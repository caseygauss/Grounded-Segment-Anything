import argparse
import os
import copy
import sys

import numpy as np
import json
import torch
import gc
from PIL import Image, ImageDraw, ImageFont

from scipy.ndimage import binary_dilation, binary_opening
from skimage.morphology import remove_small_objects

import torch.nn.functional as F

import cv2

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from scipy.ndimage import binary_fill_holes

import matplotlib
matplotlib.use('Agg')

"""
# Inside gen_mask_sam2, add the paths to the segment-anything-2 directory
sys.path.append(os.path.abspath('segment-anything-2'))
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
# Import the required modules

from sam2.sam2_image_predictor import SAM2ImagePredictor
#from model_loader import get_sam2_model
from sam2.build_sam import build_sam2

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

checkpoint_path = os.path.join(os.path.dirname(__file__), 'segment-anything-2',  'checkpoints', 'sam2_hiera_large.pt')
model_cfg = os.path.join(os.path.dirname(__file__), 'segment-anything-2', 'sam2_configs', 'sam2_hiera_l.yaml')
#
#predictor = SAM2ImagePredictor(get_sam2_model(model_cfg, checkpoint_path))


"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def measure_eye_unevenness(selected_eye_boxes):
    """
    Measures the unevenness of eyes based on their bounding boxes.
    
    :param selected_eye_boxes: A list or tensor of selected eye bounding boxes.
                               Each box should be in the format [x1, y1, x2, y2].
    :return: A dictionary containing the vertical offset, width difference, height difference, and a combined unevenness score.
    """
    if len(selected_eye_boxes) < 2:
        raise ValueError("Insufficient eye boxes to measure unevenness. Two boxes required.")
    
    # Convert to numpy array for easier manipulation if it's a tensor
    if isinstance(selected_eye_boxes, torch.Tensor):
        selected_eye_boxes = selected_eye_boxes.cpu().numpy()
    
    # Calculate the center, width, and height of each eye box
    centers = [(box[0]+box[2])/2.0 for box in selected_eye_boxes]
    widths = [abs(box[2]-box[0]) for box in selected_eye_boxes]
    heights = [abs(box[3]-box[1]) for box in selected_eye_boxes]
    vertical_centers = [(box[1]+box[3])/2.0 for box in selected_eye_boxes]
    
    # Calculate differences
    vertical_offset = abs(vertical_centers[0] - vertical_centers[1])
    width_difference = abs(widths[0] - widths[1])
    height_difference = abs(heights[0] - heights[1])
    
    # Combine the differences into a single score (simple sum or weighted sum can be considered)
    # This scoring can be adjusted based on how much weight you want to give each component
    combined_unevenness_score = vertical_offset + width_difference + height_difference

    # Print the calculated values for troubleshooting
    #print(f"Vertical Offset: {vertical_offset}")
    #print(f"Width Difference: {width_difference}")
    #print(f"Height Difference: {height_difference}")
    #print(f"Combined Unevenness Score: {combined_unevenness_score}")
    
    return {
        "vertical_offset": vertical_offset,
        "width_difference": width_difference,
        "height_difference": height_difference,
        "combined_unevenness_score": combined_unevenness_score
    }

def dilate_mask(mask, dilation_amt):
    # Create the dilation kernel
    x, y = np.meshgrid(np.arange(dilation_amt), np.arange(dilation_amt))
    center = dilation_amt // 2
    dilation_kernel = ((x - center)**2 + (y - center)**2 <= center**2).astype(np.uint8)
    dilated_binary_img = binary_dilation(mask, dilation_kernel)
    dilated_mask = Image.fromarray(dilated_binary_img.astype(np.uint8) * 255)
    return dilated_mask, dilated_binary_img

def show_mask2(mask, ax, random_color=True, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def combine_masks_for_visualization(masks):
    """
    Combine individual masks and visualize the combined mask.
    :param masks: List of individual masks as PyTorch tensors.
    :return: None
    """
    # Check if the masks list is empty
    if len(masks) == 0:
        raise ValueError("No masks provided for combination.")

    # Ensure all masks are binary and combine them
    combined_mask = torch.zeros_like(masks[0], dtype=torch.bool)
    for mask in masks:
        combined_mask |= mask > 0.5  # Logical OR to combine

    # Visualize the combined mask
    plt.imshow(combined_mask.cpu().numpy(), cmap='gray')
    plt.title('Combined Mask')
    plt.axis('off')
    plt.show()

def visualize_combined_mask(combined_mask):
    """
    Visualize the combined mask.
    :param combined_mask: Combined mask as a PyTorch tensor.
    :return: None
    """
    # Ensure mask is a 2D array by removing any singleton dimensions
    combined_mask_2d = combined_mask.squeeze()

    # Check if combined_mask_2d is still not 2D, raise an error
    if combined_mask_2d.dim() != 2:
        raise ValueError(f"Expected a 2D tensor after squeeze, but got shape {combined_mask_2d.shape}")

    # Now visualize
    plt.imshow(combined_mask_2d.cpu().numpy(), cmap='gray')
    plt.title('Combined Mask Before Filling')
    plt.axis('off')
    plt.show()


def visualize_filled_mask(combined_mask):
    """
    Visualize the filled mask.
    :param combined_mask: Combined mask as a PyTorch tensor.
    :return: None
    """
    # Convert to numpy and fill the gaps
    filled_mask = binary_fill_holes(combined_mask)

    # Visualize the filled mask
    plt.imshow(filled_mask, cmap='gray')
    plt.title('Filled Mask')
    plt.axis('off')
    plt.show()


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def resize_image(image, min_size=300):
    """
    Resize the image to ensure the smallest dimension is at least min_size while maintaining the aspect ratio.
    """
    width, height = image.size
    scale_factor = 1
    if min(width, height) < min_size:
        # Calculate the new size while maintaining aspect ratio
        if width < height:
            new_width = min_size
            new_height = int((min_size / width) * height)
            scale_factor = min_size / width
        else:
            new_height = min_size
            new_width = int((min_size / height) * width)
            scale_factor = min_size / height
        
        # Resize the image
        image = image.resize((new_width, new_height), Image.LANCZOS)
        print(f"Image resized to: {new_width}x{new_height}")
    return image, scale_factor

def scale_boxes(boxes, scale_factor):
    """
    Scale the bounding boxes back to the original image size.
    """
    print(f"\n\nScaling boxes: {scale_factor}")
    boxes = boxes / scale_factor
    
    return boxes


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    print("Image size is ", image.shape)

    try:
        model = model.to(device)
        print("Model moved to device:", device)
        
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image should be a torch tensor")
        
        image = image.to(device)
        print("Image moved to device:", device)
        print(f"Image shape: {image.shape}")
    except Exception as e:
        print(f"Error during preparation: {e}")
        return "None"

    print("Image size is ", image.shape)

    try:
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        print("Model inference completed")
    except Exception as e:
        print(f"Error during model inference: {e}")
        return "None"

    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # Debugging: Print logits and boxes shapes
    print(f"logits.shape: {logits.shape}, boxes.shape: {boxes.shape}")

    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # Debugging: Print filtered logits and boxes shapes
    print(f"logits_filt.shape: {logits_filt.shape}, boxes_filt.shape: {boxes_filt.shape}")

     # Ensure the logits_filt size is not zero before further processing
    if logits_filt.size(0) == 0:
        return torch.empty((0, 4)), []  # Return empty results if no logits are above the threshold
    
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def combine_and_fill_gaps(masks, fill_structure_size=3):
    """
    Combine individual masks and fill the gaps in the combined mask.
    :param masks: List of individual masks.
    :param fill_structure_size: Size of the structure element for gap filling.
    :return: Combined mask with gaps filled.
    """
    # Check if the masks list is empty
    # Check if the masks list is empty
    if len(masks) == 0:
        raise ValueError("No masks provided for combination.")

    # Ensure all masks are binary (bool type) and combine them
    combined_mask = torch.zeros_like(masks[0], dtype=torch.bool)
    for idx, mask in enumerate(masks):
        binary_mask = mask > 0.5  # Ensure binary mask
        combined_mask |= binary_mask  # Logical OR to combine

    # Visualize the combined mask before filling
    #visualize_combined_mask(combined_mask)

    combined_mask_2d = combined_mask.squeeze()

    # Convert to numpy and fill the gaps
    filled_mask = binary_fill_holes(combined_mask_2d)

    # Visualize the filled mask
    #visualize_filled_mask(combined_mask_2d)


    # Convert filled mask back to PyTorch tensor, maintaining the boolean data type
    filled_mask_tensor = torch.from_numpy(filled_mask).to(torch.bool)

    return filled_mask_tensor




def show_mask(mask, ax, random_color=False):
    mask = mask.astype(np.uint8)

    if random_color:
        color = np.array([200/255, 200/255, 200/255, 0.6])
    else:
        color = np.array([200, 200, 200, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def show_box2(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    


def save_mask_data(output_dir, mask_tensor, box_list, label_list, image_name, save_path, crop_x, crop_y, original_size, just_measuring=False, character_index=None, scale_factor=1):
    value = 0  # 0 for background
    background_alpha = 0.01


    # Assume mask_tensor is a PyTorch tensor of shape [1, height, width]
    mask_np = mask_tensor.cpu().numpy().squeeze()  # Convert to numpy and remove the first dimension

    # Check if image has width of 1792 or 1454
    #assert mask_np.shape[1] in [1448, 1792], f"Mask width is {mask_np.shape[1]}, expected 1454 or 1792"

    # Create an RGBA image with the same dimensions as the mask
    #mask_img = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)

    original_mask_size = (int(mask_np.shape[1] / scale_factor), int(mask_np.shape[0] / scale_factor))

    # Resize mask_np to the original mask size
    resized_mask_np = np.array(Image.fromarray(mask_np).resize(original_mask_size, Image.NEAREST))  # Resize mask down

    # Create an RGBA image with the resized mask dimensions
    mask_img_np = np.zeros((original_mask_size[1], original_mask_size[0], 4), dtype=np.uint8)

    # Set the mask color (white with full opacity)
    mask_color = np.array([225, 225, 225, 255], dtype=np.uint8)  # White color mask with full opacity


    #original_mask = np.zeros((int(mask_np.shape[0]/scale_factor), int(mask_np.shape[1]/scale_factor), 4), dtype=np.uint8)
    # Apply the mask color and alpha to where mask_np indicates (assuming mask_np is a binary mask)
    mask_img_np[resized_mask_np > 0] = mask_color  # Update this condition based on how mask_np indicates the mask

    
    # Convert the numpy array to a PIL Image
    mask_img = Image.fromarray(mask_img_np, 'RGBA')
    
    # Create a new, transparent background image
    background_img = Image.new('RGBA', original_size, (0, 0, 0, 0))
    
    # Place the mask on the background at specified coordinates
    background_img.paste(mask_img, (int(crop_x/scale_factor), int(crop_y/scale_factor)), mask_img)


    # Apply the mask color where mask_np is True
    #mask_img[mask_np] = mask_color

    if not just_measuring:
        # Save the mask image
        path_name = "mask"+save_path+".png"
        mask_image_path = os.path.join(output_dir, path_name)
        background_img.save(mask_image_path, format='PNG')

    

def is_close(box1, box2, image_width, relative_threshold=0.2):
    """
    Determine if two boxes are close to each other, based on a percentage of the image width.

    :param box1: First bounding box [x1, y1, x2, y2].
    :param box2: Second bounding box [x1, y1, x2, y2].
    :param image_width: The width of the image.
    :param relative_threshold: The percentage of the image width to use as the distance threshold.
    :return: True if the distance between the centers of box1 and box2 is less than the calculated threshold, False otherwise.
    """
    # Calculate the distance threshold as a percentage of the image width
    threshold = image_width * relative_threshold

    # Calculate the centers of the two boxes
    center_box1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center_box2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]

    # Calculate the Euclidean distance between the centers of the two boxes
    distance = ((center_box1[0] - center_box2[0]) ** 2 + (center_box1[1] - center_box2[1]) ** 2) ** 0.5

    # Determine if the boxes are close based on the calculated threshold
    return distance < threshold

def filter_face_boxes(face_boxes, hair_boxes, eye_boxes, proximity_threshold=1.0):
    filtered_face_boxes = []
    
    for face_box in face_boxes:
        close_to_hair = any(is_close(face_box, hair_box, proximity_threshold) for hair_box in hair_boxes)
        close_to_eye = any(is_close(face_box, eye_box, proximity_threshold) for eye_box in eye_boxes)
        if close_to_hair and close_to_eye:
            filtered_face_boxes.append(face_box)
    return filtered_face_boxes

def select_highest_confidence_face_box(box_label_pairs):
    """
    Selects the pair with the highest confidence score for the face.

    :param box_label_pairs: List of dictionaries, each containing a 'box' and a 'score'.
    :return: The dictionary (pair) with the highest confidence score.
    """
    if not box_label_pairs:
        return None
    
    # Find the pair with the highest score
    highest_score_pair = max(box_label_pairs, key=lambda pair: pair['score'])
    
    return highest_score_pair

def select_centermost_face_box(box_label_pairs, image_size):
    """
    Selects the pair that contains the face box closest to the center of the image.

    :param box_label_pairs: List of dictionaries, each containing a 'box'.
    :param image_size: Size of the image as a tuple (width, height).
    :return: The dictionary (pair) containing the face box closest to the center of the image.
    """
    if not box_label_pairs:
        return None
    
    y = image_size[1]
    y_mid = y / 4
    image_center = (image_size[0] / 2, y_mid * 1.5)
    min_distance = float('inf')
    centermost_pair = None

    for pair in box_label_pairs:
        box = pair['box']
        box_center_point = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        distance = np.sqrt((box_center_point[0] - image_center[0])**2 + (box_center_point[1] - image_center[1])**2)

        if distance < min_distance:
            min_distance = distance
            centermost_pair = pair

    return centermost_pair

def face_confidence_score(box_label_pairs, image_size, center_weight=0.65, confidence_weight=0.35):
    """
    Evaluates each face box based on its proximity to the image center and its confidence level.

    :param box_label_pairs: List of dictionaries, each containing a 'box' and a 'score'.
    :param image_size: Size of the image as a tuple (width, height).
    :param center_weight: Coefficient giving weight to how centered the box is.
    :param confidence_weight: Coefficient giving weight to the box's confidence score.
    :return: The dictionary (pair) with the highest combined score.
    """
    if not box_label_pairs:
        return None

    # Define the center of the image
    y = image_size[1]
    y_mid = y / 3
    image_center = (image_size[0] / 2, y_mid)
    max_score = -float('inf')
    best_pair = None

    for pair in box_label_pairs:
        box = pair['box']
        # Calculate the center point of the box
        box_center_point = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        # Calculate the Euclidean distance from the box's center to the image's center
        distance = np.sqrt((box_center_point[0] - image_center[0])**2 + (box_center_point[1] - image_center[1])**2)
        # Normalize the distance based on the image size for comparison
        normalized_distance = distance / np.sqrt(image_size[0]**2 + image_size[1]**2)
        # Calculate the centeredness score (inverted because lower distance means more centered)
        centeredness_score = 1 - normalized_distance
        # Retrieve the confidence score
        confidence_score = pair['score']
        # Calculate the combined score
        combined_score = (center_weight * centeredness_score) + (confidence_weight * confidence_score)
        
        if combined_score > max_score:
            max_score = combined_score
            best_pair = pair

    return best_pair

def face_confidence_score(box_label_pairs):
    """
    Selects the face box with the highest confidence score.

    :param box_label_pairs: List of dictionaries, each containing a 'box' and a 'score'.
    :return: The dictionary (pair) with the highest confidence score.
    """
    if not box_label_pairs:
        return None

    # Initialize variables to keep track of the best pair
    best_pair = None
    highest_score = -float('inf')

    # Iterate over all box-label pairs to find the one with the highest score
    for pair in box_label_pairs:
        confidence_score = pair['score']
        
        # Update the best pair if the current score is higher than the highest score seen so far
        if confidence_score > highest_score:
            highest_score = confidence_score
            best_pair = pair

    return best_pair


def box_center(box):
    """
    Calculate the center of a bounding box.

    :param box: Bounding box in format [x1, y1, x2, y2].
    :return: Center of the box (x_center, y_center).
    """
    # Ensure the box has four elements
    assert len(box) == 4, "Box does not have the expected format [x1, y1, x2, y2]."
    
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return x_center, y_center

def distance_between_boxes(boxA, boxB):
    """
    Calculate the Euclidean distance between the centers of two boxes.

    :param boxA: First bounding box.
    :param boxB: Second bounding box.
    :return: Euclidean distance.
    """
    centerA = box_center(boxA)
    centerB = box_center(boxB)
    distance = np.sqrt((centerA[0] - centerB[0])**2 + (centerA[1] - centerB[1])**2)

    #print("Distance between boxes:", distance)
    return distance

def edge_distance(boxA, boxB):
    """
    Calculate the minimum distance between the edges of two boxes.

    :param boxA: First bounding box.
    :param boxB: Second bounding box.
    :return: Minimum edge distance.
    """
    left = max(boxA[0], boxB[0])
    right = min(boxA[2], boxB[2])
    top = max(boxA[1], boxB[1])
    bottom = min(boxA[3], boxB[3])

    horizontal_distance = left - right
    vertical_distance = top - bottom

    #print("Horizontal distance:", horizontal_distance)
    #print("Vertical distance:", vertical_distance)

    return max(horizontal_distance, vertical_distance)

def boxes_overlap_or_close(boxA, boxB, threshold=0):
    """
    Checks if two boxes overlap or are within a certain threshold distance from each other.

    :param boxA: First bounding box in the format (left, top, right, bottom).
    :param boxB: Second bounding box in the format (left, top, right, bottom).
    :param threshold: The maximum distance between boxes to consider them as close. Default is 0 for touching.
    :return: True if boxes overlap or are within the threshold distance, False otherwise.
    """
    # Check if boxes overlap
    if boxA[2] < boxB[0] or boxA[0] > boxB[2] or boxA[3] < boxB[1] or boxA[1] > boxB[3]:
        # Boxes do not overlap, check the distance between edges for closeness
        
        # Horizontal distance
        horizontal_distance = max(0, max(boxB[0] - boxA[2], boxA[0] - boxB[2]))
        
        # Vertical distance
        vertical_distance = max(0, max(boxB[1] - boxA[3], boxA[1] - boxB[3]))
        
        # If either distance is greater than the threshold, the boxes are neither overlapping nor close
        return False if horizontal_distance > threshold or vertical_distance > threshold else True
    else:
        # Boxes overlap
        return True

def filter_boxes_near_face_box(boxes, face_box, distance_threshold):
    """
    Filter boxes that are close to the face box based on a distance threshold.

    :param boxes: List of bounding boxes.
    :param face_box: The face bounding box.
    :param distance_threshold: Distance threshold for filtering.
    :return: List of boxes close to the face box.
    """
    filtered_boxes = []
    for box in boxes:
        if distance_between_boxes(box, face_box) <= distance_threshold:
            filtered_boxes.append(box)
    return filtered_boxes

def filter_and_limit_boxes(box_label_pairs, face_box, image_width, max_count):
    """
    Filters and limits the number of boxes based on proximity to the face box.
    Each element is only selected if the distance is within 10% of the image's width.

    :param box_label_pairs: List of dictionaries, each containing a 'box' and associated data.
    :param face_box: The selected face box.
    :param image_width: The width of the image, to determine the proximity threshold.
    :param max_count: Maximum number of boxes to select.
    :return: Filtered and limited list of box-label pairs.
    """
    # Define the proximity threshold as 10% of the image's width
    proximity_threshold = 0.025 * image_width

    # Filter pairs based on distance to the face_box
    filtered_pairs = [
        pair for pair in box_label_pairs
        #Measures distance from centers of boxes
        #if distance_between_boxes(pair['box'], face_box) <= proximity_threshold

        #Looks for overlap or closeness
        if boxes_overlap_or_close(pair['box'], face_box, threshold=proximity_threshold)
    ]
    # Assuming we need a way to sort the filtered pairs since the original function did,
    # we might use a heuristic like the size of the overlap or the closeness,
    # for simplicity, keep the original distance metric for sorting
    distances = [distance_between_boxes(pair['box'], face_box) for pair in filtered_pairs]
    sorted_pairs = sorted(zip(distances, filtered_pairs), key=lambda x: x[0])

    # Limit the number of pairs based on max_count
    limited_pairs = [pair for _, pair in sorted_pairs][:max_count]
    
    return limited_pairs

def show_tensor_image(tensor_img):
    """ Utility function to display a tensor image """
    tensor_img = tensor_img.cpu()
    if tensor_img.ndimension() == 4:
        tensor_img = tensor_img.squeeze(0)
    # Normalize tensor values to [0, 1] for displaying with imshow
    tensor_img = (tensor_img - tensor_img.min()) / (tensor_img.max() - tensor_img.min())
    plt.imshow(tensor_img.permute(1, 2, 0).numpy())
    plt.show()

def show_image(img_pil):
    """ Utility function to display a PIL image """
    plt.imshow(np.array(img_pil))
    plt.show()

# Function to clean mask
def clean_mask(mask, min_size=500):
    mask = binary_opening(mask, structure=np.ones((3, 3)))
    mask = remove_small_objects(mask.astype(bool), min_size=min_size)
    return mask.astype(np.float32)  # Ensure the mask is in float32 format

def run_grounding_sam_demo(config_file, grounded_checkpoint, sam_version, sam_checkpoint, sam_hq_checkpoint, use_sam_hq, image_path, text_prompt, output_dir, box_threshold, text_threshold, device, character_prompt="", save_path="", just_measuring=False, negative_points=[], box_to_use_num=None, box_coordinates={}, is_poster=False, char_type="human", character_index=None,predictor=None, head_only=False):
    detection_status = "None"

    print("box to use num is", box_to_use_num)
    print("character prompt is", character_prompt)

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    print("The image path is", image_path)
    original_image_pil, original_image = load_image(image_path)
    original_size = original_image_pil.size
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    original_image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    cropped_image = original_image
    print("Early cropped image shape is", cropped_image.shape)

    crop_x = 0
    crop_y = 0

    char_save_path = save_path

    print("Character_prompt is: ", character_prompt)
    print("Box coordinates are: ", box_coordinates)
    
    if character_prompt != "" and text_prompt == "" and "Main Character" not in text_prompt:

        all_box_coordinates = {}
        
        try:
            char_boxes_filt, char_pred_phrases = get_grounding_output(
                model, original_image, character_prompt, 0.1, 0.01, device=device
            )
        except:
            print("Error getting grounding output")
            return "None"

        print("Character box count is", len(char_boxes_filt))

        # Prepare to save images for each box
        size = original_image_pil.size
        H, W = size[1], size[0]

        box_found = False

         # Extract scores and sort the boxes and phrases by score in descending order
        char_scores = [float(label.split('(')[-1].strip(')')) for label in char_pred_phrases]
        sorted_indices = sorted(range(len(char_scores)), key=lambda i: char_scores[i], reverse=True)
        sorted_char_boxes = [char_boxes_filt[i] for i in sorted_indices]
        sorted_char_pred_phrases = [char_pred_phrases[i] for i in sorted_indices]
        
        # Limit to top 3 options
        top_char_boxes = sorted_char_boxes[:5]
        top_char_pred_phrases = sorted_char_pred_phrases[:5]

        for i, (box, label) in enumerate(zip(top_char_boxes, top_char_pred_phrases)):
            score = float(label.split('(')[-1].strip(')'))  # Extract the score
            # Scale and transform box coordinates to PIL crop format (left, upper, right, lower)
            scaled_box = box * torch.Tensor([W, H, W, H])
            scaled_box[:2] -= scaled_box[2:] / 2
            scaled_box[2:] += scaled_box[:2]
            left, upper, right, lower = scaled_box.cpu().numpy().tolist()

            ##expand box by 10px in each direction if possible
            if left - 6 > 0:
                left -= 6
            if upper - 6 > 0:
                upper -= 6
            if right + 6 < W:
                right += 6
            if lower + 6 < H:
                lower += 6

            # Save coordinates
            all_box_coordinates[f"box_{i + 1}"] = {"left": left, "upper": upper, "right": right, "lower": lower}
            #print("running box coordinates: ", all_box_coordinates[f"box_{i + 1}"])

            # Crop the image to the character box
            cropped_image = original_image_pil.crop((left, upper, right, lower))
            
            file_name = f"cropped_img_{char_save_path}_option_{i + 1}.jpg"
            if head_only:
                file_name = f"cropped_head_img_{char_save_path}_option_{i + 1}.jpg"
            print("Saving the cropped images as: ", file_name)
            cropped_image.save(os.path.join(output_dir, file_name))

            crop_x, crop_y = left, upper

            # Save a box to the base cropped image
            if i == 0:
                special_file_name = f"cropped_img_{char_save_path}.jpg"
                if head_only:
                    special_file_name = f"cropped_head_img_{char_save_path}.jpg"
                cropped_image.save(os.path.join(output_dir, special_file_name))

            box_found = True

        if not box_found:
            print("No character box found")
            return "None"
        else:
            print("Returning all box coordinates: ", all_box_coordinates)
            return all_box_coordinates
        
    scale_factor = 1
    # If box coordinates exist, then create a crop of the base image using the coordinates
    if box_coordinates:
        # Crop the original image using the original coordinates
        crop_x = box_coordinates["left"]
        crop_y = box_coordinates["upper"]
        crop_width = box_coordinates["right"] - box_coordinates["left"]
        crop_height = box_coordinates["lower"] - box_coordinates["upper"]

        cropped_pil = original_image_pil.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

        # Ensure the cropped image is in RGB format
        cropped_pil = cropped_pil.convert("RGB")

        # Scale up the cropped image and adjust coordinates
        scaled_image, scaled_box_coordinates, scale_factor = scale_image_and_adjust_coordinates(
            cropped_pil, box_coordinates
        )

        # Show the cropped image for troubleshooting
        #cropped_pil.show()
        # Convert the scaled image to a tensor
        cropped_img = torch.from_numpy(np.array(scaled_image).transpose(2, 0, 1)).float().div(255.0)
        cropped_pil = scaled_image

        # Scaled coordinates
        crop_x = scaled_box_coordinates["left"]
        crop_y = scaled_box_coordinates["upper"]
        crop_width = scaled_box_coordinates["right"] - scaled_box_coordinates["left"]
        crop_height = scaled_box_coordinates["lower"] - scaled_box_coordinates["upper"]

    else:
        print("No box coordinates found, using the original image")
        cropped_img = cropped_image
        cropped_pil = original_image_pil

        if 'Emotions' in image_path:
            
            scale_factor = 0.78125
        #cropped_img = torch.from_numpy(np.array(cropped_img).transpose(2, 0, 1)).float().div(255.0)

    if text_prompt == "":
        return "Done"
    
    print(f"Original image size: {original_image_pil.size}, Scaled image size: {cropped_img.shape}")
    print(f"Scale factor: {scale_factor}")
    #print(f"Scaled box coordinates: {scaled_box_coordinates}")

    # run grounding dino model
    try:
        boxes_filt, pred_phrases = get_grounding_output(
            model, cropped_img, text_prompt, box_threshold, text_threshold, device=device
        )

        torch.cuda.empty_cache()

    except:
        print("Error getting grounding output")
        return "None"
    
    # Categorize boxes by label
    #face_boxes = [box for box, label in zip(boxes_filt, pred_phrases) if 'face' in label]
    #hair_boxes = [box for box, label in zip(boxes_filt, pred_phrases) if 'hair' in label]
    #eye_boxes = [box for box, label in zip(boxes_filt, pred_phrases) if 'eye' in label]

    # Modify the following part of your existing function
    # Categorize boxes by label and extract confidence scores
    face_boxes = []
    face_scores = []
    eye_boxes = []
    mouth_boxes = []
    ear_boxes = []
    hair_boxes = []
    main_character_boxes = []
    main_character_scores = []
    pet_boxes =[]
    other_boxes = []

    #print("Crop x and crop y are", crop_x, crop_y)

    # Convert the PIL image to a NumPy array
    image_np = np.array(cropped_pil)

    # Convert the NumPy array to RGB if it's not already (PIL images are in RGB by default)
    if image_np.shape[2] == 4:  # Check if the image has an alpha channel
        image_np = image_np[:, :, :3]  # Drop the alpha channel

    # Ensure the image is in RGB format
    #image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image=image_np

    size = cropped_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    #print("Box area threshold is", threshold_area)

    #predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint_path))

    for box, label in zip(boxes_filt, pred_phrases):
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        box_area = box_width * box_height
        box_area = box_area.item()  # Convert tensor to float
        

        ##Something wrong here not looking at the same boxes for comparison
        #if box_area > threshold_area:
        #    continue

        score = float(label.split('(')[-1].strip(')'))  # Extract the score
        info_obj = {'box': box, 'label': label, 'score': score}
        print(f"Box found - {info_obj['label']} with score {info_obj['score']}")
        if 'face' in label:
            face_boxes.append(info_obj)
            face_scores.append(score)
            #print(f"Face box: {box}, score: {score}")
        elif 'eye' in label:
            eye_boxes.append(info_obj)
        elif 'mouth' in label:
            mouth_boxes.append(info_obj)
        elif 'ear' in label:
            ear_boxes.append(info_obj)
        elif 'hair' in label:
            hair_boxes.append(info_obj)
        elif 'dog' in label or 'cat' in label or 'kitten' in label:
            pet_boxes.append(info_obj)
        elif 'main character' in label:
            print("Main character box found")
            main_character_boxes.append(info_obj)
            score = float(label.split('(')[-1].strip(')'))  # Extracting the confidence score from the label
            main_character_scores.append(score)
        else:
            other_boxes.append(info_obj)
    the_hair_box = None
    closest_hair_pair = None
    if not face_boxes and not main_character_boxes:
        if hair_boxes:

            #Calculate the middle of the image
            image_middle = (W / 2, H / 2)

            #Find the hairbox closest to the middle of the image
            closest_hair_box = None
            closest_distance = float('inf')

            # Iterate over hair box pairs instead of just boxes
            for pair in hair_boxes:
                box = pair['box']  # Extract the box from the pair
                box_middle = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                distance = np.sqrt((box_middle[0] - image_middle[0]) ** 2 + (box_middle[1] - image_middle[1]) ** 2)
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_hair_pair = pair  # Update to store the closest pair, not just the box


            #Filter boxes near the closest hair box
            if closest_hair_pair is not None and closest_distance < 1250:
                the_hair_box = [closest_hair_pair['box']]
                focal_box = [closest_hair_pair['box']]
                focal_box_label = closest_hair_pair['label']  # Optionally keep track of the label
                #print("Closest hair box:", closest_hair_pair['box'])
            else:
                #No Hair box, so we're going to skip the rest of the code
                print("No hair box found in the zone")
                #return "None"
        elif pet_boxes:
            #No hair box, so we're going to skip the rest of the code
            print("Pet box found in the zone")
            focal_box = [pet_boxes[0]['box']]
            focal_box_label = pet_boxes[0]['label']
        else:
            #No hair box, so we're going to skip the rest of the code
            print("No hair box found")
            #return "None"
        
    focal_box = None
    focal_box_label = None
    
    # Select the highest confidence face box
    if face_boxes:
        detection_status = "face_found"
        #highest_confidence_face_box = select_highest_confidence_face_box(face_boxes, face_scores)
        chosen_pair = face_confidence_score(face_boxes)
        if chosen_pair is not None:
            chosen_face_box = chosen_pair['box']
            focal_box = [chosen_face_box]
            focal_box_label = chosen_pair['label']
            #print("Chosen face box based on combined score:", chosen_face_box)
        else:
            print("No suitable face box found")
           

        # Debugging: Print the selected face box
        #print("Selected face box:", chosen_face_box)
        # Debugging: Print the number of filtered boxes
        #print("Number of boxes close to the face box:", len(close_boxes))
    elif main_character_boxes:
        detection_status = "character_found"
        # Convert the list of scores to a numpy array for easier manipulation
        # Assuming main_character_boxes is a list of dictionaries each with 'box', 'label', and 'score'
        highest_score_pair = max(main_character_boxes, key=lambda pair: pair['score'])
        chosen_face_box = highest_score_pair['box']
        focal_box = [chosen_face_box]  # Ensure focal_box is a list containing the chosen box
        focal_box_label = highest_score_pair['label']
    elif the_hair_box:
        detection_status = "hair_found"
        #print("hair was found")
    elif pet_boxes:
        detection_status = "pet_found"
        #print("pet was found")
    else:
        detection_status = "None"
        ##If close_boxes is empty, then we're going to skip the rest of the code
        return detection_status
        

    all_selected_pairs = []
    close_boxes = []

    if focal_box is not None and detection_status != 'hair_found' and not main_character_boxes and not pet_boxes:
        # Filter boxes near the selected face box with an increased threshold
        #close_boxes = filter_boxes_near_face_box(boxes_filt, highest_confidence_face_box, 200)
        # Assuming each *_boxes variable is now a list of dictionaries with 'box' and 'label' (and optionally 'score')
        selected_eye_pairs = filter_and_limit_boxes(eye_boxes, chosen_face_box, W, 2)
        selected_mouth_pairs = filter_and_limit_boxes(mouth_boxes, chosen_face_box, W, 1)
        selected_ear_pairs = filter_and_limit_boxes(ear_boxes, chosen_face_box, W, 2)
        selected_hair_pairs = filter_and_limit_boxes(hair_boxes, chosen_face_box, W, 1)

        all_selected_pairs = selected_eye_pairs + selected_mouth_pairs + selected_ear_pairs + selected_hair_pairs

        #print("Length of all selected pairs:", len(all_selected_pairs))
        # Extract just the boxes for measuring unevenness or further processing
        selected_eye_boxes = [pair['box'] for pair in selected_eye_pairs]

        # Continue with your logic
        if len(selected_eye_boxes) >= 2:
            eye_unevenness = measure_eye_unevenness(selected_eye_boxes)

        selected_mouth_boxes = [pair['box'] for pair in selected_mouth_pairs]
        selected_ear_boxes = [pair['box'] for pair in selected_ear_pairs]
        selected_hair_boxes = [pair['box'] for pair in selected_hair_pairs]

        close_boxes = [pair['box'] for pair in selected_eye_pairs + selected_mouth_pairs + selected_ear_pairs + selected_hair_pairs]
        #close_boxes.append(focal_box)
    elif the_hair_box:
        #print('The closest hair box is:', the_hair_box)
        #print('The closest hair pair is ', closest_hair_pair)
        for pair in ear_boxes:
            close_boxes.append(pair['box'])
        close_boxes.append(closest_hair_pair['box'])  # Assuming closest_hair_pair is a dict
        all_selected_pairs = [closest_hair_pair] + ear_boxes
    elif pet_boxes:
        print('Pet box found in the zone')
        for pair in pet_boxes:
            close_boxes.append(pair['box'])
        all_selected_pairs = [pair for pair in pet_boxes]
        
    if focal_box is not None and the_hair_box is None:
        close_boxes += focal_box
        #focal_box_label = focal_box_label  # Example label, adjust based on your logic
        focal_box_pair = {'box': focal_box[0], 'label': focal_box_label}  # Ensure focal_box is in the expected format
        all_selected_pairs.append(focal_box_pair)


    box_areas = {}
    for pair in all_selected_pairs:
        box = pair['box']
        label = pair['label']

        # Calculate the area of the box
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        box_area = box_width * box_height

        # Convert the tensor to a float and round to two decimal places
        box_area = round(box_area.item(), 2)

        box_areas[label] = box_area
    #print("Close boxes:", close_boxes)

    torch.cuda.empty_cache()
    gc.collect()

    masks_to_use = []
    scores_to_use = []

    if not negative_points:
        try:
            #boxes_filt = torch.stack(close_boxes)
            a=1
        except:
            print("Did not find any boxes")
            return "None"

        print("Close boxes length is", len(close_boxes))
        close_boxes_np = [box.cpu().numpy() for box in close_boxes]
        #print("Length of close boxes", len(close_boxes_np))
        print("Close boxes type:", type(close_boxes_np))
        print("Close boxes length:", len(close_boxes_np))
        # Set the image for the predictor

        boxes_tensor = torch.stack(close_boxes).unsqueeze(0).to(device)
        box_batch = [boxes_tensor.squeeze(0).cpu().numpy()]
        print("Box batch type:", type(box_batch))
        print("Box batch length:", len(box_batch))

        boxes_to_use = [np.array(close_boxes_np)]
        print("Boxes to use:", boxes_to_use)
        print("Length of boxes to use", len(boxes_to_use))

        # Convert the tensor (CHW) back to a NumPy array (HWC)
        cropped_img_np = cropped_img.permute(1, 2, 0).cpu().numpy()  # CHW to HWC

        # Ensure the values are in the range [0, 255] and in RGB format
        cropped_img_np = (cropped_img_np * 255).astype(np.uint8)  # Convert to uint8 format

        # Pass the corrected NumPy array to the predictor
        predictor.set_image_batch([cropped_img_np])

        multi_mask_output = False
        if 'Main Character' in text_prompt:
            multi_mask_output = False

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            masks_batch, scores_batch, _ = predictor.predict_batch(
                None,
                None,
                box_batch=boxes_to_use,
                multimask_output=multi_mask_output
            )
        print(f"Image shape: {image.shape}")
        print("Length of masks_batch:", len(masks_batch))
        print("Length of scores_batch:", len(scores_batch))

        masks_to_use = []

        for masks, scores, boxes in zip(masks_batch, scores_batch, boxes_to_use):
            print("Masks type:", type(masks))
            print("Scores type:", type(scores))
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                print("Mask shape:", mask.shape)
                print("Mask is: ", mask)
                # Only squeeze if the first dimension has size 1
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                show_mask2(mask, plt.gca(), random_color=True)
                masks_to_use.append(mask)
            #for box in boxes:
                #show_box2(box, plt.gca())
            #plt.show()
            plt.close()

        
        print("Length of masks_to_use:", len(masks_to_use))
    else:
    
        print("Initializing negative labels and points...")
        negative_labels = [0] * len(negative_points)
        negative_labels = np.array(negative_labels)
        negative_points = np.array(negative_points)

        # Convert boxes to a tensor with shape [1, M, 4]
        print("Converting boxes to tensor...")
        boxes_tensor = torch.stack(close_boxes).unsqueeze(0).to(device)
        print("Boxes tensor shape:", boxes_tensor.shape)

        # Convert points and labels to tensors and expand them to match the number of boxes
        print("Converting points and labels to tensors...")
        points_tensor = torch.tensor(negative_points, dtype=torch.float32).unsqueeze(0).to(device)
        expanded_points_tensor = points_tensor.repeat(len(close_boxes), 1, 1)
        print("Points tensor shape:", expanded_points_tensor.shape)

        labels_tensor = torch.tensor(negative_labels, dtype=torch.long).unsqueeze(0).to(device)
        expanded_labels_tensor = labels_tensor.repeat(len(close_boxes), 1)
        print("Labels tensor shape:", expanded_labels_tensor.shape)

        # Convert the tensor (CHW) back to a NumPy array (HWC)
        cropped_img_np = cropped_img.permute(1, 2, 0).cpu().numpy()  # CHW to HWC

        # Ensure the values are in the range [0, 255] and in RGB format
        cropped_img_np = (cropped_img_np * 255).astype(np.uint8)  # Convert to uint8 format

        # Pass the corrected NumPy array to the predictor
        predictor.set_image_batch([cropped_img_np])

        
        box_batch = [boxes_tensor.squeeze(0).cpu().numpy()]

        # Pass these to the model
        print("Passing data to predictor...")
        masks_batch, scores_batch, logits_batch = predictor.predict_batch(
            point_coords_batch=[expanded_points_tensor.cpu().numpy()],
            point_labels_batch=[expanded_labels_tensor.cpu().numpy()],
            box_batch=box_batch,
            multimask_output=True
        )

        print("Processing prediction results...")
        masks_to_use = []

         # Iterate over each set of masks and scores
        for masks_set, scores_set in zip(masks_batch, scores_batch):
            scores_tensor = torch.tensor(scores_set)
            
            # Initialize best mask selection
            best_mask = None
            best_score = -1

            for mask, score_set in zip(masks_set, scores_tensor):

                best_score = -1

                for i in range(mask.shape[0]):
                    mask_np = mask[i].astype(np.float32)
                    if mask_np.max() > 1:
                        mask_np /= 255.0

                    # Handle extra dimensions if needed
                    if mask_np.shape[0] == 3:
                        mask_np = mask_np.transpose(1, 2, 0)
                        mask_np = mask_np[:, :, 0]

                    # Reshape mask if needed
                    if mask_np.size == image.shape[0] * image.shape[1]:
                        mask_np = mask_np.reshape(image.shape[0], image.shape[1])
                    else:
                        print(f"Error reshaping mask: expected {image.shape[0] * image.shape[1]} but got {mask_np.size}")
                        continue

                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    try:
                        show_mask2(mask_np, plt.gca(), random_color=True)
                    except ValueError as ve:
                        print(f"Error reshaping mask: {ve}")
                        plt.close()
                        continue
                    plt.title(f"Mask {i+1}, Score: {score_set[i].item():.3f}", fontsize=18)
                    plt.axis('off')
                    #plt.show()
                    plt.close()

                # Select the best mask based on the highest score
                highest_score_index = torch.argmax(score_set).item()
                if highest_score_index < mask.shape[0] and score_set[highest_score_index].item() > best_score:
                    best_score = score_set[highest_score_index].item()
                    best_mask = mask[highest_score_index]

                if best_mask is not None:
                    best_mask = clean_mask(best_mask)  # Clean the best mask
                    masks_to_use.append(torch.tensor(best_mask).to(device))
                else:
                    print("No valid mask found for this set")

    # Dilate each mask to add padding
    print(f"Number of selected masks: {len(masks_to_use)}")
    dilation_amt = 20  # Adjust this as needed

    if is_poster:
        dilation_amt = 10

    if char_type == "dog" or char_type == "cat":
        dilation_amt = 5
    
    padded_masks = []


    if 'Main Character sauce' not in text_prompt:
        print("yoooooo")
        for mask in masks_to_use:
            mask_np = mask  # Convert the mask to a numpy array
            dilated_mask, _ = dilate_mask(mask_np, dilation_amt)
            padded_masks.append(torch.from_numpy(np.array(dilated_mask)).unsqueeze(0))
        
    else:
        for mask in masks_to_use:
            mask_np = mask  # Convert the mask to a numpy array
            if "emotion" not in image_path:
                dilated_mask, _ = dilate_mask(mask_np, 5)
                padded_masks.append(torch.from_numpy(np.array(dilated_mask)).unsqueeze(0))
            else:
                # Do not dilate the mask but add it to padded_masks
                padded_masks.append(torch.from_numpy(np.array(mask_np)).unsqueeze(0))
                print("Proper main character mask added")
        save_path = "cover"

    area_dict = {}
    binary_padded_masks = []
    #for mask in padded_masks:
    #    binary_mask = mask > 0.5  # Thresholding
    #    binary_padded_masks.append(binary_mask)
    for i, (mask, pair) in enumerate(zip(padded_masks, all_selected_pairs)):
        binary_mask = mask > 0.5  # Thresholding
        binary_padded_masks.append(binary_mask)

        # Calculate the area of the current element
        area = torch.sum(binary_mask).item()
        label = pair['label']
        area_dict[label] = area

    #combined_masks_tensor = torch.stack(binary_padded_masks)

    filled_combined_mask = combine_and_fill_gaps(padded_masks)
    #print("Filled combined mask:", filled_combined_mask)

    # Normalize image data
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # draw output image
    plt.figure(figsize=(10, 10))

    plt.imshow(image)

    for mask in padded_masks:
        mask_np = mask.cpu().numpy()[0]
        mask_np = mask_np.astype(np.float32)
        if mask_np.max() > 1:
            mask_np /= 255.0

        for point in negative_points:
            plt.plot(point[0], point[1], 'ro', markersize=5)

        show_mask(mask_np, plt.gca(), random_color=True)

    print('all selected pairs length: ', len(all_selected_pairs))
    for pair in all_selected_pairs:
        box = pair['box'].numpy()  # Ensure the box is converted to numpy array if it's a tensor
        label = pair['label']  # Use the label directly from the pair
        show_box(box, plt.gca(), label)

    save_label = save_path
    if just_measuring:
        save_label += "_inpainted"

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, f"grounded_sam_output_{save_label}.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    plt.close()

    save_mask_data(output_dir, filled_combined_mask, boxes_filt, pred_phrases, image_path, save_path, crop_x, crop_y, original_size, just_measuring, character_index, scale_factor)

    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleaned - main")

    if detection_status == 'None' or detection_status == 'hair_found':
        return detection_status
    else:
        return box_areas
    
# Function to scale an image and adjust coordinates
def scale_image_and_adjust_coordinates(image, box_coordinates, min_width=600, max_dim=2000):
    W, H = image.size  # Original image dimensions

    # Calculate the width of the cropped box
    crop_width = box_coordinates["right"] - box_coordinates["left"]

    # Determine the scale factor based on minimum width or max dimension
    if crop_width < min_width:
        scale_factor = min_width / crop_width
    else:
        scale_factor = 1  # No scaling needed if width is already larger than min_width

    # Ensure the scaled image doesn't exceed the max dimension
    if max(W * scale_factor, H * scale_factor) > max_dim:
        scale_factor = max_dim / max(W, H)  # Scale by the maximum dimension

    # Scale the entire image
    new_width = int(W * scale_factor)
    new_height = int(H * scale_factor)

    # Prevent scaling down to a very small image size
    if new_width < min_width:
        new_width = min_width
        scale_factor = new_width / W
        new_height = int(H * scale_factor)

    scaled_image = image.resize((new_width, new_height))

    # Adjust the box coordinates based on the scale factor
    scaled_box_coordinates = {
        "left": max(0, box_coordinates["left"] * scale_factor),
        "upper": max(0, box_coordinates["upper"] * scale_factor),
        "right": min(new_width, box_coordinates["right"] * scale_factor),
        "lower": min(new_height, box_coordinates["lower"] * scale_factor),
    }

    return scaled_image, scaled_box_coordinates, scale_factor

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--character_prompt", type=str, default="", help="middle boy standing")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--save_path", type=str, default="", help="ending to file")
    parser.add_argument("--just_measuring", type=bool, default=False, help="just measuring the area of specified boxes or not")
    parser.add_argument("--negative_points", type=list, default=[], help="points cords to be excluded from the mask")
    parser.add_argument("--box_to_use_num", type=int, default=None, help="box number of main char to use for masking")
    parser.add_argument("--is_poster", type=bool, default=False, help="is poster or not")
    parser.add_argument("--char_type", type=str, default="human", help="character type")
    parser.add_argument("--character_index", type=int, default=None, help="character index")
    parser.add_argument("--predictor", type=str, default=None, help="predictor")
    parser.add_argument("--head_only", type=bool, default=False, help="head only or not")
    args = parser.parse_args()

    # Call the new function with the parsed arguments
    run_grounding_sam_demo(
        args.config,
        args.grounded_checkpoint,
        args.sam_version,
        args.sam_checkpoint,
        args.sam_hq_checkpoint,
        args.use_sam_hq,
        args.input_image,
        args.text_prompt,
        args.output_dir,
        args.box_threshold,
        args.text_threshold,
        args.device,
        args.character_prompt,
        args.save_path,
        args.just_measuring,
        args.negative_points,
        args.box_to_use_num,
        args.is_poster,
        args.char_type,
        args.character_index,
        args.predictor,
        args.head_only
    )
