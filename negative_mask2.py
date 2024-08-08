import argparse
import os
import sys
import random

import numpy as np
import gc
import torch
from PIL import Image, ImageDraw, ImageFont

from scipy.ndimage import binary_dilation, binary_opening
from skimage.morphology import remove_small_objects

import cv2

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt

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

# Dilate mask function
def dilate_mask(mask, dilation_amt):
    structure = np.ones((dilation_amt, dilation_amt), dtype=bool)
    dilated_binary_img = binary_dilation(mask, structure)
    return dilated_binary_img, structure


# Clean mask function
def clean_mask(mask, min_size=500):
    mask = binary_opening(mask, structure=np.ones((3, 3)))
    mask = remove_small_objects(mask.astype(bool), min_size=min_size)
    return mask.astype(np.float32) # Ensure the same dtype is retained

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
    #plt.imshow(combined_mask_2d.cpu().numpy(), cmap='gray')
    #plt.title('Combined Mask Before Filling')
    #plt.axis('off')
    #plt.show()


def visualize_filled_mask(combined_mask):
    """
    Visualize the filled mask.
    :param combined_mask: Combined mask as a PyTorch tensor.
    :return: None
    """
    # Convert to numpy and fill the gaps
    filled_mask = binary_fill_holes(combined_mask)

    # Visualize the filled mask
    #plt.imshow(filled_mask, cmap='gray')
    #plt.title('Filled Mask')
    #plt.axis('off')
    #plt.show()


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


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

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

# Function to show masks
def show_mask(mask, ax, random_color=False, borders = True):
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

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_tensor, box_list, label_list, image_name, save_path, crop_x, crop_y, original_size, just_measuring=False):
    value = 0  # 0 for background
    background_alpha = 0.01

    # Assume mask_tensor is a PyTorch tensor of shape [1, height, width]
    mask_np = mask_tensor.cpu().numpy().squeeze()  # Convert to numpy and remove the first dimension

    # Check if image has width of 1792 or 1454
    #assert mask_np.shape[1] in [1448, 1792], f"Mask width is {mask_np.shape[1]}, expected 1454 or 1792"

    # Create an RGBA image with the same dimensions as the mask
    #mask_img = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)

    # Set the mask color and alpha
    mask_color = np.array([225, 225, 225, 255], dtype=np.uint8)  # White color mask with full opacity
    mask_img_np = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)
    # Apply the mask color and alpha to where mask_np indicates (assuming mask_np is a binary mask)
    mask_img_np[mask_np > 0] = mask_color  # Update this condition based on how mask_np indicates the mask

    
    # Convert the numpy array to a PIL Image
    mask_img = Image.fromarray(mask_img_np, 'RGBA')
    
    # Create a new, transparent background image
    background_img = Image.new('RGBA', original_size, (0, 0, 0, 0))
    
    # Place the mask on the background at specified coordinates
    background_img.paste(mask_img, (int(crop_x), int(crop_y)), mask_img)


    # Apply the mask color where mask_np is True
    #mask_img[mask_np] = mask_color

    if not just_measuring:
        # Save the mask image
        path_name = "mask"+save_path+"_negative.png"
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

def generate_candidate_points(box, num_points=10):
    """Generate candidate points within the box."""
    x_min, y_min, x_max, y_max = box
    candidate_points = []
    for _ in range(num_points):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)  # Fixed typo: should be y instead of x
        candidate_points.append((x, y))
    return candidate_points

def is_within_excluded_region(point, excluded_boxes):
    """Check if the point is within any of the excluded regions."""
    x, y = point
    for ex_box in excluded_boxes:
        ex_x_min, ex_y_min, ex_x_max, ex_y_max = ex_box
        if ex_x_min <= x <= ex_x_max and ex_y_min <= y <= ex_y_max:
            return True
    return False

def find_best_point(box, excluded_boxes, num_points=5):
    """Find the best point within the box that is not in the excluded regions."""
    candidate_points = generate_candidate_points(box, num_points)
    valid_points = [point for point in candidate_points if not is_within_excluded_region(point, excluded_boxes)]
    #print(f"Candidate points for box {box}: {candidate_points}")
    #print(f"Valid points for box {box}: {valid_points}")
    if valid_points:
        # Return the point closest to the center of the box
        x_min, y_min, x_max, y_max = box
        center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        best_point = min(valid_points, key=lambda p: (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)
        return best_point
    # If no suitable point is found, return the center of the box as a fallback
    x_min, y_min, x_max, y_max = box
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)

# Function to check if a mask covers any of the points in face_points_avoid
def mask_covers_points(mask, points):
    for point in points:
        x, y = map(int, point)  # Ensure x and y are integers
        if mask[y, x] > 0.5:  # Assuming binary mask with threshold 0.5
            return True
    return False

def scale_boxes(boxes, scale_factor):
    """
    Scale the bounding boxes back to the original image size.
    """
    boxes = boxes / scale_factor
    return boxes

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
        image = image.resize((new_width, new_height), Image.ANTIALIAS)
        print(f"Image resized to: {new_width}x{new_height}")
    return image, scale_factor

def run_grounding_sam_demo_negative(config_file, grounded_checkpoint, sam_version, sam_checkpoint, sam_hq_checkpoint, use_sam_hq, image_path, text_prompt, output_dir, box_threshold, text_threshold, device, character_prompt="", save_path="", just_measuring=False, box_to_use_num=None, box_coordinates={}, predictor=None):
    detection_status = "None"

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    original_image_pil, original_image = load_image(image_path)
    original_size = original_image_pil.size
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    original_image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    cropped_image = original_image

    crop_x = 0
    crop_y = 0

    cropped_area = 0

    char_save_path = save_path.split("_")[0]

    if character_prompt != "" and not box_coordinates:

        all_box_coordinates = {}
        
        char_boxes_filt, char_pred_phrases = get_grounding_output(
            model, original_image, character_prompt, 0.1, 0.01, device=device
        )

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
        
        # Limit to top 4 options
        top_char_boxes = sorted_char_boxes[:4]
        top_char_pred_phrases = sorted_char_pred_phrases[:4]

        for i, (box, label) in enumerate(zip(top_char_boxes, top_char_pred_phrases)):
            score = float(label.split('(')[-1].strip(')'))  # Extract the score
            # Scale and transform box coordinates to PIL crop format (left, upper, right, lower)
            scaled_box = box * torch.Tensor([W, H, W, H])
            scaled_box[:2] -= scaled_box[2:] / 2
            scaled_box[2:] += scaled_box[:2]
            left, upper, right, lower = scaled_box.cpu().numpy().tolist()

            # Save coordinates
            all_box_coordinates[f"box_{i + 1}"] = {"left": left, "upper": upper, "right": right, "lower": lower}
            #print("running box coordinates: ", all_box_coordinates[f"box_{i + 1}"])

            # Crop the image to the character box
            cropped_image = original_image_pil.crop((left, upper, right, lower))
            #print("Setting the cropped options...")
            file_name = f"cropped_img_{char_save_path}_option_{i + 1}.jpg"
            cropped_image.save(os.path.join(output_dir, file_name))

            crop_x, crop_y = left, upper

            # Save a box to the base cropped image
            if i == 0:
                special_file_name = f"cropped_img_{char_save_path}.jpg"
                cropped_image.save(os.path.join(output_dir, special_file_name))

            box_found = True

        if not box_found:
            print("No character box found")
            return "None"
        else:
            print("Returning all box coordinates: ", all_box_coordinates)
            return all_box_coordinates
        
    # If box coordinates exist, then create a crop of the base image using the coordinates
    if box_coordinates:
        crop_x = box_coordinates["left"]
        crop_y = box_coordinates["upper"]
        crop_width = box_coordinates["right"] - box_coordinates["left"]
        crop_height = box_coordinates["lower"] - box_coordinates["upper"]
        cropped_pil = original_image_pil.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

        # Display the cropped image
        cropped_pil = cropped_pil.convert("RGB")

        # Resize the cropped image if necessary because the model needs a specific size
        cropped_pil, scale_factor = resize_image(cropped_pil, min_size=300)

        # Convert to tensor if needed
        cropped_img = torch.from_numpy(np.array(cropped_pil).transpose(2, 0, 1)).float().div(255.0)

        # Show the tensor image for verification
        #show_tensor_image(cropped_img)

    #Calculate the area of the cropped image
    cropped_area = cropped_pil.size[0] * cropped_pil.size[1]

    if text_prompt == "":
        return "Done"

    # run grounding dino model
    try:
        boxes_filt, pred_phrases = get_grounding_output(
            model, cropped_img, text_prompt, box_threshold, text_threshold, device=device
        )

        # Scale the bounding boxes back to the original image size
        boxes_filt = scale_boxes(boxes_filt, scale_factor)

        torch.cuda.empty_cache()

    except:
        print("Error getting grounding output")
        return "None"

    #predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint_path))
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
    other_boxes = []
    hand_boxes = []
    arm_boxes = []
    shirt_boxes = []
    clothes_boxes = []
    negative_boxes = []

    #print("Crop x and crop y are", crop_x, crop_y)

    # Convert the PIL image to a NumPy array
    image_np = np.array(cropped_pil)

    # Convert the NumPy array to RGB if it's not already (PIL images are in RGB by default)
    if image_np.shape[2] == 4:  # Check if the image has an alpha channel
        image_np = image_np[:, :, :3]  # Drop the alpha channel

    # Ensure the image is in RGB format
    #image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    image = image_np

    words_from_prompt = [word.strip() for word in text_prompt.split(".") if word.strip()]
    print("Words from prompt are", words_from_prompt)

    size = cropped_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    
    threshold_area = 0.5 * cropped_area
    #print("Box area threshold is", threshold_area)

    list_of_other_labels = []

    for box, label in zip(boxes_filt, pred_phrases):
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        box_area = box_width * box_height
        box_area = box_area.item()  # Convert tensor to float
        #print(f"Box {label} area is {box_area}")

        # Debugging: print the label to understand its format
        #print(f"Raw label: {label}")

        # Handle cases where label might not follow expected format
        try:
            score = float(label.split('(')[-1].strip(')'))
            label_text = label.split('(')[0].strip()
        except ValueError:
            score = 0.0
            label_text = label.strip()

        info_obj = {'box': box, 'label': label_text, 'score': score}
        #print(f"Info object is {info_obj}")
        if 'face' in label or 'chin' in label:
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
        elif 'main character' in label:
            main_character_boxes.append(info_obj)
            score = float(label.split('(')[-1].strip(')'))  # Extracting the confidence score from the label
            main_character_scores.append(score)

        elif 'hand' in label:
            negative_boxes.append(info_obj)
        elif 'arm' in label:
            negative_boxes.append(info_obj)
        elif 'on head' in label:
            negative_boxes.append(info_obj)
        elif 'clothes' in label or 'shirt' in label or "body" in label or "human" in label:
            negative_boxes.append(info_obj)
        elif label.split('(')[0] in words_from_prompt:

            ##If label is not already in the list of other labels, then add it
            if label not in list_of_other_labels:
                list_of_other_labels.append(label)

            print("Label is", label)
            negative_boxes.append(info_obj)
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
            if closest_hair_pair is not None and closest_distance < 250:
                the_hair_box = [closest_hair_pair['box']]
                focal_box = [closest_hair_pair['box']]
                focal_box_label = closest_hair_pair['label']  # Optionally keep track of the label
                #print("Closest hair box:", closest_hair_pair['box'])
            else:
                #No Hair box, so we're going to skip the rest of the code
                print("No hair box found in the zone")
                #return "None"
        elif negative_boxes:
            detection_status = "negative_found"
            
            #print("negative was found")
            for pair in negative_boxes:
                #print("Negative box:", box)
                box = pair['box']
                label = pair['label']
                box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                #print(f"Center of {label} box: {box_center}")


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
        print("hair was found")
    elif negative_boxes:
        detection_status = "negative_found"
        print("negative was found")
    else:
        detection_status = "None"
        ##If close_boxes is empty, then we're going to skip the rest of the code
        return detection_status
    if negative_boxes:
        detection_status = "negative_found"
            
        print("negative was found")
        for pair in negative_boxes:
            #print("Negative box:", box)
            box = pair['box']
            label = pair['label']
            box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            #print(f"Center of {label} box: {box_center}")
        

    all_selected_pairs = []
    close_boxes = []
    excluded_boxes = []

    # Process focal boxes and selected pairs
    if focal_box is not None and detection_status != 'hair_found' and not main_character_boxes:
        selected_eye_pairs = filter_and_limit_boxes(eye_boxes, chosen_face_box, W, 2)
        selected_mouth_pairs = filter_and_limit_boxes(mouth_boxes, chosen_face_box, W, 1)
        selected_ear_pairs = filter_and_limit_boxes(ear_boxes, chosen_face_box, W, 2)
        selected_hair_pairs = filter_and_limit_boxes(hair_boxes, chosen_face_box, W, 1)

        excluded_boxes = [pair['box'] for pair in face_boxes + hair_boxes + ear_boxes]

        all_selected_pairs = selected_eye_pairs + selected_mouth_pairs + selected_ear_pairs + selected_hair_pairs + negative_boxes

        selected_eye_boxes = [pair['box'] for pair in selected_eye_pairs]
        if len(selected_eye_boxes) >= 2:
            eye_unevenness = measure_eye_unevenness(selected_eye_boxes)

        selected_mouth_boxes = [pair['box'] for pair in selected_mouth_pairs]
        selected_ear_boxes = [pair['box'] for pair in selected_ear_pairs]
        selected_hair_boxes = [pair['box'] for pair in selected_hair_pairs]

        close_boxes = [pair['box'] for pair in selected_eye_pairs + selected_mouth_pairs + selected_ear_pairs + selected_hair_pairs + negative_boxes]
    elif the_hair_box:
        close_boxes = the_hair_box
        all_selected_pairs = [closest_hair_pair]

    if focal_box is not None and the_hair_box is None:
        close_boxes += focal_box
        focal_box_pair = {'box': focal_box[0], 'label': focal_box_label}
        all_selected_pairs.append(focal_box_pair)

    accepted_labels = [label for label in words_from_prompt if label not in ["hair", "face", "ear"]]

    # Ensure we have boxes for hands, arms, and shirts
    hand_arm_shirt_boxes = [pair['box'] for pair in all_selected_pairs if 'hand' in pair['label'] or 'arm' in pair['label'] or 'shirt' in pair['label'] or 'on head' in pair['label']]
    # Add head, face, and neck to excluded regions
    #print(f"Excluded boxes: {excluded_boxes}")

    # Find positive points within each hand, arm, and shirt box, avoiding excluded regions
    positive_points = []
    for box in hand_arm_shirt_boxes:
        box_np = box.numpy() if isinstance(box, torch.Tensor) else box  # Ensure the box is in numpy array format
        best_point = find_best_point(box_np, excluded_boxes)
        print('Positive point', best_point)
        positive_points.append(best_point)

    # Draw output image and plot positive points
    plt.figure(figsize=(10, 10))
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    plt.imshow(image)

    #Get points to aviod for the mask
    face_points_avoid = []

    # Visualize excluded boxes for debugging
    for ex_box in excluded_boxes:
        #print("\nyoyo")
        x_min, y_min, x_max, y_max = ex_box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')  # Red for excluded regions
        plt.gca().add_patch(rect)

        box_np = ex_box.numpy() if isinstance(ex_box, torch.Tensor) else ex_box
        best_avoid_point = find_best_point(box_np, hand_arm_shirt_boxes)
        print('Negative point', best_avoid_point)
        face_points_avoid.append(best_avoid_point)

    plt.savefig(
        os.path.join(output_dir, f"grounded_sam_output_excluded_negative.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    # Visualize candidate points and chosen points
    for box in hand_arm_shirt_boxes:
        # Generate candidate points within the box
        candidate_points = generate_candidate_points(box)
        
        # Plot candidate points (blue)
        for point in candidate_points:
            plt.plot(point[0], point[1], 'bo', markersize=3)  # Blue points for candidate points

        for point in face_points_avoid:
            plt.plot(point[0], point[1], 'yo', markersize=20)  # Yellow points for face points to avoid

        # Draw green box around each hand, arm, and shirt box
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='g', facecolor='none')  # Green for positive boxes
        plt.gca().add_patch(rect)

    # Plot chosen points (green)
    for point in positive_points:
        cx, cy = point
        plt.plot(cx, cy, 'go', markersize=25)  # Green points for chosen points

    # Save the visualization of potential points
    plt.savefig(
        os.path.join(output_dir, f"grounded_sam_output_potentials_negative.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    # Convert boxes to numpy arrays and visualize them with labels
    for pair in all_selected_pairs:
        box = pair['box'].numpy() if isinstance(pair['box'], torch.Tensor) else pair['box']
        label = pair['label']
        show_box(box, plt.gca(), label)

    save_label = save_path
    if just_measuring:
        save_label += "_inpainted"

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, f"grounded_sam_output_{save_label}_negative.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    if hand_arm_shirt_boxes:

        # Convert hand_arm_shirt_boxes to the correct format for SAM
        hand_arm_shirt_boxes_tensor = torch.stack([torch.tensor(box) for box in hand_arm_shirt_boxes])
    else:
        print("no negatives found")
        return

    if excluded_boxes:
        excluded_boxes_tensor = torch.stack([torch.tensor(box) for box in excluded_boxes])
    else:
        excluded_boxes_tensor = torch.tensor([])

    # Convert positive points to the correct format for SAM
    positive_points_np = np.array(positive_points).reshape(-1, 1, 2)

    # Define point_labels for SAM (1 for positive)
    point_labels = np.ones((positive_points_np.shape[0],), dtype=int).reshape(-1, 1)

    negative_point_np = np.array(face_points_avoid).reshape(-1, 1, 2)
    # Define point_labels for SAM (0 for negative)
    negative_point_labels = np.full((negative_point_np.shape[0],), 0, dtype=int).reshape(-1, 1)

    # Combine positive and negative points and labels
    all_points_np = np.vstack((positive_points_np, negative_point_np))
    all_point_labels = np.vstack((point_labels, negative_point_labels))

    # Ensure the points and labels are tensors
    point_coords_tensor = torch.from_numpy(all_points_np).to(device)
    point_labels_tensor = torch.from_numpy(all_point_labels).to(device)

    # Duplicate boxes to match the number of points if necessary
    num_boxes = hand_arm_shirt_boxes_tensor.size(0)
    num_points = point_coords_tensor.size(0)

    if num_boxes < num_points:
        hand_arm_shirt_boxes_tensor = hand_arm_shirt_boxes_tensor.repeat((num_points // num_boxes) + 1, 1)[:num_points]

    torch.cuda.empty_cache()
    gc.collect()

    if 1 == 1:

        # Convert boxes to numpy arrays for SAM
        batch_boxes = hand_arm_shirt_boxes_tensor.cpu().numpy()

        print("batch boxes length", len(batch_boxes))

        # Set the image for the predictor
        img_batch = [image]# Print image shape for debugging
        print(f"Image shape: {image.shape}")

        # Prepare inputs for predict_batch
        point_coords_batch = [point_coords_tensor.cpu().numpy()]
        point_labels_batch = [point_labels_tensor.cpu().numpy()]
        box_batch = [batch_boxes]

        # Set the image for the predictor
        predictor.set_image_batch(img_batch)

        try:
            # Pass these to the model
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                masks_batch, scores_batch, logits_batch = predictor.predict_batch(
                    point_coords_batch=point_coords_batch,
                    point_labels_batch=point_labels_batch,
                    box_batch=box_batch,
                    multimask_output=True
                )
        except Exception as e:
            print("No masks found:", e)
            return "None"

        print("Processing prediction results...")
        masks_to_use = []

        batch_count = 0

        # Iterate over each set of masks and scores
        for masks_set, scores_set in zip(masks_batch, scores_batch):
            scores_tensor = torch.tensor(scores_set)
            batch_count += 1
            set_count = 0
            # Print sizes for debugging
            print(f"masks_set size: {len(masks_set)}, scores_tensor size: {scores_tensor.size()}")

            best_mask = None
            best_score = -1

            for mask, score_set in zip(masks_set, scores_tensor):
                set_count += 1
                mask_count = 0
                
                # Track best score for each set
                best_score = -1

                # Each mask might contain multiple masks, iterate over them
                for i in range(mask.shape[0]):
                    mask_count += 1
                    mask_np = mask[i].astype(np.float32)
                    if mask_np.max() > 1:
                        mask_np /= 255.0

                    # Print mask shape for debugging
                    print(f"Mask {i} shape: {mask_np.shape}")

                    # If mask has extra dimensions, handle it
                    if mask_np.shape[0] == 3:
                        mask_np = mask_np.transpose(1, 2, 0)  # Transpose to (465, 586, 3)
                        mask_np = mask_np[:, :, 0]  # Assuming the first channel is the mask we need

                    # Reshape mask if needed
                    if mask_np.size == image.shape[0] * image.shape[1]:
                        mask_np = mask_np.reshape(image.shape[0], image.shape[1])
                    else:
                        print(f"Error reshaping mask: expected {image.shape[0] * image.shape[1]} but got {mask_np.size}")
                        continue

                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    try:
                        show_mask(mask_np, plt.gca())
                    except ValueError as ve:
                        print(f"Error reshaping mask: {ve}")
                        plt.close()  # Close the figure to prevent too many open figures
                        continue
                    plt.title(f"Batch{batch_count}, Set{set_count}, Mask{mask_count}, Score: {score_set[i].item():.3f}", fontsize=18)
                    plt.axis('off')
                    #plt.show()
                    plt.close()  # Close the figure to prevent too many open figures

                # Select the best mask based on the highest score
                highest_score_index = torch.argmax(score_set).item()
                if highest_score_index < mask.shape[0] and score_set[highest_score_index].item() > best_score:
                    best_score = score_set[highest_score_index].item()
                    best_mask = mask[highest_score_index]
                    print(f"Best score is {best_score} for set {set_count}")

                if best_mask is not None:
                    best_mask = clean_mask(best_mask)  # Clean the best mask
                    masks_to_use.append(torch.tensor(best_mask).to(device))
                else:
                    print("No valid mask found for this set")

    # Dilate each mask to add padding
    dilation_amt = 25  # Adjust this as needed
    if len(excluded_boxes_tensor) > 0:
        dilation_amt = 2

    padded_masks = []
    if 'Main Character' not in text_prompt:
        for mask in masks_to_use:
            mask_np = mask.cpu().numpy()  # Convert the mask to a numpy array
            print(f"Dilating mask with shape: {mask_np.shape}")  # Debugging print
            dilated_mask, _ = dilate_mask(mask_np, dilation_amt)
            padded_masks.append(torch.from_numpy(np.array(dilated_mask)).unsqueeze(0))
    else:
        for j, mask in enumerate(masks_to_use):
            mask_np = mask.cpu().numpy()  # Convert the mask to a numpy array

            if "emotion" not in image_path:
                dilated_mask, _ = dilate_mask(mask_np, 5)
                padded_masks.append(torch.from_numpy(np.array(dilated_mask)).unsqueeze(0))
            else:
                # Do not dilate the mask but add it to padded_masks
                padded_masks.append(torch.from_numpy(np.array(mask_np)).unsqueeze(0))

    ##IF there are no masks, return none
    if len(padded_masks) == 0:
        return None

    area_dict = {}
    binary_padded_masks = []
    for i, (mask, pair) in enumerate(zip(padded_masks, all_selected_pairs)):
        binary_mask = mask > 0.5  # Thresholding
        binary_padded_masks.append(binary_mask)

        # Calculate the area of the current element
        area = torch.sum(binary_mask).item()
        label = pair['label']
        area_dict[label] = area

    # Combine and fill gaps in masks
    filled_combined_mask = combine_and_fill_gaps(padded_masks)

    # Draw output image with masks
    plt.figure(figsize=(10, 10))
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    plt.imshow(image)

    center_points = []
    for mask in padded_masks:
        mask_np = mask.cpu().numpy()[0]
        mask_np = mask_np.astype(np.float32)
        if mask_np.max() > 1:
            mask_np /= 255.0

        # Find the moments of the binary mask
        moments = cv2.moments(mask_np)
        # Overlay negative points in blue
        for point in face_points_avoid:
            plt.plot(point[0], point[1], 'bo', markersize=2) 

        # Calculate the central point coordinates
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            center_points.append((cx, cy))
            #plt.plot(cx, cy, 'ro', markersize=5)
        else:
            center_points.append(None)

        show_mask(mask_np, plt.gca(), random_color=True)

    # Add a dummy point for completeness
    center_points.append((20, 20))
    #print("Center points are", center_points)

    # Convert to lists from tuples
    #center_points = [list(coord) for coord in center_points]

    # Save the output image
    save_label = save_path
    if just_measuring:
        save_label += "_inpainted"

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, f"grounded_sam_output_{save_label}_negative.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    plt.close()

    save_mask_data(output_dir, filled_combined_mask, boxes_filt, pred_phrases, image_path, save_path, crop_x, crop_y, original_size, just_measuring)

    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleaned - negative")

    return positive_points

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
    parser.add_argument("--box_to_use_num", type=int, default=None, help="the number of the character box to use")
    parser.add_argument("--predictor", type=str, default=None, help="predictor")
    args = parser.parse_args()

    # Call the new function with the parsed arguments
    run_grounding_sam_demo_negative(
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
        args.box_to_use_num,
        args.predictor
    )
