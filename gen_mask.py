import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_closing

import cv2

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt



def dilate_mask(mask, dilation_amt):
    # Create the dilation kernel
    x, y = np.meshgrid(np.arange(dilation_amt), np.arange(dilation_amt))
    center = dilation_amt // 2
    dilation_kernel = ((x - center)**2 + (y - center)**2 <= center**2).astype(np.uint8)
    dilated_binary_img = binary_dilation(mask, dilation_kernel)
    dilated_mask = Image.fromarray(dilated_binary_img.astype(np.uint8) * 255)
    return dilated_mask, dilated_binary_img

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


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
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


def save_mask_data(output_dir, mask_tensor, box_list, label_list, image_name, save_path):
    value = 0  # 0 for background
    background_alpha = 0.01

    # Assume mask_tensor is a PyTorch tensor of shape [1, height, width]
    mask_np = mask_tensor.cpu().numpy().squeeze()  # Convert to numpy and remove the first dimension

    # Check if image has width of 1792 or 1454
    #assert mask_np.shape[1] in [1448, 1792], f"Mask width is {mask_np.shape[1]}, expected 1454 or 1792"

    # Create an RGBA image with the same dimensions as the mask
    mask_img = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)

    # Set the mask color and alpha
    mask_color = np.array([225, 225, 225, 255], dtype=np.uint8)  # White color mask with full opacity

    # Apply the mask color where mask_np is True
    mask_img[mask_np] = mask_color

    # Save the mask image
    path_name = "mask"+save_path+".png"
    mask_image_path = os.path.join(output_dir, path_name)
    Image.fromarray(mask_img).save(mask_image_path, format='PNG')

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

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
    proximity_threshold = 0.05 * image_width

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

def run_grounding_sam_demo(config_file, grounded_checkpoint, sam_version, sam_checkpoint, sam_hq_checkpoint, use_sam_hq, image_path, text_prompt, output_dir, box_threshold, text_threshold, device, save_path=""):
    detection_status = "None"

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

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

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
        print("Using SAM-HQ")
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    for box, label in zip(boxes_filt, pred_phrases):
        score = float(label.split('(')[-1].strip(')'))  # Extract the score
        info_obj = {'box': box, 'label': label, 'score': score}
        if 'face' in label:
            face_boxes.append(info_obj)
            face_scores.append(score)
            print(f"Face box: {box}, score: {score}")
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
        else:
            other_boxes.append(info_obj)

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
            else:
                #No Hair box, so we're going to skip the rest of the code
                print("No hair box found in the zone")
                return "None"
        else:
            #No hair box, so we're going to skip the rest of the code
            print("No hair box found")
            return "None"
        
    focal_box = None
    focal_box_label = None
    # Select the highest confidence face box
    if face_boxes:
        detection_status = "face_found"
        #highest_confidence_face_box = select_highest_confidence_face_box(face_boxes, face_scores)
        """
        chosen_pair = select_centermost_face_box(face_boxes, (W, H))
        if chosen_pair is not None:
            chosen_face_box = chosen_pair['box']
            focal_box = [chosen_face_box]
            focal_box_label = chosen_pair['label']
            print("Chosen face box:", chosen_face_box)
        else:
            print("Going with the highest confidence face box")
            chosen_pair = select_highest_confidence_face_box(face_boxes)
            chosen_face_box = chosen_pair['box']
            focal_box = [chosen_face_box]
            focal_box_label = chosen_pair['label']
        """
        chosen_pair = face_confidence_score(face_boxes, (W, H), center_weight=0.65, confidence_weight=0.35)
        if chosen_pair is not None:
            chosen_face_box = chosen_pair['box']
            focal_box = [chosen_face_box]
            focal_box_label = chosen_pair['label']
            print("Chosen face box based on combined score:", chosen_face_box)
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
    else:
        detection_status = "None"
        return detection_status
        close_boxes = []

    all_selected_pairs = []
    close_boxes = []
    if focal_box is not None and not main_character_boxes:
        # Filter boxes near the selected face box with an increased threshold
        #close_boxes = filter_boxes_near_face_box(boxes_filt, highest_confidence_face_box, 200)
        # Assuming each *_boxes variable is now a list of dictionaries with 'box' and 'label' (and optionally 'score')
        selected_eye_pairs = filter_and_limit_boxes(eye_boxes, chosen_face_box, W, 2)
        selected_mouth_pairs = filter_and_limit_boxes(mouth_boxes, chosen_face_box, W, 1)
        selected_ear_pairs = filter_and_limit_boxes(ear_boxes, chosen_face_box, W, 2)
        selected_hair_pairs = filter_and_limit_boxes(hair_boxes, chosen_face_box, W, 1)

        all_selected_pairs = selected_eye_pairs + selected_mouth_pairs + selected_ear_pairs + selected_hair_pairs


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

        
    if focal_box is not None:
        close_boxes += focal_box
        #focal_box_label = focal_box_label  # Example label, adjust based on your logic
        focal_box_pair = {'box': focal_box[0], 'label': focal_box_label}  # Ensure focal_box is in the expected format
        all_selected_pairs.append(focal_box_pair)
    
    boxes_filt = torch.stack(close_boxes)

    #print("Later boxes_filt:", boxes_filt)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
    
    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # Dilate each mask to add padding
    
    dilation_amt = 25  # Adjust this as needed
    
    padded_masks = []
    for mask in masks:
        mask_np = mask.cpu().numpy()[0]  # Convert the mask to a numpy array

        if 'Main Character' not in text_prompt:
            print("yoooooo")
            dilated_mask, _ = dilate_mask(mask_np, dilation_amt)
            padded_masks.append(torch.from_numpy(np.array(dilated_mask)).unsqueeze(0))
        else:
            if "emotion" not in image_path:
                dilated_mask, _ = dilate_mask(mask_np, 5)
                padded_masks.append(torch.from_numpy(np.array(dilated_mask)).unsqueeze(0))
            else:
                # Do not dilate the mask but add it to padded_masks
                padded_masks.append(torch.from_numpy(np.array(mask_np)).unsqueeze(0))
                #dilated_mask, _ = dilate_mask(mask_np, -10)
                #padded_masks.append(torch.from_numpy(np.array(dilated_mask)).unsqueeze(0))

    binary_padded_masks = []
    for mask in padded_masks:
        binary_mask = mask > 0.5  # Thresholding
        binary_padded_masks.append(binary_mask)

    #combined_masks_tensor = torch.stack(binary_padded_masks)

    filled_combined_mask = combine_and_fill_gaps(padded_masks)
    #print("Filled combined mask:", filled_combined_mask)

    # draw output image
    plt.figure(figsize=(10, 10))
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    plt.imshow(image)

    for mask in padded_masks:
        mask_np = mask.cpu().numpy()[0]
        mask_np = mask_np.astype(np.float32)
        if mask_np.max() > 1:
            mask_np /= 255.0

        show_mask(mask_np, plt.gca(), random_color=True)

    """
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    """
    for pair in all_selected_pairs:
        box = pair['box'].numpy()  # Ensure the box is converted to numpy array if it's a tensor
        label = pair['label']  # Use the label directly from the pair
        show_box(box, plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, f"cropped_{save_path}.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    save_mask_data(output_dir, filled_combined_mask, boxes_filt, pred_phrases, image_path, save_path)

    return detection_status



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

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--save_path", type=str, default="", help="ending to file")
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
        args.save_path
    )
