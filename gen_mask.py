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
    print(f"Vertical Offset: {vertical_offset}")
    print(f"Width Difference: {width_difference}")
    print(f"Height Difference: {height_difference}")
    print(f"Combined Unevenness Score: {combined_unevenness_score}")
    
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
    assert mask_np.shape[1] in [1448, 1792], f"Mask width is {mask_np.shape[1]}, expected 1454 or 1792"

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

def is_close(box1, box2, threshold=1.0):
    # Calculate the distance between the centers of two boxes
    center_box1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center_box2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    distance = ((center_box1[0] - center_box2[0]) ** 2 + (center_box1[1] - center_box2[1]) ** 2) ** 0.5
    return distance < threshold

def filter_face_boxes(face_boxes, hair_boxes, eye_boxes, proximity_threshold=1.0):
    filtered_face_boxes = []
    for face_box in face_boxes:
        close_to_hair = any(is_close(face_box, hair_box, proximity_threshold) for hair_box in hair_boxes)
        close_to_eye = any(is_close(face_box, eye_box, proximity_threshold) for eye_box in eye_boxes)
        if close_to_hair and close_to_eye:
            filtered_face_boxes.append(face_box)
    return filtered_face_boxes

def select_highest_confidence_face_box(boxes, scores):
    """
    Selects the box with the highest confidence score for the face.

    :param boxes: List or array of bounding boxes.
    :param scores: List or array of confidence scores corresponding to each box.
    :return: The box with the highest confidence score.
    """
    max_score_index = scores.argmax()
    return boxes[max_score_index]

def select_centermost_face_box(boxes, image_size):
    """
    Selects the face box that is closest to the center of the image.

    :param boxes: List or array of bounding boxes.
    :param image_size: Size of the image as a tuple (width, height).
    :return: The face box closest to the center of the image.
    """
    y = image_size[1]
    y_mid = y / 4
    image_center = (image_size[0] / 2, y_mid * 1.5)
    min_distance = float('inf')
    centermost_box = None

    for box in boxes:
        box_center_point = box_center(box)
        distance = np.sqrt((box_center_point[0] - image_center[0])**2 + (box_center_point[1] - image_center[1])**2)

        if distance < min_distance:
            min_distance = distance
            centermost_box = box

    return centermost_box


def box_center(box):
    """
    Calculate the center of a bounding box.

    :param box: Bounding box in format [x1, y1, x2, y2].
    :return: Center of the box (x_center, y_center).
    """
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

def filter_and_limit_boxes(boxes, face_box, max_count):
    """
    Filters and limits the number of boxes based on proximity to the face box.

    :param boxes: List of bounding boxes.
    :param face_box: The selected face box.
    :param max_count: Maximum number of boxes to select.
    :return: Filtered and limited list of boxes.
    """
    distances = [distance_between_boxes(box, face_box) for box in boxes]
    sorted_boxes = sorted(zip(distances, boxes), key=lambda x: x[0])
    return [box for _, box in sorted_boxes][:max_count]

def run_grounding_sam_demo(config_file, grounded_checkpoint, sam_version, sam_checkpoint, sam_hq_checkpoint, use_sam_hq, image_path, text_prompt, output_dir, box_threshold, text_threshold, device, save_path=""):
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
        if 'face' in label:
            face_boxes.append(box)
            score = float(label.split('(')[-1].strip(')'))
            face_scores.append(score)
            print(f"Face box: {box}, score: {score}")
        elif 'eye' in label:
            eye_boxes.append(box)
        elif 'mouth' in label:
            mouth_boxes.append(box)
        elif 'ear' in label:
            ear_boxes.append(box)
        elif 'hair' in label:
            hair_boxes.append(box)
        elif 'main character' in label:
            main_character_boxes.append(box)
            score = float(label.split('(')[-1].strip(')'))  # Extracting the confidence score from the label
            main_character_scores.append(score)
        else:
            other_boxes.append(box)

    # Select the highest confidence face box
    if face_boxes:
        face_scores = np.array(face_scores)
        #highest_confidence_face_box = select_highest_confidence_face_box(face_boxes, face_scores)

        highest_confidence_face_box = select_centermost_face_box(face_boxes, (W, H))

        selected_face_boxes = [highest_confidence_face_box] if highest_confidence_face_box is not None else []


        # Debugging: Print the selected face box
        print("Selected face box:", highest_confidence_face_box)

        # Filter boxes near the selected face box with an increased threshold
        #close_boxes = filter_boxes_near_face_box(boxes_filt, highest_confidence_face_box, 200)
        selected_eye_boxes = filter_and_limit_boxes(eye_boxes, highest_confidence_face_box, 2)
        selected_mouth_boxes = filter_and_limit_boxes(mouth_boxes, highest_confidence_face_box, 1)
        selected_ear_boxes = filter_and_limit_boxes(ear_boxes, highest_confidence_face_box, 2)
        selected_hair_boxes = filter_and_limit_boxes(hair_boxes, highest_confidence_face_box, 1)

        #Measure the unevenness of the eyes
        #make sure there are at least two eye boxes
        if len(selected_eye_boxes) >= 2:
            eye_unevenness = measure_eye_unevenness(selected_eye_boxes)

        close_boxes = selected_eye_boxes + selected_mouth_boxes + selected_ear_boxes + selected_hair_boxes + selected_face_boxes
        # Debugging: Print the number of filtered boxes
        #print("Number of boxes close to the face box:", len(close_boxes))
    elif main_character_boxes:
        # Convert the list of scores to a numpy array for easier manipulation
        main_character_scores = np.array(main_character_scores)
        
        # Find the index of the highest confidence score
        highest_confidence_index = np.argmax(main_character_scores)
        
        # Select the box with the highest confidence score
        highest_confidence_box = main_character_boxes[highest_confidence_index]
        close_boxes = [highest_confidence_box]
    
    else:
        close_boxes = other_boxes

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
            dilated_mask, _ = dilate_mask(mask_np, 5)
            padded_masks.append(torch.from_numpy(np.array(dilated_mask)).unsqueeze(0))

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

    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "grounded_sam_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    save_mask_data(output_dir, filled_combined_mask, boxes_filt, pred_phrases, image_path, save_path)



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
