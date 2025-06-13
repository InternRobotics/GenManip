from dotenv import load_dotenv

load_dotenv()

import cv2
from datetime import datetime
from flask import Flask, request, jsonify
import json
import logging
import numpy as np
import os
import random
import requests
import traceback

from utils.gpt_utils import request_gpt, parse_gpt_response
from utils.prompt import (
    construct_gpt_prompt,
    construct_gpt_prompt_CtoF,
    construct_grab_point_prompt,
    construct_path_planning_prompt,
    construct_path_planning_prompt_P2P,
)
from utils.utils import save_dict_to_json, save_image
from utils.draw_utils import show_anns, draw_grid, get_selected_grid_label
from utils.mask_utils import nms

LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
SAM2_URL = "http://localhost:5002/generate_masks"


class Config:
    FIXED_RESIZE_DIMENSIONS = (800, 800)
    GRID_ROWS = 9
    GRID_COLS = 16
    GRID_ALPHA = 0.3
    MASK_SELECTION_LIMIT = 3
    ORIGINAL_IMAGE_HEIGHT = 720
    ORIGINAL_IMAGE_WIDTH = 1280


logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB limit


def generate_masks(image_rgb: np.ndarray):
    """Request mask from SAM2

    Args:
        image_rgb (np.ndarray): image with RGB order.

    Returns:
        list of dict: A list of dictionaries where each dictionary represents a mask.
    """
    try:
        _, img_encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        )
        files = {"image": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}
        response = requests.post(SAM2_URL, files=files)
        response.raise_for_status()
        masks = response.json().get("masks", [])
        logging.debug(f"Number of masks received: {len(masks)}")
        return masks

    except requests.RequestException as e:
        logging.error(f"Error calling SAM2 API: {e}")
        logging.debug(traceback.format_exc())
        return []


def SoM(image_rgb, masks, prompt_text, config):
    attemps = 0
    filtered_masks = nms(masks)
    filtered_masks = sorted(filtered_masks, key=lambda x: x["area"], reverse=True)
    annotated_image = show_anns(
        image_rgb.copy(), filtered_masks, borders=True, label_masks=True
    )
    save_image(
        annotated_image, os.path.join(config["OUTPUT_DIR"], "annotated_masks.jpg")
    )

    while True:
        identified_object = select_object(
            prompt_text, image_rgb, annotated_image, config
        )
        if identified_object["number"] != -1:
            break
        attemps += 1
        if attemps > Config.MASK_SELECTION_LIMIT:
            identified_object["number"] = 1
            identified_object["object_name"] = "default"
            logging.warning("Exceeded maximum retry attempts for mask selection")
            break

    selected_mask = None
    for idx, mask in enumerate(filtered_masks, 1):
        if idx == int(identified_object["number"]):
            selected_mask = mask
            break
    if not selected_mask:
        return {"error": f'Mask number {identified_object["number"]} not found.'}, 400
    return selected_mask, filtered_masks, identified_object


def select_object(instruction, original_image, annotated_image, config):
    """
    Select an object based on an instruction using GPT.

    Args:
        instruction (str): The instruction for selecting the object.
        original_image (np.ndarray): The original image.
        annotated_image (np.ndarray): The annotated image.

    Returns:
        dict: The identified object information.
    """
    prompt = construct_gpt_prompt(instruction)
    gpt_response = request_gpt(
        message=prompt,
        images=[original_image, annotated_image],
        local_image=True,
        model_name=config["model_name"],
    )
    js = {}
    js["prompt"] = prompt
    js["response"] = gpt_response
    save_dict_to_json(js, os.path.join(config["OUTPUT_DIR"], "object_selection.json"))
    logging.debug(f"GPT Object Identification Response: {gpt_response}")

    identified_object = parse_gpt_response(gpt_response)
    logging.info(f"Identified Object: {identified_object}")
    return identified_object


def crop_and_resize(image_rgb, selected_mask, resize_dimensions):
    """
    Crop an image to the selected mask and resize it.

    Args:
        image_rgb (np.ndarray): The input RGB image.
        selected_mask (dict): The selected mask containing the segmentation.
        resize_dimensions (tuple): The dimensions to resize the cropped image to.

    Returns:
        tuple: The resized image, resized mask, and bounding box of the cropped area.
    """
    segmentation = np.array(selected_mask["segmentation"], dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(segmentation)
    cropped_image = image_rgb[y : y + h, x : x + w]
    cropped_mask = segmentation[y : y + h, x : x + w]
    resized_image = cv2.resize(cropped_image, resize_dimensions)
    resized_mask = cv2.resize(
        cropped_mask, resize_dimensions, interpolation=cv2.INTER_NEAREST
    )
    return resized_image, resized_mask, (x, y, w, h)


def CtoF_SoM(resized_image, resized_mask, prompt_text, config):
    """
    Generate and select a mask using the provided image and mask.

    Args:
        resized_image (np.ndarray): The resized image.
        resized_mask (np.ndarray): The resized mask.
        prompt_text (str): The prompt text for identifying the object.

    Returns:
        np.ndarray: The selected mask.
    """
    masks_CtoF = generate_masks(resized_image)
    if len(masks_CtoF) != 0:
        filtered_masks_CtoF = nms(masks_CtoF, CtoF=True)
        filtered_masks_CtoF = sorted(
            filtered_masks_CtoF, key=lambda x: x["area"], reverse=True
        )
        annotated_image_CtoF = show_anns(
            resized_image.copy(), filtered_masks_CtoF, borders=True, label_masks=True
        )
        save_image(
            annotated_image_CtoF,
            os.path.join(config["OUTPUT_DIR"], "annotated_masks_CtoF.jpg"),
        )
        identified_object_CtoF = select_object_CtoF(
            prompt_text, resized_image, annotated_image_CtoF, config
        )
        if identified_object_CtoF["number"] == -1:
            identified_object_CtoF["number"] = 1
            identified_object_CtoF["part_name"] = "default"
        object_number_CtoF = identified_object_CtoF.get("number")
        object_name_CtoF = identified_object_CtoF.get("part_name")
        if object_name_CtoF and object_name_CtoF.lower() != "not_found":
            logging.info(
                f"Selected Part - Mask Number: {object_number_CtoF}, Part Name: {object_name_CtoF}"
            )
        else:
            raise ValueError("No suitable part identified.")

        selected_mask = None
        for idx, mask in enumerate(filtered_masks_CtoF, 1):
            if idx == int(object_number_CtoF):
                selected_mask = mask
                break

        if not selected_mask:
            raise ValueError(f"Mask number {object_number_CtoF} not found.")

        selected_mask = np.array(selected_mask["segmentation"], dtype=np.uint8).astype(
            bool
        )
        return selected_mask

    return resized_mask


def select_object_CtoF(instruction, original_image, annotated_image, config):
    """
    Select an object based on an instruction using GPT with a 'CtoF' prompt construction.

    Args:
        instruction (str): The instruction for selecting the object.
        original_image (np.ndarray): The original image.
        annotated_image (np.ndarray): The annotated image.

    Returns:
        dict: The identified object information.
    """
    prompt = construct_gpt_prompt_CtoF(instruction)
    gpt_response = request_gpt(
        message=prompt,
        images=[original_image, annotated_image],
        local_image=True,
        model_name=config["model_name"],
    )
    js = {}
    js["prompt"] = prompt
    js["response"] = gpt_response
    save_dict_to_json(
        js, os.path.join(config["OUTPUT_DIR"], "object_selection_CtoF.json")
    )
    logging.debug(f"GPT Object Identification Response: {gpt_response}")

    identified_object = parse_gpt_response(gpt_response)
    logging.info(f"Identified Object CtoF: {identified_object}")
    return identified_object


def sample_and_annotate_points(
    resized_image, resized_mask, sample_size, bbox, resize_dimensions
):
    """
    Sample points within the mask and annotate them on the image.

    Args:
        resized_image (np.ndarray): The resized image.
        resized_mask (np.ndarray): The resized mask.
        sample_size (int): The number of points to sample within the mask.
        bbox (tuple): The bounding box of the cropped area.
        resize_dimensions (tuple): The dimensions to resize the cropped image to.

    Returns:
        tuple: The annotated image and the list of labeled points.
    """
    x, y, w, h = bbox

    mask_indices = np.where(resized_mask == 1)
    points = list(zip(mask_indices[1], mask_indices[0]))  # (x, y)

    if len(points) < sample_size:
        sample_points = points
    else:
        sample_points = random.sample(points, sample_size)

    labeled_points = []
    for idx, (px, py) in enumerate(sample_points, 1):
        original_color = resized_image[py, px]
        inverse_color = (
            255 - int(original_color[0]),
            255 - int(original_color[1]),
            255 - int(original_color[2]),
        )

        cv2.circle(resized_image, (px, py), 10, inverse_color, -1)
        cv2.putText(
            resized_image,
            str(idx),
            (px + 5, py - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            inverse_color,
            1,
        )

        original_px = int(px * (w / float(resize_dimensions[0])) + x)
        original_py = int(py * (h / float(resize_dimensions[1])) + y)
        labeled_points.append((str(idx), (original_px, original_py)))
    labeled_points = [point[1] for point in labeled_points]

    return resized_image, labeled_points


def select_grab_point(object_name, sample_points, labeled_image, config):
    """
    Select a grab point for a specified object based on sample points using GPT.

    Args:
        object_name (str): The name of the object for which the grab point is to be selected.
        sample_points (list): A list of sample points to consider for selecting the grab point.
        labeled_image (np.ndarray): The labeled image containing the object.

    Returns:
        dict: The selected grab point information, including the point and the reason for selection.
    """
    prompt = construct_grab_point_prompt(object_name, sample_points)
    gpt_response = request_gpt(
        message=prompt,
        images=labeled_image,
        local_image=True,
        model_name=config["model_name"],
    )
    js = {}
    js["prompt"] = prompt
    js["response"] = gpt_response
    save_dict_to_json(js, os.path.join(config["OUTPUT_DIR"], "grab_point.json"))
    logging.debug(f"GPT Grab Point Selection Response: {gpt_response}")

    grab_point_data = parse_gpt_response(gpt_response)
    logging.info(
        f"Selected Grab Point: {grab_point_data.get('selected_point')}, Reason: {grab_point_data.get('reason')}"
    )
    selected_point_num = grab_point_data.get("selected_point")
    reason = grab_point_data.get("reason")

    if not selected_point_num or not (1 <= selected_point_num <= len(sample_points)):
        return {"error": "Invalid grab point selected."}, 400
    grab_point = sample_points[selected_point_num - 1]
    logging.info(f"Grab Point Coordinates: {grab_point}")
    return grab_point, grab_point_data


def plan_path_P2P(
    instruction,
    object_name,
    grid_image,
    original_image,
    config,
):
    """
    Plan a path for a specified object based on the given instruction using GPT.

    Args:
        instruction (str): The instruction for path planning.
        object_name (str): The name of the object for which the path is to be planned.
        start_point (tuple): The starting point for the path planning.
        grid_image (np.ndarray): The grid image used for path planning.
        original_image (np.ndarray): The original image.
        masked_image (np.ndarray): The masked image.
        filtered_masks (list): A list of filtered masks.

    Returns:
        dict or list: The planned path data.
    """
    prompt = construct_path_planning_prompt_P2P(instruction, object_name)
    gpt_response = request_gpt(
        message=prompt,
        images=[original_image, grid_image],
        local_image=True,
        model_name=config["model_name"],
    )
    js = {}
    js["prompt"] = prompt
    js["response"] = gpt_response
    save_dict_to_json(js, os.path.join(config["OUTPUT_DIR"], "MPP2P.json"))
    logging.debug(f"GPT Path Planning Response: {gpt_response}")
    path_data = parse_gpt_response(gpt_response)
    processed_data = path_data["selected_point"]
    if processed_data["type"] == "grid":
        path_data = [
            {
                "grid_number": processed_data["label"],
                "height_m": 0.3,
                "claw_orientation": "down",
            }
        ]
    return path_data


def plan_path_grid(
    instruction,
    object_name,
    start_point,
    grid_image,
    config,
):
    """
    Plan a path for a specified object based on the given instruction using GPT.

    Args:
        instruction (str): The instruction for path planning.
        object_name (str): The name of the object for which the path is to be planned.
        start_point (tuple): The starting point for the path planning.
        grid_image (np.ndarray): The grid image used for path planning.
        original_image (np.ndarray): The original image.
        masked_image (np.ndarray): The masked image.
        filtered_masks (list): A list of filtered masks.

    Returns:
        dict or list: The planned path data.
    """
    prompt = construct_path_planning_prompt(instruction, object_name, start_point)
    gpt_response = request_gpt(
        message=prompt,
        images=grid_image,
        local_image=True,
        model_name=config["model_name"],
    )
    js = {}
    js["prompt"] = prompt
    js["response"] = gpt_response
    save_dict_to_json(js, os.path.join(config["OUTPUT_DIR"], "MPPath.json"))
    logging.debug(f"GPT Path Planning Response: {gpt_response}")

    path_data = parse_gpt_response(gpt_response)
    path_data = path_data["path"]
    logging.info(
        f"Generated Path Planning Instructions: {json.dumps(path_data, indent=2)}"
    )
    return path_data


def map_grid_to_points(image, path_steps):
    """
    Map grid labels to image coordinates and update steps with points.

    Args:
        image (np.ndarray): The image used to determine the grid dimensions.
        path_steps (list): The list of steps in the path, where each step includes grid numbers or specific points.
    """
    # 创建一个网格字典，用于将网格标签映射到图像坐标
    grid_dict = {}
    rows_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    cols_labels = list(range(1, 17))
    rows = Config.GRID_ROWS
    cols = Config.GRID_COLS
    height, width = image.shape[:2]
    cell_width = width // cols
    cell_height = height // rows

    for r in range(rows):
        for c in range(cols):
            label = f"{rows_labels[r]}{int(c)+1}"
            center_x = c * cell_width + cell_width // 2
            center_y = r * cell_height + cell_height // 2
            grid_dict[label] = (center_x, center_y)

    # 更新路径中的网格坐标
    for step in path_steps:
        if "point" not in step:
            grid_number = step["grid_number"].upper()
            if grid_number in grid_dict:
                step["point"] = grid_dict[grid_number]
            else:
                logging.warning(f"Grid number {grid_number} not found in grid labels.")


def visualize_path(image, path_steps):
    """
    Visualize the planned path on the given image.

    Args:
        image (np.ndarray): The image on which to visualize the path.
        path_steps (list): The list of steps in the path, where each step includes points.
    """
    # 提取路径中的点
    path_points = [step["point"] for step in path_steps if "point" in step]

    # 绘制路径
    for i in range(1, len(path_points)):
        start_point = path_points[i - 1]
        end_point = path_points[i]
        cv2.arrowedLine(
            image, start_point, end_point, (0, 255, 0), 2, tipLength=0.05  # 绿色箭头
        )

    # 标记路径点
    for idx, point in enumerate(path_points, 1):
        cv2.circle(image, point, 7, (0, 0, 255), -1)  # 红色圆点
        cv2.putText(
            image,
            str(idx),
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return image
    # 保存可视化路径后的图像


# =========================
# Processing Workflow
# =========================


def resize_back(result):
    coordinates = result["selected_grab_point"]["coordinates"]
    coordinates_list = list(coordinates)
    coordinates_list[0] /= 1280 / Config.ORIGINAL_IMAGE_WIDTH
    coordinates_list[1] /= 720 / Config.ORIGINAL_IMAGE_HEIGHT
    result["selected_grab_point"]["coordinates"] = tuple(coordinates_list)
    # result["selected_grab_point"]["coordinates"][0] /= 1280/Config.ORIGINAL_IMAGE_WIDTH
    # result["selected_grab_point"]["coordinates"][1] /= 720/Config.ORIGINAL_IMAGE_HEIGHT
    for point in result["path_planning_instructions"]:
        p = point["point"]
        p_list = list(p)
        p_list[0] /= 1280 / Config.ORIGINAL_IMAGE_WIDTH
        p_list[1] /= 720 / Config.ORIGINAL_IMAGE_HEIGHT
        point["point"] = tuple(p_list)


def process_image_and_prompt(image_rgb, prompt_text, config):
    """
    Process the given image and prompt to identify an object, select a grab point, and plan a path.

    Args:
        image_rgb (np.ndarray): The input image as a numpy array with RGB order.
        prompt_text (str): The prompt text for object identification and path planning.

    Returns:
        dict: A dictionary containing the results of the processing.
        int: The HTTP status code indicating success or failure.
    """
    try:
        Config.ORIGINAL_IMAGE_HEIGHT = image_rgb.shape[0]
        Config.ORIGINAL_IMAGE_WIDTH = image_rgb.shape[1]
        image_rgb = cv2.resize(image_rgb, (1280, 720))
        save_image(image_rgb, os.path.join(config["OUTPUT_DIR"], "original_image.jpg"))

        masks = generate_masks(image_rgb)
        if not masks:
            return {"error": "No masks generated by SAM2."}, 400
        selected_mask, filtered_masks, identified_object = SoM(
            image_rgb, masks, prompt_text, config
        )
        resized_image, resized_mask, bbox = crop_and_resize(
            image_rgb, selected_mask, Config.FIXED_RESIZE_DIMENSIONS
        )

        if config["CtoF"]:
            resized_mask = CtoF_SoM(resized_image, resized_mask, prompt_text, config)

        annotated_image, sample_points_coords = sample_and_annotate_points(
            resized_image, resized_mask, 5, bbox, Config.FIXED_RESIZE_DIMENSIONS
        )
        save_image(
            annotated_image,
            os.path.join(config["OUTPUT_DIR"], "cropped_sampled_mask.jpg"),
        )
        grab_point, grab_point_data = select_grab_point(
            identified_object["object_name"],
            sample_points_coords,
            annotated_image,
            config,
        )

        grid_image = image_rgb.copy()
        grid_image = draw_grid(grid_image)
        save_image(grid_image, os.path.join(config["OUTPUT_DIR"], "grid_overlay.jpg"))
        selected_grid_label = get_selected_grid_label(grab_point, image_rgb)
        movement_steps = [
            {
                "grid_number": selected_grid_label,
                "height_m": 0.3,
                "claw_orientation": "down",
            }
        ]
        cv2.circle(grid_image, grab_point, 7, (0, 0, 255), -1)  # 红色圆点
        cv2.putText(
            grid_image,
            "start point",
            (grab_point[0] + 10, grab_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        if config["P2P"]:
            path_data = plan_path_P2P(
                prompt_text,
                identified_object["object_name"],
                grid_image,
                image_rgb,
                config,
            )
        else:
            path_data = plan_path_grid(
                prompt_text,
                identified_object["object_name"],
                selected_grid_label,
                grid_image,
                config,
            )

        if isinstance(path_data, list):
            map_grid_to_points(grid_image, path_data)
            visualize_path(grid_image, path_data)
            save_image(
                grid_image, os.path.join(config["OUTPUT_DIR"], "path_visualization.jpg")
            )

        result = {
            "identified_object": identified_object,
            "selected_grab_point": {
                "point_number": grab_point_data.get("selected_point"),
                "coordinates": grab_point,
                "reason": grab_point_data.get("reason"),
            },
            "path_planning_instructions": path_data,
            "movement_steps": movement_steps,
            "grid_label": selected_grid_label,
        }
        resize_back(result)
        return result, 200

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        logging.debug(traceback.format_exc())
        return {"error": "An unexpected error occurred during processing."}, 500


@app.route("/process", methods=["POST"])
def process():
    """
    Process of flask, by input data with keys ["image", "prompt" ,"uuid"].
    "image": three channel image with order of RGB, with shape (width, height, 3), in list type.
    "prompt": instruction of the task, in string type.
    "uuid": the unique uid of the task, logs will save in folder "/logs/uuid/datetime"

    Returns:
        jsonify: result of the process, data or error.
        status code: 200 for success and 400 for failure.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON payload provided."}), 400

    if "image" not in data:
        return jsonify({"error": "No image data provided."}), 400

    image_data = data["image"]
    prompt_text = data.get("prompt", "")
    uuid = data.get("uuid", "test")
    config = data.get("config", {})
    model_name = config.get("model_name", "gpt-4o-2024-05-13")
    config["model_name"] = model_name
    folder_name = ""
    if config["CtoF"]:
        folder_name += "CtoF"
    else:
        folder_name += "noCtoF"
    if config["P2P"]:
        folder_name += "_P2P"
    else:
        folder_name += "_noP2P"
    config["OUTPUT_DIR"] = (
        f"logs/{uuid}/{folder_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(config["OUTPUT_DIR"], exist_ok=True)
    if not image_data:
        return jsonify({"error": "Empty image data."}), 400

    if not prompt_text:
        return jsonify({"error": "No prompt provided."}), 400

    try:
        image_array = np.array(image_data, dtype=np.uint8)
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            return (
                jsonify(
                    {"error": "Image array must be a 3D array with 3 channels (RGB)."}
                ),
                400,
            )
    except Exception as e:
        logging.error(f"Failed to convert image data to numpy array: {e}")
        logging.debug(traceback.format_exc())
        return jsonify({"error": "Invalid image data format."}), 400

    # 处理图像和提示
    result, status_code = process_image_and_prompt(image_array, prompt_text, config)
    return jsonify(result), status_code


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, threaded=False, processes=64)
