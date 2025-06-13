import os
import logging
import json
import cv2
import numpy as np


def save_image(image: np.ndarray, filepath):
    """Save an image to the specified filename in the output directory.

    Args:
        image (np.ndarray): The image to be saved with RGB order.
        filename (str): The name of the file to save the image as.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, image_bgr)
    logging.debug(f"Image saved: {filepath}")


def save_dict_to_json(data, file_path):
    """Save a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to be saved.
        file_path (str): The path to the file where the JSON will be saved.
    """
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print(f"JSON saved: {file_path}")
