from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, jsonify
import logging
import numpy as np
import os
import traceback
from utils.prompt import construct_gpt_subtask_split_prompt
from utils.gpt_utils import request_gpt, encode_image, parse_gpt_response


class Config:
    LOG_LEVEL = logging.DEBUG
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    SAVE_FLAG = True


logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)

app = Flask(__name__)


@app.route("/process", methods=["POST"])
def process_request():
    try:
        data = request.json
        instruction = data.get("prompt")
        image_data = data.get("image")
        model_name = data.get("config", {}).get("model_name", "gpt-4o-2024-05-13")
        if not image_data:
            return jsonify({"error": "Empty image data."}), 400
        if not instruction:
            return jsonify({"error": "No instruction provided."}), 400
        try:
            image_array = np.array(image_data, dtype=np.uint8)

            if image_array.ndim != 3 or image_array.shape[2] != 3:
                return (
                    jsonify(
                        {
                            "error": "Image array must be a 3D array with 3 channels (RGB)."
                        }
                    ),
                    400,
                )
        except Exception as e:
            logging.error(f"Failed to convert image data to numpy array: {e}")
            logging.debug(traceback.format_exc())
            return jsonify({"error": "Invalid image data format."}), 400
        if not instruction or not image_data:
            return jsonify({"error": "Instruction and image data are required."}), 400
        prompt = construct_gpt_subtask_split_prompt(instruction)
        gpt_response = request_gpt(
            message=prompt, images=[image_array], local_image=True, model_name=model_name
        )
        result = parse_gpt_response(gpt_response)
        return jsonify({"result": result})

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        logging.debug(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5008, threaded=False, processes=64)
