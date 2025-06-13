import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

app = Flask(__name__)

mask_generator = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2.1-hiera-large")

def process_image(image_data):
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

@app.route("/generate_masks", methods=["POST"])
def generate_masks():
    try:
        image_data = request.files['image'].read()
        image_rgb = process_image(image_data)
        masks = mask_generator.generate(image_rgb)
        mask_list = [{"segmentation": mask['segmentation'].tolist(), "area": mask['area']} for mask in masks]
        return jsonify({"masks": mask_list}), 200

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, threaded=False)
