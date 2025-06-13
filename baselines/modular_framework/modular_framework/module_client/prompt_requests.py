import json
import numpy as np
import requests
import os
from configs import Config

debug = Config.debug_text

def request_server(
    uuid: str, image_array: np.ndarray, prompt: str, type: str, config=None
):
    if type == "prompt_pipeline":
        if debug:
            print("--场景理解阶段--")
        response = get_response(
            uuid,
            image_array,
            prompt,
            config=config,
            address=Config.PROMPT_PIPELINE_ADDRESS,
            port=Config.PROMPT_PIPELINE_PORT,
        )
        if debug:
            print(f"自然语言输出: 抓取目标为: {response['identified_object']['object_name']}")
            print(f"自然语言输出: 抓取点位为: {response['selected_grab_point']['coordinates']}, 选择原因为: {response['selected_grab_point']['reason']}")
        
        return response
    elif type == "check_finished":
        if debug:
            print("--检查是否完成--")
        response = get_response(
            uuid,
            image_array,
            prompt,
            config=config,
            address=Config.CHECK_FINISHED_PIPELINE_ADDRESS,
            port=Config.CHECK_FINISHED_PIPELINE_PORT,
        )
        return response
    elif type == "task_split":
        if debug:
            print("--任务拆解阶段--")
        response = get_response(
            uuid,
            image_array,
            prompt,
            config=config,
            address=Config.TASK_SPLIT_PIPELINE_ADDRESS,
            port=Config.TASK_SPLIT_PIPELINE_PORT,
        )
        if debug:
            print("自然语言输出: 任务列表: \n", response["result"]["subtasks"])
        return response


def get_response(
    uuid, image_array, prompt, config=None, address="127.0.0.1", port="5000"
):
    try:
        if not isinstance(image_array, np.ndarray):
            raise ValueError("Image must be a numpy array.")
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError("Image array must be a 3D array with 3 channels (RGB).")
        image_list = image_array.tolist()
        if config is None:
            config = {}
        payload = {
            "image": image_list,
            "prompt": prompt,
            "uuid": uuid,
            "config": config,
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"http://{address}:{port}/process",
            data=json.dumps(payload),
            headers=headers,
        )
        response.raise_for_status()
        response_json = response.json()
        print("Response Status Code:", response.status_code)
        print("Response JSON:", json.dumps(response_json, indent=2, ensure_ascii=False))
        return response_json
    except FileNotFoundError:
        print(f"Image file not found.")
        return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during the test: {e}")
        return None
