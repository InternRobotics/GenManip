from io import BytesIO
import base64
import time
from PIL import Image
import requests
import os
import logging
import json
import traceback
import numpy as np

DEFAULT_LLM_MODEL_NAME = "gpt-4o-2024-05-13"
DEFAULT_VLM_MODEL_NAME = "gpt-4o-2024-05-13"
TIMEOUT = 50

model_route = {
    "gpt-4.5-preview": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "gpt-4.5-preview-2025-02-27": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "gpt-4o": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "gpt-4o-2024-05-13": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "gpt-4o-2024-08-06": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "gpt-4o-2024-11-20": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "gpt-4o-mini": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "gpt-4o-mini-2024-07-18": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "Qwen/Qwen2.5-VL-72B-Instruct": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "claude-3-5-sonnet-20240620": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "claude-3-5-sonnet-20241022": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "claude-3-7-sonnet-20250219": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "claude-3-7-sonnet-20250219-thinking": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "gemini-2.0-flash": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "gemini-2.0-pro-exp-02-05": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
    "gemini-2.0-flash-thinking-exp": ["DEFAULT_SHOP_URL", "DEFAULT_SHOP_KEY"],
}


def encode_image(image: np.array):
    """encode np array into base64 string

    Args:
        image (np.array): 3 channel image with order RGB

    Returns:
        str: encoded image in string type
    """
    buffered = BytesIO()
    pil_image = Image.fromarray(image)  # 将 numpy 数组转换为 PIL Image
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def prepare_gpt_payload(
    messages,
    images=None,
    meta_prompt="You are an assistant.",
    model_name=None,
    local_image=False,
):
    """prepare payload for gpt request

    Args:
        messages (string): The message for GPT
        images (List, optional): A List of image/A image for GPT. Defaults to None.
        meta_prompt (str, optional): Meta prompt for GPT. Defaults to "You are an assistant.".
        model_name (str, optional): Used GPT model name. Defaults to None.
        local_image (bool, optional): True when using np.array, false when using url with string type. Defaults to False.

    Returns:
        Dict: prepared payload.
    """
    user_content = []

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        content = {
            "type": "text",
            "text": message,
        }
        user_content.append(content)

    if images is not None:
        if not isinstance(images, list):
            images = [images]

        for image in images:
            if local_image:
                # 假设这里的 image 是一个 numpy 数组
                base64_image = encode_image(image)
                image_url = f"data:image/jpeg;base64,{base64_image}"
            else:
                image_url = image  # 这里的 image 应该是一个有效的 URL

            content = {
                "type": "image_url",
                "image_url": {"url": image_url, "detail": "high"},
            }
            user_content.append(content)

    payload = {
        "model": model_name if model_name else DEFAULT_LLM_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": meta_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "max_tokens": 4000,
        "response_format": {"type": "json_object"},
    }

    return payload


def get_api_key(model_name):
    if model_name in model_route:
        return os.getenv(model_route[model_name][1]), os.getenv(
            model_route[model_name][0]
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def request_gpt(
    message,
    images=None,
    meta_prompt="You are an assistant that helps design pick and place operations.",
    model_name=None,
    local_image=False,
    max_retries=50,
):
    if not isinstance(images, list):
        images = [images]
    if model_name == "internvl-latest":
        from .internvl_utils import request_gpt as request_gpt_internvl
        from datetime import datetime
        import hashlib
        from pathlib import Path
        from PIL import Image

        image_path_list = []
        for i, image in enumerate(images):
            hash_image_name = hashlib.md5(
                str(datetime.now().strftime("%Y%m%d%H%M%S%f")).encode("utf-8")
            ).hexdigest()
            Path(os.path.join(os.getcwd(), "images")).mkdir(parents=True, exist_ok=True)
            image_path = os.path.join(os.getcwd(), f"images/{hash_image_name}_{i}.jpg")
            Image.fromarray(image).save(image_path)
            print(image_path)
            image_path_list.append(image_path)
        return request_gpt_internvl(
            message,
            images=None,
            meta_prompt=meta_prompt,
            model_name=model_name,
            local_image=local_image,
            max_retries=max_retries,
            image_path=image_path_list,
        )
    else:
        return request_gpt_normal(
            message,
            images=images,
            meta_prompt=meta_prompt,
            model_name=model_name,
            local_image=local_image,
            max_retries=max_retries,
        )


def request_gpt_normal(
    message,
    images=None,
    meta_prompt="You are an assistant that helps design pick and place operations.",
    model_name="gpt-4o",
    local_image=False,
    max_retries=10,
):
    """send request to GPT and get response

    Args:
        message (str): _description_
        images (_type_, optional): _description_. Defaults to None.
        meta_prompt (str, optional): _description_. Defaults to "You are an assistant that helps design pick and place operations.".
        model_name (_type_, optional): _description_. Defaults to None.
        local_image (bool, optional): _description_. Defaults to False.
        max_retries (int, optional): _description_. Defaults to 5.

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    API_KEY, BOYUE_URL = get_api_key(model_name)
    payload = prepare_gpt_payload(
        messages=message,
        images=images,
        meta_prompt=meta_prompt,
        model_name=model_name,
        local_image=local_image,
    )
    from openai import OpenAI

    client = OpenAI(
        base_url=BOYUE_URL,
        api_key=API_KEY,
    )
    for attempt in range(max_retries):
        try:
            if model_name == "Qwen/Qwen2.5-VL-72B-Instruct":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=payload["messages"],
                    max_tokens=payload["max_tokens"],
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=payload["messages"],
                    max_tokens=payload["max_tokens"],
                    response_format=payload["response_format"],
                )
            res = response.choices[0].message.content
            parse_gpt_response(res)
            return res
        except Exception as e:
            attempt += 1
            logging.error(f"{model_name} request attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logging.info("Retrying...")
            else:
                raise RuntimeError(
                    "Exceeded maximum retry attempts for GPT requests"
                ) from e


def parse_gpt_response(response):
    """
    解析 GPT 响应以提取 JSON 数据。
    """
    try:
        if "```json" in response.lower():
            json_str = response.lower().split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1]
        else:
            json_str = response

        operations = json.loads(json_str)

        if isinstance(operations, dict):
            return operations
        elif isinstance(operations, list):
            return operations
        else:
            raise ValueError(
                "The operations format in GPT response is not a list or dict."
            )

    except Exception as e:
        logging.error("Unable to parse GPT response into JSON.")
        logging.debug(f"Original response content: {response}")
        raise e
