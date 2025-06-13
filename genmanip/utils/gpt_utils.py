import base64
from io import BytesIO
import json
import logging
import numpy as np
import os
from PIL import Image
import requests
import time
import traceback


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
    if not OPENAI_API_KEY:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    GPT_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    DEFAULT_LLM_MODEL_NAME = "gpt-4o-2024-05-13"
    DEFAULT_VLM_MODEL_NAME = "gpt-4o-2024-05-13"
    TIMEOUT = 50


def encode_image(image: np.array):
    buffered = BytesIO()
    pil_image = Image.fromarray(image)  # 将 numpy 数组转换为 PIL Image
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def parse_gpt_response(response):
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


def prepare_gpt_payload(
    messages,
    images=None,
    meta_prompt="You are an assistant.",
    model_name=None,
    local_image=False,
):
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
                base64_image = encode_image(image)
                image_url = f"data:image/jpeg;base64,{base64_image}"
            else:
                image_url = image
            content = {
                "type": "image_url",
                "image_url": {"url": image_url, "detail": "high"},
            }
            user_content.append(content)
    payload = {
        "model": model_name if model_name else Config.DEFAULT_LLM_MODEL_NAME,
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
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
    }
    return payload


def request_gpt(
    message,
    images=None,
    meta_prompt="You are an assistant that helps design pick and place operations.",
    model_name=None,
    local_image=False,
    max_retries=50,
):
    if model_name is None:
        model_name = (
            Config.DEFAULT_VLM_MODEL_NAME
            if images is not None
            else Config.DEFAULT_LLM_MODEL_NAME
        )
    payload = prepare_gpt_payload(
        messages=message,
        images=images,
        meta_prompt=meta_prompt,
        model_name=model_name,
        local_image=local_image,
    )
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{Config.OPENAI_BASE_URL}/chat/completions",
                headers=Config.GPT_HEADERS,
                json=payload,
                timeout=Config.TIMEOUT,
            )
            response.raise_for_status()
            res = response.json()["choices"][0]["message"]["content"]
            return res
        except requests.exceptions.Timeout:
            attempt += 1
            logging.error(f"Attempt {attempt}: The request timed out. Retrying...")
            if attempt < max_retries - 1:
                time.sleep(2)
        except requests.RequestException as e:
            logging.error(f"{model_name} request attempt {attempt + 1} failed: {e}")
            logging.debug(traceback.format_exc())
            if attempt < max_retries - 1:
                logging.info("Retrying...")
            else:
                raise RuntimeError(
                    "Exceeded maximum retry attempts for GPT requests"
                ) from e
