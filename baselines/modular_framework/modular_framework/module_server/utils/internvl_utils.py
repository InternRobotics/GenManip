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
import oss2
import logging
import uuid


# class Config:
DEFAULT_SHOP_KEY = os.getenv("DEFAULT_SHOP_KEY")
DEFAULT_SHOP_URL = os.getenv("DEFAULT_SHOP_URL", "")
if not DEFAULT_SHOP_KEY:
    raise ValueError("Please set the DEFAULT_SHOP_KEY environment variable.")
GPT_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEFAULT_SHOP_KEY}",
}
DEFAULT_LLM_MODEL_NAME = "gpt-4o-2024-05-13"
DEFAULT_VLM_MODEL_NAME = "gpt-4o-2024-05-13"
TIMEOUT = 50


      
import requests
import json
import logging
import ast

vlm_url=""
vlm_token=""
vlm_model="internvl-latest"
oss_bucket_name=""
oss_access_key_id=""
oss_access_key_secret=""
oss_endpoint=""
oss_root_folder="uploads"



def upload_to_oss(file_path, bucket_name, object_name, access_key_id, access_key_secret, endpoint):
    # try:
        # 创建Bucket对象
        auth = oss2.Auth(access_key_id, access_key_secret)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)
        # 上传文件
        bucket.put_object_from_file(object_name, file_path)
        logging.info(f"File {file_path} uploaded to OSS as {object_name}")

        # 获取文件的URL
        url = f"https://{bucket_name}.{endpoint}/{object_name}"
        return url
    # except Exception as e:
    #     logging.error(f"Failed to upload {file_path} to OSS: {e}")
    #     return None
def convert_messages(messages):
    converted = []
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            converted.append(msg)
        else:
            if i == len(messages) - 1:  # Only modify the last message
                converted.extend(msg.to_openai_vlm_format())
            else:
                converted.extend(msg.to_openai_llm_format())
    return converted

class InternVL:
    def __init__(self, api_url, api_key, model="internvl-latest"):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def infer(self, prompt=None, image_url=None, stream=False, callback=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        content = [
            {
                "type": "text",
                "text": prompt, 
            }
        ]
        for i in image_url:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                            "url": i
                    }
                }
            )
        # messages = [{
        #     "role": "system",
        #     "content": "You are an assistant that helps design pick and place operations."
        # },{
        #     "role": "user",
        #     "content": content,
        # }]
        messages = [{
            "role": "user",
            "content": content,
        }]

        data = {
            "model": self.model,
            "messages": messages,
            "n": 1,
            "temperature": 0.8,
            "top_p": 0.9,
            "stream": stream,
        }

        json_data = json.dumps(data, ensure_ascii=True)
        
        # try:
        response = requests.post(self.api_url, headers=headers, data=json_data, stream=stream)
        return response
        full_output = ""
        
        # if stream:
        #     for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b'\n'):
        #         if not chunk:
        #             continue
        #         decoded = chunk.decode('utf-8')
        #         if not decoded.startswith("data:"):
        #             raise Exception(f"Error message: {decoded}")
        #         decoded = decoded.strip("data:").strip()
        #         if "[DONE]" == decoded:
        #             logging.debug("Response finished!")
        #             break
        #         output = json.loads(decoded)
        #         if output.get("object") == "error":
        #             raise Exception(f"Logic error: {output}")
        #         content = output["choices"][0]["delta"].get("content", "")
        #         full_output += content
        #         if callback:
        #             callback(content)
        # else:
        output = response.json()
        print("""""""""""""""""""""""""""""", output)
        if output.get("object") == "error":
            raise Exception(f"Logic error: {output}")
        content = output["choices"][0]["message"].get("content", "")
        full_output += content
        if callback:
            callback(content)
                
        return full_output
        # except Exception as e:
        #     logging.error(f"Error in API call: {e}")
        #     return None




#     def to_openai_vlm_format(self):
#         result = [
#             {"role": "user", "content": [
#                 {"type": "text", "text": self.request},
#                 {"type": "image_url", "image_url": {"url": self.image}}
#             ]}
#         ]


internvl = InternVL(api_url=vlm_url, api_key=vlm_token, model=vlm_model)


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
    image_path=['./current.jpg'],
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
    for attempt in range(max_retries):
        try:
            oss_list = []
            for path in image_path: 
                if "remote" in path:
                    object_name = f"{oss_root_folder}/{path}"
                    oss_url = f"https://{oss_bucket_name}.{oss_endpoint}/{object_name}"
                else:
                    object_name = f"{oss_root_folder}/remote_{path}"
                    oss_url = upload_to_oss(path, oss_bucket_name, object_name, oss_access_key_id, oss_access_key_secret, oss_endpoint)
                oss_list.append(oss_url)
            if oss_url:
                res = internvl.infer(prompt=message, image_url=oss_list, callback=None)
            else:
                logging.error("Failed to upload image to OSS.")
            print(res)
            res = res.json()["choices"][0]["message"]["content"]
            parse_gpt_response(res)
            return res
        except requests.exceptions.Timeout:
            attempt += 1
            logging.error(f"Attempt {attempt}: The request timed out. Retrying...")
            if attempt < max_retries - 1:
                time.sleep(1)  # 等待 2 秒后重试
        except requests.RequestException as e:
            logging.error(f"{model_name} request attempt {attempt + 1} failed: {e}")
            logging.debug(traceback.format_exc())
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
