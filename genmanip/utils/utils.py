import hashlib
import logging
import random
import os


def generate_hash(text, algorithm="sha256"):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    hash_object = hashlib.sha256()
    hash_object.update(text.encode("utf-8"))
    return hash_object.hexdigest()


def get_nth_item_from_dict(d: dict, n: int):
    if n < 0 or n >= len(d):
        raise IndexError("Index out of range")
    return list(d.items())[n]


def process_check_finished(result):
    while isinstance(result, dict):
        if "finished" in result:
            result = result["finished"]
        else:
            for res in result:
                result = result[res]
                break
    return result


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def to_list(data):
    res = []
    if data is not None:
        res = [_ for _ in data]
    return res


def process_subgoal(goal_list, instruction):
    obj1_uid_list = goal_list["obj1_uid"]
    obj1_list = goal_list["obj1"]
    obj2_uid_list = goal_list["obj2_uid"]
    obj2_list = goal_list["obj2"]
    position_list = goal_list["position"]
    if not isinstance(obj1_uid_list, list):
        obj1_uid_list = [obj1_uid_list]
    if not isinstance(obj2_uid_list, list):
        obj2_uid_list = [obj2_uid_list]
    if not isinstance(obj1_list, list):
        obj1_list = [obj1_list]
    if not isinstance(obj2_list, list):
        obj2_list = [obj2_list]
    if not isinstance(position_list, list):
        position_list = [position_list]
    obj1_list_zip = list(zip(obj1_list, obj1_uid_list))
    obj2_list_zip = list(zip(obj2_list, obj2_uid_list))
    obj1, obj1_uid = random.choice(obj1_list_zip)
    obj2, obj2_uid = random.choice(obj2_list_zip)
    position = random.choice(position_list)
    if len(obj1_list) > 1 or len(obj2_list) > 1 or len(position_list) > 1:
        instruction = f"put the {obj1} on the {position} of the {obj2}"
    else:
        instruction = instruction
    random_goal = {
        "obj1": obj1,
        "obj1_uid": obj1_uid,
        "obj2": obj2,
        "obj2_uid": obj2_uid,
        "position": position,
    }
    return random_goal, instruction


def process_goal_list_to_random_goal(task_data):
    instruction = task_data["instruction"]
    single_task_path = random.choice(task_data["goal"])
    task_data["goal"] = [single_task_path]
    has_list = False
    for goal in task_data["goal"][0]:
        if isinstance(goal["obj1"], list):
            has_list = True
            break
        if isinstance(goal["obj2"], list):
            has_list = True
            break
        if isinstance(goal["position"], list):
            has_list = True
            break
        if isinstance(goal["obj1_uid"], list):
            has_list = True
            break
        if isinstance(goal["obj2_uid"], list):
            has_list = True
            break
    if has_list:
        print("has list")
        instruction = ""
        for idx, goal in enumerate(task_data["goal"][0]):
            random_goal, sub_instruction = process_subgoal(goal, instruction)
            task_data["goal"][0][idx] = random_goal
            if idx == 0:
                instruction = sub_instruction
            else:
                instruction += ", and " + sub_instruction
        task_data["instruction"] = instruction
    return task_data


def parse_demogen_config(config):
    demogen_config_list = config["demonstration_configs"]
    return demogen_config_list


def parse_eval_config(config):
    eval_config_list = config["evaluation_configs"]
    return eval_config_list


def parse_evalgen_config(config):
    evalgen_config_list = config["evaluation_configs"]
    return evalgen_config_list


def compare_articulation_status(status1, status2):
    if type(status1) == list:
        for s1, s2 in zip(status1, status2):
            if not compare_articulation_status(s1, s2):
                return False
        return True
    else:
        if not (type(status1) == float and type(status2) == list and len(status2) == 2):
            return False
        else:
            return status2[0] <= status1 <= status2[1]


def parse_usda(usda_path):
    with open(usda_path, "r") as f:
        usda_content = f.read()
    usda_content = usda_content.split("\n")
    for line in usda_content:
        if "prepend payload = @" in line:
            usda_path = line.split("@")[1]
            break
    return usda_path


def check_usda_exist(default_config, demogen_config):
    usd_path = parse_usda(
        os.path.join(
            default_config["ASSETS_DIR"],
            f"{demogen_config['usd_name']}.usda",
        )
    )
    usd_path = os.path.join(
        default_config["ASSETS_DIR"], *demogen_config["usd_name"].split("/")[:-1], usd_path
    )
    if not os.path.exists(usd_path):
        return False
    else:
        return True


def check_proxy_exist():
    if (
        os.environ.get("http_proxy") is not None
        or os.environ.get("https_proxy") is not None
        or os.environ.get("all_proxy") is not None
        or os.environ.get("HTTP_PROXY") is not None
        or os.environ.get("HTTPS_PROXY") is not None
        or os.environ.get("ALL_PROXY") is not None
    ):
        return True
    else:
        return False
