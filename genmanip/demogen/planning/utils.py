import os
import random
from copy import deepcopy
from filelock import SoftFileLock
from genmanip.utils.file_utils import make_dir, load_dict_from_pkl


def check_demogen_finished(demogen_config, default_config):
    if os.path.exists(
        os.path.join(default_config["DEMONSTRATION_DIR"], demogen_config["task_name"])
    ):
        generated_traj_num = len(
            os.listdir(
                os.path.join(
                    default_config["DEMONSTRATION_DIR"], demogen_config["task_name"]
                )
            )
        )
        if generated_traj_num >= demogen_config["num_episode"]:
            return True
    return False


def check_planning_finished(demogen_config, default_config):
    lock_file = os.path.join(
        os.path.join(
            default_config["DEMONSTRATION_DIR"],
            demogen_config["task_name"],
            "trajectory",
            "log_soft.lock",
        )
    )
    try:
        with SoftFileLock(lock_file, timeout=600.0):
            log_pkl_path = os.path.join(
                os.path.join(
                    default_config["DEMONSTRATION_DIR"],
                    demogen_config["task_name"],
                    "trajectory",
                    "log.pkl",
                )
            )
            if os.path.exists(log_pkl_path):
                log = load_dict_from_pkl(log_pkl_path)
                if "success" in log and log["success"] >= demogen_config["num_episode"]:
                    return True
            return False
    except:
        raise Exception(
            f"Filelock timeout, try to delete the lock file by python standalone_tools/cleanup_lockfiles.py"
        )


def check_evalgen_finished(evalgen_config, default_config):
    lock_file = os.path.join(
        os.path.join(
            default_config["TASKS_DIR"], evalgen_config["task_name"], "log_soft.lock"
        )
    )
    try:
        with SoftFileLock(lock_file, timeout=600.0):
            log_pkl_path = os.path.join(
                os.path.join(
                    default_config["TASKS_DIR"], evalgen_config["task_name"], "log.pkl"
                )
            )
            if os.path.exists(log_pkl_path):
                log = load_dict_from_pkl(log_pkl_path)
                if "success" in log and log["success"] >= evalgen_config["num_test"]:
                    return True
            return False
    except:
        raise Exception(
            f"Filelock timeout, try to delete the lock file by python standalone_tools/cleanup_lockfiles.py"
        )


def check_eval_finished(eval_config, default_config):
    lock_file = os.path.join(
        default_config["EVAL_RESULT_DIR"], eval_config["task_name"], "eval_soft.lock"
    )
    try:
        with SoftFileLock(lock_file, timeout=600.0):
            task_dir = os.path.join(
                default_config["EVAL_RESULT_DIR"],
                eval_config["task_name"],
            )
            if os.path.exists(task_dir):
                evaluation_num = len(os.listdir(task_dir)) - 1
                if evaluation_num >= eval_config["num_test"]:
                    if os.path.exists(lock_file):
                        os.remove(lock_file)
                    return -1
                else:
                    make_dir(os.path.join(task_dir, str(evaluation_num).zfill(3)))
                    return evaluation_num
            return 0
    except:
        raise Exception(
            f"Filelock timeout, try to delete the lock file by python standalone_tools/cleanup_lockfiles.py"
        )


def random_choice_from_object_or_list(object_or_list):
    if isinstance(object_or_list, list):
        return random.choice(object_or_list)
    else:
        return object_or_list


def get_random_position_candidate():
    return ["top", "near", "left", "right", "front", "back"]


def corse_process_task_data(demogen_config):
    task_data = {}
    task_data["goal"] = deepcopy(demogen_config["generation_config"]["goal"])
    if "long_horizon_meta_info" in demogen_config:
        task_data["long_horizon_meta_info"] = demogen_config["long_horizon_meta_info"]
    for goal in task_data["goal"]:
        for subgoal in goal:
            assert demogen_config["mode"] != "benchmark" or (
                (
                    (
                        isinstance(subgoal["obj1_uid"], list)
                        and len(subgoal["obj1_uid"]) == 1
                    )
                    or (not isinstance(subgoal["obj1_uid"], list))
                )
                and (
                    (
                        isinstance(subgoal["obj2_uid"], list)
                        and len(subgoal["obj2_uid"]) == 1
                    )
                    or (not isinstance(subgoal["obj2_uid"], list))
                )
            ), "obj1_uid and obj2_uid must be string or a list with only one element in benchmark mode"
            if "obj1_uid" not in subgoal or "obj2_uid" not in subgoal:
                continue
            if "obj1" in subgoal:
                if not isinstance(subgoal["obj1"], list):
                    subgoal["obj1"] = [subgoal["obj1"]]
            if not isinstance(subgoal["obj1_uid"], list):
                subgoal["obj1_uid"] = [subgoal["obj1_uid"]]
            obj1_idx = random_choice_from_object_or_list(
                [i for i in range(len(subgoal["obj1_uid"]))]
            )
            if "obj1" in subgoal:
                subgoal["obj1"] = subgoal["obj1"][obj1_idx]
            subgoal["obj1_uid"] = subgoal["obj1_uid"][obj1_idx]
            if "obj2" in subgoal:
                if not isinstance(subgoal["obj2"], list):
                    subgoal["obj2"] = [subgoal["obj2"]]
            if not isinstance(subgoal["obj2_uid"], list):
                subgoal["obj2_uid"] = [subgoal["obj2_uid"]]
            obj2_idx = random_choice_from_object_or_list(
                [i for i in range(len(subgoal["obj2_uid"]))]
            )
            if "obj2" in subgoal:
                subgoal["obj2"] = subgoal["obj2"][obj2_idx]
            subgoal["obj2_uid"] = subgoal["obj2_uid"][obj2_idx]
            subgoal["position"] = random_choice_from_object_or_list(subgoal["position"])
    return task_data


def concat_instruction(task_data, sequence=None):
    instruction = ""
    goals = task_data["goal"][0]
    if sequence is not None:
        if isinstance(sequence, int):
            goals = [goals[sequence]]
        elif isinstance(sequence, (list, tuple)):
            goals = [goals[i] for i in sequence]
        else:
            raise TypeError("sequence must be int or list/tuple")
    for i, subgoal in enumerate(goals):
        if "obj1_uid" in subgoal and "obj2_uid" in subgoal:
            obj1_caption = (
                task_data["object_infos"][subgoal["obj1_uid"]]["caption"]
                .lower()
                .replace(".", "")
            )
            obj2_caption = (
                task_data["object_infos"][subgoal["obj2_uid"]]["caption"]
                .lower()
                .replace(".", "")
            )
            position = subgoal["position"]
            if i == 0:
                instruction += (
                    f"Move the {obj1_caption} to the {position} of the {obj2_caption}"
                )
            else:
                instruction += f", and move the {obj1_caption} to the {position} of the {obj2_caption}"
        elif "obj1_uid" in subgoal:
            obj1_caption = (
                task_data["object_infos"][subgoal["obj1_uid"]]["caption"]
                .lower()
                .replace(".", "")
            )
            if i == 0:
                instruction += f"Open the {obj1_caption}"
            else:
                instruction += f", open the {obj1_caption}"
    instruction += "."
    return instruction


def rewrite_instruction(task_data, demogen_config):
    if demogen_config["domain_randomization"].get("rewrite_instruction", False):
        sequence = demogen_config["domain_randomization"].get("rewrite_sequece", None)
        task_data["instruction"] = concat_instruction(task_data, sequence)
    else:
        task_data["instruction"] = demogen_config["instruction"]


def get_action_list(task_data, demogen_config):
    task_data["task_path"] = random.choice(task_data["goal"])
    if demogen_config["generation_config"].get("is_shuffle", False):
        random.shuffle(task_data["task_path"])
    return task_data


def refine_task_data(task_data, demogen_config):
    rewrite_instruction(task_data, demogen_config)
    task_data = get_action_list(task_data, demogen_config)
    return task_data
