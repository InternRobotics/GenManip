def construct_gpt_subtask_split_prompt(instruction):
    """
    构建发送给 GPT 的提示，以拆分长时任务为子任务。
    """
    prompt = (
        f"""
You are an assistant specialized in breaking down complex pick-and-place tasks into simpler, manageable subtasks. Analyze the given task instruction and divide it into a list of subtasks.

**Task instruction**: {instruction}

**Requirements**:
1. Return the subtasks in JSON format as a list.
2. If the task is simple and does not require splitting, return a list with a single subtask.
3. Each subtask should consist of a single pick-and-place operation.
4. The subtasks will be executed sequentially by downstream modules, so ensure the instructions are clear and complete, with all necessary information.
5. Consider the possibility that the task might be partly executed before this request. Use the provided image to identify the completed parts and only output the remaining subtasks.
6. It is normal for most tasks to have only one subtask.

**Example output**:
```json
{{
  "subtasks": [
    "Pick the red apple and place it on the table.",
    "Pick the blue cup and place it next to the apple."
  ]
}}
```
"""
        # "You are an assistant that can split complex pick and place tasks into simpler subtasks. Analyze the following task instruction and divide it into a list of subtasks.\n"
        # f"Task instruction: {instruction}.\n"
        # "Return the subtasks in JSON format as a list. If the task is simple and does not require splitting, return a list with a single subtask.\n"
        # "Each subtask should consist of a single pick and place operation. The subtask will be executed in sequence by down-stream modules, so make sure the instructions are complete and clear with all the necessary information.\n"
        # "The task might be partly executed before this request, consider with the image and make sure you only output those uncompleted subtasks.\n"
        # "It is normal for most tasks to have only one subtask.\n"
        # "Example output:\n"
        # "```json\n"
        # "{\n"
        # '"subtasks":\n'
        # "[\n"
        # '    "Pick the red apple and place it on the table.",\n'
        # '    "Pick the blue cup and place it next to the apple."\n'
        # "]\n"
        # "}\n"
        # "```\n"
    )
    return prompt


def construct_gpt_completions_tracking_prompt(instruction):
    """
    构建发送给 GPT 的提示，以追踪任务完成情况。
    """
    prompt = (
        "You are an assistant that checks whether a task is completed. For the given instruction, the provided image describes the current scene, and you need to descibe the given scene and determine whether the task is finished.\n"
        f"Task instruction: {instruction}.\n"
        "Example output:\n"
        "```json\n"
        "{\n"
        '"scene_description": "The description of the scene"'
        '"finished": true or false\n'
        "}\n"
        "```\n"
    )
    return prompt


def construct_gpt_prompt(instruction):
    """
    构建发送给 GPT 的提示，以进行对象识别。
    """
    prompt = (
        "You are an assistant designed for creating pick and place operations. The image shows a scene currently observed by the camera, "
        "and another image segmented the original image into different parts using a segmentation model with annotations on the corresponding masks. You need to understand the task "
        "instructions and select the object required for the current task.\n"
        f"Task instruction: {instruction}. Pay attention to the task instruction after 'pick' which usually indicates the target object. \n"
        "Based on the above information, output the selected object in JSON format. If none of the masks include the object you need to select, "
        'output the number as -1 and the object name as "not_found."\n'
        "Example output:\n"
        "```json\n"
        "{\n"
        '    "number": 3,\n'
        '    "object_name": "apple"\n'
        '    "color": "red"\n'
        "}\n"
        "```\n"
    )
    return prompt


def construct_gpt_prompt_CtoF(instruction):
    """
    构建发送给 GPT 的提示，以进行对象识别。
    """
    prompt = (
        "You are an assistant designed for fine-grained part picking operations. The image shows the object that needs to be operated on, "
        "and another image segmented the original image into different parts using a segmentation model with annotations on the corresponding masks. "
        "You need to understand the task instructions and select the specific part of the object that needs to be picked.\n"
        f"Task instruction: {instruction}.\n"
        "Based on the above information, output the selected part in JSON format. If none of the masks include the part you need to select, "
        'output the number as -1 and the part name as "not_found."\n'
        "Example output:\n"
        "```json\n"
        "{\n"
        '    "number": 3,\n'
        '    "part_name": "handle"\n'
        '    "color": "red"\n'
        "}\n"
        "```\n"
    )
    return prompt


def construct_grab_point_prompt(object_name, points):
    """
    构建发送给 GPT 的提示，以选择最佳抓取点。
    """
    prompt = (
        f"You need to grab the object: {object_name}. The image below are the original image and five sampled points.\n\n"
        "Please select the most appropriate grab point by specifying the point number (1-5) and provide a brief reason.\n"
        "Output the result in JSON format as follows:\n"
        "{\n"
        '  "selected_point": 3,\n'
        '  "reason": "Point 3 provides the most stable grip based on the object\'s orientation."\n'
        "}"
    )
    return prompt


def construct_path_planning_prompt(instruction, object_name, start_point):
    """
    构建发送给 GPT 的提示，以生成机器人手臂的运动路径。
    """
    prompt = (
        "You are a motion planning agent. For the task I provide, you need to plan a series of waypoints and guide the robotic arm to the final position to complete the task.\n"
        f"For the task {instruction}, with the previously identified object {object_name} and the grab points (marked in the image), you need to output the following series of details as described.\n"
        f"The whole path should begin with start point {start_point}\n"
        "Each movement should include the grid cell number, the height above the table (0.1 to 0.5 meters), and the claw's orientation.\n\n"
        "Output the list in JSON format where each item contains:\n"
        "- grid_number: e.g., 'A1'\n"
        "- height_m: float value between 0.1 and 0.5\n"
        "- claw_orientation: one of ['up', 'down', 'left', 'right', 'front', 'back']\n\n"
        "Example Output:\n"
        '{"path":[\n'
        '  {"grid_number": "A1", "height_m": 0.3, "claw_orientation": "down"},\n'
        '  {"grid_number": "B3", "height_m": 0.2, "claw_orientation": "front"},\n'
        "  ...\n"
        "]}\n"
    )
    return prompt


def construct_path_planning_prompt_P2P(instruction, object_name):
    """
    构建发送给 GPT 的提示，以生成机器人手臂的运动路径。
    """
    prompt = (
        "You are an assistant that selects a single point for the robot to move to. Pay attention to the task instruction after 'place' which usually indicates the target position. \n"
        f"For the task {instruction}, with the previously identified object {object_name} and the grab points (marked in the image), you need to output the grid cell label to place the object at.\n"
        "Based on the original image and the grid overlay, please select a grid cell label that represents the target point (where to place the object) for the robot.\n"
        "Output the selected point in JSON format as follows:\n"
        "{\n"
        '    "selected_point": {"type": "grid", "label": "C4"}\n'
        "}\n"
    )
    return prompt


def construct_release_point_prompt(instruction):
    """
    构建发送给 GPT 的提示，以选择最佳抓取点。
    """
    prompt = (
        "You are an assistant that selects a single point for the robot to move to. \n"
        f"For the task {instruction},  you need to output the grid cell label to place the object at.\n"
        "Based on the original image and the grid overlay, please select a grid cell label that represents the target point (where to place the object) for the robot.\n"
        "Please select the most appropriate target point which provides the most proper and stable release position.\n"
        "Output the selected point in JSON format as follows:\n"
        "{\n"
        '    "selected_point": {"type": "grid", "label": "D9"}\n'
        "}\n"
    )
    return prompt
