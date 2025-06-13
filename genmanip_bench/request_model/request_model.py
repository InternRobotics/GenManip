from genmanip_bench.request_model.socket_utils import send_message, wait_message


def request_action(
    camera_data,
    franka_hand_pose,
    franka_pose,
    instruction,
    joint_position_state,
    key_action,
    step,
    obj_is_grasped,
    send_port,
    receive_port,
):
    reset = step == 0
    data = {
        "camera_data": camera_data,
        "instruction": instruction,
        "joint_position_state": joint_position_state,
        "key_action": key_action,
        "franka_hand_pose": franka_hand_pose,
        "franka_pose": franka_pose,
        "timestep": step,
        "obj_is_grasped": obj_is_grasped,
        "reset": reset,
    }
    send_message(send_port, data)
    response = wait_message(receive_port)
    return response["action"]
