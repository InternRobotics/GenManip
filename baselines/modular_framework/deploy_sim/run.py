import numpy as np
import sys
import os

projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
if projectroot not in sys.path:
    sys.path.append(projectroot)

import argparse
from configs.utils import load_config

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--send_port", type=int, default=12345)
parser.add_argument("-r", "--receive_port", type=int, default=12346)
parser.add_argument("--config", type=str, default="deploy_sim.configs.default.Config")
parser.add_argument("--model_name", type=str, default="gpt-4o-2024-05-13")
parser.add_argument("--P2P", type=str, default="True")
parser.add_argument("--CtoF", type=str, default="True")
load_config(parser.parse_args().config)
args = parser.parse_args()
import time
from deploy_sim.planner.planner import SimPlanner
from modular_framework.modular_framework import ViLAAgent
from deploy_sim.utils.serial_utils import (
    create_send_port_and_wait,
    create_receive_port_and_attach,
    wait_message,
    send_message,
)

args.P2P = args.P2P == "True"
args.CtoF = args.CtoF == "True"
send_socket = create_send_port_and_wait(port=args.send_port)
time.sleep(1)
receive_socket = create_receive_port_and_attach(port=args.receive_port)
while True:
    data = wait_message(receive_socket)
    if data["reset"]:
        if "simPlanner" in locals() or "simPlanner" in globals():
            del simPlanner
        if "agent" in locals() or "agent" in globals():
            del agent
        current_joint_position = data["joint_position_state"][:7]
        simPlanner = SimPlanner(
            data["franka_pose"],
            data["franka_hand_pose"],
            anygrasp_url="10.6.3.75",
            intrinsics=data["camera_data"]["obs_camera"]["intrinsics_matrix"],
            camera_pose=(data["camera_data"]["obs_camera"]["p"], data["camera_data"]["obs_camera"]["q"]),
        )
        agent = ViLAAgent(
            simPlanner, model_name=args.model_name, P2P=args.P2P, CtoF=args.CtoF
        )
        agent.initialize(data["instruction"])
    try:
        joint_position = agent.get_next_step(
            data["camera_data"]["obs_camera"]["rgb"],
            data["camera_data"]["obs_camera"]["depth"],
            data["joint_position_state"],
        )
        delta_joint_position = np.array(joint_position[:7]) - np.array(
            current_joint_position
        )
        current_joint_position = joint_position[:7]
        print(delta_joint_position)
        return_data = {
            "action": (
                delta_joint_position[:7].tolist()
                + ([0.0, 0.0] if simPlanner.get_gripper_status() else [0.04, 0.04])
            )
        }
    except Exception as e:
        print(e)
        return_data = {"action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4]}
    send_message(send_socket, return_data)
