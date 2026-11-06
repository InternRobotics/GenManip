"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import argparse
import os
import sys
import pydantic
import torch
import numpy
from pydantic import BaseModel, Field
from isaacsim import SimulationApp  # type: ignore[import-untyped]

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)


def parse_args() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
        default="configs/tasks/minimal.yml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--record",
        type=str,
        required=False,
        default="just for record",
        help="Helps to record user name for monitoring in htop/nvidia-smi/nvitop etc.",
    )

    parser.add_argument(
        "--eval",
        default=False,
        action="store_true",
        help="Run in eval mode, generate tasks in 'evaluation_configs' and save to 'tasks' folder",
    )
    parser.add_argument(
        "-wop",
        "--without_planning",
        default=False,
        action="store_true",
        help="Run in without planning mode, only generate layout and save first frame results, for vlm data generation",
    )

    parser.add_argument(
        "-l",
        "--local",
        default=False,
        action="store_true",
        help="Run in local mode, a quick command to enable Isaac Sim GUI and use local anygrasp server",
    )
    parser.add_argument(
        "-ad",
        "--action_debug_mode",
        default=False,
        action="store_true",
        help="Run in action debug mode, reload the task config from the first demonstration config every episode",
    )
    parser.add_argument(
        "-gv",
        "--genmanip_verbose",
        default=False,
        action="store_true",
        help="Verbose mode for genmanip",
    )
    args = parser.parse_args()
    if args.genmanip_verbose:
        os.environ["GENMANIP_VERBOSE"] = "1"
    return args


args = parse_args()

cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
first_visible = cuda_visible.split(",")[0].strip() if cuda_visible else "0"
active_gpu = int(first_visible) if first_visible.isdigit() else 0

simulation_app = SimulationApp({
                    "headless": not args.local,
                    "multi_gpu": False,
                    "active_gpu": active_gpu,  # Omniverse renderer: physical GPU id
                    "physics_gpu": 0,
                })
simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)

from genmanip.demogen.workflow.demogen import DemoGenWorkflow

if __name__ == "__main__":
    workflow = DemoGenWorkflow(args, simulation_app, current_dir)
    workflow.run()
