"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import argparse
import os
import sys

# early import to avoid conflict with isaacsim
import pydantic
import torch
import numpy
from pydantic import BaseModel, Field

from isaacsim import SimulationApp  # type: ignore[import-untyped]

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg",
        "--config",
        default="configs/tasks/minimal.yml",
        type=str,
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
        "-l",
        "--local",
        action="store_true",
        help="Run in local mode, a quick command to enable Isaac Sim GUI",
    )

    # Rendering options
    parser.add_argument(
        "-r",
        "--render_first_frame",
        action="store_true",
        help="Only render the first frame",
    )
    parser.add_argument(
        "-d",
        "--downsample",
        type=int,
        default=1,
        help="Downsample the rendering frame rate",
    )
    parser.add_argument(
        "--high_quality", action="store_true", help="High quality rendering"
    )

    # Camera/Annotator options
    parser.add_argument(
        "-a",
        "--add_random_position_camera",
        action="store_true",
        help="Add random position camera",
    )
    parser.add_argument(
        "-ac",
        "--add_cycle_camera",
        action="store_true",
        help="Add cycle camera",
    )
    parser.add_argument(
        "-p",
        "--save_pointcloud",
        action="store_true",
        help="Save pointcloud",
    )
    parser.add_argument(
        "-wod",
        "--without_depth",
        action="store_true",
        help="Without render and save depth info",
    )

    # Domain randomization options
    parser.add_argument(
        "-prr",
        "--process_room_randomization",
        action="store_true",
        help="Process room randomization",
    )
    return parser.parse_args()


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
simulation_app._carb_settings.set("/app/omni.graph.scriptnode/opt_in", True)

from genmanip.demogen.workflow.render import RenderWorkflow

if __name__ == "__main__":
    workflow = RenderWorkflow(args, simulation_app, current_dir)
    workflow.run()
