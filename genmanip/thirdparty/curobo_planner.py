from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from curobo.types.base import TensorDeviceType
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.util.usd_helper import UsdHelper
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
import os
import numpy as np
from genmanip.utils.file_utils import load_yaml


class CuroboFrankaPlanner:
    def __init__(self, robot_cfg, world, franka_prim_path):
        self.world = world
        self.franka_prim_path = franka_prim_path
        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(self.world.stage)
        self.robot_cfg = robot_cfg
        self.world_cfg = WorldConfig()
        self.tensor_args = TensorDeviceType()
        self.pose_metric = PoseCostMetric.create_grasp_approach_metric(
            offset_position=0.15, tstep_fraction=0.8
        )
        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_opt=True,
            use_nn_ik_seed=False,
            need_graph_success=False,
            max_attempts=30,
            timeout=30.0,
            enable_graph_attempt=5,
            ik_fail_return=5,
            pose_cost_metric=None,
            enable_finetune_trajopt=True,
            parallel_finetune=True,
            finetune_dt_scale=1.0,
            finetune_dt_decay=1.01,
            finetune_attempts=50,
            check_start_validity=True,
        )
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            interpolation_dt=0.02,
            collision_activation_distance=0.01,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            self_collision_check=True,
            collision_cache={"obb": 3000, "mesh": 3000},
        )
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup(warmup_js_trajopt=False)
        self.motion_gen.clear_world_cache()
        self.motion_gen.reset(reset_seed=False)
        self.ordered_js_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=False,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
            regularization=True,
        )
        self.ik_solver = IKSolver(self.ik_config)

    def update(self, ignore_list=[]):
        obstacles = self.usd_helper.get_obstacles_from_stage(
            ignore_substring=["franka", "Camera"] + ignore_list,
            reference_prim_path=self.franka_prim_path,
        ).get_collision_check_world()
        self.motion_gen.update_world(obstacles)

    def plan(
        self,
        ee_translation_goal: np.array,
        ee_orientation_goal: np.array,
        sim_js: JointState,
        grasp: bool = False,
    ):
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal),
            quaternion=self.tensor_args.to_device(ee_orientation_goal),
        )
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=self.ordered_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        plan_config = self.plan_config.clone()
        if grasp:
            plan_config.pose_cost_metric = self.pose_metric
        else:
            plan_config.pose_cost_metric = None
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
        if result.success.item():
            cmd_plan = result.get_interpolated_plan()
            cmd_plan = cmd_plan.get_ordered_joint_state(self.ordered_js_names)
            position_list = []
            for idx in range(len(cmd_plan.position)):
                joint_positions = cmd_plan.position[idx].cpu().numpy()
                position_list.append(joint_positions[:7])
            return position_list
        else:
            return None

    def ik_single(self, target_pose: np.array, cur_joint_positions: np.array):
        retract_config = self.tensor_args.to_device(cur_joint_positions.reshape(1, -1))
        seed_config = self.tensor_args.to_device(cur_joint_positions.reshape(1, 1, -1))
        pose = Pose(
            self.tensor_args.to_device(target_pose[:3]),
            self.tensor_args.to_device(target_pose[3:]),
        )
        ik_result = self.ik_solver.solve_single(
            pose, retract_config=retract_config, seed_config=seed_config
        )
        if not ik_result.success.item():
            return None
        return ik_result.js_solution.position.cpu().numpy().squeeze()


def get_curobo_planner(robot, robot_type, scene, current_dir):
    if robot_type == "franka":
        franka_cfg = load_yaml(
            os.path.join(current_dir, "assets/robots/configs/franka.yml")
        )
        planner = CuroboFrankaPlanner(franka_cfg, scene["world"], robot.prim_path)
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")
    return planner
