"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
import numpy as np
import torch

from omni.isaac.core.utils.types import JointsState as SimJointState  # type: ignore
from omni.isaac.core.utils.stage import get_current_stage  # type: ignore


class CuroboPlanner:
    ik_position_threshold = 0.005
    ik_rotation_threshold = 0.05
    branch_hold_position_threshold = 0.01
    branch_hold_rotation_threshold = 0.1
    ik_state_reset_joint_distance = 0.25
    ik_state_reset_position_jump = 0.1
    ik_state_reset_rotation_jump = 0.5
    ik_null_space_weight = 100.0

    # A secondary servo-speed bound for the exceptional case where every
    # successful IK candidate is on a distant branch.  The primary fix is the
    # continuity-regularized solve below; this bound is only a last fallback.
    max_ik_joint_step = 0.2

    def __init__(self, robot_cfg: dict, robot_prim_path: str) -> None:
        self.robot_prim_path = robot_prim_path
        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(get_current_stage())
        self.robot_cfg = robot_cfg
        self.world_cfg = WorldConfig()
        self.tensor_args = TensorDeviceType()
        self.pose_metric = PoseCostMetric.create_grasp_approach_metric(
            offset_position=0.15, tstep_fraction=0.8
        )
        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=7,
            max_attempts=10,
            pose_cost_metric=None,
            enable_finetune_trajopt=True,
            time_dilation_factor=1.0,
        )
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            interpolation_dt=0.01,
            collision_activation_distance=0.001,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            self_collision_check=True,
            collision_cache={"obb": 3000, "mesh": 3000},
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            optimize_dt=True,
            trajopt_dt=None,
            trim_steps=None,
            project_pose_to_goal_frame=False,
        )
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup(warmup_js_trajopt=False)
        self.motion_gen.clear_world_cache()
        self.motion_gen.reset(reset_seed=False)
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            rotation_threshold=self.ik_rotation_threshold,
            position_threshold=self.ik_position_threshold,
            num_seeds=128,
            self_collision_check=True,
            self_collision_opt=False,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
            regularization=True,
            gradient_file="gradient_ik.yml",
        )
        self.ik_solver = IKSolver(self.ik_config)
        # cuRobo's default/autotuned IK optimizer leaves this at zero even when
        # regularization=True, so the solve objective does not penalize a
        # multi-radian move away from retract_config.  Apply the documented
        # null-space regularization to both the particle and gradient stages.
        for optimizer in self.ik_solver.solver.optimizers:
            optimizer.rollout_fn.bound_cost.null_space_weight.fill_(
                self.ik_null_space_weight
            )
        self.ordered_js_names = []
        self.dof_len = 7
        self.raw_js_names = []
        self._last_ik_solution: torch.Tensor | None = None
        self._last_ik_target_pose: torch.Tensor | None = None

    def update(self, ignore_list: list[str] = []) -> None:
        robot_name = self.robot_prim_path.split("/")[-1]
        obstacles = self.usd_helper.get_obstacles_from_stage(
            ignore_substring=[robot_name, "Camera"] + ignore_list,
            reference_prim_path=self.robot_prim_path,
        ).get_collision_check_world()
        self.motion_gen.update_world(obstacles)

    def plan(
        self,
        ee_translation_goal: np.ndarray,
        ee_orientation_goal: np.ndarray,
        sim_js: SimJointState,
        dof_names: list | None = None,
        grasp: bool = False,
    ) -> list[np.ndarray] | None:
        if os.environ.get("GENMANIP_VERBOSE") == "1":
            print("goal pos:", ee_translation_goal)
            print(
                "goal quat:",
                ee_orientation_goal,
                "norm=",
                np.linalg.norm(ee_orientation_goal),
            )
            print(
                "js len:",
                len(sim_js.positions),
                "names len:",
                len(self.ordered_js_names),
            )
            print(
                "finite:",
                np.isfinite(sim_js.positions).all(),
                np.isfinite(ee_translation_goal).all(),
                np.isfinite(ee_orientation_goal).all(),
            )

        if len(self.raw_js_names) == 0:
            self.raw_js_names = self.ordered_js_names
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal),
            quaternion=self.tensor_args.to_device(ee_orientation_goal),
        )
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=self.ordered_js_names if dof_names is None else dof_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.ordered_js_names)
        plan_config = self.plan_config.clone()
        if grasp:
            plan_config.pose_cost_metric = self.pose_metric
        else:
            plan_config.pose_cost_metric = None
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)

        if os.environ.get("GENMANIP_VERBOSE") == "1":
            for k in ["status", "message", "error_code", "reason", "valid", "feasible"]:
                if hasattr(result, k):
                    print(k, getattr(result, k))

        if result.success is not None and result.success.item():
            cmd_plan = result.get_interpolated_plan()
            cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
            position_list = []
            for idx in range(len(cmd_plan.position)):
                joint_positions = cmd_plan.position[idx].cpu().numpy()  # type: ignore
                position_list.append(joint_positions[: self.dof_len])
            return position_list
        else:
            return None

    @staticmethod
    def _quaternion_angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = first / torch.linalg.vector_norm(first).clamp_min(1e-12)
        second = second / torch.linalg.vector_norm(second).clamp_min(1e-12)
        quaternion_dot = torch.abs(torch.sum(first * second))
        return 2.0 * torch.acos(quaternion_dot.clamp(max=1.0))

    def _store_ik_state(
        self,
        solution: torch.Tensor,
        target_position: torch.Tensor,
        target_quaternion: torch.Tensor,
    ) -> None:
        self._last_ik_solution = solution.detach().clone()
        self._last_ik_target_pose = torch.cat(
            [target_position, target_quaternion]
        ).detach().clone()

    def ik_single(
        self, target_pose: np.ndarray, cur_joint_positions: np.ndarray
    ) -> np.ndarray | None:
        current = self.tensor_args.to_device(cur_joint_positions.reshape(-1))
        target_position = self.tensor_args.to_device(target_pose[:3])
        target_quaternion = self.tensor_args.to_device(target_pose[3:])

        # The measured joints lag the previous position target by one physics
        # step.  Preserve the previously commanded IK result as the continuity
        # reference while the robot is still tracking it.  Large joint or pose
        # discontinuities indicate an episode reset or a genuinely new command
        # and re-anchor the state to the measurement automatically.
        reference = current
        last_solution = getattr(self, "_last_ik_solution", None)
        last_target_pose = getattr(self, "_last_ik_target_pose", None)
        if (
            last_solution is not None
            and last_target_pose is not None
            and last_solution.shape == current.shape
        ):
            joint_tracking_error = torch.amax(torch.abs(current - last_solution))
            target_position_jump = torch.linalg.vector_norm(
                target_position - last_target_pose[:3]
            )
            target_rotation_jump = self._quaternion_angle(
                target_quaternion, last_target_pose[3:]
            )
            if (
                joint_tracking_error.item() <= self.ik_state_reset_joint_distance
                and target_position_jump.item() <= self.ik_state_reset_position_jump
                and target_rotation_jump.item() <= self.ik_state_reset_rotation_jump
            ):
                reference = last_solution

        # cuRobo's seed is only an optimizer initialisation; it is not included
        # among the returned candidates.  Record whether the continuity
        # reference is itself a valid zero-motion solution, so it can be
        # preferred specifically when cuRobo attempts a distant branch switch.
        # Do not return it eagerly: doing that for every in-tolerance target
        # would introduce a 5-mm deadband into normal EEPose tracking.
        reference_fk = self.ik_solver.fk(reference.unsqueeze(0))
        reference_position = reference_fk.ee_position.reshape(-1, 3)[0]
        reference_quaternion = reference_fk.ee_quaternion.reshape(-1, 4)[0]
        position_error = torch.linalg.vector_norm(reference_position - target_position)
        rotation_error = self._quaternion_angle(
            reference_quaternion, target_quaternion
        )
        reference_reaches_target = (
            position_error.item() <= self.ik_position_threshold
            and rotation_error.item() <= self.ik_rotation_threshold
        )
        reference_is_near_target = (
            position_error.item() <= self.branch_hold_position_threshold
            and rotation_error.item() <= self.branch_hold_rotation_threshold
        )

        retract_config = reference.reshape(1, -1)
        # solve_single fills any missing seeds randomly.  Supplying only the
        # measured state therefore left 127/128 seeds free to converge to
        # unrelated branches on every control frame.  For servo IK, seed every
        # optimizer instance from the same continuity reference instead.
        seed_config = reference.reshape(1, 1, -1).repeat(
            1, self.ik_config.num_seeds, 1
        )
        pose = Pose(
            target_position,
            target_quaternion,
        )
        ik_result = self.ik_solver.solve_single(
            pose,
            retract_config=retract_config,
            seed_config=seed_config,
            return_seeds=self.ik_config.num_seeds,
        )
        success = ik_result.success.reshape(-1)
        successful_indices = torch.nonzero(success, as_tuple=False).flatten()
        if successful_indices.numel() == 0:
            if reference_reaches_target:
                self._store_ik_state(
                    reference, target_position, target_quaternion
                )
                return reference.detach().cpu().numpy()  # type: ignore
            self._store_ik_state(current, target_position, target_quaternion)
            return None

        # cuRobo normally returns the solution with the lowest aggregate pose
        # cost.  For IK servoing, that can switch between distant kinematic
        # branches even though the current configuration was supplied as both
        # seed and retract configuration.  All returned candidates have already
        # passed cuRobo's pose, joint-limit, and collision checks, so select the
        # successful one closest to the continuity reference instead.
        solutions = ik_result.js_solution.position.reshape(
            -1, ik_result.js_solution.position.shape[-1]
        )
        controlled_dof = current.shape[-1]
        successful_solutions = solutions[successful_indices]
        successful_deltas = successful_solutions[:, :controlled_dof] - reference
        max_joint_deltas = torch.amax(torch.abs(successful_deltas), dim=-1)
        nearest_index = torch.argmin(max_joint_deltas)
        solution = successful_solutions[nearest_index].clone()

        # Optimizer stages can occasionally all converge to distant branches.
        # If the reference already reaches the target, retain that valid
        # zero-motion solution instead of changing kinematic branch.  Otherwise
        # approach the nearest valid solution with a bounded servo step.
        delta = solution[:controlled_dof] - reference
        max_joint_delta = torch.amax(torch.abs(delta))
        if max_joint_delta.item() > self.max_ik_joint_step:
            if reference_is_near_target:
                self._store_ik_state(
                    reference, target_position, target_quaternion
                )
                return reference.detach().cpu().numpy()  # type: ignore
            delta *= self.max_ik_joint_step / max_joint_delta
            solution[:controlled_dof] = reference + delta

        self._store_ik_state(
            solution[:controlled_dof], target_position, target_quaternion
        )
        return solution.detach().cpu().numpy()  # type: ignore

    def fk_single(self, joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joint_positions_tensor = torch.from_numpy(
            joint_positions.astype(np.float32)
        ).to(self.tensor_args.device)
        result = self.ik_solver.fk(joint_positions_tensor.unsqueeze(0))
        position = result.ee_position.cpu().numpy().squeeze()
        orientation = result.ee_quaternion.cpu().numpy().squeeze()
        return position, orientation
