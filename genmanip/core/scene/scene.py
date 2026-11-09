"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os

from omni.isaac.core import World  # type: ignore
from omni.isaac.core.articulations import Articulation  # type: ignore
from omni.isaac.core.materials.omni_pbr import OmniPBR  # type: ignore
from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.sensor import Camera  # type: ignore

from genmanip.core.metrics.metrics_manager import MetricsManager
from genmanip.core.robot.base import BaseEmbodiment
from genmanip.core.robot.utils import RobotFactory
from genmanip.core.scene.scene_config import SceneConfig
from genmanip.utils.loader.scene import (
    capture_particle_layout_references,
    clean_prim_velocity,
    create_camera_list,
    get_object_list,
    load_articulation_data,
    load_object_pool,
    load_world_xform_prim,
    preprocess_scene,
    add_articulation_to_scene,
    setup_walls_and_materials,
)
from genmanip.utils.loader.utils import (
    collect_articulation_list,
    collect_world_pose_list,
    reset_object_xyz,
)
from genmanip.utils.pointcloud.pointcloud import objectList2meshList
from genmanip.utils.standalone.file_utils import load_yaml
from genmanip.utils.usd_utils import get_src, setup_physics_scene
from genmanip.utils.pointcloud.utils import MeshInfo, PointCloudInfo
from genmanip.utils.loader.scene import relate_object_from_data


class AssetsLibrary:
    def __init__(self):
        self.wall_textures_paths: list[str] = []
        self.domelights_paths: list[str] = []
        self.table_mdl_paths: list[str] = []
        self.table_textures_paths: list[str] = []
        self.table_paths: list[str] = []

    def initialize(self, assets_dir: str):
        if os.path.exists(os.path.join(assets_dir, "miscs/hdrs")):
            self.domelights_paths = os.listdir(f"{assets_dir}/miscs/hdrs")
        else:
            self.domelights_paths = []
        if os.path.exists(os.path.join(assets_dir, "miscs/textures")):
            self.wall_textures_paths = os.listdir(f"{assets_dir}/miscs/textures")
        else:
            self.wall_textures_paths = []
        if os.path.exists(
            os.path.join(assets_dir, "object_usds/grutopia_usd/Table/Materials")
        ):
            self.table_mdl_paths = os.listdir(
                f"{assets_dir}/object_usds/grutopia_usd/Table/Materials"
            )
        else:
            self.table_mdl_paths = []
        if os.path.exists(
            os.path.join(assets_dir, "object_usds/grutopia_usd/Table/table")
        ):
            self.table_paths = os.listdir(
                f"{assets_dir}/object_usds/grutopia_usd/Table/table"
            )
            self.table_paths = [
                os.path.join(
                    assets_dir,
                    "object_usds/grutopia_usd/Table/table",
                    table_path,
                    "instance.usd",
                )
                for table_path in self.table_paths
            ]
        else:
            self.table_paths = []


class CacheLibrary:
    def __init__(self):
        self.mesh_dict: dict[str, MeshInfo] = {}
        self.pointcloud_dict: dict[str, PointCloudInfo] = {}
        self.preloaded_object_list: dict[str, XFormPrim] = {}
        self.preloaded_object_path_list: dict[str, str] = {}
        self.preloaded_object_uid_list: dict[str, list[str]] = {}
        self.preload_hash_feature: dict[str, str] = {}
        self.preload_object_meta_info: dict[str, dict[str, bool | float | None]] = {}
        self.meta_to_fine_projection: dict[str, str] = {}


class Scene:
    def __init__(self, scene_config: SceneConfig):
        self.scene_config = scene_config
        self.object_list: dict[str, XFormPrim] = {}
        self.robot_list: list[BaseEmbodiment] = []
        self.camera_list: dict[str, Camera] = {}
        self.metric_manager: MetricsManager
        self.articulation_list: dict[str, Articulation] = {}
        self.articulation_part_list: dict[str, XFormPrim] = {}
        self.background: dict[str, OmniPBR] = {}
        self.assets_library: AssetsLibrary = AssetsLibrary()
        self.cache_library: CacheLibrary = CacheLibrary()
        self.scene_xform: XFormPrim = None
        self.uuid: str = ""
        self.meta_infos: dict = {}
        self.world: World = World()

    def _initialize_load_infomation(self) -> None:
        """
        Initialize the infomation of the scene.
        - assets_library: assets path information
        - object_pool: object annotation data
        - articulation_data: articulation annotation data
        """
        self.assets_library.initialize(self.default_config["ASSETS_DIR"])
        self.object_pool = load_object_pool(
            self.scene_config, self.default_config["current_dir"]
        )
        self.articulation_data = load_articulation_data(
            self.scene_config, self.default_config["current_dir"]
        )

    def _initialize_before_reset(
        self,
        physics_dt: float = 1 / 60.0,
        rendering_dt: float = 1 / 60.0,
        is_render: bool = False,
        only_depth_rep_for_camera: bool = False,
        only_color_rep_for_camera: bool = False,
    ) -> None:
        self.scene_xform, self.uuid = load_world_xform_prim(
            os.path.join(
                self.default_config["ASSETS_DIR"],
                f"{self.scene_config.usd_name}.usda",
            )
        )
        self.world = World(physics_dt=physics_dt, rendering_dt=rendering_dt)
        setup_physics_scene(self.scene_config.physics_scene_config)

        # TODO: HACK, remove room collider to avoid error physics simulation
        from genmanip.utils.usd_utils.collision_utils import remove_colliders

        remove_colliders(f"/World/{self.uuid}/room")

        # Get object list
        self.object_list = get_object_list(
            self.uuid, self.scene_xform, self.scene_config.table_uid
        )
        self.meta_infos["world_pose_list"] = collect_world_pose_list(self.object_list)
        capture_particle_layout_references(self)

        # Load articulation data
        self.articulation_list = {}
        self.articulation_part_list = {}
        self._parse_articulation()

        # Create robot list
        self.robot_list = [
            RobotFactory.build(
                robot_config.type,
                scene_uid=self.uuid,
                default_config=self.default_config,
                robot_config=robot_config,
            )
            for robot_config in self.scene_config.robots
        ]
        for robot in self.robot_list:
            self.world.scene.add(robot.robot)

        # Load camera information
        camera_info = self.scene_config.domain_randomization.cameras
        if camera_info.type == "fixed":
            camera_data = load_yaml(
                os.path.join(
                    self.default_config["current_dir"], camera_info.config_path
                )
            )
        else:
            raise ValueError(f"Unsupported camera type: {camera_info.type}")
        if is_render:
            if "camera1" in camera_data:
                camera_data["camera1"]["resolution"] = [640, 480]

        # Create camera list
        self.camera_list = create_camera_list(
            camera_data,
            self.uuid,
            rendering_dt,
            only_depth_rep_for_camera,
            only_color_rep_for_camera,
        )

    def _parse_articulation(self) -> None:
        for key in self.scene_config.object_config:
            if (
                self.scene_config.object_config[key].type == "existed_object"
                and self.scene_config.object_config[key].is_articulated
            ):
                for uid in self.scene_config.object_config[key].uid_list:
                    info = {}
                    info["target_positions"] = self.scene_config.object_config[
                        key
                    ].target_positions
                    info["is_articulated"] = self.scene_config.object_config[
                        key
                    ].is_articulated
                    self.scene_config.generation_config.articulation[uid] = info
                    if (
                        self.scene_config.object_config[key].articulation_info
                        is not None
                    ):
                        self.articulation_data[uid] = self.scene_config.object_config[
                            key
                        ].articulation_info
                    if self.articulation_data[uid]["is_articulated"]:
                        self.articulation_list[uid] = add_articulation_to_scene(
                            key, self.uuid, self.world
                        )
                    else:
                        self.articulation_list[uid] = self.object_list[uid]

        self.world.reset()

        for key, articulation in self.articulation_list.items():
            if self.articulation_data[key]["is_articulated"]:
                articulation._articulation_view.initialize()
        self.world.initialize_physics()

        for arti_id, articulation in self.articulation_list.items():
            if self.articulation_data[arti_id]["is_articulated"]:
                if (
                    arti_id in self.scene_config.generation_config.articulation
                    and self.scene_config.generation_config.articulation[arti_id][
                        "target_positions"
                    ]
                    is not None
                ):
                    articulation._articulation_view.set_joint_positions(
                        self.scene_config.generation_config.articulation[arti_id][
                            "target_positions"
                        ]
                    )

        for _ in range(10):
            self.world.step(render=False)

        for arti_id, articulation in self.articulation_list.items():
            arti_parts = self.articulation_data[arti_id]["part"]
            arti_prim = self.object_list[arti_id]
            arti_prim_path = arti_prim.prim_path
            for part_name, part_group in arti_parts.items():
                if part_group[0] != "/":
                    part_prim_path = arti_prim_path + f"/Instance/{part_group}"
                else:
                    part_prim_path = arti_prim_path + part_group
                arti_part = f"{arti_id}_{part_name}"
                self.object_list[arti_part] = relate_object_from_data(part_prim_path)
                self.articulation_part_list[arti_part] = relate_object_from_data(
                    part_prim_path
                )
            self.object_list.pop(arti_id)

    def _initialize_after_reset(
        self,
        is_render: bool = False,
        save_pointcloud: bool = False,
    ) -> None:

        for robot, robot_cfg in zip(self.robot_list, self.scene_config.robots):
            robot.initialize(default_joint_positions=robot_cfg.default_joint_positions)
        for key, articulation in self.articulation_list.items():
            if self.articulation_data[key]["is_articulated"]:
                articulation._articulation_view.initialize()
        if not is_render or save_pointcloud:
            self.cache_library.mesh_dict = objectList2meshList(
                self.object_list,
                os.path.join(
                    self.default_config["ASSETS_DIR"],
                    "mesh_data",
                    self.scene_config.task_name,
                ),
            )
        if self.scene_config.domain_randomization.random_environment.has_wall:
            self.background["wall"], self.background["wall_textures"] = (
                setup_walls_and_materials(self.uuid, self.world, self.object_list)
            )
        else:
            self.background["wall"] = None
            self.background["wall_textures"] = None
        self.tcp_configs = {}
        if (
            self.robot_list[0].embodiment_name == "franka"
            and self.robot_list[0].gripper_name == "robotiq"
        ):
            self.tcp_configs["franka"] = load_yaml(
                os.path.join(
                    self.default_config["current_dir"],
                    "configs/robots/tcp/franka_robotiq_tcp.yaml",
                )
            )
        else:
            self.tcp_configs["franka"] = load_yaml(
                os.path.join(
                    self.default_config["current_dir"],
                    "configs/robots/tcp/franka_tcp.yaml",
                )
            )
        reset_object_xyz(self.object_list, self.meta_infos["world_pose_list"])
        for key in self.object_list:
            clean_prim_velocity(self.object_list[key].prim_path)

    def _preprocess_scene(self) -> None:
        preprocess_scene(self, self.scene_config)

    def _warmup_world(self, physics_steps: int = 100) -> None:
        while any(
            camera._custom_annotators["distance_to_image_plane"] is not None
            and get_src(camera, "depth") is None
            for camera in self.camera_list.values()
        ):
            self.world.step()
        for _ in range(physics_steps):
            self.world.step(render=False)

    def collect_meta_infos(self) -> None:
        self.meta_infos["world_pose_list"] = collect_world_pose_list(self.object_list)
        self.meta_infos["articulation_pose_list"] = collect_articulation_list(
            self, self.articulation_list
        )
        self.meta_infos["robot_pose_list"] = [
            robot.robot.get_world_pose() for robot in self.robot_list
        ]
        self.meta_infos["robot_tcp_list"] = [
            robot.fk_single(robot.robot.get_joint_positions())
            for robot in self.robot_list
        ]
        self.meta_infos["joint_positions"] = [
            robot.robot.get_joint_positions() for robot in self.robot_list
        ]
        self.meta_infos["joint_velocities"] = [
            robot.robot.get_joint_velocities() for robot in self.robot_list
        ]

    def initialize(
        self,
        default_config: dict,
        physics_dt: float = 1 / 60.0,
        rendering_dt: float = 1 / 60.0,
        is_render: bool = False,
        save_pointcloud: bool = False,
        only_depth_rep_for_camera: bool = False,
        only_color_rep_for_camera: bool = False,
    ):
        self.default_config = default_config
        self._initialize_load_infomation()
        self._initialize_before_reset(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            is_render=is_render,
            only_depth_rep_for_camera=only_depth_rep_for_camera,
            only_color_rep_for_camera=only_color_rep_for_camera,
        )
        self.world.reset()
        self._initialize_after_reset(
            is_render=is_render,
            save_pointcloud=save_pointcloud,
        )
        self._preprocess_scene()

    def post_initialize(self, skip_warmup: bool = False) -> None:
        if not skip_warmup:
            self._warmup_world()
        self.collect_meta_infos()

    def build_metrics_manager(
        self,
        goal: list,
        skip_steps: int = 1,
        succ_cnts: int = 0,
        never_reset: bool = False,
    ):
        ori_goal = goal.copy()

        def _transform_goal(goal: list | dict):
            if isinstance(goal, list):
                return [_transform_goal(g) for g in goal]
            elif isinstance(goal, dict):
                return {
                    "type": goal.get(
                        "type", "manip/default/sr_based_genmanip_relationship"
                    ),
                    "skip_steps": goal.get("skip_steps", skip_steps),
                    "succ_cnts": goal.get("succ_cnts", succ_cnts),
                    "never_reset": goal.get("never_reset", never_reset),
                    "sub_goal_setting": goal,
                }

        self.metric_manager = MetricsManager(_transform_goal(ori_goal))

    def step(self, render: bool = True) -> float | None:
        self.world.step(render=render)
        score = 0.0
        if self.metric_manager is not None:
            score = self.metric_manager.step(self)
            if os.environ.get("GENMANIP_DEBUG", "0") == "1":
                print("Score: ", score)
        return score

    def get_meta_infos(self) -> dict:
        return self.meta_infos

    def get_scene_config(self):
        return self.scene_config
