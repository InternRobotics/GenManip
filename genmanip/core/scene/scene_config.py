from typing import Any
import numpy as np
from pydantic import BaseModel, Field, model_validator


class CameraConfig(BaseModel):
    config_path: str = Field(..., description="Path to the camera configuration")
    type: str = Field(..., description="Type of the camera")


class RobotConfig(BaseModel):
    type: str = Field(..., description="Type of the robot")
    usd_path: str | None = Field(
        default=None, description="Optional robot USD path relative to ASSETS_DIR"
    )
    default_joint_positions: list[float] | None = Field(
        default=None, description="Default joint positions"
    )
    position: list[float] | None = Field(
        default=None, description="Position of the robot"
    )
    orientation: list[float] | None = Field(
        default=None, description="Orientation of the robot"
    )


class RobotBasePositionConfig(BaseModel):
    random_range: float = Field(
        default=0.05, description="Random range of the robot base position"
    )


class RandomVisualConfig(BaseModel):
    prim_path: str = Field(default="", description="Prim path of the visual")
    type: str = Field(default="", description="Type of the visual")
    assets_pattern: str = Field(default="", description="Assets pattern of the visual")


class MaterialOverrideConfig(BaseModel):
    prim_path: str = Field(..., description="Absolute path of the shader prim")
    attribute: str = Field(
        default="inputs:diffuseColor",
        description="Shader color attribute to override",
    )
    value: tuple[float, float, float] = Field(..., description="RGB color value")


class RandomEnvironmentConfig(BaseModel):
    has_wall: bool = Field(default=False, description="Whether to have a wall")
    hdr: bool = Field(default=False, description="Whether to have a HDR environment")
    robot_base_position: bool | RobotBasePositionConfig = Field(
        default=False, description="Whether to have a random robot base position"
    )
    robot_eepose: bool = Field(
        default=False, description="Whether to have a random robot end-effector pose"
    )
    table_texture: bool = Field(
        default=False, description="Whether to have a random table texture"
    )
    table_type: bool = Field(
        default=False, description="Whether to have a random table type"
    )
    wall_texture: bool = Field(
        default=False, description="Whether to have a random wall texture"
    )
    room_randomization: bool = Field(
        default=False, description="Whether to have a random room environment"
    )
    random_visuals: list[RandomVisualConfig] = Field(
        default=[], description="List of random visuals"
    )


class ActionPathConfig(BaseModel):
    mode: str = Field(default="auto", description="Mode of the action path")
    robot: int = Field(default=0, description="Index of the robot")
    actions: list[dict] | None = Field(default=None, description="List of actions")


class ObjectConfig(BaseModel):
    type: str = Field(..., description="Type of the object")
    uid_list: list[str] = Field(default=[], description="List of object UUIDs")
    option: list[str] = Field(default=[], description="List of options")
    fixed_size: float | list[float] | None = Field(
        default=None, description="Fixed size of the object"
    )
    fixed_scale: float | list[float] | None = Field(
        default=None, description="Fixed scale of the object"
    )
    relative_scale: float | list[float] | None = Field(
        default=None, description="Relative scale of the object"
    )
    clip_range: dict[str, float] | None = Field(
        default=None, description="Clip range of the object"
    )

    # For load object from path and add additional object from path
    path: str = Field(default="", description="Path to the object")
    filter_rule: list[str] = Field(default=[], description="Filter rule of the object")
    max_cached_num: int = Field(
        default=10**9, description="Max cached number of the object"
    )
    uid: str = Field(default="", description="UUID of the object")
    replace_existed_object: list[str] = Field(
        default=[], description="List of replaced object UUIDs"
    )

    # Physical properties
    without_colliders: bool = Field(
        default=False, description="Whether to have no colliders"
    )
    is_not_rigid: bool = Field(
        default=False, description="Whether the object is not rigid"
    )
    mass: float | None = Field(default=None, description="Mass of the object")

    # For articulated object
    is_articulated: bool = Field(
        default=False, description="Whether the object is articulated"
    )
    target_positions: list[float] | None = Field(
        default=None, description="Target positions of the object"
    )
    articulation_info: dict | None = Field(
        default=None, description="Articulation information of the object"
    )
    reset_joint_positions: bool = Field(
        default=True,
        description="Whether to restore the articulation joint positions during scene reset",
    )


class GenerationConfig(BaseModel):
    action_path: ActionPathConfig = Field(
        default=ActionPathConfig(), description="Action path configurations"
    )
    goal: list = Field(default=[[]], description="List of goals")
    articulation: dict = Field(default={}, description="List of articulation names")
    mode: str = Field(default="manual", description="Mode of the generation")
    planner: str = Field(default="curobo", description="Planner of the generation")
    is_shuffle: bool = Field(
        default=False, description="Whether to shuffle the action list"
    )
    update_planner: bool = Field(
        default=False, description="Whether to update the planner"
    )
    smooth: bool = Field(default=False, description="Whether to smooth the motion")
    reset_tcp: tuple[list[float], list[float]] | float | bool | None = Field(
        default=None, description="Reset TCP"
    )
    aug_distance: float = Field(default=0.0, description="Augmented distance")
    randomization_hack_flag: bool = Field(
        default=False, description="Randomization hack flag"
    )


class DomainRandomizationConfig(BaseModel):
    cameras: CameraConfig = Field(..., description="Camera configurations")
    object_data_path: str | None = Field(
        default=None, description="Path to the object data"
    )
    articulation_data_path: str | None = Field(
        default=None, description="Path to the articulation data"
    )
    random_environment: RandomEnvironmentConfig = Field(
        default=RandomEnvironmentConfig(),
        description="Random environment configurations",
    )
    rewrite_instruction: bool = Field(
        default=False, description="Whether to rewrite the instruction"
    )
    rewrite_sequece: list[int] | None = Field(
        default=None, description="Sequence of the rewrite instruction"
    )
    record_arm_info: bool = Field(
        default=False, description="Whether to record the arm information"
    )
    camera_randomization: dict[str, dict] | None = Field(
        default=None, description="Camera randomization configurations"
    )
    material_overrides: list[MaterialOverrideConfig] = Field(
        default=[], description="Fixed material overrides applied after randomization"
    )


class LayoutConfig(BaseModel):
    ignored_objects: list[str] = Field(
        default=[], description="List of ignored objects"
    )
    type: str | None = Field(default=None, description="Type of the layout")
    partial_ignore: dict = Field(default={}, description="Partial ignore of the layout")

    # For random all range
    random_range_x: float = Field(default=-0.5 * np.inf, description="Random range x")
    random_range_y: float = Field(default=-0.5 * np.inf, description="Random range y")
    random_range_w: float = Field(default=0.5 * np.inf, description="Random range w")
    random_range_h: float = Field(default=0.5 * np.inf, description="Random range h")
    random_range_angle: float = Field(default=0.0, description="Random range angle")
    force_no_check: bool = Field(
        default=False,
        description="Force random_obj1_range placement to succeed without validity checks",
    )

    # For random centric range
    angle_bilateral: bool = Field(
        default=False, description="Whether to have a bilateral angle"
    )
    angle: float = Field(default=0.0, description="Angle of the random centric range")
    w: float = Field(default=0.0, description="Width of the random centric range")
    h: float = Field(default=0.0, description="Height of the random centric range")

    # For random custom tableset
    custom_tableset: dict[str, dict] | list[dict[str, dict]] | None = Field(
        default=None, description="Custom tableset"
    )
    in_order: bool = Field(
        default=False, description="Whether to in order the custom tableset"
    )

    # For scene graph placement
    scene_graph: list[dict] = Field(default=[], description="Scene graph")


class SceneConfig(BaseModel):
    task_name: str = Field(..., description="Name of the task")
    usd_name: str = Field(..., description="Name of the USD file")
    table_uid: str = Field(..., description="UUID of the table")
    mode: str = Field(default="manual", description="Mode of the scene")
    robots: list[RobotConfig] = Field(..., description="List of robot configurations")
    domain_randomization: DomainRandomizationConfig = Field(
        ..., description="Domain randomization configurations"
    )
    generation_config: GenerationConfig = Field(
        default=GenerationConfig(), description="Generation configurations"
    )
    object_config: dict[str, ObjectConfig] = Field(
        default={}, description="Object configurations"
    )
    physics_scene_config: dict[str, Any] = Field(
        default={}, description="physicsScene configurations"
    )
    particle_layout_references: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Particle parent prim paths mapped to object IDs whose initial XY "
            "layout translation was shared with those parents during recording"
        ),
    )
    preprocess_config: list[dict] = Field(
        default=[], description="Preprocess configurations"
    )
    layout_config: LayoutConfig = Field(
        default=LayoutConfig(), description="Layout configurations"
    )
    restart_per_success: int = Field(default=10**9, description="Restart per success")
    restart_per_failed: int = Field(default=10**9, description="Restart per failed")
    num_episode: int | None = Field(default=None, description="Number of episodes")
    num_test: int | None = Field(default=None, description="Number of tests")
    num_steps: int = Field(default=600, description="Number of steps")
    preserve_score_on_invalid_state: bool = Field(
        default=False,
        description="Keep the existing metric score when invalid robot state ends evaluation",
    )
    warmup_steps: int = Field(
        default=50, ge=0, description="Number of warmup steps before evaluation"
    )
    instruction: str = Field(default="", description="Instruction")
    physics_dt: float = Field(default=1 / 30, description="physics_dt")
    rendering_dt: float = Field(default=1 / 30, description="physics_dt")
    env_vars: dict[str, str] = Field(default={}, description="Environment variables")

    @model_validator(mode="after")
    def fill_physics_scene_defaults(self):
        DEFAULT_PHYSICS_SCENE_CONFIG = {
            "EnableGPUDynamics": False,
            "EnableStabilization": True,
            "EnableCCD": False,
            "BroadphaseType": "GPU",
            "SolverType": "TGS",
            "GpuTotalAggregatePairsCapacity": 10 * 1024 * 1024,
            "GpuFoundLostAggregatePairsCapacity": 10 * 1024 * 1024,
        }
        for k, v in DEFAULT_PHYSICS_SCENE_CONFIG.items():
            self.physics_scene_config.setdefault(k, v)
        return self
