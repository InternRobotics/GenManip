from typing import Any
from pydantic import BaseModel, Field

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.scientific_experiments.task09_readout import (
    readout_in_range,
)
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
    calculate_distance_between_two_point_clouds,
    is_inside,
)


CONTACT_TOLERANCE = 0.05
LOWER_LIMIT_OF_JOINT_THRESHOLD = 0.001


class SolidWeighingConfig(BaseModel):
    weighing_bottle_body_uid: str = Field(
        ..., description="Prim name or path of the weighing bottle body object"
    )
    analytical_balance_Chassis_uid: str = Field(
        ..., description="Prim name or path of the analytical_balance_Chassis object"
    )
    articulation_obj_uid: str = Field(..., description="uid of the articulation root")
    joint_name: str = Field(..., description="DOF name of the articulation joint")
    condition_type: str = Field(..., description="condition type")
    require_supported_inventory: bool = False
    require_empty: bool = False
    require_stable: bool = True
    reading_range_g: tuple[float, float] | None = Field(
        default=None,
        description="Inclusive displayed gram bounds; None preserves legacy scoring",
    )
    sample_uids: list[str] = Field(
        default_factory=lambda: [f"grain_{i:02d}" for i in range(1, 6)]
    )


@MetricFactory.register("manip/scientific_experiments/task_09_solid_weighing")
class SolidWeighing(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = SolidWeighingConfig(**sub_goal_setting)
        self._joint_dof_index = None

    def check_status(self, scene):
        if self.setting.require_supported_inventory:
            from genmanip.extensions.metrics.scientific_experiments.repair_checks import (
                runtime_ok,
            )

            if not runtime_ok(scene):
                return False
            import json

            balance = scene.world.stage.GetPrimAtPath(
                f"/World/{scene.uuid}/obj_{self.setting.articulation_obj_uid}"
            )

            def read(name):
                attr = balance.GetAttribute("balance:" + name) if balance else None
                return attr.Get() if attr else None

            paths = read("supported_paths") or []
            if isinstance(paths, str):
                paths = json.loads(paths)
            bottle = scene.object_list.get(self.setting.weighing_bottle_body_uid)
            supported = bool(bottle and str(bottle.prim_path) in paths)
            valid = read("valid") is True
            if self.setting.condition_type == "not_weighing":
                return not supported and read("valid_empty_bottle_tare") is True
            if not (supported and valid):
                return False
            if (
                self.setting.require_empty
                or self.setting.condition_type == "touch_button"
            ):
                objects = [
                    self.setting.weighing_bottle_body_uid,
                    *self.setting.sample_uids,
                ]
                clouds = SRBasedGenmanipRelationship.get_target_pc_list(scene, objects)
                container = clouds.get(self.setting.weighing_bottle_body_uid)
                if container is None or not len(container):
                    return False
                for uid in self.setting.sample_uids:
                    points = clouds.get(uid)
                    if points is None or not len(points):
                        return False
                    if is_inside(
                        src_pts=points, target_pts=container, thresh=0.1, scene=scene
                    ):
                        return False
            if self.setting.condition_type == "touch_button":
                return (
                    read("valid_empty_bottle_tare") is True
                    and (read("tare_event_count") or 0) > 0
                )
            if self.setting.condition_type == "weighing":
                if self.setting.require_empty:
                    return True
                if self.setting.reading_range_g is not None:
                    readout = read("lcd_readout")
                    if not (
                        read("tared") is True
                        and read("valid_empty_bottle_tare") is True
                        and (read("tare_event_count") or 0) > 0
                        and not read("error")
                        and readout_in_range(readout, self.setting.reading_range_g)
                    ):
                        return False
                    # Keep the accepted display value in the native log; camera
                    # frames can lag a rapidly changing device display.
                    snapshot = {
                        "lcd_readout": readout,
                        "range_g": list(self.setting.reading_range_g),
                        "stable": read("stable"),
                        "valid": valid,
                        "bottle_supported": supported,
                        "metric_step": self._step_cnt,
                        "world_step": getattr(
                            scene.world, "current_time_step_index", None
                        ),
                    }
                    print(
                        "TASK09_READOUT_RANGE_PASS " + json.dumps(snapshot),
                        flush=True,
                    )
                return not self.setting.require_stable or read("stable") is True
            return False

        target_uids = [
            self.setting.weighing_bottle_body_uid,
            self.setting.analytical_balance_Chassis_uid,
        ]
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)
        weighing_bottle_vertices = pclist.get(self.setting.weighing_bottle_body_uid)
        analytical_balance_vertices = pclist.get(
            self.setting.analytical_balance_Chassis_uid
        )

        if (
            weighing_bottle_vertices is None
            or analytical_balance_vertices is None
            or len(weighing_bottle_vertices) == 0
            or len(analytical_balance_vertices) == 0
        ):
            return False

        if self.setting.condition_type == "weighing":
            is_above_balance = (
                weighing_bottle_vertices[:, 2].min()
                >= analytical_balance_vertices[:, 2].max() - CONTACT_TOLERANCE
            )
            distance = calculate_distance_between_two_point_clouds(
                weighing_bottle_vertices,
                analytical_balance_vertices,
                scene=scene,
            )
            return bool(is_above_balance and distance <= CONTACT_TOLERANCE)

        if self.setting.condition_type == "not_weighing":
            no_x_overlap = (
                weighing_bottle_vertices[:, 0].max()
                < analytical_balance_vertices[:, 0].min()
                or analytical_balance_vertices[:, 0].max()
                < weighing_bottle_vertices[:, 0].min()
            )
            no_y_overlap = (
                weighing_bottle_vertices[:, 1].max()
                < analytical_balance_vertices[:, 1].min()
                or analytical_balance_vertices[:, 1].max()
                < weighing_bottle_vertices[:, 1].min()
            )
            return bool(no_x_overlap or no_y_overlap)

        if self.setting.condition_type == "touch_button":
            _articulation = scene.articulation_list[self.setting.articulation_obj_uid]
            if self._joint_dof_index is None:
                if self.setting.joint_name not in _articulation.dof_names:
                    raise RuntimeError(
                        f"Joint '{self.setting.joint_name}' not found in articulation, available joints: {_articulation.dof_names}"
                    )

                self._joint_dof_index = _articulation.get_dof_index(
                    self.setting.joint_name
                )

            joint_position = _articulation.get_joint_positions(self._joint_dof_index)[0]
            return joint_position > LOWER_LIMIT_OF_JOINT_THRESHOLD

        return False
