from typing import Any
from pydantic import BaseModel, Field

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory


class TitrationExperimentConfig(BaseModel):
    titration_station_prim_path: str = Field(
        ..., description="Prim path of the titration station object"
    )
    condition_type: str = Field(..., description="condition type")
    require_physical_completion: bool = False


@MetricFactory.register("manip/scientific_experiments/task_10_titration_experiment")
class TitrationExperiment(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = TitrationExperimentConfig(**sub_goal_setting)
        self._joint_dof_index = None

    def check_status(self, scene):
        titration_station_prim = scene.world.stage.GetPrimAtPath(
            self.setting.titration_station_prim_path
        )

        if self.setting.require_physical_completion:
            from genmanip.extensions.metrics.scientific_experiments.repair_checks import (
                runtime_ok,
            )

            if not runtime_ok(scene):
                return False
            from genmanip.extensions.metrics.scientific_experiments.repair_checks import (
                valid_titration_completion,
            )

            def read(name):
                attr = titration_station_prim.GetAttribute("titration:" + name)
                return attr.Get() if attr else None

            if read("pbd_valid") is not True or read("pbd_error"):
                return False
            if self.setting.condition_type == "check_success":
                return valid_titration_completion(
                    {
                        name: read(name)
                        for name in (
                            "pbd_valid",
                            "pbd_error",
                            "task_success",
                            "completion_volume_ml",
                            "completion_hold_seconds",
                        )
                    }
                )

        if self.setting.condition_type == "open_valve":
            valve_state = titration_station_prim.GetAttribute(
                "titration:valve_state"
            ).Get()
            return valve_state in ("DRIP", "FINE", "OPEN")

        if self.setting.condition_type == "close_valve":
            valve_state = titration_station_prim.GetAttribute(
                "titration:valve_state"
            ).Get()
            return valve_state == "CLOSED"

        if self.setting.condition_type == "check_success":
            indicator_phase = titration_station_prim.GetAttribute(
                "titration:indicator_phase"
            ).Get()
            return indicator_phase == "endpoint_pale_pink"

        return False
