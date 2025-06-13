from modular_framework.module_client.prompt_requests import request_server
from modular_framework.module_client.utils import process_check_finished
from configs import Config

debug = Config.debug_text


class ViLAAgent:
    def __init__(
        self,
        planner,
        uuid=None,
        retry_limit=3,
        model_name="gpt-4o-2024-05-13",
        P2P=True,
        CtoF=True,
    ) -> None:
        self.subtasks = []
        self.planner = planner
        self.uuid = "ViLAAgent" if uuid is None else uuid
        self.is_finished = False
        self.retry_limit = retry_limit
        self.model_name = model_name
        self.P2P = P2P
        self.CtoF = CtoF
        self.retry_cnt = 0

    def initialize(self, instruction):
        self.instruction = instruction

    def get_next_step(self, color, depth, joint_position):
        if debug:
            print("status: ", self.planner.status)
        if self.retry_cnt > self.retry_limit:
            print("Out of Limit!")
            self.is_finished = True
        if self.is_finished:
            return joint_position
        if self.planner.status != "finished" or len(self.planner.action_list) > 0:
            print(f"Planner finished stage {self.planner.status}")
            return self.planner.get_next_stage(joint_position)
        elif len(self.subtasks) > 0:
            current_task = self.subtasks.pop(0)
            if debug:
                print("current subtask: ", current_task)
            config = {}
            config["P2P"] = self.P2P
            config["CtoF"] = self.CtoF
            config["model_name"] = self.model_name
            prompt_response = request_server(
                self.uuid, color, current_task, type="prompt_pipeline", config=config
            )
            self.planner.initialize(prompt_response, color, depth)
            return joint_position
        else:
            config = {}
            config["P2P"] = self.P2P
            config["CtoF"] = self.CtoF
            config["model_name"] = self.model_name
            self.is_finished = process_check_finished(
                request_server(
                    self.uuid,
                    color,
                    self.instruction,
                    type="check_finished",
                    config=config,
                )
            )
            if not self.is_finished:
                config = {}
                config["P2P"] = self.P2P
                config["CtoF"] = self.CtoF
                config["model_name"] = self.model_name
                self.subtasks = request_server(
                    self.uuid, color, self.instruction, type="task_split", config=config
                )["result"]["subtasks"]
            self.retry_cnt += 1
            return joint_position

    def get_status(self):
        return len(self.subtasks)

    def finished(self):
        return self.is_finished
