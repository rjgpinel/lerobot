# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Protocol

from lerobot.common.robot_devices.controllers.configs import (
    ControllerConfig,
    AudioGripperControllerConfig,
)


class Controller(Protocol):
    def connect(self): ...
    def disconnect(self): ...
    def get_command(self): ...
    def stop(self): ...


def make_controllers_config(controller_type: str, **kwargs) -> ControllerConfig:
    if controller_type == "audiogripper":
        return AudioGripperControllerConfig(**kwargs)
    else:
        raise ValueError(f"The controller type '{controller_type}' is not valid.")
    
def make_controller_from_config(configs: dict[str, ControllerConfig]) -> list[Controller]:
    controllers = {}
    for key, cfg in configs.items():
        if cfg.type == "audiogripper":
            from lerobot.common.robot_devices.controllers.audiogripper import AudioGripperController

            controllers[key] = AudioGripperController(cfg)

        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return controllers

def make_controllers(controller_type: str, **kwargs) -> list[Controller]:
    config = make_controllers_config(controller_type, **kwargs)
    return make_controller_from_config(config)
