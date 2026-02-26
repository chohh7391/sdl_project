from typing import Optional, Dict, List, Any
import torch
import numpy as np
from curobo.geom.types import Cuboid, Cylinder, Obstacle
from curobo.types.base import TensorDeviceType
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import unit_quat
from cutamp.tamp_domain import HandEmpty, On
from cutamp.utils.shapes import MultiSphere

def load_Rearranging_env(
    entities: Dict[str, Any],
    movables: List[Obstacle],
    statics: List[Obstacle],
    ex_collision: List[Obstacle],
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> TAMPEnvironment:
    """물체(target_object)에 따라 다른 위치로 치우는(rearrange) 환경"""
    
    # 1. 타겟 물체의 이름 확인
    target_name = movables[0].name

    # 2. 타겟 물체에 따라 rearrange_region의 목표 위치(pose) 다르게 설정
    # pose 배열: [x, y, z, q_w, q_x, q_y, q_z] 형태 (위치 + 쿼터니언 회전)
    if target_name == "beaker":
        # 비커를 치워둘 임시 위치 (예시 좌표)
        entities["rearrange_region"].pose = [0.3, 0.3, 0.085, 1.0, 0.0, 0.0, 0.0]
    elif target_name == "flask":
        # 플라스크를 치워둘 임시 위치 (예시 좌표)
        entities["rearrange_region"].pose = [0.22, 0.2, 0.085, 1.0, 0.0, 0.0, 0.0]
    elif target_name == "box":
        # 박스를 치워둘 임시 위치 (예시 좌표)
        entities["rearrange_region"].pose = [0.4, 0.0, 0.085, 1.0, 0.0, 0.0, 0.0]
    # else:
    #     # 그 외 물체들의 기본 치우기 위치
    #     entities["rearrange_region"].pose = [0.25, 0.25, 0.085, 1.0, 0.0, 0.0, 0.0]

    # 3. TAMP 환경 정의
    env = TAMPEnvironment(
        name="rearrange",
        movables=movables,
        statics=statics,
        ex_collision=ex_collision,
        type_to_objects={
            "Movable": movables,
            "Surface": [entities["table"], entities["rearrange_region"]],
            "ExCollision": [entities["rearrange_region"]]
        },
        goal_state=frozenset(
            {
                HandEmpty.ground(),
                On.ground(target_name, entities["rearrange_region"].name), 
            }
        )
    )

    return env