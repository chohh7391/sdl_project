from typing import Dict, List, Any, Tuple, Optional
from curobo.geom.types import Obstacle
from curobo.types.base import TensorDeviceType
from cutamp.envs import TAMPEnvironment
from cutamp.tamp_domain import HandEmpty, On


def load_rearrange_env(
    entities: Dict[str, Any],
    movables: List[Obstacle],
    statics: List[Obstacle],
    ex_collision: List[Obstacle],
    rearrange_grid: str | None = None,
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> TAMPEnvironment:
    """물체(target_object)에 따라 다른 위치로 치우는(rearrange) 환경"""
    
    # 1. 타겟 물체의 이름 확인
    target_name = movables[0].name

    # calcalate rearrange position from grid name
    if rearrange_grid is not None:
        rearrange_x, rearrange_y = get_grid_xy(rearrange_grid)
        entities["rearrange_region"].pose = [rearrange_x, rearrange_y, 0.02, 1.0, 0.0, 0.0, 0.0]
    
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

def get_grid_xy(name: str) -> Tuple[Optional[float], Optional[float]]:
    """
    그리드 이름을 입력받아 해당 영역의 중심 좌표 (x, y)를 반환합니다.
    """
    # G1: x(-0.3 ~ 0), y(-0.75 ~ -0.45) -> center(-0.15, -0.6)
    if name == "G1":
        return -0.15, -0.6
    
    # G2: x(-0.2 ~ -0.1), y(0.55 ~ 0.65) -> center(-0.15, 0.6)
    elif name == "G2":
        return -0.15, 0.6

    # x-range: 0.0 ~ 0.3 (center 0.15)
    elif name == "G3": return 0.15, -0.6
    elif name == "G4": return 0.15, -0.3
    elif name == "G5": return 0.15, 0.0
    elif name == "G6": return 0.15, 0.3
    elif name == "G7": return 0.15, 0.6

    # x-range: 0.3 ~ 0.6 (center 0.45)
    elif name == "G8":  return 0.45, -0.6
    elif name == "G9":  return 0.45, -0.3
    elif name == "G10": return 0.45, 0.0
    elif name == "G11": return 0.45, 0.3
    elif name == "G12": return 0.45, 0.6

    # 정의되지 않은 이름일 경우
    return None, None
