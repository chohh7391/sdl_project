from typing import List, Optional
import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper

class SlowParallelGripper(ParallelGripper):
    """
    ParallelGripper를 상속받아, open/close 명령 시 즉시 이동하지 않고
    update()가 호출될 때마다 정해진 step 수(기본 3회)에 걸쳐 부드럽게 이동하는 그리퍼.
    """

    def __init__(
        self,
        end_effector_prim_path: str,
        joint_prim_names: List[str],
        joint_opened_positions: np.ndarray,
        joint_closed_positions: np.ndarray,
        action_deltas: np.ndarray = None,
        use_mimic_joints: bool = False,
        steps_to_execute: int = 3,  # 3번에 나눠서 실행
    ) -> None:
        super().__init__(
            end_effector_prim_path=end_effector_prim_path,
            joint_prim_names=joint_prim_names,
            joint_opened_positions=joint_opened_positions,
            joint_closed_positions=joint_closed_positions,
            action_deltas=action_deltas,
            use_mimic_joints=use_mimic_joints,
        )
        
        # 내부 상태 변수
        self._steps_total = steps_to_execute
        self._steps_remaining = 0
        self._current_step_delta = None
        self._target_direction = 0  # 1: Open, -1: Close
        
        # FR5.initialize()에서 호출될 때 함수들이 세팅됩니다.
        self._articulation_apply_action_func = None
        self._get_joint_positions_func = None

    def initialize(
        self,
        articulation_apply_action_func,
        get_joint_positions_func,
        set_joint_positions_func,
        dof_names,
        physics_sim_view=None,
    ) -> None:
        """
        FR5.initialize()에서 호출되어 로봇의 제어 함수들을 연결합니다.
        """
        super().initialize(
            articulation_apply_action_func=articulation_apply_action_func,
            get_joint_positions_func=get_joint_positions_func,
            set_joint_positions_func=set_joint_positions_func,
            dof_names=dof_names,
            physics_sim_view=physics_sim_view,
        )
        self._articulation_apply_action_func = articulation_apply_action_func
        self._get_joint_positions_func = get_joint_positions_func

    def open(self) -> None:
        """
        Open 명령을 예약합니다. 실제 움직임은 update()에서 발생합니다.
        """
        self._prepare_movement(direction=1) # 1 for Open

    def close(self) -> None:
        """
        Close 명령을 예약합니다. 실제 움직임은 update()에서 발생합니다.
        """
        self._prepare_movement(direction=-1) # -1 for Close

    def _prepare_movement(self, direction: int) -> None:
        """이동을 위한 델타값 계산 및 상태 초기화"""
        if self._action_deltas is None:
            # 델타가 없으면 기본 동작(즉시 이동)을 위해 부모 메서드 호출하려 했으나,
            # 여기서는 구조상 그냥 무시하거나 경고를 띄우는 것이 안전함
            return

        self._steps_remaining = self._steps_total
        self._target_direction = direction
        
        # 전체 이동해야 할 델타를 스텝 수로 나눔
        # action_deltas는 [finger_1_delta, finger_2_delta] 형태
        self._current_step_delta = self._action_deltas / float(self._steps_total)

    def update(self) -> None:
        """
        FR5.pre_step()에서 매 프레임 호출되는 함수.
        남은 스텝이 있다면 관절을 조금씩 움직입니다.
        """
        # 1. 움직일 필요가 없으면 리턴
        if self._steps_remaining <= 0 or self._current_step_delta is None:
            return

        # 2. 현재 관절 위치 가져오기
        if self._get_joint_positions_func is None:
            return
        
        current_positions = self._get_joint_positions_func()
        
        # 3. 다음 목표 위치 계산
        target_joint_positions = [None] * self._articulation_num_dofs
        
        for i, joint_idx in enumerate(self.active_joint_indices):
            current_pos = current_positions[joint_idx]
            
            # Open이면 더하고, Close면 뺍니다.
            if self._target_direction == 1: # Open
                next_pos = current_pos + self._current_step_delta[i]
            else: # Close
                next_pos = current_pos - self._current_step_delta[i]
                
            target_joint_positions[joint_idx] = next_pos

        # 4. 액션 적용
        action = ArticulationAction(joint_positions=target_joint_positions)
        if self._articulation_apply_action_func:
            self.apply_action(action)
        
        # 5. 남은 스텝 감소
        self._steps_remaining -= 1