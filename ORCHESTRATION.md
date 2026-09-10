# ORCHESTRATION.md — 오케스트레이션 계층 재설계 spec

`REFACTOR.md`의 "🏗️ 오케스트레이션 계층 재설계" 결정을 구체화한 **설계 문서**. 구현 subagent는
이 spec을 따른다. 목표 = **재현 가능·결정론적·배치 가능·무-silent-failure** 실험 파이프라인
(R1#8 재현성 + D3 견고성 측정). **계산 코어는 유지, 오케스트레이션/상태/실험 glue만 재작성.**

원칙: 증분 교체 — 새 오케스트레이터를 옛 것과 **나란히** 만들고, 매 단계 **작동하는 transfer
end-to-end + seeded 배치**로 검증한 뒤 옛 드라이버를 하나씩 제거. 옛 코드는 검증 전까지 레퍼런스로 유지.

---

## A. 보존 (KEEP — 계산 코어 + 그 인터페이스; 절대 재작성 X)
- cuTAMP planning(`run_cutamp`/`setup_cutamp`, `TAMPEnvironment`, `TAMPConfiguration`), cuRobo `MotionGen`.
- LLM `ActionReasoner.predict` (계약: 입력 `{current_xdl,next_xdl,obstacle_info,candidate_grids}` → 출력 `(main_tool, is_rearrange, rearrange_tool, rearrange_grid)`).
- Isaac Sim 씬 + skill(II collision / III attach / IV pour), OmniGraph 로봇 I/O, `sim_control` 확장(`get_entity_state`/`set_simulation_state`), 2-process/DDS 분리 + launch.
- **plan-step 스키마**(cuTAMP `motion_solver.py` 산출): `{"type":"trajectory","plan":JointState,"dt","op_name"}`, `{"type":"gripper","action":"close"|"open","target":obj}`.
- q_init(`/isaac_joint_states`)→cuTAMP, `get_q_home(robot)` fallback. GT poses(`get_entity_state`) = 유일한 pose oracle.
- 진행 중 cuTAMP 로버스트니스 fix(upright θ≤5°, pour-region).

## B. 재작성 (REBUILD — 순수 오케스트레이션 glue)

### B1. 단일 정규 오케스트레이터 (최우선)
현재 **드라이버 4개 중복**(`tamp_client`/`tamp_xdl_parser`/`trials/trial_driver`/`automate_run`+`run_trials`)을
**하나의 오케스트레이터**로 통합. 정규 task-drive 시퀀스 1곳: `(tool→cfg) → env → plan → execute`.
배치 harness·LLM/XDL·인터랙티브가 **모두 이 하나**를 사용. tool_change choreography·`set_tamp_cfg`·`set_tamp_env`
로직의 중복/drift 제거.

### B2. 설정/환경 레지스트리 (코드 아닌 데이터)
env별 movable/static/ex_collision 리스트, tool별 `set_tamp_cfg` 기본값(grasp_dof, time_dilation 등)을
**한 곳(config 모듈/파일)에서 1번만** 정의. 현재 4곳에 서로 다르게 박혀 있음(client `transfer=[beaker,flask]`,
parser `from/to_vessel`, trial_driver 별도, server `__main__` 또 별도) → 단일 소스화.

### B3. (대부분) 무상태 planner
`plan(env,cfg,q_init) → plan_object`, `execute(plan_object)`. 서버측 숨은 상태(`self.plan_to_execute`,
`self.last_operator`, `env_manager` in-place mutation) 제거/최소화. ROS srv 특성상 남는 상태는 명시·문서화.
(sim측 `current_tool`/attach/scale accumulator는 물리 상태라 sim측 유지.)

### B4. pouring을 명시적 plan step으로
현재 pouring이 execute 루프에서 `op_name=="Move_to_Surface"` **문자열 매칭**으로 삽입됨(숨은 제어흐름).
→ cuTAMP가 내는 **명시적 step**(예: `{"type":"pour",...}` 또는 skill 호출)으로. 실행 루프는 step 타입만 dispatch.

### B5. 상태 lifecycle 근본 수정
- **q_init/pause**: 현재 `tamp_plan_cb`가 sim을 pause(state=2)하고 execute까지 미복귀 → paused timeline이라
  `/isaac_joint_states` 끊겨 q_init empty(빈 q0 크래시 근원). **plan을 위해 sim을 pause할 필요 없음**(planning은
  planner 프로세스). → **pause 제거**(또는 q_init을 `get_entity_state` GT 관절에서 취득, 또는 pause 전 읽기).
  근원 수정, q_home fallback에 의존 X.
- **config-switch 오염**(fr5→ag95→fr5 IK 붕괴): cuRobo/CUDA 지속 GPU 상태(CUDA-graph capture, cached IK)가
  robot config 간 미해제 = **유지-존(cuRobo) 내부**. → **planner 프로세스당 robot config 1개**를 원칙으로,
  오케스트레이터가 **config별 프로세스 lifecycle을 깨끗이 관리**(적절한 spawn/teardown, `pkill -f` 금지).
  단일-config 배치(transfer/ag95)는 한 프로세스로 문제없음; 다중-config(LLM tool change)는 config마다 fresh
  planner. (cuRobo teardown 직접 수정은 깊고 위험 → 후순위.)
- **tool-change 월드 teardown**(sim `world.clear()` 전체 재구성): 작동하니 우선 유지, 빈도 최소화. 후순위 최적화.

### B6. 결정론적 seeded 배치 러너
- 프로세스 격리(배치별 `ROS_DOMAIN_ID`), `pkill -f`(아무 프로세스나 죽임) 금지 → 정확한 프로세스 그룹 관리.
- 구조적 CSV 로깅(스키마: `seed,task,robot_cfg,plan_success,planning_time_s,num_satisfying,execute_success,max_transport_tilt_deg,failure_reason,...`).
- 하드코딩 경로 X(project-relative), LLM 출력 test-override 제거(`rearrange_grid="G12"`, `target="flask"` 상수 하드코딩 제거 → 실제 LLM 출력 사용, config로 토글).

## C. 마이그레이션 순서 (증분·검증 기반)
1. 설정/환경 레지스트리(B2) + 무상태 planner API(B3) — 가장 근간, 나머지가 여기 의존.
2. 단일 오케스트레이터(B1) — 정규 task-drive; 먼저 **transfer 배치 경로**로 검증(작동하는 end-to-end 대조).
3. pause 제거(B5 q_init) + pouring 명시 step(B4).
4. 결정론 배치 러너(B6) — 옛 `run_trials`/`automate_run` 대체.
5. LLM/XDL 경로를 같은 오케스트레이터로 통합(중복 choreography 제거).
매 단계: 옛 코드 유지 → 새 것이 transfer end-to-end + seeded 배치 재현 → 옛 것 제거.

## D. 저자 확인 필요 설계 결정 (기본값=권장)
> **확정(2026-09-01): 저자가 세 항목 모두 권장안으로 승인.** 대부분 무상태 planner,
> process-per-config, LLM/XDL 경로까지 단일 오케스트레이터에 통합한다.

1. **무상태 planner**(B3): 리팩터 가치 있음 vs 최소 상태 유지. **권장: 대부분 무상태**(테스트·재현 용이).
2. **config-switch**(B5): **프로세스-per-config**(깨끗이 관리) vs cuRobo teardown 투자. **권장: 프로세스-per-config**(cuRobo 내부 수정은 깊고 위험).
3. **범위**: LLM/XDL 드라이버도 통합 vs 배치/실험 경로만 먼저. **권장: 통합**(중복 tool_change/cfg/env glue가 문제의 핵심 — 하나의 오케스트레이터가 둘 다 서비스).

## E. 구현 진행 로그
- **C1 B2+B3 ✅ (2026-09-01):** immutable 환경/planner 레지스트리 + 명시적 planning result API,
  env 요청마다 fresh `TAMPEnvManager`, 첫 planning 뒤 robot-config 변경 fail-closed. batch/client/XDL이
  동일 레지스트리 사용. 설정 drift 진단 2건: transfer legacy collision 목록과 AG95 `grasp_dof=6`을
  검증 경로와 대조해 각각 canonical collision 목록, `grasp_dof=4`로 고정. seed 0 실제 E2E:
  plan 51.06 s, 60 satisfying, execute 성공, transport tilt 1.61°.
- **C2 B1 ✅ (transfer 경로):** `TaskOrchestrator`가 `(tool→cfg)→env→plan→execute` 정규 시퀀스의
  단일 batch 엔트리포인트. seed 1 실제 E2E: plan 26.64 s, 194 satisfying, execute 성공, tilt 3.95°.
- **C3 B5(q_init)+B4 ✅:** planning pause 제거(실제 joint-state 유지), plan adapter가 explicit `pour`
  step 삽입, executor는 type dispatch. seed 2 실제 E2E: plan 39.27 s, 200 satisfying, execute 성공,
  tilt 3.77°; q_home fallback 없음, explicit pour 로그 확인.
- 위 수치는 증분 회귀 검증값이며 headline 통계가 아니다. 레지스트리 정규화 중 실패한 seed 0 두 run
  (77.98 s/81.92 s, 0 satisfying)도 CSV에 보존했다.
