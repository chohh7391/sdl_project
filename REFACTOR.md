# REFACTOR.md — sdl_project 코드/워크스페이스 정리

`sdl_project` 코드·워크스페이스를 재현 가능하고 견고하게 정리하는 엔지니어링 플랜.
논문 쪽 계획은 `PLAN.md`, 원칙/가드레일은 `CLAUDE.md`. 이 문서는 그 둘의 **코드 파트**를
구체화한 것이며, 여기서 개선한 코드가 `PLAN.md`의 D3(robustness)·§3-B(harness)·§3-C(config)·
R1#8(재현성)의 정직한 숫자를 만든다.

리뷰 근거: 아래 문제들은 모두 코드에서 `파일:줄`로 확인됨. 원칙 = **숫자는 항상 개선된
코드의 실제 측정을 따른다**(`CLAUDE.md`).

## 실행 방식 (저자 지정)
- 실제 수정/구현은 **구현 subagent**에 위임. 메인 세션(Claude)은 **매니저**: 태스크 분해·시퀀싱·통합·
  REFACTOR.md/PLAN.md 관리.
- 마일스톤마다 **감독(리뷰) subagent**로 피드백 받고 반영 후 진행.
- 같은 워크스페이스/빌드를 건드리는 구현 태스크는 충돌 방지 위해 **직렬**로. 검증은 실제 실행 기반(날조 금지).

## 🎯 핵심 진단: planning 실패의 근본 원인 (STAGE A, 2026-09-01)
저자 보고 "planning 엄청 안됨"의 원인 규명. **GT-pose→cuTAMP 입력은 정상**(물체 제약 1017-1024/1024 통과). 실패는 robot/grasp 쪽 3계층:
1. **기본 `fr5`(그리퍼 없음) = 진짜 불가능**: 잡은 beaker가 wrist sphere 관통 → `robot_to_movables 0/1024`. transfer plan은 반드시 **gripper config(`fr5_ag95`)**로. README 데모가 `tool_change ag95`를 빠뜨림.
2. **⭐ 핵심 버그**: `fr5_ag95`면 cuTAMP가 **실제로 푼다**(147-319/1024 satisfying). 그런데 `TAMP/cuTAMP/cutamp/motion_solver.py`의 `solve_curobo`(algorithm.py:493 경유)가 **빈 trajectory를 `JointState.from_position(...[-1:])`→reshape[1,6]** 하다 크래시(`invalid for input of size 0`, Pick op). 그리고 **`tamp_server.py:140-142` bare `except`가 삼켜 `total_num_satisfying=0`** → **성공 plan을 실패로 보고**, 3회 재시도. = "연결 glue 버그"의 정체(pose 아님, curobo 반환 경로).
3. **config 전환 상태 오염**: `set_tamp_cfg`(fr5→ag95→fr5) 후 IK가 552-736 → 0/1024 붕괴. tool_change 흐름이 반복 호출 → 실제 데모에서 planning 저하.
**수정 우선순위**: (i) motion_solver 빈-trajectory 가드(Pick/MoveHolding/Place ~110/132/146), (ii) tamp_server:140-142 exception 표면화(silent swallow 금지), (iii) transfer가 gripper config 쓰게, (iv) config-switch 오염 조사(fresh IK/world per cfg). **이게 D3·논문 100% planning success의 핵심.**

> **✅ 해결·검증(2026-09-01) — 실제 근본원인은 더 깊었음:** 빈 텐서는 빈 trajectory가 아니라 **`q0`(로봇 초기
> config) 자체**. `tamp_server`가 `q_init = joint_states.position[:6]`인데 `/isaac_joint_states`는 sim **playing**일
> 때만 발행되고 `tamp_plan_cb`가 sim을 pause하고 안 되살려 → q_init 비어있음 → `[1,6]` reshape 크래시. (sim playing
> → `q0.shape=(6,)`, 크래시 소멸로 증명.) = 저자가 의심한 sim↔cuTAMP **play-state/joint-state 피드 버그**.
> **수정:** `motion_solver.py` 빈-q0 복구(`world.q_init`/`get_q_home`)+빈-segment 가드(8곳); `tamp_server.py`
> exception 표면화(`_log.exception`, silent swallow 제거)+`time_dilation_factor 0.0→0.5`(파생 ZeroDivisionError 수정)
> +q_init 비면 q_home fallback; transfer는 `set_tamp_cfg fr5_ag95` 선행(문서화). (source+colcon-install 양쪽 적용.)
> **검증:** transfer **plan 2/2 성공**(satisfying 125·302/1024, 28–33s, 6 op 전부), **execute 성공**(34.8s,
> grasp→carry→pour→place; v_cmd ≤0.5 clamp 확인). `/raw_scale_data`=0은 가짜-scale 한계(IV-3).
> **⚠️ layer-3 미해결**: 2nd/3rd `set_tamp_cfg` → IK 0/1024 붕괴 → **STAGE B에서 처리**(trial마다 fresh server 또는 fix).

## STAGE B1 — randomized harness + 정직한 실측 (2026-09-01)
harness 구축 완료: seeded 위치 randomization(`SDL_SEED`, PCG64, ±5cm/±180°/±10°, 결정론적, `task.py:_randomize_layout`),
MAJOR-1 수정(파지가 **planner 의도 물체** 타깃 — `/set_grasp_target` latched, `simulation.py grasp_target_cb`),
per-trial CSV(`_2026__IEEE_Access/revision/analysis/data/transfer_trials.csv`), 드라이버(`scripts/run_trials.sh`, trial마다 fresh sim+server로 layer-3 우회).
**실측(transfer, seed 0-4 + nominal):** plan **3/5**(seed 1,3,4 성공; 0,2 실패), execute 3/3. nominal 성공 → **2/5는 layout 유발(회귀 아님)**.
**🔴 랜덤화가 드러낸 2가지 (논문 주장과 충돌 — 저자 결정: 코드 개선/D3):**
1. **planning 위치-비견고**: seed 0,2 `pour_region_in_xy = 0/1024`(flask 위 pour 배치 도달 불가) → 논문 "100% planning" 미재현.
2. **운반 upright 미보장**: seed 4 **tilt 90°**(4-DOF 파지가 기운 상대pose로 옆운반) → 논문 "θ≤5° 매 step" 위반(창발적 확증).
**수정 계획:** (A) MoveHolding에 **hard held-object upright 제약(θ≤5°)** 추가(cuRobo `hold_partial_pose` 또는 post-plan FK guard) → seed 4 + θ_max 주장 정직화; (B) pour-region 도달성 진단·개선(IK infeasible vs collision vs 도달불가 randomization) → seed 0,2. 그 뒤 **재측정**.

> **✅ 해결·검증(2026-09-01) — transfer 3/5 → 5/5:**
> - **FIX A(upright)** = cuRobo `hold_partial_pose`(carry op에서 ee roll/pitch 고정; 4-DOF 파지+upright endpoint라 yaw만
>   남아 물체 수직 유지 — 충분조건) + **post-plan FK guard(운반 max tilt 측정)**. `motion_solver.py`(플래그 `SDL_UPRIGHT_TRANSPORT`, 기본 on).
>   → seed 4 **90.24° → standalone 0.04-0.72°**, live seed 0/1/2 ≤4.01°. **θ_max 주장 이제 정직하게 참(강제+측정).**
> - **FIX B(pour-region)** = **cuTAMP 수학버그**: `approximate_goal_aabb`(`utils/common.py`)가 yawed surface AABB를
>   **대각 2코너만** 변환해 bounds 반전(코드에 `# TODO: non-axis-aligned` 있던 자리). flask yaw −179°/−126°(seed 0,2)에서
>   pour_region 항상 out-of-bounds. **→ 8코너 min/max로 수정**(loosening 아님, 임의 회전 정확). seed 0:0→265, seed 2:0→245 satisfying, live plan+execute 성공.
> - **재측정(both on, live):** seed0 plan✓/tilt4.01°, seed1 ✓/3.47°(was 7.19°), seed2 ✓/3.60° — **0,2 실패→성공**.
>   seed3,4,nominal은 지시로 live 미재실행(회귀 기전 없음). 이 session 편집 = `motion_solver.py`+`common.py`(둘 다 cuTAMP 유지-존).
> - **"천차만별"의 상당부분이 실제 버그였음** → 근본 수정(layout-general). **B2(30-seed×3-task) 준비됨** — 단, 클린 harness(오케스트레이션 재설계 후)에서 돌리는 게 재현성상 맞음.

## 🏗️ 오케스트레이션 계층 재설계 (저자 결정 2026-09-01)
**결정: 계산 코어는 유지, 목표를 막는 오케스트레이션/상태/실험 계층만 깨끗이 재작성.** 목표=재현 가능한 위치-견고 실험(R1#8·D3), 마감 리스크 통제.
- **유지 (검증됨):** cuTAMP planning + cuRobo, LLM(XDL/action reasoner) 모델, II/III/IV Isaac Sim 씬+skill, **2-env/2-process 분리 + launch/interface**(py3.12/3.10 강제, DDS 검증됨), 진행 중 cuTAMP 로버스트니스 fix.
- **재작성:** 거대 stateful `tamp_server` + 흩어진 `tamp_client`/`tamp_xdl_parser`/`automate_run` → **깨끗한 실험 오케스트레이터**(seed 결정론적 → plan → execute → 로깅, 한 세션에서 배치). **상태 lifecycle 근본 수정**: tool-change 월드 전체 파괴 제거, **config-switch 오염 근본 수정**(solver/world 재초기화), sim playing 유지(joint state), 예외 표면화. **명시적·문서화된 sim↔planner API**(ROS 서비스 정리, stateless 지향).
- **원칙:** 결정론·배치 가능·무-silent-failure·깨끗한 상태 리셋·재현성.
- **진행:** (1) 현 오케스트레이션 정밀 매핑(read-only) → (2) 매니저가 설계 spec → (3) **작동하는 transfer end-to-end에 대해 증분 교체·검증**(옛 코드는 레퍼런스 유지).

> **증분 구현 C1–C3 완료(2026-09-01):** 저자가 ORCHESTRATION §D 권장안 3개 전부 확정.
> canonical registry/명시적 planner result/fresh env snapshot/config-switch fail-closed → 단일
> `TaskOrchestrator` transfer 경로 → sim pause 제거 + explicit pour step 순서로 구현. 각 단계 실제
> transfer E2E 성공(seed 0: 51.06 s/60/tilt 1.61°, seed 1: 26.64 s/194/3.95°,
> seed 2: 39.27 s/200/3.77°; 모두 execute 성공). 상세/실패 진단은 `ORCHESTRATION.md §E`.

> **⛔ 되감기(2026-09-08):** 위 C1–C3 검증(09-01 17:54) **이후** Codex 세션들이 문서화 없이 넣은 변경 전부를 저자 지시로 되감았다
> (initial-support contact-evidence 프로토콜, ag95 6-DOF side-grasp, cuRobo 코어 패치, collision_semantics, scene_mode 등 — 라이브 E2E 실패 상태였음).
> 현재 코드 = C1–C3 검증 상태 + q_home fallback 2줄 수정. **이 메커니즘들을 다시 도입하지 말 것.** 상세·방법·재검증 수치는 `SESSION_LOG.md` 2026-09-08.

## 🤏 측면 파지(side grasp) — 논문 서술과 코드의 불일치 해소 (2026-09-08)

저자 지시: "자연스럽게 잡게 돼야 해." 논문은 Transfer를 *"2-finger gripper ... to enable side-grasp
pouring"*으로 서술하는데, 검증 경로의 코드는 **위에서 내려잡는 top 파지**였다. 아래는 모두 실측/코드 확인:

**진단 (왜 top 파지가 틀렸나)**
1. **pour가 물리적으로 불가능.** `tamp_server.pouring()`은 `last_idx = 5`, 즉 joint 6을 돌린다. FK로 확인:
   joint 6은 **EE +z(접근축)**를 정확히 회전시킨다(0.2 rad → 0.2 rad, 축 (0,0,1)).
   물체를 정확히 수직으로 둔 뒤 이 관절을 돌리면 — **top 파지: 기울기 0.00°**(10/30/60/90° 모두),
   **side 파지: 관절 각도와 1:1**(10°→10.00°, 90°→90.00°). top 파지의 접근축이 곧 비커 자신의 축이라
   비커를 제자리에서 돌리기만 한다. sim의 `compute_scale_data`가 **손목각 프록시**로 가짜 질량을 올리기 때문에
   이 사실이 지금까지 드러나지 않았다(즉 sim pour는 한 번도 실제로 기울인 적이 없다 — R1#2/R3#1에 반영 필요).
2. **top 파지 계열은 원소가 4개뿐.** `grasp_4dof_sampler`는 yaw만 뽑고 `sample_yaw(num_faces=4)`는
   `{0,90,180,270}°` **정확히 4개**를 낸다(연속 아님). 즉 1024 입자가 서로 다른 파지 4개를 탐색한다.
   또한 grasp는 **최적화 대상이 아니다**(`optimize_plan.types_to_optimize = {Pose, Conf}`) → 샘플러 품질이 전부.
3. **오프라인 IK 전수조사(seed 0–29, statics-only world, grasp+pre-grasp 모두 성공 요구):**
   top 4개 중 도달 가능 **1–3개**, **seed 3·5·18·20은 0개 → 구조적 실패 하한(입자 수로 해결 불가)**.
   side(방위 36 × 높이 3 = 108 후보)는 **30/30 seed 도달 가능**, seed별 후보 26–44개, 방위 19–29/36.
   → "성공률이 물체 위치에 따라 천차만별"의 진짜 원인. (배치 재현기는 live CSV seed 0–2와 **0.05 mm 이내** 일치로 검증.)
4. **09-02 Codex의 6-DOF 시도가 실패한 이유**: `grasp_6dof_sampler`는 자기 docstring대로 **bookshelf 도메인용** —
   pitch를 0~2π 균일 샘플링하고 파지점을 박스 **내부**에서 뽑는다. 그 샘플이 그대로 cuRobo로 가서 `IK_FAIL`
   ("Failed to plan from approach", 244.7 s/3회). 아이디어가 아니라 **샘플러**가 문제였다.

**수정 (최소 범위 — 샘플러 하나 + 레지스트리 한 줄)**
- `cutamp/samplers.py`: **`grasp_side_sampler`** 신규. 자유변수 2개 = 방위 φ~U[−π,π), 높이 h.
  tool 프레임 = **+z 반경 외향(접근축), +y 접선(손가락 개폐축), +x 아래**, 원점은 **vessel 축 위**
  (4-DOF와 같은 규약: tool 원점 = 파지 중심). h 범위는 그리퍼 구체의 수직 여유(below/above)에서 유도해
  **rim 위·base 아래로 절대 안 나가게** 자동 계산(임의 vessel 높이에 일반). 회전은 `roma.rotmat_to_euler("XYZ")`로
  기존 6-DOF action 표현에 정확히 왕복(오차 2e-7) → 다른 코드 경로 무변경.
- `particle_initialization.py`: Pick의 **샘플링 지점 한 곳**만 `grasp_dof != 4` → `grasp_side_sampler` 디스패치.
  (변환 분기 4곳은 이미 `grasp_dof`로 올바르게 갈라져 있어 무변경.)
- `orchestration/registry.py`: `ag95` **grasp_dof 4 → 6**(side는 전체 6-DOF 포즈 필요). vgc10/dh3는 4-DOF top 유지
  (suction/3-finger = 위에서 집기, 논문 Table과 일치).
- **무변경 확인:** FIX A(upright)는 `hold_partial_pose`로 **ee가 world z축 회전만** 허용하므로 파지 방식과 무관하게
  유효(엔드포인트가 수직이면 경로 전체 수직 유지). pour 후 복귀도 기존 `untilt_waypoints`가 처리.
  `approach_offset`(−5 cm, ee z)은 side에서 자연히 **반경 5 cm 후퇴**가 되어 특수처리 불필요
  (Codex의 `side_grasp_approach_clearance` 같은 예외 없음).
- `TAMP/tamp/test/test_side_grasp.py` 신규 5개 테스트로 불변식 고정(축 위 파지 / body 범위 / 수평 접근+수평 개폐축 /
  방위 전주 커버 / **pour 관절 → top 0° vs side 1:1**). 전부 통과.

**남은 항목**
- pour 방향: joint 6은 접근축(반경) 회전이라 비커 입구가 **접선 방향**으로 기울어진다. sim은 유체가 가짜라 무관하지만
  실제 rig에서는 스트림 낙하점이 flask 중심에서 벗어날 수 있음 → §6(실제 pouring 세션)에서 확인/보정.
- 논문 반영: (a) sim pour가 top 파지에서 실제로 기울지 않았다는 사실은 "sim은 유체 정확도 주장 X" 서술을 더 강하게
  뒷받침, (b) side 파지가 top보다 **위치-견고**하다는 측정(30/30 vs 26/30)은 D3의 근거로 사용 가능.

## 🫱 릴리스(놓기) — 운반은 성공했는데 비커가 눕는 문제 (2026-09-09)

B3 배치(측면 파지 + 직선 접근 + flange 용접)에서 과제 성공 **23/30 = 76.7% [Wilson 59.1–88.2]**.
실패 7건은 **전부 최종 tilt 정확히 90.00°**(비커가 옆으로 누움)인데, 그중 5건은 운반 tilt가 1.37–4.06°로
정상이었다 → 운반이 아니라 **놓는 순간**의 문제. 원인을 추측하지 않고 sim에 계측을 넣어 확정했다.

**계측 (`RELEASE_TRACE`, `simulation.py`)**
`carried_tilt_deg`는 용접이 풀리는 즉시 −1을 내보내므로 **정확히 문제 구간이 관측 불가**였다.
그리퍼 open 핸들러에 릴리스 전 / 해제 후 / 개방 후 스냅샷을, `step_cb`에 릴리스 후 3 s(6 step마다)
위치·tilt·|v|·|w| 로그를 추가. seed 5 실측:

| 시점 | z | tilt | \|w\| |
|---|---|---|---|
| pre_release | 0.0851 | 0.30° | 0.026 |
| after_detach | 0.0851 | 0.30° | 0.026 |
| **after_open** | 0.0736 | **28.85°** | **1.98** |
| t+012 | 0.0265 | 91.06° | — |

**원인 1 — 해제 순서가 뒤집혀 있었다.** 용접은 "손가락이 잡고 있다"를 대신하는 모델이므로 **손가락이 놓은 뒤**
풀어야 한다. 기존 코드는 `_detach_object()`를 먼저 호출해, 비커가 **닫힌 손가락 안의 자유물체**가 된 상태에서
손가락이 벌어지며 비커를 긁었다(위 표의 after_open). `SDL_RELEASE_ORDER`로 전환 가능하게 하고 기본값을
**`open_first`**(개방 → 해제)로 변경. 같은 seed에서 개방 후 **0.06° / 0.015 rad/s**.

**원인 2 — 비커를 15 mm 공중에서 놓고 있었다.** `goal_region`은 **Isaac Sim에 prim이 없는 플래너 전용 표면**이고
(`task.py`가 스폰하는 것은 table과 box_goal뿐), 그 xy의 실제 지지면은 테이블 윗면 z=0이다. 그런데
`cost_function`의 배치 목표는 `surface_top + activation_distance + 2 mm`이고, 반대로 movable의 플래너 포즈는
`update_entities`가 **+0.01 m 들어올린다**(초기 접촉 플래그 회피용). 합치면 실제 비커 바닥은
`0.020 + 0.002 − 0.010 = 0.012 m` 공중(실측 15.2 mm, 차이는 구 근사/PhysX rest offset).
→ `envs/constants.py`에 **`PLANNER_Z_LIFT`**를 명시하고, `transfer.py`가 `goal_region` z를
**테이블 윗면 + lift**에서 유도하도록 변경(하드코드 0.015 → 계산값 0.005). 실측 낙하 **3.8–9.6 mm**.

**원인 3 — 릴리스 후 retract가 놓인 비커를 쳤다.** 최종 retract는 방금 놓은 물체를 충돌 제외한 채
(열린 그리퍼가 아직 물체를 감싸고 있으므로 필요) **자유 계획**이라, 손이 비커를 통과·타격할 수 있었다.
파지 접근에서 이미 겪은 것과 같은 계획/시뮬 불일치이고, 이번엔 **릴리스 후**여서 세워둔 비커를 넘어뜨린다
(seed 3: 0.00°로 정지 후 t+174에 **10.44°**, 1.2 cm 밀림 — 드라이버 측정 시점을 지나서 발생).
→ 파지 접근과 동일한 **직선·자세고정 retract**(`grasp_plan_config`)로 교체, 실패 시 자유 계획 폴백 + 경고 로그.

**원인 4 — 목표 영역 제약이 발자국을 무시했다.** `StablePlacement`의 `_in_xy`는 물체 **구 중심**만 AABB에
넣으므로 비커 절반이 영역 밖으로 걸친 배치가 합법이었고, 실행 오차 몇 mm가 영역 밖으로 밀어냈다
(seed 5: 0.414, −0.293 = 중심에서 6.4 cm, 반폭 5 cm 영역). → AABB를 **각 구의 반지름만큼 축소**해
"발자국이 영역 안"을 뜻하게 변경. 물체가 표면보다 크면 경계가 뒤집히므로 표면 중심점으로 클램프.
비용은 미미: `pour_region_in_xy` 967–995/1024, `goal_region_in_xy` 1004–1016/1024.
구현은 `costs.dist_from_bounds_inset`로 분리해 단위 테스트 가능하게 했고, `TAMP/tamp/test/test_placement_geometry.py`(신규 4개)로 ②·④의 불변식을 고정했다 — 표면 윗면 = 테이블 + lift, 경계에 중심을 둔 배치는 inset에서 거부, 표면보다 큰 물체는 중심점으로 수렴.

**검증 (실패 3 + 기존 성공 2 = seed 5·3·1·0·2)**: **5/5 task_success**, 최종 tilt 전부 0.00°,
목표 중심 오차 0.4–1.3 cm, 릴리스 후 3 s 재교란 없음, 직선 retract 폴백 0회, `GO_HOME_FALLBACK` 0회(B3는 12/30).
전체 배치 B4 결과는 `SESSION_LOG.md` §0.

## 🧰 move / stir를 randomized harness에 올리기 (2026-09-10)

transfer만 harness에 있었고 move/stir는 **한 번도 실행된 적이 없었다**. 올리면서 드러난 것들:

**harness 일반화**
- `run_trials.sh`에 `TASK`/`ROBOT` 추가. 기본 도구는 `content/configs/xdl/tool_map.yml`을 따른다
  (transfer→ag95, move→**vgc10 흡착**, stir→**dh3 3지**). `sim_boot_failed` 행의 하드코딩된 `transfer,fr5_ag95`도 제거.
- `trial_driver.py`의 성공 판정을 task별 표(`TASK_OUTCOMES`)로 분리. 목표는 **엔티티로** 지정해 랜덤화를 따라가게 했다
  (stir 목표는 랜덤화되는 stirrer 위에 있다). 새 컬럼 `target_obj`,`goal_err_mm`,`aux_check` — 판정 임계값을
  나중에 조여도 배치를 다시 돌리지 않게 **원시 오차를 기록**한다.

**move: 목표가 시작 위치와 같았다(과제가 t=0에 이미 성공)**
- `box`의 nominal (−0.126,−0.575)이 목표 트레이 `box_goal`(G1 = −0.15,−0.6, 0.15 m)의 **발자국 안**이었다(중심간 3.5 cm).
  게다가 box는 바닥 z=0으로 스폰되고 트레이 윗면은 z=0.010 → **10 mm 관통**. 즉 초기 상태가 물리적으로도 무효.
- 트레이를 **G3(0.15,−0.6)**로 옮겨 0.30 m 운반 과제로 만들었다. 트레이는 `_RANDOMIZE_ORDER`에 없어
  어떤 seed 레이아웃도 바뀌지 않는다(검증: transfer seed 0의 beaker_xy가 B4와 동일).
- `box_region`(플래너 전용 표면)도 두 가지가 틀려 있었다: xy가 **0.2 m로 트레이(0.15 m)보다 크고**,
  z가 `tray_z+0.02`여서 box를 **7 mm 공중**에 놓았다. 둘 다 트레이의 실제 dims/윗면에서 유도하도록 고쳤다.
- 흡착 도구는 용접을 하지 않아 `carried_tilt_deg`가 **전 구간 −1**이었다(운반 tilt 미계측 = 논문 실행기준 (iii) 불가).
  `SurfaceGripper.is_closed()`(PhysX 상태)로 실제 파지 중일 때만 tilt를 publish하게 했다 → seed 0에서 1.87–1.94° 계측.

**stir: 홈 자세가 dh3에서 자기충돌 → 전 seed 계획 불가**
- 첫 모션 구간부터 `INVALID_START_STATE_SELF_COLLISION`. 실패 상태 q는 **홈 자세 그 자체**였다.
  링크 쌍까지 특정: `shoulder_link` ↔ `finger1_link`/`finger1_tip_link`(자기충돌 버퍼 포함 5–10 mm 침투).
  공유 홈은 j3=−2.18로 팔꿈치를 접어 도구를 어깨 옆에 세우는데, 3지 그리퍼는 여기서 여유가 없다.
  **명령된** 홈은 30 seed 모두 통과하지만 **실제로 정착한** 자세는 통과하지 못한다(seed 0 로그 q: constraint 1.3111).
  fr5/ag95/vgc10은 같은 노이즈에서 0/600 → dh3 한정 문제.
- `_TOOL_HOME_OFFSET`에 dh3만 **j3 +0.10 rad(5.7°)**. 측정으로 선택: ±10/20/40 mrad 추종노이즈에서 0/450 불가능
  (0 오프셋은 4–5/450). 0.15/0.20/0.30도 동일하게 깨끗해 knife-edge가 아니다.
- 적용 지점 주의: `tool_change_cb`가 교체 전 관절값을 복원하므로 `joints_default_state`만 바꾸면 무시된다.
  복원 시 **두 도구 오프셋의 차분**을 더해 반복 교체에도 누락·누적이 없게 했다.
- 이 수정으로 stir는 **계획·실행된다**(seed 0: 33.3 s, 823 particles).

**stir 남은 문제(미해결, 측정만)**
- 릴리스 시 잔류속도: flask |v|=0.36 m/s·|w|=4.9 rad/s. `SDL_PLACE_SETTLE_STEPS`(기본 15 step)로
  마지막 명령 자세를 유지해 |v|는 0.089로 줄었지만 **|w|는 7.3 rad/s로 오히려 커졌다**.
- 릴리스 후 flask·magnet이 **위로 떠오른다**(flask z 0.158→0.207, magnet 0.258→0.304, |w| 7–8 rad/s).
  dh3가 3지를 1.16 rad로 용접된 물체에 조여 솔버가 자기 자신과 싸우는 형태 → 해제 시 에너지 방출로 보인다.
  최종적으로 flask는 stirrer에서 183 mm 밖, 90°로 누웠다. **다음 사이클의 대상.**
- 운반 tilt 집계 오류도 여기서 발견: stir는 vessel과 stir bar를 차례로 들어 `carried_tilt_deg`가 섞였고
  bar의 30°가 vessel tilt로 보고됐다. sim이 `carried_obj`(이름)를 함께 publish하고 드라이버가
  **대상 vessel의 샘플만** 집계하게 고쳤다 → 같은 seed에서 29.98° → **2.26°**.

## 🖐 dh3 파지 안정화 — 그리퍼가 애초에 다 열리지 않았다 (2026-09-10)

stir 실행에서 해제 후 flask가 **일정 속도로 떠올랐다**(z 0.158 → 0.207 m, 약 25 mm/s, xy는 거의 고정).
폭발적 물리 불안정이라면 등속 상승이 나올 수 없다 → **여전히 잡힌 채 팔이 들어올린 것**으로 좁혀졌다.

**원인**: 병렬 그리퍼는 호출당 위치 **델타**로 구동된다. dh3는 닫힘 1.16 rad, 델타 1.16/5 = 0.232 rad인데
`gripper_commands_cb`의 open 분기가 `num_repeat = 3`(주석: "for grasping magnet (small object)")이었다.
3 × 0.232 = 0.696 < 1.16 → **손가락이 약 0.46 rad에서 멈춰 계속 쥐고 있었다.**
게다가 관절이 명령을 지연 추종하므로 실제로는 5 step보다 훨씬 많이 필요하다 — 실측 **dh3 22–23 step, ag95 11 step**.
즉 ag95의 `num_repeat = 10`도 **한 step 모자랐다**(transfer가 동작한 것은 그 시점에 이미 놓을 만큼은 열렸기 때문).

**수정**: 횟수를 세는 대신 **열림 목표 위치에 도달할 때까지 구동하고 검증**하는 `_open_gripper()`로 교체
(상한 40 step, 허용오차 0.02 rad). 도달하면 step 수와 최종 관절값을 INFO로, 도달하지 못하면
"still holding the object" WARNING을 남긴다. 흡착(SurfaceGripper)은 관절이 없으므로 1회 open.
→ 손으로 맞춘 상수에 의존하던 이 부류의 버그가 세 도구 모두에서 사라진다.

**결과(stir seed 0–4, 5/5)**: planning 31.9–61.4 s, 운반 tilt 2.28–3.16°(전부 ≤5°),
**flask 최종 tilt 0.01–0.04°**, 전부 upright·in_goal, stirrer 중심 오차 1.1–24.2 mm.
무회귀 확인: transfer seed 0·1 성공(최종 tilt 0.00°), move seed 0·1 성공(최종 tilt 0.00°).

**남은 것은 파지가 아니라 형상이다 — stir bar가 vessel에 들어갈 수 없다**
- 5 seed 전부 magnet이 z=0.2274에 정지, flask 입구는 0.2124 → 정확히 **+15.0 mm = magnet 반높이**.
  즉 magnet은 flask **상자 윗면 위에** 놓인다. flask 콜라이더는 `create_box_collider_rigid`의
  **속이 찬 단일 박스(70×70×120 mm)**라 내부가 없다. 논문의 Stir 종료조건 "stir bar inside the vessel"은
  현재 모델에서 **원리적으로 발생할 수 없다**(transfer의 pour와 같은 부류의 발견).
- 속을 파도 현재 치수로는 거의 불가능: 벽 두께별 내부 개구부 vs 45 mm 각봉의 최악 yaw 대각선 63.6 mm →
  2 mm 벽 +2.4 mm, 3 mm 벽 +0.4 mm, 4 mm 벽 **−1.6 mm**. 실제 마그네틱 stir bar는 8 mm급이며
  그 경우 필요 개구부는 11.3 mm로 여유가 크다.
- **저자 결정: (a) 속 빈 flask + 현실적 stir bar 치수** — 논문 문구를 지키는 쪽.

**적용 (2026-09-10)**
- `object.create_hollow_box_collider_rigid` / `_add_hollow_box_collider` 신설: 바닥 슬래브 + 벽 4장.
  **바깥 외피는 그대로**(cuTAMP Cuboid와 동일)라 플래너 모델·외부 접촉·정착 높이는 불변이고 내부만 실재하게 된다.
  `Task.FLASK_WALL_M = 0.003` → clear opening 64 mm. beaker는 내부에 무엇도 넣지 않으므로 속 찬 상자 유지.
- stir bar: `Task.STIR_BAR_DIMS = [0.010, 0.010, 0.035]`, `ENTITIES["magnet"].dims`도 동일하게(둘은 일치해야 한다).
  45 mm 각봉은 최악 yaw 대각선 63.6 mm로 64 mm 개구부를 통과할 수 없었다 — **어떤 vessel을 쓰든 불가능**했다.
- `_MOVABLE_FOOTPRINT["magnet"]`은 **의도적으로 45 mm 유지**. 이 표는 레이아웃 샘플링의 비중첩 기각에만 쓰이므로
  실제보다 크게 잡는 것은 보수적(여유만 늘 뿐 중첩을 놓치지 않는다)이고, 고정해 두면 **모든 seed 레이아웃이
  이전 배치와 비트 단위로 동일**하게 유지된다.
- `beaker_region`(bar를 놓는 영역)의 xy를 vessel의 **clear opening**에서 유도. 기존 80 mm 정사각형은 70 mm vessel보다
  넓어 bar가 축에서 최대 17.5 mm까지 벗어나도 합법이었다(실측 9–34 mm = rim 위). 이제 F3 inset과 합쳐
  "bar가 입구를 통과한다"를 뜻한다. 공유 상수 `envs/constants.VESSEL_WALL_M`.
- 판정도 실제 개구부 기준으로: `MAGNET_IN_VESSEL_RADIUS_M` 0.035 → **0.032**.
- **실측(stir seed 0)**: `magnet_in_vessel=True`, bar가 입구 0.2124에서 떨어져 **z=0.1129**에 정지 —
  내부 바닥(0.0954) + 반높이(0.0175)와 정확히 일치. 릴리스 |w| 7.8 → **0.03 rad/s**.
- **비용**: magnet은 transfer의 statics라 플래너 충돌 문제가 바뀐다 → 코드 버전 일치를 위해 transfer 재실행 예정.

---

## 결정 사항
- **Isaac Sim = pip 고정설치 + standalone 구동** (소스빌드/별도 ROS ws/벤더 런처/LD_PRELOAD 폐기).
- 버전: Isaac Sim **6.0.1** (pip 패키지 표기 `isaacsim==6.0.0.1`), **Python 3.12**, torch **2.10.0**/torchvision **0.25.0** (cu128).
  근거: IsaacLab v3.0.0-beta2 pip 설치 가이드.
- 환경: RTX 5070(Blackwell, 12 GB, driver 580) → CUDA 12.8 계열.
- 옛 버전 참조(draft 4.2.0, README 5.1.0)는 전부 **6.0.1로 통일** (`PLAN.md` §3-C, R1#8).
- ✅ **torch 충돌 없음(검증)**: 아래 2-env 격리 구조라 Isaac Sim의 torch(2.11/cu130)와 cuRobo의 torch(2.7/cu128)는
  서로 다른 env → 절대 한 프로세스에서 안 만남. 두 env는 DDS로만 통신.
- 실제 유체는 실제 pouring 세션에서만 측정(`PLAN.md` D1). sim pouring은 물리적 유체를 주장하지 않음.

### 검증된 2-env 구조 (2026-08-31)
| env | python | 용도 | 상태 |
|---|---|---|---|
| `.venv-sdl` | 3.12 | Isaac Sim 6.0.0.1 + `simulation.py` (내부 Humble rclpy) | isaacsim 6.0.0.1 ✅, 내부 rclpy import ✅ |
| conda `sdl` | 3.10 | tamp_server·tamp_xdl_parser·perception + cuTAMP/cuRobo | torch 2.7/cu128 ✅, cutamp/curobo ✅, 시스템 humble rclpy ✅ |

둘 다 ROS2 **Humble** on **DDS** → 같은 `ROS_DOMAIN_ID`/RMW로 wire 호환. `simulation.py`는 rclpy 코드 유지하되
**시스템 ROS를 배제하고 Isaac Sim 내부 rclpy 경로**(`isaacsim.ros2.core/humble/rclpy` + `.../humble/lib`)를 쓰게 launch만 정리.

### I-1 정확한 설치 recipe (IsaacLab v3.0.0-beta2 기준)
```bash
# 권장: uv (페이지 권장). conda를 쓰면 env 생성만 conda로, 나머지는 pip.
uv venv --python 3.12 --seed env_isaaclab && source env_isaaclab/bin/activate
uv pip install --upgrade pip
uv pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "isaacsim[all,extscache]==6.0.0.1" \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
isaacsim   # 첫 구동/라이선스 동의 확인
# conda 대안: conda create -n sdl python=3.12 && conda activate sdl && (uv 없이 pip일 땐 --index-strategy/--prerelease 생략, 필요시 --pre)
```

---

## I. 설치 / 워크스페이스 (한 번에 정리)

- [x] **I-1 pip Isaac Sim 6.0.1 (py 3.12) 설치 완료.** `.venv-sdl`에 isaacsim 6.0.0.1 설치·확인 ✅.
      conda `sdl`(py3.10) 앱-노드 env(torch 2.7/cu128, cutamp/curobo)도 정상 ✅. → R1#8 버전 6.0.1 확정.
- [~] **I-2 (RESOLVED) Isaac Sim(py3.12) ↔ ROS2 Humble(py3.10) 통합 = 내부 rclpy + DDS.**
      **판정 완료**: Isaac Sim 6.0.0.1이 py3.12용 Humble rclpy를 번들
      (`isaacsim.ros2.core/humble/rclpy/.../_rclpy_pybind11.cpython-312-*.so`), 시스템 ROS 배제 후 import 검증됨.
      → **`simulation.py`의 rclpy 코드는 유지**, 시스템 `/opt/ros/humble`(py3.10)를 배제하고 **Isaac Sim
      내부 Humble rclpy 경로**를 쓰게 **launch만 정리**. 앱 노드는 시스템 Humble(py3.10)+conda `sdl`.
      **검증 상태(2026-08-31):** `.venv-sdl` 내부 ROS로 `rclpy.init()`+`std_msgs` 정상 ✅.
      **남은 통합 관문 = 커스텀 인터페이스**: `tamp_interfaces`(+`perception_interfaces`)가 py3.12엔 없음
      (`/home/home/sdl_ws/install` 미빌드). `simulation.py`가 `tamp_interfaces.srv` import.
      남은 작업:
      - [ ] **콜콘 워크스페이스 빌드** (시스템 Humble/py3.10): `tamp_interfaces`, `perception_interfaces`,
        `tamp`, `perception_*` → 앱-노드 측(conda `sdl`) 사용.
      - [ ] **커스텀 인터페이스 py3.12 생성**: `tamp_interfaces`/`perception_interfaces`를 Isaac Sim 내부
        Humble(rosidl)로 생성해 `.venv-sdl`에서 import 가능하게. (또는 simulation.py의 커스텀 서비스를
        표준 타입/다른 IPC로 대체하는 재설계 검토 — V 참조.)
        **결정: 빌드 워크스페이스는 `~/`가 아니라 프로젝트 폴더 안**(예: `sdl_project/ros2_isaacsim_ws/`)에
        두고 project-relative로 사용 → self-contained, 홈 하드코딩 제거. 무거운/생성 산출물은 `.gitignore`.
        NVIDIA IsaacSim-ros_workspaces 전체를 vendoring하기보다 **커스텀 인터페이스만** 빌드하는 최소 ws 선호.
      - [ ] **clean launch 스크립트** 2개: (1) `.venv-sdl`에서 시스템 ROS 배제 +
        `PYTHONPATH/LD_LIBRARY_PATH`에 내부 humble rclpy/lib + py3.12 인터페이스 추가 + `ROS_DOMAIN_ID`/
        `RMW_IMPLEMENTATION` 설정 후 `python simulation.py`; (2) conda `sdl` + `source /opt/ros/humble`에서
        tamp/perception 실행. (기존 NVIDIA 런처 + `LD_PRELOAD` + 하드코딩 경로 대체.)
      - [ ] README:42-57 소스빌드, `IsaacSim-ros_workspaces`(3중 overlay, D3), 벤더 `isaacsim/` 런처(D1) 폐기.

      **구현 상태(2026-08-31):** 구현 subagent가 `scripts/run_isaacsim.sh`·`run_tamp.sh` +
      프로젝트-로컬 `ros2_isaacsim_ws/`(네트워크 없는 py3.12 크로스빌드) 작성. **감독 리뷰 결과**:
      대부분 PASS이나 🔴CRITICAL — 인터페이스가 실제로 **py3.10 링크**(파일명만 cpython-312) →
      문자열 필드 publish 시 core-dump. 수정(검증됨): `build_interfaces.sh` cmake에 구식
      `-DPYTHON_INCLUDE_DIR`/`-DPYTHON_LIBRARY` 추가 후 재빌드. 🟠MAJOR: `--check`가 직렬화를 안 해서
      false-pass → serialize 경로 태우게 강화. 🟡MINOR: 주석/README(Demo)/`run_tamp` set -e.
      → **수정 완료·검증됨** (readelf `libpython3.12` ✅, `--check` 직렬화 round-trip 통과 ✅, py3.10 overlay 무손상).
      **검증(감독+headless 실행, 2026-08-31):** ✅ 크로스-인터프리터 DDS interop 증명(py3.12↔py3.10,
      fastrtps 2.6.10↔2.6.12 skew OK, `ToolChange` 문자열 왕복); ✅ headless Isaac Sim이 RTX 5070에서
      기동(Vulkan/CUDA12.9/sm_120, exit 0). → **배관 레벨 통합 완료.** headless 가능 = behavioral 수정을
      자동 검증 가능(사용자 GUI 매번 불필요).
      **남은 항목:** (1) ✅ `simulation.py` **headless 설정화**(`SDL_HEADLESS` env, 기본 GUI 유지) +
      `run_isaacsim.sh --headless` 완료. 전체 씬 브링업 = **Isaac Sim 6.x API 마이그레이션** 필요:
      첫 블로커 `task.py:16` `read_camera_info`가 `isaacsim.ros2.bridge`→`isaacsim.ros2.core`로 이동
      (6.x에서 bridge 확장 분리). 첫 런 ~236s(셰이더 컴파일), **웜캐시 ~9s**. 에셋루트는 S3 해결(네트워크 필요,
      프로젝트 에셋은 로컬). **→ 완료·검증됨**: `task.py:16` 한 줄 수정으로 전체 씬 headless 로드,
      `"Simulation Start"` 도달, 토픽(`raw_ft_data`/`raw_scale_data`/camera/tf)·서비스(get_robot_info/
      tool_change/get_tool_info/isaac_gripper_commands) 전부 기동, 라이브 FT/scale 스트리밍 확인(웜캐시 ~30s).
      **⚠️ 이 환경 기본 `ROS_DOMAIN_ID=100`** — sim↔tamp 두 터미널 도메인 일치 필요.
      **부수 발견(behavioral로 이월):** `world.usd`가 `/home/home/git_clone/LabUtopia/...` 하드코딩 참조
      → I-3(LabUtopia)는 **에셋 레벨**까지 확장; PhysX 경고(삼각메시→convexHull, `box_goal` 음수질량/무효관성)
      → §II 물리 항목 확인. **플랫폼 준비 완료 → behavioral 수정 착수 가능.**
      (2) `tamp`/`perception_manager`/`apriltag_ros` **노드 패키지 colcon 빌드**(py3.10 overlay);
      (3) README **install 섹션(12-97줄)** 재작성(Demo 섹션은 갱신됨); (4) 최종 **실제 GUI end-to-end**(사용자).
- [ ] **I-3 LabUtopia 제거.** README:206-212의 `git clone .../LabUtopia`(rendering 우회, 통합 절차
      없음, E1) 삭제. 실제로 쓰는 asset/설정이 있으면 그것만 커밋 핀 + 출처 표기해 vendor.
- [ ] **I-4 하드코딩 경로/유저명 제거.** `/home/home/...`(README:110, `automate_run.sh:26-27`),
      conda shebang(README:204), 잘못된 `exclude_install_path:=home/home/...`(맨 `/` 누락) → env 변수 /
      ament resource / 상대경로로.
- [ ] **I-5 libstdc++ LD_PRELOAD 핵 제거**(README:128·132, `automate_run.sh:36·38`) — 단일 env로
      정리되면 불필요.
- [ ] **I-6 README/문서 정합.** task/tool 이름을 코드에 맞춤(`move`/`stir`/`ag95`/`dh3`/`vgc10`;
      README의 `pouring`/`stirring`/`2f85`는 오류). python 버전 배지 vs env 통일. `environment.yml` 핀 정리.
- [ ] **I-7 실험 하네스 교체.** `automate_run.sh`(무한 bash 루프 + `pkill` + `sleep`, seed·로깅 없음,
      D5) → **Python 하네스**(seed 0–29, per-trial 위치 randomization, 구조적 CSV 로깅). `PLAN.md` §3-B와 동일.

## II. 물리 · collision 충실도 (P1–P5)

- [ ] **II-1 PhysX scene 설정 명시.** `physics_dt`, substeps, solver iteration, gravity, GPU dynamics
      (`simulation.py`에 현재 전무). 값 기록 → `PLAN.md` §3-C.
- [ ] **II-2 collision을 planner primitive와 일치.** 현재 물체는 raw USD로 스폰(`task.py:256-294`),
      `object.py`의 근사 collision은 **호출 안 됨(dead)**. → 실제 스폰에 cuTAMP와 **동일한 박스 collider** 부여.
      **정정(검증됨): cuTAMP는 `TAMP/tamp/src/envs/utils.py`의 `ENTITIES`에서 모든 엔티티를 `Cuboid`(박스)로
      계획한다** → sim collider도 **cylinder가 아니라 박스**로, 정확히 매칭: beaker `[0.05,0.05,0.135]`,
      flask `[0.07,0.07,0.12]`, box `[0.108,0.108,0.08]`, stirrer `[0.18,0.18,0.09]`. (기존 `create_hybrid_beaker`의
      side 0.055·h 0.12는 cuTAMP와 불일치 → 0.05·0.05·0.135로 수정. visual은 USD 유지, collider만 교정.)
      비고: 실제 beaker는 원형이나 cuTAMP가 박스로 계획하므로 sim도 박스로 맞춰 **계획-실행 자기일관성** 확보;
      원형↔박스 차이는 실제 하드웨어(실제 pouring) 몫으로 남기는 sim-to-real 갭.
- [ ] **II-3 PhysicsMaterial + friction.** finger pad ↔ vessel collision에 static/dynamic friction,
      restitution 0 설정(현재 0건). → 파지 미끄러짐(위치 민감성)의 직접 해결.
- [ ] **II-4 mass/inertia 현실화.** 0.1 kg 하드코딩 → 실제값·CoM. (액체는 sim에서 미모델링 명시.)

> **✅ Phase II 완료·검증됨(2026-09-01, 매니저 spot-check 승인).** object.py 재작성
> (`create_box_collider_rigid/_static`, `_disable_visual_physics` RemoveAPI, `_set_solver_iterations`),
> task.py 스폰/z-offset 교정, simulation.py `configure_physics()`(dt 1/60·TGS·g−9.81·GPU dynamics·iter 8/1).
> collider = cuTAMP 박스 full-extent 정확 일치(검증). PhysX 경고: convexHull 3→0, 음수질량 1→0. 객체 안정 안착.
> **미완/이월**: II-3 friction/material·II-4 현실 mass → **III(grasp)로 이동**(파지 홀딩과 직결); magnet box 일치·LabUtopia world.usd(I-3)·현실 mass는 후속.

## III. Grasping (G1–G5)

- [ ] **III-1 파지 = fixed-joint attach (저자 결정 2026-09-01).** 현재 = 힘/접촉 없는 위치제어 과폐쇄 +
      마찰만(attach 없음, `simulation.py:209 gripper_commands_cb`). → **close 시 대상 물체(ee 근접)를 그리퍼
      링크에 임시 `UsdPhysics.FixedJoint`로 붙이고, open 시 제거.** 결정론적·위치 전반 견고(D3).
      **논문 영향(반드시 반영):** 응답서 실행-성공 기준 (ii) *"파지 물체 grasp frame 5mm 이내 유지"*는
      attach 시 자명해지므로 **삭제/대체** → "무충돌 + 말단조건 도달"로 재정의; execution을 **rigid-body attach
      모델링**으로 명시, 실제 파지 충실도는 실제 하드웨어 몫. (PLAN.md R1#2와 연결.)
- [ ] **III-2 grasp 성공 판정.** FT(`raw_ft_data`) 또는 contact report로 파지 성공/실패 이벤트화
      (현재 판정 없음, G4). 실패 시 fail-closed.
- [ ] **III-3 gripper 코드 단일화.** `SlowParallelGripper`(gripper.py, 미사용) vs `ParallelGripper`
      (fr5.py, 실사용) 중 하나로 통일, 나머지 제거. close 반복횟수(ag95 10/dh3 16)와 delta를
      물리 파지에 맞게 재설계.

> **✅ Phase III 완료·검증됨(2026-09-01).** 새 `isaacsim/scripts/standalone/utils/grasp.py`(FixedJoint 헬퍼) +
> `simulation.py gripper_commands_cb` 통합: close→ee 최근접 물체(≤0.12m) FixedJoint attach(body0=ee링크,
> body1=물체, 현재 상대pose 잠금, excludeFromArticulation), open→해제, `tool_change_cb`에서 선-detach.
> 검증: attach 시 rel drift 0.05mm(rigid), release 시 0.286m 이탈, 근접 없으면 `success=False` fail-closed.
> III-1 ✅ / III-2 ✅(근접 기반, FT 아님) / III-3 미완(SlowParallelGripper 잔존). II-3 friction은 attach라 생략.
> **캐비엇:** 급 slew 시 순간 compliance ~2.9cm(정상상태 sub-mm); 랜덤화 도입 시 threshold(0.12) < 물체간격 유지.
> **논문 반영 예약:** 실행-성공 기준 (ii) 재정의(→무충돌+말단조건) + rigid-body attach 명시 (PLAN R1#2).
>
> **감독 리뷰(2026-09-01) — MAJOR 2건:**
> - ✅ **MAJOR-2 (해결·검증):** attach 스냅 수정. **리뷰 진단(USD↔PhysX 프레임 불일치)은 오진** — probe 결과
>   두 프레임 byte-동일(finger-tip CoM=0), `grasp.py` 수학은 원래 정확. **진짜 원인 = flush 안 된 pose로 author**
>   (+ 테스트가 13.5cm 물체 중심을 fingertip에 teleport한 penetration 아티팩트). **수정 = `simulation.py
>   _attach_nearest_object`에서 `world.step()` flush 후 ee/object pose 재읽기→author**(grasp.py 무변경).
>   증명: disjointed 경고 1→0, attach 시 world 이동 **0.43mm**, rigid drift 0.0mm, release/fail-closed 통과.
>   ✅ **MINOR magnet:** `DynamicCylinder→DynamicCuboid[0.045,0.045,0.03]`(cuTAMP 일치). **Phase III 최종 승인.**
> - 🟠 **MAJOR-1 (랜덤화 단계로):** threshold 0.12m가 ±5cm 랜덤화 시 오물체 파지 위험(실제 기본 간격 0.157m,
>   랜덤화 시 ~0.057m). grasp.py 주석의 "≥0.30m grid"는 오류. **정답: geometric-nearest 대신 planner 의도 물체
>   이름 전달** → I-7 randomization harness와 함께 구현(그전엔 현 배치에서 안전).
> - MINOR: magnet collider가 cuTAMP 박스 `[0.045,0.045,0.03]`와 불일치(파지 후보 → 정합); SlowParallelGripper 잔존; box 안착 ~1cm.

## IV. Pouring (붓는 위치 포함)

- [ ] **IV-1 붓는 위치 정의 명확화.** pour target = `cutamp/envs/pouring.py`의 `pour_region`(Cuboid,
      line 50) + `tamp_server.py:107`의 `pour_region_pose`. 현재 애매 → **대상 vessel(flask) 실제 pose +
      rim 높이 + clearance**로 pour point를 정의하고, 주둥이(spout) 정렬을 명시. MoveHolding→pour-config
      종단조건을 이 정의에 맞춤.
- [ ] **IV-2 θ_max + tilt-rate 제한 구현.** `tamp_server.py pouring()`(:695)에 현재 없음 → 5° 상한 +
      각속도 saturation. `PLAN.md` §3-B/pouring, R3#2.
- [ ] **IV-3 sim 저울 신호 정리.** `compute_scale_data`(task.py:513)는 wrist각의 합성 누적(가짜, P5).
      → sim에서는 "유체 정확도" 주장하지 않도록 스코프, 실제 세션 신호로 대체 준비. PD gain(Kp/Kd)은
      실제 latency에 맞춰 재튜닝(§PLAN 6.2).

> **✅ Phase IV 완료·검증(정적, 2026-09-01).** 실제 pour 경로 = `transfer` env(`pouring.py`는 dead).
> - **IV-1 붓는 위치 ✅:** `TAMP/tamp/src/envs/transfer.py:19`의 `pour_region.z += 0.11`(rim 무관 매직상수,
>   0.12m flask에서만 우연 일치)를 **rim 유도**로 교정: `pour_region = flask.pose; z = flask.z + flask.dims[2]/2 + 0.05`.
>   xy=flask 입구, 임의 vessel 높이 추종. (nominal flask는 z=0.18 동일 → 기존 동작 보존.)
> - **IV-2 tilt-rate ✅:** `tamp_server.py pouring()`에 `v_cmd` clamp ±0.5 rad/s(≈28.6°/s) + 누적 tilt 안전캡
>   2.0 rad. 증명: 무제한 시 927° 폭주 → 제한 시 114.6° clean stop.
> - **θ_max=5° 🔶 논문-코드 불일치 (R3#2):** cuTAMP에 **hard θ≤5° 없음**. 4-DOF placement라 엔드포인트는 정확 수직,
>   IK tol ~2.9°; MoveHolding 경로 중간 orientation 제약 없음 → **창발적**. 응답서 "매 control step θ≤5°"는 강제 아님.
>   **결정 데이터화**: harness 단계에서 **운반 중 실제 tilt 측정** → ≤5° 유지되면 claim 정직 성립(측정), 아니면
>   post-plan FK guard(경로 max-tilt>5° 기각) 추가 또는 claim 축소. 삽입점: `motion_solver.py` FK(:267/:367) 또는
>   cuRobo `PoseCostMetric hold_partial_pose`.
> - IV-3 가짜 scale: 그대로(유체=실제 세션). **전체 plan→execute pour 검증은 live 파이프라인 필요**(다음 단계).

## V. Dead-code / 구조 정리

- [ ] **V-1 dead code 처리.** `object.py`의 `create_hybrid_*`(→ II-2에서 실사용으로 살림),
      `gripper.py`(→ III-3에서 통일), `task.py:56-60` random_grid/noise(→ I-7 harness로 실사용),
      `simulation.py:199 arm_commands_cb`(제거).
- [ ] **V-2 `set_robot` 테이블 구동 리팩터.** 4개 툴 복붙 블록(`task.py:120-217`) → 데이터 테이블 +
      단일 생성 경로. `joints_default_state` 길이 하드코딩 제거.
- [ ] **V-3 단일 소스화.** `get_grid_xy` if/elif 사다리(`task.py:546`, `rearrange.py` 중복) →
      한 곳에서 정의. 매직넘버 상수화.
- [ ] **V-4 타입/견고성.** `set_object -> FR5`(None 반환) 등 잘못된 시그니처 수정; 서비스 콜백 내
      `world.step` 동기 블로킹 재검토.

## VI. 실행 순서 / 의존성

1. **I-1 pip Isaac Sim** → sim 구동 (모든 검증의 전제).
2. **I-2~I-6 워크스페이스/문서/LabUtopia 정리** (sim 독립 — 1과 병행 가능).
3. **II 물리/collision/material** → 파지·pour의 물리 토대.
4. **III grasping** (II 위에서).
5. **IV pouring 위치 + θ_max**.
6. **I-7 randomization+seed+logging 하네스** → 위치 전반 재측정.
7. **재측정 → `PLAN.md` 숫자 갱신** (D3, 정직한 headline).

> 각 코드 변경은 sim 구동(1) 이후 in-sim 검증 필요. 검증 전 항목은 "작성됨/미검증"으로 표시.
