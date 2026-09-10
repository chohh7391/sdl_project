# HANDOFF.md — 인수인계 브리핑 (다음 에이전트/Codex용)

이 문서는 이 작업을 이어받는 에이전트를 위한 것이다. **먼저 이 문서를 끝까지 읽고, 그다음
아래 "필독 문서"를 순서대로 읽어라.** 이 저장소의 4개 관리 문서(CLAUDE/PLAN/REFACTOR/ORCHESTRATION)가
진짜 소스이며, 이 HANDOFF는 그 지도를 준다.

> **2026-09-08 갱신:** §4·§5의 "설계결정 3개 확인 요청"은 09-01에 확정·구현 완료(ORCHESTRATION §D/§E). 그 뒤 09-01 18:27~09-03에
> Codex가 넣은 비의도 변경(contact-evidence 프로토콜, 6-DOF side-grasp 등)은 **2026-09-08 전부 되감음** — 현재 코드 = 09-01 17:54 검증 상태.
> 최신 상태는 항상 `SESSION_LOG.md §0`. 작업 트리가 문서와 다르면 `~/.codex/sessions`·`~/.claude/projects` 기록으로 출처를 먼저 확인할 것.

---

## 0. 이게 무슨 프로젝트인가 (한 문단)

**IEEE Access 논문 재제출(revision) 작업**이다. 논문 `Access-2026-33062` "LLM-Guided Tool-Aware
Task and Motion Planning for Chemistry Lab Automation"이 3명 리뷰어의 지적으로 reject되었고,
**단 1번의 재제출 기회**가 주어졌다(binary review — 모든 concern을 닫아야 accept). 논문의 코드가
`sdl_project/`(LLM→XDL→cuTAMP TAMP→Isaac Sim 실행 파이프라인)인데, **코드가 부실해서 논문이 주장하는
숫자(예: 100% planning success)를 현재로선 재현 못 한다.** 그래서 목표는: **코드를 정직하게 고쳐서
재현 가능·위치-견고한 실제 숫자를 만들고**, 그 숫자로 리뷰어 concern에 대응해 재제출하는 것.

**절대 원칙(CLAUDE.md): 숫자를 지어내지 않는다.** 논문의 모든 값은 "개선된 코드를 실제로 돌린
측정치"여야 한다. 못 고치면 주장을 현실에 맞게 축소한다.

---

## 1. 필독 문서 (순서대로, `/home/home/sdl_ws/src/sdl_project/`)

1. **`CLAUDE.md`** — 불변 가드레일. 마커 규칙(`\nd{}`=빨간 미완 placeholder, `\rev{}`=노란 하이라이트),
   무결성 규칙, 재도입 금지 과장주장 4개, 전략결정 D1–D6. **매 세션 자동 로드됨.**
2. **`IEEE_ACCESS_review.md`** — 3 리뷰어의 실제 지적(입력). 무엇을 대응해야 하는지의 원천.
3. **`PLAN.md`** (한글) — 논문 revision 마스터 플랜: 리뷰어 concern 16개 → 작업 매핑표, A/B/C 실험
   카탈로그, 실제 pouring 세션 프로토콜, 최종 제출 게이트.
4. **`REFACTOR.md`** (한글) — **코드 작업의 핵심 로그.** 리뷰에서 발견한 코드 문제(P1-P5 등), 실행방식,
   🎯핵심 진단(planning 실패 근본원인), Phase I(설치)~IV(pouring) 진행상태, STAGE A/B 결과, robustness 수정
   결과, 그리고 오케스트레이션 재설계 결정. **현재 상태는 여기가 가장 정확.**
5. **`ORCHESTRATION.md`** (한글) — **바로 다음에 구현할 것의 설계 spec.** 오케스트레이션 계층 재설계.
6. 논문 파일: `_2026__IEEE_Access/access_revised.tex`(수정 원고, `\nd{}` 빨간 placeholder ~210개),
   `_2026__IEEE_Access/revision/response_to_reviewers.tex`(응답서), `_2026__IEEE_Access/access.tex`(원본, 수정금지).

---

## 2. 환경 / 실행법 (검증됨)

**2-env 구조 (py 버전이 달라 강제 분리, DDS로 통신):**
- **`.venv-sdl`** (py3.12): `/home/home/sdl_ws/src/sdl_project/.venv-sdl`. **Isaac Sim 6.0.1**(pip,
  `isaacsim==6.0.0.1`) + `simulation.py`. Isaac Sim 내부 Humble ROS(rclpy py3.12) 사용.
- **conda `sdl`** (py3.10): `/home/home/anaconda3/envs/sdl`. torch 2.7/cu128 + **cuTAMP/cuRobo** +
  tamp_server + 시스템 ROS2 Humble.
- 시스템 ROS2 = **Humble/py3.10** (`/opt/ros/humble`). **이 환경 기본 `ROS_DOMAIN_ID=100`** (sim↔tamp 두 터미널 일치 필요).
- GPU: RTX 5070(Blackwell, 12GB, CUDA 12.x). `~/INTACT`의 무관한 IsaacLab job이 GPU 공유할 수 있음 — `nvidia-smi` 먼저.

**실행 스크립트 (`sdl_project/scripts/`):**
```bash
# 터미널 1 — Isaac Sim (headless 가능)
SDL_HEADLESS=1 bash scripts/run_isaacsim.sh --headless   # --check 로 ROS env 자가검증
# 터미널 2 — TAMP (conda sdl)
bash scripts/run_tamp.sh                                  # --parser 로 xdl parser
# py3.12 커스텀 인터페이스 재빌드(필요시): bash ros2_isaacsim_ws/build_interfaces.sh
```
- 첫 Isaac Sim 부팅 ~4분(RTX 셰이더 컴파일, 1회성), 이후 **웜캐시 ~30s**. headless로 전체 씬 로드+`"Simulation Start"` 검증됨.
- **transfer plan→execute 구동법**: sim + tamp_server 띄운 뒤(둘 다 `ROS_DOMAIN_ID=100`),
  `set_tamp_cfg fr5_ag95 → set_tamp_env transfer → plan → execute`를 ROS2 서비스로 호출.
  (참고 드라이버가 scratchpad에 있었으나 세션별로 사라짐 — `TAMP/tamp/scripts/client/tamp_client.py`,
  `scripts/trials/trial_driver.py` 참고. **주의: transfer는 반드시 `fr5_ag95`(그리퍼) config**; bare `fr5`는 파지 불가로 planning 0/1024.)

**네트워크 주의**: 에이전트 툴 셸은 외부 네트워크가 막힘(HTTP 000). `git clone`/`pip`/`apt`는 사용자가
터미널(`!`)에서 실행해야 함. 로컬 빌드/실행/편집은 문제없음.

---

## 3. 지금까지 한 일 (요약 — 자세한 건 REFACTOR.md)

- **설치/launch/플랫폼 통합 ✅**: 소스빌드/별도 ROS ws/LabUtopia clone/LD_PRELOAD 핵/하드코딩 경로 폐기 →
  **pip Isaac Sim 6.0.1 + 2-env DDS**. py3.12용 커스텀 인터페이스를 **네트워크 없이 크로스빌드**
  (`ros2_isaacsim_ws/build_interfaces.sh`). clean launch 스크립트. 전체 씬 headless 구동·검증.
- **Phase II 물리/collision ✅**: 물체 collider를 cuTAMP `ENTITIES` 박스 치수에 정확 매칭(`object.py`
  `create_box_collider_rigid/_static`), PhysX 경고(삼각메시 convexHull, box_goal 음수질량) 제거,
  `simulation.py configure_physics()`(dt 1/60, TGS, g, GPU dynamics).
- **Phase III 파지=fixed-joint attach ✅** (저자 결정): `grasp.py` + `simulation.py gripper_commands_cb`.
  close→ee 근접/planner-지정 물체를 FixedJoint weld, open→해제, fail-closed. attach 스냅 버그도 수정
  (원인=flush 안 된 pose; `world.step()` 후 재읽기). **MAJOR-1**: 파지가 planner 의도 물체를 타깃(`/set_grasp_target`).
- **Phase IV pouring ✅**: 붓는 위치를 flask rim 유도로 교정(`transfer.py`), pouring tilt-rate clamp(±0.5 rad/s)+안전캡(`tamp_server.py`).
- **🎯 planning 실패 근본수정 ✅**: 원인 = ①bare fr5는 진짜 불가능(그리퍼 필요) ②`tamp_plan_cb`가 sim을
  pause해서 `/isaac_joint_states` 끊김→`q_init` 빈 q0→reshape 크래시(→motion_solver q0 복구+q_home fallback)
  ③bare `except`가 성공 plan을 실패로 삼킴(→exception 표면화) ④`time_dilation_factor=0` ZeroDivision(→0.5 기본).
  결과: **transfer plan→execute end-to-end 성공.**
- **STAGE B1 harness ✅**: seeded 위치 randomization(`SDL_SEED`, ±5cm/±180°/±10°, `task.py`), per-trial CSV
  (`_2026__IEEE_Access/revision/analysis/data/transfer_trials.csv`), fresh-server-per-trial 배치.
- **robustness 수정 ✅ (transfer 3/5 → 5/5)**: **FIX A** upright — cuRobo `hold_partial_pose`로 carry op에서
  ee roll/pitch 고정 + post-plan FK guard로 운반 tilt 측정(`motion_solver.py`) → seed4 90°→≤5°, **θ_max 주장
  정직화**. **FIX B** — cuTAMP `approximate_goal_aabb`(`utils/common.py`) **수학버그**(yawed surface AABB를
  대각 2코너만 변환→bounds 반전)를 8코너 min/max로 수정 → seed 0,2 실패→성공. **"위치별 천차만별"의 상당부분이 실제 버그였음.**

---

## 4. 지금 정확히 어디인가 (현재 상태)

- **cuTAMP robustness 수정 완료·검증**: transfer seed 0,1,2 live로 plan+execute 성공, tilt 전부 ≤5°.
  seed 3,4,nominal은 live 미재실행(회귀 기전 없음). **B2(30-seed×3-task) 전수평가 준비됨.**
- **오케스트레이션 계층 재설계 = 결정+설계 완료, 구현 전.** 저자가 "오케스트레이션 계층만 재작성"으로
  결정. 설계 spec = `ORCHESTRATION.md`. **바로 여기서 이어받으면 됨.**
- 저자에게 마지막으로 확인 요청한 3개 설계결정(기본=권장): ①대부분 무상태 planner ②config-switch는
  process-per-config ③LLM/XDL 경로도 통합. **사용자 확인 답을 받고 시작할 것.**

---

## 5. 바로 다음에 할 일 (우선순위)

1. **[확인]** 위 3개 설계결정을 사용자에게 확정받는다(ORCHESTRATION.md §D).
2. **오케스트레이션 재설계 구현** (`ORCHESTRATION.md` 순서 C): 설정/환경 레지스트리(B2)+무상태 planner API(B3)
   → 단일 오케스트레이터(B1) → pause 제거+pouring 명시 step(B5/B4) → 결정론 배치 러너(B6) → LLM/XDL 통합.
   **매 단계 작동하는 transfer end-to-end + seeded 배치로 검증**. 옛 코드는 검증 전까지 레퍼런스 유지.
3. **B2 전수평가** — 클린 harness에서 **transfer/move/stir × seed 0-29**(+nominal) 무인 실행 → CSV →
   **논문 헤드라인 숫자**(planning/execution 성공률, planning time, 운반 tilt).
4. **오프라인 통계** — Wilson CI, (PDDLStream 오면) exact McNemar, KM/restricted-mean time-to-solution.
   `_2026__IEEE_Access/revision/analysis/`에.
5. **논문 반영** — `access_revised.tex`/`response_to_reviewers.tex`의 빨간 `\nd{}`를 실제 측정치로 채우고,
   아래 §7 "논문에 반영할 발견"을 반영. `grep -n '\nd{'`가 두 파일에서 비어야 제출.
6. **저자 인바운드 대기 중**: **PDDLStream baseline**(`experiments/pddlstream/`에 이미 있음 — R1#3/#4) +
   **LLM 인프라**(`LLM/llama/script/experiments/`: test set, split, validator 벤치, action reasoner) —
   재구현 말고 **통합·검증만**.
7. **실제 하드웨어(사용자/lab)**: RS-232 저울로 **실제 pouring 세션 1회**(R2#2, 유체는 sim에서 가짜라 실제 측정
   필요) — `PLAN.md §6/§7`. 코드는 ingest/표만 준비.

---

## 6. 작업 규칙 (반드시 지킬 것)

- **무결성**: 숫자는 개선된 코드의 실제 측정. **날조 금지.** 측정 안 됐으면 `\nd{}`로 빨갛게 둔다.
- **마커 규칙**: `\nd{}`=빨간 미완(제출 전 `grep '\nd{'` 두 tex에서 0), `\rev{}`=노란 하이라이트(원본 대비 변경).
  빌드 2개: `\highlighttrue`(Highlighted PDF), `\highlightfalse`(clean). 응답서는 `\showstatusfalse` 전 정리.
- **주장은 증거 범위로 축소**(리뷰어 핵심 불만이 과장). 재도입 금지 4개 과장: "complete coverage",
  "no additional failure modes", "decisive factor across workflows", "reduced variance".
- **원본 `access.tex` 수정 금지**(하이라이트 diff 기준). 저자 목록 변경 금지.
- **실행방식(저자 지정)**: 구현은 subagent(또는 네 실행)로, 계획/통합은 관리. 마일스톤마다 감독 리뷰.
  워크스페이스/빌드 건드리는 태스크는 충돌 방지 위해 직렬로. **검증은 실제 실행 기반, 날조 금지.**

---

## 7. 논문에 반영할 발견 (누적 — §5.5에서 처리)

- **θ_max=5°**: 이제 코드가 **실제로 강제+측정**(FIX A). 응답서 실행-성공 기준을 이에 맞춰 서술 가능.
- **파지=fixed-joint attach** (저자 결정): 실행-성공 기준 (ii) "5mm 유지"는 attach 시 자명 → **"무충돌+말단조건"으로 재정의**
  + "rigid-body attach 모델링" 명시. (PLAN R1#2)
- **sim scale은 가짜**(유체 미모델링, `compute_scale_data`가 tilt 프록시) → sim pouring은 **유체 정확도 주장 X**;
  실제 pouring 세션이 유일한 유체 근거(R1#2/R3#1).
- **planning "100%"**: 현 실측은 소규모 5/5(수정 후). **B2 전수로 정직한 헤드라인** 확정 후 기재.
- **world.usd에 LabUtopia 하드코딩 경로**(`/home/home/git_clone/LabUtopia/...`) → 재질 fallback으로 로드됨(비치명).
  정리 대상(REFACTOR I-3); 논문엔 무관.
- **Isaac Sim 버전 = 6.0.1**(draft의 4.2.0/README 5.1.0은 6.0.1로 통일; R1#8). 카메라 실제 = L515 640×480(draft D435와 불일치 → 실제로).

---

## 8. 함정 / 운영 노트

- **subagent async-batch 함정**: GPU 배치를 background로 던지고 waiter로 기다리면 subagent가 계속 멈춘다.
  → **trial을 동기(foreground blocking)로 하나씩** 돌려라.
- **config-switch 오염**: 한 planner 프로세스에서 `set_tamp_cfg`를 2회+ 하면 cuRobo GPU 상태(CUDA-graph/cached IK)
  오염으로 IK가 0/1024로 붕괴. → **config마다 fresh planner 프로세스**(재설계에서 근본 처리; 그전엔 fresh-per-trial).
- **sim pause**: `tamp_plan_cb`가 sim을 pause하고 execute까지 안 되살림 → joint state 끊김. 재설계에서 pause 제거 예정.
- **sim pouring scale=0**: 정상(가짜 scale). pour 루프가 target 못 채워 안전캡으로 정지 — 유체 없음의 결과.
- **`pkill -f "isaacsim"`** 류(옛 `automate_run.sh`)는 무관 프로세스도 죽임 — 쓰지 말 것.
- 파일 편집은 **소스 + colcon-install 사본** 양쪽 신경(예: `tamp`는 `/home/home/sdl_ws/install/tamp/lib/tamp/`에도).
  `cutamp`는 editable-install이라 소스 편집이 바로 반영.

---

## 9. 이번 세션에 편집된 핵심 파일 (git status로 확인)
- `isaacsim/scripts/standalone/{simulation.py, task.py, utils/object.py, utils/grasp.py(신규)}` (II/III + headless + read_camera_info 6.x)
- `TAMP/cuTAMP/cutamp/{motion_solver.py, utils/common.py}` (planning-fix + robustness FIX A/B)
- `TAMP/tamp/scripts/server/tamp_server.py` (exception 표면화, q_home fallback, time_dilation, grasp_target, tilt clamp)
- `TAMP/tamp/src/envs/transfer.py` (pour-region rim)
- `scripts/{run_isaacsim.sh, run_tamp.sh, run_trials.sh, trials/*}`, `ros2_isaacsim_ws/*`(신규 인터페이스 빌드)
- 관리 문서: `CLAUDE.md, PLAN.md, REFACTOR.md, ORCHESTRATION.md, HANDOFF.md(이 문서)`
</content>
