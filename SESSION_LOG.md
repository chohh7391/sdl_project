# sdl 세션 로그

컨텍스트가 끊긴 뒤에도 이어갈 수 있도록 남기는 인수인계 문서.
좌표·실행법·고유 규칙은 `CLAUDE.md` 에 있다 (자동 로드됨).
**이 파일은 "무엇을 왜 결정했고 무엇이 열려 있나"** 만 담는다.

---

## 0. 지금 상태

**세 task 모두 randomized harness에서 실측 완료: transfer 30/30, stir 30/30, move 29/30. 단 move(B5)는 이전 코드 버전이라 재실행 필요. 그리고 구매한 실물 유리기구와 시뮬 모델이 불일치(저자 결정 대기).**
- stir B6 `stir_b6_20260910.csv` (**stir 최초 완주**): **30/30 [Wilson 88.6–100]**, planning 중앙 33.5 s(30/30 ≤60 s),
  운반 tilt 최대 4.40°(30/30 ≤5°), vessel 최종 tilt 최대 0.02°, stirrer 중심 오차 중앙 5.9 mm,
  **stir bar가 vessel 안에 30/30**(축에서 중앙 6.5 mm, 입구보다 100–112 mm 아래 = 내부 바닥).
- transfer B7 `transfer_b7_20260910.csv` (B4 재실행, 한 코드 버전으로 통일): **30/30**, 최종 tilt 전부 0.00°,
  운반 tilt ≤5° **27/30**(B4는 24/30), planning 중앙 55.3 s(60 s 내 22/30), 목표 오차 중앙 13.8 mm.
- move B5 `move_b5_20260910.csv`: 29/30(실행된 29건은 29/29; 1건은 Isaac Sim `world.stop()` 크래시).
  **단 B5는 그리퍼 개방 수렴·속 빈 flask·stir bar 변경 이전 코드** → 버전 일치를 위해 재실행 필요.
- stir가 막혀 있던 것은 파지가 아니라 ①dh3 홈 자기충돌 ②**그리퍼가 다 열리지 않음**(dh3 open 3 step, 실측 필요 22–23;
  ag95도 10 vs 필요 11) ③flask가 속 찬 상자여서 bar가 들어갈 수 없음 ④`beaker_region_in_xy` 임계값 1e-1(다른 영역 1e-3).
  전부 계측으로 확정하고 고쳤다(§1 2026-09-10 (2), REFACTOR).
- harness 신뢰성: 죽인 배치의 sim이 살아남아 같은 ROS 도메인에 응답해 **드라이버가 다른 seed의 월드를 읽는 사고**가 있었다
  (seed 3개가 연속으로 seed 4의 좌표를 읽음). seed마다 stray 프로세스를 정리하고 **읽어온 포즈 vs 스폰 포즈 정합성 검사**를 추가.
  B6·B7은 stray 0, mismatch 0.
- **저자 결정 대기**: 구매 실물(beaker 높이 72 mm·외경 60 mm·100 mL, flask 높이 160 mm·입구외경 38 mm·300 mL)과
  시뮬 모델(beaker 50×50×135 mm, flask 70×70×120 mm·개구부 64 mm)이 다르다. stir bar 10×35 mm는 실물 입구(내경 약 34 mm)를
  여유롭게 통과하므로 **magnet은 변경 불필요**하지만, 유리기구를 실물에 맞추면 B4–B7 전부 재실행이 필요하다.
- 열려 있는 것: 위 결정, move 재실행, PDDLStream 베이스라인·AprilTag in-the-loop 미착수.
- **레이아웃 변경(2026-09-10)**: 원고·문서·`.conventions`를 모두 `sdl_project/` 안으로 옮겼다.
  이제 저장소 루트 = 프로젝트 루트 = 세션 루트 = `/home/home/sdl_ws/src/sdl_project`(remote가 clone할 수 있는 형태).
  코드에서 깨진 참조는 `run_trials.sh`·`run_nominal_control.sh`의 기본 CSV 경로 2곳뿐이었고 수정했다.
  **아직 커밋/푸시 안 함** — `.gitignore`가 얇아(`build/ install/ log/ .vscode/ *.pyc`) 미추적 약 4 GB가
  그대로 잡히고, 100 MB 초과 파일 2개(`LLM/llama/script/experiments/ActionReasoner.zip`,
  `experiments/pddlstream/downward/builds/release/bin/downward`)는 GitHub가 거부한다.

## 1. 기록

<!--
  세션별 상세를 append. `### YYYY-MM-DD` 헤더.
  200줄 넘으면 docs/sessions/<YYYY-MM-DD>.md 로 밀어내고 여기엔 한 줄 색인만 남긴다:
    - 2026-08-03~05 dense 스케일 조사, α 지형 문제 발견 → docs/sessions/2026-08-05.md
-->

### 2026-09-01
- ORCHESTRATION §D 세 결정을 모두 권장안으로 확정하고 C1–C3 구현·실측 완료.
- canonical 설정을 찾는 과정에서 legacy XDL의 collision 목록과 AG95 grasp_dof=6 drift가 실제
  seed 0 실패를 일으킴. 검증된 목록과 grasp_dof=4로 고정 후 E2E 회복.
- planning 실패 시 None plan encode로 원인이 가려지는 오류도 fail-closed 처리.

### 2026-09-08
- **발견:** SESSION_LOG(09-01 17:57) 이후 Codex 세션이 09-01 18:27 → 09-03 11:31까지 sdl 코드를 대량 수정했으나 문서화 0건,
  라이브 성공 기록 0건(마지막 성공 = 09-01 18:24 `/tmp/gui_transfer.csv` 39 s/54/✓/3.69°, 검증 상태 직후). 저자 확인: **의도한 변경 아님 → 되돌리기.**
- **09-02 코드 스모크(되감기 전):** transfer seed0 planning 3회 모두 `IK_FAIL`("Failed to plan from approach"; 6-DOF side-grasp 접근점 도달 불가), 244.7 s.
- **되감기 방법:** `~/.codex/sessions/**` 253개 rollout에서 `item_completed/FileChange`(파일별 unified diff) 478건 수집 → 시각순 원장 →
  cutoff(2026-09-01T08:54:42Z, seed2 E2E 성공) 이후 456건 중 문서 4개(SESSION_LOG/REFACTOR/ORCHESTRATION/AGENTS)와 q_home 2줄 수정만 제외하고
  스테이징 복사본에 `patch -R -F0`로 역순 적용(fails=0, offsets=0; add된 14개 파일은 역적용 후 원본 내용과 정확히 일치 → 삭제) → 실제 트리 반영.
  FileChange 없는 apply_patch 21건은 모두 "verification failed"(무효)로 확인. 검증 실행 로그의 문자열이 복원 코드와 1:1 일치.
- **USD:** Codex가 pxr 스크립트로 `/home/home/git_clone/LabUtopia/` 접두어를 `third_party/LabUtopia/` 상대경로로 재작성(world/heat_device/stirrer.usd).
  검증 실행 sim 로그에 git_clone 경로 경고 260건 → 원본 사용 확인 → world/stirrer.usd `git checkout`. heat_device.usd는 미추적이라 재작성본 유지(렌더 재질 참조만 영향).
- **인터페이스:** srv 되감기 후 py3.10(`colcon build --packages-select tamp_interfaces`)·py3.12(`ros2_isaacsim_ws/build_interfaces.sh`) 둘 다 클린 재빌드, `--check` 통과.
  09-02에 sdl_project 안에 생긴 stray `build/ install/ log/`(colcon), `.pytest_cache`, `__pycache__` 제거.
- **복원 검증(라이브, `scratchpad/verify/transfer_verify.csv`):** seed0 54.18 s/73/1 attempt/✓/3.55°, seed1 60.21 s/166/1/✓/3.03°, seed2 63.42 s/515/1/✓/3.94°.
  09-01 검증(51.06/26.64/39.27 s) 대비 planning time 1.5–2.3배: 시간 분해로 확인 — 최적화 1000 step **29 s(33.9 it/s) → 47 s(21.0 it/s)**, load/warmup도 +30%.
  동시에 무관 프로세스 `/home/home/env_isaaclab/bin/python`(866 MiB, CPU 109%)가 GPU 71% 점유 중이었음 → 코드 회귀가 아닌 GPU 공유로 판단.
  (cuRobo .so가 검증 실행 뒤 09-01 23:36에 재빌드된 점은 남은 변수 — GPU 단독일 때 seed 0–2 재측정으로 확정할 것.)
- **cuRobo .so:** 09-01 23:36 재빌드는 pyproject.toml(setuptools_scm fallback) 변경에 따른 `pip install -e` 재실행. .cu/.cpp 변경 없음 → .so는 그대로 둠, pyproject만 복원.
- **열린 문제(변경 없음, 기록만):** ① 실패 시 3×~26 s 재시도 → 논문 60 s 예산과 불일치(budget 기반 종료 필요). ② 검증 경로는 planner 전용 z-lift
  `offset=0.01`(envs/utils.py)로 초기충돌을 피함 — Codex는 이를 제거하고 beaker 전용 contact-evidence로 대체했다가 실패. move/stir 확장 시 일반 해법 필요.
  ③ move/stir는 randomized harness로 아직 0회 실행.

### 2026-09-08 (2) — 측면 파지 구현 + B2 transfer 배치
- **저자 지시**: "자연스럽게 잡게 돼야 해" → 논문의 side-grasp pouring과 코드(top 파지)의 불일치 해소. 상세 진단·수정은 `REFACTOR.md` "측면 파지".
- **핵심 발견(FK 실측)**: pour 관절(joint 6)은 EE 접근축을 돌린다 → top 파지에서 비커 기울기는 **어떤 각도에서도 0.00°**,
  side에서만 관절과 1:1. sim의 가짜 저울(손목각 프록시)이 이 사실을 가려왔다. **논문이 주장한 실험은 어떤 설정으로도 재현 불가였다.**
- top 파지 계열은 `sample_yaw(num_faces=4)`로 **서로 다른 파지가 4개뿐**이고 grasp는 최적화 대상이 아니다
  (`types_to_optimize={Pose,Conf}`). IK 전수조사: 4개 중 1–3개만 도달, **seed 3·5·18·20은 0개** = 구조적 실패 하한.
  측면(방위×높이 108후보)은 30/30 도달, Pick+pour+place 동시 만족 후보 122–154개(top은 중앙값 2개).
- **막고 있던 것 = "그리퍼는 잡는 물체에 닿는다"를 두 계층이 충돌로 취급**: ①cuTAMP `robot_to_movables` ②cuRobo 모션 세계.
  top은 손가락이 rim 위에 멈춰 우연히 둘을 피했다. 각각 시점별/구간별로 예외 처리해 해결.
- 접근 경사 β를 측정으로 도입: grasp+pre-grasp 동시 도달률 **46.9%(수평) → 69.2%(30°)** → β~U[10°,35°].
- pre-grasp는 cuTAMP가 제약하지 않아 "최적" 파지가 접근 불가일 수 있었다 → 초기화에서 pre-grasp IK도 풀어 입자 재배정
  (배치크기 동일 → CUDA 그래프 유지). 실측 도달 651–706/1024.
- upright 가드를 **경고 → 기각**으로. `hold_partial_pose`는 soft cost라 trajopt가 위반 계획을 반환했다(s3 154°).
- pour 후 rate 제한 un-tilt 추가(이전엔 56° 기울어진 채 놓았다). 홈 복귀 실패는 과제 외이므로 `GO_HOME_FALLBACK`로 기록·유지.
- 실행 재생 주기를 `SDL_EXEC_DT`로 설정화(기본 0.04 유지). **seed 3에서 0.04→0.08 s로 실행 tilt 9.91°→2.50°** → 배치는 0.08 사용.
- **배치 결과(위 §0)**: planning 30/30. 실행 tilt는 미해결 — 추종 오차(속도 의존)와 전도(s22) 두 원인이 섞여 있다.
- seed 0은 1차 배치에서 Isaac Sim이 `Simulation Start` 전에 죽어 행 없음(인프라 실패) → 재실행해 30/30 완성.
- **논문 반영 예정**: planning 30/30은 방어 가능. θ≤5°는 **문서화된 실행 속도에서의 실측**으로 기술해야 하며 현재 값으론 미성립.
  execution 성공 기준(R1#2)에 tilt·배치 조건을 넣어야 한다(현재 판정은 궤적 완주만 본다).

### 2026-09-09 — 릴리스(놓기) 실패 규명과 수정, B4 배치
- **문제**: B3에서 실패 7/30이 모두 최종 tilt 정확히 90.00°. 운반 tilt는 5건이 1.37–4.06°로 정상이었다 → 운반이 아닌 릴리스 문제로 좁힘.
- **계측 추가**: `simulation.py`에 `RELEASE_TRACE` — 릴리스 전/해제 후/개방 후 + 3 s 동안 대상 물체의 위치·tilt·|v|·|w| 기록.
  이전에는 `carried_tilt_deg`가 해제 즉시 −1을 내보내 **정확히 중요한 구간이 관측 불가**였다.
- **관측(seed 5, 수정 전)**: pre_release z=0.0851 tilt 0.30° |w|=0.026 → after_detach 동일 → **after_open tilt 28.85° |w|=1.98**
  → t+012에 91.06°, 7.5 cm 이동 후 정지. 즉 **손가락 개방이 비커를 긁어 넘어뜨린다**.
- **수정 1 (순서)**: 용접은 "손가락이 잡고 있다"를 대신하므로 손가락이 놓은 뒤에 풀어야 한다. `detach_first` → **`open_first`**
  (`SDL_RELEASE_ORDER`로 전환 가능). 실측: 같은 seed에서 개방 후 0.06° / 0.015 rad/s.
- **수정 2 (낙하 높이)**: `goal_region`은 **Isaac Sim에 prim이 없는 플래너 전용 표면**이고 그 윗면이 0.020,
  거기에 배치 목표가 `+2 mm`, 반대로 movable의 플래너 포즈는 `PLANNER_Z_LIFT=0.01`만큼 올라가 있어
  실제 비커 바닥은 테이블 위 **12 mm(실측 15.2 mm)**에서 놓였다. `goal_region` z를 **테이블 윗면 + lift**에서 유도하도록 바꿈
  (`envs/constants.py` 신설, `transfer.py`에서 계산 → z=0.005). 실측 낙하 **3.8–9.6 mm**.
- **수정 3 (retract)**: 최종 retract는 놓인 비커를 충돌 제외한 채 자유 계획이라 손이 비커를 통과·타격할 수 있었다
  (seed 3: 정지 후 t+174에 10.44°, 1.2 cm 밀림). 파지 접근과 동일한 **직선(자세 고정) retract**로 교체, 실패 시 자유 계획 폴백.
- **수정 4 (목표 영역)**: `StablePlacement`의 `_in_xy`는 **구 중심**만 영역 안에 넣으므로 비커 절반이 걸친 배치가 합법이었고,
  실행 오차 몇 mm에 영역 밖으로 나갔다(seed 5: 0.414, −0.293 = 중심에서 6.4 cm). AABB를 구 반지름만큼 축소해
  **발자국이 영역 안**을 뜻하게 바꿈. 물체가 표면보다 크면 중심점으로 수렴하도록 클램프. 비용은 미미(pour_region 967–995/1024).
- **회귀 검증(seed 5·3·1·0·2, 실패 3 + 성공 2)**: **5/5 task_success**, 최종 tilt 전부 0.00°,
  목표 중심 오차 0.4–1.3 cm, 릴리스 후 3 s 동안 재교란 없음. 직선 retract 폴백 0회, `GO_HOME_FALLBACK` 0회(B3는 12/30).
- **B4 배치**: 30 seed, `SDL_EXEC_DT=0.08`. 결과는 §0.

### 2026-09-10 — move/stir를 harness에 올림, move 배치 B5
- harness를 task별로 일반화(`TASK`/`ROBOT`, `TASK_OUTCOMES`, 새 CSV 컬럼 3개). 상세는 REFACTOR "move / stir".
- **move는 목표가 시작 위치와 동일**했고 box가 트레이를 10 mm 관통해 스폰됐다 → 트레이를 G3로 옮겨 0.30 m 운반 과제로 만들었다.
  트레이는 랜덤화 대상이 아니라 seed 레이아웃은 불변(transfer seed 0으로 확인).
- 흡착은 용접이 없어 운반 tilt가 전 구간 미계측이었다 → `SurfaceGripper.is_closed()` 기준으로 publish.
- **stir 차단 원인은 홈 자세**: `shoulder_link`↔`finger1_*`가 dh3에서 겹친다. 명령된 홈은 통과하나 실제 정착 자세는 실패.
  dh3 전용 j3 +0.10 rad(측정으로 선택). 적용은 `tool_change_cb`의 관절 복원 지점에 **오프셋 차분**으로.
- 릴리스 전 정지(`SDL_PLACE_SETTLE_STEPS`=15)를 추가. stir에서 |v| 0.36→0.089 m/s로 줄었지만 |w|는 7 rad/s로 남았다.
- 운반 tilt가 stir에서 vessel과 stir bar를 섞고 있었다 → sim이 `carried_obj`를 함께 publish, 드라이버가 대상만 집계(29.98°→2.26°).
- B5 결과는 §0. `summarize_trials.py`도 task별 판정 문구/지표를 쓰도록 고쳤다(B4 숫자는 불변 확인).

### 2026-09-10 (2) — dh3 파지 안정화
- stir 해제 후 flask가 **등속으로 상승**(25 mm/s, xy 고정) → 물리 폭발이 아니라 **아직 잡혀서 팔이 들어올린 것**.
- 원인: 병렬 그리퍼는 호출당 델타 구동인데 open 분기가 고정 횟수였다. dh3는 3 step(필요 5 step 이상, 지연 포함 실측 22–23),
  ag95도 10 step(실측 필요 11). → 횟수 대신 **열림 목표 도달까지 구동+검증**(`_open_gripper`, 상한 40, 미도달 시 WARNING).
- 검증: stir seed 0–4 파지·배치 5/5(flask 최종 tilt 0.01–0.04°), transfer 0·1 무회귀, move 0·1 무회귀.
- 남은 것은 형상 문제(§0): flask가 속 찬 박스라 stir bar가 들어갈 수 없다. 결정 대기.

### 2026-09-10 (3) — stir 배치 B6 + transfer 재실행 B7
- stir 30/30, transfer 30/30. 상세 수치는 §0, 코드/설정은 각 로그 디렉터리의 RUN_INFO.txt.
- stir를 막던 4개 원인과 harness 오염 사고는 REFACTOR "dh3 파지 안정화" 절에 계측치와 함께 기록.
- 실물 유리기구 치수를 저자가 알려줌 → magnet(10×35 mm)은 실물 입구 38 mm를 통과하므로 유지. 유리기구 모델 갱신은 결정 대기.
