# PLAN.md — IEEE Access 리비전 (Access-2026-33062)

재제출을 위한 마스터 플랜 + 실험 카탈로그 + 상태표. 가드레일(마커 규칙, 무결성 규칙,
철회한 주장)은 먼저 `CLAUDE.md`를 볼 것. **재제출 기회는 단 1번** — 세 리뷰어의 모든
concern을 닫지 못하면 reject (binary review).

**제출 빌드 전 하드 게이트:** `_2026__IEEE_Access/access_revised.tex` 와
`_2026__IEEE_Access/revision/response_to_reviewers.tex` 두 파일 모두에서
`grep -n '\nd{'` 가 **아무것도 안 나와야** 함.

상태 범례: `[ ]` 할 일 · `[~]` 진행 중 · `[x]` 완료 · `[!]` 막힘/입력 대기
담당: **C** = Claude(코드/편집/분석) · **U** = 저자(값·판단 제공) · **L** = 실제 실험 세션

---

## 0. 현재 상태 (기준점, 2026-08-31)

- **리비전은 이미 초안 상태.** `access_revised.tex`(수정 원고, `\rev{}` 하이라이트 ~89개,
  빨간 `\nd{}` placeholder **~210개**), `revision/response_to_reviewers.tex`(point-by-point
  응답, **~95개**). 문구는 대부분 완성되어 있고, **빨간 부분 = 아직 실제로 만들어내야 하는
  숫자/값.**
- **실험 코드는 프로젝트에 모두 존재.**
  - PDDLStream baseline: `experiments/pddlstream/` (FastDownward 포함). 우리 도메인 전용
    FR5 구현 `examples/pybullet/fr5_obstacle/` — `run_{transfer,move,stir}_trials.py`
    (`--seed`, trial 루프, per-trial CSV 로깅), `domain_stir.pddl`, `stream_stir.pddl`,
    `fr5_primitives.py`, `fr5_utils.py`.
  - LLM 인프라: `LLM/llama/script/experiments/` — `XDL_generator/`(test set, validator),
    `ActionReasoner/`(4-token 데이터셋 3000 = train 2400 / test 600, `split_dataset.py`,
    학습·평가 스크립트, checkpoint).
  - cuTAMP: `TAMP/cuTAMP/cutamp/` (`config.py`: 1024 particles·1000 steps·lr 7e-3,
    `scripts/utils.py`: cost weights). 실행 파이프라인 `TAMP/tamp/scripts/server/tamp_server.py`,
    `scripts/xdl/tamp_xdl_parser.py`, envs `TAMP/tamp/src/envs/{transfer,move,stir,rearrange}.py`.
  - perception(`perception/apriltag_ros`, `perception_manager`), Isaac Sim(`isaacsim/scripts/standalone/`).
- **따라서 남은 본질은 "빨간 숫자 채우기"가 아니라, 이 코드를 실제로 돌려서 draft가 주장하는
  값을 정직하게 만들어내는 것.** 그리고 그 과정에서 드러난 코드 이슈를 고치는 것.
- **아직 실재하는 코드 이슈 (이번 작업의 핵심):**
  1. **성능이 물체 위치에 민감(천차만별)** → success rate가 randomized 배치 전반에서 재현
     가능해야 함 (D3).
  2. Isaac-Sim 파이프라인에 **randomization·seed·logging harness가 없음** (`task.py`의 위치는
     하드코딩, `tamp_server.py`의 `_log_planning_result()` 호출은 주석 처리).
  3. pouring 컨트롤러(`tamp_server.py pouring()`)에 **θ_max·tilt-rate 제한이 없음**.
  4. baseline은 **PyBullet**, cuTAMP는 **Isaac Sim** → 같은 조건 비교인지 검증 필요.
- **저자와 합의한 결정 (D1–D5, `CLAUDE.md`와 동일):**
  - **D1** 실제 하드웨어 = **adaptive-pouring 데모만** (RS-232 저울). AprilTag perception
    스냅샷은 같은 rig에서 함께 취득. 그 외 하드웨어 실험 없음.
  - **D2** AprilTag perception을 **Isaac Sim 안에서 in-the-loop**로 (렌더 카메라 → 검출 →
    fusion → World State). ground-truth 추출 방식 대체. → R1#1의 실제 해법.
  - **D3** 코드를 개선해 success 숫자가 **randomized 위치 전반에서 재현 가능**하도록. 숫자는
    항상 개선된 코드의 실제 측정을 따름 (좋게 나온 배치만 쓰지 않음).
  - **D4** baseline과 LLM 인프라는 **이미 프로젝트에 있음** → 재구현 X, **통합·검증**만.
  - **D5** draft가 **과약속**한 부분(전체 하드웨어 perception + 실제 20회 end-to-end Transfer,
    Section IV-F의 A1/A2)은 D1/D2에 맞게 **재작성**. §5 참고.

> 응답서가 참조하는 `HARDWARE.md`는 실제로 없음. 그 A/B/C 항목 카탈로그는 여기 §3에 있음.
> 응답서의 Status 노트("Needs A1/B2/…")는 §3을 가리키는 것으로 해석.

---

## 1. 환경 셋업 (모든 sim 실행의 전제)

> 코드/워크스페이스 정리 전체 계획은 **`REFACTOR.md`** 참조. 아래는 그 요약.

- [ ] **1.1 Isaac Sim = pip 고정설치 6.0.1 + standalone** (결정됨). Python **3.12**,
      `isaacsim[all,extscache]==6.0.0.1`, torch 2.10.0(cu128). 정확한 recipe는 `REFACTOR.md` §I-1.
      소스빌드/별도 ROS ws/벤더 런처/LD_PRELOAD 폐기. **버전 6.0.1로 통일**(draft 4.2.0·README 5.1.0
      대체, R1#8). ⚠️ torch 2.10 ↔ cuRobo(2.7) 호환 검증 필요. GPU: RTX 5070(Blackwell).
- [ ] **1.2 headless 배치 실행 확인.** seeded 배치를 GUI 없이 돌릴 수 있어야 함
      (`isaacsim/scripts/run_isaacsim.py`에 `headless` 인자 있음; standalone은 `headless:False`로
      하드코딩되어 있음).
- [ ] **1.3 conda env 구성** (`environment.yml`) — cuTAMP/cuRobo + LLM 스택이 현재 GPU/CUDA에서
      import 되는지 확인.

---

## 2. 리뷰어 concern → 작업 맵 (마스터 표)

각 concern에 대해 draft가 약속한 것과 실제로 필요한 것. `()`의 숫자는 draft의 목표값이며
**타이핑이 아니라 측정으로 얻어야 함.**

| # | Concern (요약) | 필요한 것 | 담당 | 상태 |
|---|---|---|---|---|
| **R1#1** | perception이 end-to-end로 안 쓰임 | D2: Isaac Sim 안 AprilTag in-loop(위치오차·occlusion·planning 영향) + 실제 스냅샷(§3-A1'); sim 결과를 ground-truth-state로 명시; bootstrap noise 주입 재실행(§3-B1) | C (+L) | [ ] |
| **R1#2** | 실행 계층 과장 | "추가 실패 없음" 삭제(draft 반영됨 — 검증); 실행 성공기준 4가지 정의; 실제 pouring 정확도(§6) | C (+L) | [~] |
| **R1#3** | planner 비교 재현 불가 | baseline 존재(§3) — **PyBullet vs Isaac Sim 동일 조건 검증**; Appendix A 양쪽 planner; system-level wall-clock 프레이밍 | C | [~] |
| **R1#4** | 통계: 성공 trial만의 latency | seeded 벤치 재실행으로 per-trial 로그 생성(§4.1); Wilson CI + exact McNemar(paired) + KM/restricted-mean time-to-solution; "variance 감소" 문구 삭제 | C | [ ] |
| **R1#5** | generator/validator 혼동, test 독립성 | 인프라 존재(§3) — 100-set에 field-level 채점; split 문서화 | C | [~] |
| **R1#6** | validator "완전 coverage" 과장 | 주장 축소(텍스트 — 지금 가능); 선택: capacity·device-placement 체크 + 벤치 확장(§3-B5) | C (+U) | [ ] |
| **R1#7** | Action Reasoner 일반화 과장 | split 존재(3000/2400/600, §3) — unseen 레벨 정의; 주장을 workspace로 제한; 3-step ablation(§3-B7) | C | [~] |
| **R1#8** | 방법론 재현성 | Section IV-A4 + Appendix A에 실제 설정값(§3-C); randomization+seed harness 구축(§3-B, 현재 없음) | C | [ ] |
| **R2#2** | 실제 로봇 실험 | 실제 pouring 세션(§6); Section IV-F를 D1/D2로 재작성(§5) | L + C | [ ] |
| **R2#3** | Author Response Table 누락 | 이 응답 문서 자체(존재 — 마무리) | C | [~] |
| **R2#4** | LLM hallucination 방어 | Section III-B4 containment(초안됨 — 검증); 선택: adversarial 테스트(§3-B8) | C | [~] |
| **R3#1** | 유체역학 / sim-to-real | property별 Section V-A(텍스트) + 실제 pouring 숫자(§6) | C + L | [ ] |
| **R3#2** | θ_max=5° 근거 | pouring 코드에 θ_max·tilt-rate 구현(§3-B/pouring); spillage 근거(sim sweep + 실제 확인)로 채움 | C (+L) | [ ] |
| **R3#3** | perception 견고성 | 3-단계 fallback 텍스트(초안됨 — 검증); 5초 대기 + fail-closed를 서술대로 구현; occlusion 숫자는 D2에서 | C | [ ] |
| **R3#4a** | 표 단위 표기 | 편집 감사(초안됨 — 검증) | C | [~] |
| **R3#4b** | Fig 1/3 표기 일관성 | 프로세스 블록 라벨 통일해 재작도 | C | [ ] |
| **R3#5** | 추천 참고문헌 [21],[32] | 추가됨(배치 검증) | C | [x] |

---

## 3. 실험/값 카탈로그 (A=하드웨어, B=sim/코드, C=설정값)

### A — 실제 하드웨어 (1회 세션, 담당 L; Claude가 수집·표 준비)
- [ ] **A3 (핵심) — Adaptive pouring 정확도.** 대상 vessel 아래 RS-232 저울, FR5로 실제
      closed-loop pour. 목표량 5/10/20 mL × 10회, 물 + 점성 액체. mean-abs 이송오차, overshoot,
      settling time 보고; scale latency 측정. → §6.
- [ ] **A1' — 실제 perception 스냅샷** (같은 rig): dual-camera AprilTag 위치오차 몇 개 pose +
      occlusion 1개 조건. 작은 표본, 정직하게.
- [x] ~~A2 실제 하드웨어 20회 end-to-end~~ → **폐기**(D1); Isaac Sim으로 이동(B2e).
- [x] ~~A4 전체 tilt×fill spillage sweep(하드웨어)~~ → **축소**: sim sweep + 실제 spot-check.
- [x] ~~A5 실제 fallback 시연~~ → **폐기**; fallback은 sim 검증 + 서술.

### B — 시뮬레이션 / 코드 (담당 C)
- [ ] **B/harness (가장 먼저) — randomization + seed + logging.**
      `task.py`의 물체 위치는 하드코딩, CSV logger는 주석 처리 상태. 구현: per-trial 물체 위치
      randomization(±5 cm / ±180° yaw) + robot init(±10°), 고정 **seed 정책(0–29)**,
      `_log_planning_result()` 재활성화. → R1#4·R1#8·D3 및 모든 "randomized trial" 숫자의 토대.
- [ ] **B/robustness (D3) — cuTAMP를 위치 전반에서 성공하게.** 위치 민감성 원인 조사
      (`cutamp/scripts/utils.py` cost weights, particle init, IK seeding `TODO: seeding`), seed
      집합 전반에서 재현되도록 개선. 재측정한 새 숫자가 기존 headline 값을 대체.
- [ ] **B1 — perception noise 주입.** 측정된 perception 오차에서 6-DoF residual을 bootstrap
      해 sim World State에 주입, seeded planning eval 재실행. (R1#1)
- [ ] **B2 — time-to-solution 분석** (생성된 로그로): KM solved-fraction 곡선, budget 내
      restricted mean, cuTAMP time band. (R1#4)
- [ ] **B2e — 렌더 perception 기반 end-to-end (D2):** 자연어 명령 → World State를 Isaac Sim
      AprilTag 파이프라인으로 grounding → planning + execution 성공률 보고. (R1#1, R2#2)
- [ ] **B3/B4 — generator field-level 정확도 + split** — `LLM/llama/script/experiments/`의 100
      test set으로 field별 채점(§4.3), split 문서화. (R1#5)
- [ ] **B5 — validator 확장 (선택):** `validator.py`에 capacity·device-placement 체크 추가,
      벤치마크 6-class로 확장. 생략 시 해당 빨간 span 2개 삭제하고 축소된 주장 유지. (R1#6)
      - [ ] 먼저 **injected-error 벤치(20 valid/80 invalid)가 코드로 생성되는지 확인**
            (`exper.py`/`test_data_gen.py`), 없으면 구축.
- [ ] **B6/B7 — Action Reasoner unseen 레벨 정의 + 3-step ablation** — split(3000/2400/600)에서
      unseen 기준 도출; two-step 루프(`tamp_xdl_parser.py run_xdl`)를 three-step으로 확장. (R1#7)
- [ ] **B8 — adversarial 명령 스트레스 테스트 (선택):** 70개 명령 / 7 class를 validator에 투입.
      생략 시 빨간 span 삭제; 5-메커니즘 텍스트는 유지. (R2#4)
- [ ] **B9 — 256-particle cuTAMP ablation (선택):** "추가 연산" 논거 보강. (R1#3)
- [ ] **B/pouring — θ_max=5° + tilt-rate 제한 구현** (`tamp_server.py pouring()`에 현재 없음)
      → 실제 데모와 주장이 일치하도록; 5° 근거용 sim spillage sweep. (R3#2)

### C — 기록할 설정값 (고친 코드에서 → Section IV-A4 + Appendix A)
- [ ] Isaac Sim 버전(§1.1로 확정); PhysX timestep/substeps(현재 미설정 — 설정).
- [ ] workspace/table 치수 + 12-grid geometry(`envs/utils.py`, `tamp_xdl_parser.py`),
      randomization 범위(새 harness에서), collision margin, cost weights(`cutamp/scripts/utils.py`),
      particle 수/steps/LR(`config.py`), controller rate(sim 60 Hz; 실제 rig rate 별도 기록),
      joint vel/accel limit(`fr5.yml`), 카메라 사양(`camera.py` — **주의: 코드 L515 640×480 vs
      draft D435 1280×720, 실제 쓰는 것으로 통일**), 저울 사양, seed 정책.
- [ ] **draft의 모든 설정값을 코드와 대조** — 여러 빨간 span(D435, 4.2.0, ±5 cm, 10 mm margin,
      125 Hz, 60 Hz control)이 코드와 안 맞음. 실제 값으로 통일.

---

## 4. 오프라인 분석 스크립트 (Claude, 시뮬레이터 불필요)

`_2026__IEEE_Access/revision/analysis/`(신규)에 둘 것.
- [ ] **4.1 seeded 벤치 로그 생성/수집:** cuTAMP(Isaac Sim)와 PDDLStream(PyBullet
      `run_*_trials.py --seed`)을 **동일 30개 seed**로 돌려 per-trial CSV(`success`,
      `planning_time_sec`, attempts) 산출. (dep: §1 env + §3-B harness)
- [ ] **4.2 통계 스크립트:** 각 성공률의 Wilson 95% CI; paired seed에 대한 exact McNemar;
      budget에서 우측 censoring한 KM solved-fraction + restricted-mean time-to-solution.
      표에 들어갈 정확한 값 출력.
- [ ] **4.3 XDL field 채점 스크립트:** 생성 protocol vs reference를 canonicalize 후 field별
      비교(operator seq / vessel id / 수치+단위 / 순서 / 전체 프로그램). (R1#5)
- [ ] **4.4 validator 벤치 러너:** injected-error 집합에 대한 confusion matrix + class별 recall.

---

## 5. 원고 재작성 (D1/D2에 맞춰 축소 — Claude)

draft가 실제로 할 것보다 많은 하드웨어를 약속함. 논문이 우리가 하는 것만 주장하도록 수정.
- [ ] **5.1 Section IV-F:** "실제 pouring + perception 스냅샷" + "Isaac Sim 내 perception-in-loop
      및 end-to-end"로 재구성. 실제 20회 Transfer, 전체 실제 spillage sweep 제거; pouring(실제)
      + perception(sim + 소량 실제) + sim end-to-end 유지.
- [ ] **5.2 응답 R1#1 & R2#2:** perception-in-Isaac-Sim + 실제 pouring을 주장하고, end-to-end는
      렌더/외부공급 state로 scope(R1#1이 명시적으로 허용한 선택지)하도록 재작성.
- [ ] **5.3 Abstract / Intro / Conclusion:** 하드웨어 문장을 축소된 scope에 맞춤.
- [ ] **5.4 이미 초안된 모든 `\rev{}` 변경**(주장 축소, 단위, containment §III-B4, fallback
      §V-B, 참고문헌)이 최종 scope와 맞는지 검증.

---

## 6. 실제 pouring 세션 프로토콜 (담당 L; Claude 준비·기입)

- [ ] **6.1 RS-232 저울 확보;** 스트리밍 rate + 해상도 확인 (실제 모델명 기록 — `A&D EK-610i`
      빨간 placeholder는 실제 그 기기일 때만 그것으로 대체).
- [ ] **6.2 closed-loop pour가 실제 저울 스트림을 읽는지 구현/검증;** 측정된 latency에 맞춰 PD
      gain(`tamp_server.py`의 `Kp/Kd`) 튜닝; θ_max + tilt-rate 적용(§3-B/pouring).
- [ ] **6.3 실행 + 촬영:** 물 + 점성 액체 × {5,10,20} mL × 10회. 목표 vs 실제 질량, overshoot,
      settling time 기록; end-to-end scale latency 측정.
- [ ] **6.4 perception 스냅샷** (같은 rig, A1'): 몇 개 pose 위치오차 + occlusion 1개.
- [ ] **6.5 표에 반영;** 두 파일의 pouring/perception 빨간 span 채움.
- [ ] **6.6 실제 실험 보충 영상.**

---

## 7. 마무리 + 완료 정의

- [ ] **7.1 모든 `\nd{}`를** 실제 측정/실행/기록으로 채움 (두 파일).
- [ ] **7.2 숫자 일관성 스윕:** 원고 + 응답 + 표에 나오는 각 값이 어디서나 동일. 바뀐 값마다
      두 파일 전체를 grep.
- [ ] **7.3 철회한 주장 재검:** 4개 과장 주장 중 어느 것도 재도입 안 됨 (`CLAUDE.md`).
- [ ] **7.4 게이트:** 두 파일에서 `grep -n '\nd{'` 비어 있음.
- [ ] **7.5 제출물 3종 빌드:** 응답 PDF(`\showstatusfalse`), Highlighted PDF(`\highlighttrue`),
      clean Main Manuscript PDF(`\highlightfalse`). 모두 컴파일 확인.
- [ ] **7.6 교차 확인:** 모든 리뷰어 concern이 응답 + 대응되는 하이라이트 원고 변경을 가짐.

---

## 8. 열린 리스크 / 감시 목록

- **표 번호 재배치:** III-B4에 요약 표가 추가되어 원래 Table 1–8 → 2–9. 모든 `\ref` 일관성
  유지; 응답서는 이미 새 번호를 전제.
- **baseline 공정성 (R1#3의 급소):** baseline은 PyBullet, cuTAMP는 Isaac Sim. **동일 30개
  seed 초기배치 + 동일 collision geometry + 동일 upright 제약**을 쓰는지 반드시 검증. 안 맞으면
  비교가 confounded → 조건을 맞추거나 차이를 명시. 맞출 수 없으면 in-repo rejection-sampling
  ablation과의 비교로 재프레이밍(그리고 R1#3/#4 재작성). 재현 불가능한 비교는 제출 금지.
- **perception-in-sim vs 리뷰어 기대:** R1#1이 "supplied state로 제한" 선택지를 명시적으로
  줬으므로 perception-in-Isaac-Sim + 소량 실제 표본은 방어 가능 — 단 R2가 실제 실험을
  "essential"이라 했으므로 실제 pouring은 확실히.
- **카메라/하드웨어 사양 불일치** (코드 L515/640×480 vs draft D435/1280×720): placeholder가
  아니라 실제 셋업을 보고.
