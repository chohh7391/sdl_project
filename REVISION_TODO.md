# REVISION_TODO.md — 남은 리비전 작업 목록

Access-2026-33062 재제출까지 남은 일. **마스터 카탈로그는 `PLAN.md`**(리뷰어 concern → 작업 맵,
실험 A/B/C 카탈로그)이고, 이 파일은 그 위에 **"지금 무엇이 실측으로 뒷받침되고 무엇이 아직 아닌가"**를
얹은 현황 문서다. 두 파일이 어긋나면 `PLAN.md`가 우선한다.

기준 시점: 2026-09-11. 근거 수치의 출처는 `_2026__IEEE_Access/revision/analysis/data/*.csv`와
각 배치의 `scripts/trials/logs/b*/RUN_INFO.txt`. 상세 진단은 `REFACTOR.md`, 경과는 `SESSION_LOG.md`.

---

## 0. 한 줄 요약

시뮬 TAMP 성공률(실험 계열 1)은 **실측으로 뒷받침되는 상태가 됐다.**
나머지 4개 계열(PDDLStream 비교 / perception / LLM 인프라 / 실물 하드웨어)은 **미착수**이고,
같은 계열 안에서도 **latency·예산·Move 배치 허용오차는 논문 서술과 어긋난다는 것이 측정으로 드러났다.**

`\nd{}` 잔여: 원고 **213**, 응답 **97**. `PLAN.md` 체크박스 **3/42**.

---

## 1. 무엇이 진실이 되었나 (계열 1)

논문은 세 task의 cuTAMP 100 %를 `\nd{}` 빨간 span이 아니라 **확정된 사실로** 적고 있었으나, 코드는
그것을 재현하지 못했다. 더 나아가 **서술된 실험 자체가 시뮬레이터 안에서 일어날 수 없는 경우가 셋**이었다.

| | 서술된 실험 | 시작 시점의 실제 |
|---|---|---|
| Transfer | 2-finger 측면 파지로 기울여 붓기 | top 파지 구현 → pour 관절이 비커 자기 축을 돌려 **어떤 각도에서도 기울기 0.00°**(가짜 저울이 은폐) |
| Stir | stir bar를 좁은 입구에 정밀 삽입 | vessel이 **속 찬 상자** → bar는 항상 뚜껑 위에 정지. 45 mm 각봉은 64 mm 개구부도 통과 불가 |
| Move | 용기를 목표 구역으로 이송 | 목표 트레이가 상자 **시작 칸 아래** → t=0에 이미 성공, 게다가 상자가 트레이를 10 mm 관통한 채 스폰 |

**현재(동일 코드 버전, 동일 30 seed):**

| task | 배치 | 과제 성공 | planning 중앙 (≤60 s) | 운반 tilt 최대 (≤5°) | 최종 tilt | 목표 오차 중앙/최대 |
|---|---|---|---|---|---|---|
| transfer | B7 | **30/30** [Wilson 88.6–100] | 55.3 s (22/30) | 11.38° (27/30) | 전부 0.00° | 13.8 / 23.6 mm |
| move | B8 | **30/30** | 23.6 s (30/30) | 2.35° (30/30) | 전부 0.00° | 13.1 / 16.9 mm |
| stir | B6 | **30/30** | 33.5 s (30/30) | 4.40° (30/30) | 최대 0.02° | 5.9 / 13.5 mm |

stir는 **stir bar가 vessel 안에 30/30** 들어갔다(축에서 중앙 6.5 mm, 입구보다 100–112 mm 아래 = 내부 바닥).
세 배치 모두 stray 프로세스 0, 포즈 정합성 불일치 0.

과제 성공 판정은 "궤적 완주"가 아니라 **task별 종료조건**(붓기 실행 · 똑바로 놓임 · 목표 영역 · bar 삽입)이다 — R1#2가 요구한 것.

---

## 2. 무엇이 아직 논문과 어긋나는가 (계열 1 안에서)

| 항목 | 논문 | 실측 | 필요한 조치 |
|---|---|---|---|
| latency (Transfer) | 31.18 ± 1.44 s | **69.90 ± 35.92 s** (최대 167.58) | 실측으로 교체 |
| latency (Move) | 17.60 ± 3.55 s | 24.39 ± 4.73 s (최대 46.39) | 실측으로 교체 |
| latency (Stir) | 28.83 ± 1.00 s | 34.48 ± 4.56 s (최대 53.75) | 실측으로 교체 |
| 60 s 예산 timeout | 0.0 % | **Transfer 8/30 초과** — 그 예산이면 성공 22/30 = 73.3 % | 예산 재서술 또는 censoring 반영 |
| Move 종료조건 | `\nd{10 mm}` | **4/30** (15 mm 26/30, 20 mm 30/30) | 임계값 결정 or 중심 유인 비용 추가 |
| 파지 슬립 | `\nd{5 mm}` | **측정 항목 자체가 없음** (tilt만 계측) | 계측 추가 or 기준 삭제 |

Move 오차는 추종오차가 아니다 — 배치 제약이 "발자국이 영역 안"까지만 요구하고 중심으로 끌지 않는다.

---

## 3. 남은 작업

### A. 데이터가 이미 있어 지금 쓸 수 있는 것

- [ ] **A1. Table 4 (`tab:tamp_comparison`) cuTAMP 행** — 성공률 유지, **latency 실측 교체**, timeout/early-term 열 재작성 · R1#4
- [ ] **A2. 60 s 예산 서술** — 예산을 실측에 맞추거나 censoring 반영 time-to-solution으로 전환 · R1#4
- [ ] **A3. Appendix A + Experiment Setup 설정값** — Isaac Sim 6.0.1, PhysX 설정, 랜덤화 범위(±5 cm / ±180° yaw / ±10° 관절), seed 정책, 입자 1024 × 1000 step, 실행 주기 0.08 s, 허용오차, upright 한계 · R1#8 · `\nd{}` 38 + 36개
- [ ] **A4. 실행 성공기준(IV)** — task별 종료조건을 실제 정의·측정한 것으로 교체 · R1#2
      ⚠ **결정 필요**: Move `10 mm`(4/30), 파지 슬립 `5 mm`(미측정)
- [ ] **A5. θ_max = 5° 근거** — spill 기하 유도 + 실행 tilt 실측 · R3#2

### B. 실험이 남은 것

- [ ] **B1. PDDLStream 베이스라인** — 표의 66.7 / 83.3 / 60 %와 McNemar p값이 **전부 미검증 하드코딩**.
      동일 30 seed 실행 + 설정 기록 + paired exact McNemar + KM/RMTS · R1#3, R1#4 · **가장 취약**
- [ ] **B2. AprilTag perception in-the-loop** (렌더 카메라 → 검출 → 융합 → World State) + perception noise 주입
      재실행(표 마지막 열 `\nd{}`) · R1#1 · **R1 최우선 concern**
- [ ] **B3. XDL generator field-level 채점** (100-set: operator / 인자 / 객체 / 수치 / 순서) + split 문서화 · R1#5
- [ ] **B4. Action Reasoner unseen 레벨 정의** + 3-step ablation · R1#7
- [ ] **B5. validator** — 주장 축소(텍스트) 또는 capacity·device-placement 체크 확장 · R1#6
- [ ] **B6. 실물 pouring 세션** + perception 스냅샷 · R2#2, R3#1 · **담당: 저자**

### C. 텍스트만으로 되는 것

- [ ] **C1. Section IV-F 재작성** ⚠ 드래프트가 **하지 않을 실험을 여전히 약속** 중
      (`access_revised.tex` line 752 "In 8 of the 20 trials…" 등 20회 하드웨어 trial).
      CLAUDE.md 전략 4번이 요구하는 축소가 미반영 · **위험도 높음**
- [ ] **C2. perception fallback** — 서술대로 5초 대기 + fail-closed 구현 · R3#3
- [ ] **C3. 단위 표기 감사** · R3#4a / **Fig 1·3 라벨 통일 재작도** · R3#4b
- [ ] **C4. Author Response Table** · R2#3 / **containment §III-B4 검증** · R2#4
- [x] **C5. 철회 주장 4개** — "complete and non-overlapping", "decisive factor", "reduced planning-time variance"
      제거 확인. "no additional failure modes"는 scoped 버전(`\rev{}`)으로만 남아 있어 허용

### D. 마무리 게이트

- [ ] **D1.** `\nd{}` 원고 213 / 응답 97 → **0**
- [ ] **D2.** 숫자 일관성 스윕 (원고 ↔ 응답 ↔ 표)
- [ ] **D3.** PDF 3종 빌드 — 응답(`\showstatusfalse`) / Highlighted(`\highlighttrue`) / clean(`\highlightfalse`)
- [ ] **D4.** concern ↔ 응답 블록 ↔ 하이라이트 원고 변경 교차확인

---

## 4. 권고 순서

```
A → C1 → B1 → B2 → B3·B4 → B6
```

- **A 먼저**: 값이 이미 손에 있어 `\nd{}`를 가장 크게 줄인다. A4의 두 결정이 다른 서술에 물려 있다.
- **그 다음 C1**: 실행하지 않을 실험을 약속한 상태를 방치하는 것이 가장 위험하다.
- **그 다음 B1**: 표의 핵심 비교가 통째로 미검증이라 재현 요구에 가장 취약하다.
- **그 다음 B2**: R1 최우선 concern.

---

## 5. 열린 결정 (저자 확인 필요)

1. **Move 종료조건 임계값** — `10 mm`를 실측(4/30)에 맞춰 완화할지, 배치에 중심 유인 비용을 넣어 실제로 달성할지
2. **파지 슬립 `5 mm`** — 계측을 추가해 측정할지, 기준에서 뺄지
3. **60 s 예산** — 예산 값을 바꿀지, censoring 반영 분석으로 서술을 바꿀지
4. **시뮬 asset** — 구매 실물(beaker 72 mm/외경 60 mm, flask 160 mm/입구 외경 38 mm)과 모델
   (beaker 50×50×135 mm, flask 70×70×120 mm·개구부 64 mm)이 다르다. **저자 지시로 asset은 그대로 둠** →
   한계로 서술할지 결정 필요. stir bar 10×35 mm는 실물 입구도 통과한다.
5. **git push** — 로컬 커밋 3개 완료, 푸시 미실행
