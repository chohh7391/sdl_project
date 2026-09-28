# RERUN.md — 논문 수치 재실험 런북

이 문서 하나로 새 머신(RTX 5080)에서 **논문의 시뮬레이션·오프라인 수치를 전부 다시 측정하고, 감사하고, 원고에 반영할 수 있게** 정리했습니다. 위에서부터 순서대로 따라가면 됩니다.

Claude 세션에 맡길 때는 저장소 루트에서 세션을 열고 "RERUN.md대로 진행해"라고 하면 됩니다. [CLAUDE.md](CLAUDE.md)의 규칙은 여기서도 그대로 적용됩니다.

---

## 0. 절대 규칙

1. **논문 수치는 기존 시뮬레이션 scene에서만 측정합니다.** 실기 scene(`transfer_real`: 저울, 받침대, 낮춘 벤치, 수평 파지, 실측 배치)은 실기 재생용 궤적을 만드는 용도이고 논문 수치에는 쓰지 않습니다(§10). 이 규칙은 스크립트가 강제합니다.
   - `SDL_*` 환경변수가 **하나라도** 설정돼 있으면 모든 재실험 스크립트가 실행을 거부합니다. 실기 scene의 스위치는 전부 `SDL_*`입니다.
   - 사전 점검이 코드의 scene 기본값(유리기구, 벤치 높이, 받침대, 파지 경사 밴드, 환경 정의)을 논문 값과 대조합니다.
   - 배치가 끝날 때마다 시뮬레이터 로그에서 실기 scene 표지 문구를 찾습니다. 발견되면 그 배치는 오염으로 처리됩니다.
2. **로봇 모델도 바꾸지 않습니다.** `hardware/fr5_d435_mount/README.md`에 적힌 TCP +4 mm는 실기용이라 논문 시뮬레이션에는 반영하지 않습니다.
3. **GPU를 공유하지 않습니다.** 시작 시점에 다른 연산 프로세스가 있으면 거부하고, 실행 중에는 30초마다 기록합니다. 다른 프로세스와 겹친 **시행**은 감사에서 걸러집니다(§7).
4. **보고할 예산은 실행 전에 선언합니다.** 첫 사전 점검에서 고정되고, 그 뒤로는 바꿀 수 없습니다. 결과를 보고 cuTAMP에 유리한 예산을 고르는 일을 막기 위해서입니다.
5. **숫자는 `RESULTS.md`와 `results/`에서만 가져옵니다.** 감사가 NOT CLEAN인 시행의 수치는 결과가 아닙니다. 측정 안 된 값은 `\nd{}`로 남깁니다.
6. `access.tex`는 절대 수정하지 않습니다. 반영은 `access_revised.tex`와 `revision/response_to_reviewers.tex`에 합니다(§9).
7. 물리 실기 실험(PLAN.md의 A 항목: 실제 붓기, 실제 인식 스냅샷)은 이 런북의 범위가 아닙니다.

---

## 1. 한눈에 보기

| 단계 | 스크립트 | 내용 | 소요 (RTX 5070 기준) | 원고·응답 위치 |
|---|---|---|---|---|
| 0 | `scripts/rerun/00_preflight.sh` | 환경, scene, GPU 점검 → `RUN_INFO.md` | 5분 (+스모크 5분) | §IV-A4, 부록 A |
| 1 | `scripts/rerun/10_cutamp.sh` | cuTAMP: Transfer·Move·Stir 정답 상태 × 3회, Transfer 인식 상태 × 3회 (각 30개 레이아웃) | 약 15시간 | Table 5, 9, §IV-C, §IV-E, R1#1·#3·#4 |
| 2 | `scripts/rerun/20_pddlstream.sh` | PDDLStream: 3개 작업 × 플래너 시드 5개 × 30개 레이아웃, 최대 180 s | 약 5–6시간 | Table 5, §IV-C, R1#3·#4 |
| 3 | `scripts/rerun/30_perception.sh` (선택) | 렌더 카메라 인식 정확도, 30개 레이아웃 | 약 2시간 | R1#1 |
| 4 | `scripts/rerun/40_llm.sh` | XDL 생성기, 검증기, Action Reasoner (정확도와 추론 시간) | 약 30분 | Table 2–4, 6–7, §IV-B, §IV-D1, R1#5·#6·#7 |
| 5 | `scripts/rerun/90_analyse.sh` | 감사 → 분석 → `RESULTS.md` | 1분 | — |

`scripts/rerun/run_all.sh`가 0→1→2→(3)→4→5를 순서대로 실행하고, 중단된 지점부터 이어 갑니다. 전체는 **하루 정도** 걸립니다.

---

## 2. 클론

colcon 오버레이를 `<ws>/install`에서 찾기 때문에 **`<ws>/src/sdl_project` 위치에** 클론해야 합니다.

```bash
mkdir -p ~/sdl_ws/src
```

```bash
cd ~/sdl_ws/src && git clone -b revision-access git@github.com:chohh7391/sdl_project.git
```

이하 명령은 `~/sdl_ws/src/sdl_project`에서 실행한다고 가정합니다.

---

## 3. 저장소 밖 준비물

클론만으로는 따라오지 않는 것들입니다. 환경 3개는 이 머신(5080)에서 **새로 만들어야** 하고, 자산과 가중치는 원래 머신에서 **옮겨 옵니다.** `envs/`에 원래 머신의 정확한 패키지 목록이 잠금 파일로 들어 있으니, 다 만든 뒤 `pip freeze`와 비교해 보세요.

### 3.1 시스템

Ubuntu 22.04, NVIDIA 드라이버(원래 머신: 580.178.04), CUDA 12.8 툴킷(cuRobo 빌드용), gcc-11, ROS 2 Humble, conda, uv, git-lfs가 필요합니다.

```bash
sudo apt install build-essential gcc-11 g++-11 git-lfs python3-rosdep ros-humble-desktop
```

### 3.2 Isaac Sim 6.0.0.1 (`.venv-sdl`, Python 3.12)

```bash
uv venv --python 3.12 --seed .venv-sdl
```

```bash
uv pip install --python .venv-sdl/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

```bash
uv pip install --python .venv-sdl/bin/python "isaacsim[all,extscache]==6.0.0.1" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
```

처음에 한 번 실행해서 EULA에 동의합니다(`.venv-sdl/bin/isaacsim`). 첫 구동은 셰이더 컴파일 때문에 4분쯤 걸립니다. 기준 목록은 `envs/venv-sdl.freeze.txt`입니다.

그다음 Isaac Sim 내장 ROS용(py3.12) 인터페이스를 빌드합니다.

```bash
bash ros2_isaacsim_ws/build_interfaces.sh
```

### 3.3 conda `sdl` (Python 3.10: cuTAMP, cuRobo, LLM)

```bash
conda env create -f environment.yml
```

```bash
conda activate sdl && pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

```bash
conda activate sdl && pip install -e TAMP/cuTAMP
```

cuRobo 빌드는 20분까지 걸리고 CUDA 12.8이 필요합니다.

```bash
conda activate sdl && cd TAMP/cuTAMP/curobo && pip install -e . --no-build-isolation
```

기준 목록은 `envs/conda-sdl.pip-freeze.txt`입니다. unsloth가 들어 있어야 LLM 단계가 돕니다.

### 3.4 colcon 워크스페이스 (시스템 ROS 2 Humble)

rosdep을 처음 쓰는 머신이면 `sudo rosdep init && rosdep update`를 먼저 합니다.

```bash
cd ~/sdl_ws && source /opt/ros/humble/setup.bash && rosdep install --from-paths src --ignore-src -y
```

```bash
cd ~/sdl_ws && source /opt/ros/humble/setup.bash && colcon build
```

`tamp_interfaces`, `perception_manager`, `apriltag_ros`가 빌드돼야 합니다. conda 환경을 비활성화한 상태에서 빌드하세요.

### 3.5 PDDLStream 기준선 (`.venv-pddl`, Python 3.10)

```bash
python3.10 -m venv .venv-pddl && .venv-pddl/bin/pip install numpy==2.2.6 pybullet==3.2.7
```

기준선 러너(`experiments/pddlstream/examples/pybullet/fr5_paired/`)와 PyBullet용 FR5 모델은 이제 git에 들어 있습니다. Fast Downward는 들어 있지 않으니 §3.6에서 소스를 옮겨 온 뒤 빌드합니다.

```bash
cd experiments/pddlstream && ./downward/build.py
```

### 3.6 원래 머신에서 옮겨 올 것

`SRC`는 원래 머신의 저장소 경로입니다. 예: `home@<원래 머신 주소>:/home/home/sdl_ws/src/sdl_project`

```bash
SRC=home@<원래 머신 주소>:/home/home/sdl_ws/src/sdl_project
```

| 대상 | 크기 | 이유 |
|---|---|---|
| `third_party/LabUtopia/` | 111 MB | scene USD가 참조하는 자산. 업스트림 `github.com/Rui-li023/LabUtopia`에서 받을 수도 있지만 사용한 커밋이 기록돼 있지 않음 |
| `LLM/llama/model/checkpoint/xdl_generator/checkpoint/` | 약 61 MB | 논문 XDL 생성기 수치를 낸 가중치. git에 있는 `xdl_llm/`과는 **다른 모델**임 (r=16 대 r=8) |
| `experiments/pddlstream/downward/` (소스만) | 12 MB | Fast Downward. 업스트림은 `caelan/downward`의 서브모듈인데 커밋이 기록돼 있지 않음 |

```bash
rsync -a "$SRC/third_party/LabUtopia/" third_party/LabUtopia/
```

```bash
rsync -a "$SRC/LLM/llama/model/checkpoint/xdl_generator/checkpoint/" LLM/llama/model/checkpoint/xdl_generator/checkpoint/
```

```bash
rsync -a --exclude builds "$SRC/experiments/pddlstream/downward/" experiments/pddlstream/downward/
```

XDL 생성기 가중치는 사전 점검이 sha256으로 확인합니다(`2cf8d597f2cc…`). 해시가 다르면 논문이 평가한 모델이 아닙니다. LLM 베이스 모델(`unsloth/llama-3.2-1b-bnb-4bit`)은 첫 실행 때 HuggingFace에서 자동으로 받습니다. 인터넷 연결이 필요합니다.

---

## 4. 실행 전 결정

기본값으로 두면 원래 캠페인과 같은 설계가 됩니다. **바꿀 거면 사전 점검 전에 정하세요.** 예산은 첫 사전 점검에서 고정됩니다.

| 변수 | 기본값 | 의미 |
|---|---|---|
| `RERUN_TAG` | (필수) | 캠페인 이름. 예: `rtx5080_20261001`. 결과는 `_2026__IEEE_Access/revision/analysis/data/rerun/<TAG>/`에 쌓임 |
| `RERUN_BUDGETS` | `60,120,180` | 보고할 벽시계 예산 [s]. 이 모든 예산에서 보고하고 KM 곡선 전체도 같이 냄. **저자 결정** |
| `RERUN_REPS` | `3` | cuTAMP 반복 횟수 (레이아웃 30개씩) |
| `RERUN_PDDL_STREAMS` | `5` | PDDLStream 플래너 시드 개수 |
| `RERUN_PDDL_MAX_TIME` | `180` | PDDLStream 시간 제한 [s]. `RERUN_BUDGETS`의 최댓값 이상이어야 함 (사전 점검이 확인) |
| `RERUN_PLAN_TIMEOUT` | `600` | 드라이버가 cuTAMP를 기다리는 시간. 전체 풀이 시간을 기록하려고 넉넉하게 잡음. 예산은 분석에서 검열로 적용 |
| `RERUN_PERCEPTION` | `0` | `1`이면 선택 단계 3(인식 정확도)도 실행 |
| `RERUN_EXPECT_GPU` | `5080` | GPU 이름에 이 문자열이 없으면 경고 |

**원래 캠페인(2026-09-12/13)과 다른 점.** scene은 같습니다(legacy 유리기구, 높인 태그 마운트). 하지만 그 뒤로 **기본으로 켜진 동작**이 세 가지 있어서, 이번 측정은 개선된 현재 코드를 재는 것입니다. 사전 점검이 `RUN_INFO.md`에 모두 기록합니다.

| 스위치 | 기본값 | 추가 시점 | 영향 |
|---|---|---|---|
| `SDL_POUR_CONTINUATIONS` | 5 | 09-14, 붓기 경로를 끝까지 이어 가는 재시드 | Transfer 실행과 붓기 |
| `SDL_RECOVERY` / `SDL_RECOVERY_RETREAT` | 1 / 1 | 09-16, 태그를 놓치면 재검출을 기다리고, 안 되면 홈으로 물러나 다시 봄 | 인식 상태 Transfer |
| `SDL_PLAN_HOLD_SIM` | 1 | 09-28, cuTAMP가 계획하는 동안 시뮬레이터가 스텝(물리·렌더링)을 멈춤. 빈 scene 기준 5080 GPU 사용률: 스텝 중 98 %, timeline PAUSED만으로는 24–25 %(렌더링이 계속됨), 스텝 정지 6 %, 시뮬레이터 없음 8–13 % | cuTAMP 계획 시간 (세 작업 모두) |

폭 기준 **도구 선택 규칙**(09-14, `tool_rule.py`)은 실행 경로에 연결된 스위치가 아니라 분석 결과입니다. 시뮬레이션 시행은 작업마다 로봇을 고정하므로 영향이 없고, LLM 단계가 이 규칙을 같은 정답 라벨과 대조한 결과를 따로 냅니다. 원고에 "학습 대신 계산"으로 쓸지는 저자 결정입니다.

---

## 5. 사전 점검

```bash
export RERUN_TAG=rtx5080_$(date +%Y%m%d)
```

```bash
scripts/rerun/00_preflight.sh --smoke
```

모든 줄이 `[ok]`여야 합니다. `--smoke`는 Transfer 시행 1개를 끝까지 돌려서 scene 표지가 없는지까지 봅니다. 자주 나오는 실패는 이렇습니다.

| 실패 | 조치 |
|---|---|
| `Refusing to run: these SDL_* variables are set` | 새 셸을 열거나 `unset`. `~/.bashrc`에 `SDL_*`가 있는지 확인 |
| `uncommitted changes` | 추적되는 파일이 바뀐 것임. 커밋하거나 되돌림. 커밋 해시가 실행된 코드를 전부 설명해야 함. 추적 안 되는 새 파일은 이 검사에 걸리지 않음 |
| `other processes are on the GPU` | 목록에 나온 프로세스를 끝내고 다시 점검. 캠페인 내내 그 상태를 유지 |
| `this tag's data came from <commit>` | 코드가 바뀐 것임. 새 `RERUN_TAG`로 시작 |
| scene 항목 `[FAIL]` | 코드의 기본값이 논문 scene에서 벗어난 것임. 원인을 고치기 전에는 진행하지 말 것 |
| `XDL generator weights ... sha256 differs` | 다른 모델임. §3.6에서 다시 옮겨 옴 |

---

## 6. 실행

tmux 안에서 돌리세요. 터미널이 끊겨도 계속됩니다. `-e`로 태그를 넘기는 이유는, tmux 서버가 이미 떠 있으면 새 세션이 지금 셸의 환경변수를 물려받지 않기 때문입니다. 콘솔 로그는 **저장소 밖**에 씁니다. 저장소 안에 쓰면 재개할 때 사전 점검이 그 파일을 커밋 안 한 변경으로 보고 막습니다.

```bash
tmux new -s rerun -e RERUN_TAG="$RERUN_TAG" "scripts/rerun/run_all.sh 2>&1 | tee -a ~/rerun_${RERUN_TAG}.log"
```

단계별로 따로 돌려도 됩니다(`10_cutamp.sh`, `20_pddlstream.sh`, …). 순서는 지켜야 하고, **두 단계를 동시에 돌리면 안 됩니다.** 서로 GPU와 CPU를 나눠 쓰게 돼서 시간 측정이 의미를 잃습니다. 스크립트도 시뮬레이터 배치가 돌고 있으면 PDDLStream과 LLM 단계의 시작을 거부합니다.

**중단되면** 같은 `RERUN_TAG`로 다시 실행하세요. 끝난 배치는 건너뛰고, 중간에 끊긴 배치는 빠진 시드만 채웁니다.

**진행 상황 보기:**

```bash
tail -f _2026__IEEE_Access/revision/analysis/data/rerun/$RERUN_TAG/rerun.log
```

`batches.tsv`에는 배치마다 시작·종료·상태가, `gpu_monitor.csv`에는 30초마다 GPU 사용률·메모리·부하와 다른 GPU 프로세스가 기록됩니다.

**실행 중 하지 말 것:** 다른 GPU 작업(학습, 다른 Isaac 세션), `pkill -f`처럼 넓은 패턴으로 프로세스 죽이기(자기 셸까지 죽습니다), 코드 수정(다음 단계가 커밋 불일치로 거부합니다).

---

## 7. 분석과 감사

`run_all.sh`의 마지막 단계에서 자동으로 돌지만, 언제든 따로 돌려도 됩니다. 데이터는 읽기만 합니다.

```bash
scripts/rerun/90_analyse.sh
```

결과는 `…/rerun/<TAG>/RESULTS.md`이고 **맨 위에 감사 판정**이 나옵니다.

- **CLEAN:** 모든 배치가 끝났고, 모든 시행이 GPU를 독점했고, 논문 scene에서 돌았고, 레이아웃이 빠짐없이 한 번씩 있습니다.
- **NOT CLEAN:** 굵게 표시된 시행의 수치는 쓰지 마세요. GPU를 공유한 시행은 지우고 다시 돌릴 수 있습니다. 원본 CSV는 `.flagged-<시각>`으로 보존됩니다.

```bash
python3 _2026__IEEE_Access/revision/analysis/rerun_audit.py _2026__IEEE_Access/revision/analysis/data/rerun/$RERUN_TAG --drop-flagged
```

```bash
scripts/rerun/run_all.sh
```

감사는 **시행 단위**로 판정합니다. cuTAMP 시행의 구간은 직전 행이 기록된 시각부터 이번 행이 기록된 시각까지이고(한 배치 안에서 시행은 차례로 돕니다), 최대 15분으로 제한합니다. PDDLStream 시행은 자기 계획 시간이 구간입니다.

`results/`에는 다음이 들어갑니다.

| 파일 | 내용 |
|---|---|
| `audit.md` | 배치 상태, 시행별 GPU 독점 여부, 완전성 |
| `planner.md` | 예산별 성공률 [Wilson 95 %], RMST, 반복별 짝지은 exact McNemar, 성공분 풀이 시간. 두 플래너가 같은 레이아웃을 봤는지 시드별로 검사하고, 다르면 짝짓기를 거부함 |
| `state_source.md` | Transfer 작업 성공: 정답 상태 대 인식 상태, 반복별 exact McNemar |
| `trials_*.txt` | 배치별 요약 (성공률, 풀이 시간 분포, 배치 오차) |
| `pour_*.txt` | 붓기 립 오차 분해 |
| `perception.txt` | 인식 정확도 (선택 단계를 돌린 경우) |
| `llm.md` | XDL 필드별 정확도·추론 시간, 검증기 클래스별 재현율, Action Reasoner 정확도·추론 시간, 도구 규칙 |

**통계 방식:** 비율에는 Wilson 95 % 구간을 씁니다. 플래너 비교는 cuTAMP 반복 r과 PDDLStream 시드 r을 같은 30개 레이아웃에서 짝짓고, 짝마다 exact McNemar를 한 번씩 합니다. 같은 레이아웃의 반복끼리는 서로 독립이 아니어서 90쌍을 합쳐 한 번에 검정하지 않습니다. 시간은 예산에서 검열한 Kaplan–Meier와 RMST로 봅니다. 작업끼리는 합치지 않습니다(CLAUDE.md 통계 규칙).

---

## 8. 결과 커밋

로그(`logs/`)는 커서 git에서 제외됩니다. CSV, `RUN_INFO.md`, `gpu_monitor.csv`, `batches.tsv`, `results/`, `RESULTS.md`는 커밋합니다.

```bash
git add _2026__IEEE_Access/revision/analysis/data/rerun/$RERUN_TAG && git commit -m "Re-run on RTX 5080: $RERUN_TAG"
```

```bash
git push origin revision-access
```

---

## 9. 원고·응답 반영

**감사가 CLEAN인 수치만** 넣습니다. 같은 값이 원고, 응답, 표에 여러 번 나오면 **전부** 같은 값으로 바꾸고, 넣은 뒤에는 값으로 두 파일을 grep해서 확인합니다(CLAUDE.md 규칙 3). `\nd{}`는 이 수치로만 교체합니다.

| 결과 (`results/`) | `access_revised.tex` | `response_to_reviewers.tex` |
|---|---|---|
| RUN_INFO: GPU, CPU, Isaac Sim 버전, torch, 커밋 | §IV-A4 (`sec:4.1.4`), 부록 A. 지금 `\nd{4.2.0}`, `\nd{RTX 4090}` 자리 | R1#3 "Common conditions", R1#8 |
| `planner.md` 성공률·CI·McNemar·RMST, 예산별 | Table 5 (`tab:tamp_comparison`), §IV-C (`sec:4.3`), KM 그림 | R1#4 전체, R1#3의 기준선 문단 |
| `planner.md` PDDLStream 실패 유형과 성공분 시간 | §IV-C | R1#3 "early termination" 논거. 실측으로는 조기 종료가 0건이고 실패가 전부 예산 소진이라 **논거 자체를 교체**해야 함 |
| `state_source.md` | 인식 기반 end-to-end. §IV-E (`sec:4.5`), 그리고 PLAN.md §5.1에 따라 렌더 인식 기반으로 재작성할 §IV-F (`sec:4.6`) | R1#1 "Influence on planning success", R2#2 |
| `trials_*.txt` 작업 성공 (정답 상태) | Table 9 (`tab:end_to_end`), §IV-E | R1#2의 실행 성공 해석 |
| `llm.md` XDL 필드별 정확도 | Table 2 (`tab:xdl_generation`), §IV-B1 | R1#5 |
| `llm.md` XDL 추론 시간 | §IV-B1의 "1.57 s" 문장 | — |
| `llm.md` 검증기 | Table 3–4 (`tab:xdl_validator_cm`, `tab:xdl_injected_errors`), §IV-B2 | R1#6 |
| `llm.md` Action Reasoner, 도구 규칙 | Table 6 (`tab:tool_selection`), Table 7 (`tab:action_reasoner`), §IV-D1 (`sec:4.4.1`) | R1#7 |
| `llm.md` Action Reasoner 추론 시간 | §IV-D1의 "0.0665 s" 문장. 원래 값은 지금은 git에 없는 옛 `tool_move` 모델에서 나온 것 | — |

**지울 것.** 아래는 5080 결과와 상관없이 현재 자료가 이미 반박하는 확정값입니다. 새 수치로 교체하거나 문장째 지워야 합니다.

- `response_to_reviewers.tex` R1#4의 PDDLStream 66.7/83.3/60.0 %, cuTAMP 100 % on all three, McNemar 2.0e-3 / 0.063 / 4.9e-4, RMST 24.8/10.1/24.5 대 31.2/17.6/28.8, "29–34 s band", 7.21 ± 10.26 대 31.18 ± 1.44
- R1#3의 "same 60 s wall-clock budget", "early terminations (20.0 % on Transfer, 13.3 % on Stir)"
- 원고 Table 5와 §IV-C의 같은 값들

**플래너 비교를 서술하는 방식.** 결론이 예산에 따라 달라질 수 있습니다. 5070 자료에서는 60 s에서 기준선이, 180 s에서 cuTAMP가 우세했습니다. 선언한 모든 예산에서 보고하고, 우세가 바뀌면 바뀐다고 쓰세요. 한 예산만 골라 보고하지 않습니다.

---

## 10. 실기 scene (`transfer_real`) — 논문 수치에 쓰지 않음

실기 FR5에서 재생할 궤적을 만드는 경로입니다. 논문용 재실험과 스크립트, 출력 위치가 완전히 분리돼 있습니다. 출력은 `_2026__IEEE_Access/` 밖에만 쓸 수 있고, 안쪽을 지정하면 거부합니다.

```bash
scripts/real_cell/record_transfer_real.sh ~/real_cell_runs/$(date +%Y%m%d) 0 1 2 3 4 5 6 7
```

```bash
scripts/real_cell/score_real_trajectories.py ~/real_cell_runs/<날짜> --export
```

scene 값은 2026-09-23 실측이고 스크립트 머리말에 있습니다. 벤치 −13 mm, 비커(실측 50 × 50 × 70 mm, 모델 60 mm)는 (0.584, −0.298)의 5 cm 받침대 위, 플라스크는 (0.520, 0.320)의 저울 판 위, 수평 파지 0–10°, 붓고 나면 받침대 위로 복귀합니다. 채점기는 EEF 수평도, 실제 비커 테두리 아래 파지 여유, 받침대 위 직립 복귀, 붓기 립 오차, 최고 기울기로 합격 여부를 가리고, 합격한 궤적만 CSV와 메타로 내보냅니다.

**실기 scene이 다른 이유:** 수평 측면 파지에서는 손목 몸체가 파지점 아래로 약 58 mm까지 내려옵니다. 그래서 벤치에 선 비커는 뾰족한 테두리 근처만 잡을 수 있고, 받침대로 비커를 올려 파지점이 몸통에 오게 했습니다. 논문 시뮬레이션은 10–35° 밴드를 쓰기 때문에 이 문제가 없습니다.

---

## 11. 이 런북으로 돌릴 수 없는 항목

구현이 없거나 저자 결정이 남은 항목입니다. **수치를 만들어 넣지 마세요.** PLAN.md 규칙대로 `\nd{}`로 두거나, 해당 문장을 지웁니다.

| 항목 | 상태 | 응답 위치 |
|---|---|---|
| B1 인식 잡음 주입 (측정 오차 bootstrap) | 미구현 | R1#1 마지막 문장, Table 5의 잡음 열 |
| B7 3단계 재배치 ablation | 미착수 | R1#7 "Extended sequence-level evaluation" |
| B8 적대적 명령 70개 | 선택, 미구현 | R2#4 빨간 문단 |
| B9 256 파티클 ablation | 선택, 미구현 | R1#3 빨간 문장 |
| θ_max 기울기 × 충전율 쏟김 sweep | 미구현 | R3#2 |
| Action Reasoner "unseen" 주장 유지 여부 | 저자 결정 | R1#7 |
| 물리 실기 (A 항목) | 이 런북 범위 밖 | R1#1·#2, R2#2, R3#1·#3 |

---

## 12. 알려진 함정

- **cuTAMP는 시드로 재현되지 않습니다.** CUDA 비결정성 때문에 같은 시드도 매번 다르게 풀립니다. 원래 머신과 숫자가 조금씩 다른 게 정상이고, 그래서 반복을 3회 합니다.
- **`ROS_DOMAIN_ID`는 100이 기본입니다.** 같은 네트워크의 다른 ROS 시스템과 겹치면 바꾸세요.
- **ROS setup 스크립트는 `set -u`와 충돌합니다.** 직접 스크립트를 쓸 때 `set -u`를 켜지 마세요.
- **정착 오차.** 시행 시작 시점의 용기 위치는 물리 안착 때문에 0.1 mm쯤 흔들립니다. 분석의 레이아웃 비교는 5 mm까지 허용하는데, 서로 다른 레이아웃은 cm 단위로 차이 나므로 둘을 확실히 구분합니다.
- **unsloth 컴파일 캐시**는 작업 디렉터리에 생깁니다. 스크립트 옆의 캐시는 git에 추적되고 있어서, LLM 단계는 `logs/llm_cwd/`에서 실행합니다. 스크립트 폴더에서 직접 돌리면 작업 트리가 더러워집니다.
- **`compare_planners.py`는 한 번 실행한 결과끼리만 비교합니다.** 반복 실행 결과는 `planner_comparison.py`로 비교합니다. 전자는 시드당 여러 행이 있는 파일을 거부합니다(예전에는 경고 없이 마지막 행만 남겼습니다).
- **첫 Isaac Sim 구동은 셰이더 컴파일로 4분쯤 걸립니다.** 사전 점검의 `--smoke`가 그 시간을 먼저 치러 둡니다.
