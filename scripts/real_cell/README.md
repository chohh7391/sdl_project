# Real cell: 인식 → (recovery) → 원격 플래닝 → 재생·붓기 → RViz 모니터링

두 비커 셀(beaker1 → beaker2 transfer)을 실제 FR5로 돌리는 순서. **논문 수치의 출처가 아니다**
— 여기서 만드는 궤적은 로봇용이고, `record_transfer_real.sh`는 논문 데이터 디렉터리 안에는 쓰지 않는다.

```
 local (노트북: FR5, 카메라 3대, 저울)                        rci-pc (100.95.230.128, RTX 5070)
 ─────────────────────────────────────                        ───────────────────────────────
 1. side_1 + side_2 fusion 인식 ──┐
 2. 못 보면 occlusion recovery ───┤ beaker1/2 xy + plate 방향
                                  └──── ssh (Tailscale) ────▶ 3. Isaac Sim + cuTAMP 플래닝
                                  ┌──── scp ◀──────────────── 궤적 CSV + meta + 점수
 4. 궤적 재생, 붓기는 pouring_controller (무게 피드백)
 5. 전 과정을 RViz에서 모니터링 (monitor.sh)
```

## 0. 한 번만 하는 준비

| 항목 | 방법 |
|---|---|
| Tailscale | `curl -fsSL https://tailscale.com/install.sh \| sh` → `sudo tailscale up` (rci-pc와 같은 계정) → `tailscale ping 100.95.230.128` |
| SSH 키 | `ssh-copy-id home@100.95.230.128` (비밀번호 없이 `ssh 100.95.230.128 true`가 되어야 한다) |
| 같은 커밋 | `plan_remote.sh`는 rci-pc의 `sdl_project`가 로컬 HEAD와 다르면 거부한다. 로컬에서 커밋 후: `git push ssh://100.95.230.128/home/home/sdl_ws/src/sdl_project HEAD:refs/heads/laptop-sync` → rci-pc에서 `git merge --ff-only laptop-sync` |
| 셀 워크스페이스 | 코어 `~/ros2_ws` (`cho_robot_project` v0.1.0, `ros2_cell_ws/cell.repos`) 위의 overlay `ros2_cell_ws`. 빌드는 [ros2_cell_ws/README.md](../../ros2_cell_ws/README.md) Build. 스크립트는 `ros2_cell_ws/install/setup.bash`를 source한다(코어 것도 같이 들어온다). 다른 곳이면 `CELL_WS` |

## 1. 셀 배치 (리그 측정값, 2026-09-29)

`record_transfer_real.sh`의 `SDL_TARGET_VESSEL=beaker2` 기본값이 이 배치다. 바뀌면 거기(또는 환경변수)를 고친다.

| 대상 | 값 |
|---|---|
| 벤치 | robot base 기준면보다 13 mm 아래 (`SDL_TABLE_Z_M=-0.013`) |
| beaker1 받침 | bar, x 200 mm × y 70 mm × 높이 70 mm, 밑면 중심 base_link (0.36, −0.30, −0.013). beaker1은 그 위 아무 곳(인식값) |
| 저울 | base_link와 같은 방향(회전 고정, 인식 안 함), pan은 +x 끝, pan 중심 (0.520, 0.320), pan 윗면 90 mm |
| beaker2 | 저울 pan 위. 인식 위치로 계획하되 pan 중심에서 15 mm 이내여야 한다 |
| 태그 | beaker1 = tag 0, beaker2 = tag 1 (70 mm, 36h11) |
| 태그 마운트 | `hardware/tag_mount/square88_circle_connector_r20_c110.stl`: 비커가 서는 pad(R 20, 두께 2 mm)와 plate 중심 사이 110 mm. **태그는 인쇄 기준 아래 변이 bridge 쪽** (offset [0, −0.110, 0]). plate는 플래너에 장애물로 들어간다 |
| 비커 | 50 × 50 × 70 mm (beaker1 모델은 파지 대역 때문에 60 mm) |

**그리퍼 adapter flange (2026-09-29).** AG-95가 약 15 mm flange 위에 달려 있다. pour controller는
`grasp_depth` 0.285와 `hand_boxes` +15 mm로 반영했다(`sdl_cell_bringup/config/real/pouring_controller.yaml`).
URDF의 `fr5.urdf.xacro gripper_flange`(기본 0.015)는 노트북 코어 checkout에만 있었고 코어 v0.1.0에는 없다. 지금 로컬
MoveIt·RViz의 그리퍼는 실제보다 15 mm 짧다(5절).
플래너 쪽은 논문 scene을 건드리지 않도록 `SDL_GRIPPER_FLANGE_M`(`record_transfer_real.sh` 기본 0.015)이 켜진
실제 셀 run에서만 cuRobo URDF·sphere(`cutamp/robots/flange.py`)와 Isaac Sim 그리퍼(`task.py _apply_gripper_flange`)를
15 mm 밀고, 점수와 RViz 경로도 `SCENE_ENV.txt`의 값을 쓴다.

## 2. 로컬 bringup 순서

각 줄은 별도 터미널. ROS 환경은 `source /opt/ros/humble/setup.bash && source ~/sdl_ws/src/sdl_project/ros2_cell_ws/install/setup.bash`
(셀 overlay. 코어 `~/ros2_ws`도 같이 들어온다).

```bash
# 로봇 + MoveIt + pouring_controller (시작하면 그리퍼가 자동으로 열린다 — 잡고 있는 비커를 받칠 것).
# 코어 fr5.config.yaml 위에 셀의 config/real/fr5_config_overrides.yaml(그리퍼 force 60 %)을 얹어 넘긴다.
# 재생 시작 자세와 recovery 시작 home은 기본으로 MoveIt이 계획한다 (2026-09-30).
# MoveIt 없이(moveit:=false) 띄우면 plan_remote에 --home-via direct를 준다.
ros2 launch sdl_cell_bringup bringup_real_cell.launch.py moveit:=true
ros2 launch cho_bringup_fr5 camera_extrinsics.launch.py \
  extrinsics_config:=$(ros2 pkg prefix --share sdl_cell_bringup)/config/real/camera_extrinsics.yaml

# 카메라: D435 두 대는 셀의 d435.launch.py (기본 stream:=both: 태그용 infra1 848x480@30은 코어
# stream:=infra1 그대로 + 보기용 컬러 640x480@15)
CFG=$(ros2 pkg prefix --share cho_bringup_fr5)/config/real
ros2 launch sdl_cell_bringup d435.launch.py camera_namespace:=side_2 camera_name:=side_2 serial_no:=_332322072253 camera_config:=$CFG/d435_side.yaml
ros2 launch sdl_cell_bringup d435.launch.py camera_namespace:=wrist  camera_name:=wrist  serial_no:=_844212070094 camera_config:=$CFG/d435_wrist.yaml
ros2 launch cho_oak oak.launch.py name:=side_1

# 저울
ros2 launch hansung_scale_driver scale.launch.py
ros2 run sdl_cell_tasks scale_relay

# 인식: 두 비커 테이블 하나로 detector + pose node (fusion_mode 기본 intersect, 한 카메라만 봐도 publish)
TABLE=$(ros2 pkg prefix --share sdl_cell_tasks)/config/perception/pour_beakers.yaml
CAMS=$(ros2 pkg prefix --share cho_object_pose)/config/cameras.yaml
ros2 launch cho_object_pose detectors.launch.py objects_config:=$TABLE cameras_config:=$CAMS
ros2 launch cho_object_pose object_pose.launch.py robot_type:=fr5 objects_config:=$TABLE cameras_config:=$CAMS min_cameras:=1

# 모니터링 (RViz + monitor_node): 끝까지 띄워 둔다
~/sdl_ws/src/sdl_project/scripts/real_cell/monitor.sh
```

`pour_beakers.yaml`은 beaker1을 `/perception/object_pose/held_beaker`로 publish한다. `pouring_controller`가
marker를 이 topic에서 읽기 때문(`pouring_controller.yaml` `vessel_pose_topic`)이며, 인식·recovery·붓기가 pose node 하나를 쓴다.
tag 0/1은 `vessel_detect.yaml`·`pour_vessel.yaml`과 겹치므로 **테이블은 하나만** 띄운다.

## 3. 실행

```bash
cd ~/sdl_ws/src/sdl_project
scripts/real_cell/plan_remote.sh --perceive --beakers --recover beaker1 --material granular --grams 10 --replay 0 1 2
```

(설탕이면 `granular`, 물이면 `liquid`. 뒤의 숫자는 플래닝 seed 목록. `--recover beaker1`은 standing 카메라가 값을
주더라도 beaker1을 wrist로 다시 잰다.)

**팔이 움직이기 전에 묻지 않는다** (2026-09-29, 저자 요청). recovery와 재생이 확인 없이 바로 시작된다.
묻게 하려면 `--confirm`. **비커 위치는 perception으로만 얻는다** — `--beaker1/--beaker2/--receiver` 같은 직접 지정과
기본 배치(예전 (0.584, −0.298), (0.36, −0.30))는 없앴고, `record_transfer_real.sh`도 `SDL_LAYOUT_XY`에 인식값이
없으면 거부한다. riser·저울 위치는 가구라서 그대로다.

단계별로 일어나는 일 — 진행 상태는 RViz 상태 줄(`~/real_cell_runs/STATUS`)에 나온다.

1. **인식** — `measure_vessels.py`가 beaker1, beaker2의 pose를 모은다(side_1 + side_2 fusion). 비커 하나당
   샘플 10개 이상, 퍼짐 5 mm / 방향 5° 이내일 때만 채택한다. 결과는 xy, 바닥 높이 z, plate 방향(bearing).
   - **인식한 위치 유지 (hold) — 지금 코어에는 없다.** `pour_beakers.yaml`의 `hold_unseen: true`는 노트북 코어
     checkout의 cho_object_pose(6e364e1)가 읽던 키다: 한 번 인식된 비커는 카메라가 놓쳐도 마지막 pose를 계속
     publish하고, 재생에서 jaws가 beaker1을 잡은 직후 `<topic>/release`로 해제했다. 코어 v0.1.0의 pose node는 이
     키를 무시하므로, 지금은 카메라가 못 보는 비커에 pose가 없다(5절). 재생의 release/restore 호출은 서비스가 없어
     실패하고 그대로 넘어간다.
2. **occlusion recovery (못 본 비커, 또는 `--recover NAME`)** — 예: beaker1 안의 설탕이 side 카메라에서 태그를 가림.
   셀의 `sdl_occlusion_recovery`가 팔을 움직인다. 다시 잴 비커만 담은 sweep 표로 훑는다.
   - 경로: home(`--home-via`, 기본 MoveIt) → 공구 약 0.55 m 높이에서 손목 카메라를 아래로 향한 raster → home.
   - 설정: task_params `vessel_table: pour_beakers.yaml`, `sweep_config: fr5_bench_beakers.yaml`의 부분 표
     (`~/real_cell_runs/live_logs/recovery_*.task_params.yaml`).
   - 그 사이 `measure_vessels.py --latch --camera wrist --min-margin 55`가 **wrist가 직접 잰** pose만 받는다
     (hold 값이나 side 카메라 단독 값은 버린다).
   - 로그: `~/real_cell_runs/live_logs/recovery_*.log`.
3. **원격 플래닝** — 인식값을 `SDL_LAYOUT_XY`(beaker, beaker2), `SDL_TAG_PLATES`로 rci-pc에 보낸다.
   rci-pc에서는 `record_transfer_real.sh`(headless Isaac Sim + cuTAMP, seed마다 한 번)와
   `score_real_trajectories.py --export`가 돈다.
   - 원격 작업은 ssh 세션과 분리돼 있다. 연결이 끊기면 `--attach <run_id>`로 다시 붙는다.
   - 통과한 궤적은 `~/real_cell_runs/<run_id>/deliver/`로 가져온다. 계획 위치가 요청값에서 2 mm 넘게
     벗어난 궤적은 버린다. 재생용 cell layout(`*_cell.yaml`)도 만든다.
4. **재생 전 검사** — `check_replay.py`가 재생 트리를 오프라인으로 만든다(명령 없음). 걸리는 것:
   layout gate, 붓기 splice, 속도 한계에 따른 시간 늘리기. 트리를 만든 task_params를
   `<run>/deliver/traj_realcell_seed<N>.task_params.yaml`에 쓰고, 재생은 그 파일을 `task_params:=`로 받는다.
5. **재생** — 두 비커 셀에서 붓기가 있으면 셀의 `sdl_beaker_transfer` task를 실행한다(2026-09-30).
   플래닝이 정한 것만 한다: task_params의 `transfer_plan: <run>/deliver/traj_realcell_seed<N>` 하나로 궤적·cell layout·
   붓는 대상(`*_cell.yaml`의 `pour:` 블록: beaker2 rim, beaker1 바닥 중심, 제자리에 놓기)을 그 plan 파일에서 읽는다.
   붓는 양과 물질(`replay_pour_grams`, `replay_pour_material`)만 따로 준다. 그 밖의 경우는 `sdl_trajectory_replay`.
   `--grasp-only`는 `sdl_grasp_trial`.
   궤적 중 `pouring` 구간은 재생하지 않는다.
   그 구간은 `pouring_controller`가 저울 무게로 붓고(`--material`, `--grams`), 붓기 시작 자세로 돌아온 다음
   남은 궤적을 이어서 재생한다.
   - **붓는 대상 위치는 인식값이다.** beaker2의 인식 바닥 중심 (x, y, z)에 높이 70 mm(`BEAKER2_HEIGHT_M`)를
     더한 rim 중심이 plan의 `pour.receiver_rim`이 되고(`sdl_trajectory_replay`면 task_params
     `replay_pour_receiver: [x, y, z]`), 트리가 Pour goal의 `receiver_rim_center`에 싣는다.
     컨트롤러(`pour_geometry: measured`)는 이 값이 없는 goal을 거부하며, 설정 파일에는 받는 용기 위치가 없다
     (`receiver_rim_center`를 설정에 넣으면 configure가 실패한다).
   - RViz에는 이 rim이 자홍색 원(`pour target`)으로, 상태 줄에 `pour into the rim at (...)`로 나온다.
   - **grasp marker도 인식값이다.** jaws가 닫힌 뒤 topic을 새로 읽지 않고, 플래닝에 쓴 beaker1 인식 바닥 중심
     (plan의 `pour.grasp_marker`, 또는 `replay_pour_grasp_marker`)을 그 자리의 관절값과 짝지운다. side 카메라가 나쁜 각도에서 따로 잰 값이
     붓기 기하를 옮기지 않게 하기 위해서다.

그 밖의 실행 방법:

```bash
scripts/real_cell/plan_remote.sh --fetch <run_id> --beakers --material liquid --grams 20   # 이미 끝난 run 다시 가져와 검사
scripts/real_cell/plan_remote.sh --attach <run_id>                                        # 끊긴 원격 run에 다시 붙기
scripts/real_cell/measure_vessels.py beaker1 beaker2                                      # 인식만 확인
```

옵션 요약: `--recover NAME`, `--no-recover`, `--plate1 DEG` / `--plate2 DEG` (plate 방향만 덮어씀),
`--home-via moveit|direct` (moveit은 `bringup_real_moveit` 필요), `--flow-index F`, `--confirm` (움직이기 전에 묻기).
전체는 `scripts/real_cell/plan_remote.sh -h`.

## 4. 안전장치 (거부 조건)

| 조건 | 이유 |
|---|---|
| rci-pc 커밋 ≠ 로컬 HEAD | 궤적은 플래너·URDF·scene 코드에 의존 |
| 두 비커 셀에 인식(또는 직접 지정) 위치가 없음 | 기본 배치로 대신 계획하지 않는다 |
| beaker2가 pan 밖 (중심에서 x 또는 y로 40 mm 초과) | 저울 위에 있어야 무게를 잰다. 220 mm 튀면 태그가 plate에서 돌아간 것 (`BEAKER2_ON_PAN_M`) |
| 붓기에 받는 용기 rim이 없음 | `sdl_beaker_transfer`는 plan에 `pour:` 블록이 없으면 트리 빌드에서 거부. 실기 컨트롤러(`pour_geometry: measured`)도 rim 없는 goal 거부 (`sdl_trajectory_replay`는 MuJoCo의 joint 붓기 때문에 트리에서는 요구하지 않는다) |
| 인식 샘플 부족·퍼짐 초과 | 움직이는 pose로 계획하지 않는다 |
| 계획 위치 ≠ 요청 위치 (> 2 mm) | 요청이 플래너에 닿지 않은 것 |
| 재생 트리 빌드 실패 | layout / splice / 속도 문제는 움직이기 전에 거부 |
| 팔이 움직이는 단계 | 묻지 않는다(저자 요청). `--confirm`이면 터미널에서 `yes` |
| 인식값 없는 비커 | 직접 지정·기본 배치 없음: `--perceive` 필수, 원격 `record_transfer_real.sh`도 거부 |

## 5. 남은 문제 / 주의

- **beaker2 높이**: 인식한 beaker2 바닥 z는 약 0.087 m로, 모델(벤치 −0.013 + pan 0.090 + 패드 0.002 = 0.079)보다
  8 mm 높다. rim은 인식값을 쓰므로 붓는 높이가 그만큼 높아지는 쪽(안전)이다. 원인(저울 높이 또는 태그 z 편향)은 미확인.
- **side_1 지연**: side_1 AprilTag 처리가 CPU를 많이 쓰고, pose node가 한 코어를 100% 쓴다. side_1 샘플이
  "stale" 또는 TF extrapolation으로 버려지는 일이 잦다. 태그가 한쪽 카메라에만 보이면 인식이 끊긴다
  (그때 recovery가 돈다).
- **home 이동**: recovery와 `--home-via direct` 재생은 충돌 검사 없는 관절 직선 이동이다. 팔이 home에서 멀면
  크게 돈다(예: j5 +189°). 가능하면 `bringup_real_moveit` + `--home-via moveit`.
- **가루 붓기 기울기 한계**: 재생 경로에서는 최대 기울기가 1.2 rad(68.8°)로 제한된다. 덜 찬 비커는 안 나올 수 있다.
- **점수 기준**: `score_real_trajectories.py`는 녹화된 붓기(lip, 최대 기울기)도 보는데, `--material`을 쓰면
  실제 붓기는 컨트롤러가 한다. 이 기준 때문에 버려지는 seed가 있다.
- **sweep 표**: `fr5_bench_beakers.yaml`은 리그에서 검증된 `fr5_bench.yaml`과 waypoint가 같고 이름만 다르다
  (2026-09-29에 다시 풀면 waypoint 36개, 2개 도달 불가로 결과가 달라서 새로 푼 값은 쓰지 않았다).
- **저장소 반영 (2026-10-26)**: 이 흐름이 쓰던 노트북 cho_robot_project 변경(GitHub에 없던 29개 커밋)은
  ros2_cell_ws로 옮겼다 — task(`sdl_beaker_transfer`, `sdl_grasp_trial`, `sdl_occlusion_recovery`의 `vessel_table`),
  표(`pour_beakers.yaml`, `fr5_bench_beakers.yaml`), pour controller 코드와 실기 값, camera extrinsics, 그리퍼 force,
  `d435.launch.py stream:=both`(셀 launch). 목록은 [ros2_cell_ws/README.md](../../ros2_cell_ws/README.md)
  "What came from the laptop". 코어 자체를 고치는 다음 네 가지는 옮기지 않았고, 코어 v0.1.0으로 실기를 돌리면 다르다:
  - **15 mm flange 모델 없음**: MoveIt 충돌 검사와 RViz의 그리퍼가 실제보다 15 mm 짧다. 플래너(cuRobo/Isaac)는
    `SDL_GRIPPER_FLANGE_M`으로 따로 반영하므로 계획 궤적에는 영향이 없고, MoveIt home 이동과 RViz에만 해당한다.
  - **pose node hold 없음**: 카메라가 놓친 비커는 pose가 사라진다. recovery로 잰 wrist 값도 유지되지 않는다.
    Level은 tag 기울기(`in_hand`) 대신 grasp 축으로 세운다.
  - **joint_trajectory_controller path tolerance가 0.10 rad**(노트북은 0.30): 재생 pick에서 팔이 명령의 50-70 %로
    따라갈 때 abort되고, 그 abort가 FR5를 main=1 sub=24/25로 멈춰 전원을 다시 넣어야 했다(2026-09-29/30, 8번 중 4번).
  - **fairino 드라이버 ServoJ 큐 skip 없음**: 큐가 6을 넘을 때 전송을 건너뛰던 처리와 5초 타이밍 로그가 없다.

## 6. 문제 해결

| 증상 | 조치 |
|---|---|
| `ServoJ refused with code -2`, `robot state could not be read` | 로봇 통신 끊김. bringup 재시작(그리퍼가 열린다). 계속되면 펜던트 오류 확인 |
| 그리퍼 명령이 ABORTED, 폭이 그대로 | 위와 같은 원인. bringup 로그의 `gripper worker alive: ... sent` 확인 |
| 비커가 RViz에 안 뜸 | `~/real_cell_runs/live_logs/object_pose.log`의 `[beaker1]` 줄에서 카메라별 상태 확인 |
| 인식 pose가 크게 흔들림 | 비커를 옮기는 중이었는지 확인. 멈춘 뒤 다시 측정 |
| 백그라운드 노드가 SIGINT로 안 꺼짐 | 비대화형 셸 `&` 작업은 SIGINT를 무시한다. SIGTERM(`kill <pid>`) |
| 두 비커 플래닝이 Pick 접근에서 실패 | 넓은 받침 윗면 위로 수평 파지 손목(파지점 아래 약 58 mm)이 지나가지 못함. 좁은 받침(bar) 사용 |
