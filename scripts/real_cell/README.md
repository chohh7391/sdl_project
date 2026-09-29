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
| cho 워크스페이스 | `~/ros2_ws` (`cho_robot_project`). `cho_task_manager`는 소스에 링크된 개발 설치라 소스 수정이 바로 반영된다 |

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

## 2. 로컬 bringup 순서

각 줄은 별도 터미널. ROS 환경은 `source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash`.

```bash
# 로봇 (시작하면 그리퍼가 자동으로 열린다 — 잡고 있는 비커를 받칠 것)
ros2 launch cho_bringup_fr5 bringup_real_robot.launch.py        # MoveIt 홈 이동이 필요하면 bringup_real_moveit
ros2 launch cho_bringup_fr5 camera_extrinsics.launch.py

# 카메라: D435 두 대는 stream:=both (태그용 infra1 848x480@30 그대로 + 보기용 컬러 640x480@15)
CFG=$(ros2 pkg prefix --share cho_bringup_fr5)/config/real
ros2 launch cho_realsense d435.launch.py camera_namespace:=side_2 camera_name:=side_2 serial_no:=_332322072253 camera_config:=$CFG/d435_side.yaml stream:=both
ros2 launch cho_realsense d435.launch.py camera_namespace:=wrist  camera_name:=wrist  serial_no:=_844212070094 camera_config:=$CFG/d435_wrist.yaml stream:=both
ros2 launch cho_oak oak.launch.py name:=side_1

# 저울
ros2 launch hansung_scale_driver scale.launch.py
ros2 run cho_control_tools scale_relay

# 인식: 두 비커 테이블 하나로 detector + pose node (fusion_mode 기본 intersect, 한 카메라만 봐도 publish)
TABLE=~/ros2_ws/src/cho_robot_project/cho_task_manager/config/perception/pour_beakers.yaml
CAMS=$(ros2 pkg prefix --share cho_object_pose)/config/cameras.yaml
ros2 launch cho_object_pose detectors.launch.py objects_config:=$TABLE cameras_config:=$CAMS
ros2 launch cho_object_pose object_pose.launch.py robot_type:=fr5 objects_config:=$TABLE cameras_config:=$CAMS min_cameras:=1

# 모니터링 (RViz + monitor_node): 끝까지 띄워 둔다
~/sdl_ws/src/sdl_project/scripts/real_cell/monitor.sh
```

`pour_beakers.yaml`은 beaker1을 `/perception/object_pose/held_beaker`로 publish한다. `pouring_controller`가
marker를 이 topic에서 읽기 때문(`controllers.yaml` `vessel_pose_topic`)이며, 인식·recovery·붓기가 pose node 하나를 쓴다.
tag 0/1은 `vessel_detect.yaml`·`pour_vessel.yaml`과 겹치므로 **테이블은 하나만** 띄운다.

## 3. 실행

```bash
cd ~/sdl_ws/src/sdl_project
scripts/real_cell/plan_remote.sh --perceive --beakers --material granular --grams 10 0 1 2 --home-via direct --replay
```

(설탕이면 `granular`, 물이면 `liquid`. 뒤의 숫자는 플래닝 seed 목록.)

단계별로 일어나는 일 — 진행 상태는 RViz 상태 줄(`~/real_cell_runs/STATUS`)에 나온다.

1. **인식** — `measure_vessels.py`가 beaker1, beaker2의 pose를 모은다(side_1 + side_2 fusion). 비커 하나당
   샘플 10개 이상, 퍼짐 5 mm / 방향 5° 이내일 때만 채택한다. 결과는 xy와 plate 방향(bearing).
2. **occlusion recovery (못 본 비커가 있을 때만)** — 예: beaker1 안의 설탕이 side 카메라에서 태그를 가림.
   `yes` 확인 뒤 `cho_task_manager occlusion_recovery`가 팔을 움직인다.
   - 경로: home(관절 직선 이동) → 공구 약 0.55 m 높이에서 손목 카메라를 아래로 향한 raster → home.
   - 설정: `vessel_table:=pour_beakers.yaml`, `sweep_config:=fr5_bench_beakers.yaml`.
   - 그 사이 `measure_vessels.py --latch`가 손목 카메라가 태그를 잡는 순간 값을 받는다.
   - 로그: `~/real_cell_runs/live_logs/recovery_*.log`.
3. **원격 플래닝** — 인식값을 `SDL_LAYOUT_XY`(beaker, beaker2), `SDL_TAG_PLATES`로 rci-pc에 보낸다.
   rci-pc에서는 `record_transfer_real.sh`(headless Isaac Sim + cuTAMP, seed마다 한 번)와
   `score_real_trajectories.py --export`가 돈다.
   - 원격 작업은 ssh 세션과 분리돼 있다. 연결이 끊기면 `--attach <run_id>`로 다시 붙는다.
   - 통과한 궤적은 `~/real_cell_runs/<run_id>/deliver/`로 가져온다. 계획 위치가 요청값에서 2 mm 넘게
     벗어난 궤적은 버린다. 재생용 cell layout(`*_cell.yaml`)도 만든다.
4. **재생 전 검사** — `check_replay.py`가 재생 트리를 오프라인으로 만든다(명령 없음). 걸리는 것:
   layout gate, 붓기 splice, 속도 한계에 따른 시간 늘리기.
5. **재생** — `yes` 확인 뒤 `trajectory_replay`를 실행한다. 궤적 중 `pouring` 구간은 재생하지 않는다.
   그 구간은 `pouring_controller`가 저울 무게로 붓고(`--material`, `--grams`), 붓기 시작 자세로 돌아온 다음
   남은 궤적을 이어서 재생한다.

그 밖의 실행 방법:

```bash
scripts/real_cell/plan_remote.sh --fetch <run_id> --beakers --material liquid --grams 20   # 이미 끝난 run 다시 가져와 검사
scripts/real_cell/plan_remote.sh --attach <run_id>                                        # 끊긴 원격 run에 다시 붙기
scripts/real_cell/measure_vessels.py beaker1 beaker2                                      # 인식만 확인
```

옵션 요약: `--beaker1 X,Y` / `--beaker2 X,Y` / `--plate1 DEG` / `--plate2 DEG` (직접 지정), `--no-recover`,
`--home-via moveit|direct` (moveit은 `bringup_real_moveit` 필요), `--flow-index F`, `--yes` (확인 생략, 비추천).
전체는 `scripts/real_cell/plan_remote.sh -h`.

## 4. 안전장치 (거부 조건)

| 조건 | 이유 |
|---|---|
| rci-pc 커밋 ≠ 로컬 HEAD | 궤적은 플래너·URDF·scene 코드에 의존 |
| 두 비커 셀에 인식(또는 직접 지정) 위치가 없음 | 기본 배치로 대신 계획하지 않는다 |
| beaker2가 pan 중심에서 15 mm 초과 | 컨트롤러가 받는 용기를 pan 중심으로 가정 (`BEAKER2_TOL_M`) |
| 인식 샘플 부족·퍼짐 초과 | 움직이는 pose로 계획하지 않는다 |
| 계획 위치 ≠ 요청 위치 (> 2 mm) | 요청이 플래너에 닿지 않은 것 |
| 재생 트리 빌드 실패 | layout / splice / 속도 문제는 움직이기 전에 거부 |
| 팔이 움직이는 모든 단계 | 터미널에서 `yes` (recovery, 재생) |

## 5. 남은 문제 / 주의

- **컨트롤러의 받는 용기 위치**: `controllers.yaml`의 `receiver_rim_center`는 [0.520, 0.320, 0.180]으로 고정돼 있고
  인식값을 쓰지 않는다. 인식으로 추정한 beaker2 rim은 약 (0.513, 0.328, 0.158)이라, 컨트롤러가 rim을 22 mm 높게 안다
  (더 높은 곳에서 붓는 쪽이라 안전). 자로 잰 값으로 바꾸고 bringup을 재시작해야 한다.
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
- **저장소 반영**: 이 흐름이 쓰는 `cho_robot_project` 변경은 그 저장소 `main`에 있다 — `3829cd9`
  (`d435.launch.py` `stream:=both`), `108cae3` (`pour_beakers.yaml`, recovery `vessel_table`,
  `fr5_bench_beakers.yaml`).

## 6. 문제 해결

| 증상 | 조치 |
|---|---|
| `ServoJ refused with code -2`, `robot state could not be read` | 로봇 통신 끊김. bringup 재시작(그리퍼가 열린다). 계속되면 펜던트 오류 확인 |
| 그리퍼 명령이 ABORTED, 폭이 그대로 | 위와 같은 원인. bringup 로그의 `gripper worker alive: ... sent` 확인 |
| 비커가 RViz에 안 뜸 | `~/real_cell_runs/live_logs/object_pose.log`의 `[beaker1]` 줄에서 카메라별 상태 확인 |
| 인식 pose가 크게 흔들림 | 비커를 옮기는 중이었는지 확인. 멈춘 뒤 다시 측정 |
| 백그라운드 노드가 SIGINT로 안 꺼짐 | 비대화형 셸 `&` 작업은 SIGINT를 무시한다. SIGTERM(`kill <pid>`) |
| 두 비커 플래닝이 Pick 접근에서 실패 | 넓은 받침 윗면 위로 수평 파지 손목(파지점 아래 약 58 mm)이 지나가지 못함. 좁은 받침(bar) 사용 |
