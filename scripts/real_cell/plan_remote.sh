#!/usr/bin/env bash
# Plan real-cell Transfer trajectories on a REMOTE GPU machine over SSH (e.g. over
# Tailscale), bring the accepted ones back, and replay one on the FR5 attached to
# THIS machine.
#
#   scripts/real_cell/plan_remote.sh --perceive --pan 0.520,0.320 0 1 2 3
#   scripts/real_cell/plan_remote.sh --perceive --beakers --material liquid --grams 20 4
#   scripts/real_cell/plan_remote.sh --attach 20260929-141203     # re-join after the ssh dropped
#   scripts/real_cell/plan_remote.sh --fetch  20260929-141203     # a finished run: copy + check only
#   add --replay to also launch trajectory_replay here. Nothing asks before the arm
#   moves (the author, 2026-09-29); --confirm brings the yes/N prompts back.
#   --use-seed N replays seed N instead of the best-scored one; it has to be one
#   of the accepted seeds (e.g. when the best one passes close to a camera).
#   --grasp-only replays only the plan's grasp of beaker1 (task grasp_trial:
#   reach, close, check the jaws stopped on it, open, back off, home), for a
#   perception-to-grasp trial series; no --material is needed.
#   --measure-only perceives (and recovers, with --recover) and stops there:
#   no planning. Run after a --grasp-only replay it measures where the jaws
#   left beaker1.
#   --repeat N pours N times at the pour start, one trial after another, for a
#   pour's statistics; each pour after the first waits for the operator's go:
#   scripts/real_cell/next_trial.sh (ros2 topic pub --once /real_cell/trial_go).
#
# What runs where:
#   remote  record_transfer_real.sh (headless Isaac Sim + tamp_server, one trial
#           per seed) -> score_real_trajectories.py --export. Detached from the
#           ssh session, so a dropped connection does not kill a 15-minute batch.
#   local   copy deliver/ back, check every meta was planned against the positions
#           given here, write the cell layout the replay gate reads, and print (or
#           run) the trajectory_replay launch.
#
# Scene (base_link metres). WHERE THE BEAKERS ARE comes from perception only
# (--perceive, required): no option or default gives a beaker's position.
#   --pan X,Y       the balance pan centre the target stands on (--flask is the same)
#                                                                            -> SDL_SCALE_PAN_XY
#   --beakers       the two-beaker cell: a second beaker, beaker2, stands at the
#                   centre of the pan instead of the flask   -> SDL_TARGET_VESSEL=beaker2
#                   beaker2 is planned, and poured into, where it was perceived. It
#                   may stand anywhere on the 130 mm pan with all of its 50 mm on
#                   it: within BEAKER2_ON_PAN_M (default 0.040) of the pan centre
#                   along x and along y.
#   The rim the pouring controller pours into is derived, never given: two-beaker
#   cell, beaker2's PERCEIVED bottom centre raised by its height (BEAKER2_HEIGHT_M,
#   0.070); flask cell, the pan centre at the flask's measured rim height
#   (FLASK_RIM_Z_M, 0.180, measured 2026-09-23). The controller has no receiver of
#   its own: a pour goal without one is refused. beaker1's perceived bottom centre
#   is the replay's grasp marker (replay_pour_grasp_marker).
#   --plate1 DEG, --plate2 DEG
#                   two-beaker cell: the bearing (base frame, from the vessel's
#                   axis) of each beaker's tag-mount plate, an obstacle beside
#                   its base -> SDL_TAG_PLATES. Unset: straight away from the robot.
#   --perceive      measure beaker1 and beaker2, with their plate bearings, with
#                   THIS machine's standing cameras, side_1 and side_2 fused by the
#                   pose node (running on cho_task_manager's pour_beakers.yaml).
#                   A beaker neither can see -- occluded, or out of view -- is
#                   found by OCCLUSION RECOVERY: cho_task_manager occlusion_recovery
#                   rasters the wrist camera over the bench (the arm moves; asked
#                   first) while measure_vessels.py --latch catches the pose.
#   --no-recover    fail instead of recovering
#   --recover NAME  re-measure NAME (beaker1, beaker2) by occlusion recovery even
#                   though the standing cameras -- or the pose node's HELD pose, kept
#                   while nothing moved the vessel -- report it: e.g. when side_1 and
#                   side_2 disagree on it and only one of them is publishing. Repeatable.
#   Only the vessels being recovered are swept for (a subset of CHO_SWEEP), and
#   only poses the wrist camera itself measured count (measure_vessels.py --camera
#   wrist, decode margin >= RECOVERY_MIN_MARGIN, the sweep table's 55).
# Omitted positions take record_transfer_real.sh's defaults, and the check below
# compares against those.
#
# Pour, for --replay: the recorded pour is handed to the pouring controller,
# which pours by weight on the balance:
#   --material liquid|granular   -> replay_pour_material (granular rolls on j6 with taps)
#   --grams G                    -> replay_pour_grams
#   --flow-index F               -> replay_pour_flow_index, 0 (water, dry sugar) .. 1
#   --pour-timeout S             -> replay_pour_timeout: the pour goal's timeout, and
#                                how long the task waits for it (0/unset: the
#                                controller's, 120 s liquid, 300 s granular). A
#                                thick liquid needs more: syrup at flow_index 1
#                                took 70 s for 10 g (2026-10-03)
# Without --material the recorded pour is replayed as recorded, weighing nothing.
#   --home-via moveit|direct     how the arm is taken to the recording's start pose,
#                                and to home at the start of a recovery sweep:
#                                moveit (default, collision-checked) needs
#                                bringup_real_moveit; direct is a straight joint move
# Before anything moves, check_replay.py builds the replay tree offline, so a
# layout mismatch or a pour that cannot be handed over is refused up front.
#
# Env:
#   SDL_PLAN_HOST    ssh destination (default 100.95.230.128; set User in ~/.ssh/config)
#   SDL_PLAN_ROOT    the sdl_project checkout on the remote; relative = under its $HOME
#                    (default sdl_ws/src/sdl_project)
#   REAL_CELL_RUNS   runs directory on BOTH sides; relative = under $HOME (default real_cell_runs)
#   CHO_WS           local ROS 2 workspace holding cho_task_manager (default ~/ros2_ws)
#   REPLAY_ARGS      extra launch args for the replay, e.g. "home_via:=direct"
#   BEAKER2_ON_PAN_M how far beaker2's centre may sit off the pan centre, along x
#                    and along y (0.040: pan half-width 65 mm less the beaker's 25)
#   BEAKER2_HEIGHT_M beaker2's height, bottom to rim (0.070)
#   FLASK_RIM_Z_M    the flask cell's rim height (0.180)
#   CHO_TABLE, CHO_SWEEP   the pose node's object table and the recovery raster
#                    (default: cho_task_manager's pour_beakers.yaml, fr5_bench_beakers.yaml)
#   RECOVERY_WAIT_S  longest a recovery may take (360)
#   RECOVERY_MIN_MARGIN  the wrist's decode margin a recovered pose needs (55)
#   ALLOW_COMMIT_MISMATCH=1   plan on a remote checkout that is not this HEAD
# Forwarded to the remote when set here: SDL_TABLE_Z_M SDL_BEAKER_RISER_M
#   SDL_BEAKER_RISER_XY_M SDL_BEAKER_RISER_Y_M SDL_RISER_XY SDL_SCALE_TOP_M
#   SDL_SCALE_BODY_BEARING_DEG SDL_GRIPPER_FLANGE_M
#   SDL_GRASP_BETA_MIN_DEG SDL_GRASP_BETA_MAX_DEG SDL_GRASP_H_FRAC PLAN_TIMEOUT
#   CUDA_VISIBLE_DEVICES
#
# Real-cell only, like record_transfer_real.sh: never a source of paper numbers.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${SDL_PLAN_HOST:-100.95.230.128}"
ROOT_R="${SDL_PLAN_ROOT:-sdl_ws/src/sdl_project}"
RUNS="${REAL_CELL_RUNS:-real_cell_runs}"
CHO_WS="${CHO_WS:-$HOME/ros2_ws}"
CHO_TP="${CHO_SRC:-$CHO_WS/src/cho_robot_project}/cho_task_manager/config"
CHO_TABLE="${CHO_TABLE:-$CHO_TP/perception/pour_beakers.yaml}"    # the pose node's table
CHO_SWEEP="${CHO_SWEEP:-$CHO_TP/sweep/fr5_bench_beakers.yaml}"    # the recovery raster
# Planned vs. requested position: both are the same numbers passed through the
# scene, so anything beyond rounding means the request did not reach the planner.
MATCH_TOL_M=0.002
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

q() { printf '%q ' "$@"; }
rpath() { case "$1" in /*) echo "$1" ;; *) echo "\$HOME/$1" ;; esac; }  # expanded remotely
# The stage this script is at, for monitor.sh's rviz status line (RUNS/STATUS),
# written whole and renamed so the monitor never reads half a line.
LOCAL_RUNS="$(eval echo "$(rpath "$RUNS")")"
status() {
  mkdir -p "$LOCAL_RUNS" && printf '%s  %s\n' "$(date +%H:%M:%S)" "$*" > "$LOCAL_RUNS/STATUS.tmp" \
    && mv -f "$LOCAL_RUNS/STATUS.tmp" "$LOCAL_RUNS/STATUS"
}
die() { echo "plan_remote: $*" >&2; status "STOPPED: ${1%%$'\n'*}"; exit 1; }

BEAKER="" BEAKER_Z="" FLASK="" BEAKER2="" BEAKER2_Z="" RECEIVER="" GRASP_MARKER=""
PLATE1="" PLATE2="" TARGET=flask PERCEIVE=0
MATERIAL="" GRAMS="" FLOW="" POUR_TIMEOUT="" HOME_VIA=moveit RECOVER=1
MODE=new STAMP="" REPLAY=0 YES=1 SEEDS=() FORCE_RECOVER=() USE_SEED= REPEAT=1 GRASP_ONLY=0 MEASURE_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --beaker|--beaker1|--beaker2|--receiver)
                die "$1: a beaker's position comes from perception only (--perceive)" ;;
    --flask|--pan)      FLASK="${2:?$1 X,Y}"; shift 2 ;;
    --beakers)  TARGET=beaker2; shift ;;
    --plate1)   PLATE1="${2:?--plate1 DEG}"; TARGET=beaker2; shift 2 ;;
    --plate2)   PLATE2="${2:?--plate2 DEG}"; TARGET=beaker2; shift 2 ;;
    --perceive) PERCEIVE=1; shift ;;
    --no-recover) RECOVER=0; shift ;;
    --recover)  FORCE_RECOVER+=("${2:?--recover beaker1|beaker2}"); shift 2 ;;
    --material) MATERIAL="${2:?--material liquid|granular}"; shift 2 ;;
    --grams)    GRAMS="${2:?--grams G}"; shift 2 ;;
    --flow-index) FLOW="${2:?--flow-index F}"; shift 2 ;;
    --pour-timeout) POUR_TIMEOUT="${2:?--pour-timeout S}"; shift 2
                [[ $POUR_TIMEOUT =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--pour-timeout must be seconds: $POUR_TIMEOUT" ;;
    --home-via) HOME_VIA="${2:?--home-via moveit|direct}"; shift 2
                [[ $HOME_VIA == moveit || $HOME_VIA == direct ]] || die "--home-via must be moveit or direct" ;;
    --attach) MODE=attach STAMP="${2:?--attach RUN_ID}"; shift 2 ;;
    --fetch)  MODE=fetch  STAMP="${2:?--fetch RUN_ID}"; shift 2 ;;
    --replay) REPLAY=1; shift ;;
    --grasp-only) GRASP_ONLY=1; shift ;;
    --measure-only) MEASURE_ONLY=1; shift ;;
    --repeat)   REPEAT="${2:?--repeat N}"; shift 2
                [[ $REPEAT =~ ^[1-9][0-9]*$ ]] || die "--repeat must be a whole number of pours, 1 or more: $REPEAT" ;;
    --use-seed) USE_SEED="${2:?--use-seed N}"; shift 2
                [[ $USE_SEED =~ ^[0-9]+$ ]] || die "--use-seed must be an integer: $USE_SEED" ;;
    --yes)    YES=1; shift ;;
    --confirm) YES=0; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    -*) die "unknown option $1" ;;
    *) [[ "$1" =~ ^[0-9]+$ ]] || die "seed must be an integer: $1"; SEEDS+=("$1"); shift ;;
  esac
done
case "$MATERIAL" in
  "") [[ -z "$GRAMS$FLOW" ]] || die "--grams/--flow-index need --material liquid|granular" ;;
  liquid|granular)
    python3 -c "import sys; sys.exit(not float(sys.argv[1]) > 0)" "$GRAMS" 2>/dev/null \
      || die "--material needs --grams G > 0" ;;
  *) die "--material must be liquid or granular (the pouring controller knows no other): $MATERIAL" ;;
esac
[[ -z "$FLOW" ]] || python3 -c "import sys; sys.exit(not 0 <= float(sys.argv[1]) <= 1)" "$FLOW" 2>/dev/null \
  || die "--flow-index must be in 0..1"

if [[ $MODE == new && $PERCEIVE == 1 ]]; then
  names=(beaker1); [[ $TARGET == beaker2 ]] && names+=(beaker2)
  ros_env() { set +u; source /opt/ros/humble/setup.bash && source "$CHO_WS/install/setup.bash"; }
  echo "=== perceiving ${names[*]} with the standing cameras (side_1 + side_2, fused)"
  status "perceiving ${names[*]} with side_1 + side_2 (fused)"
  measured="$(ros_env && python3 "$ROOT/scripts/real_cell/measure_vessels.py" "${names[@]}")"
  missing=()
  for n in "${names[@]}"; do grep -q "^$n " <<< "$measured" || missing+=("$n"); done
  for n in "${FORCE_RECOVER[@]}"; do
    [[ " ${names[*]} " == *" $n "* ]] || die "--recover $n: not a vessel of this cell (${names[*]})"
    if grep -q "^$n " <<< "$measured"; then
      echo "=== $n: $(grep "^$n " <<< "$measured") from the standing cameras (or held); re-measuring it (--recover)"
      measured="$(grep -v "^$n " <<< "$measured")"
      missing+=("$n")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    # Occluded, or out of both standing cameras' view: the wrist camera goes and
    # looks (cho_task_manager occlusion_recovery), on the pose node already
    # running, while measure_vessels latches each vessel the moment the wrist
    # holds it in view. This MOVES THE ARM.
    [[ $RECOVER == 1 ]] || die "not perceived: ${missing[*]} (and --no-recover was given)"
    echo "=== to recover with the wrist camera: ${missing[*]}"
    status "occlusion recovery: the wrist camera looks for ${missing[*]}"
    if [[ $YES != 1 ]]; then
      [[ -t 0 ]] || die "occlusion recovery moves the arm; run interactively or pass --yes"
      echo "The arm will move: to its home pose (a straight joint move), then a raster over"
      echo "the bench with the tool ~0.55 m up and the wrist camera looking down, then home."
      read -r -p "Area clear, e-stop in reach? [yes/N] " ans
      [[ "$ans" == yes ]] || die "occlusion recovery not confirmed; nothing moved"
    fi
    mkdir -p "$LOCAL_RUNS/live_logs"
    rstamp="$(date +%Y%m%d-%H%M%S)"
    rlog="$LOCAL_RUNS/live_logs/recovery_$rstamp.log"
    rout="$(mktemp)"
    # Sweep only for the vessels being recovered: the raster ends once they are
    # found, and it does not go looking for a vessel that is already measured.
    rsweep="$LOCAL_RUNS/live_logs/sweep_$rstamp.yaml"
    python3 - "$CHO_SWEEP" "$rsweep" "${missing[@]}" <<'PY' || die "could not write the recovery's sweep table"
import sys, yaml
src, dst, names = sys.argv[1], sys.argv[2], sys.argv[3:]
table = yaml.safe_load(open(src))
table["sweeps"] = [s for s in table["sweeps"] if s.get("object") in names]
missing = sorted(set(names) - {s["object"] for s in table["sweeps"]})
if missing:
    sys.exit("the sweep table %s has no sweep for %s" % (src, missing))
yaml.safe_dump(table, open(dst, "w"), sort_keys=False)
PY
    (ros_env && exec python3 "$ROOT/scripts/real_cell/measure_vessels.py" "${missing[@]}" \
        --latch --seconds "${RECOVERY_WAIT_S:-360}" \
        --camera wrist --min-margin "${RECOVERY_MIN_MARGIN:-55}") > "$rout" 2>> "$rlog" &
    mpid=$!
    (ros_env && exec timeout "${RECOVERY_WAIT_S:-360}" ros2 launch cho_task_manager \
        run_task_manager.launch.py task:=occlusion_recovery robot_type:=fr5 \
        "vessel_table:=$CHO_TABLE" "sweep_config:=$rsweep" "home_via:=$HOME_VIA") >> "$rlog" 2>&1
    echo "recovery tree: $(grep -a -o 'Task finished with status: [A-Za-z.]*' "$rlog" | tail -1)" \
      "(log $rlog)"
    for _ in $(seq 1 10); do kill -0 "$mpid" 2>/dev/null || break; sleep 1; done
    kill "$mpid" 2>/dev/null; wait "$mpid" 2>/dev/null
    grep -a -E '^(beaker1|beaker2): ' "$rlog" | tail -4
    measured="$measured"$'\n'"$(cat "$rout")"; rm -f "$rout"
    for n in "${missing[@]}"; do
      grep -q "^$n " <<< "$measured" || die "occlusion recovery did not find $n either (see $rlog)"
    done
  fi
  while read -r name x y z bearing; do
    case "$name" in
      beaker1) BEAKER="$x,$y"; BEAKER_Z="$z"; [[ -n "$PLATE1" ]] || PLATE1="$bearing" ;;
      beaker2) BEAKER2="$x,$y"; BEAKER2_Z="$z"; [[ -n "$PLATE2" ]] || PLATE2="$bearing" ;;
    esac
  done <<< "$(grep -E '^beaker[12] ' <<< "$measured")"
fi
if [[ $MEASURE_ONLY == 1 ]]; then
  [[ $PERCEIVE == 1 ]] || die "--measure-only measures: give --perceive"
  echo "=== measured: beaker1 ${BEAKER:-not seen} z ${BEAKER_Z:--} plate ${PLATE1:--} deg;" \
       "beaker2 ${BEAKER2:-not seen} z ${BEAKER2_Z:--} plate ${PLATE2:--} deg"
  status "measured, not planned (--measure-only)"
  exit 0
fi
if [[ $MODE == new ]]; then
  [[ $PERCEIVE == 1 ]] || die "the beakers' positions come from perception only: use --perceive"
  [[ -n "$BEAKER" ]] || die "beaker1 was not perceived; nothing stands in for it"
  [[ $TARGET != beaker2 || -n "$BEAKER2" ]] || die "beaker2 was not perceived; nothing stands in for it"
  # The replay grasps beaker1 where it was perceived, and measures the grasp there.
  GRASP_MARKER="$BEAKER,$BEAKER_Z"
fi
for v in "$BEAKER" "$FLASK" "$BEAKER2"; do
  [[ -z "$v" || "$v" =~ ^-?[0-9.]+,-?[0-9.]+$ ]] || die "position must be X,Y in metres: $v"
done
for v in "$PLATE1" "$PLATE2"; do
  [[ -z "$v" || "$v" =~ ^-?[0-9.]+$ ]] || die "a plate bearing is degrees: $v"
done
plates=""
[[ -n "$PLATE1" ]] && plates="beaker:$PLATE1"
[[ -n "$PLATE2" ]] && plates="${plates:+$plates;}beaker2:$PLATE2"
PAN="${FLASK:-0.520,0.320}"   # record_transfer_real.sh's default pan
if [[ -n "$BEAKER2" ]]; then
  # beaker2 is planned, and poured into, where it was perceived. It only has to
  # stand on the pan -- the balance weighs the pour -- which is square to the
  # robot base (SDL_SCALE_BODY_BEARING_DEG=180), so the pan's edges run along x
  # and y. A pose off the pan is a tag turned on its plate (220 mm out) or a
  # beaker that is not on the balance, and neither is poured into.
  python3 -c '
import sys
b = [float(v) for v in sys.argv[1].split(",")]; p = [float(v) for v in sys.argv[2].split(",")]
lim = float(sys.argv[3]); dx, dy = b[0] - p[0], b[1] - p[1]
print("beaker2 at (%.4f, %.4f): %+.1f / %+.1f mm from the pan centre (%.4f, %.4f) along x / y; "
      "on the pan within %.0f mm" % (b[0], b[1], dx * 1e3, dy * 1e3, p[0], p[1], lim * 1e3))
sys.exit(abs(dx) > lim or abs(dy) > lim)' "$BEAKER2" "$PAN" "${BEAKER2_ON_PAN_M:-0.040}" \
    || die "beaker2 is not standing on the balance pan (or its tag is turned on its plate)"
fi
# Where the pour goes: the receiver's rim centre, which the pouring controller
# takes from the goal and has no default for.
if [[ $MODE != new ]]; then
  # A re-joined run grasps and pours where it was planned to: the perceived
  # positions its request recorded.
  req="$(eval echo "$(rpath "$RUNS/$STAMP")")/REQUEST.env"
  RECEIVER="$(sed -n 's/^RECEIVER_RIM=//p' "$req" 2>/dev/null)"
  GRASP_MARKER="$(sed -n 's/^GRASP_MARKER=//p' "$req" 2>/dev/null)"
  grep -q '^SDL_TARGET_VESSEL=beaker2$' "$req" 2>/dev/null && TARGET=beaker2
fi
if [[ -n "$MATERIAL" && -z "$RECEIVER" ]]; then
  if [[ $TARGET == beaker2 ]]; then
    [[ -n "$BEAKER2" && -n "$BEAKER2_Z" ]] || die "the pour needs beaker2's rim, and only perception gives it
  (--perceive; for a re-joined run, its REQUEST.env RECEIVER_RIM)"
    RECEIVER="$(python3 -c 'import sys; x, y = sys.argv[1].split(","); z = float(sys.argv[2]) + float(sys.argv[3])
print("%.4f,%.4f,%.4f" % (float(x), float(y), z))' "$BEAKER2" "$BEAKER2_Z" "${BEAKER2_HEIGHT_M:-0.070}")"
    echo "pour into beaker2's rim at ($RECEIVER): perceived base z=$BEAKER2_Z + ${BEAKER2_HEIGHT_M:-0.070} m"
  else
    RECEIVER="$PAN,${FLASK_RIM_Z_M:-0.180}"
    echo "pour into the flask's rim at ($RECEIVER): the pan centre at the measured rim height"
  fi
fi
[[ -z "$RECEIVER" || "$RECEIVER" =~ ^-?[0-9.]+,-?[0-9.]+,-?[0-9.]+$ ]] \
  || die "the receiver rim must be X,Y,Z in metres: $RECEIVER"
[[ -z "$GRASP_MARKER" || "$GRASP_MARKER" =~ ^-?[0-9.]+,-?[0-9.]+,-?[0-9.]+$ ]] \
  || die "beaker1's grasp marker must be X,Y,Z in metres: $GRASP_MARKER"

# --- reach the planner ---------------------------------------------------------
"${SSH[@]}" "$HOST" true 2>/dev/null || die "cannot ssh to $HOST without a password.
  tailscale status          # is this machine on the tailnet, and is $HOST online?
  ssh-copy-id <user>@$HOST  # once, so BatchMode ssh works (then set User in ~/.ssh/config)"

if [[ $MODE == new ]]; then
  [[ ${#SEEDS[@]} -gt 0 ]] || die "give at least one seed"
  here="$(git -C "$ROOT" rev-parse HEAD)"
  there="$("${SSH[@]}" "$HOST" "git -C $(rpath "$ROOT_R") rev-parse HEAD" 2>/dev/null)" \
    || die "no sdl_project checkout at $ROOT_R on $HOST (set SDL_PLAN_ROOT)"
  if [[ "$here" != "$there" ]]; then
    msg="remote checkout is at ${there:0:9}, this one at ${here:0:9}. The trajectory
  depends on the planner, URDF and scene code, so plan with the code you replay with:
  git push, then on $HOST: git -C $ROOT_R pull"
    [[ "${ALLOW_COMMIT_MISMATCH:-0}" == 1 ]] && echo "WARNING: $msg" >&2 || die "$msg"
  fi
  dirty="$("${SSH[@]}" "$HOST" "git -C $(rpath "$ROOT_R") status --porcelain --untracked-files=no")"
  [[ -z "$dirty" ]] || echo "WARNING: the remote checkout has uncommitted changes:
$dirty" >&2
  "${SSH[@]}" "$HOST" "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader" \
    2>/dev/null | sed 's/^/remote GPU: /'

  envs=()
  layout=""
  [[ -n "$BEAKER" ]] && layout="beaker:$BEAKER"
  [[ -n "$BEAKER2" ]] && layout="${layout:+$layout;}beaker2:$BEAKER2"
  [[ -n "$layout" ]] && envs+=("SDL_LAYOUT_XY=$layout")
  [[ -n "$FLASK" ]] && envs+=("SDL_SCALE_PAN_XY=$FLASK")
  [[ $TARGET == beaker2 ]] && envs+=("SDL_TARGET_VESSEL=beaker2")
  [[ -n "$plates" ]] && envs+=("SDL_TAG_PLATES=$plates")
  for k in SDL_TABLE_Z_M SDL_BEAKER_RISER_M SDL_BEAKER_RISER_XY_M SDL_BEAKER_RISER_Y_M SDL_RISER_XY \
           SDL_SCALE_TOP_M SDL_SCALE_BODY_BEARING_DEG SDL_GRIPPER_FLANGE_M \
           SDL_GRASP_BETA_MIN_DEG SDL_GRASP_BETA_MAX_DEG \
           SDL_GRASP_H_FRAC PLAN_TIMEOUT CUDA_VISIBLE_DEVICES; do
    [[ -n "${!k:-}" ]] && envs+=("$k=${!k}")
  done
  STAMP="$(date +%Y%m%d-%H%M%S)"
  echo "=== planning on $HOST, run $STAMP, seeds ${SEEDS[*]}: ${envs[*]:-(record_transfer_real.sh defaults)}"
  status "planning on $HOST (Isaac Sim + cuTAMP), run $STAMP, seeds ${SEEDS[*]}"
  # The quoted heredoc is the remote script; its arguments are quoted with %q
  # because ssh joins its command line into one string for the remote shell.
  "${SSH[@]}" "$HOST" "bash -s -- $(q "$(rpath "$ROOT_R")" "$(rpath "$RUNS/$STAMP")" "${envs[@]}" -- "${SEEDS[@]}")" <<'REMOTE' \
    || die "could not start the remote run"
set -eo pipefail
root=$(eval echo "$1"); run=$(eval echo "$2"); shift 2
envs=(); while [[ "$1" != -- ]]; do envs+=("$1"); shift; done; shift
[[ -e "$run/PID" ]] && { echo "run directory already used: $run" >&2; exit 3; }
mkdir -p "$run" && cd "$root"
env "${envs[@]}" setsid nohup bash -c '
  run=$1; shift
  scripts/real_cell/record_transfer_real.sh "$run" "$@" &&
    python3 scripts/real_cell/score_real_trajectories.py "$run" --export
  echo $? > "$run/EXIT"' _ "$run" "$@" > "$run/remote.log" 2>&1 < /dev/null &
echo $! > "$run/PID"
REMOTE
fi
[[ -n "$STAMP" ]] || die "no run id"
RUN_R="$(rpath "$RUNS/$STAMP")"
LOCAL="$(eval echo "$(rpath "$RUNS/$STAMP")")"
mkdir -p "$LOCAL"
# What THIS side asked for, so the check below can tell a request that never
# reached the planner from one that did (SCENE_ENV.txt alone shows only the latter).
[[ $MODE == new ]] && printf '%s\n' "${envs[@]}" "HOST=$HOST" "RECEIVER_RIM=$RECEIVER" "GRASP_MARKER=$GRASP_MARKER" > "$LOCAL/REQUEST.env"

# --- follow it to the end ------------------------------------------------------
if [[ $MODE != fetch ]]; then
  hint="$0 --attach $STAMP"; [[ $REPLAY == 1 ]] && hint+=" --replay"
  trap 'echo; echo "detached; the remote run continues. Re-join with: $hint" >&2; exit 130' INT
  echo "=== following $HOST:$RUNS/$STAMP/remote.log (Ctrl-C only detaches; the run goes on)"
  status "planning on $HOST, run $STAMP: following its log"
  "${SSH[@]}" "$HOST" "run=$RUN_R; [ -e \$run/PID ] || { echo 'no such run' >&2; exit 3; }
    tail -n +1 -F \$run/remote.log 2>/dev/null & t=\$!
    while [ ! -e \$run/EXIT ] && kill -0 \$(cat \$run/PID) 2>/dev/null; do sleep 2; done
    sleep 1; kill \$t"
  rc=$?
  trap - INT
  [[ $rc -eq 3 ]] && die "no run $STAMP on $HOST"
  [[ $rc -eq 0 ]] || die "lost the connection; the remote run continues. Re-join with: $hint"
fi
code="$("${SSH[@]}" "$HOST" "cat $RUN_R/EXIT 2>/dev/null")"
[[ -n "$code" ]] || die "the remote run ended without writing EXIT (killed?); see $HOST:$RUNS/$STAMP/remote.log"
[[ "$code" == 0 ]] || die "the remote run failed (exit $code); see $HOST:$RUNS/$STAMP/remote.log"

# --- bring the trajectories home and check them --------------------------------
for f in remote.log SCENE_ENV.txt COMMIT transfer_real.csv deliver; do
  scp -q -r -o BatchMode=yes "$HOST:$RUNS/$STAMP/$f" "$LOCAL/" 2>/dev/null || true
done
compgen -G "$LOCAL/deliver/*.csv" >/dev/null || die "no trajectory was accepted; the score table is in $LOCAL/remote.log:
$(grep -A40 '^seed ' "$LOCAL/remote.log" 2>/dev/null | head -40)"

status "fetched run $STAMP; checking the trajectories against the request"
BEST="$(python3 - "$LOCAL" "$MATCH_TOL_M" "$USE_SEED" <<'PY'
"""Check each meta against the positions the run was asked to plan for, and write
the cell layout the replay gate compares it with."""
import glob, json, math, os, re, sys
run, tol = sys.argv[1], float(sys.argv[2])
def read_env(name):
    path = os.path.join(run, name)
    if not os.path.exists(path):
        return {}
    return dict(l.rstrip("\n").split("=", 1) for l in open(path) if "=" in l)
env, asked = read_env("SCENE_ENV.txt"), read_env("REQUEST.env")
for k in ("SDL_LAYOUT_XY", "SDL_SCALE_PAN_XY", "SDL_TARGET_VESSEL", "SDL_TAG_PLATES"):
    if k in asked and env.get(k) != asked[k]:
        sys.exit("the remote ran with %s=%r, but %r was asked for" % (k, env.get(k), asked[k]))
# The pour target stands at the pan centre: the flask, or beaker2 in the
# two-beaker cell.
target = env.get("SDL_TARGET_VESSEL", "flask") or "flask"
want = {}
for item in env.get("SDL_LAYOUT_XY", "").split(";"):
    name, _, xy = item.strip().partition(":")
    if name in ("beaker", "beaker2") and xy:
        want[name] = [float(v) for v in xy.split(",")[:2]]
if target not in want and env.get("SDL_SCALE_PAN_XY"):
    want[target] = [float(v) for v in env["SDL_SCALE_PAN_XY"].split(",")]
ok = []
for meta_path in sorted(glob.glob(os.path.join(run, "deliver", "*.meta.json"))):
    base = meta_path[:-len(".meta.json")]
    lay = json.load(open(meta_path)).get("layout_the_trajectory_assumes") or {}
    bad = []
    for name, xy in want.items():
        got = lay.get(name, {}).get("xy")
        if got is None or math.hypot(got[0] - xy[0], got[1] - xy[1]) > tol:
            bad.append("%s planned at %s, asked for %s" % (name, got, xy))
    if bad:
        print("REFUSED %s: %s" % (os.path.basename(base), "; ".join(bad)), file=sys.stderr)
        continue
    lines = ["# The cell layout %s.csv assumes, from its meta (run %s)."
             % (os.path.basename(base), os.path.basename(run)),
             "# beaker and %s are the positions sent to the planner; the rest are copied" % target,
             "# from the meta. The replay senses nothing: the beaker must stand ON its",
             "# riser, and the %s at the centre of the balance pan, exactly here." % target,
             "layout:"]
    for name in sorted(lay):
        x, y = lay[name]["xy"]
        if name in want:
            x, y = want[name]
        lines.append("  %s:\n    xy: [%.4f, %.4f]" % (name, x, y))
        if name not in ("beaker", "flask", "beaker2") and lay[name].get("yaw_deg") is not None:
            lines.append("    yaw_deg: %.1f" % lay[name]["yaw_deg"])
    # The pour the plan was made for, where cho_task_manager's beaker_transfer
    # reads it: beaker2's rim and beaker1's bottom centre as perception gave them
    # to the planner, and beaker1 set back where it was taken from.
    rim, marker = asked.get("RECEIVER_RIM", ""), asked.get("GRASP_MARKER", "")
    if target == "beaker2" and rim and marker:
        lines += ["# The pour this plan was made for (REQUEST.env): beaker2's rim centre and",
                  "# beaker1's bottom centre as perception measured them, base_link, m.",
                  "pour:",
                  "  receiver_rim: [%s]" % ", ".join(v.strip() for v in rim.split(",")),
                  "  grasp_marker: [%s]" % ", ".join(v.strip() for v in marker.split(",")),
                  "  put_back: start"]
    open(base + "_cell.yaml", "w").write("\n".join(lines) + "\n")
    ok.append(re.search(r"seed(\d+)$", base).group(1))
# score_real_trajectories.py prints the accepted seeds best (smallest lip error) first
m = re.search(r"best first: (.*)", open(os.path.join(run, "remote.log"), errors="ignore").read())
order = re.findall(r"seed (\d+)", m.group(1)) if m else []
best = next((s for s in order if s in ok), ok[0] if ok else "")
use = sys.argv[3] if len(sys.argv) > 3 else ""
if use:
    if use not in ok:
        sys.exit("--use-seed %s: not an accepted seed (accepted: %s)" % (use, ", ".join(ok) or "none"))
    if use != best:
        print("replaying seed %s as asked (--use-seed), not the best-scored seed %s" % (use, best),
              file=sys.stderr)
    best = use
print(best)
PY
)" || die "checking the trajectories failed"
[[ -n "$BEST" ]] || die "every trajectory was refused (see above); nothing to replay"

CSV="$LOCAL/deliver/traj_realcell_seed${BEST}.csv"
META="${CSV%.csv}.meta.json"; CELL="${CSV%.csv}_cell.yaml"
echo
echo "=== ready: $CSV"
echo "the physical objects must stand here (base_link, m):"
awk '/^  [a-z]/ { keep = /^  (beaker|beaker2|riser|flask|scale|tag_beaker|tag_beaker2):/ } keep' "$CELL"
# The two-beaker cell's own task when it pours: beaker_transfer reads the
# trajectory, the cell and the pour target from the plan's files.
TRANSFER=""
[[ -n "$MATERIAL" && $TARGET == beaker2 ]] && grep -q '^pour:' "$CELL" && TRANSFER=1
if [[ $GRASP_ONLY == 1 ]]; then
  [[ -z "$MATERIAL" && $REPEAT == 1 ]] || die "--grasp-only grasps and lets go; it pours nothing (no --material, --repeat)"
  LAUNCH=(ros2 launch cho_task_manager run_task_manager.launch.py task:=grasp_trial robot_type:=fr5
          "transfer_plan:=${CSV%.csv}")
  echo "grasp only: reach, close on beaker1, check, open, back off, home"
elif [[ -n "$TRANSFER" ]]; then
  LAUNCH=(ros2 launch cho_task_manager run_task_manager.launch.py task:=beaker_transfer robot_type:=fr5
          "transfer_plan:=${CSV%.csv}" "replay_pour_material:=$MATERIAL" "replay_pour_grams:=$GRAMS")
  [[ -n "$FLOW" ]] && LAUNCH+=("replay_pour_flow_index:=$FLOW")
  [[ -n "$POUR_TIMEOUT" ]] && LAUNCH+=("replay_pour_timeout:=$POUR_TIMEOUT")
  echo "pour: $GRAMS g of $MATERIAL by weight, handed to the pouring controller, into the rim at ($RECEIVER)"
else
LAUNCH=(ros2 launch cho_task_manager run_task_manager.launch.py task:=trajectory_replay robot_type:=fr5
        "replay_trajectory:=$CSV" "replay_meta:=$META" "replay_layout:=$CELL")
if [[ -n "$MATERIAL" ]]; then
  LAUNCH+=("replay_pour_material:=$MATERIAL" "replay_pour_grams:=$GRAMS"
           "replay_pour_receiver:=$RECEIVER")
  [[ -n "$GRASP_MARKER" ]] && LAUNCH+=("replay_pour_grasp_marker:=$GRASP_MARKER")
  # The two-beaker transfer sets beaker1 back where it took it from, so the pose
  # node holds the pose it had (the wrist's, if a recovery measured it) again
  # rather than believing whatever a standing camera makes of its tag next.
  PUT_BACK=elsewhere; [[ $TARGET == beaker2 ]] && PUT_BACK=start
  LAUNCH+=("replay_pour_put_back:=$PUT_BACK")
  [[ -n "$FLOW" ]] && LAUNCH+=("replay_pour_flow_index:=$FLOW")
  [[ -n "$POUR_TIMEOUT" ]] && LAUNCH+=("replay_pour_timeout:=$POUR_TIMEOUT")
  echo "pour: $GRAMS g of $MATERIAL by weight, handed to the pouring controller, into the rim at ($RECEIVER)"
else
  echo "pour: replayed as recorded (no --material), nothing is weighed"
fi
fi
LAUNCH+=("home_via:=$HOME_VIA" ${REPLAY_ARGS:-})
if [[ $REPEAT != 1 ]]; then
  [[ -n "$MATERIAL" ]] || die "--repeat $REPEAT repeats a pour, and there is none without --material"
  LAUNCH+=("replay_pour_repeat:=$REPEAT")
  echo "pour: $REPEAT trials, each after the first on scripts/real_cell/next_trial.sh"
fi
# Build the replay tree here, without ticking it: whatever cho_task_manager
# refuses at build time (layout gate, pour splice, velocity limits) is refused
# before anyone is asked to stand back from the arm.
check=(python3 "$ROOT/scripts/real_cell/check_replay.py" "${CSV%.csv}" --home-via "$HOME_VIA"
       --repeat "$REPEAT")
if [[ $GRASP_ONLY == 1 ]]; then
  check+=(--grasp-only)
elif [[ -n "$TRANSFER" ]]; then
  check+=(--transfer --grams "$GRAMS" --material "$MATERIAL" --flow-index "${FLOW:-0}")
elif [[ -n "$MATERIAL" ]]; then
  check+=(--grams "$GRAMS" --material "$MATERIAL" --flow-index "${FLOW:-0}"
          --receiver "$RECEIVER" --grasp-marker "$GRASP_MARKER"
          --put-back "$([[ $TARGET == beaker2 ]] && echo start || echo elsewhere)")
fi
(set +u; source /opt/ros/humble/setup.bash && source "$CHO_WS/install/setup.bash" && cd /tmp && "${check[@]}") \
  || die "the replay tree does not build for this trajectory (see above); nothing moved"
echo
echo "with the real bringup running (home_via:=moveit needs bringup_real_moveit; on"
echo "bringup_real_robot use --home-via direct), the scale relay and the pose node:"
echo "  $(q "${LAUNCH[@]}")"
# The trajectory monitor.sh shows from here on.
printf '%s\n' "${CSV%.csv}" > "$LOCAL_RUNS/CURRENT.tmp" && mv -f "$LOCAL_RUNS/CURRENT.tmp" "$LOCAL_RUNS/CURRENT"
status "ready: run $STAMP seed $BEST checked; waiting for the replay to be confirmed"
[[ $REPLAY == 1 ]] || { status "ready: run $STAMP seed $BEST (not replayed)"; exit 0; }

if [[ $YES != 1 ]]; then
  [[ -t 0 ]] || die "--replay moves the arm; confirm interactively or pass --yes"
  read -r -p "The arm will move. Vessels exactly as above, bringup up, area clear? [yes/N] " ans
  [[ "$ans" == yes ]] || die "not confirmed; nothing moved"
fi
# NOT set -u around these: the ROS setup scripts read unbound variables.
set +u
source /opt/ros/humble/setup.bash
source "$CHO_WS/install/setup.bash"
status "REPLAYING run $STAMP seed $BEST on the FR5${MATERIAL:+, pouring $GRAMS g of $MATERIAL by weight into the rim at ($RECEIVER)}"
"${LAUNCH[@]}"
rc=$?
status "replay of run $STAMP seed $BEST ended (exit $rc)"
exit $rc
