#!/usr/bin/env bash
# Plan real-cell Transfer trajectories on a REMOTE GPU machine over SSH (e.g. over
# Tailscale), bring the accepted ones back, and replay one on the FR5 attached to
# THIS machine.
#
#   scripts/real_cell/plan_remote.sh --beaker 0.584,-0.298 --flask 0.520,0.320 0 1 2 3
#   scripts/real_cell/plan_remote.sh --perceive --beakers --material liquid --grams 20 4
#   scripts/real_cell/plan_remote.sh --attach 20260929-141203     # re-join after the ssh dropped
#   scripts/real_cell/plan_remote.sh --fetch  20260929-141203     # a finished run: copy + check only
#   add --replay to also launch trajectory_replay here (it asks before moving the arm)
#
# What runs where:
#   remote  record_transfer_real.sh (headless Isaac Sim + tamp_server, one trial
#           per seed) -> score_real_trajectories.py --export. Detached from the
#           ssh session, so a dropped connection does not kill a 15-minute batch.
#   local   copy deliver/ back, check every meta was planned against the positions
#           given here, write the cell layout the replay gate reads, and print (or
#           run) the trajectory_replay launch.
#
# Scene (base_link metres):
#   --beaker X,Y    the source beaker, on its riser (--beaker1 is the same)  -> SDL_LAYOUT_XY
#   --pan X,Y       the balance pan centre the target stands on (--flask is the same)
#                                                                            -> SDL_SCALE_PAN_XY
#   --beakers       the two-beaker cell: a second beaker, beaker2, stands at the
#                   centre of the pan instead of the flask   -> SDL_TARGET_VESSEL=beaker2
#   --beaker2 X,Y   where beaker2 was MEASURED (implies --beakers). It is not
#                   planned against: beaker2 is fixed at the pan centre, and a
#                   measurement more than BEAKER2_TOL_M (default 0.015) off it
#                   refuses the run -- re-centre the beaker, or move --pan.
#   --perceive      measure beaker1 (and beaker2) with THIS machine's cameras, from
#                   /perception/object_pose/beaker1|beaker2 (measure_vessels.py);
#                   needs the pose node running on the beakers table
# Omitted positions take record_transfer_real.sh's defaults, and the check below
# compares against those.
#
# Pour, for --replay: the recorded pour is handed to the pouring controller,
# which pours by weight on the balance:
#   --material liquid|granular   -> replay_pour_material (granular rolls on j6 with taps)
#   --grams G                    -> replay_pour_grams
#   --flow-index F               -> replay_pour_flow_index, 0 (water, dry sugar) .. 1
# Without --material the recorded pour is replayed as recorded, weighing nothing.
#   --home-via moveit|direct     how the arm is taken to the recording's start pose:
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
#   BEAKER2_TOL_M    how far a measured beaker2 may sit off the pan centre (0.015)
#   ALLOW_COMMIT_MISMATCH=1   plan on a remote checkout that is not this HEAD
# Forwarded to the remote when set here: SDL_TABLE_Z_M SDL_BEAKER_RISER_M
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
# Planned vs. requested position: both are the same numbers passed through the
# scene, so anything beyond rounding means the request did not reach the planner.
MATCH_TOL_M=0.002
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

die() { echo "plan_remote: $*" >&2; exit 1; }
q() { printf '%q ' "$@"; }
rpath() { case "$1" in /*) echo "$1" ;; *) echo "\$HOME/$1" ;; esac; }  # expanded remotely

BEAKER="" FLASK="" BEAKER2="" TARGET=flask PERCEIVE=0 MATERIAL="" GRAMS="" FLOW="" HOME_VIA=moveit
MODE=new STAMP="" REPLAY=0 YES=0 SEEDS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --beaker|--beaker1) BEAKER="${2:?$1 X,Y}"; shift 2 ;;
    --flask|--pan)      FLASK="${2:?$1 X,Y}"; shift 2 ;;
    --beaker2)  BEAKER2="${2:?--beaker2 X,Y}"; TARGET=beaker2; shift 2 ;;
    --beakers)  TARGET=beaker2; shift ;;
    --perceive) PERCEIVE=1; shift ;;
    --material) MATERIAL="${2:?--material liquid|granular}"; shift 2 ;;
    --grams)    GRAMS="${2:?--grams G}"; shift 2 ;;
    --flow-index) FLOW="${2:?--flow-index F}"; shift 2 ;;
    --home-via) HOME_VIA="${2:?--home-via moveit|direct}"; shift 2
                [[ $HOME_VIA == moveit || $HOME_VIA == direct ]] || die "--home-via must be moveit or direct" ;;
    --attach) MODE=attach STAMP="${2:?--attach RUN_ID}"; shift 2 ;;
    --fetch)  MODE=fetch  STAMP="${2:?--fetch RUN_ID}"; shift 2 ;;
    --replay) REPLAY=1; shift ;;
    --yes)    YES=1; shift ;;
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
  echo "=== measuring ${names[*]} with this machine's cameras"
  measured="$(set +u; source /opt/ros/humble/setup.bash && source "$CHO_WS/install/setup.bash" &&
              python3 "$ROOT/scripts/real_cell/measure_vessels.py" "${names[@]}")" \
    || die "perception did not give a stable pose for every vessel (see above)"
  while read -r name x y; do
    case "$name" in
      beaker1) [[ -z "$BEAKER" ]] || die "--perceive and --beaker both give beaker1"; BEAKER="$x,$y" ;;
      beaker2) [[ -z "$BEAKER2" ]] || die "--perceive and --beaker2 both give beaker2"; BEAKER2="$x,$y" ;;
    esac
  done <<< "$measured"
fi
for v in "$BEAKER" "$FLASK" "$BEAKER2"; do
  [[ -z "$v" || "$v" =~ ^-?[0-9.]+,-?[0-9.]+$ ]] || die "position must be X,Y in metres: $v"
done
if [[ -n "$BEAKER2" ]]; then
  # beaker2 is planned at the pan centre; the measurement only has to agree.
  pan="${FLASK:-0.520,0.320}"   # record_transfer_real.sh's default pan
  python3 -c '
import math, sys
b = [float(v) for v in sys.argv[1].split(",")]; p = [float(v) for v in sys.argv[2].split(",")]
tol = float(sys.argv[3]); d = math.hypot(b[0] - p[0], b[1] - p[1])
print("beaker2 measured at (%.4f, %.4f), %.1f mm from the pan centre (%.4f, %.4f); tolerance %.0f mm"
      % (b[0], b[1], d * 1e3, p[0], p[1], tol * 1e3))
sys.exit(d > tol)' "$BEAKER2" "$pan" "${BEAKER2_TOL_M:-0.015}" \
    || die "beaker2 is not at the pan centre the scene stands it on"
fi

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
  [[ -n "$BEAKER" ]] && envs+=("SDL_LAYOUT_XY=beaker:$BEAKER")
  [[ -n "$FLASK" ]] && envs+=("SDL_SCALE_PAN_XY=$FLASK")
  [[ $TARGET == beaker2 ]] && envs+=("SDL_TARGET_VESSEL=beaker2")
  for k in SDL_TABLE_Z_M SDL_BEAKER_RISER_M SDL_GRASP_BETA_MIN_DEG SDL_GRASP_BETA_MAX_DEG \
           SDL_GRASP_H_FRAC PLAN_TIMEOUT CUDA_VISIBLE_DEVICES; do
    [[ -n "${!k:-}" ]] && envs+=("$k=${!k}")
  done
  STAMP="$(date +%Y%m%d-%H%M%S)"
  echo "=== planning on $HOST, run $STAMP, seeds ${SEEDS[*]}: ${envs[*]:-(record_transfer_real.sh defaults)}"
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
[[ $MODE == new ]] && printf '%s\n' "${envs[@]}" "HOST=$HOST" > "$LOCAL/REQUEST.env"

# --- follow it to the end ------------------------------------------------------
if [[ $MODE != fetch ]]; then
  hint="$0 --attach $STAMP"; [[ $REPLAY == 1 ]] && hint+=" --replay"
  trap 'echo; echo "detached; the remote run continues. Re-join with: $hint" >&2; exit 130' INT
  echo "=== following $HOST:$RUNS/$STAMP/remote.log (Ctrl-C only detaches; the run goes on)"
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

BEST="$(python3 - "$LOCAL" "$MATCH_TOL_M" <<'PY'
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
for k in ("SDL_LAYOUT_XY", "SDL_SCALE_PAN_XY", "SDL_TARGET_VESSEL"):
    if k in asked and env.get(k) != asked[k]:
        sys.exit("the remote ran with %s=%r, but %r was asked for" % (k, env.get(k), asked[k]))
# The pour target stands at the pan centre: the flask, or beaker2 in the
# two-beaker cell.
target = env.get("SDL_TARGET_VESSEL", "flask") or "flask"
want = {}
for item in env.get("SDL_LAYOUT_XY", "").split(";"):
    if item.strip().startswith("beaker:"):
        want["beaker"] = [float(v) for v in item.split(":", 1)[1].split(",")[:2]]
if env.get("SDL_SCALE_PAN_XY"):
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
    open(base + "_cell.yaml", "w").write("\n".join(lines) + "\n")
    ok.append(re.search(r"seed(\d+)$", base).group(1))
# score_real_trajectories.py prints the accepted seeds best (smallest lip error) first
m = re.search(r"best first: (.*)", open(os.path.join(run, "remote.log"), errors="ignore").read())
order = re.findall(r"seed (\d+)", m.group(1)) if m else []
best = next((s for s in order if s in ok), ok[0] if ok else "")
print(best)
PY
)" || die "checking the trajectories failed"
[[ -n "$BEST" ]] || die "every trajectory was refused (see above); nothing to replay"

CSV="$LOCAL/deliver/traj_realcell_seed${BEST}.csv"
META="${CSV%.csv}.meta.json"; CELL="${CSV%.csv}_cell.yaml"
echo
echo "=== ready: $CSV"
echo "the physical objects must stand here (base_link, m):"
awk '/^  [a-z]/ { keep = /^  (beaker|beaker2|riser|flask|scale):/ } keep' "$CELL"
LAUNCH=(ros2 launch cho_task_manager run_task_manager.launch.py task:=trajectory_replay robot_type:=fr5
        "replay_trajectory:=$CSV" "replay_meta:=$META" "replay_layout:=$CELL")
if [[ -n "$MATERIAL" ]]; then
  LAUNCH+=("replay_pour_material:=$MATERIAL" "replay_pour_grams:=$GRAMS")
  [[ -n "$FLOW" ]] && LAUNCH+=("replay_pour_flow_index:=$FLOW")
  echo "pour: $GRAMS g of $MATERIAL by weight, handed to the pouring controller"
else
  echo "pour: replayed as recorded (no --material), nothing is weighed"
fi
LAUNCH+=("home_via:=$HOME_VIA" ${REPLAY_ARGS:-})
# Build the replay tree here, without ticking it: whatever cho_task_manager
# refuses at build time (layout gate, pour splice, velocity limits) is refused
# before anyone is asked to stand back from the arm.
check=(python3 "$ROOT/scripts/real_cell/check_replay.py" "${CSV%.csv}" --home-via "$HOME_VIA")
[[ -n "$MATERIAL" ]] && check+=(--grams "$GRAMS" --material "$MATERIAL" --flow-index "${FLOW:-0}")
(set +u; source /opt/ros/humble/setup.bash && source "$CHO_WS/install/setup.bash" && cd /tmp && "${check[@]}") \
  || die "the replay tree does not build for this trajectory (see above); nothing moved"
echo
echo "with the real bringup running (home_via:=moveit needs bringup_real_moveit; on"
echo "bringup_real_robot use --home-via direct), the scale relay and the pose node:"
echo "  $(q "${LAUNCH[@]}")"
[[ $REPLAY == 1 ]] || exit 0

if [[ $YES != 1 ]]; then
  [[ -t 0 ]] || die "--replay moves the arm; confirm interactively or pass --yes"
  read -r -p "The arm will move. Vessels exactly as above, bringup up, area clear? [yes/N] " ans
  [[ "$ans" == yes ]] || die "not confirmed; nothing moved"
fi
# NOT set -u around these: the ROS setup scripts read unbound variables.
set +u
source /opt/ros/humble/setup.bash
source "$CHO_WS/install/setup.bash"
exec "${LAUNCH[@]}"
