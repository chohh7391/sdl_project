#!/usr/bin/env bash
# Plan real-cell Transfer trajectories on a REMOTE GPU machine over SSH (e.g. over
# Tailscale), bring the accepted ones back, and replay one on the FR5 attached to
# THIS machine.
#
#   scripts/real_cell/plan_remote.sh --beaker 0.584,-0.298 --flask 0.520,0.320 0 1 2 3
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
# --beaker X,Y   where the beaker stands (on its riser), base_link metres -> SDL_LAYOUT_XY
# --flask  X,Y   the balance pan centre the flask stands on               -> SDL_SCALE_PAN_XY
# Either may be omitted; record_transfer_real.sh's own defaults then apply and
# the check below compares against those defaults.
#
# Env:
#   SDL_PLAN_HOST    ssh destination (default 100.95.230.128; set User in ~/.ssh/config)
#   SDL_PLAN_ROOT    the sdl_project checkout on the remote; relative = under its $HOME
#                    (default sdl_ws/src/sdl_project)
#   REAL_CELL_RUNS   runs directory on BOTH sides; relative = under $HOME (default real_cell_runs)
#   CHO_WS           local ROS 2 workspace holding cho_task_manager (default ~/ros2_ws)
#   REPLAY_ARGS      extra launch args for the replay, e.g. "replay_pour_grams:=20"
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

BEAKER="" FLASK="" MODE=new STAMP="" REPLAY=0 YES=0 SEEDS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --beaker) BEAKER="${2:?--beaker X,Y}"; shift 2 ;;
    --flask)  FLASK="${2:?--flask X,Y}"; shift 2 ;;
    --attach) MODE=attach STAMP="${2:?--attach RUN_ID}"; shift 2 ;;
    --fetch)  MODE=fetch  STAMP="${2:?--fetch RUN_ID}"; shift 2 ;;
    --replay) REPLAY=1; shift ;;
    --yes)    YES=1; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    -*) die "unknown option $1" ;;
    *) [[ "$1" =~ ^[0-9]+$ ]] || die "seed must be an integer: $1"; SEEDS+=("$1"); shift ;;
  esac
done
for v in "$BEAKER" "$FLASK"; do
  [[ -z "$v" || "$v" =~ ^-?[0-9.]+,-?[0-9.]+$ ]] || die "position must be X,Y in metres: $v"
done

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
for k in ("SDL_LAYOUT_XY", "SDL_SCALE_PAN_XY"):
    if k in asked and env.get(k) != asked[k]:
        sys.exit("the remote ran with %s=%r, but %r was asked for" % (k, env.get(k), asked[k]))
want = {}
for item in env.get("SDL_LAYOUT_XY", "").split(";"):
    if item.strip().startswith("beaker:"):
        want["beaker"] = [float(v) for v in item.split(":", 1)[1].split(",")[:2]]
if env.get("SDL_SCALE_PAN_XY"):
    want["flask"] = [float(v) for v in env["SDL_SCALE_PAN_XY"].split(",")]
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
             "# beaker/flask are the positions measured on the rig and sent to the planner;",
             "# the rest are copied from the meta. The replay senses nothing: the beaker",
             "# must stand ON its riser, the flask on the balance pan, exactly here.",
             "layout:"]
    for name in sorted(lay):
        x, y = lay[name]["xy"]
        if name in want:
            x, y = want[name]
        lines.append("  %s:\n    xy: [%.4f, %.4f]" % (name, x, y))
        if name not in ("beaker", "flask") and lay[name].get("yaw_deg") is not None:
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
awk '/^  [a-z]/ { keep = /^  (beaker|riser|flask|scale):/ } keep' "$CELL"
LAUNCH=(ros2 launch cho_task_manager run_task_manager.launch.py task:=trajectory_replay robot_type:=fr5
        "replay_trajectory:=$CSV" "replay_meta:=$META" "replay_layout:=$CELL" ${REPLAY_ARGS:-})
echo
echo "with the real bringup running (ros2 launch cho_bringup_fr5 bringup_real_moveit.launch.py ...):"
echo "  $(q "${LAUNCH[@]}")"
[[ $REPLAY == 1 ]] || exit 0

if [[ $YES != 1 ]]; then
  [[ -t 0 ]] || die "--replay moves the arm; confirm interactively or pass --yes"
  read -r -p "The arm will move. Beaker on its riser and flask on the pan as above, bringup up, area clear? [yes/N] " ans
  [[ "$ans" == yes ]] || die "not confirmed; nothing moved"
fi
# NOT set -u around these: the ROS setup scripts read unbound variables.
set +u
source /opt/ros/humble/setup.bash
source "$CHO_WS/install/setup.bash"
exec "${LAUNCH[@]}"
