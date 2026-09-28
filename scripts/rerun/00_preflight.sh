#!/usr/bin/env bash
# Pre-flight for a re-run campaign. Checks the machine, the scene and the GPU,
# and writes RUN_INFO.md. Nothing runs an experiment until this passes.
#
#   export RERUN_TAG=rtx5080_20261001
#   scripts/rerun/00_preflight.sh            # checks + RUN_INFO.md
#   scripts/rerun/00_preflight.sh --smoke    # ...and one Transfer trial end to end
#
# Re-running it on an existing tag is safe: it refuses if the code changed since
# the tag's first data, or if the declared budgets would change.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rr_scene_guard

SMOKE=0; [[ "${1:-}" == "--smoke" ]] && SMOKE=1
FAIL=0
ok()   { printf '  [ok]   %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; FAIL=1; }
warn() { printf '  [warn] %s\n' "$*"; }

echo "== re-run pre-flight: tag $RERUN_TAG"
echo "   repo $RR_ROOT"

# --- 1. where the repository sits ------------------------------------------------
echo "-- layout"
[[ "$(basename "$(dirname "$RR_ROOT")")" == "src" ]] \
  && ok "repo is <ws>/src/sdl_project (ws = $RR_WS)" \
  || bad "repo must be cloned as <ws>/src/sdl_project; the colcon overlay is looked up at <ws>/install"

# --- 2. the code under test ---------------------------------------------------------
echo "-- code"
COMMIT="$(git -C "$RR_ROOT" rev-parse HEAD)"
# Long-form pathspec: the short ':!' form misparses a path starting with '_',
# git then exits with an error, prints nothing on stdout, and an empty result
# would read as "clean". So a failing git is itself a failure here.
# Tracked files only: an untracked file (a log someone left, a cache) does
# not change the code under test, and counting it would block every resume.
if ! DIRTY="$(git -C "$RR_ROOT" status --porcelain --untracked-files=no -- . \
          ':(exclude)_2026__IEEE_Access/revision/analysis/data/rerun' 2>&1)"; then
  bad "git status failed: $DIRTY"
elif [[ -n "$DIRTY" ]]; then
  DIRTY="$(echo "$DIRTY" | head -5)"
  bad "uncommitted changes -- the commit hash would not describe the code that ran:"
  echo "$DIRTY" | sed 's/^/           /'
else
  ok "clean at $COMMIT"
fi
if [[ -f "$RR_OUT/COMMIT" && "$(cat "$RR_OUT/COMMIT")" != "$COMMIT" && -z "${RERUN_ALLOW_COMMIT_CHANGE:-}" ]]; then
  bad "this tag's data came from $(cat "$RR_OUT/COMMIT"); HEAD is now $COMMIT. Use a new RERUN_TAG."
fi

# --- 3. budgets, declared BEFORE any data exists ------------------------------------
echo "-- declared budgets"
BUDGETS="${RERUN_BUDGETS:-60,120,180}"
if [[ -f "$RR_OUT/BUDGETS" ]]; then
  [[ "$(cat "$RR_OUT/BUDGETS")" == "$BUDGETS" ]] \
    && ok "budgets $BUDGETS s (as declared)" \
    || bad "budgets were declared as $(cat "$RR_OUT/BUDGETS"); refusing to change them to $BUDGETS after the fact"
else
  ok "declaring budgets $BUDGETS s (fixed from now on for this tag)"
fi
MAXB="$(echo "$BUDGETS" | tr ',' '\n' | sort -n | tail -1)"
PDDLT="${RERUN_PDDL_MAX_TIME:-180}"
if python3 -c "import sys; sys.exit(0 if float('$PDDLT') >= float('$MAXB') else 1)"; then
  ok "PDDLStream limit ${PDDLT} s covers the largest budget ${MAXB} s"
else
  bad "PDDLStream is stopped at ${PDDLT} s but a budget of ${MAXB} s is declared -- it could not be censored fairly"
fi

# --- 4. the paper's scene, read from the code itself ---------------------------------
echo "-- scene (must be the paper's, not the real cell's)"
rr_conda_sdl
SCENE_OUT="$(cd "$RR_ROOT" && PYTHONPATH="$RR_ROOT/TAMP/tamp/src:$RR_ROOT/TAMP/cuTAMP" python - <<'PY' 2>&1
import math
from envs.constants import (glassware_set, vessel_dims, TABLE_Z_OFFSET,
                            BEAKER_RISER_M, vessel_height_override)
from envs.utils import ENTITIES
from orchestration.registry import get_environment_spec
from cutamp import samplers
checks = [
    ("glassware set", glassware_set(), "legacy"),
    ("beaker dims", vessel_dims("beaker"), [0.05, 0.05, 0.135]),
    ("flask dims", vessel_dims("flask"), [0.07, 0.07, 0.12]),
    ("beaker extra height", vessel_height_override("beaker"), 0.0),
    ("bench offset", TABLE_Z_OFFSET, 0.0),
    ("beaker riser", BEAKER_RISER_M, 0.0),
    ("table pose z", ENTITIES["table"].pose[2], -0.01),
    ("grasp beta min [deg]", round(math.degrees(samplers.BETA_MIN), 6), 10.0),
    ("grasp beta max [deg]", round(math.degrees(samplers.BETA_MAX), 6), 35.0),
    ("grasp height bias", samplers.GRASP_H_FRAC, 0.0),
    ("transfer entities", get_environment_spec("transfer").entities, ("beaker", "flask", "magnet")),
    ("transfer statics", get_environment_spec("transfer").statics,
     ("table", "goal_region", "stirrer", "magnet")),
]
bad = 0
for name, got, want in checks:
    same = got == want
    bad += not same
    print("%s %-22s %s%s" % ("ok  " if same else "FAIL", name, got, "" if same else "   (paper scene: %s)" % (want,)))
raise SystemExit(1 if bad else 0)
PY
)"
SCENE_RC=$?
echo "$SCENE_OUT" | sed 's/^ok  /  [ok]   /; s/^FAIL/  [FAIL]/'
[[ $SCENE_RC -eq 0 ]] || FAIL=1

# --- 5. environments -------------------------------------------------------------------
echo "-- environments"
python -c "import torch, curobo, cutamp; assert torch.cuda.is_available()" 2>/dev/null \
  && ok "conda $(basename "$CONDA_PREFIX"): torch $(python -c 'import torch;print(torch.__version__)'), cuRobo, cuTAMP, CUDA" \
  || bad "conda sdl: torch/curobo/cutamp import or CUDA failed"
python -c "import unsloth" >/dev/null 2>&1 \
  && ok "conda sdl: unsloth (LLM experiments)" || bad "conda sdl: unsloth import failed (LLM experiments)"
conda deactivate 2>/dev/null

[[ -x "$RR_ROOT/.venv-sdl/bin/python" ]] \
  && ok ".venv-sdl: $("$RR_ROOT/.venv-sdl/bin/python" -m pip show isaacsim 2>/dev/null | awk '/^Version/{print "isaacsim "$2}')" \
  || bad ".venv-sdl (Isaac Sim) missing"
bash "$RR_ROOT/scripts/run_isaacsim.sh" --check > "$RR_OUT/logs/preflight_isaac_check.log" 2>&1 \
  && ok "run_isaacsim.sh --check (py3.12 ROS interfaces)" \
  || bad "run_isaacsim.sh --check failed -- see logs/preflight_isaac_check.log"

if [[ -f "$RR_WS/install/setup.bash" ]]; then
  ( source /opt/ros/humble/setup.bash && source "$RR_WS/install/setup.bash" &&
    ros2 interface show tamp_interfaces/srv/Plan >/dev/null &&
    ros2 pkg prefix perception_manager >/dev/null && ros2 pkg prefix apriltag_ros >/dev/null ) 2>/dev/null \
    && ok "colcon overlay: tamp_interfaces, perception_manager, apriltag_ros" \
    || bad "colcon overlay at $RR_WS/install lacks tamp_interfaces / perception_manager / apriltag_ros"
else
  bad "no colcon overlay at $RR_WS/install"
fi

"$RR_ROOT/.venv-pddl/bin/python" -c "import pybullet" 2>/dev/null \
  && ok ".venv-pddl: pybullet" || bad ".venv-pddl with pybullet missing"
[[ -x "$RR_ROOT/experiments/pddlstream/downward/builds/release/bin/downward" ]] \
  && ok "Fast Downward built" || bad "Fast Downward not built (experiments/pddlstream/downward/build.py)"
[[ -f "$RR_ANALYSIS/data/seed_layouts.json" ]] \
  && ok "seed_layouts.json (the 30 layouts PDDLStream shares with Isaac Sim)" || bad "seed_layouts.json missing"
[[ -d "$RR_ROOT/third_party/LabUtopia" ]] \
  && ok "third_party/LabUtopia (scene assets)" || bad "third_party/LabUtopia missing"

XDL_ADAPTER="$RR_ROOT/LLM/llama/model/checkpoint/xdl_generator/checkpoint/adapter_model.safetensors"
XDL_SHA=2cf8d597f2cccd511fb08c0893db72394db2ef249f08e376171601f559935a74
if [[ -f "$XDL_ADAPTER" ]]; then
  [[ "$(sha256sum "$XDL_ADAPTER" | cut -d' ' -f1)" == "$XDL_SHA" ]] \
    && ok "XDL generator weights (sha256 matches the evaluated model)" \
    || bad "XDL generator weights present but NOT the evaluated model (sha256 differs)"
else
  bad "XDL generator weights missing: $XDL_ADAPTER (not in git -- see RERUN.md)"
fi
for d in model_split2400 model_unseen; do
  [[ -f "$RR_ROOT/LLM/llama/script/experiments/ActionReasoner/$d/checkpoint/adapter_model.safetensors" ]] \
    && ok "Action Reasoner $d" || bad "Action Reasoner $d checkpoint missing"
done

# --- 6. the GPU ---------------------------------------------------------------------------
echo "-- GPU"
GPU="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
[[ -n "$GPU" ]] && ok "$GPU" || bad "nvidia-smi failed"
[[ "$GPU" == *"${RERUN_EXPECT_GPU:-5080}"* ]] \
  || warn "GPU is not an RTX ${RERUN_EXPECT_GPU:-5080}; the paper states the GPU, so this run must state this one"
FOREIGN="$(rr_gpu_foreign)"
if [[ -n "$FOREIGN" ]]; then
  bad "other processes are on the GPU -- timings would be contended:"
  echo "$FOREIGN" | sed 's/^/           /'
else
  ok "no other compute process on the GPU"
fi

# --- verdict -------------------------------------------------------------------------------
if [[ $FAIL -ne 0 ]]; then
  echo "== pre-flight FAILED; nothing was recorded for this tag."
  exit 1
fi

# First successful pre-flight fixes the commit and the budgets for the tag.
[[ -f "$RR_OUT/COMMIT" ]] || echo "$COMMIT" > "$RR_OUT/COMMIT"
[[ -f "$RR_OUT/BUDGETS" ]] || echo "$BUDGETS" > "$RR_OUT/BUDGETS"

# --- RUN_INFO.md ---------------------------------------------------------------------------
rr_conda_sdl
TORCH="$(python -c 'import torch;print(torch.__version__, "CUDA", torch.version.cuda)' 2>/dev/null)"
CUROBO="$(python -c 'import curobo;print(getattr(curobo,"__version__","?"))' 2>/dev/null)"
conda deactivate 2>/dev/null
default_of() { grep -hoE "environ\.get\(\"$1\", \"[^\"]*\"\)" "$RR_ROOT"/TAMP/tamp/scripts/server/tamp_server.py \
                 "$RR_ROOT"/TAMP/cuTAMP/cutamp/*.py "$RR_ROOT"/TAMP/tamp/src/envs/*.py \
                 "$RR_ROOT"/isaacsim/scripts/standalone/*.py 2>/dev/null \
                 | head -1 | sed -E 's/.*, "([^"]*)"\)/\1/'; }
{
  echo "# RUN_INFO — re-run campaign \`$RERUN_TAG\`"
  echo
  echo "Written by \`scripts/rerun/00_preflight.sh\` on $(date '+%F %T %Z')."
  echo
  echo "| | |"
  echo "|---|---|"
  echo "| commit | \`$(cat "$RR_OUT/COMMIT")\` (branch \`$(git -C "$RR_ROOT" branch --show-current)\`) |"
  echo "| host / user | $(hostname) / $(whoami) |"
  echo "| GPU | $GPU |"
  echo "| CUDA (nvidia-smi) | $(nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' | head -1) |"
  echo "| CPU | $(LC_ALL=C lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2; exit}') ($(nproc) threads) |"
  echo "| RAM | $(LC_ALL=C free -g | awk '/^Mem:/{print $2" GB"}') |"
  echo "| OS | $(. /etc/os-release; echo "$PRETTY_NAME"), kernel $(uname -r) |"
  echo "| Isaac Sim | $("$RR_ROOT/.venv-sdl/bin/python" -m pip show isaacsim 2>/dev/null | awk '/^Version/{print $2}') |"
  echo "| torch (conda sdl) | $TORCH |"
  echo "| cuRobo | $CUROBO |"
  echo "| declared budgets | $(cat "$RR_OUT/BUDGETS") s |"
  echo "| cuTAMP repetitions | ${RERUN_REPS:-3} x 30 layouts per task |"
  echo "| PDDLStream | ${RERUN_PDDL_STREAMS:-5} planner seeds x 30 layouts, max ${RERUN_PDDL_MAX_TIME:-180} s |"
  echo "| GPU exclusive at start | yes (pre-flight found no other compute process) |"
  echo
  echo "## Behaviour switches, at their code defaults (no SDL_* variable was set)"
  echo
  echo "Pour continuations, the recovery ladder and the planning hold were added AFTER"
  echo "the 2026-09-12/13 campaign, so this run measures them and that campaign did not."
  echo "The other rows describe the same scene that campaign ran."
  echo
  echo "| switch | default | added |"
  echo "|---|---|---|"
  echo "| SDL_POUR_CONTINUATIONS | $(default_of SDL_POUR_CONTINUATIONS) | 09-14, re-seeded pour-path continuation |"
  echo "| SDL_RECOVERY | $(default_of SDL_RECOVERY) | 09-16, wait for a re-detection before a missed tag fails |"
  echo "| SDL_RECOVERY_RETREAT | $(default_of SDL_RECOVERY_RETREAT) | 09-16, retreat to home and look again |"
  echo "| SDL_RECOVERY_SCAN | $(default_of SDL_RECOVERY_SCAN) | off: needs the wrist camera |"
  echo "| SDL_PLAN_HOLD_SIM | $(default_of SDL_PLAN_HOLD_SIM) | 09-28, the simulator stops stepping (physics + rendering) while cuTAMP plans |"
  echo "| SDL_UPRIGHT_TRANSPORT | $(default_of SDL_UPRIGHT_TRANSPORT) | upright bound on carry segments |"
  echo "| SDL_TAG_MOUNT | $(default_of SDL_TAG_MOUNT) | 09-12, tags on a raised mount |"
  echo "| SDL_GLASSWARE | $(default_of SDL_GLASSWARE) | the paper's glassware |"
  echo
  echo "## Scene check"
  echo
  echo '```'
  echo "$SCENE_OUT"
  echo '```'
} > "$RR_OUT/RUN_INFO.md"
echo "== pre-flight passed; wrote $RR_OUT/RUN_INFO.md"

# --- optional: one Transfer trial end to end -------------------------------------------------
if [[ $SMOKE -eq 1 ]]; then
  echo "== smoke: one Transfer trial (seed 0), ground-truth state"
  mkdir -p "$RR_OUT/smoke"
  rm -f "$RR_OUT/smoke/transfer_smoke.csv"
  SDL_STATE_SOURCE=ground_truth CSV="$RR_OUT/smoke/transfer_smoke.csv" TASK=transfer \
    ROBOT=fr5_ag95 LOGDIR="$RR_OUT/logs/smoke" PLAN_TIMEOUT=600 \
    bash "$RR_ROOT/scripts/run_trials.sh" 0 > "$RR_OUT/logs/smoke.out" 2>&1
  python3 "$RR_ANALYSIS/summarize_trials.py" "$RR_OUT/smoke/transfer_smoke.csv" --budget 600 2>&1 \
    | grep -E "task success|planning success|time \[s\]"
  rr_check_scene_logs "$RR_OUT/logs/smoke" && echo "   scene markers: none (paper scene)"
fi
