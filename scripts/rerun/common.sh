# shellcheck shell=bash
# Shared by scripts/rerun/*.sh. SOURCED, not run. See RERUN.md at the repo root.
#
# Everything here exists so that a re-run on another machine produces numbers
# that can be defended: the scene is the paper's (not the real-cell one), the
# GPU was not shared, every batch says when it ran, and an interrupted campaign
# resumes without duplicating trials.

# NOT set -u: /opt/ros/humble/setup.bash reads unbound variables.
set -o pipefail

RR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # .../src/sdl_project
RR_WS="$(cd "$RR_ROOT/../.." && pwd)"                            # the colcon workspace
RR_ANALYSIS="$RR_ROOT/_2026__IEEE_Access/revision/analysis"

if [[ -z "${RERUN_TAG:-}" ]]; then
  echo "RERUN_TAG is not set. Pick one name per campaign and keep it, e.g." >&2
  echo "    export RERUN_TAG=rtx5080_$(date +%Y%m%d)" >&2
  exit 2
fi
RR_OUT="$RR_ANALYSIS/data/rerun/$RERUN_TAG"
mkdir -p "$RR_OUT/logs" "$RR_OUT/results"

rr_log()  { echo "[$(date '+%F %T')] $*" | tee -a "$RR_OUT/rerun.log"; }
rr_die()  { rr_log "ERROR: $*"; exit 1; }

# --- the scene the PAPER is evaluated in, and nothing else -------------------
# The real-cell scene (transfer_real: balance, riser, lowered bench, level grasp,
# measured layout) exists for replaying trajectories on the robot. It must never
# leak into a paper number, and every one of its switches is an SDL_* variable.
# So a re-run refuses to start if ANY SDL_* variable is set: the scripts set the
# few they need themselves, per call, and the code defaults describe the paper's
# scene.
rr_scene_guard() {
  local bad
  bad="$(compgen -e | grep -E '^SDL_' || true)"
  if [[ -n "$bad" ]]; then
    echo "Refusing to run: these SDL_* variables are set in the environment:" >&2
    for v in $bad; do echo "    $v=${!v}" >&2; done
    echo "The paper's scene is the code defaults. Unset them (or open a fresh shell)." >&2
    exit 3
  fi
}

# Lines the simulator prints ONLY when a real-cell switch is active. Checked in
# every batch's sim logs after the batch, so a scene change that got in some
# other way than an environment variable is still caught. The wrist camera is
# not among them: it has been part of the paper's scene since 09-30.
RR_REAL_SCENE_MARKERS='measured layout:|\[Task\] balance:|beaker riser|bench offset|flask stands on the pan'
rr_check_scene_logs() {
  local dir="$1" hits
  hits="$(grep -alE "$RR_REAL_SCENE_MARKERS" "$dir"/sim_seed*.log 2>/dev/null || true)"
  if [[ -n "$hits" ]]; then
    rr_log "SCENE CONTAMINATION: real-cell markers in: $hits"
    echo "$dir" >> "$RR_OUT/SCENE_CONTAMINATED"
    return 1
  fi
  return 0
}

# --- GPU exclusivity ----------------------------------------------------------
# A process on the GPU that is not part of this campaign. Ours are recognised by
# their command line or working directory being inside this workspace; anything
# unreadable counts as foreign, which errs on the side of flagging.
rr_gpu_foreign() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
  while IFS=, read -r pid mem; do
    pid="${pid// /}"; mem="${mem// /}"
    [[ -z "$pid" ]] && continue
    local cmd cwd
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
    cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null)"
    case "$cmd $cwd" in *"$RR_WS"*) continue ;; esac
    echo "pid=$pid mem=${mem}MiB cmd=${cmd:0:140}"
  done
}

# One sample every RERUN_MONITOR_S seconds for the whole campaign: GPU use, load
# average, and any foreign GPU process. 90_analyse.sh intersects it with the
# batch windows, so contention is found afterwards even if nobody was watching.
rr_monitor_start() {
  local pidf="$RR_OUT/.monitor.pid"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then return 0; fi
  local csvf="$RR_OUT/gpu_monitor.csv"
  [[ -s "$csvf" ]] || echo "epoch,gpu_util_pct,gpu_mem_mib,loadavg_1m,n_foreign,foreign" > "$csvf"
  (
    while :; do
      read -r util mem < <(nvidia-smi --query-gpu=utilization.gpu,memory.used \
                              --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')
      load="$(cut -d' ' -f1 /proc/loadavg)"
      f="$(rr_gpu_foreign)"
      n="$(printf '%s' "$f" | grep -c . || true)"
      printf '%s,%s,%s,%s,%s,"%s"\n' "$(date +%s)" "${util:-}" "${mem:-}" "$load" "$n" \
        "$(printf '%s' "$f" | tr '\n' '|' | tr '"' "'")" >> "$csvf"
      sleep "${RERUN_MONITOR_S:-30}"
    done
  ) > /dev/null 2>&1 &
  echo $! > "$pidf"
  disown 2>/dev/null || true
}
rr_monitor_stop() {
  local pidf="$RR_OUT/.monitor.pid"
  [[ -f "$pidf" ]] && kill "$(cat "$pidf")" 2>/dev/null
  rm -f "$pidf"
}

# --- batches ------------------------------------------------------------------
# name <TAB> start epoch <TAB> end epoch <TAB> status, one line per attempt.
rr_batch_begin() { RR_BATCH_T0="$(date +%s)"; rr_log "BEGIN $1"; }
rr_batch_end()   {
  printf '%s\t%s\t%s\t%s\n' "$1" "$RR_BATCH_T0" "$(date +%s)" "$2" >> "$RR_OUT/batches.tsv"
  rr_log "END   $1 ($2)"
}

# Seeds 0..n-1 not yet in a trial CSV, space separated (empty when complete).
rr_missing_seeds() {
  python3 - "$1" "$2" <<'PY'
import csv, os, sys
path, n = sys.argv[1], int(sys.argv[2])
have = set()
if os.path.exists(path) and os.path.getsize(path) > 0:
    have = {int(r["seed"]) for r in csv.DictReader(open(path)) if r.get("seed", "") != ""}
print(" ".join(str(s) for s in range(n) if s not in have))
PY
}

# Same, for one planner seed of a multi-stream PDDLStream CSV.
rr_missing_pddl_seeds() {
  python3 - "$1" "$2" "$3" <<'PY'
import csv, os, sys
path, ps, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
have = set()
if os.path.exists(path) and os.path.getsize(path) > 0:
    have = {int(r["seed"]) for r in csv.DictReader(open(path)) if r.get("planner_seed") == ps}
print(" ".join(str(s) for s in range(n) if s not in have))
PY
}

# A CSV must hold each seed exactly once; a duplicate means a resumed batch
# appended to a partial one and the analysis would double-count it.
rr_check_complete() {
  python3 - "$1" "$2" "${3:-}" <<'PY'
import csv, collections, sys
path, n, ps = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = list(csv.DictReader(open(path)))
if ps:
    rows = [r for r in rows if r.get("planner_seed") == ps]
c = collections.Counter(int(r["seed"]) for r in rows)
dup = sorted(s for s, k in c.items() if k > 1)
missing = sorted(set(range(n)) - set(c))
if dup or missing:
    print("INCOMPLETE %s%s: duplicates %s, missing %s" % (path, " ps=" + ps if ps else "", dup, missing))
    sys.exit(1)
PY
}

# conda `sdl` (py3.10): cuTAMP, cuRobo and the LLM stack.
rr_conda_sdl() {
  local base
  base="$(conda info --base 2>/dev/null)" || rr_die "conda not found"
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
  conda activate "${CONDA_SDL_ENV:-sdl}" || rr_die "conda env ${CONDA_SDL_ENV:-sdl} missing"
}
