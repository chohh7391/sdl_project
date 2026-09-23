#!/usr/bin/env python3
"""Plan an environment OFFLINE: the TAMP server's own planner, no simulator.

Exercises exactly the path a trial takes once the scene is up --
update_env -> load_env(ENV) -> run_cutamp -- against the layout a real sim log
says the scene had. A failure costs one planning call instead of a five-minute
trial, the planner's world is printed so a wrong pose is visible before it is
planned against, and the constraint table in the output names the constraint
that is binding.

    # needs the same environment the server runs in
    source /opt/ros/humble/setup.bash && source <ws>/install/setup.bash
    conda activate sdl
    SDL_GLASSWARE=real SDL_TABLE_Z_M=-0.013 SDL_BEAKER_RISER_M=0.05 \
      scripts/trials/offline_plan.py scripts/trials/logs/sim_seed6_*.log

ENV (default transfer_real) and ATTEMPTS (default 1) are environment variables.
Poses are rebuilt from the sim log's "effective layout" block, i.e. what the
simulator placed, not what it settled to; that is the right input for a
planning question and not a substitute for a trial.
"""
import math, os, re, sys, time
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, SRC + "/TAMP/tamp/scripts/server")
sys.path.insert(0, SRC + "/TAMP/tamp/src")
sys.path.insert(0, SRC + "/TAMP/cuTAMP")

log = sys.argv[1]
txt = open(log, errors="ignore").read()
# the LAST block of each object is the effective layout
lay = {}
for m in re.finditer(r"\[Task\]\s+(beaker|flask|magnet|box|stirrer)\s+xy=\(([-\d.]+),([-\d.]+)\)\s+"
                     r"yaw=([-\d.]+)deg\s+z=([-\d.]+)", txt):
    lay[m.group(1)] = [float(m.group(i)) for i in (2, 3, 5, 4)]   # x, y, z, yaw
for m in re.finditer(r"\[Task\]\s+(scale|riser)\s+xy=\(([-\d.]+),([-\d.]+)\)\s+yaw=([-\d.]+)deg\s+"
                     r"dims=\[([-\d.,\s]+)\]", txt):
    dims = [float(v) for v in m.group(5).split(",")]
    dz = float(os.environ.get("SDL_TABLE_Z_M", "0"))
    lay[m.group(1)] = [float(m.group(2)), float(m.group(3)), dz + dims[2] / 2.0, float(m.group(4))]
home = [float(v) for v in re.search(r"home_arm\(rad\) = \[([-\d.,\s]+)\]", txt).group(1).split(",")]

def pose(x, y, z, yaw_deg):
    h = math.radians(yaw_deg) / 2.0
    return [x, y, z, math.cos(h), 0.0, 0.0, math.sin(h)]

from orchestration.registry import get_environment_spec
from tamp_server import TAMP, default_config
from dataclasses import replace

ENV = os.environ.get("ENV", "transfer_real")
spec = get_environment_spec(ENV)
poses = {n: pose(*lay[n]) for n in spec.entities if n in lay}
print("layout:")
for n, p in poses.items():
    print("   %-8s xyz=(%.4f, %.4f, %.4f)" % (n, p[0], p[1], p[2]))
print("q_init:", home)

cfg = replace(default_config(), robot="fr5_ag95", grasp_dof=6)
tamp = TAMP(config=cfg, use_tetris_tuned_weights=None)
tamp.max_attempts = int(os.environ.get("ATTEMPTS", "1"))
tamp.update_env(name=ENV, poses=poses, movables=list(spec.movables),
                statics=list(spec.statics), ex_collision=list(spec.ex_collision))

print("\nplanner world (what the collision checker sees):")
for kind, objs in (("movable", tamp.env.movables), ("static", tamp.env.statics),
                   ("ex_coll", tamp.env.ex_collision)):
    for o in objs:
        print("   %-8s %-12s xyz=(%+.4f, %+.4f, %+.4f) dims=%s"
              % (kind, o.name, o.pose[0], o.pose[1], o.pose[2],
                 [round(float(v), 4) for v in o.dims]))
t0 = time.time()
res = tamp.plan(home, None)
print("\nplanned in %.1f s: success=%s satisfying=%s attempts=%s reason=%s"
      % (time.time() - t0, res.success, res.total_num_satisfying, res.attempts,
         res.failure_reason or "none"))
