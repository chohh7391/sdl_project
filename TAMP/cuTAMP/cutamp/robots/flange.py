"""REAL-CELL only: an adapter flange between the FR5's tool flange and the AG-95.

The real cell's gripper is bolted to an adapter flange (author, 2026-09-29:
"about 1.5 cm"), so it reaches that much further along the tool axis than the
shared fr5_ag95 model, which the paper's simulation uses unchanged.
SDL_GRIPPER_FLANGE_M (metres, default 0) moves the gripper out by that much
wherever the model is loaded -- here for cuRobo (URDF and collision spheres),
in isaacsim/scripts/standalone/task.py for the simulator -- and only when it
is set, which scripts/real_cell/record_transfer_real.sh does. The shared files
are never edited: a patched copy is written beside each, named for the offset.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ENV = "SDL_GRIPPER_FLANGE_M"
#: The joint that carries the gripper on the tool flange (tool0 -> gripper_base_link).
GRIPPER_JOINT = "gripper_base_joint"
#: The FR5 tool flange face in wrist3_link, where the gripper's spheres begin.
FLANGE_FACE_Z = 0.100
#: Longer than any adapter this cell will see; a typo in metres (15) is refused.
MAX_FLANGE_M = 0.05


def flange_m():
    """The adapter flange's length from SDL_GRIPPER_FLANGE_M, or 0."""
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return 0.0
    value = float(raw)
    if not 0.0 <= value <= MAX_FLANGE_M:
        raise ValueError("%s=%r: the flange is metres, 0..%.2f" % (ENV, raw, MAX_FLANGE_M))
    return value


def _tag(flange):
    return "flange_%gmm" % round(flange * 1e3, 3)


def patched_urdf(src, flange):
    """A copy of *src* with the gripper moved *flange* metres out along the tool axis.

    Written beside *src*, so package:// and relative mesh paths resolve as before.
    """
    src = Path(src)
    tree = ET.parse(src)
    joints = [j for j in tree.getroot().findall("joint") if j.get("name") == GRIPPER_JOINT]
    if len(joints) != 1:
        raise ValueError("%s: expected one joint %r" % (src, GRIPPER_JOINT))
    origin = joints[0].find("origin")
    if origin is None:
        origin = ET.SubElement(joints[0], "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    x, y, z = (float(v) for v in origin.get("xyz", "0 0 0").split())
    # tool0's z is the tool axis (the chain wrist3 -> flange -> tool0 turns and
    # turns back), so the gripper goes out along the joint's own z.
    origin.set("xyz", "%r %r %r" % (x, y, z + flange))
    dst = src.with_name("%s.%s%s" % (src.stem, _tag(flange), src.suffix))
    tree.write(dst)
    return dst


def patched_spheres(src, flange):
    """A copy of the sphere table *src* with the gripper's wrist3_link spheres moved out.

    The gripper body is covered by wrist3_link spheres past the flange face;
    they move out by *flange*, and the nearest one is also kept where it was, so
    the flange itself stays covered. The finger links' spheres ride their own
    links, which the URDF joint moves.
    """
    src = Path(src)
    table = yaml.safe_load(open(src))
    spheres = table["collision_spheres"]["wrist3_link"]
    past = [s for s in spheres if s["center"][2] >= FLANGE_FACE_Z]
    if not past:
        raise ValueError("%s: no wrist3_link sphere past the flange face" % src)
    keep = min(past, key=lambda s: s["center"][2])
    moved = [dict(s, center=[s["center"][0], s["center"][1], s["center"][2] + flange])
             for s in past]
    table["collision_spheres"]["wrist3_link"] = (
        [s for s in spheres if s["center"][2] < FLANGE_FACE_Z] + [dict(keep)] + moved)
    dst = src.with_name("%s.%s%s" % (src.stem, _tag(flange), src.suffix))
    with open(dst, "w") as f:
        yaml.safe_dump(table, f, sort_keys=False)
    return dst


def apply_to_curobo_cfg(cfg):
    """Point a cuRobo robot config at flange-patched copies, when a flange is set."""
    flange = flange_m()
    if flange == 0.0:
        return cfg
    kin = cfg["robot_cfg"]["kinematics"]
    urdf = Path(kin["external_asset_path"]) / kin["urdf_path"]
    spheres = Path(kin["external_robot_configs_path"]) / kin["collision_spheres"]
    kin["urdf_path"] = str(patched_urdf(urdf, flange).relative_to(kin["external_asset_path"]))
    kin["collision_spheres"] = str(
        patched_spheres(spheres, flange).relative_to(kin["external_robot_configs_path"]))
    print("[cutamp] %s=%g: gripper %g mm out on its adapter flange (%s, %s)"
          % (ENV, flange, flange * 1e3, kin["urdf_path"], kin["collision_spheres"]))
    return cfg
