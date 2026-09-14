"""Shared prompt for the Action Reasoner, and the geometry grounding option.

train.py and eval_action_reasoner.py each carried their own copy of the prompt.
They are the same string today, and they must stay the same or a model is
evaluated on a prompt it was not trained on, so both now import it from here.

AR_GEOMETRY=1 adds the width of every object the step mentions, and states the
two-finger gripper's opening as a number. Without it the tool rule is only
stated in words -- "objects exceeding gripper span" -- while the input carries
nothing but names, so the only thing a model can learn is the association
box/bottle -> vgc10. That is exactly what fails on an unseen object class:
holding bottle and box out of training takes main-tool accuracy from 100% to
79.6% and auxiliary-tool accuracy from 88.7% to 61.7%. With the widths present
the rule is computable from the input instead of memorised.

Widths are not invented. beaker 50 mm, flask 70 mm, box 108 mm and the stirrer
plate 180 mm are the cuboids cuTAMP itself plans with (TAMP/tamp/src/envs/
utils.py ENTITIES); bottle 108 mm is the bounding width of FluidBottle.usd; and
the 92 mm opening is measured between the ag95's finger tips in its own URDF.
"""
import os

# object class -> width in millimetres
OBJECT_WIDTH_MM = {
    "beaker": 50,
    "flask": 70,
    "box": 108,
    "bottle": 108,
    "plate": 180,
}
AG95_OPENING_MM = 92

_TOOL_RULES_PLAIN = """- dh3 (3-finger): Move, Stir for cylindrical vessels (beaker, flask, tube)
- ag95 (2-finger): All Transfer operations
- vgc10 (suction): Objects exceeding gripper span (box, plate, bottle)"""

_TOOL_RULES_GEOMETRIC = """- dh3 (3-finger): Move, Stir for cylindrical vessels
- ag95 (2-finger): All Transfer operations
- vgc10 (suction): objects wider than the two-finger opening of %d mm""" % AG95_OPENING_MM

_BODY = """
Given the current and next XDL steps, the obstacle status, and candidate grids,
output exactly four tokens separated by commas:
main_tool, need_rearrange, aux_tool, target_grid

Allowed values:
- main_tool: dh3, ag95, vgc10
- aux_tool: dh3, ag95, vgc10, None
- need_rearrange: True, False
- target_grid: grid ID (e.g., G5) or None
"""

_TAIL = """
Current XDL:
{current_xdl}

Next XDL:
{next_xdl}

Obstacle:
{obstacle_info}

Candidate Grids:
{candidate_grids}

Output:
"""


def geometry_enabled():
    return os.environ.get("AR_GEOMETRY", "0").strip() == "1"


def build_prompt(inst, geometry=None):
    if geometry is None:
        geometry = geometry_enabled()
    head = ("You are a tool and rearrangement planner for laboratory automation.\n"
            "\nTool Rules:\n")
    head += (_TOOL_RULES_GEOMETRIC if geometry else _TOOL_RULES_PLAIN) + "\n"
    body = _BODY
    tail = _TAIL
    if geometry:
        body += "\nObject Widths:\n{object_widths}\n"
    prompt = head + body + tail
    fields = dict(current_xdl=inst["current_xdl"], next_xdl=inst["next_xdl"],
                  obstacle_info=inst["obstacle_info"],
                  candidate_grids=inst["candidate_grids"])
    if geometry:
        fields["object_widths"] = inst.get("object_widths", "unknown")
    return prompt.format(**fields)
