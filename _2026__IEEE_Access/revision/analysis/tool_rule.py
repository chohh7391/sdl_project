#!/usr/bin/env python3
"""Tool selection as a rule over measured width, not a learned association.

Read off the 3000-item set, the labelling function has no exceptions:
  main_tool = ag95                       if the step is a Transfer
            = vgc10                      if the target is wider than the
                                         two-finger opening (92 mm)
            = dh3                        otherwise
  aux_tool  = vgc10 / dh3 by the same width test on the obstacle,
            = None                       when no rearrangement is needed

The fine-tuned model cannot be made to apply that test. Giving it each object's
width changes nothing, and training the rule on wide variants of a seen class
still leaves it keyed on the class token: renaming the unseen classes to a seen
one, with the widths untouched, moves suction recall from 1/316 to 124/316.
So the rule is computed here from the width perception already measures, which
makes it exact for any object vocabulary, and the model is left with the parts
that are not rules -- whether to rearrange, and where to put the obstacle.
"""
import json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_prompt import OBJECT_WIDTH_MM, AG95_OPENING_MM

IDENT = re.compile(r"\b(beaker|flask|tube|bottle|box|plate|tray)(_[A-Z])?\b")


def widths_from_instruction(inst):
    """name -> width, taking the object_widths field when the prompt carries it."""
    out = {}
    for ln in (inst.get("object_widths") or "").split("\n"):
        if ":" in ln:
            name, rest = ln.split(":", 1)
            digits = "".join(c for c in rest if c.isdigit())
            if digits:
                out[name.strip()] = int(digits)
    return out


def _class_of(name):
    m = IDENT.match(name)
    return m.group(1) if m else None


def _width(name, widths):
    if name in widths:
        return widths[name]
    cls = _class_of(name)
    return OBJECT_WIDTH_MM.get(cls)


def main_tool(inst):
    xdl = inst.get("current_xdl") or ""
    if xdl.lstrip().startswith("<Transfer"):
        return "ag95"
    widths = widths_from_instruction(inst)
    m = re.search(r'object="([^"]+)"', xdl) or re.search(r'vessel="([^"]+)"', xdl)
    if not m:
        return "dh3"
    w = _width(m.group(1), widths)
    if w is None:
        return "dh3"
    return "vgc10" if w > AG95_OPENING_MM else "dh3"


def aux_tool(inst, need_rearrange):
    if not need_rearrange:
        return "None"
    widths = widths_from_instruction(inst)
    m = IDENT.search(inst.get("obstacle_info") or "")
    if not m:
        return "None"
    w = _width(m.group(0), widths)
    if w is None:
        return "dh3"
    return "vgc10" if w > AG95_OPENING_MM else "dh3"


if __name__ == "__main__":
    path = sys.argv[1]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    mt = sum(1 for r in rows if main_tool(r["instruction"]) == r["output"].split(",")[0])
    ax = sum(1 for r in rows
             if aux_tool(r["instruction"], r["output"].split(",")[1] == "True")
             == r["output"].split(",")[2])
    print("%-52s n=%-5d main %4d/%d = %6.2f%%   aux %4d/%d = %6.2f%%"
          % (os.path.basename(path), n, mt, n, 100.0 * mt / n, ax, n, 100.0 * ax / n))
