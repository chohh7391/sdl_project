#!/usr/bin/env python3
"""Add an object_widths field to a split, so the tool rule can be computed.

Every object identifier the step mentions gets the width of its class. Nothing
about the scene or the answer is added -- only a property of the object that
perception measures anyway -- so this grounds the rule rather than leaking it.
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_prompt import OBJECT_WIDTH_MM

IDENT = re.compile(r"\b(beaker|flask|tube|bottle|box|plate|tray)(_[A-Z])?\b")


def widths_for(inst):
    """Widths of every object the step mentions.

    A size-curriculum item carries _width_overrides, giving the width of the
    variant it introduced; that variant's class word is one already in the
    table, so without the override it would report the class's nominal width
    and the lesson would be lost.
    """
    text = " ".join(str(inst.get(k) or "") for k in
                    ("current_xdl", "next_xdl", "obstacle_info", "candidate_grids"))
    overrides = inst.get("_width_overrides") or {}
    seen, out = set(), []
    for m in IDENT.finditer(text):
        name = m.group(0)
        cls = m.group(1)
        if name in seen:
            continue
        if name in overrides:
            width = overrides[name]
        elif cls in OBJECT_WIDTH_MM:
            width = OBJECT_WIDTH_MM[cls]
        else:
            continue
        seen.add(name)
        out.append("%s: %d mm" % (name, width))
    return "\n".join(out) if out else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    a = ap.parse_args()
    n = miss = 0
    with open(a.src) as fh, open(a.dst, "w") as out:
        for ln in fh:
            if not ln.strip():
                continue
            r = json.loads(ln)
            w = widths_for(r["instruction"])
            if w == "none":
                miss += 1
            r["instruction"]["object_widths"] = w
            r["instruction"].pop("_width_overrides", None)
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print("%s -> %s : %d records, %d with no recognised object" % (a.src, a.dst, n, miss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
