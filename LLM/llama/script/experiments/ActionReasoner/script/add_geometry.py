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
    text = " ".join(str(inst.get(k) or "") for k in
                    ("current_xdl", "next_xdl", "obstacle_info", "candidate_grids"))
    seen, out = set(), []
    for m in IDENT.finditer(text):
        name = m.group(0)
        cls = m.group(1)
        if name in seen or cls not in OBJECT_WIDTH_MM:
            continue
        seen.add(name)
        out.append("%s: %d mm" % (name, OBJECT_WIDTH_MM[cls]))
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
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print("%s -> %s : %d records, %d with no recognised object" % (a.src, a.dst, n, miss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
