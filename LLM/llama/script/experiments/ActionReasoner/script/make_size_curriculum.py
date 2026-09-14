#!/usr/bin/env python3
"""Teach the tool rule from size, so it can transfer to an unseen class.

Holding bottle and box out of training removes every item whose answer is
vgc10, so the model never emits that token and scores exactly 1 - (vgc10 share)
on the unseen-class level. Putting each object's width in the input does not
help, because there is still nothing connecting a width to that output.

This adds the missing supervision without reintroducing the held-out classes:
existing training items are copied with one object renamed to a SIZE VARIANT of
a class that is already seen, and only the output token that object determines
is changed. The labelling function was read off the data, not assumed -- across
all 3000 items aux_tool is a function of the obstacle's class (beaker/flask ->
dh3, bottle/box -> vgc10, no exceptions) and main_tool is ag95 for every
Transfer and otherwise a function of the target's class.

Two shortcuts are deliberately blocked:
  * the suffix carries no information -- wide and narrow variants draw from the
    same pool of suffixes, so "_W" cannot stand in for "wide"
  * the width is not a single magic number -- wide variants are drawn from
    100-130 mm and narrow ones from 40-65 mm, so the model has to compare
    against the 92 mm opening rather than memorise one value

At test time bottle and box are 108 mm under a class word never seen in
training, so a model that learned the comparison should transfer and a model
that memorised names should not.
"""
import argparse, collections, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_prompt import AG95_OPENING_MM

WIDE_MM = [100, 110, 120, 130]
NARROW_MM = [40, 55, 65]
SUFFIXES = ["U", "V", "W", "X"]
BASE_CLASSES = ["beaker", "flask"]          # classes that remain in training

OBJ_ATTR = re.compile(r'object="([a-z]+)(_[A-Z])?"')
OBSTACLE = re.compile(r"\b(beaker|flask|bottle|box)(_[A-Z])?\b")


def variant_name(rng, cls):
    return "%s_%s" % (cls, rng.choice(SUFFIXES))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-kind", type=int, default=120,
                    help="augmented items per (wide/narrow) x (main/aux) cell")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.train) if l.strip()]
    rng = random.Random(a.seed)
    out = [dict(r) for r in rows]
    made = collections.Counter()

    # --- aux_tool: rename the OBSTACLE to a size variant ---------------------
    aux_pool = [r for r in rows
                if r["output"].split(",")[1] == "True"
                and OBSTACLE.search(r["instruction"].get("obstacle_info") or "")]
    for wide in (True, False):
        widths = WIDE_MM if wide else NARROW_MM
        for _ in range(a.per_kind):
            if not aux_pool:
                break
            src = rng.choice(aux_pool)
            inst = json.loads(json.dumps(src["instruction"]))
            m = OBSTACLE.search(inst["obstacle_info"])
            cls = rng.choice(BASE_CLASSES)
            new = variant_name(rng, cls)
            w = rng.choice(widths)
            inst["obstacle_info"] = (inst["obstacle_info"][:m.start()] + new
                                    + inst["obstacle_info"][m.end():])
            inst["_width_overrides"] = {new: w}
            toks = src["output"].split(",")
            toks[2] = "vgc10" if w > AG95_OPENING_MM else "dh3"
            out.append({"instruction": inst, "output": ",".join(toks)})
            made["aux_wide" if wide else "aux_narrow"] += 1

    # --- main_tool: rename a MOVE target to a size variant --------------------
    main_pool = [r for r in rows
                 if (r["instruction"].get("current_xdl") or "").lstrip().startswith("<Move")
                 and OBJ_ATTR.search(r["instruction"]["current_xdl"])]
    for wide in (True, False):
        widths = WIDE_MM if wide else NARROW_MM
        for _ in range(a.per_kind):
            if not main_pool:
                break
            src = rng.choice(main_pool)
            inst = json.loads(json.dumps(src["instruction"]))
            m = OBJ_ATTR.search(inst["current_xdl"])
            cls = rng.choice(BASE_CLASSES)
            new = variant_name(rng, cls)
            w = rng.choice(widths)
            inst["current_xdl"] = (inst["current_xdl"][:m.start(1)] + new
                                   + inst["current_xdl"][m.end(2) if m.group(2) else m.end(1):])
            inst["_width_overrides"] = {new: w}
            toks = src["output"].split(",")
            toks[0] = "vgc10" if w > AG95_OPENING_MM else "dh3"
            out.append({"instruction": inst, "output": ",".join(toks)})
            made["main_wide" if wide else "main_narrow"] += 1

    with open(a.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("base %d + augmented %d = %d" % (len(rows), sum(made.values()), len(out)))
    print("  ", dict(made))
    tools = collections.Counter(r["output"].split(",")[0] for r in out)
    aux = collections.Counter(r["output"].split(",")[2] for r in out)
    print("   main tool distribution:", dict(tools))
    print("   aux  tool distribution:", dict(aux))
    return 0


if __name__ == "__main__":
    sys.exit(main())
