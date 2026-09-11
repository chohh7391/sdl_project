#!/usr/bin/env python3
"""Build the three held-out splits the manuscript already defines.

Section 4.3 states that the Action Reasoner's test scenarios are unseen at
three levels -- an object class absent from fine-tuning, a layout whose
occupancy pattern does not occur in training, and ordered operator pairs held
out entirely. None of the three holds for the shipped 80/20 random split: the
600 test items contain no unseen class, no unseen occupancy pattern and no
unseen operator pair. Rather than drop the definition, this builds the split it
describes.

  L1  unseen object class    every item mentioning bottle or box. All 247
                             training items answered vgc10 mention one of them,
                             so holding these out removes the suction tool from
                             supervision entirely: the model must infer it from
                             the tool rules in the prompt.
  L3  unseen operator pair   four ordered (current, next) operator pairs,
                             chosen among the mid-frequency ones so the level
                             is neither trivial nor degenerate
  L2  unseen layout          nine occupancy patterns, where a pattern is the
                             set of (object class, grid) the obstacle field
                             names -- relabelling identifiers does not make a
                             new pattern
  C   in-distribution        a random holdout from what remains, as the
                             baseline each level's drop is measured against

An item belonging to several families goes to the most specific (L1 > L3 > L2),
and training excludes every test item. The split is written only after it is
verified: training must contain no held-out class, pair or pattern, and no test
instruction may appear verbatim in training.
"""
import argparse, collections, json, os, random, re, sys

CLS = re.compile(r"\b(beaker|flask|tube|bottle|box|plate|tray)(?:_[A-Z])?\b")
UNSEEN_CLASSES = {"bottle", "box"}


def classes(rec):
    return set(CLS.findall(json.dumps(rec["instruction"])))


def occupancy(rec):
    s = rec["instruction"].get("obstacle_info") or ""
    return frozenset((c, g) for c, g in
                     re.findall(r"([a-zA-Z_]+?)(?:_[A-Z])?\s+at\s+(G\d+)", s))


def op_pair(rec):
    a = re.search(r"<(\w+)", rec["instruction"].get("current_xdl") or "")
    b = re.search(r"<(\w+)", rec["instruction"].get("next_xdl") or "")
    return (a.group(1) if a else "None", b.group(1) if b else "None")


def build(rows, n_pairs, n_patterns, n_control, seed):
    rng = random.Random(seed)
    pair_counts = collections.Counter(op_pair(r) for r in rows)
    pat_counts = collections.Counter(occupancy(r) for r in rows)

    pair_pool = sorted(p for p, c in pair_counts.items() if 20 <= c <= 200)
    held_pairs = set(rng.sample(pair_pool, n_pairs))
    pat_pool = sorted((p for p, c in pat_counts.items() if c >= 10),
                      key=lambda p: sorted(p))
    held_pats = set(rng.sample(pat_pool, n_patterns))

    L1, L3, L2, rest = [], [], [], []
    for i, r in enumerate(rows):
        if classes(r) & UNSEEN_CLASSES:
            L1.append(i)
        elif op_pair(r) in held_pairs:
            L3.append(i)
        elif occupancy(r) in held_pats:
            L2.append(i)
        else:
            rest.append(i)
    rng.shuffle(rest)
    control, train = rest[:n_control], rest[n_control:]
    return {"train": train, "L1_unseen_class": L1, "L3_unseen_pair": L3,
            "L2_unseen_layout": L2, "control_in_distribution": control}, \
        held_pairs, held_pats


def verify(rows, split, held_pairs, held_pats):
    ok = True
    tr = [rows[i] for i in split["train"]]
    bad = sum(1 for r in tr if classes(r) & UNSEEN_CLASSES)
    print("  training items mentioning a held-out class: %d" % bad); ok &= bad == 0
    bad = sum(1 for r in tr if op_pair(r) in held_pairs)
    print("  training items using a held-out operator pair: %d" % bad); ok &= bad == 0
    bad = sum(1 for r in tr if occupancy(r) in held_pats)
    print("  training items using a held-out occupancy pattern: %d" % bad); ok &= bad == 0
    vg = sum(1 for r in tr if r["output"].split(",")[0] == "vgc10")
    print("  training items answered vgc10 (suction): %d" % vg); ok &= vg == 0
    tr_keys = {json.dumps(r["instruction"], sort_keys=True) for r in tr}
    for name in ("L1_unseen_class", "L3_unseen_pair", "L2_unseen_layout",
                 "control_in_distribution"):
        dup = sum(1 for i in split[name]
                  if json.dumps(rows[i]["instruction"], sort_keys=True) in tr_keys)
        print("  %-24s verbatim in training: %d" % (name, dup)); ok &= dup == 0
    return ok


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data", default=os.path.join(
        here, "dataset", "action_reasoner_dataset.jsonl"))
    ap.add_argument("--outdir", default=os.path.join(here, "dataset", "unseen"))
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--patterns", type=int, default=9)
    ap.add_argument("--control", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    split, held_pairs, held_pats = build(rows, a.pairs, a.patterns, a.control,
                                         a.seed)
    print("source: %s (%d items)" % (a.data, len(rows)))
    print("held-out operator pairs: %s" % sorted(held_pairs))
    print("held-out occupancy patterns: %d" % len(held_pats))
    for k, v in split.items():
        tools = collections.Counter(rows[i]["output"].split(",")[0] for i in v)
        print("  %-24s %4d   main tools %s" % (k, len(v), dict(sorted(tools.items()))))
    print("\nverification:")
    if not verify(rows, split, held_pairs, held_pats):
        print("SPLIT REJECTED -- not written")
        return 1
    os.makedirs(a.outdir, exist_ok=True)
    for k, v in split.items():
        p = os.path.join(a.outdir, "%s.jsonl" % k)
        with open(p, "w") as fh:
            for i in v:
                fh.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        print("  wrote %s (%d)" % (p, len(v)))
    json.dump({"seed": a.seed,
               "held_out_pairs": sorted("%s>%s" % p for p in held_pairs),
               "held_out_patterns": [sorted("%s@%s" % t for t in p)
                                     for p in sorted(held_pats, key=lambda q: sorted(q))],
               "sizes": {k: len(v) for k, v in split.items()}},
              open(os.path.join(a.outdir, "split_manifest.json"), "w"), indent=1)
    print("  wrote %s" % os.path.join(a.outdir, "split_manifest.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
