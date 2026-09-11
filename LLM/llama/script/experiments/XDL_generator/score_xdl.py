#!/usr/bin/env python3
"""Field-level scoring of generated XDL against the recovered ground truth.

The existing experiment reports only the validator pass rate, which asks
whether the generated XML is well formed and schema-legal -- not whether it
says what the instruction said. A procedure that drops a step, swaps two steps,
or writes 25 mL where the instruction said 50 mL can still pass. Reviewer 1's
concern is exactly that, so this scores the fields:

  operator   the tag at each position
  object     vessel / from_vessel / to_vessel / object / place
  reagent    the substance
  quantity   volume / time / temp -- the numbers
  order      whether the operator SEQUENCE matches, separated from whether the
             same operators were produced at all (a multiset match with a
             sequence mismatch is an ordering error, not a content error)

Field accuracies are computed over positionally aligned steps whose operator
matches, because comparing a Stir's time against an Add's volume would measure
nothing. Steps that have no counterpart (a length mismatch) are counted as
errors in the step-count and exact-match figures rather than being dropped.
"""
import argparse, json, os, sys
import xml.etree.ElementTree as ET

OBJECT_FIELDS = ("vessel", "from_vessel", "to_vessel", "object", "place")
REAGENT_FIELDS = ("reagent",)
QUANTITY_FIELDS = ("volume", "time", "temp")
SCHEMA_FIELDS = ("active",)


def parse_xml(text):
    """Steps from generated XML, or None if it will not parse."""
    try:
        root = ET.fromstring(text.strip())
    except ET.ParseError:
        return None
    if root.tag != "procedure":
        return None
    out = []
    for el in root:
        step = {"op": el.tag}
        step.update(dict(el.attrib))
        out.append(step)
    return out


def wilson(k, n, z=1.959964):
    import math
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h) * 100, min(1.0, c + h) * 100


def score(labels, preds):
    n = len(labels)
    tot = {
        "parsed": 0, "exact": 0, "step_count": 0,
        "op_seq": 0, "op_multiset": 0,
        "op_pos_ok": 0, "op_pos_tot": 0,
        "object_ok": 0, "object_tot": 0,
        "reagent_ok": 0, "reagent_tot": 0,
        "quantity_ok": 0, "quantity_tot": 0,
        "schema_ok": 0, "schema_tot": 0,
    }
    per_op = {}
    for lab in labels:
        gold = lab["steps"]
        pred_xml = preds.get(str(lab["index"]), preds.get(lab["index"]))
        pred = parse_xml(pred_xml) if pred_xml else None
        if pred is None:
            continue
        tot["parsed"] += 1
        if len(pred) == len(gold):
            tot["step_count"] += 1
        gops = [s["op"] for s in gold]
        pops = [s["op"] for s in pred]
        if gops == pops:
            tot["op_seq"] += 1
        if sorted(gops) == sorted(pops):
            tot["op_multiset"] += 1

        exact = len(pred) == len(gold)
        for i, g in enumerate(gold):
            tot["op_pos_tot"] += 1
            p = pred[i] if i < len(pred) else None
            if p is None:
                exact = False
                continue
            if p["op"] == g["op"]:
                tot["op_pos_ok"] += 1
            else:
                exact = False
                continue    # fields of a different operator are not comparable
            d = per_op.setdefault(g["op"], {"n": 0, "ok": 0})
            d["n"] += 1
            step_ok = True
            for k, v in g.items():
                if k == "op":
                    continue
                bucket = ("object" if k in OBJECT_FIELDS else
                          "reagent" if k in REAGENT_FIELDS else
                          "quantity" if k in QUANTITY_FIELDS else
                          "schema" if k in SCHEMA_FIELDS else None)
                if bucket is None:
                    continue
                tot[bucket + "_tot"] += 1
                if p.get(k) == v:
                    tot[bucket + "_ok"] += 1
                else:
                    step_ok = False
                    exact = False
            # an attribute the gold step does not have is still an error
            for k in p:
                if k != "op" and k not in g:
                    step_ok = False
                    exact = False
            if step_ok:
                d["ok"] += 1
        if exact:
            tot["exact"] += 1
    return n, tot, per_op


def pct(k, n):
    if n == 0:
        return "   n/a        "
    lo, hi = wilson(k, n)
    return "%5.1f%% [%.1f, %.1f]" % (100.0 * k / n, lo, hi)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--labels", default=os.path.join(here, "xdl_labels.json"))
    ap.add_argument("--preds", required=True,
                    help='JSON mapping instruction index -> generated XML')
    a = ap.parse_args()

    labels = json.load(open(a.labels))["labels"]
    preds = json.load(open(a.preds))
    preds = preds.get("generations", preds)
    n, tot, per_op = score(labels, preds)

    print("instructions scored: %d   generations parsed as <procedure>: %s"
          % (n, pct(tot["parsed"], n)))
    print()
    print("  exact procedure match   %s   (all steps, all fields)"
          % pct(tot["exact"], n))
    print("  step count correct      %s" % pct(tot["step_count"], n))
    print("  operator sequence       %s   (right operators in the right order)"
          % pct(tot["op_seq"], n))
    print("  operator multiset       %s   (right operators, order ignored)"
          % pct(tot["op_multiset"], n))
    print()
    print("  per-step operator       %s   (n=%d steps)"
          % (pct(tot["op_pos_ok"], tot["op_pos_tot"]), tot["op_pos_tot"]))
    print("  object fields           %s   (n=%d)"
          % (pct(tot["object_ok"], tot["object_tot"]), tot["object_tot"]))
    print("  reagent fields          %s   (n=%d)"
          % (pct(tot["reagent_ok"], tot["reagent_tot"]), tot["reagent_tot"]))
    print("  quantity fields         %s   (n=%d)"
          % (pct(tot["quantity_ok"], tot["quantity_tot"]), tot["quantity_tot"]))
    print("  schema-only fields      %s   (n=%d, HeatChill active)"
          % (pct(tot["schema_ok"], tot["schema_tot"]), tot["schema_tot"]))
    print()
    print("  per-operator step accuracy (all fields of that step correct):")
    for op in sorted(per_op):
        d = per_op[op]
        print("    %-12s %s   (n=%d)" % (op, pct(d["ok"], d["n"]), d["n"]))
    ordering_only = tot["op_multiset"] - tot["op_seq"]
    print("\n  ordering errors (right operators, wrong order): %d" % ordering_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
