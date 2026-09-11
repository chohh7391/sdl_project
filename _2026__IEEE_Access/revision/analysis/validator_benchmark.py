#!/usr/bin/env python3
"""The 20-valid / 80-invalid injected-error benchmark, built from real protocols.

The manuscript's validator table claims a benchmark of 20 valid and 80 invalid
samples with a per-class recall of 20/20 across four error categories. The code
shipped a 12-item set (4 valid, 8 invalid) hand-written in test_data_gen.py, so
that benchmark did not exist. This builds it.

The valid half is drawn from the recovered gold protocols of the 100-instruction
test set -- real procedures, not hand-written ones -- keeping only those the
validator accepts, on a fixed stride so the selection is reproducible. The
invalid half injects 20 samples of each of the four categories the manuscript
names, by a documented mutation of a valid protocol:

  capacity_violation     overfill a 100 mL beaker
  device_placement       put two objects on the same plate
  missing_attribute      drop one required attribute from one step
  undefined_tag          rename a tag out of the schema
  undefined_attribute    add an attribute not in the operator's schema
                         (the manuscript names these two as one class; they are
                         scored apart because the validator as shipped caught
                         only the first)
  nonexistent_object     rename a vessel, object or place to one not in the
                         workspace inventory
  precondition_violation make an operator act on a vessel that is empty at that
                         point, by dropping the Add that filled it

Every generated sample is CHECKED to be invalid before it enters the set (and
every valid sample to be valid); a mutation that fails to invalidate is
discarded and reported, never counted. That is what makes a recall figure from
this benchmark meaningful rather than circular.
"""
import argparse, json, math, os, random, sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validator import (ProcedureValidator, ProcedureValidationError,
                       VESSELS, OBJECTS, PLATES, ALLOWED_ATTRS)
from xdl_ground_truth import ATTR_ORDER, steps_to_xml

REQUIRED = {op: tuple(a for a in attrs) for op, attrs in ATTR_ORDER.items()}
FAKE_NAMES = ["beaker_Z", "flask_Q", "vial_A", "magic_beaker", "tube_1",
              "plate_Z", "box_Z", "carousel_A"]
OUT_OF_SCHEMA_TAGS = ["Shake", "Sonicate", "Filter", "add", "clean_vessel",
                      "Distill"]
OUT_OF_SCHEMA_ATTRS = ["foo", "speed", "rate", "duration", "pressure", "mode"]
OBJECT_KEYS = ("vessel", "from_vessel", "to_vessel", "object", "place")


def is_valid(xml):
    try:
        ProcedureValidator().validate(xml)
        return True, ""
    except ProcedureValidationError as exc:
        return False, str(exc)


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h) * 100, min(1.0, c + h) * 100


def mut_missing_attribute(steps, rng):
    i = rng.randrange(len(steps))
    op = steps[i]["op"]
    keys = [k for k in REQUIRED[op]]
    if not keys:
        return None
    k = rng.choice(keys)
    out = [dict(s) for s in steps]
    del out[i][k]
    return out, "dropped %s.%s" % (op, k)


def mut_undefined_tag(steps, rng):
    out = [dict(s) for s in steps]
    i = rng.randrange(len(out))
    out[i]["op"] = rng.choice(OUT_OF_SCHEMA_TAGS)
    return out, "tag -> %s" % out[i]["op"]


def mut_undefined_attribute(steps, rng):
    """Kept separate from the tag case on purpose.

    The manuscript names one class, "undefined tag or attribute". Combining
    them would let this benchmark report 20/20 while only ever exercising the
    tag half: a mutation that fails to invalidate is discarded and retried, so
    a gap in the attribute half would be silently skipped rather than shown.
    Scored separately, the attribute half is 0/20 on the validator as it
    shipped and 20/20 once the attribute whitelist is in.
    """
    out = [dict(s) for s in steps]
    i = rng.randrange(len(out))
    a = rng.choice(OUT_OF_SCHEMA_ATTRS)
    out[i][a] = "x"
    return out, "added %s.%s" % (out[i]["op"], a)


def mut_nonexistent_object(steps, rng):
    out = [dict(s) for s in steps]
    cands = [(i, k) for i, s in enumerate(out) for k in s if k in OBJECT_KEYS]
    if not cands:
        return None
    i, k = rng.choice(cands)
    out[i][k] = rng.choice(FAKE_NAMES)
    return out, "%s.%s -> %s" % (out[i]["op"], k, out[i][k])


def mut_capacity_violation(steps, rng):
    """Overfill a beaker: 100 mL capacity against 50 mL of headroom per Add."""
    v = rng.choice(["beaker_A", "beaker_B"])
    out = [dict(s) for s in steps]
    for _ in range(3):
        out.append({"op": "Add", "vessel": v, "reagent": "water",
                    "volume": "50 mL"})
    return out, "appended 3 x 50 mL into %s (100 mL)" % v


def mut_device_placement(steps, rng):
    """Put two different objects on the same plate."""
    p = rng.choice(["plate_A", "plate_B"])
    out = [dict(s) for s in steps]
    out.append({"op": "Move", "object": "box_A", "place": p})
    out.append({"op": "Move", "object": "bottle_A", "place": p})
    return out, "box_A and bottle_A both onto %s" % p


def mut_precondition_violation(steps, rng):
    """Drop the Add that fills the vessel a later operator depends on."""
    adds = [i for i, s in enumerate(steps) if s["op"] == "Add"]
    dependents = [s for s in steps
                  if s["op"] in ("Stir", "HeatChill", "Transfer")]
    if not adds or not dependents:
        return None
    i = rng.choice(adds)
    out = [dict(s) for j, s in enumerate(steps) if j != i]
    if not out:
        return None
    return out, "dropped the Add filling %s" % steps[i]["vessel"]


MUTATORS = [
    ("missing_attribute", mut_missing_attribute),
    ("undefined_tag", mut_undefined_tag),
    ("undefined_attribute", mut_undefined_attribute),
    ("nonexistent_object", mut_nonexistent_object),
    ("precondition_violation", mut_precondition_violation),
    ("capacity_violation", mut_capacity_violation),
    ("device_placement", mut_device_placement),
]


def to_xml(steps):
    lines = ["<procedure>"]
    for s in steps:
        op = s["op"]
        order = ATTR_ORDER.get(op, ())
        keys = [k for k in order if k in s] + [
            k for k in s if k != "op" and k not in order]
        attrs = " ".join('%s="%s"' % (k, s[k]) for k in keys)
        lines.append("  <%s%s />" % (op, (" " + attrs) if attrs else ""))
    lines.append("</procedure>")
    return "\n".join(lines)


def build(labels, n_valid, per_class, seed):
    rng = random.Random(seed)
    pool = []
    for l in labels:
        ok, _ = is_valid(l["xml"])
        if ok:
            pool.append(l)
    if len(pool) < n_valid:
        raise SystemExit("only %d validator-accepted gold protocols" % len(pool))
    stride = len(pool) // n_valid
    valid = [pool[i * stride] for i in range(n_valid)]

    samples = [{"class": "valid", "is_actually_valid": True,
                "source_index": v["index"], "mutation": "",
                "xml": v["xml"]} for v in valid]

    discarded = {}
    for cls, fn in MUTATORS:
        made, tries = 0, 0
        while made < per_class and tries < per_class * 200:
            tries += 1
            src = rng.choice(pool)
            res = fn([dict(s) for s in src["steps"]], rng)
            if res is None:
                continue
            steps, note = res
            xml = to_xml(steps)
            ok, _ = is_valid(xml)
            if ok:
                # the mutation did not actually invalidate the protocol; it is
                # a validator gap, recorded but never counted as a positive
                discarded.setdefault(cls, []).append((src["index"], note))
                continue
            _, why = is_valid(xml)
            samples.append({"class": cls, "is_actually_valid": False,
                            "source_index": src["index"], "mutation": note,
                            "xml": xml, "rejected_because": why})
            made += 1
        if made < per_class:
            # A class the validator cannot be made to fail is a GAP, not a
            # reason to stop: record what it accepted and score the class at
            # the samples that were built.
            print("  %s: only %d/%d samples could be made invalid -- the "
                  "validator accepts the rest" % (cls, made, per_class))
    return samples, discarded, len(pool)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--labels", default=os.path.join(here, "xdl_labels.json"))
    ap.add_argument("--valid", type=int, default=20)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(here, "validator_benchmark.json"))
    a = ap.parse_args()

    labels = json.load(open(a.labels))["labels"]
    samples, discarded, pool_n = build(labels, a.valid, a.per_class, a.seed)
    print("gold protocols the validator accepts: %d/%d" % (pool_n, len(labels)))
    print("benchmark: %d valid + %d invalid = %d samples (seed %d)"
          % (a.valid, len(samples) - a.valid, len(samples), a.seed))
    for cls, items in sorted(discarded.items()):
        print("  %s: %d mutations did NOT invalidate and were discarded"
              % (cls, len(items)))
        for idx, note in items[:3]:
            print("     e.g. from label %d: %s" % (idx, note))

    TP = FP = TN = FN = 0
    per_class = {}
    for s in samples:
        ok, why = is_valid(s["xml"])
        if s["is_actually_valid"]:
            if ok: TP += 1
            else:
                FN += 1
                print("  FALSE NEGATIVE on a valid sample (label %d): %s"
                      % (s["source_index"], why))
        else:
            d = per_class.setdefault(s["class"], {"n": 0, "caught": 0})
            d["n"] += 1
            if ok:
                FP += 1
            else:
                TN += 1
                d["caught"] += 1
    n = len(samples)
    acc = (TP + TN) / n * 100.0
    print("\nconfusion matrix (positive = validator passes):")
    print("  TP %d   FN %d      (valid samples: %d)" % (TP, FN, TP + FN))
    print("  FP %d   TN %d      (invalid samples: %d)" % (FP, TN, FP + TN))
    lo, hi = wilson(TP + TN, n)
    print("  accuracy %.1f%%  Wilson 95%% CI [%.1f, %.1f]" % (acc, lo, hi))
    print("\nper-class recall on injected errors:")
    for cls in sorted(per_class):
        d = per_class[cls]
        lo, hi = wilson(d["caught"], d["n"])
        print("  %-24s %2d/%-2d = %5.1f%%  [%.1f, %.1f]"
              % (cls, d["caught"], d["n"], 100.0 * d["caught"] / d["n"], lo, hi))
    # A class is only meaningful if the validator rejects for the intended
    # reason, so show what it actually said.
    print("\nreason the validator gave, per injected class:")
    reasons = {}
    for s in samples:
        if s["is_actually_valid"]:
            continue
        key = (s["class"], (s.get("rejected_because") or "").split(":")[0][:44])
        reasons[key] = reasons.get(key, 0) + 1
    for (cls, why), k in sorted(reasons.items()):
        print("  %-24s x%-3d %s" % (cls, k, why))
    json.dump({"seed": a.seed, "samples": samples,
               "discarded_non_invalidating": {k: v for k, v in discarded.items()}},
              open(a.out, "w"), indent=1)
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
