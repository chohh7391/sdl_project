#!/usr/bin/env python3
"""Recover the ground-truth XDL for each of the 100 test instructions.

test_data_gen.py builds each instruction from structured steps and then throws
the structure away, keeping only the joined sentences, so the test set ships
with no labels -- which is why the existing experiment can only measure whether
the generated XML is schema-valid, never whether it says what the instruction
said. A schema-valid but wrong procedure scores as a success there.

The labels are recoverable rather than guessable: the generator draws from a
closed vocabulary (4 vessels, 2 objects, 2 plates, 2 reagents, 3 volumes,
3 stir times, 3 temperatures) and a closed set of 23 sentence templates, so
inverting those templates reconstructs exactly the step the generator drew.
Every sentence must match exactly one template; anything unmatched or ambiguous
raises instead of being guessed at, and --check reports the tally so the
recovery can be trusted before it is used to score a model.

active="true" on HeatChill is not stated by any instruction -- it is required by
the validator's schema -- so it is emitted in the label and scored as a schema
field rather than an instruction-derived one.
"""
import argparse, json, os, re, sys

VESSELS = ["beaker_A", "beaker_B", "flask_A", "flask_B"]
OBJECTS = ["box_A", "bottle_A"]
PLATES = ["plate_A", "plate_B"]
REAGENTS = ["water", "ethanol"]
VOLUMES = ["10 mL", "25 mL", "50 mL"]
STIR_TIMES = ["30 s", "1 min", "2 min"]
TEMPS = ["0 C", "25 C", "60 C"]

def alt(vals):
    return "(?:%s)" % "|".join(re.escape(v) for v in vals)

V, O, P, R = alt(VESSELS), alt(OBJECTS), alt(PLATES), alt(REAGENTS)
VOL, T, TE = alt(VOLUMES), alt(STIR_TIMES), alt(TEMPS)

# (compiled pattern, operator, field names in group order) -- one entry per
# template in test_data_gen.py's nl_* functions.
TEMPLATES = []

def add(pat, op, fields):
    TEMPLATES.append((re.compile("^" + pat + "$"), op, fields))

for p in (r"Add (%s) of (%s) to (%s)\." % (VOL, R, V),
          r"Pour (%s) of (%s) into (%s)\." % (VOL, R, V),
          r"Introduce (%s) of (%s) into the (%s)\." % (VOL, R, V),
          r"Place (%s) of (%s) into (%s)\." % (VOL, R, V)):
    add(p, "Add", ("volume", "reagent", "vessel"))

for p in (r"Stir (%s) for (%s)\." % (V, T),
          r"Mix the contents of (%s) for (%s)\." % (V, T),
          r"Agitate (%s) for (%s)\." % (V, T),
          r"Stir the solution in (%s) for (%s)\." % (V, T)):
    add(p, "Stir", ("vessel", "time"))

for p in (r"Heat (%s) to (%s)\." % (V, TE),
          r"Set the temperature of (%s) to (%s)\." % (V, TE),
          r"Adjust (%s) to (%s)\." % (V, TE),
          r"Bring (%s) to (%s)\." % (V, TE)):
    add(p, "HeatChill", ("vessel", "temp"))

for p in (r"Transfer (%s) from (%s) to (%s)\." % (VOL, V, V),
          r"Pour (%s) from (%s) into (%s)\." % (VOL, V, V),
          r"Send (%s) from (%s) to (%s)\." % (VOL, V, V)):
    add(p, "Transfer", ("volume", "from_vessel", "to_vessel"))

for p in (r"Clean (%s)\." % V, r"Wash (%s)\." % V,
          r"Rinse and clean (%s)\." % V, r"Perform cleaning on (%s)\." % V):
    add(p, "CleanVessel", ("vessel",))

for p in (r"Move (%s) to (%s)\." % (O, P),
          r"Place (%s) on (%s)\." % (O, P),
          r"Relocate (%s) onto (%s)\." % (O, P),
          r"Put (%s) on top of (%s)\." % (O, P)):
    add(p, "Move", ("object", "place"))


def split_sentences(instruction):
    """Split on sentence boundaries. No vocabulary item contains a period."""
    parts = [s.strip() for s in re.split(r"(?<=\.)\s+", instruction.strip())]
    return [s for s in parts if s]


def parse_sentence(sent):
    hits = []
    for pat, op, fields in TEMPLATES:
        m = pat.match(sent)
        if m:
            step = {"op": op}
            step.update(dict(zip(fields, m.groups())))
            if op == "HeatChill":
                step["active"] = "true"
            hits.append(step)
    if not hits:
        raise ValueError("no template matches: %r" % sent)
    if len(hits) > 1:
        ops = {h["op"] for h in hits}
        if len(ops) > 1 or hits[0] != hits[1]:
            raise ValueError("ambiguous, %d templates match: %r" % (len(hits), sent))
    return hits[0]


def parse_instruction(instruction):
    return [parse_sentence(s) for s in split_sentences(instruction)]


ATTR_ORDER = {
    "Add": ("vessel", "reagent", "volume"),
    "Stir": ("vessel", "time"),
    "HeatChill": ("vessel", "temp", "active"),
    "Transfer": ("from_vessel", "to_vessel", "volume"),
    "CleanVessel": ("vessel",),
    "Move": ("object", "place"),
}


def steps_to_xml(steps):
    lines = ["<procedure>"]
    for s in steps:
        attrs = " ".join('%s="%s"' % (k, s[k]) for k in ATTR_ORDER[s["op"]])
        lines.append("  <%s %s />" % (s["op"], attrs))
    lines.append("</procedure>")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "test_data.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    data = json.load(open(a.data))
    instrs = data["test_instructions"]
    labels, ops, lens, failures = [], {}, {}, []
    for i, ins in enumerate(instrs):
        try:
            steps = parse_instruction(ins)
        except ValueError as exc:
            failures.append((i, str(exc)))
            continue
        labels.append({"index": i, "instruction": ins, "steps": steps,
                       "xml": steps_to_xml(steps)})
        lens[len(steps)] = lens.get(len(steps), 0) + 1
        for s in steps:
            ops[s["op"]] = ops.get(s["op"], 0) + 1

    print("instructions: %d   recovered: %d   unrecovered: %d"
          % (len(instrs), len(labels), len(failures)))
    for i, why in failures:
        print("  [%d] %s" % (i, why))
    print("steps per instruction: %s"
          % ", ".join("%d->%d" % kv for kv in sorted(lens.items())))
    print("operator counts: %s"
          % ", ".join("%s %d" % kv for kv in sorted(ops.items())))
    print("total steps: %d" % sum(len(l["steps"]) for l in labels))

    if a.check:
        # Every label must also pass the project's own validator, or the
        # recovery disagrees with the schema the model is asked to produce.
        sys.path.insert(0, os.path.dirname(os.path.abspath(a.data)))
        from validator import ProcedureValidator, ProcedureValidationError
        bad = 0
        for l in labels:
            try:
                ProcedureValidator().validate(l["xml"])
            except ProcedureValidationError as exc:
                bad += 1
                if bad <= 5:
                    print("  label %d fails the validator: %s" % (l["index"], exc))
        print("labels rejected by the validator: %d/%d" % (bad, len(labels)))

    if a.out:
        json.dump({"source": a.data, "labels": labels}, open(a.out, "w"), indent=1)
        print("wrote %s" % a.out)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
