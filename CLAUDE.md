# CLAUDE.md — IEEE Access revision guardrails

This repo hosts the **revision** of IEEE Access manuscript **Access-2026-33062**,
"LLM-Guided Tool-Aware Task and Motion Planning for Chemistry Lab Automation."
The paper was rejected with an invitation to resubmit (IEEE Access binary review).
**We get exactly one resubmission** — every concern from all three reviewers must be
addressed, or the paper is rejected with no further chance. Do not drift from that goal.

Read `PLAN.md` at the start of every session; it is the master checklist and the
experiment catalog. Keep its checkboxes current as items are completed.

---

## Revision strategy (decided with the author — do not silently change)

1. **Real-hardware effort is limited to the adaptive-pouring demo.** One physical session:
   an RS232 electronic scale under the target vessel, real closed-loop pouring on the FR5,
   filmed. Real AprilTag perception snapshots (localization + occlusion) are captured on that
   same rig while it is set up, at low extra cost. **No other experiment runs on real hardware**
   — not the full 20-trial end-to-end Transfer, not the spillage sweep at every tilt.
2. **AprilTag perception must be exercised *in the loop in Isaac Sim* (rendered cameras →
   detection → fusion → World State), not read from simulator ground truth.** This is the
   real fix for Reviewer 1's Concern #1. End-to-end and perception-accuracy numbers come from
   this rendered, perception-in-the-loop path in Isaac Sim, plus the real snapshots from (1).
3. **The `sdl_project` code is weak and its success rate varies strongly with object
   placement.** A first-class goal is to *improve* the planner/pipeline so the reported
   success numbers are genuinely and repeatably achieved across randomized object positions
   — not obtained on lucky layouts. If robustness cannot be reached, the claims are scoped
   down to the measured reality; the numbers **always** follow the improved code's actual
   measured performance, never the aspiration.
4. **Consequence for the drafts:** the current `response_to_reviewers.tex` and
   `access_revised.tex` over-commit to a *full* physical perception + 20 end-to-end hardware
   trials (Section IV-F, A1/A2). Those commitments must be **rewritten** to match the scope
   above (pouring real; perception-in-Isaac-Sim; end-to-end claims scoped to externally
   supplied / rendered-perception state, which Reviewer 1 explicitly offered as an option).
   Track this in PLAN.md; do not leave the drafts promising experiments that will not be run.
5. **Key experiments already exist in the project; do not reinvent them.** The **PDDLStream
   baseline** (`experiments/pddlstream/`, FR5 impl at `examples/pybullet/fr5_obstacle/`) and the
   **LLM infrastructure** (`LLM/llama/script/experiments/`: XDL-generator test set + validator,
   4-token Action Reasoner with a real train/test split) are present. Your job is to **integrate,
   run, and verify they reproduce the claimed numbers over the full input set**, then fill the red
   spans and record settings — not to rebuild them. Only build a substitute if something is truly
   missing (fallback for the baseline if the PyBullet-vs-Isaac-Sim comparison can't be made fair:
   reframe vs. the in-repo rejection-sampling ablation).
6. **Isaac Sim version is unresolved.** The repo's expected version differs from `~/isaacsim`
   (and the drafts say 4.2.0 while the README says 5.1.0). Either vendor the matching version
   inside the project path or port to the current one; then **pin and document the exact
   version** (also required by Reviewer 1 #8). Do not cite a version until it is confirmed.

---

## File roles (source of truth)

| File | Role | Editable? |
|---|---|---|
| `_2026__IEEE_Access/access.tex` | **Original** submitted manuscript. Frozen reference for the diff/highlight. | **No — never edit.** |
| `_2026__IEEE_Access/access_revised.tex` | The **working revised manuscript**. All manuscript edits go here. | Yes |
| `_2026__IEEE_Access/revision/response_to_reviewers.tex` | Point-by-point response to reviewers. | Yes |
| `IEEE_ACCESS_review.md`, `_2026__IEEE_Access/revision/review/review.txt` | The reviewers' comments (input, read-only). | No |
| `sdl_project/` | The code the paper describes (cuTAMP, LLM, perception, Isaac Sim). | Only when an item in PLAN.md requires it |
| `PLAN.md` | Master execution plan + A/B/C experiment catalog + status. | Yes |

---

## Marker discipline (the house rules — do not break these)

The manuscript and the response use two LaTeX markers. Both are defined in the preambles.

- **`\nd{...}` = red placeholder.** A value that depends on an experiment **not yet run**
  or a configuration value **not yet recorded**. It is red on purpose.
  - **NEVER invent, estimate, or "reasonably guess" a number to fill an `\nd{}`.**
    A red span is replaced **only** by a real measurement, a real simulation/analysis
    output, or a value read from the code/lab records. If the number does not exist yet,
    it stays red.
  - `grep -n '\nd{'` over **both** `access_revised.tex` and `response_to_reviewers.tex`
    **must return nothing** before any submission build. This is the hard gate.

- **`\rev{...}` = yellow highlight.** Marks text changed versus the original manuscript.
  Every substantive change to `access_revised.tex` must be wrapped in `\rev{}`
  (or `\revbox{}` for a whole table/figure float) so the Highlighted PDF shows it.
  Two builds from the same file: `\highlighttrue` → Highlighted PDF; `\highlightfalse`
  → clean Main Manuscript.

- The response also uses `\ifshowstatus` (blue heading = concern complete, red heading =
  pending). Set `\showstatusfalse` before submission so status notes vanish.

---

## Integrity rules (non-negotiable)

1. **No fabricated results.** If it was not measured, run, or recorded, it stays `\nd{}`.
   The real-pouring numbers come from the physical rig; you cannot produce them — see PLAN.md.
   Simulation/perception numbers must come from an actual run of the (improved) code over the
   full randomized seed set, not from the best-case layout. A headline success rate that the
   code cannot reproduce across the randomized positions is a fabrication, even if it once
   appeared on screen.
2. **Scope every claim to what was tested.** The reviewers' central complaint was
   over-claiming. Prefer "under the evaluated conditions" / "within the six-operator,
   12-grid workspace" / "for the phenomena the simulator models." **Never re-introduce**
   the retracted global claims: *"complete and non-overlapping coverage,"* *"the Skill
   Library introduces no additional failure modes,"* *"procedural context is the decisive
   factor across multi-step chemical workflows,"* *"reduced planning-time variance."*
3. **Number consistency.** A value appearing in both the manuscript and the response (and
   in any table) must be identical everywhere. When a measured value replaces a red span,
   update **every** occurrence in the same pass. Grep the value across both files.
4. **Statistics honesty.** Proportions get Wilson 95% CIs; the paired planner comparison
   uses the exact McNemar test on shared seeds; time-to-solution is censoring-aware
   (Kaplan–Meier / restricted mean over the budget). Never characterize successful trials
   only, and report tasks separately (no pooling across tasks of different difficulty).

---

## Scope discipline (what NOT to do)

- This is a **revision, not a new paper.** Do not add contributions, sections, or claims
  beyond what a reviewer concern requires. Every change must trace to a concern in PLAN.md.
- **Do not edit `access.tex`** (the original) — it is the baseline the highlight diffs against.
- **Author list is fixed.** A byline change needs a formal editor request; do not touch it.
- Reviewer-suggested references are optional; add one only if it genuinely strengthens the
  work (R3's two refs are already in as [21],[32]). Do not pad the bibliography.
- Do not run the physical-lab (A-item) experiments yourself — specify the protocol, prepare
  the ingest/analysis, and fill numbers once the authors provide them.

---

## Definition of done (whole revision)

- [ ] `grep '\nd{'` empty in `access_revised.tex` **and** `response_to_reviewers.tex`.
- [ ] Every reviewer concern has (a) a response block and (b) a matching, highlighted
      manuscript change.
- [ ] Numbers identical across manuscript ↔ response ↔ tables.
- [ ] Three deliverables build: response PDF (`\showstatusfalse`), Highlighted PDF
      (`\highlighttrue`), clean Main Manuscript PDF (`\highlightfalse`).
- [ ] No retracted over-claim reintroduced; every new claim scoped to its evidence.

---

## Working conventions

- Verify a file/parameter actually exists in `sdl_project/` before citing it in the paper
  or claiming a value; do not describe code that isn't there.
- Chat equations: plain-text Unicode, no LaTeX block math (per global guidance).
- When in doubt about whether a change belongs in this revision, check it against the
  concern it serves in PLAN.md before writing it.

---

@./.conventions/project.md
@./.conventions/session-log.md
