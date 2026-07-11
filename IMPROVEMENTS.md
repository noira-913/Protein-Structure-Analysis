# ALMA — Known Limitations & Implementation Roadmap

## Status

All P1 (physics model), P2 (coverage), P3 (analysis), and P4 (performance)
milestones are complete, including GPU/CPU parity on real hardware and a CI release
pipeline that actually verifies what it ships. Force-field **accuracy is validated
against real structures but not a closed matter** — see item #1 below, which stays
open on purpose. Three further items remain, deliberately deferred as low-priority
(not blocking, not forgotten) — see below.

## Remaining Work

**1. Force-field accuracy — validated to a first bar, not "done"**
`tests/accuracy_test.py` currently passes 14/15 real proteins by a specific,
deliberately modest bar: MC sampling starting from the true native structure stays
within ~2x the AlphaFold-vs-crystal RMSD baseline. That's a real, meaningful check
(it has already caught and fixed multiple real energy-function bugs — see the
hard-core repulsion history and the nucleic-acid filtering fix below), but it is
not the same as being competitive with production MD/scoring tools, and it doesn't
by itself prove the energy function is *accurate*, only that it isn't *badly
wrong* for the cases tested. Concretely still open:
  - Broaden the test set further (more folds, more disulfide/metal/ligand cases,
    larger multi-domain assemblies) — every new real structure tested so far has
    found at least one genuine issue (the hard-core cap, the nucleic-acid filter),
    so more coverage should be expected to keep finding real gaps, not just
    confirming what's already known.
  - The current pass bar (~2x AlphaFold RMSD) is a sanity check, not a target —
    consider tightening it or adding a second, independent accuracy metric (e.g.
    decoy discrimination: does the energy function actually rank near-native
    conformations below deliberately perturbed ones by a wide margin, not just a
    plausible one).
  - Real-world sessions keep surfacing accuracy-adjacent issues in structures the
    test set doesn't cover (disordered linkers being compared against ordered
    references, co-crystallized ligands/nucleic acids contaminating a reference
    structure) — each one so far turned out to be either genuine biology or a
    real parsing bug, not a red flag on the physics itself, but there's no reason
    to assume that streak continues without more testing.

  **Attempted: decoy-discrimination metric — infrastructure built, but surfaced
  a real limitation in the move set rather than closing the question.** Added
  `--decoy-discrimination` to `tests/accuracy_test.py`: generates a hot-MC
  "decoy" (`generate_ensemble` at elevated T) and a same-computational-budget
  "relaxed-native" reference (same step count, native T=0.6) from the same
  starting structure, then compares `calculate_potential()` scores. The
  same-budget design matters — comparing a decoy that got a long relaxation
  budget against the raw, un-relaxed crystal energy (only a few hundred
  near-native steps in the existing test) would conflate "more steps relaxed
  local crystal-packing strain" with "the decoy is a genuinely better fold"; an
  early version of this test made exactly that mistake (1LYZ's raw crystal
  energy is 187,484.6 kcal/mol — 20,000 relaxation steps alone drop that to
  ~10,000 kcal/mol regardless of temperature) before being corrected to the
  same-budget comparison.

  **Real finding: `generate_ensemble` cannot currently produce a structurally
  divergent decoy at any temperature.** Tested T=2.5 (1LYZ, 1UBQ) and T=8.0
  (~13x native T=0.6, 1UBQ) at 20,000 steps each. Every decoy's Calpha RMSD
  from native landed in the same 0.4–1.0 Å range as the same-budget T=0.6
  relaxed-native reference's own RMSD — statistically indistinguishable, not a
  divergent alternative conformation. Root cause, confirmed by reading the
  code (not assumed): per-move rotation is hard-capped at `ANGLE_MAX = 0.50`
  rad in every MC loop in `physics_engine.cpp`, so raising the nominal
  temperature only saturates move-acceptance faster, it never permits a larger
  single move; and the move mix (uniform over all rotatable bonds, plus 25%
  crankshaft) is dominated by sidechain torsions, which don't move Cα at all,
  with crankshaft pairs specifically designed (see Part 4/6 history above) to
  preserve downstream backbone geometry via a compensating rotation rather
  than reorganize the fold. There is currently no move type in this codebase
  capable of a large, sustained backbone reorganization.

  **Consequence: the metric's "margin" numbers are not currently meaningful
  evidence about the energy function's ranking ability** — a negative margin
  (decoy scores lower than native) reflects two near-identical near-native
  ensembles differing by noise, not the force field failing to discriminate a
  real decoy. **Decision: kept as infrastructure, not removed** (the
  same-budget relaxed-native design and RMSD sanity check are both real,
  reusable engineering, unlike Part 2's crankshaft-coupled move, which was
  production physics code that actively made a target metric worse) — a
  runtime caveat prints whenever `--decoy-discrimination` is used, and the
  module docstring records the finding, so this isn't silently mistaken for a
  working, validated check. **Follow-up needed before this closes**: a real
  divergent-decoy generator — e.g. a dedicated large-angle/uncapped move mode,
  rigid-body backbone-segment perturbation, or fragment-threading — none of
  which existed in the move-set inventory checked this session.

**2. IDP/disorder landscape classification — real-IDP calibration in progress, multiple real bugs found and fixed, not yet closed**
The population-weighted, multi-branch IDP classifier (see "Completed Work" →
Analysis for the original design) was calibrated only against stable, ordered
proteins before this round. Testing it against a genuine, literature-documented
partially-disordered protein (1XQ8, human α-synuclein — helical when
micelle-bound, but its C-terminal tail, residues ~98-140, and in truth the
whole free/monomeric protein, are intrinsically disordered) surfaced three real,
previously-undiscovered bugs in the classification logic itself, in addition to
the one pre-existing parsing bug the same test run exposed:

  - **Parsing bug (fixed): multi-model NMR files were silently duplicated.**
    `_parse_pdb`'s main atom loop used `st.get_atoms()`, which walks *every*
    MODEL in a file. For a single-model X-ray structure this is harmless, but a
    multi-model NMR ensemble (e.g. 1XQ8, ~40 conformers) would have every
    conformer's atoms concatenated into one structure — duplicating the whole
    system ~40x and corrupting bond topology/Cα maps. The other three PDB-reading
    helpers in the file (`_ca_map_from_pdb`, `_heavy_map_from_pdb`,
    `_ca_residues_from_pdb`) already guarded against this (`for model in st: ...;
    break`); the main parser was the one place that didn't. Fixed by changing to
    `st[0].get_atoms()`. Since real experimentally-determined IDP structures are
    overwhelmingly NMR-derived (multi-model) by nature, this bug would have hit
    the live GUI for any real disordered-protein structure a user opened, not
    just this test harness.

  - **Classification bug #1 (fixed): asymmetric "competitive basin" comparison.**
    `competitive = [c for c in sig if min(energies[c]) - best_e < COMPETITIVE_KT
    * kT]` used `best_e` = the minimum energy *inside the dominant (most
    populous) basin*, not the true global minimum. The comparison is one-sided:
    any basin with *lower* energy than the dominant basin's own minimum passes
    trivially, no matter how much lower, and the dominant basin always passes
    against itself (diff=0). Confirmed by hand against the exact logged numbers
    for 1UBQ: 4 basins were counted "competitive" purely because they had lower
    minimum energy than the dominant basin, with real energy spreads of up to
    ~96-223 kcal/mol between them — nowhere close to the intended ~20kT (11.84
    kcal/mol) thermal-reach window. This explains why every calibration run
    (including on stable, ordered proteins) had been landing on "POSSIBLY
    DISORDERED": the filter was structurally guaranteed to over-count competitive
    basins. Fixed by comparing every non-dominant basin's own minimum against the
    true global best energy (itself corrected to be the lowest minimum among
    *significant* basins only — see next bug), and excluding the dominant basin
    from counting as its own competitor.

  - **Classification bug #2 (fixed): global reference point contaminated by
    noise/transient snapshots.** The first fix's naive `global_best_e = min(all
    pooled snapshot energies)` includes DBSCAN "noise" points — one-off
    transient configurations that never formed a persistent basin. A single
    lucky transient snapshot can dip below any real basin's minimum by chance,
    dragging the whole reference point down and making every real basin look
    artificially "too far away." Fixed by anchoring to the lowest minimum among
    *significant* (`sig`) basins only.

  - **Calibration issue (fixed): the 20kT threshold was tuned for one protein
    size and didn't generalize.** With both bugs above fixed, re-running the
    same 1UBQ/1XQ8 pair flipped 1XQ8 (a genuine IDP, 97.9% RMSF-disordered) to
    "0 competitive basins / ORDERED" — the opposite failure mode. `COMPETITIVE_KT
    = 20` (≈11.84 kcal/mol) had implicitly been calibrated against 76-residue
    ubiquitin; a real conformational reorganization in a larger/more flexible
    140-residue chain costs more raw kcal/mol even when it's an equally
    legitimate alternate state, because energy is extensive but the original
    constant was flat. Fixed by scaling `COMPETITIVE_KT = 20 * sqrt(n_res / 76)`
    — sqrt because the *spread* from ordinary thermal jitter over N weakly-
    correlated degrees of freedom grows as √N (central-limit-type argument), not
    linearly and not at all. Re-tested against 1UBQ (76 res), 1XQ8 (140 res),
    and 1LYZ (129 res, hen egg-white lysozyme, another real ordered control):
    1LYZ correctly reads ORDERED; 1UBQ now flags a real, literature-documented
    flexible region (its C-terminal tail, residues ~72-76) rather than a
    fabricated one, though the top-line label still reads "POSSIBLY DISORDERED"
    for what may be an overly alarmist label on a real but minor regional
    finding; 1XQ8 still reads ORDERED, consistently, across 4 repeat runs.

  - **Root cause of the remaining 1XQ8 miscall, diagnosed but not yet fixed:
    basin-minimum energy is a noisy extreme-value statistic under short
    sampling.** Repeat runs of 1XQ8 confirmed this is *not* run-to-run flip-
    flopping (it read ORDERED in all 4 runs) — it's structural. Concrete
    example from one run: three basins split 41%/33%/22% of the trajectory (an
    unusually flat population split — itself a strong disorder signal), yet the
    33%-populated basin's own *minimum* sampled energy sat 502 kcal/mol above
    the true global minimum (found in the 22% basin). A third of the entire
    ensemble visited that basin, but its single luckiest snapshot still wasn't
    very low — under short (80-step) branches, "the lowest point a basin
    happened to sample" is a high-variance statistic, while a basin's
    *population* (an average over many samples) is comparatively stable.
    Tuning `COMPETITIVE_KT` further cannot fix this — the comparison quantity
    itself is the wrong one for basins this broad and flat.

  **Decision (2026-07-07): OR the two criteria.** Rather than replacing the
  energy-gap test with a population-derived one (`ΔF = -kT·ln(pop_i /
  pop_dominant)`, the standard Boltzmann relative-free-energy relation, and a
  natural fit given the classifier's dominant-basin selection is already
  population-weighted), a basin will count as competitive if it passes *either*
  the energy-gap test *or* the population-ratio test. Each statistic fails in a
  different direction under short sampling — energy-gap misses well-populated
  basins with an unlucky minimum (the 1XQ8 case above); population-ratio alone
  could miss a genuinely low-energy basin that was simply under-explored. ORing
  them is the more conservative choice against missing a real disorder signal,
  at the cost of being slightly more permissive (a marginally higher risk of
  false-positive IDP labels on truly ordered proteins) — an acceptable trade
  since two ordered controls (1UBQ, 1LYZ) are already in the calibration set to
  catch it if it goes too far. AND-ing was considered and rejected (strictly
  stricter than either alone, and would still miss the 1XQ8 case since it fails
  the energy test outright).

  **First implementation (flat ratio, rejected):** "basin population >= 25% of
  the dominant basin's population" broke both ordered controls — 1UBQ and 1LYZ
  both flipped to POSSIBLY DISORDERED (8/9 and 4/5 basins counted
  "competitive"). Root cause: a flat ratio of the dominant's *raw* population
  degenerates whenever the dominant itself is only modestly ahead of the noise
  floor (common in exactly the flat, borderline-disordered landscapes this is
  meant to catch) — e.g. dominant=17% * 25% = 4.25%, below the 5% SIG_FLOOR
  already required for "significant" at all, making the test a no-op (every
  basin that survived `sig` passed automatically).

  **Second implementation (floor-relative dynamic scaling, current):** instead
  of a flat ratio of the dominant's raw population, scale relative to how far
  the dominant sits *above* `SIG_FLOOR` (its "excess"): `pop_threshold =
  SIG_FLOOR + 0.4 * (pop_dominant - SIG_FLOOR)`. This is what actually adapts
  to landscape shape — a dominant basin barely above the noise floor (flat,
  disorder-leaning landscape) drags the competitor bar down with it; a sharply
  peaked dominant raises the bar correspondingly. The 0.4 constant was picked
  as a reasoned middle value, not hand-tuned to force perfect separation on 3
  samples (risk of overfitting, compounded by the already-confirmed run-to-run
  MC/DBSCAN variance — precisely tuning to a knife-edge boundary would likely
  just flip on the next run's random seed). Re-tested against 1UBQ/1XQ8/1LYZ:
  one run had 2/3 correct (1UBQ → ORDERED, 1XQ8 → POSSIBLY DISORDERED, both
  matching ground truth; 1LYZ → POSSIBLY DISORDERED, RMSF says 0%). The 1LYZ
  miss flagged only the very N-terminus (residues 1-3, 1-5) at modest
  population (18%, 11%) — initially looked like the same pattern as ubiquitin's
  real C-terminal tail finding (a plausible minor terminus flexibility rather
  than a fabricated result), but a repeat-run consistency check ruled that out.

  **Conclusion: 1LYZ is a coin flip across runs, and it's genuine sampling
  noise, not a real minor finding.** 4 total runs: 2 correctly ORDERED (funnel
  0.13, 0.21), 2 incorrectly POSSIBLY DISORDERED (funnel 0.22, 0.10) — and the
  flagged "competitive" region was *different every time* (residues 1-3/1-5 →
  1-6/1-129/1-129 → none → none), never the same feature twice. A real
  structural feature would show up consistently; this doesn't. More telling:
  **funnel never exceeded 0.24 in any of the 4 runs**, for hen egg-white
  lysozyme — a textbook-rigid, 4-disulfide-stabilized protein that a
  well-converged run should show as strongly single-basin-dominant. It never
  did, in any run. That points at the sampling depth itself (3 branches × 40
  walkers × 80 steps) as the actual remaining bottleneck, not anything left to
  fix in the classification logic — consistent with the earlier discussion
  that plain short-run Metropolis MC mixes slowly on a per-run basis and
  shouldn't be trusted for population estimates without either more
  steps/branches or an enhanced-sampling method (e.g. replica exchange).
  Further classification-threshold tuning is very unlikely to help past this
  point; the next real fix here is sampling depth, tracked as a follow-up, not
  yet started.

  **Convergence diagnostic, proof of concept: multi-chain R-hat (Gelman-
  Rubin).** Rather than requiring a manual repeat-run spot-check to notice a
  run like the 1LYZ case above, `LandscapeWorker.run()` now computes a
  per-run R-hat for free by reusing the 3 branches it already runs: split the
  pooled energy trace back into its 3 per-branch chains (truncated to the
  shortest branch's length), compare between-chain variance to within-chain
  variance (standard Gelman-Rubin `R-hat = sqrt(((n-1)/n * W + B/n) / W)`).
  R-hat near 1.0 means the branches agree on the distribution; well above 1
  (conventionally >1.1) means they've settled into different pictures and
  haven't mixed — exactly the failure mode already demonstrated empirically.
  Surfaced in the progress log line and in the emitted result dict (`r_hat`
  key, `None` if fewer than 2 branches or a degenerate zero-variance chain).
  Validated by re-running the same 1LYZ repeat-run check this diagnostic was
  designed to catch, to confirm R-hat actually reads high on the runs already
  known to be unconverged.

  **Validation result: inconclusive/not yet useful as implemented.** 4 fresh
  1LYZ runs all misclassified as POSSIBLY DISORDERED (RMSF confirms all 4
  should read ORDERED). R-hat values: 1.28, 1.25, **1.01**, 1.22. The run with
  R-hat=1.01 — by the conventional <1.1 rule of thumb, "converged" — was
  exactly as wrong as the other three. Likely reason: R-hat computed on the
  pooled *energy* trace measures whether branches agree on the overall energy
  distribution, but the classification is actually driven by DBSCAN cluster
  populations in PC-projected space, which is a related but distinct quantity
  — two branches can have statistically similar energy distributions (low
  R-hat) while still assigning snapshots to different clusters, since
  clustering with only ~40 points per branch is noisy in a way a marginal
  energy R-hat doesn't see. Next step tried: compute R-hat on the PC1
  projection (the coordinate DBSCAN actually clusters on) instead of raw
  energy, since that's more directly tied to what actually varies between
  mislabeled runs.

  **Second attempt (PC1-based R-hat) also inconclusive — and mildly against
  the hypothesis.** Moved the R-hat computation to after the PCA `layout` is
  built and reran it on `layout[:, 0]` instead of energy. 4 fresh 1LYZ runs:
  R-hat = 1.48 (wrong: POSSIBLY DISORDERED), 1.32 (wrong), 3.32 (wrong), 2.29
  (**correct**: ORDERED). The one run that actually got the right answer had
  the *highest* R-hat in the batch -- the opposite of what the diagnostic
  should show if it tracked convergence quality relevant to the
  classification. Combined across both variants (8 total runs, energy-based
  and PC1-based), neither shows the expected pattern (low R-hat correlating
  with a correct/converged label). With only 4 samples per variant this could
  still be noise, but two independent attempts both failing to show the
  hoped-for direction is itself informative.

  **Conclusion: stop iterating on R-hat variants; this line of investigation
  is not converging.** A single scalar convergence summary (whether on energy
  or PC1) doesn't appear to reliably predict classification correctness at
  this sample size, and it isn't obvious a third choice of observable would
  do better. The r_hat field is kept in the codebase (informational,
  harmless) but should not be treated as a trustworthiness signal. This
  reinforces rather than replaces the original diagnosis: the actual fix
  needed is more MC sampling depth (more steps/branches, or an
  enhanced-sampling method), not a better diagnostic layered on top of
  under-sampled chains — a diagnostic can't distinguish "converged" from
  "not converged" when neither branch has actually converged yet. The
  within-branch block-stability check (see below) was deferred rather than
  tried next, given this pattern.

  **Deferred to a later pass: within-branch block stability.** A second,
  complementary diagnostic was scoped but not implemented yet: split each
  individual branch's trajectory in half and compare early-half vs. late-half
  population/energy estimates. This catches a branch that's still drifting
  (hasn't equilibrated within itself) even in the edge case where all 3
  branches happen to agree with each other by coincidence -- a failure mode
  R-hat alone wouldn't catch, since R-hat only measures *agreement between*
  chains, not whether any one of them has stopped moving. Given neither R-hat
  variant showed a usable signal, this is now lower-priority than simply
  increasing sampling depth directly and re-measuring label stability.

  **Broader roster confirms this isn't size-specific.** Filled the one real
  gap in the calibration roster (large + stable — genuine large IDPs mostly
  have no single deposited structure at all, so that quadrant stays
  uncovered) with 1YPI (triosephosphate isomerase, 494 residues, already
  vetted in `tests/accuracy_test.py`'s own protein list). 4 runs: funnel
  0.21/0.10/0.17/0.10, RMSF a clean 0.0% every time, but the discrete label
  still split 2/4 ORDERED vs. POSSIBLY DISORDERED — and R-hat (1.84, 2.55,
  2.38, 2.11) tracked correctness no better here than on 1LYZ, the third
  independent case where it didn't. Notably, funnel scores here were no
  higher than on the much smaller 129-residue lysozyme, despite this being a
  substantially larger, equally rigid, classic stable fold — real evidence
  that the fixed 40-snapshot-per-branch budget becomes proportionally more
  inadequate as protein size grows (bigger conformational space, same fixed
  sample count), supporting a sampling depth that scales with protein size
  rather than one flat constant for every protein, going into the sampling-
  depth optimization work this item's conclusion points to.

  **Sampling depth increased (small/fast proteins first) — validated as a
  success on 1UBQ and 1LYZ.** Rather than guessing a step-count multiplier,
  measured the real integrated autocorrelation time (τ) of the MC energy
  trace directly with a one-off instrumented probe
  (`scratchpad/autocorr_probe.py`, calls `run_landscape_trajectory()` with
  `steps_per_snapshot=1` -- no engine changes needed, the API already
  supports arbitrary thinning). Result: **τ ≈ 500 steps on both 1UBQ (76 res)
  and 1XQ8 (140 res)** -- apparently close to size-independent in this range
  (not yet checked at 1YPI's 494-residue scale). The old production settings
  (`LandscapeWorker.N_SNAPSHOTS=120` total ÷ 3 branches = 40 snapshots × 80
  steps = 3200 raw steps/branch) gave `N_eff = steps/(2τ) ≈ 3.2` effective
  independent samples per branch -- each branch's 40 "saved snapshots" were
  carrying only ~3 genuinely independent draws. This is the root, now-
  quantified cause of the label instability documented above.

  First-pass fix, deliberately targeting ~10x the old N_eff (not the full
  30-50x long-run target -- too large a first jump to validate safely):
  `N_SNAPSHOTS` redefined as a **per-branch** target (was previously divided
  by branch count, so *more* branches meant *less* depth per branch --
  backwards) and raised to 30; `STEPS_PER_SNAP` raised from 80 to 1200
  (~2.4×τ, so each saved snapshot now carries meaningfully more independent
  information instead of being a near-duplicate of its neighbor). Net: ~36,000
  raw steps/branch, ~11x the old depth, targeting `N_eff ≈ 30-40`/branch.

  To keep this affordable, the 3 branches (already fully statistically
  independent) now run **concurrently** instead of in the old sequential
  `for` loop, via `QThreadPool`/`QRunnable` (new `_LandscapeBranchRunnable`
  class, `python/gui_main.py`) -- the first use of any concurrency primitive
  in this codebase beyond the top-level `QThread` workers. Confirmed via
  codebase research before implementing: `run_landscape_trajectory` already
  releases the GIL (`py::call_guard<py::gil_scoped_release>()`,
  `physics_engine.cpp`), so real concurrent execution is possible, but the
  CPU engine has one shared mutable member (`std::mt19937 gen`) touched every
  MC step -- unsafe to share one engine instance across concurrent branches.
  Each branch now gets its own fresh `PhysicsEngine()` instance (cheap to
  construct), following the same convention the codebase already used
  elsewhere ("fresh engine instance to avoid thread contention" at
  `ComparisonWorker`). `self.engine` itself is left untouched during the
  parallel section since `_dedicated_subsearch` reuses it sequentially
  afterward.

  **Validation, 1UBQ (4 runs total: 1 sanity + 3 repeats):** ORDERED every
  single time, funnel tightly clustered at 0.61-0.67 -- a dramatic contrast
  to the old scattered 0.07-0.24 range. Runtime ~160-200s (up from ~40-90s,
  but nowhere near the naive ~11x the step increase alone would imply --
  parallelization is doing real work).

  **Validation, 1LYZ (3 repeats):** landed on POSSIBLY DISORDERED all 3
  times, funnel consistently ~0.30-0.32 with 3 basins splitting the
  trajectory almost exactly 30/30/30 every run -- worse-looking than before
  at face value (old runs were ~50/50 ORDERED/DISORDERED by chance), but
  qualitatively different: **reproducible, not noisy**. Investigated before
  concluding it was a regression: the flagged region (residues ~1-5, near
  the N-terminus) matches literature-documented real HEWL flexibility --
  "the N-terminus of HEWL is very flexible, and can be stabilized through
  interactions with polyoxotungstate molecules," and even the most rigid
  crystal form has "approximately one third of the side chains [existing]
  in more than one conformation" (Hen Egg-White Lysozyme Crystallisation...,
  PMC4498469). Conclusion: this is very likely the deeper sampling correctly
  resolving a *real* local flexibility that the old noisy/shallow sampling
  couldn't reliably surface (the old scattered results look more like random
  noise than consistent detection of anything) -- not a new bug. What
  remains is the same open question already flagged for ubiquitin's real
  C-terminal tail finding: whether "POSSIBLY DISORDERED" is the right
  top-line label for a protein that's overwhelmingly rigid (4 disulfides, low
  B-factors everywhere else) but has one small, real, genuinely flexible
  region -- a labeling-semantics question, not a sampling or clustering bug.

  **Status: first-pass sampling-depth increase counted as a success at this
  scale.** Both test proteins now produce *reproducible, physically
  defensible* results (1UBQ: real single dominant fold; 1LYZ: real
  N-terminal flexibility, correctly and consistently detected) instead of
  noisy, unstable ones.

  **Labeling-semantics fix: gate the top-line classification on structural
  displacement magnitude, not just population/energy competitiveness.**
  Closed the open question above the same day. A basin can be statistically
  "competitive" (population/energy) yet not represent real disorder -- a
  sidechain rotamer flip or normal terminal wobble is not the same thing as
  genuine large-scale conformational heterogeneity. Real examples measured
  this session split cleanly on displacement magnitude (Kabsch-aligned max
  Cα displacement between a competitive basin's representative and the
  dominant basin's representative, reusing the existing
  `_kabsch_align_points` helper): minor local flexibility (1LYZ N-terminus
  ~0.9-4.4 Å across runs, ubiquitin's real C-terminal tail ~2.2-5.2 Å) vs.
  genuine large-scale disorder (1XQ8, real IDP: ~10.5-25.9 Å) -- a wide,
  consistent gap between the two groups. SS-diff fraction was checked and
  rejected as the discriminator (small in *both* groups, doesn't separate
  them). Implemented `STRUCTURAL_DISP_THRESHOLD = 5.0` Å (picked as roughly
  the midpoint of the observed gap; calibrated from a small number of real
  cases, same caveat as every other constant tuned this session) as an
  additional filter: only basins whose representative differs from the
  dominant one by at least this much count toward `IDP`/`POSSIBLY DISORDERED`
  (`competitive_structural`, computed right after `competitive`, reusing the
  dominant-basin representative now hoisted earlier in `run()` for this
  purpose). The raw, unfiltered `competitive` list is left untouched
  everywhere else (sub-candidate/region-flagging, `basin_summary`), so real
  minor flexibility is still correctly *surfaced* in the UI/logs -- only the
  top-line label stopped treating it as evidence of disorder. Progress log
  now shows both counts, e.g. "2 competitive, 0 structural", for visibility
  into which basins actually drove the label.

  **Re-validated on 1UBQ and 1LYZ (3 repeats each) with the gate active:
  6/6 correct.** 1UBQ: 3/3 ORDERED (funnel 0.32-0.97 across runs, 0-1 of 2
  competitive basins crossing the structural threshold). 1LYZ: **3/3 now
  correctly ORDERED** (previously 0/3 with the depth increase alone) --
  every run shows displacements of 1.8-4.4 Å, consistently under the 5 Å
  threshold, while the flagged N-terminal region (e.g. "region 5-129",
  "region 1-1") is still visible in the sub-candidate output for anyone
  inspecting details.

  **Small-scale phase of the sampling-depth/classification investigation is
  now closed.** Explicitly not yet done, tracked as follow-ups: validating/
  tuning at the 1YPI (494-residue) scale, GPU-path-specific timing, adaptive
  per-protein depth, and the deferred within-branch block-stability
  diagnostic.

  **Large-scale (1YPI, 494 res) follow-up — size-adaptive sampling depth,
  closed as a success.** Picked up the deferred 1YPI-scale item directly.
  The original plan was to re-measure the autocorrelation time (τ) at 494
  residues and derive `STEPS_PER_SNAP` from it, the same way the small-scale
  fix above was derived. A dedicated probe (`scratchpad/autocorr_probe.py`,
  same `steps_per_snapshot=1` fine-thinning technique as the original τ
  measurement) instead found **τ does not converge** — it grows with trace
  length instead of settling (2k steps → τ≈508, 20k steps → τ≈3995, 100k
  steps → τ≈26228, all reported with the Sokal auto-window still capped,
  i.e. even the largest trace didn't see the ACF decay below noise). A
  100k-step 1UBQ trace's first-half vs second-half mean energy also
  drifted by -1.54 standard deviations — direct evidence the MC chain at
  T=0.6 hasn't reached a stationary distribution within any affordable
  step budget, so there is no well-defined τ to build a
  `STEPS_PER_SNAP ∝ τ^p` formula on. This means the earlier τ≈500 estimate
  (which drove the small-scale fix above) was itself very likely an
  underestimate from a too-short probe run, not a true converged mixing
  time — though that fix still empirically worked (6/6 correct on
  1UBQ/1LYZ), so the depth increase it produced was directionally right
  even if the τ number motivating its exact magnitude wasn't reliable.
  This is the same failure pattern already seen twice with the R-hat
  convergence diagnostic (both energy- and PC1-based variants failed to
  track classification correctness and were abandoned, above): a
  single-scalar mixing/convergence statistic keeps proving unreliable for
  this specific sampler, on both counts now attempted (R-hat and τ).

  **Two alternatives considered and rejected at this decision point:**
  - *Investigate the non-stationarity/drift itself as a separate, higher-
    priority research question* (e.g. whether plain Metropolis MC needs
    replacing with simulated annealing or replica exchange to actually
    reach equilibrium at T=0.6) — rejected for now, not because it's
    wrong, but because it's a substantially larger, open-ended effort than
    the immediate 1YPI-scale goal, and the existing structural-displacement
    gate + empirical repeat-run validation already gives a working, tested
    answer without resolving it first. Left as a standing question rather
    than closed — the non-stationarity finding itself is real and doesn't
    go away just because the depth-scaling fix worked around it.
  - *Proceed with the original τ-derived formula anyway, using the flawed
    short-trace τ≈500 as a rough anchor* (i.e. keep going as if the small-
    scale fix's number were reliable, matching what that earlier session
    effectively did) — rejected as building the next fix on a measurement
    already caught being wrong, when a measurement-independent mechanism
    (grow `N_SNAPSHOTS` for clustering density) was available and testable
    on its own merits instead.

  **Pragmatic pivot (decided over the flawed-τ formula): only grow
  `N_SNAPSHOTS` with size; bound `STEPS_PER_SNAP` by a runtime budget
  instead of a mixing-time formula.** `N_SNAPSHOTS` scaling is untouched by
  the τ failure — it targets a different, independently-valid mechanism
  (DBSCAN clustering density: PCA feature dimension = 3·n_ca grows ~6.5×
  from 76→494 residues while the pooled sample count stayed flat, so a
  real single basin fragments in the higher-dimensional space regardless
  of chain mixing). Implemented in `LandscapeWorker._adaptive_depth()`
  (`python/gui_main.py`): `n_snapshots = round(30 * sqrt(n_res/76))`,
  clamped to `[30, 60]`; `steps_per_snap` held at the validated 1200
  baseline unless total work (`n_snapshots * steps_per_snap * n_res`,
  since per-step cost is ~O(n_res)) exceeds 3× the anchor's work budget,
  in which case `steps_per_snap` shrinks first (down to a floor of 600) to
  protect the snapshot count, which is the actual target of this fix. Net
  effect at 1YPI: 60 snapshots × 600 steps/branch — the same total
  36,000-step-per-branch budget as the unscaled anchor (so no runtime
  regression versus the old flat scheme), but double the pooled points
  fed to DBSCAN across the 3 concurrent branches (90→180).

  **Validation: new `tests/landscape_stability_test.py` (repeat-run label
  stability, no automated classifier test existed before this).** Runs
  `LandscapeWorker` directly (synchronous `run()` call, no QThread event
  loop) across 3 concurrent branches per protein, matching real GUI usage.
  Result: **1UBQ 3/3 ORDERED, 1LYZ 3/3 ORDERED, 1YPI 4/4 ORDERED** — the
  target fix (1YPI previously split 2/4 ORDERED/POSSIBLY-DISORDERED under
  the flat-depth scheme). 1YPI funnel scores stayed low (0.133-0.144,
  similar to the pre-fix 0.10-0.21 range) but the *discrete label* is now
  reproducible, which is the actual acceptance criterion — consistent with
  the existing pattern where the structural-displacement gate, not funnel
  magnitude alone, drives the top-line label. Runtime: 1YPI averaged
  ~1033s/run (696-1366s across repeats) via the concurrent 3-branch
  dispatch — acceptable, and identical in raw step count to what the old
  flat scheme already spent per branch at this size.

  **Status: large-protein-scale phase of the sampling-depth investigation
  is now closed as a success**, on the same "reproducible across repeats"
  bar as the small-scale phase. Explicitly not yet done at the time, tracked
  as follow-ups: ~~GPU-path-specific timing at this scale (validation above
  used the CPU engine only)~~ **-- since done**, see the "Compute-time
  follow-up" entry further below (measured real GPU timing at 1YPI's scale,
  found and fixed a 34% regression from a naive flat branch-count default);
  a genuine large single-chain IDP test case (still no known suitable
  deposited structure, same gap noted in the small-scale broader-roster
  check above); and the deferred within-branch block-stability diagnostic,
  now doubly de-prioritized given both R-hat and τ have failed as
  convergence signals for this sampler.

  **Investigate the MC chain's non-stationarity/drift directly — still
  open; both originally-named alternatives have since been tried and
  closed (2026-07-09 update).** The 100k-step 1UBQ probe found the chain
  still drifting (-1.54σ first-half vs second-half mean energy) at T=0.6
  with no sign of reaching a stationary distribution within any affordable
  step budget; the shipped fix (structural-displacement gate + empirical
  repeat-run validation, plus the later sampling-budget cut) works around
  this without resolving it. This note originally proposed "simulated
  annealing or replica exchange" as the way to address the root cause —
  both were tried later this session, not just proposed:
    - **Simulated annealing** (as a burn-in accelerant): tested, rejected —
      made 1UBQ's funnel score lower and more variable than no annealing at
      all, and using it for actual production sampling would bias the walk
      toward the minimum-energy state, corrupting the population-weighted
      basin estimates this classifier depends on.
    - **Replica exchange**: fully implemented (temperature ladder, segmented
      execution, swap logic — CPU and GPU), tuned, a real step-size-tuning
      bug found and fixed, retested against all 4 ground-truth proteins —
      mixed result (real win on 1LYZ, no clear win elsewhere, a reproduced
      ~40% misclassification risk on 1XQ8 after the bug fix). Kept off by
      default (`USE_REPLICA_EXCHANGE = False`).
  Both closed, not abandoned as untried. **What's still genuinely open:
  a different enhanced-sampling approach entirely** — not another tuning
  pass on the PT design already tried, and not annealing. Candidates not
  yet attempted: larger/smarter MC moves (e.g. explicit basin-hopping
  jumps between already-discovered basins rather than relying on thermal
  fluctuation to cross barriers), or a different exchange scheme (e.g.
  Hamiltonian replica exchange varying the potential rather than
  temperature, if the flat-temperature-ladder PT's specific failure mode
  turns out to be more about *which* coordinate is being enhanced than
  *whether* enhancement helps at all).

  **Diagnostic (2026-07-09): measured which energy term actually blocks
  rejected moves, before building either candidate above — decisive,
  size- and move-type-dependent result.** Rather than guessing between
  Hamiltonian REST (right fix if the bottleneck is torsional barrier
  height) and coordinated/concerted moves (right fix if it's a
  correlated steric clash the current single/double-DOF move set can't
  route around), added an instrumented probe
  (`PhysicsEngine::run_mc_diagnostic`, `physics_engine.cpp`) that
  separates the hard-core (steric) contribution from the rest of the
  non-bonded sum via a new `pair_e_diag`/`cross_e_diag` pair (duplicated
  from `pair_e`/the existing `cross_e` lambda, not modifying either --
  same precedent as `run_landscape_segment`), and logs every attempted
  move's proposed angle magnitude, accept/reject outcome, and full
  energy-component breakdown. Ran on 1UBQ (76 res, small) and 1YPI (494
  res, large), bucketed proposed moves by `|delta|` into small (routine
  in-basin jitter) vs. large (top quartile -- the moves that could
  plausibly cross a basin boundary), and checked among **rejected,
  large-bucket** moves whether `|d_hardcore|` or `|d_dih|` was the
  bigger contributor:

  | Protein | Move type | n rejected-large | Steric-dominant | Torsional-dominant |
  |---|---|---|---|---|
  | 1UBQ (76 res) | torsion | 2266 | 0.0% | 100.0% |
  | 1UBQ (76 res) | crankshaft | 1206 | 8.1% | 91.9% |
  | 1YPI (494 res) | torsion | 297 | 1.0% | 99.0% |
  | 1YPI (494 res) | crankshaft | 196 | **90.8%** | 9.2% |

  **Single-bond torsion moves are dihedral-barrier-limited at every size
  tested** -- essentially never blocked by steric clash, regardless of
  protein size. **But the concerted crankshaft moves (φ/ψ pair sharing a
  Cα, the codebase's existing coordinated-move mechanism) flip
  completely on the large protein**: torsional-dominant on 1UBQ (91.9%,
  consistent with the torsion-move pattern), but overwhelmingly
  steric-dominant on 1YPI (90.8%) -- and severely so: median hard-core
  penalty among rejected large crankshaft moves on 1YPI is **46,213
  kcal/mol** (vs. 0.00 on 1UBQ), meaning genuine deep atomic overlaps,
  not marginal near-misses. Physically sensible: 1YPI is far more
  densely packed, so even a 2-degree-of-freedom concerted move doesn't
  have enough room to route a large backbone displacement around its
  neighbors -- exactly the "single/double-DOF move can't find the real
  multi-body path" mechanism, and specifically on the large-protein case
  that's been the recurring hard case throughout this whole
  investigation (PT, sampling depth -- always 1YPI).

  **Verdict: Hamiltonian REST is not well-motivated by this data** for
  the actual hard case -- the large-protein rejections that matter
  (crankshaft, large-magnitude, i.e. the ones that could plausibly cross
  a basin) are steric, not torsional. **The coordinated/concerted-move
  family is the evidence-backed next step**, specifically because the
  existing 2-DOF crankshaft mechanism already isn't enough room on a
  densely-packed large protein -- candidates: more/larger coordinated
  move sets (beyond the current single φ/ψ-pair case), loop-closure-style
  moves guaranteed valid by construction, or explicit basin-jump moves
  reusing the existing multi-branch/Kabsch-comparison infrastructure.
  Not yet implemented as of this entry -- this diagnostic's job was to
  pick the right direction before investing in either, not to build it.

  **Runtime-optimization follow-up (2026-07-09): what was tried, what was
  rejected, what was adopted.** Prompted by a GPU-speedup question, this
  investigated why `run_landscape_trajectory` isn't faster on this
  machine's real hardware (RTX 4070, CUDA 13.2), and whether the
  non-stationarity finding above could be worked around cheaply.

  - **GPU rebuild + benchmark: modest 1.48–1.68× speedup, not an order of
    magnitude.** The CUDA extension wasn't present at session start and
    was rebuilt (`python setup.py build_ext --inplace`, confirmed against
    real hardware). Measured on 1UBQ (76 res) and 1LYZ (129 res) at
    production adaptive-depth settings: CPU 398/205 steps/s vs GPU
    589/345 steps/s. Modest because the bottleneck is structural (see
    next finding), not a build/config issue.
  - **OpenMP was silently disabled by the build — fixed, then found not
    to matter.** The build log read "OpenMP undetected -> single-thread
    CPU build." Root-caused directly: `setup.py`'s `has_openmp()` compiles
    a test file with a bare `cl.exe` that has no `INCLUDE`/`LIB` set
    unless invoked inside a VS Developer environment
    (`vcvars64.bat`) — confirmed by reproducing the exact compile error
    ("`omp.h`: no include path set") and then reproducing a clean
    "OpenMP detected" build by loading the VS environment first. Re-
    benchmarked 1LYZ with OpenMP now genuinely enabled (32 logical
    cores available): **no measurable change** (201 vs 205 steps/s
    before). Root cause: torsion-angle Metropolis MC is sequential
    *within* a chain (step N+1 depends on step N's accept/reject), so
    OpenMP can only parallelize the non-bonded energy sum for one step's
    moved atoms (~N/2 atoms) — at these system sizes (600-3800 atoms,
    further thinned by the O(N) cell list) there isn't enough per-step
    parallel work to amortize 32-way thread synchronization overhead.
    Very likely the same underlying reason the GPU speedup above was
    modest rather than dramatic: same sequential-chain bottleneck, just
    on different hardware.
  - **Simulated annealing (as a burn-in accelerant): tested, rejected —
    didn't help, and conceptually risky for this use case anyway.**
    Tested on 1UBQ: an annealed burn-in (4000 steps, T 1.5->0.6) followed
    by a 4x-reduced production budget gave 3/3 ORDERED but a *lower and
    more variable* funnel (0.40-0.73) than the identical reduced budget
    with no annealing at all (0.82-1.00), and cost more wall time (33s vs
    25s avg). Conceptual reason this was always a narrow tool at best:
    plain SA (used for the *production* sampling, not just burn-in)
    deliberately biases the walk toward the single lowest-energy state,
    which is the wrong ensemble for population-weighted basin
    classification — only real replica exchange (which preserves each
    replica's own canonical ensemble) avoids that bias, and that was
    *not* what was tested here.
  - **The one deliberate, validated win: cutting the sampling budget
    ~4x holds up on every ground-truth case available, not just the easy
    one.** Tested reduced budget (~half of both `n_snapshots` and
    `steps_per_snap`, i.e. ~4x fewer total steps) against full production
    budget on all 4 available ground-truth proteins:

    | Protein | Ground truth | Full → reduced label | Funnel (full vs reduced) | Time (full → reduced) |
    |---|---|---|---|---|
    | 1UBQ (76 res) | ORDERED | ORDERED, ORDERED | — vs 0.82-1.00 | — |
    | 1LYZ (129 res) | ORDERED | ORDERED, ORDERED | 0.31-0.33 vs 0.32-0.53 | 230s → 60.6s |
    | 1YPI (494 res) | ORDERED | ORDERED, ORDERED | 0.13-0.14 vs 0.12-0.21 | 1033s → 244s |
    | 1XQ8 (140 res) | real IDP | POSSIBLY DISORDERED, POSSIBLY DISORDERED | 0.33-0.34 vs 0.32-0.35 | 508s → 120s |

    12/12 repeat runs correct across all four sizes and both
    classification directions (ordered *and* the one real-disorder case
    available), at ~4x less wall time, funnel scores in the same range
    either way (not degraded). This closes the one gap the earlier
    sampling-depth validation had — 1XQ8 (the real IDP) had never
    actually been re-tested against the current adaptive-depth code
    before this check.

  **Decision: adopt the ~4x reduced budget as the new production
  default** (halving `BASE_N_SNAPSHOTS`/`BASE_STEPS_PER_SNAP` and their
  dependent clamps in `LandscapeWorker`) — real, already-validated
  speedup with no algorithm change. **Separately, proceed with a full
  design for replica exchange (parallel tempering)** as the actual fix
  for the underlying non-stationarity problem, scoped as its own larger
  follow-up rather than bundled with the budget-cut adoption. Confirmed
  during design: the C++ engine's Metropolis acceptance is `exp(-ΔE/T)`
  with `T` already in kcal/mol (`physics_engine.cpp:64,68`), so the
  standard swap-acceptance formula `p_swap = min(1, exp((E_i-E_j) *
  (1/T_i - 1/T_j)))` applies directly with no unit conversion. Full
  architecture (segmented execution reusing the block-chaining technique
  already proven by the annealing test above, kept orthogonal to the
  existing 3-branch population-coverage structure, new tunables:
  replica count, temperature ladder, swap interval) is scoped in the
  session plan; implementation not yet started as of this entry.

  **Reduced budget adopted.** `LandscapeWorker.N_SNAPSHOTS`/`STEPS_PER_SNAP`
  halved (30→15, 1200→600) and dependent clamps (`N_SNAPSHOTS_CAP`
  60→30, `STEPS_PER_SNAP_MIN` 600→300) re-derived to match. Re-ran the
  full `tests/landscape_stability_test.py` suite (now covering all 4
  ground-truth proteins, 1XQ8 newly added) against the new defaults:
  **13/13 repeat runs correct** (1UBQ 3/3, 1LYZ 3/3, 1YPI 4/4, 1XQ8 3/3),
  confirming the change as adopted, not just measured once.

  **Replica exchange (parallel tempering): implemented
  (`_ReplicaExchangeBranchRunnable`/`_ReplicaSegmentRunnable`,
  `LandscapeWorker.USE_REPLICA_EXCHANGE` toggle, default off) and
  validated against all 4 ground-truth proteins — mixed result, kept
  disabled.** Segmented execution (block-chaining the same way as the
  annealing test), inner `QThreadPool` per branch, alternating-pair swap
  scheme with the confirmed `p_swap = min(1, exp((E_i-E_j)*(1/T_i-1/T_j)))`
  formula (starting ladder: 4 replicas, ratio 1.6, swap every 5
  snapshots — untuned starting guesses). Compared PT against the
  already-adopted reduced budget on all 4 cases, same slot-0 sample
  budget each time:

  | Protein | Plain funnel / time | PT funnel / time | Verdict |
  |---|---|---|---|
  | 1UBQ (76 res) | 0.667-1.000 / 26.6s | 0.778-1.000 / 50.7s | no clear win, ~1.9x cost |
  | 1LYZ (129 res) | 0.300-0.633 / 86.0s | 0.750-1.000 / 117.2s | real win, ~1.4x cost |
  | 1XQ8 (140 res, IDP) | 0.333-0.383 / 72.7s | 0.350-0.450 / 178.9s | marginal, ~2.5x cost |
  | 1YPI (494 res) | 0.144-0.189 / 236.1s | 0.122-0.278 / 450.5s | worse (wider/noisier), ~1.9x cost |

  15/15 PT runs stayed correctly labeled (never broke anything), but
  only 1LYZ showed a genuine accuracy/consistency benefit (tighter,
  higher funnel range vs. plain MC's 0.30-0.63 — real signal on the
  historically hardest, most flip-flop-prone case in this whole
  investigation). The other 3 cases showed no clear win for a real
  1.4-2.5x wall-time cost, and on 1YPI specifically — the original
  motivating case for the whole large-scale investigation — PT actually
  made the funnel score *noisier*, not tighter. This is the "neither
  shows up" outcome the design explicitly flagged as a real possibility,
  materializing on 3 of 4 cases rather than being ruled out. **Decision:
  keep `USE_REPLICA_EXCHANGE = False`** — implemented and available as an
  opt-in for future tuning (untuned ladder/swap-interval constants could
  plausibly change this picture), but not adopted as the production
  default given the mixed evidence.

  **Follow-up: found and fixed a real bug in the PT implementation
  (segment-scoped step-size reset), retested — did not change the
  overall verdict, and surfaced a new accuracy risk.** Root cause:
  `run_landscape_trajectory` (`physics_engine.cpp`) has its own online
  step-size tuning (`cur_max`, rescaled every 200 steps toward a
  28-52% acceptance target) that is call-scoped -- it resets to the base
  `max_angle` and is discarded at the end of every call. The plain-MC
  path calls this once per branch (full trajectory to adapt); PT's
  segmented execution called it once per short swap-interval segment
  (~3000 steps), so `cur_max` restarted from scratch every segment,
  getting only ~15 tuning windows before being thrown away -- plausibly
  hurting 1YPI's much larger conformational space the most, which lined
  up with 1YPI being the one case that got *worse* under PT.

  Fixed by adding a new C++ method, `run_landscape_segment`
  (`physics_engine.cpp`, next to `run_landscape_trajectory`) that also
  returns the final tuned `cur_max`, so `_ReplicaExchangeBranchRunnable`
  (`python/gui_main.py`) can thread it forward as the next segment's
  `max_angle` (and swap it alongside state/energy on an accepted
  replica swap, since the tuned step size belongs to the physical
  configuration, not the temperature slot). Duplicated the ~220-line
  function body rather than extracting a shared helper, matching this
  file's own stated precedent for `run_landscape_trajectory` vs.
  `generate_ensemble` (duplicate to avoid touching an already-verified
  hot path).

  **GPU parity added (2026-07-09).** `physics_engine_cuda.cu` had the
  identical reset bug (`S.cur_max = max_angle` at the top of every call)
  -- fixed the same way, adding a `run_landscape_segment` there too that
  also returns the final `S.cur_max`. Much cheaper to duplicate on this
  side: the GPU engine's `run_landscape_trajectory` is already factored
  into `init_chain`/`run_mc_steps` helpers, so the whole method is ~20
  lines, not the CPU engine's ~220. Python's PT orchestration
  (`_ReplicaExchangeBranchRunnable`) needed no changes -- it already
  calls `engine.run_landscape_segment(...)` polymorphically regardless
  of which physics module constructed the engine. Smoke-tested end-to-end
  on GPU (1UBQ, `USE_REPLICA_EXCHANGE=True`): runs correctly, `cur_max`
  threading confirmed working. Note this confirms GPU PT *runs*
  correctly, not that its accuracy has been separately re-validated
  against the ground-truth proteins the same way the CPU path was above
  -- the verdict (`USE_REPLICA_EXCHANGE = False` by default) is
  unchanged either way, since CPU PT already didn't clear the bar.

  **Retest result: helped 1YPI a little, but removed 1LYZ's one clear
  win, and introduced a real ~40% misclassification rate on 1XQ8.**

  | Protein | Old (buggy) PT | Fixed PT | Plain MC (reference) |
  |---|---|---|---|
  | 1YPI (494 res) | funnel 0.122-0.278 / 450.5s avg | funnel 0.133-0.244 / 397.2s avg (narrower, faster, but still worse than plain) | funnel 0.144-0.189 / 236.1s |
  | 1LYZ (129 res) | funnel 0.750-1.000 (real win) | funnel 0.233-0.483 (win gone -- back to plain-MC-like) | funnel 0.300-0.633 |
  | 1XQ8 (140 res, IDP) | 3/3 correct POSSIBLY DISORDERED | **2/5 wrong (ORDERED)** across two repeat batches | 3/3 correct |

  The 1YPI improvement is real but partial (narrower funnel range, ~12%
  faster) and doesn't close the gap to plain MC. More importantly,
  1LYZ's funnel dropped right back down once `cur_max` was allowed to
  settle and stop resetting -- suggesting the *bug itself* (aggressive,
  repeated step-size reinitialization) was accidentally providing
  exploration diversity that helped 1LYZ, an ironic result: "fixing" a
  bug that looked purely like wasted-effort busywork actually removed
  an accidental benefit in one case, while not clearly helping the
  motivating case (1YPI) enough to matter. The 1XQ8 result is the
  clearest signal: a real, reproduced (not one-off) ~40% wrong-label
  rate on the one case that checks the classifier still detects genuine
  disorder, appearing only after the fix -- both the buggy PT and plain
  MC got this case right every time.

  **Final decision: `USE_REPLICA_EXCHANGE` stays `False`.** The
  step-size-reset fix is correct engineering (kept in the code -- there's
  no reason to revert a genuine bug fix) but does not change the
  practical verdict on replica exchange as implemented: it does not
  reliably outperform the already-shipped budget cut on any case, and
  now has a demonstrated accuracy risk on the real-disorder case. Not
  recommended for further tuning along this exact design without a
  more fundamental rethink (e.g. cost-normalized comparison, a properly
  tuned temperature ladder, or a different enhanced-sampling method
  entirely) rather than incremental fixes to the current architecture.

  **Compute-time follow-up: GPU auto-selection + size-dependent branch
  count — real, unconditional speedup, no accuracy tradeoff (unlike
  PT).** With PT closed out as not worth pursuing further, returned to
  the original compute-time question. Two findings:

  - **GPU was opt-in only** (a "Yes/No" modal at GUI startup,
    `ProteinApp.__init__`) despite being validated to within 0.06% of
    CPU (see "Completed Work" -> Performance) and already having a
    runtime fallback to CPU on failure (`gpu_fallback` signal) -- a pure
    speed win gated behind a click every session for no remaining
    reason. Changed to auto-select GPU when detected.
  - **GPU branch concurrency is far cheaper than expected, but only at
    small-to-medium protein sizes.** Directly measured (not assumed):
    running N `_LandscapeBranchRunnable`s concurrently via `QThreadPool`
    on GPU, wall time barely grows with branch count at 1LYZ's scale
    (129 res): 1 branch 59.8s, 3 branches 66.1s (~1.11x), 6 branches
    79.7s (~1.33x) -- per-branch amortized cost actually *improves*
    (59.8s -> 22.0s -> 13.3s) as more branches queue up. Root cause
    (confirmed by reading `physics_engine_cuda.cu`): no `cudaStream_t`
    is ever created, everything shares the implicit default stream: at
    this atom count each kernel is small enough that CPU-side
    launch-dispatch overhead dominates over actual kernel execution
    time, and that overhead pipelines across threads even without real
    simultaneous kernel execution on the device.

  **First attempt (flat 6 branches on GPU, rejected): regressed 1YPI.**
  Naively fixed `GPU_BRANCH_COUNT = 6` unconditionally. Validated
  end-to-end (real `PipelineWorker`-generated ensemble -> real
  `_start_landscape`-style top-K selection -> real `LandscapeWorker`
  run) on all 4 ground-truth proteins: 1UBQ/1LYZ/1XQ8 all correct and
  at least as fast as the CPU 3-branch baseline (1LYZ notably faster:
  71.8s -> 64.8s with *double* the branches), but **1YPI (494 res) got
  34% slower** (194.5s -> 260.1s), despite being correctly labeled every
  time. Root cause: the "nearly free" concurrency measured at 76-140 res
  is a launch-dispatch-overhead artifact, not real device-level
  parallelism -- at 494 res (~4x the atoms), each kernel is large enough
  to genuinely occupy the GPU's compute resources on its own, so
  "concurrent" branches start actually competing for the same SMs/cores
  instead of hiding behind cheap dispatch overhead. A structural
  crossover tied to kernel size, not thermal throttling.

  **Fix: branch count scales down with protein size instead of a flat
  constant.** `_start_landscape` (`python/gui_main.py`) now computes
  `branch_count = max(3, min(6, round(6 / sqrt(max(n_res,140)/140))))`
  on GPU (CPU unchanged at flat 3) -- stays at the confirmed-good max
  (6) up to 140 res (the largest size directly validated at 6 branches),
  then decays smoothly above it, landing at 3 (matching CPU) for 1YPI's
  494 res by construction. The exact crossover between 140 and 494 res
  is unmeasured -- this is a smooth, principled approximation (same
  sqrt-decay style as `_adaptive_depth`'s size scaling), not a
  independently validated threshold. Also bumped the initial ensemble's
  `n_cand` from 5 to 8 (cheap -- 300 steps/candidate) so enough distinct
  top-K candidates exist to actually fill 6 branches when selected.

  **Re-validated end-to-end after the fix, all 4 ground-truth proteins,
  real ensemble-generation -> branch-selection -> landscape-run path
  (not the isolated pieces): 4/4 correct.** 1UBQ (76 res, 6 branches):
  ORDERED, 40.6s. 1LYZ (129 res, 6 branches): ORDERED, 89.3s. 1XQ8
  (140 res, 6 branches): POSSIBLY DISORDERED, 124.3s. 1YPI (494 res,
  branch count now correctly drops to 3 via the formula): ORDERED,
  223.1s -- back in line with the CPU 3-branch baseline (194.5s), not
  the regressed 260.1s the flat-6 version produced. The size-dependent
  scaling closes the regression while keeping the win on every smaller
  case.

  **Follow-up re-verification (2026-07-09): 1YPI funnel/label stability
  re-checked (4 repeats) after all of this session's GPU/SSL/knot
  changes landed, not just a single spot-check.** The earlier
  re-validation above only ran 1YPI once; real production behavior had
  changed further since (GPU auto-selected by default, SSL bundling
  fixed, knot classifier fixed) without a repeat-run check the way
  every other validation this session got. Ran the real production
  path end-to-end (`generate_ensemble` -> size-dependent branch
  selection -> `LandscapeWorker`, GPU engine) 4x: **4/4 correct
  ORDERED**, funnel range [0.167, 0.322], avg time 131.7s (branch_count
  correctly resolved to 3 at 494 res). Funnel range is a little wider
  than the prior data points on record (0.122-0.222), driven by one run
  at 0.322 -- not a regression: labels stayed correct in all 4, funnel
  has been documented as a noisy statistic under this budget all
  session, and a *higher* funnel is if anything more consistent with
  1YPI's ground truth (rigid TIM-barrel, single-basin dominance
  expected) than a lower one would be. GPU-default production path
  confirmed still correct at this scale.

  **MC mixing bottleneck diagnosis (2026-07-09): steric, not torsional,
  and size-dependent.** With PT closed out as not worth pursuing
  further, went back to the open non-stationarity question (100k-step
  1UBQ probe: -1.54σ drift, chain never converges within any affordable
  budget). Two candidate mechanisms were on the table -- Hamiltonian
  replica exchange (scale the dihedral term; the right fix if the
  bottleneck is torsional-barrier height) vs. coordinated/concerted
  moves (the right fix if the bottleneck is a correlated steric clash
  the current single/double-DOF move set can't route around by chance).
  Built an instrumented diagnostic (`run_mc_diagnostic`,
  `pair_e_diag`/`cross_e_diag` -- splits the hard-core steric term out
  from the rest of the non-bonded sum for every *attempted* move, not
  just accepted ones) and ran it on 1UBQ (76 res) and 1YPI (494 res),
  bucketing proposed moves by size and asking whether `|d_hardcore|` or
  `|d_dih|` dominates among rejected large moves:

  | Protein | torsion: steric-dominant | crankshaft: steric-dominant |
  |---|---|---|
  | 1UBQ (76 res) | 0.0% | 8.1% |
  | 1YPI (494 res) | 1.0% | 90.8% (median hard-core penalty 46,213 kcal/mol) |

  **Decisive and size-dependent**: single-bond torsion moves are
  dihedral-barrier-limited at every size tested (not a steric problem).
  The existing 2-DOF backbone crankshaft move flips from
  torsional-dominant on the small protein to overwhelmingly
  steric-dominant (genuine deep atomic overlaps, not marginal
  near-misses) on the large, densely-packed one. **Verdict: pursue
  concerted/coordinated moves next, not Hamiltonian REST** -- the
  blocking energy on the case that actually matters (the large protein)
  is in the non-bonded term, which dihedral scaling wouldn't touch.

  **Implemented: concerted sidechain-pair moves — real result:
  no measurable improvement on the motivating bottleneck.** Added a
  third MC move type, `try_concerted_sidechain`, to both landscape
  production loops (`run_landscape_trajectory`, `run_landscape_segment`
  -- `generate_ensemble` deliberately out of scope, its short relaxation
  runs aren't attempting basin crossings). Design: at parse time,
  `BondTopology::identify_concerted_sidechain_pairs()` finds pairs of
  sidechain rotatable bonds with disjoint `rot_bond_sides` (fully
  independent rotation groups, unlike φ/ψ which are hierarchically
  nested) within 6.0 Å of each other (476 candidates on 1UBQ, 2728 on
  1YPI); the new move proposes independent simultaneous deltas on both
  bonds (15% selection probability when candidates exist), so a
  sidechain whose swing would clash with a neighbor gets the chance to
  have that neighbor move out of the way in the same proposal. Detailed
  balance holds with no correction (both deltas drawn from the same
  symmetric distribution every other move already uses, state-independent
  pair selection). Correctly distinguishes the two independent groups
  from the shared crankshaft/torsion exclusion machinery: reuses
  `dihedral_e_boundary`/`ss_e_side` as-is via the existing boolean
  `in_side` (their "any atom moved vs. any fixed" semantics don't care
  which group), but adds a separate local `side_group`/`cross_e_concerted`
  for the pairwise non-bonded sum specifically (needed so A-B cross-terms
  between the two groups are correctly *included*, not skipped the way a
  naive boolean union would).

  Rebuilt (CPU+GPU, OpenMP confirmed detected, clean compile), smoke-tested
  (finite/bounded energies on 1UBQ and 1YPI, no explosion). **Re-ran the
  Part 4 diagnostic with the new move active** (added an unlogged copy of
  the same move into `run_mc_diagnostic`'s state-evolution loop, since the
  question is whether crankshaft's rejection profile changes once the
  chain is actually sampling with the new move mixed in):

  | Protein | crankshaft steric-dominant, before | after |
  |---|---|---|
  | 1UBQ (76 res) | 8.1% | 6.2% |
  | 1YPI (494 res) | 90.8% (median 46,213 kcal/mol) | 91.8% (median 55,541 kcal/mol) |

  **No measurable improvement on the motivating bottleneck** -- crankshaft's
  steric-dominant rejection rate on 1YPI is unchanged within noise, and the
  median hard-core penalty among rejected moves is, if anything, slightly
  higher. Plausible explanation: the new move fires on a *different*,
  independently-selected sidechain pair than whichever crankshaft move is
  attempted next -- it doesn't target the specific local packing a given
  crankshaft proposal is about to run into, so it doesn't function as the
  "make room" mechanism the design intended. A pair-selection strategy
  correlated with the crankshaft move actually being attempted (rather than
  a uniformly random independent pick) might be needed to realize the
  intended effect -- not attempted here.

  **Regression + stability validation: no harm, all ground truth held.**
  `tests/landscape_stability_test.py` (all 4 proteins, real production
  path): **13/13 correct labels**, no change in direction from the
  pre-Part-6 baseline -- 1UBQ 3/3 ORDERED (funnel 0.822-1.000), 1LYZ 3/3
  ORDERED (funnel 0.250-0.700), 1YPI 4/4 ORDERED (funnel 0.167-0.267,
  actually *tighter* than the 0.167-0.322 pre-Part-6 range), 1XQ8 3/3
  POSSIBLY DISORDERED (funnel 0.317-0.333). Near-native stability
  spot-check (mirrors `accuracy_test.py`'s own RMSD-during-MC method, but
  against `run_landscape_trajectory` directly -- the function actually
  modified, since `accuracy_test.py` only exercises `generate_ensemble`,
  which Part 6 left untouched): 1UBQ Cα RMSD from the native crystal
  structure stayed in [0.71, 1.35] Å over a full run (energy -534
  kcal/mol, i.e. genuinely relaxing, not just wandering); 1YPI stayed in
  [0.24, 2.33] Å (energy -3229 kcal/mol) -- both comfortably within the
  few-Å near-native envelope the force field is expected to hold, no
  RMSD blowup from the new move.

  **Decision: keep the move (harmless, cheap, occasionally tightens
  funnel consistency) but it did not resolve the motivating steric
  bottleneck** -- large-protein crankshaft rejections remain
  ~91% steric-dominant with or without it. Unlike PT, there's no
  accuracy regression to weigh against keeping it, so it stays in the
  code (no `USE_...`-style off switch was added; it's controlled purely
  by `concerted_sidechain_pairs` availability + the 15% in-C++
  selection probability). The real bottleneck this session set out to
  fix (1YPI-scale crankshaft steric rejections) remains open --
  candidate next step per the negative-result analysis above:
  correlate pair selection with the specific crankshaft move about to
  be attempted, rather than picking independently at random.

  **Tried the candidate next step (`try_crankshaft_coupled`) -- also
  negative, reverted.** Implemented a fourth move type, added to all
  three of `run_landscape_trajectory`, `run_landscape_segment`, and
  `run_mc_diagnostic` (unlogged variant): instead of coupling the
  crankshaft to a randomly-chosen independent sidechain pair, it
  precomputed (`BondTopology::identify_crankshaft_sidechain_neighbors()`)
  which SIDECHAIN rotatable bonds sit spatially near *that specific
  crankshaft pair's own Cα pivot*, and proposed the crankshaft rotation
  together with one of those neighbors simultaneously -- directly
  targeting the "random pair has no relationship to the crankshaft
  move being attempted" explanation from the negative result above.
  Same `side_group`/`cross_e_concerted` correctness pattern as
  `try_concerted_sidechain`, same detailed-balance argument (independent
  symmetric deltas, state-independent selection). Built clean (CPU+GPU),
  smoke-tested (finite/bounded energies, 70/73 crankshaft pairs
  couplable on 1UBQ, 456/480 on 1YPI).

  Re-ran the Part 4 diagnostic with this move active (mixed into
  `run_mc_diagnostic`'s state evolution, unlogged, same as before):

  | Protein | crankshaft steric-dominant | median \|d_hardcore\| among rejected-large |
  |---|---|---|
  | original baseline (no coupled moves) | 90.8% | 46,213 kcal/mol |
  | + `try_concerted_sidechain` only | 91.8% | 55,541 kcal/mol |
  | + `try_crankshaft_coupled` also | **94.0%** | **53,257 kcal/mol** |

  **Worse, not better** -- the correlated-pair-selection hypothesis was
  the specific fix this move was built to test, and it didn't pan out;
  crankshaft's steric-dominant rejection rate went up, not down (1UBQ's
  own crankshaft steric-dominant rate also rose, 8.1%→11.2%, though n is
  small there). `tests/landscape_stability_test.py` still passed 13/13
  across all 4 ground-truth proteins (1UBQ 3/3, 1LYZ 3/3, 1YPI 4/4,
  1XQ8 3/3, funnel ranges consistent with prior runs) and the
  near-native RMSD spot-check stayed bounded (1UBQ max 3.22 Å, 1YPI max
  0.85 Å, both genuinely relaxing) -- so it caused no correctness harm,
  but also delivered no benefit, on real duplicated code across three
  call sites with no GPU parity. **Decision: reverted** (unlike
  `try_concerted_sidechain`, which was kept because it's cheap, causes
  no harm, and occasionally tightens funnel consistency -- this move
  had none of those upsides to offset its complexity, and directly made
  the one number it was designed to move worse). `physics_engine.cpp`
  and `gui_main.py` are back to the pre-this-addition state (matches
  the `try_concerted_sidechain` commit exactly, verified via
  `git diff --stat` showing no change after the revert).

  Both attempts at "give a specific crankshaft pair a way to relieve
  its own steric clash" (random independent pairing, then spatially/
  move-correlated pairing) failed to move the needle. This suggests
  the bottleneck may not be fixable by adding one more simultaneous
  DOF to a 2-DOF backbone move at all -- on a protein packed as densely
  as 1YPI's TIM-barrel core, resolving a genuine deep overlap plausibly
  needs a real multi-body relaxation path (several neighbors moving in
  a coordinated sequence, not one extra single rotation), which points
  toward heavier mechanisms (loop-closure-style guaranteed-valid moves,
  local minimization after a proposed move, or accepting that
  single-temperature Metropolis MC with local moves has a hard sampling
  ceiling on large, tightly packed proteins) rather than another
  variant of "propose one more independent bond." Not attempted this
  session -- item #2 is left open here, with this negative result
  recorded so a future session doesn't re-try the same two ideas.

  **Tried a third mechanism instead: a coarse structural-pivot branch, built as
  its own branch rather than a per-step move — partial win, real limitation
  found.** The two attempts above both tried to get a large structural change
  accepted by ordinary per-step Metropolis MC, and both failed for the same
  reason: the immediate post-move energy of a large backbone rotation is
  dominated by transient steric clashes, so it gets rejected almost by
  construction (this is exactly what basin-hopping's standard shape avoids —
  judge acceptance on the *relaxed* energy after a kick, never the raw
  post-kick energy). Implemented as a new, dedicated branch in the landscape
  classifier (`LandscapeWorker.USE_PIVOT_BRANCH`, `python/gui_main.py`)
  rather than a per-step move: `_coarse_pivot()` applies one unconditional
  (non-Metropolis) large (1.0-3.0 rad, well beyond `ANGLE_MAX=0.50`) rotation
  to a random backbone φ/ψ bond (reusing the existing `concerted_pairs`
  φ/ψ pairing) on a fresh copy of the top-ranked candidate structure, then
  `_PivotBranchRunnable` relaxes it via the *same* `run_landscape_trajectory`
  MC and step budget every other branch already uses. No C++ changes were
  needed — `BondTopology.rot_bonds`/`.rot_bond_sides`/`.concerted_pairs` and
  `Particle.x/y/z` were already exposed read-write to Python
  (`src/physics_engine.cpp:3651-3682`), so the pivot itself is pure Python,
  touching no verified hot path. Same output contract as every other branch
  runnable, so zero changes were needed anywhere in the downstream PCA/DBSCAN/
  basin-significance pipeline.

  Validated (`<scratchpad>/pivot_branch_validation.py`, reusing
  `tests/landscape_stability_test.py`'s exact headless pattern) across all 4
  ground-truth proteins, pivot on vs. off, 3-4 repeats each:

  | Protein | labels (pivot on) | pivot branch's own relaxed energy | vs. reference |
  |---|---|---|---|
  | 1UBQ (76 res) | 3/3 ORDERED (matches off) | ~2,900-3,040 kcal/mol | at or below native's 3,713.8 — genuinely relaxed |
  | 1LYZ (129 res) | 3/3 ORDERED (matches off) | 61,476-1,937,427 kcal/mol | ~6-180x a relaxed-native reference (~10,666) — never recovered |
  | 1YPI (494 res) | 4/4 ORDERED (matches off) | 1.9-9.2 billion kcal/mol | never remotely relaxed (native-scale is tens of thousands) |
  | 1XQ8 (140 res, IDP) | 3/3 POSSIBLY DISORDERED (matches off) | ~11,745-11,935 kcal/mol | still elevated, closer than 1LYZ/1YPI |

  **No label regression anywhere — 13/13 correct with the pivot branch active,
  matching pivot-off exactly.** The existing basin-significance/competitive-
  energy filtering correctly excludes an unrelaxed, astronomically-high-energy
  pivot branch from ever counting as a "competitive" alternative, so the
  concern flagged before validation (a spurious kick flipping a correct
  ORDERED label) did not materialize on any tested case.

  **But the mechanism's real success is narrow, and splits exactly along the
  same size/packing line every other finding this session has hit.** On small,
  loosely-packed 1UBQ, the shared per-branch step budget is enough to relax
  the kick to a genuinely competitive (sometimes lower-than-native) energy,
  and it visibly adds real alternative basins (`n_sig` rose from a 1-3 range
  to 2-5 across repeats). On 1LYZ and especially 1YPI — the actual motivating
  cases from the crankshaft-coupled investigation — the same shared budget
  isn't remotely enough to relax a kick this large; the branch just sits at
  an unphysical energy for its entire trajectory, correctly filtered out as
  non-competitive (safe), but also non-functional (it never finds anything).
  It also costs real wall time on the large cases for zero benefit: 1YPI runs
  with the pivot branch active took up to 1658s vs. 403s worst-case without
  it (extra neighbor-list-rebuild churn from the wildly displaced coordinates
  plausibly compounds the cost, not just one more parallel branch's raw step
  count). 1XQ8 sits in between — an IDP's more open structure relaxes further
  than 1LYZ/1YPI but still nowhere near native-scale.

  **Decision: kept, default OFF** (`USE_PIVOT_BRANCH = False`, same posture as
  `USE_REPLICA_EXCHANGE`) — real, validated, harmless infrastructure that does
  what it was designed to do on small proteins, but doesn't yet solve the
  large-protein basin-diversity problem it was built for, and adds cost there
  without benefit. The bottleneck is the same one identified for
  `try_crankshaft_coupled`: MC-only relaxation (no gradient minimizer exists
  in this codebase, confirmed by grep) is simply too slow to recover from a
  large kick within an affordable step budget on a densely packed structure.
  **Candidate next step, not attempted**: give the pivot branch its own larger
  step budget instead of reusing the shared `N_per`/`S` (it starts much
  farther from equilibrium than every other branch, so an apples-to-apples
  budget was never obviously the right choice — this was an explicit open
  question going into validation, and the data now argues for scaling it up,
  at least on large proteins), or apply several smaller pivots instead of one
  large one to keep displacement high while landing in a more recoverable
  starting clash.

  **Tested greedy (near-zero-temperature) MC as a cheap, zero-new-code
  relaxation alternative — mixed result, worse on the case that matters.**
  Reused `run_landscape_trajectory` exactly as-is with `T=0.01` (only
  energy-decreasing moves get accepted) instead of the pivot branch's normal
  `T=0.6`, over the same shared step budget, starting from the same
  post-kick structure (`<scratchpad>/greedy_mc_relax_test.py`). On 1LYZ the
  two ended close (greedy 16,135 vs. normal 15,622 kcal/mol) but greedy was
  still dropping at the end of the budget while normal MC had already
  plateaued — inconclusive which wins with more steps. On **1YPI, greedy was
  clearly worse** (95,233 vs. normal MC's 82,623 kcal/mol) and had already
  plateaued, while normal T=0.6 MC was still steadily dropping (−1,646
  kcal/mol over the last 5 snapshots, nowhere near converged). This is the
  classic greedy-descent failure mode: rejecting every uphill move forfeits
  the ability to cross a small barrier to reach a better basin, and on the
  densely-packed 1YPI landscape that cost more than it saved — regular
  Metropolis's thermal fluctuations found a better path greedy couldn't.
  **Conclusion: greedy MC is not a good substitute for more normal-MC
  budget.** The more useful signal is that normal T=0.6 MC had *not*
  converged at the current shared step budget on 1YPI — direct evidence
  for the "give the pivot branch a larger relax budget" lever noted above,
  not a temperature-schedule change. Not yet implemented (budget increase
  itself untested) — next concrete step if this thread continues.

  **Queued candidate, to look at regardless of how the budget/greedy-MC
  experiments above turn out: a torsion-angle-space gradient minimizer.**
  Every relaxation mechanism tried this session (plain MC, PT, the pivot
  branch's relax phase) is MC-only — there is no gradient of the energy
  function anywhere in this codebase (confirmed by grep), so relaxing a
  large kick means waiting for enough random accepted moves to happen in
  the right order, rather than walking directly downhill. A minimizer
  (steepest descent / conjugate gradient / L-BFGS) in torsion-angle space
  would fit this engine's existing move representation (every move here
  already works by rotating a bond and propagating the downstream side —
  same as `rodrigues()`/`rot_bond_sides`), but needs analytic dE/dθ per
  rotatable bond, chain-ruled through the nonbonded (LJ + electrostatics +
  GB) and dihedral energy terms — real, nontrivial derivative math to
  derive and verify per term, not a small addition. A Cartesian-space
  minimizer would be more standard for MD-style tools but conflicts with
  this engine's design, which deliberately keeps bond lengths/angles rigid
  by construction (see item #3 below) — unconstrained per-atom movement
  would need bond/angle constraints bolted on. Explicitly not contingent on
  the pivot-branch budget/greedy-MC experiments succeeding or failing —
  worth evaluating on its own terms as the deeper fix if MC-only relaxation
  turns out to have a hard ceiling regardless of budget or acceptance
  temperature.

**3. Bond stretching + angle bending energy terms (P1.4c)**
Torsion moves preserve bond lengths/angles by construction, so these terms sit at
their equilibrium minima and contribute < 0.1 kcal/mol/step — safe to defer
indefinitely unless a future move type (e.g. bond-length perturbation) needs them.

**4. Full GAFF2 force field for organic ligands/cofactors — bond connectivity
done for ATP/heme/NAD⁺/FAD/PLP, partial charges still deliberately unfixed**
Currently: HETATM ligands get an element-based fallback (radius/ε by element only,
zero partial charge) rather than real small-molecule parameters. A full fix means
shipping a GAFF2 atom-type table and either calling `antechamber` programmatically
or caching parameters for common cofactors (ATP, heme, NAD⁺, FAD, PLP…). Metal ions
already have proper tabulated parameters (`amber_params.ION_PARAMS`) — only organic
ligands are affected.

**Scoping decision (2026-07-09): no cheminformatics dependency in this codebase.**
Checked for RDKit/OpenBabel/antechamber before starting — none present anywhere in
source, `requirements.txt`, or the PyInstaller build (`alma.spec`/
`build_portable_exe.bat`). Calling `antechamber` programmatically would mean bundling
a full AmberTools installation into what's currently a self-contained, dependency-free
portable `.exe` — a real architecture departure, not a small addition. Hand-tabulated
parameters for a curated cofactor list (the item's own suggested alternative) fits the
existing pattern much better (mirrors `ION_PARAMS`, which is exactly this approach for
metal ions already).

**ATP bond connectivity added (`bond_templates()`, `physics_engine.cpp`) — partial
charges intentionally NOT added.** Started with ATP only (most tractable: moderate
size, extremely well-studied). Two separable pieces: (a) bond connectivity — standard
organic chemistry, an established structural fact, not a fitted parameter, so
high-confidence to add directly; (b) partial charges/VDW radii — real fitted
parameters (RESP charges, e.g. from Meagher, Redman & Carlson 2003, J. Comput. Chem.
24, 1016 for the triphosphate group) that need an independently verifiable source.
Tried to fetch/verify the standard reference (Manchester AMBER parameter database,
the primary paper via two different hosts) — all three attempts failed (connection
error, 403, no direct data). **Decided not to hand-transcribe charges from training
recall and present them as verified real values** — for a physics tool, a
plausible-looking-but-wrong number is worse than the current honest, visibly
approximate element-based fallback. Shipped the bond-connectivity half only, which is
independently valuable: ligand atoms previously had *zero* bonded topology at all
(`BondTopology::build()` silently skips any residue not in `bond_templates()`), so
covalently-bonded atoms *within* a ligand had no 1-2/1-3 exclusion from the
non-bonded sum — the same category of hard-core-repulsion-blowup bug already fixed
for protein termini/disulfides, just never noticed for ligands because none had a
bond template at all yet.

**Validated the fix is functionally necessary, not cosmetic**, using a real deposited
ATP molecule (extracted from PDB entry 2P0X, the smallest ATP-containing structure by
atom count, 31 heavy atoms — hydrogens present in that particular file were excluded
from this test since ATP has no H-bond template yet, a separate smaller gap):
calculated potential energy **with** the new bond template: 271.4 kcal/mol (8.76
kcal/mol/atom, comfortably in the sane range). Re-ran the identical coordinates with
the residue renamed to something unrecognized (simulating the old no-template
behavior): **1,956,212 kcal/mol (63,104 kcal/mol/atom)** — a ~7,200x hard-core
blowup, confirming this was a real, live bug for any ATP-containing structure, not a
hypothetical one. `tests/bridge_test.py` re-run clean (purely additive change, no
existing `bond_templates()` entries touched).

Rotatable-bond classification (the separate `RotSpec` table used for MC torsion move
selection) was **not** added for ATP — out of scope for this fix, which only targets
non-bonded exclusion correctness. ATP atoms currently can't be moved by torsion MC
moves (they're static during sampling), which is a reasonable, deliberate boundary,
not an oversight.

Remaining for this item: real partial charges/VDW for ATP (blocked on sourcing, see
above — welcome a pointer to a verifiable parameter file/table), then heme/NAD⁺/FAD/
PLP bond templates + charges following the same two-step pattern.

**Follow-up sourcing attempt (2026-07-09), same wall, more thoroughly confirmed.**
Revisited this specifically to add ATP's charges. Four more routes tried, all dead
ends: the Manchester AMBER parameter database (still unreachable), the Meagher et al.
2003 paper via a Wayback Machine snapshot (blocked), a targeted GitHub search for a
redistributed `ATP.mol2`/`.prep`/`.lib` file with real RESP charges (no direct hit),
and RCSB's own Chemical Component Dictionary CIF for ATP (`_chem_comp_atom.charge` —
confirmed this is formal charge only, all zeros, not usable as a partial-charge force
field parameter). **Decision (re-confirmed): stop here, keep the element-based
fallback.** Not a quick problem to unblock with another search attempt — if picked up
again, either bring a verified source/file directly, or deliberately choose the
"chemically-reasoned approximation, clearly labeled as such" path discussed and
declined this round (as opposed to presenting unverified numbers as real RESP values).

**Heme b / NAD⁺ / FAD / PLP bond connectivity added (2026-07-10) — same
pattern as ATP, this time sourced from RCSB's Chemical Component Dictionary
directly instead of memory.** ATP's connectivity was hand-transcribed from
standard organic chemistry knowledge; for these four (larger, more complex
heterocycles — a real chance of a subtle atom-naming slip, exactly the kind
of risk the ATP charges decision was already cautious about), fetched
`files.rcsb.org/ligands/view/<CODE>.cif` directly and parsed the
`_chem_comp_bond` loop's heavy-atom (non-hydrogen) pairs programmatically —
this record type is official structural data (not a fitted parameter), so
it's independently verifiable in a way ATP's RESP charges never were.
Verified two ways before trusting it: every ligand's extracted atom count
matches its `_chem_comp.formula` heavy-atom count exactly (HEM 43/43, NAD
44/44, FAD 53/53, PLP 16/16), and each bond graph forms a single connected
component (no isolated fragments — a real molecule shouldn't have any).

**Validated functionally necessary, same method as ATP**: extracted each
ligand's real deposited HETATM block (HEM from 1C53, NAD from 7YW4, FAD from
4E0H, PLP from 1G76) and compared `calculate_potential()` with vs. without
the template (residue renamed to simulate the old no-template fallback):

| Ligand | with template | without template | blowup |
|---|---|---|---|
| HEM (43 atoms) | 82.2 kcal/mol (1.91/atom) | 3,784,824.8 kcal/mol | 46,066x |
| NAD (44 atoms) | 261.0 kcal/mol (5.93/atom) | 3,419,873.5 kcal/mol | 13,105x |
| FAD (53 atoms) | 371.6 kcal/mol (7.01/atom) | 4,022,912.7 kcal/mol | 10,825x |
| PLP (16 atoms) | 68.7 kcal/mol (4.29/atom) | 681,356.3 kcal/mol | 9,922x |

Same category of confirmed-real bug as ATP: without a bond template, every
covalently-bonded intra-ligand pair goes through the full non-bonded sum
with no 1-2/1-3 exclusion, and the hard-core repulsion term blows up.
`tests/bridge_test.py` re-run clean (purely additive, no existing template
entries touched). Charges/VDW for all four still go through
`amber_params.py`'s element-based fallback, same open gap as ATP —
rotatable-bond classification also not added (out of scope, same boundary
as ATP: these ligands stay static during MC torsion sampling).

**5. Membrane/lipid slab model**
Implicit solvent assumes uniform water (ε=78.5) everywhere. Membrane proteins need
a low-ε bilayer-region model; no such model exists yet. Low priority — no membrane
protein test cases in current use.

**6. False-positive knot detection on larger backbones — root-caused and
fixed (2026-07-09)**
While running the 1YPI (triosephosphate isomerase, 494 residues) calibration
data for item #2, the knot classifier called it a trefoil (3₁, 6 crossings)
at only 58-92% closure-vote confidence across 4 runs. TIM-barrel folds are a
classic, textbook *unknotted* topology in the structural biology literature —
this doesn't match. For comparison, the validated cases in "Completed Work" →
Analysis (1LYZ = unknot, 1J85 = documented deep trefoil) both hit 97-100%
confidence.

**Root cause: 1YPI is a homodimer (2 chains, 247 residues each), and the
classifier's call site pooled both chains' Cα coordinates together before
ever reaching the closure/KMT algorithm.** `PipelineWorker.run()`
(`gui_main.py`) built `ca_coords` directly from `ca_map.values()` with no
chain-break awareness — measured directly: a real within-chain Cα-Cα
distance of 3.86 Å vs. a **65.94 Å** artificial "bond" at the chain A→B
boundary, a ~17x-oversized non-bond the closure/crossing-detection code had
no way to know wasn't real backbone. This isn't a numerical robustness
issue in the closure or KMT reduction step (the original hypothesis) — a
knot is only a well-defined property of a *single continuous curve*, so
concatenating two separate, non-covalently-linked chains into one "backbone"
is topologically meaningless, not just noisy input. The low, inconsistent
confidence (58-92%, vs. 97-100% on the genuinely single-chain controls) is
exactly what a single artificial long segment inconsistently registering as
a crossing under different random projections would look like.

Confirmed by direct reproduction: classifying 1YPI's chain A alone (247 res,
a real continuous backbone, no chain break) gave **unknot at 75-88%
confidence across 4 independent seeds** — matching the textbook TIM-barrel
topology exactly, at confidence comparable to the other validated single-chain
cases.

**Fix:** `PipelineWorker.run()` now groups `ca_map` by chain and classifies
only the largest chain, with a log message when other chains are skipped
("knot type isn't well-defined across separate, non-bonded chains") rather
than silently dropping data. Re-verified against the real production code
path (`_parse_pdb` → grouped `ca_map` → `classify_backbone_knot`) at the
GUI's actual `n_trials=12`: unknot, 75-83% confidence, consistent across 3
seeds.

**Added 1YPI to `tests/knot_test.py`** as a third case (large-backbone
unknotted negative control) — the gap the original investigation flagged
("1YPI is NOT in the test suite... likely why this false positive went
unnoticed") is now closed. `fetch_ca_trace` in that file had the identical
chain-pooling bug (harmless before now, since both existing cases — 1LYZ,
1J85 — are single-chain) and was fixed the same way. Full suite re-run:
3/3 PASS (1LYZ unknot 97%, 1J85 trefoil 97%, 1YPI unknot 88%).

*(Also: `tests/accuracy_test.py`'s auto-offset alignment can't handle a PDB entry
whose crystal numbering doesn't correspond linearly to full-length UniProt numbering
at any single offset — hit once, for chymotrypsin inhibitor 2 (2CI2). This is a
limitation of that test harness's alignment heuristic, not the physics engine.)*

**7. IDP ensemble characterization — in progress, not yet wired into the GUI
or validated (branch `idp-handling`)**
Item #2's classifier only ever produces a binary ORDERED/POSSIBLY-DISORDERED/
IDP label plus a funnel score — once something is flagged disordered, there's
no further description of *what the ensemble actually looks like*. This item
is the first of three originally-scoped directions (ensemble characterization,
disorder-aware sampling, GUI/visualization) for actually giving the user a
real IDP result rather than just a label; the other two are not started.

Inventory taken before writing any code (avoid duplicating existing
infrastructure): confirmed no radius of gyration, end-to-end distance, or
contact-map computation exists anywhere in this codebase (C++ or Python);
confirmed `sasa_nonpolar()` (`physics_engine.cpp`) is a private scalar-total
helper baked into the energy function, not exposed to Python or usable as a
general per-atom SASA routine; confirmed the one existing per-bond MC
step-size hook (`BondTopology.rot_bond_scale`) is driven by a structural
lever-arm heuristic, `def_readonly`, and never touched by IUPred/RMSF
anywhere — i.e. there is currently zero feedback from the sequence-based
disorder predictor into MC sampling.

**Built so far (`python/gui_main.py`), all operating on `LandscapeWorker`'s
existing pooled `snapshots` — no new C++ needed for any of it:**
  - `_compute_radius_of_gyration()`, `_compute_end_to_end()` — per-snapshot,
    Cα-based, rotation/translation-invariant (no alignment needed, unlike
    RMSF).
  - `_compute_internal_scaling()` — the polymer-physics internal-distance
    scaling law `<R_ij> ~ |i-j|^ν` (Marsh & Forman-Kay 2010, Biophys J),
    fit via ensemble-mean Cα-Cα distance binned by sequence separation, then
    log-log least squares. Unlike an Rg-vs-chain-length fit, this works from
    a *single* protein's own ensemble (no need to compare across differently-
    sized proteins) — ν≈0.33 indicates a compact globule, ν≈0.5 an ideal/
    random-coil chain, ν≈0.6 an expanded/self-avoiding disordered chain.
    Loops per-snapshot to bound memory to O(n_ca²) rather than
    O(n_snaps·n_ca²) — the naive broadcast form would need a multi-GB array
    at 1YPI's scale (494 residues).
  - `_compute_contact_map()` — per-residue-pair contact frequency across the
    ensemble (Cα-Cα distance < 8 Å) plus a per-residue "contact variance"
    summary (mean Bernoulli variance `p(1-p)` over non-adjacent partners).
    Deliberately complementary to RMSF, not a duplicate: RMSF needs a
    Kabsch-aligned reference frame, which becomes shaky for a residue deep
    in a genuinely disordered region (no single "mean position" is
    meaningful); contact frequency needs no global alignment at all, only
    per-snapshot relative distances, so it stays meaningful exactly where
    RMSF's assumption is weakest.
  - **Real bug fixed along the way, not just new metrics**: `_compute_rmsf`'s
    Kabsch fit was computed from *all* Cα atoms, including any genuinely
    disordered ones. When a large flexible region is present, the
    least-squares fit itself gets dragged around by that region's motion
    ("the alignment chases the tail"), contaminating the reference frame
    used to measure fluctuation everywhere — including the truly rigid core.
    Fixed by adding a `fit_mask` parameter to `_kabsch_align_points()`: the
    rotation is still *applied* to every atom, but is now *fitted* only from
    IUPred-predicted-ordered residues (score < 0.5) when available — a
    signal that exists pre-MC, purely from sequence, so it can't be
    circularly biased by the RMSF it's used to compute. Standard practice in
    MD ensemble analysis for exactly this reason (e.g. restricting alignment
    to a `backbone and resid <core>` selection). Falls back to the original
    all-atom fit if fewer than 3 residues pass the core threshold (Kabsch
    needs ≥3 non-degenerate points).

**Validated so far (synthetic + smoke tests, not yet the full ground-truth
suite):**
  - Synthetic core/tail contamination test: a rigid "core" cluster plus a
    "tail" cluster given an independent extra rotation on top of a shared
    rigid-body motion. Core-restricted fit recovered the true core rotation
    to floating-point precision (residual ~2e-15 Å); fitting on the whole
    set (old behavior) showed real, measurable contamination (1.18 Å
    residual) — confirms the fix does what it's supposed to, not just that
    it runs.
  - Real smoke test on 1UBQ (76 res, 20-snapshot MC trajectory): Rg
    11.61±0.02 Å (matches ubiquitin's known compactness), tight end-to-end
    distribution, contact-map per-residue variance near zero (0.0002 mean —
    correct for a short, near-native, rigid trajectory), core-aligned vs.
    plain RMSF correlated at 0.93 with the expected direction of difference
    on a synthetically-tagged "disordered" tail.
  - Timing at 1YPI's scale (494 residues, 180 synthetic snapshots matching
    production adaptive-depth budget): every new metric completes in under
    3 seconds (`_compute_internal_scaling`/`_compute_contact_map`, the two
    O(n_ca²)-per-snapshot ones, ~2.5s each) — negligible next to the MC
    sampling itself.

**GUI wiring completed.** A new `ENSEMBLE` page (`_view_stack` index 3,
toggle button next to `DISORDER`) shows a 4-panel figure — Rg histogram,
end-to-end histogram, the internal-scaling log-log fit (with ν/R²
annotated), and a contact-frequency heatmap — following
`_draw_disorder_profile`'s existing matplotlib-in-Qt embedding pattern
exactly (same `Figure`/`FigureCanvas` construction, same page-toggle
mechanics as the existing `DISORDER`/`LANDSCAPE` pages). Wired into
`_on_landscape_done` right after the existing RMSF computation, reusing the
same `data["snapshots"]`/`self._ca_indices` already available there — no new
data plumbing needed. Verified end-to-end headlessly (construct `ProteinApp`
without showing a window, feed it a real MC result dict, confirm no
exceptions and that the toggle button/stats label update correctly) before
running the real validation below.

**Validated against all 4 ground-truth proteins — a real, decisive signal on
two of the four metrics, a genuine negative finding on the third.** Single
production run per protein (not a repeat-stability suite — this is a first
qualitative check, matching the same adaptive-depth budget and plain-MC path
used everywhere else in this file, i.e. `USE_ALTERNATING_RELAX`/
`USE_REPLICA_EXCHANGE`/`USE_PIVOT_BRANCH` all default `False` — none of the
`minimize_torsion` cost-wall machinery from the closed-out alternating-GM/MC
branch is involved here):

| Protein | Direction | Label | Rg (Å) | ν | R² | contact_var |
|---|---|---|---|---|---|---|
| 1UBQ (76 res) | ordered | ORDERED | 11.55±0.04 | 0.168 | 0.29 | 0.0006 |
| 1LYZ (129 res) | ordered | ORDERED | 13.86±0.03 | 0.105 | 0.17 | 0.0010 |
| 1YPI (494 res) | ordered | ORDERED | 24.26±0.02 | 0.276 | 0.93 | 0.0001 |
| 1XQ8 (140 res, real IDP) | IDP | POSSIBLY DISORDERED | 48.30±1.34 | **0.716** | **0.98** | 0.0001 |

**Rg and ν both show a sharp, real separation, exactly matching the
literature-grounded expectation.** 1XQ8's Rg (48.3 Å) is dramatically larger
than any of the ordered proteins despite being the second-*smallest* chain
in the set (140 res, smaller than 1YPI's 494) — a folded protein this size
would never reach that radius. Its Rg spread is also far wider in absolute
terms (std 1.34 Å vs. 0.02-0.04 Å for the ordered cases). Most decisively:
ν=0.716 with R²=0.978 (an excellent power-law fit) lands squarely in
expanded/self-avoiding-chain territory, sharply distinct from the ordered
proteins' compact-globule range (ν=0.105-0.276). 1YPI's own ν (0.276,
R²=0.933) is higher and better-fit than 1UBQ/1LYZ's (0.105-0.168,
R²=0.17-0.29) — plausibly because its larger, more extended shape gives
genuine polymer-like scaling behavior over a longer range of sequence
separations, while the two smaller compact globules are mostly just noise
at their scale (too small to show clean internal scaling at all).

**`contact_var` did not discriminate as motivated — a real, mechanistic
negative finding, not a bug.** It stayed low (0.0001-0.0010) across *all
four* proteins, including 1XQ8, with no clear ordered-vs-IDP separation.
Root cause, understood after the fact: `contact_var` is a Bernoulli-variance
statistic (`p·(1-p)`, maximized at contact probability `p=0.5`), but the two
regimes it was meant to distinguish both drive `p` toward the *extremes*,
not toward 0.5, just via different mechanisms — a compact ordered protein
has most residue pairs *permanently* in contact or permanently far apart
(`p≈1` or `p≈0`), while 1XQ8's fully-expanded ensemble has most pairs
*almost never* in contact at all (`p≈0` too, just via chain extension rather
than fixed structure). The metric only lights up for residues with
genuinely *intermediate, fluctuating* contact behavior across the ensemble
(e.g. molten-globule-like transient contacts, or a residue oscillating
between a compact and an open sub-state) — a real, different physical
question from the general ordered/disordered axis it was introduced to
detect. Kept in the codebase (the underlying computation is correct and the
metric may be useful for that narrower question), but not treated as a
reliable IDP-detection signal on its own.

**Status: ensemble characterization (item #7's first of three originally-
scoped directions) is functionally complete and validated.** Rg + internal
scaling exponent are real, working, literature-grounded IDP descriptors,
now visible in the GUI. `contact_var` is shipped as correct infrastructure
with an honestly-documented limitation rather than oversold. Explicitly not
done: disorder-aware sampling and further GUI/visualization-support work
(the other two originally-scoped directions) — not started.

---

## Completed Work

### Physics model
- **AMBER ff14SB atom typing** — `(resname, atomname)` → ~35 heavy-atom types →
  full VDW radii/ε (`amber_params.py`, integrated in `gui_main._parse_pdb`).
- **RESP partial charges** — full ff14SB charges for all 20 amino acids + HIS
  protonation variants.
- **Bonded topology + dihedral energy** — covalent bond graph built from residue
  templates (`BondTopology` in `physics_engine.cpp`); dihedral (Ramachandran)
  energy over all rotatable bonds; 1-2/1-3 non-bonded exclusions.
- **Torsion-angle MC moves** — Rodrigues rotation of one rotatable bond's j-side
  per step (50% backbone φ/ψ, 40% sidechain χ, 10% other), replacing the earlier
  physically-invalid Cartesian-translation sampler. See "Technical Notes" below
  for the move-schedule pseudocode.
- **Hard-core repulsion term** — went through three real bugs before landing on
  its current form: (a) miscalibrated threshold caused legitimate 1-4/H···H
  contacts to blow up to 10⁸–10⁹ kcal/mol (fixed: `HARD_CUTOFF_FRAC = 0.6` +
  terminal/protonation-variant bond patching); (b) the threshold was a hard
  branch, discontinuous at the boundary, so a contact landing a fraction of a
  percent to either side got wildly different energies (fixed: smooth additive
  penalty, continuous in value and slope at `r_cut`); (c) that smooth penalty was
  still unbounded as `r→0`, so a genuine deep clash in an old, low-resolution
  structure could still blow up to billions of kcal/mol (fixed: `HARD_CAP = 5000`
  kcal/mol ceiling on the penalty term). Both CPU (`physics_engine.cpp`) and CUDA
  (`physics_engine_cuda.cu`) engines carry all three fixes.
- **Disulfide bonds** — SG–SG pairs < 2.5 Å detected during parsing, excluded from
  non-bonded energy, harmonic restraint applied (K=600 kcal/mol/Å², r₀=2.044 Å).
- **United-atom charge correction for missing hydrogens** — real PDB structures
  (X-ray, most SWISS-MODEL/AlphaFold outputs) essentially never resolve hydrogen
  positions, so `_parse_pdb` only ever creates Particles for the heavy atoms
  actually present in the file. `PARTIAL_CHARGES` holds full all-atom ff14SB
  charges (heavy atom + attached H charged separately), so when the H particle is
  never created, its charge was previously dropped on the floor — every residue's
  heavy atoms then carried a large, spurious net charge that should have been
  cancelled by a hydrogen that was never instantiated (e.g. ALA heavy-only summed
  to -0.5351 instead of its true net 0.0000; LYS heavy-only summed to -0.8815
  instead of its true net +1.0000). This corrupted electrostatics/GB for every
  atom in every structure ever analyzed, not just an edge case. Fixed in
  `amber_params.py`/`gui_main.py`: when a hydrogen that would be bonded to a
  given heavy atom (per static residue templates mirroring `bond_templates()` in
  `physics_engine.cpp`) is absent from the specific residue instance being
  parsed, its charge is folded onto its heavy-atom parent (a standard "united
  atom" approximation) — `amber_params.missing_hydrogen_charge()`. If the
  hydrogen IS present (e.g. a neutron/NMR structure with explicit H), no
  correction is applied for that atom, since its own Particle already carries
  the charge. Verified: corrected per-residue net charges now match true formal
  charges exactly (ALA → 0.0, LYS → +1.0, ASP/GLU → -1.0, etc.). Re-ran the full
  `tests/accuracy_test.py` suite (all 15 proteins, CPU+GPU, 200/1000/5000 MC
  steps) after the fix: **14/15 still pass** at the same ~2x-AlphaFold-RMSD bar
  (2CI2 is still the pre-existing N/A numbering-alignment case, not a new
  failure) — the corrected electrostatics changes per-atom energetics
  substantially but doesn't regress near-native stability on this test set,
  and it removes a real, systematic source of error from every future
  accuracy measurement.

### Coverage
- **Ligands/metals/cofactors** — HETATM no longer dropped; metal ions get proper
  tabulated parameters (`ION_PARAMS`); other ligands get element-based fallback
  (see "Remaining Work" #2 for the full fix).
- **Nucleic acid residues filtered** — DNA/RNA residues (often recorded as
  standard `ATOM` records, not `HETATM`) were being parsed as unrecognized protein
  residues, with no bond template to exclude their real covalent bonds from the
  non-bonded sum — the same hard-core blowup as the physics-model bugs above, just
  triggered by a co-crystallized nucleic acid (e.g. a SWISS-MODEL template for a
  DNA-binding protein) instead of a parsing gap. Both `_parse_pdb` (energy
  calculation) and `_aligned_pdb_text` (layered-view rendering) now filter
  `_NUCLEOTIDE_RESNAMES` the same way water is filtered.

### Analysis
- **Sequence-based disorder prediction** — `python/iupred.py`, a 20×20 statistical
  potential over a sliding window (see "Technical Notes"), computed immediately on
  parse (before any MC run) and shown alongside the trajectory-based RMSF view.
- **All-heavy-atom RMSD** — Kabsch alignment on the full heavy-atom set (not just
  Cα), catching sidechain-packing differences that Cα-only RMSD misses.
- **Real-protein accuracy validation (first bar passed, not closed — see "Remaining
  Work" #1)** — `tests/accuracy_test.py` fetches real RCSB crystal structures +
  AlphaFold predictions, establishes the AlphaFold-vs-crystal RMSD as an accuracy
  baseline, and checks whether MC sampling starting from the true native structure
  stays near-native or drifts away. **15 proteins tested (521–8200 atoms, 65–1021
  residues, 0–17 disulfides, α/β/mixed folds), 14/15 pass** — MC sampling stays
  within a fraction of an Ångström of native, well inside the AlphaFold baseline,
  across every fold type and size tested. (The one non-pass, 2CI2, is the
  numbering-alignment limitation noted above, not an accuracy failure.) Needs
  internet access; run with `python tests/accuracy_test.py`.
- **Landscape exploration branches from the best MC candidate** — `_start_landscape()`
  used to always start from the raw parsed input; it now starts from the
  lowest-energy candidate from the initial pipeline run, so exploring the
  landscape reveals what conformational sub-states are reachable from the most
  likely structure, not from the unrelaxed input coordinates.
- **Backbone knot topology classification (Alexander polynomial)** — motivated
  by a club member's own knot-theory research writeup (matching Calpha traces
  to knot types via Alexander/Jones polynomials, Khovanov homology).
  Structurally similar proteins can be topologically distinct, with real (if
  niche) biological relevance: chaperone dependence, mechanical unfolding
  resistance, disease-linked misfolding in knotted proteins. `python/knot_analysis.py`
  implements the standard protein-topology pipeline: (1) stochastic chain
  closure — extend both termini out to a distant point along independent
  random directions and join them, repeated across many trials with majority
  voting, to avoid the well-known closure-direction bias in protein knot
  detection; (2) KMT (Koniaris-Muthukumar-Taylor) triangle-elimination
  reduction directly in 3D, shrinking a closed polygon of hundreds/thousands of
  vertices down to a minimal-vertex representative of the same knot type with
  no projection-plane bias; (3) generic 2D projection + crossing detection with
  depth-based over/under assignment; (4) Wirtinger presentation of the knot
  group from the crossings, reduced to an Alexander matrix via Fox calculus;
  Alexander polynomial extracted as the determinant of an (n-1)×(n-1) minor,
  computed by evaluating at roots of unity and inverse-FFT (a naive
  Vandermonde-solve interpolation was tried first and found numerically
  unstable past ~15 crossings; roots-of-unity sampling turns this into an
  ordinary, stable DFT); (5) compared against a small table of known Alexander
  polynomials — only the unknot, trefoil (3₁), and figure-eight (4₁) are named
  with confidence; anything else is reported as "unidentified" with its
  crossing number and raw polynomial rather than risk mislabeling from an
  unverified tabulated value. Scoped to the Alexander polynomial only — the
  chirality-sensitive Jones polynomial (needs a Kauffman bracket recursion) and
  Khovanov homology (needs a fully categorified chain complex) are real,
  separate follow-up efforts, not attempted here. Wired into `gui_main.py`:
  `PipelineWorker.run()` classifies the Calpha trace's topology in the
  background thread right after parsing (a few seconds, non-blocking) and logs
  the result; stored in the `extra` dict and on `self._knot_result` for future
  UI surfacing (no dedicated panel/plot yet — currently visible via the
  existing process log). Validated three ways: synthetic parametric curves
  with known ground truth (circle/trefoil/figure-eight, all correct across 25
  random projection/closure trials each, despite the raw crossing count
  varying widely trial to trial — exactly the invariance a working topological
  invariant must show); real proteins with literature-known knot status (hen
  egg-white lysozyme 1LYZ = unknot, YibK methyltransferase 1J85 = documented
  deep trefoil, both correct at 97-100% closure-vote confidence,
  `tests/knot_test.py`); and an end-to-end smoke test through the actual
  `PipelineWorker.run()` path. Known limitation: only 3 knot types are named
  explicitly; real knotted proteins occasionally have deeper knots (5₂, 6₁,
  6₂, 6₃ documented in the literature) that this module currently reports as
  "unidentified knot" with the correct Alexander polynomial rather than a
  name — extending `KNOWN_KNOTS` needs exact tabulated Alexander polynomials
  cross-checked against a reference (e.g. KnotInfo), deferred rather than
  risking a wrong label.

### Performance
- **O(N) cell-list neighbor search** — replaces the earlier O(N²) all-pairs scan;
  3×3×3 cell neighborhood, cell side = `NL_CUTOFF + NL_SKIN`.
- **GPU-resident MC trajectories** — `run_landscape_trajectory()` runs the entire
  N-snapshot Markov chain in one C++/CUDA call instead of round-tripping
  Python↔C++ once per snapshot; device buffers stay resident for the whole chain
  instead of being reallocated every call.
- **GPU/CPU torsion-move parity** — the CUDA engine now runs the same
  torsion-angle Metropolis MC as the CPU engine (previously it ran a
  physically-invalid Cartesian move and silently ignored topology). Build- and
  run-verified on real hardware (RTX 4070, CUDA 13.2): `calculate_potential()`
  parity between engines is within 0.06% on all 11 bundled test structures.

### Release / CI infrastructure
- **GPU runtime fallback** — a GPU kernel-launch failure (as opposed to an
  import/device-detection failure at startup) used to crash the whole analysis.
  `PipelineWorker`/`LandscapeWorker` now catch engine failures specifically,
  retry on a fresh CPU engine, and downgrade to CPU for the rest of the session
  via a `gpu_fallback` signal — verified by simulating the exact failure mode
  through the real code path.
- **Fixed the release build shipping a broken CUDA extension** — root-caused a
  chain of five real bugs: a stale `protein_physics_cuda.cp312...pyd` accidentally
  committed a month earlier was masking a genuine CUDA 12.4/MSVC incompatibility
  on `windows-latest`; the build-verify step only checked that *a* file with the
  right name existed, not that it was fresh or complete. Fixed by removing the
  stale tracked artifacts, migrating CI to CUDA 13.2.0 (matching a real,
  extensively-verified local dev setup), and hardening the verify step to check
  file freshness and confirm all 4 target architectures (`sm_75/86/89/90`)
  actually compiled via `cuobjdump`. Re-verified end-to-end against a real tagged
  release (`v0.5.5`/`v0.5.6`): the shipped extension now has all 4 architectures,
  first time in at least a month.
- **Fixed HTTPS/SSL completely broken in the portable `dist/ALMA.exe`** —
  every RCSB/AlphaFold/SWISS-MODEL fetch failed with "Can't connect to HTTPS
  URL because the SSL module is not available," while the identical code
  worked fine from the unfrozen source. Root-caused by building the actual
  portable exe and testing it (not just re-running source-level scripts):
  PyInstaller's automatic dependency walker detects `_ssl.pyd` (it shows up
  in the build's own dependency graph) but does **not** detect or bundle
  `libssl-*.dll`/`libcrypto-*.dll`, the OpenSSL DLLs `_ssl.pyd` actually
  links against — confirmed by grepping the build's xref/warning output for
  both filenames and finding zero mentions. A known category of gap with
  `python-build-standalone`-style DLL layouts (this project's `.venv` is
  `uv`-managed), since PyInstaller's binary scanner is more commonly
  exercised against a standard python.org installer's layout. Fixed in
  `alma.spec` by bundling both DLLs explicitly (glob-matched from
  `sys.base_prefix/DLLs`, not a hardcoded filename, so a future OpenSSL
  version bump doesn't silently reintroduce this). Re-verified end-to-end
  against the real rebuilt exe: fetched the exact UniProt ID (P05067) that
  failed before the fix — AlphaFold API query now succeeds and proceeds
  into parsing (6209 bonds, 9 disulfides, 6091 atoms), no SSL error.

---

## Technical Notes

### Torsion move implementation
```
before MC loop:
  build covalent graph G from residue templates
  identify rotatable bonds B = {(i,j) : bond not in ring, not terminal}
  label each bond as backbone-phi, backbone-psi, or sidechain-chi

each MC step:
  pick bond (i,j) from B with weights (50% bb, 40% sc, 10% rigid)
  propose δφ ~ U[-maxδ, +maxδ]  (maxδ ≈ 5° for backbone, 30° for sidechain)
  rotate all atoms on j-side around axis (i→j) by δφ  [O(N/2) worst case]
  recompute non-bonded ΔE for moved atoms only
  Metropolis accept/reject
```

### IUPred algorithm summary
```
energy_sum(i) = Σ_{|j-i|≤10} E_pair(aa[i], aa[j])
  where E_pair is a 20×20 statistical potential (400 floats, one-time load)
disorder_score(i) = 1 / (1 + exp(a * energy_sum(i) + b))
  threshold: score > 0.5 → disordered
```

### Bond topology source
Residue template connectivity is in AMBER's `prep` files (one per residue). The
bond list for each of the 20 amino acids is static and small (10–25 bonds per
residue). Encoded directly in C++ (`bond_templates()` in `physics_engine.cpp`).

### Hard-core repulsion term (current form)
```
r_cut = HARD_CUTOFF_FRAC * sigma          (HARD_CUTOFF_FRAC = 0.6)
E = edh + egb + elj                        (always computed)
if r < r_cut:
    penalty = HARD_SCALE * [(r_cut/r)^12 - 1]^2
    E += min(penalty, HARD_CAP)            (HARD_SCALE = 1e4, HARD_CAP = 5e3)
```
Continuous (value and slope both zero) at `r = r_cut`; bounded for arbitrarily
small `r`. Both `physics_engine.cpp` and `physics_engine_cuda.cu` implement this
identically (`HARD_CAP_F` in the CUDA engine).

### References
- ff14SB: Maier et al. (2015) JCTC 11, 3696-3713
- HCT GB: Hawkins, Cramer & Truhlar (1995) J. Phys. Chem. 99, 11663
- Kabsch: Kabsch (1978) Acta Cryst. A34, 827-828
- IUPred2A: Mészáros et al. (2018) Nucleic Acids Res. 46, W329-W337
