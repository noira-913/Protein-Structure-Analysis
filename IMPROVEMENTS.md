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

**3. Bond stretching + angle bending energy terms (P1.4c)**
Torsion moves preserve bond lengths/angles by construction, so these terms sit at
their equilibrium minima and contribute < 0.1 kcal/mol/step — safe to defer
indefinitely unless a future move type (e.g. bond-length perturbation) needs them.

**4. Full GAFF2 force field for organic ligands/cofactors**
Currently: HETATM ligands get an element-based fallback (radius/ε by element only,
zero partial charge) rather than real small-molecule parameters. A full fix means
shipping a GAFF2 atom-type table and either calling `antechamber` programmatically
or caching parameters for common cofactors (ATP, heme, NAD⁺, FAD, PLP…). Metal ions
already have proper tabulated parameters (`amber_params.ION_PARAMS`) — only organic
ligands are affected.

**5. Membrane/lipid slab model**
Implicit solvent assumes uniform water (ε=78.5) everywhere. Membrane proteins need
a low-ε bilayer-region model; no such model exists yet. Low priority — no membrane
protein test cases in current use.

*(Also: `tests/accuracy_test.py`'s auto-offset alignment can't handle a PDB entry
whose crystal numbering doesn't correspond linearly to full-length UniProt numbering
at any single offset — hit once, for chymotrypsin inhibitor 2 (2CI2). This is a
limitation of that test harness's alignment heuristic, not the physics engine.)*

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
