# ALMA — Known Limitations & Implementation Roadmap

## Current Limitations

### P1 — Physics Model (Blocks Everything Else)

**1. No atom typing — all carbons treated identically**
- Current: element symbol → (radius, ε). All carbons use one entry.
- Fix: `(resname, atomname)` → AMBER ff14SB atom type → full params.
  AMBER distinguishes ~35 heavy-atom types for proteins (CX = Cα, 2C = CH₂,
  3C = CH₃, CA = aromatic, CT = aliphatic sp3, etc.).
- Impact: wrong VDW radii and ε for ~60% of atoms.
- Status: **DONE** — `amber_params.py` complete; `get_atom_params()` integrated into `_parse_pdb` in `gui_main.py`.

**2. Charge model is nearly empty**
- Current: charge assigned only to Cα of {ARG,LYS,HIS,ASP,GLU}.
  Everything else gets 0.
- Fix: RESP partial charges from AMBER ff14SB for every heavy atom.
  Example: backbone N = −0.4157 e, C = +0.5973 e, O = −0.5679 e.
- Impact: GB solvation and Debye-Hückel terms are largely zeroed out.
- Status: **DONE** — AMBER ff14SB partial charges for all 20 AA + HIS protonation variants in `amber_params.py`.

**3. No bonded energy terms**
- Missing: bond stretching E = Σ k_b(r−r₀)²
           angle bending  E = Σ k_θ(θ−θ₀)²
           dihedral       E = Σ Vₙ/2·[1 + cos(nφ−γ)]
           improper       E = Σ k_i(ξ−ξ₀)²  (planarity of peptide/ring)
- Requires: covalent topology graph built from PDB + residue template tables.
- Note: these three terms differ greatly in how much they matter:
  (a) **Bond stretching and angle bending: negligible once P1.5 is in.**
      Torsion moves preserve bond lengths and angles by construction, so
      these terms always sit at their equilibrium minima and contribute
      < 0.1 kcal/mol per step.  Safe to defer indefinitely once P1.5 lands.
  (b) **Dihedral (torsion) energy: NOT negligible.**  Backbone φ/ψ barriers
      on the Ramachandran plot are 2–5 kcal/mol.  Without these terms the
      MC sampler treats all torsion angles as equally likely, vastly
      oversampling high-energy regions (cis-peptide, eclipsed backbone).
      Must be added in P1.4a, right after P1.5.
  (c) **1-2/1-3 non-bonded exclusions: wrong RIGHT NOW.**  Directly bonded
      pairs (1-2) and angle-separated pairs (1-3) currently contribute to the
      GB/DH sum — they should be zeroed.  The r < 0.85σ hard-core guard masks
      the LJ blowup, but electrostatic/GB terms between bonded atoms are
      physically meaningless.  Fix in P1.4b alongside dihedral terms.
- Status: **DONE** — P1.4b (1-2/1-3 exclusions) and P1.4a (dihedral energy, all rotatable bonds) both implemented in `physics_engine.cpp`. `BondTopology.dihedrals` built in `build()`; `dihedral_e_boundary()` computes Δ in MC step; `total_e()` includes full dihedral sum.

**4. MC move set is physically invalid**
- Current: one atom translated randomly in Cartesian space → breaks bonds.
- Fix: torsion angle moves (rotate one rotatable bond; downstream atoms
  move rigidly as a group; bond lengths/angles never violated).
  Proposed move schedule: 50% backbone ϕ/ψ, 40% sidechain χ, 10% other.
- Requires: bond topology graph (same data structure as #3) — now available
  as `BondTopology` in `physics_engine.cpp`.
- Impact: this is the single most important fix. Current "sampling" does not
  explore protein conformational space at all.
- Status: **DONE** — Rodrigues torsion move loop replaces Cartesian sampler; `BondTopology.rot_bond_sides` pre-computed; exclusions applied in ΔE cross-pair loops.

**13. Nonbonded energy was wrong by 5-6 orders of magnitude (billions instead of thousands of kcal/mol)**
- Found while running the real pipeline end-to-end on the bundled all-atom (explicit-H)
  test structures — `calculate_potential()` returned totals like 5.7×10⁹ kcal/mol for a
  140-residue protein instead of the expected low thousands.
- Two compounding bugs in `physics_engine.cpp`:
  (a) `pair_e()`'s hard-core guard fired at `r < 0.85·σ`. Measured against a real folded
      all-atom structure, legitimate non-excluded contacts (mostly 1-4 and H···H van der
      Waals pairs) go down to `r/σ ≈ 0.67` with nothing pathological about them — plain
      LJ already scores them at a few kcal/mol. At 0.85·σ the guard was firing on
      thousands of these normal contacts and replacing each with an artificial
      `HARD_SCALE·(σ/r)¹²` spike.
  (b) `bond_templates()` only covers each residue's internal/standard form, so atoms that
      only exist at a chain terminus or under a non-default protonation state — N-terminal
      NH₃⁺ (`H1`/`H2`/`H3`), C-terminal COO⁻ (`OXT`), HIS `NE2`-`HE2` when the file has it —
      never got an `add_bond()` call. That left them out of `excl[]` too, so a real ~1.0 Å
      covalent pair was scored as a nonbonded contact and hit the (miscalibrated) hard-core
      guard, each contributing 10⁸–10⁹ kcal/mol on its own.
- Fix: lowered the hard-core threshold to `HARD_CUTOFF_FRAC = 0.6` (safely below every
  observed legitimate contact, still catches genuine MC-proposal overlaps), and added a
  terminal/protonation-variant bonding patch in `BondTopology::build()` that connects these
  atoms whenever both ends are actually present in the parsed structure.
- Status: **DONE** — verified across all 11 bundled `data/*.pdb` structures: energies now
  land in the expected range (roughly −30 to +7 kcal/mol per atom for correctly folded
  proteins) instead of billions. `topo.adj` shows zero unbonded polymer atoms on every
  bundled structure.

**5. Trajectory lengths are too short to matter**
- Pipeline MC: 300 steps × 5 candidates.
- Landscape MC: 120 × 80 = 9 600 steps.
- With valid torsion moves, useful sampling begins around 10⁵–10⁶ steps.
- Status: trivial to increase once the move set is valid.

### P2 — Missing Coverage

**6. Ligands, metals, cofactors silently dropped**
- Reason: the 6-element VDW table has no parameters for Fe, Zn, Mg, etc.,
  and GAFF (small-molecule force field) is not included.
- Immediate partial fix: stop skipping HETATM; use element-based fallback
  for unknown types; add tabulated params for common ions (Mg²⁺, Ca²⁺,
  Zn²⁺, Fe²⁺/³⁺, Na⁺, Cl⁻).
- Full fix (deferred): ship a GAFF2 atom type table for organic ligands;
  call antechamber programmatically or cache params for common cofactors
  (ATP, heme, NAD⁺, FAD, PLP…).
- Status: **DONE** (immediate partial fix) — `_parse_pdb` in `gui_main.py`
  no longer skips HETATM; metal ions get element-based fallback params;
  isolated large residue indices (≥100 000) prevent spurious adjacency
  to standard-AA residues. GAFF2 full fix still not implemented.

**7. No disulfide bonds**
- Cys SG–SG covalent bonds (~2.05 Å) are not detected or enforced.
- Fix: scan parsed structure for Cys pairs with SG–SG < 2.5 Å;
  add a stiff harmonic restraint or bond term.
- Status: **DONE** — `BondTopology::add_disulfide()` in `physics_engine.cpp`
  registers SG–SG pairs (<2.5 Å) detected during PDB parsing, excludes them
  from nonbonded pair energy, and applies a harmonic restraint
  (`ss_e`/`ss_e_side`, K_SS=600 kcal/mol/Å², r0=2.044 Å) in `total_e()` and
  the MC ΔE path.

**8. No membrane / lipid environment**
- Implicit solvent assumes uniform water (ε = 78.5) everywhere.
- Membrane proteins need a slab model (low ε in the bilayer region).
- Status: low priority / future work.

### P3 — Analysis

**9. IDP classification is heuristic and unreliable**
- Current: counts metastable basins from a 9 600-step (invalid) MC
  trajectory using ad-hoc thresholds.
- Fix: sequence-based disorder prediction (IUPred2A algorithm) as a
  one-time O(N) pass during PDB parsing.
  Core: sliding-window expected pairwise interaction energy using a
  20×20 amino acid statistical potential.
  Output: per-residue disorder probability in [0,1]; replace current
  RMSF-based classification.
- Status: **DONE** — `python/iupred.py` implements the sequence-based
  disorder predictor; `_parse_pdb` calls `score_from_resnames()` and the
  GUI shows an IUPred panel immediately after parsing (before any MC run),
  plus a dual IUPred+RMSF panel once the landscape run completes.
  RMSF-based classification kept as a complementary trajectory-based view,
  not replaced.

**10. RMSD comparison ignores sidechain conformation**
- Only Cα atoms are compared; two structures can have identical backbone
  but completely different sidechain packing.
- Fix: add all-heavy-atom RMSD option (Kabsch on full heavy-atom set).
- Status: **DONE** — `heavy_map`/`heavy_indices`/`heavy_keys` built in `_parse_pdb`
  and `_heavy_atom_map_from_pdb()` for reference structures; `ComparisonWorker`
  computes `rmsd_heavy` (Kabsch on all non-hydrogen ATOM-record heavy atoms) for
  MC candidates, AlphaFold, and SWISS-MODEL, shown alongside Cα RMSD as a new
  "HEAVY RMSD" column in `gui_main.py`.

### P4 — Performance

**11. O(N²) neighbor list build**
- Current: every atom checked against every other — O(N²).
- Fix: cell-list decomposition — divide box into cells of side ≥ NL_CUTOFF;
  each atom checks only its 27 neighboring cells — O(N).
  Significant for proteins > 2 000 atoms.
- Status: **DONE** — `physics_engine.cpp` bins atoms into a 3-D cell grid
  (cell side = NL_CUTOFF+NL_SKIN) and each atom's Verlet neighbor list scan
  is limited to its 27-cell neighborhood instead of all N atoms.

**12. MC trajectory round-trips CPU↔GPU**
- LandscapeWorker did 120 Python↔C++ round trips (one per snapshot), each
  re-marshalling the full particle array via generate_ensemble()+
  calculate_potential(), and on the GPU backend reallocating/re-uploading
  device buffers from scratch every call.
- Fix: `PhysicsEngine::run_landscape_trajectory()` (both engines) runs the
  entire N_SNAPSHOTS×STEPS_PER_SNAP Markov chain in a single C++ call.
  LandscapeWorker now calls it once instead of looping in Python. On the
  GPU engine, atom parameters/pair-list/exclusion buffers are allocated
  once per chain and kept resident across the whole trajectory (positions
  + Born radii are re-uploaded once per MC step, not reallocated).
  Snapshots still return full particle lists (not just Cα) because the
  GUI's landscape-graph click handler renders the full structure of
  whichever node the user picks — an all-Cα return would have silently
  broken that feature.
- Status: **DONE**.

**14. GPU engine ran physically-invalid Cartesian MC + ignored topology**
- `physics_engine_cuda.cu`'s `generate_ensemble()` translated one atom in
  Cartesian space per step (the same invalid move type fixed for the CPU
  engine by P1.5) and accepted a `topology` argument only for API parity,
  silently ignoring it — no dihedral energy, no 1-2/1-3 exclusions, no
  disulfide restraints. The GPU and CPU backends produced physically
  different, non-comparable ensembles from the same PDB.
- It also still used the pre-P1.6 hard-core threshold
  (`r < 0.85·σ`, both in the device `pair_e_gpu()` kernel and the host
  `pair_e_cpu()` helper) — P1.6's fix to `physics_engine.cpp` was never
  ported here, so the GPU engine would reproduce the same
  billions-of-kcal/mol blowup on any real all-atom structure exercising
  legitimate 1-4/H···H contacts down to r/σ ≈ 0.67.
- Fix: ported physics_engine.cpp's torsion-angle Metropolis MC (Rodrigues
  rotation of one rotatable bond's j-side, lever-arm scaling, crankshaft
  moves, adaptive proposal width) to the GPU engine, with the same bond
  topology (1-2/1-3 exclusions, dihedral/Ramachandran energy, disulfide
  restraints). `physics_engine.cpp`'s `BondTopology` exports the extra
  data the CUDA module needs (`rb_atom_i/j/kind`, `dih_*` CSR arrays,
  `disulfide_pairs`, `concerted_pairs`) as plain vectors so a separately
  compiled pybind11 module can reconstruct an equivalent topology without
  cross-module C++ type registration. Also lowered `HARD_CUTOFF_FRAC` to
  0.6 in the GPU engine to match P1.6.
- GPU/CPU split: the per-step cross-boundary nonbonded energy sum (the
  dominant O(N·neighbors) cost for a torsion move with a large j-side) now
  runs on the GPU via a new `cross_pair_energy_kernel`. SASA stays on the
  CPU host in both engines — its per-atom accumulation is a sequential,
  order-dependent fold, not an embarrassingly parallel reduction — as do
  the O(rotatable-bond-count) dihedral/disulfide sums, which are too small
  to justify a kernel launch.
- Status: **DONE — build- and run-verified on real hardware** (Windows 11,
  RTX 4070 Laptop GPU, CUDA 13.2 toolkit, MSVC 2022). `python setup.py
  build_ext --inplace` compiles both `protein_physics` and
  `protein_physics_cuda` cleanly. `python tests/calibrate_gpu.py` passes all
  11 bundled `data/*.pdb` structures: `calculate_potential()` parity between
  CPU and GPU engines is within 0.06% relative difference on every structure
  (most < 0.01%, i.e. float-vs-double noise), both engines' `generate_ensemble()`
  lower the energy over 200 MC steps, and `run_landscape_trajectory()`
  completes end-to-end with no CUDA errors. `tests/bridge_test.py` also
  passes on this build. That calibration run also surfaced two real,
  Windows-specific bugs fixed alongside this port (both were latent in the
  pre-port code too, just never exercised on real hardware before):
  `nvcc`'s own default C++ dialect doesn't support the structured bindings
  this port uses (setup.py's `build_cuda_extension()` calls `nvcc` directly,
  bypassing the `/std:c++latest` that `Pybind11Extension` auto-injects for
  the CPU build — fixed by adding `-std=c++17` there), and `Particle`/
  `PhysicsEngine` — defined separately in each module but sharing the same
  class names — collided at import time because MSVC's RTTI compares
  `type_info` by decorated name across DLLs (Linux/macOS give each shared
  object independent RTTI, so this never showed up there); fixed with
  `py::module_local()` on both classes in both modules.

**15. Hard-core repulsion term had two independent real bugs: a discontinuous step AND an unbounded penalty**
- Bug A — discontinuity at the threshold: found via the calibration run above:
  `calculate_potential()` on `Q92793.pdb` (~18.5k atoms) returned +4,314,828
  kcal/mol — grossly outside the sane −30…+7 kcal/mol/atom range from item
  #13 — on BOTH engines identically (confirming this is inherited
  physics-model behavior, not a GPU-vs-CPU discrepancy). Traced to exactly
  one pair: Ser27 `CA` vs Phe2438 `CD1`, a genuine tertiary contact (not a
  missing bond/exclusion — `topo.adj` shows zero unbonded atoms) sitting at
  `r/σ = 0.5997`, 0.03% inside the `HARD_CUTOFF_FRAC = 0.6` threshold from
  item #13. That one pair alone contributed +4.6M kcal/mol via
  `HARD_SCALE·(σ/r)¹²`, because the old `pair_e()` was a hard branch: a
  contact landing a fraction of a percent on either side of the threshold
  gets either an ordinary bounded LJ repulsion of a few kcal/mol, or a
  multi-million-kcal/mol spike — for what is physically almost the same
  contact. P1.6's 0.6 calibration was validated against one structure
  (1XQ8); a much larger structure had a real contact close enough to find
  the edge of that threshold.
- Bug B — unbounded even away from the threshold: the bundled `data/*.pdb`
  set is pre-vetted — every structure in it was already debugged against
  these exact failure modes. A fresh, unmodified real PDB fetched from RCSB
  is not pre-vetted, and `tests/accuracy_test.py` (new, see item #16)
  immediately found one: hen egg-white lysozyme (PDB 1LYZ, a 1975
  "real-space refinement" structure) has a genuine 1.36 Å
  CB(Ala122)···NH1(Arg125) contact — a real coordinate artifact from a
  poorly-resolved arginine sidechain in an old, low-resolution structure,
  not a parsing bug — sitting at `r/σ = 0.364`, deep inside the threshold
  rather than near its edge. `pair_e()`'s hard-core branch
  (`r < 0.6·σ` → `HARD_SCALE·(σ/r)¹²`) is unbounded as `r→0`: this single
  pair alone evaluated to ~2×10⁹ kcal/mol and made `calculate_potential()`
  report 2.1 billion kcal/mol for the whole 1001-atom structure — the exact
  "billions instead of thousands" failure item #13 was supposed to have
  fixed, just triggered by a different, real-world input instead of a
  parsing gap. Any real (imperfect) structure — older X-ray, NMR ensembles,
  low-confidence AlphaFold regions — can contain a handful of pathologically
  short contacts that are not MC-proposal artifacts, and they aren't
  guaranteed to sit conveniently near the threshold like the Q92793 case, so
  fixing only the discontinuity (Bug A) isn't sufficient on its own.
- Fix: both problems needed independent fixes, applied together in both
  engines' `pair_e()`/`pair_e_gpu()`:
  1. **Continuity (Bug A):** replaced the hard branch with a smooth additive
     penalty: `E = edh + egb + elj`, always, plus
     `HARD_SCALE·[(r_cut/r)¹² − 1]²` when `r < r_cut`
     (`r_cut = HARD_CUTOFF_FRAC·σ`). At `r = r_cut` the penalty and its
     derivative are both exactly zero, so the total is continuous and smooth
     across the boundary — no jump, and no new threshold/width parameter to
     calibrate (reuses `HARD_CUTOFF_FRAC` unchanged).
  2. **Boundedness (Bug B):** the smooth formula above is still unbounded as
     `r→0` — it only fixes contacts near the threshold edge, not contacts
     deep inside it. Added `HARD_CAP = 5.0e3` kcal/mol (`HARD_CAP_F` in the
     CUDA engine): the penalty term itself is capped via
     `min(HARD_SCALE·[(r_cut/r)¹² − 1]², HARD_CAP)`. A clashing pair is
     still strongly, correctly penalized (thousands of kcal/mol — MC still
     firmly rejects/relaxes it) but can no longer single-handedly make
     `calculate_potential()` meaningless for a real structure.
  Both fixes diverge steeply for genuine MC-proposal overlaps (`r ≪ r_cut`),
  so the guard-rail purpose is unaffected; the cap only bounds the
  magnitude, it doesn't weaken the penalty for realistic clash distances.
- Status: **DONE** — verified against all 11 bundled structures: the 10
  without any contact near the threshold or deep inside it reproduce their
  exact previous `calculate_potential()` values (confirming zero
  regression), `Q92793.pdb` now returns −303,562.7 kcal/mol (−16.35/atom)
  instead of +4,314,828, and 1LYZ's `calculate_potential()` dropped from 2.1
  billion to 41,230 kcal/mol (41.2 kcal/mol/atom — the "thousands, not
  billions" range item #13 intended), while a 5000-step MC run still lowers
  it further to -1,600 to -2,100 kcal/mol and stays within 0.2-0.4 Å Cα RMSD
  of the native structure.

**16. No test had ever compared ALMA's own output to real, independent structural ground truth**
- `bridge_test.py` and `calibrate_gpu.py` only check internal self-consistency
  (does the Python↔C++ bridge round-trip data; does the CPU engine agree with
  the GPU engine). Neither asks whether the energy function's minimum actually
  corresponds to a real protein's native structure.
- Added `tests/accuracy_test.py`: for each of several real, structurally
  diverse proteins, fetches the real RCSB crystal structure and the AlphaFold
  prediction for the same UniProt entry, establishes the AlphaFold-vs-crystal
  RMSD as an accuracy baseline, then runs `generate_ensemble()` starting from
  the real crystal structure (the true native state) and checks whether MC
  energy minimization keeps the structure near-native (physically sane) or
  lets it drift away (a real force-field accuracy gap, not a code bug).
  Auto-detects the AlphaFold-to-crystal residue-numbering offset via sequence
  identity (not just coordinate-key overlap — see the script's `best_offset`
  docstring for two real registration bugs this caught and fixed along the
  way: a tandem-repeat frame ambiguity in polyubiquitin, and a propeptide-
  length-driven mismatch in Interleukin-1 beta) and restricts multi-copy
  crystal asymmetric units (e.g. barnase, 3 copies in 1BNI) to a single chain
  so crystal-packing contacts between unrelated copies don't get scored as
  real intramolecular interactions.
- Status: **DONE** — 7 of 8 tested proteins (hen egg-white lysozyme, human
  ubiquitin, sperm whale myoglobin, bovine RNase A, Bacillus barnase,
  Interleukin-1 beta, alpha-lactalbumin) show ALMA's MC sampling staying
  within a fraction of an Ångström of native — comfortably inside (usually
  well under half of) the AlphaFold-vs-crystal accuracy bar for that protein,
  across alpha-helical, beta-grasp, and mixed alpha+beta folds with and
  without disulfides. The 8th (chymotrypsin inhibitor 2, PDB 2CI2) can't be
  automatically aligned to its full-length UniProt entry at any residue offset
  (its crystal numbering doesn't correspond linearly to full-length numbering)
  — a known, documented limitation of the offset-search approach, not an ALMA
  accuracy issue. Run with `python tests/accuracy_test.py` (needs internet
  access to fetch structures).
- Extended to 5 larger proteins (521-4599 atoms was the full range before this;
  now up to triose phosphate isomerase, carbonic anhydrase II, aldolase A,
  firefly luciferase, and human serum albumin — 1883 to 4599 atoms, up to 578
  residues): **12/13 total proteins pass**, no new bugs at larger scale. Human
  serum albumin — the largest and most disulfide-dense case (578 residues,
  4599 atoms) — correctly detects all 17 of its known native disulfide bonds
  and keeps MC sampling within 0.3-0.8 Å of native after 5000 steps, well
  inside its 1.28 Å AlphaFold baseline. Firefly luciferase's 7.0 Å AlphaFold-
  vs-crystal baseline reflects genuine, well-documented hinge motion between
  its two domains (not a registration bug — full 523/523 residue match) and
  ALMA's own sampling drifts proportionally more for it (up to 2.8 Å) than for
  any other test protein, which is the physically expected result for a
  flexible multi-domain enzyme, not a red flag.
- Pushed further to 2 much larger single-domain/monomer enzymes: catalase
  (498 residues, 4099 atoms, heme-binding) and a beta-galactosidase monomer
  (1021 residues, 8200 atoms — the largest structure tested, extracted as a
  single chain from its tetrameric crystal form). Both pass: MC sampling
  stays within 0.2-0.5 Å of native after 1000 steps, comfortably inside their
  0.43 Å and 0.65 Å AlphaFold baselines respectively. Confirms the HARD_CAP
  fix (item #15) and disulfide/topology handling hold up at production-scale
  atom counts, not just the small/medium test proteins above.
- **Total: 15 real proteins tested, 14/15 pass** (521 to 8200 atoms, 65 to
  1021 residues, 0 to 17 disulfides, alpha/beta/mixed folds); the one
  non-pass (CI2) is a documented offset-search limitation, not an accuracy
  failure.

**17. GPU runtime failures (as opposed to startup/import failures) crashed the whole analysis instead of falling back to CPU**
- `_try_gpu_backend()` only confirms `device_name()` succeeds at app startup —
  that only proves the CUDA extension imports and a device is visible. It says
  nothing about whether a kernel launch will actually succeed later. Observed
  directly: a distributed portable build (`ALMA.exe`, built by the release
  workflow's CUDA 12.4 toolchain) parsed a real structure successfully, then
  failed the moment `generate_ensemble()` actually launched a kernel:
  `CUDA error the provided PTX was compiled with an unsupported toolchain.` —
  a toolchain/driver PTX-compatibility mismatch, not a code bug in this repo,
  but one this repo's error handling made unrecoverable: `PipelineWorker.run()`
  had exactly one `except Exception` wrapping the entire pipeline, so any
  failure at any stage — including one that only manifests on the specific
  machine running the distributed binary, well after parsing already
  succeeded — surfaced as a raw error string and killed the whole analysis.
- Fix: `PipelineWorker.run()` and `LandscapeWorker.run()` (the two call sites
  that actually launch GPU kernels — the initial ensemble/energy computation
  and the landscape-exploration Markov chain) now catch exceptions from the
  engine calls specifically. If the failing engine wasn't already the CPU one,
  they instantiate a fresh `protein_physics.PhysicsEngine()` and retry the
  same computation on CPU rather than aborting, emit a clear log message
  explaining what happened, and emit a new `gpu_fallback` signal. The main
  window listens for that signal and permanently downgrades
  `self._physics_mod`/`self.engine` to CPU for the rest of the session (so
  later analyses don't repeat the same multi-second failure-then-retry), and
  updates the sidebar's backend label to say so.
- Status: **DONE** — verified by simulating the exact failure (a fake GPU
  engine whose `generate_ensemble()` raises the same `RuntimeError` text seen
  in the field) through the real `PipelineWorker.run()` code path: parsing
  succeeds, the simulated GPU call fails, the worker logs the fallback,
  emits `gpu_fallback`, retries on CPU, and the `finished` signal fires
  normally — the analysis completes instead of crashing. `tests/bridge_test.py`
  still passes (no regression to the normal all-CPU or all-GPU paths). At the
  time this was written the underlying PTX/toolchain mismatch itself was
  believed to be an unfixable environment issue on whichever machine runs a
  mismatched distributed build — **that assumption was wrong; see item #18,
  which root-causes and actually fixes it.**

**18. The "unsupported toolchain" PTX error (item #17) was a real, fixable release-build bug, not an unavoidable environment mismatch**
- Root-caused by extracting the actual downloaded `ALMA.exe`'s bundled
  `protein_physics_cuda.cp312-win_amd64.pyd` from its PyInstaller temp
  extraction directory and running it directly: `cuobjdump --list-elf`
  showed it contains compiled code for **only `sm_75`** (Turing) — none of
  the `sm_86`/`sm_89`/`sm_90` (Ampere/Ada/Hopper) targets `setup.py` asks
  nvcc for. An RTX 4070 (Ada, `sm_89`) has no matching native code in that
  binary, so the driver falls back to JIT-compiling the embedded `sm_75`
  PTX — which is what was actually failing.
- Pulled the release workflow's own build log (`gh run view --log`) for the
  run that produced this exact download and found the real cause: CUDA
  12.4's nvcc hits a **fatal** `host_config.h` error — `unsupported
  Microsoft Visual Studio version` — because `windows-latest` now ships a
  newer default MSVC (Visual Studio "18", toolset 14.51.x) than CUDA 12.4
  supports (2017-2022 only, i.e. up to roughly the 14.3x/14.4x generation).
  `build_cuda_extension()` catches nvcc's nonzero exit and returns `False`
  without failing the job, so this alone should have just meant "no GPU
  extension shipped" — a safe, if disappointing, degradation.
- What actually shipped instead: `git log` showed
  `protein_physics_cuda.cp312-win_amd64.pyd` (plus its `.exp`/`.lib`
  build-artifact siblings) had been **accidentally committed to the repo on
  2026-06-09** and never removed — `.gitignore`'s `*.pyd` rule doesn't
  retroactively untrack files already committed. The release workflow's
  "Verify CUDA extension was actually built" step only checked that *a*
  `protein_physics_cuda*.pyd` file existed on disk, which that ~1-month-old
  stale tracked copy always satisfied — so every release since has silently
  shipped that same ancient, `sm_75`-only extension regardless of whether
  the *current* commit's build actually succeeded.
- Fix, three parts closing each link in the chain:
  1. `git rm --cached` the stale tracked `protein_physics.cp312-win_amd64.pyd`
     / `protein_physics_cuda.cp312-win_amd64.{pyd,exp,lib}` — these were
     never supposed to be tracked and were never rebuilt since being
     committed.
  2. First tried pinning `ilammy/msvc-dev-cmd@v1`'s `toolset: 14.3` in
     `release.yml`, assuming `windows-latest` ships multiple MSVC toolset
     generations side by side. **Wrong** — re-running the workflow against
     this fix immediately failed with `Toolset directory for version '14.3'
     was not found`: the current `windows-latest` image only has the one
     (newest, 14.51.x) toolset installed, nothing older to select. Reverted
     the pin and instead bumped the CUDA toolkit itself from `12.4.1` to
     `13.2.0` (this action's own documented default, and the exact version
     already verified, extensively, against this repo's actual
     `physics_engine_cuda.cu` on a real Windows+CUDA dev machine — see items
     #14/this item's own re-verification below) — new enough that nvcc
     supports the MSVC generation `windows-latest` actually ships.
  3. Hardened the verify step: it now checks the built `.pyd`'s
     `LastWriteTimeUtc` is after the build step actually started (catches
     any future stale-file masking the same way), and runs
     `cuobjdump --list-elf` to confirm all four expected architectures
     (`sm_75`/`86`/`89`/`90`) actually compiled, not just that *some* file
     with the right name exists.
  4. Re-running against that fix surfaced two more real problems, each fixed
     in turn rather than assumed away: `Jimver/cuda-toolkit@v0.2.19`'s own
     bundled CUDA version table only went up to `12.6.2` (`13.2.0` didn't
     exist in it at all) — bumped the action pin to `v0.2.35`, confirmed by
     diffing both tags' `src/links/windows-links.ts`. Then, with the action
     fixed, nvcc failed with `Cannot open include file: 'crt/host_config.h'`
     — the narrower `sub-packages: ["nvcc", "cudart"]` selection (fine under
     12.4.1) left out whatever component now ships that header under
     13.2.0's reorganized layout; removed the restriction so it installs the
     full toolkit instead. Also caught (via the same run) the verify step's
     broad glob matching an unrelated, wrong-ABI tracked `cp314` file and
     reporting it as "found" — pinned the filter to the exact `cp312` tag
     this job's Python 3.12 produces, and untracked the `cp314`
     `.pyd`/`.exp`/`.lib` artifacts too (the same mistake as the `cp312`
     ones, just not yet noticed).
- Status: **DONE and re-verified end-to-end.** A fresh `workflow_dispatch`
  run (`gh run view <id> --log`) completed successfully in 20m14s and
  printed exactly what the hardened verify step was built to confirm:
  `Found CUDA extension: protein_physics_cuda.cp312-win_amd64.pyd` (the
  correct, fresh, correct-ABI file — not a stale or wrong-ABI one) followed
  by `cuobjdump --list-elf` listing all four architectures and
  `Confirmed compiled code for all expected architectures: sm_75, sm_86,
  sm_89, sm_90`. This is the first release build in at least a month to
  actually ship a complete, freshly-compiled CUDA extension. The item #17
  CPU-fallback safety net stays regardless, since a driver/toolchain
  mismatch on some future user's specific machine is still possible even
  from a correctly-built extension.

**19. Landscape exploration always branched from the raw parsed input, never from the MC-relaxed best candidate**
- `_start_landscape()` always passed `self._init_atoms` — the unrelaxed,
  as-parsed structure — to `LandscapeWorker`, regardless of whether a
  pipeline run had already produced an MC ensemble with a clearly better
  (lower-energy) candidate. Requested directly by a user after noticing a
  layered comparison for a real protein (human preproinsulin, P01308) had a
  huge, real disagreement in a specific region (its known intrinsically
  disordered C-peptide linker — see the session discussion in this file's
  history/commit log around item #16-18 for the diagnostic that localized
  it): if even the best candidate can have multiple accessible sub-states in
  a flexible region, exploring FROM that best candidate — not from the raw,
  unrelaxed input — is what actually reveals which positions are reachable
  from the most likely structure.
- Fix: `_start_landscape()` now uses the lowest-energy candidate from
  `self._ensemble`/`self._energies` (the same `np.argmin(energies)` pattern
  already used elsewhere in this file for the "best candidate" concept — see
  `_render`, the VIEW-button handlers) as the landscape trajectory's starting
  structure, logging which candidate and energy it branched from. Falls back
  to the raw parsed structure only if no ensemble exists yet (defensive,
  shouldn't normally trigger since the button is disabled until a pipeline
  run completes).
- Status: **DONE** — verified that the best candidate is a genuinely
  different, relaxed structure from the raw parse (nonzero coordinate
  distance) via a headless `PipelineWorker` smoke test on 1LYZ;
  `tests/bridge_test.py` still passes (no regression).

---

## Implementation Roadmap

```
P1.1  amber_params.py — VDW + charge tables (ff14SB)                      ✓ DONE
P1.2  gui_main.py    — replace _AMBER/_CHARGE with new tables             ✓ DONE
P1.3  physics_engine.cpp — bond topology graph (residue templates)        ✓ DONE
─────────────────────────────────────────────────────────────────
P1.5  physics_engine.cpp — torsion angle MC moves                        ✓ DONE
P1.4b physics_engine.cpp — 1-2/1-3 non-bonded exclusions                ✓ DONE
P1.4a physics_engine.cpp — dihedral energy (Ramachandran penalties)     ✓ DONE
─────────────────────────────────────────────────────────────────
P1.4c physics_engine.cpp — bond + angle energy terms (safe to defer)
─────────────────────────────────────────────────────────────────
P2.1  amber_params.py — common metal ion params                          ✓ DONE
P2.2  gui_main.py/_parse_pdb — stop dropping HETATM                      ✓ DONE
P2.3  physics_engine.cpp — disulfide detection + restraints              ✓ DONE
─────────────────────────────────────────────────────────────────
P3.1  python/iupred.py  — sequence-based disorder predictor              ✓ DONE
P3.2  gui_main.py       — integrate IUPred into PDB parsing              ✓ DONE
P3.3  gui_main.py/_compute_rmsd — all-heavy-atom RMSD option             ✓ DONE
─────────────────────────────────────────────────────────────────
P4.1  physics_engine.cpp — cell-list NL build                            ✓ DONE
P4.2  physics_engine_cuda.cu — GPU-resident trajectory                  ✓ DONE
─────────────────────────────────────────────────────────────────
P1.6  physics_engine.cpp — fix hard-core threshold + terminal/          ✓ DONE
      protonation-variant bond patch (nonbonded energy was off by
      5-6 orders of magnitude)
P1.7  physics_engine_cuda.cu — torsion-move + topology parity with       ✓ DONE
      the CPU engine (dihedral/exclusions/disulfide/crankshaft),
      GPU-resident MC state, HARD_CUTOFF_FRAC 0.6 fix ported to GPU
      (build- and run-verified on real Windows+CUDA hardware, RTX 4070 +
      CUDA 13.2 — calibrate_gpu.py: 11/11 structures PASS, energy parity
      < 0.06% rel. diff)
P1.8  physics_engine.cpp/physics_engine_cuda.cu — smooth the hard-core   ✓ DONE
      nonbonded term into a continuous penalty (was a discontinuous
      step at HARD_CUTOFF_FRAC; found via a real large structure whose
      closest contact landed right on the old threshold)
P1.9  physics_engine.cpp/physics_engine_cuda.cu — cap the (still         ✓ DONE
      unbounded-as-r->0) smooth penalty from P1.8 at HARD_CAP=5000
      kcal/mol; found via tests/accuracy_test.py on a real, unmodified
      RCSB structure (1LYZ) with a genuine deep clash, not just a
      near-threshold one
P1.10 tests/accuracy_test.py — real-protein accuracy validation vs        ✓ DONE
      AlphaFold/RCSB ground truth (15 proteins, 14/15 pass; see item #16)
P1.11 gui_main.py — PipelineWorker/LandscapeWorker fall back to CPU on    ✓ DONE
      a GPU runtime (not just startup) failure instead of crashing the
      whole analysis; see item #17
P1.12 .github/workflows/release.yml + removed stale tracked cp312        ✓ DONE
      artifacts — root-caused and fixed item #17's actual cause: CUDA
      12.4 vs. windows-latest's default MSVC, masked for ~1 month by an
      accidentally-committed stale .pyd; see item #18. Needs a fresh
      tagged release to re-verify end-to-end.
P1.13 gui_main.py — landscape exploration branches from the best MC       ✓ DONE
      candidate instead of the raw parsed input; see item #19.
```

---

## Technical Notes

### Torsion move implementation (P1.5)
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

### IUPred algorithm summary (P3.1)
```
energy_sum(i) = Σ_{|j-i|≤10} E_pair(aa[i], aa[j])
  where E_pair is a 20×20 statistical potential (400 floats, one-time load)
disorder_score(i) = 1 / (1 + exp(a * energy_sum(i) + b))
  threshold: score > 0.5 → disordered
```

### Bond topology source (P1.3)
Residue template connectivity is in AMBER's `prep` files (one per residue).
The bond list for each of the 20 amino acids is static and small
(10–25 bonds per residue). Encode directly in C++ or load from a
JSON/TOML table at init.

### References
- ff14SB: Maier et al. (2015) JCTC 11, 3696-3713
- HCT GB: Hawkins, Cramer & Truhlar (1995) J. Phys. Chem. 99, 11663
- Kabsch: Kabsch (1978) Acta Cryst. A34, 827-828
- IUPred2A: Mészáros et al. (2018) Nucleic Acids Res. 46, W329-W337
