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
- Status: TODO.

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
- Status: **DONE** — not build-verified in this session (no CUDA
  toolkit/GPU available in this environment; same caveat as prior GPU-
  touching sessions in this repo). Needs a Windows+CUDA build/run to
  confirm it compiles and to compare GPU vs CPU energies on the bundled
  `data/*.pdb` structures before this is trusted for production use.

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
      (not build-verified — no CUDA toolkit/GPU in this environment)
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
