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
- Full fix: ship a GAFF2 atom type table for organic ligands; call
  antechamber programmatically or cache params for common cofactors
  (ATP, heme, NAD⁺, FAD, PLP…).
- Status: TODO.

**7. No disulfide bonds**
- Cys SG–SG covalent bonds (~2.05 Å) are not detected or enforced.
- Fix: scan parsed structure for Cys pairs with SG–SG < 2.5 Å;
  add a stiff harmonic restraint or bond term.
- Status: TODO.

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
- Status: TODO.

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
- LandscapeWorker does 120 Python↔GPU transfers (one per snapshot).
- Fix: keep entire particle array on GPU device memory; only transfer Cα
  coordinate snapshots (tiny) to host.
- Status: TODO — requires CUDA kernel changes.

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
P2.1  amber_params.py — common metal ion params
P2.2  gui_main.py/_parse_pdb — stop dropping HETATM
P2.3  physics_engine.cpp — disulfide detection + restraints
─────────────────────────────────────────────────────────────────
P3.1  python/iupred.py  — sequence-based disorder predictor
P3.2  gui_main.py       — integrate IUPred into PDB parsing
─────────────────────────────────────────────────────────────────
P4.1  physics_engine.cpp — cell-list NL build
P4.2  physics_engine_cuda.cu — GPU-resident trajectory
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
