# ALMA — Atomistic Local Motion Analyzer

An implicit-solvent protein physics engine with a PyQt6 desktop front end.
ALMA fetches a protein structure (by PDB ID or UniProt accession), scores
and samples its conformational space with a Metropolis Monte Carlo engine
written in C++, and visualizes the resulting ensemble, energy landscape,
and disorder profile.

The physics core (`protein_physics`) is a pybind11 extension: an
OpenMP-parallelized CPU implementation with an optional CUDA GPU backend.
The GUI (`gui_main.py`) never talks to raw floats — it hands PDB-derived
particle lists to the C++ engine and gets back energies, ensembles, and
trajectories.

---

## Table of Contents

- [Background](#background)
- [Theoretical Foundation](#theoretical-foundation)
  - [The Energy Model](#the-energy-model)
  - [Monte Carlo Sampling](#monte-carlo-sampling)
  - [Bond Topology and Move Validity](#bond-topology-and-move-validity)
  - [Force Field Parameters](#force-field-parameters)
  - [Disulfide Bonds](#disulfide-bonds)
  - [Sequence-Based Disorder Prediction](#sequence-based-disorder-prediction)
  - [Structural Comparison](#structural-comparison)
  - [Backbone Knot Topology Classification](#backbone-knot-topology-classification)
- [Architecture](#architecture)
- [Application Workflow](#application-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Project Layout](#project-layout)
- [Performance Notes](#performance-notes)
- [Known Limitations](#known-limitations)
- [References](#references)

---

## Background

Proteins are linear chains of amino acids that spontaneously fold into a
compact, specific three-dimensional structure — the **native fold** — in
aqueous solution. This is one of the central problems of structural
biology: the folded structure determines almost everything about a
protein's function, and misfolding is implicated in diseases ranging from
Alzheimer's to cystic fibrosis.

Folding is thermodynamically driven: the native structure is
(approximately) the global minimum of the protein's **free energy** as a
function of atomic coordinates. Computing that free energy exactly
requires modeling every water molecule, every ion, and quantum-mechanical
electronic structure — computationally intractable for anything but the
smallest systems. **Implicit-solvent models** sidestep this by replacing
the explicit water bath with a continuum dielectric medium and a
surface-area-dependent hydrophobic term. This is the approach ALMA takes:
it is not a full molecular dynamics package, but an approximate,
fast-to-evaluate scoring function paired with a physically-motivated
Monte Carlo sampler, aimed at conformational exploration and comparative
analysis rather than picosecond-accurate trajectories.

The motivating use case: given a folded structure (crystallographic PDB
entry or an AlphaFold prediction), how much do local backbone/sidechain
torsions matter to its stability, how does it compare energetically and
geometrically to other structural predictions of the same sequence, and
does its energy landscape look like an ordered globular protein or an
intrinsically disordered one?

---

## Theoretical Foundation

### The Energy Model

ALMA scores a conformation as the sum of four additive terms (all
energies in kcal/mol, distances in Å):

**1. Generalized Born electrostatics (GB/HCT) — polar solvation**

Water has a high dielectric constant (ε ≈ 78.5) that screens
electrostatic interactions between charged/polar atoms; the protein
interior is treated as a low-dielectric medium (ε ≈ 1). The GB model
estimates the free energy released when a partial charge moves from the
low-dielectric interior into the high-dielectric solvent, as a function
of how deeply that atom is buried. ALMA uses the **HCT (Hawkins, Cramer &
Truhlar, 1995)** formulation of effective Born radii, which shrink as an
atom becomes more surrounded by other atoms (more "buried").

**2. Debye–Hückel screened Coulomb (DH) — ionic screening**

At physiological ionic strength (~150 mM NaCl), mobile ions in solution
cluster around charged residues and screen long-range Coulomb
interactions. This is modeled with an exponential screening factor
`exp(-κr)` on top of the bare Coulomb term, where `κ⁻¹` (the Debye
length) is on the order of 8 Å at physiological salt concentration.

**3. Lennard-Jones 12-6 potential — van der Waals**

Combines short-range steric repulsion (`r⁻¹²`, preventing atomic overlap)
with longer-range London dispersion attraction (`r⁻⁶`). Per-atom σ
(combined VDW radius) and ε (well depth) are looked up from **AMBER
ff14SB** atom-type tables (see [Force Field Parameters](#force-field-parameters)).
A hard-core guard is applied when `r < 0.85σ` to prevent numerical
blow-up during large-scale MC proposals.

**4. SASA nonpolar term — hydrophobic burial**

Nonpolar surface area exposed to solvent carries an energetic penalty
proportional to solvent-accessible surface area (SASA), using a
surface-tension-style linear model (`γ·SASA + β`). This is the dominant
thermodynamic driver of hydrophobic core formation — nonpolar sidechains
"want" to bury themselves away from water, which is a major part of why
proteins fold into compact globules rather than remaining extended.

**5. Disulfide restraint (when present)**

Covalently bonded cysteine pairs (Cys SG–SG, detected at parse time when
the SG–SG distance is under 2.5 Å) are modeled with a stiff harmonic
restraint, `E = K_SS·(r − r₀)²`, `K_SS = 600 kcal/mol/Å²`,
`r₀ = 2.044 Å` (the AMBER ff14SB CYX equilibrium bond length). This bond
is excluded from the nonbonded (GB/DH/LJ) pair sum, since it is a
covalent, not nonbonded, interaction.

**6. Dihedral (torsion) energy**

Backbone φ/ψ and sidechain χ dihedrals carry a Fourier-series torsional
potential, `E = Σ (Vₙ/2)·[1 + cos(nφ − γ)]`, parameterized per bond type
(backbone φ, backbone ψ, and per-heavy-atom sidechain defaults). Without
this term, an MC sampler would treat every torsion angle as equally
likely — grossly oversampling physically implausible conformations like
eclipsed backbone or cis-peptide geometries, since the Ramachandran
φ/ψ energy barriers (2–5 kcal/mol) are large relative to thermal
energy at room temperature.

**1-2/1-3 exclusions:** atoms directly bonded (1-2, i.e. covalently
adjacent) or separated by exactly one intervening atom (1-3, i.e. an
angle away) are excluded from the nonbonded pair sum entirely — those
interactions are physically meaningless as "nonbonded" contacts and are
already implicitly accounted for by the bonded/dihedral terms.

### Monte Carlo Sampling

Conformational space is explored with **torsion-angle Metropolis Monte
Carlo**, not Cartesian atom displacement. This distinction matters: a
naive MC scheme that perturbs one atom's (x, y, z) coordinates directly
will, almost always, break a covalent bond or distort a bond angle —
producing energetically absurd, chemically invalid structures. Instead:

1. A rotatable bond `(i → j)` is picked at random from the protein's
   precomputed bond topology graph, weighted 50% backbone φ/ψ, 40%
   sidechain χ, 10% other/rigid moves.
2. A random rotation angle `δφ ~ U[−maxΔ, +maxΔ]` is proposed (small for
   backbone bonds, larger for sidechain bonds — backbone motions are more
   energetically consequential per degree).
3. Every atom on the "downstream" (j-) side of the bond is rigidly
   rotated about the `i → j` axis by `δφ`, via the **Rodrigues rotation
   formula**. Because this is a rigid rotation of a rigid subtree, bond
   lengths and bond angles are preserved *exactly*, by construction — the
   move is always chemically valid.
4. Only the energy terms that actually change (cross-boundary pairs
   between the rotated and unrotated atom sets, plus the SASA and
   dihedral terms local to the moved bond) are recomputed — a full
   O(N²) re-evaluation per proposed move would be prohibitively slow.
5. **Metropolis acceptance criterion:** if `ΔE < 0`, the move is always
   accepted (it lowers energy). If `ΔE ≥ 0`, it is accepted with
   probability `exp(−ΔE / T)`. Rejected moves are reverted (atom
   positions and cached Born radii restored) with no side effects.

At effective temperature `T = 0.6 kcal/mol` (roughly room temperature,
since `kT ≈ 0.592 kcal/mol` at 300 K), this Markov chain converges to
the **canonical (Boltzmann) ensemble**, `P(conformation) ∝ exp(−E/kT)` —
meaning low-energy conformations are visited more often, in exact
proportion to their Boltzmann weight, given enough sampling steps.

Two Monte Carlo regimes are used in the application:

- **Pipeline MC** (short): generates a handful of candidate conformations
  per structure, primarily to explore local energy minima near the input
  structure.
- **Landscape MC** (long): runs a much longer Markov chain (hundreds of
  snapshots × tens of steps each) specifically to characterize the shape
  of the energy landscape — see [Application Workflow](#application-workflow).

### Bond Topology and Move Validity

To generate torsion moves, ALMA needs a covalent connectivity graph for
every residue. This is built once per structure from a table of
per-residue bond templates (one static list of atom-atom bonds per
standard amino acid, encoding χ angle definitions and known chemistry —
e.g. `SER: CA-CB, CB-OG` where `CB-OG` is the χ1 hydroxyl-orientation
bond). From this graph, `BondTopology` derives:

- The set of **rotatable bonds** (any bond not in a ring, not terminal).
- For each rotatable bond, which atoms lie on the "i-side" vs. "j-side" —
  used both to perform the rigid rotation and to identify which
  nonbonded pairs cross the rotation boundary (and thus need ΔE
  recomputed).
- The **1-2/1-3 exclusion list**, so directly-bonded and angle-adjacent
  atom pairs never contribute spurious nonbonded energy.
- **Dihedral records** (4-atom sequences + Fourier terms) for every
  rotatable bond, used by the dihedral energy term above.
- Any detected **disulfide pairs** (see below), added as additional
  excluded/restrained pairs.

Peptide bonds between consecutive residues, and residue adjacency in
general, are inferred from a per-atom "residue index" assigned during
PDB parsing — consecutive integer indices trigger automatic peptide-bond
detection, while HETATM residues (ligands, ions) are assigned isolated
indices (≥ 100,000) specifically so they are never mistaken for being
covalently adjacent to a neighboring amino acid.

### Force Field Parameters

Nonbonded parameters (VDW radius, well depth ε, partial charge) are drawn
from **AMBER ff14SB** (Maier et al., 2015), which distinguishes roughly
35 heavy-atom types for standard amino acids (e.g. `CX` = Cα, `2C` = sp3
CH₂, `3C` = sp3 CH₃, `CA` = aromatic carbon, `CT` = generic aliphatic
sp3) rather than treating "carbon" as a single monolithic type. Partial
charges are the corresponding **RESP** charges from the same force
field, covering every heavy atom of all 20 standard amino acids plus the
common histidine protonation-state variants (HID/HIE/HIP).

Real PDB structures (X-ray, most SWISS-MODEL/AlphaFold outputs) essentially
never resolve hydrogen positions, so parsing only ever creates particles for
the heavy atoms actually present in a file. Since ff14SB's RESP charges are
full all-atom charges (heavy atom and each attached hydrogen carrying its own
partial charge), naively dropping unresolved hydrogens would leave every
residue with a large, spurious net charge — e.g. alanine's heavy atoms alone
sum to −0.535 rather than the true neutral 0.000; lysine's sum to −0.882
rather than the true +1.000. ALMA corrects for this with a **united-atom
charge fold-in**: for every hydrogen that would be bonded to a given heavy
atom (per the same static residue templates used for bond topology) but is
absent from the parsed structure, that hydrogen's charge is added onto its
heavy-atom parent. A hydrogen that genuinely is present (e.g. a neutron or NMR
structure) is left alone, since its own particle already carries the charge.

Non-amino-acid HETATM records are not silently discarded:

- **Water** (HOH, WAT, TIP3, SOL, …) is always dropped — explicit water
  is unnecessary and counterproductive under an implicit-solvent model;
  including it would inflate the O(N²)-scaling nonbonded sum and distort
  Born radii.
- **Metal ions** (Mg²⁺, Ca²⁺, Zn²⁺, Fe²⁺/³⁺, Na⁺, Cl⁻, …) are kept, using
  a small tabulated ion parameter set.
- **Other ligands/cofactors** are kept, using element-symbol-based
  fallback VDW parameters (full GAFF2 small-molecule parameterization is
  not yet implemented — see [Known Limitations](#known-limitations)).

### Disulfide Bonds

Cysteine SG–SG covalent crosslinks are structurally and thermodynamically
important — they're a primary stabilizing feature in many extracellular
and secreted proteins. ALMA scans every pair of cysteine SG atoms in a
parsed structure; any pair closer than 2.5 Å (the ~2.0–2.1 Å covalent
S–S bond length plus a tolerance margin) is registered as a disulfide
bond. Each detected pair is (a) excluded from the nonbonded pair sum,
exactly like any other covalent 1-2 pair, and (b) given a harmonic
restraint term in the total energy and in every Monte Carlo ΔE
evaluation, so that MC torsion moves cannot arbitrarily stretch a
covalent S–S bond while exploring conformational space.

### Sequence-Based Disorder Prediction

Not every protein has a well-defined native fold — **intrinsically
disordered proteins (IDPs)** and disordered regions lack a single stable
structure, and this is itself biologically meaningful (many are involved
in signaling and regulation, precisely because they can adopt multiple
conformations). ALMA estimates per-residue disorder probability directly
from sequence, independent of any MC simulation, using an
**IUPred-inspired** statistical approach:

```
energy_sum(i) = Σ_{|j-i|≤10} E_pair(aa[i], aa[j])
disorder_score(i) = 1 / (1 + exp(a·energy_sum(i) + b))
```

where `E_pair` is a pairwise interaction potential over a sliding window
around each residue, and residues with `disorder_score > 0.5` are
flagged as likely disordered. Because this only requires the amino acid
sequence, it is computed immediately after PDB parsing — before any MC
sampling has run — and is later shown alongside a second,
trajectory-based disorder signal (RMSF, below) once the longer Landscape
MC run completes.

### Structural Comparison

Given an analyzed structure, ALMA concurrently fetches independent
structural predictions of the same protein — an AlphaFold model (via the
AlphaFold DB REST API, resolved through a PDB→UniProt mapping when
starting from a PDB ID) and a SWISS-MODEL homology model — and compares
all of them (including ALMA's own MC candidates) against the reference
structure using **Kabsch superposition**:

1. Center both coordinate sets on their centroids.
2. Compute the cross-covariance matrix `H = PᵀQ`.
3. Take its SVD: `U, Σ, Vᵀ = svd(H)`.
4. Correct for reflection ambiguity: `d = sign(det(VᵀUᵀ))`,
   `R = Vᵀ · diag(1, 1, d) · Uᵀ`.
5. `RMSD = sqrt(mean(‖R·q − p‖²))` over all matched atoms.

By default this is computed over **Cα atoms only** (backbone trace);
an **all-heavy-atom RMSD** option is also available, using the exact
same Kabsch procedure over every non-hydrogen atom keyed by
`(chain, residue, atom name)` rather than by residue alone — this is
more sensitive to sidechain packing differences that a Cα-only
comparison cannot see (two structures can have an identical backbone
trace with completely different rotamer conformations).

Where available, AlphaFold's **pLDDT** (predicted local distance
difference test, a per-residue confidence score in [0, 100]) is shown
alongside the RMSD, giving a sense of how much weight to put on
disagreements with the AlphaFold prediction in low-confidence regions.

### Backbone Knot Topology Classification

Two proteins can share nearly identical secondary/tertiary structure and
still be **topologically distinct** — one's backbone trace forms a knot,
the other doesn't — and that distinction carries real biological weight:
knotted proteins fold via more constrained pathways (some depend on
molecular chaperones that unknotted homologs don't need), resist mechanical
unfolding and proteolytic degradation differently, and knot-related
misfolding is implicated in some disease-linked proteins. ALMA classifies
the Cα backbone trace's knot type using the same approach used by
established protein-topology tools (KnotProt, Topoly):

1. **Stochastic closure** — an open chain has no well-defined knot type on
   its own, so both termini are extended out to a distant point along
   independent random directions and joined, repeated across many trials
   with majority voting. This avoids the well-known closure-direction bias
   that a single arbitrary closure can introduce.
2. **KMT reduction** (Koniaris–Muthukumar–Taylor) — repeatedly replaces a
   vertex of the closed polygon with a direct chord to its neighbors
   whenever that chord doesn't pass through any other part of the chain,
   shrinking a polygon of hundreds/thousands of vertices down to a
   minimal-vertex representative of the same knot type, entirely in 3D
   (no projection-plane bias).
3. **2D projection + crossing detection**, recording over/under strand
   order from projection-axis depth at each crossing.
4. **Wirtinger presentation → Alexander polynomial** — the knot group is
   built from the crossings and reduced via Fox calculus to an Alexander
   matrix; the Alexander polynomial is the determinant of an (n−1)×(n−1)
   minor, extracted via roots-of-unity sampling + inverse FFT (a naive
   Vandermonde-solve approach was found to be numerically unstable past
   ~15 crossings).
5. **Classification** against a small table of known Alexander
   polynomials — currently the unknot, trefoil (3₁), and figure-eight
   (4₁) are named with confidence; anything else is reported as
   "unidentified" along with its crossing number and raw polynomial
   rather than risk a mislabeled result.

This is scoped to the Alexander polynomial only. The chirality-sensitive
**Jones polynomial** (needs a Kauffman bracket recursion) and **Khovanov
homology** (a fully categorified chain-complex invariant) are real,
larger follow-up efforts, not yet implemented — see
[Known Limitations](#known-limitations). Classification runs once per
structure in `PipelineWorker`'s background thread, right after parsing
(a few seconds, non-blocking), and is currently surfaced via the process
log (no dedicated visualization panel yet).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  python/gui_main.py  (PyQt6 desktop application)                 │
│                                                                    │
│   PipelineWorker  ──►  fetch PDB, parse, run short MC             │
│   ComparisonWorker ──► fetch AlphaFold/SWISS-MODEL, Kabsch RMSD   │
│   LandscapeWorker  ──► long MC chain, PCA, graph, IDP classify    │
│                                                                    │
│   All heavy lifting (energy evaluation, MC sampling) delegates    │
│   to the C++ extension below via pybind11 calls.                  │
└─────────────────────────┬──────────────────────────────────────┘
                           │ import protein_physics
┌─────────────────────────▼──────────────────────────────────────┐
│  protein_physics  (pybind11 C++ extension)                       │
│                                                                    │
│   src/physics_engine.cpp   — CPU engine (OpenMP-parallelized)     │
│     • Particle, BondTopology, PhysicsEngine classes               │
│     • GB/DH/LJ/SASA energy, dihedral + disulfide terms            │
│     • Torsion-angle Metropolis MC (generate_ensemble)             │
│     • Cell-list neighbor search (O(N), not O(N²))                 │
│                                                                    │
│   src/physics_engine_cuda.cu — optional GPU backend                │
│     • Compiled separately by nvcc as protein_physics_cuda          │
│     • gui_main.py imports it opportunistically, falls back to CPU  │
└────────────────────────────────────────────────────────────────┘
```

Supporting Python modules:

- `python/amber_params.py` — AMBER ff14SB VDW/charge lookup tables,
  metal ion parameters, water residue-name set, united-atom charge
  correction for missing hydrogens.
- `python/iupred.py` — sequence-based disorder predictor.
- `python/knot_analysis.py` — backbone knot topology classification
  (Alexander polynomial).

The GPU extension is a separate build artifact from the CPU extension;
it is not wired into `setuptools`, and the GUI degrades silently to the
CPU path if it cannot be imported. Currently the GPU path still
round-trips the full particle array through Python on every MC snapshot
during the Landscape run rather than keeping the trajectory resident on
device memory — see [Known Limitations](#known-limitations).

---

## Application Workflow

```
1. Input
   User enters a 4-character PDB ID (crystallographic structure) or a
   UniProt accession (AlphaFold-predicted structure).

2. PipelineWorker (background thread)
   a. Resolve target → local file, RCSB PDB, or AlphaFold DB REST API.
   b. Parse PDB → AMBER-parameterized Particle list + BondTopology
      (atom typing, partial charges, disulfide detection, rotatable
      bonds, dihedral records — all in one pass).
   c. Run short Metropolis MC (generate_ensemble) to produce a handful
      of candidate conformations; the GIL is released during the C++
      call so the GUI thread stays responsive.
   d. Evaluate final GB/DH/LJ/SASA/dihedral/disulfide energy for each
      candidate.

3. ComparisonWorker (background thread, runs concurrently)
   Fetches AlphaFold and/or SWISS-MODEL structures for the same protein
   and computes Kabsch RMSD (Cα and all-heavy-atom) plus pLDDT (where
   available) against the reference structure, for every candidate.

4. LandscapeWorker (triggered on demand)
   Runs a much longer MC Markov chain, projects the resulting Cα
   trajectory to 2D via PCA (capturing dominant collective motions),
   builds a conformational graph (nodes = snapshots, edges = consecutive
   transitions weighted by |ΔE|), detects metastable basins via greedy
   modularity community detection, and classifies the protein as
   ordered / possibly-disordered / IDP based on basin count and energy
   spread. Also produces per-residue RMSF (root-mean-square
   fluctuation across the trajectory) as a trajectory-based disorder
   signal, shown alongside the sequence-based IUPred prediction.
```

---

## Installation

Requires a C++ compiler with OpenMP support (MSVC on Windows, GCC/Clang
elsewhere) and, optionally, the CUDA toolkit for the GPU backend.

```bash
pip install -e .                        # CPU-only, editable install
python setup.py build_ext --inplace     # Builds CPU + GPU extensions if CUDA is found
```

Python dependencies are listed in `requirements.txt` (PyQt6, NumPy,
BioPython, scikit-learn, NetworkX, requests, pybind11).

### Portable Windows build

Every tagged release (`vX.Y.Z`) publishes a single-file `ALMA.exe` via
[GitHub Actions](.github/workflows/release.yml) — download it from the
repo's Releases page and run it directly, no Python or installer needed.
It creates a `data/` folder next to itself on first run to cache
downloaded PDB structures.

To build it locally on Windows:

```bat
build_portable_exe.bat
```

which builds the `protein_physics` extension for the active interpreter
and packages `python/gui_main.py` with PyInstaller (`alma.spec`) into
`dist\ALMA.exe`. The CI build is CPU-only; a locally built exe will also
include the CUDA backend if the CUDA toolkit is present at build time.

## Usage

```bash
python python/gui_main.py
```

Enter a 4-character PDB ID (e.g. `1XQ8`) or a UniProt accession (e.g.
`P05067`) and press run. The sidebar shows live progress; once the
pipeline completes, the structure viewer, comparison table, landscape
graph, and disorder profile panels become available.

To validate the Python↔C++ data bridge without the GUI:

```bash
python tests/bridge_test.py
```

## Project Layout

```
src/physics_engine.cpp        C++ CPU physics engine (pybind11 module: protein_physics)
src/physics_engine_cuda.cu    Optional CUDA GPU backend (protein_physics_cuda)
src/main.cpp                  Minimal pybind11 example/sanity module
python/gui_main.py            PyQt6 desktop application
python/amber_params.py        AMBER ff14SB atom-type / charge / ion parameter tables
python/iupred.py              Sequence-based disorder predictor
python/knot_analysis.py       Backbone knot topology classification (Alexander polynomial)
tests/bridge_test.py          Python↔C++ data bridge integrity tests
tests/knot_test.py            Knot classification validation (synthetic + real proteins)
tests/accuracy_test.py        Real-protein force-field accuracy validation (RCSB + AlphaFold)
data/                         Cached/example PDB structures
IMPROVEMENTS.md               Known limitations and implementation roadmap
```

## Performance Notes

- Nonbonded pair energy uses a **cell-list neighbor search** (cells sized
  to the interaction cutoff, 3×3×3 cell neighborhood per atom) rather
  than a brute-force O(N²) all-pairs scan, and is parallelized over
  OpenMP with independent per-thread RNG streams during ensemble
  generation.
- Monte Carlo ΔE evaluation only recomputes energy terms that actually
  change on a given move (cross-boundary nonbonded pairs, local
  dihedral/SASA/disulfide terms) rather than re-scoring the whole
  structure per step.

## Known Limitations

See `IMPROVEMENTS.md` for the full, maintained roadmap. Notable open
items:

- **GPU-resident trajectories**: the Landscape MC run still transfers
  the full particle array between Python and the GPU on every snapshot,
  rather than keeping the trajectory resident in device memory.
- **Small-molecule ligands**: non-ion HETATM records use element-based
  fallback parameters rather than a full GAFF2 force field; common
  cofactors (ATP, heme, NAD⁺, FAD, PLP) are not specially parameterized.
- **No membrane/lipid environment**: the implicit-solvent model assumes
  uniform bulk water everywhere, so membrane protein regions are not
  treated with a lower-dielectric slab.
- **Bond/angle harmonic terms** are deferred — since torsion moves
  preserve bond lengths and angles by construction, they always sit at
  their equilibrium values and contribute negligible energy under the
  current move set.
- **Knot classification is Alexander-polynomial-only**: chirality
  (left/right-handed knots) is indistinguishable without the Jones
  polynomial, and only 3 knot types (unknot, 3₁, 4₁) are named
  explicitly — deeper knots are reported as "unidentified" with their
  correct-but-unnamed polynomial rather than a possibly-wrong label.

## References

- Maier, J.A. et al. (2015). *ff14SB: Improving the Accuracy of Protein
  Side Chain and Backbone Parameters from ff99SB*. J. Chem. Theory
  Comput. 11, 3696–3713.
- Hawkins, G.D., Cramer, C.J., Truhlar, D.G. (1995). *Pairwise Solute
  Descreening of Solute Charges from a Dielectric Medium*. J. Phys.
  Chem. 99, 11663.
- Kabsch, W. (1978). *A discussion of the solution for the best rotation
  to relate two sets of vectors*. Acta Cryst. A34, 827–828.
- Mészáros, B. et al. (2018). *IUPred2A: context-dependent prediction of
  protein disorder as a function of redox state and protein binding*.
  Nucleic Acids Res. 46, W329–W337.
- Koniaris, K., Muthukumar, M. (1991). *Self-entanglement in ring
  polymers*. J. Chem. Phys. 95, 2873.
- Alexander, J.W. (1928). *Topological invariants of knots and links*.
  Trans. Amer. Math. Soc. 30, 275–306.
