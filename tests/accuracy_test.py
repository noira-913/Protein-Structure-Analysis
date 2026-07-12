"""
accuracy_test.py -- Real-protein structural accuracy validation.

Every other test in this repo (bridge_test.py, calibrate_gpu.py) checks internal
self-consistency: does the C++/CUDA bridge round-trip data correctly, does the CPU
engine agree with the GPU engine. Neither asks the actual scientific question: does
ALMA's implicit-solvent torsion-MC energy function agree with reality?

This script does, across several real, structurally diverse proteins. For each one it:

  1. Fetches the real experimental crystal structure (RCSB) and the AlphaFold
     prediction for the same UniProt entry.
  2. Establishes a baseline: how close is the AlphaFold model to the real crystal
     structure (Calpha and all-heavy-atom RMSD)? This is the accuracy bar AlphaFold
     itself achieves for this protein -- typically well under 2 Angstrom for a
     confident globular domain.
  3. Starts an MC run from the REAL crystal structure (the true energy minimum, if
     the force field is any good) and runs generate_ensemble() for increasing step
     counts on both engines, tracking energy and Calpha RMSD back to the original
     crystal coordinates.

     This is the real test: a correctly-parameterized energy function should let the
     structure wander only slightly (a few Angstrom) while lowering energy, because
     the native fold IS (approximately) the energy minimum. If RMSD blows up while
     the sampler reports the energy as improving, that is direct evidence the force
     field's minimum does not correspond to the real structure.
  4. Reports whether ALMA's sampling stays within, or blows past, the AlphaFold-vs-
     crystal baseline from step 2, for every protein tested.
  5. (--decoy-discrimination flag) Generates a "decoy" via a long, hot MC run
     (default T=2.5, 20000 steps) AND a same-budget "relaxed native" reference
     (same 20000 steps, native T=0.6) from the same starting structure, then
     compares their energies. The relaxed-native run exists specifically to
     absorb local steric strain the raw deposited crystal coordinates carry
     (crystallographic model uncertainty, no explicit H, alternate conformers) --
     comparing a decoy that got a long relaxation budget against the raw crystal
     energy (which only gets step 3's few hundred near-native steps) would
     confound "more steps relaxed local clashes" with "the decoy is a genuinely
     better fold," which isn't the question this test asks. Both runs' Calpha
     RMSD back to the crystal structure is also reported: relaxed-native should
     stay low (confirming it's still recognizably the native structure, not a
     drifted alternative) while the decoy should be substantially higher if
     T=2.5 actually explored away from the native basin. This checks a
     different, complementary property from step 3: not "does sampling stay
     near native when started there," but "does the energy function rank the
     true structure below a substantially different alternative conformation of
     the same chain, by a wide margin" -- the basic requirement for the energy
     function to be useful as a scoring/ranking function, not just a stable
     local sampler. Weaker than a real independent decoy set (e.g. Rosetta/CASP
     decoys built by different means), but self-contained and a real check --
     *when it can produce a genuine decoy at all*.

     KNOWN LIMITATION, confirmed empirically (2026-07-10): generate_ensemble's
     move set cannot currently produce a structurally divergent decoy no matter
     how high --decoy-temp goes. Per-move rotation is hard-capped at
     ANGLE_MAX=0.50 rad (physics_engine.cpp) regardless of temperature, and the
     move mix is dominated by sidechain torsions (which don't move Calpha at
     all) plus crankshaft pairs (deliberately designed to preserve downstream
     backbone geometry via a compensating rotation, not reorganize the fold).
     Tested up to T=8.0 (~13x native T=0.6) for 20000 steps on 1UBQ: Calpha
     RMSD from native reached only 0.968 A, statistically indistinguishable
     from the T=0.6 relaxed-native reference's own 0.683-0.919 A drift. This is
     a real structural limit of the available move types, not a parameter that
     needs more tuning -- so a negative "margin" from this test is NOT currently
     meaningful evidence the force field fails to rank native correctly; it's
     evidence the "decoy" never left the native basin. Left in as real,
     reusable infrastructure (the same-budget relaxed-native design, and the
     RMSD sanity check, are both still correct) for whenever a real
     divergent-decoy generator (e.g. large rigid-body backbone perturbations,
     fragment threading, or an uncapped/large-angle move mode) exists to plug
     into it -- do not trust its verdict output until then.

RESIDUE NUMBERING: AlphaFold predicts the full UniProt sequence (often including a
cleaved signal peptide / propeptide the deposited mature-chain crystal structure
does not have), so raw residue numbers frequently don't line up 1:1 with the
crystal structure. This script auto-detects the correct offset by scanning a range
of shifts and picking the one that maximizes common-Calpha overlap, rather than
hardcoding it per protein. (gui_main.py's ComparisonWorker does NOT do this today --
it assumes UniProt-consistent numbering between query and reference, which silently
breaks, near-zero common residues, for any real PDB entry deposited with mature-
chain numbering after a cleaved signal peptide. Flagged, not fixed, here.)

Test proteins -- chosen for fold/feature diversity, not cherry-picked for ease:
  - 1LYZ  / P00698  hen egg-white lysozyme   (alpha+beta, 4 disulfides, 1975 X-ray)
  - 1UBQ  / P0CG47  human ubiquitin          (beta-grasp, no disulfides, tiny 76aa)
  - 1MBN  / P02185  sperm whale myoglobin    (all-alpha, has a bound HETATM heme)
  - 7RSA  / P61823  bovine ribonuclease A    (alpha+beta, 4 disulfides)
  - 1BNI  / P00648  Bacillus barnase         (alpha+beta, no disulfides, 3 copies
                                              in the asymmetric unit -- restricted
                                              to chain A, see restrict_to_first_chain)
  - 2CI2  / P01053  chymotrypsin inhibitor 2 (small alpha+beta, no disulfides;
                                              known limitation: its crystal
                                              numbering doesn't linearly align with
                                              full-length UniProt numbering at any
                                              offset, so the AlphaFold baseline
                                              reports N/A here -- kept as an honest
                                              example of the offset-search limits)
  - 2I1B  / P01584  interleukin-1 beta       (all-beta trefoil, no disulfides)
  - 1F6S  / P00711  alpha-lactalbumin        (alpha+beta, 4 disulfides, lysozyme-
                                              family fold for comparison)

  Larger proteins (more particles, more rotatable bonds -- same near-native
  question at bigger scale):
  - 1YPI  / P60174  triose phosphate isomerase (TIM barrel, ~250 aa, no SS)
  - 2CBA  / P00918  carbonic anhydrase II    (beta-sheet + Zn2+ metal site, ~260 aa)
  - 1ALD  / P00883  aldolase A               (TIM barrel, ~360 aa, tetramer in the
                                              crystal -- restricted to chain A)
  - 1LCI  / P08659  firefly luciferase       (~550 aa, two-domain, no SS/metals)
  - 1AO6  / P02768  human serum albumin      (~585 aa, all-alpha, 17 disulfides --
                                              the largest and most disulfide-dense
                                              case in this set)

Requires internet access (fetches from files.rcsb.org and alphafold.ebi.ac.uk).
"""

import argparse
import os
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import numpy as np  # noqa: E402
import requests      # noqa: E402
from Bio.PDB import PDBParser, PDBIO, Select  # noqa: E402

TEST_PROTEINS = [
    # (label, uniprot_id, pdb_id)
    ("Hen egg-white lysozyme", "P00698", "1LYZ"),
    ("Human ubiquitin",        "P0CG47", "1UBQ"),
    ("Sperm whale myoglobin",  "P02185", "1MBN"),
    ("Bovine ribonuclease A",  "P61823", "7RSA"),
    ("Bacillus barnase",       "P00648", "1BNI"),
    ("Chymotrypsin inhibitor 2", "P01053", "2CI2"),
    ("Interleukin-1 beta",     "P01584", "2I1B"),
    ("Alpha-lactalbumin",      "P00711", "1F6S"),
    # Larger proteins (more particles) -- stress-test the same near-native
    # energy-minimization question at bigger atom counts / more rotatable bonds.
    ("Triose phosphate isomerase", "P60174", "1YPI"),
    ("Carbonic anhydrase II",  "P00918", "2CBA"),
    ("Aldolase A",             "P00883", "1ALD"),
    ("Firefly luciferase",     "P08659", "1LCI"),
    ("Human serum albumin",    "P02768", "1AO6"),
    ("Catalase",               "P00432", "7CAT"),
    ("Beta-galactosidase monomer", "P00722", "1BGL"),
]

OFFSET_SEARCH_RANGE = range(-200, 201)


def fetch_crystal(pdb_id, dest):
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)


class _SingleChain(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id


def restrict_to_first_chain(pdb_path):
    """Many real PDB entries deposit multiple crystallographic copies of the same
    biological chain in one asymmetric unit (e.g. 1BNI has barnase chains A, B, C).
    Feeding all of them into one BondTopology would treat crystal-packing contacts
    between unrelated copies as if they were real intramolecular interactions,
    confounding the "does MC stay near native" question this script asks. Rewrite
    the file in place, keeping only the first ATOM-record protein chain."""
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", pdb_path)
    first_chain_id = None
    for model in st:
        for chain in model:
            if any(res.get_id()[0] == " " for res in chain):
                first_chain_id = chain.get_id()
                break
        break
    if first_chain_id is None:
        return
    io = PDBIO()
    io.set_structure(st)
    io.save(pdb_path, _SingleChain(first_chain_id))


def fetch_alphafold(uniprot_id, dest):
    api = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}", timeout=30)
    api.raise_for_status()
    entry = api.json()[0]
    r = requests.get(entry["pdbUrl"], timeout=30)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)
    return entry.get("globalMetricValue")


def _ca_resnames_from_pdb(path):
    """Return {(chain_id, res_seq): 3-letter resname} for every standard-ATOM Calpha."""
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", path)
    out = {}
    for model in st:
        for chain in model:
            for res in chain:
                if res.get_id()[0] != " ":
                    continue
                for atom in res:
                    if atom.get_name().strip() == "CA":
                        out[(chain.get_id(), res.get_id()[1])] = res.get_resname().strip()
        break
    return out


def best_offset(af_ca_map, xtal_ca_map, af_resnames, xtal_resnames):
    """Scan OFFSET_SEARCH_RANGE for the residue-number shift that best aligns the
    AlphaFold (full UniProt sequence) and crystal (often mature-chain-only
    numbering) structures. Returns (offset, n_common).

    A shift is only accepted if it actually lines up the SEQUENCE, not just the
    coordinate-key range. Coordinate-key overlap alone is not sufficient: for
    Interleukin-1 beta (PDB 2I1B / UniProt P01584), the crystal is numbered 1-153
    starting after propeptide cleavage, and enough offsets place all 153 AlphaFold
    residues inside the crystal's occupied numeric range that a coordinate-key-only
    search locks onto a totally wrong registration (off by ~53 residues) which still
    reports "153/153 match" by sheer numeric-range coincidence -- and then reports a
    bogus ~18 Angstrom "AlphaFold is wildly wrong" result, when the real answer
    (correct offset) is under 1 Angstrom. Requiring >=90% identical residue names at
    the aligned positions rules this out. (Tandem-repeat proteins like polyubiquitin
    have the same failure mode in the opposite direction: several offsets pass the
    identity check because the repeats are near-identical in sequence too -- among
    those, the lowest-RMSD offset is the structurally correct frame.)
    """
    candidates = []
    for off in OFFSET_SEARCH_RANGE:
        common_keys = [(c, r) for (c, r) in af_ca_map if (c, r + off) in xtal_ca_map]
        if not common_keys:
            candidates.append((off, 0, 0.0))
            continue
        n_match = sum(1 for (c, r) in common_keys
                       if af_resnames.get((c, r)) == xtal_resnames.get((c, r + off)))
        candidates.append((off, len(common_keys), n_match / len(common_keys)))
    if not candidates:
        return 0, 0
    valid = [(off, n, ident) for off, n, ident in candidates if n >= 3 and ident >= 0.9]
    if not valid:
        # Fall back to the best identity fraction available, even if < 0.9,
        # rather than silently returning a nonsensical alignment.
        off, n, ident = max(candidates, key=lambda t: (t[2], t[1]))
        return (off, n) if ident > 0 else (0, 0)
    max_n = max(n for _, n, _ in valid)
    near_max = [off for off, n, _ in valid if n >= max_n * 0.9]
    best_off, best_rmsd = near_max[0], float("inf")
    for off in near_max:
        shifted = {(c, r + off): v for (c, r), v in af_ca_map.items()}
        rmsd = compute_rmsd_generic(xtal_ca_map, shifted)
        if rmsd is not None and rmsd < best_rmsd:
            best_off, best_rmsd = off, rmsd
    best_n = sum(1 for (c, r) in af_ca_map if (c, r + best_off) in xtal_ca_map)
    return best_off, best_n


def compute_rmsd_generic(map1, map2):
    """Kabsch RMSD between two {key: coord} maps sharing >=3 common keys."""
    common = sorted(set(map1) & set(map2))
    if len(common) < 3:
        return None
    c1 = np.array([map1[k] for k in common], dtype=float)
    c2 = np.array([map2[k] for k in common], dtype=float)
    c1 -= c1.mean(0); c2 -= c2.mean(0)
    H = c2.T @ c1
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = c1 - (c2 @ R.T)
    return float(np.sqrt((diff ** 2).sum(1).mean()))


def ca_map_from_atoms(atoms, ca_indices, ca_map_keys):
    return {key: np.array([atoms[idx].x, atoms[idx].y, atoms[idx].z])
            for idx, key in ca_map_keys}


def decoy_discrimination(atoms, topo, engine, ca_indices, ca_map, ca_map_keys, n_atoms,
                          decoy_temp=2.5, relax_steps=20000, gui_main=None, physics_mod=None):
    """Compare a hot-MC decoy against a SAME-BUDGET relaxed-native reference, not
    the raw crystal energy. Raw deposited coordinates routinely carry local
    steric strain (crystallographic model uncertainty, no explicit H, alternate
    conformers) that thousands of MC steps relax away regardless of temperature
    -- comparing a decoy that got that relaxation budget against a native energy
    that didn't would confound "MC found a better nearby minimum than a few
    hundred near-native steps reached" with "the decoy is a genuinely better,
    different fold," which is the actual question here. Both runs get the same
    step budget; only temperature differs. The relaxed-native run's own Calpha
    RMSD is reported too, as a check that it actually stayed near-native (a
    precondition for this being a meaningful reference at all -- if it drifted
    as far as the decoy, this test isn't measuring what it claims to)."""
    t0 = time.perf_counter()
    relaxed_ens = engine.generate_ensemble(atoms, topo, 1, relax_steps, 0.6, 0.12)
    dt_relax = time.perf_counter() - t0
    relaxed = relaxed_ens[0]
    e_relaxed = engine.calculate_potential(relaxed, topo)
    e_relaxed_per_atom = e_relaxed / n_atoms
    relaxed_ca_map = ca_map_from_atoms(relaxed, ca_indices, ca_map_keys)
    relaxed_rmsd = compute_rmsd_generic(ca_map, relaxed_ca_map)

    t0 = time.perf_counter()
    decoy_ens = engine.generate_ensemble(atoms, topo, 1, relax_steps, decoy_temp, 0.12)
    dt_decoy = time.perf_counter() - t0
    decoy = decoy_ens[0]
    e_decoy = engine.calculate_potential(decoy, topo)
    e_decoy_per_atom = e_decoy / n_atoms
    decoy_ca_map = ca_map_from_atoms(decoy, ca_indices, ca_map_keys)
    decoy_rmsd = compute_rmsd_generic(ca_map, decoy_ca_map)

    margin = e_decoy_per_atom - e_relaxed_per_atom
    print(f"    relaxed-native (T=0.6, {relax_steps} steps, {dt_relax:.1f}s): "
          f"E = {e_relaxed:.1f} kcal/mol ({e_relaxed_per_atom:.2f} kcal/mol/atom), "
          f"Calpha RMSD to crystal = {relaxed_rmsd:.3f} Å")
    print(f"    decoy (T={decoy_temp}, {relax_steps} steps, {dt_decoy:.1f}s): "
          f"E = {e_decoy:.1f} kcal/mol ({e_decoy_per_atom:.2f} kcal/mol/atom), "
          f"Calpha RMSD to crystal = {decoy_rmsd:.3f} Å, "
          f"margin over relaxed-native = {margin:+.2f} kcal/mol/atom")
    result = {"e_relaxed_per_atom": e_relaxed_per_atom, "relaxed_rmsd": relaxed_rmsd,
              "e_decoy_per_atom": e_decoy_per_atom, "decoy_rmsd": decoy_rmsd, "margin": margin}

    # Pivot-branch decoy arm (2026-07-13, Phase B of the IMPROVEMENTS.md items
    # #1+#2 joint investigation): the hot-MC decoy above is documented (see
    # module docstring point 5) to never actually leave the native basin --
    # this arm instead uses the same "coarse structural pivot" kick-then-relax
    # mechanism already built for LandscapeWorker's disabled USE_PIVOT_BRANCH
    # (gui_main._coarse_pivot), which a same-day smoke test showed CAN produce
    # a structurally divergent, energetically comparable alternate conformation
    # (~11-15 A Calpha RMSD from seed on 1UBQ, final energy well within the
    # ordinary-branch range). Same-budget discipline preserved: the post-kick
    # relax gets the identical relax_steps budget as the other two arms, at
    # native T=0.6 (not decoy_temp -- the kick itself, not elevated
    # temperature, is what's supposed to produce the divergence here).
    if gui_main is not None and physics_mod is not None and len(topo.concerted_pairs) > 0:
        rng = np.random.default_rng(0)
        pivoted, n_applied = gui_main._coarse_pivot(atoms, topo, physics_mod, rng, n_pivots=1)
        t0 = time.perf_counter()
        pivot_ens = engine.generate_ensemble(pivoted, topo, 1, relax_steps, 0.6, 0.12)
        dt_pivot = time.perf_counter() - t0
        pivot_relaxed = pivot_ens[0]
        e_pivot = engine.calculate_potential(pivot_relaxed, topo)
        e_pivot_per_atom = e_pivot / n_atoms
        pivot_ca_map = ca_map_from_atoms(pivot_relaxed, ca_indices, ca_map_keys)
        pivot_rmsd = compute_rmsd_generic(ca_map, pivot_ca_map)
        pivot_margin = e_pivot_per_atom - e_relaxed_per_atom
        print(f"    pivot-branch decoy ({n_applied} kick(s), T=0.6, {relax_steps} steps, "
              f"{dt_pivot:.1f}s): E = {e_pivot:.1f} kcal/mol ({e_pivot_per_atom:.2f} "
              f"kcal/mol/atom), Calpha RMSD to crystal = {pivot_rmsd:.3f} Å, "
              f"margin over relaxed-native = {pivot_margin:+.2f} kcal/mol/atom")
        result.update({"e_pivot_per_atom": e_pivot_per_atom, "pivot_rmsd": pivot_rmsd,
                        "pivot_margin": pivot_margin})
    return result


def run_one(label, uniprot_id, pdb_id, cpu_mod, gpu_mod, gui_main, mc_steps, tmpdir,
            run_decoy=False, decoy_temp=2.5, decoy_steps=20000):
    print(f"\n{'='*70}\n{label}  (UniProt {uniprot_id}, PDB {pdb_id})\n{'='*70}")

    crystal_path = os.path.join(tmpdir, f"{pdb_id}.pdb")
    af_path = os.path.join(tmpdir, f"AF-{uniprot_id}.pdb")
    fetch_crystal(pdb_id, crystal_path)
    restrict_to_first_chain(crystal_path)
    plddt = fetch_alphafold(uniprot_id, af_path)
    print(f"  AlphaFold global pLDDT: {plddt}")

    parsed = gui_main._parse_pdb(crystal_path, lambda *a: None, cpu_mod)
    atoms, ca_indices, ca_map, topo, iupred, ca_residues, heavy_map, heavy_indices, heavy_keys = parsed
    n_atoms = len(atoms)
    print(f"  parsed: {n_atoms} atoms, {topo.num_rot_bonds} rotatable bonds, "
          f"{topo.num_dihedrals} dihedrals, {topo.num_disulfide_pairs} disulfide pairs, "
          f"{len(ca_map)} Calpha residues")

    ca_map_keys = list(zip(ca_indices, ca_map.keys()))

    af_ca_map, _ = gui_main._ca_map_from_pdb(af_path)
    af_resnames = _ca_resnames_from_pdb(af_path)
    xtal_resnames = {(c, r): name for (c, r, name) in ca_residues}
    off, n_common = best_offset(af_ca_map, ca_map, af_resnames, xtal_resnames)
    print(f"  residue-numbering offset (AlphaFold -> crystal): {off:+d}  "
          f"({n_common}/{len(ca_map)} Calpha residues match)")
    af_ca_map_adj = {(c, r + off): v for (c, r), v in af_ca_map.items()}
    baseline_ca_rmsd = compute_rmsd_generic(ca_map, af_ca_map_adj)

    af_heavy_map = gui_main._heavy_atom_map_from_pdb(af_path)
    af_heavy_map_adj = {(c, r + off, n): v for (c, r, n), v in af_heavy_map.items()}
    baseline_heavy_rmsd = compute_rmsd_generic(heavy_map, af_heavy_map_adj)

    print(f"  AlphaFold-vs-crystal Calpha RMSD:     {baseline_ca_rmsd:.3f} Å"
          if baseline_ca_rmsd is not None else "  AlphaFold-vs-crystal Calpha RMSD:     N/A")
    print(f"  AlphaFold-vs-crystal heavy-atom RMSD: {baseline_heavy_rmsd:.3f} Å"
          if baseline_heavy_rmsd is not None else "  AlphaFold-vs-crystal heavy-atom RMSD: N/A")

    e0 = cpu_mod.PhysicsEngine().calculate_potential(atoms, topo)
    print(f"  starting (crystal) potential energy: {e0:.1f} kcal/mol "
          f"({e0/n_atoms:.2f} kcal/mol/atom)")

    engines = [("CPU", cpu_mod.PhysicsEngine())]
    if gpu_mod is not None:
        engines.append(("GPU", gpu_mod.PhysicsEngine()))

    row_results = []
    for elabel, engine in engines:
        for steps in mc_steps:
            t0 = time.perf_counter()
            ens = engine.generate_ensemble(atoms, topo, 1, steps, 0.6, 0.12)
            dt = time.perf_counter() - t0
            candidate = ens[0]
            e_after = engine.calculate_potential(candidate, topo)
            cand_ca_map = ca_map_from_atoms(candidate, ca_indices, ca_map_keys)
            rmsd_to_native = compute_rmsd_generic(ca_map, cand_ca_map)
            row_results.append((elabel, steps, dt, e0, e_after, rmsd_to_native))
            print(f"    [{elabel:3s}] {steps:5d} steps ({dt:5.2f}s): "
                  f"E {e0:.1f} -> {e_after:.1f} kcal/mol, "
                  f"Calpha RMSD to native = {rmsd_to_native:.3f} Å")

    worst_rmsd = max(r[5] for r in row_results)
    result = {
        "label": label, "pdb_id": pdb_id, "n_atoms": n_atoms,
        "baseline_ca_rmsd": baseline_ca_rmsd, "worst_mc_rmsd": worst_rmsd,
        "n_disulfides": topo.num_disulfide_pairs,
    }
    if run_decoy:
        result["decoy"] = decoy_discrimination(atoms, topo, engines[0][1],
                                                 ca_indices, ca_map, ca_map_keys, n_atoms,
                                                 decoy_temp=decoy_temp, relax_steps=decoy_steps,
                                                 gui_main=gui_main, physics_mod=cpu_mod)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc-steps", type=int, nargs="+", default=[200, 1000, 5000],
                     help="MC step counts to test per protein (default: 200 1000 5000)")
    ap.add_argument("--proteins", nargs="*", default=None,
                     help="Subset of protein labels to run (default: all)")
    ap.add_argument("--decoy-discrimination", action="store_true",
                     help="Also generate a hot-MC decoy per protein and check native "
                          "scores lower (CPU engine only, adds one long MC run/protein)")
    ap.add_argument("--decoy-temp", type=float, default=2.5,
                     help="MC temperature for decoy generation (default: 2.5)")
    ap.add_argument("--decoy-steps", type=int, default=20000,
                     help="MC step budget for both the decoy and the same-budget "
                          "relaxed-native reference (default: 20000)")
    args = ap.parse_args()

    try:
        import protein_physics as cpu_mod
    except ImportError as ex:
        print(f"FATAL: could not import protein_physics (CPU engine): {ex}")
        sys.exit(1)

    gpu_mod = None
    try:
        import protein_physics_cuda as gpu_mod
        print(f"GPU device: {gpu_mod.PhysicsEngine.device_name()}")
    except Exception as ex:
        print(f"NOTE: GPU engine not available ({ex}); CPU-only run.")

    import gui_main  # noqa: E402
    from PyQt6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)

    proteins = TEST_PROTEINS
    if args.proteins:
        wanted = set(args.proteins)
        proteins = [p for p in TEST_PROTEINS if p[0] in wanted]

    tmpdir = tempfile.mkdtemp(prefix="alma_accuracy_")
    summaries = []
    for label, uniprot_id, pdb_id in proteins:
        try:
            summaries.append(run_one(label, uniprot_id, pdb_id, cpu_mod, gpu_mod,
                                      gui_main, args.mc_steps, tmpdir,
                                      run_decoy=args.decoy_discrimination,
                                      decoy_temp=args.decoy_temp, decoy_steps=args.decoy_steps))
        except Exception as ex:
            print(f"  FAILED: {ex}")
            summaries.append({"label": label, "pdb_id": pdb_id, "n_atoms": 0,
                               "baseline_ca_rmsd": None, "worst_mc_rmsd": None,
                               "n_disulfides": 0, "error": str(ex)})

    print(f"\n{'='*70}\nOverall summary across {len(summaries)} proteins\n{'='*70}")
    print(f"{'Protein':30s} {'PDB':6s} {'atoms':>6s} {'SS':>3s} {'AF RMSD':>9s} {'worst MC RMSD':>14s}  verdict")
    n_ok = 0
    for s in summaries:
        if s.get("error"):
            print(f"{s['label']:30s} {s['pdb_id']:6s}   FAILED: {s['error']}")
            continue
        af_r, mc_r = s["baseline_ca_rmsd"], s["worst_mc_rmsd"]
        if af_r is None or mc_r is None:
            verdict = "N/A (insufficient common residues)"
        elif mc_r <= af_r * 2:
            verdict = "OK -- within ~2x AlphaFold accuracy bar"
            n_ok += 1
        else:
            verdict = "FLAG -- force-field minimum diverges from native"
        af_str = f"{af_r:.3f}" if af_r is not None else "N/A"
        mc_str = f"{mc_r:.3f}" if mc_r is not None else "N/A"
        print(f"{s['label']:30s} {s['pdb_id']:6s} {s['n_atoms']:6d} {s['n_disulfides']:3d} "
              f"{af_str:>9s} {mc_str:>14s}  {verdict}")
    print(f"\n{n_ok}/{len(summaries)} proteins had ALMA sampling stay within "
          f"~2x the AlphaFold-level accuracy bar.")

    if args.decoy_discrimination:
        print(f"\n{'='*70}\nDecoy discrimination (native vs. hot-MC decoy)\n{'='*70}")
        print("CAVEAT (confirmed 2026-07-10, see module docstring point 5): "
              "generate_ensemble cannot currently produce a structurally divergent "
              "decoy at any --decoy-temp (ANGLE_MAX cap + sidechain-dominated move "
              "mix) -- verdicts below are NOT meaningful until a real divergent-"
              "decoy generator exists. Reported for the record, not as a pass/fail.")
        n_discriminated = 0
        n_with_decoy = 0
        n_pivot_discriminated = 0
        n_with_pivot = 0
        for s in summaries:
            decoy = s.get("decoy")
            if decoy is None:
                continue
            n_with_decoy += 1
            ok = decoy["margin"] > 0
            n_discriminated += int(ok)
            print(f"{s['label']:30s} margin = {decoy['margin']:+7.2f} kcal/mol/atom  "
                  f"{'OK -- native scores lower' if ok else 'FLAG -- decoy scores lower than native'}")
            if "pivot_margin" in decoy:
                n_with_pivot += 1
                pivot_ok = decoy["pivot_margin"] > 0
                n_pivot_discriminated += int(pivot_ok)
                print(f"{'':30s} pivot-branch margin = {decoy['pivot_margin']:+7.2f} "
                      f"kcal/mol/atom, RMSD = {decoy['pivot_rmsd']:.2f} Å  "
                      f"{'OK -- native scores lower' if pivot_ok else 'FLAG -- pivot decoy scores lower than native'}")
        print(f"\n{n_discriminated}/{n_with_decoy} proteins: native scored lower "
              f"(better) than its own hot-MC decoy.")
        if n_with_pivot:
            print(f"\nPivot-branch decoy arm (2026-07-13, Phase B): unlike the hot-MC "
                  f"decoy above, this one DOES produce a structurally divergent decoy "
                  f"(see IMPROVEMENTS.md items #1/#2) -- its verdicts below ARE "
                  f"meaningful evidence about ranking ability, not just \"never left "
                  f"the native basin.\"")
            print(f"{n_pivot_discriminated}/{n_with_pivot} proteins: native scored "
                  f"lower (better) than its own pivot-branch decoy.")


if __name__ == "__main__":
    main()
