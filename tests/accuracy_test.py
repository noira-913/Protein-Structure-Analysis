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


def run_one(label, uniprot_id, pdb_id, cpu_mod, gpu_mod, gui_main, mc_steps, tmpdir):
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
    return {
        "label": label, "pdb_id": pdb_id, "n_atoms": n_atoms,
        "baseline_ca_rmsd": baseline_ca_rmsd, "worst_mc_rmsd": worst_rmsd,
        "n_disulfides": topo.num_disulfide_pairs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc-steps", type=int, nargs="+", default=[200, 1000, 5000],
                     help="MC step counts to test per protein (default: 200 1000 5000)")
    ap.add_argument("--proteins", nargs="*", default=None,
                     help="Subset of protein labels to run (default: all)")
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
                                      gui_main, args.mc_steps, tmpdir))
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


if __name__ == "__main__":
    main()
