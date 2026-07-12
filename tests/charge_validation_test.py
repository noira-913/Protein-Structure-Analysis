"""
charge_validation_test.py -- automated validation for the real RESP partial
charges added to amber_params.PARTIAL_CHARGES for HETATM cofactors (ATP,
ADP, GTP, GDP, HEM, NAD -- see IMPROVEMENTS.md item #4).

No such check existed before this: bridge_test.py's "test_charges" is
unrelated arbitrary data, and accuracy_test.py has no charge-related checks
at all. This packages what the original ATP work did once, by hand, into a
reusable script covering every ligand added since.

Three checks per ligand:
  1. Formal-charge-sum: do this ligand's united-atom charges in
     PARTIAL_CHARGES sum to its documented expected net formal charge?
  2. Atom-name coverage: does a real deposited structure's HETATM record for
     this ligand have exactly the atom names PARTIAL_CHARGES expects (no
     missing, no extra)?
  3. A/B potential-energy sanity: does enabling the real charges (vs. forcing
     them to zero, same structure/topology) produce a finite, physically
     modest per-atom energy shift -- not a blowup/NaN?

Requires internet access (fetches real structures from files.rcsb.org, same
as tests/accuracy_test.py).
"""
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import requests  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402

# (ligand, expected net formal charge, source note)
EXPECTED_CHARGE = {
    "ATP": (-4.0, "ATP4-, fully deprotonated triphosphate (Meagher/Redman/Carlson 2003)"),
    "ADP": (-3.0, "ADP3-, fully deprotonated diphosphate, same lineage as ATP4-"),
    "GTP": (-4.0, "GTP4-, fully deprotonated triphosphate, same lineage as ATP4-"),
    "GDP": (-3.0, "GDP3-, fully deprotonated diphosphate, same lineage as ATP4-"),
    "HEM": (-2.0, "heme b, both propionates deprotonated (physiological pH)"),
    "NAD": (-1.0, "NAD+: +1 pyridinium, two singly-ionized phosphodiester linkages"),
}

# (ligand, real deposited PDB entry known to contain it -- verified via the
# RCSB search API / PDBe ligand_monomers API before use, not guessed)
REFERENCE_PDB = {
    "ATP": "2P0X",
    "ADP": "1ATR",
    "GTP": "2RAP",
    "GDP": "1Q21",
    "HEM": "1C53",
    "NAD": "7YW4",
}

CHARGE_SUM_TOL = 1e-3
# Loosely calibrated off ATP's own +0.36 kcal/mol/atom real-charge shift on a
# 1079-atom structure (IMPROVEMENTS.md item #4) -- this is a blowup/NaN
# sanity check, not a tight accuracy claim.
MAX_SANE_PER_ATOM_SHIFT = 20.0


def fetch_pdb(pdb_id, dest):
    if os.path.exists(dest):
        return
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)


def check_charge_sum(ap, ligand):
    expected, note = EXPECTED_CHARGE[ligand]
    total = sum(v for (r, a), v in ap.PARTIAL_CHARGES.items() if r == ligand)
    n = sum(1 for (r, a) in ap.PARTIAL_CHARGES if r == ligand)
    ok = abs(total - expected) < CHARGE_SUM_TOL
    print(f"  [charge-sum] {ligand}: n_atoms={n} sum={total:+.4f} "
          f"expected={expected:+.4f} ({note})  [{'PASS' if ok else 'FAIL'}]")
    return ok


def check_atom_coverage(ap, ligand, pdb_path):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(ligand, pdb_path)
    model = struct[0]
    found = None
    for chain in model:
        for residue in chain:
            if residue.get_resname().strip() == ligand:
                found = residue
                break
        if found:
            break
    if found is None:
        print(f"  [atom-coverage] {ligand}: FAIL -- residue not found in {pdb_path}")
        return False
    pdb_atoms = {a.get_name().strip() for a in found if a.element != "H"}
    expected_atoms = {a for (r, a) in ap.PARTIAL_CHARGES if r == ligand}
    missing = expected_atoms - pdb_atoms
    extra = pdb_atoms - expected_atoms
    ok = not missing and not extra
    print(f"  [atom-coverage] {ligand}: pdb_atoms={len(pdb_atoms)} "
          f"expected={len(expected_atoms)} missing={sorted(missing)} "
          f"extra={sorted(extra)}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def check_ab_energy(ap, gui_main, protein_physics, ligand, pdb_path):
    parsed = gui_main._parse_pdb(pdb_path, lambda *a: None, protein_physics)
    atoms_real, ca_indices, ca_map, topo = parsed[0], parsed[1], parsed[2], parsed[3]
    n_atoms = len(atoms_real)
    engine = protein_physics.PhysicsEngine()
    e_real = engine.calculate_potential(atoms_real, topo)

    saved = {(r, a): v for (r, a), v in ap.PARTIAL_CHARGES.items() if r == ligand}
    try:
        for key in saved:
            ap.PARTIAL_CHARGES[key] = 0.0
        parsed_zero = gui_main._parse_pdb(pdb_path, lambda *a: None, protein_physics)
        atoms_zero, topo_zero = parsed_zero[0], parsed_zero[3]
        e_zero = engine.calculate_potential(atoms_zero, topo_zero)
    finally:
        ap.PARTIAL_CHARGES.update(saved)

    import math
    delta = e_real - e_zero
    per_atom = delta / n_atoms
    finite = math.isfinite(delta)
    sane = finite and abs(per_atom) < MAX_SANE_PER_ATOM_SHIFT
    print(f"  [A/B energy] {ligand} ({os.path.basename(pdb_path)}, {n_atoms} atoms): "
          f"E(real charges)={e_real:.1f}  E(zeroed)={e_zero:.1f}  "
          f"delta={delta:+.1f} kcal/mol ({per_atom:+.3f} kcal/mol/atom)  "
          f"[{'PASS' if sane else 'FAIL'}]")
    return sane


def main():
    import protein_physics  # noqa: E402
    import gui_main  # noqa: E402
    import amber_params as ap  # noqa: E402

    ligands = sorted(EXPECTED_CHARGE)
    n_fail = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for ligand in ligands:
            print(f"=== {ligand} ===")
            ok = check_charge_sum(ap, ligand)
            n_fail += 0 if ok else 1

            pdb_id = REFERENCE_PDB[ligand]
            path = os.path.join(tmpdir, f"{pdb_id}.pdb")
            fetch_pdb(pdb_id, path)

            ok = check_atom_coverage(ap, ligand, path)
            n_fail += 0 if ok else 1

            ok = check_ab_energy(ap, gui_main, protein_physics, ligand, path)
            n_fail += 0 if ok else 1
            print()

    if n_fail:
        print(f"{n_fail} check(s) FAILED.")
        sys.exit(1)
    print(f"All checks PASSED for {len(ligands)} ligand(s).")


if __name__ == "__main__":
    main()
