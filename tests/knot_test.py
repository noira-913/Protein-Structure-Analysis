"""
knot_test.py -- validate python/knot_analysis.py against real proteins with known
knot status from the literature.

Negative control: hen egg-white lysozyme (PDB 1LYZ) -- a small, thoroughly studied,
unknotted globular protein (also used elsewhere in this repo's accuracy_test.py).

Positive control: YibK (PDB 1J85), a SPOUT-family RNA methyltransferase and one of
the classic examples of a deeply knotted protein backbone (trefoil knot, 3_1) --
extensively documented in the protein-knot literature (Mallam & Jackson and others)
and listed in the KnotProt database.

Requires internet access (fetches from files.rcsb.org).
"""
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import numpy as np  # noqa: E402
import requests      # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402

from knot_analysis import classify_backbone_knot  # noqa: E402

CASES = [
    ("Hen egg-white lysozyme", "1LYZ", "unknot"),
    ("YibK methyltransferase (SPOUT family)", "1J85", "3_1 (trefoil)"),
]


def fetch_ca_trace(pdb_id, tmp_path):
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30)
    r.raise_for_status()
    with open(tmp_path, "wb") as f:
        f.write(r.content)
    parser = PDBParser(QUIET=True)
    st = parser.get_structure(pdb_id, tmp_path)
    coords = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.get_id()[0] != " ":
                    continue
                if "CA" in res:
                    coords.append(res["CA"].get_coord().copy())
        break
    return np.array(coords, dtype=float)


def main():
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="knot_test_")
    n_fail = 0
    for label, pdb_id, expected in CASES:
        print(f"=== {label} ({pdb_id}) -- expecting {expected} ===")
        path = os.path.join(tmpdir, f"{pdb_id}.pdb")
        ca = fetch_ca_trace(pdb_id, path)
        print(f"  {len(ca)} Calpha atoms")
        t0 = time.time()
        result = classify_backbone_knot(ca, n_trials=32, seed=0)
        dt = time.time() - t0
        status = "PASS" if result.name == expected else "FAIL"
        if status == "FAIL":
            n_fail += 1
        print(f"  result: {result.name}  (crossing_number={result.crossing_number}, "
              f"confidence={result.confidence:.2f}, {dt:.1f}s)  [{status}]")
        print()

    if n_fail:
        print(f"{n_fail}/{len(CASES)} case(s) FAILED.")
        sys.exit(1)
    print(f"All {len(CASES)} case(s) PASSED.")


if __name__ == "__main__":
    main()
