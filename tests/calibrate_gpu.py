"""
calibrate_gpu.py — CPU vs GPU physics engine calibration/validation script.

Run this on a machine with an NVIDIA GPU + CUDA toolkit after building BOTH
extensions:

    python setup.py build_ext --inplace   # from the repo root; needs CUDA
                                           # toolkit + cl.exe on PATH to also
                                           # produce protein_physics_cuda*.pyd
    python tests/calibrate_gpu.py         # this script

What it checks (see IMPROVEMENTS.md item #14 for context — the GPU engine
was recently ported from a physically-invalid Cartesian MC move to the same
torsion-angle MC + bond topology the CPU engine uses):

  1. protein_physics_cuda actually imports and reports a device name — i.e.
     the extension built AND a CUDA-capable GPU is visible at runtime.
  2. calculate_potential() parity: same parsed structure + same BondTopology
     fed to both engines. This is the one deterministic, exact comparison
     available (MC itself is stochastic in both engines — no seed knob was
     added — so post-MC energies are only checked for "reasonable", not
     "identical"). A large relative difference here means the GPU port has
     a real bug (wrong exclusion mask, wrong Born radii, a dropped energy
     term, the hard-core threshold, etc.), not just float-vs-double noise.
  3. Short independent MC run on each engine (generate_ensemble): confirms
     the sampler actually lowers energy and stays in a physically sane
     range (catches gross bugs like a miscalibrated hard-core threshold,
     which previously produced billions-of-kcal/mol energies).
  4. run_landscape_trajectory() on the GPU engine: confirms the GPU-resident
     multi-snapshot path runs end-to-end without CUDA errors.
  5. Wall-clock timing CPU vs GPU for the same MC workload — the actual
     "does the GPU offload help" data point, since kernel-launch overhead
     can dominate for small proteins.

Prints a PASS/FAIL summary per structure at the end. Paste the full output
back if anything is flagged FAIL or looks physically wrong (e.g. energies
in the billions, or GPU/CPU disagreeing by orders of magnitude).
"""

import argparse
import glob
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import numpy as np  # noqa: E402


def relative_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdbs", nargs="*", default=None,
                     help="Specific data/*.pdb files to test (default: all bundled structures)")
    ap.add_argument("--mc-steps", type=int, default=200,
                     help="Steps for the short generate_ensemble sanity run (default: 200)")
    ap.add_argument("--landscape-snapshots", type=int, default=5,
                     help="Snapshots for the run_landscape_trajectory smoke test (default: 5)")
    ap.add_argument("--landscape-steps", type=int, default=20,
                     help="Steps per snapshot for the landscape smoke test (default: 20)")
    ap.add_argument("--energy-tolerance", type=float, default=0.02,
                     help="Relative difference above which calculate_potential parity FAILs (default: 0.02 = 2%%)")
    args = ap.parse_args()

    try:
        import protein_physics as cpu_mod
    except ImportError as ex:
        print(f"FATAL: could not import protein_physics (CPU engine): {ex}")
        print("Build it first: python setup.py build_ext --inplace")
        sys.exit(1)

    try:
        import protein_physics_cuda as gpu_mod
    except ImportError as ex:
        print(f"FATAL: could not import protein_physics_cuda (GPU engine): {ex}")
        print("This means either the CUDA extension didn't build (check for")
        print("protein_physics_cuda*.pyd/.so next to protein_physics.*), or")
        print("it built but the interpreter can't load it (missing cudart DLL")
        print("on PATH, mismatched Python ABI, etc.).")
        sys.exit(1)

    try:
        device_name = gpu_mod.PhysicsEngine.device_name()
    except Exception as ex:
        print(f"FATAL: protein_physics_cuda imported but device_name() failed: {ex}")
        print("This usually means no CUDA-capable GPU is visible at runtime")
        print("(driver not installed, wrong CUDA_VISIBLE_DEVICES, etc.).")
        sys.exit(1)
    print(f"GPU device: {device_name}")
    print(f"CPU threads: {cpu_mod.PhysicsEngine().num_threads()}  "
          f"GPU-module reported threads: {gpu_mod.PhysicsEngine().num_threads()}")
    print()

    import gui_main  # noqa: E402  (heavy import — PyQt6 etc. — done after the fast checks above)
    from PyQt6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)

    if args.pdbs:
        pdb_paths = args.pdbs
    else:
        pdb_paths = sorted(glob.glob(os.path.join(REPO_ROOT, "data", "*.pdb")))
    if not pdb_paths:
        print("FATAL: no PDB files found (pass --pdbs or check data/*.pdb)")
        sys.exit(1)

    results = []
    for pdb_path in pdb_paths:
        name = os.path.basename(pdb_path)
        print(f"=== {name} ===")
        try:
            parsed = gui_main._parse_pdb(pdb_path, lambda *a: None, cpu_mod)
        except Exception as ex:
            print(f"  PARSE FAILED: {ex}")
            results.append((name, "FAIL (parse)"))
            continue
        atoms, ca_indices, ca_map, topo, *_rest = parsed
        n_atoms = len(atoms)
        print(f"  atoms={n_atoms}  rot_bonds={topo.num_rot_bonds}  "
              f"dihedrals={topo.num_dihedrals}  disulfides={topo.num_disulfide_pairs}")

        row_ok = True

        # 1. calculate_potential() parity — the one deterministic, exact check.
        cpu_engine = cpu_mod.PhysicsEngine()
        gpu_engine = gpu_mod.PhysicsEngine()
        e_cpu = cpu_engine.calculate_potential(atoms, topo)
        e_gpu = gpu_engine.calculate_potential(atoms, topo)
        rdiff = relative_diff(e_cpu, e_gpu)
        status = "OK" if rdiff <= args.energy_tolerance else "MISMATCH"
        if status != "OK":
            row_ok = False
        print(f"  calculate_potential: CPU={e_cpu:.3f}  GPU={e_gpu:.3f}  "
              f"rel_diff={rdiff:.4f}  [{status}]")

        # Sanity: energy should be in the "thousands, not billions" range per atom
        # (see IMPROVEMENTS.md item #13/#14 — this is exactly what was broken before).
        for label, e in (("CPU", e_cpu), ("GPU", e_gpu)):
            per_atom = abs(e) / max(n_atoms, 1)
            if per_atom > 1000:
                print(f"  ⚠ {label} energy/atom = {per_atom:.1f} kcal/mol — "
                      f"looks like a hard-core/exclusion blowup, not a real value")
                row_ok = False

        # 2. Short independent MC run on each engine.
        for label, engine in (("CPU", cpu_engine), ("GPU", gpu_engine)):
            t0 = time.perf_counter()
            ens = engine.generate_ensemble(atoms, topo, 1, args.mc_steps, 0.6, 0.12)
            dt = time.perf_counter() - t0
            e_after = engine.calculate_potential(ens[0], topo)
            e_before = e_cpu if label == "CPU" else e_gpu
            direction = "lower" if e_after < e_before else "HIGHER (unexpected)"
            print(f"  {label} generate_ensemble({args.mc_steps} steps): "
                  f"{dt:.2f}s  E_before={e_before:.1f}  E_after={e_after:.1f}  ({direction})")
            if e_after > e_before:
                row_ok = False

        # 3. GPU-resident landscape trajectory smoke test.
        try:
            t0 = time.perf_counter()
            snaps, energies = gpu_engine.run_landscape_trajectory(
                atoms, topo, args.landscape_snapshots, args.landscape_steps, 0.6, 0.12)
            dt = time.perf_counter() - t0
            print(f"  GPU run_landscape_trajectory("
                  f"{args.landscape_snapshots}x{args.landscape_steps}): {dt:.2f}s  "
                  f"energies={[round(e, 1) for e in energies]}")
            if len(snaps) != args.landscape_snapshots or len(snaps[0]) != n_atoms:
                print("  ⚠ unexpected snapshot shape")
                row_ok = False
        except Exception as ex:
            print(f"  run_landscape_trajectory FAILED: {ex}")
            row_ok = False

        results.append((name, "PASS" if row_ok else "FAIL"))
        print()

    print("=== Summary ===")
    for name, status in results:
        print(f"  {status:5s}  {name}")
    n_fail = sum(1 for _, s in results if s != "PASS")
    if n_fail:
        print(f"\n{n_fail}/{len(results)} structure(s) flagged — paste this whole output back for a fix.")
        sys.exit(1)
    print(f"\nAll {len(results)} structure(s) OK.")


if __name__ == "__main__":
    main()
