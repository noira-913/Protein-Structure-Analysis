"""
landscape_stability_test.py -- repeat-run label stability check for the IDP/
disorder landscape classifier (LandscapeWorker in python/gui_main.py).

Context: the classifier has no automated test today; validation has always been
manual repeat-run spot checks (see IMPROVEMENTS.md item #2's whole history of
1UBQ/1LYZ/1XQ8/1YPI runs). This script automates exactly that spot check so a
future change to LandscapeWorker (sampling depth, classification thresholds,
etc.) can be validated by running one command instead of re-running the GUI by
hand N times per protein.

This specifically validates the 2026-07-08 size-adaptive sampling depth change
(LandscapeWorker._adaptive_depth): 1YPI (494 res, a rigid TIM-barrel, RMSF 0%)
should now read reproducibly ORDERED (it previously split 2/4 ORDERED vs
POSSIBLY DISORDERED under the old flat-depth scheme), while 1UBQ (76 res) and
1LYZ (129 res) must not regress from their already-validated 3/3 ORDERED.

Also covers 1XQ8 (140 res, human alpha-synuclein -- a genuine, literature-
documented IDP, 97.9% RMSF-disordered), added 2026-07-09 alongside the
~4x sampling-budget reduction (see IMPROVEMENTS.md item #2's "Runtime-
optimization follow-up"). Every case up to that point had only checked the
ORDERED direction; 1XQ8 is the one case available that checks the classifier
still correctly detects real disorder, not just correctly stays quiet on
rigid proteins.

No Qt event loop is started -- LandscapeWorker.run() is called directly
(synchronous) rather than via QThread.start(), and its pyqtSignal connections
fire synchronously in that direct call since sender and receiver share a
thread. A QApplication instance is still constructed because gui_main imports
QWebEngineView, which some Qt backends require an application instance to
initialize even when no window is ever shown.

Requires no internet access -- uses the bundled PDBs under data/.
"""
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

DATA_DIR = os.path.join(REPO_ROOT, "data")

# (label, pdb_file, expected_label, n_repeats)
CASES = [
    ("1UBQ (ubiquitin, 76 res)",        "1UBQ.pdb", "ORDERED", 3),
    ("1LYZ (lysozyme, 129 res)",        "1LYZ.pdb", "ORDERED", 3),
    ("1YPI (triosephosphate isomerase, 494 res)", "1YPI.pdb", "ORDERED", 4),
    ("1XQ8 (alpha-synuclein, 140 res, real IDP)", "1XQ8.pdb", "POSSIBLY DISORDERED", 3),
    # Added 2026-07-13 (IMPROVEMENTS.md item #2 follow-up): a second real IDP,
    # found via a database-driven DisProt search after 1XQ8 was the only real
    # IDP in the calibration roster. PopZ-Delta134-177 (Caulobacter vibrioides
    # polar organizing protein Z, N-terminal domain, PDB 6XRY, 141 res per
    # SEQRES) -- solution NMR of the truncated, non-self-assembling construct,
    # i.e. the genuine free/unbound state (not induced by a micelle or binding
    # partner, unlike 1XQ8's own micelle-bound deposit). Literature-documented
    # as unstructured except one ~8-residue amphipathic MoRF helix (M10-I17)
    # (Holmes et al. 2020, J Mol Biol, PMC7736533).
    #
    # KNOWN FLAKY (2026-07-13, merge-readiness audit): the original 3/3
    # result this case shipped with did not reproduce -- 9 further
    # independent repeats (two batches, unmodified code) landed at 6/9
    # correct (67%), funnel range 0.272-0.358 straddling the classifier's
    # own decision boundary (the *same* funnel value produced both labels
    # in different runs). Root cause is the same already-documented,
    # already-deferred short-MC-chain sampling-depth limitation as pre-fix
    # 1LYZ, not a bug in this test or the classification logic -- see
    # IMPROVEMENTS.md item #2's 2026-07-13 correction note. A FAIL on this
    # one case alone is expected/known, not necessarily a new regression --
    # check whether 1UBQ/1LYZ/1YPI/1XQ8 also regressed before concluding
    # something broke.
    ("6XRY (PopZ N-terminal domain, 141 res, real IDP)", "6XRY.pdb", "POSSIBLY DISORDERED", 3),
]


def run_one(gui_main, protein_physics, QApplication_unused, atoms, ca_indices, topo, seed,
            iupred_scores=None):
    import numpy as np

    engine = protein_physics.PhysicsEngine()
    # Mirror real usage (_start_landscape): 3 concurrent branches. We don't have a
    # precomputed top-3 ensemble here, so all 3 branches start from the same
    # structure -- each still gets an independently-seeded fresh PhysicsEngine
    # (per _LandscapeBranchRunnable), so their MC trajectories diverge on their
    # own. This matters because the fix under test (more snapshots per branch)
    # is about pooled-point density for DBSCAN across all branches, not a
    # single trajectory -- testing with only 1 branch would under-test it.
    worker = gui_main.LandscapeWorker(
        engine, atoms, ca_indices, topo, protein_physics,
        extra_seeds=[atoms, atoms], iupred_scores=iupred_scores)

    collected = {}

    def on_result(data):
        collected["data"] = data

    def on_progress(msg):
        pass  # keep stdout quiet; uncomment print(msg) for verbose debugging

    worker.result.connect(on_result)
    worker.progress.connect(on_progress)

    np.random.seed(seed)
    t0 = time.perf_counter()
    worker.run()  # synchronous -- not worker.start()
    dt = time.perf_counter() - t0

    data = collected.get("data", {})
    return data.get("idp_label"), data.get("funnel"), dt


def main():
    # gui_main imports QtWebEngineWidgets, which must be imported (or
    # Qt.AA_ShareOpenGLContexts set) before any QCoreApplication/QApplication is
    # constructed -- so import gui_main first, then create the QApplication.
    import protein_physics  # noqa: E402
    import gui_main  # noqa: E402
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])

    n_fail = 0
    for label, pdb_file, expected, n_repeats in CASES:
        path = os.path.join(DATA_DIR, pdb_file)
        print(f"=== {label} -- expecting {n_repeats}/{n_repeats} {expected} ===")
        atoms, ca_indices, ca_map, topo, iupred_scores, *_rest = gui_main._parse_pdb(
            path, lambda *a: None, protein_physics)
        n_res = len(ca_indices)
        # Thread real IUPred scores through to _adaptive_depth/LandscapeWorker
        # (2026-07-13 fix, IMPROVEMENTS.md item #2 re-test): this harness used
        # to discard _parse_pdb's iupred_scores via `*_rest`, so every case --
        # including 1XQ8, the one real IDP this suite is meant to validate --
        # was tested at the un-boosted, size-only depth rather than the real
        # disorder-aware depth _start_landscape actually uses in the GUI. A
        # stale in-code comment on _adaptive_depth claimed this harness "has
        # no sequence info at all" -- not true; _parse_pdb already computes
        # real sequence-based IUPred scores from the parsed PDB, they just
        # weren't being passed through.
        disorder_frac = (gui_main._iupred.fraction_disordered(iupred_scores)
                          if iupred_scores else 0.0)
        n_snap, steps = gui_main.LandscapeWorker._adaptive_depth(n_res, disorder_frac)
        print(f"  n_res={n_res}  disorder_frac={disorder_frac:.3f}  "
              f"adaptive depth: {n_snap} snapshots x {steps} steps/branch")

        labels, funnels, times = [], [], []
        for i in range(n_repeats):
            idp_label, funnel, dt = run_one(
                gui_main, protein_physics, QApplication, atoms, ca_indices, topo, seed=i,
                iupred_scores=iupred_scores)
            labels.append(idp_label)
            funnels.append(funnel)
            times.append(dt)
            print(f"  run {i+1}/{n_repeats}: label={idp_label}  funnel={funnel:.3f}  "
                  f"time={dt:.1f}s")

        n_correct = sum(1 for lb in labels if lb == expected)
        status = "PASS" if n_correct == n_repeats else "FAIL"
        if status == "FAIL":
            n_fail += 1
        print(f"  --> {n_correct}/{n_repeats} correct, funnel range "
              f"[{min(funnels):.3f}, {max(funnels):.3f}], "
              f"avg time {sum(times)/len(times):.1f}s  [{status}]")
        print()

    if n_fail:
        print(f"{n_fail}/{len(CASES)} protein(s) FAILED label-stability check.")
        sys.exit(1)
    print(f"All {len(CASES)} protein(s) PASSED label-stability check.")


if __name__ == "__main__":
    main()
