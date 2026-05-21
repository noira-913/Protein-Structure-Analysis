"""
gui_main.py — ALMA: Atomistic Local Motion Analyzer
=====================================================
PyQt6 desktop application for protein structure retrieval, implicit-solvent
Monte Carlo simulation, and conformational analysis.

High-level workflow:
  1. User enters a 4-char PDB ID or a UniProt accession in the sidebar.
  2. PipelineWorker (QThread) fetches the PDB file (RCSB or AlphaFold),
     maps atoms to AMBER ff parameters, then runs generate_ensemble() via
     the C++ protein_physics extension.
  3. ComparisonWorker (QThread) concurrently fetches AlphaFold / SWISS-MODEL
     structures and computes Kabsch-superimposed Cα RMSD vs the reference.
  4. LandscapeWorker (QThread) runs a longer MC Markov chain, projects
     conformations to 2D via PCA, builds a NetworkX conformational graph,
     detects metastable basins via greedy modularity, and classifies the
     protein as ordered / possibly-disordered / IDP.
  5. Residue flexibility is shown as per-Cα RMSF from the landscape trajectory.

3D structures are rendered in a QWebEngineView using the 3Dmol.js library
served from a local temp HTML file (avoids CORS issues with file:// URLs).

GPU acceleration: if protein_physics_cuda is importable at startup, the user
is offered a choice; both modules expose the same Particle / PhysicsEngine API.
"""

import sys, os, requests, traceback, tempfile
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox, QFrame,
    QProgressBar, QSplitter, QGridLayout, QScrollArea, QStackedWidget,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtCore import QUrl
from Bio.PDB import PDBParser, PDBList
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import protein_physics


def _try_gpu_backend():
    """Attempt to import the CUDA extension.  Returns (module, gpu_name) or (None, None)."""
    try:
        import protein_physics_cuda as cuda_mod
        name = cuda_mod.PhysicsEngine.device_name()
        return cuda_mod, name
    except Exception:
        return None, None

os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

# ═══════════════════════════════════════════════════════════════════
#  Light theme
# ═══════════════════════════════════════════════════════════════════
STYLE = """
QMainWindow, QWidget {
    background-color: #f1f5f9;
    color: #1e293b;
    font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace;
    font-size: 12px;
}
QFrame#panel {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
}
QLabel#heading {
    color: #1d4ed8;
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 2px;
    padding: 6px 8px 2px 8px;
}
QLabel#metric-val {
    color: #0f172a;
    font-size: 20px;
    font-weight: bold;
    padding: 0 8px;
}
QLabel#metric-unit {
    color: #64748b;
    font-size: 10px;
    padding: 0 8px 4px 8px;
}
QLabel#status-ok  { color: #16a34a; font-size: 11px; font-weight: bold; }
QLabel#status-run { color: #d97706; font-size: 11px; font-weight: bold; }
QLabel#status-err { color: #dc2626; font-size: 11px; font-weight: bold; }

QLineEdit {
    background-color: #ffffff;
    color: #1e293b;
    border: 1.5px solid #cbd5e1;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 14px;
    selection-background-color: #bfdbfe;
}
QLineEdit:focus { border-color: #1d4ed8; }

QPushButton#run-btn {
    background-color: #1d4ed8;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 10px 20px;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: bold;
}
QPushButton#run-btn:hover  { background-color: #1e40af; }
QPushButton#run-btn:disabled { background-color: #e2e8f0; color: #94a3b8; }

QPushButton#sec-btn {
    background-color: transparent;
    color: #1d4ed8;
    border: 1.5px solid #1d4ed8;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 10px;
}
QPushButton#sec-btn:hover { background-color: #eff6ff; }
QPushButton#sec-btn:disabled { color: #94a3b8; border-color: #cbd5e1; }

QTextEdit {
    background-color: #f8fafc;
    color: #334155;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 10px;
    font-size: 11px;
    line-height: 1.6;
}
QScrollBar:vertical { background: #f1f5f9; width: 8px; border: none; }
QScrollBar::handle:vertical { background: #94a3b8; border-radius: 4px; min-height: 20px; }
QProgressBar {
    background-color: #e2e8f0;
    border: none;
    border-radius: 2px;
    height: 4px;
}
QProgressBar::chunk { background-color: #1d4ed8; }
QSplitter::handle { background: #e2e8f0; }
"""

# ═══════════════════════════════════════════════════════════════════
#  AMBER parameters
# ═══════════════════════════════════════════════════════════════════
# Simplified AMBER ff99SB-like radii (Å) and LJ well-depths (kcal/mol)
# keyed by element symbol.  Fallback: (1.9 Å, 0.1 kcal/mol).
_AMBER = {
    "C": (1.908, 0.086), "N": (1.824, 0.170),
    "O": (1.661, 0.210), "S": (2.000, 0.250),
    "H": (0.600, 0.015), "P": (2.100, 0.200),
}
# Residue-level formal charges (in electron units) assigned to the Cα atom.
# Only charged/titratable side-chains are listed; all others default to 0.
_CHARGE = {"ARG": +1.0, "LYS": +1.0, "HIS": +0.5, "ASP": -1.0, "GLU": -1.0}

def _atom_params(atom):
    """Return (charge, radius, epsilon) for a BioPython Atom object.

    Charge is assigned to Cα atoms only; all other atoms get 0.
    Element is inferred from atom.element or the first character of atom name.
    """
    res  = atom.get_parent().get_resname().strip()
    name = atom.get_name().strip()
    elem = (atom.element or "").strip().upper()
    elem = elem if len(elem) == 1 else name[0].upper()
    charge = _CHARGE.get(res, 0.0) if name == "CA" else 0.0
    r, e = _AMBER.get(elem, (1.9, 0.1))
    return charge, r, e

def _parse_pdb(path, log, physics_mod):
    """Parse a PDB file and build a particle list for the physics engine.

    Returns (particles, ca_indices, ca_map):
      particles   — list of physics_mod.Particle for all valid heavy atoms
      ca_indices  — index into particles for each Cα atom (in residue order)
      ca_map      — dict {(chain_id, res_seq): coord_array} for RMSD reference

    HETATM records (ligands, waters) are skipped (get_id()[0] != ' ').
    Atoms with NaN/Inf coordinates are skipped and a warning is logged.
    """
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("prot", path)
    atoms, skipped = [], 0
    ca_indices, ca_map = [], {}
    for atom in st.get_atoms():
        if atom.get_parent().get_id()[0] != " ":
            continue
        coord = atom.get_coord()
        if not np.all(np.isfinite(coord)):
            skipped += 1
            continue
        if atom.get_name().strip() == "CA":
            ca_indices.append(len(atoms))
            key = (atom.get_parent().get_parent().get_id(),
                   atom.get_parent().get_id()[1])
            ca_map[key] = coord.copy()
        charge, r, e = _atom_params(atom)
        atoms.append(physics_mod.Particle(
            float(coord[0]), float(coord[1]), float(coord[2]),
            charge, r, e, False))
    if skipped:
        log(f"  ⚠  {skipped} atoms skipped (invalid coords)")
    return atoms, ca_indices, ca_map

def _parse_pdb_atoms_only(path, physics_mod):
    """Lightweight PDB parser — returns only the particle list, no index maps.
    Used by ComparisonWorker to quickly evaluate AlphaFold / SWISS-MODEL energies.
    """
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("prot", path)
    atoms = []
    for atom in st.get_atoms():
        if atom.get_parent().get_id()[0] != " ":
            continue
        coord = atom.get_coord()
        if not np.all(np.isfinite(coord)):
            continue
        charge, r, e = _atom_params(atom)
        atoms.append(physics_mod.Particle(
            float(coord[0]), float(coord[1]), float(coord[2]),
            charge, r, e, False))
    return atoms

def _ca_map_from_pdb(path):
    """Return (ca_map, avg_bfactor). avg_bfactor = avg pLDDT for AlphaFold files."""
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", path)
    ca_map, bfactors = {}, []
    for model in st:
        for chain in model:
            for res in chain:
                if res.get_id()[0] != " ":
                    continue
                for atom in res:
                    if atom.get_name().strip() == "CA":
                        key = (chain.get_id(), res.get_id()[1])
                        ca_map[key] = atom.get_coord().copy()
                        bfactors.append(atom.get_bfactor())
        break
    avg_b = float(np.mean(bfactors)) if bfactors else None
    return ca_map, avg_b

def _compute_rmsd(ca_map1, ca_map2):
    """Kabsch-superimposed Cα RMSD (Å) between two residue-keyed coordinate maps.

    Only residues present in both maps are compared.  Returns None when fewer
    than 3 common residues exist (Kabsch requires at least 3 points).
    The Kabsch rotation handles the reflection ambiguity via SVD sign correction.
    """
    common = sorted(set(ca_map1.keys()) & set(ca_map2.keys()))
    if len(common) < 3:
        return None
    c1 = np.array([ca_map1[k] for k in common], dtype=float)
    c2 = np.array([ca_map2[k] for k in common], dtype=float)
    c1 -= c1.mean(0); c2 -= c2.mean(0)
    H = c1.T @ c2
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = c1 - (c2 @ R.T)
    return float(np.sqrt((diff ** 2).sum(1).mean()))

def _compute_rmsf(snapshots, ca_indices):
    """Per-Cα root-mean-square fluctuation (Å) across all trajectory snapshots.

    snapshots   — list of particle-lists from LandscapeWorker
    ca_indices  — index of each Cα atom inside each particle list

    Returns a 1-D array of length len(ca_indices).  High RMSF values indicate
    flexible or disordered regions.
    """
    n_snaps = len(snapshots)
    n_ca    = len(ca_indices)
    if n_snaps == 0 or n_ca == 0:
        return np.array([])
    coords = np.zeros((n_snaps, n_ca, 3), dtype=float)
    for si, particles in enumerate(snapshots):
        for ci, pidx in enumerate(ca_indices):
            if pidx < len(particles):
                p = particles[pidx]
                coords[si, ci] = [p.x, p.y, p.z]
    mean_pos = coords.mean(axis=0)                          # [n_ca, 3]
    diff     = coords - mean_pos[np.newaxis]                # [n_snaps, n_ca, 3]
    return np.sqrt((diff ** 2).sum(axis=2).mean(axis=0))   # [n_ca]

def _extract_ca_residues(pdb_path):
    """Return [(chain_id, res_seq, res_name3)] for Cα residues in parse order."""
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", pdb_path)
    residues = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.get_id()[0] != " ":
                    continue
                for atom in res:
                    if atom.get_name().strip() == "CA":
                        residues.append(
                            (chain.get_id(), res.get_id()[1], res.get_resname()))
                        break
        break
    return residues

# ═══════════════════════════════════════════════════════════════════
#  PipelineWorker — download + parse + MC in background
# ═══════════════════════════════════════════════════════════════════
class PipelineWorker(QThread):
    """Background thread that runs the full analysis pipeline.

    Signals:
      progress(str)   — log message for the sidebar process log
      metrics(dict)   — partial metrics to update sidebar counters
      finished(...)   — emitted on success with (ensemble, energies, pdb_path,
                        ca_indices, ca_map, init_atoms)
      error(str)      — emitted on unrecoverable failure

    Steps (run() method):
      1. _fetch()     — retrieve PDB from disk, RCSB, or AlphaFold API
      2. _parse_pdb() — build Particle list + Cα index map
      3. generate_ensemble() — run MC via C++ engine (GIL released)
      4. calculate_potential() — compute final energies for each candidate
    """
    progress = pyqtSignal(str)
    metrics  = pyqtSignal(dict)
    # ensemble, energies, pdb_path, ca_indices, ca_map, init_atoms
    finished = pyqtSignal(object, object, str, object, object, object)
    error    = pyqtSignal(str)

    def __init__(self, engine, target, physics_mod, n_cand=5, steps=300):
        super().__init__()
        self.engine      = engine
        self.target      = target
        self.physics_mod = physics_mod
        self.n_cand      = n_cand
        self.steps       = steps

    def _fetch(self, target):
        """Resolve target to a local PDB file path.

        Search order:
          1. data/<target>.pdb on disk (exact or lower-case)
          2. RCSB PDB (4-char IDs only, via Bio.PDB.PDBList)
          3. AlphaFold DB REST API (UniProt accession IDs)
          4. Versioned AlphaFold model URLs (v4 → v3 → v2 fallback)
        Returns a path string or None on failure.
        """
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        for cand in [
            os.path.join(data_dir, f"{target}.pdb"),
            os.path.join(data_dir, f"{target.lower()}.pdb"),
            f"{target}.pdb", f"{target.lower()}.pdb",
        ]:
            if os.path.exists(cand):
                self.progress.emit(f"  Local file: {cand}")
                return cand

        dest = os.path.join(data_dir, f"{target}.pdb")
        if len(target) == 4:
            self.progress.emit("  Connecting to RCSB PDB…")
            try:
                pdbl = PDBList(verbose=False)
                raw = pdbl.retrieve_pdb_file(
                    target.lower(), pdir=data_dir, file_format="pdb", overwrite=True)
                if not raw or not os.path.exists(raw):
                    return None
                if os.path.exists(dest):
                    os.remove(dest)
                os.rename(raw, dest)
                return dest
            except Exception as ex:
                self.progress.emit(f"  RCSB failed: {ex}")
                return None
        else:
            self.progress.emit("  Querying AlphaFold API…")
            try:
                api = requests.get(
                    f"https://alphafold.ebi.ac.uk/api/prediction/{target}", timeout=15)
                if api.status_code == 200:
                    entries = api.json()
                    if entries and "pdbUrl" in entries[0]:
                        r = requests.get(entries[0]["pdbUrl"], timeout=30)
                        if r.status_code == 200:
                            with open(dest, "w") as f:
                                f.write(r.text)
                            return dest
            except Exception as ex:
                self.progress.emit(f"  AlphaFold API error: {ex}")
            self.progress.emit("  Trying versioned AlphaFold URLs…")
            for ver in ("v4", "v3", "v2"):
                url = f"https://alphafold.ebi.ac.uk/files/AF-{target}-F1-model_{ver}.pdb"
                try:
                    r = requests.get(url, timeout=15)
                    if r.status_code == 200:
                        with open(dest, "w") as f:
                            f.write(r.text)
                        self.progress.emit(f"  Found at model_{ver}")
                        return dest
                except Exception:
                    pass
            self.progress.emit("  AlphaFold: no structure found")
            return None

    def run(self):
        try:
            path = self._fetch(self.target)
            if not path:
                self.error.emit("Structure retrieval failed.")
                return
            self.progress.emit("  Parsing PDB + AMBER forcefield mapping…")
            atoms, ca_indices, ca_map = _parse_pdb(path, self.progress.emit, self.physics_mod)
            if not atoms:
                self.error.emit("No valid protein atoms found.")
                return
            self.metrics.emit({"n_atoms": len(atoms), "threads": self.engine.num_threads()})
            self.progress.emit(f"  {len(atoms)} atoms · {self.engine.num_threads()} threads")
            self.progress.emit(
                f"  Running MC: {self.n_cand} candidates × {self.steps} steps…")
            ensemble = self.engine.generate_ensemble(
                atoms, self.n_cand, self.steps, 0.6, 0.3)
            self.progress.emit("  Computing ensemble free energies…")
            energies = [self.engine.calculate_potential(s) for s in ensemble]
            self.metrics.emit({"best_e": min(energies), "n_cand": self.n_cand})
            self.finished.emit(ensemble, energies, path, ca_indices, ca_map, atoms)
        except Exception as ex:
            self.error.emit(str(ex))

# ═══════════════════════════════════════════════════════════════════
#  ComparisonWorker — fetch AlphaFold + SWISS-MODEL, compute RMSD
# ═══════════════════════════════════════════════════════════════════
class ComparisonWorker(QThread):
    progress = pyqtSignal(str)
    result   = pyqtSignal(list)

    def __init__(self, target, pdb_path, ca_indices, ref_ca_map,
                 ensemble, energies, engine, physics_mod):
        super().__init__()
        self.target      = target
        self.pdb_path    = pdb_path
        self.ca_indices  = ca_indices
        self.ref_ca_map  = ref_ca_map
        self.ensemble    = ensemble
        self.energies    = energies
        self.engine      = engine
        self.physics_mod = physics_mod

    def _pdb_to_uniprot(self, pdb_id):
        try:
            url = (f"https://rest.uniprot.org/uniprotkb/search"
                   f"?query=database%3Apdb%3A{pdb_id}&format=list&size=1")
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and r.text.strip():
                return r.text.strip().split("\n")[0]
        except Exception:
            pass
        return None

    def _fetch_alphafold(self, uniprot_id):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        dest = os.path.join(data_dir, f"AF_{uniprot_id}.pdb")
        if os.path.exists(dest):
            return dest
        try:
            api = requests.get(
                f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}", timeout=15)
            if api.status_code == 200:
                entries = api.json()
                if entries and "pdbUrl" in entries[0]:
                    r = requests.get(entries[0]["pdbUrl"], timeout=30)
                    if r.status_code == 200:
                        with open(dest, "w") as f:
                            f.write(r.text)
                        return dest
        except Exception as ex:
            self.progress.emit(f"  [CMP] AlphaFold fetch error: {ex}")
        return None

    def _fetch_swissmodel(self, uniprot_id):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        dest = os.path.join(data_dir, f"SM_{uniprot_id}.pdb")
        if os.path.exists(dest):
            return dest
        try:
            r = requests.get(
                f"https://swissmodel.expasy.org/repository/uniprot/{uniprot_id}.json",
                timeout=20)
            if r.status_code == 200:
                data = r.json()
                structs = data.get("result", {}).get("structures", [])
                if structs:
                    best = max(structs, key=lambda s: s.get("gmqe", 0))
                    coord_url = best.get("coordinates")
                    if coord_url:
                        r2 = requests.get(coord_url, timeout=30)
                        if r2.status_code == 200:
                            with open(dest, "w") as f:
                                f.write(r2.text)
                            return dest
        except Exception as ex:
            self.progress.emit(f"  [CMP] SWISS-MODEL error: {ex}")
        return None

    def _mc_ca_map(self, cand_idx):
        keys     = list(self.ref_ca_map.keys())
        particles = self.ensemble[cand_idx]
        ca_map   = {}
        for j, key in enumerate(keys):
            if j < len(self.ca_indices):
                pidx = self.ca_indices[j]
                if pidx < len(particles):
                    p = particles[pidx]
                    ca_map[key] = np.array([p.x, p.y, p.z])
        return ca_map

    def _energy_for(self, path):
        try:
            atoms = _parse_pdb_atoms_only(path, self.physics_mod)
            if atoms:
                return self.engine.calculate_potential(atoms)
        except Exception:
            pass
        return None

    def run(self):
        results  = []
        best_idx = int(np.argmin(self.energies))

        for i, energy in enumerate(self.energies):
            mc_ca = self._mc_ca_map(i)
            rmsd  = _compute_rmsd(self.ref_ca_map, mc_ca) if mc_ca else None
            results.append({
                "source": f"MC  C{i+1}", "is_mc": True, "mc_idx": i,
                "is_best": i == best_idx, "energy": energy,
                "rmsd": rmsd, "plddt": None, "path": self.pdb_path,
            })

        is_pdb_id  = len(self.target) == 4
        uniprot_id = None
        if is_pdb_id:
            self.progress.emit("  [CMP] Mapping PDB → UniProt…")
            uniprot_id = self._pdb_to_uniprot(self.target)
            if uniprot_id:
                self.progress.emit(f"  [CMP] UniProt: {uniprot_id}")
        else:
            uniprot_id = self.target

        if is_pdb_id and uniprot_id:
            self.progress.emit("  [CMP] Fetching AlphaFold structure…")
            af_path = self._fetch_alphafold(uniprot_id)
            if af_path:
                af_ca, avg_plddt = _ca_map_from_pdb(af_path)
                results.append({
                    "source": "AlphaFold", "is_mc": False, "is_best": False,
                    "energy": self._energy_for(af_path),
                    "rmsd": _compute_rmsd(self.ref_ca_map, af_ca),
                    "plddt": avg_plddt, "path": af_path,
                })

        if uniprot_id:
            self.progress.emit("  [CMP] Fetching SWISS-MODEL homology model…")
            sm_path = self._fetch_swissmodel(uniprot_id)
            if sm_path:
                sm_ca, _ = _ca_map_from_pdb(sm_path)
                results.append({
                    "source": "Homology  (SWISS-MODEL)", "is_mc": False, "is_best": False,
                    "energy": self._energy_for(sm_path),
                    "rmsd": _compute_rmsd(self.ref_ca_map, sm_ca),
                    "plddt": None, "path": sm_path,
                })

        self.progress.emit(f"  [CMP] Done — {len(results)} sources")
        self.result.emit(results)

# ═══════════════════════════════════════════════════════════════════
#  LandscapeWorker — MC trajectory → conformational graph
# ═══════════════════════════════════════════════════════════════════
class LandscapeWorker(QThread):
    progress = pyqtSignal(str)
    result   = pyqtSignal(dict)

    N_SNAPSHOTS    = 120   # total trajectory length
    STEPS_PER_SNAP = 80    # MC steps between each snapshot

    def __init__(self, engine, init_atoms, ca_indices, T=0.6, maxd=0.3):
        super().__init__()
        self.engine     = engine
        self.init_atoms = init_atoms
        self.ca_indices = ca_indices
        self.T          = T
        self.maxd       = maxd

    def _ca_vec(self, particles):
        """Flatten Cα coordinates of one snapshot into a 1-D vector."""
        v = []
        for i in self.ca_indices:
            if i < len(particles):
                p = particles[i]
                v += [p.x, p.y, p.z]
        return np.array(v, dtype=float)

    def run(self):
        import networkx as nx
        from sklearn.decomposition import PCA

        N, S = self.N_SNAPSHOTS, self.STEPS_PER_SNAP
        self.progress.emit(
            f"  [LANDSCAPE] Running {N}×{S}-step Markov chain…")

        snapshots, energies = [], []
        current = self.init_atoms
        for i in range(N):
            if i % 20 == 0:
                self.progress.emit(f"  [LANDSCAPE] Snapshot {i}/{N}…")
            # Each call advances the chain by S MC steps
            current = self.engine.generate_ensemble(
                current, 1, S, self.T, self.maxd)[0]
            snapshots.append(current)
            energies.append(self.engine.calculate_potential(current))

        energies = np.array(energies, dtype=float)

        self.progress.emit("  [LANDSCAPE] Building conformational graph…")

        # PCA layout — project the high-dim coordinate space to 2D
        coord_mat = np.array([self._ca_vec(s) for s in snapshots])
        n_feat = coord_mat.shape[1]
        n_pc   = min(2, N - 1, n_feat)
        pca    = PCA(n_components=n_pc)
        layout = pca.fit_transform(coord_mat)
        if layout.ndim == 1 or layout.shape[1] < 2:
            layout = np.column_stack([layout.reshape(-1, 1),
                                      np.zeros((N, 1))])
        var_exp = pca.explained_variance_ratio_.tolist()

        # Build the sequential conformational graph:
        # each snapshot = node, consecutive snapshots = edge
        G = nx.Graph()
        for i in range(N):
            G.add_node(i, energy=float(energies[i]))
        for i in range(N - 1):
            G.add_edge(i, i + 1,
                       weight=float(abs(energies[i + 1] - energies[i])))

        # Community detection — clusters are metastable basins
        communities = list(
            nx.algorithms.community.greedy_modularity_communities(G))
        node_comm = {node: ci
                     for ci, comm in enumerate(communities)
                     for node in comm}

        # IDP classification
        kT  = 0.592          # kcal/mol at 300 K
        sig = [c for c in communities if len(c) >= 0.05 * N]
        e_spread = 0.0
        if len(sig) > 1:
            mins = [min(float(energies[i]) for i in c) for c in sig]
            e_spread = float(max(mins) - min(mins))

        best_comm  = min(communities,
                         key=lambda c: min(float(energies[i]) for i in c))
        funnel     = len(best_comm) / N

        if len(sig) >= 3 and e_spread < 5 * kT:
            idp_label, idp_color = "IDP", "#dc2626"
        elif len(sig) >= 2 or funnel < 0.5:
            idp_label, idp_color = "POSSIBLY DISORDERED", "#d97706"
        else:
            idp_label, idp_color = "ORDERED", "#16a34a"

        self.progress.emit(
            f"  [LANDSCAPE] Done · {len(sig)} significant basins · "
            f"funnel={funnel:.2f} · {idp_label}")

        self.result.emit({
            "snapshots":    snapshots,
            "energies":     energies,
            "layout":       layout,
            "communities":  communities,
            "node_comm":    node_comm,
            "var_exp":      var_exp,
            "n_sig":        len(sig),
            "funnel":       float(funnel),
            "e_spread":     float(e_spread),
            "idp_label":    idp_label,
            "idp_color":    idp_color,
        })

# ═══════════════════════════════════════════════════════════════════
#  Helper widgets
# ═══════════════════════════════════════════════════════════════════
def _panel():
    f = QFrame(); f.setObjectName("panel"); return f

def _heading(text):
    l = QLabel(text.upper()); l.setObjectName("heading"); return l

def _metric_widget(label):
    w = QWidget()
    v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
    v.addWidget(_heading(label))
    val = QLabel("—"); val.setObjectName("metric-val")
    v.addWidget(val)
    return w, val

def _sep():
    l = QFrame(); l.setFrameShape(QFrame.Shape.HLine)
    l.setStyleSheet("color: #e2e8f0;"); return l

# ═══════════════════════════════════════════════════════════════════
#  Main GUI
# ═══════════════════════════════════════════════════════════════════
class ProteinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALMA — Protein Structure Analysis")
        self.setMinimumSize(1300, 800)

        self._physics_mod = protein_physics
        cuda_mod, gpu_name = _try_gpu_backend()
        if cuda_mod is not None:
            reply = QMessageBox.question(
                self, "GPU Detected",
                f"GPU found: {gpu_name}\n\nUse GPU acceleration?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._physics_mod = cuda_mod
        self._backend = (
            f"GPU  {gpu_name}" if self._physics_mod is not protein_physics else "CPU"
        )

        try:
            self.engine = self._physics_mod.PhysicsEngine()
        except Exception as ex:
            QMessageBox.critical(self, "Fatal",
                f"Failed to initialise physics engine ({self._backend}):\n{ex}")
            sys.exit(1)

        self._ensemble           = []
        self._energies           = []
        self._view_mode          = "layered"
        self._current_cand_idx   = 0
        self._comp_worker        = None
        self._landscape_worker   = None
        self._init_atoms         = None
        self._pdb_path           = None
        self._ca_indices         = []
        self._landscape_snaps    = []
        self._landscape_energies = np.array([])
        self._rmsf               = None
        self._rmsf_residues      = []
        self._rmsf_n_disordered  = 0
        self._rmsf_pct           = 0.0
        self._build_ui()
        self.setStyleSheet(STYLE)

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self):
        root  = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # ── Sidebar ──────────────────────────────────────────────
        sidebar = QVBoxLayout()
        sidebar.setSpacing(8)

        title = QLabel("ALMA")
        title.setStyleSheet("color:#d97706;font-size:22px;font-weight:bold;"
                            "letter-spacing:6px;padding:8px 0 2px 8px;")
        sub = QLabel("Atomistic Local Motion Analyzer")
        sub.setStyleSheet("color:#64748b;font-size:10px;letter-spacing:1px;padding:0 0 2px 8px;")
        backend_lbl = QLabel(f"⚙  {self._backend}")
        backend_lbl.setStyleSheet("color:#94a3b8;font-size:9px;letter-spacing:1px;padding:0 0 8px 8px;")
        sidebar.addWidget(title); sidebar.addWidget(sub)
        sidebar.addWidget(backend_lbl); sidebar.addWidget(_sep())

        inp_panel = _panel()
        inp_v = QVBoxLayout(inp_panel)
        inp_v.setContentsMargins(8, 4, 8, 10)
        inp_v.addWidget(_heading("Target"))
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("PDB ID  /  UniProt ID")
        self.id_input.returnPressed.connect(self._start)
        inp_v.addWidget(self.id_input)
        sidebar.addWidget(inp_panel)

        self.run_btn = QPushButton("▶  RUN ANALYSIS")
        self.run_btn.setObjectName("run-btn")
        self.run_btn.clicked.connect(self._start)
        sidebar.addWidget(self.run_btn)

        self.best_btn = QPushButton("SHOW BEST STRUCTURE")
        self.best_btn.setObjectName("sec-btn")
        self.best_btn.clicked.connect(self._show_best)
        self.best_btn.setEnabled(False)
        sidebar.addWidget(self.best_btn)

        self.landscape_start_btn = QPushButton("◈  EXPLORE LANDSCAPE")
        self.landscape_start_btn.setObjectName("sec-btn")
        self.landscape_start_btn.clicked.connect(self._start_landscape)
        self.landscape_start_btn.setEnabled(False)
        sidebar.addWidget(self.landscape_start_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setTextVisible(False)
        sidebar.addWidget(self.progress_bar)
        sidebar.addWidget(_sep())

        met_panel = _panel()
        met_g = QGridLayout(met_panel)
        met_g.setContentsMargins(4, 4, 4, 8); met_g.setSpacing(4)
        self._mw_atoms,   self._mv_atoms   = _metric_widget("ATOMS")
        self._mw_threads, self._mv_threads = _metric_widget("THREADS")
        self._mw_energy,  self._mv_energy  = _metric_widget("BEST ENERGY")
        self._mw_cand,    self._mv_cand    = _metric_widget("CANDIDATES")
        met_g.addWidget(self._mw_atoms,   0, 0)
        met_g.addWidget(self._mw_threads, 0, 1)
        met_g.addWidget(self._mw_energy,  1, 0)
        met_g.addWidget(self._mw_cand,    1, 1)
        sidebar.addWidget(met_panel); sidebar.addWidget(_sep())

        self.status_lbl = QLabel("IDLE")
        self.status_lbl.setObjectName("status-ok")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar.addWidget(self.status_lbl)

        log_panel = _panel()
        log_v = QVBoxLayout(log_panel)
        log_v.setContentsMargins(6, 4, 6, 6)
        log_v.addWidget(_heading("Process Log"))
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        log_v.addWidget(self.log)
        sidebar.addWidget(log_panel); sidebar.addStretch()

        # ── Viewer panel ─────────────────────────────────────────
        viewer_panel = _panel()
        viewer_v = QVBoxLayout(viewer_panel)
        viewer_v.setContentsMargins(0, 0, 0, 0); viewer_v.setSpacing(0)

        # Header bar
        viewer_header = QWidget(); viewer_header.setFixedHeight(36)
        vh_layout = QHBoxLayout(viewer_header)
        vh_layout.setContentsMargins(12, 0, 12, 0); vh_layout.setSpacing(8)
        viewer_title = QLabel("3D STRUCTURE VIEWER")
        viewer_title.setStyleSheet("color:#64748b;font-size:10px;letter-spacing:2px;")
        vh_layout.addWidget(viewer_title)
        vh_layout.addStretch()
        self.viewer_cand_lbl = QLabel("")
        self.viewer_cand_lbl.setStyleSheet("font-size:10px;letter-spacing:1px;")
        self.viewer_cand_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vh_layout.addWidget(self.viewer_cand_lbl)
        vh_layout.addStretch()
        self.view_mode_btn = QPushButton("◧  SIDE-BY-SIDE")
        self.view_mode_btn.setObjectName("sec-btn")
        self.view_mode_btn.clicked.connect(self._toggle_view_mode)
        self.view_mode_btn.setFixedHeight(24)
        vh_layout.addWidget(self.view_mode_btn)
        self.landscape_toggle_btn = QPushButton("◈  LANDSCAPE")
        self.landscape_toggle_btn.setObjectName("sec-btn")
        self.landscape_toggle_btn.clicked.connect(self._toggle_landscape)
        self.landscape_toggle_btn.setFixedHeight(24)
        self.landscape_toggle_btn.setEnabled(False)
        vh_layout.addWidget(self.landscape_toggle_btn)
        self.disorder_toggle_btn = QPushButton("⊛  DISORDER")
        self.disorder_toggle_btn.setObjectName("sec-btn")
        self.disorder_toggle_btn.clicked.connect(self._toggle_disorder)
        self.disorder_toggle_btn.setFixedHeight(24)
        self.disorder_toggle_btn.setEnabled(False)
        vh_layout.addWidget(self.disorder_toggle_btn)
        self._candidate_btns = []
        viewer_v.addWidget(viewer_header)

        # ── QStackedWidget: structure view ↔ landscape view ──────
        self._view_stack = QStackedWidget()

        # Page 0: 3Dmol web viewer
        self.web = QWebEngineView()
        self.web.setStyleSheet("border:none;")
        self.web.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self.web.loadFinished.connect(self._on_load_finished)
        self._html_tmpfile = os.path.join(tempfile.gettempdir(), "alma_viewer.html")
        self._render_empty()
        self._view_stack.addWidget(self.web)          # index 0

        # Page 1: energy landscape (matplotlib canvas + IDP badge)
        landscape_page = QWidget()
        lp_v = QVBoxLayout(landscape_page)
        lp_v.setContentsMargins(0, 0, 0, 0); lp_v.setSpacing(0)

        # IDP indicator bar
        idp_bar = QWidget(); idp_bar.setFixedHeight(30)
        idp_bar.setStyleSheet("background:#ffffff;border-bottom:1px solid #e2e8f0;")
        idp_h = QHBoxLayout(idp_bar)
        idp_h.setContentsMargins(16, 0, 16, 0)
        idp_title = QLabel("CONFORMATIONAL LANDSCAPE")
        idp_title.setStyleSheet("color:#64748b;font-size:10px;letter-spacing:2px;")
        idp_h.addWidget(idp_title); idp_h.addStretch()
        self.idp_hint_lbl = QLabel("click a node to view that conformation")
        self.idp_hint_lbl.setStyleSheet("color:#94a3b8;font-size:9px;letter-spacing:1px;")
        idp_h.addWidget(self.idp_hint_lbl)
        idp_h.addSpacing(20)
        idp_badge_lbl = QLabel("CLASSIFICATION:")
        idp_badge_lbl.setStyleSheet("color:#64748b;font-size:9px;letter-spacing:1px;")
        idp_h.addWidget(idp_badge_lbl)
        self.idp_status_lbl = QLabel("—")
        self.idp_status_lbl.setStyleSheet("font-size:10px;font-weight:bold;color:#94a3b8;margin-left:6px;")
        idp_h.addWidget(self.idp_status_lbl)
        lp_v.addWidget(idp_bar)

        # Matplotlib canvas
        self._landscape_fig = Figure(facecolor="#f8fafc", tight_layout=True)
        self._landscape_canvas = FigureCanvas(self._landscape_fig)
        self._landscape_canvas.setStyleSheet("border:none;")
        self._landscape_fig.canvas.mpl_connect("pick_event", self._on_graph_pick)
        lp_v.addWidget(self._landscape_canvas)

        self._view_stack.addWidget(landscape_page)   # index 1

        # Page 2: disorder / RMSF profile
        disorder_page = QWidget()
        dp_v = QVBoxLayout(disorder_page)
        dp_v.setContentsMargins(0, 0, 0, 0); dp_v.setSpacing(0)

        dp_hdr = QWidget(); dp_hdr.setFixedHeight(30)
        dp_hdr.setStyleSheet("background:#ffffff;border-bottom:1px solid #e2e8f0;")
        dp_h = QHBoxLayout(dp_hdr); dp_h.setContentsMargins(16, 0, 16, 0)
        dp_title = QLabel("RESIDUE FLEXIBILITY PROFILE")
        dp_title.setStyleSheet("color:#64748b;font-size:10px;letter-spacing:2px;")
        dp_h.addWidget(dp_title); dp_h.addStretch()
        self.disorder_stats_lbl = QLabel("—")
        self.disorder_stats_lbl.setStyleSheet("color:#94a3b8;font-size:9px;letter-spacing:1px;")
        dp_h.addWidget(self.disorder_stats_lbl)
        dp_h.addSpacing(16)
        self.flex_render_btn = QPushButton("COLOR BY FLEXIBILITY")
        self.flex_render_btn.setObjectName("sec-btn")
        self.flex_render_btn.setFixedHeight(22)
        self.flex_render_btn.setStyleSheet(
            "background:transparent;color:#7c3aed;border:1.5px solid #7c3aed;"
            "border-radius:4px;padding:2px 10px;font-size:9px;letter-spacing:1px;")
        self.flex_render_btn.clicked.connect(self._render_colored_by_rmsf)
        dp_h.addWidget(self.flex_render_btn)
        dp_v.addWidget(dp_hdr)

        self._disorder_fig    = Figure(facecolor="#f8fafc", tight_layout=True)
        self._disorder_canvas = FigureCanvas(self._disorder_fig)
        self._disorder_canvas.setStyleSheet("border:none;")
        dp_v.addWidget(self._disorder_canvas)

        self._view_stack.addWidget(disorder_page)    # index 2
        viewer_v.addWidget(self._view_stack)

        # Candidate energy bar
        self.ebar_widget = QWidget(); self.ebar_widget.setFixedHeight(40)
        self.ebar_widget.setVisible(False)
        ebar_layout = QHBoxLayout(self.ebar_widget)
        ebar_layout.setContentsMargins(12, 4, 12, 4); ebar_layout.setSpacing(6)
        self.ebar_legend_lbl = QLabel("")
        self.ebar_legend_lbl.setStyleSheet("font-size:9px;")
        ebar_layout.addWidget(self.ebar_legend_lbl)
        ebar_vsep = QFrame(); ebar_vsep.setFrameShape(QFrame.Shape.VLine)
        ebar_vsep.setStyleSheet("color:#e2e8f0;")
        ebar_layout.addWidget(ebar_vsep)
        self.ebar_labels = []
        viewer_v.addWidget(self.ebar_widget)

        # ── Comparison panel ──────────────────────────────────────
        self.comp_panel = QFrame()
        self.comp_panel.setObjectName("panel")
        self.comp_panel.setVisible(False)
        self.comp_panel.setMaximumHeight(200)
        comp_v = QVBoxLayout(self.comp_panel)
        comp_v.setContentsMargins(0, 0, 0, 4); comp_v.setSpacing(0)

        comp_hdr = QWidget(); comp_hdr.setFixedHeight(26)
        comp_hdr_layout = QHBoxLayout(comp_hdr)
        comp_hdr_layout.setContentsMargins(12, 0, 12, 0)
        comp_title_lbl = QLabel("STRUCTURE COMPARISON")
        comp_title_lbl.setStyleSheet("color:#64748b;font-size:10px;letter-spacing:2px;")
        comp_hdr_layout.addWidget(comp_title_lbl); comp_hdr_layout.addStretch()
        self.comp_ref_lbl = QLabel("")
        self.comp_ref_lbl.setStyleSheet("color:#94a3b8;font-size:9px;letter-spacing:1px;")
        comp_hdr_layout.addWidget(self.comp_ref_lbl)
        self.comp_status_lbl = QLabel("Fetching…")
        self.comp_status_lbl.setStyleSheet("color:#d97706;font-size:9px;margin-left:12px;")
        comp_hdr_layout.addWidget(self.comp_status_lbl)
        comp_v.addWidget(comp_hdr)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.comp_table_w = QWidget()
        self.comp_table_layout = QGridLayout(self.comp_table_w)
        self.comp_table_layout.setContentsMargins(12, 2, 12, 2)
        self.comp_table_layout.setSpacing(2)
        self.comp_table_layout.setColumnStretch(0, 3)
        self.comp_table_layout.setColumnStretch(1, 2)
        self.comp_table_layout.setColumnStretch(2, 2)
        self.comp_table_layout.setColumnStretch(3, 1)
        self.comp_table_layout.setColumnStretch(4, 1)
        for col, text in enumerate(["SOURCE", "ENERGY (kcal/mol)", "RMSD vs REF", "pLDDT", ""]):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "color:#94a3b8;font-size:9px;letter-spacing:1px;font-weight:bold;"
                "border-bottom:1px solid #e2e8f0;padding-bottom:2px;")
            self.comp_table_layout.addWidget(lbl, 0, col)
        scroll.setWidget(self.comp_table_w)
        comp_v.addWidget(scroll)
        viewer_v.addWidget(self.comp_panel)

        # ── Final assembly ────────────────────────────────────────
        left_w = QWidget(); left_w.setFixedWidth(280)
        left_w.setLayout(sidebar)
        outer.addWidget(left_w); outer.addWidget(viewer_panel)

    # ── Workflow ──────────────────────────────────────────────────

    def _start(self):
        target = self.id_input.text().strip().upper()
        if not target:
            return

        # Stop any still-running background workers
        for w in (getattr(self, "_comp_worker", None),
                  getattr(self, "_landscape_worker", None)):
            if w is not None and w.isRunning():
                w.terminate()
                w.wait(500)

        self.run_btn.setEnabled(False)
        self.best_btn.setEnabled(False)
        self.landscape_start_btn.setEnabled(False)
        self.landscape_toggle_btn.setEnabled(False)
        self.disorder_toggle_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_lbl.setText("RUNNING")
        self.status_lbl.setStyleSheet("color:#d97706;font-size:11px;font-weight:bold;")
        self.log.clear()
        self.comp_panel.setVisible(False)

        # Reset all secondary panels
        self.idp_status_lbl.setText("—")
        self.idp_status_lbl.setStyleSheet("font-size:10px;font-weight:bold;color:#94a3b8;")
        self.disorder_stats_lbl.setText("—")
        self._rmsf = None

        # Reset view to structure page and fix button labels
        self._view_stack.setCurrentIndex(0)
        self.view_mode_btn.setVisible(True)
        self.view_mode_btn.setText(
            "⊞  LAYERED" if self._view_mode == "sidebyside" else "◧  SIDE-BY-SIDE")
        self.landscape_toggle_btn.setText("◈  LANDSCAPE")
        self.disorder_toggle_btn.setText("⊛  DISORDER")

        self.landscape_start_btn.setText("◈  EXPLORE LANDSCAPE")
        self._log(f"[{target}] Analysis initiated")

        self.worker = PipelineWorker(self.engine, target, self._physics_mod)
        self.worker.progress.connect(self._log)
        self.worker.metrics.connect(self._on_metrics)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_metrics(self, d):
        if "n_atoms"  in d: self._mv_atoms.setText(str(d["n_atoms"]))
        if "threads"  in d: self._mv_threads.setText(str(d["threads"]))
        if "best_e"   in d: self._mv_energy.setText(f"{d['best_e']:.0f}")
        if "n_cand"   in d: self._mv_cand.setText(str(d["n_cand"]))

    def _on_done(self, ensemble, energies, pdb_path, ca_indices, ca_map, init_atoms):
        self._ensemble    = ensemble
        self._energies    = energies
        self._init_atoms  = init_atoms
        self._ca_indices  = ca_indices
        self._pdb_path    = pdb_path
        self.run_btn.setEnabled(True)
        self.best_btn.setEnabled(True)
        self.landscape_start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_lbl.setText("COMPLETE")
        self.status_lbl.setStyleSheet("color:#16a34a;font-size:11px;font-weight:bold;")

        best_idx = int(np.argmin(energies))
        self._log("─" * 36)
        for i, e in enumerate(energies):
            tag = " ◀ BEST" if i == best_idx else ""
            self._log(f"  Candidate {i+1:02d}  {e:>12.2f} kcal/mol{tag}")
        self._log("─" * 36)

        self._build_candidate_bar(energies, best_idx)
        self._render(best_idx)

        target = self.id_input.text().strip().upper()
        ref_label = "Crystal PDB" if len(target) == 4 else "AlphaFold input"
        self.comp_ref_lbl.setText(f"RMSD ref: {ref_label}")
        self.comp_status_lbl.setText("Fetching…")
        self.comp_status_lbl.setStyleSheet("color:#d97706;font-size:9px;margin-left:12px;")
        self.comp_panel.setVisible(True)
        self._clear_comp_table_rows()

        self._comp_worker = ComparisonWorker(
            target, pdb_path, ca_indices, ca_map,
            ensemble, energies, self.engine, self._physics_mod)
        self._comp_worker.progress.connect(self._log)
        self._comp_worker.result.connect(self._on_comparison_result)
        self._comp_worker.start()

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_lbl.setText("ERROR")
        self.status_lbl.setStyleSheet("color:#dc2626;font-size:11px;font-weight:bold;")
        self._log(f"[ERROR] {msg}")

    def _show_best(self):
        if not self._ensemble:
            return
        if self._view_stack.currentIndex() != 0:
            self._view_stack.setCurrentIndex(0)
        self._render(int(np.argmin(self._energies)))

    def _log(self, msg):
        self.log.append(msg)

    def _on_load_finished(self, ok):
        if not ok:
            self._log("[WEB] loadFinished → FAILED")

    def _set_html(self, html):
        with open(self._html_tmpfile, "w", encoding="utf-8") as f:
            f.write(html)
        self.web.setUrl(QUrl.fromLocalFile(self._html_tmpfile))

    # ── Comparison table ──────────────────────────────────────────

    def _clear_comp_table_rows(self):
        for row in range(1, self.comp_table_layout.rowCount()):
            for col in range(self.comp_table_layout.columnCount()):
                item = self.comp_table_layout.itemAtPosition(row, col)
                if item and item.widget():
                    item.widget().deleteLater()

    def _on_comparison_result(self, results):
        self.comp_status_lbl.setText(f"{len(results)} sources")
        self.comp_status_lbl.setStyleSheet("color:#16a34a;font-size:9px;margin-left:12px;")
        self._clear_comp_table_rows()
        for row_i, entry in enumerate(results, start=1):
            is_best = entry.get("is_best", False)
            is_mc   = entry.get("is_mc", False)
            prefix  = "★ " if is_best else "   "
            src_lbl = QLabel(prefix + entry["source"])
            color   = "#16a34a" if is_best else ("#7c3aed" if not is_mc else "#475569")
            src_lbl.setStyleSheet(
                f"color:{color};font-size:10px;"
                + ("font-weight:bold;" if is_best else ""))
            self.comp_table_layout.addWidget(src_lbl, row_i, 0)

            energy = entry.get("energy")
            e_lbl  = QLabel(f"{energy:.1f}" if energy is not None else "—")
            e_lbl.setStyleSheet("font-size:10px;color:#1e293b;")
            self.comp_table_layout.addWidget(e_lbl, row_i, 1)

            rmsd  = entry.get("rmsd")
            r_lbl = QLabel(f"{rmsd:.2f} Å" if rmsd is not None else "—")
            r_lbl.setStyleSheet("font-size:10px;color:#1e293b;")
            self.comp_table_layout.addWidget(r_lbl, row_i, 2)

            plddt  = entry.get("plddt")
            p_lbl  = QLabel(f"{plddt:.1f}" if plddt is not None else "—")
            if plddt is not None:
                pcol = "#16a34a" if plddt >= 70 else ("#d97706" if plddt >= 50 else "#dc2626")
                p_lbl.setStyleSheet(f"font-size:10px;color:{pcol};font-weight:bold;")
            else:
                p_lbl.setStyleSheet("font-size:10px;color:#94a3b8;")
            self.comp_table_layout.addWidget(p_lbl, row_i, 3)

            view_btn = QPushButton("VIEW")
            view_btn.setFixedSize(48, 20)
            view_btn.setStyleSheet(
                "background:transparent;color:#1d4ed8;border:1px solid #1d4ed8;"
                "border-radius:3px;font-size:9px;padding:0;letter-spacing:1px;")
            en = entry.copy()
            view_btn.clicked.connect(lambda _, e=en: self._render_source(e))
            self.comp_table_layout.addWidget(view_btn, row_i, 4)

    def _render_source(self, entry):
        if self._view_stack.currentIndex() != 0:
            self._view_stack.setCurrentIndex(0)
            self.view_mode_btn.setVisible(True)
            self.landscape_toggle_btn.setText("◈  LANDSCAPE")
        if entry.get("is_mc"):
            self._render(entry["mc_idx"])
        else:
            path = entry.get("path")
            if path and os.path.exists(path):
                self._render_external_pdb(path, entry["source"])

    def _render_external_pdb(self, path, source_name):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                pdb_text = f.read()
        except Exception as ex:
            self._log(f"[EXT] Cannot read {path}: {ex}"); return

        pdb_esc = pdb_text.replace("\\", "\\\\").replace("`", "\\`")
        is_af   = "AlphaFold" in source_name
        color   = "#7c3aed" if is_af else "#d97706"
        label   = source_name.upper()
        n_atoms = pdb_text.count("\nATOM")

        html = f"""<!DOCTYPE html><html><head>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ margin:0;padding:0;box-sizing:border-box; }}
  body {{ background:#f8fafc;overflow:hidden; }}
  #v {{ width:100vw;height:100vh; }}
  #info {{
    position:absolute;top:12px;left:16px;font-family:monospace;font-size:11px;
    letter-spacing:1px;color:#1e293b;pointer-events:none;
    background:rgba(255,255,255,0.88);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0;
  }}
  #legend {{
    position:absolute;bottom:12px;left:16px;font-family:monospace;font-size:10px;
    letter-spacing:1px;color:#475569;pointer-events:none;
    background:rgba(255,255,255,0.88);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0;
  }}
</style></head><body>
<div id="v"></div>
<div id="info">{label} &nbsp;&middot;&nbsp; {n_atoms} ATOMS</div>
<div id="legend"><span style="color:{color}">&#9632;</span> {label}</div>
<script>
(function(){{
  var v=$3Dmol.createViewer("v",{{backgroundColor:"#f8fafc"}});
  var m=v.addModel(`{pdb_esc}`,"pdb");
  m.setStyle({{}},{{cartoon:{{color:"{color}",thickness:0.8,opacity:1.0}},
                    sphere:{{color:"{color}",radius:0.55,opacity:1.0}}}});
  v.zoomTo(); v.zoom(0.85); v.render();
  setInterval(function(){{ v.rotate(1,'y'); v.render(); }},50);
}})();
</script></body></html>"""
        self.viewer_cand_lbl.setText(
            f'<span style="color:{color};font-weight:bold;">{label}</span>')
        self._set_html(html)

    # ── Energy landscape ──────────────────────────────────────────

    def _start_landscape(self):
        if self._init_atoms is None:
            return
        if self._landscape_worker is not None and self._landscape_worker.isRunning():
            return
        self.landscape_start_btn.setEnabled(False)
        self.landscape_start_btn.setText("◈  COMPUTING…")
        self.idp_status_lbl.setText("…")
        self.idp_status_lbl.setStyleSheet("font-size:10px;font-weight:bold;color:#d97706;")
        self._log("[LANDSCAPE] Starting Markov-chain exploration…")

        # Fresh engine instance to avoid thread contention with ComparisonWorker
        try:
            ls_engine = self._physics_mod.PhysicsEngine()
        except Exception as ex:
            self._log(f"[LANDSCAPE] Engine init failed: {ex}")
            self.landscape_start_btn.setEnabled(True)
            self.landscape_start_btn.setText("◈  EXPLORE LANDSCAPE")
            return

        self._landscape_worker = LandscapeWorker(
            ls_engine, self._init_atoms, self._ca_indices)
        self._landscape_worker.progress.connect(self._log)
        self._landscape_worker.result.connect(self._on_landscape_done)
        self._landscape_worker.start()

    def _on_landscape_done(self, data):
        self._landscape_snaps    = data["snapshots"]
        self._landscape_energies = data["energies"]
        self.landscape_start_btn.setText("◈  RE-EXPLORE")
        self.landscape_start_btn.setEnabled(True)
        self.landscape_toggle_btn.setEnabled(True)

        lbl   = data["idp_label"]
        color = data["idp_color"]
        self.idp_status_lbl.setText(lbl)
        self.idp_status_lbl.setStyleSheet(
            f"font-size:10px;font-weight:bold;color:{color};margin-left:6px;")

        self._draw_landscape(data)

        # Compute RMSF from trajectory and draw disorder profile
        if self._ca_indices and self._pdb_path:
            rmsf     = _compute_rmsf(data["snapshots"], self._ca_indices)
            residues = _extract_ca_residues(self._pdb_path)
            self._rmsf          = rmsf
            self._rmsf_residues = residues
            self._draw_disorder_profile(rmsf, residues)
            self.disorder_toggle_btn.setEnabled(True)

        self._log(f"[LANDSCAPE] Classification: {lbl}  ·  "
                  f"{data['n_sig']} metastable basins  ·  "
                  f"funnel={data['funnel']:.2f}")

    def _draw_landscape(self, data):
        """Render the conformational graph on the matplotlib canvas."""
        self._landscape_fig.clear()
        ax = self._landscape_fig.add_subplot(111)
        ax.set_facecolor("#f8fafc")
        self._landscape_fig.patch.set_facecolor("#f8fafc")

        layout      = data["layout"]           # (N, 2)
        energies    = data["energies"]          # (N,)
        communities = data["communities"]
        N           = len(energies)
        cmap_comm   = plt.cm.tab10

        # ── Trajectory path (thin grey edges) ──────────────────
        for i in range(N - 1):
            ax.plot([layout[i, 0], layout[i + 1, 0]],
                    [layout[i, 1], layout[i + 1, 1]],
                    color="#cbd5e1", lw=0.5, alpha=0.35, zorder=1)

        # ── Community convex hulls ──────────────────────────────
        try:
            from scipy.spatial import ConvexHull
            from matplotlib.patches import Polygon as MplPolygon
            for ci, comm in enumerate(communities):
                if len(comm) < 3:
                    continue
                pts = layout[sorted(comm)]
                try:
                    hull = ConvexHull(pts)
                    verts = pts[hull.vertices]
                    col = cmap_comm(ci % 10)
                    poly = MplPolygon(verts, alpha=0.07,
                                      facecolor=col, edgecolor=col,
                                      linewidth=1.2, linestyle="--", zorder=2)
                    ax.add_patch(poly)
                except Exception:
                    pass
        except ImportError:
            pass   # scipy optional

        # ── Nodes (scatter, colored by energy, pickable) ────────
        sc = ax.scatter(
            layout[:, 0], layout[:, 1],
            c=energies, cmap="RdYlGn_r",
            s=55, zorder=3, picker=8,
            edgecolors="#ffffff", linewidths=0.4, alpha=0.88)

        # ── Special markers ─────────────────────────────────────
        ax.scatter(*layout[0],  c="#22c55e", s=130, zorder=5, marker="s",
                   edgecolors="#fff", linewidths=0.8, label="Start")
        ax.scatter(*layout[-1], c="#f97316", s=130, zorder=5, marker="D",
                   edgecolors="#fff", linewidths=0.8, label="End")
        best_i = int(np.argmin(energies))
        ax.scatter(*layout[best_i], c="#1d4ed8", s=200, zorder=6, marker="*",
                   edgecolors="#fff", linewidths=0.8,
                   label=f"Min E (#{best_i + 1})")

        # ── Basin centroid labels ────────────────────────────────
        for ci, comm in enumerate(communities):
            pts = layout[sorted(comm)]
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            col = cmap_comm(ci % 10)
            ax.text(cx, cy, f"B{ci + 1}", fontsize=7, color=col,
                    fontweight="bold", ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="#ffffffcc", ec=col, lw=0.8, alpha=0.9))

        # ── Colorbar ────────────────────────────────────────────
        cbar = self._landscape_fig.colorbar(
            sc, ax=ax, shrink=0.65, pad=0.01, aspect=20)
        cbar.set_label("Energy (kcal/mol)", fontsize=7, color="#64748b")
        cbar.ax.tick_params(labelsize=6, colors="#94a3b8")

        # ── Labels & style ───────────────────────────────────────
        var = data.get("var_exp", [0, 0])
        ax.set_xlabel(f"PC1  ({var[0]*100:.1f}% var.)",
                      fontsize=8, color="#64748b", labelpad=4)
        ax.set_ylabel(f"PC2  ({var[1]*100:.1f}% var.)",
                      fontsize=8, color="#64748b", labelpad=4)
        ax.set_title(
            f"Energy Landscape  ·  {data['n_sig']} metastable basins  ·  "
            f"funnel={data['funnel']:.2f}  ·  {data['idp_label']}",
            fontsize=9, color="#1e293b", fontweight="bold", pad=8)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#e2e8f0"); sp.set_linewidth(0.8)
        ax.legend(fontsize=7, framealpha=0.88, edgecolor="#e2e8f0",
                  facecolor="#ffffff", labelcolor="#1e293b", loc="best")

        self._landscape_canvas.draw()

    def _on_graph_pick(self, event):
        """User clicked a node in the landscape graph — load that snapshot."""
        if not hasattr(event, "ind") or len(event.ind) == 0:
            return
        snap_idx = int(event.ind[0])
        if snap_idx >= len(self._landscape_snaps):
            return

        particles = self._landscape_snaps[snap_idx]
        energy    = float(self._landscape_energies[snap_idx])
        pdb_str   = self._build_pdb_str(particles)
        n_atoms   = len(particles)

        html = f"""<!DOCTYPE html><html><head>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ margin:0;padding:0;box-sizing:border-box; }}
  body {{ background:#f8fafc;overflow:hidden; }}
  #v {{ width:100vw;height:100vh; }}
  #info {{
    position:absolute;top:12px;left:16px;font-family:monospace;font-size:11px;
    letter-spacing:1px;color:#1e293b;pointer-events:none;
    background:rgba(255,255,255,0.88);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0;
  }}
  #legend {{
    position:absolute;bottom:12px;left:16px;font-family:monospace;font-size:10px;
    letter-spacing:1px;color:#475569;pointer-events:none;
    background:rgba(255,255,255,0.88);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0;
  }}
</style></head><body>
<div id="v"></div>
<div id="info">SNAPSHOT #{snap_idx + 1} &nbsp;&middot;&nbsp; {energy:.1f} kcal/mol \
&nbsp;&middot;&nbsp; {n_atoms} ATOMS</div>
<div id="legend"><span style="color:#0891b2">&#9632;</span> LANDSCAPE SNAPSHOT</div>
<script>
(function(){{
  var v=$3Dmol.createViewer("v",{{backgroundColor:"#f8fafc"}});
  var m=v.addModel(`{pdb_str}`,"pdb");
  m.setStyle({{}},{{cartoon:{{color:"#0891b2",thickness:0.8,opacity:1.0}},
                    sphere:{{color:"#0891b2",radius:0.55,opacity:1.0}}}});
  v.zoomTo(); v.zoom(0.85); v.render();
  setInterval(function(){{ v.rotate(1,'y'); v.render(); }},50);
}})();
</script></body></html>"""

        self.viewer_cand_lbl.setText(
            f'<span style="color:#0891b2;font-weight:bold;">'
            f'SNAPSHOT #{snap_idx + 1}</span> · {energy:.1f} kcal/mol')
        self._set_html(html)
        # Auto-switch to structure view so the user sees the loaded conformation
        self._view_stack.setCurrentIndex(0)
        self.view_mode_btn.setVisible(True)
        self.landscape_toggle_btn.setText("◈  LANDSCAPE")

    def _toggle_landscape(self):
        if self._view_stack.currentIndex() == 0:
            self._view_stack.setCurrentIndex(1)
            self.view_mode_btn.setVisible(False)
            self.landscape_toggle_btn.setText("⊡  STRUCTURE")
        else:
            self._view_stack.setCurrentIndex(0)
            self.view_mode_btn.setVisible(True)
            self.landscape_toggle_btn.setText("◈  LANDSCAPE")

    # ── Candidate energy bar ──────────────────────────────────────

    def _build_candidate_bar(self, energies, best_idx):
        layout = self.ebar_widget.layout()
        for btn in self._candidate_btns:
            layout.removeWidget(btn); btn.deleteLater()
        self._candidate_btns.clear()

        e_min, e_max = min(energies), max(energies)
        e_range      = max(abs(e_max - e_min), 1.0)
        best_col     = f"#{22:02x}{163:02x}{74:02x}"
        worst_col    = f"#{min(22+210,255):02x}{max(163-120,0):02x}{0:02x}"
        self.ebar_legend_lbl.setText(
            f'<span style="color:{best_col}">■</span>'
            f' LOWEST ENERGY (BEST) &nbsp;·····&nbsp; '
            f'<span style="color:{worst_col}">■</span>'
            f' HIGHEST ENERGY (WORST) &nbsp;&nbsp;'
            f'<span style="color:#94a3b8">'
            f'{e_min:.0f} → {e_max:.0f} kcal/mol</span>'
        )
        for i, e in enumerate(energies):
            norm  = (e - e_min) / e_range
            r, g, b = int(22+210*norm), int(163-120*norm), int(74*(1-norm))
            color = f"#{r:02x}{g:02x}{b:02x}"
            label = f"★ C{i+1}" if i == best_idx else f"C{i+1}"
            btn   = QPushButton(label)
            if i == best_idx:
                btn.setStyleSheet(
                    f"background:{color};color:#fff;border:none;border-radius:3px;"
                    f"font-size:10px;font-weight:bold;padding:4px 10px;")
            else:
                btn.setStyleSheet(
                    f"background:#ffffff;color:{color};"
                    f"border:1.5px solid {color};border-radius:3px;"
                    f"font-size:10px;padding:4px 10px;")
            idx = i
            btn.clicked.connect(lambda _, ii=idx: self._render(ii))
            layout.addWidget(btn)
            self._candidate_btns.append(btn)
        layout.addStretch()
        self.ebar_widget.setVisible(True)

    # ── 3D rendering ──────────────────────────────────────────────

    def _render_empty(self):
        self.web.setHtml("""<!DOCTYPE html><html>
<body style="margin:0;background:#f8fafc;display:flex;align-items:center;
             justify-content:center;height:100vh;">
  <div style="text-align:center;font-family:monospace;">
    <div style="color:#cbd5e1;font-size:48px;letter-spacing:8px;">◈</div>
    <div style="color:#94a3b8;font-size:11px;letter-spacing:3px;margin-top:16px;">
      AWAITING STRUCTURE</div>
  </div>
</body></html>""")

    def _toggle_view_mode(self):
        if self._view_mode == "layered":
            self._view_mode = "sidebyside"
            self.view_mode_btn.setText("⊞  LAYERED")
        else:
            self._view_mode = "layered"
            self.view_mode_btn.setText("◧  SIDE-BY-SIDE")
        if self._ensemble:
            self._render(self._current_cand_idx)

    def _build_pdb_str(self, particles):
        lines = []
        for i, p in enumerate(particles):
            lines.append(
                f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00  0.50           C")
        return "\n".join(lines)

    def _update_candidate_bar_selection(self, active_idx):
        if not self._candidate_btns or not self._energies:
            return
        best_idx = int(np.argmin(self._energies))
        n = len(self._energies)
        if active_idx < n:
            e = self._energies[active_idx]
            tag_html = (
                '&nbsp;&nbsp;<span style="color:#16a34a;font-weight:bold;">★ BEST</span>'
                if active_idx == best_idx
                else f'&nbsp;&nbsp;<span style="color:#94a3b8;">(best: C{best_idx+1})</span>')
            self.viewer_cand_lbl.setText(
                f'<b>CANDIDATE {active_idx+1} / {n}</b>'
                f'&nbsp;&nbsp;·&nbsp;&nbsp;{e:.1f} kcal/mol{tag_html}')
        e_min, e_max = min(self._energies), max(self._energies)
        e_range = max(abs(e_max - e_min), 1.0)
        for i, btn in enumerate(self._candidate_btns):
            if i >= len(self._energies):
                break
            norm = (self._energies[i] - e_min) / e_range
            r, g, bv = int(22+210*norm), int(163-120*norm), int(74*(1-norm))
            col = f"#{r:02x}{g:02x}{bv:02x}"
            is_best = (i == best_idx); is_active = (i == active_idx)
            if is_best and is_active:
                btn.setStyleSheet(
                    f"background:{col};color:#fff;border:2px solid #fff;"
                    f"border-radius:3px;font-size:10px;font-weight:bold;padding:4px 10px;")
            elif is_best:
                btn.setStyleSheet(
                    f"background:{col};color:#fff;border:none;"
                    f"border-radius:3px;font-size:10px;font-weight:bold;padding:4px 10px;")
            elif is_active:
                btn.setStyleSheet(
                    f"background:#eff6ff;color:{col};border:2px solid {col};"
                    f"border-radius:3px;font-size:10px;font-weight:bold;padding:4px 10px;")
            else:
                btn.setStyleSheet(
                    f"background:#ffffff;color:{col};border:1.5px solid {col};"
                    f"border-radius:3px;font-size:10px;padding:4px 10px;")

    def _render(self, cand_idx=0):
        self._log(f"[RENDER] mode={self._view_mode}  cand={cand_idx}  "
                  f"ensemble={len(self._ensemble)}")
        self._current_cand_idx = cand_idx
        if self._energies:
            self._update_candidate_bar_selection(cand_idx)
        if not self._ensemble:
            self._log("[RENDER] ensemble empty — aborting"); return
        try:
            if self._view_mode == "sidebyside":
                self._render_sidebyside(cand_idx)
            else:
                self._render_layered(cand_idx)
        except Exception as ex:
            self._log(f"[RENDER ERROR] {ex}\n{traceback.format_exc()}")

    def _render_layered(self, selected_idx):
        best_idx      = int(np.argmin(self._energies))
        best_particles = self._ensemble[best_idx]
        n_atoms        = len(best_particles)
        pdb_best       = self._build_pdb_str(best_particles)
        best_e = (f"{self._energies[best_idx]:.1f} kcal/mol"
                  if best_idx < len(self._energies) else "")
        sel_e  = (f"{self._energies[selected_idx]:.1f} kcal/mol"
                  if selected_idx < len(self._energies) else "")
        best_js = (
            f'  var mBest=v.addModel(`{pdb_best}`,"pdb");\n'
            f'  mBest.setStyle({{}},{{cartoon:{{color:"#1d4ed8",thickness:0.8,opacity:1.0}},'
            f'sphere:{{color:"#1d4ed8",radius:0.55,opacity:1.0}}}});')
        if selected_idx != best_idx:
            pdb_sel = self._build_pdb_str(self._ensemble[selected_idx])
            sel_js  = (
                f'  var mSel=v.addModel(`{pdb_sel}`,"pdb");\n'
                f'  mSel.setStyle({{}},{{cartoon:{{color:"#0891b2",thickness:0.6,opacity:0.65}},'
                f'sphere:{{color:"#0891b2",radius:0.50,opacity:0.60}}}});')
            label  = (f"LAYERED &nbsp; C{selected_idx+1} ({sel_e})"
                      f" &nbsp;over&nbsp; C{best_idx+1} BEST ({best_e})")
            legend = ('<div id="legend">OVERLAY &nbsp; '
                      '<span style="color:#1d4ed8">&#9632;</span> BEST &nbsp;'
                      '<span style="color:#0891b2">&#9632;</span> SELECTED</div>')
        else:
            sel_js = ""
            label  = f"&#9733; BEST &nbsp; C{best_idx+1} &nbsp;&middot;&nbsp; {best_e}"
            legend = ('<div id="legend">'
                      '<span style="color:#1d4ed8">&#9632;</span> BEST CANDIDATE</div>')
        html = f"""<!DOCTYPE html><html><head>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ margin:0;padding:0;box-sizing:border-box; }}
  body {{ background:#f8fafc;overflow:hidden; }}
  #v {{ width:100vw;height:100vh; }}
  #info {{ position:absolute;top:12px;left:16px;font-family:monospace;font-size:11px;
    letter-spacing:1px;color:#1e293b;pointer-events:none;
    background:rgba(255,255,255,0.88);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0; }}
  #legend {{ position:absolute;bottom:12px;left:16px;font-family:monospace;font-size:10px;
    letter-spacing:1px;color:#475569;pointer-events:none;
    background:rgba(255,255,255,0.88);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0; }}
</style></head><body>
<div id="v"></div>
<div id="info">{label} &nbsp;&middot;&nbsp; {n_atoms} ATOMS</div>
{legend}
<script>
(function(){{
  var v=$3Dmol.createViewer("v",{{backgroundColor:"#f8fafc"}});
{best_js}
{sel_js}
  v.zoomTo(); v.zoom(0.85); v.render();
  setInterval(function(){{ v.rotate(1,'y'); v.render(); }},50);
}})();
</script></body></html>"""
        self._log(f"[LAYERED] html={len(html.encode())} bytes")
        self._set_html(html)

    def _render_sidebyside(self, selected_idx):
        best_idx       = int(np.argmin(self._energies))
        best_particles = self._ensemble[best_idx]
        sel_particles  = self._ensemble[selected_idx]
        n_atoms        = len(best_particles)
        pdb_best       = self._build_pdb_str(best_particles)
        best_e = (f"{self._energies[best_idx]:.1f} kcal/mol"
                  if best_idx < len(self._energies) else "")
        sel_e  = (f"{self._energies[selected_idx]:.1f} kcal/mol"
                  if selected_idx < len(self._energies) else "")
        left_js = (
            f'  var mL=vL.addModel(`{pdb_best}`,"pdb");\n'
            f'  mL.setStyle({{}},{{cartoon:{{color:"#1d4ed8",thickness:0.8,opacity:1.0}},'
            f'sphere:{{color:"#1d4ed8",radius:0.55,opacity:1.0}}}});')
        if selected_idx != best_idx and len(sel_particles) == len(best_particles):
            devs = [
                ((p.x-r.x)**2 + (p.y-r.y)**2 + (p.z-r.z)**2) ** 0.5
                for p, r in zip(sel_particles, best_particles)]
            max_dev = max(devs) if max(devs) > 0 else 1.0
            threshold = max_dev * 0.3
            sim_lines, diff_lines = [], []
            for i, p in enumerate(sel_particles):
                line = (f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                        f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00  0.50           C")
                (diff_lines if devs[i] > threshold else sim_lines).append(line)
            pdb_sim  = "\n".join(sim_lines)
            pdb_diff = "\n".join(diff_lines)
            right_parts = []
            if sim_lines:
                right_parts.append(
                    f'  var mSim=vR.addModel(`{pdb_sim}`,"pdb");\n'
                    f'  mSim.setStyle({{}},{{cartoon:{{color:"#374151",thickness:0.55,opacity:0.8}},'
                    f'sphere:{{color:"#374151",radius:0.45,opacity:0.75}}}});')
            if diff_lines:
                right_parts.append(
                    f'  var mDiff=vR.addModel(`{pdb_diff}`,"pdb");\n'
                    f'  mDiff.setStyle({{}},{{cartoon:{{color:"#f97316",thickness:0.75,opacity:1.0}},'
                    f'sphere:{{color:"#f97316",radius:0.62,opacity:1.0}}}});')
            right_js     = "\n".join(right_parts)
            right_info   = f"C{selected_idx+1} &nbsp;&middot;&nbsp; {sel_e} &nbsp;&middot;&nbsp; {n_atoms} ATOMS"
            right_legend = (f'<span style="color:#374151">&#9632;</span> SIMILAR &nbsp;'
                            f'<span style="color:#f97316">&#9632;</span> DIFFERS ({len(diff_lines)} atoms)')
        else:
            right_js     = (
                f'  var mR=vR.addModel(`{pdb_best}`,"pdb");\n'
                f'  mR.setStyle({{}},{{cartoon:{{color:"#1d4ed8",thickness:0.8,opacity:1.0}},'
                f'sphere:{{color:"#1d4ed8",radius:0.55,opacity:1.0}}}});')
            right_info   = f"&#9733; C{best_idx+1} (BEST) &nbsp;&middot;&nbsp; {best_e} &nbsp;&middot;&nbsp; {n_atoms} ATOMS"
            right_legend = '<span style="color:#1d4ed8">&#9632;</span> BEST CANDIDATE'
        html = f"""<!DOCTYPE html><html><head>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ margin:0;padding:0;box-sizing:border-box; }}
  body {{ background:#f8fafc;overflow:hidden;display:flex;width:100vw;height:100vh; }}
  .vpane {{ flex:1;position:relative;height:100%; }}
  .divider {{ width:2px;background:#e2e8f0;flex-shrink:0; }}
  .info {{ position:absolute;top:12px;left:12px;font-family:monospace;font-size:11px;
    letter-spacing:1px;color:#1e293b;pointer-events:none;
    background:rgba(255,255,255,0.9);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0;z-index:10; }}
  .legend {{ position:absolute;bottom:12px;left:12px;font-family:monospace;font-size:10px;
    letter-spacing:1px;color:#475569;pointer-events:none;
    background:rgba(255,255,255,0.9);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0;z-index:10; }}
  .pane-lbl {{ position:absolute;top:12px;right:12px;font-family:monospace;font-size:9px;
    letter-spacing:2px;color:#94a3b8;pointer-events:none;z-index:10; }}
</style></head><body>
<div class="vpane">
  <div id="vL" style="width:100%;height:100%;"></div>
  <div class="info">&#9733; C{best_idx+1} (BEST) &nbsp;&middot;&nbsp; {best_e} &nbsp;&middot;&nbsp; {n_atoms} ATOMS</div>
  <div class="pane-lbl">BEST CANDIDATE</div>
  <div class="legend"><span style="color:#1d4ed8">&#9632;</span> BEST CANDIDATE</div>
</div>
<div class="divider"></div>
<div class="vpane">
  <div id="vR" style="width:100%;height:100%;"></div>
  <div class="info">{right_info}</div>
  <div class="pane-lbl">COMPARISON</div>
  <div class="legend">{right_legend}</div>
</div>
<script>
(function(){{
  var vL=$3Dmol.createViewer("vL",{{backgroundColor:"#f8fafc"}});
{left_js}
  vL.zoomTo(); vL.zoom(0.85); vL.render();
  setInterval(function(){{ vL.rotate(1,'y'); vL.render(); }},50);
  var vR=$3Dmol.createViewer("vR",{{backgroundColor:"#f8fafc"}});
{right_js}
  vR.zoomTo(); vR.zoom(0.85); vR.render();
  setInterval(function(){{ vR.rotate(1,'y'); vR.render(); }},50);
}})();
</script></body></html>"""
        self._log(f"[SBS] html={len(html.encode())} bytes")
        self._set_html(html)

    # ── Disorder / RMSF profile ───────────────────────────────────

    def _draw_disorder_profile(self, rmsf, residues):
        """Plot per-residue RMSF; green = ordered, red = disordered (≥2 Å)."""
        THRESHOLD = 2.0   # Å — standard disorder cutoff
        n_res = min(len(rmsf), len(residues))
        if n_res == 0:
            return

        x = np.arange(1, n_res + 1)
        y = rmsf[:n_res]

        n_dis = int((y > THRESHOLD).sum())
        pct   = 100.0 * n_dis / n_res
        self._rmsf_n_disordered = n_dis
        self._rmsf_pct          = pct
        self.disorder_stats_lbl.setText(
            f"DISORDERED: {n_dis} / {n_res} residues  ({pct:.1f}%)")

        self._disorder_fig.clear()
        ax = self._disorder_fig.add_subplot(111)
        ax.set_facecolor("#f8fafc")
        self._disorder_fig.patch.set_facecolor("#f8fafc")

        ax.fill_between(x, 0, y, where=(y <= THRESHOLD),
                        color="#16a34a", alpha=0.22, label="Ordered (<2 Å)")
        ax.fill_between(x, 0, y, where=(y > THRESHOLD),
                        color="#dc2626", alpha=0.22, label="Disordered (≥2 Å)")
        ax.plot(x, y, color="#1e293b", lw=1.2, zorder=3)
        ax.axhline(THRESHOLD, color="#dc2626", lw=0.9, linestyle="--",
                   alpha=0.65, label="2 Å cutoff")

        if residues:
            step = max(1, n_res // 20)
            tick_pos = list(range(1, n_res + 1, step))
            tick_lbl = [
                str(residues[tp - 1][1]) if tp - 1 < len(residues) else str(tp)
                for tp in tick_pos
            ]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl, fontsize=6, rotation=45, ha="right")

        ax.set_xlim(1, n_res)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Residue", fontsize=8, color="#64748b", labelpad=4)
        ax.set_ylabel("RMSF (Å)", fontsize=8, color="#64748b", labelpad=4)
        ax.set_title(
            f"Residue Flexibility  ·  {n_dis}/{n_res} disordered ({pct:.1f}%)",
            fontsize=9, color="#1e293b", fontweight="bold", pad=8)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#e2e8f0"); sp.set_linewidth(0.8)
        ax.legend(fontsize=7, framealpha=0.88, edgecolor="#e2e8f0",
                  facecolor="#ffffff", labelcolor="#1e293b", loc="upper right")
        self._disorder_canvas.draw()

    def _render_colored_by_rmsf(self):
        """Load the MC best structure into 3Dmol colored by per-residue RMSF (B-factor)."""
        if self._rmsf is None or not self._ensemble:
            return
        best_idx  = int(np.argmin(self._energies))
        particles = self._ensemble[best_idx]
        rmsf      = self._rmsf
        residues  = self._rmsf_residues
        n_ca      = min(len(self._ca_indices), len(rmsf))

        lines = []
        for j in range(n_ca):
            pidx = self._ca_indices[j]
            if pidx >= len(particles):
                continue
            p     = particles[pidx]
            bval  = float(rmsf[j])
            chain = residues[j][0] if j < len(residues) else "A"
            resno = residues[j][1] if j < len(residues) else j + 1
            lines.append(
                f"ATOM  {j+1:5d}  CA  ALA {chain}{resno:4d}    "
                f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00{bval:6.2f}           C")
        pdb_str = "\n".join(lines)
        pdb_esc = pdb_str.replace("\\", "\\\\").replace("`", "\\`")

        html = f"""<!DOCTYPE html><html><head>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ margin:0;padding:0;box-sizing:border-box; }}
  body {{ background:#f8fafc;overflow:hidden; }}
  #v {{ width:100vw;height:100vh; }}
  #info {{
    position:absolute;top:12px;left:16px;font-family:monospace;font-size:11px;
    letter-spacing:1px;color:#1e293b;pointer-events:none;
    background:rgba(255,255,255,0.88);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0;
  }}
  #legend {{
    position:absolute;bottom:12px;left:16px;font-family:monospace;font-size:10px;
    letter-spacing:1px;color:#475569;pointer-events:none;
    background:rgba(255,255,255,0.88);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0;
  }}
</style></head><body>
<div id="v"></div>
<div id="info">FLEXIBILITY MAP &nbsp;&middot;&nbsp; BEST CANDIDATE &nbsp;&middot;&nbsp; {n_ca} C&alpha; atoms</div>
<div id="legend">
  <span style="color:#2563eb">&#9632;</span> RIGID (low RMSF)
  &nbsp;&middot;&middot;&middot;&middot;&middot;&nbsp;
  <span style="color:#dc2626">&#9632;</span> FLEXIBLE / DISORDERED (high RMSF)
</div>
<script>
(function(){{
  var v=$3Dmol.createViewer("v",{{backgroundColor:"#f8fafc"}});
  var m=v.addModel(`{pdb_esc}`,"pdb");
  m.setStyle({{}},{{
    cartoon:{{
      colorscheme:{{prop:"b",gradient:"bwr",min:0,max:4.0}},
      thickness:0.8,opacity:1.0
    }},
    sphere:{{
      colorscheme:{{prop:"b",gradient:"bwr",min:0,max:4.0}},
      radius:0.55,opacity:1.0
    }}
  }});
  v.zoomTo(); v.zoom(0.85); v.render();
  setInterval(function(){{ v.rotate(1,'y'); v.render(); }},50);
}})();
</script></body></html>"""

        self.viewer_cand_lbl.setText(
            '<span style="color:#7c3aed;font-weight:bold;">FLEXIBILITY MAP</span>'
            ' &nbsp;·&nbsp; blue=rigid &nbsp;·&nbsp; red=disordered')
        self._set_html(html)
        self._view_stack.setCurrentIndex(0)
        self.view_mode_btn.setVisible(True)
        self.disorder_toggle_btn.setText("⊛  DISORDER")

    def _toggle_disorder(self):
        if self._view_stack.currentIndex() == 2:
            self._view_stack.setCurrentIndex(0)
            self.view_mode_btn.setVisible(True)
            self.disorder_toggle_btn.setText("⊛  DISORDER")
            self.landscape_toggle_btn.setText("◈  LANDSCAPE")
        else:
            self._view_stack.setCurrentIndex(2)
            self.view_mode_btn.setVisible(False)
            self.disorder_toggle_btn.setText("⊡  STRUCTURE")
            self.landscape_toggle_btn.setText("◈  LANDSCAPE")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        win = ProteinApp()
        win.show()
        sys.exit(app.exec())
    except SystemExit:
        raise
    except Exception:
        with open("error_log.txt", "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        input("Error — press Enter to exit…")
