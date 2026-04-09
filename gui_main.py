import sys
import os
import requests
import numpy as np
import traceback
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox,
)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWebEngineWidgets import QWebEngineView
from Bio.PDB import PDBParser, PDBList
import protein_physics

os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

# --- Parameters ---
_AMBER_PARAMS = {
    "C": (1.908, 0.086), "N": (1.824, 0.170),
    "O": (1.661, 0.210), "S": (2.000, 0.250),
    "H": (0.600, 0.015), "P": (2.100, 0.200),
}
_DEFAULT_PARAMS = (1.9, 0.1)
_RESIDUE_CHARGE = {
    "ARG": +1.0, "LYS": +1.0, "HIS": +0.5,
    "ASP": -1.0, "GLU": -1.0,
}

def _atom_params(atom):
    res_name = atom.get_parent().get_resname().strip()
    atom_name = atom.get_name().strip()
    raw_elem = (atom.element or "").strip().upper()
    element = raw_elem if len(raw_elem) == 1 else atom_name[0].upper()
    charge = _RESIDUE_CHARGE.get(res_name, 0.0) if atom_name == "CA" else 0.0
    radius, epsilon = _AMBER_PARAMS.get(element, _DEFAULT_PARAMS)
    return charge, radius, epsilon

def _parse_pdb(path: str, log_fn) -> list:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("prot", path)
    atoms, skipped = [], 0
    for atom in struct.get_atoms():
        if atom.get_parent().get_id()[0] != " ": continue
        coord = atom.get_coord()
        if not np.all(np.isfinite(coord)):
            skipped += 1
            continue
        charge, radius, epsilon = _atom_params(atom)
        atoms.append(protein_physics.Particle(
            float(coord[0]), float(coord[1]), float(coord[2]),
            charge, radius, epsilon, False))
    if skipped:
        log_fn(f"[*] Skipped {skipped} atoms with invalid coordinates")
    return atoms

class FullPipelineWorker(QThread):
    finished = pyqtSignal(list, list)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, engine, target: str, n_cand=5, steps=200):
        super().__init__()
        self.engine = engine
        self.target = target
        self.n_cand = n_cand
        self.steps = steps

    def _fetch(self, target: str) -> str | None:
        for cand in [f"{target}.pdb", f"{target.lower()}.pdb"]:
            if os.path.exists(cand):
                self.progress.emit(f"[*] Using local file: {cand}")
                return cand
        if len(target) == 4:
            self.progress.emit("[*] Downloading from RCSB PDB...")
            try:
                pdbl = PDBList(verbose=False)
                raw = pdbl.retrieve_pdb_file(target.lower(), pdir=".", file_format="pdb", overwrite=True)
                dest = f"{target}.pdb"
                if os.path.exists(dest): os.remove(dest)
                os.rename(raw, dest)
                return dest
            except: return None
        else:
            self.progress.emit(f"[*] Searching AlphaFold DB: {target}")
            url = f"https://alphafold.ebi.ac.uk/files/AF-{target}-F1-model_v4.pdb"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    dest = f"{target}.pdb"
                    with open(dest, "w", encoding="utf-8") as f: f.write(r.text)
                    return dest
            except: return None
        return None

    def run(self):
        try:
            path = self._fetch(self.target)
            if not path:
                self.error.emit("Could not retrieve structure data.")
                return
            self.progress.emit("[*] Mapping forcefield parameters...")
            atoms = _parse_pdb(path, self.progress.emit)
            self.progress.emit(f"[*] MC Sampling: {self.n_cand} candidates...")
            ensemble = self.engine.generate_ensemble(atoms, self.n_cand, self.steps, 0.6, 0.3)
            self.progress.emit("[*] Calculating free energies...")
            energies = [self.engine.calculate_potential(s) for s in ensemble]
            self.finished.emit(ensemble, energies)
        except Exception as e:
            self.error.emit(str(e))

class AdvancedProteinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro-Topology: Physics & UniProt Explorer")
        self.setGeometry(100, 100, 1400, 850)
        
        # [ORIGINAL CSS - UNTOUCHED]
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { background-color: #121212; color: #E0E0E0; }
            QLineEdit { background-color: #1F1F1F; color: #00FF00; border: 1px solid #333; padding: 8px; }
            QTextEdit { background-color: #0A0A0A; color: #BBBBBB; font-family: 'Consolas'; }
            QPushButton { background-color: #3D5AFE; color: white; border-radius: 4px; padding: 10px; font-weight: bold; }
            QPushButton:disabled { background-color: #555; }
        """)

        try:
            self.engine = protein_physics.PhysicsEngine()
        except:
            QMessageBox.critical(self, "Error", "Failed to load C++ module!")
            sys.exit()

        self.init_ui()

    def init_ui(self):
        # [ORIGINAL LAYOUT - UNTOUCHED]
        layout = QHBoxLayout()
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        side = QVBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("PDB ID (e.g. 1XQ8) or UniProt (e.g. P37840)")
        self.btn = QPushButton("ANALYZE & GENERATE ENSEMBLE")
        self.btn.clicked.connect(self.start_workflow)
        
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        
        side.addWidget(QLabel("PROTEIN IDENTIFIER"))
        side.addWidget(self.input)
        side.addWidget(self.btn)
        side.addWidget(QLabel("PROCESS LOG"))
        side.addWidget(self.log)
        layout.addLayout(side, 1)

        self.web = QWebEngineView()
        layout.addWidget(self.web, 3)

    def start_workflow(self):
        target = self.input.text().strip().upper()
        if not target: return
        
        self.btn.setEnabled(False)
        self.log.append(f"[*] Starting Analysis: {target}")

        self.worker = FullPipelineWorker(self.engine, target)
        self.worker.progress.connect(self.log.append)
        self.worker.finished.connect(self._on_complete)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_complete(self, ensemble, energies):
        self.btn.setEnabled(True)
        best_idx = int(np.argmin(energies))
        self.log.append("[*] Complete:")
        for i, en in enumerate(energies):
            tag = " <- BEST" if i == best_idx else ""
            self.log.append(f"  Candidate {i+1}: {en:.2f} kcal/mol{tag}")
        self._render(ensemble[best_idx])

    def _on_error(self, msg):
        self.btn.setEnabled(True)
        self.log.append(f"[!] Error: {msg}")

    def _render(self, particles):
        lines = []
        for i, p in enumerate(particles):
            lines.append(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00  0.00           C")
        pdb_content = "\\n".join(lines)
        html = f"""
        <html><body style="margin:0; background:#000;">
            <div id="v" style="width:100vw; height:100vh;"></div>
            <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
            <script>
                let v = $3Dmol.createViewer("v", {{backgroundColor:"black"}});
                v.addModel(`{pdb_content}`, "pdb");
                v.setStyle({{}}, {{"cartoon":{{"color":"spectrum"}}}});
                v.zoomTo(); v.render();
            </script>
        </body></html>
        """
        self.web.setHtml(html)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        window = AdvancedProteinApp()
        window.show()
        sys.exit(app.exec())
    except Exception:
        with open("error_log.txt", "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        