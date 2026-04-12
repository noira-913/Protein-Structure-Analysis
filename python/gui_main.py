import sys, os, requests, traceback, tempfile
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox, QFrame,
    QProgressBar, QSplitter, QGridLayout,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtCore import QUrl
from Bio.PDB import PDBParser, PDBList
import protein_physics


def _try_gpu_backend():
    """Return (module, gpu_name) if protein_physics_cuda is built and a GPU exists."""
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
QLineEdit:focus {
    border-color: #1d4ed8;
}

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
QPushButton#run-btn:hover {
    background-color: #1e40af;
}
QPushButton#run-btn:disabled {
    background-color: #e2e8f0;
    color: #94a3b8;
}

QPushButton#sec-btn {
    background-color: transparent;
    color: #1d4ed8;
    border: 1.5px solid #1d4ed8;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 10px;
}
QPushButton#sec-btn:hover {
    background-color: #eff6ff;
}

QTextEdit {
    background-color: #f8fafc;
    color: #334155;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 10px;
    font-size: 11px;
    line-height: 1.6;
}
QScrollBar:vertical {
    background: #f1f5f9; width: 8px; border: none;
}
QScrollBar::handle:vertical {
    background: #94a3b8; border-radius: 4px; min-height: 20px;
}
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
_AMBER = {
    "C": (1.908, 0.086), "N": (1.824, 0.170),
    "O": (1.661, 0.210), "S": (2.000, 0.250),
    "H": (0.600, 0.015), "P": (2.100, 0.200),
}
_CHARGE = {"ARG":+1.0,"LYS":+1.0,"HIS":+0.5,"ASP":-1.0,"GLU":-1.0}

def _atom_params(atom):
    res  = atom.get_parent().get_resname().strip()
    name = atom.get_name().strip()
    elem = (atom.element or "").strip().upper()
    elem = elem if len(elem)==1 else name[0].upper()
    charge = _CHARGE.get(res, 0.0) if name == "CA" else 0.0
    r, e = _AMBER.get(elem, (1.9, 0.1))
    return charge, r, e

def _parse_pdb(path, log, physics_mod):
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("prot", path)
    atoms, skipped = [], 0
    for atom in st.get_atoms():
        if atom.get_parent().get_id()[0] != " ": continue
        coord = atom.get_coord()
        if not np.all(np.isfinite(coord)):
            skipped += 1; continue
        charge, r, e = _atom_params(atom)
        atoms.append(physics_mod.Particle(
            float(coord[0]), float(coord[1]), float(coord[2]),
            charge, r, e, False))
    if skipped: log(f"  ⚠  {skipped} atoms skipped (invalid coords)")
    return atoms

# ═══════════════════════════════════════════════════════════════════
#  Worker  (download + parse + physics in QThread)
# ═══════════════════════════════════════════════════════════════════
class PipelineWorker(QThread):
    progress  = pyqtSignal(str)
    metrics   = pyqtSignal(dict)
    finished  = pyqtSignal(list, list)
    error     = pyqtSignal(str)

    def __init__(self, engine, target, physics_mod, n_cand=5, steps=300):
        super().__init__()
        self.engine     = engine
        self.target     = target
        self.physics_mod = physics_mod
        self.n_cand     = n_cand
        self.steps      = steps

    def _fetch(self, target):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(data_dir, exist_ok=True)

        # Check for cached local file in data/ or root
        for cand in [
            os.path.join(data_dir, f"{target}.pdb"),
            os.path.join(data_dir, f"{target.lower()}.pdb"),
            f"{target}.pdb",
            f"{target.lower()}.pdb",
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
                if not raw or not os.path.exists(raw): return None
                if os.path.exists(dest): os.remove(dest)
                os.rename(raw, dest)
                return dest
            except Exception as ex:
                self.progress.emit(f"  RCSB failed: {ex}"); return None
        else:
            # 1. AlphaFold REST API — returns the canonical versioned URL
            self.progress.emit("  Querying AlphaFold API…")
            try:
                api = requests.get(
                    f"https://alphafold.ebi.ac.uk/api/prediction/{target}",
                    timeout=15)
                if api.status_code == 200:
                    entries = api.json()
                    if entries and "pdbUrl" in entries[0]:
                        pdb_url = entries[0]["pdbUrl"]
                        self.progress.emit(f"  Fetching: {pdb_url}")
                        r = requests.get(pdb_url, timeout=30)
                        if r.status_code == 200:
                            with open(dest, "w") as f: f.write(r.text)
                            return dest
            except Exception as ex:
                self.progress.emit(f"  AlphaFold API error: {ex}")

            # 2. Fallback: versioned direct URLs
            self.progress.emit("  Trying versioned AlphaFold URLs…")
            for ver in ("v4", "v3", "v2"):
                url = f"https://alphafold.ebi.ac.uk/files/AF-{target}-F1-model_{ver}.pdb"
                try:
                    r = requests.get(url, timeout=15)
                    if r.status_code == 200:
                        with open(dest, "w") as f: f.write(r.text)
                        self.progress.emit(f"  Found at model_{ver}")
                        return dest
                except Exception:
                    pass
            self.progress.emit("  AlphaFold: no structure found for this ID")
            return None

    def run(self):
        try:
            path = self._fetch(self.target)
            if not path:
                self.error.emit("Structure retrieval failed."); return

            self.progress.emit("  Parsing PDB + AMBER forcefield mapping…")
            atoms = _parse_pdb(path, self.progress.emit, self.physics_mod)
            if not atoms:
                self.error.emit("No valid protein atoms found."); return

            self.metrics.emit({"n_atoms": len(atoms),
                               "threads": self.engine.num_threads()})
            self.progress.emit(
                f"  {len(atoms)} atoms · {self.engine.num_threads()} threads")
            self.progress.emit(
                f"  Running MC: {self.n_cand} candidates × {self.steps} steps…")

            ensemble = self.engine.generate_ensemble(
                atoms, self.n_cand, self.steps, 0.6, 0.3)

            self.progress.emit("  Computing ensemble free energies…")
            energies = [self.engine.calculate_potential(s) for s in ensemble]

            self.metrics.emit({"best_e": min(energies), "n_cand": self.n_cand})
            self.finished.emit(ensemble, energies)
        except Exception as ex:
            self.error.emit(str(ex))

# ═══════════════════════════════════════════════════════════════════
#  Helper widgets
# ═══════════════════════════════════════════════════════════════════
def _panel():
    f = QFrame(); f.setObjectName("panel"); return f

def _heading(text):
    l = QLabel(text.upper()); l.setObjectName("heading"); return l

def _metric_widget(label):
    w = QWidget()
    v = QVBoxLayout(w); v.setContentsMargins(0,0,0,0); v.setSpacing(0)
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

        # Backend selection: GPU if available and user agrees, else CPU
        self._physics_mod = protein_physics
        cuda_mod, gpu_name = _try_gpu_backend()
        if cuda_mod is not None:
            reply = QMessageBox.question(
                self, "GPU Detected",
                f"GPU found: {gpu_name}\n\nUse GPU acceleration for energy calculations?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._physics_mod = cuda_mod
        self._backend = (
            f"GPU  {gpu_name}" if self._physics_mod is not protein_physics
            else "CPU"
        )

        try:
            self.engine = self._physics_mod.PhysicsEngine()
        except Exception as ex:
            QMessageBox.critical(self, "Fatal",
                f"Failed to initialise physics engine ({self._backend}):\n{ex}")
            sys.exit(1)

        self._ensemble = []
        self._energies = []
        self._view_mode = "layered"       # "layered" | "sidebyside"
        self._current_cand_idx = 0
        self._build_ui()
        self.setStyleSheet(STYLE)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # ── Sidebar ───────────────────────────
        sidebar = QVBoxLayout()
        sidebar.setSpacing(8)

        title = QLabel("ALMA")
        title.setStyleSheet("color:#d97706;font-size:22px;font-weight:bold;"
                            "letter-spacing:6px;padding:8px 0 2px 8px;")
        sub = QLabel("Atomistic Local Motion Analyzer")
        sub.setStyleSheet("color:#64748b;font-size:10px;letter-spacing:1px;padding:0 0 2px 8px;")
        backend_lbl = QLabel(f"⚙  {self._backend}")
        backend_lbl.setStyleSheet("color:#94a3b8;font-size:9px;letter-spacing:1px;padding:0 0 8px 8px;")
        sidebar.addWidget(title)
        sidebar.addWidget(sub)
        sidebar.addWidget(backend_lbl)
        sidebar.addWidget(_sep())

        # Input panel
        inp_panel = _panel()
        inp_v = QVBoxLayout(inp_panel)
        inp_v.setContentsMargins(8,4,8,10)
        inp_v.addWidget(_heading("Target"))
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("PDB ID  /  UniProt ID")
        self.id_input.returnPressed.connect(self._start)
        inp_v.addWidget(self.id_input)
        sidebar.addWidget(inp_panel)

        # Buttons
        self.run_btn = QPushButton("▶  RUN ANALYSIS")
        self.run_btn.setObjectName("run-btn")
        self.run_btn.clicked.connect(self._start)
        sidebar.addWidget(self.run_btn)

        self.best_btn = QPushButton("SHOW BEST STRUCTURE")
        self.best_btn.setObjectName("sec-btn")
        self.best_btn.clicked.connect(self._show_best)
        self.best_btn.setEnabled(False)
        sidebar.addWidget(self.best_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0,0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setTextVisible(False)
        sidebar.addWidget(self.progress_bar)

        sidebar.addWidget(_sep())

        # Metrics panel
        met_panel = _panel()
        met_g = QGridLayout(met_panel)
        met_g.setContentsMargins(4,4,4,8)
        met_g.setSpacing(4)

        self._mw_atoms, self._mv_atoms = _metric_widget("ATOMS")
        self._mw_threads, self._mv_threads = _metric_widget("THREADS")
        self._mw_energy, self._mv_energy = _metric_widget("BEST ENERGY")
        self._mw_cand, self._mv_cand = _metric_widget("CANDIDATES")

        met_g.addWidget(self._mw_atoms,   0, 0)
        met_g.addWidget(self._mw_threads, 0, 1)
        met_g.addWidget(self._mw_energy,  1, 0)
        met_g.addWidget(self._mw_cand,    1, 1)
        sidebar.addWidget(met_panel)

        sidebar.addWidget(_sep())

        # Status
        self.status_lbl = QLabel("IDLE")
        self.status_lbl.setObjectName("status-ok")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar.addWidget(self.status_lbl)

        # Log
        log_panel = _panel()
        log_v = QVBoxLayout(log_panel)
        log_v.setContentsMargins(6,4,6,6)
        log_v.addWidget(_heading("Process Log"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        log_v.addWidget(self.log)
        sidebar.addWidget(log_panel)
        sidebar.addStretch()

        # ── Viewer panel ──────────────────────
        viewer_panel = _panel()
        viewer_v = QVBoxLayout(viewer_panel)
        viewer_v.setContentsMargins(0,0,0,0)
        viewer_v.setSpacing(0)

        viewer_header = QWidget()
        viewer_header.setFixedHeight(36)
        vh_layout = QHBoxLayout(viewer_header)
        vh_layout.setContentsMargins(12,0,12,0)
        vh_layout.setSpacing(8)
        viewer_title = QLabel("3D STRUCTURE VIEWER")
        viewer_title.setStyleSheet("color:#64748b;font-size:10px;letter-spacing:2px;")
        vh_layout.addWidget(viewer_title)
        vh_layout.addStretch()
        # Centre label — updated whenever a candidate is selected
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
        self._candidate_btns = []
        viewer_v.addWidget(viewer_header)

        self.web = QWebEngineView()
        self.web.setStyleSheet("border:none;")
        # Allow local temp-file pages to load 3Dmol.js from the CDN
        self.web.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self.web.loadFinished.connect(self._on_load_finished)
        self._html_tmpfile = os.path.join(tempfile.gettempdir(), "alma_viewer.html")
        self._render_empty()
        viewer_v.addWidget(self.web)

        # Candidate energy bar
        self.ebar_widget = QWidget()
        self.ebar_widget.setFixedHeight(40)
        self.ebar_widget.setVisible(False)
        ebar_layout = QHBoxLayout(self.ebar_widget)
        ebar_layout.setContentsMargins(12,4,12,4)
        ebar_layout.setSpacing(6)
        # Permanent legend label — text is filled in by _build_candidate_bar
        self.ebar_legend_lbl = QLabel("")
        self.ebar_legend_lbl.setStyleSheet("font-size:9px;")
        ebar_layout.addWidget(self.ebar_legend_lbl)
        ebar_vsep = QFrame()
        ebar_vsep.setFrameShape(QFrame.Shape.VLine)
        ebar_vsep.setStyleSheet("color:#e2e8f0;")
        ebar_layout.addWidget(ebar_vsep)
        self.ebar_labels = []
        viewer_v.addWidget(self.ebar_widget)

        # Layout
        left_w = QWidget()
        left_w.setFixedWidth(280)
        left_w.setLayout(sidebar)

        outer.addWidget(left_w)
        outer.addWidget(viewer_panel)

    # ── Workflow ──────────────────────────────

    def _start(self):
        target = self.id_input.text().strip().upper()
        if not target: return
        self.run_btn.setEnabled(False)
        self.best_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_lbl.setText("RUNNING")
        self.status_lbl.setStyleSheet("color:#d97706;font-size:11px;font-weight:bold;")
        self.log.clear()
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

    def _on_done(self, ensemble, energies):
        self._ensemble = ensemble
        self._energies = energies
        self.run_btn.setEnabled(True)
        self.best_btn.setEnabled(True)
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

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_lbl.setText("ERROR")
        self.status_lbl.setStyleSheet("color:#dc2626;font-size:11px;font-weight:bold;")
        self._log(f"[ERROR] {msg}")

    def _show_best(self):
        if not self._ensemble: return
        best_idx = int(np.argmin(self._energies))
        self._render(best_idx)

    def _log(self, msg):
        self.log.append(msg)

    def _on_load_finished(self, ok):
        self._log(f"[WEB] loadFinished → {'OK' if ok else 'FAILED'}")

    def _set_html(self, html):
        """Write HTML to a temp file and load via setUrl — avoids setHtml's 2MB data: URL limit."""
        with open(self._html_tmpfile, "w", encoding="utf-8") as f:
            f.write(html)
        self.web.setUrl(QUrl.fromLocalFile(self._html_tmpfile))

    # ── Candidate energy bar ──────────────────

    def _build_candidate_bar(self, energies, best_idx):
        layout = self.ebar_widget.layout()
        for btn in self._candidate_btns:
            layout.removeWidget(btn); btn.deleteLater()
        self._candidate_btns.clear()

        e_min, e_max = min(energies), max(energies)
        e_range = max(abs(e_max - e_min), 1.0)

        # Build the best-colour (green) and worst-colour (red) for the legend
        best_col  = f"#{22:02x}{163:02x}{74:02x}"          # #16a34a  (lowest energy)
        worst_col = f"#{min(22+210,255):02x}{max(163-120,0):02x}{0:02x}"  # #dc2b00 (highest energy)
        self.ebar_legend_lbl.setText(
            f'<span style="color:{best_col}">■</span>'
            f' LOWEST ENERGY (BEST) &nbsp;·····&nbsp; '
            f'<span style="color:{worst_col}">■</span>'
            f' HIGHEST ENERGY (WORST) &nbsp;&nbsp;'
            f'<span style="color:#94a3b8">'
            f'{e_min:.0f} → {e_max:.0f} kcal/mol'
            f'</span>'
        )

        for i, e in enumerate(energies):
            norm = (e - e_min) / e_range   # 0=best, 1=worst
            # Green (best) → Red (worst)
            r = int(22 + 210 * norm)
            g = int(163 - 120 * norm)
            b = int(74 * (1 - norm))
            color = f"#{r:02x}{g:02x}{b:02x}"
            label = f"★ C{i+1}" if i == best_idx else f"C{i+1}"
            btn = QPushButton(label)
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

    # ── 3D rendering ──────────────────────────

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
                f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00  0.50           C"
            )
        return "\n".join(lines)

    def _update_candidate_bar_selection(self, active_idx):
        if not self._candidate_btns or not self._energies:
            return
        best_idx = int(np.argmin(self._energies))
        n = len(self._energies)

        # Update the centre title label in the viewer header
        if active_idx < n:
            e = self._energies[active_idx]
            if active_idx == best_idx:
                tag_html = '&nbsp;&nbsp;<span style="color:#16a34a;font-weight:bold;">★ BEST</span>'
            else:
                tag_html = f'&nbsp;&nbsp;<span style="color:#94a3b8;">(best: C{best_idx+1})</span>'
            self.viewer_cand_lbl.setText(
                f'<b>CANDIDATE {active_idx+1} / {n}</b>'
                f'&nbsp;&nbsp;·&nbsp;&nbsp;{e:.1f} kcal/mol{tag_html}'
            )

        e_min, e_max = min(self._energies), max(self._energies)
        e_range = max(abs(e_max - e_min), 1.0)
        for i, btn in enumerate(self._candidate_btns):
            if i >= len(self._energies):
                break
            norm = (self._energies[i] - e_min) / e_range
            r = int(22 + 210 * norm)
            g = int(163 - 120 * norm)
            bv = int(74 * (1 - norm))
            col = f"#{r:02x}{g:02x}{bv:02x}"
            is_best   = (i == best_idx)
            is_active = (i == active_idx)
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
                  f"ensemble={len(self._ensemble)}  energies={len(self._energies)}")
        self._current_cand_idx = cand_idx
        if self._energies:
            self._update_candidate_bar_selection(cand_idx)
        if not self._ensemble:
            self._log("[RENDER] ensemble empty — aborting")
            return
        try:
            if self._view_mode == "sidebyside":
                self._render_sidebyside(cand_idx)
            else:
                self._render_layered(cand_idx)
        except Exception as ex:
            self._log(f"[RENDER ERROR] {ex}\n{traceback.format_exc()}")

    def _render_layered(self, selected_idx):
        self._log(f"[LAYERED] entry  selected_idx={selected_idx}  "
                  f"ensemble_len={len(self._ensemble)}")
        best_idx = int(np.argmin(self._energies))
        best_particles = self._ensemble[best_idx]
        n_atoms = len(best_particles)
        self._log(f"[LAYERED] best_idx={best_idx}  n_atoms={n_atoms}")

        pdb_best = self._build_pdb_str(best_particles)
        self._log(f"[LAYERED] pdb_best size={len(pdb_best.encode())} bytes")
        best_e = (f"{self._energies[best_idx]:.1f} kcal/mol"
                  if best_idx < len(self._energies) else "")
        sel_e  = (f"{self._energies[selected_idx]:.1f} kcal/mol"
                  if selected_idx < len(self._energies) else "")

        best_model_js = (
            f'  var mBest=v.addModel(`{pdb_best}`,"pdb");\n'
            f'  mBest.setStyle({{}},{{cartoon:{{color:"#1d4ed8",thickness:0.8,opacity:1.0}},sphere:{{color:"#1d4ed8",radius:0.55,opacity:1.0}}}});'
        )

        if selected_idx != best_idx:
            pdb_sel = self._build_pdb_str(self._ensemble[selected_idx])
            self._log(f"[LAYERED] pdb_sel size={len(pdb_sel.encode())} bytes")
            sel_model_js = (
                f'  var mSel=v.addModel(`{pdb_sel}`,"pdb");\n'
                f'  mSel.setStyle({{}},{{cartoon:{{color:"#0891b2",thickness:0.6,opacity:0.65}},sphere:{{color:"#0891b2",radius:0.50,opacity:0.60}}}});'
            )
            label = (f"LAYERED &nbsp; C{selected_idx+1} ({sel_e})"
                     f" &nbsp;over&nbsp; C{best_idx+1} BEST ({best_e})")
            legend_html = (
                '<div id="legend">'
                'OVERLAY &nbsp; '
                '<span style="color:#1d4ed8">&#9632;</span> BEST &nbsp;'
                '<span style="color:#0891b2">&#9632;</span> SELECTED'
                '</div>'
            )
        else:
            sel_model_js = ""
            label = f"&#9733; BEST &nbsp; C{best_idx+1} &nbsp;&middot;&nbsp; {best_e}"
            legend_html = (
                '<div id="legend">'
                '<span style="color:#1d4ed8">&#9632;</span> BEST CANDIDATE'
                '</div>'
            )

        html = f"""<!DOCTYPE html><html><head>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ margin:0;padding:0;box-sizing:border-box; }}
  body {{ background:#f8fafc;overflow:hidden; }}
  #v {{ width:100vw;height:100vh; }}
  #info {{
    position:absolute;top:12px;left:16px;
    font-family:monospace;font-size:11px;letter-spacing:1px;
    color:#1e293b;pointer-events:none;
    background:rgba(255,255,255,0.88);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0;
  }}
  #legend {{
    position:absolute;bottom:12px;left:16px;
    font-family:monospace;font-size:10px;letter-spacing:1px;
    color:#475569;pointer-events:none;
    background:rgba(255,255,255,0.88);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0;
  }}
</style>
</head><body>
<div id="v"></div>
<div id="info">{label} &nbsp;&middot;&nbsp; {n_atoms} ATOMS</div>
{legend_html}
<script>
(function(){{
  var v=$3Dmol.createViewer("v",{{backgroundColor:"#f8fafc"}});
{best_model_js}
{sel_model_js}
  v.zoomTo(); v.zoom(0.85); v.render();
  setInterval(function(){{ v.rotate(1,'y'); v.render(); }},50);
}})();
</script></body></html>"""
        html_bytes = len(html.encode())
        self._log(f"[LAYERED] html size={html_bytes} bytes "
                  f"({'OVER' if html_bytes > 2_000_000 else 'under'} 2MB limit)")
        self._set_html(html)
        self._log("[LAYERED] setUrl called")

    def _render_sidebyside(self, selected_idx):
        self._log(f"[SBS] entry  selected_idx={selected_idx}  "
                  f"ensemble_len={len(self._ensemble)}")
        best_idx = int(np.argmin(self._energies))
        best_particles = self._ensemble[best_idx]
        sel_particles  = self._ensemble[selected_idx]
        n_atoms = len(best_particles)
        self._log(f"[SBS] best_idx={best_idx}  n_atoms={n_atoms}")

        pdb_best = self._build_pdb_str(best_particles)
        self._log(f"[SBS] pdb_best size={len(pdb_best.encode())} bytes")
        best_e = (f"{self._energies[best_idx]:.1f} kcal/mol"
                  if best_idx < len(self._energies) else "")
        sel_e  = (f"{self._energies[selected_idx]:.1f} kcal/mol"
                  if selected_idx < len(self._energies) else "")

        # Left panel: always the best candidate (deep blue)
        left_model_js = (
            f'  var mL=vL.addModel(`{pdb_best}`,"pdb");\n'
            f'  mL.setStyle({{}},{{cartoon:{{color:"#1d4ed8",thickness:0.8,opacity:1.0}},sphere:{{color:"#1d4ed8",radius:0.55,opacity:1.0}}}});'
        )

        # Right panel: split into two separate models — similar atoms (grey)
        # and differing atoms (orange) — using model.setStyle, which is the
        # correct 3Dmol API (vR.setStyle({resi:[...]}, ...) is unreliable).
        if selected_idx != best_idx and len(sel_particles) == len(best_particles):
            devs = [
                ((p.x - r.x)**2 + (p.y - r.y)**2 + (p.z - r.z)**2) ** 0.5
                for p, r in zip(sel_particles, best_particles)
            ]
            max_dev   = max(devs) if max(devs) > 0 else 1.0
            threshold = max_dev * 0.3
            sim_lines, diff_lines = [], []
            for i, p in enumerate(sel_particles):
                line = (f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                        f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00  0.50           C")
                (diff_lines if devs[i] > threshold else sim_lines).append(line)
            pdb_sim  = "\n".join(sim_lines)
            pdb_diff = "\n".join(diff_lines)
            n_diff   = len(diff_lines)
            right_parts = []
            if sim_lines:
                right_parts.append(
                    f'  var mSim=vR.addModel(`{pdb_sim}`,"pdb");\n'
                    f'  mSim.setStyle({{}},{{cartoon:{{color:"#374151",thickness:0.55,opacity:0.8}},sphere:{{color:"#374151",radius:0.45,opacity:0.75}}}});'
                )
            if diff_lines:
                right_parts.append(
                    f'  var mDiff=vR.addModel(`{pdb_diff}`,"pdb");\n'
                    f'  mDiff.setStyle({{}},{{cartoon:{{color:"#f97316",thickness:0.75,opacity:1.0}},sphere:{{color:"#f97316",radius:0.62,opacity:1.0}}}});'
                )
            right_model_js = "\n".join(right_parts)
            right_info   = f"C{selected_idx+1} &nbsp;&middot;&nbsp; {sel_e} &nbsp;&middot;&nbsp; {n_atoms} ATOMS"
            right_legend = (
                f'<span style="color:#374151">&#9632;</span> SIMILAR &nbsp;'
                f'<span style="color:#f97316">&#9632;</span> DIFFERS FROM BEST ({n_diff} atoms)'
            )
        else:
            right_model_js = (
                f'  var mR=vR.addModel(`{pdb_best}`,"pdb");\n'
                f'  mR.setStyle({{}},{{cartoon:{{color:"#1d4ed8",thickness:0.8,opacity:1.0}},sphere:{{color:"#1d4ed8",radius:0.55,opacity:1.0}}}});'
            )
            right_info   = f"&#9733; C{best_idx+1} (BEST) &nbsp;&middot;&nbsp; {best_e} &nbsp;&middot;&nbsp; {n_atoms} ATOMS"
            right_legend = '<span style="color:#1d4ed8">&#9632;</span> BEST CANDIDATE'

        html = f"""<!DOCTYPE html><html><head>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ margin:0;padding:0;box-sizing:border-box; }}
  body {{ background:#f8fafc;overflow:hidden;display:flex;width:100vw;height:100vh; }}
  .vpane {{ flex:1;position:relative;height:100%; }}
  .divider {{ width:2px;background:#e2e8f0;flex-shrink:0; }}
  .info {{
    position:absolute;top:12px;left:12px;
    font-family:monospace;font-size:11px;letter-spacing:1px;
    color:#1e293b;pointer-events:none;
    background:rgba(255,255,255,0.9);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0;z-index:10;
  }}
  .legend {{
    position:absolute;bottom:12px;left:12px;
    font-family:monospace;font-size:10px;letter-spacing:1px;
    color:#475569;pointer-events:none;
    background:rgba(255,255,255,0.9);padding:5px 10px;
    border-radius:5px;border:1px solid #e2e8f0;z-index:10;
  }}
  .pane-lbl {{
    position:absolute;top:12px;right:12px;
    font-family:monospace;font-size:9px;letter-spacing:2px;
    color:#94a3b8;pointer-events:none;z-index:10;
  }}
</style>
</head><body>
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
{left_model_js}
  vL.zoomTo(); vL.zoom(0.85); vL.render();
  setInterval(function(){{ vL.rotate(1,'y'); vL.render(); }},50);

  var vR=$3Dmol.createViewer("vR",{{backgroundColor:"#f8fafc"}});
{right_model_js}
  vR.zoomTo(); vR.zoom(0.85); vR.render();
  setInterval(function(){{ vR.rotate(1,'y'); vR.render(); }},50);
}})();
</script></body></html>"""
        html_bytes = len(html.encode())
        self._log(f"[SBS] html size={html_bytes} bytes "
                  f"({'OVER' if html_bytes > 2_000_000 else 'under'} 2MB limit)")
        self._set_html(html)
        self._log("[SBS] setUrl called")


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
        with open("error_log.txt","w") as f: traceback.print_exc(file=f)
        traceback.print_exc()
        input("Error — press Enter to exit…")
