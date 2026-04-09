import sys, os, requests, traceback
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox, QFrame,
    QProgressBar, QSplitter, QGridLayout,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWebEngineWidgets import QWebEngineView
from Bio.PDB import PDBParser, PDBList
import protein_physics

os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

# ═══════════════════════════════════════════════════════════════════
#  디자인 토큰 (가시성 및 세련미 개선 버전)
# ═══════════════════════════════════════════════════════════════════
STYLE = """
QMainWindow, QWidget {
    background-color: #1a1d26;  /* 배경을 약간 밝게 하여 가시성 확보 */
    color: #e2e4ed;           /* 기본 글자색을 더 밝은 실버로 변경 */
    font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace;
    font-size: 12px;
}
QFrame#panel {
    background-color: #232733;  /* 패널 배경을 위젯보다 살짝 밝게 하여 입체감 부여 */
    border: 1px solid #32394d;
    border-radius: 6px;
}
QLabel#heading {
    color: #f6ad55;           /* 헤딩 색상을 좀 더 선명한 오렌지 골드로 변경 */
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 2px;
    padding: 6px 8px 2px 8px;
}
QLabel#metric-val {
    color: #ffffff;           /* 수치 데이터는 완전한 흰색으로 강조 */
    font-size: 20px;
    font-weight: bold;
    padding: 0 8px;
}
QLabel#metric-unit {
    color: #a0aec0;           /* 단위는 차분한 회색으로 유지 */
    font-size: 10px;
    padding: 0 8px 4px 8px;
}
QLabel#status-ok  { color: #68d391; font-size: 11px; font-weight: bold; }
QLabel#status-run { color: #f6ad55; font-size: 11px; font-weight: bold; }
QLabel#status-err { color: #fc8181; font-size: 11px; font-weight: bold; }

QLineEdit {
    background-color: #11141d;  /* 입력창은 대비를 위해 더 어둡게 */
    color: #f6ad55;           /* 입력 텍스트는 골드 색상으로 명확히 표기 */
    border: 1px solid #4a5568;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 14px;
    selection-background-color: #2c5282;
}
QLineEdit:focus { 
    border-color: #f6ad55; 
    background-color: #151926;
}

QPushButton#run-btn {
    background-color: #f6ad55;  /* 버튼을 채우기 방식으로 변경하여 가시성 극대화 */
    color: #1a1d26;           /* 글자는 배경색과 동일하게 하여 대비 생성 */
    border: none;
    border-radius: 4px;
    padding: 10px 20px;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: bold;
}
QPushButton#run-btn:hover {
    background-color: #ed8936;
}
QPushButton#run-btn:disabled {
    background-color: #2d3748;
    color: #4a5568;
}

QPushButton#sec-btn {
    background-color: transparent;
    color: #63b3ed;           /* 보조 버튼 글자를 밝은 스카이 블루로 변경 */
    border: 1px solid #4299e1;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 10px;
}
QPushButton#sec-btn:hover { 
    background-color: rgba(66, 153, 225, 0.1); 
}

QTextEdit {
    background-color: #0f1219;  /* 로그 영역 가독성 향상 */
    color: #cbd5e0;           /* 로그 글자를 밝은 회백색으로 변경 */
    border: 1px solid #2d3748;
    border-radius: 4px;
    padding: 10px;
    font-size: 11px;
    line-height: 1.6;
}
QScrollBar:vertical {
    background: #1a1d26; width: 8px; border: none;
}
QScrollBar::handle:vertical {
    background: #4a5568; border-radius: 4px; min-height: 20px;
}
QProgressBar {
    background-color: #11141d;
    border: 1px solid #2d3748;
    border-radius: 2px;
    height: 4px;
}
QProgressBar::chunk { background-color: #f6ad55; }
QSplitter::handle { background: #2d3748; }
"""

# ═══════════════════════════════════════════════════════════════════
#  AMBER 파라미터
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

def _parse_pdb(path, log):
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("prot", path)
    atoms, skipped = [], 0
    for atom in st.get_atoms():
        if atom.get_parent().get_id()[0] != " ": continue
        coord = atom.get_coord()
        if not np.all(np.isfinite(coord)):
            skipped += 1; continue
        charge, r, e = _atom_params(atom)
        atoms.append(protein_physics.Particle(
            float(coord[0]), float(coord[1]), float(coord[2]),
            charge, r, e, False))
    if skipped: log(f"  ⚠  {skipped} atoms skipped (invalid coords)")
    return atoms

# ═══════════════════════════════════════════════════════════════════
#  Worker  (다운로드 + 파싱 + 물리 계산 전부 QThread 안)
# ═══════════════════════════════════════════════════════════════════
class PipelineWorker(QThread):
    progress  = pyqtSignal(str)
    metrics   = pyqtSignal(dict)          # 실시간 메트릭
    finished  = pyqtSignal(list, list)    # ensemble, energies
    error     = pyqtSignal(str)

    def __init__(self, engine, target, n_cand=5, steps=300):
        super().__init__()
        self.engine  = engine
        self.target  = target
        self.n_cand  = n_cand
        self.steps   = steps

    def _fetch(self, target):
        for cand in [f"{target}.pdb", f"{target.lower()}.pdb"]:
            if os.path.exists(cand):
                self.progress.emit(f"  Local file: {cand}")
                return cand
        if len(target) == 4:
            self.progress.emit("  Connecting to RCSB PDB…")
            try:
                pdbl = PDBList(verbose=False)
                raw = pdbl.retrieve_pdb_file(
                    target.lower(), pdir=".", file_format="pdb", overwrite=True)
                if not raw or not os.path.exists(raw): return None
                dest = f"{target}.pdb"
                if os.path.exists(dest): os.remove(dest)
                os.rename(raw, dest)
                return dest
            except Exception as ex:
                self.progress.emit(f"  RCSB failed: {ex}"); return None
        else:
            self.progress.emit(f"  Querying AlphaFold DB…")
            url = f"https://alphafold.ebi.ac.uk/files/AF-{target}-F1-model_v4.pdb"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    dest = f"{target}.pdb"
                    with open(dest, "w") as f: f.write(r.text)
                    return dest
                self.progress.emit(f"  AlphaFold HTTP {r.status_code}")
            except Exception as ex:
                self.progress.emit(f"  AlphaFold failed: {ex}")
            return None

    def run(self):
        try:
            path = self._fetch(self.target)
            if not path:
                self.error.emit("Structure retrieval failed."); return

            self.progress.emit("  Parsing PDB + AMBER forcefield mapping…")
            atoms = _parse_pdb(path, self.progress.emit)
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
#  헬퍼 위젯
# ═══════════════════════════════════════════════════════════════════
def _panel():
    f = QFrame(); f.setObjectName("panel"); return f

def _heading(text):
    l = QLabel(text.upper()); l.setObjectName("heading"); return l

def _metric_widget(label):
    """레이블 + 값 + 단위 묶음"""
    w = QWidget()
    v = QVBoxLayout(w); v.setContentsMargins(0,0,0,0); v.setSpacing(0)
    v.addWidget(_heading(label))
    val = QLabel("—"); val.setObjectName("metric-val")
    v.addWidget(val)
    return w, val

def _sep():
    l = QFrame(); l.setFrameShape(QFrame.Shape.HLine)
    l.setStyleSheet("color: #1e2535;"); return l

# ═══════════════════════════════════════════════════════════════════
#  메인 GUI
# ═══════════════════════════════════════════════════════════════════
class ProteinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALMA — Protein Structure Analysis")
        self.setMinimumSize(1300, 800)

        try:
            self.engine = protein_physics.PhysicsEngine()
        except Exception:
            QMessageBox.critical(self, "Fatal",
                "Failed to load protein_physics module.\nRebuild with: pip install .")
            sys.exit(1)

        self._ensemble = []
        self._energies = []
        self._build_ui()
        self.setStyleSheet(STYLE)

    # ── UI 빌드 ─────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # ── 왼쪽 사이드바 ─────────────────────
        sidebar = QVBoxLayout()
        sidebar.setSpacing(8)

        # 헤더
        title = QLabel("ALMA")
        title.setStyleSheet("color:#e8b84b;font-size:22px;font-weight:bold;"
                            "letter-spacing:6px;padding:8px 0 2px 8px;")
        sub = QLabel("Atomistic Local Motion Analyzer")
        sub.setStyleSheet("color:#2a3349;font-size:10px;letter-spacing:1px;padding:0 0 8px 8px;")
        sidebar.addWidget(title)
        sidebar.addWidget(sub)
        sidebar.addWidget(_sep())

        # 입력 패널
        inp_panel = _panel()
        inp_v = QVBoxLayout(inp_panel)
        inp_v.setContentsMargins(8,4,8,10)
        inp_v.addWidget(_heading("Target"))
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("PDB ID  /  UniProt ID")
        self.id_input.returnPressed.connect(self._start)
        inp_v.addWidget(self.id_input)
        sidebar.addWidget(inp_panel)

        # 실행 버튼
        self.run_btn = QPushButton("▶  RUN ANALYSIS")
        self.run_btn.setObjectName("run-btn")
        self.run_btn.clicked.connect(self._start)
        sidebar.addWidget(self.run_btn)

        self.best_btn = QPushButton("SHOW BEST STRUCTURE")
        self.best_btn.setObjectName("sec-btn")
        self.best_btn.clicked.connect(self._show_best)
        self.best_btn.setEnabled(False)
        sidebar.addWidget(self.best_btn)

        # 진행 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0,0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setTextVisible(False)
        sidebar.addWidget(self.progress_bar)

        sidebar.addWidget(_sep())

        # 메트릭 패널
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

        # 상태
        self.status_lbl = QLabel("IDLE")
        self.status_lbl.setObjectName("status-ok")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar.addWidget(self.status_lbl)

        # 로그
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

        # ── 오른쪽 뷰어 ───────────────────────
        viewer_panel = _panel()
        viewer_v = QVBoxLayout(viewer_panel)
        viewer_v.setContentsMargins(0,0,0,0)
        viewer_v.setSpacing(0)

        viewer_header = QWidget()
        viewer_header.setFixedHeight(32)
        vh_layout = QHBoxLayout(viewer_header)
        vh_layout.setContentsMargins(12,0,12,0)
        viewer_title = QLabel("3D STRUCTURE VIEWER")
        viewer_title.setStyleSheet("color:#2a3349;font-size:10px;letter-spacing:2px;")
        vh_layout.addWidget(viewer_title)
        vh_layout.addStretch()
        self._candidate_btns = []
        viewer_v.addWidget(viewer_header)

        self.web = QWebEngineView()
        self.web.setStyleSheet("border:none;background:#080a0e;")
        self._render_empty()
        viewer_v.addWidget(self.web)

        # ── 에너지 바 ─────────────────────────
        self.ebar_widget = QWidget()
        self.ebar_widget.setFixedHeight(40)
        self.ebar_widget.setVisible(False)
        ebar_layout = QHBoxLayout(self.ebar_widget)
        ebar_layout.setContentsMargins(12,4,12,4)
        self.ebar_labels = []
        viewer_v.addWidget(self.ebar_widget)

        # 레이아웃 조합
        left_w = QWidget()
        left_w.setFixedWidth(280)
        left_w.setLayout(sidebar)

        outer.addWidget(left_w)
        outer.addWidget(viewer_panel)

    # ── 워크플로 ─────────────────────────────

    def _start(self):
        target = self.id_input.text().strip().upper()
        if not target: return
        self.run_btn.setEnabled(False)
        self.best_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_lbl.setText("RUNNING")
        self.status_lbl.setObjectName("status-run")
        self.status_lbl.setStyleSheet("color:#e8b84b;font-size:11px;")
        self.log.clear()
        self._log(f"[{target}] Analysis initiated")

        self.worker = PipelineWorker(self.engine, target)
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
        self.status_lbl.setStyleSheet("color:#48bb78;font-size:11px;")

        best_idx = int(np.argmin(energies))
        self._log("─" * 36)
        for i, e in enumerate(energies):
            tag = " ◀ BEST" if i == best_idx else ""
            self._log(f"  Candidate {i+1:02d}  {e:>12.2f} kcal/mol{tag}")
        self._log("─" * 36)

        self._build_candidate_bar(energies, best_idx)
        self._render(ensemble[best_idx], best_idx)

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_lbl.setText("ERROR")
        self.status_lbl.setStyleSheet("color:#fc8181;font-size:11px;")
        self._log(f"[ERROR] {msg}")

    def _show_best(self):
        if not self._ensemble: return
        best_idx = int(np.argmin(self._energies))
        self._render(self._ensemble[best_idx], best_idx)

    def _log(self, msg):
        self.log.append(msg)

    # ── 에너지 후보 바 ────────────────────────

    def _build_candidate_bar(self, energies, best_idx):
        layout = self.ebar_widget.layout()
        # 기존 위젯 제거
        for btn in self._candidate_btns:
            layout.removeWidget(btn); btn.deleteLater()
        self._candidate_btns.clear()

        e_min, e_max = min(energies), max(energies)
        e_range = max(abs(e_max - e_min), 1.0)

        for i, e in enumerate(energies):
            norm = (e - e_min) / e_range   # 0=best, 1=worst
            r = int(72 + 183 * norm)
            g = int(184 - 100 * norm)
            b = int(75 + 50 * norm)
            color = f"#{r:02x}{g:02x}{b:02x}"
            btn = QPushButton(f"C{i+1}")
            if i == best_idx:
                btn.setStyleSheet(
                    f"background:{color};color:#000;border:none;border-radius:2px;"
                    f"font-size:10px;font-weight:bold;padding:4px 8px;")
            else:
                btn.setStyleSheet(
                    f"background:transparent;color:{color};"
                    f"border:1px solid {color};border-radius:2px;"
                    f"font-size:10px;padding:4px 8px;")
            idx = i
            btn.clicked.connect(lambda _, ii=idx: self._render(self._ensemble[ii], ii))
            layout.addWidget(btn)
            self._candidate_btns.append(btn)
        layout.addStretch()
        self.ebar_widget.setVisible(True)

    # ── 3D 렌더링 ────────────────────────────

    def _render_empty(self):
        self.web.setHtml("""<!DOCTYPE html><html>
<body style="margin:0;background:#080a0e;display:flex;align-items:center;
             justify-content:center;height:100vh;">
  <div style="text-align:center;font-family:monospace;">
    <div style="color:#1e2535;font-size:48px;letter-spacing:8px;">◈</div>
    <div style="color:#1e2535;font-size:11px;letter-spacing:3px;margin-top:16px;">
      AWAITING STRUCTURE</div>
  </div>
</body></html>""")

    def _render(self, particles, cand_idx=0):
        lines = []
        for i, p in enumerate(particles):
            lines.append(
                f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00  0.00           C")
        pdb = "\n".join(lines)
        html = f"""<!DOCTYPE html><html><head>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ margin:0;padding:0;box-sizing:border-box; }}
  body {{ background:#080a0e;overflow:hidden; }}
  #v {{ width:100vw;height:100vh; }}
  #info {{
    position:absolute;bottom:16px;left:16px;
    font-family:monospace;font-size:11px;
    color:#2a3349;letter-spacing:1px;pointer-events:none;
  }}
</style>
</head><body>
<div id="v"></div>
<div id="info">CANDIDATE {cand_idx+1:02d} &nbsp;·&nbsp; {len(particles)} ATOMS</div>
<script>
(function(){{
  const v=$3Dmol.createViewer("v",{{backgroundColor:"#080a0e"}});
  v.addModel(`{pdb}`,"pdb");
  v.setStyle({{}},{{"cartoon":{{
    "color":"spectrum",
    "thickness":0.4,
    "opacity":0.92
  }}}});
  v.addStyle({{}},{{"stick":{{
    "radius":0.08,"opacity":0.35,"colorscheme":"Carbon"
  }}}});
  v.zoomTo();
  v.zoom(0.85);
  v.render();
}})();
</script></body></html>"""
        self.web.setHtml(html)


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