"""
gui_main.py — ALMA: Atomistic Local Motion Analyzer
=====================================================
PyQt6 desktop application for protein structure retrieval, implicit-solvent
Monte Carlo simulation, and conformational analysis.
단백질 구조 검색, 암묵적 용매 몬테 카를로 시뮬레이션, 구조 분석을 위한
PyQt6 데스크톱 애플리케이션.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 전체 흐름 (High-level Workflow)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. [입력] 사용자가 4글자 PDB ID(결정학 구조) 또는 UniProt 접근자를 입력.
     User enters a 4-char PDB ID (crystallographic) or UniProt accession.

  2. [PipelineWorker] 백그라운드 QThread가 PDB 파일을 가져오고
     AMBER ff99SB 파라미터를 원자에 매핑한 뒤, C++ physics 엔진으로
     Metropolis MC 앙상블을 생성하고 최종 에너지를 계산한다.
     Background QThread fetches PDB (RCSB or AlphaFold), maps atoms to
     AMBER ff parameters, then calls generate_ensemble() via the C++
     protein_physics extension and evaluates final energies.

  3. [ComparisonWorker] 동시에 AlphaFold/SWISS-MODEL 구조를 가져와
     Kabsch 중첩 Cα RMSD를 참조 구조와 비교한다.
     Concurrently fetches AlphaFold / SWISS-MODEL structures and computes
     Kabsch-superimposed Cα RMSD vs the reference.
     • Kabsch 알고리즘: SVD로 최적 회전 행렬 R을 구해 두 구조를 정렬.
       이론: H = P^T Q → SVD → R = V diag(1,1,det(VUᵀ)) Uᵀ

  4. [LandscapeWorker] 더 긴 MC 마르코프 체인을 실행해 배열 공간을 탐색.
     PCA로 고차원 Cα 좌표를 2D로 투영한 뒤, 그 구조 공간에서 밀도 기반
     군집화(DBSCAN)로 준안정 분지(metastable basin)를 탐지해 단백질을
     ordered / possibly-disordered / IDP 로 분류한다.
     Runs a longer MC Markov chain, projects conformations to 2D via PCA,
     then detects metastable basins via density-based clustering (DBSCAN)
     on that structural space, and classifies the protein as ordered / IDP.
     • PCA: 고차원 구조 공간의 주요 집단 운동 방향(PC1, PC2)을 찾아
       자유 에너지 지형을 2D로 시각화.
     • IDP(Intrinsically Disordered Protein) 분류: 준안정 분지가 많고
       에너지 차이가 작으면(에너지 지형이 평탄하면) 무질서 단백질로 분류.

  5. [RMSF] 궤적의 각 Cα에 대해 RMSF(제곱평균제곱근 요동)를 계산해
     잔기별 유연성(flexibility) 프로파일을 시각화한다.
     Per-Cα RMSF from landscape trajectory shows residue flexibility.
     RMSF ≥ 2 Å 인 잔기는 무질서(disordered)로 표시.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 렌더링 (3D Rendering)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3D structures are rendered in a QWebEngineView using the 3Dmol.js library
served from a local temp HTML file (avoids CORS issues with file:// URLs).
3D 구조는 로컬 임시 HTML 파일을 통해 3Dmol.js 라이브러리를 사용해 렌더링.
(file:// URL의 CORS 문제를 피하기 위해 임시 파일 경로를 사용)

GPU acceleration: if protein_physics_cuda is importable at startup, the user
is offered a choice; both modules expose the same Particle / PhysicsEngine API.
GPU 가속: protein_physics_cuda 가 임포트 가능하면 시작 시 GPU 선택 옵션 제공.
두 모듈(CPU/GPU)은 동일한 Particle / PhysicsEngine API를 노출한다.
"""

import sys, os, io, requests, traceback, tempfile
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox, QFrame,
    QProgressBar, QSplitter, QGridLayout, QScrollArea, QStackedWidget,
    QComboBox,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QRunnable, QThreadPool
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtCore import QUrl
from Bio.PDB import PDBParser, PDBList, PDBIO, Select
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import protein_physics
from amber_params import get_atom_params as _amber_get_params, ION_PARAMS, _WATER_RESNAMES, _NUCLEOTIDE_RESNAMES
from amber_params import missing_hydrogen_charge as _amber_missing_h_charge
import iupred as _iupred
import knot_analysis as _knot

# Optional C++/OpenMP-accelerated backend for the two O(n_ca^2)-per-snapshot
# ensemble metrics (_compute_internal_scaling's mean-distance matrix,
# _compute_contact_map's contact-frequency matrix) -- see
# IMPROVEMENTS.md item #7. Falls back to the pure-Python loops in this file
# when the extension has not been built (same fallback convention as
# protein_physics_cuda's optional GPU backend).
try:
    import protein_analysis as _analysis_ext
except ImportError:
    _analysis_ext = None


def _app_base_dir():
    """Root directory for locating the sibling ``data`` folder.

    Under a PyInstaller-frozen build ``__file__`` points inside the
    temporary/onefile extraction bundle, so the persistent cache dir must
    instead sit next to the executable to survive across runs.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _vendor_asset_path(name):
    """Absolute filesystem path to a locally-vendored third-party web asset.

    Added 2026-07-13 (IMPROVEMENTS.md item #7 re-test) to replace the CDN
    <script src="https://3Dmol.org/..."> tag every 3D view used to embed:
    that external fetch intermittently failed inside this app's embedded
    QWebEngineView ($3Dmol is not defined -- root-caused as specific to
    constructing the full ProteinApp window, not a code bug in any one
    view), and vendoring removes the external-network dependency entirely.
    Source lives at python/vendor/ in dev; PyInstaller (alma.spec) bundles
    the same folder as vendor/ under sys._MEIPASS in a frozen build.
    """
    if getattr(sys, "frozen", False):
        base = os.path.join(sys._MEIPASS, "vendor")
    else:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
    return os.path.join(base, name)


# file:// URL for the vendored 3Dmol.js build, embedded via <script src="...">
# in every 3D-view HTML page generated below (see _vendor_asset_path above).
_3DMOL_JS_URL = QUrl.fromLocalFile(_vendor_asset_path("3Dmol-min.js")).toString()


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
#  AMBER ff14SB parameters  (see python/amber_params.py for full tables)
#  AMBER ff14SB 힘장 매개변수 — 전체 표는 amber_params.py 참고
# ═══════════════════════════════════════════════════════════════════
#
# 이전 방식: 6개 원소 기호 → (반경, ε);  전하는 Cα에만 형식 전하 할당.
# 새 방식:  (잔기명, 원자명) → AMBER 원자 유형 → 정확한 VDW 매개변수 +
#           RESP 부분 전하 (20개 표준 아미노산 모든 중원자).
#
# Previous: 6 element symbols → (radius, ε); charge only on Cα of 5 residues.
# New:      (resname, atomname) → AMBER ff14SB atom type → correct VDW params
#           + RESP partial charges for all heavy atoms of all 20 amino acids.

def _atom_params(atom):
    """Return (charge, radius, epsilon) for a BioPython Atom object.

    Delegates to amber_params.get_atom_params() which uses the full
    AMBER ff14SB (resname, atomname) → type → (radius, ε) + RESP charges.
    Falls back to element-based estimates for unknown residues/atoms.

    AMBER ff14SB amber_params.get_atom_params()에 위임.
    알 수 없는 잔기/원자의 경우 원소 기반 추정값으로 폴백.
    """
    res  = atom.get_parent().get_resname().strip()
    name = atom.get_name().strip()
    return _amber_get_params(res, name)

def _parse_pdb(path, log, physics_mod):
    """PDB 파일을 파싱해 물리 엔진용 파티클 목록과 결합 위상 그래프를 구성한다.
    Parse a PDB file into a Particle list + BondTopology for the physics engine.

    ── 반환값 (Returns) ──────────────────────────────────────────────────────────
      atoms         — 모든 유효 중원자(금속 이온 포함)에 대한 Particle 목록.
                      Particle list for all valid heavy atoms including metal ions.
      ca_indices    — atoms 목록 내 각 Cα 원자의 인덱스 (잔기 등장 순서).
                      Index into atoms for each Cα atom, in residue encounter order.
      ca_map        — {(chain_id, res_seq): 좌표 배열} — Kabsch RMSD 계산에 사용.
                      dict {(chain_id, res_seq): coord} for Kabsch RMSD computation.
      topo          — 이황화 구속이 등록된 BondTopology 인스턴스.
                      BondTopology with disulfide restraints registered.
      iupred_scores — IUPred 유사 예측기에서 계산한 잔기별 무질서 확률 [0, 1].
                      Per-Cα disorder probability from the sequence-based predictor.
      ca_residues   — [(chain_id, res_seq, res_name3)] — 무질서 프로파일 x축 레이블용.
                      Residue info per Cα for disorder profile x-axis labels.
      heavy_map     — {(chain_id, res_seq, atom_name): 좌표 배열} — 전체 중원자 RMSD용.
                      dict {(chain_id, res_seq, atom_name): coord} for all-heavy-atom RMSD
                      (P10 개선: 사이드체인 형태를 반영하는 RMSD, ca_map과 동일 원리이나
                      원자명까지 키에 포함해 모든 중원자를 구분한다).
                      Same idea as ca_map but keyed down to atom_name so every heavy atom
                      (not just Cα) participates in the Kabsch superposition — see the
                      "전체 중원자 RMSD" comment block below for the full rationale.
      heavy_indices — atoms 목록 내 각 중원자의 인덱스, heavy_map과 동일한 순서.
                      Index into atoms for each heavy atom, in the same order as heavy_map.
      heavy_keys    — heavy_indices와 짝이 되는 (chain_id, res_seq, atom_name) 키 목록.
                      Parallel (chain_id, res_seq, atom_name) key list matching heavy_indices
                      (MC 앙상블 후보는 좌표 배열뿐이므로, 이 인덱스로 다시 키를 붙여준다).
                      MC ensemble candidates are bare coordinate arrays with no atom
                      metadata, so this lets us re-attach (chain, res_seq, atom_name)
                      keys the same way ca_indices does for Cα-only RMSD.

    ── HETATM 처리 방침 (P2.2) ───────────────────────────────────────────────────
      BioPython에서 res.get_id()[0]의 값:
        ' '     → 표준 ATOM 레코드 (표준 아미노산)
        'H_xxx' → HETATM 레코드 (리간드, 이온 등)
        'W'     → 물 분자 (WAT, HOH)

      이전 방침: 모든 HETATM 건너뜀 → 금속 활성 부위 단백질에서 이온이 소실됨.

      현재 방침 (P2.2):
        • 물 분자 (HOH, WAT, …): 항상 건너뜀.
          이유: implicit-solvent(GB/SASA) 모델을 사용하므로 명시적 물이 불필요.
               포함 시 O(N²) 비결합 합산이 크게 느려지고 Born 반경이 왜곡됨.
        • 금속 이온 (MG, ZN, FE 등): 포함. ION_PARAMS 표에서 직접 파라미터 조회.
        • 기타 HETATM (유기 리간드): 포함. 원소 기호 기반 폴백 파라미터 사용.
          결합 위상(topo)은 알 수 없는 잔기를 조용히 건너뛰므로 안전.

      BioPython residue id interpretation:
        ' '     → standard ATOM record (standard amino acid)
        'H_xxx' → HETATM record (ligand, ion, etc.)
        'W'     → water

      New policy (P2.2):
        • Water: always dropped (implicit-solvent model; explicit water would inflate
          the O(N²) pair sum and distort Born radii).
        • Metal ions: included using pre-tabulated ION_PARAMS.
        • Other HETATM: included using element-symbol fallback params.

    ── 건너뜀 규칙 요약 (Skip rules) ────────────────────────────────────────────
      • 물 분자 (HOH, WAT, TIP3, SOL 등)
      • NaN/Inf 좌표 (결정학 미해상 루프)
      • Water residues (any name in _WATER_RESNAMES)
      • Invalid (NaN/Inf) coordinates (unresolved crystallographic loops)
    """
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("prot", path)
    atoms, skipped = [], 0
    ca_indices, ca_map = [], {}

    # ── 전체 중원자 맵 준비 (all-heavy-atom map, P10) ───────────────────────────
    # heavy_map: (chain_id, res_seq, atom_name) → 좌표. heavy_indices/heavy_keys:
    # atoms 목록 내 인덱스와 그에 대응하는 키를 병렬 배열로 저장 (ca_indices와 동일 패턴).
    # heavy_map: (chain_id, res_seq, atom_name) → coord. heavy_indices/heavy_keys are
    # parallel arrays (index into atoms + matching key), mirroring the ca_indices pattern.
    heavy_indices: list[int] = []
    heavy_keys:    list[tuple] = []
    heavy_map:     dict = {}

    # ── Cα 잔기 정보 수집 (Cα residue info collection) ──────────────────────────
    # ca_residues: 무질서 프로파일 x축 레이블용 (chain_id, res_seq, res_name3) 3-튜플.
    # ca_resnames: IUPred 서열 입력용 3글자 잔기명 목록 (Cα 순서).
    # ca_residues: (chain_id, res_seq, res_name3) for each Cα — disorder plot x-axis.
    # ca_resnames: 3-letter resnames in Cα order — sequence input for IUPred.
    ca_residues: list[tuple] = []
    ca_resnames: list[str]   = []

    # ── SG 원자 추적 (CYS SG atom tracking for disulfide detection, P2.3) ───────
    # 각 시스테인 SG 원자에 대해 (파티클 인덱스, x, y, z)를 저장한다.
    # 파싱 완료 후 SG–SG 거리를 전체 검사해 이황화 결합을 탐지한다.
    # Store (particle_idx, x, y, z) for each CYS SG atom.
    # After parsing, check all SG–SG pairs for distances < 2.5 Å.
    sg_atoms: list[tuple] = []
    n_ions = 0   # 포함된 금속 이온 수 (로그용) / included metal ion count for logging

    # ── 위상 구성용 메타데이터 배열 (BondTopology metadata arrays) ──────────────
    # BondTopology.build()에 전달하는 세 개의 평행 배열.
    # 모두 atoms 목록과 동일한 순서·길이를 유지해야 한다.
    # Three parallel arrays passed to BondTopology.build().
    # Must stay in the same order and length as the atoms list.
    meta_resnames:  list[str] = []   # 잔기명 (예: "ALA") / residue name
    meta_atomnames: list[str] = []   # 원자명 (예: "CA") / atom name
    meta_residx:    list[int] = []   # 잔기별 고유 정수 인덱스 / unique residue integer index

    # ── 잔기별 고유 정수 인덱스 할당 맵 (Residue → sequential integer index) ────
    # 키: (chain_id, res_seq, icode) — 삽입코드까지 포함해 항체 번호매김(100A, 100B)처리.
    # 값: 0-기반 정수. C++ build()에서 연속값(r, r+1)이 펩타이드 결합 탐지에 사용됨.
    # HETATM 잔기는 10만 이상의 고립 인덱스를 부여해 인접성 오해를 방지.
    # Key: (chain_id, res_seq, icode) — insertion code handles antibody numbering (100A,100B).
    # Value: 0-based integer; consecutive (r, r+1) triggers peptide bond in C++ build().
    # HETATM residues get isolated large indices (≥100 000) — no adjacency to ATOM residues.
    residue_id_map: dict[tuple, int] = {}

    # 첫 번째 MODEL만 사용 — NMR 앙상블(다중 모델) 파일을 st.get_atoms()로 순회하면
    # 모든 모델의 원자가 합쳐져 구조가 수십 배로 중복된다 (_ca_map_from_pdb 등 다른
    # 파서 헬퍼들과 동일한 규칙).
    # First MODEL only — st.get_atoms() would otherwise walk every model in an NMR
    # ensemble file, duplicating the whole structure ~N-fold (same rule already
    # applied in _ca_map_from_pdb/_heavy_map_from_pdb/_ca_residues_from_pdb).
    for atom in st[0].get_atoms():
        res      = atom.get_parent()
        # BioPython 잔기 플래그: ' '=표준AA, 'H_xxx'=HETATM, 'W'=물
        # BioPython residue flag: ' '=standard AA, 'H_xxx'=HETATM, 'W'=water
        het_flag = res.get_id()[0]
        resname  = res.get_resname().strip()
        atomname = atom.get_name().strip()

        # ── 물 분자 제거 ──────────────────────────────────────────────────────
        # implicit-solvent 모델에서 명시적 물은 불필요하고 계산 비용만 증가시킴.
        # Drop water molecules — counterproductive with our implicit-solvent model.
        if het_flag != " " and resname in _WATER_RESNAMES:
            continue

        # ── 핵산 잔기 제거 ────────────────────────────────────────────────────
        # DNA/RNA는 표준 ATOM 레코드로 기록되는 경우가 많아 het_flag만으로는
        # 걸러지지 않는다. amber_params._NUCLEOTIDE_RESNAMES의 주석 참고 —
        # 실제로 SWISS-MODEL이 단백질-DNA 복합체를 템플릿으로 골라 결합된 DNA를
        # 결과 파일에 남겨둔 사례에서 발견됨 (calculate_potential()이
        # 수천만 kcal/mol로 폭주).
        # Drop nucleic acid residues — often standard ATOM records, not HETATM,
        # so the water check above doesn't catch them. See the comment on
        # amber_params._NUCLEOTIDE_RESNAMES — found via a real case where
        # SWISS-MODEL picked a protein-DNA co-crystal as its best template and
        # left the bound DNA in the output file (calculate_potential() blew up
        # to tens of millions of kcal/mol).
        if resname in _NUCLEOTIDE_RESNAMES:
            continue

        # ── 좌표 유효성 검사 ──────────────────────────────────────────────────
        # NaN/Inf 좌표는 결정학적으로 해상도가 부족한 루프 영역에서 발생함.
        # Invalid coords occur in crystallographically unresolved loop regions.
        coord = atom.get_coord()
        if not np.all(np.isfinite(coord)):
            skipped += 1
            continue

        # ── Cα 원자 추적 (표준 ATOM 레코드만) ──────────────────────────────────
        # HETATM의 경우 atomname이 "CA"여도 Cα 탄소가 아니므로 표준 ATOM만 처리.
        # For standard ATOM records only: Cα atoms define backbone for RMSD/RMSF/IUPred.
        if het_flag == " " and atomname == "CA":
            ca_indices.append(len(atoms))
            ca_key = (res.get_parent().get_id(), res.get_id()[1])
            ca_map[ca_key] = coord.copy()
            ca_residues.append((res.get_parent().get_id(), res.get_id()[1], resname))
            ca_resnames.append(resname)

        # ── 전체 중원자 RMSD (all-heavy-atom RMSD, IMPROVEMENTS #10) ─────────────
        # Cα RMSD만 비교하면 백본이 겹쳐도 사이드체인 회전이체(rotamer)가 완전히
        # 달라진 구조를 "동일하다"고 오판할 수 있다. 예를 들어 활성 부위 잔기의
        # 사이드체인이 반대 방향을 향해도 Cα 위치는 거의 변하지 않는다. 모든
        # 중원자(백본 + 사이드체인)를 Kabsch 중첩에 포함시키면 이런 로컬 패킹
        # 차이가 RMSD 값에 직접 반영되어, 참조 구조와의 비교가 더 엄격해진다.
        #
        # Cα-only RMSD can call two structures "identical" even when a sidechain
        # rotamer flips entirely — e.g. an active-site sidechain pointing the
        # opposite way barely moves the Cα.  Including every heavy atom (backbone
        # + sidechain) in the Kabsch superposition makes local packing differences
        # show up directly in the RMSD, giving a stricter structural comparison.
        #
        # 수소 제외 이유: AlphaFold/SWISS-MODEL 등 비교 대상 참조 구조들은 대개
        # 수소를 명시적으로 포함하지 않으므로, 우리 쪽에서도 수소를 빼야 두 맵의
        # 키가 실제로 겹친다 (수소를 넣으면 공통 키가 거의 0개가 되어 RMSD가
        # 계산 불가능해진다).
        # Hydrogens excluded: reference structures (AlphaFold, SWISS-MODEL) rarely
        # include explicit hydrogens, so we drop them here too — otherwise the two
        # key sets would barely overlap and _compute_rmsd would have no common atoms.
        #
        # HETATM 제외 이유: 리간드/이온은 서로 다른 구조 소스 사이에서 1:1로
        # 대응한다는 보장이 없다 (아예 없을 수도, 다른 리간드가 결합했을 수도
        # 있음). 표준 ATOM 레코드(폴리펩티드 골격+사이드체인)만 포함해야 비교가
        # 의미를 가진다.
        # HETATM excluded: ligands/ions aren't guaranteed to correspond 1:1 across
        # different structure sources (may be absent, or a different ligand may be
        # bound) — only standard ATOM records (backbone + sidechain) make a
        # meaningful heavy-atom comparison.
        if het_flag == " " and not atomname.startswith("H"):
            heavy_key = (res.get_parent().get_id(), res.get_id()[1], atomname)
            heavy_indices.append(len(atoms))
            heavy_keys.append(heavy_key)
            heavy_map[heavy_key] = coord.copy()

        # ── CYS SG 원자 추적 (이황화 탐지용, P2.3) ───────────────────────────
        # 파티클 목록에 추가되기 직전의 인덱스 len(atoms)를 함께 저장한다.
        # Store particle index BEFORE appending (= current len(atoms)) + coordinates.
        if het_flag == " " and resname == "CYS" and atomname == "SG":
            sg_atoms.append((len(atoms), float(coord[0]), float(coord[1]), float(coord[2])))

        # 금속 이온 카운트 (로그 표시용)
        # Count ion atoms for the log message.
        if het_flag != " " and resname in ION_PARAMS:
            n_ions += 1

        # ── 잔기 고유 인덱스 할당 ────────────────────────────────────────────
        # 표준 AA: 등장 순서대로 연속 정수 부여 → C++에서 펩타이드 결합 자동 탐지.
        # HETATM: 10만 + 카운터로 고립 인덱스 부여 → 표준 AA와 인접하지 않도록.
        # Standard AA: sequential integer in encounter order → C++ detects peptide bonds.
        # HETATM:  isolated large index (≥100 000) → no adjacency to standard AA.
        res_key = (res.get_parent().get_id(), res.get_id()[1], res.get_id()[2])
        if res_key not in residue_id_map:
            if het_flag == " ":
                residue_id_map[res_key] = len(residue_id_map)
            else:
                residue_id_map[res_key] = 100000 + len(residue_id_map)

        meta_resnames.append(resname)
        meta_atomnames.append(atomname)
        meta_residx.append(residue_id_map[res_key])

        # AMBER ff14SB 파라미터 조회 → Particle 생성.
        # _atom_params() → amber_params.get_atom_params() 위임:
        #   이온 잔기 → ION_PARAMS 직접 반환
        #   표준 AA  → (잔기명, 원자명) → AMBER 원자 유형 → VDW + RESP 전하
        #   알 수 없는 잔기 → 원소 기호 기반 폴백
        # Create Particle: _atom_params() → get_atom_params() which handles ions,
        # standard AA (full AMBER params), and unknown residues (element fallback).
        charge, r, e = _atom_params(atom)
        atoms.append(physics_mod.Particle(
            float(coord[0]), float(coord[1]), float(coord[2]),
            charge, r, e, False))

    if skipped:
        log(f"  ⚠  {skipped} atoms skipped (invalid coords)")
    if n_ions:
        log(f"  ions: {n_ions} metal/ion atoms included")

    # ── 수소 결손 전하 보정 (United-atom charge correction) ───────────────────
    # PDB 결정구조는 거의 항상 수소를 해상하지 않으므로, 위 루프에서 만든 원자
    # 목록에는 애초에 수소 Particle이 없다. PARTIAL_CHARGES는 전원자(all-atom)
    # 전하이므로, 수소가 없으면 그 수소가 상쇄했어야 할 전하가 통째로 누락되어
    # 모든 잔기가 물리적으로 잘못된 순전하를 갖게 된다 (예: ALA는 원래 중성인데
    # 중원자만 합산하면 -0.535, LYS는 +1이어야 하는데 -0.88). amber_params.
    # missing_hydrogen_charge()로 이 결손을 모(母) 중원자에 되돌려준다("united
    # atom" 근사) — 단, 그 잔기 인스턴스에 실제로 해당 수소가 있으면 이미 자체
    # Particle이 전하를 갖고 있으므로 보정하지 않는다.
    #
    # Real PDB structures almost never resolve hydrogens, so the atom list
    # built above never had hydrogen Particles to begin with. PARTIAL_CHARGES
    # holds full all-atom charges, so omitting hydrogens silently drops the
    # charge they would have carried, leaving every residue with an
    # unphysical net charge (e.g. ALA reads -0.535 instead of neutral; LYS
    # reads -0.88 instead of +1). Fold each missing hydrogen's charge back
    # onto its parent heavy atom (a standard "united atom" approximation) via
    # amber_params.missing_hydrogen_charge() — skipped for any hydrogen that
    # IS actually present in this residue instance, since its own Particle
    # already carries that charge.
    residue_atomnames: dict[int, set] = {}
    for _ridx, _aname in zip(meta_residx, meta_atomnames):
        residue_atomnames.setdefault(_ridx, set()).add(_aname)
    for _i, (_resname, _atomname, _ridx) in enumerate(
            zip(meta_resnames, meta_atomnames, meta_residx)):
        _extra = _amber_missing_h_charge(_resname, _atomname, residue_atomnames[_ridx])
        if _extra:
            atoms[_i].charge += _extra

    # ── 결합 위상 그래프 구성 (Build covalent bond topology) ─────────────────
    # BondTopology는 HETATM 잔기(bond_templates에 없음)를 조용히 건너뛴다.
    # BondTopology.build() silently skips HETATM residues (not in bond_templates).
    topo = protein_physics.BondTopology()
    topo.build(meta_resnames, meta_atomnames, meta_residx)
    # 곁사슬 협동 이동 후보 쌍 식별 (2026-07-09, IMPROVEMENTS.md 항목 #2 참고):
    # build()는 좌표가 필요 없어 여기서 별도로 호출한다 — 좌표(atoms)가 이 시점에
    # 이미 준비되어 있으므로 build() 직후가 자연스러운 호출 지점이다.
    #
    # Concerted sidechain-pair candidate identification (2026-07-09, see
    # IMPROVEMENTS.md item #2): needs coordinates, so build() doesn't do this
    # itself -- atoms are already available here, right after build() is the
    # natural place to call it.
    topo.identify_concerted_sidechain_pairs(atoms)
    log(f"  topology: {topo.num_bonds} bonds · {topo.num_rot_bonds} rotatable"
        f" · {topo.num_concerted_pairs} crankshaft pairs"
        f" · {len(topo.concerted_sidechain_pairs)} sidechain-pair candidates")

    # ── 이황화 결합 탐지 (Disulfide bond detection, P2.3) ─────────────────────
    # 모든 CYS SG 원자 쌍 중 SG–SG < 2.5 Å인 쌍을 이황화 결합으로 간주한다.
    # 탐지 기준 2.5 Å = 이황화 결합 공유 결합 거리(2.0–2.1 Å) + 0.5 Å 여유.
    # 탐지된 쌍마다 topo.add_disulfide(i, j) 호출:
    #   1) disulfide_pairs에 등록 → 하모닉 구속 에너지에 기여
    #   2) excl에 추가 → 비결합 합산에서 제외 (1-2 쌍 배제)
    #
    # Scan all CYS SG–SG pairs for distances < 2.5 Å (= SG-SG bond length 2.0-2.1 Å
    # plus 0.5 Å tolerance).  For each detected pair, add_disulfide():
    #   1) Registers in disulfide_pairs → contributes harmonic restraint energy
    #   2) Adds to excl[] → excluded from the non-bonded pair sum (1-2 exclusion)
    n_ss = 0
    for ai, (ia, ax, ay, az) in enumerate(sg_atoms):
        for ib, bx, by, bz in (sg_atoms[k] for k in range(ai + 1, len(sg_atoms))):
            dx, dy, dz = ax - bx, ay - by, az - bz
            if dx*dx + dy*dy + dz*dz < 6.25:   # 2.5² = 6.25 Å²
                topo.add_disulfide(ia, ib)
                n_ss += 1
    if n_ss:
        log(f"  disulfide bonds: {n_ss} detected")

    # ── IUPred 무질서 예측 (Sequence-based disorder prediction, P3.1) ────────
    # 서열만으로 잔기별 무질서 확률을 계산한다. O(N × WINDOW) 시간, 즉각 처리.
    # 이 값은 무질서 패널에 즉시 표시되어 landscape MC 결과를 기다릴 필요 없음.
    # Pure sequence-based prediction — O(N × WINDOW), computed immediately on parse.
    # Enables the disorder panel before the landscape MC run completes.
    iupred_scores: list[float] = []
    if ca_resnames:
        iupred_scores = _iupred.score_from_resnames(ca_resnames)

    return (atoms, ca_indices, ca_map, topo, iupred_scores, ca_residues,
            heavy_map, heavy_indices, heavy_keys)

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

def _heavy_atom_map_from_pdb(path):
    """Return {(chain_id, res_seq, atom_name): coord} for a reference PDB file
    (AlphaFold / SWISS-MODEL). Mirrors _ca_map_from_pdb but keeps every
    standard-ATOM-record non-hydrogen heavy atom instead of Cα only.

    _ca_map_from_pdb과 동일한 구조이나 Cα뿐 아니라 모든 표준 ATOM 중원자를 수집한다.
    전체 중원자 RMSD(P10)에서 참조 구조 쪽 맵을 만들 때 사용한다.

    수소·HETATM을 제외하는 이유는 _parse_pdb의 "전체 중원자 RMSD" 주석 참고:
    참조 구조는 수소가 없고, 리간드/이온은 소스 간 1:1 대응이 보장되지 않는다.
    See the "all-heavy-atom RMSD" comment block in _parse_pdb for why hydrogens
    and HETATM records are excluded (reference files lack explicit hydrogens;
    ligands/ions aren't guaranteed to correspond 1:1 across sources).
    """
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", path)
    heavy_map = {}
    for model in st:
        for chain in model:
            for res in chain:
                if res.get_id()[0] != " ":
                    continue
                for atom in res:
                    name = atom.get_name().strip()
                    if name.startswith("H"):
                        continue
                    key = (chain.get_id(), res.get_id()[1], name)
                    heavy_map[key] = atom.get_coord().copy()
        break
    return heavy_map

def _compute_rmsd(ca_map1, ca_map2):
    """Kabsch-superimposed Cα RMSD (Å) between two residue-keyed coordinate maps.
    두 잔기 키 좌표 맵 사이의 Kabsch 중첩 Cα RMSD (Å).

    Only residues present in both maps are compared.  Returns None when fewer
    than 3 common residues exist (Kabsch requires at least 3 points).
    두 맵에 공통으로 있는 잔기만 비교. 공통 잔기 < 3개면 None 반환.

    Kabsch 알고리즘 이론 (Kabsch Algorithm Theory):
    ─────────────────────────────────────────────
    두 구조를 최적으로 겹치는 회전 행렬 R을 SVD로 구한다 (R을 Q에 적용해 P에 맞춘다):
      1. 각 구조의 무게중심을 원점으로 이동 (centroid subtraction)
      2. 공분산 행렬 H = Q^T P 계산 (Q=이동할 구조, P=기준 구조 — 순서가 중요함)
      3. SVD: U, S, Vt = svd(H)
      4. 반사(reflection) 보정: d = sign(det(Vᵀ Uᵀ)) → 행렬식이 -1이면 반사이므로 보정
         d = sign(det(Vt.T @ U.T)); R = Vt.T @ diag(1, 1, d) @ U.T
      5. RMSD = sqrt(mean(||Q·Rᵀ - P||²))

    The Kabsch rotation handles the reflection ambiguity via SVD sign correction.
    Kabsch 회전은 SVD 부호 보정으로 반사 모호성을 처리한다.

    H must be Q^T P (mobile^T @ reference), not P^T Q — swapping the order
    yields the covariance's transpose, which SVD happily factors into a
    *different* (generally wrong) rotation. The resulting RMSD is silently
    inflated whenever the two structures are actually rotated relative to each
    other — exactly the case for AlphaFold/SWISS-MODEL/PDB structures, each
    solved/predicted in its own independent coordinate frame.
    """
    common = sorted(set(ca_map1.keys()) & set(ca_map2.keys()))
    if len(common) < 3:
        return None
    c1 = np.array([ca_map1[k] for k in common], dtype=float)   # P: reference
    c2 = np.array([ca_map2[k] for k in common], dtype=float)   # Q: mobile
    c1 -= c1.mean(0); c2 -= c2.mean(0)
    H = c2.T @ c1
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = c1 - (c2 @ R.T)
    return float(np.sqrt((diff ** 2).sum(1).mean()))

def _kabsch_fit(ref_map, mobile_map):
    """공통 Cα 키를 이용해 mobile_map을 ref_map 프레임에 겹치는 (R, ref_centroid,
    mobile_centroid)를 반환. 공통 잔기 < 3개면 None.
    Return (R, ref_centroid, mobile_centroid) that superimposes mobile_map onto
    ref_map's frame via their common Cα keys. None if fewer than 3 are shared.

    _compute_rmsd와 동일한 SVD 유도 — 반환값을 재사용해 스칼라 RMSD뿐 아니라
    전체 원자 좌표 변환에도 같은 정렬을 적용할 수 있게 분리했다.
    Same SVD derivation as _compute_rmsd, factored out so the same alignment
    can be applied to whole-structure coordinates, not just the RMSD scalar.
    """
    common = sorted(set(ref_map.keys()) & set(mobile_map.keys()))
    if len(common) < 3:
        return None
    p = np.array([ref_map[k] for k in common], dtype=float)
    q = np.array([mobile_map[k] for k in common], dtype=float)
    p_c, q_c = p.mean(0), q.mean(0)
    p0, q0 = p - p_c, q - q_c
    H = q0.T @ p0   # Q^T P (mobile^T @ reference) — see _compute_rmsd note
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, p_c, q_c

def _ensemble_ca_map(ref_ca_map, ca_indices, particles):
    """ref_ca_map과 동일한 잔기 키로 MC 앙상블 후보의 Cα 좌표 맵을 재구성.
    Rebuild an MC ensemble candidate's Cα coordinate map, keyed the same way
    as ref_ca_map, so it can be Kabsch-fit against it (same logic as
    CompareWorker._mc_ca_map, needed here too for the layered 3D view).
    """
    keys = list(ref_ca_map.keys())
    ca_map = {}
    for j, key in enumerate(keys):
        if j < len(ca_indices):
            pidx = ca_indices[j]
            if pidx < len(particles):
                p = particles[pidx]
                ca_map[key] = np.array([p.x, p.y, p.z])
    return ca_map

def _kabsch_align_points(ref_pts, mobile_pts, fit_mask=None):
    """ref_pts, mobile_pts: (n,3) arrays of the same points in
    correspondence (same order, no keys needed). Returns mobile_pts
    optimally Kabsch-rotated/translated onto ref_pts' frame.

    Same SVD derivation as _kabsch_fit/_compute_rmsd, but operating
    directly on arrays instead of residue-keyed maps -- needed for
    landscape basin comparisons, where two representative snapshots can
    have drifted to completely different absolute positions/orientations
    (torsion-angle MC has no reason to keep a molecule's overall pose
    fixed) even when their actual shape is nearly identical. Comparing raw
    coordinates without this conflates "real conformational difference"
    with "same shape, different orientation" -- exactly the bug already
    fixed once for the external layered 3D view (see _render_external_layered).

    fit_mask: optional boolean array (same length as ref_pts/mobile_pts)
    selecting which points determine the fitted rotation -- e.g. an
    IUPred-predicted rigid "core" subset. The rotation is still APPLIED to
    every point in mobile_pts; only the least-squares fit itself is
    restricted. Needed whenever part of the point set is genuinely mobile
    (a disordered region): fitting on the full set lets that motion
    contaminate the rotation itself ("the alignment chases the tail"),
    which can both inflate apparent RMSF in the truly rigid part and mask
    the disordered region's own real motion. Falls back to using every
    point if fit_mask is None or selects fewer than 3 points (Kabsch needs
    >=3 non-degenerate points to define a rotation).
    """
    if fit_mask is not None and np.count_nonzero(fit_mask) >= 3:
        fit_ref, fit_mob = ref_pts[fit_mask], mobile_pts[fit_mask]
    else:
        fit_ref, fit_mob = ref_pts, mobile_pts
    ref_c, mob_c = fit_ref.mean(0), fit_mob.mean(0)
    p0, q0 = fit_ref - ref_c, fit_mob - mob_c
    H = q0.T @ p0
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (mobile_pts - mob_c) @ R.T + ref_c

def _backbone_ncac_indices(ca_map, heavy_map, heavy_indices):
    """ca_map과 동일한 잔기 순서로 N/CA/C 원자 인덱스를 반환한다 — 어떤 후보의
    원시 입자 좌표에서도 phi/psi 이면각을 계산할 수 있게 (2차 구조 근사용,
    _secondary_structure_string 참고).
    Per-residue backbone N/CA/C atom indices, in the same residue order as
    ca_map, so phi/psi dihedral angles can be computed from any candidate's
    raw particle coordinates (used for the secondary-structure abstraction —
    see _secondary_structure_string).
    """
    heavy_idx_by_key = {key: idx for key, idx in zip(heavy_map.keys(), heavy_indices)}
    n_idx, ca_idx, c_idx = [], [], []
    for (chain, resseq) in ca_map.keys():
        n_idx.append(heavy_idx_by_key.get((chain, resseq, "N")))
        ca_idx.append(heavy_idx_by_key.get((chain, resseq, "CA")))
        c_idx.append(heavy_idx_by_key.get((chain, resseq, "C")))
    return n_idx, ca_idx, c_idx

def _dihedral_angle(p0, p1, p2, p3):
    """4개 원자 좌표로부터 이면각(도) 계산 — 표준 공식."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))

def _secondary_structure_string(particles, n_idx, ca_idx, c_idx):
    """잔기별 2차 구조 근사 문자열('H'=나선, 'E'=가닥, 'C'=코일/루프) —
    골격 phi/psi 이면각만으로 분류한다.

    완전한 DSSP가 아니다(DSSP는 골격 수소결합 패턴도 함께 쓴다) — 훨씬
    저렴한 표준 라마찬드란 영역 근사다: (phi, psi)가 알파나선 또는
    베타가닥의 핵심 영역에 들어가는지만 본다. 이는 분지들을 원자 좌표
    차이가 아니라 "범주" 차이로 비교하기 위한 성긴 구조적 추상화로
    충분하다 — 매 잔기가 항상 라벨을 얻으며(알려진 안정 구조 라이브러리와
    맞춰볼 필요가 전혀 없다), 논의된 "안정 구조와 비교 불가능한 IDP 구간"
    문제 자체가 애초에 발생하지 않는다.

    Simplified per-residue secondary-structure label ('H'=helix, 'E'=strand,
    'C'=coil/turn) from backbone phi/psi dihedral angles alone.

    This is NOT full DSSP (which additionally uses backbone H-bonding
    geometry) -- it's the much cheaper, standard Ramachandran-region proxy:
    classify (phi, psi) into the core alpha-helix or beta-strand basins,
    else coil. Good enough as a coarse structural abstraction for comparing
    basins by *category* rather than raw atom displacement -- every
    conformation always gets a label this way, with no comparison against
    an external structure library required (the "IDP region with no similar
    stable structure available" problem doesn't arise in the first place).
    """
    n = len(ca_idx)
    labels = ["C"] * n

    def pos(idx):
        if idx is None or idx >= len(particles):
            return None
        p = particles[idx]
        return np.array([p.x, p.y, p.z])

    for i in range(1, n - 1):
        c_prev, n_i, ca_i, c_i, n_next = (
            pos(c_idx[i - 1]), pos(n_idx[i]), pos(ca_idx[i]), pos(c_idx[i]), pos(n_idx[i + 1]))
        if any(v is None for v in (c_prev, n_i, ca_i, c_i, n_next)):
            continue
        phi = _dihedral_angle(c_prev, n_i, ca_i, c_i)
        psi = _dihedral_angle(n_i, ca_i, c_i, n_next)
        if -100 <= phi <= -30 and -67 <= psi <= -7:
            labels[i] = "H"
        elif -180 <= phi <= -45 and (90 <= psi <= 180 or -180 <= psi <= -150):
            labels[i] = "E"
    return "".join(labels)

def _ss_diff_fraction(ss_a, ss_b):
    """두 2차 구조 문자열 사이의 잔기별 불일치 비율."""
    n = min(len(ss_a), len(ss_b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if ss_a[i] != ss_b[i]) / n

class _NotNucleicAcidOrWater(Select):
    """Bio.PDB Select filter: drop nucleic acid / water residues from output.

    핵산/물 잔기를 출력에서 제외하는 Bio.PDB Select 필터.

    _parse_pdb()가 이미 이런 잔기를 건너뛰어 calculate_potential()에는
    영향을 주지 않지만(item #20), _aligned_pdb_text()는 별도로 원본 파일을
    통째로 읽어 렌더링용 PDB 텍스트를 만들기 때문에 같은 필터가 필요하다.
    그렇지 않으면 에너지는 고쳐졌어도 레이어드 3D 뷰에는 여전히 결합된
    DNA/RNA(예: SWISS-MODEL이 단백질-DNA 복합체를 템플릿으로 골랐을 때)가
    남아 있어 시각적으로는 여전히 "이상"해 보인다.

    _parse_pdb() already skips these residues so calculate_potential() is
    unaffected (item #20), but _aligned_pdb_text() separately re-reads the
    raw file to build the rendering PDB text, so it needs the same filter.
    Without it, the energy is fixed but the layered 3D view still shows the
    bound DNA/RNA (e.g. when SWISS-MODEL picked a protein-DNA co-crystal as
    its template) tangled through the protein, still looking "wrong" visually
    even though the underlying physics is now correct.
    """

    def accept_residue(self, residue):
        resname = residue.get_resname().strip()
        return resname not in _NUCLEOTIDE_RESNAMES and resname not in _WATER_RESNAMES


def _aligned_pdb_text(path, ref_ca_map):
    """외부 구조(AlphaFold/SWISS-MODEL)를 ref_ca_map 프레임에 Kabsch 정렬한 뒤
    전체 원자 PDB 텍스트로 반환(핵산/물 잔기는 제외). 공통 Cα < 3개면 원본
    텍스트를 그대로 반환.
    Kabsch-align an external structure (AlphaFold/SWISS-MODEL) onto ref_ca_map's
    frame and return the whole-structure PDB text, excluding nucleic acid and
    water residues (see _NotNucleicAcidOrWater). Falls back to the raw file
    text when fewer than 3 Cα residues are shared (alignment impossible).

    레이어드 뷰에서 두 구조를 겹쳐 보려면 같은 좌표계에 있어야 한다. 외부
    구조는 독립적으로 계산/실험된 구조라 참조 구조와 좌표계가 전혀 다르므로,
    RMSD 계산에 쓰인 것과 동일한 정렬을 원자 전체에 적용해야 겹쳐 보인다.
    Overlaying two structures in the layered view only makes sense once they
    share a coordinate frame. External structures are independently solved/
    predicted, so their frame has no relation to the reference — applying the
    same alignment used for the RMSD figure to every atom is what makes the
    overlay visually meaningful.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()
    ca_map, _ = _ca_map_from_pdb(path)
    fit = _kabsch_fit(ref_ca_map, ca_map)
    if fit is None:
        return raw_text
    R, ref_c, mob_c = fit
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("ext", path)
    for atom in st.get_atoms():
        coord = atom.get_coord()
        atom.set_coord(((coord - mob_c) @ R.T + ref_c).astype(np.float32))
    buf = io.StringIO()
    writer = PDBIO()
    writer.set_structure(st)
    writer.save(buf, select=_NotNucleicAcidOrWater())
    return buf.getvalue()

def _compute_rmsf(snapshots, ca_indices, iupred_scores=None, core_threshold=0.5):
    """Per-Cα root-mean-square fluctuation (Å) across all trajectory snapshots.
    궤적 스냅샷 전체에 걸친 잔기별 Cα RMSF (Å).

    snapshots   — list of particle-lists from LandscapeWorker
                  LandscapeWorker의 MC 스냅샷 목록 (각 원소 = 입자 목록)
    ca_indices  — index of each Cα atom inside each particle list
                  각 Cα 원자의 입자 목록 내 인덱스
    iupred_scores — optional, per-residue IUPred disorder score (same order/
                  length as ca_indices). When given, the Kabsch alignment fit
                  is restricted to the IUPred-predicted-ordered "core"
                  (score < core_threshold) -- see _kabsch_align_points'
                  fit_mask docstring for why. None (default) aligns on every
                  Cα, matching the original behavior.

    Returns a 1-D array of length len(ca_indices).  High RMSF values indicate
    flexible or disordered regions.
    len(ca_indices) 길이의 1D 배열 반환. 높은 RMSF = 유연/무질서 영역.

    RMSF 이론 (Theory):
    ───────────────────
    RMSF_i = sqrt( <(r_i - <r_i>)²> )
    여기서 <> 는 궤적 시간 평균, r_i 는 잔기 i의 Cα 위치.
    MD/MC 시뮬레이션에서 RMSF ≥ 2 Å 는 일반적으로 무질서/유연 영역 기준.
    실험적으로는 X-선 결정학 B-인수(B-factor)에 해당:
      B = 8π²/3 · RMSF²  (등방성 가정)
    High RMSF ↔ high B-factor ↔ crystallographically disordered region.
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

    # 요동을 재기 전 모든 스냅샷을 첫 스냅샷 프레임에 Kabsch 중첩한다 —
    # 안 그러면 실제 내부 유연성과 "형태는 같은데 절대 위치/방향만 다름"이
    # 뒤섞인다. 토션각 MC는 궤적 내내 절대 자세를 유지할 이유가 없고,
    # 다중 분지 탐색(서로 다른 시드에서 독립적으로 뻗어나감)은 이 문제를
    # 훨씬 크게 만든다 — 각 분지가 서로 다른 절대 방향으로 표류한 채
    # 그대로 풀링되면 거의 모든 잔기가 실제로는 그대로인데도 "매우
    # 유연"한 것처럼 보인다.
    #
    # Kabsch-superpose every snapshot onto the first one before measuring
    # fluctuation -- otherwise RMSF conflates real internal flexibility
    # with whole-molecule pose drift. Torsion-angle MC has no reason to
    # keep a fixed absolute position/orientation over a trajectory, and
    # multi-branch exploration (independent drift per branch) makes this
    # much worse: pooling raw, unaligned snapshots from several
    # independently-drifted branches can make nearly every residue look
    # "highly flexible" even when the actual fold barely changes. Same
    # principle as the layered-view/basin-comparison Kabsch fixes
    # elsewhere in this file.
    #
    # Fitting the rotation itself on ALL Cα atoms has a second, separate
    # problem when a real disordered region is present: the least-squares
    # fit gets dragged around by that region's motion ("the alignment
    # chases the tail"), contaminating the reference frame used to measure
    # every residue's fluctuation -- including the genuinely rigid core.
    # Restrict the fit to IUPred-predicted-ordered residues when available
    # (a signal available pre-MC, purely sequence-derived, so it can't be
    # circularly biased by the RMSF it's used to compute), then apply that
    # fitted rotation to all Cα atoms via _kabsch_align_points' fit_mask.
    fit_mask = None
    if iupred_scores is not None and len(iupred_scores) == n_ca:
        fit_mask = np.asarray(iupred_scores) < core_threshold
    ref = coords[0]
    aligned = np.empty_like(coords)
    aligned[0] = ref
    for si in range(1, n_snaps):
        aligned[si] = _kabsch_align_points(ref, coords[si], fit_mask=fit_mask)

    mean_pos = aligned.mean(axis=0)                          # [n_ca, 3]
    diff     = aligned - mean_pos[np.newaxis]                # [n_snaps, n_ca, 3]
    return np.sqrt((diff ** 2).sum(axis=2).mean(axis=0))   # [n_ca]


def _compute_radius_of_gyration(snapshots, ca_indices):
    """Per-snapshot Cα radius of gyration (Å):
    Rg_t = sqrt( mean_i |r_i(t) - r_mean(t)|^2 ).

    Unweighted Cα-based, matching this codebase's other residue-level
    ensemble metrics (RMSF). Rotation/translation-invariant by
    construction -- unlike RMSF, no Kabsch alignment is needed (Rg only
    depends on each snapshot's own internal shape, not its absolute pose).

    A compact, ordered fold should show a tight Rg distribution; a real
    IDP's more open, fluctuating ensemble should show a wider one -- see
    _compute_internal_scaling for the more discriminating polymer-scaling
    companion metric.

    Returns a 1-D array of length len(snapshots).
    """
    n_snaps = len(snapshots)
    n_ca = len(ca_indices)
    if n_snaps == 0 or n_ca == 0:
        return np.array([])
    rg = np.empty(n_snaps)
    for si, particles in enumerate(snapshots):
        coords = np.array([[particles[pidx].x, particles[pidx].y, particles[pidx].z]
                            for pidx in ca_indices if pidx < len(particles)])
        if len(coords) == 0:
            rg[si] = np.nan
            continue
        centroid = coords.mean(0)
        rg[si] = np.sqrt(((coords - centroid) ** 2).sum(1).mean())
    return rg


def _compute_end_to_end(snapshots, ca_indices):
    """Per-snapshot N-to-C terminal Cα-Cα distance (Å) -- the other classic
    polymer-physics ensemble descriptor alongside Rg. Rotation/translation-
    invariant, no alignment needed.

    Returns a 1-D array of length len(snapshots).
    """
    n_snaps = len(snapshots)
    if n_snaps == 0 or len(ca_indices) < 2:
        return np.array([])
    i0, i1 = ca_indices[0], ca_indices[-1]
    dist = np.empty(n_snaps)
    for si, particles in enumerate(snapshots):
        if i0 < len(particles) and i1 < len(particles):
            p0, p1 = particles[i0], particles[i1]
            dist[si] = np.sqrt((p0.x - p1.x) ** 2 + (p0.y - p1.y) ** 2 + (p0.z - p1.z) ** 2)
        else:
            dist[si] = np.nan
    return dist


def _snapshots_to_coord_array(snapshots, ca_indices):
    """Build the (n_snaps, n_ca, 3) float64 array both accelerated ensemble
    metrics (below) need as input to protein_analysis's C++ functions --
    factored out once rather than duplicated in both fallback loops."""
    n_snaps, n_ca = len(snapshots), len(ca_indices)
    coords = np.zeros((n_snaps, n_ca, 3), dtype=float)
    for si, particles in enumerate(snapshots):
        for ci, pidx in enumerate(ca_indices):
            if pidx < len(particles):
                p = particles[pidx]
                coords[si, ci] = [p.x, p.y, p.z]
    return coords


def _compute_internal_scaling(snapshots, ca_indices, min_sep=3):
    """Ensemble-averaged internal distance scaling law: <R_ij> ~ |i-j|^nu,
    the standard polymer-physics descriptor for distinguishing compact
    globules (nu ~ 0.33), ideal/random-coil chains (nu ~ 0.5), and
    expanded/self-avoiding disordered chains (nu ~ 0.6) -- computed from a
    SINGLE protein's own conformational ensemble via its internal residue-
    pair separations (Marsh & Forman-Kay 2010, Biophys J), unlike an
    Rg-vs-chain-length scaling fit, which would need many different-length
    proteins' ensembles to fit at all.

    Computes the ensemble-mean Cα-Cα distance for every residue pair via
    protein_analysis.compute_mean_dist_matrix (C++/OpenMP) when that
    extension is built, else a pure-Python per-snapshot loop (bounding
    memory to O(n_ca^2) rather than O(n_snaps * n_ca^2) -- this codebase's
    larger cases, e.g. 1YPI at n_ca~500, would otherwise need a multi-GB
    broadcast array). Bins by sequence separation |i-j|, then fits
    log<R_ij> = log(R0) + nu*log(|i-j|) via ordinary least squares over
    separations >= min_sep (excludes the shortest separations, where local
    bond/dihedral geometry rather than long-range chain statistics
    dominates the mean distance).

    Returns (seps, mean_dists, nu, log_r0, r_squared) -- seps/mean_dists are
    1-D arrays (one entry per distinct |i-j| present), nu/log_r0 are the
    fitted scaling exponent/prefactor, r_squared is the fit's goodness of
    fit on the log-log data. Returns (array([]), array([]), None, None,
    None) if there aren't enough residues or snapshots to fit.
    """
    n_snaps = len(snapshots)
    n_ca = len(ca_indices)
    if n_snaps == 0 or n_ca < min_sep + 2:
        return np.array([]), np.array([]), None, None, None

    if _analysis_ext is not None:
        coord_arr = _snapshots_to_coord_array(snapshots, ca_indices)
        mean_dist_matrix = _analysis_ext.compute_mean_dist_matrix(coord_arr)
    else:
        mean_dist_matrix = np.zeros((n_ca, n_ca), dtype=float)
        for particles in snapshots:
            coords = np.zeros((n_ca, 3), dtype=float)
            for ci, pidx in enumerate(ca_indices):
                if pidx < len(particles):
                    p = particles[pidx]
                    coords[ci] = [p.x, p.y, p.z]
            diff = coords[:, None, :] - coords[None, :, :]         # [n_ca, n_ca, 3]
            mean_dist_matrix += np.sqrt((diff ** 2).sum(-1))
        mean_dist_matrix /= n_snaps

    # Bin by separation via np.diagonal rather than a full-matrix boolean
    # mask per separation. mean_dist_matrix is symmetric, so the old
    # `mean_dist_matrix[seps_all == sep]` approach summed BOTH the upper
    # and lower diagonal at each offset -- two copies of the same (i,i+sep)
    # values, which doesn't change the mean but costs an O(n_ca^2) mask
    # comparison per separation (O(n_ca^3) total over the whole loop).
    # np.diagonal(matrix, offset=sep) reads the same numbers directly in
    # O(n_ca-sep) with no comparison pass at all -- O(n_ca^2) total,
    # mathematically identical output (verified: max diff ~5e-17 on a
    # synthetic symmetric matrix, floating-point noise). This was the
    # actual bottleneck at large n_ca once the matrix build itself moved
    # to C++ (IMPROVEMENTS.md item #7): at n_ca=1500 the old binning loop
    # alone took 2.18s, ~9x longer than the C++-accelerated matrix build.
    seps, mean_dists = [], []
    for sep in range(min_sep, n_ca):
        diag = np.diagonal(mean_dist_matrix, offset=sep)
        if diag.size > 0:
            seps.append(sep)
            mean_dists.append(diag.mean())
    seps = np.array(seps, dtype=float)
    mean_dists = np.array(mean_dists, dtype=float)

    if len(seps) < 3:
        return seps, mean_dists, None, None, None

    log_sep, log_dist = np.log(seps), np.log(mean_dists)
    A = np.vstack([log_sep, np.ones_like(log_sep)]).T
    (nu, log_r0), _residuals, *_rest = np.linalg.lstsq(A, log_dist, rcond=None)
    pred = A @ np.array([nu, log_r0])
    ss_res = float(np.sum((log_dist - pred) ** 2))
    ss_tot = float(np.sum((log_dist - log_dist.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return seps, mean_dists, float(nu), float(log_r0), r_squared


def _compute_contact_map(snapshots, ca_indices, cutoff=8.0):
    """Per-residue-pair contact frequency across the ensemble (Cα-Cα
    distance < cutoff Å), plus a per-residue "contact variance" summary --
    a structural-promiscuity metric complementary to RMSF.

    RMSF measures how far a residue's POSITION strays from its mean in a
    fixed (Kabsch-aligned) reference frame, which becomes shakier the more
    genuinely disordered a region is -- a residue sampling many unrelated
    environments doesn't have a well-defined "mean position" even after the
    core-restricted alignment fix above. Contact frequency instead asks
    "which residues is this one near," which stays meaningful without any
    global alignment at all (only relative Cα-Cα distances, computed per
    snapshot, are used) -- a residue that contacts a shifting, inconsistent
    set of partners across the ensemble is a real disorder signal even when
    RMSF's alignment-dependent answer for it is noisy.

    Computed via protein_analysis.compute_contact_freq_matrix (C++/OpenMP)
    when that extension is built, else a pure-Python per-snapshot loop
    (same O(n_ca^2)-at-a-time pattern as _compute_internal_scaling, to
    avoid an O(n_snaps * n_ca^2) broadcast array).

    Returns (contact_freq, per_residue_variance):
    - contact_freq: [n_ca, n_ca] array, fraction of snapshots each residue
      pair is in contact.
    - per_residue_variance: [n_ca] array. For each residue i, treats each
      partner j's contact_freq[i,j] as a Bernoulli probability and averages
      p_ij*(1-p_ij) over partners -- high when a residue's contacts are
      inconsistent across the ensemble (near 0.25 per partner at p=0.5),
      low when contacts are reliably present or reliably absent (p near 0
      or 1). Excludes the diagonal and immediate sequence neighbors
      (i±1, i±2), which are always in contact by covalent-bond construction
      and would otherwise dilute every residue's score toward "consistent"
      regardless of real disorder.
    """
    n_snaps = len(snapshots)
    n_ca = len(ca_indices)
    if n_snaps == 0 or n_ca == 0:
        return np.zeros((0, 0)), np.array([])

    if _analysis_ext is not None:
        coord_arr = _snapshots_to_coord_array(snapshots, ca_indices)
        contact_freq = _analysis_ext.compute_contact_freq_matrix(coord_arr, cutoff)
    else:
        contact_freq = np.zeros((n_ca, n_ca), dtype=float)
        for particles in snapshots:
            coords = np.zeros((n_ca, 3), dtype=float)
            for ci, pidx in enumerate(ca_indices):
                if pidx < len(particles):
                    p = particles[pidx]
                    coords[ci] = [p.x, p.y, p.z]
            diff = coords[:, None, :] - coords[None, :, :]
            dist = np.sqrt((diff ** 2).sum(-1))
            contact_freq += (dist < cutoff)
        contact_freq /= n_snaps
        np.fill_diagonal(contact_freq, 0.0)

    bernoulli_var = contact_freq * (1.0 - contact_freq)
    mask = np.ones((n_ca, n_ca), dtype=bool)
    np.fill_diagonal(mask, False)
    for offset in (1, 2):
        if offset < n_ca:
            idx = np.arange(n_ca - offset)
            mask[idx, idx + offset] = False
            mask[idx + offset, idx] = False
    per_residue_variance = np.array([
        bernoulli_var[i][mask[i]].mean() if mask[i].any() else 0.0
        for i in range(n_ca)
    ])
    return contact_freq, per_residue_variance


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
    전체 분석 파이프라인을 실행하는 백그라운드 스레드 (QThread 상속).

    Qt 시그널-슬롯(signal-slot) 패턴을 사용해 GUI 스레드와 통신:
    메인 스레드를 블로킹하지 않고 긴 계산을 비동기로 수행.
    Uses Qt signal-slot pattern to communicate with the GUI thread
    without blocking it during long calculations.

    Signals / 시그널:
      progress(str)   — log message for the sidebar process log
                        사이드바 로그에 표시할 진행 메시지
      metrics(dict)   — partial metrics to update sidebar counters
                        사이드바 카운터 업데이트용 부분 메트릭
      finished(...)   — emitted on success with (ensemble, energies, pdb_path,
                        ca_indices, ca_map, init_atoms)
                        성공 시 방출 — 앙상블, 에너지, PDB 경로, Cα 인덱스 등
      error(str)      — emitted on unrecoverable failure
                        복구 불가 오류 시 방출

    Steps (run() 메서드 실행 순서):
    ────────────────────────────────
      1. _fetch()     — PDB 파일을 디스크 → RCSB → AlphaFold API 순으로 검색
                        Retrieve PDB from disk, RCSB, or AlphaFold API
      2. _parse_pdb() — BioPython으로 PDB 파싱 후 AMBER 매개변수 매핑,
                        C++ Particle 목록과 Cα 인덱스 맵 구성
                        Build Particle list + Cα index map with AMBER params
      3. generate_ensemble() — C++ 엔진으로 Metropolis MC 앙상블 생성
                               (GIL 해제 → Qt GUI 스레드 응답 유지)
                               Run MC via C++ engine (GIL released)
      4. calculate_potential() — 각 후보 배열의 최종 에너지 계산
                                  Compute final GB/DH/LJ/SASA energies
    """
    progress = pyqtSignal(str)
    metrics  = pyqtSignal(dict)
    # ensemble, energies, pdb_path, ca_indices, ca_map, init_atoms, topo, extra
    # extra dict: {"iupred_scores": [...], "ca_residues": [...],
    #              "heavy_map": {...}, "heavy_indices": [...], "heavy_keys": [...]}
    # (heavy_* added for IMPROVEMENTS #10 all-heavy-atom RMSD; kept in extra so the
    #  finished() signal arity doesn't need to change — same precedent as iupred_scores.)
    finished = pyqtSignal(object, object, str, object, object, object, object, object)
    error    = pyqtSignal(str)
    # Emitted when the GPU engine fails at runtime (not just at import/device-name
    # time) and this worker fell back to CPU to finish the current analysis. The
    # main window listens for this to permanently downgrade to the CPU engine for
    # the rest of the session, so later analyses don't repeat the same failure.
    gpu_fallback = pyqtSignal(str)

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
        data_dir = os.path.join(_app_base_dir(), "data")
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
            (atoms, ca_indices, ca_map, topo, iupred_scores, ca_residues,
             heavy_map, heavy_indices, heavy_keys) = \
                _parse_pdb(path, self.progress.emit, self.physics_mod)
            if not atoms:
                self.error.emit("No valid protein atoms found.")
                return
            self.metrics.emit({"n_atoms": len(atoms), "threads": self.engine.num_threads()})
            self.progress.emit(f"  {len(atoms)} atoms · {self.engine.num_threads()} threads")

            # ── 매듭 위상 분류 (Backbone knot topology, P5.1) ────────────────
            # Calpha 궤적을 폐곡선으로 만들어 매듭 유형을 분류한다. 서열/구조와
            # 무관하게 한 번만 계산하면 되므로 MC 실행 전, 백그라운드 스레드에서
            # 수행한다(수 초 소요 — UI를 막지 않기 위해 워커 스레드에서 실행).
            # Classify the Calpha trace's knot topology. Purely structural (not
            # MC-dependent), so computed once here in the background worker
            # thread before the MC run -- this can take a few seconds and must
            # not block the UI.
            # 다중 체인 버그 (2026-07-09): ca_map.values()를 그대로 이었더니
            # 서로 공유결합으로 연결되지 않은 체인 경계에서 인위적인 "결합"이
            # 생겼다 — 1YPI(이합체, 각 체인 247잔기)에서 실측: 체인 내
            # 정상 Cα-Cα 거리 3.86Å 대비 체인 경계에서 65.94Å짜리 가짜 세그
            # 먼트가 생겨, 텍스트북상 매듭 없음(unknot)인 TIM-배럴이 58-92%
            # 신뢰도로 트레포일(trefoil)로 오분류됐다(IMPROVEMENTS.md 항목
            # #6). 매듭은애초에 하나의 연속된 곡선에서만 정의되는 개념이라
            # 여러 체인을 이은 값 자체가 위상수학적으로 무의미하다 — 임계값
            # 조정이 아니라 체인별로 분리해 가장 큰 체인 하나만 분류하는 것이
            # 올바른 수정이다.
            #
            # Multi-chain bug (2026-07-09): concatenating ca_map.values() raw
            # created an artificial "bond" at chain boundaries between chains
            # that aren't covalently connected -- measured directly on 1YPI
            # (a homodimer, 247 res/chain): a real within-chain Ca-Ca distance
            # of 3.86 A vs. a 65.94 A fake segment at the chain boundary,
            # which got a textbook-unknotted TIM-barrel misclassified as a
            # trefoil at only 58-92% confidence (IMPROVEMENTS.md item #6).
            # A knot is only a well-defined property of a single continuous
            # curve, so concatenating separate chains isn't just noisy data --
            # it's topologically meaningless. The fix is to classify each
            # chain separately (using only the largest/primary one here, not
            # a threshold tweak), not paper over the boundary artifact.
            self.progress.emit("  Classifying backbone topology (knot analysis)…")
            try:
                ca_coords_by_chain: dict = {}
                for (chain_id, _res_seq), coord in ca_map.items():
                    ca_coords_by_chain.setdefault(chain_id, []).append(coord)
                primary_chain = max(ca_coords_by_chain,
                                     key=lambda c: len(ca_coords_by_chain[c]))
                if len(ca_coords_by_chain) > 1:
                    self.progress.emit(
                        f"  Multi-chain structure ({len(ca_coords_by_chain)} chains) -- "
                        f"classifying topology for chain {primary_chain} only "
                        f"(knot type isn't well-defined across separate, non-bonded chains)")
                ca_coords = np.array(ca_coords_by_chain[primary_chain], dtype=float)
                knot_result = _knot.classify_backbone_knot(ca_coords, n_trials=24)
                self.progress.emit(
                    f"  Topology: {knot_result.name} "
                    f"({knot_result.crossing_number} crossings, "
                    f"{knot_result.confidence*100:.0f}% closure-vote confidence)")
            except Exception as ex:
                self.progress.emit(f"  Topology classification skipped: {ex}")
                knot_result = None

            self.progress.emit(
                f"  Running MC: {self.n_cand} candidates × {self.steps} steps…")
            # ── GPU 런타임 실패 시 CPU로 자동 폴백 ───────────────────────────
            # _try_gpu_backend()은 앱 시작 시 device_name()이 성공하는지만 확인한다
            # (임포트/디바이스 감지 확인일 뿐). 실제 커널 실행은 완전히 다른 실패
            # 지점이다 — 예: 배포용 빌드가 이 머신의 드라이버가 지원하지 않는
            # 툴체인으로 컴파일된 PTX를 포함하고 있으면, 파싱까지는 멀쩡히
            # 끝나고 나서 generate_ensemble() 호출에서 처음으로 CUDA 오류가
            # 터진다. 이런 실패로 전체 분석을 중단시키는 대신, CPU 엔진으로
            # 다시 시도해 이번 실행만이라도 완료시킨다.
            # _try_gpu_backend() only confirms device_name() succeeds at startup
            # (import/device-detection only). The actual kernel launch is a
            # completely separate failure point -- e.g. a distributed build's
            # CUDA extension can contain PTX compiled by a toolchain this
            # machine's driver doesn't support, which only surfaces here, well
            # after parsing has already succeeded cleanly. Rather than letting
            # that crash the whole analysis, fall back to the CPU engine and
            # finish this run with it.
            try:
                ensemble = self.engine.generate_ensemble(
                    atoms, topo, self.n_cand, self.steps, 0.6, 0.12)
                self.progress.emit("  Computing ensemble free energies…")
                energies = [self.engine.calculate_potential(s, topo) for s in ensemble]
            except Exception as ex:
                if self.physics_mod is protein_physics:
                    raise  # already on CPU -- nothing left to fall back to
                self.progress.emit(f"  ⚠ GPU engine failed at runtime ({ex}) — falling back to CPU")
                self.gpu_fallback.emit(str(ex))
                self.physics_mod = protein_physics
                self.engine = protein_physics.PhysicsEngine()
                self.metrics.emit({"threads": self.engine.num_threads()})
                ensemble = self.engine.generate_ensemble(
                    atoms, topo, self.n_cand, self.steps, 0.6, 0.12)
                self.progress.emit("  Computing ensemble free energies… (CPU)")
                energies = [self.engine.calculate_potential(s, topo) for s in ensemble]
            self.metrics.emit({"best_e": min(energies), "n_cand": self.n_cand})
            extra = {"iupred_scores": iupred_scores, "ca_residues": ca_residues,
                      "heavy_map": heavy_map, "heavy_indices": heavy_indices,
                      "heavy_keys": heavy_keys, "knot_result": knot_result}
            self.finished.emit(ensemble, energies, path, ca_indices, ca_map, atoms, topo, extra)
        except Exception as ex:
            self.error.emit(str(ex))

# ═══════════════════════════════════════════════════════════════════
#  ComparisonWorker — fetch AlphaFold + SWISS-MODEL, compute RMSD
# ═══════════════════════════════════════════════════════════════════
class ComparisonWorker(QThread):
    progress = pyqtSignal(str)
    result   = pyqtSignal(list)

    def __init__(self, target, pdb_path, ca_indices, ref_ca_map,
                 ensemble, energies, engine, physics_mod,
                 heavy_indices=None, ref_heavy_map=None):
        super().__init__()
        self.target      = target
        self.pdb_path    = pdb_path
        self.ca_indices  = ca_indices
        self.ref_ca_map  = ref_ca_map
        self.ensemble    = ensemble
        self.energies    = energies
        self.engine      = engine
        self.physics_mod = physics_mod
        # ── 전체 중원자 RMSD용 참조 데이터 (P10) ────────────────────────────
        # heavy_indices/ref_heavy_map이 없으면(구버전 호출 등) 빈 값으로 폴백해
        # Heavy RMSD 열이 "—"로만 표시되고 나머지 기능은 그대로 동작한다.
        # Reference data for all-heavy-atom RMSD. Falls back to empty containers
        # if not supplied, so the Heavy RMSD column just shows "—" without
        # breaking anything else.
        self.heavy_indices = heavy_indices if heavy_indices is not None else []
        self.ref_heavy_map = ref_heavy_map if ref_heavy_map is not None else {}

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
        data_dir = os.path.join(_app_base_dir(), "data")
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
        data_dir = os.path.join(_app_base_dir(), "data")
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
                    # s.get("gmqe", 0) only falls back to 0 when the key is
                    # ABSENT -- SWISS-MODEL's API returns many older
                    # template-based entries with the key present but set to
                    # null (None), so most real responses mix None and float
                    # gmqe values. max() comparing None to None (or None to a
                    # float) raises TypeError, crashing this fetch outright
                    # for any protein whose SWISS-MODEL entries aren't all
                    # freshly gmqe-scored (confirmed live against P39476: all
                    # bar one of the 16 returned structures had gmqe=null).
                    best = max(structs, key=lambda s: s.get("gmqe") or 0)
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

    def _mc_heavy_map(self, cand_idx):
        """MC 후보의 전체 중원자 좌표 맵을 (chain, res_seq, atom_name) 키로 재구성.
        Rebuild an MC candidate's all-heavy-atom coordinate map keyed by
        (chain, res_seq, atom_name).

        _mc_ca_map과 동일한 원리: MC 앙상블 후보는 좌표 배열뿐이므로 파싱 시점에
        저장해 둔 heavy_indices로 원래 (잔기, 원자명) 키를 다시 붙여준다.
        Same idea as _mc_ca_map: MC ensemble candidates are bare coordinate
        arrays, so heavy_indices (captured at parse time) lets us re-attach the
        original (chain, res_seq, atom_name) key.
        """
        keys      = list(self.ref_heavy_map.keys())
        particles = self.ensemble[cand_idx]
        heavy_map = {}
        for j, key in enumerate(keys):
            if j < len(self.heavy_indices):
                pidx = self.heavy_indices[j]
                if pidx < len(particles):
                    p = particles[pidx]
                    heavy_map[key] = np.array([p.x, p.y, p.z])
        return heavy_map

    def _energy_for(self, path):
        # topo 없이 calculate_potential을 호출하면 1-2/1-3 배제가 적용되지 않아
        # 모든 공유 결합 쌍(~1.5 Å)이 비결합 항으로 평가되어 수천억 kcal/mol
        # 단위의 허구적 척력 에너지가 나온다 — AlphaFold/SWISS-MODEL 비교
        # 구조에서 관측된 비정상적으로 큰 양의 에너지의 원인이었다.
        #
        # calculate_potential without a topo skips 1-2/1-3 exclusions, so every
        # covalent bond (~1.5 Å apart) gets scored as a non-bonded contact,
        # producing spurious repulsive energies in the hundreds of billions of
        # kcal/mol — the cause of the previously-observed absurd positive
        # energies for AlphaFold/SWISS-MODEL comparison structures.
        try:
            atoms, *_, topo, _, _, _, _, _ = _parse_pdb(path, lambda *_: None, self.physics_mod)
            if atoms:
                return self.engine.calculate_potential(atoms, topo)
        except Exception:
            pass
        return None

    def run(self):
        results  = []
        best_idx = int(np.argmin(self.energies))

        for i, energy in enumerate(self.energies):
            mc_ca    = self._mc_ca_map(i)
            rmsd     = _compute_rmsd(self.ref_ca_map, mc_ca) if mc_ca else None
            mc_heavy = self._mc_heavy_map(i)
            rmsd_heavy = _compute_rmsd(self.ref_heavy_map, mc_heavy) if mc_heavy else None
            results.append({
                "source": f"MC  C{i+1}", "is_mc": True, "mc_idx": i,
                "is_best": i == best_idx, "energy": energy,
                "rmsd": rmsd, "rmsd_heavy": rmsd_heavy,
                "plddt": None, "path": self.pdb_path,
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
                af_heavy = _heavy_atom_map_from_pdb(af_path)
                results.append({
                    "source": "AlphaFold", "is_mc": False, "is_best": False,
                    "energy": self._energy_for(af_path),
                    "rmsd": _compute_rmsd(self.ref_ca_map, af_ca),
                    "rmsd_heavy": _compute_rmsd(self.ref_heavy_map, af_heavy),
                    "plddt": avg_plddt, "path": af_path,
                })

        if uniprot_id:
            self.progress.emit("  [CMP] Fetching SWISS-MODEL homology model…")
            sm_path = self._fetch_swissmodel(uniprot_id)
            if sm_path:
                sm_ca, _ = _ca_map_from_pdb(sm_path)
                sm_heavy = _heavy_atom_map_from_pdb(sm_path)
                results.append({
                    "source": "Homology  (SWISS-MODEL)", "is_mc": False, "is_best": False,
                    "energy": self._energy_for(sm_path),
                    "rmsd": _compute_rmsd(self.ref_ca_map, sm_ca),
                    "rmsd_heavy": _compute_rmsd(self.ref_heavy_map, sm_heavy),
                    "plddt": None, "path": sm_path,
                })

        self.progress.emit(f"  [CMP] Done — {len(results)} sources")
        self.result.emit(results)

class _LandscapeBranchRunnable(QRunnable):
    """Runs one landscape MC branch on its own fresh PhysicsEngine instance.

    Not safe to share a single engine instance across concurrently-running
    branches -- the CPU engine keeps one mutable member (a single mt19937
    RNG) that every MC step reads and writes, so two branches racing on the
    same instance would corrupt each other's random stream. Constructing an
    engine is cheap (trivial on CPU; the CUDA ctor just checks device count),
    so each branch gets its own rather than sharing one.

    Writes its result into `results[bi]` as (snapshots, energies, error):
    - success: (snaps, ens, None)
    - GPU call failed, fell back to CPU successfully: (snaps, ens, <the GPU
      exception>) -- snaps/ens are populated from the CPU retry, but the
      original exception is kept so the caller knows a fallback happened.
    - fatal (already on CPU, or the CPU retry also failed): (None, None, ex)

    Each `results[bi]` slot is written by exactly one runnable, so no lock is
    needed -- the caller only reads the list after `QThreadPool.waitForDone()`.
    """
    def __init__(self, bi, seed, topo, n_snapshots, steps_per_snap, T, max_angle,
                 physics_mod, results):
        super().__init__()
        self.bi = bi
        self.seed = seed
        self.topo = topo
        self.n_snapshots = n_snapshots
        self.steps_per_snap = steps_per_snap
        self.T = T
        self.max_angle = max_angle
        self.physics_mod = physics_mod
        self.results = results

    def run(self):
        try:
            engine = self.physics_mod.PhysicsEngine()
            snaps, ens = engine.run_landscape_trajectory(
                self.seed, self.topo, self.n_snapshots, self.steps_per_snap,
                self.T, self.max_angle)
            self.results[self.bi] = (snaps, ens, None)
        except Exception as ex:
            if self.physics_mod is protein_physics:
                self.results[self.bi] = (None, None, ex)
                return
            try:
                cpu_engine = protein_physics.PhysicsEngine()
                snaps, ens = cpu_engine.run_landscape_trajectory(
                    self.seed, self.topo, self.n_snapshots, self.steps_per_snap,
                    self.T, self.max_angle)
                self.results[self.bi] = (snaps, ens, ex)
            except Exception as ex2:
                self.results[self.bi] = (None, None, ex2)


def _rodrigues_rotate_atoms(atoms, atom_indices, origin, axis, angle):
    """Rotate atoms[i] for i in atom_indices around `axis` (unit vector) through
    `origin` by `angle` radians, in place. Same math as physics_engine.cpp's
    rodrigues() helper, reimplemented here in plain Python/numpy since this is
    called once per pivot branch, not inside a hot per-step MC loop."""
    ox, oy, oz = origin
    ux, uy, uz = axis
    cosA, sinA = np.cos(angle), np.sin(angle)
    for idx in atom_indices:
        px, py, pz = atoms[idx].x - ox, atoms[idx].y - oy, atoms[idx].z - oz
        dot = px * ux + py * uy + pz * uz
        cx = uy * pz - uz * py
        cy = uz * px - ux * pz
        cz = ux * py - uy * px
        rx = px * cosA + cx * sinA + ux * dot * (1 - cosA)
        ry = py * cosA + cy * sinA + uy * dot * (1 - cosA)
        rz = pz * cosA + cz * sinA + uz * dot * (1 - cosA)
        atoms[idx].x = rx + ox
        atoms[idx].y = ry + oy
        atoms[idx].z = rz + oz


def _coarse_pivot(seed_atoms, topo, physics_mod, rng, n_pivots=1, angle_range=(1.0, 3.0)):
    """Apply n_pivots large, UNCONDITIONAL (non-Metropolis) backbone phi/psi
    rotations to a fresh copy of seed_atoms, producing a genuinely different
    starting conformation for a dedicated MC branch.

    This is the basin-hopping "propose" step: no energy evaluation or
    accept/reject happens here, deliberately. IMPROVEMENTS.md item #2 records
    two prior attempts (try_crankshaft_coupled, and testing generate_ensemble
    at high temperature) that tried to get a large structural change accepted
    by ordinary per-step Metropolis MC -- both failed, because the immediate
    post-move energy of a large backbone rotation is dominated by transient
    steric clashes and gets rejected almost by construction (or, at high T,
    the ANGLE_MAX=0.50 rad cap and sidechain-dominated move mix mean no large
    move is even proposed). Standard basin-hopping avoids this by judging
    acceptance on the RELAXED energy after a kick, not the raw post-kick
    energy -- this function is the "kick" half; the caller (_PivotBranchRunnable)
    supplies the "relax" half via the existing run_landscape_trajectory MC.

    Picks a random topo.concerted_pairs entry (the existing phi/psi crankshaft
    pairing) rather than an arbitrary raw dihedral, so the axis/side-set
    relationship is already-validated structure, then rotates one of that
    pair's two bonds by a large (well beyond ANGLE_MAX) random angle.
    """
    pivoted = [physics_mod.Particle(a.x, a.y, a.z, a.charge, a.radius, a.epsilon, a.is_water)
               for a in seed_atoms]
    pairs = topo.concerted_pairs
    if not pairs:
        return pivoted, 0
    n_applied = 0
    for _ in range(n_pivots):
        phi_idx, psi_idx = pairs[rng.integers(len(pairs))]
        bond_idx = phi_idx if rng.random() < 0.5 else psi_idx
        rb = topo.rot_bonds[bond_idx]
        side = topo.rot_bond_sides[bond_idx]
        if not side:
            continue
        origin = np.array([pivoted[rb.i].x, pivoted[rb.i].y, pivoted[rb.i].z])
        target = np.array([pivoted[rb.j].x, pivoted[rb.j].y, pivoted[rb.j].z])
        axis = target - origin
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            continue
        axis /= norm
        angle = rng.uniform(*angle_range) * (1 if rng.random() < 0.5 else -1)
        _rodrigues_rotate_atoms(pivoted, side, origin, axis, angle)
        n_applied += 1
    return pivoted, n_applied


class _PivotBranchRunnable(QRunnable):
    """Coarse structural-pivot branch: applies _coarse_pivot() to the seed
    structure, then relaxes the result via the same run_landscape_trajectory
    MC every other branch uses, over the same per-branch step budget.

    Same output contract as _LandscapeBranchRunnable -- (snapshots, energies,
    error) into results[bi] -- so it plugs into the existing pooling/PCA/
    DBSCAN pipeline with zero downstream changes. See IMPROVEMENTS.md item #2
    for the design rationale (basin-hopping "kick then relax", built as its
    own branch rather than a per-step move so the kick is never itself
    Metropolis-judged).
    """
    def __init__(self, bi, seed, topo, n_snapshots, steps_per_snap, T, max_angle,
                 physics_mod, results, n_pivots=1, rng_seed=None):
        super().__init__()
        self.bi = bi
        self.seed = seed
        self.topo = topo
        self.n_snapshots = n_snapshots
        self.steps_per_snap = steps_per_snap
        self.T = T
        self.max_angle = max_angle
        self.physics_mod = physics_mod
        self.results = results
        self.n_pivots = n_pivots
        self.rng_seed = rng_seed

    def run(self):
        try:
            rng = np.random.default_rng(self.rng_seed)
            pivoted, _n_applied = _coarse_pivot(
                self.seed, self.topo, self.physics_mod, rng, n_pivots=self.n_pivots)
            engine = self.physics_mod.PhysicsEngine()
            snaps, ens = engine.run_landscape_trajectory(
                pivoted, self.topo, self.n_snapshots, self.steps_per_snap,
                self.T, self.max_angle)
            self.results[self.bi] = (snaps, ens, None)
        except Exception as ex:
            if self.physics_mod is protein_physics:
                self.results[self.bi] = (None, None, ex)
                return
            try:
                cpu_engine = protein_physics.PhysicsEngine()
                pivoted, _n_applied = _coarse_pivot(
                    self.seed, self.topo, protein_physics, np.random.default_rng(self.rng_seed),
                    n_pivots=self.n_pivots)
                snaps, ens = cpu_engine.run_landscape_trajectory(
                    pivoted, self.topo, self.n_snapshots, self.steps_per_snap,
                    self.T, self.max_angle)
                self.results[self.bi] = (snaps, ens, ex)
            except Exception as ex2:
                self.results[self.bi] = (None, None, ex2)


class _ReplicaExchangeBranchRunnable(QRunnable):
    """Runs one landscape MC branch as a parallel-tempering (replica exchange)
    ensemble instead of a single fixed-temperature chain.

    Same output contract as _LandscapeBranchRunnable: writes (snapshots,
    energies, error) into results[bi]. Everything downstream (pooling into
    all_snapshots/all_energies, PCA/DBSCAN classification) is unchanged --
    only how this one branch's samples are generated differs.

    Why PT and not plain MC or simulated annealing: the sampling-depth
    investigation (IMPROVEMENTS.md item #2) found the MC chain's integrated
    autocorrelation time never converges (it grows with trace length,
    indicating the chain is still slowly relaxing at any affordable step
    budget, not merely under-sampled). Simulated annealing was tested as a
    burn-in accelerant and empirically made things worse (lower, noisier
    funnel scores) -- and using it for actual production sampling would
    bias the walk toward the single lowest-energy state, corrupting the
    population-weighted basin estimates this classifier depends on. Replica
    exchange fixes the mixing problem without that bias: each temperature
    "slot" runs its own canonical-ensemble chain, and periodic swap moves
    between neighboring slots are themselves Metropolis-accepted with a
    criterion chosen specifically to preserve each slot's own stationary
    distribution -- so slot 0 (the physical target temperature) is still a
    valid sample from the same ensemble the plain single-temperature chain
    was already trying to sample, just better-mixed.

    Swap acceptance: physics_engine.cpp's own Metropolis step accepts moves
    with probability exp(-dE/T), and T is already in the same energy units
    as dE (kcal/mol; T=0.6 ~= kT at 300K, not literal Kelvin through a
    Boltzmann-constant conversion) -- see the "Monte Carlo Sampling" comment
    block there. That means the standard replica-swap formula between
    neighboring slots a, b (energies E_a, E_b, temperatures T_a, T_b)
    applies directly with no unit conversion:

        p_swap = min(1, exp((E_a - E_b) * (1/T_a - 1/T_b)))

    derived from requiring detailed balance against each slot's own
    pi(E) ~ exp(-E/T) -- no k_B factor needed since the engine's own
    per-step acceptance already uses this exact convention.

    Segmented execution: run_landscape_trajectory runs its entire chain in
    one C++ call with no checkpoint, so a swap needs the chain broken into
    short segments with a barrier between them. Reuses the block-chaining
    technique already proven this session (autocorr_probe.py, the
    annealing test, and _dedicated_subsearch's existing sequential-reuse
    pattern): call run_landscape_trajectory(state, topo,
    n_snapshots=PT_SWAP_INTERVAL, steps_per_snap, T_k, max_angle) per
    replica per segment, take snaps[-1] as the next segment's initial
    state, and repeat. Only slot 0's snapshots/energies from each segment
    are pooled into the branch's returned trajectory -- higher-temperature
    slots exist only to help slot 0 mix faster, never feed classification.

    Each replica keeps its own persistent PhysicsEngine across all segments
    (only ever touched by whichever thread is running that replica's
    current segment -- safe because QThreadPool.waitForDone() between
    segments is a full barrier, so handing the same engine object to a
    different pooled thread next segment has proper happens-before
    ordering, same safety argument as _LandscapeBranchRunnable's "one
    engine, one thread at a time" rule, just not literally the same OS
    thread for the whole run).
    """
    def __init__(self, bi, seed, topo, n_snapshots, steps_per_snap, T, max_angle,
                 physics_mod, results, n_replicas, ladder_ratio, swap_interval,
                 rng_seed):
        super().__init__()
        self.bi = bi
        self.seed = seed
        self.topo = topo
        self.n_snapshots = n_snapshots
        self.steps_per_snap = steps_per_snap
        self.T = T
        self.max_angle = max_angle
        self.physics_mod = physics_mod
        self.results = results
        self.n_replicas = n_replicas
        self.ladder_ratio = ladder_ratio
        self.swap_interval = swap_interval
        self.rng_seed = rng_seed

    def run(self):
        try:
            snaps, ens = self._run_pt(self.physics_mod)
            self.results[self.bi] = (snaps, ens, None)
        except Exception as ex:
            if self.physics_mod is protein_physics:
                self.results[self.bi] = (None, None, ex)
                return
            try:
                snaps, ens = self._run_pt(protein_physics)
                self.results[self.bi] = (snaps, ens, ex)
            except Exception as ex2:
                self.results[self.bi] = (None, None, ex2)

    def _run_pt(self, physics_mod):
        rng = np.random.default_rng(self.rng_seed)
        ladder = [self.T * (self.ladder_ratio ** k) for k in range(self.n_replicas)]
        engines = [physics_mod.PhysicsEngine() for _ in range(self.n_replicas)]
        states = [self.seed for _ in range(self.n_replicas)]
        energies = [None] * self.n_replicas
        # Per-replica tuned step size, threaded across segments (see
        # run_landscape_segment's C++ docstring for why this exists --
        # run_landscape_trajectory's online step-size tuning is call-scoped
        # and would otherwise restart from self.max_angle every segment).
        cur_maxes = [self.max_angle for _ in range(self.n_replicas)]

        n_segments = max(1, self.n_snapshots // self.swap_interval)
        pooled_snaps, pooled_energies = [], []

        for seg in range(n_segments):
            seg_results = [None] * self.n_replicas
            pool = QThreadPool()
            pool.setMaxThreadCount(max(1, self.n_replicas))
            for k in range(self.n_replicas):
                pool.start(_ReplicaSegmentRunnable(
                    k, engines[k], states[k], self.topo, self.swap_interval,
                    self.steps_per_snap, ladder[k], cur_maxes[k], seg_results))
            pool.waitForDone()

            for k in range(self.n_replicas):
                seg_snaps, seg_ens, seg_cur_max = seg_results[k]
                states[k] = seg_snaps[-1]
                energies[k] = seg_ens[-1]
                cur_maxes[k] = seg_cur_max
                if k == 0:
                    pooled_snaps.extend(seg_snaps)
                    pooled_energies.extend(seg_ens)

            # Alternating-pair swap scheme: (0,1),(2,3),... on even segments,
            # (1,2),(3,4),... on odd -- standard PT bookkeeping so every
            # neighbor pair gets attempted over time without needing a
            # simultaneous all-pairs resolution.
            offset = seg % 2
            for a in range(offset, self.n_replicas - 1, 2):
                b = a + 1
                dE = energies[a] - energies[b]
                d_beta = (1.0 / ladder[a]) - (1.0 / ladder[b])
                exponent = dE * d_beta
                p_swap = 1.0 if exponent >= 0 else np.exp(exponent)
                if rng.random() < p_swap:
                    states[a], states[b] = states[b], states[a]
                    energies[a], energies[b] = energies[b], energies[a]
                    # The tuned step size belongs to the physical replica
                    # (configuration) that's moving between temperature
                    # slots, same as its configuration and energy already do.
                    cur_maxes[a], cur_maxes[b] = cur_maxes[b], cur_maxes[a]

        return pooled_snaps, pooled_energies


class _ReplicaSegmentRunnable(QRunnable):
    """Runs one replica's single segment within one PT swap interval.

    Calls run_landscape_segment (not run_landscape_trajectory) specifically
    to get the final tuned cur_max back -- see that C++ method's docstring
    for why: run_landscape_trajectory's online step-size tuning is call-
    scoped and would otherwise restart from self.max_angle every segment
    instead of carrying forward what it learned.

    Writes (snapshots, energies, final_cur_max) into results[k]. Note
    QThreadPool does not propagate a worker thread's Python exception back
    to the caller -- if run_landscape_segment raises here, results[k] is
    simply left at its None sentinel. The caller
    (_ReplicaExchangeBranchRunnable._run_pt) unpacking that None then raises
    its own TypeError, which the outer _ReplicaExchangeBranchRunnable.run()
    try/except does catch (triggering the same GPU-fallback-retry-or-mark-
    fatal path as _LandscapeBranchRunnable) -- so a segment failure is not
    silently swallowed or hung, but the original exception's detail is lost
    in favor of the secondary TypeError. Acceptable for now since this path
    is not yet the production default.
    """
    def __init__(self, k, engine, state, topo, n_snapshots, steps_per_snap,
                 T, max_angle, results):
        super().__init__()
        self.k = k
        self.engine = engine
        self.state = state
        self.topo = topo
        self.n_snapshots = n_snapshots
        self.steps_per_snap = steps_per_snap
        self.T = T
        self.max_angle = max_angle
        self.results = results

    def run(self):
        snaps, ens, final_cur_max = self.engine.run_landscape_segment(
            self.state, self.topo, self.n_snapshots, self.steps_per_snap,
            self.T, self.max_angle)
        self.results[self.k] = (snaps, ens, final_cur_max)


# ═══════════════════════════════════════════════════════════════════
#  LandscapeWorker — MC trajectory → conformational graph
# ═══════════════════════════════════════════════════════════════════
class LandscapeWorker(QThread):
    progress = pyqtSignal(str)
    result   = pyqtSignal(dict)
    # See PipelineWorker.gpu_fallback -- same runtime-failure/CPU-retry pattern,
    # needed here too since this is the other long-running GPU MC call.
    gpu_fallback = pyqtSignal(str)

    # 표본 깊이 보정 (2026-07-07): 전용 계측 스크립트(scratchpad/autocorr_probe.py)로
    # 에너지 궤적의 실제 적분 자기상관시간(τ)을 직접 측정한 결과 τ≈500 스텝
    # (1UBQ 76잔기, 1XQ8 140잔기 모두 거의 동일 — 이 범위에서는 단백질
    # 크기와 거의 무관해 보임). 기존 값(40 스냅샷×80스텝=분지당 3200스텝)은
    # N_eff = steps/(2τ) ≈ 3.2 — 즉 분지 하나가 저장한 40개 스냅샷 중
    # 실질적으로 독립적인 정보는 3개 남짓이었다. 이번 세션 내내 관찰된
    # 라벨 불안정성(같은 단백질을 다시 돌리면 ORDERED/POSSIBLY DISORDERED가
    # 뒤바뀜)의 근본 원인이 바로 이것 — 자세한 경위는 IMPROVEMENTS.md 항목
    # #2 참고.
    #
    # 1차 목표로 분지당 N_eff≈30-40 (기존 대비 약 10배)까지만 올린다 — 장기
    # 목표치(N_eff≈150-300, 30-50배)는 이번 단계에서 검증하기엔 너무 크다.
    # STEPS_PER_SNAP은 τ의 2-3배 정도로 잡아 저장되는 스냅샷 하나하나가
    # 실제로 더 많은 독립 정보를 담도록 하고(기존 80은 τ의 1/6 수준이라
    # 이웃 스냅샷끼리 거의 중복이었다), N_SNAPSHOTS는 분지 수와 무관하게
    # "분지당" 목표 스냅샷 수로 재정의한다(이전에는 분지 수로 나눠 분지가
    # 늘수록 분지당 깊이가 오히려 얕아지는 구조였다).
    #
    # Sampling-depth correction (2026-07-07): a dedicated instrumentation
    # script (scratchpad/autocorr_probe.py) measured the *real* integrated
    # autocorrelation time (tau) of the energy trace directly, by calling
    # run_landscape_trajectory() with steps_per_snapshot=1. Result: tau≈500
    # steps on both 1UBQ (76 res) and 1XQ8 (140 res) -- apparently close to
    # size-independent in this range. The old values (40 snapshots x 80
    # steps = 3200 raw steps/branch) gave N_eff = steps/(2*tau) ≈ 3.2 --
    # each branch's 40 "saved snapshots" carried only ~3 genuinely
    # independent samples. This is the root, now-quantified cause of the
    # label instability (ORDERED <-> POSSIBLY DISORDERED on repeat runs of
    # the same protein) observed all session -- see IMPROVEMENTS.md item #2
    # for the full investigation.
    #
    # First-pass target: N_eff ~= 30-40 per branch (~10x the old ~3.2), not
    # the full long-run target (N_eff ~= 150-300, a 30-50x step increase --
    # too large a first jump to validate safely). STEPS_PER_SNAP is set to
    # roughly 2-3x tau so each saved snapshot actually carries more
    # independent information (the old 80 was ~tau/6, so neighboring saved
    # snapshots were mostly duplicates). N_SNAPSHOTS is now a per-branch
    # target, not a total split across branches (the old formula divided a
    # fixed total by branch count, so *more* branches meant *less* depth per
    # branch -- backwards for what multi-branch exploration is for).
    N_SNAPSHOTS    = 15     # snapshots per branch (was: total across all branches)
    STEPS_PER_SNAP = 600    # MC steps between each snapshot (was: 80; ~2.4x measured tau≈500)
    # 2026-07-09 재조정: 위 두 값은 원래 30×1200이었으나, GPU 속도 비교 조사 중
    # (아래 "런타임 최적화" 참고) 정확도 저하 없이 예산을 더 줄일 수 있는지
    # 직접 측정했다. 1UBQ/1LYZ/1YPI/1XQ8(진짜 IDP) 네 단백질 모두에서 예산을
    # 절반(스냅샷 수·스텝 수 각각 절반, 총 스텝 ~4배 감소)으로 줄여도 12/12
    # 반복 실행이 정답을 유지했고 funnel 점수도 같은 범위였다 — IMPROVEMENTS.md
    # 항목 #2 "런타임 최적화" 절 참고. 원래 30×1200 값은 이제 절반이 된 이
    # 상수들의 2배로 남겨두지 않고, 검증된 값 자체로 갱신한다.
    #
    # 2026-07-09 re-tuning: these two were originally 30×1200, halved after a
    # GPU-speedup investigation (see "Runtime optimization" below) directly
    # measured whether the budget could be cut further without losing
    # accuracy. Tested on all 4 available ground-truth proteins (1UBQ, 1LYZ,
    # 1YPI, and 1XQ8 -- a real IDP): halving both n_snapshots and steps (a
    # ~4x total step reduction) still gave 12/12 correct repeat-run labels
    # with funnel scores in the same range as the un-halved budget -- see
    # IMPROVEMENTS.md item #2's "Runtime-optimization follow-up" for the
    # full table. These are the validated values themselves, not a multiplier
    # applied on top of the original 30×1200 anchor.

    # 크기별 표본 깊이 보정 (2026-07-08): 위 상수들은 76-140잔기 범위에서만 검증됐다.
    # 1YPI(494잔기, 강체 TIM-배럴, RMSF 0%)로 재확인한 결과 funnel 점수가 129잔기
    # 리소자임과 다를 바 없이 낮았고(0.10-0.21) ORDERED/POSSIBLY DISORDERED가
    # 2/4로 갈렸다 — 단백질이 커질수록(PCA 특징 차원 = 3×n_ca가 ~6.5배 증가) DBSCAN이
    # 군집화할 표본 수는 그대로라 실제 우물이 여러 조각으로 쪼개지는 것으로 보인다.
    #
    # 애초에 τ(자기상관시간)를 다시 측정해 STEPS_PER_SNAP을 τ에 맞춰 늘리려 했으나,
    # 전용 계측(scratchpad/autocorr_probe.py)에서 τ 추정치가 궤적을 늘릴수록 계속
    # 커졌다(2천 스텝→508, 2만 스텝→3995, 10만 스텝→26228 — 한 번도 수렴하지 않음).
    # 1UBQ 10만 스텝 궤적의 전반부/후반부 평균 에너지도 -1.54σ만큼 단조 하강해,
    # T=0.6에서 사슬이 감당 가능한 예산 안에서는 정상 상태(stationary)에 도달하지
    # 못한다는 뜻이다 — 잘 정의된 τ가 없으므로 "STEPS_PER_SNAP ∝ τ" 공식 자체가
    # 성립하지 않는다(이전 세션의 τ≈500 측정도 같은 이유로 과소측정이었을 가능성이
    # 높다). 이는 IUPred R-hat 조사(에너지 기반·PC1 기반 모두 분류 정확도를 예측하지
    # 못하고 폐기됨, IMPROVEMENTS.md 항목 #2 참고)와 같은 패턴 — 단일 스칼라
    # 혼합(mixing) 진단이 이 분류기에서는 반복적으로 신뢰할 수 없는 것으로 드러났다.
    #
    # 따라서 실용적 접근으로 전환: N_SNAPSHOTS만 단백질 크기에 맞춰 늘려(군집화
    # 밀도 문제를 직접 겨냥 — τ 측정 실패와 무관하게 유효한 메커니즘) DBSCAN이
    # 더 많은 표본으로 군집을 나누게 하고, STEPS_PER_SNAP은 이론적 τ 공식 없이
    # 실행시간 예산으로만 제한한다. 검증은 이론적 N_eff 목표가 아니라 반복 실행
    # 라벨 안정성(landscape_stability_test.py)만으로 판단한다.
    #
    # Size-adaptive sampling depth (2026-07-08): the constants above were only
    # validated at 76-140 residues. Re-checking with 1YPI (494 res, a rigid TIM-
    # barrel, RMSF 0%) showed funnel scores no better than 129-res lysozyme
    # (0.10-0.21) and a 2/4 ORDERED/POSSIBLY-DISORDERED split -- as protein size
    # grows (PCA feature dim = 3*n_ca grows ~6.5x), the pooled sample count DBSCAN
    # clusters over stays fixed, so a real single basin looks fragmented.
    #
    # The original plan was to re-measure the autocorrelation time (tau) and
    # scale STEPS_PER_SNAP to match it. A dedicated probe (scratchpad/
    # autocorr_probe.py) found tau keeps growing with trace length instead of
    # converging (2k steps -> 508, 20k steps -> 3995, 100k steps -> 26228) and
    # the 100k-step 1UBQ trace's first-half/second-half mean energy drifted by
    # -1.54 std devs -- the chain at T=0.6 hasn't reached a stationary
    # distribution within any affordable budget, so there's no well-defined tau
    # to build a "STEPS_PER_SNAP ~ tau" formula on (the earlier tau~500 estimate
    # was likely undermeasured for the same reason). This matches a pattern
    # already seen with the R-hat convergence diagnostic (both energy- and
    # PC1-based variants failed to predict classification correctness and were
    # abandoned -- see IMPROVEMENTS.md item #2): single-scalar mixing
    # diagnostics have repeatedly proven unreliable for this classifier.
    #
    # Pragmatic pivot: only grow N_SNAPSHOTS with protein size (targets the
    # clustering-density problem directly, independent of the failed tau
    # measurement) and bound STEPS_PER_SNAP by a runtime budget instead of a
    # mixing-time formula. Validated empirically via repeat-run label stability
    # (tests/landscape_stability_test.py) rather than a theoretical N_eff target.
    ANCHOR_N_RES       = 76      # ubiquitin -- matches the COMPETITIVE_KT anchor below
    N_SNAPSHOTS_CAP    = 30      # 2026-07-09: halved alongside N_SNAPSHOTS (was 60 = 2x30)
    STEPS_PER_SNAP_MIN = 300     # 2026-07-09: halved alongside STEPS_PER_SNAP (was 600)
    WORK_BUDGET_MULT   = 3.0     # allow ~3x the anchor's total MC-step work

    @classmethod
    def _adaptive_depth(cls, n_res, disorder_frac=0.0):
        """Per-branch (n_snapshots, steps_per_snap) scaled for protein size
        and (optionally) IUPred-predicted disorder fraction.

        n_snapshots grows with sqrt(n_res/ANCHOR_N_RES) so DBSCAN still has enough
        pooled points to resolve real basins as the PCA feature space grows with
        size. steps_per_snap is held at the validated anchor value and only
        trimmed if the total work (n_snapshots * steps_per_snap * n_res, since
        per-step cost is ~O(n_res)) would exceed WORK_BUDGET_MULT times the
        anchor's work -- see class-level comment above for why this isn't tau-
        derived.

        disorder_frac (IMPROVEMENTS.md item #7, "disorder-aware sampling"):
        the fraction of residues IUPred predicts disordered (0.0 if
        unavailable -- e.g. a caller that never computed IUPred scores at
        all; tests/landscape_stability_test.py DOES have real sequence info
        from the parsed PDB and passes real scores through as of the
        2026-07-13 fix, see that file). A genuinely
        disordered region is, by construction, sampling a heterogeneous mix
        of sub-populations rather than fluctuating around one well -- the
        same "DBSCAN needs more pooled points to resolve what's really
        there" argument that already motivates the size-based sqrt scaling
        above, just along a second axis. Scaled linearly (not sqrt, unlike
        the size term): more disordered residues means proportionally more
        distinct heterogeneous states to resolve, not a spread-of-noise
        argument the central-limit-style sqrt reasoning applies to. Modest,
        bounded first-pass constant (DISORDER_DEPTH_MULT=1.0, i.e. a fully
        IUPred-disordered protein gets up to 2x depth) -- applied before the
        existing N_SNAPSHOTS_CAP clamp, so it can't runaway past the
        already-validated cost ceiling; not yet independently tuned beyond
        that cap, same "reasoned starting value, not a fitted optimum"
        posture as every other constant introduced this way in this file.
        """
        r = max(n_res, 1) / cls.ANCHOR_N_RES
        disorder_mult = 1.0 + cls.DISORDER_DEPTH_MULT * max(0.0, min(1.0, disorder_frac))
        n_snap = int(round(cls.N_SNAPSHOTS * r ** 0.5 * disorder_mult))
        n_snap = max(cls.N_SNAPSHOTS, min(n_snap, cls.N_SNAPSHOTS_CAP))

        steps = cls.STEPS_PER_SNAP
        work_max = cls.WORK_BUDGET_MULT * (cls.N_SNAPSHOTS * cls.STEPS_PER_SNAP * cls.ANCHOR_N_RES)
        work = n_snap * steps * n_res
        if work > work_max:
            steps = max(cls.STEPS_PER_SNAP_MIN, int(work_max / (n_snap * n_res)))
        return n_snap, steps

    # 병렬 템퍼링 (Replica exchange / parallel tempering, 2026-07-09) — 실험적,
    # 기본값 꺼짐. 표본 깊이 조사에서 발견한 비정상성(τ가 궤적 길이에 따라
    # 계속 커지고 절대 수렴하지 않음, 즉 사슬이 어떤 예산으로도 완전히
    # 평형에 도달하지 못함)을 실제로 해결하기 위한 시도 — 예산을 줄이는
    # 것(위 N_SNAPSHOTS/STEPS_PER_SNAP)은 증상을 우회했을 뿐, 원인은 그대로다.
    # 담금질(simulated annealing)은 번인 가속용으로 시험했으나 도움이 안 됐고
    # (오히려 funnel 점수가 더 낮고 불안정해짐), 실제 생산 샘플링에 쓰면
    # 최저 에너지 상태로 편향돼 개체군 가중 basin 추정을 왜곡한다 — 병렬
    # 템퍼링은 각 온도 슬롯이 자기 고유의 정준 앙상블을 유지하도록 교환
    # 수용 기준 자체가 설계되어 있어 이 편향이 없다. 4개 검증 단백질
    # (1UBQ/1LYZ/1YPI/1XQ8)로 검증한 결과는 엇갈렸다 — 1LYZ(역사적으로 가장
    # 불안정했던 사례)에서는 funnel 점수가 실제로 더 높고 좁아졌지만(0.75-1.00
    # vs 평범한 MC의 0.30-0.63), 나머지 세 단백질(1UBQ/1XQ8/1YPI)에서는
    # 뚜렷한 개선이 없었고 특히 1YPI(애초에 이 조사를 시작한 대상)에서는
    # funnel 점수가 오히려 더 불안정해졌다 — 모든 경우에서 1.4-2.5배의
    # 실행 시간 비용이 들었다. 라벨 자체는 15/15 반복 실행 모두 정확했으나
    # (한 번도 틀리지 않음), 이 정도의 엇갈린 근거로는 기본값으로 켤 이유가
    # 되지 않는다 — IMPROVEMENTS.md 항목 #2 참고. 기본 False 유지, 향후
    # 사다리 간격·교환 주기 등을 다시 튜닝해볼 수 있는 옵트인 기능으로 남김.
    #
    # Replica exchange / parallel tempering (2026-07-09) -- an attempt to
    # actually fix the non-stationarity found during the sampling-depth
    # investigation (tau keeps growing with trace length and never
    # converges -- the chain never fully equilibrates at any affordable
    # budget), rather than just working around it (the N_SNAPSHOTS/
    # STEPS_PER_SNAP budget cut above sidesteps the symptom, not the cause).
    # Simulated annealing was tried as a burn-in accelerant and didn't help
    # (lower, noisier funnel scores); using it for actual production
    # sampling would bias the walk toward the minimum-energy state and
    # corrupt population-weighted basin estimates. Replica exchange avoids
    # that bias by design -- see _ReplicaExchangeBranchRunnable's docstring
    # for the full swap-acceptance derivation. Validated against all 4
    # ground-truth proteins: mixed result. 1LYZ (historically the most
    # flip-flop-prone case) showed a real, tighter/higher funnel range
    # (0.75-1.00 vs plain MC's 0.30-0.63); 1UBQ/1XQ8/1YPI showed no clear
    # improvement, and 1YPI specifically (the original motivating case for
    # this whole investigation) got a *noisier* funnel, not tighter --
    # each case cost 1.4-2.5x more wall time regardless. Labels stayed
    # correct in all 15/15 PT runs (never broke anything), but this
    # evidence doesn't support switching the default on -- kept False,
    # available as an opt-in for future retuning (the ladder/swap-interval
    # constants below are untuned starting guesses, not validated optima).
    #
    # Follow-up (2026-07-09): found a real bug behind 1YPI's regression --
    # run_landscape_trajectory's own online step-size tuning (cur_max) is
    # call-scoped, resetting every call; PT called it once per short
    # segment instead of once per full trajectory, so cur_max never got
    # more than ~15 tuning windows before being discarded. Fixed via a new
    # C++ method, run_landscape_segment (also returns the final cur_max so
    # it can be threaded across segments -- see _ReplicaSegmentRunnable/
    # _ReplicaExchangeBranchRunnable._run_pt); physics_engine_cuda.cu got
    # the identical fix too (2026-07-09), so this works on both engines --
    # this class's own PT orchestration needed no changes either way, it
    # already calls run_landscape_segment polymorphically. Retested: 1YPI's
    # funnel narrowed a bit (0.133-0.244 vs 0.122-0.278) and got ~12%
    # faster, but still trails plain MC (0.144-0.189); 1LYZ's funnel
    # dropped back to plain-MC-like (0.233-0.483, the 0.75-1.00 win is
    # gone -- the bug's aggressive step-size reset was apparently an
    # accidental source of useful exploration diversity there); and 1XQ8
    # (the real IDP) came back wrong (ORDERED) on 2 of 5 repeats across two
    # batches, a real, reproduced regression neither the buggy PT nor
    # plain MC ever showed. Kept as a correct bug fix (no reason to revert
    # real engineering), but it does not change the verdict: still False,
    # and this specific design is not recommended for further incremental
    # tuning -- a cost-normalized comparison, a properly tuned ladder, or a
    # different enhanced-sampling method entirely would be needed to
    # revisit this, not another fix to the current architecture.
    USE_REPLICA_EXCHANGE = False
    PT_N_REPLICAS      = 4       # temperature ladder size, starting guess
    PT_LADDER_RATIO    = 1.6     # T_k = T0 * ratio**k; ~20-40% neighbor swap accept is the target
    PT_SWAP_INTERVAL   = 5       # snapshots per segment between swap attempts

    # Coarse structural-pivot branch (IMPROVEMENTS.md item #2, "should a large
    # pivot happen before branches diverge" follow-up to the crankshaft-coupled
    # and decoy-discrimination negative results): one extra branch, kicked away
    # from init_atoms via _coarse_pivot() (an unconditional large phi/psi
    # rotation, not Metropolis-judged) before its own MC relaxation runs. Off
    # by default pending validation -- same conservative posture as
    # USE_REPLICA_EXCHANGE.
    USE_PIVOT_BRANCH = False
    PIVOT_N_PIVOTS   = 1

    # Disorder-aware sampling depth (IMPROVEMENTS.md item #7) -- see
    # _adaptive_depth's docstring for the full derivation. Always active
    # (not an opt-in toggle like USE_REPLICA_EXCHANGE/USE_PIVOT_BRANCH):
    # disorder_frac defaults to 0.0 when no IUPred scores are supplied, so
    # this is a pure no-op extension of the already-shipped size-based
    # depth scaling, not a new behavior that needs its own off-switch.
    DISORDER_DEPTH_MULT = 1.0

    def __init__(self, engine, init_atoms, ca_indices, topo, physics_mod=None, T=0.6, max_angle=0.12,
                 extra_seeds=None, backbone_ncac=None, iupred_scores=None):
        super().__init__()
        self.engine      = engine
        self.init_atoms  = init_atoms
        self.ca_indices  = ca_indices
        self.topo        = topo
        self.physics_mod = physics_mod if physics_mod is not None else protein_physics
        self.T           = T
        self.max_angle   = max_angle
        # IUPred per-residue disorder scores (same order as ca_indices),
        # used only to scale sampling depth via _adaptive_depth -- see
        # DISORDER_DEPTH_MULT above. None/empty -> disorder_frac=0.0,
        # identical to the pre-existing size-only behavior.
        self.iupred_scores = iupred_scores
        # 다중 분지 탐색용 추가 시드 (예: 상위 K개 MC 후보) — 없으면 기존처럼
        # 단일 궤적으로 동작한다.
        # Extra seeds for multi-branch exploration (e.g. the next-best K MC
        # candidates) — falls back to the original single-trajectory
        # behavior when empty.
        self.extra_seeds = list(extra_seeds) if extra_seeds else []
        # (n_idx, ca_idx, c_idx) per residue, ca_map order — for the
        # secondary-structure abstraction (see _secondary_structure_string).
        self.backbone_ncac = backbone_ncac

    def _ca_vec(self, particles):
        """Flatten Cα coordinates of one snapshot into a 1-D vector."""
        v = []
        for i in self.ca_indices:
            if i < len(particles):
                p = particles[i]
                v += [p.x, p.y, p.z]
        return np.array(v, dtype=float)

    def run(self):
        from sklearn.decomposition import PCA
        from sklearn.cluster import DBSCAN

        seeds = [self.init_atoms] + self.extra_seeds
        n_seed_branches = len(seeds)
        use_pivot = self.USE_PIVOT_BRANCH and len(self.topo.concerted_pairs) > 0
        n_branches = n_seed_branches + (1 if use_pivot else 0)
        n_res = len(self.ca_indices)
        disorder_frac = (_iupred.fraction_disordered(self.iupred_scores)
                          if self.iupred_scores else 0.0)
        N_per, S = self._adaptive_depth(n_res, disorder_frac)  # per-branch target -- see class-level comment above

        # ── 다중 분지 탐색 (Multi-branch exploration) ────────────────────
        # 시드 하나(최적 후보)에서만 뻗으면, 안정 단백질에서는 문제없다
        # (우물이 하나뿐이라 어디서 시작해도 결국 거기로 수렴) — 하지만
        # 지배적 상태 자체가 없는 진짜 IDP에서는 "최적" 시드가 사실상
        # 임의의 표본일 뿐이라 거기서 뻗은 궤적 하나가 실제 인구 분포를
        # 대표하지 못한다. 상위 K개 후보에서 각각 뻗어 전부 모아 군집화
        # 하면 특정 시작점에 치우치지 않은 인구 가중 분포를 얻는다.
        #
        # Branching from a single seed (the best candidate) is fine for a
        # stable protein (one deep well -- any start converges there), but
        # for a protein with no dominant state (a real IDP) that "best"
        # seed is close to an arbitrary sample, so one trajectory from it
        # doesn't represent the real population. Branching from several
        # top candidates and pooling all of them gives a population-
        # weighted picture that isn't biased toward one starting point.
        # 분지 병렬 실행 (Branch parallelization, 2026-07-07): 분지들은 서로
        # 완전히 독립적인데도 지금까지는 순차 for 루프로 하나씩 돌았다.
        # 위의 표본 깊이 증가(분지당 스텝 수 약 11배)를 순차 실행 그대로
        # 두면 왕복 시간이 감당하기 어려울 만큼 늘어난다. run_landscape_trajectory
        # 호출은 이미 GIL을 놓으므로(physics_engine.cpp의 py::call_guard) 실제
        # 동시 실행이 가능하다 — QThreadPool/QRunnable로 각 분지를 동시에
        # 돌린다. CPU 엔진은 mt19937 gen 하나를 인스턴스 멤버로 공유하므로
        # (physics_engine.cpp) 같은 엔진 인스턴스를 여러 스레드가 동시에 쓰면
        # 안전하지 않다 — 분지마다 새 PhysicsEngine 인스턴스를 만든다
        # (ComparisonWorker가 이미 쓰던 "스레드 경합 방지용 새 엔진 인스턴스"
        # 관례와 동일). self.engine 자체는 건드리지 않고 그대로 두는데,
        # run() 뒷부분의 _dedicated_subsearch가 이 병렬 구간 이후 순차적으로
        # self.engine을 재사용하기 때문이다.
        #
        # Branches are fully independent, but used to run one after another in
        # a sequential for-loop. Combined with the ~11x per-branch step
        # increase above, sequential execution would make round-trip time
        # unworkable. run_landscape_trajectory already releases the GIL
        # (py::call_guard in physics_engine.cpp), so real concurrent execution
        # is possible -- run each branch via QThreadPool/QRunnable instead.
        # The CPU engine has one shared mutable member (mt19937 gen in
        # physics_engine.cpp), so one engine instance isn't safe to use from
        # multiple threads at once -- give each branch its own fresh
        # PhysicsEngine (mirrors the existing convention at ComparisonWorker:
        # "fresh engine instance to avoid thread contention"). self.engine
        # itself is left untouched here, since _dedicated_subsearch reuses it
        # sequentially later, after this parallel section completes.
        physics_mod_snapshot = self.physics_mod
        branch_results = [None] * n_branches
        pool = QThreadPool()
        pool.setMaxThreadCount(max(1, n_branches))
        pt_note = " [replica exchange]" if self.USE_REPLICA_EXCHANGE else ""
        for bi, seed in enumerate(seeds):
            if n_branches > 1:
                self.progress.emit(
                    f"  [LANDSCAPE] Branch {bi+1}/{n_branches}: "
                    f"running {N_per}×{S}-step chain (n_res={n_res}, parallel){pt_note}…")
            else:
                self.progress.emit(
                    f"  [LANDSCAPE] Running {N_per}×{S}-step Markov chain (n_res={n_res}){pt_note}…")
            if self.USE_REPLICA_EXCHANGE:
                pool.start(_ReplicaExchangeBranchRunnable(
                    bi, seed, self.topo, N_per, S, self.T, self.max_angle,
                    physics_mod_snapshot, branch_results,
                    self.PT_N_REPLICAS, self.PT_LADDER_RATIO, self.PT_SWAP_INTERVAL,
                    rng_seed=bi))
            else:
                pool.start(_LandscapeBranchRunnable(
                    bi, seed, self.topo, N_per, S, self.T, self.max_angle,
                    physics_mod_snapshot, branch_results))
        if use_pivot:
            pivot_bi = n_seed_branches
            self.progress.emit(
                f"  [LANDSCAPE] Branch {pivot_bi+1}/{n_branches}: "
                f"coarse structural pivot + {N_per}×{S}-step relax (n_res={n_res})…")
            pool.start(_PivotBranchRunnable(
                pivot_bi, self.init_atoms, self.topo, N_per, S, self.T, self.max_angle,
                physics_mod_snapshot, branch_results, n_pivots=self.PIVOT_N_PIVOTS,
                rng_seed=n_seed_branches))
        pool.waitForDone()

        all_snapshots, all_energies, branch_lengths = [], [], []
        gpu_fallback_error = None
        for bi in range(n_branches):
            snaps, ens, err = branch_results[bi]
            if snaps is None:
                raise err  # fatal -- matches the original single-branch-failure-aborts-run behavior
            if physics_mod_snapshot is not protein_physics and err is not None:
                gpu_fallback_error = err  # this branch's GPU call failed and fell back to CPU
            all_snapshots.extend(snaps)
            all_energies.extend(ens)
            branch_lengths.append(len(snaps))

        if gpu_fallback_error is not None:
            self.progress.emit(
                f"  ⚠ GPU engine failed at runtime ({gpu_fallback_error}) — falling back to CPU")
            self.gpu_fallback.emit(str(gpu_fallback_error))
            self.physics_mod = protein_physics
            self.engine = protein_physics.PhysicsEngine()

        snapshots = all_snapshots
        energies  = np.array(all_energies, dtype=float)
        N = len(energies)

        self.progress.emit("  [LANDSCAPE] Building conformational graph…")

        # ── PCA: 고차원 구조 공간을 2D로 투영 ────────────────────────────
        # PCA layout — project the high-dim Cα coordinate space to 2D.
        #
        # 각 스냅샷의 Cα 좌표를 이어붙여 (N × 3·n_ca) 행렬을 만든다.
        # PCA는 데이터의 분산이 가장 큰 방향(주성분, PC)을 찾는다.
        #   PC1 = 단백질의 가장 큰 집단 운동 방향 (가장 많은 분산 설명)
        #   PC2 = PC1에 직교하는 두 번째로 큰 운동 방향
        # 이를 통해 에너지 지형(free energy landscape)을 2D로 시각화할 수 있다.
        # Build (N × 3·n_ca) matrix; PCA finds directions of maximum variance.
        # PC1/PC2 capture dominant collective motions of the protein backbone.
        coord_mat = np.array([self._ca_vec(s) for s in snapshots])
        n_feat = coord_mat.shape[1]
        n_pc   = min(2, N - 1, n_feat)
        pca    = PCA(n_components=n_pc)
        layout = pca.fit_transform(coord_mat)
        if layout.ndim == 1 or layout.shape[1] < 2:
            layout = np.column_stack([layout.reshape(-1, 1),
                                      np.zeros((N, 1))])
        var_exp = pca.explained_variance_ratio_.tolist()

        # ── 수렴 진단: 다중 분지 Ȓ (Gelman-Rubin R-hat) ──────────────────
        # 분류 결과(POSSIBLY DISORDERED 등)가 실행마다 뒤집히는 문제를
        # 실증적으로 확인했다(같은 단백질을 반복 실행 → 다른 라벨) — 유비퀴틴
        # funnel이 0.07~0.24로 널뛰고, 극도로 안정적인 라이소자임조차 4번의
        # 반복 실행 중 2번은 POSSIBLY DISORDERED로 잘못 나왔다. 매번 여러 번
        # 재실행해서 눈으로 확인하는 대신, 이미 돌리고 있는 3개의 독립 분지를
        # 그대로 활용해 한 번의 실행 안에서 수렴 여부를 정량적으로 판단한다.
        #
        # 처음에는 풀링된 에너지 궤적으로 Ȓ을 계산했지만, 라이소자임 4회
        # 반복 실행 검증에서 무의미했다: 4번 모두 잘못 분류됐는데, 그중
        # Ȓ=1.01(관례상 "수렴")로 나온 실행도 나머지와 똑같이 틀렸다. 에너지
        # 궤적의 Ȓ은 분지들이 "에너지 분포"에 동의하는지만 보는데, 실제
        # 분류는 PCA 구조 공간(`layout`)에서의 DBSCAN 군집화가 결정한다 —
        # 관련은 있지만 다른 양이다. 그래서 대신 DBSCAN이 실제로 군집화하는
        # 좌표인 PC1(`layout[:, 0]`)에서 Ȓ을 계산한다 — 에너지보다 실제
        # 분류 결과와 훨씬 더 직접적으로 연결된 양이다.
        #
        # 표준 Gelman-Rubin 진단: 분지 간 분산(B)이 분지 내 분산(W)보다 훨씬
        # 크면(Ȓ이 1보다 훨씬 크면) 분지들이 서로 다른 분포로 수렴했다는
        # 뜻 — 즉 아직 다 섞이지 않은 것이다. Ȓ≈1.0이면 잘 수렴한 것.
        # 관례적 기준: Ȓ < 1.1 정도면 수렴, 그 이상이면 의심.
        #
        # Convergence diagnostic: multi-chain R-hat (Gelman-Rubin). We've
        # empirically confirmed the classification label flips between runs
        # of the *same* protein (ubiquitin's funnel swung 0.07-0.24; even
        # lysozyme, about as rigid a control as exists, read POSSIBLY
        # DISORDERED in 2 of 4 repeat runs). Rather than requiring manual
        # repeat-runs to notice this, reuse the 3 branches this function
        # already computes to get a per-run convergence signal for free.
        #
        # First tried on the pooled *energy* trace, but that was uninformative
        # on the 4-run lysozyme validation: all 4 runs misclassified, and the
        # one with R-hat=1.01 (conventionally "converged") was exactly as
        # wrong as the rest. Energy-trace R-hat only checks whether branches
        # agree on the energy *distribution*, but the classification is
        # actually decided by DBSCAN clustering in PCA structural space
        # (`layout`) -- a related but distinct quantity. So compute R-hat on
        # PC1 (`layout[:, 0]`) instead -- the coordinate DBSCAN actually
        # clusters on, and far more directly tied to the classification
        # outcome than energy is.
        #
        # Standard Gelman-Rubin: if between-chain variance (B) dominates
        # within-chain variance (W) -- R-hat well above 1 -- the branches
        # have settled into different pictures of the landscape and haven't
        # mixed yet. R-hat near 1.0 means they agree. Conventional rule of
        # thumb: R-hat < 1.1 is considered converged.
        r_hat = None
        if len(branch_lengths) >= 2 and min(branch_lengths) >= 2:
            n_chain = min(branch_lengths)
            pc1 = layout[:, 0]
            chains = []
            offset = 0
            for length in branch_lengths:
                chains.append(pc1[offset:offset + n_chain])
                offset += length
            chains = np.array(chains)  # shape (m branches, n_chain samples)
            m = chains.shape[0]
            chain_means = chains.mean(axis=1)
            grand_mean  = chain_means.mean()
            B = n_chain / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
            W = float(np.mean([np.var(c, ddof=1) for c in chains]))
            if W > 1e-9:
                var_hat = (n_chain - 1) / n_chain * W + B / n_chain
                r_hat = float(np.sqrt(var_hat / W))

        # ── 밀도 기반 군집화 → 준안정 분지 (Density Clustering → Metastable Basins) ──
        # Density-based clustering — clusters are metastable basins.
        #
        # 이전 구현은 연속 스냅샷을 잇는 체인(경로) 그래프에 그래프 모듈성
        # 군집화를 적용했다.  그러나 경로 그래프에서 탐욕적 모듈성은
        # 실제 에너지 지형과 무관하게 거의 항상 ~sqrt(N)개의 인위적인
        # 균등 크기 군집으로 쪼개지는 '해상도 한계(resolution limit)'를
        # 가진다.  그 결과 열 요동만 있는 완전히 질서 있는 단일 분지
        # 단백질도 여러 개의 가짜 분지로 쪼개져 IDP/무질서로 오분류되는
        # 체계적 편향이 있었다 (에너지 가중치를 실제로 전달하지도 않았음).
        #
        # 대신 실제 구조 공간(PCA 2D 투영, `layout`)에서 밀도 기반 군집화
        # (DBSCAN)를 수행한다: 같은 분지 내 열 요동은 조밀하게 연결되어
        # 하나의 군집을 이루고, 궤적 자체의 전형적 스텝 이동 거리보다
        # 뚜렷하게 큰 실제 구조 전이만 별도 군집(분지)으로 분리된다.
        #
        # The previous chain-graph + greedy-modularity approach suffered from
        # modularity's well-known resolution limit on path graphs: it split
        # almost any trajectory into ~sqrt(N) artefactual, near-equal-sized
        # "communities" regardless of the actual energy landscape (and never
        # even passed the |ΔE| edge weights to the algorithm) — systematically
        # misclassifying genuinely ordered, single-basin proteins as
        # multi-basin / disordered.
        #
        # Density-based clustering (DBSCAN) directly on the PCA structural
        # layout instead groups snapshots by structural proximity: thermal
        # jitter within one basin stays densely connected (one cluster),
        # while only a real conformational transition — a jump clearly
        # larger than the trajectory's own per-step noise — starts a new
        # cluster.  eps is calibrated from the trajectory's own median
        # consecutive-frame displacement, so it self-scales with however
        # much the MC step size/temperature actually moves the structure.
        # 분지 경계(서로 다른 시드에서 온 스냅샷 사이)는 실제 연속 스텝이
        # 아니므로 여기서 제외한다 — 안 그러면 다중 분지 탐색에서 시드가
        # 서로 멀리 떨어져 있을 때 가짜 "큰 이동"이 껴 eps 보정을 왜곡한다.
        # Exclude branch boundaries (between snapshots from different
        # seeds) -- they aren't real consecutive steps, and including them
        # would inject spurious "large jumps" into the eps calibration
        # whenever multi-branch seeds happen to be far apart structurally.
        diffs, offset = [], 0
        for length in branch_lengths:
            if length > 1:
                diffs.append(np.linalg.norm(
                    np.diff(layout[offset:offset + length], axis=0), axis=1))
            offset += length
        step_disp   = np.concatenate(diffs) if diffs else np.array([0.0])
        step_scale  = float(np.median(step_disp)) if step_disp.size else 1.0
        eps         = max(step_scale * 4.0, 1e-6)
        min_pts     = max(3, int(0.05 * N))
        labels      = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(layout)

        # 잡음점(label == -1, 전이 도중의 과도 상태)은 어떤 분지에도 속하지
        # 않는 것으로 취급한다.  군집이 하나도 없으면(예: N이 매우 작음)
        # 궤적 전체를 하나의 분지로 폴백한다.
        # Noise points (label == -1, transient in-between states) belong to
        # no basin.  Fall back to a single whole-trajectory basin if DBSCAN
        # finds none (e.g. very small N).
        communities = [np.flatnonzero(labels == lbl).tolist()
                       for lbl in sorted(set(labels)) if lbl != -1]
        if not communities:
            communities = [list(range(N))]
        node_comm = {node: ci
                     for ci, comm in enumerate(communities)
                     for node in comm}

        # ── IDP 분류 (IDP Classification) ────────────────────────────────
        # Intrinsically Disordered Protein (IDP) 분류 기준:
        #
        # kT = 0.592 kcal/mol @ 300K — 열 요동 에너지 스케일.
        # 유의미한 분지: 전체 스냅샷의 5% 이상을 차지하는 군집.
        #
        # "경쟁 분지" (Competitive basins): DBSCAN은 정상적인 열 요동만으로도
        # (곁사슬 회전 하나, 국소 접촉 재배열 하나) PCA 공간에서 새로운 밀집
        # 군집을 쉽게 만들어낸다 — 실제로 다른 접힘 상태가 아니어도. 5개의
        # 실제 안정 단백질(1LYZ, 1UBQ, 1MBN, 7RSA, 1BNI)로 보정해 본 결과,
        # 단순히 "유의미한 분지 수 ≥ 2"만으로 판정하면 5개 전부가
        # POSSIBLY DISORDERED로 잘못 분류됐다 — 심지어 최적 분지가 전체의
        # 66%를 차지한 미오글로빈(funnel=0.66)까지도. 실제로 경쟁하는 준안정
        # 상태라면 전역 최저 에너지에서 열적으로 실제 도달 가능한 범위 안에
        # 있어야 한다; 그보다 훨씬(수백~수만 kcal/mol) 높은 에너지의 분지는
        # 일시적인 뒤틀림/충돌 상태일 뿐 진짜 대체 접힘이 아니다 — 그런 상태를
        # 실제로 볼츠만 분포로 그 정도 비율(≥5%)만큼 방문했다는 것 자체가
        # 물리적으로 불가능하며, DBSCAN이 하나의 진짜 분지를 기하학적으로
        # 잘게 쪼갠 결과에 가깝다.
        #
        # 분류 기준:
        #   IDP:               ≥3개 "경쟁" 분지 AND e_spread < 5kT
        #                      (여러 분지가 비슷한 에너지 → 지형이 평탄 → 무질서)
        #   POSSIBLY DISORDERED: ≥2개 "경쟁" 분지 AND 최적 분지가 전체의 70% 미만
        #                        (진짜 열적으로 경쟁하는 대체 상태가 있고, 그것이
        #                        지배적이지 않을 때만 무질서 신호로 카운트)
        #   ORDERED:           그 외 (단일 지배적 분지, 또는 다른 분지가 있어도
        #                      에너지상 실제로 경쟁하지 않음)
        #
        # Competitive basins: DBSCAN readily carves a new dense cluster out of
        # perfectly normal thermal jitter (one side-chain flip, one local
        # contact rearrangement) even in a genuinely single-funnel landscape.
        # Calibrated against 5 real, textbook well-folded proteins (1LYZ,
        # 1UBQ, 1MBN, 7RSA, 1BNI): naively gating on "sig basins >= 2" alone
        # misclassified all 5 as POSSIBLY DISORDERED -- including myoglobin
        # with funnel=0.66 (two-thirds of the trajectory in one basin). A
        # real competing metastable state must have its own minimum within
        # thermal reach (generously, 20kT) of the global minimum; clusters
        # far above that are transient distortions/clashes encountered en
        # route, not real alternate folds -- visiting one at >=5% Boltzmann
        # weight would be physically impossible, so its presence just means
        # DBSCAN geometrically fragmented one real basin.
        #
        # IDP: >=3 competitive basins AND e_spread < 5kT (flat landscape)
        # POSSIBLY DISORDERED: >=2 competitive basins AND funnel < 0.7
        # ORDERED: otherwise (single dominant basin, or other basins present
        #          but not actually energetically competitive)
        kT  = 0.592          # kcal/mol at 300 K
        # 20kT를 76잔기(유비퀴틴) 기준으로 놓고 sqrt(n_res)로 스케일한다 — 에너지는
        # 원자 수에 비례해 늘어나지만, 정상적인 열 요동으로 인한 "퍼짐"은 서로
        # 약하게 상관된 자유도의 합이므로 sqrt(n_res)로 커진다(중심극한정리형
        # 논리). 고정 상수 하나로는 76잔기 유비퀴틴에서 맞던 컷오프가 140잔기
        # 알파-시누클레인(실제 IDP, 1XQ8)에서는 너무 빡빡해 진짜 경쟁 분지
        # (기저상태 대비 +13.8 kcal/mol, 인구 22%)까지 걸러내 버렸다.
        # Scale 20kT (calibrated against 76-residue ubiquitin) by sqrt(n_res) --
        # energy grows with atom count, but the *spread* from ordinary thermal
        # jitter is a sum over weakly-correlated degrees of freedom, so it grows
        # as sqrt(n_res) (central-limit-type argument), not linearly and not at
        # all. A single flat constant, tuned against 76-residue ubiquitin, was
        # too tight for the 140-residue real IDP case (1XQ8): it excluded a
        # genuinely real, well-populated (22%) alternate basin sitting only
        # 13.8 kcal/mol above the global minimum.
        n_res = max(1, len(self.ca_indices))
        COMPETITIVE_KT = 20 * (n_res / 76) ** 0.5
        SIG_FLOOR = 0.05    # "유의미한 분지"로 치는 최소 인구 비율 / min population fraction to count as a significant basin
        sig = [c for c in communities if len(c) >= SIG_FLOOR * N]

        # "지배적" 분지 = 유의미한 분지 중 가장 인구가 많은 것 (에너지가 가장
        # 낮은 단일 스냅샷이 속한 분지가 아니라). 안정 단백질에서는 보통 둘이
        # 같은 분지를 가리키지만(진짜 우물이 인구도 가장 많고 에너지도 가장
        # 낮다), 지배적 상태 자체가 없는 IDP에서는 "에너지 최솟값 하나"가
        # 우연히 방문한 희귀 요동일 뿐일 수 있다 — 인구를 기준으로 삼아야
        # 그런 경우에도 의미 있는 기준점이 된다.
        #
        # "Dominant" basin = the significant basin with the largest
        # population, not the one containing the single lowest-energy
        # snapshot. For a stable protein these usually coincide (the real
        # well is both the most populated and the lowest-energy one), but
        # for a protein with no dominant state (a real IDP), the single
        # lowest-energy point can be a rare fluctuation rather than
        # anything representative -- population is the more meaningful
        # anchor either way.
        best_comm = max(sig, key=len) if sig else max(communities, key=len)
        funnel    = len(best_comm) / N

        # 지배적 분지의 대표 구조 — 분류 단계의 구조적 변위 필터(아래)와
        # 이후 동적 후보 탐색 단계 둘 다에서 쓰므로 여기서 한 번만 계산한다.
        # Dominant basin's representative structure -- computed once here
        # since both the classification-stage structural filter (below) and
        # the later dynamic sub-candidate picking step need it.
        dom_idx = min(best_comm, key=lambda i: float(energies[i]))
        dominant_particles = snapshots[dom_idx]
        dom_vec = self._ca_vec(dominant_particles).reshape(-1, 3)

        # 경쟁 분지는 "지배적 분지"가 아니라 전역 최저 에너지 대비 열적으로
        # 도달 가능한지로 판정해야 한다. 지배적 분지 자체의 최솟값을 기준으로
        # 삼으면 비대칭 버그가 생긴다: 지배적 분지보다 에너지가 낮은 분지는
        # (얼마나 낮든) 항상 통과해 버려서 실제로는 서로 수백 kcal/mol 떨어진
        # 분지들까지 "경쟁"으로 잘못 집계됐다 (1UBQ 보정 실행에서 확인:
        # e_spread가 96~223으로 나왔는데, 정의상 20kT(~11.84) 이내여야 함).
        #
        # Competitive basins must be judged against the true global minimum,
        # not the dominant basin's own minimum. Using the dominant basin's min
        # as the reference is asymmetric: any basin with LOWER energy than the
        # dominant one always passes, no matter how much lower -- which let
        # basins hundreds of kcal/mol apart get counted as "competitive"
        # (confirmed via the 1UBQ calibration run: e_spread came out as
        # 96-223, when by definition it should be bounded by ~20kT ≈ 11.84).
        #
        # 전체 스냅샷(잡음점 포함)이 아니라 유의미한 분지(sig)들의 최솟값 중
        # 최소를 기준으로 삼는다 — 어느 분지에도 속하지 못한 잡음/과도 상태
        # 스냅샷 하나가 우연히 아주 낮은 에너지를 찍었다면, 그 하나 때문에
        # 기준점 자체가 비현실적으로 낮아져 진짜 분지들이 전부 "경쟁 불가"로
        # 밀려날 수 있다.
        # Anchor to the lowest minimum among significant basins only, not all
        # pooled snapshots (which include DBSCAN noise/transient points). A
        # single transient snapshot that never formed a persistent basin can
        # dip below any real basin's minimum by chance, which would drag the
        # reference point down and make every real basin look "too far" away.
        global_best_e = min(min(float(energies[i]) for i in c) for c in sig)

        # 에너지 갭 기준만으로는 놓치는 경우가 있다 — 인구가 많은 분지라도
        # 짧은 샘플링에서 우연히 최저점을 못 찍었을 수 있다(1XQ8 보정에서
        # 확인: 33% 인구 분지의 최솟값이 전역 최솟값보다 502 kcal/mol이나
        # 높았다). 그래서 인구 비율 기준을 OR로 추가한다: 지배적 분지 인구의
        # 25% 이상을 차지하면 그 자체로 "경쟁"으로 인정한다. kT 기반 자유
        # 에너지 공식(-kT·ln(비율))은 시도해봤지만 kT=0.592가 너무 작아
        # sig 필터(≥5% 인구)를 통과한 분지는 사실상 전부 자동으로 통과해
        # 버려 아무 것도 걸러내지 못했다 — 그래서 비율을 직접 비교한다.
        #
        # The energy-gap criterion alone misses real cases: a heavily-
        # populated basin can simply not have sampled its lowest point yet
        # under short branches (confirmed on 1XQ8: a 33%-populated basin's own
        # minimum sat 502 kcal/mol above the true global minimum). So OR in a
        # population-ratio criterion. (A kT-scaled free-energy formula,
        # -kT*ln(ratio), was tried first and rejected: kT=0.592 is so small
        # that any basin passing the 5%-of-N "sig" filter already trivially
        # satisfies it, making it a no-op filter.)
        #
        # A flat ratio of the dominant basin's raw population ("needs >=25% of
        # what the dominant has") was tried next and also rejected: it
        # degenerates whenever the dominant itself is only modestly ahead
        # (common in exactly the flat, borderline-disordered landscapes this
        # is meant to catch) -- e.g. dominant=17% * 25% = 4.25%, which sits
        # *below* the 5% SIG_FLOOR already required to be "significant" at
        # all, making the test a no-op in the other direction (everything that
        # survived `sig` passes automatically). Confirmed on real data: 1UBQ
        # and 1LYZ (both genuinely ordered) both flipped to POSSIBLY DISORDERED
        # with 8/9 and 4/5 basins respectively counted "competitive".
        #
        # 지배적 분지의 원래 인구 비율(예: 25%)로 고정하면 지배적 분지 자체의
        # 인구가 낮을 때(평탄하고 무질서에 가까운 지형에서 흔함) 문턱이
        # SIG_FLOOR보다도 낮아져 아무 것도 걸러내지 못한다. 대신 지배적 분지가
        # "잡음 문턱(SIG_FLOOR) 위로 얼마나 튀어나와 있는지" 그 초과분의
        # 비율로 판단한다 — 지배적 분지가 문턱에 가까울수록(평탄한 지형) 다른
        # 분지에 요구하는 인구도 함께 낮아지고, 지배적 분지가 뚜렷할수록
        # (뾰족한 지형) 요구치도 함께 높아진다.
        # Instead, scale relative to how far the dominant basin sits *above*
        # SIG_FLOOR (its "excess"), not its raw population -- this is what
        # actually adapts to the shape of the landscape: when the dominant is
        # only barely above the noise floor (a flat, disorder-leaning
        # landscape), the bar for a competitor drops right along with it; when
        # the dominant is sharply peaked, the bar rises correspondingly.
        POP_RATIO_THRESHOLD = 0.4
        pop_dominant   = len(best_comm) / N
        pop_threshold  = SIG_FLOOR + POP_RATIO_THRESHOLD * (pop_dominant - SIG_FLOOR)

        competitive = [c for c in sig
                       if c is not best_comm
                       and (min(float(energies[i]) for i in c) - global_best_e
                            < COMPETITIVE_KT * kT
                            or (len(c) / N) >= pop_threshold)]

        # 인구/에너지로 "경쟁"인 분지라도 실제 구조 변위가 작으면(곁사슬
        # 회전이체, 말단 곁가지의 정상적인 흔들림) 무질서가 아니다 — 이번
        # 세션에서 실측한 실제 사례들이 뚜렷하게 갈린다: 사소한 국소 유연성
        # (라이소자임 N-말단 약 0.9-2.6 A, 유비퀴틴 C-말단 꼬리 약 2.2-3.7 A)과
        # 진짜 대규모 무질서(1XQ8, 실제 IDP: 약 10.5-25.9 A) 사이에 뚜렷한
        # 간격이 있다. 라이소자임(이황화 결합 4개, RMSF~0%)이 반복 실행마다
        # 일관되게 POSSIBLY DISORDERED로 나온 원인이 바로 이것이었다 —
        # 문헌으로 확인한 결과 실제로 존재하는 N-말단 유연성이지만
        # ("N-terminus of HEWL is very flexible", HEWL 결정화 논문
        # PMC4498469), 단백질 전체가 무질서하다는 뜻은 아니다. 아래 필터는
        # 이미 sub-candidate 루프에서 쓰는 것과 동일한 Kabsch 정렬 + 최대
        # 변위 계산을 재사용한다(SS-diff는 제외 — 실측 데이터에서 사소한
        # 사례와 진짜 IDP 사례 둘 다 SS-diff가 작아 구분에 도움이 안 됐다).
        #
        # A basin can be statistically "competitive" (population/energy) yet
        # not be real disorder -- e.g. a sidechain rotamer flip or normal
        # terminal wobble. Real examples measured this session split
        # cleanly: minor local flexibility (1LYZ N-terminus ~0.9-2.6 A,
        # ubiquitin's real C-terminal tail ~2.2-3.7 A) vs. genuine large-scale
        # disorder (1XQ8, real IDP: ~10.5-25.9 A) -- a wide gap between them.
        # This is exactly why lysozyme (4 disulfides, ~0% RMSF) consistently
        # read POSSIBLY DISORDERED on repeat runs: real N-terminal
        # flexibility ("the N-terminus of HEWL is very flexible", confirmed
        # via the HEWL crystallization literature, PMC4498469), but not
        # evidence the whole protein is disordered. Reuses the same
        # Kabsch-align + max-displacement calculation already used in the
        # sub-candidate loop below (SS-diff omitted -- in the real examples
        # measured, SS-diff was small in *both* the minor and genuine-IDP
        # cases, so it doesn't discriminate here).
        STRUCTURAL_DISP_THRESHOLD = 5.0  # Å -- calibrated from the small set
        # of real cases above (~4 A ceiling on minor flexibility, ~10 A floor
        # on the one genuine-IDP case measured); may need revisiting with
        # more real test cases.
        competitive_structural = []
        for c in competitive:
            rep_idx = min(c, key=lambda i: float(energies[i]))
            if rep_idx == dom_idx:
                continue
            rep_vec = self._ca_vec(snapshots[rep_idx]).reshape(-1, 3)
            if rep_vec.shape != dom_vec.shape:
                continue
            rep_vec_aligned = _kabsch_align_points(dom_vec, rep_vec)
            disp = np.linalg.norm(rep_vec_aligned - dom_vec, axis=1)
            if float(disp.max()) >= STRUCTURAL_DISP_THRESHOLD:
                competitive_structural.append(c)

        e_spread = 0.0
        if len(competitive_structural) > 1:
            mins = [min(float(energies[i]) for i in c) for c in competitive_structural]
            e_spread = float(max(mins) - min(mins))

        if len(competitive_structural) >= 3 and e_spread < 5 * kT:
            idp_label, idp_color = "IDP", "#dc2626"
        elif len(competitive_structural) >= 2 and funnel < 0.7:
            idp_label, idp_color = "POSSIBLY DISORDERED", "#d97706"
        else:
            idp_label, idp_color = "ORDERED", "#16a34a"

        r_hat_str = f"{r_hat:.2f}" if r_hat is not None else "N/A"
        self.progress.emit(
            f"  [LANDSCAPE] Done · {len(sig)} significant basins "
            f"({len(competitive)} competitive, {len(competitive_structural)} structural) · "
            f"funnel={funnel:.2f} · R-hat={r_hat_str} · {idp_label}")

        # ── 동적 후보 탐색 (Dynamic sub-candidate picking) ────────────────
        # 경쟁 분지가 있다는 것은 어느 특정 구간(예: 유비퀴틴의 유연한
        # C-말단 꼬리)이 여러 형태를 취할 수 있다는 뜻이다. 그 구간을
        # 짚어내고, 이미 충분히 샘플링됐는지(재사용) 아니면 더 탐색해야
        # 하는지(전용 재탐색) — 혹은 둘 다인지(하이브리드) — 그 구간의
        # 인구 비율과 크기로부터 그때그때 판단한다. 정적 규칙 하나로
        # 고정하지 않는 이유: 이미 잘 샘플링된 분지를 다시 도는 것은
        # 낭비고, 거의 방문되지 않은 작은 구간은 전용 탐색이 싸고 유용하며,
        # 큰 구간은 전용 탐색이 비싸므로 기존 샘플에 의존하는 편이 낫다.
        #
        # Dynamic sub-candidate picking: a competitive basin means some
        # specific region (e.g. ubiquitin's flexible C-terminal tail) can
        # take on more than one shape. Pinpoint that region, then decide
        # per-region whether the general landscape run already sampled it
        # well enough (reuse), whether it's under-sampled and cheap to
        # refine (dedicated re-search), or both (hybrid) -- based on how
        # much of the trajectory already visited it and how large it is.
        # Not a fixed rule: re-running a well-sampled basin wastes compute;
        # a small, barely-visited region is cheap to refine directly; a
        # large region makes a dedicated re-run expensive, so lean on what
        # was already sampled instead.
        sub_candidates = []
        # dom_idx/dominant_particles/dom_vec already computed above, before
        # classification.

        # ── 2차 구조 근사 (Secondary-structure abstraction) ───────────────
        # 원자 좌표 대신 잔기별 나선/가닥/코일 범주로 분지를 비교한다 —
        # 어떤 IDP 구간도 항상 라벨을 가지므로(외부 안정 구조 라이브러리와
        # 맞출 필요가 없다), 순수 변위만으로는 못 잡는 "같은 위치가 나선에서
        # 가닥으로 바뀜" 같은 범주적 차이까지 구간 표시에 반영한다.
        # Compare basins by per-residue helix/strand/coil category instead
        # of raw atom displacement -- every region always gets a label (no
        # matching against an external stable-structure library needed),
        # catching categorical changes (e.g. helix -> strand at the same
        # position) that pure displacement alone would miss.
        ss_dominant = None
        if self.backbone_ncac is not None:
            n_idx, ca_idx, c_idx = self.backbone_ncac
            ss_dominant = _secondary_structure_string(dominant_particles, n_idx, ca_idx, c_idx)

        if len(competitive) >= 2:
            for c in competitive:
                rep_idx = min(c, key=lambda i: float(energies[i]))
                if rep_idx == dom_idx:
                    continue
                rep_vec = self._ca_vec(snapshots[rep_idx]).reshape(-1, 3)
                if rep_vec.shape != dom_vec.shape:
                    continue
                # 원시 좌표를 그대로 비교하면 실제 형태 차이와 "형태는 같은데
                # 절대 위치/방향만 다름"이 뒤섞인다 — 다중 분지 탐색에서는
                # 서로 다른 시드에서 뻗어나가므로 이 문제가 특히 크다(레이어드
                # 3D 뷰에서 이미 한 번 고친 것과 같은 종류의 버그). 비교 전에
                # Kabsch 정렬로 절대 자세 차이를 제거한다.
                # Comparing raw coordinates conflates real shape differences
                # with "same shape, different absolute pose" -- especially
                # bad with multi-branch exploration, since branches start
                # from different seeds (the same class of bug already fixed
                # once for the external layered 3D view). Kabsch-align first
                # to remove pose differences before measuring shape change.
                rep_vec = _kabsch_align_points(dom_vec, rep_vec)
                disp = np.linalg.norm(rep_vec - dom_vec, axis=1)
                # 절대 2A 문턱 대신, 이 분지 자체의 최대 변위에 상대적인 문턱을
                # 쓴다 — PCA/DBSCAN은 잔기 전체에 걸쳐 흩어진 작은 변화(각
                # 잔기는 2A를 넘지 않음)만으로도 두 구조를 구분해낼 수 있으므로,
                # 절대 문턱은 실제 "경쟁 분지"를 자주 놓친다(실측: 유비퀴틴
                # 재실행에서 2개 경쟁 분지가 있었는데도 0개 구간이 잡힘). 이
                # 분지의 최대 변위의 절반(최소 1A)을 문턱으로 삼으면 그 분지를
                # 구분 짓는 실제 원인 잔기를 항상 잡아내되, 변위가 전부 너무
                # 작을 때는(모두 <1A) 여전히 아무 구간도 표시하지 않는다.
                #
                # Use a threshold relative to this basin's own peak
                # displacement instead of a fixed 2A cutoff -- PCA/DBSCAN can
                # tell two structures apart from small changes spread across
                # many residues (none individually crossing 2A), so a fixed
                # absolute cutoff often misses real competitive basins
                # (observed: a ubiquitin re-run found 2 competitive basins
                # but flagged 0 regions). Half of this basin's own peak
                # displacement (floor 1A) always catches whatever residues
                # actually drive the difference, while basins where nothing
                # moves more than 1A still correctly flag nothing.
                threshold = max(1.0, 0.5 * float(disp.max()))
                flagged_set = set(np.flatnonzero(disp > threshold).tolist())

                ss_diff_frac = 0.0
                if ss_dominant is not None:
                    ss_rep = _secondary_structure_string(
                        snapshots[rep_idx], *self.backbone_ncac)
                    ss_diff_frac = _ss_diff_fraction(ss_dominant, ss_rep)
                    flagged_set |= {i for i in range(min(len(ss_dominant), len(ss_rep)))
                                    if ss_dominant[i] != ss_rep[i]}

                if not flagged_set:
                    continue
                region = (min(flagged_set), max(flagged_set))
                region_size = region[1] - region[0] + 1
                pop_frac = len(c) / N

                strategy = self._pick_strategy(pop_frac, region_size)
                self.progress.emit(
                    f"  [LANDSCAPE] Competitive basin: {pop_frac*100:.0f}% pop, "
                    f"residues {region[0]+1}-{region[1]+1} differ up to "
                    f"{disp.max():.1f} A (SS diff {ss_diff_frac*100:.0f}%) -> {strategy}")

                candidates = []
                if strategy in ("reuse", "hybrid"):
                    candidates += [
                        {"particles": snapshots[i], "energy": float(energies[i]),
                         "source": "landscape"} for i in c]
                if strategy in ("dedicated", "hybrid"):
                    candidates += self._dedicated_subsearch(snapshots[rep_idx])

                # Keep the best (lowest-energy) candidates first for the UI picker.
                candidates.sort(key=lambda d: d["energy"])
                sub_candidates.append({
                    "region": region, "population": float(pop_frac),
                    "strategy": strategy, "energy": float(energies[rep_idx]),
                    "ss_diff": float(ss_diff_frac),
                    "candidates": candidates,
                })

        # ── 구조 풀 (Structural pool) ──────────────────────────────────────
        # 지배적 상태가 없는 IDP라도, 그저 "무질서"라는 라벨 하나로 끝내지
        # 않고 실제로 자주 나타나는(=인구 비율이 높은) 구조들의 순위 목록을
        # 만든다 — 서브 구간과 별개로, 유의미한 분지 전부를 인구순으로.
        # Even without a dominant state, an IDP still has a pool of
        # frequently-occurring structures -- rank every significant basin
        # by population (independent of the sub-region picking above),
        # rather than collapsing everything down to just a disorder label.
        basin_summary = []
        for c in sig:
            rep_idx = min(c, key=lambda i: float(energies[i]))
            e_vals = [float(energies[i]) for i in c]
            basin_summary.append({
                "population": len(c) / N,
                "energy_min": min(e_vals), "energy_max": max(e_vals),
                "particles": snapshots[rep_idx],
                "is_dominant": c is best_comm,
                # Pooled-snapshot indices belonging to this basin (2026-07-13,
                # IMPROVEMENTS.md item #7 re-test) -- lets the ensemble overlay
                # show only this basin's conformers instead of always the whole
                # pooled trajectory. Stored here (not reconstructed from the
                # raw `communities` list) because basin_summary is built from
                # `sig` and then population-sorted, so its index order doesn't
                # match `communities`' original order.
                "member_indices": list(c),
            })
        basin_summary.sort(key=lambda b: -b["population"])

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
            "r_hat":        r_hat,
            "idp_label":    idp_label,
            "idp_color":    idp_color,
            "dominant_particles": dominant_particles,
            "sub_candidates": sub_candidates,
            "basin_summary": basin_summary,
        })

    def _pick_strategy(self, pop_frac, region_size):
        """Decide reuse vs. dedicated vs. hybrid for one competitive basin.

        pop_frac: fraction of the N landscape snapshots already in this
        basin (how well the general run already sampled it).
        region_size: number of residues flagged as differing from the
        dominant basin (a proxy for how expensive a focused re-run is).
        """
        if pop_frac >= 0.15:
            return "reuse"          # already decently sampled, don't re-spend compute
        if region_size <= 8:
            return "dedicated" if pop_frac < 0.08 else "hybrid"
        return "hybrid" if pop_frac >= 0.08 else "reuse"

    def _dedicated_subsearch(self, seed_particles):
        """Focused re-exploration seeded at a competitive basin's
        representative conformation, run at a smaller angle/temperature
        budget so it refines around that basin instead of wandering back to
        the dominant one or elsewhere.

        This is not a literal bond-restricted MC -- the C++ engine doesn't
        expose a per-bond move mask, so building one would mean
        re-implementing its adjacency/downstream-atom traversal in Python.
        Seeding from the basin's own representative plus a tighter
        temperature/max-angle achieves the same practical goal (denser
        sampling of that specific alternate conformation) without new
        native-extension work.
        """
        N_sub, S_sub = 40, 40
        T_sub, angle_sub = self.T * 0.5, self.max_angle * 0.5
        try:
            snaps, energies = self.engine.run_landscape_trajectory(
                seed_particles, self.topo, N_sub, S_sub, T_sub, angle_sub)
        except Exception as ex:
            self.progress.emit(f"  [LANDSCAPE] Dedicated sub-search failed: {ex}")
            return []
        return [{"particles": s, "energy": float(e), "source": "dedicated"}
                for s, e in zip(snaps, energies)]

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

        # GPU 자동 선택 (2026-07-09): 이전에는 매 세션 "GPU 사용?" 모달을 눌러야
        # GPU를 썼다. GPU 백엔드는 이미 CPU 대비 오차 0.06% 이내로 검증됐고
        # (완료 작업 → 성능 참고), 실행시간 실패도 이미 CPU로 자동 폴백된다
        # (gpu_fallback 시그널) — GPU가 있으면 조건 없이 이득만 있는 상황이라
        # 매번 클릭을 요구할 이유가 없다. GPU 감지 시 자동 사용, 실패하면
        # 조용히 CPU로 남는다.
        #
        # GPU auto-selection (2026-07-09): previously required clicking "Yes"
        # on a modal every session to use GPU. The GPU backend is already
        # validated to within 0.06% of CPU (see "Completed Work" -> Performance)
        # and runtime failures already fall back to CPU automatically (the
        # gpu_fallback signal) -- with no accuracy tradeoff and an existing
        # safety net, there's no reason to gate a pure speed win behind a
        # click every session. Auto-select GPU when detected; silently stays
        # on CPU if detection fails.
        self._physics_mod = protein_physics
        cuda_mod, gpu_name = _try_gpu_backend()
        if cuda_mod is not None:
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
        self._topo               = None
        self._pdb_path           = None
        self._ca_indices         = []
        self._landscape_snaps    = []
        self._landscape_energies = np.array([])
        self._sub_candidates     = []
        self._basin_summary      = []
        self._landscape_dominant_particles = None
        self._active_subcand     = None   # (region_idx, cand_idx) currently rendered
        self._rmsf               = None
        self._rmsf_residues      = []
        self._rmsf_n_disordered  = 0
        self._rmsf_pct           = 0.0
        self._iupred_scores      = []
        self._ca_residues        = []
        self._knot_result        = None
        self._ref_ca_map         = {}
        self._current_ext_entry  = None
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
        self.backend_lbl = QLabel(f"⚙  {self._backend}")
        self.backend_lbl.setStyleSheet("color:#94a3b8;font-size:9px;letter-spacing:1px;padding:0 0 8px 8px;")
        sidebar.addWidget(title); sidebar.addWidget(sub)
        sidebar.addWidget(self.backend_lbl); sidebar.addWidget(_sep())

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
        self.ensemble_toggle_btn = QPushButton("⊚  ENSEMBLE")
        self.ensemble_toggle_btn.setObjectName("sec-btn")
        self.ensemble_toggle_btn.clicked.connect(self._toggle_ensemble)
        self.ensemble_toggle_btn.setFixedHeight(24)
        self.ensemble_toggle_btn.setEnabled(False)
        vh_layout.addWidget(self.ensemble_toggle_btn)
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

        # Page 1: energy landscape -- split view: dominant/selected structure
        # (left) alongside the graph + basin panels (right), instead of
        # replacing the 3D view entirely. Landing on the landscape page used
        # to mean losing the structure view outright, and clicking any node/
        # basin/sub-candidate yanked you straight back out to the full
        # structure page just to see it -- bad navigation UX for something
        # you're meant to click through repeatedly while exploring.
        landscape_page = QWidget()
        lp_outer = QHBoxLayout(landscape_page)
        lp_outer.setContentsMargins(0, 0, 0, 0); lp_outer.setSpacing(0)

        landscape_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.landscape_web = QWebEngineView()
        self.landscape_web.setStyleSheet("border:none;")
        self.landscape_web.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self._landscape_html_tmpfile = os.path.join(
            tempfile.gettempdir(), "alma_landscape_viewer.html")
        landscape_splitter.addWidget(self.landscape_web)

        landscape_graph_col = QWidget()
        lp_v = QVBoxLayout(landscape_graph_col)
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

        # Sub-region bar — populated when the landscape run finds a
        # competitive alternate basin (see LandscapeWorker._pick_strategy).
        # One row per flagged region, each with a picker for its
        # sub-candidates (reused/dedicated/hybrid depending on how that
        # basin was explored).
        self._subregion_panel = QWidget()
        self._subregion_panel.setStyleSheet(
            "background:#fffbeb;border-bottom:1px solid #fde68a;")
        self._subregion_v = QVBoxLayout(self._subregion_panel)
        self._subregion_v.setContentsMargins(16, 4, 16, 4)
        self._subregion_v.setSpacing(3)
        self._subregion_panel.setVisible(False)
        lp_v.addWidget(self._subregion_panel)
        self._subregion_rows = []

        # Structural pool bar — population-ranked list of every significant
        # basin found, independent of the sub-region picker above. Exists
        # even when there's no single dominant state (e.g. a real IDP):
        # rather than collapsing everything to one disorder label, this
        # shows what actually recurs and how often.
        self._basinpool_panel = QWidget()
        self._basinpool_panel.setStyleSheet(
            "background:#eff6ff;border-bottom:1px solid #bfdbfe;")
        self._basinpool_v = QVBoxLayout(self._basinpool_panel)
        self._basinpool_v.setContentsMargins(16, 4, 16, 4)
        self._basinpool_v.setSpacing(3)
        self._basinpool_panel.setVisible(False)
        lp_v.addWidget(self._basinpool_panel)
        self._basinpool_rows = []

        # Matplotlib canvas
        self._landscape_fig = Figure(facecolor="#f8fafc", tight_layout=True)
        self._landscape_canvas = FigureCanvas(self._landscape_fig)
        self._landscape_canvas.setStyleSheet("border:none;")
        self._landscape_fig.canvas.mpl_connect("pick_event", self._on_graph_pick)
        lp_v.addWidget(self._landscape_canvas)

        landscape_splitter.addWidget(landscape_graph_col)
        landscape_splitter.setStretchFactor(0, 2)
        landscape_splitter.setStretchFactor(1, 3)
        # setStretchFactor only governs how *extra* space is distributed on
        # resize, not the initial split -- a bare QSplitter can collapse the
        # 3D viewer to 0 width on first show, since QWebEngineView reports a
        # tiny/zero size hint (confirmed: landscape_web measured 0x838 right
        # after the page was first shown). A minimum width plus an explicit
        # initial setSizes() keeps it visible from the start.
        self.landscape_web.setMinimumWidth(300)
        landscape_splitter.setSizes([600, 900])
        lp_outer.addWidget(landscape_splitter)

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

        # Page 3: IDP ensemble metrics (Rg, end-to-end, internal scaling
        # law, contact-map) -- see IMPROVEMENTS.md item #7. Deliberately a
        # separate page from the disorder/RMSF profile above rather than
        # folded into it: these are whole-ensemble/pairwise descriptors
        # (Rg is one value per snapshot, contact frequency is per residue
        # PAIR), a different data shape from the disorder page's per-
        # residue-only line plots.
        ensemble_page = QWidget()
        ep_v = QVBoxLayout(ensemble_page)
        ep_v.setContentsMargins(0, 0, 0, 0); ep_v.setSpacing(0)

        ep_hdr = QWidget(); ep_hdr.setFixedHeight(30)
        ep_hdr.setStyleSheet("background:#ffffff;border-bottom:1px solid #e2e8f0;")
        ep_h = QHBoxLayout(ep_hdr); ep_h.setContentsMargins(16, 0, 16, 0)
        ep_title = QLabel("IDP ENSEMBLE METRICS")
        ep_title.setStyleSheet("color:#64748b;font-size:10px;letter-spacing:2px;")
        ep_h.addWidget(ep_title); ep_h.addStretch()
        self.ensemble_stats_lbl = QLabel("—")
        self.ensemble_stats_lbl.setStyleSheet("color:#94a3b8;font-size:9px;letter-spacing:1px;")
        ep_h.addWidget(self.ensemble_stats_lbl)
        ep_h.addSpacing(16)
        self.ensemble_overlay_btn = QPushButton("SHOW ENSEMBLE OVERLAY")
        self.ensemble_overlay_btn.setObjectName("sec-btn")
        self.ensemble_overlay_btn.setFixedHeight(22)
        self.ensemble_overlay_btn.setStyleSheet(
            "background:transparent;color:#7c3aed;border:1.5px solid #7c3aed;"
            "border-radius:4px;padding:2px 10px;font-size:9px;letter-spacing:1px;")
        self.ensemble_overlay_btn.clicked.connect(self._render_ensemble_overlay)
        ep_h.addWidget(self.ensemble_overlay_btn)
        ep_v.addWidget(ep_hdr)

        self._ensemble_fig    = Figure(facecolor="#f8fafc", tight_layout=True)
        self._ensemble_canvas = FigureCanvas(self._ensemble_fig)
        self._ensemble_canvas.setStyleSheet("border:none;")
        ep_v.addWidget(self._ensemble_canvas)

        self._view_stack.addWidget(ensemble_page)    # index 3
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
        # ── 열 구성 (P10: Heavy RMSD 열 추가) ───────────────────────────────
        # 0=SOURCE 1=ENERGY 2=Cα RMSD 3=Heavy RMSD(사이드체인 반영) 4=pLDDT 5=VIEW버튼
        # Column layout (P10 adds Heavy RMSD): 0=SOURCE 1=ENERGY 2=Cα RMSD
        # 3=Heavy RMSD (sidechain-aware) 4=pLDDT 5=VIEW button.
        self.comp_table_layout.setColumnStretch(0, 3)
        self.comp_table_layout.setColumnStretch(1, 2)
        self.comp_table_layout.setColumnStretch(2, 2)
        self.comp_table_layout.setColumnStretch(3, 2)
        self.comp_table_layout.setColumnStretch(4, 1)
        self.comp_table_layout.setColumnStretch(5, 1)
        for col, text in enumerate(
                ["SOURCE", "ENERGY (kcal/mol)", "RMSD vs REF", "HEAVY RMSD", "pLDDT", ""]):
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
        self.ensemble_toggle_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_lbl.setText("RUNNING")
        self.status_lbl.setStyleSheet("color:#d97706;font-size:11px;font-weight:bold;")
        self.log.clear()
        self.comp_panel.setVisible(False)

        # Reset all secondary panels
        self.idp_status_lbl.setText("—")
        self.idp_status_lbl.setStyleSheet("font-size:10px;font-weight:bold;color:#94a3b8;")
        self.disorder_stats_lbl.setText("—")
        self.ensemble_stats_lbl.setText("—")
        self._rmsf = None

        # Reset view to structure page and fix button labels
        self._view_stack.setCurrentIndex(0)
        self.view_mode_btn.setVisible(True)
        self.view_mode_btn.setText(
            "⊞  LAYERED" if self._view_mode == "sidebyside" else "◧  SIDE-BY-SIDE")
        self.landscape_toggle_btn.setText("◈  LANDSCAPE")
        self.disorder_toggle_btn.setText("⊛  DISORDER")
        self.ensemble_toggle_btn.setText("⊚  ENSEMBLE")

        self.landscape_start_btn.setText("◈  EXPLORE LANDSCAPE")
        self._log(f"[{target}] Analysis initiated")

        # n_cand 상향 (2026-07-09): 기본 5개로는 GPU에서 늘어난 분지 수(아래
        # _start_landscape의 GPU_BRANCH_COUNT=6 참고)를 뒷받침할 후보가 부족하다.
        # 후보 1개당 300스텝(steps 기본값)뿐이라 몇 개 늘려도 초기 파이프라인
        # 실행 비용은 미미하다 — CPU/GPU 모두에 적용해도 무방.
        #
        # Bump n_cand (2026-07-09): the default 5 isn't enough to back the
        # higher GPU branch count below (GPU_BRANCH_COUNT=6 in
        # _start_landscape). Each extra candidate only costs 300 steps
        # (the default `steps` arg), so raising this is cheap regardless of
        # CPU/GPU -- applied unconditionally rather than gating it, too.
        self.worker = PipelineWorker(self.engine, target, self._physics_mod, n_cand=8)
        self.worker.progress.connect(self._log)
        self.worker.metrics.connect(self._on_metrics)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.gpu_fallback.connect(self._on_gpu_fallback)
        self.worker.start()

    def _on_gpu_fallback(self, reason: str):
        """The GPU engine failed at runtime (not just at startup detection) and
        the worker already recovered by finishing its current run on the CPU.
        Downgrade permanently for the rest of the session so later analyses
        don't repeat the same failure and its multi-second retry delay."""
        if self._physics_mod is protein_physics:
            return
        self._log(f"  GPU engine disabled for the rest of this session after a runtime "
                   f"failure ({reason}). Restart the app to retry the GPU backend.")
        self._physics_mod = protein_physics
        try:
            self.engine = protein_physics.PhysicsEngine()
        except Exception:
            pass
        self._backend = "CPU (GPU disabled after runtime failure)"
        self.backend_lbl.setText(f"⚙  {self._backend}")

    def _on_metrics(self, d):
        if "n_atoms"  in d: self._mv_atoms.setText(str(d["n_atoms"]))
        if "threads"  in d: self._mv_threads.setText(str(d["threads"]))
        if "best_e"   in d: self._mv_energy.setText(f"{d['best_e']:.0f}")
        if "n_cand"   in d: self._mv_cand.setText(str(d["n_cand"]))

    def _on_done(self, ensemble, energies, pdb_path, ca_indices, ca_map,
                 init_atoms, topo, extra):
        self._ensemble    = ensemble
        self._energies    = energies
        self._init_atoms  = init_atoms
        self._topo        = topo
        self._ca_indices  = ca_indices
        self._pdb_path    = pdb_path
        # 참조 구조 Cα 맵 저장 — AlphaFold/SWISS-MODEL을 레이어드 뷰로 겹칠 때
        # 참조 프레임에 정렬(Kabsch)하는 데 사용 (RMSD 계산과 동일한 기준).
        # Store the reference Cα map — used to Kabsch-align AlphaFold/SWISS-MODEL
        # structures onto the reference frame for the layered view (same
        # reference used for the RMSD column).
        self._ref_ca_map  = ca_map
        # ── IUPred 점수 저장 및 무질서 패널 즉시 활성화 ──────────────────────
        # IUPred 예측은 순수 서열 기반이므로 MC landscape 실행 전에도 바로 표시 가능.
        # 사용자가 무질서 버튼을 누르면 IUPred 단독 패널(단일 플롯)을 먼저 보게 되고,
        # 이후 landscape 완료 시 RMSF와 나란히 보여지는 이중 패널로 업데이트된다.
        # IUPred is sequence-only so the disorder panel can be enabled immediately.
        # Before landscape: single IUPred panel.
        # After landscape: dual-panel (IUPred top + RMSF bottom).
        self._iupred_scores  = extra.get("iupred_scores", [])
        self._ca_residues    = extra.get("ca_residues", [])
        self._knot_result    = extra.get("knot_result")
        # 전체 중원자 RMSD용 데이터 (P10) — ComparisonWorker에 그대로 전달.
        # All-heavy-atom RMSD data (P10) — forwarded as-is to ComparisonWorker.
        self._heavy_map      = extra.get("heavy_map", {})
        self._heavy_indices  = extra.get("heavy_indices", [])
        if self._iupred_scores:
            self._draw_disorder_profile(residues=self._ca_residues,
                                        iupred_scores=self._iupred_scores)
            self.disorder_toggle_btn.setEnabled(True)
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
            ensemble, energies, self.engine, self._physics_mod,
            heavy_indices=self._heavy_indices, ref_heavy_map=self._heavy_map)
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

    def _set_landscape_html(self, html):
        """Same as _set_html but targets the landscape page's own embedded
        3D viewer (separate temp file/widget) -- lets basin/sub-candidate/
        node clicks update the structure in place on the landscape page
        instead of forcing a jump back to the main structure page."""
        with open(self._landscape_html_tmpfile, "w", encoding="utf-8") as f:
            f.write(html)
        self.landscape_web.setUrl(QUrl.fromLocalFile(self._landscape_html_tmpfile))

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

            # Heavy RMSD (P10) — 사이드체인까지 포함한 전체 중원자 Kabsch RMSD.
            # 값이 없으면(구조 파싱 실패 등) 기존 RMSD 열과 동일하게 "—" 표시.
            # Heavy RMSD (P10) — all-heavy-atom Kabsch RMSD including sidechains.
            # Falls back to "—" like the Cα RMSD column when unavailable.
            rmsd_heavy = entry.get("rmsd_heavy")
            rh_lbl = QLabel(f"{rmsd_heavy:.2f} Å" if rmsd_heavy is not None else "—")
            rh_lbl.setStyleSheet("font-size:10px;color:#1e293b;")
            self.comp_table_layout.addWidget(rh_lbl, row_i, 3)

            plddt  = entry.get("plddt")
            p_lbl  = QLabel(f"{plddt:.1f}" if plddt is not None else "—")
            if plddt is not None:
                pcol = "#16a34a" if plddt >= 70 else ("#d97706" if plddt >= 50 else "#dc2626")
                p_lbl.setStyleSheet(f"font-size:10px;color:{pcol};font-weight:bold;")
            else:
                p_lbl.setStyleSheet("font-size:10px;color:#94a3b8;")
            self.comp_table_layout.addWidget(p_lbl, row_i, 4)

            view_btn = QPushButton("VIEW")
            view_btn.setFixedSize(48, 20)
            view_btn.setStyleSheet(
                "background:transparent;color:#1d4ed8;border:1px solid #1d4ed8;"
                "border-radius:3px;font-size:9px;padding:0;letter-spacing:1px;")
            en = entry.copy()
            view_btn.clicked.connect(lambda _, e=en: self._render_source(e))
            self.comp_table_layout.addWidget(view_btn, row_i, 5)

    def _render_source(self, entry):
        if self._view_stack.currentIndex() != 0:
            self._view_stack.setCurrentIndex(0)
            self.view_mode_btn.setVisible(True)
            self.landscape_toggle_btn.setText("◈  LANDSCAPE")
        if entry.get("is_mc"):
            self._current_ext_entry = None
            self._render(entry["mc_idx"])
        else:
            path = entry.get("path")
            if path and os.path.exists(path):
                self._render_external(entry)

    def _render_external(self, entry):
        """AlphaFold/SWISS-MODEL VIEW 버튼 디스패처 — 현재 뷰 모드(레이어드/
        사이드바이사이드)에 따라 최적 MC 후보와 함께 렌더링한다.
        VIEW-button dispatcher for AlphaFold/SWISS-MODEL entries — renders
        alongside the best MC candidate, honoring the current view mode
        (layered/side-by-side) toggle just like the MC candidate view does.
        """
        self._current_ext_entry = entry
        try:
            if self._view_mode == "sidebyside":
                self._render_external_sidebyside(entry)
            else:
                self._render_external_layered(entry)
        except Exception as ex:
            self._log(f"[RENDER ERROR] {ex}\n{traceback.format_exc()}")

    def _render_external_layered(self, entry):
        path, source_name = entry.get("path"), entry["source"]
        is_af     = "AlphaFold" in source_name
        ext_color = "#7c3aed" if is_af else "#d97706"
        label_ext = source_name.upper()

        # 참조 구조 프레임에 Kabsch 정렬 — 그래야 겹쳐 봤을 때 실제로 겹쳐 보인다.
        # Kabsch-align onto the reference frame first, otherwise "layered" would
        # just show two structures floating at unrelated positions/orientations.
        if self._ref_ca_map:
            pdb_ext = _aligned_pdb_text(path, self._ref_ca_map)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                pdb_ext = f.read()
        pdb_esc = pdb_ext.replace("\\", "\\\\").replace("`", "\\`")
        n_atoms = pdb_ext.count("\nATOM")

        best_js, best_e, best_idx = "", "", None
        if self._ensemble and self._energies:
            best_idx = int(np.argmin(self._energies))
            # 후보를 참조 프레임에 Kabsch 정렬 — 안 그러면 RMSD 표는 작게 나와도
            # (자체적으로 정렬 후 계산되므로) 실제 렌더링은 MC의 원본(비정렬)
            # 좌표계에 남아, 정렬된 참조 구조와 전혀 다른 위치/방향에 떠 보인다.
            # Kabsch-align the candidate onto the reference frame first — without
            # this, the RMSD table (which aligns internally before scoring) can
            # read as tiny while the rendered candidate — still in its raw,
            # unaligned MC coordinates — floats at a completely different
            # position/orientation than the aligned reference structure.
            transform = None
            if self._ref_ca_map:
                cand_ca_map = _ensemble_ca_map(
                    self._ref_ca_map, self._ca_indices, self._ensemble[best_idx])
                if cand_ca_map:
                    transform = _kabsch_fit(self._ref_ca_map, cand_ca_map)
            pdb_best = self._build_pdb_str(self._ensemble[best_idx], transform)
            best_e   = f"{self._energies[best_idx]:.1f} kcal/mol"
            best_js  = (
                f'  var mBest=v.addModel(`{pdb_best}`,"pdb");\n'
                f'  mBest.setStyle({{}},{{cartoon:{{color:"#1d4ed8",thickness:0.8,opacity:1.0}},'
                f'sphere:{{color:"#1d4ed8",radius:0.55,opacity:1.0}}}});')

        ext_js = (
            f'  var mExt=v.addModel(`{pdb_esc}`,"pdb");\n'
            f'  mExt.setStyle({{}},{{cartoon:{{color:"{ext_color}",thickness:0.6,opacity:0.65}},'
            f'sphere:{{color:"{ext_color}",radius:0.50,opacity:0.60}}}});')

        if best_js:
            label  = (f"LAYERED &nbsp; {label_ext} &nbsp;over&nbsp; "
                      f"C{best_idx+1} BEST ({best_e}) &nbsp;&middot;&nbsp; {n_atoms} ATOMS")
            legend = ('<div id="legend">OVERLAY &nbsp; '
                      '<span style="color:#1d4ed8">&#9632;</span> BEST CANDIDATE &nbsp;'
                      f'<span style="color:{ext_color}">&#9632;</span> {label_ext}</div>')
        else:
            label  = f"{label_ext} &nbsp;&middot;&nbsp; {n_atoms} ATOMS"
            legend = (f'<div id="legend">'
                      f'<span style="color:{ext_color}">&#9632;</span> {label_ext}</div>')

        html = f"""<!DOCTYPE html><html><head>
<script src="{_3DMOL_JS_URL}"></script>
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
<div id="info">{label}</div>
{legend}
<script>
(function(){{
  var v=$3Dmol.createViewer("v",{{backgroundColor:"#f8fafc"}});
{best_js}
{ext_js}
  v.zoomTo(); v.zoom(0.85); v.render();
  (function spin(){{ v.rotate(1,'y'); v.render(); requestAnimationFrame(spin); }})();
}})();
</script></body></html>"""
        self.viewer_cand_lbl.setText(
            f'<span style="color:{ext_color};font-weight:bold;">{label_ext}</span> (layered)')
        self._log(f"[EXT-LAYERED] html={len(html.encode())} bytes")
        self._set_html(html)

    def _render_external_sidebyside(self, entry):
        path, source_name = entry.get("path"), entry["source"]
        is_af     = "AlphaFold" in source_name
        ext_color = "#7c3aed" if is_af else "#d97706"
        label_ext = source_name.upper()

        # _render_external_layered already filters nucleic acid/water
        # residues and Kabsch-aligns onto the reference frame (see
        # _aligned_pdb_text) -- this side-by-side view read the raw file
        # directly instead, so a co-crystallized DNA/RNA template (e.g. a
        # protein-DNA complex used as a SWISS-MODEL homology source) showed
        # up tangled through the protein here even though the layered view
        # and the energy/RMSD numbers already excluded it correctly.
        if self._ref_ca_map:
            pdb_ext = _aligned_pdb_text(path, self._ref_ca_map)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                pdb_ext = f.read()
        pdb_esc = pdb_ext.replace("\\", "\\\\").replace("`", "\\`")
        n_atoms = pdb_ext.count("\nATOM")

        left_js = (
            f'  var mL=vL.addModel(`{pdb_esc}`,"pdb");\n'
            f'  mL.setStyle({{}},{{cartoon:{{color:"{ext_color}",thickness:0.8,opacity:1.0}},'
            f'sphere:{{color:"{ext_color}",radius:0.55,opacity:1.0}}}});')
        left_info   = f"{label_ext} &nbsp;&middot;&nbsp; {n_atoms} ATOMS"
        left_legend = f'<span style="color:{ext_color}">&#9632;</span> {label_ext}'

        right_js, right_info, right_legend = "", "NO MC ENSEMBLE", ""
        if self._ensemble and self._energies:
            best_idx = int(np.argmin(self._energies))
            pdb_best = self._build_pdb_str(self._ensemble[best_idx])
            best_e   = f"{self._energies[best_idx]:.1f} kcal/mol"
            right_js = (
                f'  var mR=vR.addModel(`{pdb_best}`,"pdb");\n'
                f'  mR.setStyle({{}},{{cartoon:{{color:"#1d4ed8",thickness:0.8,opacity:1.0}},'
                f'sphere:{{color:"#1d4ed8",radius:0.55,opacity:1.0}}}});')
            right_info   = f"&#9733; C{best_idx+1} (BEST) &nbsp;&middot;&nbsp; {best_e}"
            right_legend = '<span style="color:#1d4ed8">&#9632;</span> BEST CANDIDATE'

        html = f"""<!DOCTYPE html><html><head>
<script src="{_3DMOL_JS_URL}"></script>
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
  <div class="info">{left_info}</div>
  <div class="pane-lbl">{label_ext}</div>
  <div class="legend">{left_legend}</div>
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
  (function spinL(){{ vL.rotate(1,'y'); vL.render(); requestAnimationFrame(spinL); }})();
  var vR=$3Dmol.createViewer("vR",{{backgroundColor:"#f8fafc"}});
{right_js}
  vR.zoomTo(); vR.zoom(0.85); vR.render();
  (function spinR(){{ vR.rotate(1,'y'); vR.render(); requestAnimationFrame(spinR); }})();
}})();
</script></body></html>"""
        self.viewer_cand_lbl.setText(
            f'<span style="color:{ext_color};font-weight:bold;">{label_ext}</span> (side-by-side)')
        self._log(f"[EXT-SBS] html={len(html.encode())} bytes")
        self._set_html(html)

    # ── Energy landscape ──────────────────────────────────────────

    def _start_landscape(self):
        if self._init_atoms is None:
            return
        if self._landscape_worker is not None and self._landscape_worker.isRunning():
            return

        # ── 최적 후보에서 분지 탐색 (Branch exploration from the best candidate) ──
        # 원본 파싱 구조가 아니라, MC 앙상블에서 가장 낮은 에너지를 가진(=가장
        # 그럴듯한) 후보를 지형 탐색의 시작점으로 사용한다. 최적 후보라 해도
        # 유연한 영역(예: 무질서 링커)에서는 그 자체로 여러 준안정 상태를
        # 가질 수 있으므로, 그 후보에서부터 분지시켜 실제로 어떤 위치들이
        # 가능한지 보여준다. 앙상블이 아직 없으면(방어적으로) 원본 파싱
        # 구조로 폴백한다.
        #
        # Branch from the best (lowest-energy) MC candidate, not the raw
        # parsed input -- that candidate is the "most likely" structure, but
        # even it can have several accessible sub-states in flexible regions
        # (e.g. a disordered linker). Exploring from there reveals what
        # positions are actually reachable from that most-likely state,
        # rather than from the unrelaxed starting coordinates. Falls back to
        # the raw parsed structure if no ensemble exists yet (defensive).
        # 상위 K개 후보에서 각각 뻗어나가는 다중 분지 탐색 (Multi-branch
        # exploration from the top-K candidates) — 안정 단백질에서는 진짜
        # 우물 하나로 다 수렴하므로 사실상 단일 궤적과 다를 바 없지만,
        # 지배적 상태가 없는 단백질(IDP)에서는 "최적" 후보 하나가 우연한
        # 표본일 뿐일 수 있어 그 하나에서만 뻗으면 편향된다.
        #
        # Multi-branch exploration from the top-K candidates -- for a
        # stable protein this converges to the same real well regardless
        # (so it behaves like the old single-trajectory version), but for a
        # protein with no dominant state (an IDP) the single "best"
        # candidate can be close to an arbitrary sample, so branching from
        # it alone biases the result.
        # 분지 수 (2026-07-09): 처음엔 GPU에서 무조건 6분지로 고정했으나,
        # 1YPI(494잔기)에서 오히려 CPU 3분지보다 34% 느려지는 것을 발견했다
        # (194.5s → 260.1s). 원인: 76-140잔기에서 "거의 공짜"였던 동시성은
        # 사실 진짜 커널 동시 실행이 아니라 — 커널이 작아 실행 시간보다
        # CPU 쪽 커널 실행(launch) 오버헤드가 더 크고, 그 오버헤드가
        # 여러 스레드에 걸쳐 파이프라인되며 숨겨진 것뿐이었다(명시적
        # cudaStream_t 없이 전부 기본 스트림을 공유 — physics_engine_cuda.cu
        # 확인함). 494잔기(~4배 원자 수)에서는 커널 자체가 GPU 연산 자원을
        # 실제로 점유할 만큼 커져서, "동시" 분지들이 진짜로 같은 SM/코어를
        # 놓고 경쟁하게 된다 — 열 스로틀링이 아니라 구조적 전환점.
        #
        # 실측 4개 지점(76/129/140/494잔기)에 맞춰 크기에 따라 매끄럽게
        # 줄어드는 함수로 대체: 140잔기까지는 6분지 그대로(전부 실측 확인),
        # 그 이상은 1/sqrt(n_res/140)로 감소시켜 494잔기에서 자연히 3분지로
        # (CPU와 동일) 수렴한다 — 정확한 "전환점"은 아직 모르므로(140~494
        # 사이 데이터 없음) 하드 임계값 대신 완만한 함수로 안전하게 근사.
        #
        # Branch count (2026-07-09): originally fixed at 6 on GPU
        # unconditionally, but found 1YPI (494 res) got 34% *slower* than
        # the CPU 3-branch baseline (194.5s -> 260.1s). Root cause: the
        # "nearly free" concurrency measured at 76-140 res wasn't real
        # simultaneous kernel execution -- at that atom count each kernel is
        # small enough that CPU-side launch-dispatch overhead dominates over
        # actual kernel runtime, and that overhead pipelines across threads
        # (no explicit cudaStream_t anywhere -- confirmed in
        # physics_engine_cuda.cu -- everything shares the default stream).
        # At 494 res (~4x the atoms), kernels are big enough to genuinely
        # occupy the GPU's compute resources on their own, so "concurrent"
        # branches start really competing for the same SMs/cores -- a
        # structural crossover, not thermal throttling.
        #
        # Replaced the flat constant with a function matching the 4 real
        # measured points (76/129/140 res): stays at the max (6, all
        # confirmed good there) up to the anchor size, then decays as
        # 1/sqrt(n_res/anchor) above it -- lands at 3 (matching CPU) for
        # 1YPI's 494 res by construction. The exact crossover point between
        # 140 and 494 res is unmeasured, so this is a smooth approximation,
        # not a validated threshold -- see IMPROVEMENTS.md item #2.
        GPU_BRANCH_MAX    = 6
        GPU_BRANCH_ANCHOR = 140   # largest size where 6 branches is confirmed good
        CPU_BRANCH_COUNT  = 3
        n_res_for_branching = len(self._ca_indices) if self._ca_indices else GPU_BRANCH_ANCHOR
        if self._physics_mod is not protein_physics:
            ratio = max(n_res_for_branching, GPU_BRANCH_ANCHOR) / GPU_BRANCH_ANCHOR
            branch_count = max(CPU_BRANCH_COUNT,
                                min(GPU_BRANCH_MAX, round(GPU_BRANCH_MAX / ratio ** 0.5)))
        else:
            branch_count = CPU_BRANCH_COUNT

        start_atoms = self._init_atoms
        extra_seeds = []
        branch_note = ""
        if self._ensemble and self._energies:
            order = np.argsort(self._energies)
            top_k = order[:min(branch_count, len(order))]
            best_idx = int(top_k[0])
            start_atoms = self._ensemble[best_idx]
            extra_seeds = [self._ensemble[int(i)] for i in top_k[1:]]
            branch_note = (f" (branching from {len(top_k)} best candidates, "
                            f"best={self._energies[best_idx]:.1f} kcal/mol)")

        self.landscape_start_btn.setEnabled(False)
        self.landscape_start_btn.setText("◈  COMPUTING…")
        self.idp_status_lbl.setText("…")
        self.idp_status_lbl.setStyleSheet("font-size:10px;font-weight:bold;color:#d97706;")
        self._log(f"[LANDSCAPE] Starting Markov-chain exploration{branch_note}…")

        # Fresh engine instance to avoid thread contention with ComparisonWorker
        try:
            ls_engine = self._physics_mod.PhysicsEngine()
        except Exception as ex:
            self._log(f"[LANDSCAPE] Engine init failed: {ex}")
            self.landscape_start_btn.setEnabled(True)
            self.landscape_start_btn.setText("◈  EXPLORE LANDSCAPE")
            return

        backbone_ncac = None
        if self._ref_ca_map and self._heavy_map and self._heavy_indices:
            backbone_ncac = _backbone_ncac_indices(
                self._ref_ca_map, self._heavy_map, self._heavy_indices)

        self._landscape_worker = LandscapeWorker(
            ls_engine, start_atoms, self._ca_indices, self._topo, self._physics_mod,
            extra_seeds=extra_seeds, backbone_ncac=backbone_ncac,
            iupred_scores=self._iupred_scores)
        self._landscape_worker.progress.connect(self._log)
        self._landscape_worker.result.connect(self._on_landscape_done)
        self._landscape_worker.gpu_fallback.connect(self._on_gpu_fallback)
        self._landscape_worker.start()

    def _on_landscape_done(self, data):
        self._landscape_snaps    = data["snapshots"]
        self._landscape_energies = data["energies"]
        self._landscape_dominant_particles = data.get("dominant_particles")
        self._sub_candidates     = data.get("sub_candidates", [])
        self._basin_summary      = data.get("basin_summary", [])
        self._active_subcand     = None
        self.landscape_start_btn.setText("◈  RE-EXPLORE")
        self.landscape_start_btn.setEnabled(True)
        self.landscape_toggle_btn.setEnabled(True)

        lbl   = data["idp_label"]
        color = data["idp_color"]
        self.idp_status_lbl.setText(lbl)
        self.idp_status_lbl.setStyleSheet(
            f"font-size:10px;font-weight:bold;color:{color};margin-left:6px;")

        self._build_subregion_panel(self._sub_candidates)
        self._build_basinpool_panel(self._basin_summary)
        self._draw_landscape(data)

        # Compute RMSF from trajectory and draw disorder profile
        if self._ca_indices and self._pdb_path:
            rmsf     = _compute_rmsf(data["snapshots"], self._ca_indices,
                                      iupred_scores=self._iupred_scores)
            residues = _extract_ca_residues(self._pdb_path)
            self._rmsf          = rmsf
            self._rmsf_residues = residues
            self._draw_disorder_profile(rmsf=rmsf, residues=residues,
                                        iupred_scores=self._iupred_scores)
            self.disorder_toggle_btn.setEnabled(True)

            rg = _compute_radius_of_gyration(data["snapshots"], self._ca_indices)
            end_to_end = _compute_end_to_end(data["snapshots"], self._ca_indices)
            seps, mean_dists, nu, log_r0, r_squared = _compute_internal_scaling(
                data["snapshots"], self._ca_indices)
            contact_freq, _contact_var = _compute_contact_map(
                data["snapshots"], self._ca_indices)
            self._draw_ensemble_metrics(rg, end_to_end, seps, mean_dists, nu, log_r0,
                                        r_squared, contact_freq)
            self.ensemble_toggle_btn.setEnabled(True)

        self._log(f"[LANDSCAPE] Classification: {lbl}  ·  "
                  f"{data['n_sig']} metastable basins  ·  "
                  f"funnel={data['funnel']:.2f}")

    # ── Sub-region / sub-candidate picker ──────────────────────────

    def _build_subregion_panel(self, sub_candidates):
        """(Re)build the sub-region bar from the landscape worker's dynamic
        sub-candidate picking (see LandscapeWorker._pick_strategy). One row
        per flagged region, each with a dropdown of that region's
        sub-candidates (sourced from the general run, a dedicated
        re-search, or both) and a VIEW button."""
        for row in self._subregion_rows:
            self._subregion_v.removeWidget(row); row.deleteLater()
        self._subregion_rows.clear()

        if not sub_candidates:
            self._subregion_panel.setVisible(False)
            return

        for ri, region in enumerate(sub_candidates):
            lo, hi = region["region"]
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
            lbl = QLabel(
                f"REGION {lo+1}-{hi+1} &middot; {region['strategy']} &middot; "
                f"{region['population']*100:.0f}% pop &middot; "
                f"SS diff {region.get('ss_diff', 0)*100:.0f}% &middot; "
                f"{len(region['candidates'])} sub-candidates")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("color:#92400e;font-size:9px;letter-spacing:0.5px;")
            h.addWidget(lbl)
            combo = QComboBox()
            combo.setFixedHeight(20)
            combo.setStyleSheet("font-size:9px;")
            for ci, cand in enumerate(region["candidates"]):
                combo.addItem(f"#{ci+1}  {cand['energy']:.1f} kcal/mol "
                              f"({cand['source']})")
            h.addWidget(combo)
            btn = QPushButton("VIEW")
            btn.setFixedHeight(20)
            btn.setStyleSheet(
                "background:#7c3aed;color:#fff;border:none;border-radius:3px;"
                "font-size:9px;padding:2px 10px;")
            btn.clicked.connect(
                lambda _, ri=ri, cb=combo: self._render_subcandidate(ri, cb.currentIndex()))
            h.addWidget(btn)
            h.addStretch()
            self._subregion_v.addWidget(row)
            self._subregion_rows.append(row)

        self._subregion_panel.setVisible(True)

    def _build_basinpool_panel(self, basin_summary):
        """(Re)build the structural-pool bar: every significant basin found,
        ranked by population, with a VIEW button per row. Populated even
        when there's no single dominant state (a real IDP) -- the point is
        showing what recurs and how often, not picking one reference."""
        for row in self._basinpool_rows:
            self._basinpool_v.removeWidget(row); row.deleteLater()
        self._basinpool_rows.clear()

        if not basin_summary:
            self._basinpool_panel.setVisible(False)
            return

        for bi, basin in enumerate(basin_summary):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
            tag = " ★ DOMINANT" if basin.get("is_dominant") else ""
            lbl = QLabel(
                f"BASIN {bi+1} &middot; {basin['population']*100:.0f}% pop &middot; "
                f"E=[{basin['energy_min']:.0f}, {basin['energy_max']:.0f}]{tag}")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("color:#1d4ed8;font-size:9px;letter-spacing:0.5px;")
            h.addWidget(lbl)
            h.addStretch()
            btn = QPushButton("VIEW")
            btn.setFixedHeight(20)
            btn.setStyleSheet(
                "background:#1d4ed8;color:#fff;border:none;border-radius:3px;"
                "font-size:9px;padding:2px 10px;")
            btn.clicked.connect(lambda _, bi=bi: self._render_basin(bi))
            h.addWidget(btn)
            # OVERLAY button (2026-07-13, item #7 re-test): spaghetti-plot
            # view of just this basin's own conformers, reusing
            # _render_ensemble_overlay's existing rendering via basin_idx --
            # see _render_basin_overlay.
            overlay_btn = QPushButton("OVERLAY")
            overlay_btn.setFixedHeight(20)
            overlay_btn.setStyleSheet(
                "background:#7c3aed;color:#fff;border:none;border-radius:3px;"
                "font-size:9px;padding:2px 10px;")
            overlay_btn.clicked.connect(lambda _, bi=bi: self._render_basin_overlay(bi))
            h.addWidget(overlay_btn)
            self._basinpool_v.addWidget(row)
            self._basinpool_rows.append(row)

        self._basinpool_panel.setVisible(True)

    def _render_basin(self, basin_idx):
        if basin_idx >= len(self._basin_summary):
            return
        basin = self._basin_summary[basin_idx]
        pdb_str = self._build_ca_pdb_str(basin["particles"])
        n_atoms = len(basin["particles"])

        html = f"""<!DOCTYPE html><html><head>
<script src="{_3DMOL_JS_URL}"></script>
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
<div id="info">BASIN {basin_idx + 1} &nbsp;&middot;&nbsp; \
{basin['population']*100:.0f}% pop &nbsp;&middot;&nbsp; \
E=[{basin['energy_min']:.0f}, {basin['energy_max']:.0f}] &nbsp;&middot;&nbsp; \
{n_atoms} RESIDUES</div>
<div id="legend"><span style="color:#1d4ed8">&#9632;</span> STRUCTURAL POOL BASIN</div>
<script>
(function(){{
  var v=$3Dmol.createViewer("v",{{backgroundColor:"#f8fafc"}});
  var m=v.addModel(`{pdb_str}`,"pdb");
  m.setStyle({{}},{{cartoon:{{color:"#1d4ed8",thickness:0.8,opacity:1.0,style:"trace"}},
                    sphere:{{color:"#1d4ed8",radius:0.55,opacity:1.0}}}});
  v.zoomTo(); v.zoom(0.85); v.render();
  (function spin(){{ v.rotate(1,'y'); v.render(); requestAnimationFrame(spin); }})();
}})();
</script></body></html>"""

        self.viewer_cand_lbl.setText(
            f'<span style="color:#1d4ed8;font-weight:bold;">BASIN {basin_idx+1}</span> '
            f'· {basin["population"]*100:.0f}% pop')
        # Render into the landscape page's own embedded viewer, not the
        # main structure page -- clicking around the structural pool should
        # update the split-view structure in place, not yank the user back
        # to the full structure page every time.
        self._set_landscape_html(html)

    def _build_ca_pdb_str(self, particles, transform=None):
        """Cα-only synthetic PDB text, indexed the same way as the region
        (lo, hi) tuples from LandscapeWorker (both derive from ca_indices in
        the same order), so a flagged region maps directly onto residue
        numbers here without needing a separate atom->residue lookup."""
        lines = []
        for i, pidx in enumerate(self._ca_indices):
            if pidx >= len(particles):
                continue
            p = particles[pidx]
            x, y, z = p.x, p.y, p.z
            if transform is not None:
                R, ref_c, mob_c = transform
                x, y, z = (np.array([x, y, z]) - mob_c) @ R.T + ref_c
            lines.append(
                f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.50           C")
        return "\n".join(lines)

    def _render_subcandidate(self, region_idx, cand_idx):
        if region_idx >= len(self._sub_candidates):
            return
        region = self._sub_candidates[region_idx]
        cands  = region["candidates"]
        if not cands:
            return
        cand_idx = max(0, min(cand_idx, len(cands) - 1))
        cand = cands[cand_idx]
        self._active_subcand = (region_idx, cand_idx)

        dom_pdb = self._build_ca_pdb_str(self._landscape_dominant_particles)
        sub_pdb = self._build_ca_pdb_str(cand["particles"])
        dom_esc = dom_pdb.replace("\\", "\\\\").replace("`", "\\`")
        sub_esc = sub_pdb.replace("\\", "\\\\").replace("`", "\\`")

        lo, hi = region["region"]
        resi_list = list(range(lo + 1, hi + 2))

        html = f"""<!DOCTYPE html><html><head>
<script src="{_3DMOL_JS_URL}"></script>
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
<div id="info">SUB-CANDIDATE #{cand_idx + 1}/{len(cands)} &nbsp;&middot;&nbsp; \
REGION {lo+1}-{hi+1} &nbsp;&middot;&nbsp; {cand['energy']:.1f} kcal/mol \
&nbsp;&middot;&nbsp; {cand['source']}</div>
<div id="legend">OVERLAY &nbsp; <span style="color:#94a3b8">&#9632;</span> DOMINANT BASIN &nbsp;\
<span style="color:#7c3aed">&#9632;</span> SUB-CANDIDATE &nbsp;\
<span style="color:#dc2626">&#9632;</span> FLAGGED REGION</div>
<script>
(function(){{
  var v=$3Dmol.createViewer("v",{{backgroundColor:"#f8fafc"}});
  var mDom=v.addModel(`{dom_esc}`,"pdb");
  mDom.setStyle({{}},{{cartoon:{{color:"#94a3b8",thickness:0.6,opacity:0.45,style:"trace"}},
                      sphere:{{color:"#94a3b8",radius:0.4,opacity:0.4}}}});
  var mSub=v.addModel(`{sub_esc}`,"pdb");
  mSub.setStyle({{}},{{cartoon:{{color:"#7c3aed",thickness:0.8,opacity:0.9,style:"trace"}},
                      sphere:{{color:"#7c3aed",radius:0.5,opacity:0.85}}}});
  mSub.setStyle({{resi:{resi_list}}},{{cartoon:{{color:"#dc2626",thickness:1.0,opacity:1.0,style:"trace"}},
                      sphere:{{color:"#dc2626",radius:0.65,opacity:1.0}}}});
  v.zoomTo(); v.zoom(0.85); v.render();
  (function spin(){{ v.rotate(1,'y'); v.render(); requestAnimationFrame(spin); }})();
}})();
</script></body></html>"""

        self.viewer_cand_lbl.setText(
            f'<span style="color:#7c3aed;font-weight:bold;">SUB-CANDIDATE</span> '
            f'REGION {lo+1}-{hi+1} · #{cand_idx+1}/{len(cands)}')
        # Stays on the landscape page -- see _render_basin for why.
        self._set_landscape_html(html)

    def _draw_landscape(self, data):
        """Render the conformational graph on the matplotlib canvas.
        matplotlib 캔버스에 구조 그래프(에너지 지형)를 렌더링.

        시각화 구성 요소:
        ─────────────────
        • 얇은 회색 엣지: MC 궤적의 시간 순서 연결 (thin grey edges = time sequence)
        • 볼록 껍질(convex hull) 음영: 각 군집(분지)의 영역 표시
          (Community convex hulls = metastable basin boundaries)
        • 산점도 노드: 색상 = 에너지 (RdYlGn_r: 낮음=초록, 높음=빨강)
          (Scatter nodes coloured by energy)
        • 녹색 사각형 = 시작점, 주황 다이아몬드 = 끝점,
          파란 별 = 최소 에너지 구조 (에너지 최솟값)
        • 컬러바: 에너지 스케일 (kcal/mol)
        • 레이블 B1, B2, ...: 군집(분지) 번호
        • X축/Y축: PC1/PC2 — 설명 분산(explained variance) % 표시
        """
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
<script src="{_3DMOL_JS_URL}"></script>
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
  (function spin(){{ v.rotate(1,'y'); v.render(); requestAnimationFrame(spin); }})();
}})();
</script></body></html>"""

        self.viewer_cand_lbl.setText(
            f'<span style="color:#0891b2;font-weight:bold;">'
            f'SNAPSHOT #{snap_idx + 1}</span> · {energy:.1f} kcal/mol')
        # Renders into the landscape page's own embedded viewer (split view)
        # instead of switching away to the main structure page -- clicking
        # through nodes is meant to be repeated many times while exploring,
        # so it shouldn't force a page jump each time.
        self._set_landscape_html(html)

    def _toggle_landscape(self):
        if self._view_stack.currentIndex() == 0:
            self._view_stack.setCurrentIndex(1)
            self.view_mode_btn.setVisible(False)
            self.landscape_toggle_btn.setText("⊡  STRUCTURE")
            # Default the split view's structure panel to the dominant basin
            # (index 0 -- basin_summary is population-sorted) rather than
            # leaving it blank on first open.
            if self._basin_summary:
                self._render_basin(0)
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
        if self._current_ext_entry is not None:
            self._render_external(self._current_ext_entry)
        elif self._ensemble:
            self._render(self._current_cand_idx)

    def _build_pdb_str(self, particles, transform=None):
        """transform, given, is (R, ref_centroid, mobile_centroid) from
        _kabsch_fit — applied to each atom before writing, so a candidate can
        be rendered in an external reference's frame instead of its own raw
        MC coordinates (see _render_external_layered)."""
        lines = []
        for i, p in enumerate(particles):
            x, y, z = p.x, p.y, p.z
            if transform is not None:
                R, ref_c, mob_c = transform
                x, y, z = (np.array([x, y, z]) - mob_c) @ R.T + ref_c
            lines.append(
                f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.50           C")
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
        self._current_cand_idx  = cand_idx
        self._current_ext_entry = None
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
<script src="{_3DMOL_JS_URL}"></script>
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
  (function spin(){{ v.rotate(1,'y'); v.render(); requestAnimationFrame(spin); }})();
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
<script src="{_3DMOL_JS_URL}"></script>
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
  (function spinL(){{ vL.rotate(1,'y'); vL.render(); requestAnimationFrame(spinL); }})();
  var vR=$3Dmol.createViewer("vR",{{backgroundColor:"#f8fafc"}});
{right_js}
  vR.zoomTo(); vR.zoom(0.85); vR.render();
  (function spinR(){{ vR.rotate(1,'y'); vR.render(); requestAnimationFrame(spinR); }})();
}})();
</script></body></html>"""
        self._log(f"[SBS] html={len(html.encode())} bytes")
        self._set_html(html)

    # ── Disorder / RMSF profile ───────────────────────────────────

    def _draw_disorder_profile(self, rmsf=None, residues=None, iupred_scores=None):
        """잔기별 유연성 프로파일을 matplotlib으로 그린다.
        Plot the per-residue flexibility / disorder profile.

        ── 이중 패널 설계 (Dual-panel design) ────────────────────────────────────
        사용 가능한 데이터에 따라 1~2개의 서브플롯을 동적으로 생성한다:

          1. IUPred 패널 (PDB 파싱 직후 즉시 사용 가능):
             서열 기반 잔기별 무질서 확률 [0, 1].
             기준값(threshold) 0.5 초과 → 무질서로 예측.
             서열 특성만으로 판단하므로 MC 실행 전에 먼저 볼 수 있음.

          2. RMSF 패널 (landscape MC 궤적 완료 후):
             MC 시뮬레이션에서 각 Cα의 평균 제곱근 요동 (Å).
             RMSF ≥ 2 Å → 유연/무질서 영역 (문헌 표준 기준값).
             X-선 결정학 B-인수와 직접 비교 가능: RMSF = sqrt(3B/8π²).

        두 패널은 동일한 x축(잔기 번호)와 레이블을 공유해 비교를 용이하게 한다.

        Dynamic subplot count based on available data:
          Panel 1 — IUPred: sequence-based disorder probability [0,1] per residue.
                    Available immediately after _parse_pdb().  Threshold = 0.5.
          Panel 2 — RMSF:   per-Cα root-mean-square fluctuation (Å) from MC.
                    Available after landscape MC trajectory.  Threshold = 2 Å.
                    Directly comparable to X-ray B-factors: RMSF = sqrt(3B/8π²).

        ── 동적 업데이트 흐름 (Update sequence) ────────────────────────────────
          1. _on_done()에서 호출: iupred_scores만 있음 → IUPred 단일 패널.
          2. _on_landscape_done()에서 호출: rmsf도 있음 → 이중 패널.

        Called first from _on_done() with IUPred only → single panel.
        Called again from _on_landscape_done() with both → dual panel.

        파라미터 / Parameters
        ─────────────────────
        rmsf          — 잔기별 RMSF 배열 (Å). None이면 RMSF 패널 생략.
                        Per-residue RMSF array (Å). None → skip RMSF panel.
        residues      — [(chain_id, res_seq, res_name3)] — x축 눈금 레이블.
                        List of (chain_id, res_seq, res_name3) for x-axis tick labels.
        iupred_scores — 잔기별 무질서 확률 [0,1]. None이면 IUPred 패널 생략.
                        Per-residue disorder probability [0,1]. None → skip IUPred panel.
        """
        have_rmsf   = rmsf is not None and len(rmsf) > 0
        have_iupred = iupred_scores is not None and len(iupred_scores) > 0

        if not have_rmsf and not have_iupred:
            return

        n_res = 0
        if have_rmsf and residues:
            n_res = min(len(rmsf), len(residues))
        if have_iupred and residues:
            n_res = max(n_res, min(len(iupred_scores), len(residues) if residues else len(iupred_scores)))
        if n_res == 0:
            if have_rmsf:    n_res = len(rmsf)
            if have_iupred:  n_res = max(n_res, len(iupred_scores))
        if n_res == 0:
            return

        n_panels = int(have_rmsf) + int(have_iupred)
        self._disorder_fig.clear()
        self._disorder_fig.patch.set_facecolor("#f8fafc")
        axes = self._disorder_fig.subplots(n_panels, 1, squeeze=False,
                                           gridspec_kw={"hspace": 0.45})
        panel = 0

        def _tick_labels(ax_obj, n):
            if residues and len(residues) >= n:
                step = max(1, n // 20)
                tp   = list(range(1, n + 1, step))
                tl   = [str(residues[t-1][1]) if t-1 < len(residues) else str(t) for t in tp]
                ax_obj.set_xticks(tp); ax_obj.set_xticklabels(tl, fontsize=6, rotation=45, ha="right")

        # ── IUPred panel (sequence-based, always shown when available) ──────
        if have_iupred:
            ax = axes[panel][0]
            ax.set_facecolor("#f8fafc")
            n = min(len(iupred_scores), n_res)
            x = np.arange(1, n + 1)
            y = np.array(iupred_scores[:n])
            THOLD = 0.5
            n_dis = int((y > THOLD).sum())
            pct   = 100.0 * n_dis / n if n else 0.0
            ax.fill_between(x, 0, y, where=(y <= THOLD), color="#16a34a", alpha=0.20)
            ax.fill_between(x, 0, y, where=(y > THOLD),  color="#dc2626", alpha=0.20)
            ax.plot(x, y, color="#1e293b", lw=1.0, zorder=3)
            ax.axhline(THOLD, color="#dc2626", lw=0.8, linestyle="--", alpha=0.6)
            ax.set_xlim(1, n); ax.set_ylim(0, 1)
            ax.set_ylabel("IUPred score", fontsize=7, color="#64748b", labelpad=3)
            ax.set_title(f"Sequence disorder (IUPred)  ·  {n_dis}/{n} predicted disordered ({pct:.1f}%)",
                         fontsize=8, color="#1e293b", fontweight="bold", pad=5)
            ax.tick_params(colors="#94a3b8", labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor("#e2e8f0"); sp.set_linewidth(0.7)
            _tick_labels(ax, n)
            panel += 1

        # ── RMSF panel (simulation-based, shown when landscape is done) ─────
        if have_rmsf:
            ax = axes[panel][0]
            ax.set_facecolor("#f8fafc")
            n = min(len(rmsf), n_res)
            x = np.arange(1, n + 1)
            y = np.array(rmsf[:n])
            THOLD = 2.0
            n_dis = int((y > THOLD).sum())
            pct   = 100.0 * n_dis / n if n else 0.0
            self._rmsf_n_disordered = n_dis
            self._rmsf_pct          = pct
            self.disorder_stats_lbl.setText(
                f"RMSF: {n_dis}/{n_res} residues disordered (≥2 Å)  ({pct:.1f}%)")
            ax.fill_between(x, 0, y, where=(y <= THOLD), color="#16a34a", alpha=0.22)
            ax.fill_between(x, 0, y, where=(y > THOLD),  color="#dc2626", alpha=0.22)
            ax.plot(x, y, color="#1e293b", lw=1.0, zorder=3)
            ax.axhline(THOLD, color="#dc2626", lw=0.8, linestyle="--", alpha=0.6)
            ax.set_xlim(1, n); ax.set_ylim(bottom=0)
            ax.set_xlabel("Residue", fontsize=7, color="#64748b", labelpad=3)
            ax.set_ylabel("RMSF (Å)", fontsize=7, color="#64748b", labelpad=3)
            ax.set_title(f"MC flexibility (RMSF)  ·  {n_dis}/{n} disordered ({pct:.1f}%)",
                         fontsize=8, color="#1e293b", fontweight="bold", pad=5)
            ax.tick_params(colors="#94a3b8", labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor("#e2e8f0"); sp.set_linewidth(0.7)
            _tick_labels(ax, n)

        if not have_rmsf:
            # Update stats label for IUPred-only mode
            n = min(len(iupred_scores), n_res) if have_iupred else 0
            y = np.array(iupred_scores[:n]) if have_iupred else np.array([])
            n_dis = int((y > 0.5).sum())
            pct   = 100.0 * n_dis / n if n else 0.0
            self.disorder_stats_lbl.setText(
                f"IUPred: {n_dis}/{n} residues disordered (score>0.5)  ({pct:.1f}%)")

        self._disorder_canvas.draw()

    def _draw_ensemble_metrics(self, rg, end_to_end, seps, mean_dists, nu, log_r0, r_squared,
                                contact_freq):
        """Draw the 4-panel IDP ensemble-characterization figure (Rg
        histogram, end-to-end histogram, internal-distance scaling-law fit,
        contact-map heatmap) -- see IMPROVEMENTS.md item #7 for the design
        rationale of each metric. Any panel is skipped if its data is empty
        (e.g. the scaling-law fit needs >=3 distinct sequence separations
        and returns None for nu/r_squared when there aren't enough).
        """
        have_rg      = rg is not None and len(rg) > 0
        have_e2e     = end_to_end is not None and len(end_to_end) > 0
        have_scaling = seps is not None and len(seps) > 0 and nu is not None
        have_contact = contact_freq is not None and contact_freq.size > 0

        if not (have_rg or have_e2e or have_scaling or have_contact):
            return

        self._ensemble_fig.clear()
        self._ensemble_fig.patch.set_facecolor("#f8fafc")
        axes = self._ensemble_fig.subplots(2, 2)

        def _style(ax_obj):
            ax_obj.set_facecolor("#f8fafc")
            ax_obj.tick_params(colors="#94a3b8", labelsize=6)
            for sp in ax_obj.spines.values():
                sp.set_edgecolor("#e2e8f0"); sp.set_linewidth(0.7)

        # ── Rg histogram ─────────────────────────────────────────
        ax = axes[0][0]; _style(ax)
        if have_rg:
            ax.hist(rg, bins=min(20, max(5, len(rg) // 2)), color="#7c3aed", alpha=0.6,
                    edgecolor="#5b21b6")
            ax.set_xlabel("Rg (Å)", fontsize=7, color="#64748b", labelpad=3)
            ax.set_ylabel("count", fontsize=7, color="#64748b", labelpad=3)
            ax.set_title(f"Radius of gyration  ·  {rg.mean():.2f}±{rg.std():.2f} Å",
                         fontsize=8, color="#1e293b", fontweight="bold", pad=5)
        else:
            ax.axis("off")

        # ── End-to-end histogram ─────────────────────────────────
        ax = axes[0][1]; _style(ax)
        if have_e2e:
            ax.hist(end_to_end, bins=min(20, max(5, len(end_to_end) // 2)),
                    color="#0891b2", alpha=0.6, edgecolor="#0e7490")
            ax.set_xlabel("N-to-C distance (Å)", fontsize=7, color="#64748b", labelpad=3)
            ax.set_ylabel("count", fontsize=7, color="#64748b", labelpad=3)
            ax.set_title(f"End-to-end distance  ·  {end_to_end.mean():.2f}±{end_to_end.std():.2f} Å",
                         fontsize=8, color="#1e293b", fontweight="bold", pad=5)
        else:
            ax.axis("off")

        # ── Internal scaling law: <R_ij> ~ |i-j|^nu ──────────────
        ax = axes[1][0]; _style(ax)
        if have_scaling:
            ax.scatter(seps, mean_dists, s=10, color="#ea580c", alpha=0.7, zorder=3)
            fit_x = np.array([seps.min(), seps.max()])
            fit_y = np.exp(nu * np.log(fit_x) + log_r0)
            ax.plot(fit_x, fit_y, color="#1e293b", lw=1.2, linestyle="--", zorder=2)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("sequence separation |i-j|", fontsize=7, color="#64748b", labelpad=3)
            ax.set_ylabel("<R_ij> (Å)", fontsize=7, color="#64748b", labelpad=3)
            r2_str = f"{r_squared:.2f}" if r_squared is not None else "n/a"
            ax.set_title(f"Internal scaling  ·  ν={nu:.3f}  (R²={r2_str})",
                         fontsize=8, color="#1e293b", fontweight="bold", pad=5)
        else:
            ax.axis("off")

        # ── Contact-map heatmap ───────────────────────────────────
        ax = axes[1][1]; _style(ax)
        if have_contact:
            im = ax.imshow(contact_freq, cmap="viridis", vmin=0, vmax=1, origin="lower")
            ax.set_xlabel("residue", fontsize=7, color="#64748b", labelpad=3)
            ax.set_ylabel("residue", fontsize=7, color="#64748b", labelpad=3)
            ax.set_title("Contact frequency", fontsize=8, color="#1e293b",
                         fontweight="bold", pad=5)
            cbar = self._ensemble_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6, colors="#94a3b8")
        else:
            ax.axis("off")

        stats_bits = []
        if have_rg:
            stats_bits.append(f"Rg {rg.mean():.1f}±{rg.std():.1f} Å")
        if have_scaling:
            stats_bits.append(f"ν={nu:.2f}")
        self.ensemble_stats_lbl.setText("  ·  ".join(stats_bits) if stats_bits else "—")

        self._ensemble_canvas.draw()

    def _render_colored_by_rmsf(self):
        """Load the MC best structure into 3Dmol colored by per-residue RMSF (B-factor).
        MC 최저 에너지 구조를 잔기별 RMSF(B-인수)로 색칠해 3Dmol로 렌더링.

        RMSF 값을 PDB B-인수(온도 인수) 필드에 저장해 3Dmol의 bwr 그래디언트로 표시:
          파란색(blue) = RMSF 낮음 = 딱딱한(rigid) 구조 영역
          흰색(white)  = 중간 유연성
          빨간색(red)  = RMSF 높음 = 유연/무질서 영역

        RMSF stored in PDB B-factor field; 3Dmol's bwr gradient maps:
          blue → rigid (low RMSF), white → intermediate, red → flexible/disordered.
        Colour scale: 0 → 4.0 Å (clamped).
        """
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
<script src="{_3DMOL_JS_URL}"></script>
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
  (function spin(){{ v.rotate(1,'y'); v.render(); requestAnimationFrame(spin); }})();
}})();
</script></body></html>"""

        self.viewer_cand_lbl.setText(
            '<span style="color:#7c3aed;font-weight:bold;">FLEXIBILITY MAP</span>'
            ' &nbsp;·&nbsp; blue=rigid &nbsp;·&nbsp; red=disordered')
        self._set_html(html)
        self._view_stack.setCurrentIndex(0)
        self.view_mode_btn.setVisible(True)
        self.disorder_toggle_btn.setText("⊛  DISORDER")

    def _render_ensemble_overlay(self, basin_idx=None):
        """Overlay a sample of MC trajectory snapshots in one 3Dmol viewer --
        the classic NMR-ensemble "spaghetti plot" view of what the sampled
        conformational ensemble actually looks like in 3D, rather than only
        the scalar Rg/nu/contact-map summaries already on the ENSEMBLE page
        (see IMPROVEMENTS.md item #7 -- this is the third, previously-
        unstarted "GUI/visualization" direction).

        Each overlay frame is Kabsch-superposed onto the first snapshot,
        with the fit restricted to the IUPred-predicted ordered core when
        available (same rationale/mask as _compute_rmsf) -- otherwise a
        genuinely rigid core would look just as scattered as a real
        disordered tail, since torsion-angle MC has no reason to keep a
        fixed absolute pose across a trajectory. Frames are colored by the
        same per-residue RMSF (b-factor, bwr gradient) already used for
        COLOR BY FLEXIBILITY, so the rigid core stays visibly steady/blue
        while a real disordered region visibly frays red across frames.
        The lowest-energy dominant structure is drawn on top as an opaque
        solid reference.

        basin_idx (2026-07-13, IMPROVEMENTS.md item #7 re-test): when given,
        restricts the overlay to only that basin_summary entry's member
        snapshots (via its "member_indices", see run()'s basin_summary
        construction) instead of the whole pooled multi-branch trajectory --
        lets a user compare one basin's conformers in isolation rather than
        always seeing every basin superimposed. None (default) reproduces
        the original always-pooled behavior unchanged.
        """
        snaps = self._landscape_snaps
        if basin_idx is not None:
            if basin_idx >= len(self._basin_summary):
                return
            member_indices = self._basin_summary[basin_idx]["member_indices"]
            snaps = [snaps[i] for i in member_indices if i < len(snaps)]
        if not snaps or not self._ca_indices:
            return
        ca_indices = self._ca_indices
        n_ca = len(ca_indices)
        n_snaps = len(snaps)

        coords = np.zeros((n_snaps, n_ca, 3), dtype=float)
        for si, particles in enumerate(snaps):
            for ci, pidx in enumerate(ca_indices):
                if pidx < len(particles):
                    p = particles[pidx]
                    coords[si, ci] = [p.x, p.y, p.z]

        fit_mask = None
        if self._iupred_scores is not None and len(self._iupred_scores) == n_ca:
            fit_mask = np.asarray(self._iupred_scores) < 0.5

        ref = coords[0]
        MAX_OVERLAY = 20
        sample_idx = sorted(set(
            np.linspace(0, n_snaps - 1, min(n_snaps, MAX_OVERLAY)).astype(int).tolist()))

        bfac = (self._rmsf if self._rmsf is not None and len(self._rmsf) == n_ca
                else None)

        models_js = []
        for k, si in enumerate(sample_idx):
            frame = coords[si] if si == 0 else _kabsch_align_points(
                ref, coords[si], fit_mask=fit_mask)
            lines = []
            for i in range(n_ca):
                x, y, z = frame[i]
                b = float(bfac[i]) if bfac is not None else 0.5
                lines.append(
                    f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C")
            pdb_esc = "\n".join(lines).replace("\\", "\\\\").replace("`", "\\`")
            models_js.append(
                f'  var m{k}=v.addModel(`{pdb_esc}`,"pdb");\n'
                f'  m{k}.setStyle({{}},{{cartoon:{{'
                f'colorscheme:{{prop:"b",gradient:"bwr",min:0,max:4.0}},'
                f'thickness:0.5,opacity:0.30}}}});\n')

        ref_js = ""
        if self._landscape_dominant_particles:
            dom_pdb = self._build_ca_pdb_str(self._landscape_dominant_particles)
            dom_esc = dom_pdb.replace("\\", "\\\\").replace("`", "\\`")
            ref_js = (
                f'  var mRef=v.addModel(`{dom_esc}`,"pdb");\n'
                f'  mRef.setStyle({{}},{{cartoon:{{color:"#1d4ed8",'
                f'thickness:0.9,opacity:1.0}}}});\n')

        html = f"""<!DOCTYPE html><html><head>
<script src="{_3DMOL_JS_URL}"></script>
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
<div id="info">ENSEMBLE OVERLAY{f' &nbsp;&middot;&nbsp; BASIN {basin_idx+1}' if basin_idx is not None else ''} \
&nbsp;&middot;&nbsp; {len(sample_idx)} sampled conformers \
&nbsp;&middot;&nbsp; {n_ca} C&alpha; atoms</div>
<div id="legend">
  <span style="color:#1d4ed8">&#9632;</span> DOMINANT (reference)
  &nbsp;&middot;&middot;&middot;&middot;&middot;&nbsp;
  <span style="color:#2563eb">&#9632;</span> RIGID
  &nbsp;&middot;&nbsp;
  <span style="color:#dc2626">&#9632;</span> FLEXIBLE (per-conformer overlay)
</div>
<script>
(function(){{
  var v=$3Dmol.createViewer("v",{{backgroundColor:"#f8fafc"}});
{''.join(models_js)}{ref_js}
  v.zoomTo(); v.zoom(0.85); v.render();
  (function spin(){{ v.rotate(1,'y'); v.render(); requestAnimationFrame(spin); }})();
}})();
</script></body></html>"""

        basin_tag = f' (basin {basin_idx+1})' if basin_idx is not None else ''
        self.viewer_cand_lbl.setText(
            f'<span style="color:#7c3aed;font-weight:bold;">ENSEMBLE OVERLAY{basin_tag}</span>'
            f' &nbsp;·&nbsp; {len(sample_idx)} conformers &nbsp;·&nbsp; '
            'blue=rigid · red=disordered')
        self._set_html(html)
        self._view_stack.setCurrentIndex(0)
        self.view_mode_btn.setVisible(True)
        self.ensemble_toggle_btn.setText("⊚  ENSEMBLE")

    def _render_basin_overlay(self, basin_idx):
        """OVERLAY button in the basin pool (2026-07-13, item #7 re-test) --
        same spaghetti-plot view as _render_ensemble_overlay, restricted to
        one basin's own conformers via its member_indices (see run()'s
        basin_summary construction)."""
        self._render_ensemble_overlay(basin_idx=basin_idx)

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

    def _toggle_ensemble(self):
        if self._view_stack.currentIndex() == 3:
            self._view_stack.setCurrentIndex(0)
            self.view_mode_btn.setVisible(True)
            self.ensemble_toggle_btn.setText("⊚  ENSEMBLE")
            self.landscape_toggle_btn.setText("◈  LANDSCAPE")
        else:
            self._view_stack.setCurrentIndex(3)
            self.view_mode_btn.setVisible(False)
            self.ensemble_toggle_btn.setText("⊡  STRUCTURE")
            self.landscape_toggle_btn.setText("◈  LANDSCAPE")


# ═══════════════════════════════════════════════════════════════════
#  진입점 (Entry Point)
#  실행 흐름:
#    1. QApplication 생성 (Qt 이벤트 루프 초기화)
#    2. ProteinApp 윈도우 생성 — GPU 탐지, PhysicsEngine 초기화 포함
#    3. 윈도우 표시 후 Qt 이벤트 루프 시작 (sys.exit으로 정상 종료 코드 반환)
#    4. 예외 발생 시 error_log.txt에 스택 트레이스 저장
#  Entry flow: create QApplication → build window (GPU detect, engine init)
#              → show → Qt event loop → on exception write error_log.txt
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
