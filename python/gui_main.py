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
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtCore import QUrl
from Bio.PDB import PDBParser, PDBList, PDBIO
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import protein_physics
from amber_params import get_atom_params as _amber_get_params, ION_PARAMS, _WATER_RESNAMES
import iupred as _iupred


def _app_base_dir():
    """Root directory for locating the sibling ``data`` folder.

    Under a PyInstaller-frozen build ``__file__`` points inside the
    temporary/onefile extraction bundle, so the persistent cache dir must
    instead sit next to the executable to survive across runs.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

    for atom in st.get_atoms():
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

    # ── 결합 위상 그래프 구성 (Build covalent bond topology) ─────────────────
    # BondTopology는 HETATM 잔기(bond_templates에 없음)를 조용히 건너뛴다.
    # BondTopology.build() silently skips HETATM residues (not in bond_templates).
    topo = protein_physics.BondTopology()
    topo.build(meta_resnames, meta_atomnames, meta_residx)
    log(f"  topology: {topo.num_bonds} bonds · {topo.num_rot_bonds} rotatable"
        f" · {topo.num_concerted_pairs} crankshaft pairs")

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

def _aligned_pdb_text(path, ref_ca_map):
    """외부 구조(AlphaFold/SWISS-MODEL)를 ref_ca_map 프레임에 Kabsch 정렬한 뒤
    전체 원자 PDB 텍스트로 반환. 공통 Cα < 3개면 원본 텍스트를 그대로 반환.
    Kabsch-align an external structure (AlphaFold/SWISS-MODEL) onto ref_ca_map's
    frame and return the whole-structure PDB text. Falls back to the raw file
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
    writer.save(buf)
    return buf.getvalue()

def _compute_rmsf(snapshots, ca_indices):
    """Per-Cα root-mean-square fluctuation (Å) across all trajectory snapshots.
    궤적 스냅샷 전체에 걸친 잔기별 Cα RMSF (Å).

    snapshots   — list of particle-lists from LandscapeWorker
                  LandscapeWorker의 MC 스냅샷 목록 (각 원소 = 입자 목록)
    ca_indices  — index of each Cα atom inside each particle list
                  각 Cα 원자의 입자 목록 내 인덱스

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
                      "heavy_keys": heavy_keys}
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

# ═══════════════════════════════════════════════════════════════════
#  LandscapeWorker — MC trajectory → conformational graph
# ═══════════════════════════════════════════════════════════════════
class LandscapeWorker(QThread):
    progress = pyqtSignal(str)
    result   = pyqtSignal(dict)
    # See PipelineWorker.gpu_fallback -- same runtime-failure/CPU-retry pattern,
    # needed here too since this is the other long-running GPU MC call.
    gpu_fallback = pyqtSignal(str)

    N_SNAPSHOTS    = 120   # total trajectory length
    STEPS_PER_SNAP = 80    # MC steps between each snapshot

    def __init__(self, engine, init_atoms, ca_indices, topo, physics_mod=None, T=0.6, max_angle=0.12):
        super().__init__()
        self.engine      = engine
        self.init_atoms  = init_atoms
        self.ca_indices  = ca_indices
        self.topo        = topo
        self.physics_mod = physics_mod if physics_mod is not None else protein_physics
        self.T           = T
        self.max_angle   = max_angle

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

        N, S = self.N_SNAPSHOTS, self.STEPS_PER_SNAP
        self.progress.emit(
            f"  [LANDSCAPE] Running {N}×{S}-step Markov chain…")

        # run_landscape_trajectory() advances the whole N×S-step chain in a
        # single C++/CUDA call instead of looping here and calling
        # generate_ensemble()+calculate_potential() N times — each of those
        # Python-level iterations used to re-marshal the full particle array
        # across the Python<->C++ boundary (and, on the GPU backend,
        # reallocate device buffers from scratch) every single snapshot.
        # See IMPROVEMENTS.md item #12. The trade-off is coarser progress
        # feedback: this call blocks until all N snapshots are done rather
        # than reporting every 20 snapshots, since the loop itself now lives
        # in C++.
        try:
            snapshots, energies = self.engine.run_landscape_trajectory(
                self.init_atoms, self.topo, N, S, self.T, self.max_angle)
        except Exception as ex:
            if self.physics_mod is protein_physics:
                raise  # already on CPU -- nothing left to fall back to
            self.progress.emit(f"  ⚠ GPU engine failed at runtime ({ex}) — falling back to CPU")
            self.gpu_fallback.emit(str(ex))
            self.physics_mod = protein_physics
            self.engine = protein_physics.PhysicsEngine()
            snapshots, energies = self.engine.run_landscape_trajectory(
                self.init_atoms, self.topo, N, S, self.T, self.max_angle)
        energies = np.array(energies, dtype=float)

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
        step_disp   = np.linalg.norm(np.diff(layout, axis=0), axis=1)
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
        # e_spread = 유의미한 분지들의 최소 에너지 차이 → 지형의 '거칠기'.
        #
        # 분류 기준:
        #   IDP:               ≥3개 분지 AND e_spread < 5kT
        #                      (여러 분지가 비슷한 에너지 → 지형이 평탄 → 무질서)
        #   POSSIBLY DISORDERED: ≥2개 분지 OR 최저 분지가 전체의 50% 미만 차지
        #                        (부분적 무질서 또는 다중 접힘 상태)
        #   ORDERED:           단일 지배적 분지 (깔때기형 에너지 지형)
        #                      (funnel landscape → 단일 안정 구조)
        # IDP: ≥3 significant basins AND e_spread < 5kT (flat landscape → disordered)
        # ORDERED: single dominant basin (funnel landscape → stable native fold)
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
        self._topo               = None
        self._pdb_path           = None
        self._ca_indices         = []
        self._landscape_snaps    = []
        self._landscape_energies = np.array([])
        self._rmsf               = None
        self._rmsf_residues      = []
        self._rmsf_n_disordered  = 0
        self._rmsf_pct           = 0.0
        self._iupred_scores      = []
        self._ca_residues        = []
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
            pdb_best = self._build_pdb_str(self._ensemble[best_idx])
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
<div id="info">{label}</div>
{legend}
<script>
(function(){{
  var v=$3Dmol.createViewer("v",{{backgroundColor:"#f8fafc"}});
{best_js}
{ext_js}
  v.zoomTo(); v.zoom(0.85); v.render();
  setInterval(function(){{ v.rotate(1,'y'); v.render(); }},50);
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
  setInterval(function(){{ vL.rotate(1,'y'); vL.render(); }},50);
  var vR=$3Dmol.createViewer("vR",{{backgroundColor:"#f8fafc"}});
{right_js}
  vR.zoomTo(); vR.zoom(0.85); vR.render();
  setInterval(function(){{ vR.rotate(1,'y'); vR.render(); }},50);
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
        start_atoms = self._init_atoms
        branch_note = ""
        if self._ensemble and self._energies:
            best_idx = int(np.argmin(self._energies))
            start_atoms = self._ensemble[best_idx]
            branch_note = f" (branching from best candidate, {self._energies[best_idx]:.1f} kcal/mol)"

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

        self._landscape_worker = LandscapeWorker(
            ls_engine, start_atoms, self._ca_indices, self._topo, self._physics_mod)
        self._landscape_worker.progress.connect(self._log)
        self._landscape_worker.result.connect(self._on_landscape_done)
        self._landscape_worker.gpu_fallback.connect(self._on_gpu_fallback)
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
            self._draw_disorder_profile(rmsf=rmsf, residues=residues,
                                        iupred_scores=self._iupred_scores)
            self.disorder_toggle_btn.setEnabled(True)

        self._log(f"[LANDSCAPE] Classification: {lbl}  ·  "
                  f"{data['n_sig']} metastable basins  ·  "
                  f"funnel={data['funnel']:.2f}")

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
        if self._current_ext_entry is not None:
            self._render_external(self._current_ext_entry)
        elif self._ensemble:
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
