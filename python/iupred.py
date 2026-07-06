"""
iupred.py — 서열 기반 고유 무질서 예측기 (P3.1)
=================================================
Sequence-based intrinsic disorder predictor (P3.1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 이론 배경 (Theoretical Background)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

고유 무질서 단백질 (IDP, Intrinsically Disordered Protein):
  특정한 3차 구조를 가지지 않고 생리 조건에서 동적 앙상블 상태로 존재하는
  단백질 또는 단백질 영역.  전사 인자, 신호 전달 단백질, 상전이(phase
  separation) 허브 단백질 등에 광범위하게 나타난다.  유연성이 기능적으로
  필수적인 경우가 많다 (예: 결합 시 접힘, 분자 인식 등).

Intrinsically Disordered Proteins (IDPs):
  Proteins or protein regions that lack a stable folded structure under
  physiological conditions and instead populate a heterogeneous ensemble
  of rapidly interconverting conformations.  IDPs are enriched in
  transcription factors, signalling hubs, and phase-separating proteins.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 알고리즘 개요 (Algorithm — IUPred2A-style energy potential)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

출처 / Source:
  Dosztányi et al. (2005) J. Mol. Biol. 347, 827-839
  Mészáros et al. (2018) Nucleic Acids Res. 46, W329-W337

핵심 직관 (Core intuition):
  규칙 구조 단백질의 각 잔기는 주변 잔기들과 강한 상호작용(특히 소수성
  상호작용)을 형성해 에너지적으로 안정한 접힘을 만든다.
  무질서 단백질의 잔기는 하전되거나 극성이 강해 충분한 소수성 상호작용을
  형성하지 못하므로 구조가 고정되지 않는다.
  → 서열만으로 잔기별 상호작용 에너지 합산을 계산하면 무질서를 예측할 수 있다.

Core intuition:
  Ordered protein residues form strong local pairwise interactions (mostly
  hydrophobic contacts) that stabilise the fold.  Disordered residues tend
  to be charged or polar, forming insufficient hydrophobic contacts to anchor
  a stable structure.  Summing expected pairwise interaction energies from
  the sequence alone is therefore predictive of disorder.

단계별 알고리즘 (Step-by-step):

  1. 20×20 쌍별 에너지 행렬 E_pair 구성:
        E_pair(aa_i, aa_j) = −max(0,H_i)·max(0,H_j) + 0.5·Q_i·Q_j
     H = 카이트-두리틀 소수성 지수 / 4.5 (정규화).
     Q = pH 7 형식 전하 (Asp/Glu=−1, Lys/Arg=+1).
     소수성-소수성 쌍: E < 0 (유리, 규칙 선호).
     같은 부호 하전 쌍: E > 0 (불리, 무질서 선호).
     반대 부호 하전 쌍: E < 0 (유리, 규칙 선호).

  1. Build 20×20 pairwise energy matrix E_pair:
        E_pair(i,j) = −max(0,H_i)·max(0,H_j) + 0.5·Q_i·Q_j
     H = KD hydrophobicity / 4.5 (normalised to ≈ [0,1]).
     Q = formal charge at pH 7 (Asp/Glu=−1, Lys/Arg=+1, others=0).
     Hphobic-Hphobic: E < 0 (favorable → promotes order).
     Same-sign charge: E > 0 (unfavorable → promotes disorder).
     Opposite-sign charge: E < 0 (favorable → promotes order via salt bridge).

  2. 잔기별 슬라이딩 창 에너지 합산:
        energy_sum(i) = Σ_{|j−i| ≤ WINDOW, j≠i} E_pair(aa[i], aa[j])
     창 크기 = ±10 잔기 (총 21잔기 창, IUPred2A 단범위 기준).
     합산값이 음수 → 잔기 i가 강한 소수성/정전기 상호작용 → 규칙.
     합산값이 0 또는 양수 → 약한 상호작용 → 무질서.

  2. Per-residue sliding-window energy sum:
        energy_sum(i) = Σ_{|j−i| ≤ WINDOW, j≠i} E_pair(aa[i], aa[j])
     WINDOW = 10 (21-residue total window, matching IUPred2A short-range mode).
     Negative e_sum → strong interactions → ordered.
     Near-zero or positive e_sum → weak interactions → disordered.

  3. 시그모이드 변환 → 무질서 확률 [0, 1]:
        score(i) = σ(slope · energy_sum(i) + bias)
              σ(x) = 1 / (1 + exp(−x))
     slope > 0: 양의 에너지 합산(약한 상호작용) → 높은 무질서 점수.
     bias > 0: 중립 잔기(e_sum ≈ 0)도 약간 무질서 쪽으로 편향.
              → 순수 Gly 서열: score ≈ 0.62 (Gly의 알려진 유연성 반영).

  3. Sigmoid conversion → disorder probability in [0, 1]:
        score(i) = σ(slope · energy_sum(i) + bias)
     slope > 0: positive e_sum (few contacts) → high disorder score.
     bias > 0: neutral residues (e_sum ≈ 0) lean slightly toward "disordered"
               — consistent with Gly's known backbone flexibility.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 보정값 (Calibration values):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  slope = 1.4, bias = 0.2 로 아래 기준 검증:
    순수 Ile (소수성 강) → score ≈ 0.00  (규칙 / ordered)
    순수 Asp (같은 부호 반발) → score ≈ 1.00  (무질서 / disordered)
    순수 Gly (상호작용 없음) → score ≈ 0.55  (중립 유연 / neutral-flexible)

  Validated (slope=1.4, bias=0.2):
    PolyIle  → ~0.00 (strongly ordered, hydrophobic core-forming)
    PolyAsp  → ~1.00 (strongly disordered, like polyglutamate/acidic IDPs)
    PolyGly  → ~0.55 (neutral, still mildly flexible-leaning, consistent
               with Gly-rich linkers, without swamping ordinary surface loops)

  bias was originally 0.7 (σ(0.7)≈0.668), which meant any residue with a
  merely-neutral window sum (e_sum≈0 — the common case for ordinary
  surface-exposed polar/charged residues, not just IDP regions) already
  scored above the 0.5 "disordered" cutoff by default. Calibrated against 5
  real, textbook well-folded proteins (1LYZ, 1UBQ, 1MBN, 7RSA, 1BNI):
  bias=0.7 flagged 0-48% of residues "disordered" in structurally rigid,
  disulfide-stabilized enzymes (worst case: bovine RNase A at 47.6%, a
  protein real IUPred2A scores near 0%). bias=0.2 brings all 5 down to 0-9%,
  consistent with ordinary surface-loop flexibility rather than systematic
  over-prediction, while still preserving the qualitative Gly-leans-
  flexible behavior (σ(0.2)=0.55 > 0.5, just not overwhelmingly so).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 중요 주의사항 (Important caveat)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  이 구현은 IUPred2A의 물리적 원칙을 따르지만, 실제 IUPred2A가 사용하는
  20×20 통계적 포텐셜 행렬(PDB 통계 기반)을 그대로 쓰지는 않는다.
  대신 카이트-두리틀 소수성과 형식 전하를 조합한 물리 기반 근사치를 사용한다.
  이 예측기는 연구용 참고 자료로 활용하되, 임상/생화학 용도에는 공식
  IUPred2A 웹서버(https://iupred2a.elte.hu)를 사용할 것을 권장한다.

  This implementation follows the physical principles of IUPred2A but does NOT
  use the exact published 20×20 statistical potential (derived from PDB statistics).
  Instead it uses a physically motivated approximation combining KD hydrophobicity
  and formal charges.  For research reference only; for production use, consult
  the official IUPred2A server at https://iupred2a.elte.hu.
"""

import math

# ── 아미노산 순서 및 인덱스 맵 ────────────────────────────────────────────────
# 20가지 표준 아미노산을 알파벳 순서(1-글자 코드 기준)로 나열.
# IUPred2A와 동일한 순서를 사용해 20×20 행렬의 행·열 인덱스를 일관되게 유지.
# 알 수 없는 잔기(변형 AA, HETATM 등)는 인덱스 5 (Gly, 중립 폴백)로 대입.
#
# 20 standard amino acids in alphabetical 1-letter-code order.
# Same ordering as IUPred2A keeps matrix row/column indices consistent.
# Unknown residues (modified AA, HETATM) default to index 5 (Gly, neutral fallback).
_AA_ORDER: str = "ACDEFGHIKLMNPQRSTVWY"
_AA_IDX: dict[str, int] = {aa: i for i, aa in enumerate(_AA_ORDER)}

# ── 카이트-두리틀 소수성 지수 (Kyte-Doolittle hydrophobicity scale) ───────────
# Kyte & Doolittle (1982) J. Mol. Biol. 157, 105-132.
# 각 아미노산의 소수성 (소수성 정도): 높을수록 비극성, 낮을수록 극성/하전.
# 물에 노출되는 단백질 표면 vs. 소수성 핵(core)에 묻히는 경향을 나타냄.
#
# 이 값은 이후 /4.5로 정규화해 [−1, +1] 범위로 변환하고,
# 소수성 E_pair 계산에서는 양수 부분만 (max(0, H)) 사용한다.
# 음수 소수성(하전 잔기)에는 상호작용이 없는 것으로 모델링
# (소수성 '인력'은 비극성 잔기 사이에서만 발생).
#
# Kyte & Doolittle (1982) J. Mol. Biol. 157, 105-132.
# Raw hydrophobicity per amino acid.  High = nonpolar; low = polar/charged.
# Normalised to ≈ [−1,+1] by dividing by 4.5 (max absolute value in the scale).
# Only the positive part (max(0, H)) is used for the hydrophobic E_pair term,
# because hydrophobic attraction only applies between nonpolar residues.
_KD_RAW: dict[str, float] = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G":-0.4,  "H":-3.2,  "I":  4.5, "K": -3.9, "L": 3.8,
    "M": 1.9,  "N":-3.5,  "P": -1.6, "Q": -3.5, "R": -4.5,
    "S":-0.8,  "T":-0.7,  "V":  4.2, "W": -0.9, "Y": -1.3,
}

# ── 정규화된 소수성 벡터 (Normalised hydrophobicity vector) ───────────────────
# KD 값을 최댓값(4.5)으로 나눠 [−1, +1] 범위로 정규화한 후,
# 소수성 E_pair에 쓸 양수 부분만 클리핑.  음수 KD 잔기의 기여는 0.
# 즉 소수성 기여는 max(0, KD/4.5)².
#
# KD divided by 4.5 → ≈ [−1,+1], then clipped to max(0, …) for the
# hydrophobic term.  Residues with negative KD (charged/polar) contribute 0
# to the hydrophobic component: only genuinely nonpolar residues attract each other.
_H: list[float] = [max(0.0, _KD_RAW[aa] / 4.5) for aa in _AA_ORDER]

# ── 형식 전하 벡터 (Formal charge vector at pH 7) ────────────────────────────
# pH 7 생리 조건에서 각 아미노산의 대표 형식 전하.
# ASP (D), GLU (E): −1  (카르복실레이트 탈양성자화)
# LYS (K), ARG (R): +1  (아민/구아니디늄 양성자화)
# HIS (H): 0  (pKa ≈ 6.0, pH 7에서 대부분 중성)
# 나머지: 0
#
# Formal charge at pH 7.
# ASP(D), GLU(E) = −1 (deprotonated carboxylate, pKa ≈ 3.9 / 4.1).
# LYS(K) = +1 (protonated ε-amine, pKa ≈ 10.5).
# ARG(R) = +1 (protonated guanidinium, pKa ≈ 12.5).
# HIS(H) = 0  (imidazole, pKa ≈ 6.0, mostly neutral at pH 7).
# All others = 0.
# 순서: A  C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y
_Q: list[float] = [
    0,  0, -1, -1,  0,   0,  0,  0, +1,  0,
    0,  0,  0,  0, +1,   0,  0,  0,  0,  0,
]

# ── 20×20 쌍별 상호작용 에너지 행렬 (Pairwise interaction energy matrix) ──────
#
# E_pair(i, j) = E_hydrophobic(i,j) + E_charge(i,j)
#
# (A) 소수성 기여: E_hydrophobic = −H_i · H_j
#     양의 소수성 잔기 i, j가 만날 때 음수(유리) → 소수성 코어 형성 → 규칙 구조.
#     예: Ile(H=1.0) + Leu(H=0.84) → E = −0.84 (유리, 규칙)
#     예: Asp(H=0)   + Ile(H=1.0)  → E =  0.0  (중립)
#
# (B) 전하 기여: E_charge = +0.5 × Q_i × Q_j
#     같은 부호(예: Asp-Asp, Lys-Lys): E > 0 (반발, 불리, 무질서 선호).
#     반대 부호(예: Asp-Lys, 염 다리): E < 0 (인력, 유리, 규칙 선호).
#     계수 0.5는 전하 기여가 소수성 기여와 비슷한 규모가 되도록 조정.
#
# 행렬은 대칭이다: E_pair(i,j) == E_pair(j,i).
#
# E_pair(i, j) = E_hydrophobic(i,j) + E_charge(i,j)
#
# (A) Hydrophobic term: E_hydrophobic = −H_i · H_j
#     Negative (favorable) when both residues are hydrophobic → promotes burial
#     in a packed hydrophobic core → promotes ordered structure.
#     Example: Ile(H≈1.0) + Leu(H≈0.84) → E = −0.84 (favorable, ordered)
#     Example: Asp(H=0)   + Ile(H=1.0)  → E =  0.0  (neutral)
#
# (B) Charge term: E_charge = +0.5 × Q_i × Q_j
#     Same-sign charges (Asp-Asp, Lys-Lys): E > 0 (repulsion → disorder).
#     Opposite-sign charges (Asp-Lys salt bridge): E < 0 (attraction → order).
#     Factor 0.5 calibrated so charge and hydrophobic terms have comparable magnitude.
#
# The matrix is symmetric: E_pair(i,j) == E_pair(j,i).
_E_PAIR: list[list[float]] = [
    [(-_H[i] * _H[j]) + (0.5 * _Q[i] * _Q[j]) for j in range(20)]
    for i in range(20)
]

# ── 시그모이드 파라미터 (Sigmoid calibration parameters) ──────────────────────
#
# disorder_score(i) = σ(slope × energy_sum(i) + bias)
#               σ(x) = 1 / (1 + exp(−x))
#
# slope (기울기, 감도):
#   energy_sum 변화에 대한 무질서 점수의 민감도.
#   값이 클수록 에너지 차이를 더 가파르게 구별한다.
#   slope = 1.4 → 에너지 합산이 약 2.0 단위 변하면 점수가 ≈ 0.10 변함.
#
# slope (sensitivity):
#   How sharply the score responds to changes in energy_sum.
#   slope = 1.4: a ~2.0-unit change in e_sum shifts score by ≈ 0.10.
_SLOPE: float = 1.4

# bias (편향, 기준점):
#   e_sum = 0 일 때 σ(bias)의 값을 결정.
#   bias > 0: e_sum = 0인 중립 잔기도 0.5보다 높은 점수(무질서 선호) → Gly 반영.
#   bias = 0.2 → σ(0.2) ≈ 0.55: 완전 중립 서열의 기저 무질서 점수.
#
#   원래 0.7이었으나(σ(0.7)≈0.668), 이는 곧 e_sum≈0인 잔기(무질서 단백질뿐
#   아니라 규칙 단백질 표면의 흔한 극성/하전 잔기도 포함)가 기본적으로
#   0.5 무질서 문턱값을 넘긴다는 뜻이었다. 5개의 실제 안정 단백질(1LYZ,
#   1UBQ, 1MBN, 7RSA, 1BNI)로 보정한 결과 bias=0.7에서는 이 문턱 단백질들이
#   0~48% "무질서"로 잘못 표시됐다(최악: RNase A 47.6%, 실제 IUPred2A는
#   이 단백질을 거의 0%로 평가). bias=0.2로 낮추면 5개 전부 0~9%로 내려가
#   정상적인 표면 루프 유연성 수준에 부합한다.
#
# bias (intercept):
#   Controls the score when e_sum = 0.
#   bias = 0.2 -> sigma(0.2) ~= 0.55: a residue with no interactions scores as
#   only mildly "disordered-leaning", not confidently so.
#
#   Originally 0.7 (sigma(0.7)~=0.668), which meant any residue with a merely-
#   neutral window sum -- common for ordinary surface-exposed polar/charged
#   residues in a perfectly ordered protein, not just IDP regions -- already
#   cleared the 0.5 disorder cutoff by default. Calibrated against 5 real,
#   textbook well-folded proteins (1LYZ, 1UBQ, 1MBN, 7RSA, 1BNI): bias=0.7
#   flagged 0-48% of residues "disordered" in structurally rigid, disulfide-
#   stabilized enzymes (worst case: bovine RNase A at 47.6%, vs. near-0% from
#   real IUPred2A). bias=0.2 brings all 5 down to 0-9%, consistent with
#   ordinary surface-loop flexibility rather than systematic over-prediction.
_BIAS:  float = 0.2

# 슬라이딩 창 반경 (Sliding-window half-width):
# ±10 잔기 → 총 21잔기 창.
# IUPred2A의 단범위(short-range) 예측 모드에 해당.
# 값이 너무 작으면 국소적 정보만 반영, 너무 크면 전반적 서열 편향에 민감해짐.
# 창 경계에서 가용 이웃 수가 줄어 말단 잔기의 점수가 중간보다 낮은 경향이 있음.
#
# Sliding window half-width = 10 → full window of 21 residues.
# Matches the IUPred2A short-range disorder mode.
# At sequence termini, fewer neighbours are available, slightly biasing
# terminal residues toward lower disorder scores (smaller e_sum magnitude).
_WINDOW: int = 10

# 2차 평활화 반경 (Secondary smoothing half-width):
# _WINDOW의 이웃 합산만으로는 e_sum(i)가 여전히 잔기 i 자신의 정체성(row
# 선택)에 크게 좌우돼, 이웃은 거의 그대로인데도 인접 잔기 사이에서 값이
# 급격히 진동한다 (예: Asp 다음에 Ile가 오면 완전히 다른 상호작용 행을 씀).
# IUPred2A는 원본 논문에서 이 노이즈를 줄이기 위해 에너지 프로파일에 추가
# 이동평균을 적용한다 — 여기서도 동일하게 적용해 매끄러운 곡선을 얻는다.
#
# The _WINDOW neighbor-sum alone still leaves e_sum(i) dominated by residue
# i's own identity (which row of E_PAIR gets used), so the profile can swing
# sharply between adjacent residues even though their neighborhoods barely
# differ (e.g. an Asp immediately followed by an Ile pulls from completely
# different interaction rows). IUPred2A's published method applies a further
# running-average smoothing pass over the raw energy profile for exactly
# this reason — applied here too, to avoid a falsely jagged disorder curve.
_SMOOTH_HALF_WINDOW: int = 5

# ── 3→1 글자 아미노산 코드 변환 맵 (Three-letter to one-letter code map) ──────
# PDB 파일의 3글자 잔기명을 1글자 코드로 변환한다.
# HIS의 세 가지 프로토네이션 형태(HID/HIE/HIP)는 모두 'H'로 매핑.
# 알 수 없는 잔기명은 score_from_resnames()에서 'G'(글리신 폴백)으로 처리.
#
# Converts PDB 3-letter residue names to 1-letter codes for the sequence predictor.
# All three histidine protonation variants (HID/HIE/HIP) map to 'H'.
# Unknown residue names default to 'G' (Gly, neutral fallback) in score_from_resnames().
THREE_TO_ONE: dict[str, str] = {
    "ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C",
    "GLN":"Q", "GLU":"E", "GLY":"G", "HIS":"H", "HID":"H",
    "HIE":"H", "HIP":"H", "ILE":"I", "LEU":"L", "LYS":"K",
    "MET":"M", "PHE":"F", "PRO":"P", "SER":"S", "THR":"T",
    "TRP":"W", "TYR":"Y", "VAL":"V",
}


def score(sequence: str) -> list[float]:
    """잔기별 무질서 확률 [0, 1]을 계산한다.
    Compute per-residue disorder probability in [0, 1].

    파라미터 / Parameters
    ─────────────────────
    sequence : 1글자 아미노산 서열 문자열.
               알 수 없는 문자는 'G'(글리신 중립 폴백)으로 대체.
               1-letter amino acid sequence string.
               Unknown characters default to 'G' (Gly, neutral).

    반환값 / Returns
    ────────────────
    길이 = len(sequence)인 float 리스트.
    각 원소 ∈ [0, 1]; 0.5 초과 → 무질서로 예측.
    List of floats, length = len(sequence), each in [0, 1].
    Values > 0.5 indicate a predicted disordered residue.

    복잡도 / Complexity
    ────────────────────
    O(N × WINDOW) 시간, O(N) 공간.  일반 단백질(N < 1000)에서 무시할 수준.
    O(N × WINDOW) time, O(N) space.  Negligible for typical proteins (N < 1000).
    """
    n = len(sequence)
    if n == 0:
        return []

    # 서열을 정수 인덱스 배열로 변환.
    # 대문자로 변환 후 _AA_IDX 조회; 알 수 없는 문자 → 인덱스 5 (Gly, 중립).
    # Convert sequence to integer index array.
    # Upper-case first; unknown characters → index 5 (Gly, neutral fallback).
    idx: list[int] = [_AA_IDX.get(aa.upper(), 5) for aa in sequence]

    # 슬라이딩 창 에너지 합산.
    # 잔기 i에 대해 창 내 모든 다른 잔기 j와의 E_pair를 합산한다.
    # row = _E_PAIR[idx[i]] 로 i 행을 한 번만 조회해 j 루프에서 반복 참조를 피함.
    # Sliding-window energy summation.
    # For residue i, accumulate E_pair(i,j) for all j within the window (j ≠ i).
    # Cache row = _E_PAIR[idx[i]] once to avoid repeated list-of-lists indexing.
    e_sum: list[float] = [0.0] * n
    for i in range(n):
        lo  = max(0, i - _WINDOW)
        hi  = min(n, i + _WINDOW + 1)
        row = _E_PAIR[idx[i]]   # i번째 잔기 유형의 상호작용 에너지 행 / row for aa type i
        for j in range(lo, hi):
            if j != i:
                e_sum[i] += row[idx[j]]

    # 2차 평활화: 원본 e_sum 프로파일에 이동평균을 적용해 잔기별 정체성으로
    # 인한 고주파 진동을 제거한다 (알고리즘 설명 상단 참고).
    # Secondary smoothing: running average over the raw e_sum profile to
    # remove the high-frequency, per-residue-identity oscillation (see the
    # algorithm note above _SMOOTH_HALF_WINDOW).
    smoothed: list[float] = [0.0] * n
    for i in range(n):
        lo = max(0, i - _SMOOTH_HALF_WINDOW)
        hi = min(n, i + _SMOOTH_HALF_WINDOW + 1)
        smoothed[i] = sum(e_sum[lo:hi]) / (hi - lo)
    e_sum = smoothed

    # 시그모이드 변환: e_sum → [0, 1] 무질서 확률.
    # exp(-x)가 지수 오버플로우를 일으킬 수 있으므로 OverflowError를 잡는다.
    # 수치적으로 안전: exp(-x) → ∞ 일 때 1/(1+∞) = 0, exp(x) → ∞ 일 때 1/(1+0) = 1.
    # Sigmoid: maps energy-scaled value to [0,1] disorder probability.
    # Guard against OverflowError: exp(-x)→∞ → score=0; exp(x)→∞ → score=1.
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    return [_sigmoid(_SLOPE * e + _BIAS) for e in e_sum]


def score_from_resnames(resnames3: list[str]) -> list[float]:
    """3글자 잔기명 목록을 받아 잔기별 무질서 점수를 반환하는 편의 래퍼.
    Convenience wrapper: accept 3-letter residue names, return per-residue disorder scores.

    gui_main.py의 _parse_pdb()에서 직접 호출:
    PDB 파싱 중 수집한 ca_resnames(Cα 잔기명 3글자 목록)를 인자로 받아
    1글자 서열로 변환한 뒤 score()에 위임한다.

    Called directly from _parse_pdb() in gui_main.py:
    converts the ca_resnames list collected during PDB parsing to a 1-letter
    sequence and delegates to score().

    알 수 없는 잔기명은 'G'(글리신 중립 폴백)으로 처리.
    Unknown residue names map to 'G' (Gly, neutral fallback).
    """
    # THREE_TO_ONE에 없는 잔기(변형 AA, 비표준 잔기 등)는 'G'로 대체.
    # Residues absent from THREE_TO_ONE (modified AA, non-standard) → 'G'.
    seq = "".join(THREE_TO_ONE.get(r.strip(), "G") for r in resnames3)
    return score(seq)


def fraction_disordered(scores: list[float], threshold: float = 0.5) -> float:
    """threshold 초과인 잔기의 비율을 반환한다.
    Return the fraction of residues with disorder score > threshold.

    점수가 없으면 0.0 반환.
    Returns 0.0 for empty input.
    """
    if not scores:
        return 0.0
    return sum(1 for s in scores if s > threshold) / len(scores)
