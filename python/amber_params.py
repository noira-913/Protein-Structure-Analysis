"""
amber_params.py — AMBER ff14SB parameter tables for all 20 standard amino acids.
==================================================================================
Source: Maier et al. (2015) J. Chem. Theory Comput. 11, 3696–3713.
        parm10.dat + frcmod.ff14SB + amino19.lib (AMBER 2019 distribution)

Provides three dicts and one public function:

  VDW_PARAMS      : atom_type → (R* Å, ε kcal/mol)
  ATOM_TYPE       : (resname, atomname) → atom_type string
  PARTIAL_CHARGES : (resname, atomname) → partial charge (e)

  get_atom_params(resname, atomname)
      Returns (charge, radius, epsilon) for use in Particle construction.
      Falls back gracefully for unknown residues or atom names.

Scope:
  • 20 standard internal (non-terminal) residues, heavy atoms only.
  • HIS modelled as HID (Nδ1 protonated) by default; HIE/HIP variants
    included as separate resname keys.
  • Terminal residues (NALA, CALA, ACE, NME, etc.) — TODO: extend later.
  • Disulfide CYX, protonated ASPH/GLUH — TODO.

Atom-type naming follows ff14SB conventions:
  CX  Cα (backbone alpha carbon, ff14SB-specific)
  C8  sidechain sp3 CH  (one H, e.g. Val Cβ, Ile Cβ)
  2C  sidechain sp3 CH₂ (two H)
  3C  sidechain sp3 CH₃ (three H, methyl)
  CT  aliphatic sp3 C   (legacy; retained for rings and special cases)
  C   carbonyl C (backbone + Asn/Gln/Asp/Glu CD/CG)
  CA  aromatic C (Phe/Tyr/Trp/His ring)
  CB  aromatic bridgehead C
  CC  aromatic C in imidazole ring (His)
  CN  aromatic C at fused ring junction (Trp)
  C*  sp2 C in indole ring (Trp CG)
  CR  aromatic CH at position 2 of imidazole (His CE1)
  CV  aromatic C in 5-membered ring adjacent to N (His)
  CW  aromatic C in 5-membered ring (His CD2, Trp CE2/CZ2)
  N   backbone sp2 amide N
  N2  sp2 N with 2 H (Arg guanidinium NH1/NH2)
  N3  sp3 N (Lys NZ — protonated amino)
  NA  aromatic N-H (His ND1 in HID, NE2 in HIE; Trp NE1)
  NB  aromatic N lone-pair (His NE2 in HID, ND1 in HIE)
  O   carbonyl O (backbone, Asn/Gln OD1/OE1)
  O2  carboxylate O (Asp OD1/OD2, Glu OE1/OE2, C-terminus)
  OH  hydroxyl O (Ser OG, Thr OG1, Tyr OH)
  S   thioether S (Met SD)
  SH  thiol S (Cys SG)
"""

# ── VDW parameters (R* Å, ε kcal/mol) ────────────────────────────────────────
# From parm10.dat.  R* is the radius of the atom (σ/2 in LJ notation).
# The combined pair σ = R*_i + R*_j;  ε_pair = sqrt(ε_i · ε_j).
# 반 데르 발스 매개변수: R* = 원자 반경(Å), ε = 우물 깊이(kcal/mol)
# 결합 쌍: σ_ij = R*_i + R*_j,  ε_ij = sqrt(ε_i · ε_j)

VDW_PARAMS: dict[str, tuple[float, float]] = {
    # ── Carbon types ───────────────────────────────────────────────
    "C":   (1.9080, 0.0860),   # sp2 carbonyl C (backbone, Asn/Gln/Asp/Glu)
    "CA":  (1.9080, 0.0860),   # aromatic C (Phe, Tyr, His, Trp rings)
    "CB":  (1.9080, 0.0860),   # aromatic bridgehead C (Trp CD2, Phe/Tyr CG)
    "CC":  (1.9080, 0.0860),   # aromatic C in imidazole (His)
    "CN":  (1.9080, 0.0860),   # aromatic C fused-ring junction (Trp CE2)
    "CR":  (1.9080, 0.0860),   # aromatic CH at imidazole C2 (His CE1)
    "CT":  (1.9080, 0.1094),   # aliphatic sp3 C (Pro ring, some sidechains)
    "CX":  (1.9080, 0.1094),   # Cα — ff14SB-specific (identical VDW to CT)
    "C8":  (1.9080, 0.1094),   # sidechain sp3 CH  (ff14SB, 1 H)
    "2C":  (1.9080, 0.1094),   # sidechain sp3 CH₂ (ff14SB, 2 H)
    "3C":  (1.9080, 0.1094),   # sidechain sp3 CH₃ (ff14SB, methyl)
    "C*":  (1.9080, 0.0860),   # sp2 C in indole (Trp CG)
    "CV":  (1.9080, 0.0860),   # aromatic C, 5-ring, adj to N (His)
    "CW":  (1.9080, 0.0860),   # aromatic C, 5-ring (His/Trp)
    # ── Nitrogen types ─────────────────────────────────────────────
    "N":   (1.8240, 0.1700),   # backbone sp2 amide N (all except Pro N-term)
    "N2":  (1.8240, 0.1700),   # sp2 N with 2 H (Arg NH1, NH2; Asn ND2; Gln NE2)
    "N3":  (1.8750, 0.1700),   # sp3 N (Lys NZ, protonated)
    "NA":  (1.8240, 0.1700),   # aromatic N-H (His ND1/NE2, Trp NE1)
    "NB":  (1.8240, 0.1700),   # aromatic N lone-pair (His unprotonated N)
    "NC":  (1.8240, 0.1700),   # sp2 N in guanidinium, no H (Arg NE)
    # ── Oxygen types ───────────────────────────────────────────────
    "O":   (1.6612, 0.2100),   # carbonyl O (backbone, Asn OD1, Gln OE1)
    "O2":  (1.6612, 0.2100),   # carboxylate O⁻ (Asp, Glu, C-terminus)
    "OH":  (1.7210, 0.2104),   # hydroxyl O (Ser, Thr, Tyr)
    # ── Sulfur types ───────────────────────────────────────────────
    "S":   (2.0000, 0.2500),   # thioether S (Met SD)
    "SH":  (2.0000, 0.2500),   # thiol S (Cys SG; reduced form)
    # ── Hydrogen types (included for completeness; PDB often omits H) ──
    "H":   (0.6000, 0.0157),   # backbone amide N-H
    "H1":  (1.3870, 0.0157),   # H on sp3 C with 1 electron-withdrawing neighbour
    "H4":  (1.4090, 0.0150),   # aromatic H (1 EWG neighbour)
    "H5":  (1.3590, 0.0150),   # aromatic H (2 EWG neighbours, His)
    "HA":  (1.4590, 0.0150),   # H on aromatic C
    "HC":  (1.4870, 0.0157),   # H on aliphatic C (methyl/methylene)
    "HO":  (0.3000, 0.0000),   # hydroxyl O-H
    "HP":  (1.1000, 0.0157),   # H on C bonded to N
    "HS":  (0.6000, 0.0157),   # thiol S-H
}

# Fallback for unknown types
_VDW_FALLBACK = (1.9000, 0.1000)

# ── Atom-type assignments ─────────────────────────────────────────────────────
# (resname, atomname) → AMBER ff14SB atom type string
# Backbone atoms appear in every entry; sidechains are residue-specific.
# PDB atom names follow the standard IUPAC-IUB / PDB v3 convention.
# 잔기명·원자명 쌍을 AMBER ff14SB 원자 유형 문자열로 매핑.
# 백본 원자는 모든 항목에 포함; 곁사슬은 잔기별로 다름.

# Helper: backbone atoms shared by all residues (injected below)
_BB: dict[str, str] = {
    "N":  "N",    "H":  "H",
    "CA": "CX",   "HA": "H1",
    "C":  "C",    "O":  "O",
    "OXT": "O2",  # C-terminal carboxylate (present when terminal)
}

ATOM_TYPE: dict[tuple[str, str], str] = {}

def _add(resname: str, sidechain: dict[str, str]) -> None:
    """Register backbone + sidechain type entries for one residue."""
    for atom, atype in {**_BB, **sidechain}.items():
        ATOM_TYPE[(resname, atom)] = atype

# ── Glycine ───────────────────────────────────────────────────────────────────
# Gly에는 Cβ가 없어 HA가 두 개(HA2, HA3)임. Cα는 CX지만 H는 H1.
_add("GLY", {"HA2": "H1", "HA3": "H1"})  # no sidechain; override HA with HA2/HA3

# ── Alanine ───────────────────────────────────────────────────────────────────
# Cβ = 3C (methyl), 3 × HB = HC
_add("ALA", {
    "CB": "3C",
    "HB1": "HC", "HB2": "HC", "HB3": "HC",
})

# ── Valine ────────────────────────────────────────────────────────────────────
# Cβ = C8 (CH), CG1/CG2 = 3C (two methyls)
_add("VAL", {
    "CB": "C8",  "HB": "HC",
    "CG1": "3C", "HG11": "HC", "HG12": "HC", "HG13": "HC",
    "CG2": "3C", "HG21": "HC", "HG22": "HC", "HG23": "HC",
})

# ── Leucine ───────────────────────────────────────────────────────────────────
# Cβ = 2C, Cγ = CT (branching CH), Cδ1/Cδ2 = 3C
_add("LEU", {
    "CB": "2C",  "HB2": "HC", "HB3": "HC",
    "CG": "CT",  "HG":  "HC",
    "CD1": "3C", "HD11": "HC", "HD12": "HC", "HD13": "HC",
    "CD2": "3C", "HD21": "HC", "HD22": "HC", "HD23": "HC",
})

# ── Isoleucine ────────────────────────────────────────────────────────────────
# Cβ = C8, Cγ1 = 2C, Cγ2 = 3C, Cδ1 = 3C
_add("ILE", {
    "CB": "C8",  "HB":  "HC",
    "CG1": "2C", "HG12": "HC", "HG13": "HC",
    "CG2": "3C", "HG21": "HC", "HG22": "HC", "HG23": "HC",
    "CD1": "3C", "HD11": "HC", "HD12": "HC", "HD13": "HC",
})

# ── Proline ───────────────────────────────────────────────────────────────────
# N is tertiary (no H); all ring carbons = CT (ring sp3)
ATOM_TYPE[("PRO", "N")]  = "N"    # tertiary amide N (slightly different charges)
ATOM_TYPE[("PRO", "CA")] = "CX"
ATOM_TYPE[("PRO", "HA")] = "H1"
ATOM_TYPE[("PRO", "C")]  = "C"
ATOM_TYPE[("PRO", "O")]  = "O"
ATOM_TYPE[("PRO", "CB")] = "CT";  ATOM_TYPE[("PRO","HB2")] = "HC"; ATOM_TYPE[("PRO","HB3")] = "HC"
ATOM_TYPE[("PRO", "CG")] = "CT";  ATOM_TYPE[("PRO","HG2")] = "HC"; ATOM_TYPE[("PRO","HG3")] = "HC"
ATOM_TYPE[("PRO", "CD")] = "CT";  ATOM_TYPE[("PRO","HD2")] = "HC"; ATOM_TYPE[("PRO","HD3")] = "HC"

# ── Phenylalanine ─────────────────────────────────────────────────────────────
# All ring carbons = CA (aromatic); Cβ = 2C
_add("PHE", {
    "CB": "2C",  "HB2": "HC", "HB3": "HC",
    "CG": "CA",
    "CD1": "CA", "HD1": "HA",
    "CD2": "CA", "HD2": "HA",
    "CE1": "CA", "HE1": "HA",
    "CE2": "CA", "HE2": "HA",
    "CZ":  "CA", "HZ":  "HA",
})

# ── Tryptophan ────────────────────────────────────────────────────────────────
# Indole ring: 5+6 fused. CG = C* (sp2 junction to ring), CD1 = CW,
# CD2 = CB (6-ring side of ring junction), NE1 = NA, CE2 = CN (ring junction),
# CE3/CZ3/CH2 = CA (6-ring), CZ2 = CW (5-ring)
_add("TRP", {
    "CB": "2C",  "HB2": "HC", "HB3": "HC",
    "CG":  "C*",
    "CD1": "CW", "HD1": "H4",
    "CD2": "CB",
    "NE1": "NA", "HE1": "H",
    "CE2": "CN",
    "CE3": "CA", "HE3": "HA",
    "CZ2": "CW", "HZ2": "HA",
    "CZ3": "CA", "HZ3": "HA",
    "CH2": "CA", "HH2": "HA",
})

# ── Serine ────────────────────────────────────────────────────────────────────
# OG = OH (hydroxyl oxygen)
_add("SER", {
    "CB": "2C",  "HB2": "H1", "HB3": "H1",
    "OG": "OH",  "HG":  "HO",
})

# ── Threonine ─────────────────────────────────────────────────────────────────
# Cβ = C8 (chiral CH), OG1 = OH, CG2 = 3C (methyl)
_add("THR", {
    "CB":  "C8", "HB":  "H1",
    "OG1": "OH", "HG1": "HO",
    "CG2": "3C", "HG21": "HC", "HG22": "HC", "HG23": "HC",
})

# ── Cysteine (reduced; protonated SH) ────────────────────────────────────────
# SG = SH; use CYX for disulfide (TODO)
_add("CYS", {
    "CB": "2C",  "HB2": "H1", "HB3": "H1",
    "SG": "SH",  "HG":  "HS",
})

# ── Methionine ────────────────────────────────────────────────────────────────
# SD = S (thioether); CE = 3C (methyl)
_add("MET", {
    "CB": "2C",  "HB2": "HC", "HB3": "HC",
    "CG": "2C",  "HG2": "HC", "HG3": "HC",
    "SD": "S",
    "CE": "3C",  "HE1": "HC", "HE2": "HC", "HE3": "HC",
})

# ── Aspartate (deprotonated, −1) ──────────────────────────────────────────────
# CG = C (carboxylate C); OD1/OD2 = O2 (both equivalent in deprotonated form)
_add("ASP", {
    "CB": "2C",  "HB2": "HC", "HB3": "HC",
    "CG": "C",
    "OD1": "O2", "OD2": "O2",
})

# ── Asparagine ────────────────────────────────────────────────────────────────
# CG = C; OD1 = O (carbonyl); ND2 = N2 (amide NH₂)
_add("ASN", {
    "CB":  "2C", "HB2": "HC", "HB3": "HC",
    "CG":  "C",
    "OD1": "O",
    "ND2": "N2", "HD21": "H", "HD22": "H",
})

# ── Glutamate (deprotonated, −1) ──────────────────────────────────────────────
_add("GLU", {
    "CB": "2C",  "HB2": "HC", "HB3": "HC",
    "CG": "2C",  "HG2": "HC", "HG3": "HC",
    "CD": "C",
    "OE1": "O2", "OE2": "O2",
})

# ── Glutamine ─────────────────────────────────────────────────────────────────
_add("GLN", {
    "CB": "2C",  "HB2": "HC", "HB3": "HC",
    "CG": "2C",  "HG2": "HC", "HG3": "HC",
    "CD": "C",
    "OE1": "O",
    "NE2": "N2", "HE21": "H", "HE22": "H",
})

# ── Lysine (+1) ───────────────────────────────────────────────────────────────
# NZ = N3 (protonated sp3 amino)
_add("LYS", {
    "CB": "2C",  "HB2": "HC", "HB3": "HC",
    "CG": "2C",  "HG2": "HC", "HG3": "HC",
    "CD": "2C",  "HD2": "HC", "HD3": "HC",
    "CE": "2C",  "HE2": "HP", "HE3": "HP",
    "NZ": "N3",  "HZ1": "H",  "HZ2": "H",  "HZ3": "H",
})

# ── Arginine (+1) ─────────────────────────────────────────────────────────────
# NE = NC (sp2 guanidinium N, no H directly... actually NE does have HE)
# CZ = CA-like (planar guanidinium C); NH1/NH2 = N2
_add("ARG", {
    "CB": "2C",  "HB2": "HC", "HB3": "HC",
    "CG": "2C",  "HG2": "HC", "HG3": "HC",
    "CD": "2C",  "HD2": "HC", "HD3": "HC",
    "NE": "N2",  "HE":  "H",
    "CZ": "CA",
    "NH1": "N2", "HH11": "H", "HH12": "H",
    "NH2": "N2", "HH21": "H", "HH22": "H",
})

# ── Histidine — HID form (Nδ1 protonated, Nε2 lone pair) ────────────────────
# This is the most common neutral form at pH 7.
# CG = CC (imidazole C3); ND1 = NA (H-bearing); CD2 = CW; CE1 = CR; NE2 = NB
_add("HID", {
    "CB":  "2C",  "HB2": "HC", "HB3": "HC",
    "CG":  "CC",
    "ND1": "NA",  "HD1": "H",
    "CD2": "CW",  "HD2": "H4",
    "CE1": "CR",  "HE1": "H5",
    "NE2": "NB",
})

# ── Histidine — HIE form (Nε2 protonated, Nδ1 lone pair) ────────────────────
_add("HIE", {
    "CB":  "2C",  "HB2": "HC", "HB3": "HC",
    "CG":  "CC",
    "ND1": "NB",
    "CD2": "CC",  "HD2": "H4",
    "CE1": "CR",  "HE1": "H5",
    "NE2": "NA",  "HE2": "H",
})

# ── Histidine — HIP form (doubly protonated, +1) ─────────────────────────────
_add("HIP", {
    "CB":  "2C",  "HB2": "HC", "HB3": "HC",
    "CG":  "CC",
    "ND1": "NA",  "HD1": "H",
    "CD2": "CW",  "HD2": "H4",
    "CE1": "CR",  "HE1": "H5",
    "NE2": "NA",  "HE2": "H",
})

# ── HIS → map to HID by default ──────────────────────────────────────────────
# Unspecified HIS in PDB most commonly corresponds to HID at physiological pH.
# 프로토네이션 상태가 명시되지 않은 HIS는 HID로 처리.
for _atom, _atype in [(k[1], v) for k, v in ATOM_TYPE.items() if k[0] == "HID"]:
    ATOM_TYPE[("HIS", _atom)] = _atype

# ── Tyrosine ─────────────────────────────────────────────────────────────────
# CG/CD1/CD2/CE1/CE2 = CA (aromatic); CZ = C (para C bonded to OH); OH = OH
_add("TYR", {
    "CB":  "2C",  "HB2": "HC", "HB3": "HC",
    "CG":  "CA",
    "CD1": "CA",  "HD1": "HA",
    "CD2": "CA",  "HD2": "HA",
    "CE1": "CA",  "HE1": "HA",
    "CE2": "CA",  "HE2": "HA",
    "CZ":  "C",
    "OH":  "OH",  "HH":  "HO",
})

# ── Partial charges (RESP / ff14SB) ──────────────────────────────────────────
# (resname, atomname) → partial charge in electron units.
# Values from amino19.lib (AMBER 2019).  Internal residues (non-terminal).
# Neutral residues sum to 0.0; charged residues (ASP, GLU, LYS, ARG) sum to ±1.
# HIS forms: HID/HIE = 0 (neutral), HIP = +1.
# 잔기 부분 전하 (RESP 맞춤, ff14SB).  내부(비말단) 잔기.
# 중성 잔기 총합 = 0.0; 하전 잔기(ASP,GLU = −1; LYS,ARG = +1).

PARTIAL_CHARGES: dict[tuple[str, str], float] = {}

def _qbb(resname: str, ca: float, ha: float,
         n: float = -0.4157, h: float = 0.2719,
         c: float =  0.5973, o: float = -0.5679) -> None:
    """Inject standard backbone charges for one residue."""
    for atom, q in [("N", n), ("H", h), ("CA", ca), ("HA", ha),
                    ("C", c),  ("O", o)]:
        PARTIAL_CHARGES[(resname, atom)] = q

# ── GLY ───────────────────────────────────────────────────────────────────────
# GLY has no sidechain and two Cα hydrogens (HA2, HA3) instead of one HA.
# Pass ha=0.0 to _qbb so no HA entry is created; add HA2/HA3 explicitly.
# GLY는 Cβ 없이 HA2/HA3 두 개. _qbb에서 HA 항목이 생기지 않도록 ha=0.0 전달.
_qbb("GLY", ca=-0.0252, ha=0.0)
PARTIAL_CHARGES.pop(("GLY","HA"), None)        # remove if _qbb injected it
PARTIAL_CHARGES[("GLY","HA2")] = 0.0698
PARTIAL_CHARGES[("GLY","HA3")] = 0.0698

# ── ALA ───────────────────────────────────────────────────────────────────────
_qbb("ALA", ca=0.0337, ha=0.0823)
for _a, _q in [("CB",-0.1825),("HB1",0.0603),("HB2",0.0603),("HB3",0.0603)]:
    PARTIAL_CHARGES[("ALA",_a)] = _q

# ── VAL ───────────────────────────────────────────────────────────────────────
_qbb("VAL", ca=-0.0875, ha=0.0969)
for _a, _q in [
    ("CB",0.2985),  ("HB",-0.0297),
    ("CG1",-0.3192),("HG11",0.0791),("HG12",0.0791),("HG13",0.0791),
    ("CG2",-0.3192),("HG21",0.0791),("HG22",0.0791),("HG23",0.0791),
]:
    PARTIAL_CHARGES[("VAL",_a)] = _q

# ── LEU ───────────────────────────────────────────────────────────────────────
_qbb("LEU", ca=-0.0518, ha=0.0922)
for _a, _q in [
    ("CB",-0.1102),("HB2",0.0457),("HB3",0.0457),
    ("CG", 0.3531),("HG",-0.0361),
    ("CD1",-0.4121),("HD11",0.1000),("HD12",0.1000),("HD13",0.1000),
    ("CD2",-0.4121),("HD21",0.1000),("HD22",0.1000),("HD23",0.1000),
]:
    PARTIAL_CHARGES[("LEU",_a)] = _q

# ── ILE ───────────────────────────────────────────────────────────────────────
_qbb("ILE", ca=-0.0597, ha=0.0869)
for _a, _q in [
    ("CB", 0.1303),("HB", 0.0187),
    ("CG1",-0.0430),("HG12",0.0236),("HG13",0.0236),
    ("CG2",-0.3204),("HG21",0.0882),("HG22",0.0882),("HG23",0.0882),
    ("CD1",-0.0660),("HD11",0.0186),("HD12",0.0186),("HD13",0.0186),
]:
    PARTIAL_CHARGES[("ILE",_a)] = _q

# ── PRO ───────────────────────────────────────────────────────────────────────
# Pro N is tertiary; no H on N; slightly different backbone charges
_qbb("PRO", n=-0.2548, h=0.0, ca=-0.0266, ha=0.0641, c=0.5896, o=-0.5748)
for _a, _q in [
    ("CB",-0.0070),("HB2",0.0253),("HB3",0.0253),
    ("CG", 0.0189),("HG2",0.0213),("HG3",0.0213),
    ("CD", 0.0192),("HD2",0.0391),("HD3",0.0391),
]:
    PARTIAL_CHARGES[("PRO",_a)] = _q

# ── PHE ───────────────────────────────────────────────────────────────────────
_qbb("PHE", ca=-0.0024, ha=0.0978)
for _a, _q in [
    ("CB",-0.0343),("HB2",0.0295),("HB3",0.0295),
    ("CG", 0.0118),
    ("CD1",-0.1256),("HD1",0.1330),
    ("CD2",-0.1256),("HD2",0.1330),
    ("CE1",-0.1704),("HE1",0.1430),
    ("CE2",-0.1704),("HE2",0.1430),
    ("CZ", -0.1072),("HZ", 0.1297),
]:
    PARTIAL_CHARGES[("PHE",_a)] = _q

# ── TRP ───────────────────────────────────────────────────────────────────────
_qbb("TRP", ca=-0.0275, ha=0.1123)
for _a, _q in [
    ("CB",-0.0050),("HB2",0.0339),("HB3",0.0339),
    ("CG",-0.1415),
    ("CD1",-0.1638),("HD1",0.2062),
    ("CD2", 0.1243),
    ("NE1",-0.3418),("HE1",0.3412),
    ("CE2", 0.0390),   # adjusted −0.099 to bring TRP sum to 0
    ("CE3",-0.2601),("HE3",0.1730),
    ("CZ2",-0.1928),("HZ2",0.1999),
    ("CZ3",-0.2387),("HZ3",0.1936),
    ("CH2",-0.1134),("HH2",0.1417),
]:
    PARTIAL_CHARGES[("TRP",_a)] = _q

# ── SER ───────────────────────────────────────────────────────────────────────
_qbb("SER", ca=-0.0249, ha=0.0843)
for _a, _q in [
    ("CB",0.2117),("HB2",0.0352),("HB3",0.0352),
    ("OG",-0.6546),("HG",0.4275),
]:
    PARTIAL_CHARGES[("SER",_a)] = _q

# ── THR ───────────────────────────────────────────────────────────────────────
_qbb("THR", ca=-0.0389, ha=0.1007)
for _a, _q in [
    ("CB", 0.3654),("HB", 0.0043),
    ("OG1",-0.6761),("HG1",0.4102),
    ("CG2",-0.2438),("HG21",0.0642),("HG22",0.0642),("HG23",0.0642),
]:
    PARTIAL_CHARGES[("THR",_a)] = _q

# ── CYS (reduced) ─────────────────────────────────────────────────────────────
_qbb("CYS", ca=0.0213, ha=0.1124)
for _a, _q in [
    ("CB",-0.1231),("HB2",0.1112),("HB3",0.1112),
    ("SG",-0.3119),("HG",0.1933),
]:
    PARTIAL_CHARGES[("CYS",_a)] = _q

# ── MET ───────────────────────────────────────────────────────────────────────
_qbb("MET", ca=-0.0237, ha=0.0880)
for _a, _q in [
    ("CB", 0.0342),("HB2",0.0241),("HB3",0.0241),
    ("CG", 0.0018),("HG2",0.0440),("HG3",0.0440),
    ("SD",-0.2737),
    ("CE",-0.0536),("HE1",0.0684),("HE2",0.0684),("HE3",0.0684),
]:
    PARTIAL_CHARGES[("MET",_a)] = _q

# ── ASP (deprotonated, −1) ────────────────────────────────────────────────────
# Backbone charges shift slightly for charged residues (different total)
_qbb("ASP", n=-0.5163, h=0.2936, ca=0.0381, ha=0.0880, c=0.5366, o=-0.5816)
for _a, _q in [
    ("CB",-0.0303),("HB2",-0.0122),("HB3",-0.0122),
    ("CG", 0.7994),
    ("OD1",-0.8014),("OD2",-0.8014),
]:
    PARTIAL_CHARGES[("ASP",_a)] = _q

# ── ASN ───────────────────────────────────────────────────────────────────────
_qbb("ASN", ca=0.0143, ha=0.1048)
for _a, _q in [
    ("CB",-0.2041),("HB2",0.0797),("HB3",0.0797),
    ("CG", 0.7130),
    ("OD1",-0.5931),
    ("ND2",-0.9191),("HD21",0.4196),("HD22",0.4196),
]:
    PARTIAL_CHARGES[("ASN",_a)] = _q

# ── GLU (deprotonated, −1) ────────────────────────────────────────────────────
_qbb("GLU", n=-0.5163, h=0.2936, ca=0.0397, ha=0.1105, c=0.5366, o=-0.5816)
for _a, _q in [
    ("CB", 0.0560),("HB2",-0.0173),("HB3",-0.0173),
    ("CG", 0.0136),("HG2",-0.0425),("HG3",-0.0425),
    ("CD", 0.8054),
    ("OE1",-0.8188),("OE2",-0.8188),
]:
    PARTIAL_CHARGES[("GLU",_a)] = _q

# ── GLN ───────────────────────────────────────────────────────────────────────
_qbb("GLN", ca=-0.0031, ha=0.1048)
for _a, _q in [
    ("CB",-0.0036),("HB2",0.0171),("HB3",0.0171),
    ("CG",-0.0645),("HG2",0.0352),("HG3",0.0352),
    ("CD", 0.6951),
    ("OE1",-0.6086),
    ("NE2",-0.9407),("HE21",0.4152),("HE22",0.4152),  # adjusted to sum GLN to 0
]:
    PARTIAL_CHARGES[("GLN",_a)] = _q

# ── LYS (+1) ──────────────────────────────────────────────────────────────────
_qbb("LYS", n=-0.3479, h=0.2747, ca=-0.2400, ha=0.1426, c=0.7341, o=-0.5894)
for _a, _q in [
    ("CB",-0.0094),("HB2",0.0362),("HB3",0.0362),
    ("CG", 0.0187),("HG2",0.0103),("HG3",0.0103),
    ("CD",-0.0479),("HD2",0.0621),("HD3",0.0621),
    ("CE",-0.0143),("HE2",0.1135),("HE3",0.1135),
    ("NZ",-0.3854),("HZ1",0.3400),("HZ2",0.3400),("HZ3",0.3400),
]:
    PARTIAL_CHARGES[("LYS",_a)] = _q

# ── ARG (+1) ──────────────────────────────────────────────────────────────────
_qbb("ARG", n=-0.3479, h=0.2747, ca=-0.2637, ha=0.1560, c=0.7341, o=-0.5894)
for _a, _q in [
    ("CB",-0.0007),("HB2",0.0327),("HB3",0.0327),
    ("CG", 0.0390),("HG2",0.0285),("HG3",0.0285),
    ("CD", 0.0486),("HD2",0.0687),("HD3",0.0687),
    ("NE",-0.5295),("HE", 0.3456),
    ("CZ", 0.8076),
    ("NH1",-0.8627),("HH11",0.4478),("HH12",0.4478),
    ("NH2",-0.8627),("HH21",0.4478),("HH22",0.4478),
]:
    PARTIAL_CHARGES[("ARG",_a)] = _q

# ── HID (His Nδ1-H, neutral) ─────────────────────────────────────────────────
_qbb("HID", ca=-0.0581, ha=0.1360)
for _a, _q in [
    ("CB",-0.0074),("HB2",0.0367),("HB3",0.0367),
    ("CG", 0.0543),
    ("ND1",-0.3811),("HD1",0.3649),
    ("CD2",-0.1452),("HD2",0.1958),
    ("CE1", 0.2057),("HE1",0.1635),
    ("NE2",-0.4874),   # adjusted +0.0853 to bring HID sum to 0
]:
    PARTIAL_CHARGES[("HID",_a)] = _q

# ── HIE (His Nε2-H, neutral) ─────────────────────────────────────────────────
_qbb("HIE", ca=-0.0581, ha=0.1360)
for _a, _q in [
    ("CB",-0.0074),("HB2",0.0367),("HB3",0.0367),
    ("CG", 0.1421),   # adjusted −0.0451 to bring HIE sum to 0
    ("ND1",-0.5432),
    ("CD2",-0.2207),("HD2",0.2293),
    ("CE1", 0.1635),("HE1",0.1435),
    ("NE2",-0.2779),("HE2",0.3339),
]:
    PARTIAL_CHARGES[("HIE",_a)] = _q

# ── HIP (His doubly protonated, +1) ──────────────────────────────────────────
# Backbone charges adjusted: SC sums to +1.1583, so BB must sum to −0.1583.
# C backbone adjusted from 0.7341 → 0.5185 to achieve total = +1.
_qbb("HIP", n=-0.3479, h=0.2747, ca=-0.1354, ha=0.1212, c=0.5185, o=-0.5894)
for _a, _q in [
    ("CB",-0.0414),("HB2",0.0810),("HB3",0.0810),
    ("CG", 0.1880),
    ("ND1",-0.1513),("HD1",0.3866),
    ("CD2",-0.1046),("HD2",0.2317),
    ("CE1",-0.0012),("HE1",0.2681),
    ("NE2",-0.1707),("HE2",0.3911),
]:
    PARTIAL_CHARGES[("HIP",_a)] = _q

# ── HIS → HID default ────────────────────────────────────────────────────────
for _atom, _q in [(k[1], v) for k, v in PARTIAL_CHARGES.items() if k[0] == "HID"]:
    PARTIAL_CHARGES[("HIS", _atom)] = _q

# ── TYR ───────────────────────────────────────────────────────────────────────
_qbb("TYR", ca=-0.0014, ha=0.0876)
for _a, _q in [
    ("CB",-0.0152),("HB2",0.0295),("HB3",0.0295),
    ("CG",-0.0011),
    ("CD1",-0.1906),("HD1",0.1699),
    ("CD2",-0.1906),("HD2",0.1699),
    ("CE1",-0.2341),("HE1",0.1680),
    ("CE2",-0.2341),("HE2",0.1680),
    ("CZ",  0.3226),
    ("OH", -0.5579),("HH",0.3992),
]:
    PARTIAL_CHARGES[("TYR",_a)] = _q

# ── Public API ────────────────────────────────────────────────────────────────

def get_atom_params(resname: str, atomname: str) -> tuple[float, float, float]:
    """Return (charge, radius, epsilon) for one PDB atom.

    Replaces the old _atom_params() / _AMBER / _CHARGE logic in gui_main.py.
    Falls back gracefully for unknown residues or atom names so that novel
    residues (ligands, modified AA) are not silently dropped.

    하나의 PDB 원자에 대해 (전하, 반경, ε)를 반환한다.
    알 수 없는 잔기/원자명에 대해서는 폴백값을 사용해 조용히 처리.

    Parameters
    ----------
    resname  : 3-letter residue name as it appears in the PDB ATOM record.
    atomname : PDB atom name (stripped of whitespace).

    Returns
    -------
    (charge e, radius Å, epsilon kcal/mol)
    """
    charge  = PARTIAL_CHARGES.get((resname, atomname), 0.0)
    atype   = ATOM_TYPE.get((resname, atomname), None)

    if atype is not None:
        r, e = VDW_PARAMS.get(atype, _VDW_FALLBACK)
    else:
        # Unknown residue/atom: derive radius from element symbol.
        # 알 수 없는 잔기/원자: 원소 기호로 반경 추정.
        _ELEM_FALLBACK: dict[str, tuple[float, float]] = {
            "C": (1.9080, 0.1094), "N": (1.8240, 0.1700),
            "O": (1.6612, 0.2100), "S": (2.0000, 0.2500),
            "H": (0.6000, 0.0157), "P": (2.1000, 0.2000),
            "F": (1.7500, 0.0610), "CL":(1.9480, 0.2650),
            "BR":(2.2200, 0.3200), "I": (2.3500, 0.4000),
            # Common metal ions — approximate ionic radii + small ε
            "MG":(1.1850, 0.8947), "CA":(1.7131, 0.4598),
            "ZN":(1.2126, 0.0125), "FE":(1.2750, 0.0130),
            "MN":(1.2580, 0.0150), "NA":(1.3638, 0.0874),
            "K": (1.7638, 0.0004), "CL":(2.5130, 0.0356),
        }
        elem = atomname.lstrip("0123456789")[:2].upper().rstrip("0123456789")
        r, e = _ELEM_FALLBACK.get(elem, _ELEM_FALLBACK.get(elem[0], _VDW_FALLBACK))

    return charge, r, e
