/*
 * physics_engine.cpp
 *
 * CPU implicit-solvent protein physics engine, exposed to Python via pybind11.
 * C++ 핵심 물리 엔진 — Python(pybind11)에 노출되는 CPU 기반 암묵적 용매 시뮬레이터.
 *
 * ── 물리 이론 개요 (Theory Overview) ───────────────────────────────────────────
 *
 * 단백질은 수용액 속에서 고유한 3차원 구조(native fold)로 접힌다.
 * 이 접힘(folding)은 아미노산들이 서로, 그리고 용매(물)와 상호작용하는
 * 자유 에너지(free energy)를 최소화하는 방향으로 진행된다.
 *
 * Proteins fold into a unique native 3D structure in aqueous solution.
 * This folding minimises the total free energy of interaction between residues
 * and with the solvent.  We approximate it with four additive terms:
 *
 * ── 에너지 모델 (Energy Model, units: kcal/mol, distances: Å) ─────────────────
 *
 *   1. Generalized Born / GB-HCT  (극성 용매화 에너지 — Polar Solvation)
 *      Water has a high dielectric constant (ε≈78.5) that screens electrostatic
 *      interactions.  The GB model computes how much energy is released (negative)
 *      when polar charges are transferred from the protein interior (ε≈1) into
 *      the solvent.  We use HCT effective Born radii (Hawkins, Cramer & Truhlar
 *      1995) which shrink when an atom is buried inside other atoms.
 *      물은 높은 유전 상수(ε≈78.5)를 가져 정전기 상호작용을 차폐한다.
 *      GB 모델은 극성 전하를 단백질 내부(ε≈1)에서 용매로 옮길 때 방출되는
 *      에너지를 계산한다.  HCT 유효 Born 반경은 원자가 내부에 묻혀 있을수록
 *      더 작아져 용매화의 차폐 효과를 반영한다.
 *
 *   2. Debye–Hückel (DH)  (이온 차폐 쿨롱 — Screened Coulomb)
 *      In a physiological salt solution (≈150 mM NaCl), mobile ions accumulate
 *      around charged residues and screen the long-range Coulomb force.
 *      The screening factor exp(-κr) decays on the Debye length scale (κ⁻¹≈8 Å).
 *      생리적 염 농도(≈150 mM NaCl)에서 이동 이온들이 하전 잔기 주변에
 *      모여 장거리 쿨롱력을 차폐한다.  exp(-κr) 인자가 데바이 길이
 *      스케일(κ⁻¹≈8 Å)에서 감쇠한다.
 *
 *   3. Lennard-Jones 12-6  (반 데르 발스 — Van der Waals)
 *      Combines steric hard-core repulsion (r⁻¹² term, prevents atom overlap)
 *      with London dispersion attraction (r⁻⁶ term).  Parameters σ (radius sum)
 *      and ε (well depth) come from AMBER ff99SB-like lookup tables.
 *      입체 반발(r⁻¹², 원자 겹침 방지)과 런던 분산 인력(r⁻⁶)을 결합한다.
 *      매개변수 σ(반경 합)와 ε(우물 깊이)는 AMBER ff99SB 유사 표에서 가져온다.
 *
 *   4. SASA Nonpolar Term  (소수성 매몰 — Hydrophobic Burial)
 *      The nonpolar surface area exposed to water carries an unfavorable free
 *      energy proportional to the solvent-accessible surface area (SASA).
 *      This is the primary thermodynamic driver for the hydrophobic core formation.
 *      물에 노출된 비극성 표면적은 SASA에 비례하는 불리한 자유 에너지를 갖는다.
 *      이것이 단백질 내부 소수성 코어 형성의 주요 열역학적 구동력이다.
 *
 * ── 몬테 카를로 샘플링 (Monte Carlo Sampling) — P1.5 비틀림 이동 ──────────────
 *
 * Conformational space is explored via torsion-angle Metropolis MC:
 * 구조 공간은 비틀림 각도 기반 Metropolis MC 알고리즘으로 탐색한다:
 *
 *   • BondTopology에서 회전 가능 결합 (i→j) 하나를 무작위로 선택
 *     (Pick one rotatable bond i→j at random from BondTopology)
 *   • j-side 원자 전체를 i→j 축 주위로 δφ ~ U[-max_angle, +max_angle] 회전
 *     (Rotate all j-side atoms by δφ around axis i→j — Rodrigues formula)
 *   • 교차 쌍 에너지 ΔE + ΔSASA 계산
 *     (Compute ΔE from cross-pair energies + ΔSASA)
 *   • ΔE < 0 이면 무조건 수용 (always accept if energy decreases)
 *   • ΔE ≥ 0 이면 볼츠만 확률 exp(-ΔE/T)로 수용 (accept with Boltzmann probability)
 *   • 거부되면 j-side 원자 위치·Born 반경 복원 (revert j-side on rejection)
 *   • 결합 길이·결합각은 회전 전후 항상 보존 (bond lengths/angles identically preserved)
 *
 * 유효 온도 T=0.6 kcal/mol ≈ 300 K (kT at room temperature ≈ 0.592 kcal/mol).
 * 이 알고리즘은 볼츠만 분포 exp(-E/T)로 수렴하며 정준 앙상블을 샘플링한다.
 * This converges to the Boltzmann distribution and samples the canonical ensemble.
 *
 * ── 병렬화 (Parallelism) ──────────────────────────────────────────────────────
 *   - Per-atom Verlet neighbor list (NL_CUTOFF + NL_SKIN shell)
 *     원자별 Verlet 이웃 목록 — skin 트릭으로 O(N²) 재구축 횟수 최소화
 *   - OpenMP parallel for over pair energy loops
 *     쌍 에너지 루프에 OpenMP 병렬화 (동적 스케줄링, reduction)
 *   - Independent per-thread RNG streams in ensemble generation
 *     앙상블 생성 시 스레드별 독립 난수 스트림으로 MC 체인 병렬 실행
 *
 * Python module name: protein_physics
 * Exposed classes: Particle, BondKind, RotBond, BondTopology, PhysicsEngine
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define _USE_MATH_DEFINES  // required on MSVC to get M_PI from <cmath>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <map>
#include <unordered_map>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace py = pybind11;

// ── Physics constants ─────────────────────────────────────────────────────────
// 물리 상수 — 모든 에너지 단위는 kcal/mol, 거리 단위는 Å(옹스트롬)
//
// COULOMB   : conversion factor  e²/(4πε₀) in kcal·Å/(mol·e²)
//             전하 단위 e를 kcal/mol 에너지로 변환하는 쿨롱 상수
// EPS_WATER : dielectric constant of bulk water (78.5)
//             벌크 물의 유전 상수 — 전기장을 78.5배 차폐
// EPS_PROT  : interior dielectric constant of the protein (1.0)
//             단백질 내부 유전 상수 — 진공에 가까운 낮은 극성
// KAPPA     : Debye screening length inverse (Å⁻¹); ~0.1 Å⁻¹ at 150 mM NaCl
//             데바이 역차폐 길이 — 이온 강도가 높을수록 κ가 커져 차폐가 강해짐
// GAMMA_SA  : surface tension coefficient (kcal/mol·Å²) for SASA nonpolar term
//             SASA 비극성 항의 표면 장력 계수 — 소수성 매몰의 에너지 이득 스케일
// BETA_SA   : additive constant in SASA energy (kcal/mol)
//             SASA 에너지의 덧셈 상수 (기준 오프셋)
// PROBE_R   : solvent probe radius (1.4 Å = water molecule)
//             용매 탐침 반경 — 물 분자가 닿을 수 있는 최소 거리를 결정
// NL_CUTOFF : hard pair-energy cutoff (Å); pairs beyond this are ignored
//             이 거리 밖의 쌍은 에너지 계산에서 제외 (장거리 근사)
// NL_SKIN   : extra shell around cutoff kept in neighbor list for drift tolerance
//             이웃 목록을 컷오프보다 약간 크게 유지해 drift 허용 — 재구축 빈도 감소
// HARD_SCALE: energy scale for hard-core repulsion when atoms overlap (r < HARD_CUTOFF_FRAC·σ)
//             원자가 겹칠 때(r < HARD_CUTOFF_FRAC·σ) 적용되는 강한 척력 에너지 스케일
// HARD_CUTOFF_FRAC: fraction of σ below which pair_e() substitutes the hard-core
//             term for the ordinary LJ/GB/DH sum.  Calibrated empirically against
//             an all-atom (explicit-H) folded structure (PDB 1XQ8): real, chemically
//             valid non-excluded contacts (mostly 1-4 and H···H van der Waals pairs)
//             go down to r/σ ≈ 0.67 with nothing pathological about them — plain LJ
//             already evaluates them to a small, bounded, mildly repulsive energy.
//             A threshold of 0.85 (the previous value) caught thousands of these
//             normal contacts and replaced a ~few-kcal/mol LJ value with an
//             artificial HARD_SCALE·(σ/r)¹² spike of 10⁴–10⁹ kcal/mol each,
//             which is why whole-protein energies came out in the billions
//             instead of the expected thousands.  0.6 sits safely below every
//             observed legitimate contact while still catching genuine
//             numerical-overlap pathologies from aggressive MC proposals.
//             HARD_CUTOFF_FRAC: pair_e()가 통상적인 LJ/GB/DH 합 대신 하드코어 항을
//             대입하는 σ 대비 거리 비율. 수소를 포함한 전원자 접힘 구조(PDB 1XQ8)로
//             경험적으로 보정함: 실제로 배제되지 않는 화학적으로 정상적인 접촉
//             (대부분 1-4 쌍과 H···H 반데르발스 쌍)은 r/σ ≈ 0.67까지 내려가며 전혀
//             병리적이지 않다 — 통상적인 LJ만으로도 작고 유한한 약한 척력 값이 나온다.
//             기존 값 0.85는 이런 정상 접촉 수천 개를 붙잡아 몇 kcal/mol의 LJ 값을
//             10⁴~10⁹ kcal/mol짜리 인위적 스파이크로 대체해버렸고, 이것이 전체
//             단백질 에너지가 수천 대신 수십억 단위로 나온 원인이다. 0.6은 관측된
//             모든 정상 접촉보다 충분히 낮으면서도, 공격적인 MC 제안으로 인한 실제
//             수치적 겹침 병리 현상은 여전히 잡아낸다.
// HARD_CAP  : ceiling (kcal/mol) on the hard-core term itself. Even below
//             HARD_CUTOFF_FRAC·σ, HARD_SCALE·(σ/r)¹² is unbounded as r→0, and real
//             deposited structures do contain occasional pathological contacts that
//             are NOT MC-proposal artifacts — e.g. PDB 1LYZ (1975 "real-space
//             refinement", pre-modern crystallography) has a genuine 1.36 Å CB···NH1
//             contact between Ala122 and Arg125's flexible, poorly-resolved
//             guanidinium group. Uncapped, that single pair alone evaluates to
//             ~2×10⁹ kcal/mol and swamps calculate_potential() for the entire
//             1001-atom structure (observed total: 2.1×10⁹, i.e. one pair IS the
//             "billions" bug — see accuracy_test.py). A real force field would
//             still treat this as strongly, finitely unfavorable, not treat the
//             whole structure as if it doesn't exist. HARD_CAP bounds any single
//             hard-core contact to a large-but-finite penalty so MC still firmly
//             rejects/relaxes such contacts while calculate_potential() stays a
//             meaningful, comparable number for real (imperfect) input structures,
//             not just the hand-vetted bundled test set.
//             HARD_CAP: 하드코어 항 자체의 상한(kcal/mol). HARD_CUTOFF_FRAC·σ 밑에서도
//             HARD_SCALE·(σ/r)¹²은 r→0일 때 무한대로 발산한다. 실제 구조에는 MC 제안의
//             인위적 결과가 아닌, 진짜 병리적 접촉이 이따금 존재한다 — 예: PDB 1LYZ
//             (1975년 "real-space refinement", 현대 이전 결정학)는 Ala122와 유연하고
//             전자밀도가 약해 잘 정제되지 않은 Arg125 구아니디늄기 사이에 실제 1.36 Å
//             CB···NH1 접촉을 갖고 있다. 상한 없이는 이 한 쌍만으로 ~2×10⁹ kcal/mol이
//             나와 1001개 원자 전체 구조의 calculate_potential()을 완전히 뒤덮는다
//             (관측값: 2.1×10⁹ — 즉 이 쌍 하나가 "수십억" 버그의 전부다. accuracy_test.py
//             참고). 실제 힘장이라면 이런 접촉도 강하지만 유한한 불리함으로 취급해야지,
//             전체 구조가 존재하지 않는 것처럼 만들면 안 된다. HARD_CAP은 단일 하드코어
//             접촉을 크지만 유한한 페널티로 제한해, MC는 여전히 이런 접촉을 확실히
//             거부/완화하면서도 calculate_potential()은 미리 검증된 번들 테스트셋뿐
//             아니라 실제(불완전한) 입력 구조에서도 의미 있고 비교 가능한 값으로 남는다.
// GB_COEF   : prefactor for the GB Born term = -½(1/ε_prot - 1/ε_water)·C
//             GB Born 항의 앞인수 — 진공→물 이동 시 에너지 이득(음수)
static constexpr double COULOMB    = 332.0636;
static constexpr double EPS_WATER  = 78.5;
static constexpr double EPS_PROT   = 1.0;
static constexpr double KAPPA      = 0.1257;
static constexpr double GAMMA_SA   = 0.00542;
static constexpr double BETA_SA    = 0.92;
static constexpr double PROBE_R    = 1.4;
static constexpr double NL_CUTOFF  = 12.0;
static constexpr double NL_SKIN    = 2.0;
static constexpr double NL_RCUT2   = (NL_CUTOFF+NL_SKIN)*(NL_CUTOFF+NL_SKIN);
static constexpr double PAIR_CUT2  = NL_CUTOFF*NL_CUTOFF;
static constexpr double HALF_SKIN2 = (NL_SKIN*0.5)*(NL_SKIN*0.5);
static constexpr double HARD_SCALE = 1.0e4;
static constexpr double HARD_CAP   = 5.0e3;
static constexpr double HARD_CUTOFF_FRAC = 0.6;
static constexpr double GB_COEF    = -0.5*(1.0/EPS_PROT-1.0/EPS_WATER)*COULOMB;

// Particle: one heavy atom (or hydrogen) in the system.
// 시스템 내 중원자(또는 수소) 하나를 나타내는 구조체.
//   x,y,z    — Cartesian coordinates (Å) / 데카르트 좌표 (단위: Å)
//   charge   — partial charge (e), taken from AMBER/CHARGE lookup in gui_main.py
//              부분 전하 (단위: e) — AMBER ff99SB 기반 룩업에서 가져옴
//   radius   — van der Waals radius (Å); also used as intrinsic Born radius
//              반 데르 발스 반경; GB 계산의 고유 Born 반경으로도 쓰임
//   epsilon  — LJ well-depth (kcal/mol) / 레너드-존스 우물 깊이
//   is_water — true for explicit water molecules (currently unused; reserved)
//              명시적 물 분자 플래그 (현재 미사용; 향후 확장용)
struct Particle {
    double x,y,z,charge,radius,epsilon;
    bool is_water;
    Particle(double x,double y,double z,double charge,
             double radius=1.9,double epsilon=0.1,bool is_water=false)
        :x(x),y(y),z(z),charge(charge),radius(radius),epsilon(epsilon),is_water(is_water){}
};

// NeighborList: Verlet pair list — stores atom-j indices reachable from atom-i
// within (NL_CUTOFF + NL_SKIN).
//
// Verlet 이웃 목록 — Verlet(1967)이 분자 동역학에 도입한 핵심 최적화 기법.
// 목적: 매 MC 스텝마다 O(N²) 쌍 검색을 피하기 위해
//       컷오프보다 약간 큰 "skin" 범위 내의 이웃을 미리 캐싱한다.
//
// build() is O(N²) and called only when atoms have drifted more than NL_SKIN/2
// from their positions at the last rebuild.  This amortises the O(N²) cost
// over many MC steps (the "skin trick").
// build()는 O(N²)이지만, 원자들이 skin/2 이상 이동했을 때만 호출되므로
// 재구축 비용이 여러 MC 스텝에 걸쳐 상각(amortise)된다 — "skin 트릭".
//
//   nb[i]  — sorted list of j > i that are within NL_RCUT2 of atom i
//            원자 i에서 NL_RCUT2 거리 내에 있는 j > i 인덱스 목록
//   ref[i] — position of atom i at the last rebuild (drift-check baseline)
//            마지막 재구축 시 원자 i의 위치 — drift 검사의 기준점
struct NeighborList {
    std::vector<std::vector<size_t>> nb;
    std::vector<std::array<double,3>> ref;
    size_t N=0;
    // ── 셀 목록 이웃 목록 재구축 (Cell-list Neighbor List rebuild, P4.1) ─────────
    //
    // ── 배경: O(N²) 방식의 문제 (Why we need a cell list) ───────────────────────
    //
    // 기존 방식(모든 쌍 순환)은 N개 원자에 대해 O(N²) 연산이 필요하다.
    // N = 2000 (단백질 약 150 잔기)에서 약 2×10⁶ 쌍 검사.
    // N = 10000 (단백질 약 750 잔기)에서 약 5×10⁷ 쌍 검사.
    // 재구축 빈도는 낮지만(drift 기반) 각 재구축 자체가 병목이 된다.
    //
    // The old all-pairs loop is O(N²).  For N=2 000 (≈150 residues) that is
    // ~2×10⁶ pair checks per rebuild.  For N=10 000 (~750 residues) it is
    // ~5×10⁷ checks.  Even at low rebuild frequency (drift-triggered) this
    // becomes the bottleneck for larger proteins.
    //
    // ── 셀 목록 원리 (Cell-list principle) ──────────────────────────────────────
    //
    // 알고리즘:
    //   1. 경계 상자를 한 변 길이 cs = sqrt(NL_RCUT2) + ε 의 3D 직교 셀로 분할.
    //      cs ≥ 탐색 반경이므로, 두 원자가 탐색 반경 내에 있다면
    //      반드시 동일 셀 또는 3D에서 1셀 이내 인접 셀에 존재한다.
    //      (증명: 두 원자가 2셀 이상 떨어진 셀에 있다면 적어도 한 차원에서
    //       거리가 cs ≥ NL_RCUT 이상이므로 탐색 반경을 초과함.)
    //   2. 각 원자를 해당 셀 목록에 등록.  O(N).
    //   3. 각 원자 i에 대해 3×3×3 = 27개 이웃 셀(자기 셀 포함)의 원자 j만 검사.
    //      j > i 조건으로 중복 제거.  탐색 반경 초과 쌍은 거리 검사로 필터링.
    //
    // Cell-list algorithm:
    //   1. Partition the bounding box into cubic cells of side cs ≥ sqrt(NL_RCUT2).
    //      Key property: any two atoms within the search radius must be in the
    //      same cell or a cell that is at most 1 cell away in each dimension.
    //      Proof: if two atoms are ≥ 2 cells apart in any dimension, their
    //      separation in that dimension is ≥ cs ≥ search_radius → outside cutoff.
    //   2. Bin each atom into its cell.  O(N).
    //   3. For each atom i, scan only atoms j in the 27-cell 3×3×3 neighbourhood.
    //      Use j>i to avoid double-counting.  Filter by exact distance check.
    //
    // 복잡도 분석 (Complexity):
    //   평균 원자 밀도 ρ(atoms/Å³)에서 셀당 원자 수 ≈ ρ × cs³ = 상수.
    //   각 원자가 검사하는 쌍 수 ≈ 27 × 셀당 원자 수 = O(1).
    //   전체 재구축 복잡도: O(N).
    //   실용적 속도 향상: N = 1000에서 ≈ 40×, N = 5000에서 ≈ 200×.
    //
    // Average complexity: O(N) because each atom checks only a constant number
    // of neighbors (27 cells × ρ×cs³ atoms per cell ≈ constant).
    // Practical speedup: ≈40× at N=1000, ≈200× at N=5000.
    //
    // ── 구현 세부 (Implementation details) ──────────────────────────────────────
    //
    // cs = sqrt(NL_RCUT2) + ε:
    //   수치 오차로 경계 원자가 잘못 분류되는 것을 막기 위한 작은 여백.
    //   Small epsilon prevents floating-point boundary atoms from being misclassified.
    //
    // 경계 처리: 경계 상자를 cs 만큼 확장해 모든 원자가 유효한 셀 인덱스를 갖도록.
    //   std::clamp 로 경계 초과 인덱스를 안전하게 클램핑.
    // Boundary: expand the box by one cell (margin=cs) so all atoms get valid
    //   cell indices; clamp prevents out-of-range access.
    void build(const std::vector<Particle>& p) {
        N = p.size(); nb.assign(N, {}); ref.resize(N);
        if (N == 0) return;
        for (size_t i = 0; i < N; ++i) ref[i] = {p[i].x, p[i].y, p[i].z};
        if (N == 1) return;

        // Bounding box
        double xlo=p[0].x,xhi=p[0].x,ylo=p[0].y,yhi=p[0].y,zlo=p[0].z,zhi=p[0].z;
        for (const auto& a : p) {
            xlo=std::min(xlo,a.x); xhi=std::max(xhi,a.x);
            ylo=std::min(ylo,a.y); yhi=std::max(yhi,a.y);
            zlo=std::min(zlo,a.z); zhi=std::max(zhi,a.z);
        }
        // Cell side = full search radius ensures 3×3×3 neighborhood covers all pairs.
        const double cs = std::sqrt(NL_RCUT2) + 1e-6;
        const double mg = cs;   // expand box by one cell on each side
        xlo -= mg; ylo -= mg; zlo -= mg;
        double lx = xhi - xlo + 2.0*mg, ly = yhi - ylo + 2.0*mg, lz = zhi - zlo + 2.0*mg;
        int nx = std::max(1, (int)std::ceil(lx / cs));
        int ny = std::max(1, (int)std::ceil(ly / cs));
        int nz = std::max(1, (int)std::ceil(lz / cs));

        // Assign each atom to a cell
        std::vector<std::vector<size_t>> cells((size_t)(nx*ny*nz));
        std::vector<int> cx(N), cy(N), cz(N);
        for (size_t i = 0; i < N; ++i) {
            cx[i] = std::min(nx-1, std::max(0, (int)std::floor((p[i].x-xlo)/cs)));
            cy[i] = std::min(ny-1, std::max(0, (int)std::floor((p[i].y-ylo)/cs)));
            cz[i] = std::min(nz-1, std::max(0, (int)std::floor((p[i].z-zlo)/cs)));
            cells[(size_t)(cx[i]+nx*(cy[i]+ny*cz[i]))].push_back(i);
        }

        // For each atom i scan the 3×3×3 cell neighbourhood
        for (size_t i = 0; i < N; ++i) {
            for (int dz=-1; dz<=1; ++dz) {
                int cz2 = cz[i]+dz; if (cz2<0||cz2>=nz) continue;
                for (int dy=-1; dy<=1; ++dy) {
                    int cy2 = cy[i]+dy; if (cy2<0||cy2>=ny) continue;
                    for (int dx=-1; dx<=1; ++dx) {
                        int cx2 = cx[i]+dx; if (cx2<0||cx2>=nx) continue;
                        for (size_t j : cells[(size_t)(cx2+nx*(cy2+ny*cz2))]) {
                            if (j <= i) continue;
                            double ddx=p[i].x-p[j].x,ddy=p[i].y-p[j].y,ddz=p[i].z-p[j].z;
                            if (ddx*ddx+ddy*ddy+ddz*ddz < NL_RCUT2) nb[i].push_back(j);
                        }
                    }
                }
            }
        }
    }
    // Returns true when any atom has moved > NL_SKIN/2 since the last build,
    // meaning a pair that was just outside NL_CUTOFF could now be inside it.
    bool needs_rebuild(const std::vector<Particle>& p) const {
        for(size_t i=0;i<N;++i){
            double dx=p[i].x-ref[i][0],dy=p[i].y-ref[i][1],dz=p[i].z-ref[i][2];
            if(dx*dx+dy*dy+dz*dz>HALF_SKIN2) return true;
        }
        return false;
    }
};

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  Bond Topology  (공유 결합 위상 그래프)                                      ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// ── 목적 (Purpose) ─────────────────────────────────────────────────────────────
//
// 현재 MC 샘플러는 무작위 원자 하나를 데카르트 공간에서 이동시킨다.
// 이 방식은 공유 결합 길이와 각도를 파괴하므로 물리적으로 무의미하다.
// 예: Cα를 3 Å 이동하면 인접한 N·C·Cβ와의 결합이 늘어지거나 끊어진다.
//
// The current MC sampler translates one atom at random in Cartesian space.
// This breaks covalent bond lengths and angles and produces physically invalid
// conformations.  Example: moving Cα by 3 Å stretches or breaks its bonds to
// the neighbouring N, C, and Cβ.
//
// 해결책: 비틀림 각도 이동(torsion-angle move).
// 회전 가능한 결합 하나(i→j)를 선택하고, j-side 원자 전체를 i→j 축을 중심으로
// 작은 각도 δφ만큼 강체(rigid body) 회전시킨다.
// 이렇게 하면 분자의 모든 결합 길이와 결합 각도가 보존된다.
//
// Solution: torsion-angle MC moves.
// Pick one rotatable bond (i→j), rotate all atoms on the j-side as a rigid
// body by a small angle δφ around the i→j axis.  Bond lengths and angles are
// identically preserved throughout the move.
//
// ── 이 섹션이 제공하는 것 (What this section provides) ─────────────────────────
//
//  • BondKind  — 결합 종류 레이블 (백본 φ/ψ, 곁사슬 χ, 고정)
//                bond-kind labels (backbone φ/ψ, sidechain χ, fixed)
//  • RotBond   — 회전 가능 결합 하나의 인덱스·종류 묶음
//                (i,j) atom indices + kind for one rotatable bond
//  • bond_templates() — 20개 표준 아미노산의 잔기 내 공유 결합 쌍 정적 표
//                       static per-residue intra-residue covalent bond tables
//  • rot_specs()      — 잔기별 회전 가능 결합 목록 정적 표
//                       static per-residue rotatable bond specifications
//  • BondTopology     — PDB 파싱 직후 한 번만 구성하는 결합 그래프 클래스
//                       bond graph class built once after PDB parsing
//
// ── 향후 사용처 (Planned use in later milestones) ─────────────────────────────
//
//  P1.4 — bonded energy terms:
//    1-2 쌍 (직접 결합)·1-3 쌍 (두 결합)·1-4 쌍 (세 결합)을 adj에서 쉽게 열거.
//    표준 비결합 에너지에서 1-2/1-3 쌍은 제외, 1-4 쌍은 스케일 다운해야 한다.
//    Enumerate 1-2 (bonded), 1-3 (angle), 1-4 (dihedral) pairs from adj.
//    Standard force fields exclude 1-2/1-3 from non-bonded sums and
//    scale down 1-4 interactions.
//
//  P1.5 — torsion-angle MC moves:
//    rot_bonds 목록에서 결합 하나를 선택 → j_side()로 이동할 원자 집합을 구하고
//    i→j 축 주위의 회전 행렬 R(δφ)를 적용 → 물리적으로 유효한 구조 탐색.
//    Pick one bond from rot_bonds → get j_side() atom set → apply rotation
//    matrix R(δφ) around i→j → physically valid conformation sampling.

// ── BondKind: 결합 종류 분류 ──────────────────────────────────────────────────
//
// 각 회전 가능 결합에 붙이는 레이블.  P1.5 MC 이동 가중치 계획:
//   백본 50% (φ 25% + ψ 25%),  곁사슬 40%,  기타 10%.
// Label attached to each rotatable bond.  Planned P1.5 MC move weights:
//   backbone 50% (φ 25% + ψ 25%), sidechain 40%, other 10%.
enum class BondKind : uint8_t {
    // N→CA 결합을 축으로 하는 백본 φ(파이) 이중면체각 회전.
    // φ = 이전 C(i-1)–N–Cα–C 이중면체각; 허용 범위는 Ramachandran 도표로 규정.
    // Backbone φ dihedral axis (N→CA bond).
    // φ = C(i-1)–N–Cα–C dihedral; allowed regions defined by Ramachandran plot.
    BACKBONE_PHI,

    // CA→C 결합을 축으로 하는 백본 ψ(프사이) 이중면체각 회전.
    // ψ = N–Cα–C–N(i+1) 이중면체각.
    // Backbone ψ dihedral axis (CA→C bond).
    // ψ = N–Cα–C–N(i+1) dihedral.
    BACKBONE_PSI,

    // 곁사슬 χ 각도: χ1(CA–CB), χ2(CB–CG/OG/SG), χ3, χ4, χ5.
    // 잔기 종류에 따라 χ 각도 수가 다름 (Gly/Ala = 0, Lys/Arg = 5 등).
    // Sidechain χ dihedral bonds (χ1 = CA–CB, χ2 = CB–CG/OG/SG, etc.).
    // Number of χ angles varies by residue (Gly/Ala = 0, Lys/Arg = up to 5).
    SIDECHAIN,

    // 회전 불가 결합: 방향족 고리 내 결합, 말단 결합, 펩타이드 결합(부분 이중 결합).
    // Non-rotatable: ring bonds, terminal bonds, peptide bonds (partial double-bond
    // character due to resonance — rotation barrier ≈ 20 kcal/mol).
    FIXED
};

// ── RotBond: 회전 가능 결합 하나의 서술자 ────────────────────────────────────
//
// P1.5 MC 이동의 기본 단위.  사용 방법:
//   1. rot_bonds 목록에서 RotBond rb를 무작위로 선택.
//   2. j_side(rb.i, rb.j)로 회전할 원자 인덱스 집합 S를 구한다.
//   3. δφ ~ Uniform[-maxδ, +maxδ]를 샘플링한다.
//      maxδ ≈ 5° (backbone), 30° (sidechain) 로 이동 폭을 조절.
//   4. 축 벡터 u = (p[rb.j] - p[rb.i]).normalize() 를 구한다.
//   5. Rodrigues 회전 공식으로 S의 모든 원자를 rb.i 기준점 주위로 δφ 회전.
//   6. Metropolis 수용/거부 기준 적용.
//
// The basic unit of a P1.5 torsion MC move.  Usage:
//   1. Pick a RotBond rb at random from rot_bonds.
//   2. Compute atom set S = j_side(rb.i, rb.j) — these atoms will move.
//   3. Sample δφ ~ Uniform[-maxδ, +maxδ].
//      maxδ ≈ 5° (backbone), 30° (sidechain).
//   4. Axis vector u = normalise(p[rb.j] − p[rb.i]).
//   5. Apply Rodrigues rotation by δφ around u, anchored at p[rb.i],
//      to every atom in S.
//   6. Metropolis accept/reject.
struct RotBond {
    int      i;     // 결합의 i쪽 원자 인덱스 (회전축의 시작점)
                    // atom index on the i-side (axis origin)
    int      j;     // 결합의 j쪽 원자 인덱스 (회전축의 끝점, j-side 원자들이 회전)
                    // atom index on the j-side (axis tip; j-side atoms rotate)
    BondKind kind;  // 결합 종류 레이블 (이동 가중치 결정에 사용)
                    // bond kind label (used to determine move weight)
};

// ── P1.4a: 이중면체각 에너지 (Dihedral / Torsion energy) ──────────────────────
//
// 공식: E = Σ V2 × [1 + cos(n·φ − γ)]
//   V2  : 장벽 높이 (kcal/mol) — AMBER 표기의 Vn/2에 해당
//   n   : 주기성 (periodicity)
//   γ   : 위상 오프셋 (radians)
//
// Formula: E = Σ V2 × [1 + cos(n·φ − γ)]
//   V2 = barrier height (kcal/mol), corresponds to Vn/2 in AMBER notation.
//   n  = periodicity, γ = phase offset (radians).
//
// 파라미터 출처 (Parameter source):
//   AMBER parm99 / ff14SB Fourier terms, approximate values.
//   백본 φ/ψ: ~2.5 kcal/mol 총 장벽 → Ramachandran 도표 허용 영역 재현
//   곁사슬 χ: 원소별 대표값 사용 (C-C, C-O, C-N, C-S 분류)
//   Backbone φ/ψ: ~2.5 kcal/mol total barrier reproducing Ramachandran allowed regions.
//   Sidechain χ: element-class representative values (C-C, C-O, C-N, C-S).

struct DihTerm {
    double V2;    // Vn/2 barrier height (kcal/mol) — using AMBER convention
                  // 장벽 높이 (kcal/mol) — AMBER Vn/2 규칙
    int    n;     // periodicity / 주기성
    double gamma; // phase (radians) / 위상 (라디안)
};

// Pre-built torsion dihedral record: 4 atom indices + energy terms.
// 사전 계산된 이중면체각: 4개 원자 인덱스 + Fourier 에너지 항 목록.
// Atoms a-b-c-d define the torsion; b-c is the rotatable bond.
struct DihRecord {
    int   a, b, c, d;              // b-c는 중심 회전 결합
    std::vector<DihTerm> terms;
};

// ── 이중면체각 파라미터 표 (Dihedral parameter table) ────────────────────────
//
// BondKind과 j-side 원자명(aname_j)을 키로 하는 Fourier 항 반환.
// 키: BondKind (BACKBONE_PHI/PSI → 백본 표준 항)
//     aname_j의 첫 문자 → 원소 분류
//
// Returns Fourier terms keyed by bond kind and first character of atom_j name.
// The returned V2 values are in kcal/mol (= Vn/2 in AMBER notation).
static std::vector<DihTerm> get_dih_terms(BondKind kind,
                                           const std::string& aname_j)
{
    using K = BondKind;
    if (kind == K::BACKBONE_PHI)
        // C(prev)-N-CA-C: AMBER parm99 C-N-CT-C terms.
        // n=3 주 장벽 (α-나선/β-가닥 대칭 최솟값) + n=2 비대칭 보정.
        // n=3 primary barrier (symmetric minima at ±60° and 180°) + n=2 asymmetry correction.
        return { {0.40, 3, M_PI}, {0.29, 2, M_PI} };
    if (kind == K::BACKBONE_PSI)
        // N-CA-C-N(next): similar magnitude; n=3 zero shifted to prefer ψ=±60°, 180°.
        return { {0.40, 3, 0.0}, {0.29, 2, M_PI} };
    // Sidechain: key on element class of j-atom
    char e = aname_j.empty() ? 'C' : aname_j[0];
    if (e == 'O')   // C-O bond: SER/THR/ASN/GLN/ASP/GLU/TYR
        // Strong 2-fold prefers gauche oxygen; 3-fold minor correction.
        // 2-fold이 산소를 가우쉬 위치로 선호; 3-fold은 소폭 보정.
        return { {1.18, 2, M_PI}, {0.14, 3, 0.0} };
    if (e == 'S')   // C-S bond: CYS/MET — low barrier
        return { {0.60, 3, 0.0} };
    if (e == 'N')   // C-N bond: LYS-NZ, ARG-NE — sp3 amine
        return { {0.50, 3, 0.0}, {0.16, 2, 0.0} };
    // Generic sp3 C-C: aliphatic sidechain chi bonds (χ1-χ5)
    // AMBER CT-CT-CT-CT value
    return { {0.56, 3, 0.0} };
}

// ── 잔기 내 결합 템플릿 (Intra-residue bond template tables) ─────────────────
//
// 각 항목: (원자명_A, 원자명_B) 공유 결합 쌍 목록.
// 출처: AMBER amino19.lib (AMBER 2019 배포판) prep 파일.
// PDB v3 표준 원자명 규칙을 따름.
//
// Each entry: list of (atomname_A, atomname_B) covalent bond pairs within one
// residue.  Source: AMBER amino19.lib prep files; PDB v3 atom name convention.
//
// 설계 결정 (Design decisions):
//  • 백본 원자(N, H, CA, HA, C, O)를 잔기마다 명시적으로 포함.
//    inter-residue 펩타이드 결합 C(i)→N(i+1)은 build()에서 별도 처리.
//  • Backbone atoms included explicitly per residue.
//    The inter-residue peptide bond C(i)→N(i+1) is added separately in build().
//
//  • 고리 닫힘(ring closure) 결합은 // ring closure 주석으로 표시.
//    이 결합이 있어야 adj에서 고리 탐지가 가능하고,
//    j_side() DFS가 고리 원자를 올바르게 처리한다.
//  • Ring-closure bonds are marked with comments.  Their presence in adj
//    lets j_side() DFS correctly traverse rings without double-counting.
//
//  • 수소 원자도 포함 (결합 차수 계산, 1-2 배제 목록에 필요).
//    PDB 파일에 수소가 없더라도 lookup.find()가 실패하면 조용히 건너뜀.
//  • Hydrogens included (needed for bond-order counting and 1-2 exclusions).
//    If the PDB lacks H atoms, the lookup.find() simply misses them silently.

using BondPair = std::pair<const char*, const char*>;
using RotSpec  = std::tuple<const char*, const char*, BondKind>;

// 백본 공통 결합 매크로 (이니셜라이저 리스트 반복 방지용)
// Backbone bond macros to avoid repeating the same 5 pairs in every entry.
//
// _BB      — 표준 내부 잔기 백본: N-H, N-CA, CA-HA, CA-C, C-O
//             Standard internal residue backbone
// _BB_GLY  — Gly 전용: CB 없음, HA 대신 HA2·HA3 두 개
//             Gly-specific: no CB; two alpha-H (HA2, HA3) instead of one HA
// _BB_PRO  — Pro 전용: N에 H 없음 (3차 아민), 고리 N-CD 결합 포함
//             Pro-specific: no HN (tertiary amine), ring N-CD bond included
#define _BB  {"N","H"},{"N","CA"},{"CA","HA"},{"CA","C"},{"C","O"}
#define _BB_GLY {"N","H"},{"N","CA"},{"CA","HA2"},{"CA","HA3"},{"CA","C"},{"C","O"}
#define _BB_PRO {"N","CA"},{"N","CD"},{"CA","HA"},{"CA","C"},{"C","O"}

static const std::unordered_map<std::string, std::vector<BondPair>>&
bond_templates() {
    // static 지역 변수 → 프로그램 수명 동안 단 한 번만 초기화됨 (thread-safe in C++11).
    // Static local: initialised exactly once for the program's lifetime (C++11 guaranteed).
    static const std::unordered_map<std::string, std::vector<BondPair>> t = {

        // GLY (글리신 / Glycine) — 유일하게 Cβ가 없는 잔기.
        // α-탄소에 수소가 두 개(HA2, HA3)붙어 있어 입체화학적으로 대칭.
        // Only amino acid without Cβ; two Hα atoms (HA2, HA3) → achiral Cα.
        {"GLY", { _BB_GLY }},

        // ALA (알라닌 / Alanine) — 가장 작은 곁사슬: Cβ 메틸기 (CH₃).
        // χ각도 없음 (메틸기 회전은 에너지 변화가 미미해 MC 이동 불필요).
        // Smallest sidechain: Cβ methyl (CH₃); no χ angle (methyl rotation negligible).
        {"ALA", { _BB, {"CA","CB"},{"CB","HB1"},{"CB","HB2"},{"CB","HB3"} }},

        // VAL (발린 / Valine) — β-분지형 (β-branched). Cβ에 두 메틸기 CG1·CG2 분기.
        // χ1 = CA-CB. β-분지가 있어 나선 구조 형성을 방해하는 경향이 있음.
        // β-branched: Cβ forks into two methyls (CG1, CG2).  χ1 = CA-CB.
        // β-branching sterically disfavours helix formation.
        {"VAL", { _BB,
                  {"CA","CB"},{"CB","HB"},
                  {"CB","CG1"},{"CG1","HG11"},{"CG1","HG12"},{"CG1","HG13"},
                  {"CB","CG2"},{"CG2","HG21"},{"CG2","HG22"},{"CG2","HG23"} }},

        // LEU (류신 / Leucine) — 가장 흔한 소수성 잔기.
        // Cβ–Cγ–(Cδ1, Cδ2): 두 말단 메틸기. χ1 = CA-CB, χ2 = CB-CG.
        // Most abundant hydrophobic residue.  Two terminal methyls at Cδ.
        // χ1 = CA-CB, χ2 = CB-CG.
        {"LEU", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},{"CG","HG"},
                  {"CG","CD1"},{"CD1","HD11"},{"CD1","HD12"},{"CD1","HD13"},
                  {"CG","CD2"},{"CD2","HD21"},{"CD2","HD22"},{"CD2","HD23"} }},

        // ILE (이소류신 / Isoleucine) — β-분지형 + 긴 곁사슬.
        // Cβ → CG1(–CD1), CG2(메틸). χ1 = CA-CB, χ2 = CB-CG1.
        // β-branched + elongated sidechain.  χ1 = CA-CB, χ2 = CB-CG1.
        {"ILE", { _BB,
                  {"CA","CB"},{"CB","HB"},
                  {"CB","CG1"},{"CG1","HG12"},{"CG1","HG13"},
                  {"CG1","CD1"},{"CD1","HD11"},{"CD1","HD12"},{"CD1","HD13"},
                  {"CB","CG2"},{"CG2","HG21"},{"CG2","HG22"},{"CG2","HG23"} }},

        // PRO (프롤린 / Proline) — 유일하게 N이 백본과 곁사슬 양쪽에 결합하는 잔기.
        // 피롤리딘 5원 고리(N-CA-CB-CG-CD-N)가 φ 각도를 약 -65° 근처로 고정.
        // ψ 각도는 회전 가능. 나선 구조를 방해("나선 파괴자").
        // Only residue where N bonds to both backbone and sidechain (pyrrolidine ring).
        // The 5-membered ring (N-CA-CB-CG-CD) locks φ near -65°.
        // ψ is free.  Known as a "helix breaker".
        {"PRO", { _BB_PRO,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},{"CG","HG2"},{"CG","HG3"},
                  {"CG","CD"},{"CD","HD2"},{"CD","HD3"} }},

        // PHE (페닐알라닌 / Phenylalanine) — 6원 방향족 벤젠 고리.
        // 마지막 결합 CD2-CG는 고리 닫힘.  χ2 = CB-CG (고리 회전).
        // 방향족 고리 회전은 χ2로 표현되지만 C2 대칭으로 ±180° 회전이 동등.
        // 6-membered benzene ring.  CD2-CG closes the ring.
        // χ2 = CB-CG (ring flipping).  C2 symmetry: +180° ≡ -180° flip.
        {"PHE", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},
                  {"CG","CD1"},{"CD1","HD1"},{"CD1","CE1"},{"CE1","HE1"},
                  {"CE1","CZ"},{"CZ","HZ"},{"CZ","CE2"},{"CE2","HE2"},
                  {"CE2","CD2"},{"CD2","HD2"},{"CD2","CG"} }},   // ring closure / 고리 닫힘

        // TRP (트립토판 / Tryptophan) — 가장 큰 아미노산. 인돌 이중 고리 시스템.
        // 5원 피롤 고리(CG-CD1-NE1-CE2-CD2)와 6원 벤젠 고리(CE2-CZ2-CH2-CZ3-CE3-CD2) 융합.
        // 두 개의 고리 닫힘 결합이 있음.  χ1 = CA-CB, χ2 = CB-CG.
        // Largest amino acid.  Fused bicyclic indole (5-ring pyrrole + 6-ring benzene).
        // Two ring-closure bonds.  χ1 = CA-CB, χ2 = CB-CG.
        {"TRP", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},
                  {"CG","CD1"},{"CD1","HD1"},{"CD1","NE1"},{"NE1","HE1"},
                  {"NE1","CE2"},{"CE2","CD2"},{"CD2","CG"},            // 5-ring closure / 5원 고리 닫힘
                  {"CE2","CZ2"},{"CZ2","HZ2"},{"CZ2","CH2"},{"CH2","HH2"},
                  {"CH2","CZ3"},{"CZ3","HZ3"},{"CZ3","CE3"},{"CE3","HE3"},
                  {"CE3","CD2"} }},                                    // 6-ring closure / 6원 고리 닫힘

        // SER (세린 / Serine) — 극성 수산기 OG. 수소 결합 공여/수용 가능.
        // χ1 = CA-CB, χ2 = CB-OG (수산기 방향).
        // Polar hydroxyl OG; hydrogen bond donor and acceptor.
        // χ1 = CA-CB, χ2 = CB-OG (hydroxyl orientation).
        {"SER", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},{"CB","OG"},{"OG","HG"} }},

        // THR (트레오닌 / Threonine) — β-분지형 + 수산기. OG1 + CG2(메틸) 두 갈래.
        // χ1 = CA-CB. OG1의 수산기 방향이 중요하나 메틸 CG2는 대칭이라 χ2 불필요.
        // β-branched + hydroxyl.  OG1 (hydroxyl) + CG2 (methyl).
        // χ1 = CA-CB; CG2 methyl is symmetric so no χ2 needed.
        {"THR", { _BB,
                  {"CA","CB"},{"CB","HB"},
                  {"CB","OG1"},{"OG1","HG1"},
                  {"CB","CG2"},{"CG2","HG21"},{"CG2","HG22"},{"CG2","HG23"} }},

        // CYS (시스테인 / Cysteine) — 티올기 SG. 환원형(SH); 산화형 이황화물은 CYX(미구현).
        // χ1 = CA-CB, χ2 = CB-SG (티올 방향).  SG는 금속 배위 결합에도 관여.
        // Thiol SG.  Reduced form (SH); oxidised disulfide form is CYX (TODO).
        // χ1 = CA-CB, χ2 = CB-SG.  SG also participates in metal coordination.
        {"CYS", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},{"CB","SG"},{"SG","HG"} }},

        // MET (메티오닌 / Methionine) — 티오에테르 SD. 가장 긴 곁사슬 중 하나 (4개 χ).
        // χ1=CA-CB, χ2=CB-CG, χ3=CG-SD, χ4=SD-CE.  SD-CE 결합은 유연하고 낮은 에너지 장벽.
        // Thioether SD.  One of the longest sidechains; 4 χ angles.
        // χ1-χ4.  SD-CE bond has a low rotational barrier (~1 kcal/mol).
        {"MET", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},{"CG","HG2"},{"CG","HG3"},{"CG","SD"},
                  {"SD","CE"},{"CE","HE1"},{"CE","HE2"},{"CE","HE3"} }},

        // ASP (아스파르트산 / Aspartate) — pH 7에서 음전하(-1). 카르복실산기 CG(OD1,OD2).
        // OD1·OD2는 이온화 상태에서 공명으로 동등.  χ1=CA-CB, χ2=CB-CG.
        // Negatively charged (-1) at pH 7.  Carboxylate CG(OD1,OD2).
        // OD1 and OD2 are resonance-equivalent in the deprotonated form.
        {"ASP", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},{"CG","OD1"},{"CG","OD2"} }},

        // ASN (아스파라긴 / Asparagine) — 중성 아미드기 CG(OD1, ND2).
        // ND2의 두 수소(HD21, HD22)는 수소 결합 공여체.  χ1=CA-CB, χ2=CB-CG.
        // Neutral amide CG(OD1 carbonyl, ND2 amino).
        // ND2 hydrogens are H-bond donors.  χ1 = CA-CB, χ2 = CB-CG.
        {"ASN", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},{"CG","OD1"},
                  {"CG","ND2"},{"ND2","HD21"},{"ND2","HD22"} }},

        // GLU (글루탐산 / Glutamate) — pH 7에서 음전하(-1). ASP보다 탄소 하나 더 긴 곁사슬.
        // χ1=CA-CB, χ2=CB-CG, χ3=CG-CD.
        // Negatively charged (-1) at pH 7; one CH₂ longer than ASP.
        // χ1 = CA-CB, χ2 = CB-CG, χ3 = CG-CD.
        {"GLU", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},{"CG","HG2"},{"CG","HG3"},
                  {"CG","CD"},{"CD","OE1"},{"CD","OE2"} }},

        // GLN (글루타민 / Glutamine) — 중성 아미드기. GLU의 중성 유사체.
        // χ1=CA-CB, χ2=CB-CG, χ3=CG-CD.
        // Neutral amide; neutral counterpart of GLU.
        // χ1 = CA-CB, χ2 = CB-CG, χ3 = CG-CD.
        {"GLN", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},{"CG","HG2"},{"CG","HG3"},
                  {"CG","CD"},{"CD","OE1"},
                  {"CD","NE2"},{"NE2","HE21"},{"NE2","HE22"} }},

        // LYS (라이신 / Lysine) — pH 7에서 양전하(+1). 긴 알킬 사슬 + 말단 아미노기 NZ.
        // 5개 χ 각도(χ1-χ5). NZ의 HZ1-HZ3는 수소 결합 공여체.
        // Positively charged (+1) at pH 7.  Long alkyl chain + terminal NH₃⁺.
        // 5 χ angles (χ1-χ5).  NZ protons are strong H-bond donors.
        {"LYS", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},{"CG","HG2"},{"CG","HG3"},
                  {"CG","CD"},{"CD","HD2"},{"CD","HD3"},
                  {"CD","CE"},{"CE","HE2"},{"CE","HE3"},
                  {"CE","NZ"},{"NZ","HZ1"},{"NZ","HZ2"},{"NZ","HZ3"} }},

        // ARG (아르기닌 / Arginine) — pH 7에서 양전하(+1). 구아니디늄기(CZ-NH1-NH2)가 특징.
        // 공명 구조로 3개의 N이 전하를 공유 → 평면 구조 + 높은 pKa(≈12.5).
        // χ1=CA-CB, χ2=CB-CG, χ3=CG-CD, χ4=CD-NE. CZ-NH1·CZ-NH2는 고정(공명).
        // Positively charged (+1) at pH 7.  Guanidinium (CZ-NH1-NH2) is planar
        // due to resonance delocalisation over 3 N atoms; pKa ≈ 12.5.
        // χ1-χ4.  CZ-NH1 and CZ-NH2 are FIXED (partial double-bond character).
        {"ARG", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},{"CG","HG2"},{"CG","HG3"},
                  {"CG","CD"},{"CD","HD2"},{"CD","HD3"},
                  {"CD","NE"},{"NE","HE"},{"NE","CZ"},
                  {"CZ","NH1"},{"NH1","HH11"},{"NH1","HH12"},
                  {"CZ","NH2"},{"NH2","HH21"},{"NH2","HH22"} }},

        // HID (히스티딘 — Nδ1 프로토네이션 형태 / Histidine, Nδ1-protonated)
        // 중성, pH ~6.0 근처에서 가장 흔한 형태.  이미다졸 5원 고리.
        // ND1에 수소(HD1)가 있고 NE2는 고독쌍(lone pair) 질소(NB).
        // 고리 닫힘: CD2-CG.  χ1=CA-CB, χ2=CB-CG.
        // Neutral HID form (most common near pH 6.0).  5-membered imidazole ring.
        // ND1 carries H (HD1); NE2 is lone-pair N (NB type).
        // Ring closure: CD2-CG.  χ1 = CA-CB, χ2 = CB-CG.
        {"HID", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},
                  {"CG","ND1"},{"ND1","HD1"},{"ND1","CE1"},{"CE1","HE1"},
                  {"CE1","NE2"},{"NE2","CD2"},{"CD2","HD2"},{"CD2","CG"} }},  // ring / 고리 닫힘

        // HIE (히스티딘 — Nε2 프로토네이션 형태 / Histidine, Nε2-protonated)
        // 중성, ND1은 고독쌍 질소.  NE2에 수소(HE2)가 있음.
        // Neutral HIE form.  ND1 is lone-pair N; NE2 carries H (HE2).
        {"HIE", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},
                  {"CG","ND1"},{"ND1","CE1"},{"CE1","HE1"},
                  {"CE1","NE2"},{"NE2","HE2"},{"NE2","CD2"},{"CD2","HD2"},
                  {"CD2","CG"} }},                                             // ring / 고리 닫힘

        // HIP (히스티딘 — 이중 프로토네이션 형태, +1 / Histidine, doubly-protonated, +1)
        // ND1·NE2 둘 다 H를 보유.  pH < 6 환경이나 활성 부위에서 나타남.
        // Both ND1 and NE2 carry H.  Occurs below pH 6 or in enzyme active sites.
        {"HIP", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},
                  {"CG","ND1"},{"ND1","HD1"},{"ND1","CE1"},{"CE1","HE1"},
                  {"CE1","NE2"},{"NE2","HE2"},{"NE2","CD2"},{"CD2","HD2"},
                  {"CD2","CG"} }},                                             // ring / 고리 닫힘

        // TYR (타이로신 / Tyrosine) — PHE + 파라 수산기(OH).  pKa ≈ 10.
        // 6원 방향족 고리 + CZ-OH.  χ1=CA-CB, χ2=CB-CG.
        // Phenylalanine + para-hydroxyl (OH).  pKa ≈ 10.
        // 6-membered aromatic ring + CZ-OH.  χ1 = CA-CB, χ2 = CB-CG.
        {"TYR", { _BB,
                  {"CA","CB"},{"CB","HB2"},{"CB","HB3"},
                  {"CB","CG"},
                  {"CG","CD1"},{"CD1","HD1"},{"CD1","CE1"},{"CE1","HE1"},
                  {"CE1","CZ"},{"CZ","OH"},{"OH","HH"},
                  {"CZ","CE2"},{"CE2","HE2"},{"CE2","CD2"},{"CD2","HD2"},
                  {"CD2","CG"} }},                                             // ring / 고리 닫힘

        // ATP (아데노신 삼인산 / Adenosine triphosphate) — IMPROVEMENTS.md 항목
        // #4 (GAFF2 리간드 힘장) 첫 단계. 여기 있는 건 순수한 결합 연결성뿐이다
        // (표준 유기화학 — 어떤 원자가 어떤 원자와 공유결합하는지는 실험/합성
        // 문헌에서 확립된 사실이지 최적화로 얻는 값이 아니다). 실제 힘장 값
        // (부분 전하·반데르발스 반경/ε)은 검증 가능한 출처를 못 찾아 의도적으로
        // 보류했다 — 표준 RESP 전하 표(예: Meagher, Redman & Carlson 2003,
        // J. Comput. Chem. 24, 1016의 삼인산기 파라미터)를 이 세션 안에서
        // 다시 확인할 방법이 없었고, 검증 못 한 숫자를 "실제 값"인 것처럼
        // 적어 넣는 것은 원소 기호 기반 폴백(현재 상태, 정직하게 부정확함을
        // 알 수 있음)보다 더 위험하다고 판단했다. 이 결합 표만으로도 실질적
        // 개선이 있다: ATP 내부의 실제 공유결합 원자쌍이 이제 비결합 합에서
        // 제외되어, 전에는 없었던 "리간드 내부 하드코어 반발 폭주" 위험이
        // 사라진다(단백질 말단/이황화 결합에서 이미 겪었던 것과 같은 종류의
        // 버그). 전하/반데르발스는 여전히 amber_params.py의 원소 기반 폴백을
        // 거친다 — 부분 전하 0, 반경은 원소 기호로 추정.
        //
        // ATP (adenosine triphosphate) -- first step on IMPROVEMENTS.md item
        // #4 (GAFF2 ligand force field). What's here is pure bond
        // connectivity only (standard organic chemistry -- which atoms are
        // covalently bonded to which is an established structural fact, not
        // a fitted parameter). Real force-field values (partial charges,
        // VDW radius/epsilon) are deliberately NOT included yet -- no
        // independently verifiable source for the standard RESP charge
        // table (e.g. Meagher, Redman & Carlson 2003, J. Comput. Chem. 24,
        // 1016's triphosphate parameters) could be confirmed within this
        // session, and writing down unverified numbers as if they were real
        // values would be riskier than the current element-based fallback
        // (which is honestly, visibly approximate). This bond table alone
        // is still a real improvement: ATP's actual covalently-bonded atom
        // pairs are now excluded from the non-bonded sum, removing a
        // previously-unguarded risk of intra-ligand hard-core-repulsion
        // blowup (the same category of bug already fixed for protein
        // termini/disulfides). Charges/VDW still go through
        // amber_params.py's element-symbol fallback -- zero partial charge,
        // element-guessed radius.
        {"ATP", {
            // 아데닌 염기 (Adenine base): 6원 고리 + 5원 고리 융합 퓨린계
            {"N1","C2"},{"C2","N3"},{"N3","C4"},{"C4","C5"},{"C5","C6"},{"C6","N1"},
            {"C4","N9"},{"N9","C8"},{"C8","N7"},{"N7","C5"},        // 5-ring closure / 5원 고리 닫힘
            {"C6","N6"},                                             // exocyclic amino / 곁사슬 아미노기
            {"N9","C1'"},                                            // glycosidic bond / 글리코시드 결합
            // 리보스 (Ribose, 5원 furanose 고리 + 2'/3'-OH)
            {"C1'","C2'"},{"C2'","C3'"},{"C3'","C4'"},{"C4'","O4'"},{"O4'","C1'"},
            {"C2'","O2'"},{"C3'","O3'"},{"C4'","C5'"},
            // 삼인산기 사슬 (Triphosphate chain: alpha-beta-gamma)
            {"C5'","O5'"},{"O5'","PA"},
            {"PA","O1A"},{"PA","O2A"},{"PA","O3A"},
            {"O3A","PB"},{"PB","O1B"},{"PB","O2B"},{"PB","O3B"},
            {"O3B","PG"},{"PG","O1G"},{"PG","O2G"},{"PG","O3G"},
        }},
    };
    return t;
}

// 백본 매크로 해제 — 이 이하에서는 사용하지 않으므로 오염 방지.
// Undefine backbone macros to avoid polluting the rest of the translation unit.
#undef _BB
#undef _BB_GLY
#undef _BB_PRO

// ── 회전 가능 결합 명세 (Rotatable bond specifications) ──────────────────────
//
// 각 잔기에 대해 회전 가능한 결합을 (원자명_i, 원자명_j, BondKind) 튜플로 정의.
// build()가 이 명세를 실제 원자 인덱스로 변환해 rot_bonds를 채운다.
//
// Per-residue rotatable bonds as (atomname_i, atomname_j, BondKind) tuples.
// build() resolves these names to atom indices and populates rot_bonds.
//
// 회전 가능 기준 (Criteria for "rotatable"):
//  ① 고리 내 결합이 아닐 것 (PHE·TYR·TRP·HIS 방향족 고리, PRO 피롤리딘 고리)
//     Not part of a ring (aromatic rings of PHE/TYR/TRP/HIS; pyrrolidine of PRO)
//  ② 두 결합 모두 말단 원자가 아닐 것 (메틸기 H, 카르보닐 O 등 leaf 원자 제외)
//     Neither atom should be a leaf (methyl H, carbonyl O, etc. — nothing to rotate)
//  ③ 이중 결합 성격이 없을 것 (펩타이드 C-N: 공명으로 회전 장벽 ≈ 20 kcal/mol)
//     No partial double-bond character (peptide C-N: resonance barrier ≈ 20 kcal/mol)
//
// PRO 특별 처리 (Pro special case):
//   피롤리딘 고리가 N을 Cδ와 묶어 N-CA 결합(φ)을 -65° 근처로 고정.
//   따라서 PRO에는 BACKBONE_PSI(CA-C)만 등록; BACKBONE_PHI는 없음.
//   The pyrrolidine ring tethers N to Cδ, locking the N-CA dihedral (φ)
//   near -65°.  Only BACKBONE_PSI (CA-C) is registered for PRO; no PHI.
//
// ARG 특별 처리 (Arg special case):
//   CZ-NH1·CZ-NH2 결합은 구아니디늄 공명으로 부분 이중 결합 성격.
//   rot_specs에서 의도적으로 제외 (χ4 = CD-NE까지만 등록).
//   CZ-NH1 and CZ-NH2 have partial double-bond character due to guanidinium
//   resonance — intentionally excluded.  Only χ1-χ4 (up to CD-NE) registered.
static const std::unordered_map<std::string, std::vector<RotSpec>>&
rot_specs() {
    using K = BondKind;
    static const std::unordered_map<std::string, std::vector<RotSpec>> t = {
        // φ: N→CA,  ψ: CA→C  (모든 잔기에 공통 — 별도 주석 생략)
        // φ: N→CA,  ψ: CA→C  (universal backbone — comments omitted per-entry)
        {"GLY", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI} }},
        // ALA: 메틸 Cβ는 회전해도 에너지 변화가 없어 χ1 등록 불필요.
        //      Methyl Cβ has 3-fold symmetry; rotating it changes nothing observable.
        {"ALA", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI} }},
        // VAL: χ1 = CA-CB (두 메틸기 방향 결정)
        //      χ1 = CA-CB (determines orientation of both methyl groups)
        {"VAL", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN} }},
        // LEU: χ1=CA-CB, χ2=CB-CG (두 말단 메틸기 방향 결정)
        {"LEU", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN} }},
        // ILE: χ1=CA-CB, χ2=CB-CG1 (CG2 메틸은 대칭이라 제외)
        //      χ2 = CB-CG1 (CG2 methyl omitted — symmetric)
        {"ILE", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG1",K::SIDECHAIN} }},
        // PRO: φ 고정 (피롤리딘 고리). ψ만 등록.
        //      φ locked by pyrrolidine ring.  Only ψ registered.
        {"PRO", { {"CA","C",K::BACKBONE_PSI} }},
        // PHE: χ2=CB-CG (방향족 고리 회전). C2 대칭: +180° ≡ -180°.
        //      χ2 = CB-CG (ring flip).  C2 symmetry: +180° ≡ -180°.
        {"PHE", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN} }},
        // TRP: χ2=CB-CG (인돌 고리 방향 결정)
        //      χ2 = CB-CG (indole ring orientation)
        {"TRP", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN} }},
        // SER: χ2=CB-OG (수산기 방향 — 수소 결합 네트워크에 민감)
        //      χ2 = CB-OG (hydroxyl orientation — sensitive to H-bond network)
        {"SER", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","OG",K::SIDECHAIN} }},
        // THR: χ1=CA-CB만 등록. OG1 방향(χ2)은 중요하나 β-분지라 이동 폭이 제한됨.
        //      Only χ1. OG1 orientation matters but β-branching limits sampling.
        {"THR", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN} }},
        // CYS: χ2=CB-SG (티올 방향)
        {"CYS", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","SG",K::SIDECHAIN} }},
        // MET: χ1-χ4 (가장 유연한 곁사슬 중 하나)
        //      χ1-χ4 (one of the most flexible sidechains)
        {"MET", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN},
                  {"CG","SD",K::SIDECHAIN}, {"SD","CE",K::SIDECHAIN} }},
        // ASP: χ2=CB-CG (카르복실기 방향). OD1·OD2는 공명으로 동등.
        //      χ2 = CB-CG (carboxylate orientation).  OD1/OD2 resonance-equivalent.
        {"ASP", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN} }},
        // ASN: χ2=CB-CG (아미드 평면 방향)
        //      χ2 = CB-CG (amide plane orientation)
        {"ASN", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN} }},
        // GLU: χ3=CG-CD (카르복실기 방향)
        {"GLU", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN},
                  {"CG","CD",K::SIDECHAIN} }},
        // GLN: χ3=CG-CD (아미드 평면 방향)
        {"GLN", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN},
                  {"CG","CD",K::SIDECHAIN} }},
        // LYS: χ1-χ5. CE-NZ(χ5): 말단 아미노기 방향 (수소 결합에 중요).
        //      χ1-χ5.  CE-NZ (χ5): terminal amino orientation (critical for H-bonds).
        {"LYS", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN},
                  {"CG","CD",K::SIDECHAIN}, {"CD","CE",K::SIDECHAIN},
                  {"CE","NZ",K::SIDECHAIN} }},
        // ARG: χ1-χ4 (CD-NE까지). CZ-NH1·CZ-NH2는 공명 고정이므로 제외.
        //      χ1-χ4 (up to CD-NE).  CZ-NH bonds excluded (resonance-fixed).
        {"ARG", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN},
                  {"CG","CD",K::SIDECHAIN}, {"CD","NE",K::SIDECHAIN} }},
        // HID/HIE/HIP: χ2=CB-CG (이미다졸 고리 방향). 고리 내 결합은 제외.
        //              χ2 = CB-CG (imidazole ring orientation).  Ring bonds excluded.
        {"HID", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN} }},
        {"HIE", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN} }},
        {"HIP", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN} }},
        // TYR: χ2=CB-CG (방향족 고리 방향). C2 대칭이지만 OH로 인해 완전 동등하지는 않음.
        //      χ2 = CB-CG (ring orientation).  C2 broken by para-OH (not fully symmetric).
        {"TYR", { {"N","CA",K::BACKBONE_PHI}, {"CA","C",K::BACKBONE_PSI},
                  {"CA","CB",K::SIDECHAIN}, {"CB","CG",K::SIDECHAIN} }},
    };
    return t;
}

// ── BondTopology 클래스 ────────────────────────────────────────────────────────
//
// 하나의 단백질 구조 전체에 대한 공유 결합 그래프.
// PDB 파싱 직후 Python에서 build()를 호출해 단 한 번 구성한다.
// 이후 PhysicsEngine이 에너지 계산 및 MC 이동에 참조한다(P1.4/P1.5).
//
// Covalent bond graph for an entire protein structure.
// Built once by calling build() from Python immediately after PDB parsing.
// Subsequent PhysicsEngine calls reference it for energy terms and MC moves.
//
// 공개 필드 (Public fields):
//   N         — 전체 원자 수 (파티클 배열 크기와 동일)
//               total atom count (equals size of Particle array)
//   adj       — 비방향 인접 목록.  adj[i] = {j, k, …}: i에 직접 결합된 원자 인덱스들.
//               undirected adjacency list; adj[i] lists all atoms directly bonded to i
//   bonds     — 모든 결합 쌍 (i < j).  에너지 1-2 배제 목록 생성에 사용.
//               all bond pairs with i<j; used to build 1-2 exclusion lists
//   rot_bonds — 회전 가능 결합 목록.  MC 이동 축으로 사용.
//               rotatable bonds; each is a candidate MC move axis
class BondTopology {
public:
    int                              N = 0;
    std::vector<std::vector<int>>    adj;
    std::vector<std::pair<int,int>>  bonds;
    std::vector<RotBond>             rot_bonds;
    // rot_bond_sides[k] = j_side(rot_bonds[k].i, rot_bonds[k].j)
    // Pre-computed once in build() so the hot MC step loop can skip DFS.
    // build() 호출 시 한 번 계산해 MC 루프 내 DFS 비용을 제거한다.
    std::vector<std::vector<int>>    rot_bond_sides;

    // P1.4b — 1-2/1-3 비결합 배제 집합 (Non-bonded exclusion sets)
    //
    // AMBER 관례:
    //   1-2 쌍 (직접 결합, ~1.5 Å): 비결합 합산에서 완전히 제외.
    //   1-3 쌍 (결합각 분리, ~2.4 Å): 비결합 합산에서 완전히 제외.
    //   1-4 쌍 (세 결합): 포함, 단 LJ×½ + 전하×5/6 로 스케일 다운 (TODO P1.4b+).
    //
    // AMBER convention:
    //   1-2 pairs (direct bond, ~1.5 Å): fully excluded from non-bonded sum.
    //   1-3 pairs (angle-separated, ~2.4 Å): fully excluded.
    //   1-4 pairs (three bonds): included but scaled (LJ×½, charge×5/6) — TODO P1.4b+.
    //
    // 저장 형식: excl[i] = j > i 인 배제 파트너들의 정렬된 목록.
    //           pair_e() 호출 전에 is_excluded(i, j)로 빠르게 체크.
    // Storage:  excl[i] = sorted list of j > i excluded from pair_e with i.
    //           Check via is_excluded(i, j) before calling pair_e().
    std::vector<std::vector<int>>    excl;

    // P1.4a — 이중면체각 레코드 목록 (Pre-built dihedral energy records)
    //
    // 각 회전 가능 결합에 대해 4원자 시퀀스 (a, b=rb.i, c=rb.j, d)를 하나 저장.
    // dihedral_e()가 이 목록을 순회해 Fourier 합산을 계산.
    //
    // One DihRecord per rotatable bond.  a = first adj[rb.i] ≠ rb.j;
    // d = first adj[rb.j] ≠ rb.i.  Energy terms from get_dih_terms().
    std::vector<DihRecord>           dihedrals;

    // ── 레버암 스케일 (Lever-arm scale per rotatable bond) ───────────────────
    //
    // 같은 δφ라도 j-side 원자 수(N_down)가 클수록 평균 선형 변위(lever-arm effect)가
    // 커져 거의 모든 이동이 거부된다 (큰 단백질의 N-말단 결합에서 특히 심각).
    //
    // 완화 방법: max_angle에 scale_k = sqrt(N_ref / N_down_k) 를 곱해
    // 원자당 RMS 변위를 결합 종류에 관계없이 일정하게 유지한다.
    // N_ref = 10 (전형적인 곁사슬 j-side 크기). 범위 클램프: [0.05, 1.0].
    //
    // Each bond's scale_k = sqrt(N_ref / N_downstream), clamped to [0.05, 1.0].
    // Multiplied into max_angle before sampling δφ so that per-atom RMS
    // displacement is approximately constant regardless of bond position.
    std::vector<double>              rot_bond_scale;

    // ── 크랭크샤프트 협동 이동 쌍 (Crankshaft concerted-move pairs) ──────────
    //
    // 같은 Cα 원자를 공유하는 (φ 결합, ψ 결합) 쌍:  φ.j == ψ.i == CA 인덱스.
    //
    // MC 루프에서 +δ(φ) → −δ(ψ) 순으로 적용하면:
    //   φ: CA + 곁사슬 + 이후 백본 전체가 +δ 만큼 회전 (Rodrigues)
    //   ψ: C 이후 원자들이 −δ 만큼 복원 회전 (근사 상쇄, O(δ²) 잔여 변위)
    // 순 효과: 잔기 i의 곁사슬만 크게 이동, 이후 백본은 거의 제자리.
    // 거부율이 일반 단일 비틀림 이동보다 크게 낮아 대형 단백질 탐색에 효과적.
    //
    // Pairs (phi_rot_bond_idx, psi_rot_bond_idx) sharing the same Cα (φ.j == ψ.i).
    // Applied as +δ around φ then −δ around ψ: sidechain moves, downstream
    // backbone approximately restores (O(δ²) residual).  Yields higher acceptance
    // than single torsion moves for large proteins.
    std::vector<std::pair<int,int>>  concerted_pairs;

    // ── 곁사슬 협동 이동 쌍 (Concerted sidechain-pair moves, IMPROVEMENTS.md
    // 항목 #2의 MC 혼합 병목 진단 결과에 대한 대응) ──────────────────────────
    //
    // 위 concerted_pairs(φ/ψ 크랭크샤프트)는 백본 위에서만 작동하고, 두 결합이
    // 계층적으로 중첩되어(ψ가 φ의 하류) 안전하다. 실측 진단 결과(run_mc_diagnostic,
    // 1UBQ vs 1YPI): 단일 비틀림 이동은 크기와 무관하게 이중면체각 장벽이
    // 병목이지만, 크랭크샤프트는 1YPI(494잔기, 조밀하게 밀집)에서 거부된 대형
    // 이동의 90.8%가 입체 장애(하드코어 반발) 때문 — 중앙값 46,213 kcal/mol의
    // 심각한 원자 겹침. 기존 2-자유도 협동 이동으로는 밀집된 큰 단백질에서
    // 곁사슬이 이웃을 피해 돌아갈 공간이 부족하다는 뜻이다.
    //
    // 해법: 서로 무관한(disjoint rot_bond_sides) 두 곁사슬 회전 결합이 3차원
    // 공간에서 가까이 있으면(피벗 원자 거리 < cutoff), 하나가 이웃 쪽으로
    // 비집고 들어가려 할 때 다른 하나가 동시에 비켜줄 기회를 준다 — 두 번의
    // 독립적인 단일 결합 이동이 우연히 순서대로 일어나길 기다리는 것보다
    // 훨씬 효율적. identify_concerted_sidechain_pairs()에서 파싱 시점에 한 번만
    // 계산(좌표가 필요하므로 build() 자체에는 넣지 않음 — build()는 좌표 없이
    // 순수 위상만 다루도록 유지).
    //
    // The backbone crankshaft pairs above only work on the backbone, and are
    // safe because the two bonds are hierarchically nested (ψ downstream of φ).
    // Direct measurement (run_mc_diagnostic, 1UBQ vs 1YPI) found: single-bond
    // torsion moves are dihedral-barrier-limited regardless of size, but
    // crankshaft moves on 1YPI (494 res, densely packed) are 90.8% steric-
    // dominant among rejected large moves -- median 46,213 kcal/mol, genuine
    // deep overlaps. The existing 2-DOF concerted move doesn't have enough
    // room on a densely packed large protein for a sidechain to route around
    // its neighbours.
    //
    // Fix: two independent (disjoint rot_bond_sides) sidechain rotatable bonds
    // that are spatially close (pivot-atom distance < cutoff) get the chance
    // to move simultaneously in one proposal -- so if one sidechain's swing
    // would clash with a neighbour, that neighbour can move out of the way in
    // the same step, instead of needing two separate, uncorrelated single-bond
    // moves to happen in the right order by chance. Computed once at parse
    // time in identify_concerted_sidechain_pairs() (needs coordinates, so
    // deliberately not part of build() itself, which stays coordinate-free).
    std::vector<std::pair<int,int>>  concerted_sidechain_pairs;

    // ── 이황화 결합 쌍 및 구속 (Disulfide bond pairs + restraints, P2.3) ────────
    //
    // ── 이황화 결합 생화학 (Disulfide bond biochemistry) ─────────────────────────
    //
    // 이황화 결합(S–S bond)은 두 시스테인 잔기의 SG 원자 사이에 형성되는
    // 공유 결합이다.  산화 환경에서 형성되며 단백질의 3차 구조를 안정화시키는
    // 주요 공유 가교 역할을 한다.  면역글로불린(항체), 인슐린, 리보핵산분해효소 A
    // 등 수많은 분비 단백질과 세포외 단백질에 나타난다.
    //
    // Disulfide bonds (S–S bonds) form between the SG atoms of two Cysteine
    // residues in oxidising environments and act as covalent cross-links that
    // stabilise the tertiary structure.  They are prevalent in secreted proteins,
    // antibodies, insulin, RNase A, and many extracellular proteins.
    //
    // PDB 구조에서 이황화 결합의 특징:
    //   SG–SG 거리 ≈ 2.0–2.1 Å (공유 결합 거리).
    //   정상 범위: 1.9–2.3 Å.  2.5 Å를 초과하면 비결합 상태.
    //   탐지 기준: SG–SG < 2.5 Å (0.5 Å 여유 포함).
    //   평형 거리: r₀ = 2.044 Å (ff14SB CYX 잔기 템플릿에서).
    //
    // Characteristics of disulfide bonds in PDB structures:
    //   SG–SG distance ≈ 2.0–2.1 Å (covalent bond length).
    //   Detection threshold: < 2.5 Å (generous 0.5 Å tolerance).
    //   Equilibrium distance: r₀ = 2.044 Å (from AMBER ff14SB CYX template).
    //
    // ── 이 구현에서의 처리 (How this implementation handles them) ────────────────
    //
    // 등록 방식:
    //   Python _parse_pdb()에서 SG 원자 인덱스를 수집한 후 SG–SG 거리를 검사.
    //   2.5 Å 미만인 쌍마다 add_disulfide(i, j)를 호출해 이 목록에 등록.
    //
    // Registration:
    //   _parse_pdb() in Python collects SG atom indices, then calls
    //   add_disulfide(i,j) for every SG–SG pair closer than 2.5 Å.
    //
    // 에너지 기여:
    //   총 에너지에 하모닉 구속 E_SS = K_SS × (r − r₀)² 추가 (P2.3).
    //   K_SS = 600 kcal/mol/Å² → 약간의 변위에도 큰 에너지 페널티 → 결합 거리 유지.
    //   이 값은 AMBER의 S–S 결합 신축 상수(~166 kcal/mol/Å²)보다 크지만,
    //   MC 샘플러에서 간헐적으로 큰 torsion 이동이 이황화 결합 거리를 크게 변화시킬 수
    //   있으므로 강한 구속으로 처리한다.
    //
    // Energy contribution:
    //   A harmonic restraint E_SS = K_SS × (r − r₀)² is added to total_e() and
    //   to the MC ΔE computation.  K_SS = 600 kcal/mol/Å² is intentionally stiffer
    //   than the AMBER S–S stretching constant (≈166 kcal/mol/Å²) because a single
    //   large torsion MC move can displace SG atoms significantly; the stiff spring
    //   ensures the bond distance stays near r₀ throughout sampling.
    //
    // 1-2 배제:
    //   이황화 결합은 공유 결합이므로 비결합 에너지 합산(pair_e)에서 제외해야 한다.
    //   add_disulfide()는 쌍을 excl[]에도 추가해 1-2 쌍 배제 원칙을 준수한다.
    //
    // 1-2 exclusion:
    //   As a covalent bond, the SS pair must be excluded from the non-bonded sum.
    //   add_disulfide() inserts the pair into excl[] so is_excluded(i,j) returns true.
    std::vector<std::pair<int,int>>  disulfide_pairs;

    // add_disulfide: 이황화 결합 쌍 (i, j)를 등록한다.
    //   • 경계 검사: 유효하지 않은 인덱스나 자기 자신과의 쌍은 조용히 무시.
    //   • 정규화:   항상 i < j 로 저장 (excl과 일관성 유지).
    //   • disulfide_pairs에 추가 후 excl[i]에도 삽입(이진 탐색으로 정렬 유지).
    //
    // add_disulfide: Register a disulfide bond between atoms i and j.
    //   • Boundary check: silently ignore invalid indices or self-pairs.
    //   • Normalise: always store with i < j (consistent with excl convention).
    //   • Push to disulfide_pairs and insert j into excl[i] (keeps excl sorted).
    void add_disulfide(int i, int j) {
        if (i < 0 || j < 0 || i >= N || j >= N || i == j) return;
        if (i > j) std::swap(i, j);
        disulfide_pairs.push_back({i, j});
        // 이황화 SG-SG 쌍을 1-2 비결합 배제 목록에 추가.
        // 이진 탐색으로 정렬된 위치를 찾아 중복 없이 삽입.
        // Insert SS pair into 1-2 exclusion list (binary search keeps list sorted,
        // duplicate check prevents double-insertion).
        auto& vi = excl[i];
        auto pos = std::lower_bound(vi.begin(), vi.end(), j);
        if (pos == vi.end() || *pos != j) vi.insert(pos, j);
    }

    // ── build() ──────────────────────────────────────────────────────────────
    //
    // 파라미터 (Parameters) — 모두 길이 N, 파티클 배열과 동일 순서:
    //   resnames  — 3글자 잔기명 (예: "ALA", "GLY")
    //   atomnames — PDB 원자명 (예: "CA", "OG1")
    //   res_idx   — 잔기별 고유 순차 정수 (0, 1, 2, …).
    //               같은 잔기의 원자는 동일 값 공유.
    //               연속된 잔기는 r → r+1 차이를 가져야 펩타이드 결합이 추가됨.
    //               Python에서 (chain_id, res_seq, icode) 조합에 순번을 매겨 전달.
    //
    // All three arrays are parallel to the Particle array (length N):
    //   resnames  — 3-letter residue name (e.g. "ALA")
    //   atomnames — PDB atom name      (e.g. "CA")
    //   res_idx   — unique sequential residue integer (0, 1, 2, …).
    //               Atoms in the same residue share the same value.
    //               A difference of exactly 1 between consecutive residues
    //               triggers peptide-bond insertion.
    //               Assigned in Python from (chain_id, res_seq, icode) tuples.
    //
    // 알고리즘 단계 (Algorithm steps):
    //   1. (res_idx, atomname) → 파티클 인덱스 역방향 조회 맵 생성.
    //   2. 등장 순서대로 잔기 목록 작성; HIS → HID 정규화.
    //   3. 각 잔기에 AMBER 템플릿 결합 쌍을 적용해 잔기 내 결합 추가.
    //   4. 연속 잔기 쌍(r, r+1) 사이에 펩타이드 결합 C(r)→N(r+1) 추가.
    //   5. rot_specs 표에서 회전 가능 결합 인덱스 추출해 rot_bonds 채움.
    void build(const std::vector<std::string>& resnames,
               const std::vector<std::string>& atomnames,
               const std::vector<int>&          res_idx)
    {
        N = (int)resnames.size();
        adj.assign(N, {});
        bonds.clear();
        rot_bonds.clear();
        if (N == 0) return;

        // ── 단계 1: 역방향 조회 맵 구성 ────────────────────────────────────
        // (res_idx, atomname) → 파티클 배열 내 인덱스 k.
        // std::map 사용: pair<int,string> 비교 연산자가 기본 정의되어 있어 안전.
        // O(N log N) 구성; 이후 각 조회는 O(log N).
        //
        // Step 1: Build reverse lookup map.
        // (res_idx, atomname) → index k in the Particle array.
        // std::map used: pair<int,string> comparison is defined in the standard.
        // O(N log N) build; O(log N) per subsequent lookup.
        std::map<std::pair<int,std::string>, int> lookup;
        for (int k = 0; k < N; ++k)
            lookup[{res_idx[k], atomnames[k]}] = k;

        // ── 단계 2: 잔기 목록 작성 및 잔기명 정규화 ──────────────────────────
        // unique_res: 등장 순서대로 정렬된 잔기 인덱스 목록 (adj 순서 보장).
        // res_of:     res_idx → 정규화된 잔기명.
        //             HIS → HID: PDB의 미지정 HIS는 pH 7 우세 형태인 HID로 처리.
        //
        // Step 2: Build ordered residue list and normalise residue names.
        // unique_res: residue indices in first-encounter order (preserves chain order).
        // res_of:     res_idx → canonical resname.
        //             HIS → HID: unspecified HIS treated as the dominant neutral
        //             form at pH 7 (Nδ1-protonated).
        std::vector<int> unique_res;
        std::unordered_map<int, std::string> res_of;
        for (int k = 0; k < N; ++k) {
            int r = res_idx[k];
            if (res_of.find(r) == res_of.end()) {
                unique_res.push_back(r);
                std::string rn = resnames[k];
                if (rn == "HIS") rn = "HID";
                res_of[r] = rn;
            }
        }

        // ── 내부 헬퍼: 결합 추가 ────────────────────────────────────────────
        // i > j 인 경우 swap해 항상 i < j 순서로 저장 (bonds 중복 방지).
        // adj는 양방향으로 삽입해 비방향 그래프 유지.
        //
        // Internal helper: add a bond between atoms i and j.
        // Normalises to i < j to prevent duplicates in bonds.
        // Inserts both directions into adj for an undirected graph.
        auto add_bond = [&](int i, int j) {
            if (i == j) return;
            if (i > j) std::swap(i, j);
            adj[i].push_back(j);
            adj[j].push_back(i);
            bonds.push_back({i, j});
        };

        // ── 단계 3: 잔기 내 결합 추가 ────────────────────────────────────────
        // 각 잔기에 대해 bond_templates()에서 (원자명_A, 원자명_B) 쌍을 가져오고
        // lookup으로 실제 파티클 인덱스로 변환한다.
        // 원자가 PDB에 없으면 (수소 누락 등) lookup.find()가 end()를 반환 → 조용히 건너뜀.
        //
        // Step 3: Add intra-residue bonds.
        // For each residue, fetch (atomname_A, atomname_B) pairs from bond_templates()
        // and translate to particle indices via lookup.
        // Missing atoms (e.g. no H atoms in PDB) yield lookup misses → silently skipped.
        const auto& btmpl = bond_templates();
        for (int r : unique_res) {
            auto it = btmpl.find(res_of[r]);
            if (it == btmpl.end()) continue;  // 미지원 잔기 (리간드, 비표준 AA 등)
                                               // unsupported residue (ligand, non-standard AA)
            for (auto& [a, b] : it->second) {
                auto ia = lookup.find({r, a});
                auto ib = lookup.find({r, b});
                if (ia != lookup.end() && ib != lookup.end())
                    add_bond(ia->second, ib->second);
            }
        }

        // ── 단계 4: 잔기 간 펩타이드 결합 추가 ──────────────────────────────
        // 연속 잔기 쌍 (r1, r2)에서 r2 == r1 + 1 이어야 펩타이드 결합을 추가한다.
        // r2 ≠ r1 + 1 이면 체인 절단(chain break) 또는 다른 체인 → 결합 없음.
        // res_idx는 Python에서 (chain_id, res_seq, icode) 조합에 순번을 부여했으므로
        // 다른 체인의 잔기는 절대로 연속 정수를 공유하지 않는다.
        //
        // Step 4: Add inter-residue peptide bonds.
        // Only connect residues (r1, r2) where r2 == r1 + 1 (sequential assignment).
        // r2 ≠ r1 + 1 signals a chain break or a different chain → no bond added.
        // Because res_idx is assigned per unique (chain_id, res_seq, icode) in Python,
        // atoms from different chains can never share consecutive indices.
        for (size_t k = 0; k + 1 < unique_res.size(); ++k) {
            int r1 = unique_res[k], r2 = unique_res[k + 1];
            if (r2 != r1 + 1) continue;
            auto ic = lookup.find({r1, "C"});   // C-말단 카르보닐 C
            auto in = lookup.find({r2, "N"});   // 다음 잔기 아미드 N
            if (ic != lookup.end() && in != lookup.end())
                add_bond(ic->second, in->second);
        }

        // ── 단계 4b: 말단/양성자화 변이 원자 패치 ───────────────────────────
        //
        // bond_templates()는 잔기 "내부" 표준형만 다루므로, 사슬 말단이나
        // 비표준 양성자화 상태에서만 나타나는 원자는 어느 템플릿에도 없어
        // add_bond()로 연결되지 않는다. 그 결과 excl[]에도 빠져, 실제로는
        // ~1.0 Å 떨어진 공유 결합 쌍인데도 비결합 항(특히 하드코어 척력)에
        // 그대로 들어가 수십억 kcal/mol 단위의 허구 에너지를 만들어낸다.
        // 잔기명에 관계없이 "이 잔기에 해당 원자가 실제로 존재하는가"만
        // 보고 연결하므로, 어떤 잔기가 사슬의 첫/마지막에 오든 안전하다.
        //
        // Step 4b: patch terminal / alternate-protonation atoms.
        // bond_templates() only covers each residue's internal/standard form,
        // so atoms that only appear at a chain terminus or under a non-default
        // protonation state (N-terminal NH3+ H1/H2/H3, C-terminal COO⁻ OXT,
        // HIS NE2-HE2 when the file has it regardless of the HID default used
        // elsewhere for typing) never get an add_bond() call. That leaves them
        // out of excl[] too, so a real ~1.0 Å covalent pair ends up evaluated
        // as a nonbonded contact — tripping the hard-core term for billions of
        // kcal/mol of spurious energy. This patch bonds them whenever both
        // atoms are actually present, independent of residue identity or
        // position in the chain.
        for (int r : unique_res) {
            auto bondIfPresent = [&](const char* a, const char* b) {
                auto ia = lookup.find({r, a});
                auto ib = lookup.find({r, b});
                if (ia != lookup.end() && ib != lookup.end())
                    add_bond(ia->second, ib->second);
            };
            bondIfPresent("N", "H1");    // N-terminal NH3+
            bondIfPresent("N", "H2");
            bondIfPresent("N", "H3");
            bondIfPresent("C", "OXT");   // C-terminal COO-
            bondIfPresent("NE2", "HE2"); // HIS epsilon-protonated (HIE/HIP); ND1-HD1 already
                                         // covered by the HID template used for typing.
        }

        // ── 단계 5: 회전 가능 결합 인덱스 추출 ──────────────────────────────
        // rot_specs()의 (원자명_i, 원자명_j, BondKind) 튜플을 실제 인덱스로 변환.
        // 원자가 조회되지 않으면 (PRO에서 N-H가 없는 경우 등) 조용히 건너뜀.
        //
        // Step 5: Resolve rotatable bond atom indices.
        // Translate (atomname_i, atomname_j, BondKind) tuples from rot_specs()
        // to actual particle indices.  Unresolved atoms are silently skipped
        // (e.g. Pro has no N-H so any rot_spec referencing Pro's H would miss).
        const auto& rtmpl = rot_specs();
        for (int r : unique_res) {
            auto it = rtmpl.find(res_of[r]);
            if (it == rtmpl.end()) continue;
            for (auto& [a, b, kind] : it->second) {
                auto ia = lookup.find({r, a});
                auto ib = lookup.find({r, b});
                if (ia != lookup.end() && ib != lookup.end())
                    rot_bonds.push_back({ia->second, ib->second, kind});
            }
        }

        // ── P1.4b: 1-2/1-3 비결합 배제 집합 구성 ────────────────────────────
        //
        // AMBER 관례에 따라 직접 결합(1-2)과 결합각 분리(1-3) 쌍을
        // 비결합 에너지 합산에서 제외한다.
        //
        // 1-2 배제: bonds 목록에서 직접 추출 (이미 lo < hi 형식).
        // 1-3 배제: adj를 두 번 탐색해 i → j → k (k ≠ i) 경로를 모두 열거.
        //          (lo, hi) = (min(i,k), max(i,k)) 형식으로 excl[lo]에 추가.
        // 마지막으로 각 excl[i]를 정렬·중복 제거해 이진 탐색 가능하게 만든다.
        //
        // Build 1-2/1-3 exclusion sets per AMBER convention.
        // 1-2: direct bonds (i,j) from bonds list (already canonical i < j).
        // 1-3: two-hop path i → j → k (k ≠ i), stored as (min, max).
        // After filling: sort and deduplicate each excl[i] for O(log N) lookup.
        excl.assign(N, {});
        for (auto& [bi, bj] : bonds)           // 1-2: bonds is already i < j
            excl[bi].push_back(bj);
        for (int i = 0; i < N; ++i)            // 1-3: path of length 2
            for (int j2 : adj[i])
                for (int k : adj[j2]) {
                    if (k == i) continue;
                    int lo = i < k ? i : k;
                    int hi = i < k ? k : i;
                    excl[lo].push_back(hi);
                }
        for (auto& v : excl) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }

        // ── P1.4a: 이중면체각 레코드 구성 ────────────────────────────────────
        //
        // 각 회전 가능 결합 (rb.i → rb.j) 에 대해 4원자 시퀀스를 하나 선택:
        //   a = adj[rb.i]에서 rb.j가 아닌 첫 번째 이웃 (i-side 앵커)
        //   b = rb.i
        //   c = rb.j
        //   d = adj[rb.j]에서 rb.i가 아닌 첫 번째 이웃 (j-side 앵커)
        // 말단 결합(a 또는 d가 없음)은 건너뜀.
        //
        // For each rotatable bond (rb.i → rb.j), pick one 4-atom sequence:
        //   a = first neighbor of rb.i that is not rb.j (i-side anchor)
        //   b = rb.i,  c = rb.j
        //   d = first neighbor of rb.j that is not rb.i (j-side anchor)
        // Skip terminal bonds where a or d cannot be found.
        dihedrals.clear();
        for (size_t k = 0; k < rot_bonds.size(); ++k) {
            const RotBond& rb = rot_bonds[k];
            int a_idx = -1, b_idx = -1;
            for (int nb : adj[rb.i]) if (nb != rb.j) { a_idx = nb; break; }
            for (int nb : adj[rb.j]) if (nb != rb.i) { b_idx = nb; break; }
            if (a_idx < 0 || b_idx < 0) continue;
            auto terms = get_dih_terms(rb.kind, atomnames[rb.j]);
            if (!terms.empty())
                dihedrals.push_back({a_idx, rb.i, rb.j, b_idx, std::move(terms)});
        }

        // ── 사전 계산: 각 회전 가능 결합의 j-side 원자 집합 ─────────────────
        // MC 루프에서 매 스텝마다 DFS를 반복 실행하는 비용을 없애기 위해
        // build() 시점에 한 번만 j_side()를 계산해 캐싱한다.
        //
        // Pre-compute j-side atom sets for all rotatable bonds.
        // Avoids repeating DFS inside the hot MC step loop.
        rot_bond_sides.resize(rot_bonds.size());
        for (size_t k = 0; k < rot_bonds.size(); ++k)
            rot_bond_sides[k] = j_side(rot_bonds[k].i, rot_bonds[k].j);

        // ── 레버암 스케일 사전 계산 ──────────────────────────────────────────
        // N_down이 크면 같은 δφ에서 원자당 선형 변위가 √(N_down)에 비례해 커짐.
        // scale_k = sqrt(N_REF / N_down_k) 로 보정해 원자당 RMS 변위를 일정하게 유지.
        //
        // scale_k = sqrt(N_ref / N_downstream_k), clamped to [0.05, 1.0].
        // Reference size N_ref = 10 ≈ typical sidechain j-side.
        constexpr double LEVER_NREF = 10.0;
        rot_bond_scale.resize(rot_bonds.size());
        for (size_t k = 0; k < rot_bonds.size(); ++k) {
            double ns = std::max(1.0, (double)rot_bond_sides[k].size());
            rot_bond_scale[k] = std::min(1.0, std::max(0.05, std::sqrt(LEVER_NREF / ns)));
        }

        // ── 크랭크샤프트 협동 이동 쌍 구성 ──────────────────────────────────
        // φ 결합(i=N, j=CA)과 ψ 결합(i=CA, j=C)이 동일한 CA 원자를 공유하는 쌍.
        // φ.j == ψ.i == CA 조건으로 빠르게 매칭.
        //
        // Match φ and ψ bonds sharing the same Cα: φ.j == ψ.i == CA_atom_idx.
        {
            std::unordered_map<int,int> phi_at_ca, psi_at_ca;
            for (size_t k = 0; k < rot_bonds.size(); ++k) {
                if (rot_bonds[k].kind == BondKind::BACKBONE_PHI)
                    phi_at_ca[rot_bonds[k].j] = (int)k;
                else if (rot_bonds[k].kind == BondKind::BACKBONE_PSI)
                    psi_at_ca[rot_bonds[k].i] = (int)k;
            }
            for (auto& [ca_idx, pk] : phi_at_ca) {
                auto it = psi_at_ca.find(ca_idx);
                if (it != psi_at_ca.end())
                    concerted_pairs.push_back({pk, it->second});
            }
        }
    }

    // ── identify_concerted_sidechain_pairs() ────────────────────────────────
    //
    // build()가 좌표 없이 순수 위상만으로 concerted_pairs(백본 크랭크샤프트)를
    // 구성하는 것과 달리, 이 함수는 실제 좌표가 필요하므로 별도 함수로 분리했다
    // (build() 자체는 건드리지 않음 — 이미 검증된 함수 보호). Python에서
    // topo.build(...) 직후, 초기 원자 좌표가 준비된 시점에 한 번만 호출한다.
    //
    // 후보 쌍 조건:
    //   1. 둘 다 BondKind::SIDECHAIN (백본 φ/ψ는 이미 concerted_pairs가 처리).
    //   2. rot_bond_sides가 서로 겹치지 않음(disjoint) — 같은 잔기의 χ1/χ2처럼
    //      한쪽이 다른 쪽의 상류/하류에 있는 경우를 배제한다. 잔기 ID를 따로
    //      저장하지 않아도 이 조건만으로 "완전히 독립적으로 움직이는가"를
    //      정확히 포착한다 — 잔기 동일성보다 더 일반적이고 정확한 기준.
    //   3. 피벗 원자(j) 간 거리가 cutoff 이내 — 곁사슬이 서로 닿을 만큼 가까운지
    //      확인하는 값싼 근사(정확한 최근접-원자 거리보다 저렴; 검증 결과 너무
    //      거칠면 나중에 개선).
    //
    // O(n_rotbonds²) 1회 비용(파싱 시점, MC 스텝마다가 아님). 매우 크거나
    // 조밀한 단백질에서 쌍 개수가 지나치게 커지지 않도록 상한을 둔다.
    //
    // Unlike build() (pure topology, no coordinates), this needs real atom
    // positions, so it's a separate function -- build() itself is left
    // untouched to protect an already-verified function. Called once from
    // Python right after topo.build(...), once initial atom coordinates are
    // available.
    //
    // Candidate pair conditions:
    //   1. Both BondKind::SIDECHAIN (backbone φ/ψ is already covered by
    //      concerted_pairs above).
    //   2. Disjoint rot_bond_sides -- excludes cases like the same residue's
    //      χ1/χ2 where one bond is upstream/downstream of the other. This
    //      condition alone correctly captures "do these two move completely
    //      independently" without needing a separately-stored residue ID --
    //      more general and more accurate than a same-residue check.
    //   3. Pivot atom (j) distance within cutoff -- a cheap proxy for "close
    //      enough for their sidechains to actually reach each other" (cheaper
    //      than an exact closest-atom-pair distance; revisit if validation
    //      shows this proxy is too coarse).
    //
    // O(n_rotbonds²) one-time cost at parse time, not per MC step. Capped to
    // avoid an unbounded pair count on very large/dense proteins.
    void identify_concerted_sidechain_pairs(const std::vector<Particle>& init_coords,
                                             double cutoff = 6.0) {
        concerted_sidechain_pairs.clear();
        constexpr size_t MAX_PAIRS = 4000;  // safety valve for very large/dense proteins
        const double cutoff2 = cutoff * cutoff;
        const int nrb = (int)rot_bonds.size();

        auto sides_disjoint = [&](int a, int b) -> bool {
            // rot_bond_sides entries are typically small (a handful to a few
            // dozen atoms); linear-scan intersection check is fine here since
            // this whole function only runs once at parse time.
            const auto& sa = rot_bond_sides[a];
            const auto& sb = rot_bond_sides[b];
            const auto& small  = (sa.size() <= sb.size()) ? sa : sb;
            const auto& big    = (sa.size() <= sb.size()) ? sb : sa;
            for (int idx : small)
                if (std::find(big.begin(), big.end(), idx) != big.end())
                    return false;
            return true;
        };

        for (int a = 0; a < nrb && concerted_sidechain_pairs.size() < MAX_PAIRS; ++a) {
            if (rot_bonds[a].kind != BondKind::SIDECHAIN) continue;
            const Particle& pa = init_coords[rot_bonds[a].j];
            for (int b = a + 1; b < nrb && concerted_sidechain_pairs.size() < MAX_PAIRS; ++b) {
                if (rot_bonds[b].kind != BondKind::SIDECHAIN) continue;
                const Particle& pb = init_coords[rot_bonds[b].j];
                double dx = pa.x - pb.x, dy = pa.y - pb.y, dz = pa.z - pb.z;
                if (dx*dx + dy*dy + dz*dz > cutoff2) continue;
                if (!sides_disjoint(a, b)) continue;
                concerted_sidechain_pairs.push_back({a, b});
            }
        }
    }

    // ── bonded() ─────────────────────────────────────────────────────────────
    //
    // 원자 i와 j가 직접 공유 결합으로 연결되어 있으면 true 반환.
    // 인접 목록 adj[i]에 대한 선형 탐색.  단백질 원자의 평균 결합 차수는 2-4이므로
    // O(1)에 가까운 성능이며 이진 탐색이나 해시 오버헤드보다 유리하다.
    //
    // Returns true iff atoms i and j are directly covalently bonded.
    // Linear scan of adj[i].  Average bond degree in a protein atom is 2-4,
    // so this is effectively O(1) — faster in practice than hash or binary search.
    bool bonded(int i, int j) const noexcept {
        if (i < 0 || i >= N) return false;
        for (int k : adj[i]) if (k == j) return true;
        return false;
    }

    // ── is_excluded() ────────────────────────────────────────────────────────
    //
    // 원자 i와 j가 1-2 또는 1-3 비결합 배제 쌍이면 true 반환.
    // 쌍 (i, j)를 pair_e()에 전달하기 전에 이 함수로 필터링해야 한다.
    //
    // Returns true if pair (i, j) is a 1-2 or 1-3 excluded pair and should
    // be skipped in the non-bonded energy sum.
    //
    // 구현: excl[min(i,j)]에서 max(i,j)를 이진 탐색 — O(log E) ≈ O(1).
    // Implementation: binary search in excl[lo] for hi — O(log E) ≈ O(1).
    bool is_excluded(int i, int j) const noexcept {
        if (i < 0 || j < 0 || i >= N || j >= N) return false;
        if (i > j) std::swap(i, j);
        const auto& v = excl[i];
        return std::binary_search(v.begin(), v.end(), j);
    }

    // ── j_side() ─────────────────────────────────────────────────────────────
    //
    // 결합 (bi→bj)의 j-side 원자 인덱스 집합을 DFS로 반환한다.
    // 이 집합이 P1.5 비틀림 각 MC 이동에서 실제로 회전하는 원자들이다.
    //
    // Returns the set of atom indices on the j-side of bond (bi→bj) via DFS.
    // This set is exactly the atoms that rotate in a P1.5 torsion MC move.
    //
    // 알고리즘 (Algorithm):
    //   visited[bi] = true  → bi를 방문됨으로 표시해 bi 방향으로의 역방향 탐색을 차단.
    //   그 다음 bj에서 DFS 시작 → bi를 넘어가지 않고 bj에서 도달 가능한 모든 원자 수집.
    //   이로써 bi-side 원자들(고정된 쪽)은 결과에 포함되지 않는다.
    //   고리 원자(PHE·TYR·HIS·TRP·PRO)는 두 경로로 연결되어 있지만,
    //   visited 배열이 중복 방문을 막아 올바르게 처리된다.
    //
    //   Set visited[bi] = true to block backtracking through the bond axis.
    //   Then DFS from bj collects all atoms reachable without crossing bi.
    //   The bi-side (fixed atoms) are thus excluded from the result.
    //   Ring atoms (PHE/TYR/HIS/TRP/PRO) are connected via two paths, but
    //   the visited array prevents double-visiting and handles them correctly.
    //
    // 반환값 (Return value):
    //   회전할 원자들의 인덱스 벡터.  bi·bj 모두 범위 밖이면 빈 벡터 반환.
    //   Vector of atom indices that rotate.  Returns empty if bi or bj is out of range.
    std::vector<int> j_side(int bi, int bj) const {
        std::vector<int> side;
        if (bi < 0 || bi >= N || bj < 0 || bj >= N) return side;
        std::vector<bool> visited(N, false);
        visited[bi] = true;   // bi를 장벽으로 설정 / set bi as traversal barrier
        std::vector<int> stk = {bj};
        while (!stk.empty()) {
            int cur = stk.back(); stk.pop_back();
            if (visited[cur]) continue;
            visited[cur] = true;
            side.push_back(cur);
            for (int nb : adj[cur]) stk.push_back(nb);
        }
        return side;
    }
};

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  P1.4 — Bonded Energy Terms  (결합 에너지 항) — NOT YET IMPLEMENTED         ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// ── 왜 아직 구현하지 않는가? (Why deferred?) ───────────────────────────────────
//
// 세 결합 항의 중요도가 서로 다르다:
//
// (a) 결합 신축 · 결합각 굽힘 (Bond stretching and angle bending) ← 안전하게 연기 가능
//   공식:   E_bond  = Σ k_b · (r − r₀)²        [Hookean spring]
//           E_angle = Σ k_θ · (θ − θ₀)²
//
//   P1.5 비틀림 이동이 구현되면 결합 길이와 결합각은 항상 r₀ / θ₀ 에 머문다.
//   이 항들의 ΔE는 스텝당 < 0.1 kcal/mol 이어서 볼츠만 가중치에 거의 영향을 주지 않는다.
//   구현 우선순위: P1.4c (마지막으로 미룸).
//
//   Once P1.5 torsion moves are in, bonds and angles never leave their equilibrium
//   values, so these terms contribute < 0.1 kcal/mol per step and can be safely
//   deferred.  Priority: P1.4c — last.
//
//   파라미터 출처 (Parameter source): AMBER ff14SB bond/angle tables.
//   전형적인 값 (Typical values):
//     C–N bond:  k_b ≈ 490 kcal/mol/Å²,   r₀ ≈ 1.335 Å
//     C–CA bond: k_b ≈ 317 kcal/mol/Å²,   r₀ ≈ 1.522 Å
//     N–CA–C:    k_θ ≈ 63 kcal/mol/rad²,  θ₀ ≈ 111.5°
//
// (b) 이중면체각 에너지 (Dihedral / Torsion energy) ← 중요 — P1.4a로 먼저 구현
//   공식:   E_dih = Σ Vₙ/2 · [1 + cos(n·φ − γ)]
//           n = 1,2,3,4 (n=3 이 주기, n=2 가 펩타이드 평면성에 중요)
//
//   물리적 의미: 백본 φ/ψ 는 Ramachandran 도표의 허용 영역을 정의한다.
//   α-나선 (φ≈−57°, ψ≈−47°) 과 β-가닥 (φ≈−120°, ψ≈+125°) 사이의 에너지 장벽은
//   2–5 kcal/mol 이다.  이 항 없이는 금지 영역(cis-펩타이드, eclipsed 백본)이
//   볼츠만 분포에서 올바르게 억제되지 않아 앙상블 전체가 물리적으로 잘못된다.
//
//   Physical meaning: φ/ψ barriers define the Ramachandran plot.
//   α-helix (φ≈−57°, ψ≈−47°) vs β-strand (φ≈−120°, ψ≈+125°) barriers are
//   2–5 kcal/mol.  Without this term, forbidden backbone geometries are not
//   suppressed correctly, corrupting the entire ensemble.
//   Priority: P1.4a — first after P1.5.
//
//   구현 계획 (Implementation plan):
//     1. adj를 통해 4-atom 시퀀스 (i,j,k,l)를 열거하고 φ = dihedral(i,j,k,l) 를 계산.
//     2. Fourier 급수 Σ Vₙ/2·[1+cos(n·φ−γ)]를 합산 (n=1..4).
//     3. 1-4 쌍의 비결합 에너지를 0.5×로 스케일 다운 (AMBER 관례).
//     Enumerate 4-atom sequences via adj, compute φ = dihedral(i,j,k,l),
//     accumulate Fourier series, scale down 1-4 non-bonded interactions.
//
// (c) 부적절 이중면체각 (Improper dihedral) — 고리·펩타이드 평면성 강제
//   공식:   E_imp = Σ k_ξ · (ξ − ξ₀)²   [harmonic]
//
//   목적: 펩타이드 결합 O=C–N–Cα 의 공명 평면성(ξ₀=0)을 강제.
//         방향족 고리 원자들의 평면성도 강제.
//   구현 우선순위: P1.4c와 같이 (본딩 에너지 항 전체가 완성될 때).
//   Purpose: enforce planarity of peptide O=C–N–Cα (ξ₀=0) and aromatic rings.
//   Priority: alongside P1.4c.
//
// (d) 1-2/1-3 비결합 배제 (1-2/1-3 non-bonded exclusions) ← 현재 잘못됨
//
//   현재 pair_e()는 모든 이웃 원자 쌍을 처리한다.
//   AMBER 관례: 직접 결합 쌍(1-2)과 결합각 분리 쌍(1-3)은 비결합 합산에서 제외.
//   1-4 쌍(세 결합으로 분리)은 포함하되 LJ는 1/2, 전하항은 5/6 로 스케일 다운.
//
//   Currently pair_e() processes all neighbor pairs including directly bonded (1-2)
//   and angle-separated (1-3) atoms — these must be excluded.
//   1-4 pairs are included with LJ scaled by ½ and charge terms by 5/6 (AMBER convention).
//
//   구현 계획 (Implementation plan):
//     build() 후 BondTopology.adj를 BFS로 탐색해 각 원자에 대해
//     1-2, 1-3, 1-4 집합을 미리 계산해 캐싱.
//     pair_e() 호출 전에 집합 멤버십 체크.
//     After build(), BFS from each atom to pre-compute 1-2/1-3/1-4 sets.
//     Check membership before pair_e() to skip excluded pairs.
//   Priority: P1.4b — alongside dihedral terms.

// PhysicsEngine: the core simulation object.
// One instance owns an RNG (gen) used by generate_ensemble.
// All heavy computation is in private static helpers so they can run without
// a 'this' pointer inside OpenMP parallel sections.
class PhysicsEngine {
private:
    std::mt19937 gen;

    // Squared Euclidean distance between two particles.
    static inline double d2(const Particle& a,const Particle& b) noexcept {
        double dx=a.x-b.x,dy=a.y-b.y,dz=a.z-b.z;return dx*dx+dy*dy+dz*dz;
    }

    // Rodrigues rotation of point (px,py,pz) around unit axis (ux,uy,uz)
    // anchored at (ox,oy,oz) by angle with precomputed (cosD, sinD).
    //
    // 로드리게스 회전: 앵커 (ox,oy,oz)를 기준으로 단위 축 u 주위로 cosD·sinD 각도만큼
    // 점 p를 강체(rigid-body) 회전시킨다.
    //
    // 공식 (formula):  v = p − anchor
    //   p_new = anchor + v·cosD + (u×v)·sinD + u·(u·v)·(1−cosD)
    //
    // P1.5 비틀림 이동의 핵심 연산:
    //   j-side 원자 집합의 모든 원자를 rb.i→rb.j 축 주위로 δφ 회전한다.
    //   회전 후 결합 길이·결합각은 정확히 보존된다 (강체 회전 특성).
    static inline void rodrigues(double& px, double& py, double& pz,
                                  double ox, double oy, double oz,
                                  double ux, double uy, double uz,
                                  double cosD, double sinD) noexcept
    {
        double vx = px-ox, vy = py-oy, vz = pz-oz;
        double dot = ux*vx + uy*vy + uz*vz;
        double cx  = uy*vz - uz*vy;
        double cy  = uz*vx - ux*vz;
        double cz  = ux*vy - uy*vx;
        double k   = 1.0 - cosD;
        px = ox + vx*cosD + cx*sinD + ux*dot*k;
        py = oy + vy*cosD + cy*sinD + uy*dot*k;
        pz = oz + vz*cosD + cz*sinD + uz*dot*k;
    }

    // Signed torsion angle (radians) for the 4-atom sequence a-b-c-d.
    // Uses the Praxedesova-Husak convention: angle between planes (abc) and (bcd).
    //
    // 4원자 a-b-c-d 의 부호 있는 비틀림각 (라디안).
    // 벡터 b1 = a-b, b2 = c-b, b3 = d-c 로 정의.
    // n1 = b1×b2, n2 = b2×b3 이 두 평면의 법선 벡터.
    // atan2(m·n2, n1·n2) 로 부호를 결정한다 (m = n1 × b2).
    static inline double dihedral_angle(const std::vector<Particle>& p,
                                         int a, int b, int c, int d) noexcept {
        double b1x = p[a].x-p[b].x, b1y = p[a].y-p[b].y, b1z = p[a].z-p[b].z;
        double b2x = p[c].x-p[b].x, b2y = p[c].y-p[b].y, b2z = p[c].z-p[b].z;
        double b3x = p[d].x-p[c].x, b3y = p[d].y-p[c].y, b3z = p[d].z-p[c].z;
        // n1 = b1 × b2  (normal to plane a-b-c)
        double n1x = b1y*b2z-b1z*b2y, n1y = b1z*b2x-b1x*b2z, n1z = b1x*b2y-b1y*b2x;
        // n2 = b2 × b3  (normal to plane b-c-d)
        double n2x = b2y*b3z-b2z*b3y, n2y = b2z*b3x-b2x*b3z, n2z = b2x*b3y-b2y*b3x;
        // m1 = n1 × b2  (in-plane reference for sign)
        double m1x = n1y*b2z-n1z*b2y, m1y = n1z*b2x-n1x*b2z, m1z = n1x*b2y-n1y*b2x;
        double x = n1x*n2x+n1y*n2y+n1z*n2z;
        double y = m1x*n2x+m1y*n2y+m1z*n2z;
        return std::atan2(y, x);
    }

    // Sum of torsion energy over all DihRecord entries that cross the
    // i-side / j-side boundary (exactly one side of each record in j_side).
    // Only records with at least one atom in each side contribute to ΔE.
    //
    // 이중면체각 에너지 합산.  경계를 가로지르는 레코드(j-side와 i-side가 혼재)만 포함.
    // 공식: E = Σ V2 · [1 + cos(n·φ − γ)]
    static double dihedral_e_boundary(const std::vector<Particle>& p,
                                       const std::vector<DihRecord>& dihs,
                                       const std::vector<bool>& in_side) noexcept {
        double E = 0.0;
        for (const auto& dr : dihs) {
            bool s_a = in_side[dr.a], s_b = in_side[dr.b],
                 s_c = in_side[dr.c], s_d = in_side[dr.d];
            bool any_side  = s_a||s_b||s_c||s_d;
            bool any_fixed = !s_a||!s_b||!s_c||!s_d;
            if (!any_side || !any_fixed) continue;
            double phi = dihedral_angle(p, dr.a, dr.b, dr.c, dr.d);
            for (const auto& t : dr.terms)
                E += t.V2 * (1.0 + std::cos((double)t.n * phi - t.gamma));
        }
        return E;
    }

    // Full dihedral energy sum over all records (for total_e).
    // 전체 이중면체각 에너지 합산 (total_e에서 사용).
    static double dihedral_e(const std::vector<Particle>& p,
                              const std::vector<DihRecord>& dihs) noexcept {
        double E = 0.0;
        for (const auto& dr : dihs) {
            double phi = dihedral_angle(p, dr.a, dr.b, dr.c, dr.d);
            for (const auto& t : dr.terms)
                E += t.V2 * (1.0 + std::cos((double)t.n * phi - t.gamma));
        }
        return E;
    }

    // ── 이황화 결합 하모닉 구속 에너지 (Disulfide harmonic restraint energy, P2.3) ─
    //
    // 이황화 결합을 하모닉 스프링으로 모델링한다.
    //   E_SS = K_SS × (r_SG-SG − r₀_SS)²
    //
    // 각 항의 의미 (Term definitions):
    //   r_SG-SG — 두 시스테인 SG 원자 사이의 현재 거리 (Å)
    //             Current distance between the two Cys SG atoms (Å)
    //   r₀_SS   — 평형 SG–SG 결합 거리 (2.044 Å).
    //             ff14SB CYX 잔기 템플릿에서 가져온 값.
    //             Equilibrium SG–SG bond distance from AMBER ff14SB CYX template.
    //   K_SS    — 힘 상수 (600 kcal/mol/Å²).
    //             AMBER S–S 신축 상수(≈166 kcal/mol/Å²)보다 의도적으로 크게 설정.
    //             이유: 단일 torsion MC 이동이 SG 원자를 크게 이동시킬 수 있으므로
    //             강한 구속으로 이황화 결합 거리를 효과적으로 유지해야 한다.
    //             Force constant (600 kcal/mol/Å²).  Intentionally larger than
    //             the AMBER S-S stretching constant (≈166 kcal/mol/Å²) because
    //             a single large torsion MC step can move SG far; the stiffer spring
    //             keeps the bond distance near r₀ even with large proposals.
    //
    // ss_e(): 모든 이황화 쌍에 대해 구속 에너지를 합산한다. total_e()에서 사용.
    //         Sum restraint energy over all SS pairs.  Called from total_e().
    //
    // ss_e_side(): MC 이동 중 ΔE 계산 최적화:
    //   in_side 원자가 하나도 없는 SS 쌍은 이동에 의해 영향받지 않으므로 건너뜀.
    //   이유: 강체 torsion 회전은 j-side 내부 거리를 보존하고,
    //          i-side 내부 거리도 변하지 않는다.  변하는 것은 오직
    //          i-side 원자 ↔ j-side 원자 사이의 거리뿐이다.
    //   따라서 SS 쌍 (a, b)에서 a와 b 모두 같은 쪽(in_side 모두 true 또는 모두 false)이면
    //   그 쌍의 에너지는 이동 전후로 변하지 않으므로 ΔE 계산에서 제외해도 된다.
    //
    // ss_e_side(): Optimised ΔE computation for MC moves:
    //   Skip SS pairs where NEITHER or BOTH atoms are in in_side — their
    //   pairwise distance is unchanged by a rigid torsion rotation (the
    //   rotation preserves intra-side distances; only cross-side pairs change).
    //   Only pairs where exactly one atom is in in_side (i.e. cross-side) change.
    static constexpr double K_SS  = 600.0;    // 이황화 구속 힘 상수 (kcal/mol/Å²)
                                               // SS restraint force constant
    static constexpr double R0_SS = 2.044;    // SG–SG 평형 거리 (Å) / equilibrium SG-SG distance

    // ss_e: 전체 이황화 구속 에너지 합산.  total_e()에서 호출.
    // ss_e: Total disulfide restraint energy.  Called from total_e().
    static double ss_e(const std::vector<Particle>& p,
                        const std::vector<std::pair<int,int>>& ss) noexcept {
        double E = 0.0;
        for (const auto& [i, j] : ss) {
            double dx=p[i].x-p[j].x, dy=p[i].y-p[j].y, dz=p[i].z-p[j].z;
            double dr = std::sqrt(dx*dx+dy*dy+dz*dz) - R0_SS;
            E += K_SS * dr * dr;
        }
        return E;
    }

    // ss_e_side: MC 이동 ΔE에서 변화하는 이황화 구속 에너지만 합산.
    // in_side 원자가 하나만 포함된 SS 쌍(cross-side)만 계산한다.
    //
    // ss_e_side: Partial SS restraint energy for MC ΔE — only cross-side pairs
    // (exactly one atom in in_side) contribute to ΔE under a torsion rotation.
    static double ss_e_side(const std::vector<Particle>& p,
                              const std::vector<std::pair<int,int>>& ss,
                              const std::vector<bool>& in_side) noexcept {
        double E = 0.0;
        for (const auto& [i, j] : ss) {
            // 두 원자 모두 in_side이거나 둘 다 아니면 → 이동에 의해 거리 불변 → 건너뜀.
            // Both in same side → rigid rotation preserves their distance → skip.
            if (!in_side[i] && !in_side[j]) continue;
            double dx=p[i].x-p[j].x, dy=p[i].y-p[j].y, dz=p[i].z-p[j].z;
            double dr = std::sqrt(dx*dx+dy*dy+dz*dz) - R0_SS;
            E += K_SS * dr * dr;
        }
        return E;
    }

    // HCT pairwise Born integral.
    // HCT(Hawkins-Cramer-Truhlar) 쌍별 Born 적분.
    //
    // Computes the contribution of atom j (radius rj) to the Born desolvation
    // sum of atom i (radius ri) at separation r.  Formula from Hawkins,
    // Cramer & Truhlar (1995).  Returns 0 when j is fully buried within i.
    //
    // 원자 j(반경 rj)가 원자 i(반경 ri)의 Born 탈용매화 합산에 기여하는 양을 계산.
    // 직관: j가 i 주변의 용매를 "차단"할수록 i의 유효 Born 반경이 커져
    //       GB 에너지가 줄어든다 (더 잘 차폐됨 = 덜 불리).
    // j가 i에 완전히 묻혀 있으면 0 반환.
    static inline double hct(double r,double r2,double ri,double rj) noexcept {
        double L=std::max(std::abs(r-rj),ri),U=r+rj;
        if(ri>=U) return 0.0;
        return 1.0/L-1.0/U+(r2-rj*rj+ri*ri)/(2.0*r*ri*ri)*std::log(L/U)*0.5/r;
    }

    // Compute effective Born radii for all atoms.
    // 모든 원자의 유효 Born 반경 계산.
    //
    // Each radius a[i] = 1/(1/r_i - 0.5·Σ_j hct(i,j)), clamped to ≥ 0.5 Å.
    // a[i] = 1 / (1/r_i - 0.5 × Σ_j hct(i,j)), 최소값 0.5 Å로 클램프.
    //
    // Larger a[i] means atom i is more buried (better shielded from solvent).
    // a[i]가 클수록 원자 i가 더 깊이 묻혀 있어 용매로부터 더 잘 차폐됨.
    //
    // 물리적 의미: 노출된 원자(a ≈ r_i)는 GB 에너지가 크고(불리),
    //             묻힌 원자(a >> r_i)는 GB 에너지가 작다(유리).
    static std::vector<double> born_radii(const std::vector<Particle>& p,const NeighborList& nl){
        size_t N=p.size();
        std::vector<double> sum(N,0.0);
        for(size_t i=0;i<N;++i)
            for(size_t j:nl.nb[i]){
                double r2=d2(p[i],p[j]),r=std::sqrt(r2);
                sum[i]+=hct(r,r2,p[i].radius,p[j].radius);
                sum[j]+=hct(r,r2,p[j].radius,p[i].radius);
            }
        std::vector<double> a(N);
        for(size_t i=0;i<N;++i){
            double inv=1.0/p[i].radius-0.5*sum[i];
            a[i]=1.0/std::max(inv,2.0);
        }
        return a;
    }

    // Incremental Born radius update for one atom (idx) after it moves.
    // Cheaper than recomputing all radii; used every MC step in generate_ensemble.
    static void update_born(size_t idx,const std::vector<Particle>& p,
                             const NeighborList& nl,std::vector<double>& a){
        double ri=p[idx].radius,sum=0.0;
        for(size_t j:nl.nb[idx]){
            double r2=d2(p[idx],p[j]),r=std::sqrt(r2);
            sum+=hct(r,r2,ri,p[j].radius);
        }
        a[idx]=1.0/std::max(1.0/ri-0.5*sum,2.0);
    }

    // SASA nonpolar energy.
    // SASA 비극성 에너지.
    //
    // Approximates the solvent-accessible surface area of each atom using a
    // spherical-cap subtraction model, then scales by GAMMA_SA.
    // This captures the hydrophobic burial penalty without an expensive numerical
    // surface calculation.
    //
    // 구면 캡(spherical-cap) 차감 모델로 각 원자의 용매 접근 가능 표면적(SASA)을
    // 근사하고 GAMMA_SA(표면 장력)를 곱한다.
    //
    // 물리: 비극성(소수성) 표면이 물에 노출되면 인접 물 분자들이
    //       수소결합 네트워크를 재배열해야 해 엔트로피 비용이 발생한다.
    //       (소수성 효과, hydrophobic effect)
    //       단백질이 접히면 이 비용이 줄어 전체 에너지가 낮아진다.
    static double sasa_nonpolar(const std::vector<Particle>& p,const NeighborList& nl){
        size_t N=p.size();double E=BETA_SA;
        for(size_t i=0;i<N;++i){
            double ri=p[i].radius+PROBE_R,sa=4.0*M_PI*ri*ri;
            for(size_t j:nl.nb[i]){
                double rj=p[j].radius+PROBE_R;
                double dx=p[i].x-p[j].x,dy=p[i].y-p[j].y,dz=p[i].z-p[j].z;
                double r=std::sqrt(dx*dx+dy*dy+dz*dz),dc=ri+rj;
                if(r>=dc) continue;
                double h=(dc-r)/(2.0*ri);
                sa-=std::min(sa*0.85,2.0*M_PI*ri*ri*h);
            }
            E+=GAMMA_SA*std::max(0.0,sa);
        }
        return E;
    }

    // Total pairwise energy for one atom pair (i, j) with precomputed Born radii.
    // 원자 쌍 (i, j)의 전체 쌍별 에너지 (유효 Born 반경 사전 계산됨).
    //
    // Combines:
    //   edh  — Debye–Hückel screened electrostatics
    //          데바이-휘켈 차폐 정전기: q_i·q_j·exp(-κr) / (ε_w·r)
    //          이온 용액에서 하전 잔기 간 장거리 쿨롱력이 지수적으로 감쇠
    //
    //   egb  — Generalized Born electrostatic solvation: GB_COEF · q_i·q_j / f_GB
    //          일반화 Born 용매화: 전하 쌍이 물 속으로 들어갈 때의 에너지 이득
    //          f_GB = sqrt(r²+ a_i·a_j·exp(-r²/4a_i a_j)) — GB 거리 함수
    //          (r→∞ 시 f_GB→r, r→0 시 f_GB→sqrt(a_i·a_j))
    //
    //   elj  — Lennard-Jones 12-6 van der Waals
    //          4ε[(σ/r)¹² - (σ/r)⁶] : 12항=척력(steric), 6항=분산(인력)
    //
    // Hard-core term: a SMOOTH penalty added on top of edh+egb+elj when atoms
    // overlap (r < HARD_CUTOFF_FRAC·σ), not a branch that replaces them.
    //
    //   E_hard(r) = HARD_SCALE · [ (r_cut/r)¹² − 1 ]²      where r_cut = HARD_CUTOFF_FRAC·σ
    //
    // At r = r_cut this is exactly 0 in BOTH value and slope (the bracket and
    // its derivative both vanish there), so the total energy is continuous
    // and smooth across the boundary — unlike the old design, which jumped
    // straight from the ordinary LJ/GB/DH formula to HARD_SCALE·(σ/r)¹²,
    // discarding it entirely. That discontinuity was harmless for every
    // bundled structure checked at the time (P1.6), but real folded proteins
    // can have a genuine, unremarkable tertiary contact land a fraction of a
    // percent on either side of the threshold — found via a large predicted
    // structure (Q92793, ~18.5k atoms) whose closest contact sat at
    // r/σ = 0.5997, 0.03% inside the old cutoff: that ONE pair alone
    // contributed +4.6M kcal/mol under the old formula (a value on the other
    // side of the boundary would have scored a normal, bounded LJ repulsion
    // of a few kcal/mol for what is physically the same contact). The new
    // formula gives that same pair a ~2.5 kcal/mol penalty instead — smooth,
    // bounded, and negligible next to the rest of the protein's energy — while
    // still diverging steeply for genuine MC-proposal overlaps (r ≪ r_cut).
    //
    // 하드코어 항: r < HARD_CUTOFF_FRAC·σ 일 때 edh+egb+elj를 대체하는 대신
    // 그 위에 "매끄러운" 벌점을 더한다. r=r_cut에서 값과 기울기가 모두 0이 되도록
    // 구성되어 있어 경계에서 에너지가 불연속적으로 튀지 않는다.
    //
    // HARD_CAP: the smooth formula above fixes continuity AT the r_cut boundary,
    // but [(r_cut/r)^12 - 1]^2 is still unbounded as r->0 -- it only helps
    // contacts near the edge of the cutoff (like Q92793's r/sigma=0.5997 case
    // above), not contacts deep inside it. Real (imperfect) structures can have
    // both: PDB 1LYZ (a 1975 X-ray structure) has a genuine 1.36 Angstrom
    // CB(Ala122)...NH1(Arg125) contact at r/sigma=0.364, a poorly-resolved Arg
    // sidechain artifact, not an MC overlap. Uncapped, that single pair alone
    // evaluates to ~1.6e9 kcal/mol even under the smooth formula. HARD_CAP
    // bounds the penalty term itself so one such pair can no longer swamp
    // calculate_potential() for the whole structure, while still being a much
    // larger penalty than any real non-excluded contact should ever incur.
    //
    // HARD_CAP: 위 매끄러운 공식은 r_cut 경계에서의 연속성만 고치고, r->0일 때
    // [(r_cut/r)^12 - 1]^2 자체는 여전히 무한히 발산한다 — 경계 부근 접촉에는
    // 도움이 되지만 경계 훨씬 안쪽의 접촉에는 소용없다. 실제(불완전한) 구조는
    // 둘 다 가질 수 있다: PDB 1LYZ(1975년 X선 구조)는 r/σ=0.364인 실제 1.36 Å
    // CB(Ala122)···NH1(Arg125) 접촉을 갖고 있는데, 이는 잘 정제되지 않은 Arg
    // 곁사슬의 결과물이지 MC 겹침이 아니다. 상한 없이는 이 한 쌍만으로도 매끄러운
    // 공식 하에서 ~1.6×10⁹ kcal/mol이 나온다. HARD_CAP은 벌점 항 자체를 제한해
    // 이런 접촉 하나가 전체 구조의 calculate_potential()을 뒤덮지 못하게 하면서도,
    // 실제 배제되지 않는 정상 접촉보다는 훨씬 큰 페널티를 유지한다.
    static inline double pair_e(const Particle& pi,const Particle& pj,double ai,double aj) noexcept {
        double dx=pi.x-pj.x,dy=pi.y-pj.y,dz=pi.z-pj.z;
        double r2=dx*dx+dy*dy+dz*dz,r=std::sqrt(r2),sig=pi.radius+pj.radius;
        double qp=pi.charge*pj.charge;
        double edh=(COULOMB*qp)/(EPS_WATER*r)*std::exp(-KAPPA*r);
        double fgb=std::sqrt(r2+ai*aj*std::exp(-r2/(4.0*ai*aj)));
        double egb=GB_COEF*qp/fgb;
        double eps=std::sqrt(pi.epsilon*pj.epsilon),s6=std::pow(sig/r,6);
        double elj=4.0*eps*(s6*s6-s6);
        double E=edh+egb+elj;
        double r_cut=sig*HARD_CUTOFF_FRAC;
        if(r<r_cut){
            double x=std::pow(r_cut/r,12.0)-1.0;
            E+=std::min(HARD_SCALE*x*x, HARD_CAP);
        }
        return E;
    }

    // pair_e_diag: diagnostic-only variant of pair_e() above, for the MC
    // rejection-cause investigation (IMPROVEMENTS.md item #2 -- is the real
    // mixing bottleneck a torsional barrier or a correlated steric clash?).
    // pair_e() fuses electrostatics+GB+LJ+hard-core into one returned number
    // with no way to see the hard-core (steric) contribution separately --
    // this variant returns the identical total (same edh+egb+elj+hard-core
    // sum, byte-for-byte the same formula) but also writes just the
    // hard-core sub-term into hardcore_out, so the caller can tell "was
    // this move blocked by real atomic overlap" apart from "was it blocked
    // by ordinary electrostatics/LJ/GB energetics." Duplicated rather than
    // adding an out-parameter to pair_e() itself, to avoid touching an
    // already-verified hot path called from every other energy computation
    // in this file -- same precedent as run_landscape_segment vs.
    // run_landscape_trajectory elsewhere in this file.
    static inline double pair_e_diag(const Particle& pi,const Particle& pj,double ai,double aj,
                                      double& hardcore_out) noexcept {
        double dx=pi.x-pj.x,dy=pi.y-pj.y,dz=pi.z-pj.z;
        double r2=dx*dx+dy*dy+dz*dz,r=std::sqrt(r2),sig=pi.radius+pj.radius;
        double qp=pi.charge*pj.charge;
        double edh=(COULOMB*qp)/(EPS_WATER*r)*std::exp(-KAPPA*r);
        double fgb=std::sqrt(r2+ai*aj*std::exp(-r2/(4.0*ai*aj)));
        double egb=GB_COEF*qp/fgb;
        double eps=std::sqrt(pi.epsilon*pj.epsilon),s6=std::pow(sig/r,6);
        double elj=4.0*eps*(s6*s6-s6);
        double E=edh+egb+elj;
        double r_cut=sig*HARD_CUTOFF_FRAC;
        hardcore_out = 0.0;
        if(r<r_cut){
            double x=std::pow(r_cut/r,12.0)-1.0;
            hardcore_out = std::min(HARD_SCALE*x*x, HARD_CAP);
            E+=hardcore_out;
        }
        return E;
    }

    // Sum of SASA + dihedral + all pair energies within NL_CUTOFF.
    // Skips 1-2/1-3 excluded pairs when topo != nullptr (P1.4b).
    // Includes dihedral energy when topo != nullptr and topo has dihedrals (P1.4a).
    // OpenMP parallel-for over atom i with dynamic scheduling and energy reduction.
    static double total_e(const std::vector<Particle>& p, const NeighborList& nl,
                          const std::vector<double>& a,
                          const BondTopology* topo = nullptr) {
        double E = sasa_nonpolar(p, nl);
        if (topo && !topo->dihedrals.empty())
            E += dihedral_e(p, topo->dihedrals);
        // 이황화 결합 구속 에너지: build()에 등록된 SG-SG 쌍에 대한 하모닉 에너지 합산.
        // 이황화 결합이 없는 단백질에서는 disulfide_pairs가 비어 있어 추가 비용 없음.
        // Disulfide restraint energy: harmonic sum over registered SG-SG pairs.
        // Zero cost for proteins without disulfide bonds (empty vector short-circuits).
        if (topo && !topo->disulfide_pairs.empty())
            E += ss_e(p, topo->disulfide_pairs);
        // MSVC's OpenMP implementation only supports the OpenMP 2.0 canonical
        // for-loop form, which requires a SIGNED loop variable -- size_t (i's
        // natural type here, matching p.size()) fails to compile under
        // MSVC+/openmp with C3016. Use a signed ptrdiff_t for the loop counter
        // and cast back to size_t for indexing.
        const std::ptrdiff_t N = static_cast<std::ptrdiff_t>(p.size());
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic,8) reduction(+:E)
#endif
        for (std::ptrdiff_t i = 0; i < N; ++i)
            for (size_t j : nl.nb[(size_t)i]) {
                if (topo && topo->is_excluded((int)i, (int)j)) continue;
                double dx = p[i].x-p[j].x, dy = p[i].y-p[j].y, dz = p[i].z-p[j].z;
                if (dx*dx+dy*dy+dz*dz > PAIR_CUT2) continue;
                E += pair_e(p[i], p[j], a[i], a[j]);
            }
        return E;
    }
public:
    PhysicsEngine():gen(std::random_device{}()){}

    // calculate_potential: full GB/DH/LJ/SASA energy for an arbitrary particle list.
    // Rebuilds the neighbor list from scratch on every call — use only for final
    // evaluation, not inside the inner MC loop.
    // Pass topo != nullptr to enable 1-2/1-3 exclusions (P1.4b).
    double calculate_potential(const std::vector<Particle>& particles,
                               const BondTopology* topo = nullptr) {
        if (particles.empty()) return 0.0;
        NeighborList nl; nl.build(particles);
        auto a = born_radii(particles, nl);
        return total_e(particles, nl, a, topo);
    }

    // generate_ensemble: run 'ncand' independent torsion-angle MC trajectories.
    // Returns a vector of ncand final conformations (the MC "ensemble").
    //
    // P1.5 비틀림 각도 이동 MC — ncand개의 독립 궤적을 각각 steps 스텝 실행.
    // 추가 기능 (additions vs. original P1.5):
    //   • 레버암 적응형 스케일: rot_bond_scale[k] = sqrt(N_ref/N_down_k)
    //     (큰 단백질 N-말단 결합의 치솟는 거부율 완화)
    //   • 크랭크샤프트 협동 이동 (25%): 같은 Cα의 φ+ψ 쌍에 +δ/−δ 적용
    //     → 곁사슬 이동, 이후 백본 근사 복원 → 수용율 향상
    //   • 온라인 수용율 추적 + 자동 조정: TUNE_FREQ 스텝마다 cur_max 조절
    //     목표 수용율 [TARGET_LO, TARGET_HI] 유지
    //
    // Additions over original P1.5:
    //   • Lever-arm scale: rot_bond_scale[k] = sqrt(N_ref/N_downstream_k) prevents
    //     lever-arm rejection catastrophe for bonds near the N-terminus.
    //   • Crankshaft moves (25% of steps): concerted +δ/−δ on φ/ψ sharing one Cα.
    //     Sidechain displaces; downstream backbone approximately restores → higher acceptance.
    //   • Online acceptance tracking: adjust cur_max every TUNE_FREQ steps toward
    //     target acceptance window [TARGET_LO, TARGET_HI].
    //
    // Metropolis detailed balance is preserved:
    //   torsion move  — symmetric U[-δ,+δ] proposal.
    //   crankshaft    — symmetric U[-δ,+δ] for +δ/−δ pair; reverse proposal identical probability.
    std::vector<std::vector<Particle>> generate_ensemble(
        const std::vector<Particle>& init,
        const BondTopology& topo,
        int ncand, int steps,
        double T = 0.6,
        double max_angle = 0.12)
    {
        if (init.empty()) throw std::invalid_argument("initial_state empty");
        if (ncand <= 0 || steps <= 0) throw std::invalid_argument("ncand/steps must be positive");
        if (topo.rot_bonds.empty()) throw std::invalid_argument("topology has no rotatable bonds");
        size_t N   = init.size();
        int    nrb = (int)topo.rot_bonds.size();
        int    ncp = (int)topo.concerted_pairs.size();
        std::vector<std::vector<Particle>> ens(ncand);

        auto chain = [&](int c, std::mt19937& rng) {
            std::vector<Particle> st = init;
            NeighborList nl; nl.build(st);
            auto a = born_radii(st, nl);
            double curE = total_e(st, nl, a, &topo);

            std::uniform_int_distribution<int>    pick_rb(0, nrb - 1);
            std::uniform_int_distribution<int>    pick_cp(0, std::max(0, ncp - 1));
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            std::vector<bool> in_side(N, false);  // reused every step; cleared after each move

            // ── 적응형 제안 폭 상태 (Adaptive proposal width state) ──────────
            // TUNE_FREQ 스텝마다 수용율을 측정해 cur_max를 조절한다.
            // 목표 수용율: [TARGET_LO, TARGET_HI]. 이 창을 벗어나면 SCALE_UP/DOWN 적용.
            double cur_max = max_angle;
            int    acc_win = 0, tot_win = 0;
            constexpr int    TUNE_FREQ  = 200;
            constexpr double TARGET_LO  = 0.28, TARGET_HI = 0.52;
            constexpr double SCALE_UP   = 1.06,  SCALE_DOWN = 0.94;
            constexpr double ANGLE_MAX  = 0.50,  ANGLE_MIN  = 0.004; // radians

            // ── 크로스-쌍 에너지 계산 헬퍼 ──────────────────────────────────
            // in_side가 설정된 상태에서 호출; i-side vs j-side 교차 쌍만 합산.
            // Helper: sum cross-pair energies for the current in_side mask.
            auto cross_e = [&]() -> double {
                double E = 0.0;
                for (size_t i = 0; i < N; ++i)
                    for (size_t j : nl.nb[i]) {
                        if (in_side[i] == in_side[j]) continue;
                        if (topo.is_excluded((int)i, (int)j)) continue;
                        double dx = st[i].x-st[j].x, dy = st[i].y-st[j].y, dz = st[i].z-st[j].z;
                        if (dx*dx+dy*dy+dz*dz > PAIR_CUT2) continue;
                        E += pair_e(st[i], st[j], a[i], a[j]);
                    }
                return E;
            };

            // ── 표준 비틀림 이동 람다 ─────────────────────────────────────
            // Standard torsion-angle MC move with lever-arm scaling.
            // Returns {accepted, did_move}.
            auto try_torsion = [&]() -> std::pair<bool,bool> {
                int            rb_idx = pick_rb(rng);
                const RotBond& rb     = topo.rot_bonds[rb_idx];
                const std::vector<int>& side = topo.rot_bond_sides[rb_idx];
                if (side.empty()) return {false, false};

                // 회전축 계산 / Compute normalised rotation axis.
                double ax = st[rb.j].x-st[rb.i].x, ay = st[rb.j].y-st[rb.i].y, az = st[rb.j].z-st[rb.i].z;
                double al = std::sqrt(ax*ax+ay*ay+az*az);
                if (al < 1e-10) return {false, false};
                ax/=al; ay/=al; az/=al;

                // 레버암 스케일 + 결합 종류 가중치 적용
                // Apply lever-arm scale and bond-kind weighting.
                double base_d = (rb.kind == BondKind::BACKBONE_PHI ||
                                 rb.kind == BondKind::BACKBONE_PSI)
                                ? cur_max : cur_max * 2.5;
                base_d *= topo.rot_bond_scale[rb_idx];
                double delta = std::uniform_real_distribution<double>(-base_d, base_d)(rng);
                double cosD  = std::cos(delta), sinD = std::sin(delta);

                for (int k : side) in_side[k] = true;

                std::vector<std::array<double,3>> old_pos(side.size());
                std::vector<double>               old_born(side.size());
                for (size_t k = 0; k < side.size(); ++k) {
                    int idx = side[k];
                    old_pos[k]  = {st[idx].x, st[idx].y, st[idx].z};
                    old_born[k] = a[idx];
                }

                double old_cross = cross_e();                     // 비결합 교차 쌍 에너지 (이동 전)
                double old_sasa  = sasa_nonpolar(st, nl);        // 비극성 SASA 에너지 (이동 전)
                double old_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);  // 이중면체각 에너지 (이동 전)
                // 이황화 구속 에너지 (이동 전): cross-side SG-SG 쌍만 계산.
                // 이황화 결합 없으면 0.0 (단락 평가).
                // SS restraint before move: only cross-side pairs.  0.0 if no SS bonds.
                double old_ss    = topo.disulfide_pairs.empty() ? 0.0
                                 : ss_e_side(st, topo.disulfide_pairs, in_side);

                // ── Rodrigues 회전 적용 + Born 반경 갱신 ─────────────────────────────
                double ox = st[rb.i].x, oy = st[rb.i].y, oz = st[rb.i].z;
                for (int k : side)
                    rodrigues(st[k].x, st[k].y, st[k].z, ox, oy, oz, ax, ay, az, cosD, sinD);
                for (int k : side) update_born((size_t)k, st, nl, a);

                double new_cross = cross_e();                     // 비결합 교차 쌍 에너지 (이동 후)
                double new_sasa  = sasa_nonpolar(st, nl);        // 비극성 SASA 에너지 (이동 후)
                double new_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);  // 이중면체각 에너지 (이동 후)
                // 이황화 구속 에너지 (이동 후).
                // SS restraint after move.
                double new_ss    = topo.disulfide_pairs.empty() ? 0.0
                                 : ss_e_side(st, topo.disulfide_pairs, in_side);

                // ΔE = 모든 에너지 항의 변화량 합산 (이동 후 − 이동 전).
                // ΔE = sum of all energy component changes (after - before move).
                bool accepted = false;
                double dE = (new_cross-old_cross) + (new_sasa-old_sasa)
                          + (new_dih-old_dih)   + (new_ss-old_ss);
                if (dE < 0.0 || uni(rng) < std::exp(-dE / T)) {
                    curE += dE;
                    accepted = true;
                } else {
                    for (size_t k = 0; k < side.size(); ++k) {
                        int idx = side[k];
                        st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                        a[idx]    = old_born[k];
                    }
                }
                for (int k : side) in_side[k] = false;
                return {accepted, true};
            };

            // ── 크랭크샤프트 협동 이동 람다 ──────────────────────────────
            // Crankshaft move: +δ on φ then −δ on ψ for the same Cα.
            //   Phase 1 (φ, +δ): CA + sidechain + all downstream rotate.
            //   Phase 2 (ψ, −δ): C + O + all downstream rotate back (≈ cancels).
            //   Net displacement: sidechain of residue i; downstream backbone O(δ²).
            // Returns {accepted, did_move}.
            auto try_crankshaft = [&]() -> std::pair<bool,bool> {
                if (ncp == 0) return {false, false};
                int cp_idx = (ncp > 1) ? pick_cp(rng) : 0;
                int phi_k  = topo.concerted_pairs[cp_idx].first;
                int psi_k  = topo.concerted_pairs[cp_idx].second;
                const std::vector<int>& phi_side = topo.rot_bond_sides[phi_k];
                const std::vector<int>& psi_side = topo.rot_bond_sides[psi_k];
                if (phi_side.empty() || psi_side.empty()) return {false, false};

                // in_side: φ side (includes sidechain + downstream backbone)
                for (int k : phi_side) in_side[k] = true;

                // φ 축 계산 / φ rotation axis.
                const RotBond& phi_rb = topo.rot_bonds[phi_k];
                double p1ox = st[phi_rb.i].x, p1oy = st[phi_rb.i].y, p1oz = st[phi_rb.i].z;
                double p1ax = st[phi_rb.j].x-p1ox, p1ay = st[phi_rb.j].y-p1oy, p1az = st[phi_rb.j].z-p1oz;
                double p1l  = std::sqrt(p1ax*p1ax+p1ay*p1ay+p1az*p1az);
                if (p1l < 1e-10) { for (int k : phi_side) in_side[k] = false; return {false, false}; }
                p1ax/=p1l; p1ay/=p1l; p1az/=p1l;

                // 이전 에너지 + 상태 저장 / Save old energy and positions.
                std::vector<std::array<double,3>> old_pos(phi_side.size());
                std::vector<double>               old_born(phi_side.size());
                for (size_t k = 0; k < phi_side.size(); ++k) {
                    int idx = phi_side[k];
                    old_pos[k]  = {st[idx].x, st[idx].y, st[idx].z};
                    old_born[k] = a[idx];
                }
                double old_cross = cross_e();                    // 크로스-쌍 에너지 (이동 전) / cross-pair energy before
                double old_sasa  = sasa_nonpolar(st, nl);       // 비극성 SASA (이동 전) / nonpolar SASA before
                double old_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side); // 이중면체각 (이동 전)
                // 이황화 구속 (이동 전): cross-side 쌍만 포함. 이황화 없으면 0.0.
                // SS restraint before both rotations: cross-side pairs only. 0.0 if no SS.
                double old_ss    = topo.disulfide_pairs.empty() ? 0.0
                                 : ss_e_side(st, topo.disulfide_pairs, in_side);

                // δ 샘플링: 크랭크샤프트는 실효 이동이 국소적이므로 스케일 없이 cur_max 사용.
                // No lever-arm scale for crankshaft: the effective displacement is local
                // regardless of chain position (downstream atoms approximately cancel).
                double delta = std::uniform_real_distribution<double>(-cur_max, cur_max)(rng);
                double cosD  = std::cos(delta), sinD = std::sin(delta);

                // Phase 1: φ 회전 (+δ), φ-side 전체
                for (int k : phi_side)
                    rodrigues(st[k].x, st[k].y, st[k].z, p1ox, p1oy, p1oz, p1ax, p1ay, p1az, cosD, sinD);

                // Phase 2: ψ 회전 (−δ), ψ-side (post-φ 축 좌표 사용)
                // ψ axis recomputed from current (post-φ) atom positions.
                const RotBond& psi_rb = topo.rot_bonds[psi_k];
                double p2ox = st[psi_rb.i].x, p2oy = st[psi_rb.i].y, p2oz = st[psi_rb.i].z;
                double p2ax = st[psi_rb.j].x-p2ox, p2ay = st[psi_rb.j].y-p2oy, p2az = st[psi_rb.j].z-p2oz;
                double p2l  = std::sqrt(p2ax*p2ax+p2ay*p2ay+p2az*p2az);
                if (p2l < 1e-10) {
                    // φ 회전 복원 후 스킵 / Revert φ and skip.
                    for (size_t k = 0; k < phi_side.size(); ++k) {
                        int idx = phi_side[k];
                        st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                    }
                    for (int k : phi_side) in_side[k] = false;
                    return {false, false};
                }
                p2ax/=p2l; p2ay/=p2l; p2az/=p2l;
                // Apply −δ: cosD same (cos symmetric), −sinD flips rotation direction.
                for (int k : psi_side)
                    rodrigues(st[k].x, st[k].y, st[k].z, p2ox, p2oy, p2oz, p2ax, p2ay, p2az, cosD, -sinD);

                // φ-side 전체 Born 반경 업데이트 (일부 원자는 제자리 복귀, 모두 갱신)
                for (int k : phi_side) update_born((size_t)k, st, nl, a);

                double new_cross = cross_e();                    // 크로스-쌍 에너지 (이동 후) / cross-pair after
                double new_sasa  = sasa_nonpolar(st, nl);       // 비극성 SASA (이동 후) / SASA after
                double new_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side); // 이중면체각 (이동 후)
                // 이황화 구속 (두 회전 후). 하류 잔기가 근사 복귀하므로 ΔE는 작다.
                // SS restraint after both rotations.  Small ΔE because downstream approximately restores.
                double new_ss    = topo.disulfide_pairs.empty() ? 0.0
                                 : ss_e_side(st, topo.disulfide_pairs, in_side);

                // 전체 ΔE = 모든 항의 변화량 합산.
                // Total ΔE: sum of all component changes across both phase rotations.
                bool accepted = false;
                double dE = (new_cross-old_cross) + (new_sasa-old_sasa)
                          + (new_dih-old_dih)   + (new_ss-old_ss);
                if (dE < 0.0 || uni(rng) < std::exp(-dE / T)) {
                    curE += dE;
                    accepted = true;
                } else {
                    for (size_t k = 0; k < phi_side.size(); ++k) {
                        int idx = phi_side[k];
                        st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                        a[idx]    = old_born[k];
                    }
                }
                for (int k : phi_side) in_side[k] = false;
                return {accepted, true};
            };

            // ── 메인 MC 루프 ──────────────────────────────────────────────
            for (int s = 0; s < steps; ++s) {
                // drift > NL_SKIN/2 이면 이웃 목록·Born 반경 전체 재구축
                if (nl.needs_rebuild(st)) {
                    nl.build(st);
                    a    = born_radii(st, nl);
                    curE = total_e(st, nl, a, &topo);
                }

                // 이동 종류 선택: 25% 크랭크샤프트, 75% 표준 비틀림
                // Move type: 25% crankshaft (when pairs exist), 75% standard torsion.
                bool do_crank = ncp > 0 && uni(rng) < 0.25;
                auto [accepted, did_move] = do_crank ? try_crankshaft() : try_torsion();

                // ── 수용율 추적 및 cur_max 자동 조정 ──────────────────────
                // Track acceptance; tune cur_max every TUNE_FREQ moves.
                if (did_move) {
                    acc_win += accepted ? 1 : 0;
                    if (++tot_win == TUNE_FREQ) {
                        double rate = (double)acc_win / TUNE_FREQ;
                        if      (rate > TARGET_HI) cur_max = std::min(ANGLE_MAX, cur_max * SCALE_UP);
                        else if (rate < TARGET_LO) cur_max = std::max(ANGLE_MIN, cur_max * SCALE_DOWN);
                        acc_win = tot_win = 0;
                    }
                }
            }
            ens[c] = std::move(st);
        };

#ifdef _OPENMP
        #pragma omp parallel
        {
            std::mt19937 lg(std::random_device{}() ^ (std::hash<int>{}(omp_get_thread_num()) << 16));
            #pragma omp for schedule(dynamic)
            for (int c = 0; c < ncand; ++c) chain(c, lg);
        }
#else
        for (int c = 0; c < ncand; ++c) chain(c, gen);
#endif
        return ens;
    }

    // run_landscape_trajectory: run ONE torsion-MC chain for
    // n_snapshots * steps_per_snapshot total steps, recording a full snapshot
    // (particle list + energy) every steps_per_snapshot steps.
    //
    // This exists so LandscapeWorker (Python) can advance the whole 120×80-step
    // Markov chain with a single call into C++ instead of looping 120 times,
    // each iteration re-marshalling the full particle array across the
    // Python↔C++ boundary via generate_ensemble(...)[0] + calculate_potential(...)
    // (see IMPROVEMENTS.md item #12). The per-step physics is identical to
    // generate_ensemble's single-chain body (duplicated rather than shared via
    // a helper, to avoid touching that already-verified hot path).
    std::pair<std::vector<std::vector<Particle>>, std::vector<double>>
    run_landscape_trajectory(
        const std::vector<Particle>& init,
        const BondTopology& topo,
        int n_snapshots, int steps_per_snapshot,
        double T = 0.6,
        double max_angle = 0.12)
    {
        if (init.empty()) throw std::invalid_argument("initial_state empty");
        if (n_snapshots <= 0 || steps_per_snapshot <= 0)
            throw std::invalid_argument("n_snapshots/steps_per_snapshot must be positive");
        if (topo.rot_bonds.empty()) throw std::invalid_argument("topology has no rotatable bonds");

        size_t N   = init.size();
        int    nrb = (int)topo.rot_bonds.size();
        int    ncp = (int)topo.concerted_pairs.size();

        std::vector<std::vector<Particle>> snapshots;
        std::vector<double>                energies;
        snapshots.reserve(n_snapshots);
        energies.reserve(n_snapshots);

        std::vector<Particle> st = init;
        NeighborList nl; nl.build(st);
        auto a = born_radii(st, nl);
        double curE = total_e(st, nl, a, &topo);

        std::uniform_int_distribution<int>    pick_rb(0, nrb - 1);
        std::uniform_int_distribution<int>    pick_cp(0, std::max(0, ncp - 1));
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<bool> in_side(N, false);

        double cur_max = max_angle;
        int    acc_win = 0, tot_win = 0;
        constexpr int    TUNE_FREQ  = 200;
        constexpr double TARGET_LO  = 0.28, TARGET_HI = 0.52;
        constexpr double SCALE_UP   = 1.06,  SCALE_DOWN = 0.94;
        constexpr double ANGLE_MAX  = 0.50,  ANGLE_MIN  = 0.004;

        auto cross_e = [&]() -> double {
            double E = 0.0;
            for (size_t i = 0; i < N; ++i)
                for (size_t j : nl.nb[i]) {
                    if (in_side[i] == in_side[j]) continue;
                    if (topo.is_excluded((int)i, (int)j)) continue;
                    double dx = st[i].x-st[j].x, dy = st[i].y-st[j].y, dz = st[i].z-st[j].z;
                    if (dx*dx+dy*dy+dz*dz > PAIR_CUT2) continue;
                    E += pair_e(st[i], st[j], a[i], a[j]);
                }
            return E;
        };

        auto try_torsion = [&]() -> std::pair<bool,bool> {
            int            rb_idx = pick_rb(gen);
            const RotBond& rb     = topo.rot_bonds[rb_idx];
            const std::vector<int>& side = topo.rot_bond_sides[rb_idx];
            if (side.empty()) return {false, false};

            double ax = st[rb.j].x-st[rb.i].x, ay = st[rb.j].y-st[rb.i].y, az = st[rb.j].z-st[rb.i].z;
            double al = std::sqrt(ax*ax+ay*ay+az*az);
            if (al < 1e-10) return {false, false};
            ax/=al; ay/=al; az/=al;

            double base_d = (rb.kind == BondKind::BACKBONE_PHI ||
                             rb.kind == BondKind::BACKBONE_PSI)
                            ? cur_max : cur_max * 2.5;
            base_d *= topo.rot_bond_scale[rb_idx];
            double delta = std::uniform_real_distribution<double>(-base_d, base_d)(gen);
            double cosD  = std::cos(delta), sinD = std::sin(delta);

            for (int k : side) in_side[k] = true;

            std::vector<std::array<double,3>> old_pos(side.size());
            std::vector<double>               old_born(side.size());
            for (size_t k = 0; k < side.size(); ++k) {
                int idx = side[k];
                old_pos[k]  = {st[idx].x, st[idx].y, st[idx].z};
                old_born[k] = a[idx];
            }

            double old_cross = cross_e();
            double old_sasa  = sasa_nonpolar(st, nl);
            double old_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double old_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            double ox = st[rb.i].x, oy = st[rb.i].y, oz = st[rb.i].z;
            for (int k : side)
                rodrigues(st[k].x, st[k].y, st[k].z, ox, oy, oz, ax, ay, az, cosD, sinD);
            for (int k : side) update_born((size_t)k, st, nl, a);

            double new_cross = cross_e();
            double new_sasa  = sasa_nonpolar(st, nl);
            double new_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double new_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            bool accepted = false;
            double dE = (new_cross-old_cross) + (new_sasa-old_sasa)
                      + (new_dih-old_dih)   + (new_ss-old_ss);
            if (dE < 0.0 || uni(gen) < std::exp(-dE / T)) {
                curE += dE;
                accepted = true;
            } else {
                for (size_t k = 0; k < side.size(); ++k) {
                    int idx = side[k];
                    st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                    a[idx]    = old_born[k];
                }
            }
            for (int k : side) in_side[k] = false;
            return {accepted, true};
        };

        auto try_crankshaft = [&]() -> std::pair<bool,bool> {
            if (ncp == 0) return {false, false};
            int cp_idx = (ncp > 1) ? pick_cp(gen) : 0;
            int phi_k  = topo.concerted_pairs[cp_idx].first;
            int psi_k  = topo.concerted_pairs[cp_idx].second;
            const std::vector<int>& phi_side = topo.rot_bond_sides[phi_k];
            const std::vector<int>& psi_side = topo.rot_bond_sides[psi_k];
            if (phi_side.empty() || psi_side.empty()) return {false, false};

            for (int k : phi_side) in_side[k] = true;

            const RotBond& phi_rb = topo.rot_bonds[phi_k];
            double p1ox = st[phi_rb.i].x, p1oy = st[phi_rb.i].y, p1oz = st[phi_rb.i].z;
            double p1ax = st[phi_rb.j].x-p1ox, p1ay = st[phi_rb.j].y-p1oy, p1az = st[phi_rb.j].z-p1oz;
            double p1l  = std::sqrt(p1ax*p1ax+p1ay*p1ay+p1az*p1az);
            if (p1l < 1e-10) { for (int k : phi_side) in_side[k] = false; return {false, false}; }
            p1ax/=p1l; p1ay/=p1l; p1az/=p1l;

            std::vector<std::array<double,3>> old_pos(phi_side.size());
            std::vector<double>               old_born(phi_side.size());
            for (size_t k = 0; k < phi_side.size(); ++k) {
                int idx = phi_side[k];
                old_pos[k]  = {st[idx].x, st[idx].y, st[idx].z};
                old_born[k] = a[idx];
            }
            double old_cross = cross_e();
            double old_sasa  = sasa_nonpolar(st, nl);
            double old_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double old_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            double delta = std::uniform_real_distribution<double>(-cur_max, cur_max)(gen);
            double cosD  = std::cos(delta), sinD = std::sin(delta);

            for (int k : phi_side)
                rodrigues(st[k].x, st[k].y, st[k].z, p1ox, p1oy, p1oz, p1ax, p1ay, p1az, cosD, sinD);

            const RotBond& psi_rb = topo.rot_bonds[psi_k];
            double p2ox = st[psi_rb.i].x, p2oy = st[psi_rb.i].y, p2oz = st[psi_rb.i].z;
            double p2ax = st[psi_rb.j].x-p2ox, p2ay = st[psi_rb.j].y-p2oy, p2az = st[psi_rb.j].z-p2oz;
            double p2l  = std::sqrt(p2ax*p2ax+p2ay*p2ay+p2az*p2az);
            if (p2l < 1e-10) {
                for (size_t k = 0; k < phi_side.size(); ++k) {
                    int idx = phi_side[k];
                    st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                }
                for (int k : phi_side) in_side[k] = false;
                return {false, false};
            }
            p2ax/=p2l; p2ay/=p2l; p2az/=p2l;
            for (int k : psi_side)
                rodrigues(st[k].x, st[k].y, st[k].z, p2ox, p2oy, p2oz, p2ax, p2ay, p2az, cosD, -sinD);

            for (int k : phi_side) update_born((size_t)k, st, nl, a);

            double new_cross = cross_e();
            double new_sasa  = sasa_nonpolar(st, nl);
            double new_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double new_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            bool accepted = false;
            double dE = (new_cross-old_cross) + (new_sasa-old_sasa)
                      + (new_dih-old_dih)   + (new_ss-old_ss);
            if (dE < 0.0 || uni(gen) < std::exp(-dE / T)) {
                curE += dE;
                accepted = true;
            } else {
                for (size_t k = 0; k < phi_side.size(); ++k) {
                    int idx = phi_side[k];
                    st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                    a[idx]    = old_born[k];
                }
            }
            for (int k : phi_side) in_side[k] = false;
            return {accepted, true};
        };

        // ── try_concerted_sidechain (2026-07-09, IMPROVEMENTS.md item #2) ────
        // Direct measurement (run_mc_diagnostic) found the crankshaft move
        // above is 90.8% steric-dominant among rejected large moves on 1YPI
        // (494 res, densely packed) vs. 8.1% on 1UBQ (76 res) -- the existing
        // 2-DOF concerted move doesn't have enough room to route a sidechain
        // around its neighbours on a large, tightly packed structure. This
        // move gives two independent (disjoint rot_bond_sides), spatially
        // close sidechain bonds (topo.concerted_sidechain_pairs, computed once
        // at parse time in identify_concerted_sidechain_pairs()) the chance to
        // move *simultaneously* -- so if one sidechain's swing would clash
        // with its neighbour, that neighbour can move out of the way in the
        // same proposal, rather than requiring two separate, uncorrelated
        // single-bond moves to happen in the right order by chance.
        //
        // Detailed balance: the two deltas are drawn independently from the
        // same symmetric distribution every other move in this file already
        // uses, and pair/move selection is state-independent (a fixed,
        // precomputed candidate list, chosen uniformly) -- so q(x->y) =
        // f(δ_A)f(δ_B) = f(-δ_A)f(-δ_B) = q(y->x) exactly, and plain
        // Metropolis acceptance is correct with no importance-weighting
        // needed, same proof structure as try_torsion/try_crankshaft above.
        //
        // side_group (0=unmoved, 1=group A, 2=group B) is local to this move
        // only -- try_torsion/try_crankshaft above are untouched and keep
        // using the existing boolean in_side. dihedral_e_boundary/ss_e_side
        // only need "any atom moved vs. any atom fixed" (group identity
        // doesn't matter to them), so the existing in_side -- with both
        // groups' atoms marked true -- is reused as-is for those two calls.
        // Only the pairwise non-bonded sum needs to distinguish A from B: the
        // existing cross_e()'s `in_side[i]==in_side[j]` skip would (if fed a
        // union of both groups under one boolean) incorrectly skip the A-B
        // cross term entirely -- exactly the interaction this move exists to
        // capture (do the two proposed sidechain moves clash with each
        // other) -- hence the separate cross_e_concerted() below.
        std::vector<int> side_group(N, 0);
        auto cross_e_concerted = [&]() -> double {
            double E = 0.0;
            for (size_t i = 0; i < N; ++i)
                for (size_t j : nl.nb[i]) {
                    if (side_group[i] == side_group[j]) continue;
                    if (topo.is_excluded((int)i, (int)j)) continue;
                    double dx = st[i].x-st[j].x, dy = st[i].y-st[j].y, dz = st[i].z-st[j].z;
                    if (dx*dx+dy*dy+dz*dz > PAIR_CUT2) continue;
                    E += pair_e(st[i], st[j], a[i], a[j]);
                }
            return E;
        };
        const int ncsp = (int)topo.concerted_sidechain_pairs.size();
        std::uniform_int_distribution<int> pick_csp(0, std::max(0, ncsp - 1));

        auto try_concerted_sidechain = [&]() -> std::pair<bool,bool> {
            if (ncsp == 0) return {false, false};
            int   pair_idx = (ncsp > 1) ? pick_csp(gen) : 0;
            int   bond_a   = topo.concerted_sidechain_pairs[pair_idx].first;
            int   bond_b   = topo.concerted_sidechain_pairs[pair_idx].second;
            const RotBond& rb_a = topo.rot_bonds[bond_a];
            const RotBond& rb_b = topo.rot_bonds[bond_b];
            const std::vector<int>& side_a = topo.rot_bond_sides[bond_a];
            const std::vector<int>& side_b = topo.rot_bond_sides[bond_b];
            if (side_a.empty() || side_b.empty()) return {false, false};

            double axA = st[rb_a.j].x-st[rb_a.i].x, ayA = st[rb_a.j].y-st[rb_a.i].y, azA = st[rb_a.j].z-st[rb_a.i].z;
            double alA = std::sqrt(axA*axA+ayA*ayA+azA*azA);
            if (alA < 1e-10) return {false, false};
            axA/=alA; ayA/=alA; azA/=alA;
            double axB = st[rb_b.j].x-st[rb_b.i].x, ayB = st[rb_b.j].y-st[rb_b.i].y, azB = st[rb_b.j].z-st[rb_b.i].z;
            double alB = std::sqrt(axB*axB+ayB*ayB+azB*azB);
            if (alB < 1e-10) return {false, false};
            axB/=alB; ayB/=alB; azB/=alB;

            // Both bonds are BondKind::SIDECHAIN by construction (see
            // identify_concerted_sidechain_pairs) -- same sidechain-style
            // 2.5x lever-arm scaling as try_torsion's non-backbone branch.
            double deltaA = std::uniform_real_distribution<double>(
                -cur_max*2.5*topo.rot_bond_scale[bond_a], cur_max*2.5*topo.rot_bond_scale[bond_a])(gen);
            double deltaB = std::uniform_real_distribution<double>(
                -cur_max*2.5*topo.rot_bond_scale[bond_b], cur_max*2.5*topo.rot_bond_scale[bond_b])(gen);
            double cosA = std::cos(deltaA), sinA = std::sin(deltaA);
            double cosB = std::cos(deltaB), sinB = std::sin(deltaB);

            for (int k : side_a) { in_side[k] = true; side_group[k] = 1; }
            for (int k : side_b) { in_side[k] = true; side_group[k] = 2; }

            std::vector<int> all_idx;
            all_idx.reserve(side_a.size() + side_b.size());
            all_idx.insert(all_idx.end(), side_a.begin(), side_a.end());
            all_idx.insert(all_idx.end(), side_b.begin(), side_b.end());
            std::vector<std::array<double,3>> old_pos(all_idx.size());
            std::vector<double>               old_born(all_idx.size());
            for (size_t k = 0; k < all_idx.size(); ++k) {
                int idx = all_idx[k];
                old_pos[k]  = {st[idx].x, st[idx].y, st[idx].z};
                old_born[k] = a[idx];
            }

            double old_cross = cross_e_concerted();
            double old_sasa  = sasa_nonpolar(st, nl);
            double old_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double old_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            double oxA = st[rb_a.i].x, oyA = st[rb_a.i].y, ozA = st[rb_a.i].z;
            for (int k : side_a)
                rodrigues(st[k].x, st[k].y, st[k].z, oxA, oyA, ozA, axA, ayA, azA, cosA, sinA);
            double oxB = st[rb_b.i].x, oyB = st[rb_b.i].y, ozB = st[rb_b.i].z;
            for (int k : side_b)
                rodrigues(st[k].x, st[k].y, st[k].z, oxB, oyB, ozB, axB, ayB, azB, cosB, sinB);
            for (int k : all_idx) update_born((size_t)k, st, nl, a);

            double new_cross = cross_e_concerted();
            double new_sasa  = sasa_nonpolar(st, nl);
            double new_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double new_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            bool accepted = false;
            double dE = (new_cross-old_cross) + (new_sasa-old_sasa)
                      + (new_dih-old_dih)   + (new_ss-old_ss);
            if (dE < 0.0 || uni(gen) < std::exp(-dE / T)) {
                curE += dE;
                accepted = true;
            } else {
                for (size_t k = 0; k < all_idx.size(); ++k) {
                    int idx = all_idx[k];
                    st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                    a[idx]    = old_born[k];
                }
            }
            for (int k : all_idx) { in_side[k] = false; side_group[k] = 0; }
            return {accepted, true};
        };

        // 이동 종류 선택 확률 (Move-type selection probabilities): concerted
        // sidechain 15% (있을 때만) -> 남은 확률의 25%가 crankshaft -> 나머지가
        // torsion. 실효 확률(concerted 후보가 있을 때): concerted 15%,
        // crankshaft (1-0.15)*0.25=21.25%, torsion 63.75%. 후보가 없는 단백질은
        // 기존과 완전히 동일하게 동작(순차 확인이라 자동으로 하위 분기로 감).
        //
        // Effective probabilities when concerted-sidechain candidates exist:
        // concerted 15%, crankshaft (1-0.15)*0.25=21.25%, torsion 63.75%.
        // Proteins with zero candidate pairs behave exactly as before (the
        // sequential check falls through automatically).
        constexpr double CONCERTED_SIDECHAIN_PROB = 0.15;

        for (int snap = 0; snap < n_snapshots; ++snap) {
            for (int s = 0; s < steps_per_snapshot; ++s) {
                if (nl.needs_rebuild(st)) {
                    nl.build(st);
                    a    = born_radii(st, nl);
                    curE = total_e(st, nl, a, &topo);
                }
                bool do_concerted = ncsp > 0 && uni(gen) < CONCERTED_SIDECHAIN_PROB;
                bool do_crank     = !do_concerted && ncp > 0 && uni(gen) < 0.25;
                auto [accepted, did_move] = do_concerted ? try_concerted_sidechain()
                                           : do_crank     ? try_crankshaft()
                                                           : try_torsion();
                if (did_move) {
                    acc_win += accepted ? 1 : 0;
                    if (++tot_win == TUNE_FREQ) {
                        double rate = (double)acc_win / TUNE_FREQ;
                        if      (rate > TARGET_HI) cur_max = std::min(ANGLE_MAX, cur_max * SCALE_UP);
                        else if (rate < TARGET_LO) cur_max = std::max(ANGLE_MIN, cur_max * SCALE_DOWN);
                        acc_win = tot_win = 0;
                    }
                }
            }
            snapshots.push_back(st);
            // Recorded energy matches the old Python-level LandscapeWorker exactly:
            // a from-scratch calculate_potential() call (fresh NL + fresh Born radii),
            // not the loop's incrementally-tracked curE (which only rebuilds Born
            // radii for the atoms that just moved, and would drift slightly from
            // a true from-scratch value over thousands of steps).
            energies.push_back(calculate_potential(st, &topo));
        }
        return {snapshots, energies};
    }

    // run_landscape_segment: identical to run_landscape_trajectory above, except
    // it also returns the final tuned cur_max. Exists for replica-exchange
    // (parallel tempering, python/gui_main.py's _ReplicaExchangeBranchRunnable):
    // run_landscape_trajectory's online step-size tuning (cur_max, rescaled every
    // TUNE_FREQ steps toward [TARGET_LO, TARGET_HI] acceptance) is call-scoped --
    // it resets to max_angle and is discarded at the end of every call. PT calls
    // the trajectory function once per short swap-interval segment rather than
    // once for the whole run, so without this, cur_max would restart from
    // scratch every segment instead of carrying forward what it learned -- a
    // real bug found empirically (1YPI, the largest test protein, got a
    // *noisier* funnel score under PT than plain MC, plausibly because its
    // bigger conformational space needs more than ~15 tuning windows per
    // segment to find a good step size). This function lets the Python caller
    // thread cur_max through as the next segment's max_angle argument.
    //
    // Deliberately a full duplicate of run_landscape_trajectory's body, not a
    // shared private helper that both call -- matching this file's own
    // established precedent (see run_landscape_trajectory's own docstring
    // above: its body is "duplicated rather than shared via a helper" relative
    // to generate_ensemble, specifically to avoid touching an already-verified
    // hot path). Refactoring a shared helper here would require editing
    // run_landscape_trajectory itself, which this precedent exists to avoid.
    //
    // CPU only -- physics_engine_cuda.cu's run_landscape_trajectory has the
    // identical reset behavior (S.cur_max = max_angle at the top of every
    // call) but does not yet have a GPU equivalent of this fix; deferred.
    std::tuple<std::vector<std::vector<Particle>>, std::vector<double>, double>
    run_landscape_segment(
        const std::vector<Particle>& init,
        const BondTopology& topo,
        int n_snapshots, int steps_per_snapshot,
        double T = 0.6,
        double max_angle = 0.12)
    {
        if (init.empty()) throw std::invalid_argument("initial_state empty");
        if (n_snapshots <= 0 || steps_per_snapshot <= 0)
            throw std::invalid_argument("n_snapshots/steps_per_snapshot must be positive");
        if (topo.rot_bonds.empty()) throw std::invalid_argument("topology has no rotatable bonds");

        size_t N   = init.size();
        int    nrb = (int)topo.rot_bonds.size();
        int    ncp = (int)topo.concerted_pairs.size();

        std::vector<std::vector<Particle>> snapshots;
        std::vector<double>                energies;
        snapshots.reserve(n_snapshots);
        energies.reserve(n_snapshots);

        std::vector<Particle> st = init;
        NeighborList nl; nl.build(st);
        auto a = born_radii(st, nl);
        double curE = total_e(st, nl, a, &topo);

        std::uniform_int_distribution<int>    pick_rb(0, nrb - 1);
        std::uniform_int_distribution<int>    pick_cp(0, std::max(0, ncp - 1));
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<bool> in_side(N, false);

        double cur_max = max_angle;
        int    acc_win = 0, tot_win = 0;
        constexpr int    TUNE_FREQ  = 200;
        constexpr double TARGET_LO  = 0.28, TARGET_HI = 0.52;
        constexpr double SCALE_UP   = 1.06,  SCALE_DOWN = 0.94;
        constexpr double ANGLE_MAX  = 0.50,  ANGLE_MIN  = 0.004;

        auto cross_e = [&]() -> double {
            double E = 0.0;
            for (size_t i = 0; i < N; ++i)
                for (size_t j : nl.nb[i]) {
                    if (in_side[i] == in_side[j]) continue;
                    if (topo.is_excluded((int)i, (int)j)) continue;
                    double dx = st[i].x-st[j].x, dy = st[i].y-st[j].y, dz = st[i].z-st[j].z;
                    if (dx*dx+dy*dy+dz*dz > PAIR_CUT2) continue;
                    E += pair_e(st[i], st[j], a[i], a[j]);
                }
            return E;
        };

        auto try_torsion = [&]() -> std::pair<bool,bool> {
            int            rb_idx = pick_rb(gen);
            const RotBond& rb     = topo.rot_bonds[rb_idx];
            const std::vector<int>& side = topo.rot_bond_sides[rb_idx];
            if (side.empty()) return {false, false};

            double ax = st[rb.j].x-st[rb.i].x, ay = st[rb.j].y-st[rb.i].y, az = st[rb.j].z-st[rb.i].z;
            double al = std::sqrt(ax*ax+ay*ay+az*az);
            if (al < 1e-10) return {false, false};
            ax/=al; ay/=al; az/=al;

            double base_d = (rb.kind == BondKind::BACKBONE_PHI ||
                             rb.kind == BondKind::BACKBONE_PSI)
                            ? cur_max : cur_max * 2.5;
            base_d *= topo.rot_bond_scale[rb_idx];
            double delta = std::uniform_real_distribution<double>(-base_d, base_d)(gen);
            double cosD  = std::cos(delta), sinD = std::sin(delta);

            for (int k : side) in_side[k] = true;

            std::vector<std::array<double,3>> old_pos(side.size());
            std::vector<double>               old_born(side.size());
            for (size_t k = 0; k < side.size(); ++k) {
                int idx = side[k];
                old_pos[k]  = {st[idx].x, st[idx].y, st[idx].z};
                old_born[k] = a[idx];
            }

            double old_cross = cross_e();
            double old_sasa  = sasa_nonpolar(st, nl);
            double old_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double old_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            double ox = st[rb.i].x, oy = st[rb.i].y, oz = st[rb.i].z;
            for (int k : side)
                rodrigues(st[k].x, st[k].y, st[k].z, ox, oy, oz, ax, ay, az, cosD, sinD);
            for (int k : side) update_born((size_t)k, st, nl, a);

            double new_cross = cross_e();
            double new_sasa  = sasa_nonpolar(st, nl);
            double new_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double new_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            bool accepted = false;
            double dE = (new_cross-old_cross) + (new_sasa-old_sasa)
                      + (new_dih-old_dih)   + (new_ss-old_ss);
            if (dE < 0.0 || uni(gen) < std::exp(-dE / T)) {
                curE += dE;
                accepted = true;
            } else {
                for (size_t k = 0; k < side.size(); ++k) {
                    int idx = side[k];
                    st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                    a[idx]    = old_born[k];
                }
            }
            for (int k : side) in_side[k] = false;
            return {accepted, true};
        };

        auto try_crankshaft = [&]() -> std::pair<bool,bool> {
            if (ncp == 0) return {false, false};
            int cp_idx = (ncp > 1) ? pick_cp(gen) : 0;
            int phi_k  = topo.concerted_pairs[cp_idx].first;
            int psi_k  = topo.concerted_pairs[cp_idx].second;
            const std::vector<int>& phi_side = topo.rot_bond_sides[phi_k];
            const std::vector<int>& psi_side = topo.rot_bond_sides[psi_k];
            if (phi_side.empty() || psi_side.empty()) return {false, false};

            for (int k : phi_side) in_side[k] = true;

            const RotBond& phi_rb = topo.rot_bonds[phi_k];
            double p1ox = st[phi_rb.i].x, p1oy = st[phi_rb.i].y, p1oz = st[phi_rb.i].z;
            double p1ax = st[phi_rb.j].x-p1ox, p1ay = st[phi_rb.j].y-p1oy, p1az = st[phi_rb.j].z-p1oz;
            double p1l  = std::sqrt(p1ax*p1ax+p1ay*p1ay+p1az*p1az);
            if (p1l < 1e-10) { for (int k : phi_side) in_side[k] = false; return {false, false}; }
            p1ax/=p1l; p1ay/=p1l; p1az/=p1l;

            std::vector<std::array<double,3>> old_pos(phi_side.size());
            std::vector<double>               old_born(phi_side.size());
            for (size_t k = 0; k < phi_side.size(); ++k) {
                int idx = phi_side[k];
                old_pos[k]  = {st[idx].x, st[idx].y, st[idx].z};
                old_born[k] = a[idx];
            }
            double old_cross = cross_e();
            double old_sasa  = sasa_nonpolar(st, nl);
            double old_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double old_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            double delta = std::uniform_real_distribution<double>(-cur_max, cur_max)(gen);
            double cosD  = std::cos(delta), sinD = std::sin(delta);

            for (int k : phi_side)
                rodrigues(st[k].x, st[k].y, st[k].z, p1ox, p1oy, p1oz, p1ax, p1ay, p1az, cosD, sinD);

            const RotBond& psi_rb = topo.rot_bonds[psi_k];
            double p2ox = st[psi_rb.i].x, p2oy = st[psi_rb.i].y, p2oz = st[psi_rb.i].z;
            double p2ax = st[psi_rb.j].x-p2ox, p2ay = st[psi_rb.j].y-p2oy, p2az = st[psi_rb.j].z-p2oz;
            double p2l  = std::sqrt(p2ax*p2ax+p2ay*p2ay+p2az*p2az);
            if (p2l < 1e-10) {
                for (size_t k = 0; k < phi_side.size(); ++k) {
                    int idx = phi_side[k];
                    st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                }
                for (int k : phi_side) in_side[k] = false;
                return {false, false};
            }
            p2ax/=p2l; p2ay/=p2l; p2az/=p2l;
            for (int k : psi_side)
                rodrigues(st[k].x, st[k].y, st[k].z, p2ox, p2oy, p2oz, p2ax, p2ay, p2az, cosD, -sinD);

            for (int k : phi_side) update_born((size_t)k, st, nl, a);

            double new_cross = cross_e();
            double new_sasa  = sasa_nonpolar(st, nl);
            double new_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double new_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            bool accepted = false;
            double dE = (new_cross-old_cross) + (new_sasa-old_sasa)
                      + (new_dih-old_dih)   + (new_ss-old_ss);
            if (dE < 0.0 || uni(gen) < std::exp(-dE / T)) {
                curE += dE;
                accepted = true;
            } else {
                for (size_t k = 0; k < phi_side.size(); ++k) {
                    int idx = phi_side[k];
                    st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                    a[idx]    = old_born[k];
                }
            }
            for (int k : phi_side) in_side[k] = false;
            return {accepted, true};
        };

        // try_concerted_sidechain: identical to the copy in run_landscape_trajectory
        // above (see the detailed comment there for the physical motivation and the
        // side_group/cross_e_concerted design rationale) -- duplicated here rather
        // than shared, matching this pair of functions' existing precedent.
        std::vector<int> side_group(N, 0);
        auto cross_e_concerted = [&]() -> double {
            double E = 0.0;
            for (size_t i = 0; i < N; ++i)
                for (size_t j : nl.nb[i]) {
                    if (side_group[i] == side_group[j]) continue;
                    if (topo.is_excluded((int)i, (int)j)) continue;
                    double dx = st[i].x-st[j].x, dy = st[i].y-st[j].y, dz = st[i].z-st[j].z;
                    if (dx*dx+dy*dy+dz*dz > PAIR_CUT2) continue;
                    E += pair_e(st[i], st[j], a[i], a[j]);
                }
            return E;
        };
        const int ncsp = (int)topo.concerted_sidechain_pairs.size();
        std::uniform_int_distribution<int> pick_csp(0, std::max(0, ncsp - 1));

        auto try_concerted_sidechain = [&]() -> std::pair<bool,bool> {
            if (ncsp == 0) return {false, false};
            int   pair_idx = (ncsp > 1) ? pick_csp(gen) : 0;
            int   bond_a   = topo.concerted_sidechain_pairs[pair_idx].first;
            int   bond_b   = topo.concerted_sidechain_pairs[pair_idx].second;
            const RotBond& rb_a = topo.rot_bonds[bond_a];
            const RotBond& rb_b = topo.rot_bonds[bond_b];
            const std::vector<int>& side_a = topo.rot_bond_sides[bond_a];
            const std::vector<int>& side_b = topo.rot_bond_sides[bond_b];
            if (side_a.empty() || side_b.empty()) return {false, false};

            double axA = st[rb_a.j].x-st[rb_a.i].x, ayA = st[rb_a.j].y-st[rb_a.i].y, azA = st[rb_a.j].z-st[rb_a.i].z;
            double alA = std::sqrt(axA*axA+ayA*ayA+azA*azA);
            if (alA < 1e-10) return {false, false};
            axA/=alA; ayA/=alA; azA/=alA;
            double axB = st[rb_b.j].x-st[rb_b.i].x, ayB = st[rb_b.j].y-st[rb_b.i].y, azB = st[rb_b.j].z-st[rb_b.i].z;
            double alB = std::sqrt(axB*axB+ayB*ayB+azB*azB);
            if (alB < 1e-10) return {false, false};
            axB/=alB; ayB/=alB; azB/=alB;

            double deltaA = std::uniform_real_distribution<double>(
                -cur_max*2.5*topo.rot_bond_scale[bond_a], cur_max*2.5*topo.rot_bond_scale[bond_a])(gen);
            double deltaB = std::uniform_real_distribution<double>(
                -cur_max*2.5*topo.rot_bond_scale[bond_b], cur_max*2.5*topo.rot_bond_scale[bond_b])(gen);
            double cosA = std::cos(deltaA), sinA = std::sin(deltaA);
            double cosB = std::cos(deltaB), sinB = std::sin(deltaB);

            for (int k : side_a) { in_side[k] = true; side_group[k] = 1; }
            for (int k : side_b) { in_side[k] = true; side_group[k] = 2; }

            std::vector<int> all_idx;
            all_idx.reserve(side_a.size() + side_b.size());
            all_idx.insert(all_idx.end(), side_a.begin(), side_a.end());
            all_idx.insert(all_idx.end(), side_b.begin(), side_b.end());
            std::vector<std::array<double,3>> old_pos(all_idx.size());
            std::vector<double>               old_born(all_idx.size());
            for (size_t k = 0; k < all_idx.size(); ++k) {
                int idx = all_idx[k];
                old_pos[k]  = {st[idx].x, st[idx].y, st[idx].z};
                old_born[k] = a[idx];
            }

            double old_cross = cross_e_concerted();
            double old_sasa  = sasa_nonpolar(st, nl);
            double old_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double old_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            double oxA = st[rb_a.i].x, oyA = st[rb_a.i].y, ozA = st[rb_a.i].z;
            for (int k : side_a)
                rodrigues(st[k].x, st[k].y, st[k].z, oxA, oyA, ozA, axA, ayA, azA, cosA, sinA);
            double oxB = st[rb_b.i].x, oyB = st[rb_b.i].y, ozB = st[rb_b.i].z;
            for (int k : side_b)
                rodrigues(st[k].x, st[k].y, st[k].z, oxB, oyB, ozB, axB, ayB, azB, cosB, sinB);
            for (int k : all_idx) update_born((size_t)k, st, nl, a);

            double new_cross = cross_e_concerted();
            double new_sasa  = sasa_nonpolar(st, nl);
            double new_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double new_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            bool accepted = false;
            double dE = (new_cross-old_cross) + (new_sasa-old_sasa)
                      + (new_dih-old_dih)   + (new_ss-old_ss);
            if (dE < 0.0 || uni(gen) < std::exp(-dE / T)) {
                curE += dE;
                accepted = true;
            } else {
                for (size_t k = 0; k < all_idx.size(); ++k) {
                    int idx = all_idx[k];
                    st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                    a[idx]    = old_born[k];
                }
            }
            for (int k : all_idx) { in_side[k] = false; side_group[k] = 0; }
            return {accepted, true};
        };

        constexpr double CONCERTED_SIDECHAIN_PROB = 0.15;

        for (int snap = 0; snap < n_snapshots; ++snap) {
            for (int s = 0; s < steps_per_snapshot; ++s) {
                if (nl.needs_rebuild(st)) {
                    nl.build(st);
                    a    = born_radii(st, nl);
                    curE = total_e(st, nl, a, &topo);
                }
                bool do_concerted = ncsp > 0 && uni(gen) < CONCERTED_SIDECHAIN_PROB;
                bool do_crank     = !do_concerted && ncp > 0 && uni(gen) < 0.25;
                auto [accepted, did_move] = do_concerted ? try_concerted_sidechain()
                                           : do_crank     ? try_crankshaft()
                                                           : try_torsion();
                if (did_move) {
                    acc_win += accepted ? 1 : 0;
                    if (++tot_win == TUNE_FREQ) {
                        double rate = (double)acc_win / TUNE_FREQ;
                        if      (rate > TARGET_HI) cur_max = std::min(ANGLE_MAX, cur_max * SCALE_UP);
                        else if (rate < TARGET_LO) cur_max = std::max(ANGLE_MIN, cur_max * SCALE_DOWN);
                        acc_win = tot_win = 0;
                    }
                }
            }
            snapshots.push_back(st);
            energies.push_back(calculate_potential(st, &topo));
        }
        return {snapshots, energies, cur_max};
    }

    // run_mc_diagnostic: MC-rejection-cause investigation (IMPROVEMENTS.md item #2's
    // remaining open question -- is the sampler's mixing bottleneck a torsional
    // barrier or a correlated steric clash the current single-bond/single-pair move
    // set can't route around?). Runs n_steps of the same torsion/crankshaft MC used
    // everywhere else in this file, but for every *attempted* move (accepted or
    // rejected) logs: move type, the proposed angle magnitude (to separate small
    // routine in-basin jitter from large potential basin-crossing attempts during
    // analysis), whether accepted, and the energy-component breakdown split out via
    // pair_e_diag/cross_e_diag above (hard-core isolated from the rest of the
    // non-bonded sum). Returned as parallel vectors, not a vector-of-structs, for
    // easy numpy analysis on the Python side.
    //
    // Duplicated from the same try_torsion/try_crankshaft skeleton used by
    // run_landscape_trajectory/run_landscape_segment above rather than sharing code
    // with them -- identical reasoning as those two: avoid touching an
    // already-verified hot path for a change that's purely additive logging.
    // CPU only (an introspection tool, no GPU counterpart needed).
    struct McDiagnosticResult {
        std::vector<int>    move_type;       // 0 = torsion, 1 = crankshaft
        std::vector<double> proposed_delta;   // |sampled angle|, radians, pre-scaling
        std::vector<int>    accepted;         // 0/1 (not vector<bool> -- pybind11/numpy friendliness)
        std::vector<double> d_hardcore;
        std::vector<double> d_nonbonded_soft; // edh+egb+elj delta, hard-core excluded
        std::vector<double> d_sasa;
        std::vector<double> d_dih;
        std::vector<double> d_ss;
    };

    McDiagnosticResult run_mc_diagnostic(
        const std::vector<Particle>& init,
        const BondTopology& topo,
        int n_steps,
        double T = 0.6,
        double max_angle = 0.12)
    {
        if (init.empty()) throw std::invalid_argument("initial_state empty");
        if (n_steps <= 0) throw std::invalid_argument("n_steps must be positive");
        if (topo.rot_bonds.empty()) throw std::invalid_argument("topology has no rotatable bonds");

        size_t N   = init.size();
        int    nrb = (int)topo.rot_bonds.size();
        int    ncp = (int)topo.concerted_pairs.size();

        std::vector<Particle> st = init;
        NeighborList nl; nl.build(st);
        auto a = born_radii(st, nl);
        double curE = total_e(st, nl, a, &topo);

        std::uniform_int_distribution<int>    pick_rb(0, nrb - 1);
        std::uniform_int_distribution<int>    pick_cp(0, std::max(0, ncp - 1));
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<bool> in_side(N, false);

        double cur_max = max_angle;
        int    acc_win = 0, tot_win = 0;
        constexpr int    TUNE_FREQ  = 200;
        constexpr double TARGET_LO  = 0.28, TARGET_HI = 0.52;
        constexpr double SCALE_UP   = 1.06,  SCALE_DOWN = 0.94;
        constexpr double ANGLE_MAX  = 0.50,  ANGLE_MIN  = 0.004;

        McDiagnosticResult out;
        out.move_type.reserve(n_steps);
        out.proposed_delta.reserve(n_steps);
        out.accepted.reserve(n_steps);
        out.d_hardcore.reserve(n_steps);
        out.d_nonbonded_soft.reserve(n_steps);
        out.d_sasa.reserve(n_steps);
        out.d_dih.reserve(n_steps);
        out.d_ss.reserve(n_steps);

        // cross_e_diag: same pair-loop shape as cross_e() above, but also
        // accumulates the hard-core-only sub-total via pair_e_diag.
        auto cross_e_diag = [&](double& hardcore_sum) -> double {
            double E = 0.0;
            hardcore_sum = 0.0;
            for (size_t i = 0; i < N; ++i)
                for (size_t j : nl.nb[i]) {
                    if (in_side[i] == in_side[j]) continue;
                    if (topo.is_excluded((int)i, (int)j)) continue;
                    double dx = st[i].x-st[j].x, dy = st[i].y-st[j].y, dz = st[i].z-st[j].z;
                    if (dx*dx+dy*dy+dz*dz > PAIR_CUT2) continue;
                    double hc = 0.0;
                    E += pair_e_diag(st[i], st[j], a[i], a[j], hc);
                    hardcore_sum += hc;
                }
            return E;
        };

        auto try_torsion_diag = [&]() -> bool {
            int            rb_idx = pick_rb(gen);
            const RotBond& rb     = topo.rot_bonds[rb_idx];
            const std::vector<int>& side = topo.rot_bond_sides[rb_idx];
            if (side.empty()) return false;

            double ax = st[rb.j].x-st[rb.i].x, ay = st[rb.j].y-st[rb.i].y, az = st[rb.j].z-st[rb.i].z;
            double al = std::sqrt(ax*ax+ay*ay+az*az);
            if (al < 1e-10) return false;
            ax/=al; ay/=al; az/=al;

            double base_d = (rb.kind == BondKind::BACKBONE_PHI ||
                             rb.kind == BondKind::BACKBONE_PSI)
                            ? cur_max : cur_max * 2.5;
            base_d *= topo.rot_bond_scale[rb_idx];
            double delta = std::uniform_real_distribution<double>(-base_d, base_d)(gen);
            double cosD  = std::cos(delta), sinD = std::sin(delta);

            for (int k : side) in_side[k] = true;

            std::vector<std::array<double,3>> old_pos(side.size());
            std::vector<double>               old_born(side.size());
            for (size_t k = 0; k < side.size(); ++k) {
                int idx = side[k];
                old_pos[k]  = {st[idx].x, st[idx].y, st[idx].z};
                old_born[k] = a[idx];
            }

            double hc_old = 0.0;
            double old_cross = cross_e_diag(hc_old);
            double old_sasa  = sasa_nonpolar(st, nl);
            double old_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double old_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            double ox = st[rb.i].x, oy = st[rb.i].y, oz = st[rb.i].z;
            for (int k : side)
                rodrigues(st[k].x, st[k].y, st[k].z, ox, oy, oz, ax, ay, az, cosD, sinD);
            for (int k : side) update_born((size_t)k, st, nl, a);

            double hc_new = 0.0;
            double new_cross = cross_e_diag(hc_new);
            double new_sasa  = sasa_nonpolar(st, nl);
            double new_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double new_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            double d_hc  = hc_new - hc_old;
            double d_soft = (new_cross - hc_new) - (old_cross - hc_old);
            double d_sasa_ = new_sasa - old_sasa;
            double d_dih_  = new_dih - old_dih;
            double d_ss_   = new_ss - old_ss;
            double dE = (new_cross-old_cross) + d_sasa_ + d_dih_ + d_ss_;

            bool accepted = false;
            if (dE < 0.0 || uni(gen) < std::exp(-dE / T)) {
                curE += dE;
                accepted = true;
            } else {
                for (size_t k = 0; k < side.size(); ++k) {
                    int idx = side[k];
                    st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                    a[idx]    = old_born[k];
                }
            }
            for (int k : side) in_side[k] = false;

            out.move_type.push_back(0);
            out.proposed_delta.push_back(std::fabs(delta));
            out.accepted.push_back(accepted ? 1 : 0);
            out.d_hardcore.push_back(d_hc);
            out.d_nonbonded_soft.push_back(d_soft);
            out.d_sasa.push_back(d_sasa_);
            out.d_dih.push_back(d_dih_);
            out.d_ss.push_back(d_ss_);

            // acceptance-window tuning, identical to run_landscape_segment
            acc_win += accepted ? 1 : 0;
            if (++tot_win == TUNE_FREQ) {
                double rate = (double)acc_win / TUNE_FREQ;
                if      (rate > TARGET_HI) cur_max = std::min(ANGLE_MAX, cur_max * SCALE_UP);
                else if (rate < TARGET_LO) cur_max = std::max(ANGLE_MIN, cur_max * SCALE_DOWN);
                acc_win = tot_win = 0;
            }
            return true;
        };

        auto try_crankshaft_diag = [&]() -> bool {
            if (ncp == 0) return false;
            int cp_idx = (ncp > 1) ? pick_cp(gen) : 0;
            int phi_k  = topo.concerted_pairs[cp_idx].first;
            int psi_k  = topo.concerted_pairs[cp_idx].second;
            const std::vector<int>& phi_side = topo.rot_bond_sides[phi_k];
            const std::vector<int>& psi_side = topo.rot_bond_sides[psi_k];
            if (phi_side.empty() || psi_side.empty()) return false;

            for (int k : phi_side) in_side[k] = true;

            const RotBond& phi_rb = topo.rot_bonds[phi_k];
            double p1ox = st[phi_rb.i].x, p1oy = st[phi_rb.i].y, p1oz = st[phi_rb.i].z;
            double p1ax = st[phi_rb.j].x-p1ox, p1ay = st[phi_rb.j].y-p1oy, p1az = st[phi_rb.j].z-p1oz;
            double p1l  = std::sqrt(p1ax*p1ax+p1ay*p1ay+p1az*p1az);
            if (p1l < 1e-10) { for (int k : phi_side) in_side[k] = false; return false; }
            p1ax/=p1l; p1ay/=p1l; p1az/=p1l;

            std::vector<std::array<double,3>> old_pos(phi_side.size());
            std::vector<double>               old_born(phi_side.size());
            for (size_t k = 0; k < phi_side.size(); ++k) {
                int idx = phi_side[k];
                old_pos[k]  = {st[idx].x, st[idx].y, st[idx].z};
                old_born[k] = a[idx];
            }
            double hc_old = 0.0;
            double old_cross = cross_e_diag(hc_old);
            double old_sasa  = sasa_nonpolar(st, nl);
            double old_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double old_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            double delta = std::uniform_real_distribution<double>(-cur_max, cur_max)(gen);
            double cosD  = std::cos(delta), sinD = std::sin(delta);

            for (int k : phi_side)
                rodrigues(st[k].x, st[k].y, st[k].z, p1ox, p1oy, p1oz, p1ax, p1ay, p1az, cosD, sinD);

            const RotBond& psi_rb = topo.rot_bonds[psi_k];
            double p2ox = st[psi_rb.i].x, p2oy = st[psi_rb.i].y, p2oz = st[psi_rb.i].z;
            double p2ax = st[psi_rb.j].x-p2ox, p2ay = st[psi_rb.j].y-p2oy, p2az = st[psi_rb.j].z-p2oz;
            double p2l  = std::sqrt(p2ax*p2ax+p2ay*p2ay+p2az*p2az);
            if (p2l < 1e-10) {
                for (size_t k = 0; k < phi_side.size(); ++k) {
                    int idx = phi_side[k];
                    st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                }
                for (int k : phi_side) in_side[k] = false;
                return false;
            }
            p2ax/=p2l; p2ay/=p2l; p2az/=p2l;
            for (int k : psi_side)
                rodrigues(st[k].x, st[k].y, st[k].z, p2ox, p2oy, p2oz, p2ax, p2ay, p2az, cosD, -sinD);

            for (int k : phi_side) update_born((size_t)k, st, nl, a);

            double hc_new = 0.0;
            double new_cross = cross_e_diag(hc_new);
            double new_sasa  = sasa_nonpolar(st, nl);
            double new_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double new_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            double d_hc  = hc_new - hc_old;
            double d_soft = (new_cross - hc_new) - (old_cross - hc_old);
            double d_sasa_ = new_sasa - old_sasa;
            double d_dih_  = new_dih - old_dih;
            double d_ss_   = new_ss - old_ss;
            double dE = (new_cross-old_cross) + d_sasa_ + d_dih_ + d_ss_;

            bool accepted = false;
            if (dE < 0.0 || uni(gen) < std::exp(-dE / T)) {
                curE += dE;
                accepted = true;
            } else {
                for (size_t k = 0; k < phi_side.size(); ++k) {
                    int idx = phi_side[k];
                    st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                    a[idx]    = old_born[k];
                }
            }
            for (int k : phi_side) in_side[k] = false;

            out.move_type.push_back(1);
            out.proposed_delta.push_back(std::fabs(delta));
            out.accepted.push_back(accepted ? 1 : 0);
            out.d_hardcore.push_back(d_hc);
            out.d_nonbonded_soft.push_back(d_soft);
            out.d_sasa.push_back(d_sasa_);
            out.d_dih.push_back(d_dih_);
            out.d_ss.push_back(d_ss_);

            acc_win += accepted ? 1 : 0;
            if (++tot_win == TUNE_FREQ) {
                double rate = (double)acc_win / TUNE_FREQ;
                if      (rate > TARGET_HI) cur_max = std::min(ANGLE_MAX, cur_max * SCALE_UP);
                else if (rate < TARGET_LO) cur_max = std::max(ANGLE_MIN, cur_max * SCALE_DOWN);
                acc_win = tot_win = 0;
            }
            return true;
        };

        // Plain (undiagnosed) concerted-sidechain move -- Part 6's evidence-backed fix
        // (physics_engine.cpp run_landscape_trajectory/run_landscape_segment), included
        // here so this diagnostic's *sampled states* reflect the same mixing the new
        // move gives production runs -- otherwise re-running this diagnostic couldn't
        // show whether crankshaft's steric-dominant rejection rate actually drops.
        // Not logged into `out` (its arrays are torsion(0)/crankshaft(1) only, and
        // this move's own d_hardcore/d_dih split isn't the question being re-tested
        // here) -- it just advances the chain like an accepted/rejected torsion or
        // crankshaft move would, silently, same as the existing degenerate-retry path.
        std::vector<int> side_group(N, 0);
        auto cross_e_concerted = [&]() -> double {
            double E = 0.0;
            for (size_t i = 0; i < N; ++i)
                for (size_t j : nl.nb[i]) {
                    if (side_group[i] == side_group[j]) continue;
                    if (topo.is_excluded((int)i, (int)j)) continue;
                    double dx = st[i].x-st[j].x, dy = st[i].y-st[j].y, dz = st[i].z-st[j].z;
                    if (dx*dx+dy*dy+dz*dz > PAIR_CUT2) continue;
                    E += pair_e(st[i], st[j], a[i], a[j]);
                }
            return E;
        };
        const int ncsp = (int)topo.concerted_sidechain_pairs.size();
        std::uniform_int_distribution<int> pick_csp(0, std::max(0, ncsp - 1));
        constexpr double CONCERTED_SIDECHAIN_PROB = 0.15;

        auto try_concerted_sidechain_plain = [&]() -> bool {
            if (ncsp == 0) return false;
            int   pair_idx = (ncsp > 1) ? pick_csp(gen) : 0;
            int   bond_a   = topo.concerted_sidechain_pairs[pair_idx].first;
            int   bond_b   = topo.concerted_sidechain_pairs[pair_idx].second;
            const RotBond& rb_a = topo.rot_bonds[bond_a];
            const RotBond& rb_b = topo.rot_bonds[bond_b];
            const std::vector<int>& side_a = topo.rot_bond_sides[bond_a];
            const std::vector<int>& side_b = topo.rot_bond_sides[bond_b];
            if (side_a.empty() || side_b.empty()) return false;

            double axA = st[rb_a.j].x-st[rb_a.i].x, ayA = st[rb_a.j].y-st[rb_a.i].y, azA = st[rb_a.j].z-st[rb_a.i].z;
            double alA = std::sqrt(axA*axA+ayA*ayA+azA*azA);
            if (alA < 1e-10) return false;
            axA/=alA; ayA/=alA; azA/=alA;
            double axB = st[rb_b.j].x-st[rb_b.i].x, ayB = st[rb_b.j].y-st[rb_b.i].y, azB = st[rb_b.j].z-st[rb_b.i].z;
            double alB = std::sqrt(axB*axB+ayB*ayB+azB*azB);
            if (alB < 1e-10) return false;
            axB/=alB; ayB/=alB; azB/=alB;

            double deltaA = std::uniform_real_distribution<double>(
                -cur_max*2.5*topo.rot_bond_scale[bond_a], cur_max*2.5*topo.rot_bond_scale[bond_a])(gen);
            double deltaB = std::uniform_real_distribution<double>(
                -cur_max*2.5*topo.rot_bond_scale[bond_b], cur_max*2.5*topo.rot_bond_scale[bond_b])(gen);
            double cosA = std::cos(deltaA), sinA = std::sin(deltaA);
            double cosB = std::cos(deltaB), sinB = std::sin(deltaB);

            for (int k : side_a) { in_side[k] = true; side_group[k] = 1; }
            for (int k : side_b) { in_side[k] = true; side_group[k] = 2; }

            std::vector<int> all_idx;
            all_idx.reserve(side_a.size() + side_b.size());
            all_idx.insert(all_idx.end(), side_a.begin(), side_a.end());
            all_idx.insert(all_idx.end(), side_b.begin(), side_b.end());
            std::vector<std::array<double,3>> old_pos(all_idx.size());
            std::vector<double>               old_born(all_idx.size());
            for (size_t k = 0; k < all_idx.size(); ++k) {
                int idx = all_idx[k];
                old_pos[k]  = {st[idx].x, st[idx].y, st[idx].z};
                old_born[k] = a[idx];
            }

            double old_cross = cross_e_concerted();
            double old_sasa  = sasa_nonpolar(st, nl);
            double old_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double old_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            double oxA = st[rb_a.i].x, oyA = st[rb_a.i].y, ozA = st[rb_a.i].z;
            for (int k : side_a)
                rodrigues(st[k].x, st[k].y, st[k].z, oxA, oyA, ozA, axA, ayA, azA, cosA, sinA);
            double oxB = st[rb_b.i].x, oyB = st[rb_b.i].y, ozB = st[rb_b.i].z;
            for (int k : side_b)
                rodrigues(st[k].x, st[k].y, st[k].z, oxB, oyB, ozB, axB, ayB, azB, cosB, sinB);
            for (int k : all_idx) update_born((size_t)k, st, nl, a);

            double new_cross = cross_e_concerted();
            double new_sasa  = sasa_nonpolar(st, nl);
            double new_dih   = dihedral_e_boundary(st, topo.dihedrals, in_side);
            double new_ss    = topo.disulfide_pairs.empty() ? 0.0
                             : ss_e_side(st, topo.disulfide_pairs, in_side);

            double dE = (new_cross-old_cross) + (new_sasa-old_sasa)
                      + (new_dih-old_dih)   + (new_ss-old_ss);
            if (dE < 0.0 || uni(gen) < std::exp(-dE / T)) {
                curE += dE;
            } else {
                for (size_t k = 0; k < all_idx.size(); ++k) {
                    int idx = all_idx[k];
                    st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
                    a[idx]    = old_born[k];
                }
            }
            for (int k : all_idx) { in_side[k] = false; side_group[k] = 0; }
            return true;
        };

        for (int s = 0; s < n_steps; ++s) {
            if (nl.needs_rebuild(st)) {
                nl.build(st);
                a    = born_radii(st, nl);
                curE = total_e(st, nl, a, &topo);
            }
            if (ncsp > 0 && uni(gen) < CONCERTED_SIDECHAIN_PROB) {
                if (!try_concerted_sidechain_plain()) --s;  // degenerate -- retry, don't log
                continue;
            }
            bool do_crank = ncp > 0 && uni(gen) < 0.25;
            bool did_move = do_crank ? try_crankshaft_diag() : try_torsion_diag();
            if (!did_move) --s;  // degenerate proposal (empty side/zero-length axis) -- retry, don't log
        }
        return out;
    }

    // lowest_energy_structure: scan an ensemble and return the conformation with the
    // minimum total potential energy.  Calls calculate_potential on every member,
    // so this is O(ncand · N²) and should only be called once after MC finishes.
    std::vector<Particle> lowest_energy_structure(const std::vector<std::vector<Particle>>& ens){
        if(ens.empty()) throw std::invalid_argument("ensemble empty");
        return *std::min_element(ens.begin(),ens.end(),
            [this](const auto& a,const auto& b){return calculate_potential(a)<calculate_potential(b);});
    }

    // num_threads: returns the number of OpenMP threads available (1 if not compiled with OpenMP).
    int num_threads() const {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }
};

// ── pybind11 module ───────────────────────────────────────────────────────────
// Exposes Particle and PhysicsEngine to Python as 'protein_physics'.
// py::call_guard<py::gil_scoped_release>() releases the Python GIL during
// long C++ calls so that the Qt GUI thread remains responsive.
PYBIND11_MODULE(protein_physics,m){
    m.doc()="High-perf implicit-solvent engine (Verlet NL·HCT-GB·SASA·OpenMP)";
    // module_local(): protein_physics_cuda defines its own, separately-compiled
    // "Particle"/"PhysicsEngine" classes with the same names. On Linux/macOS
    // these are distinct C++ types (separate RTTI per shared object) and
    // pybind11 would register them independently regardless; on Windows, MSVC's
    // RTTI compares type_info by decorated NAME across DLLs, so pybind11 sees
    // them as the SAME C++ type and refuses the second module's registration
    // with "generic_type: type is already registered!" the moment both modules
    // are imported in one process (gui_main.py always imports protein_physics,
    // then conditionally protein_physics_cuda to probe for a GPU at startup).
    // module_local() keeps each module's registration in its own per-module
    // table instead of the shared global one, which is exactly what's needed
    // here since Particle/PhysicsEngine instances are never passed between
    // the two engines (gui_main.py always uses one engine module consistently
    // for a whole session, never mixing).
    py::class_<Particle>(m,"Particle", py::module_local())
        .def(py::init<double,double,double,double,double,double,bool>(),
             py::arg("x"),py::arg("y"),py::arg("z"),py::arg("charge"),
             py::arg("radius")=1.9,py::arg("epsilon")=0.1,py::arg("is_water")=false)
        .def_readwrite("x",&Particle::x).def_readwrite("y",&Particle::y)
        .def_readwrite("z",&Particle::z).def_readwrite("charge",&Particle::charge)
        .def_readwrite("radius",&Particle::radius).def_readwrite("epsilon",&Particle::epsilon)
        .def_readwrite("is_water",&Particle::is_water);
    py::enum_<BondKind>(m,"BondKind")
        .value("BACKBONE_PHI", BondKind::BACKBONE_PHI)
        .value("BACKBONE_PSI", BondKind::BACKBONE_PSI)
        .value("SIDECHAIN",    BondKind::SIDECHAIN)
        .value("FIXED",        BondKind::FIXED);
    py::class_<RotBond>(m,"RotBond")
        .def_readonly("i",    &RotBond::i)
        .def_readonly("j",    &RotBond::j)
        .def_readonly("kind", &RotBond::kind);
    py::class_<BondTopology>(m,"BondTopology")
        .def(py::init<>())
        .def("build",    &BondTopology::build,
             py::arg("resnames"), py::arg("atomnames"), py::arg("res_idx"))
        .def("bonded",   &BondTopology::bonded, py::arg("i"), py::arg("j"))
        .def("j_side",   &BondTopology::j_side, py::arg("bond_i"), py::arg("bond_j"))
        .def("is_excluded", &BondTopology::is_excluded, py::arg("i"), py::arg("j"))
        .def_readonly("adj",            &BondTopology::adj)
        .def_readonly("bonds",          &BondTopology::bonds)
        .def_readonly("rot_bonds",      &BondTopology::rot_bonds)
        .def_readonly("rot_bond_sides", &BondTopology::rot_bond_sides)
        .def_readonly("excl",             &BondTopology::excl)
        .def_readonly("rot_bond_scale",   &BondTopology::rot_bond_scale)
        .def_readonly("disulfide_pairs",  &BondTopology::disulfide_pairs)
        .def_readonly("concerted_pairs",  &BondTopology::concerted_pairs)
        .def_readonly("concerted_sidechain_pairs", &BondTopology::concerted_sidechain_pairs)
        .def("identify_concerted_sidechain_pairs",
             &BondTopology::identify_concerted_sidechain_pairs,
             py::arg("init_coords"), py::arg("cutoff") = 6.0)
        .def("add_disulfide", &BondTopology::add_disulfide,
             py::arg("atom_i"), py::arg("atom_j"))
        // ── Flat/CSR exports (rb_*, dih_*) ──────────────────────────────────
        //
        // rot_bonds/dihedrals hold custom C++ structs (RotBond/DihRecord) that
        // are bound as pybind11 types only in THIS module. A second, separately
        // compiled pybind11 module (protein_physics_cuda) cannot cast a Python
        // BondTopology's .rot_bonds/.dihedrals into its own local struct types
        // without re-registering py::class_<RotBond>/py::class_<DihRecord> —
        // which pybind11 forbids for the same C++ type across modules and is
        // meaningless here anyway (RotBond/DihRecord are distinct C++ types per
        // translation unit). Exporting the same data as plain vectors of
        // int/double lets any module reconstruct it via ordinary STL casts.
        .def_property_readonly("rb_atom_i",
            [](const BondTopology& t){
                std::vector<int> v; v.reserve(t.rot_bonds.size());
                for (const auto& rb : t.rot_bonds) v.push_back(rb.i);
                return v;
            })
        .def_property_readonly("rb_atom_j",
            [](const BondTopology& t){
                std::vector<int> v; v.reserve(t.rot_bonds.size());
                for (const auto& rb : t.rot_bonds) v.push_back(rb.j);
                return v;
            })
        .def_property_readonly("rb_kind",
            [](const BondTopology& t){
                std::vector<int> v; v.reserve(t.rot_bonds.size());
                for (const auto& rb : t.rot_bonds) v.push_back(static_cast<int>(rb.kind));
                return v;
            })
        .def_property_readonly("dih_a",
            [](const BondTopology& t){
                std::vector<int> v; v.reserve(t.dihedrals.size());
                for (const auto& d : t.dihedrals) v.push_back(d.a);
                return v;
            })
        .def_property_readonly("dih_b",
            [](const BondTopology& t){
                std::vector<int> v; v.reserve(t.dihedrals.size());
                for (const auto& d : t.dihedrals) v.push_back(d.b);
                return v;
            })
        .def_property_readonly("dih_c",
            [](const BondTopology& t){
                std::vector<int> v; v.reserve(t.dihedrals.size());
                for (const auto& d : t.dihedrals) v.push_back(d.c);
                return v;
            })
        .def_property_readonly("dih_d",
            [](const BondTopology& t){
                std::vector<int> v; v.reserve(t.dihedrals.size());
                for (const auto& d : t.dihedrals) v.push_back(d.d);
                return v;
            })
        // CSR layout: dih_term_offsets has one entry per dihedral plus a final
        // sentinel (length = num_dihedrals + 1); terms for dihedral k live in
        // [offsets[k], offsets[k+1]) of the flat v2/n/gamma arrays.
        .def_property_readonly("dih_term_offsets",
            [](const BondTopology& t){
                std::vector<int> v; v.reserve(t.dihedrals.size() + 1);
                int off = 0; v.push_back(0);
                for (const auto& d : t.dihedrals) { off += (int)d.terms.size(); v.push_back(off); }
                return v;
            })
        .def_property_readonly("dih_term_v2",
            [](const BondTopology& t){
                std::vector<double> v;
                for (const auto& d : t.dihedrals)
                    for (const auto& term : d.terms) v.push_back(term.V2);
                return v;
            })
        .def_property_readonly("dih_term_n",
            [](const BondTopology& t){
                std::vector<int> v;
                for (const auto& d : t.dihedrals)
                    for (const auto& term : d.terms) v.push_back(term.n);
                return v;
            })
        .def_property_readonly("dih_term_gamma",
            [](const BondTopology& t){
                std::vector<double> v;
                for (const auto& d : t.dihedrals)
                    for (const auto& term : d.terms) v.push_back(term.gamma);
                return v;
            })
        .def_property_readonly("num_dihedrals",
            [](const BondTopology& t){ return (int)t.dihedrals.size(); })
        .def_property_readonly("num_atoms",
            [](const BondTopology& t){ return t.N; })
        .def_property_readonly("num_bonds",
            [](const BondTopology& t){ return (int)t.bonds.size(); })
        .def_property_readonly("num_rot_bonds",
            [](const BondTopology& t){ return (int)t.rot_bonds.size(); })
        .def_property_readonly("num_concerted_pairs",
            [](const BondTopology& t){ return (int)t.concerted_pairs.size(); })
        .def_property_readonly("num_disulfide_pairs",
            [](const BondTopology& t){ return (int)t.disulfide_pairs.size(); });
    py::class_<PhysicsEngine>(m,"PhysicsEngine", py::module_local())
        .def(py::init<>())
        .def("calculate_potential",
             [](PhysicsEngine& self,
                const std::vector<Particle>& particles,
                py::object topo_obj) -> double {
                 const BondTopology* topo = nullptr;
                 if (!topo_obj.is_none())
                     topo = py::cast<BondTopology*>(topo_obj);
                 py::gil_scoped_release release;
                 return self.calculate_potential(particles, topo);
             },
             py::arg("particles"),
             py::arg("topology") = py::none())
        .def("generate_ensemble",&PhysicsEngine::generate_ensemble,
             py::arg("initial_state"),py::arg("topology"),
             py::arg("n_candidates"),py::arg("steps_per_cand"),
             py::arg("temperature")=0.6,py::arg("max_angle")=0.12,
             py::call_guard<py::gil_scoped_release>())
        .def("lowest_energy_structure",&PhysicsEngine::lowest_energy_structure,
             py::arg("ensemble"),py::call_guard<py::gil_scoped_release>())
        .def("run_landscape_trajectory",&PhysicsEngine::run_landscape_trajectory,
             py::arg("initial_state"),py::arg("topology"),
             py::arg("n_snapshots"),py::arg("steps_per_snapshot"),
             py::arg("temperature")=0.6,py::arg("max_angle")=0.12,
             py::call_guard<py::gil_scoped_release>())
        .def("run_landscape_segment",&PhysicsEngine::run_landscape_segment,
             py::arg("initial_state"),py::arg("topology"),
             py::arg("n_snapshots"),py::arg("steps_per_snapshot"),
             py::arg("temperature")=0.6,py::arg("max_angle")=0.12,
             py::call_guard<py::gil_scoped_release>())
        .def("run_mc_diagnostic",
             [](PhysicsEngine& self, const std::vector<Particle>& init,
                const BondTopology& topo, int n_steps, double T, double max_angle) {
                 PhysicsEngine::McDiagnosticResult r;
                 {
                     py::gil_scoped_release release;
                     r = self.run_mc_diagnostic(init, topo, n_steps, T, max_angle);
                 }  // GIL re-acquired here, before building the Python-visible tuple below
                 return std::make_tuple(r.move_type, r.proposed_delta, r.accepted,
                                         r.d_hardcore, r.d_nonbonded_soft, r.d_sasa,
                                         r.d_dih, r.d_ss);
             },
             py::arg("initial_state"),py::arg("topology"),py::arg("n_steps"),
             py::arg("temperature")=0.6,py::arg("max_angle")=0.12)
        .def("num_threads",&PhysicsEngine::num_threads);
}