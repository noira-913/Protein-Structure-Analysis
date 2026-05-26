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
 * ── 몬테 카를로 샘플링 (Monte Carlo Sampling) ─────────────────────────────────
 *
 * Conformational space is explored via the Metropolis MC algorithm:
 * 구조 공간은 Metropolis MC 알고리즘으로 탐색한다:
 *
 *   • 무작위로 원자 하나를 선택해 [-maxd, +maxd]³ 범위에서 이동 제안
 *     (Select one atom at random; propose a displacement in [-maxd, +maxd]³)
 *   • ΔE < 0 이면 무조건 수용 (always accept if energy decreases)
 *   • ΔE ≥ 0 이면 볼츠만 확률 exp(-ΔE/T)로 수용 (accept with Boltzmann probability)
 *   • 거부되면 원자를 원래 위치로 복원 (revert atom position if rejected)
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
 * Exposed classes: Particle, PhysicsEngine
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
// HARD_SCALE: energy scale for hard-core repulsion when atoms overlap (r < 0.85σ)
//             원자가 겹칠 때(r < 0.85σ) 적용되는 강한 척력 에너지 스케일
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
    // O(N²) full rebuild: refresh pair lists and snapshot reference positions.
    void build(const std::vector<Particle>& p){
        N=p.size();nb.assign(N,{});ref.resize(N);
        for(size_t i=0;i<N;++i){
            ref[i]={p[i].x,p[i].y,p[i].z};
            for(size_t j=i+1;j<N;++j){
                double dx=p[i].x-p[j].x,dy=p[i].y-p[j].y,dz=p[i].z-p[j].z;
                if(dx*dx+dy*dy+dz*dz<NL_RCUT2) nb[i].push_back(j);
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
    // Hard-core repulsion replaces all three when atoms overlap (r < 0.85·σ).
    // r < 0.85σ 원자 겹침 시 HARD_SCALE·(σ/r)¹²로 모든 항을 대체 (충돌 방지).
    static inline double pair_e(const Particle& pi,const Particle& pj,double ai,double aj) noexcept {
        double dx=pi.x-pj.x,dy=pi.y-pj.y,dz=pi.z-pj.z;
        double r2=dx*dx+dy*dy+dz*dz,r=std::sqrt(r2),sig=pi.radius+pj.radius;
        if(r<sig*0.85) return HARD_SCALE*std::pow(sig/r,12.0);
        double qp=pi.charge*pj.charge;
        double edh=(COULOMB*qp)/(EPS_WATER*r)*std::exp(-KAPPA*r);
        double fgb=std::sqrt(r2+ai*aj*std::exp(-r2/(4.0*ai*aj)));
        double egb=GB_COEF*qp/fgb;
        double eps=std::sqrt(pi.epsilon*pj.epsilon),s6=std::pow(sig/r,6);
        double elj=4.0*eps*(s6*s6-s6);
        return edh+egb+elj;
    }

    // Sum of SASA + all pair energies within NL_CUTOFF.
    // OpenMP parallel-for over atom i with dynamic scheduling and energy reduction.
    static double total_e(const std::vector<Particle>& p,const NeighborList& nl,const std::vector<double>& a){
        double E=sasa_nonpolar(p,nl);
        size_t N=p.size();
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic,8) reduction(+:E)
#endif
        for(size_t i=0;i<N;++i)
            for(size_t j:nl.nb[i]){
                double dx=p[i].x-p[j].x,dy=p[i].y-p[j].y,dz=p[i].z-p[j].z;
                if(dx*dx+dy*dy+dz*dz>PAIR_CUT2) continue;
                E+=pair_e(p[i],p[j],a[i],a[j]);
            }
        return E;
    }
public:
    PhysicsEngine():gen(std::random_device{}()){}

    // calculate_potential: full GB/DH/LJ/SASA energy for an arbitrary particle list.
    // Rebuilds the neighbor list from scratch on every call — use only for final
    // evaluation, not inside the inner MC loop.
    double calculate_potential(const std::vector<Particle>& particles){
        if(particles.empty()) return 0.0;
        NeighborList nl;nl.build(particles);
        auto a=born_radii(particles,nl);
        return total_e(particles,nl,a);
    }

    // generate_ensemble: run 'ncand' independent MC trajectories of 'steps' steps each.
    // Returns a vector of ncand final conformations (the MC "ensemble").
    //
    // ncand개의 독립 MC 궤적을 각각 steps 스텝 실행하고,
    // 최종 배열(conformation) ncand개를 벡터로 반환한다.
    //
    // Parameters / 매개변수:
    //   init   — starting conformation (same for all chains)
    //            모든 체인의 공통 초기 배열
    //   ncand  — number of independent trajectories (candidates)
    //            독립 MC 궤적 수 (후보 배열 수)
    //   steps  — MC steps per trajectory / 궤적당 MC 스텝 수
    //   T      — effective temperature in kcal/mol (0.6 ≈ 300 K)
    //            유효 온도 (kcal/mol) — 열 요동의 크기를 조절
    //   maxd   — max per-atom displacement per step (Å)
    //            스텝당 최대 원자 이동 거리 (Å)
    //
    // Algorithm (Metropolis Monte Carlo) / 알고리즘:
    //   ① 무작위 원자 idx 선택 → [-maxd, +maxd]³ 이동 제안
    //      Pick random atom idx; propose displacement in [-maxd,+maxd]³
    //   ② 이동 전 idx의 이웃 쌍 에너지 합산 (O(이웃 수), not O(N²))
    //      Compute old pair energy sum for idx (local, O(neighbors))
    //   ③ Born 반경 업데이트 → 이동 후 새 쌍 에너지 합산
    //      Update Born radius for idx; compute new pair energy sum
    //   ④ ΔE = new - old: 음수면 무조건 수용,
    //      양수면 exp(-ΔE/T) 확률로 수용, 나머지는 복원
    //      Accept if ΔE<0, else accept with prob exp(-ΔE/T), else revert
    //   ⑤ drift > NL_SKIN/2 이면 이웃 목록 전체 재구축 (O(N²))
    //      Rebuild NL (O(N²)) when any atom drifts >NL_SKIN/2
    //
    // 수렴 이론: Metropolis 기준은 세밀 균형(detailed balance) 조건을 만족하므로
    // 충분히 긴 체인은 볼츠만 분포 p ∝ exp(-E/T)로 수렴 (에르고딕 가정 하).
    // Convergence: Metropolis satisfies detailed balance → Boltzmann distribution.
    //
    // Parallelism / 병렬화: OpenMP가 체인들을 병렬로 실행.
    // 각 스레드는 하드웨어 난수 XOR 스레드 ID로 독립 RNG를 초기화 → 상관 방지.
    // Each OpenMP thread gets a unique seed (hw_random XOR thread_id) to prevent
    // correlated draws across chains.
    std::vector<std::vector<Particle>> generate_ensemble(
        const std::vector<Particle>& init,int ncand,int steps,double T=0.6,double maxd=0.3)
    {
        if(init.empty()) throw std::invalid_argument("initial_state empty");
        if(ncand<=0||steps<=0) throw std::invalid_argument("ncand/steps must be positive");
        size_t N=init.size();
        std::vector<std::vector<Particle>> ens(ncand);
        auto chain=[&](int c,std::mt19937& rng){
            std::vector<Particle> st=init;
            NeighborList nl;nl.build(st);
            auto a=born_radii(st,nl);
            double curE=total_e(st,nl,a);
            std::uniform_real_distribution<double> disp(-maxd,maxd);
            std::uniform_real_distribution<double> uni(0.0,1.0);
            std::uniform_int_distribution<size_t> pick(0,N-1);
            for(int s=0;s<steps;++s){
                // Rebuild NL and recompute Born radii when drift exceeds skin/2
                if(nl.needs_rebuild(st)){nl.build(st);a=born_radii(st,nl);curE=total_e(st,nl,a);}
                size_t idx=pick(rng);
                double ox=st[idx].x,oy=st[idx].y,oz=st[idx].z;
                // Propose displacement
                st[idx].x+=disp(rng);st[idx].y+=disp(rng);st[idx].z+=disp(rng);
                // Collect old pair energy for atom idx (at its original position)
                double old_p=0.0;
                for(size_t j:nl.nb[idx]){
                    double dx=ox-st[j].x,dy=oy-st[j].y,dz=oz-st[j].z;
                    if(dx*dx+dy*dy+dz*dz>PAIR_CUT2) continue;
                    Particle tmp=st[idx];tmp.x=ox;tmp.y=oy;tmp.z=oz;
                    old_p+=pair_e(tmp,st[j],a[idx],a[j]);
                }
                // Update Born radius for moved atom and accumulate new pair energy
                double old_a=a[idx];
                update_born(idx,st,nl,a);
                double new_p=0.0;
                for(size_t j:nl.nb[idx]){
                    double dx=st[idx].x-st[j].x,dy=st[idx].y-st[j].y,dz=st[idx].z-st[j].z;
                    if(dx*dx+dy*dy+dz*dz>PAIR_CUT2) continue;
                    new_p+=pair_e(st[idx],st[j],a[idx],a[j]);
                }
                double dE=new_p-old_p;
                // Metropolis acceptance criterion (볼츠만 수용 기준):
                // ΔE<0 → 에너지 감소 → 항상 수용 (always accept)
                // ΔE≥0 → 확률 exp(-ΔE/T)로 수용 → 열 요동으로 에너지 장벽 극복 가능
                // 거부(rejected) → 원래 좌표와 Born 반경으로 복원
                if(dE<0.0||uni(rng)<std::exp(-dE/T)){curE+=dE;}
                else{st[idx].x=ox;st[idx].y=oy;st[idx].z=oz;a[idx]=old_a;}
            }
            ens[c]=std::move(st);
        };
#ifdef _OPENMP
        #pragma omp parallel
        {
            // Each OpenMP thread gets a unique seed to prevent correlated draws
            std::mt19937 lg(std::random_device{}()^(std::hash<int>{}(omp_get_thread_num())<<16));
            #pragma omp for schedule(dynamic)
            for(int c=0;c<ncand;++c) chain(c,lg);
        }
#else
        for(int c=0;c<ncand;++c) chain(c,gen);
#endif
        return ens;
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
    py::class_<Particle>(m,"Particle")
        .def(py::init<double,double,double,double,double,double,bool>(),
             py::arg("x"),py::arg("y"),py::arg("z"),py::arg("charge"),
             py::arg("radius")=1.9,py::arg("epsilon")=0.1,py::arg("is_water")=false)
        .def_readwrite("x",&Particle::x).def_readwrite("y",&Particle::y)
        .def_readwrite("z",&Particle::z).def_readwrite("charge",&Particle::charge)
        .def_readwrite("radius",&Particle::radius).def_readwrite("epsilon",&Particle::epsilon)
        .def_readwrite("is_water",&Particle::is_water);
    py::class_<PhysicsEngine>(m,"PhysicsEngine")
        .def(py::init<>())
        .def("calculate_potential",&PhysicsEngine::calculate_potential,
             py::arg("particles"),py::call_guard<py::gil_scoped_release>())
        .def("generate_ensemble",&PhysicsEngine::generate_ensemble,
             py::arg("initial_state"),py::arg("n_candidates"),py::arg("steps_per_cand"),
             py::arg("temperature")=0.6,py::arg("max_disp")=0.3,
             py::call_guard<py::gil_scoped_release>())
        .def("lowest_energy_structure",&PhysicsEngine::lowest_energy_structure,
             py::arg("ensemble"),py::call_guard<py::gil_scoped_release>())
        .def("num_threads",&PhysicsEngine::num_threads);
}