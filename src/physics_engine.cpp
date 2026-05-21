/*
 * physics_engine.cpp
 *
 * CPU implicit-solvent protein physics engine, exposed to Python via pybind11.
 *
 * Energy model (all in kcal/mol, distances in Angstroms):
 *   1. Generalized Born (GB/HCT)  — polar solvation / electrostatics in water
 *   2. Debye–Hückel               — screened long-range Coulomb in ionic solution
 *   3. Lennard-Jones 12-6         — van der Waals / steric packing
 *   4. SASA nonpolar term         — hydrophobic burial penalty (surface-area model)
 *
 * Parallelism:
 *   - Per-atom neighbor list (Verlet list, NL_CUTOFF + NL_SKIN shell)
 *   - OpenMP parallel for over pair loops when available
 *   - Monte Carlo ensemble generation with per-thread RNG streams
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
// COULOMB   : conversion factor  e²/(4πε₀) in kcal·Å/(mol·e²)
// EPS_WATER : dielectric constant of bulk water (78.5)
// EPS_PROT  : interior dielectric constant of the protein (1.0)
// KAPPA     : Debye screening length inverse (Å⁻¹); ~0.1 Å⁻¹ at 150 mM NaCl
// GAMMA_SA  : surface tension coefficient (kcal/mol·Å²) for SASA nonpolar term
// BETA_SA   : additive constant in SASA energy (kcal/mol)
// PROBE_R   : solvent probe radius (1.4 Å = water molecule)
// NL_CUTOFF : hard pair-energy cutoff (Å); pairs beyond this are ignored
// NL_SKIN   : extra shell around cutoff kept in neighbor list for drift tolerance
// HARD_SCALE: energy scale for hard-core repulsion when atoms overlap (r < 0.85σ)
// GB_COEF   : prefactor for the GB Born term = -½(1/ε_prot - 1/ε_water)·C
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
//   x,y,z    — Cartesian coordinates (Å)
//   charge   — partial charge (e), taken from AMBER/CHARGE lookup in gui_main.py
//   radius   — van der Waals radius (Å); also used as intrinsic Born radius
//   epsilon  — LJ well-depth (kcal/mol)
//   is_water — true for explicit water molecules (currently unused; reserved)
struct Particle {
    double x,y,z,charge,radius,epsilon;
    bool is_water;
    Particle(double x,double y,double z,double charge,
             double radius=1.9,double epsilon=0.1,bool is_water=false)
        :x(x),y(y),z(z),charge(charge),radius(radius),epsilon(epsilon),is_water(is_water){}
};

// NeighborList: Verlet pair list — stores atom-j indices reachable from atom-i
// within (NL_CUTOFF + NL_SKIN). build() is O(N²) and called only when atoms
// have drifted more than NL_SKIN/2 from their positions at the last rebuild.
// This amortises the O(N²) rebuild cost over many MC steps (the "skin trick").
//   nb[i]  — sorted list of j > i that are within NL_RCUT2 of atom i
//   ref[i] — position of atom i at the last rebuild (drift-check baseline)
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
    // Computes the contribution of atom j (radius rj) to the Born desolvation
    // sum of atom i (radius ri) at separation r/r2.  Formula from Hawkins,
    // Cramer & Truhlar (1995).  Returns 0 when j is fully buried within i.
    static inline double hct(double r,double r2,double ri,double rj) noexcept {
        double L=std::max(std::abs(r-rj),ri),U=r+rj;
        if(ri>=U) return 0.0;
        return 1.0/L-1.0/U+(r2-rj*rj+ri*ri)/(2.0*r*ri*ri)*std::log(L/U)*0.5/r;
    }

    // Compute effective Born radii for all atoms.
    // Each radius a[i] = 1/(1/r_i - 0.5·Σ_j hct(i,j)), clamped to ≥ 0.5 Å.
    // Larger a[i] means atom i is more buried (better shielded from solvent).
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
    // Approximates the solvent-accessible surface area of each atom using a
    // spherical-cap subtraction model, then scales by GAMMA_SA.
    // This captures the hydrophobic burial penalty without an expensive numerical
    // surface calculation.
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
    // Combines:
    //   edh  — Debye–Hückel screened electrostatics
    //   egb  — Generalized Born electrostatic solvation (GB_COEF·q_i·q_j/f_GB)
    //   elj  — Lennard-Jones 12-6 van der Waals
    // Hard-core repulsion replaces all three when atoms overlap (r < 0.85·σ).
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
    // Returns a vector of ncand final conformations.
    //
    // Parameters:
    //   init   — starting conformation (same for all chains)
    //   ncand  — number of independent trajectories (candidates)
    //   steps  — MC steps per trajectory
    //   T      — effective temperature in kcal/mol (0.6 ≈ 300 K)
    //   maxd   — max per-atom displacement per step (Å)
    //
    // Algorithm (Metropolis Monte Carlo):
    //   Each step: pick a random atom, displace it uniformly in [-maxd,+maxd]³,
    //   compute ΔE from the local pair sum change (O(neighbors) not O(N²)),
    //   accept if ΔE<0 or with probability exp(-ΔE/T), else revert.
    //   NL is rebuilt (O(N²)) when any atom has drifted >NL_SKIN/2.
    //
    // Parallelism: OpenMP runs chains in parallel with thread-local RNGs derived
    // from a hardware random seed XOR'd with the thread ID to avoid correlation.
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
                // Metropolis acceptance criterion
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