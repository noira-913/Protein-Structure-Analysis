/*
 * physics_engine_cuda.cu
 *
 * CUDA C++ port of physics_engine.cpp.
 * Exposes a pybind11 module named `protein_physics_cuda` with the same
 * Particle / BondTopology / PhysicsEngine Python-facing API as the CPU
 * version (protein_physics), plus PhysicsEngine::device_name() (static).
 *
 * ── P4.2 status ──────────────────────────────────────────────────────────
 * This engine now runs the SAME torsion-angle Metropolis MC as the CPU
 * engine (Rodrigues rotation of one rotatable bond's j-side, full bond
 * topology: 1-2/1-3 exclusions, dihedral (Ramachandran) energy, disulfide
 * restraints, lever-arm scaling, crankshaft moves, adaptive proposal width)
 * instead of the old physically-invalid single-atom Cartesian move. The
 * GPU accelerates the per-step cross-pair nonbonded energy sum (the
 * dominant O(N·neighbors) cost of a torsion move touching a large j-side,
 * e.g. a backbone bond near the N-terminus) via a resident-buffer kernel;
 * SASA and dihedral/disulfide terms stay on the CPU host exactly as in
 * physics_engine.cpp (SASA's per-atom accumulation is a sequential,
 * order-dependent fold — not embarrassingly parallel — and dihedral/SS
 * sums are O(rotatable bonds), too small to be worth a kernel launch).
 *
 * generate_ensemble()/run_landscape_trajectory() keep one persistent set of
 * GPU buffers per MC chain for its entire duration (atom parameters
 * uploaded once; neighbor-pair/exclusion buffers rebuilt only when the
 * Verlet skin is exceeded; positions + Born radii re-uploaded once per
 * step). run_landscape_trajectory() additionally lets LandscapeWorker
 * (Python) advance the whole 120x80-step Markov chain with a single call
 * into C++, instead of looping 120 times and re-marshalling the full
 * particle array across the Python<->C++ boundary each iteration.
 *
 * Build requirements:
 *   nvcc  -arch=sm_XX  --compiler-options -O2
 *         -I<pybind11/include>  -I<Python/include>
 *         physics_engine_cuda.cu  -o protein_physics_cuda.<ext>
 *         -lpython3X
 *   (optionally add -Xcompiler /openmp  or  -Xcompiler -fopenmp)
 */

/* ── standard headers ────────────────────────────────────────────────── */
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <string>
#include <cstdint>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

/* ═══════════════════════════════════════════════════════════════════════
 * 1.  PHYSICS CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── double-precision CPU-side constants ─────────────────────────────── */
/* Note: pair energy / Born radii are computed exclusively on the GPU in this
 * engine (see gpu_total_energy_topo / cross_pair_energy_kernel below), so
 * only the constants their CPU-side neighbors (NeighborList, SASA, disulfide
 * restraints) actually need live here. COULOMB/EPS_WATER/EPS_PROT/KAPPA/
 * HARD_SCALE/HARD_CUTOFF_FRAC/GB_COEF only fed a double-precision pair_e_cpu()
 * that existed for the old Cartesian-move engine; removed along with it. */
static constexpr double GAMMA_SA   = 0.00542;
static constexpr double BETA_SA    = 0.92;
static constexpr double PROBE_R    = 1.4;
static constexpr double NL_CUTOFF  = 12.0;
static constexpr double NL_SKIN    = 2.0;
static constexpr double NL_RCUT2   = (NL_CUTOFF + NL_SKIN) * (NL_CUTOFF + NL_SKIN);
static constexpr double HALF_SKIN2 = (NL_SKIN * 0.5) * (NL_SKIN * 0.5);
static constexpr double K_SS       = 600.0;   /* disulfide restraint force constant (kcal/mol/Ang^2) */
static constexpr double R0_SS      = 2.044;   /* equilibrium SG-SG distance (Ang) */

/* ── single-precision GPU-side constants (#define for device code) ───── */
#define COULOMB_F      332.0636f
#define EPS_WATER_F    78.5f
#define KAPPA_F        0.1257f
#define HARD_SCALE_F   1.0e4f
/* HARD_CUTOFF_FRAC_F: kept in sync with physics_engine.cpp's P1.6 fix (see
 * IMPROVEMENTS.md item #13). This GPU engine previously used the old,
 * miscalibrated 0.85 threshold — this file had never been touched by that
 * bugfix — which would have reproduced the same billions-of-kcal/mol
 * blowup on any real all-atom structure the moment this engine exercised
 * real (non-excluded) 1-4/H..H contacts down to r/sigma ~ 0.67. */
#define HARD_CUTOFF_FRAC_F 0.6f
/* HARD_CAP_F: ceiling (kcal/mol) on the hard-core term. Mirrors the CPU
 * engine's fix in physics_engine.cpp (see the HARD_CAP comment there) --
 * real deposited structures (e.g. PDB 1LYZ, a 1975 structure with a genuine
 * 1.36 Ang CB...NH1 contact) can contain pathological-but-real short
 * contacts that are not MC-proposal artifacts. Left uncapped, HARD_SCALE_F *
 * (sigma/r)^12 is unbounded as r->0 and a single such pair can swamp the
 * whole structure's energy by 5-6 orders of magnitude. */
#define HARD_CAP_F 5.0e3f
/* GB_COEF_F = -0.5 * (1/1 - 1/78.5) * 332.0636  */
#define GB_COEF_F      (-0.5f * (1.0f - 1.0f / 78.5f) * 332.0636f)
#define PAIR_CUT2_F    144.0f          /* 12^2 */
#define NL_RCUT2_F     196.0f          /* (12+2)^2 */

/* ═══════════════════════════════════════════════════════════════════════
 * 2.  CUDA ERROR HELPER
 * ═══════════════════════════════════════════════════════════════════════ */
static void cuda_check(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "CUDA error %s at %s:%d",
                      cudaGetErrorString(err), file, line);
        throw std::runtime_error(buf);
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

/* ═══════════════════════════════════════════════════════════════════════
 * 3.  RAII GPU BUFFER
 * ═══════════════════════════════════════════════════════════════════════ */
template<typename T>
struct GpuBuf {
    T*     ptr = nullptr;
    size_t n   = 0;

    explicit GpuBuf() = default;
    explicit GpuBuf(size_t count) { alloc(count); }
    ~GpuBuf() { free(); }

    /* Non-copyable, movable */
    GpuBuf(const GpuBuf&)            = delete;
    GpuBuf& operator=(const GpuBuf&) = delete;
    GpuBuf(GpuBuf&& o) noexcept : ptr(o.ptr), n(o.n) { o.ptr = nullptr; o.n = 0; }
    GpuBuf& operator=(GpuBuf&& o) noexcept {
        if (this != &o) { free(); ptr = o.ptr; n = o.n; o.ptr = nullptr; o.n = 0; }
        return *this;
    }

    void alloc(size_t count) {
        free();
        n = count;
        if (n) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
    }
    void free() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
        n = 0;
    }
    void upload(const T* host, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* host, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host, ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() {
        CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T)));
    }
};

/* ═══════════════════════════════════════════════════════════════════════
 * 4.  PARTICLE STRUCT
 * ═══════════════════════════════════════════════════════════════════════ */
struct Particle {
    double x, y, z, charge, radius, epsilon;
    bool   is_water;
    Particle(double x, double y, double z, double charge,
             double radius = 1.9, double epsilon = 0.1, bool is_water = false)
        : x(x), y(y), z(z), charge(charge),
          radius(radius), epsilon(epsilon), is_water(is_water) {}
};

/* ═══════════════════════════════════════════════════════════════════════
 * 5.  NEIGHBOR LIST  (extends CPU version with flat pair arrays)
 * ═══════════════════════════════════════════════════════════════════════ */
struct NeighborList {
    /* Per-atom adjacency list (CPU incremental MC) */
    std::vector<std::vector<size_t>> nb;
    /* Reference positions for drift check */
    std::vector<std::array<double, 3>> ref;
    size_t N = 0;

    /* Flat pair arrays for GPU kernels */
    std::vector<int> pi;   /* index of atom i in each pair */
    std::vector<int> pj;   /* index of atom j in each pair */

    void build(const std::vector<Particle>& p) {
        N = p.size();
        nb.assign(N, {});
        ref.resize(N);
        pi.clear();
        pj.clear();

        for (size_t i = 0; i < N; ++i) {
            ref[i] = {p[i].x, p[i].y, p[i].z};
            for (size_t j = i + 1; j < N; ++j) {
                double dx = p[i].x - p[j].x;
                double dy = p[i].y - p[j].y;
                double dz = p[i].z - p[j].z;
                if (dx*dx + dy*dy + dz*dz < NL_RCUT2) {
                    nb[i].push_back(j);
                    pi.push_back(static_cast<int>(i));
                    pj.push_back(static_cast<int>(j));
                }
            }
        }
    }

    bool needs_rebuild(const std::vector<Particle>& p) const {
        for (size_t i = 0; i < N; ++i) {
            double dx = p[i].x - ref[i][0];
            double dy = p[i].y - ref[i][1];
            double dz = p[i].z - ref[i][2];
            if (dx*dx + dy*dy + dz*dz > HALF_SKIN2) return true;
        }
        return false;
    }
};

/* ═══════════════════════════════════════════════════════════════════════
 * 6.  CPU HELPER FUNCTIONS  (renamed with _cpu suffix)
 *     Used for per-step incremental energy in the MC loop.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double d2_cpu(const Particle& a, const Particle& b) noexcept {
    double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

static inline double hct_cpu(double r, double r2, double ri, double rj) noexcept {
    double L = std::max(std::abs(r - rj), ri);
    double U = r + rj;
    if (ri >= U) return 0.0;
    return 1.0/L - 1.0/U
           + (r2 - rj*rj + ri*ri) / (2.0*r*ri*ri) * std::log(L/U) * 0.5 / r;
}

static void update_born_cpu(size_t idx,
                             const std::vector<Particle>& p,
                             const NeighborList& nl,
                             std::vector<double>& a)
{
    double ri = p[idx].radius, sum = 0.0;
    for (size_t j : nl.nb[idx]) {
        double r2 = d2_cpu(p[idx], p[j]);
        double r  = std::sqrt(r2);
        sum += hct_cpu(r, r2, ri, p[j].radius);
    }
    a[idx] = 1.0 / std::max(1.0 / ri - 0.5 * sum, 2.0);
}

static double sasa_nonpolar_cpu(const std::vector<Particle>& p,
                                 const NeighborList& nl)
{
    size_t N = p.size();
    double E = BETA_SA;
    for (size_t i = 0; i < N; ++i) {
        double ri = p[i].radius + PROBE_R;
        double sa = 4.0 * M_PI * ri * ri;
        for (size_t j : nl.nb[i]) {
            double rj = p[j].radius + PROBE_R;
            double dx = p[i].x - p[j].x;
            double dy = p[i].y - p[j].y;
            double dz = p[i].z - p[j].z;
            double r  = std::sqrt(dx*dx + dy*dy + dz*dz);
            double dc = ri + rj;
            if (r >= dc) continue;
            double h = (dc - r) / (2.0 * ri);
            sa -= std::min(sa * 0.85, 2.0 * M_PI * ri * ri * h);
        }
        E += GAMMA_SA * std::max(0.0, sa);
    }
    return E;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 7.  BOND TOPOLOGY  (host-side; mirrors BondTopology in physics_engine.cpp)
 *
 * BondTopology itself is a pybind11 type registered ONLY in the CPU module
 * (protein_physics) — gui_main.py always builds it via
 * `protein_physics.BondTopology()` even when the GPU engine is selected.
 * A second, separately compiled pybind11 module cannot cast that Python
 * object into a locally-defined C++ struct without re-registering
 * py::class_<BondTopology> for the identical C++ type, which pybind11
 * forbids (and which would be the wrong C++ type here anyway, since this
 * translation unit never includes physics_engine.cpp's definitions).
 *
 * Instead, physics_engine.cpp exports the pieces this engine needs as
 * plain vector<int>/vector<double> properties (rb_atom_i/j/kind, dih_*
 * CSR arrays, disulfide_pairs, concerted_pairs, plus the already-existing
 * rot_bond_sides/rot_bond_scale/excl) and extract_topology() below
 * reconstructs an equivalent local HostTopology from those via ordinary
 * pybind11 STL casts — no shared type registration needed.
 * ═══════════════════════════════════════════════════════════════════════ */

enum : int { BK_BACKBONE_PHI = 0, BK_BACKBONE_PSI = 1, BK_SIDECHAIN = 2, BK_FIXED = 3 };

struct RotBond {
    int i, j;
    int kind;   /* one of the BK_* constants above */
};

struct DihTerm {
    double V2;
    int    n;
    double gamma;
};

struct DihRecord {
    int a, b, c, d;
    std::vector<DihTerm> terms;
};

struct HostTopology {
    std::vector<RotBond>            rot_bonds;
    std::vector<std::vector<int>>   rot_bond_sides;
    std::vector<double>             rot_bond_scale;
    std::vector<std::vector<int>>   excl;
    std::vector<DihRecord>          dihedrals;
    std::vector<std::pair<int,int>> disulfide_pairs;
    std::vector<std::pair<int,int>> concerted_pairs;

    /* Identical algorithm to BondTopology::is_excluded in physics_engine.cpp. */
    bool is_excluded(int i, int j) const noexcept {
        if (i < 0 || j < 0 || i >= (int)excl.size() || j >= (int)excl.size()) return false;
        if (i > j) std::swap(i, j);
        const auto& v = excl[i];
        return std::binary_search(v.begin(), v.end(), j);
    }
};

static HostTopology extract_topology(const py::object& topo_obj)
{
    HostTopology t;

    auto rb_i = topo_obj.attr("rb_atom_i").cast<std::vector<int>>();
    auto rb_j = topo_obj.attr("rb_atom_j").cast<std::vector<int>>();
    auto rb_k = topo_obj.attr("rb_kind").cast<std::vector<int>>();
    t.rot_bonds.resize(rb_i.size());
    for (size_t k = 0; k < rb_i.size(); ++k)
        t.rot_bonds[k] = RotBond{rb_i[k], rb_j[k], rb_k[k]};

    t.rot_bond_sides = topo_obj.attr("rot_bond_sides").cast<std::vector<std::vector<int>>>();
    t.rot_bond_scale = topo_obj.attr("rot_bond_scale").cast<std::vector<double>>();
    t.excl           = topo_obj.attr("excl").cast<std::vector<std::vector<int>>>();
    t.disulfide_pairs = topo_obj.attr("disulfide_pairs").cast<std::vector<std::pair<int,int>>>();
    t.concerted_pairs = topo_obj.attr("concerted_pairs").cast<std::vector<std::pair<int,int>>>();

    auto da   = topo_obj.attr("dih_a").cast<std::vector<int>>();
    auto db   = topo_obj.attr("dih_b").cast<std::vector<int>>();
    auto dc   = topo_obj.attr("dih_c").cast<std::vector<int>>();
    auto dd   = topo_obj.attr("dih_d").cast<std::vector<int>>();
    auto doff = topo_obj.attr("dih_term_offsets").cast<std::vector<int>>();
    auto dv2  = topo_obj.attr("dih_term_v2").cast<std::vector<double>>();
    auto dn   = topo_obj.attr("dih_term_n").cast<std::vector<int>>();
    auto dg   = topo_obj.attr("dih_term_gamma").cast<std::vector<double>>();
    t.dihedrals.resize(da.size());
    for (size_t k = 0; k < da.size(); ++k) {
        DihRecord& r = t.dihedrals[k];
        r.a = da[k]; r.b = db[k]; r.c = dc[k]; r.d = dd[k];
        for (int m = doff[k]; m < doff[k + 1]; ++m)
            r.terms.push_back({dv2[m], dn[m], dg[m]});
    }
    return t;
}

/* ── Rodrigues rotation (identical to physics_engine.cpp) ─────────────── */
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

static inline double dihedral_angle(const std::vector<Particle>& p,
                                     int a, int b, int c, int d) noexcept
{
    double b1x = p[a].x-p[b].x, b1y = p[a].y-p[b].y, b1z = p[a].z-p[b].z;
    double b2x = p[c].x-p[b].x, b2y = p[c].y-p[b].y, b2z = p[c].z-p[b].z;
    double b3x = p[d].x-p[c].x, b3y = p[d].y-p[c].y, b3z = p[d].z-p[c].z;
    double n1x = b1y*b2z-b1z*b2y, n1y = b1z*b2x-b1x*b2z, n1z = b1x*b2y-b1y*b2x;
    double n2x = b2y*b3z-b2z*b3y, n2y = b2z*b3x-b2x*b3z, n2z = b2x*b3y-b2y*b3x;
    double m1x = n1y*b2z-n1z*b2y, m1y = n1z*b2x-n1x*b2z, m1z = n1x*b2y-n1y*b2x;
    double x = n1x*n2x+n1y*n2y+n1z*n2z;
    double y = m1x*n2x+m1y*n2y+m1z*n2z;
    return std::atan2(y, x);
}

static double dihedral_e(const std::vector<Particle>& p,
                          const std::vector<DihRecord>& dihs) noexcept
{
    double E = 0.0;
    for (const auto& dr : dihs) {
        double phi = dihedral_angle(p, dr.a, dr.b, dr.c, dr.d);
        for (const auto& t : dr.terms)
            E += t.V2 * (1.0 + std::cos((double)t.n * phi - t.gamma));
    }
    return E;
}

static double dihedral_e_boundary(const std::vector<Particle>& p,
                                   const std::vector<DihRecord>& dihs,
                                   const std::vector<bool>& in_side) noexcept
{
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

static double ss_e(const std::vector<Particle>& p,
                    const std::vector<std::pair<int,int>>& ss) noexcept
{
    double E = 0.0;
    for (const auto& [i, j] : ss) {
        double dx=p[i].x-p[j].x, dy=p[i].y-p[j].y, dz=p[i].z-p[j].z;
        double dr = std::sqrt(dx*dx+dy*dy+dz*dz) - R0_SS;
        E += K_SS * dr * dr;
    }
    return E;
}

static double ss_e_side(const std::vector<Particle>& p,
                         const std::vector<std::pair<int,int>>& ss,
                         const std::vector<bool>& in_side) noexcept
{
    double E = 0.0;
    for (const auto& [i, j] : ss) {
        if (!in_side[i] && !in_side[j]) continue;
        double dx=p[i].x-p[j].x, dy=p[i].y-p[j].y, dz=p[i].z-p[j].z;
        double dr = std::sqrt(dx*dx+dy*dy+dz*dz) - R0_SS;
        E += K_SS * dr * dr;
    }
    return E;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 8.  CUDA DEVICE FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════════ */

__device__ __forceinline__
float hct_gpu(float r, float r2, float ri, float rj)
{
    float absval = r - rj;
    if (absval < 0.0f) absval = -absval;
    float L = (absval > ri) ? absval : ri;
    float U = r + rj;
    if (ri >= U) return 0.0f;
    float logLU = __logf(L / U);
    return 1.0f/L - 1.0f/U
           + (r2 - rj*rj + ri*ri) / (2.0f*r*ri*ri) * logLU * 0.5f / r;
}

/*
 * Hard-core term is a SMOOTH penalty added on top of edh+egb+elj when atoms
 * overlap (r < HARD_CUTOFF_FRAC_F*sig), not a branch that replaces them --
 * see the matching comment on pair_e() in physics_engine.cpp for the full
 * rationale (a genuine tertiary contact landing a fraction of a percent
 * inside the old hard threshold used to score a multi-million-kcal/mol
 * discontinuous jump instead of an ordinary bounded LJ repulsion). At
 * r == r_cut the penalty and its derivative are both exactly zero, so the
 * total is continuous and smooth across the boundary. The penalty is also
 * capped at HARD_CAP_F: the smooth formula alone is still unbounded as
 * r->0, which matters for contacts deep inside the cutoff (not just near
 * the boundary) -- see the HARD_CAP comment in physics_engine.cpp.
 */
__device__ __forceinline__
float pair_e_gpu(float xi, float yi, float zi, float qi, float ri, float epsi, float ai,
                 float xj, float yj, float zj, float qj, float rj, float epsj, float aj)
{
    float dx  = xi - xj;
    float dy  = yi - yj;
    float dz  = zi - zj;
    float r2  = dx*dx + dy*dy + dz*dz;
    float r   = sqrtf(r2);
    float sig = ri + rj;

    float qp  = qi * qj;
    float edh = (COULOMB_F * qp) / (EPS_WATER_F * r) * __expf(-KAPPA_F * r);

    float aiaj = ai * aj;
    float fgb  = sqrtf(r2 + aiaj * __expf(-r2 / (4.0f * aiaj)));
    float egb  = GB_COEF_F * qp / fgb;

    float eps = sqrtf(epsi * epsj);
    float sr  = sig / r;
    float s6  = sr * sr * sr * sr * sr * sr;
    float elj = 4.0f * eps * (s6*s6 - s6);

    float e = edh + egb + elj;

    float r_cut = sig * HARD_CUTOFF_FRAC_F;
    if (r < r_cut) {
        float ratio  = r_cut / r;
        float ratio6 = ratio * ratio * ratio * ratio * ratio * ratio;
        float x      = ratio6 * ratio6 - 1.0f;
        /* HARD_CAP_F: the smooth formula above is continuous at r_cut but still
         * unbounded as r->0 -- see the matching HARD_CAP comment on pair_e() in
         * physics_engine.cpp (a real 1LYZ contact at r/sigma=0.364 evaluates to
         * ~1.6e9 even under this smooth formula without a cap). */
        e += fminf(HARD_SCALE_F * x * x, HARD_CAP_F);
    }

    return e;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 9.  CUDA KERNELS
 * ═══════════════════════════════════════════════════════════════════════ */

__global__
void born_sum_kernel(int npairs,
                     const int*   d_pi,
                     const int*   d_pj,
                     const float* d_x,
                     const float* d_y,
                     const float* d_z,
                     const float* d_r,
                     float*       d_sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npairs) return;

    int   i  = d_pi[tid];
    int   j  = d_pj[tid];
    float dx = d_x[i] - d_x[j];
    float dy = d_y[i] - d_y[j];
    float dz = d_z[i] - d_z[j];
    float r2 = dx*dx + dy*dy + dz*dz;
    float r  = sqrtf(r2);
    float ri = d_r[i];
    float rj = d_r[j];

    float cij = hct_gpu(r, r2, ri, rj);
    float cji = hct_gpu(r, r2, rj, ri);

    atomicAdd(&d_sum[i], cij);
    atomicAdd(&d_sum[j], cji);
}

__global__
void born_finalize_kernel(int natoms,
                          const float* d_r,
                          const float* d_sum,
                          float*       d_a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;
    float inv = 1.0f / d_r[i] - 0.5f * d_sum[i];
    if (inv < 2.0f) inv = 2.0f;
    d_a[i] = 1.0f / inv;
}

/*
 * pair_energy_kernel
 * One thread per (i,j) pair from the neighbor list. Skips pairs beyond
 * PAIR_CUT2_F and pairs flagged excluded (1-2/1-3 bonded, per topology).
 * d_excluded is never null; callers upload an all-zero mask when no
 * topology applies (calculate_potential with topology=None).
 */
__global__
void pair_energy_kernel(int npairs,
                        const int*     d_pi,
                        const int*     d_pj,
                        const uint8_t* d_excluded,
                        const float*   d_x,
                        const float*   d_y,
                        const float*   d_z,
                        const float*   d_q,
                        const float*   d_r,
                        const float*   d_eps,
                        const float*   d_a,
                        float*         d_etot)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npairs) return;
    if (d_excluded[tid]) return;

    int   i  = d_pi[tid];
    int   j  = d_pj[tid];
    float dx = d_x[i] - d_x[j];
    float dy = d_y[i] - d_y[j];
    float dz = d_z[i] - d_z[j];
    float r2 = dx*dx + dy*dy + dz*dz;

    if (r2 > PAIR_CUT2_F) return;

    float e = pair_e_gpu(d_x[i],  d_y[i],  d_z[i],  d_q[i],  d_r[i],  d_eps[i], d_a[i],
                         d_x[j],  d_y[j],  d_z[j],  d_q[j],  d_r[j],  d_eps[j], d_a[j]);
    atomicAdd(d_etot, e);
}

/*
 * cross_pair_energy_kernel
 * Same as pair_energy_kernel, but only sums pairs that straddle the
 * i_side/j_side boundary of the atom set that just rotated (d_inside[i] !=
 * d_inside[j]) — mirrors physics_engine.cpp's cross_e() lambda exactly,
 * just evaluated in parallel over the whole pair list on the GPU instead
 * of a serial CPU loop. This is the dominant per-MC-step cost for a
 * torsion move whose j-side is large (e.g. a backbone bond near the
 * N-terminus rotates roughly half the protein).
 */
__global__
void cross_pair_energy_kernel(int npairs,
                              const int*     d_pi,
                              const int*     d_pj,
                              const uint8_t* d_excluded,
                              const uint8_t* d_inside,
                              const float*   d_x,
                              const float*   d_y,
                              const float*   d_z,
                              const float*   d_q,
                              const float*   d_r,
                              const float*   d_eps,
                              const float*   d_a,
                              float*         d_etot)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npairs) return;
    if (d_excluded[tid]) return;

    int i = d_pi[tid];
    int j = d_pj[tid];
    if (d_inside[i] == d_inside[j]) return;

    float dx = d_x[i] - d_x[j];
    float dy = d_y[i] - d_y[j];
    float dz = d_z[i] - d_z[j];
    float r2 = dx*dx + dy*dy + dz*dz;
    if (r2 > PAIR_CUT2_F) return;

    float e = pair_e_gpu(d_x[i],  d_y[i],  d_z[i],  d_q[i],  d_r[i],  d_eps[i], d_a[i],
                         d_x[j],  d_y[j],  d_z[j],  d_q[j],  d_r[j],  d_eps[j], d_a[j]);
    atomicAdd(d_etot, e);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 10. HOST FUNCTION: gpu_total_energy_topo
 *     Orchestrates Born-radius + pairwise-energy kernels for a full,
 *     from-scratch energy evaluation (used at NL rebuild and for
 *     calculate_potential). Exclusion-aware; optionally returns the
 *     computed Born radii (as double) so the caller can adopt them as its
 *     incremental per-step state.
 * ═══════════════════════════════════════════════════════════════════════ */
static double gpu_total_energy_topo(const std::vector<Particle>& p,
                                     const NeighborList&           nl,
                                     const std::vector<uint8_t>&   excluded,
                                     std::vector<double>*          a_out)
{
    const int N      = static_cast<int>(p.size());
    const int npairs = static_cast<int>(nl.pi.size());

    std::vector<float> hx(N), hy(N), hz(N), hq(N), hr(N), heps(N);
    for (int i = 0; i < N; ++i) {
        hx[i]   = static_cast<float>(p[i].x);
        hy[i]   = static_cast<float>(p[i].y);
        hz[i]   = static_cast<float>(p[i].z);
        hq[i]   = static_cast<float>(p[i].charge);
        hr[i]   = static_cast<float>(p[i].radius);
        heps[i] = static_cast<float>(p[i].epsilon);
    }

    GpuBuf<float> d_x(N),    d_y(N),    d_z(N);
    GpuBuf<float> d_q(N),    d_r(N),    d_eps_buf(N);
    GpuBuf<float> d_sum(N),  d_a(N);
    GpuBuf<float> d_etot(1);
    GpuBuf<int>     d_pi_buf(npairs > 0 ? npairs : 1);
    GpuBuf<int>     d_pj_buf(npairs > 0 ? npairs : 1);
    GpuBuf<uint8_t> d_excl_buf(npairs > 0 ? npairs : 1);

    d_x.upload(hx.data(),   N);
    d_y.upload(hy.data(),   N);
    d_z.upload(hz.data(),   N);
    d_q.upload(hq.data(),   N);
    d_r.upload(hr.data(),   N);
    d_eps_buf.upload(heps.data(), N);

    if (npairs > 0) {
        d_pi_buf.upload(nl.pi.data(), npairs);
        d_pj_buf.upload(nl.pj.data(), npairs);
        d_excl_buf.upload(excluded.data(), npairs);
    }

    d_sum.zero();
    d_etot.zero();

    constexpr int BLOCK = 256;

    if (npairs > 0) {
        int grid = (npairs + BLOCK - 1) / BLOCK;
        born_sum_kernel<<<grid, BLOCK>>>(
            npairs, d_pi_buf.ptr, d_pj_buf.ptr,
            d_x.ptr, d_y.ptr, d_z.ptr, d_r.ptr, d_sum.ptr);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        int grid = (N + BLOCK - 1) / BLOCK;
        born_finalize_kernel<<<grid, BLOCK>>>(N, d_r.ptr, d_sum.ptr, d_a.ptr);
        CUDA_CHECK(cudaGetLastError());
    }

    if (npairs > 0) {
        int grid = (npairs + BLOCK - 1) / BLOCK;
        pair_energy_kernel<<<grid, BLOCK>>>(
            npairs, d_pi_buf.ptr, d_pj_buf.ptr, d_excl_buf.ptr,
            d_x.ptr, d_y.ptr, d_z.ptr,
            d_q.ptr, d_r.ptr, d_eps_buf.ptr,
            d_a.ptr, d_etot.ptr);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    float h_etot = 0.0f;
    d_etot.download(&h_etot, 1);

    if (a_out) {
        std::vector<float> ha(N);
        d_a.download(ha.data(), N);
        a_out->resize(N);
        for (int i = 0; i < N; ++i) (*a_out)[i] = static_cast<double>(ha[i]);
    }

    double sasa = sasa_nonpolar_cpu(p, nl);
    return sasa + static_cast<double>(h_etot);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 11. CHAIN GPU STATE  (P4.2 GPU residency)
 *
 *     One instance per MC chain, allocated once and reused for the whole
 *     chain's lifetime (steps_per_cand steps, or the full landscape
 *     trajectory) instead of reallocating/re-uploading device buffers on
 *     every call as the old per-call gpu_total_energy() did. Atom
 *     parameters (charge/radius/epsilon) never change and are uploaded
 *     once; the neighbor-pair/exclusion buffers are rebuilt only when the
 *     Verlet skin is exceeded; positions and Born radii are re-uploaded
 *     once per MC step (O(N), not O(N^2) — cheap next to the O(pairs)
 *     kernel work they feed).
 * ═══════════════════════════════════════════════════════════════════════ */
struct ChainGpuState {
    int N = 0;
    int npairs = 0;

    GpuBuf<float>   d_q, d_r, d_eps;     /* static for the whole chain */
    GpuBuf<float>   d_x, d_y, d_z, d_a;  /* refreshed every MC step */
    GpuBuf<uint8_t> d_inside;            /* refreshed every MC step */
    GpuBuf<int>     d_pi, d_pj;          /* rebuilt on NL rebuild only */
    GpuBuf<uint8_t> d_excluded;          /* rebuilt on NL rebuild only */
    GpuBuf<float>   d_etot;

    void init(int n) {
        N = n;
        d_q.alloc(n); d_r.alloc(n); d_eps.alloc(n);
        d_x.alloc(n); d_y.alloc(n); d_z.alloc(n); d_a.alloc(n);
        d_inside.alloc(n);
        d_etot.alloc(1);
    }

    void upload_static(const std::vector<float>& q,
                       const std::vector<float>& r,
                       const std::vector<float>& eps) {
        d_q.upload(q.data(), N);
        d_r.upload(r.data(), N);
        d_eps.upload(eps.data(), N);
    }

    void rebuild_pairs(const std::vector<int>& pi,
                       const std::vector<int>& pj,
                       const std::vector<uint8_t>& excluded) {
        npairs = static_cast<int>(pi.size());
        d_pi.alloc(npairs > 0 ? npairs : 1);
        d_pj.alloc(npairs > 0 ? npairs : 1);
        d_excluded.alloc(npairs > 0 ? npairs : 1);
        if (npairs > 0) {
            d_pi.upload(pi.data(), npairs);
            d_pj.upload(pj.data(), npairs);
            d_excluded.upload(excluded.data(), npairs);
        }
    }

    void upload_state(const std::vector<float>& x, const std::vector<float>& y,
                      const std::vector<float>& z, const std::vector<float>& a,
                      const std::vector<uint8_t>& inside) {
        d_x.upload(x.data(), N);
        d_y.upload(y.data(), N);
        d_z.upload(z.data(), N);
        d_a.upload(a.data(), N);
        d_inside.upload(inside.data(), N);
    }

    double cross_energy() {
        if (npairs == 0) return 0.0;
        d_etot.zero();
        constexpr int BLOCK = 256;
        int grid = (npairs + BLOCK - 1) / BLOCK;
        cross_pair_energy_kernel<<<grid, BLOCK>>>(
            npairs, d_pi.ptr, d_pj.ptr, d_excluded.ptr, d_inside.ptr,
            d_x.ptr, d_y.ptr, d_z.ptr,
            d_q.ptr, d_r.ptr, d_eps.ptr, d_a.ptr,
            d_etot.ptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float h = 0.0f;
        d_etot.download(&h, 1);
        return static_cast<double>(h);
    }
};

/* Uploads the full current state and returns the cross-boundary pairwise
 * energy for the given in_side mask. Called twice per MC move (before and
 * after the Rodrigues rotation) with the same mask, mirroring
 * physics_engine.cpp's cross_e(). */
static double gpu_cross_energy(ChainGpuState& gpu,
                               const std::vector<Particle>& st,
                               const std::vector<double>& a,
                               const std::vector<bool>& in_side)
{
    int N = static_cast<int>(st.size());
    std::vector<float>   hx(N), hy(N), hz(N), ha(N);
    std::vector<uint8_t> hin(N);
    for (int i = 0; i < N; ++i) {
        hx[i]  = static_cast<float>(st[i].x);
        hy[i]  = static_cast<float>(st[i].y);
        hz[i]  = static_cast<float>(st[i].z);
        ha[i]  = static_cast<float>(a[i]);
        hin[i] = in_side[i] ? 1 : 0;
    }
    gpu.upload_state(hx, hy, hz, ha, hin);
    return gpu.cross_energy();
}

/* ═══════════════════════════════════════════════════════════════════════
 * 12. MC CHAIN STATE + STEP LOGIC  (torsion-angle Metropolis MC, P1.5
 *     parity port). Mirrors physics_engine.cpp's generate_ensemble chain
 *     lambda / try_torsion / try_crankshaft exactly, with the cross-pair
 *     nonbonded sum offloaded to the GPU via ChainGpuState.
 * ═══════════════════════════════════════════════════════════════════════ */
struct MCState {
    std::vector<Particle> st;
    HostTopology           topo;
    NeighborList           nl;
    std::vector<double>    a;
    ChainGpuState          gpu;
    std::vector<bool>      in_side;
    double curE    = 0.0;
    double cur_max = 0.12;
    int    acc_win = 0, tot_win = 0;
    std::mt19937 rng{std::random_device{}()};
};

static void rebuild_state(MCState& S)
{
    S.nl.build(S.st);
    std::vector<uint8_t> excluded(S.nl.pi.size());
    for (size_t k = 0; k < S.nl.pi.size(); ++k)
        excluded[k] = S.topo.is_excluded(S.nl.pi[k], S.nl.pj[k]) ? 1 : 0;

    std::vector<double> a_new;
    double e = gpu_total_energy_topo(S.st, S.nl, excluded, &a_new);
    if (!S.topo.dihedrals.empty())       e += dihedral_e(S.st, S.topo.dihedrals);
    if (!S.topo.disulfide_pairs.empty()) e += ss_e(S.st, S.topo.disulfide_pairs);

    S.curE = e;
    S.a    = std::move(a_new);
    S.gpu.rebuild_pairs(S.nl.pi, S.nl.pj, excluded);
}

static void init_chain(MCState& S)
{
    int N = static_cast<int>(S.st.size());
    S.gpu.init(N);
    std::vector<float> hq(N), hr(N), heps(N);
    for (int i = 0; i < N; ++i) {
        hq[i]   = static_cast<float>(S.st[i].charge);
        hr[i]   = static_cast<float>(S.st[i].radius);
        heps[i] = static_cast<float>(S.st[i].epsilon);
    }
    S.gpu.upload_static(hq, hr, heps);
    S.in_side.assign(N, false);
    rebuild_state(S);
}

/* Standard torsion-angle MC move with lever-arm scaling (P1.5 parity). */
static std::pair<bool,bool> try_torsion(MCState& S, double T)
{
    int nrb = static_cast<int>(S.topo.rot_bonds.size());
    std::uniform_int_distribution<int> pick_rb(0, nrb - 1);
    int rb_idx = pick_rb(S.rng);
    const RotBond& rb = S.topo.rot_bonds[rb_idx];
    const std::vector<int>& side = S.topo.rot_bond_sides[rb_idx];
    if (side.empty()) return {false, false};

    auto& st = S.st;
    double ax = st[rb.j].x-st[rb.i].x, ay = st[rb.j].y-st[rb.i].y, az = st[rb.j].z-st[rb.i].z;
    double al = std::sqrt(ax*ax+ay*ay+az*az);
    if (al < 1e-10) return {false, false};
    ax/=al; ay/=al; az/=al;

    double base_d = (rb.kind == BK_BACKBONE_PHI || rb.kind == BK_BACKBONE_PSI)
                    ? S.cur_max : S.cur_max * 2.5;
    base_d *= S.topo.rot_bond_scale[rb_idx];
    double delta = std::uniform_real_distribution<double>(-base_d, base_d)(S.rng);
    double cosD  = std::cos(delta), sinD = std::sin(delta);

    for (int k : side) S.in_side[k] = true;

    std::vector<std::array<double,3>> old_pos(side.size());
    std::vector<double>               old_born(side.size());
    for (size_t k = 0; k < side.size(); ++k) {
        int idx = side[k];
        old_pos[k]  = {st[idx].x, st[idx].y, st[idx].z};
        old_born[k] = S.a[idx];
    }

    double old_cross = gpu_cross_energy(S.gpu, st, S.a, S.in_side);
    double old_sasa  = sasa_nonpolar_cpu(st, S.nl);
    double old_dih   = dihedral_e_boundary(st, S.topo.dihedrals, S.in_side);
    double old_ss    = S.topo.disulfide_pairs.empty() ? 0.0
                     : ss_e_side(st, S.topo.disulfide_pairs, S.in_side);

    double ox = st[rb.i].x, oy = st[rb.i].y, oz = st[rb.i].z;
    for (int k : side)
        rodrigues(st[k].x, st[k].y, st[k].z, ox, oy, oz, ax, ay, az, cosD, sinD);
    for (int k : side) update_born_cpu((size_t)k, st, S.nl, S.a);

    double new_cross = gpu_cross_energy(S.gpu, st, S.a, S.in_side);
    double new_sasa  = sasa_nonpolar_cpu(st, S.nl);
    double new_dih   = dihedral_e_boundary(st, S.topo.dihedrals, S.in_side);
    double new_ss    = S.topo.disulfide_pairs.empty() ? 0.0
                     : ss_e_side(st, S.topo.disulfide_pairs, S.in_side);

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    bool accepted = false;
    double dE = (new_cross-old_cross) + (new_sasa-old_sasa)
              + (new_dih-old_dih)   + (new_ss-old_ss);
    if (dE < 0.0 || uni(S.rng) < std::exp(-dE / T)) {
        S.curE += dE;
        accepted = true;
    } else {
        for (size_t k = 0; k < side.size(); ++k) {
            int idx = side[k];
            st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
            S.a[idx]  = old_born[k];
        }
    }
    for (int k : side) S.in_side[k] = false;
    return {accepted, true};
}

/* Crankshaft concerted move: +delta on phi then -delta on psi sharing one Ca. */
static std::pair<bool,bool> try_crankshaft(MCState& S, double T)
{
    int ncp = static_cast<int>(S.topo.concerted_pairs.size());
    if (ncp == 0) return {false, false};
    std::uniform_int_distribution<int> pick_cp(0, std::max(0, ncp - 1));
    int cp_idx = (ncp > 1) ? pick_cp(S.rng) : 0;
    int phi_k  = S.topo.concerted_pairs[cp_idx].first;
    int psi_k  = S.topo.concerted_pairs[cp_idx].second;
    const std::vector<int>& phi_side = S.topo.rot_bond_sides[phi_k];
    const std::vector<int>& psi_side = S.topo.rot_bond_sides[psi_k];
    if (phi_side.empty() || psi_side.empty()) return {false, false};

    auto& st = S.st;
    for (int k : phi_side) S.in_side[k] = true;

    const RotBond& phi_rb = S.topo.rot_bonds[phi_k];
    double p1ox = st[phi_rb.i].x, p1oy = st[phi_rb.i].y, p1oz = st[phi_rb.i].z;
    double p1ax = st[phi_rb.j].x-p1ox, p1ay = st[phi_rb.j].y-p1oy, p1az = st[phi_rb.j].z-p1oz;
    double p1l  = std::sqrt(p1ax*p1ax+p1ay*p1ay+p1az*p1az);
    if (p1l < 1e-10) { for (int k : phi_side) S.in_side[k] = false; return {false, false}; }
    p1ax/=p1l; p1ay/=p1l; p1az/=p1l;

    std::vector<std::array<double,3>> old_pos(phi_side.size());
    std::vector<double>               old_born(phi_side.size());
    for (size_t k = 0; k < phi_side.size(); ++k) {
        int idx = phi_side[k];
        old_pos[k]  = {st[idx].x, st[idx].y, st[idx].z};
        old_born[k] = S.a[idx];
    }
    double old_cross = gpu_cross_energy(S.gpu, st, S.a, S.in_side);
    double old_sasa  = sasa_nonpolar_cpu(st, S.nl);
    double old_dih   = dihedral_e_boundary(st, S.topo.dihedrals, S.in_side);
    double old_ss    = S.topo.disulfide_pairs.empty() ? 0.0
                     : ss_e_side(st, S.topo.disulfide_pairs, S.in_side);

    double delta = std::uniform_real_distribution<double>(-S.cur_max, S.cur_max)(S.rng);
    double cosD  = std::cos(delta), sinD = std::sin(delta);

    for (int k : phi_side)
        rodrigues(st[k].x, st[k].y, st[k].z, p1ox, p1oy, p1oz, p1ax, p1ay, p1az, cosD, sinD);

    const RotBond& psi_rb = S.topo.rot_bonds[psi_k];
    double p2ox = st[psi_rb.i].x, p2oy = st[psi_rb.i].y, p2oz = st[psi_rb.i].z;
    double p2ax = st[psi_rb.j].x-p2ox, p2ay = st[psi_rb.j].y-p2oy, p2az = st[psi_rb.j].z-p2oz;
    double p2l  = std::sqrt(p2ax*p2ax+p2ay*p2ay+p2az*p2az);
    if (p2l < 1e-10) {
        for (size_t k = 0; k < phi_side.size(); ++k) {
            int idx = phi_side[k];
            st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
        }
        for (int k : phi_side) S.in_side[k] = false;
        return {false, false};
    }
    p2ax/=p2l; p2ay/=p2l; p2az/=p2l;
    for (int k : psi_side)
        rodrigues(st[k].x, st[k].y, st[k].z, p2ox, p2oy, p2oz, p2ax, p2ay, p2az, cosD, -sinD);

    for (int k : phi_side) update_born_cpu((size_t)k, st, S.nl, S.a);

    double new_cross = gpu_cross_energy(S.gpu, st, S.a, S.in_side);
    double new_sasa  = sasa_nonpolar_cpu(st, S.nl);
    double new_dih   = dihedral_e_boundary(st, S.topo.dihedrals, S.in_side);
    double new_ss    = S.topo.disulfide_pairs.empty() ? 0.0
                     : ss_e_side(st, S.topo.disulfide_pairs, S.in_side);

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    bool accepted = false;
    double dE = (new_cross-old_cross) + (new_sasa-old_sasa)
              + (new_dih-old_dih)   + (new_ss-old_ss);
    if (dE < 0.0 || uni(S.rng) < std::exp(-dE / T)) {
        S.curE += dE;
        accepted = true;
    } else {
        for (size_t k = 0; k < phi_side.size(); ++k) {
            int idx = phi_side[k];
            st[idx].x = old_pos[k][0]; st[idx].y = old_pos[k][1]; st[idx].z = old_pos[k][2];
            S.a[idx]  = old_born[k];
        }
    }
    for (int k : phi_side) S.in_side[k] = false;
    return {accepted, true};
}

static void run_mc_steps(MCState& S, int steps, double T)
{
    int ncp = static_cast<int>(S.topo.concerted_pairs.size());
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    constexpr int    TUNE_FREQ  = 200;
    constexpr double TARGET_LO  = 0.28, TARGET_HI = 0.52;
    constexpr double SCALE_UP   = 1.06,  SCALE_DOWN = 0.94;
    constexpr double ANGLE_MAX  = 0.50,  ANGLE_MIN  = 0.004;

    for (int s = 0; s < steps; ++s) {
        if (S.nl.needs_rebuild(S.st)) rebuild_state(S);

        bool do_crank = ncp > 0 && uni(S.rng) < 0.25;
        auto [accepted, did_move] = do_crank ? try_crankshaft(S, T) : try_torsion(S, T);

        if (did_move) {
            S.acc_win += accepted ? 1 : 0;
            if (++S.tot_win == TUNE_FREQ) {
                double rate = (double)S.acc_win / TUNE_FREQ;
                if      (rate > TARGET_HI) S.cur_max = std::min(ANGLE_MAX, S.cur_max * SCALE_UP);
                else if (rate < TARGET_LO) S.cur_max = std::max(ANGLE_MIN, S.cur_max * SCALE_DOWN);
                S.acc_win = S.tot_win = 0;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * 13. PhysicsEngine CLASS
 * ═══════════════════════════════════════════════════════════════════════ */
class PhysicsEngine {
public:
    PhysicsEngine() {
        int ndev = 0;
        cudaError_t err = cudaGetDeviceCount(&ndev);
        if (err != cudaSuccess || ndev == 0) {
            throw std::runtime_error(
                "protein_physics_cuda: No CUDA-capable GPU found. "
                "Use the CPU module (protein_physics) instead.");
        }
    }

    static std::string device_name() {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        return std::string(prop.name);
    }

    /* calculate_potential: full GB/DH/LJ/SASA(+dihedral+SS if topo given)
     * energy for an arbitrary particle list. Rebuilds the neighbor list
     * from scratch every call — use only for final evaluation, not inside
     * a hot loop. */
    double calculate_potential(const std::vector<Particle>& particles,
                               const HostTopology* topo = nullptr) {
        if (particles.empty()) return 0.0;
        NeighborList nl; nl.build(particles);
        std::vector<uint8_t> excluded(nl.pi.size(), 0);
        if (topo)
            for (size_t k = 0; k < nl.pi.size(); ++k)
                excluded[k] = topo->is_excluded(nl.pi[k], nl.pj[k]) ? 1 : 0;

        double E = gpu_total_energy_topo(particles, nl, excluded, nullptr);
        if (topo && !topo->dihedrals.empty())       E += dihedral_e(particles, topo->dihedrals);
        if (topo && !topo->disulfide_pairs.empty()) E += ss_e(particles, topo->disulfide_pairs);
        return E;
    }

    /* generate_ensemble: run 'ncand' independent torsion-angle MC chains,
     * each for 'steps' steps. Returns the ncand final conformations. */
    std::vector<std::vector<Particle>> generate_ensemble(
        const std::vector<Particle>& init,
        const HostTopology& topo,
        int ncand, int steps,
        double T = 0.6,
        double max_angle = 0.12)
    {
        if (init.empty()) throw std::invalid_argument("initial_state empty");
        if (ncand <= 0 || steps <= 0) throw std::invalid_argument("ncand/steps must be positive");
        if (topo.rot_bonds.empty()) throw std::invalid_argument("topology has no rotatable bonds");

        std::vector<std::vector<Particle>> ens(ncand);
        auto run_one = [&](int c) {
            MCState S;
            S.st      = init;
            S.topo    = topo;
            S.cur_max = max_angle;
            init_chain(S);
            run_mc_steps(S, steps, T);
            ens[c] = std::move(S.st);
        };

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < ncand; ++c) run_one(c);
#else
        for (int c = 0; c < ncand; ++c) run_one(c);
#endif
        return ens;
    }

    /* run_landscape_trajectory: run ONE torsion-MC chain for
     * n_snapshots * steps_per_snapshot total steps, recording a full
     * snapshot (particle list + energy) every steps_per_snapshot steps —
     * the GPU-resident equivalent of physics_engine.cpp's method of the
     * same name (see IMPROVEMENTS.md item #12). Lets LandscapeWorker
     * (Python) advance the whole chain with one call instead of looping
     * 120 times, each iteration re-marshalling the full particle array
     * and reallocating GPU buffers from scratch across the Python<->C++
     * boundary. */
    std::pair<std::vector<std::vector<Particle>>, std::vector<double>>
    run_landscape_trajectory(
        const std::vector<Particle>& init,
        const HostTopology& topo,
        int n_snapshots, int steps_per_snapshot,
        double T = 0.6,
        double max_angle = 0.12)
    {
        if (init.empty()) throw std::invalid_argument("initial_state empty");
        if (n_snapshots <= 0 || steps_per_snapshot <= 0)
            throw std::invalid_argument("n_snapshots/steps_per_snapshot must be positive");
        if (topo.rot_bonds.empty()) throw std::invalid_argument("topology has no rotatable bonds");

        MCState S;
        S.st      = init;
        S.topo    = topo;
        S.cur_max = max_angle;
        init_chain(S);

        std::vector<std::vector<Particle>> snapshots;
        std::vector<double>                energies;
        snapshots.reserve(n_snapshots);
        energies.reserve(n_snapshots);

        for (int snap = 0; snap < n_snapshots; ++snap) {
            run_mc_steps(S, steps_per_snapshot, T);
            snapshots.push_back(S.st);
            // Fresh from-scratch evaluation, matching the old Python-level
            // LandscapeWorker's calculate_potential() call exactly (not the
            // loop's incrementally-tracked S.curE, which would drift).
            energies.push_back(calculate_potential(S.st, &S.topo));
        }
        return {snapshots, energies};
    }

    /* lowest_energy_structure: scan an ensemble, return the lowest-energy
     * conformation. O(ncand) calculate_potential calls (no topology, same
     * as the CPU engine's behaviour) — call only once after MC finishes. */
    std::vector<Particle> lowest_energy_structure(
        const std::vector<std::vector<Particle>>& ens)
    {
        if (ens.empty())
            throw std::invalid_argument("ensemble empty");
        return *std::min_element(
            ens.begin(), ens.end(),
            [this](const auto& a, const auto& b) {
                return calculate_potential(a) < calculate_potential(b);
            });
    }

    int num_threads() const {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }
};

/* ═══════════════════════════════════════════════════════════════════════
 * 14. PYBIND11 MODULE
 * ═══════════════════════════════════════════════════════════════════════ */
PYBIND11_MODULE(protein_physics_cuda, m)
{
    m.doc() = "GPU-accelerated implicit-solvent engine "
              "(CUDA Born/pair-energy kernels + torsion-angle MC, "
              "topology-aware: dihedral/exclusion/disulfide parity with protein_physics)";

    // module_local(): protein_physics (the CPU module) defines its own,
    // separately-compiled "Particle"/"PhysicsEngine" classes with the same
    // names. On Windows, MSVC's RTTI compares type_info by decorated NAME
    // across DLLs (unlike Linux/macOS, where each shared object has its own
    // independent RTTI), so pybind11 sees these as the SAME C++ type and
    // refuses this module's registration with "generic_type: type is already
    // registered!" the moment both modules are imported in one process
    // (gui_main.py always imports protein_physics, then conditionally
    // protein_physics_cuda to probe for a GPU at startup). module_local()
    // keeps this module's registration in its own per-module table instead
    // of the shared global one -- safe here since Particle/PhysicsEngine
    // instances are never passed between the two engines.
    py::class_<Particle>(m, "Particle", py::module_local())
        .def(py::init<double, double, double, double, double, double, bool>(),
             py::arg("x"),
             py::arg("y"),
             py::arg("z"),
             py::arg("charge"),
             py::arg("radius")    = 1.9,
             py::arg("epsilon")   = 0.1,
             py::arg("is_water")  = false)
        .def_readwrite("x",        &Particle::x)
        .def_readwrite("y",        &Particle::y)
        .def_readwrite("z",        &Particle::z)
        .def_readwrite("charge",   &Particle::charge)
        .def_readwrite("radius",   &Particle::radius)
        .def_readwrite("epsilon",  &Particle::epsilon)
        .def_readwrite("is_water", &Particle::is_water);

    py::class_<PhysicsEngine>(m, "PhysicsEngine", py::module_local())
        .def(py::init<>())
        .def("calculate_potential",
             [](PhysicsEngine& self, const std::vector<Particle>& particles,
                py::object topo_obj) -> double {
                 if (topo_obj.is_none()) {
                     py::gil_scoped_release release;
                     return self.calculate_potential(particles, nullptr);
                 }
                 HostTopology topo = extract_topology(topo_obj);
                 py::gil_scoped_release release;
                 return self.calculate_potential(particles, &topo);
             },
             py::arg("particles"),
             py::arg("topology") = py::none())
        .def("generate_ensemble",
             [](PhysicsEngine& self,
                const std::vector<Particle>& init, py::object topo_obj,
                int ncand, int steps,
                double T, double max_angle) -> std::vector<std::vector<Particle>> {
                 HostTopology topo = extract_topology(topo_obj);
                 py::gil_scoped_release release;
                 return self.generate_ensemble(init, topo, ncand, steps, T, max_angle);
             },
             py::arg("initial_state"),
             py::arg("topology"),
             py::arg("n_candidates"),
             py::arg("steps_per_cand"),
             py::arg("temperature") = 0.6,
             py::arg("max_angle")   = 0.12)
        .def("run_landscape_trajectory",
             [](PhysicsEngine& self,
                const std::vector<Particle>& init, py::object topo_obj,
                int n_snapshots, int steps_per_snapshot,
                double T, double max_angle)
                -> std::pair<std::vector<std::vector<Particle>>, std::vector<double>> {
                 HostTopology topo = extract_topology(topo_obj);
                 py::gil_scoped_release release;
                 return self.run_landscape_trajectory(
                     init, topo, n_snapshots, steps_per_snapshot, T, max_angle);
             },
             py::arg("initial_state"),
             py::arg("topology"),
             py::arg("n_snapshots"),
             py::arg("steps_per_snapshot"),
             py::arg("temperature") = 0.6,
             py::arg("max_angle")   = 0.12)
        .def("lowest_energy_structure",
             [](PhysicsEngine& self, const std::vector<std::vector<Particle>>& ens) {
                 py::gil_scoped_release release;
                 return self.lowest_energy_structure(ens);
             },
             py::arg("ensemble"))
        .def("num_threads",
             &PhysicsEngine::num_threads)
        .def_static("device_name",
                    &PhysicsEngine::device_name);
}
