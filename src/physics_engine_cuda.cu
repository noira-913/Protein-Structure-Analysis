/*
 * physics_engine_cuda.cu
 *
 * CUDA C++ port of physics_engine.cpp.
 * Exposes a pybind11 module named `protein_physics_cuda` with the same
 * Particle and PhysicsEngine API as the CPU version, plus:
 *   - PhysicsEngine::device_name()  (static)
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

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

/* ═══════════════════════════════════════════════════════════════════════
 * 1.  PHYSICS CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── double-precision CPU-side constants ─────────────────────────────── */
static constexpr double COULOMB    = 332.0636;
static constexpr double EPS_WATER  = 78.5;
static constexpr double EPS_PROT   = 1.0;
static constexpr double KAPPA      = 0.1257;
static constexpr double GAMMA_SA   = 0.00542;
static constexpr double BETA_SA    = 0.92;
static constexpr double PROBE_R    = 1.4;
static constexpr double NL_CUTOFF  = 12.0;
static constexpr double NL_SKIN    = 2.0;
static constexpr double NL_RCUT2   = (NL_CUTOFF + NL_SKIN) * (NL_CUTOFF + NL_SKIN);
static constexpr double PAIR_CUT2  = NL_CUTOFF * NL_CUTOFF;
static constexpr double HALF_SKIN2 = (NL_SKIN * 0.5) * (NL_SKIN * 0.5);
static constexpr double HARD_SCALE = 1.0e4;
static constexpr double GB_COEF    = -0.5 * (1.0 / EPS_PROT - 1.0 / EPS_WATER) * COULOMB;

/* ── single-precision GPU-side constants (#define for device code) ───── */
#define COULOMB_F      332.0636f
#define EPS_WATER_F    78.5f
#define KAPPA_F        0.1257f
#define HARD_SCALE_F   1.0e4f
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

static inline double pair_e_cpu(const Particle& pi_p, const Particle& pj_p,
                                 double ai, double aj) noexcept
{
    double dx  = pi_p.x - pj_p.x;
    double dy  = pi_p.y - pj_p.y;
    double dz  = pi_p.z - pj_p.z;
    double r2  = dx*dx + dy*dy + dz*dz;
    double r   = std::sqrt(r2);
    double sig = pi_p.radius + pj_p.radius;
    if (r < sig * 0.85)
        return HARD_SCALE * std::pow(sig / r, 12.0);
    double qp  = pi_p.charge * pj_p.charge;
    double edh = (COULOMB * qp) / (EPS_WATER * r) * std::exp(-KAPPA * r);
    double fgb = std::sqrt(r2 + ai * aj * std::exp(-r2 / (4.0 * ai * aj)));
    double egb = GB_COEF * qp / fgb;
    double eps = std::sqrt(pi_p.epsilon * pj_p.epsilon);
    double s6  = std::pow(sig / r, 6.0);
    double elj = 4.0 * eps * (s6*s6 - s6);
    return edh + egb + elj;
}

static std::vector<double> born_radii_cpu(const std::vector<Particle>& p,
                                           const NeighborList& nl)
{
    size_t N = p.size();
    std::vector<double> sum(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j : nl.nb[i]) {
            double r2 = d2_cpu(p[i], p[j]);
            double r  = std::sqrt(r2);
            sum[i] += hct_cpu(r, r2, p[i].radius, p[j].radius);
            sum[j] += hct_cpu(r, r2, p[j].radius, p[i].radius);
        }
    }
    std::vector<double> a(N);
    for (size_t i = 0; i < N; ++i) {
        double inv = 1.0 / p[i].radius - 0.5 * sum[i];
        a[i] = 1.0 / std::max(inv, 2.0);
    }
    return a;
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
 * 7.  CUDA DEVICE FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * HCT Born-integral kernel  (float)
 * Computes the HCT pairwise contribution to the Born sum of atom i
 * from atom j with vdW radii ri, rj and interatomic distance r / r2.
 */
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
 * Pairwise energy  (float)
 * Returns the total pairwise interaction energy between atoms i and j,
 * including Debye–Hückel electrostatics, Generalized Born,
 * Lennard-Jones, and a hard-core repulsion.
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

    /* Hard-core repulsion */
    if (r < sig * 0.85f) {
        float ratio = sig / r;
        float r6    = ratio * ratio * ratio * ratio * ratio * ratio;
        return HARD_SCALE_F * r6 * r6;
    }

    float qp  = qi * qj;

    /* Debye–Hückel */
    float edh = (COULOMB_F * qp) / (EPS_WATER_F * r) * __expf(-KAPPA_F * r);

    /* Generalized Born */
    float aiaj = ai * aj;
    float fgb  = sqrtf(r2 + aiaj * __expf(-r2 / (4.0f * aiaj)));
    float egb  = GB_COEF_F * qp / fgb;

    /* Lennard-Jones */
    float eps = sqrtf(epsi * epsj);
    float sr  = sig / r;
    float s6  = sr * sr * sr * sr * sr * sr;
    float elj = 4.0f * eps * (s6*s6 - s6);

    return edh + egb + elj;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 8.  CUDA KERNELS
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * born_sum_kernel
 * One thread per (i,j) pair from the neighbor list.
 * Atomically accumulates HCT integrals into per-atom Born sums.
 *
 * Params:
 *   npairs   — number of pairs
 *   d_pi     — pair atom-i indices  [npairs]
 *   d_pj     — pair atom-j indices  [npairs]
 *   d_x/y/z  — atom positions       [natoms]
 *   d_r      — atom vdW radii       [natoms]
 *   d_sum    — Born sum accumulators [natoms], zero-initialised by caller
 */
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

    /* Contribution of j to i's sum and i to j's sum */
    float cij = hct_gpu(r, r2, ri, rj);
    float cji = hct_gpu(r, r2, rj, ri);

    atomicAdd(&d_sum[i], cij);
    atomicAdd(&d_sum[j], cji);
}

/*
 * born_finalize_kernel
 * One thread per atom.
 * Converts per-atom Born sums to effective Born radii.
 *
 *   a[i] = 1 / max( 1/r[i] - 0.5*sum[i],  2 )
 */
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
 * One thread per (i,j) pair from the neighbor list.
 * Skips pairs beyond PAIR_CUT2_F.
 * Atomically accumulates pairwise energy into a scalar.
 */
__global__
void pair_energy_kernel(int npairs,
                        const int*   d_pi,
                        const int*   d_pj,
                        const float* d_x,
                        const float* d_y,
                        const float* d_z,
                        const float* d_q,
                        const float* d_r,
                        const float* d_eps,
                        const float* d_a,
                        float*       d_etot)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npairs) return;

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

/* ═══════════════════════════════════════════════════════════════════════
 * 9.  HOST FUNCTION: gpu_total_energy
 *     Orchestrates the three GPU kernels and adds the CPU-computed SASA.
 * ═══════════════════════════════════════════════════════════════════════ */
static double gpu_total_energy(const std::vector<Particle>& p,
                                const NeighborList&           nl)
{
    const int N      = static_cast<int>(p.size());
    const int npairs = static_cast<int>(nl.pi.size());

    /* ── Build SoA float arrays on host ──────────────────────────────── */
    std::vector<float> hx(N), hy(N), hz(N), hq(N), hr(N), heps(N);
    for (int i = 0; i < N; ++i) {
        hx[i]   = static_cast<float>(p[i].x);
        hy[i]   = static_cast<float>(p[i].y);
        hz[i]   = static_cast<float>(p[i].z);
        hq[i]   = static_cast<float>(p[i].charge);
        hr[i]   = static_cast<float>(p[i].radius);
        heps[i] = static_cast<float>(p[i].epsilon);
    }

    /* ── Allocate GPU buffers ─────────────────────────────────────────── */
    GpuBuf<float> d_x(N),    d_y(N),    d_z(N);
    GpuBuf<float> d_q(N),    d_r(N),    d_eps_buf(N);
    GpuBuf<float> d_sum(N),  d_a(N);
    GpuBuf<float> d_etot(1);
    GpuBuf<int>   d_pi_buf(npairs > 0 ? npairs : 1);
    GpuBuf<int>   d_pj_buf(npairs > 0 ? npairs : 1);

    /* ── Upload atom data ─────────────────────────────────────────────── */
    d_x.upload(hx.data(),   N);
    d_y.upload(hy.data(),   N);
    d_z.upload(hz.data(),   N);
    d_q.upload(hq.data(),   N);
    d_r.upload(hr.data(),   N);
    d_eps_buf.upload(heps.data(), N);

    /* ── Upload pair indices ──────────────────────────────────────────── */
    if (npairs > 0) {
        d_pi_buf.upload(nl.pi.data(), npairs);
        d_pj_buf.upload(nl.pj.data(), npairs);
    }

    /* ── Zero accumulators ────────────────────────────────────────────── */
    d_sum.zero();
    d_etot.zero();

    constexpr int BLOCK = 256;

    /* ── Kernel 1: Born sums ─────────────────────────────────────────── */
    if (npairs > 0) {
        int grid = (npairs + BLOCK - 1) / BLOCK;
        born_sum_kernel<<<grid, BLOCK>>>(
            npairs,
            d_pi_buf.ptr, d_pj_buf.ptr,
            d_x.ptr, d_y.ptr, d_z.ptr,
            d_r.ptr,
            d_sum.ptr);
        CUDA_CHECK(cudaGetLastError());
    }

    /* ── Kernel 2: Born finalize ─────────────────────────────────────── */
    {
        int grid = (N + BLOCK - 1) / BLOCK;
        born_finalize_kernel<<<grid, BLOCK>>>(
            N,
            d_r.ptr,
            d_sum.ptr,
            d_a.ptr);
        CUDA_CHECK(cudaGetLastError());
    }

    /* ── Kernel 3: Pairwise energy ───────────────────────────────────── */
    if (npairs > 0) {
        int grid = (npairs + BLOCK - 1) / BLOCK;
        pair_energy_kernel<<<grid, BLOCK>>>(
            npairs,
            d_pi_buf.ptr, d_pj_buf.ptr,
            d_x.ptr, d_y.ptr, d_z.ptr,
            d_q.ptr, d_r.ptr, d_eps_buf.ptr,
            d_a.ptr,
            d_etot.ptr);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    /* ── Download scalar result ───────────────────────────────────────── */
    float h_etot = 0.0f;
    d_etot.download(&h_etot, 1);

    /* ── Add CPU-computed SASA (kept on CPU as specified) ─────────────── */
    double sasa = sasa_nonpolar_cpu(p, nl);

    return sasa + static_cast<double>(h_etot);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 10. PhysicsEngine CLASS
 * ═══════════════════════════════════════════════════════════════════════ */
class PhysicsEngine {
private:
    std::mt19937 gen_;

    /* ── Total energy using GPU for pair sums, CPU for SASA ─────────── */
    static double total_e_gpu(const std::vector<Particle>& p,
                               const NeighborList& nl)
    {
        return gpu_total_energy(p, nl);
    }

    /* ── Total energy using CPU helpers only (fallback / incremental) ── */
    static double total_e_cpu(const std::vector<Particle>& p,
                               const NeighborList& nl,
                               const std::vector<double>& a)
    {
        double E = sasa_nonpolar_cpu(p, nl);
        size_t N = p.size();
        for (size_t i = 0; i < N; ++i) {
            for (size_t j : nl.nb[i]) {
                double dx = p[i].x - p[j].x;
                double dy = p[i].y - p[j].y;
                double dz = p[i].z - p[j].z;
                if (dx*dx + dy*dy + dz*dz > PAIR_CUT2) continue;
                E += pair_e_cpu(p[i], p[j], a[i], a[j]);
            }
        }
        return E;
    }

public:
    /* ── Constructor: verify a CUDA device exists ───────────────────── */
    PhysicsEngine() : gen_(std::random_device{}()) {
        int ndev = 0;
        cudaError_t err = cudaGetDeviceCount(&ndev);
        if (err != cudaSuccess || ndev == 0) {
            throw std::runtime_error(
                "protein_physics_cuda: No CUDA-capable GPU found. "
                "Use the CPU module (protein_physics) instead.");
        }
    }

    /* ── device_name: returns the name of GPU 0 ─────────────────────── */
    static std::string device_name() {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        return std::string(prop.name);
    }

    /* ── calculate_potential ─────────────────────────────────────────── */
    double calculate_potential(const std::vector<Particle>& particles) {
        if (particles.empty()) return 0.0;
        NeighborList nl;
        nl.build(particles);
        return total_e_gpu(particles, nl);
    }

    /* ── generate_ensemble  (MC with GPU full-energy evaluations) ────── */
    std::vector<std::vector<Particle>> generate_ensemble(
        const std::vector<Particle>& init,
        int    ncand,
        int    steps,
        double T    = 0.6,
        double maxd = 0.3)
    {
        if (init.empty())
            throw std::invalid_argument("initial_state empty");
        if (ncand <= 0 || steps <= 0)
            throw std::invalid_argument("ncand/steps must be positive");

        const size_t N = init.size();
        std::vector<std::vector<Particle>> ens(ncand);

        /*
         * Chain lambda: one independent MC trajectory.
         * Full-energy (when NL is rebuilt) uses GPU via total_e_gpu.
         * Per-step incremental energy uses CPU helpers (too fine-grained for GPU).
         */
        auto chain = [&](int c, std::mt19937& rng) {
            std::vector<Particle> st = init;
            NeighborList nl;
            nl.build(st);
            /* Compute initial Born radii on CPU (needed for incremental MC) */
            auto a = born_radii_cpu(st, nl);
            /* Use GPU for the initial full energy */
            double curE = total_e_gpu(st, nl);

            std::uniform_real_distribution<double> disp(-maxd, maxd);
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            std::uniform_int_distribution<size_t>  pick(0, N - 1);

            for (int s = 0; s < steps; ++s) {
                /* Rebuild NL when atoms have drifted; recompute full energy on GPU */
                if (nl.needs_rebuild(st)) {
                    nl.build(st);
                    a    = born_radii_cpu(st, nl);
                    curE = total_e_gpu(st, nl);
                }

                size_t idx = pick(rng);
                double ox = st[idx].x, oy = st[idx].y, oz = st[idx].z;

                /* Propose move */
                st[idx].x += disp(rng);
                st[idx].y += disp(rng);
                st[idx].z += disp(rng);

                /* Old pairwise contribution of atom idx (CPU incremental) */
                double old_p = 0.0;
                for (size_t j : nl.nb[idx]) {
                    double dx = ox - st[j].x;
                    double dy = oy - st[j].y;
                    double dz = oz - st[j].z;
                    if (dx*dx + dy*dy + dz*dz > PAIR_CUT2) continue;
                    Particle tmp = st[idx];
                    tmp.x = ox; tmp.y = oy; tmp.z = oz;
                    old_p += pair_e_cpu(tmp, st[j], a[idx], a[j]);
                }

                /* Update Born radius for the moved atom (CPU incremental) */
                double old_a = a[idx];
                update_born_cpu(idx, st, nl, a);

                /* New pairwise contribution of atom idx (CPU incremental) */
                double new_p = 0.0;
                for (size_t j : nl.nb[idx]) {
                    double dx = st[idx].x - st[j].x;
                    double dy = st[idx].y - st[j].y;
                    double dz = st[idx].z - st[j].z;
                    if (dx*dx + dy*dy + dz*dz > PAIR_CUT2) continue;
                    new_p += pair_e_cpu(st[idx], st[j], a[idx], a[j]);
                }

                double dE = new_p - old_p;

                /* Metropolis acceptance */
                if (dE < 0.0 || uni(rng) < std::exp(-dE / T)) {
                    curE += dE;
                } else {
                    /* Reject: restore position and Born radius */
                    st[idx].x = ox;
                    st[idx].y = oy;
                    st[idx].z = oz;
                    a[idx]    = old_a;
                }
            }
            ens[c] = std::move(st);
        }; /* end chain lambda */

        /* ── Run chains, parallel if OpenMP available ──────────────── */
#ifdef _OPENMP
        #pragma omp parallel
        {
            std::mt19937 lg(
                std::random_device{}() ^
                (static_cast<uint32_t>(std::hash<int>{}(omp_get_thread_num())) << 16u));
            #pragma omp for schedule(dynamic)
            for (int c = 0; c < ncand; ++c) chain(c, lg);
        }
#else
        for (int c = 0; c < ncand; ++c) chain(c, gen_);
#endif
        return ens;
    }

    /* ── lowest_energy_structure ─────────────────────────────────────── */
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

    /* ── num_threads ─────────────────────────────────────────────────── */
    int num_threads() const {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }
};

/* ═══════════════════════════════════════════════════════════════════════
 * 11. PYBIND11 MODULE
 * ═══════════════════════════════════════════════════════════════════════ */
PYBIND11_MODULE(protein_physics_cuda, m)
{
    m.doc() = "GPU-accelerated implicit-solvent engine "
              "(CUDA Born·pair kernels + CPU SASA·incremental-MC)";

    py::class_<Particle>(m, "Particle")
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

    py::class_<PhysicsEngine>(m, "PhysicsEngine")
        .def(py::init<>())
        // topology arg accepted for API parity with CPU module but not used (GPU Cartesian MC).
        .def("calculate_potential",
             [](PhysicsEngine& self, const std::vector<Particle>& particles,
                py::object /* topo */) -> double {
                 py::gil_scoped_release release;
                 return self.calculate_potential(particles);
             },
             py::arg("particles"),
             py::arg("topology") = py::none())
        // Wrapper that matches the CPU module's torsion-MC signature.
        // The GPU backend accepts 'topology' but ignores it — Cartesian moves
        // remain in use until P4.2 (GPU-resident torsion trajectory).
        // max_angle (rad) is converted to a comparable Cartesian maxd (Å).
        .def("generate_ensemble",
             [](PhysicsEngine& self,
                const std::vector<Particle>& init,
                py::object /* topo: accepted, ignored until P4.2 */,
                int ncand, int steps,
                double T, double max_angle) -> std::vector<std::vector<Particle>> {
                 // ~0.12 rad × 3 Å typical bond arm ≈ 0.36 Å; clamp to [0.05, 1.0]
                 double maxd = std::min(1.0, std::max(0.05, max_angle * 3.0));
                 py::gil_scoped_release release;
                 return self.generate_ensemble(init, ncand, steps, T, maxd);
             },
             py::arg("initial_state"),
             py::arg("topology"),
             py::arg("n_candidates"),
             py::arg("steps_per_cand"),
             py::arg("temperature") = 0.6,
             py::arg("max_angle")   = 0.12)
        .def("lowest_energy_structure",
             &PhysicsEngine::lowest_energy_structure,
             py::arg("ensemble"),
             py::call_guard<py::gil_scoped_release>())
        .def("num_threads",
             &PhysicsEngine::num_threads)
        .def_static("device_name",
                    &PhysicsEngine::device_name);
}
