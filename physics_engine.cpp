#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// ─────────────────────────────────────────────
//  데이터 구조
// ─────────────────────────────────────────────

struct Particle {
    double x, y, z;
    double charge;
    double radius;      // 원자별 반지름 (Å) — LJ 시그마 계산에 사용
    double epsilon;     // 원자별 LJ 우물 깊이 (kcal/mol)
    bool   is_water;

    Particle(double x, double y, double z,
             double charge,
             double radius  = 1.9,   // 기본값: 평균 C-alpha 반지름
             double epsilon = 0.1,
             bool   is_water = false)
        : x(x), y(y), z(z),
          charge(charge), radius(radius),
          epsilon(epsilon), is_water(is_water) {}
};

// ─────────────────────────────────────────────
//  PhysicsEngine
// ─────────────────────────────────────────────

class PhysicsEngine {
private:
    std::mt19937 gen;

    // ── 내부 유틸 ──────────────────────────────

    static inline double dist2(const Particle& a, const Particle& b) noexcept {
        double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        return dx*dx + dy*dy + dz*dz;
    }

    // Generalized Born 유효 Born 반지름 근사 (Still 1990)
    // 간단한 Coulomb 적분 근사: 1/R_i ≈ 1/r_i - Σ_j f(r_ij)
    std::vector<double> compute_born_radii(const std::vector<Particle>& p) const {
        const size_t N = p.size();
        std::vector<double> alpha(N);

        for (size_t i = 0; i < N; ++i) {
            double sum = 0.0;
            double ri  = p[i].radius;

            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;
                double r2  = dist2(p[i], p[j]);
                double r   = std::sqrt(r2);
                double rj  = p[j].radius;

                // Hawkins-Cramer-Truhlar (HCT) 근사
                double L = std::max(std::abs(r - rj), ri);
                double U = r + rj;
                if (ri < U) {
                    sum += (1.0/L - 1.0/U
                            + (r2 - rj*rj + ri*ri) / (2.0*r*ri*ri)  // ∂ 보정항
                            * std::log(L/U) * 0.5 / r);
                }
            }
            // 유효 Born 반지름: 최소값 0.5Å 클램프
            double inv_alpha = 1.0/ri - 0.5 * sum;
            alpha[i] = 1.0 / std::max(inv_alpha, 1.0 / 0.5);
        }
        return alpha;
    }

public:
    PhysicsEngine() : gen(std::random_device{}()) {}

    // ── 1. 포텐셜 에너지 ──────────────────────

    /**
     * 총 포텐셜 에너지 = E_elec(DH) + E_GB + E_LJ + E_hard
     *
     *  E_elec  : Debye-Hückel screened Coulomb  (장거리 차폐)
     *  E_GB    : Generalized Born 암시적 용매   (탈용매화 자유에너지)
     *  E_LJ    : Lennard-Jones 12-6             (원자별 ε, σ 사용)
     *  E_hard  : 하드 코어 반발                  (r < σ_ij 초과 시)
     */
    double calculate_potential(const std::vector<Particle>& particles) {
        if (particles.empty()) return 0.0;

        const size_t N       = particles.size();
        const double eps_w   = 78.5;   // 물의 유전율
        const double eps_p   = 1.0;    // 단백질 내부 유전율
        const double kappa   = 0.125;  // Debye 역길이 (Å⁻¹), 150 mM 이온 강도
        const double GB_coef = -0.5 * (1.0/eps_p - 1.0/eps_w) * 332.0; // kcal/mol 단위

        // Generalized Born 유효 반지름 선계산
        auto alpha = compute_born_radii(particles);

        double energy = 0.0;

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic) reduction(+:energy)
#endif
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                double r2  = dist2(particles[i], particles[j]);
                double r   = std::sqrt(r2);

                // ─ 하드 코어 반발 ─────────────────────
                double sig_ij = particles[i].radius + particles[j].radius;
                if (r < sig_ij * 0.9) {
                    energy += 1.0e4 * std::pow((sig_ij / r), 12.0);
                    continue;  // 극단적 겹침이면 다른 항 생략
                }

                // ─ Debye-Hückel ──────────────────────
                double q_prod = particles[i].charge * particles[j].charge;
                energy += (q_prod * 332.0) / (eps_w * r) * std::exp(-kappa * r);

                // ─ Generalized Born ──────────────────
                double f_gb = std::sqrt(r2
                              + alpha[i] * alpha[j]
                              * std::exp(-r2 / (4.0 * alpha[i] * alpha[j])));
                energy += GB_coef * q_prod / f_gb;

                // ─ Lennard-Jones (Lorentz-Berthelot 혼합 규칙) ──
                double eps_ij = std::sqrt(particles[i].epsilon * particles[j].epsilon);
                double sig6   = std::pow(sig_ij / r, 6);
                energy += 4.0 * eps_ij * (sig6 * sig6 - sig6);
            }
        }
        return energy;
    }

    // ── 2. Monte Carlo 앙상블 생성 ─────────────

    /**
     * Metropolis-Hastings 샘플링
     *  - 온도 T (kcal/mol 단위, 기본 0.6 ≈ 300 K)
     *  - x/y/z 모두 교란 (이전 버전은 x만 이동)
     *  - 후보군 병렬 생성 (OpenMP)
     */
    std::vector<std::vector<Particle>> generate_ensemble(
        const std::vector<Particle>& initial_state,
        int    n_candidates,
        int    steps_per_cand,
        double temperature  = 0.6,
        double max_disp     = 0.5)
    {
        if (initial_state.empty())
            throw std::invalid_argument("initial_state가 비어 있습니다.");
        if (n_candidates <= 0 || steps_per_cand <= 0)
            throw std::invalid_argument("n_candidates 와 steps_per_cand 는 양수여야 합니다.");

        const size_t N = initial_state.size();
        std::vector<std::vector<Particle>> ensemble(n_candidates);

        // 각 스레드마다 독립적인 난수 엔진 사용
#ifdef _OPENMP
        #pragma omp parallel
        {
            std::mt19937 local_gen(std::random_device{}()
                                   ^ (std::hash<int>{}(omp_get_thread_num()) << 16));
            std::uniform_real_distribution<double> disp(-max_disp, max_disp);
            std::uniform_real_distribution<double> prob(0.0, 1.0);
            std::uniform_int_distribution<size_t>  pick(0, N - 1);

            #pragma omp for schedule(dynamic)
            for (int c = 0; c < n_candidates; ++c) {
                std::vector<Particle> state = initial_state;
                double cur_E = calculate_potential(state);

                for (int s = 0; s < steps_per_cand; ++s) {
                    size_t idx   = pick(local_gen);
                    double ox = state[idx].x, oy = state[idx].y, oz = state[idx].z;
                    state[idx].x += disp(local_gen);
                    state[idx].y += disp(local_gen);
                    state[idx].z += disp(local_gen);

                    double new_E = calculate_potential(state);
                    double dE    = new_E - cur_E;

                    if (dE < 0.0 || prob(local_gen) < std::exp(-dE / temperature)) {
                        cur_E = new_E;
                    } else {
                        state[idx].x = ox;
                        state[idx].y = oy;
                        state[idx].z = oz;
                    }
                }
                ensemble[c] = std::move(state);
            }
        }
#else
        // OpenMP 없는 환경 (LG 그램 등) — 순차 실행
        std::uniform_real_distribution<double> disp(-max_disp, max_disp);
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        std::uniform_int_distribution<size_t>  pick(0, N - 1);

        for (int c = 0; c < n_candidates; ++c) {
            std::vector<Particle> state = initial_state;
            double cur_E = calculate_potential(state);

            for (int s = 0; s < steps_per_cand; ++s) {
                size_t idx   = pick(gen);
                double ox = state[idx].x, oy = state[idx].y, oz = state[idx].z;
                state[idx].x += disp(gen);
                state[idx].y += disp(gen);
                state[idx].z += disp(gen);

                double new_E = calculate_potential(state);
                double dE    = new_E - cur_E;

                if (dE < 0.0 || prob(gen) < std::exp(-dE / temperature)) {
                    cur_E = new_E;
                } else {
                    state[idx].x = ox;
                    state[idx].y = oy;
                    state[idx].z = oz;
                }
            }
            ensemble[c] = std::move(state);
        }
#endif
        return ensemble;
    }

    // ── 3. 유틸리티 ──────────────────────────

    /** 앙상블에서 최저 에너지 구조 반환 */
    std::vector<Particle> lowest_energy_structure(
        const std::vector<std::vector<Particle>>& ensemble)
    {
        if (ensemble.empty()) throw std::invalid_argument("앙상블이 비어 있습니다.");
        return *std::min_element(ensemble.begin(), ensemble.end(),
            [this](const std::vector<Particle>& a, const std::vector<Particle>& b) {
                return calculate_potential(a) < calculate_potential(b);
            });
    }

    /** 현재 OpenMP 스레드 수 (디버그용) */
    int num_threads() const {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }
};

// ─────────────────────────────────────────────
//  Python 바인딩
// ─────────────────────────────────────────────

PYBIND11_MODULE(protein_physics, m) {
    m.doc() = "Implicit-solvent protein physics engine (OpenMP-accelerated)";

    py::class_<Particle>(m, "Particle")
        .def(py::init<double, double, double, double, double, double, bool>(),
             py::arg("x"), py::arg("y"), py::arg("z"),
             py::arg("charge"),
             py::arg("radius")   = 1.9,
             py::arg("epsilon")  = 0.1,
             py::arg("is_water") = false)
        .def_readwrite("x",        &Particle::x)
        .def_readwrite("y",        &Particle::y)
        .def_readwrite("z",        &Particle::z)
        .def_readwrite("charge",   &Particle::charge)
        .def_readwrite("radius",   &Particle::radius)
        .def_readwrite("epsilon",  &Particle::epsilon)
        .def_readwrite("is_water", &Particle::is_water);

    py::class_<PhysicsEngine>(m, "PhysicsEngine")
        .def(py::init<>())
        // py::call_guard<py::gil_scoped_release>:
        // GIL을 해제하여 QThread 호출 중 Qt 이벤트 루프가 멈추지 않게 함
        .def("calculate_potential",     &PhysicsEngine::calculate_potential,
             py::arg("particles"),
             py::call_guard<py::gil_scoped_release>())
        .def("generate_ensemble",       &PhysicsEngine::generate_ensemble,
             py::arg("initial_state"),
             py::arg("n_candidates"),
             py::arg("steps_per_cand"),
             py::arg("temperature") = 0.6,
             py::arg("max_disp")    = 0.5,
             py::call_guard<py::gil_scoped_release>())
        .def("lowest_energy_structure", &PhysicsEngine::lowest_energy_structure,
             py::arg("ensemble"),
             py::call_guard<py::gil_scoped_release>())
        .def("num_threads",             &PhysicsEngine::num_threads);
}