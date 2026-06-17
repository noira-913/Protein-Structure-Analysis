/*
 * analysis_ext.cpp  —  ALMA landscape analysis C++ backend
 * ──────────────────────────────────────────────────────────
 * Exposes two functions as the `protein_analysis` Python extension:
 *
 *   compute_rqa(coords, eps_quantile=0.15, min_line=2) → (DET, LAM)
 *     Full O(N²) recurrence quantification — pairwise distances + diagonal
 *     and vertical line scans — all in C++, OpenMP-parallel when available.
 *
 *   compute_acf(series) → ndarray[n//2]
 *     Normalised autocorrelation function via direct O(N²) inner product,
 *     OpenMP-parallel over lag index.
 *
 * Both return identical numerical results to the pure-Python fallbacks in
 * landscape_metrics.py; the Python layer falls back silently when this
 * extension has not been built.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;


// ─── RQA (DET + LAM) ─────────────────────────────────────────────────────────

py::tuple compute_rqa(
    py::array_t<double, py::array::c_style | py::array::forcecast> coords,
    double eps_quantile,
    int    min_line)
{
    auto buf = coords.request();
    if (buf.ndim != 2)
        throw std::invalid_argument("coords must be a 2-D array (n_snaps × n_dims)");

    const int n   = static_cast<int>(buf.shape[0]);
    const int dim = static_cast<int>(buf.shape[1]);
    const double* data = static_cast<const double*>(buf.ptr);

    if (n < 4) return py::make_tuple(0.0, 0.0);

    // ── 1. Pairwise Euclidean distance matrix ──────────────────────────────
    // Outer-loop parallelism is race-free: thread i writes dist[i*n+j] and
    // dist[j*n+i] where j > i, so no two threads ever write the same cell.
    std::vector<double> dist(static_cast<std::size_t>(n) * n, 0.0);

#pragma omp parallel for schedule(dynamic) if(n > 200)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < dim; ++k) {
                double df = data[i * dim + k] - data[j * dim + k];
                s += df * df;
            }
            s = std::sqrt(s);
            dist[i * n + j] = s;
            dist[j * n + i] = s;
        }
    }

    // ── 2. eps = eps_quantile of upper-triangle distances ──────────────────
    std::vector<double> upper;
    upper.reserve(static_cast<std::size_t>(n) * (n - 1) / 2);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            upper.push_back(dist[i * n + j]);

    auto kidx = static_cast<std::ptrdiff_t>(eps_quantile * static_cast<double>(upper.size()));
    if (kidx >= static_cast<std::ptrdiff_t>(upper.size()))
        kidx = static_cast<std::ptrdiff_t>(upper.size()) - 1;
    std::nth_element(upper.begin(), upper.begin() + kidx, upper.end());
    const double eps = upper[static_cast<std::size_t>(kidx)];

    // ── 3. Recurrence matrix ───────────────────────────────────────────────
    std::vector<std::int8_t> R(static_cast<std::size_t>(n) * n, 0);
    long long total_rec = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j && dist[i * n + j] < eps) {
                R[i * n + j] = 1;
                ++total_rec;
            }

    if (total_rec == 0) return py::make_tuple(0.0, 0.0);

    // ── 4. DET: diagonal line scan ─────────────────────────────────────────
    long long diag_pts = 0;
#pragma omp parallel for reduction(+:diag_pts) schedule(dynamic) if(n > 200)
    for (int d = -(n - min_line); d <= n - min_line; ++d) {
        const int r0  = (d < 0) ? -d : 0;
        const int c0  = (d >= 0) ?  d : 0;
        const int len = n - (d < 0 ? -d : d);
        int run = 0, pts = 0;
        for (int k = 0; k < len; ++k) {
            if (R[(r0 + k) * n + (c0 + k)]) {
                ++run;
            } else {
                if (run >= min_line) pts += run;
                run = 0;
            }
        }
        if (run >= min_line) pts += run;
        diag_pts += pts;
    }

    // ── 5. LAM: vertical line scan ─────────────────────────────────────────
    long long vert_pts = 0;
#pragma omp parallel for reduction(+:vert_pts) if(n > 200)
    for (int col = 0; col < n; ++col) {
        int run = 0, pts = 0;
        for (int row = 0; row < n; ++row) {
            if (R[row * n + col]) {
                ++run;
            } else {
                if (run >= min_line) pts += run;
                run = 0;
            }
        }
        if (run >= min_line) pts += run;
        vert_pts += pts;
    }

    const double det = std::min(1.0, static_cast<double>(diag_pts) / total_rec);
    const double lam = std::min(1.0, static_cast<double>(vert_pts) / total_rec);
    return py::make_tuple(det, lam);
}


// ─── ACF ─────────────────────────────────────────────────────────────────────

py::array_t<double> compute_acf(
    py::array_t<double, py::array::c_style | py::array::forcecast> series)
{
    auto buf = series.request();
    if (buf.ndim != 1)
        throw std::invalid_argument("series must be a 1-D array");

    const int n   = static_cast<int>(buf.shape[0]);
    const double* q = static_cast<const double*>(buf.ptr);
    const int max_lag = n / 2;

    auto result = py::array_t<double>(max_lag);
    double* out = static_cast<double*>(result.request().ptr);

    double q2 = 0.0;
    for (int i = 0; i < n; ++i) q2 += q[i] * q[i];

    if (q2 < 1e-12) {
        std::fill(out, out + max_lag, 1.0);
        return result;
    }

    // Parallelising over lag is safe: each lag writes to a unique out[lag].
#pragma omp parallel for if(max_lag > 100)
    for (int lag = 0; lag < max_lag; ++lag) {
        double s = 0.0;
        for (int i = 0; i < n - lag; ++i)
            s += q[i] * q[i + lag];
        out[lag] = s / q2;
    }
    return result;
}


// ─── module ───────────────────────────────────────────────────────────────────

PYBIND11_MODULE(protein_analysis, m) {
    m.doc() = "ALMA landscape analysis C++ backend (OpenMP-accelerated RQA + ACF)";

    m.def("compute_rqa", &compute_rqa,
          py::arg("coords"),
          py::arg("eps_quantile") = 0.15,
          py::arg("min_line")     = 2,
          "Recurrence Quantification Analysis.\n\n"
          "coords : (n_snaps, n_dims) float64 array (e.g. PCA layout).\n"
          "Returns (DET, LAM) — recurrence determinism and laminarity.");

    m.def("compute_acf", &compute_acf,
          py::arg("series"),
          "Normalised autocorrelation function of a mean-subtracted 1-D series.\n"
          "Returns float64 ndarray of length n//2.");
}
