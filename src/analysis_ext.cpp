/*
 * analysis_ext.cpp  —  ALMA ensemble-metrics C++ backend
 * ──────────────────────────────────────────────────────────
 * Exposes two functions as the `protein_analysis` Python extension,
 * accelerating the two O(n_ca^2)-per-snapshot loops in gui_main.py's
 * IDP ensemble-characterization metrics (IMPROVEMENTS.md item #7) —
 * everything else there (Rg, end-to-end distance, the log-log scaling-law
 * fit, the contact-variance post-processing) is already sub-0.1s and stays
 * in Python.
 *
 *   compute_mean_dist_matrix(coords) → (n_ca, n_ca) ndarray
 *     Ensemble-mean Cα-Cα distance matrix — the heavy part of
 *     _compute_internal_scaling's polymer-scaling-law fit.
 *
 *   compute_contact_freq_matrix(coords, cutoff) → (n_ca, n_ca) ndarray
 *     Fraction of snapshots each residue pair is within `cutoff` Å of each
 *     other — the heavy part of _compute_contact_map.
 *
 * Both return identical numerical results (within floating-point rounding)
 * to gui_main.py's pure-Python per-snapshot loops; gui_main.py falls back
 * to those loops silently when this extension has not been built.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;


// Shared coords-array validation + shape extraction. Both functions take
// the same (n_snaps, n_ca, 3) layout — the Python side already builds this
// once from LandscapeWorker's pooled Particle snapshots.
static void check_coords(const py::buffer_info &buf) {
    if (buf.ndim != 3 || buf.shape[2] != 3)
        throw std::invalid_argument("coords must be a 3-D array (n_snaps, n_ca, 3)");
}


// ─── Ensemble-mean Cα-Cα distance matrix ──────────────────────────────────
//
// For each snapshot (sequential outer loop — different snapshots all
// accumulate into the SAME shared matrix, so parallelizing over snapshots
// would need a reduction), the O(n_ca^2) inner pairwise-distance pass is
// parallelized over residue index i: thread i only ever writes
// sum_mat[i*n+j]/sum_mat[j*n+i] for j > i, so no two threads touch the same
// cell within one snapshot's parallel region — same race-free argument as
// analysis_ext.cpp's original compute_rqa distance-matrix pass, just
// accumulating (+=) across snapshots instead of assigning once.
py::array_t<double> compute_mean_dist_matrix(
    py::array_t<double, py::array::c_style | py::array::forcecast> coords)
{
    auto buf = coords.request();
    check_coords(buf);

    const int n_snaps = static_cast<int>(buf.shape[0]);
    const int n_ca     = static_cast<int>(buf.shape[1]);
    const double* data = static_cast<const double*>(buf.ptr);

    auto result = py::array_t<double>({n_ca, n_ca});
    double* sum_mat = static_cast<double*>(result.request().ptr);
    std::fill(sum_mat, sum_mat + static_cast<std::size_t>(n_ca) * n_ca, 0.0);

    if (n_snaps == 0 || n_ca == 0) return result;

    for (int s = 0; s < n_snaps; ++s) {
        const double* snap = data + static_cast<std::size_t>(s) * n_ca * 3;
#pragma omp parallel for schedule(dynamic) if(n_ca > 100)
        for (int i = 0; i < n_ca; ++i) {
            for (int j = i + 1; j < n_ca; ++j) {
                double dx = snap[i * 3 + 0] - snap[j * 3 + 0];
                double dy = snap[i * 3 + 1] - snap[j * 3 + 1];
                double dz = snap[i * 3 + 2] - snap[j * 3 + 2];
                double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                sum_mat[i * n_ca + j] += d;
                sum_mat[j * n_ca + i] += d;
            }
        }
    }

    const double inv_n = 1.0 / static_cast<double>(n_snaps);
    for (std::size_t k = 0; k < static_cast<std::size_t>(n_ca) * n_ca; ++k)
        sum_mat[k] *= inv_n;

    return result;
}


// ─── Ensemble contact-frequency matrix ────────────────────────────────────
//
// Same accumulation pattern as compute_mean_dist_matrix, counting a
// contact (distance < cutoff) instead of summing raw distance.
py::array_t<double> compute_contact_freq_matrix(
    py::array_t<double, py::array::c_style | py::array::forcecast> coords,
    double cutoff)
{
    auto buf = coords.request();
    check_coords(buf);

    const int n_snaps = static_cast<int>(buf.shape[0]);
    const int n_ca     = static_cast<int>(buf.shape[1]);
    const double* data = static_cast<const double*>(buf.ptr);

    auto result = py::array_t<double>({n_ca, n_ca});
    double* freq = static_cast<double*>(result.request().ptr);
    std::fill(freq, freq + static_cast<std::size_t>(n_ca) * n_ca, 0.0);

    if (n_snaps == 0 || n_ca == 0) return result;

    const double cutoff2 = cutoff * cutoff;
    for (int s = 0; s < n_snaps; ++s) {
        const double* snap = data + static_cast<std::size_t>(s) * n_ca * 3;
#pragma omp parallel for schedule(dynamic) if(n_ca > 100)
        for (int i = 0; i < n_ca; ++i) {
            for (int j = i + 1; j < n_ca; ++j) {
                double dx = snap[i * 3 + 0] - snap[j * 3 + 0];
                double dy = snap[i * 3 + 1] - snap[j * 3 + 1];
                double dz = snap[i * 3 + 2] - snap[j * 3 + 2];
                if (dx * dx + dy * dy + dz * dz < cutoff2) {
                    freq[i * n_ca + j] += 1.0;
                    freq[j * n_ca + i] += 1.0;
                }
            }
        }
    }

    const double inv_n = 1.0 / static_cast<double>(n_snaps);
    for (std::size_t k = 0; k < static_cast<std::size_t>(n_ca) * n_ca; ++k)
        freq[k] *= inv_n;

    return result;
}


// ─── module ───────────────────────────────────────────────────────────────

PYBIND11_MODULE(protein_analysis, m) {
    m.doc() = "ALMA ensemble-metrics C++ backend (OpenMP-accelerated)";

    m.def("compute_mean_dist_matrix", &compute_mean_dist_matrix,
          py::arg("coords"),
          "Ensemble-mean Cα-Cα distance matrix.\n\n"
          "coords : (n_snaps, n_ca, 3) float64 array.\n"
          "Returns (n_ca, n_ca) float64 ndarray.");

    m.def("compute_contact_freq_matrix", &compute_contact_freq_matrix,
          py::arg("coords"),
          py::arg("cutoff") = 8.0,
          "Fraction of snapshots each residue pair is within `cutoff` A.\n\n"
          "coords : (n_snaps, n_ca, 3) float64 array.\n"
          "Returns (n_ca, n_ca) float64 ndarray (diagonal is 0).");
}
