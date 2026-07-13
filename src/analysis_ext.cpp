/*
 * analysis_ext.cpp  —  ALMA ensemble-metrics + knot-analysis C++ backend
 * ──────────────────────────────────────────────────────────
 * Exposes the `protein_analysis` Python extension. Two groups of functions:
 *
 * IDP ensemble-characterization metrics (IMPROVEMENTS.md item #7) —
 * accelerates the two O(n_ca^2)-per-snapshot loops in gui_main.py; everything
 * else there (Rg, end-to-end distance, the log-log scaling-law fit, the
 * contact-variance post-processing) is already sub-0.1s and stays in Python.
 *
 *   compute_mean_dist_matrix(coords) → (n_ca, n_ca) ndarray
 *     Ensemble-mean Cα-Cα distance matrix — the heavy part of
 *     _compute_internal_scaling's polymer-scaling-law fit.
 *
 *   compute_contact_freq_matrix(coords, cutoff) → (n_ca, n_ca) ndarray
 *     Fraction of snapshots each residue pair is within `cutoff` Å of each
 *     other — the heavy part of _compute_contact_map.
 *
 * Backbone-knot classification (knot_analysis.py) — accelerates the two
 * genuinely O(n^2)-per-trial (or worse) Python loops in that module's KMT
 * reduction + crossing-detection pipeline, run 12+ times per real landscape
 * analysis (IMPROVEMENTS.md item #7's C++-porting follow-up):
 *
 *   kmt_reduce(poly, max_passes=200) → (m, 3) ndarray
 *     Same algorithm as knot_analysis.kmt_reduce -- port, not a rewrite;
 *     see that function's own docstring for the KMT algorithm description.
 *
 *   find_crossings(poly, axis) → (ok, under_pos, over_pos, sign)
 *     Same algorithm as knot_analysis.find_crossings, but takes the random
 *     projection axis as an explicit argument instead of an RNG -- the
 *     random draw itself stays in Python (knot_analysis._random_unit_vector)
 *     so trial-to-trial RNG consumption is identical whether this C++ path
 *     is used or not, and so the two implementations are directly,
 *     deterministically comparable given the same (poly, axis) input.
 *
 * All four return identical numerical results (within floating-point
 * rounding) to their pure-Python counterparts; the Python side falls back
 * to those loops silently when this extension has not been built.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace std;

// ─── shared 3-vector helper (knot-analysis functions only) ───────────────
namespace {
struct Vec3 {
    double x, y, z;
    Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
    double dot(const Vec3 &o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3 &o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
};
}  // namespace


// Shared coords-array validation + shape extraction. Both functions take
// the same (n_snaps, n_ca, 3) layout — the Python side already builds this
// once from LandscapeWorker's pooled Particle snapshots.
static void check_coords(const py::buffer_info &buf) {
    if (buf.ndim != 3 || buf.shape[2] != 3)
        throw invalid_argument("coords must be a 3-D array (n_snaps, n_ca, 3)");
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
    fill(sum_mat, sum_mat + static_cast<size_t>(n_ca) * n_ca, 0.0);

    if (n_snaps == 0 || n_ca == 0) return result;

    for (int s = 0; s < n_snaps; ++s) {
        const double* snap = data + static_cast<size_t>(s) * n_ca * 3;
#pragma omp parallel for schedule(dynamic) if(n_ca > 100)
        for (int i = 0; i < n_ca; ++i) {
            for (int j = i + 1; j < n_ca; ++j) {
                double dx = snap[i * 3 + 0] - snap[j * 3 + 0];
                double dy = snap[i * 3 + 1] - snap[j * 3 + 1];
                double dz = snap[i * 3 + 2] - snap[j * 3 + 2];
                double d = sqrt(dx * dx + dy * dy + dz * dz);
                sum_mat[i * n_ca + j] += d;
                sum_mat[j * n_ca + i] += d;
            }
        }
    }

    const double inv_n = 1.0 / static_cast<double>(n_snaps);
    for (size_t k = 0; k < static_cast<size_t>(n_ca) * n_ca; ++k)
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
    fill(freq, freq + static_cast<size_t>(n_ca) * n_ca, 0.0);

    if (n_snaps == 0 || n_ca == 0) return result;

    const double cutoff2 = cutoff * cutoff;
    for (int s = 0; s < n_snaps; ++s) {
        const double* snap = data + static_cast<size_t>(s) * n_ca * 3;
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
    for (size_t k = 0; k < static_cast<size_t>(n_ca) * n_ca; ++k)
        freq[k] *= inv_n;

    return result;
}


// ─── KMT (Koniaris-Muthukumar-Taylor) triangle-elimination reduction ──────
//
// Direct port of knot_analysis.py's kmt_reduce/_segment_triangle_intersect
// (Möller-Trumbore ray/triangle test). Not parallelized: each pass's vertex
// deletions are sequentially dependent (removing vertex i changes the
// neighbor set for the next check), same reason the Python original is a
// single-threaded loop -- the win here is purely from replacing per-element
// numpy call overhead (np.cross/np.dot on 3-vectors, ~10-100ns of Python/
// numpy dispatch each) with raw double arithmetic, not from parallelism.
static bool segment_triangle_intersect(const Vec3 &p0, const Vec3 &p1,
                                        const Vec3 &a, const Vec3 &b, const Vec3 &c,
                                        double eps = 1e-9) {
    Vec3 d = p1 - p0;
    Vec3 e1 = b - a;
    Vec3 e2 = c - a;
    Vec3 pvec = d.cross(e2);
    double det = e1.dot(pvec);
    if (abs(det) < eps) return false;  // segment parallel to triangle plane
    double inv_det = 1.0 / det;
    Vec3 tvec = p0 - a;
    double u = tvec.dot(pvec) * inv_det;
    if (u < -eps || u > 1 + eps) return false;
    Vec3 qvec = tvec.cross(e1);
    double v = d.dot(qvec) * inv_det;
    if (v < -eps || u + v > 1 + eps) return false;
    double t = e2.dot(qvec) * inv_det;
    return eps < t && t < 1 - eps;
}

py::array_t<double> kmt_reduce(
    py::array_t<double, py::array::c_style | py::array::forcecast> poly,
    int max_passes = 200)
{
    auto buf = poly.request();
    if (buf.ndim != 2 || buf.shape[1] != 3)
        throw invalid_argument("poly must be a (n,3) array");
    const int n0 = static_cast<int>(buf.shape[0]);
    const double *data = static_cast<const double *>(buf.ptr);

    vector<Vec3> pts(n0);
    for (int i = 0; i < n0; ++i)
        pts[i] = {data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2]};

    for (int pass = 0; pass < max_passes; ++pass) {
        int n = static_cast<int>(pts.size());
        if (n <= 3) break;
        bool removed_any = false;
        int i = 0;
        // Mirrors the Python `while i < len(pts) and len(pts) > 3` loop
        // exactly, including NOT incrementing i after a deletion (the
        // vertex that shifted into position i gets re-examined immediately
        // -- intentional: removing one vertex can make its new neighbor
        // removable too, within the same pass).
        while (i < static_cast<int>(pts.size()) && pts.size() > 3) {
            n = static_cast<int>(pts.size());
            const int im1 = (i - 1 + n) % n;
            const int ip1 = (i + 1) % n;
            const Vec3 prev_p = pts[im1];
            const Vec3 cur_p  = pts[i];
            const Vec3 next_p = pts[ip1];
            bool blocked = false;
            for (int j = 0; j < n; ++j) {
                if (j == im1 || j == i || j == ip1) continue;
                const int jp1 = (j + 1) % n;
                if (jp1 == im1) continue;
                if (segment_triangle_intersect(pts[j], pts[jp1], prev_p, cur_p, next_p)) {
                    blocked = true;
                    break;
                }
            }
            if (!blocked) {
                pts.erase(pts.begin() + i);
                removed_any = true;
            } else {
                ++i;
            }
        }
        if (!removed_any) break;
    }

    auto result = py::array_t<double>({static_cast<int>(pts.size()), 3});
    double *out = static_cast<double *>(result.request().ptr);
    for (size_t i = 0; i < pts.size(); ++i) {
        out[i * 3 + 0] = pts[i].x;
        out[i * 3 + 1] = pts[i].y;
        out[i * 3 + 2] = pts[i].z;
    }
    return result;
}


// ─── Crossing detection on a 2D projection ────────────────────────────────
//
// Direct port of knot_analysis.py's find_crossings/_segment_intersect_2d.
// Takes the projection axis as an explicit argument (drawn in Python via
// _random_unit_vector before calling this) rather than an RNG, so the two
// implementations are deterministically comparable given the same input --
// see this file's header comment. O(n^2) pair scan, embarrassingly
// parallel (each pair only ever appends to its own local result, no shared
// mutable state) but left single-threaded here: n is the KMT-*reduced*
// vertex count, typically small enough (order 1-100) that OpenMP's
// per-thread dispatch overhead would dominate over the actual work --
// unlike compute_mean_dist_matrix/compute_contact_freq_matrix above, where
// n_ca reaches into the hundreds-to-thousands.
py::tuple find_crossings(
    py::array_t<double, py::array::c_style | py::array::forcecast> poly,
    py::array_t<double, py::array::c_style | py::array::forcecast> axis_arr)
{
    auto buf = poly.request();
    if (buf.ndim != 2 || buf.shape[1] != 3)
        throw invalid_argument("poly must be a (n,3) array");
    const int n = static_cast<int>(buf.shape[0]);
    const double *data = static_cast<const double *>(buf.ptr);

    auto axbuf = axis_arr.request();
    if (axbuf.size != 3)
        throw invalid_argument("axis must be a length-3 array");
    const double *axd = static_cast<const double *>(axbuf.ptr);
    const Vec3 axis = {axd[0], axd[1], axd[2]};

    const Vec3 tmp = (abs(axis.x) < 0.9) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 u_ = axis.cross(tmp);
    const double u_norm = sqrt(u_.dot(u_));
    u_ = {u_.x / u_norm, u_.y / u_norm, u_.z / u_norm};
    const Vec3 v_ = axis.cross(u_);

    vector<double> proj_u(n), proj_v(n), depth(n);
    for (int i = 0; i < n; ++i) {
        const Vec3 p = {data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2]};
        proj_u[i] = p.dot(u_);
        proj_v[i] = p.dot(v_);
        depth[i]  = p.dot(axis);
    }

    auto degenerate = [] {
        return py::make_tuple(false, py::array_t<double>(0),
                               py::array_t<double>(0), py::array_t<int>(0));
    };

    vector<double> under_pos, over_pos;
    vector<int> sign;
    const double eps = 1e-9;

    for (int i = 0; i < n; ++i) {
        const int ip1 = (i + 1) % n;
        const double p0x = proj_u[i], p0y = proj_v[i];
        const double d1x = proj_u[ip1] - p0x, d1y = proj_v[ip1] - p0y;
        for (int j = i + 1; j < n; ++j) {
            const int jp1 = (j + 1) % n;
            if (j == i || jp1 == i || ip1 == j) continue;  // adjacent segments
            const double q0x = proj_u[j], q0y = proj_v[j];
            const double d2x = proj_u[jp1] - q0x, d2y = proj_v[jp1] - q0y;

            const double denom = d1x * d2y - d1y * d2x;
            if (abs(denom) < eps) continue;
            const double diffx = q0x - p0x, diffy = q0y - p0y;
            const double t = (diffx * d2y - diffy * d2x) / denom;
            const double u = (diffx * d1y - diffy * d1x) / denom;
            if (!(t > eps && t < 1 - eps && u > eps && u < 1 - eps)) continue;

            const double depth_i = depth[i] * (1 - t) + depth[ip1] * t;
            const double depth_j = depth[j] * (1 - u) + depth[jp1] * u;
            if (abs(depth_i - depth_j) < 1e-9) return degenerate();

            const double cross_z = d1x * d2y - d1y * d2x;
            if (abs(cross_z) < 1e-12) return degenerate();

            double op, unp;
            int sgn;
            if (depth_i > depth_j) {
                op = i + t; unp = j + u;
                sgn = (cross_z > 0) ? 1 : -1;
            } else {
                op = j + u; unp = i + t;
                sgn = (cross_z < 0) ? 1 : -1;
            }
            under_pos.push_back(unp);
            over_pos.push_back(op);
            sign.push_back(sgn);
        }
    }

    py::array_t<double> under_arr(under_pos.size());
    py::array_t<double> over_arr(over_pos.size());
    py::array_t<int> sign_arr(sign.size());
    copy(under_pos.begin(), under_pos.end(), static_cast<double *>(under_arr.request().ptr));
    copy(over_pos.begin(), over_pos.end(), static_cast<double *>(over_arr.request().ptr));
    copy(sign.begin(), sign.end(), static_cast<int *>(sign_arr.request().ptr));
    return py::make_tuple(true, under_arr, over_arr, sign_arr);
}


// ─── module ───────────────────────────────────────────────────────────────

PYBIND11_MODULE(protein_analysis, m) {
    m.doc() = "ALMA ensemble-metrics + knot-analysis C++ backend (OpenMP-accelerated)";

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

    m.def("kmt_reduce", &kmt_reduce,
          py::arg("poly"), py::arg("max_passes") = 200,
          "KMT triangle-elimination reduction of a closed 3D polygon.\n\n"
          "poly : (n, 3) float64 array.\n"
          "Returns (m, 3) float64 ndarray, m <= n.");

    m.def("find_crossings", &find_crossings,
          py::arg("poly"), py::arg("axis"),
          "2D-projected self-crossing detection.\n\n"
          "poly : (n, 3) float64 array. axis : length-3 float64 array "
          "(the random projection direction -- drawn in Python).\n"
          "Returns (ok, under_pos, over_pos, sign): ok=False means a "
          "degenerate projection (caller should retry with a new axis).");
}
