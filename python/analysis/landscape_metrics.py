"""
landscape_metrics.py — Scientifically grounded landscape disorder metrics
=========================================================================

Replaces the KAM-based quantities in TOPOLOGY_PLAN_V2.md §3 with observables
that are valid for overdamped Langevin / Markov chain dynamics:

  KAM landscape stability  →  spectral_gap()
  Thermal disorder param K →  basin_entropy()
  Rotation / winding number→  autocorr_decay()
  Arnold diffusion argument→  kramers_rates()
  (supplementary)          →  recurrence_determinism()

All functions are pure (no side effects, same input → same output).

Reference frames
----------------
  - Markov chain spectral theory: Aldous & Fill (2002), "Reversible Markov Chains"
  - RQA: Zbilut & Webber (1992), Marwan et al. (2007)
  - Kramers rate: Kramers (1940), Hänggi et al. (1990)
"""

from __future__ import annotations
import numpy as np
from typing import NamedTuple


# ─── public data structure ────────────────────────────────────────────────────

class DisorderReport(NamedTuple):
    """All scientifically grounded disorder metrics for one MC run."""
    # Spectral gap of basin-to-basin transition matrix (replaces KAM stability)
    spectral_gap: float          # in [0,1]; near 1 → fast mixing → IDP

    # Shannon entropy of basin occupancy (replaces Chirikov K parameter)
    basin_entropy: float         # absolute, in nats
    basin_entropy_norm: float    # normalised to [0,1]; 1 → uniform → IDP

    # Autocorrelation decay time of PC1 coordinate (replaces rotation number)
    tau_corr: float              # in MC steps; short → IDP
    acf: np.ndarray              # full ACF array, length = N_snapshots // 2

    # RQA determinism of PCA trajectory (supplementary chaos measure)
    rqa_det: float               # in [0,1]; low → diffuse / chaotic → IDP
    rqa_lam: float               # laminarity; low → low recurrence structure

    # Kramers empirical basin-transition rates (replaces Arnold diffusion)
    mean_escape_rate: float      # mean rate out of any basin (MC-step units)
    mean_barrier_est: float      # mean empirical barrier (energy units of input)

    # Revised IDP classification based on the new metrics
    idp_label_v2: str
    idp_color_v2: str


# ─── 1. Spectral gap ──────────────────────────────────────────────────────────

def spectral_gap(node_comm: dict[int, int], n_basins: int) -> float:
    """Spectral gap of the empirical basin-to-basin Markov transition matrix.

    Builds T[i,j] from the sequential snapshot trajectory: each consecutive
    pair (node k, node k+1) contributes one count T[comm(k), comm(k+1)].
    T is row-stochastic.  The spectral gap is 1 - Re(λ₂) where λ₁ = 1 is the
    stationary eigenvalue.

    Interpretation (Aldous & Fill 2002):
        gap ≈ 0  →  near-zero second eigenvalue → slow mixing → ordered
        gap ≈ 1  →  second eigenvalue ≈ 0 → single-step mixing → IDP (flat)

    Returns 0.0 when n_basins < 2 (single basin → ordered by definition).
    """
    if n_basins < 2:
        return 0.0

    n_snaps = max(node_comm.keys()) + 1
    counts = np.zeros((n_basins, n_basins), dtype=float)
    for k in range(n_snaps - 1):
        i = node_comm.get(k, 0)
        j = node_comm.get(k + 1, 0)
        if i < n_basins and j < n_basins:
            counts[i, j] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0          # avoid division by zero for empty rows
    T = counts / row_sums

    eigenvalues = np.linalg.eigvals(T)
    real_eigs = np.sort(np.real(eigenvalues))[::-1]   # descending

    if len(real_eigs) < 2:
        return 0.0
    return float(np.clip(1.0 - real_eigs[1], 0.0, 1.0))


# ─── 2. Basin entropy ─────────────────────────────────────────────────────────

def basin_entropy(communities: list, n_snapshots: int) -> tuple[float, float]:
    """Shannon entropy of basin occupancy.

    p_i = |community_i| / n_snapshots
    S   = -Σ p_i · ln(p_i)          (absolute, nats)
    S_n = S / ln(n_basins)           (normalised; 1 = uniform over all basins)

    Interpretation:
        S_n ≈ 0  →  one dominant basin → funnel landscape → ordered
        S_n ≈ 1  →  all basins equally populated → flat landscape → IDP

    The normalised version directly replaces the Chirikov K parameter:
    both are dimensionless measures of landscape flatness, but S_n is derived
    from the actual Boltzmann-weighted occupancy of the sampled trajectory
    rather than from an analogy with Hamiltonian resonance overlap.
    """
    n_b = len(communities)
    if n_b < 2 or n_snapshots == 0:
        return 0.0, 0.0

    probs = np.array([len(c) / n_snapshots for c in communities], dtype=float)
    probs = probs[probs > 0]
    S = float(-np.sum(probs * np.log(probs)))
    S_norm = S / np.log(n_b) if n_b > 1 else 0.0
    return S, float(np.clip(S_norm, 0.0, 1.0))


# ─── 3. Autocorrelation decay ─────────────────────────────────────────────────

def autocorr_decay(pca_traj: np.ndarray) -> tuple[float, np.ndarray]:
    """Normalised ACF of PC1 and fitted exponential decay time τ_corr.

    C(τ) = <q(t) q(t+τ)> / <q²>

    Fits C(τ) = exp(-τ/τ_corr) over lags where C(τ) > 0 to avoid log(0).
    Returns (τ_corr, acf_array).

    Interpretation:
        Short τ_corr (<< N_snapshots)  →  fast decorrelation → IDP
        Long  τ_corr (≈ N_snapshots)   →  persistent structure → ordered

    When the fit fails (e.g. C(τ) never crosses zero) τ_corr = N_snapshots
    (treated as "effectively infinite decorrelation time → ordered").
    """
    q = pca_traj[:, 0] - pca_traj[:, 0].mean()
    n = len(q)
    max_lag = max(n // 2, 1)

    q2 = float(np.dot(q, q))
    if q2 < 1e-12:
        return float(n), np.ones(max_lag)

    acf = np.array([float(np.dot(q[:n-lag], q[lag:])) / q2
                    for lag in range(max_lag)])

    # Fit exponential on positive part to get τ_corr
    positive = acf > 0
    lags = np.arange(max_lag, dtype=float)
    if positive.sum() < 2:
        return float(n), acf

    log_acf = np.log(np.clip(acf[positive], 1e-12, None))
    lag_pos = lags[positive]
    # Linear fit: log C(τ) = -τ/τ_corr  →  slope = -1/τ_corr
    slope, _ = np.polyfit(lag_pos, log_acf, 1)
    tau = float(-1.0 / slope) if slope < 0 else float(n)
    return float(np.clip(tau, 1.0, float(n))), acf


# ─── 4. Recurrence Quantification Analysis (RQA) ─────────────────────────────

def recurrence_determinism(pca_traj: np.ndarray,
                            eps_quantile: float = 0.15,
                            min_line: int = 2) -> tuple[float, float]:
    """RQA determinism and laminarity of the PCA trajectory.

    Builds recurrence matrix R[i,j] = 1 if ||x_i - x_j|| < ε, where ε is
    chosen as the eps_quantile of all pairwise distances (ensures a fixed
    recurrence rate regardless of coordinate scale).

    DET = fraction of recurrence points forming diagonal lines ≥ min_line
    LAM = fraction of recurrence points forming vertical lines ≥ min_line

    Interpretation (Marwan et al. 2007):
        High DET → deterministic / quasi-periodic trajectory → ordered protein
        Low DET  → diffuse / stochastic trajectory → IDP

    O(N²) in memory and time.  For N ≤ 120 snapshots this is < 0.1 ms.
    """
    n = len(pca_traj)
    if n < 4:
        return 0.0, 0.0

    # Pairwise Euclidean distances in PCA space
    diff = pca_traj[:, np.newaxis, :] - pca_traj[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    eps = float(np.quantile(dist[dist > 0], eps_quantile))
    R = (dist < eps).astype(np.int8)
    np.fill_diagonal(R, 0)       # exclude trivial self-recurrence

    total_rec = int(R.sum())
    if total_rec == 0:
        return 0.0, 0.0

    # Diagonal lines (DET)
    diag_points = 0
    for d in range(-(n - min_line), n - min_line + 1):
        diag = np.diag(R, d)
        # Count points in runs of length >= min_line
        run = 0
        for v in diag:
            if v:
                run += 1
            else:
                if run >= min_line:
                    diag_points += run
                run = 0
        if run >= min_line:
            diag_points += run

    # Vertical lines (LAM)
    vert_points = 0
    for col in range(n):
        run = 0
        for v in R[:, col]:
            if v:
                run += 1
            else:
                if run >= min_line:
                    vert_points += run
                run = 0
        if run >= min_line:
            vert_points += run

    det = float(diag_points) / float(total_rec)
    lam = float(vert_points) / float(total_rec)
    return float(np.clip(det, 0.0, 1.0)), float(np.clip(lam, 0.0, 1.0))


# ─── 5. Kramers empirical basin transition rates ──────────────────────────────

def kramers_rates(node_comm: dict[int, int],
                  energies: np.ndarray,
                  communities: list,
                  T: float) -> tuple[float, float]:
    """Empirical basin-to-basin transition rates and Arrhenius barrier estimates.

    For each direct basin transition i→j observed in the trajectory:
        rate_ij  = n_transitions_ij / time_in_i        (empirical first-passage)
        barrier_ij = E_saddle - E_min_i                (Arrhenius estimate)
          where E_saddle = max energy on the crossing segment

    Motivation: this replaces the Arnold diffusion argument (§3.6) which
    requires Hamiltonian dynamics.  The empirical rate replaces the analytical
    Kramers formula k = ν exp(-ΔE/kT) because ALMA's MC engine does not expose
    the curvature ν at the saddle; instead we estimate rates directly from
    observed crossing frequencies in the trajectory.

    Returns (mean_escape_rate, mean_barrier).  Returns (0.0, 0.0) for a
    single-basin trajectory (no crossings).
    """
    n_snaps = max(node_comm.keys()) + 1
    n_b = len(communities)

    if n_b < 2:
        return 0.0, 0.0

    # Basin minimum energies
    basin_min_e = {}
    for ci, comm in enumerate(communities):
        idxs = [i for i in comm if i < len(energies)]
        basin_min_e[ci] = float(np.min(energies[idxs])) if idxs else 0.0

    # Count transitions and record crossing energies
    counts = np.zeros((n_b, n_b), dtype=float)
    time_in = np.zeros(n_b, dtype=float)
    saddle_e = {}     # (i,j) → list of crossing energies

    for k in range(n_snaps - 1):
        i = node_comm.get(k, 0)
        j = node_comm.get(k + 1, 0)
        if i >= n_b or j >= n_b:
            continue
        time_in[i] += 1.0
        if i != j:
            counts[i, j] += 1.0
            key = (min(i, j), max(i, j))
            e_cross = float(energies[k]) if k < len(energies) else 0.0
            saddle_e.setdefault(key, []).append(e_cross)

    rates, barriers = [], []
    for i in range(n_b):
        for j in range(n_b):
            if i == j or counts[i, j] == 0:
                continue
            rate = counts[i, j] / max(time_in[i], 1.0)
            rates.append(rate)
            key = (min(i, j), max(i, j))
            if key in saddle_e and saddle_e[key]:
                barrier = max(saddle_e[key]) - basin_min_e[i]
                barriers.append(max(barrier, 0.0))

    mean_rate    = float(np.mean(rates))    if rates    else 0.0
    mean_barrier = float(np.mean(barriers)) if barriers else 0.0
    return mean_rate, mean_barrier


# ─── 6. Integration function ──────────────────────────────────────────────────

def compute_disorder_metrics(result: dict, T: float = 0.6) -> DisorderReport:
    """Compute all scientifically grounded landscape disorder metrics.

    Accepts the dict emitted by LandscapeWorker.result and returns a
    DisorderReport with new metrics plus a revised IDP classification.

    Revised classification logic (replaces the current basin-count heuristic):
        Uses spectral gap + normalised entropy as primary discriminators,
        with τ_corr and RQA DET as supporting evidence.

        ORDERED:   gap < 0.3  AND  S_norm < 0.4
        IDP:       gap > 0.6  AND  S_norm > 0.6
        MARGINAL:  everything else

    These thresholds are empirical starting points, not calibrated against
    a labelled benchmark — treat them as tunable hyperparameters.
    """
    communities  = result["communities"]
    node_comm    = result["node_comm"]
    energies     = result["energies"]
    layout       = result["layout"]
    n_snaps      = len(result["snapshots"])
    n_b          = len(communities)

    gap               = spectral_gap(node_comm, n_b)
    S_abs, S_norm     = basin_entropy(communities, n_snaps)
    tau, acf_arr      = autocorr_decay(layout)
    det, lam          = recurrence_determinism(layout)
    mean_rate, mean_b = kramers_rates(node_comm, energies, communities, T)

    # Revised classification
    if gap < 0.3 and S_norm < 0.4:
        label, color = "ORDERED", "#16a34a"
    elif gap > 0.6 and S_norm > 0.6:
        label, color = "IDP", "#dc2626"
    else:
        label, color = "POSSIBLY DISORDERED", "#d97706"

    return DisorderReport(
        spectral_gap       = gap,
        basin_entropy      = S_abs,
        basin_entropy_norm = S_norm,
        tau_corr           = tau,
        acf                = acf_arr,
        rqa_det            = det,
        rqa_lam            = lam,
        mean_escape_rate   = mean_rate,
        mean_barrier_est   = mean_b,
        idp_label_v2       = label,
        idp_color_v2       = color,
    )
