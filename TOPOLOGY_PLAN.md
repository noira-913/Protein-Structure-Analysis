# Topological Mapping of Protein Conformational Space
## Theoretical Background & Implementation Plan

> Branch: `feature/topological-mapping`  
> Project: ALMA — Atomistic Local Motion Analyzer  
> Scope: Lyapunov chaos analysis · KAM stability · Knot theory · Jones polynomial · Khovanov homology · Chern-Simons TQFT

---

## Table of Contents

1. [Motivation & Core Thesis](#1-motivation--core-thesis)
2. [Dynamical Chaos & Lyapunov Exponents](#2-dynamical-chaos--lyapunov-exponents)
3. [KAM Theory & Conformational Stability](#3-kam-theory--conformational-stability)
4. [Algebraic Foundations — Lambda-Style](#4-algebraic-foundations--lambda-style)
5. [Category Theory Layer](#5-category-theory-layer)
6. [Knot Theory of Protein Backbones](#6-knot-theory-of-protein-backbones)
7. [Jones Polynomial via Skein Relations](#7-jones-polynomial-via-skein-relations)
8. [Khovanov Homology](#8-khovanov-homology)
9. [Chern-Simons Field Theory](#9-chern-simons-field-theory)
10. [Connecting Chaos to Topology — IDP Hypothesis](#10-connecting-chaos-to-topology--idp-hypothesis)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Module Architecture & Data Flow](#12-module-architecture--data-flow)
13. [Open Questions & Research Directions](#13-open-questions--research-directions)
14. [References](#14-references)

---

## 1. Motivation & Core Thesis

### 1.1 The Problem with Current IDP Classification

ALMA's current IDP classifier (in `gui_main.py:LandscapeWorker`) works by:

1. Running a 120 × 80 = 9,600-step MC Markov chain on the Cα coordinate space.
2. Projecting snapshots to 2D via PCA.
3. Building a sequential conformational graph (NetworkX).
4. Applying greedy modularity community detection → counting "metastable basins".
5. Classifying as IDP if ≥ 3 significant basins AND energy spread < 5 kT.

This has three fundamental weaknesses:

**Weakness 1 — Statistical**: 9,600 steps is far too short to establish reliable basin
counts. Basins found at this scale are mostly noise from the short trajectory combined
with the ad-hoc modularity threshold. The classification is highly sensitive to the
random seed of the MC chain.

**Weakness 2 — Dynamical**: counting basins is a static snapshot of the energy landscape.
It does not measure how sensitively the protein *responds to perturbation* — the hallmark
of chaos. Two proteins can have the same number of basins but one exponentially amplifies
small conformational differences (IDP) while the other contracts them back to the native
state (ordered).

**Weakness 3 — Topological**: the current approach is entirely blind to the *topology*
of the backbone fold. Two proteins can have identical energy landscapes — same number of
basins, same energy spreads — yet have completely different fold topologies: one forms a
trefoil knot in its backbone, the other does not. Knottedness is a *topological invariant*
that cannot be detected by any dynamical measurement alone.

### 1.2 The Two-Pillar Thesis

The core claim of this project:

> **IDP-ness is characterized by two independent, complementary invariants:**
>
> **(a) Dynamical invariant**: positive Lyapunov exponent λ₁ > 0 in the MC trajectory,
> meaning the conformational dynamics is chaotic — small perturbations to the initial
> structure grow exponentially rather than relaxing to a stable native fold.
>
> **(b) Topological invariant**: trivial backbone knot type — Jones polynomial V_K(t) = 1
> and Khovanov homology Kh^{i,j}(K) ≅ ℤ (unknot homology) — meaning the protein
> backbone, when closed into a loop, carries no persistent topological complexity.
>
> An ordered protein satisfies *neither*: its dynamics contracts to a stable basin
> (λ₁ ≤ 0, KAM tori intact) and its fold topology is non-trivial (non-trivial Jones
> polynomial for knotted folds, or at minimum persistent writhe pattern under projection).

These two invariants are *logically independent* — a protein could in principle be
dynamically stable but topologically trivial (a well-folded unknotted protein), or
dynamically chaotic but transiently knotted (a rare theoretical case). The joint
measurement is strictly more informative than either alone.

### 1.3 Connection to ALMA's Existing Physics

ALMA already computes four energy terms in `physics_engine.cpp`:
- GB-HCT polar solvation (generalized Born)
- Debye-Hückel screened Coulomb
- Lennard-Jones 12-6
- SASA nonpolar burial

And it already runs torsion-angle Metropolis MC (P1.5 in `IMPROVEMENTS.md`).
The topology module does *not* replace this — it reads from the MC trajectory output
(`LandscapeWorker.result` snapshots and `PipelineWorker.finished` ensemble) and
computes additional invariants on top of the existing machinery.

### 1.4 The Deep Bridge

The connection between dynamical chaos and knot topology runs through a chain of
exact mathematical results (not analogies):

```
Double-pendulum chaos
    │  (Lyapunov exponent measures rate of phase-space separation)
    ▼
Hamiltonian dynamics on 2n-dimensional phase space
    │  (Arnold-Liouville: integrable systems have invariant tori 𝕋ⁿ)
    ▼
KAM theory: invariant tori are Lagrangian submanifolds of phase space
    │  (tori are topological objects; their persistence = topological stability)
    ▼
Topology of curves and surfaces in ℝ³
    │  (protein backbone = curve in ℝ³; its topology = knot type)
    ▼
Knot invariants: Jones polynomial V_K(t) ∈ ℤ[t^{1/2}, t^{-1/2}]
    │  (Witten 1989: exact result, not analogy)
    ▼
Chern-Simons TQFT: V_K(t) = ⟨W_{1/2}(K)⟩_CS
    │  (Khovanov 2000: categorification lifts polynomial to chain complex)
    ▼
Khovanov homology: bigraded ℤ-modules Kh^{i,j}(K)
    (Euler characteristic = Jones polynomial; contains strictly more info)
```

The Witten and Khovanov steps are *theorems*, not speculations. The KAM-to-topology
step goes through the Arnold-Maslov index, which is explained in §9.4.

---

## 2. Dynamical Chaos & Lyapunov Exponents

### 2.1 Sensitivity to Initial Conditions — Formal Definition

Let Φ: X → X be the time-evolution map of a dynamical system on phase space X.
For a discrete map (like our MC chain), Φ advances the state by one step.

The **largest Lyapunov exponent** (LLE) is:

```
λ₁ = lim_{n→∞} (1/n) ln ||DΦⁿ(x₀) · v||
```

where DΦⁿ is the Jacobian of the n-step map and v is a generic tangent vector.
The limit exists and is independent of v (for Lebesgue-almost-all v) by the
multiplicative ergodic theorem (Oseledets 1968).

Practically: start two trajectories at x₀ and x₀ + ε·v̂, evolve both, measure
the separation after n steps:

```
||δxₙ|| ≈ ε · e^{λ₁ n}    (for large n, generic v̂)
```

Physical interpretation for proteins:
- λ₁ < 0: nearby conformations converge → **funnel landscape → ordered protein**
- λ₁ = 0: separation grows at most polynomially → **KAM regime, marginally stable**
- λ₁ > 0: separation grows exponentially → **chaotic, flat landscape → IDP**

### 2.2 The Lyapunov Spectrum

The full **Lyapunov spectrum** {λ₁ ≥ λ₂ ≥ ... ≥ λ_n} where n = 3N (dimension of
the Cα coordinate space for N residues) describes expansion and contraction in all
independent directions of phase space simultaneously.

**Pairing symmetry**: for Hamiltonian systems, λᵢ + λ_{n+1-i} = 0 (exponents come
in +/- pairs). The MC chain is not Hamiltonian (it is a Markov chain satisfying
detailed balance), but in the limit of small step size and many steps, it approximates
Langevin dynamics with dissipation + noise. The dissipation breaks the pairing, making
all exponents ≤ 0 for stable basins (the chain is contracting, on average).

**Sum of Lyapunov exponents** = time-averaged divergence of the vector field:
```
Σᵢ λᵢ = ⟨∇·f⟩  (for flow ẋ = f(x))
```
For a dissipative system, Σλᵢ < 0 (phase space volume contracts over time).
For an IDP, many λᵢ are positive (expansion in many directions), but the sum
must still be negative (the chain is ergodic and mixing).

### 2.3 Benettin Algorithm — Full Derivation

For the discrete MC map Φ, we cannot compute the Jacobian DΦ analytically (the
torsion-angle update + Metropolis acceptance is not differentiable). Instead we
use a **finite-difference approximation** to the tangent map:

```
DΦ(x) · v ≈ (Φ(x + ε·v) - Φ(x)) / ε    for small ε
```

This requires running a **perturbed shadow trajectory** alongside the reference.

**Full Benettin algorithm for the LLE**:

```
Input:  MC chain generator mc_step(particles, topo, T, max_angle)
        initial conformation x₀ (list of Particles)
        perturbation scale ε = 1e-6 Å (in Cα coordinate space)
        number of steps N = 1000 (minimum for convergence)

Initialize:
  x    ← x₀                          (reference trajectory)
  δx   ← ε · random_unit_vector()    (initial perturbation, Cα only)
  λ_acc ← 0.0

For n = 1 to N:
  x'   ← x + δx                      (perturbed state)
  x    ← mc_step(x, ...)             (advance reference)
  x'   ← mc_step(x', ...)            (advance perturbed — same RNG seed!)
  δx   ← [x'[i].pos - x[i].pos for each Cα i]
  growth ← ||δx|| / ε
  λ_acc += ln(growth)                 (accumulate log-expansion)
  δx   ← ε · δx / ||δx||             (renormalize to ε)

Return λ₁ = λ_acc / N
```

**Critical detail**: the perturbed and reference trajectories must use the *same
random number sequence* at each step. If they use independent RNG streams, we measure
the variance of the MC stationary distribution, not the sensitivity to initial
conditions. In ALMA's C++ engine, this means passing the same seed to both calls
of `generate_ensemble`.

**Convergence**: λ₁ converges as O(1/√N) after a burn-in of ~100 steps. For proteins
with N ≈ 100–500 residues, 1000–5000 steps is typically sufficient.

### 2.4 Full Spectrum via QR / Gram-Schmidt

To compute the full Lyapunov spectrum {λ₁,...,λₖ} (first k exponents):

```
Initialize:
  Q ← n×k matrix of orthonormal tangent vectors (random initialization)
  λ_acc[1..k] ← 0

For each step n:
  Evolve each column v_i: Q' ← DΦ · Q  (using finite differences)
  QR decompose Q': Q, R ← qr(Q')       (Gram-Schmidt orthonormalization)
  λ_acc[i] += ln|R[i,i]|               (diagonal = local expansion rates)

Return λᵢ = λ_acc[i] / N
```

For protein conformational analysis, the first k=5 exponents are sufficient.
The full 3N-dimensional spectrum is not needed (and would require 3N shadow
trajectories — impractical for large proteins).

### 2.5 Kaplan-Yorke Dimension

The **Kaplan-Yorke (Lyapunov) dimension** estimates the fractal dimension of
the attractor from the spectrum:

```
D_KY = k + (λ₁ + λ₂ + ... + λ_k) / |λ_{k+1}|
```

where k is the largest index such that the partial sum Σᵢ₌₁ᵏ λᵢ ≥ 0.

For proteins:
- **Ordered protein** (deep funnel): all λᵢ < 0. k=0 by convention, D_KY = 0.
  The attractor is a point (the native basin).
- **Marginally disordered**: some λᵢ ≥ 0, small D_KY ≈ 1–2.
  The attractor is a low-dimensional curve or surface.
- **IDP**: many λᵢ > 0, large D_KY >> 1.
  The attractor is high-dimensional — the protein samples a large
  fraction of its available conformational space.

D_KY is a single displayable scalar (like RMSF or the IDP label) that
summarizes the complexity of the conformational attractor.

### 2.6 Finite-Time Lyapunov Exponents (FTLE)

For the short trajectories in ALMA (120 snapshots), the infinite-time limit
is not achievable. Instead we compute **finite-time Lyapunov exponents**:

```
FTLE(x₀, T) = (1/T) ln ||DΦᵀ(x₀) · v||
```

FTLE varies along the trajectory and gives a *local* measure of chaos:
- High FTLE at snapshot i → that conformation is in a chaotic region
- Low FTLE → that conformation is near a stable basin

The **FTLE field** on the landscape PCA plot (color-coding each node by its
FTLE value) gives a visual map of which regions of conformational space are
chaotic and which are stable. This directly enhances the existing landscape
visualization in `gui_main.py`.

### 2.7 Connecting to ALMA's MC Engine

ALMA's `generate_ensemble` in `physics_engine.cpp` performs torsion-angle MC.
Each call advances a chain by `steps` MC steps. The Lyapunov computation wraps
this:

```python
# In topology/lyapunov.py:
def benettin_lle(engine, topo, init_atoms, n_steps=1000, eps=1e-6, T=0.6):
    x = copy_particles(init_atoms)
    dx = random_ca_perturbation(init_atoms, eps)   # perturb Cα coords by eps
    x_pert = apply_perturbation(x, dx)
    
    log_sum = 0.0
    for _ in range(n_steps):
        x      = engine.generate_ensemble(x, topo, 1, 1, T, 0.12)[0]
        x_pert = engine.generate_ensemble(x_pert, topo, 1, 1, T, 0.12)[0]
        sep = ca_separation(x, x_pert)              # ||δx|| in Cα space
        log_sum += np.log(sep / eps)
        x_pert = renormalize_perturbation(x, x_pert, eps)  # rescale to eps
    
    return log_sum / n_steps
```

The key question — whether to use the same or different RNG seeds for x and
x_pert — is handled by the C++ engine's per-thread RNG streams. We pass
n_cand=1 so the same thread handles both, ensuring the Metropolis accept/reject
decisions are coupled (same uniform random draws).

---

## 3. KAM Theory & Conformational Stability

### 3.1 Integrable Systems and Action-Angle Variables

A Hamiltonian system H(p,q) with n degrees of freedom is **integrable** (in the
Liouville sense) if it has n independent conserved quantities {F₁=H, F₂,...,Fₙ}
in involution ({Fᵢ, Fⱼ} = 0, where {} is the Poisson bracket).

By the **Arnold-Liouville theorem**, the level sets of the conserved quantities
{Fᵢ = cᵢ} that are compact and connected are invariant n-dimensional tori 𝕋ⁿ.
On each torus, one can introduce **action-angle variables** (I, θ) ∈ ℝⁿ × 𝕋ⁿ
such that:

```
H = H(I₁,...,Iₙ)    (depends only on actions, not angles)
İᵢ = -∂H/∂θᵢ = 0   (actions are constants of motion)
θ̇ᵢ = ∂H/∂Iᵢ = ωᵢ(I)  (angles evolve linearly)
```

The motion on each torus is quasi-periodic with frequency vector ω(I).
The key number is the **frequency ratio** ω₁/ω₂ (in 2D; in nD, the ratio vector ω):
- Rational ω: periodic orbit on the torus (resonance)
- Irrational ω: dense orbit on the torus (quasi-periodic, ergodic on the torus)

For protein normal modes: the action variables Iᵢ are the amplitudes of the
i-th normal mode (quasi-harmonic), and θᵢ are their phases. The unperturbed
Hamiltonian is the harmonic approximation to the potential energy well.

### 3.2 The KAM Theorem — Precise Statement

**Theorem** (Kolmogorov 1954, Arnold 1963, Moser 1962):

Let H(I,θ) = H₀(I) + ε H₁(I,θ,ε) where:
- H₀ is integrable, ∂²H₀/∂I² is non-degenerate (the **twist condition**)
- H₁ is an analytic perturbation of size ε
- ω(I) = ∂H₀/∂I satisfies the **Diophantine condition**:

```
|ω(I) · k| ≥ γ / |k|^τ    for all k ∈ ℤⁿ \ {0}
```

for constants γ > 0 and τ > n-1.

**Conclusion**: for ε sufficiently small (ε < ε_c(γ,τ,H₀,H₁)), there exists a
near-identity canonical transformation mapping 𝕋ⁿ to a *perturbed torus*
𝕋ⁿ_ε that is still invariant under the full Hamiltonian H.

**What this means**: most invariant tori survive small perturbations. The set
of surviving tori has measure → 1 as ε → 0. The "holes" between tori (resonance
zones) are filled with chaotic layers that grow as ε increases.

### 3.3 The Chirikov Resonance Overlap Criterion

The KAM theorem guarantees torus survival for small ε but does not give the
critical ε_c explicitly. The **Chirikov overlap criterion** (1979) gives a
practical estimate:

For two resonant tori with frequency ratios p₁/q₁ and p₂/q₂, their resonance
zones (stochastic layers) have widths:

```
δI₁ ≈ 2√(ε V_k / (∂²H₀/∂I²))    (island half-width in action space)
```

When δI₁ + δI₂ > |I₁ - I₂| (the islands overlap), KAM breakdown occurs and
the region between them becomes globally chaotic:

```
K_Chirikov = δI / ΔI ≈ 1    →    onset of global chaos
```

For proteins, ε ↔ kT/ΔE_barrier (thermal noise relative to barrier height),
and the "resonance" corresponds to the protein sampling multiple basins:

```
K_protein = kT / min{ΔE_ij over all adjacent basins i,j}

K_protein < 0.5:  well within KAM regime, ordered
K_protein ∈ [0.5, 1.5]:  approaching breakdown, partially disordered
K_protein > 1.5:  past Chirikov threshold, globally chaotic, IDP
```

This is computable from the landscape MC trajectory: the barrier heights
ΔE_ij are estimated as the energy difference between basin minima and the
saddle point (highest energy snapshot on a path between basins).

### 3.4 Resonances and the Poincaré-Birkhoff Theorem

Near a rational frequency ratio ω₁/ω₂ = p/q, the KAM torus breaks up into
a chain of **p** elliptic fixed points (stable basins) and **q** hyperbolic
fixed points (saddle points / transition states) — the **Poincaré-Birkhoff
theorem**.

For proteins, this directly maps to the conformational basin structure:
- **Elliptic fixed points** ↔ metastable conformational basins (PCA clusters)
- **Hyperbolic fixed points** ↔ transition state structures (saddle points in
  the energy landscape, highest-energy conformations between basins)
- **Resonance order p/q** ↔ the ratio of inter-basin transition rates

The Poincaré section (a 2D slice of phase space transverse to the flow) for
the ALMA landscape MC is effectively the PCA projection — each point in the
PCA plot is one snapshot's position on a 2D "Poincaré section" of the
3N-dimensional conformational space.

### 3.5 Winding Numbers and Rotation Numbers

In the 2D PCA projection of the ALMA landscape, define a **rotation number**
(winding number) for the trajectory around each basin center:

```
ρ = lim_{n→∞} (1/n) Σᵢ₌₀ⁿ⁻¹ Δθᵢ / (2π)
```

where Δθᵢ is the angular increment around the basin center at step i.

By KAM theory:
- **Irrational ρ**: quasi-periodic orbit, KAM torus intact → ordered protein
- **Rational ρ = p/q**: resonant orbit, broken into p elliptic + q hyperbolic points
- **Undefined ρ** (non-convergent): chaotic orbit → IDP

The rotation number is directly computable from the `layout` array (PCA coordinates)
in `LandscapeWorker.result`.

### 3.6 Arnold Diffusion in Higher Dimensions

In systems with n ≥ 3 degrees of freedom, KAM tori have dimension n while the
energy surface has dimension 2n-1. For n ≥ 3, tori do NOT separate the energy
surface (codimension 1 argument fails): a (2n-1)-dimensional energy surface is
not separated by n-dimensional tori when 2n-1 > n+1, i.e., n > 2.

This means that even when KAM tori exist, orbits can slowly *diffuse* through
the gaps between them — **Arnold diffusion** — with diffusion rate exponentially
small in ε⁻¹. For proteins (n >> 2), Arnold diffusion is always possible in
principle, meaning:

- No protein is *absolutely* ordered in the KAM sense
- But the Arnold diffusion timescale can be astronomically long (>> protein lifetime)
- In practice: ordered protein = Arnold diffusion timescale >> experimental observation time

This motivates the use of the **finite-time Lyapunov exponent** (§2.6) rather
than the asymptotic LLE — we care about chaos on the timescale of the MC run,
not in infinite time.

---

## 4. Algebraic Foundations — Lambda-Style

### 4.1 Why Functional / Lambda-Calculus Style?

The mathematical structures needed for Khovanov homology — groups, rings, modules,
chain complexes, functors — are naturally **algebraic types with pure operations**.
They have no mutable state; they are defined entirely by their operations and axioms.

This maps directly to the **functional programming** paradigm derived from
lambda calculus (Church 1936):

| Lambda calculus concept | Python implementation | Mathematical role |
|---|---|---|
| Term / type | `@dataclass(frozen=True)` | algebraic structure |
| Abstraction λx.M | `lambda x: ...` or `def f(x):` | operation definition |
| Application M N | `f(x)` | applying an operation |
| Composition | `lambda f, g: lambda x: f(g(x))` | function composition ∘ |
| Currying | `functools.partial` | partial application |
| Fixed point Y | recursive `def` | inductive definitions |

The design rule: **every function in `topology/` is pure** (no side effects, no
mutation, same input → same output). This is not an aesthetic choice — it enables:

1. **Equational reasoning**: we can substitute equals for equals, just as in math.
2. **Testability**: pure functions are trivially unit-testable (no setup/teardown).
3. **Memoization**: pure functions can be cached safely (`@functools.lru_cache`).
   Critical for the Kauffman bracket recursion (exponential state space, heavy reuse).
4. **Parallelism**: pure functions have no data races (relevant for computing Jones
   polynomials of multiple MC snapshots in parallel).

### 4.2 Sets as Types

In lambda calculus, a **set** S is modeled as its characteristic function:
```
S : U → {True, False}    (predicate on a universe U)
```

In Python:
```python
# A set is a predicate — a pure function from elements to bool
Set = Callable[[Any], bool]

# Set operations via function composition
intersection = lambda A, B: lambda x: A(x) and B(x)
union        = lambda A, B: lambda x: A(x) or  B(x)
complement   = lambda A:    lambda x: not A(x)
subset       = lambda A, B: all(B(x) for x in universe if A(x))
```

For finite sets (relevant to chain complex bases), we use `frozenset` for
immutability + hashability, which is required by `@dataclass(frozen=True)`.

### 4.3 Groups

A **group** (G, ·, e, ⁻¹) satisfies four axioms:
1. **Closure**: ∀a,b ∈ G: a·b ∈ G
2. **Associativity**: (a·b)·c = a·(b·c)
3. **Identity**: ∃e ∈ G: e·a = a·e = a ∀a
4. **Inverses**: ∀a ∈ G ∃a⁻¹ ∈ G: a·a⁻¹ = e

```python
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, FrozenSet

G = TypeVar('G')

@dataclass(frozen=True)
class Group(Generic[G]):
    elements:  FrozenSet[G]
    mul:       Callable[[G, G], G]      # · : G × G → G
    identity:  G                        # e ∈ G
    inv:       Callable[[G], G]         # ⁻¹ : G → G

    def verify_axioms(self) -> bool:
        E = list(self.elements)
        # Closure
        assert all(self.mul(a,b) in self.elements for a in E for b in E)
        # Identity
        assert all(self.mul(self.identity, a) == a for a in E)
        # Inverses
        assert all(self.mul(a, self.inv(a)) == self.identity for a in E)
        return True
```

**Groups essential to this project**:

**ℤ (integers under addition)** — the coefficient ring for all chain complexes:
```python
Z_group = Group(
    elements  = integers,          # infinite, represented lazily
    mul       = lambda a, b: a+b,  # addition
    identity  = 0,
    inv       = lambda a: -a
)
```

**Braid group Bₙ** — fundamental to knot theory. Generated by σ₁,...,σₙ₋₁
(crossing the i-th strand over the (i+1)-th strand) with relations:
```
σᵢ σⱼ = σⱼ σᵢ           (|i-j| ≥ 2, distant crossings commute)
σᵢ σᵢ₊₁ σᵢ = σᵢ₊₁ σᵢ σᵢ₊₁  (braid relation, Yang-Baxter equation)
```
Braids are relevant because every knot/link is the **closure** of a braid
(Alexander's theorem), and the Jones polynomial was originally defined via
braid representations (Jones' original 1985 construction through von Neumann
algebras / Hecke algebras).

**Symmetric group Sₙ** — permutations of n elements. Appears as a quotient
of Bₙ (by adding σᵢ² = 1) and in the representation theory underlying the
colored Jones polynomial.

**ℤ/2ℤ** — integers mod 2. Used for Khovanov homology over 𝔽₂ (simpler:
no sign issues, no torsion in homology groups). Useful for quick sanity checks
before working over ℤ.

### 4.4 Rings

A **ring** (R, +, ·, 0, 1) is an abelian group (R,+,0) with a second binary
operation · (multiplication) that is associative, distributes over +, and has
identity 1. Commutativity of · is NOT required (though most rings here are commutative).

```python
R = TypeVar('R')

@dataclass(frozen=True)
class Ring(Generic[R]):
    add_group:   Group[R]           # (R, +, 0) — underlying abelian group
    mul:         Callable[[R,R], R] # · : R × R → R
    one:         R                  # multiplicative identity
    # zero = add_group.identity

    zero = property(lambda self: self.add_group.identity)
    add  = property(lambda self: self.add_group.mul)
    neg  = property(lambda self: self.add_group.inv)
```

**The Laurent polynomial ring ℤ[t, t⁻¹]** is the coefficient ring for the
Jones polynomial. Elements are finite sums Σₙ aₙ tⁿ where aₙ ∈ ℤ and n ∈ ℤ
(including negative exponents):

```python
# Represent as dict[int, int]: exponent → coefficient
# e.g., -t⁻⁴ + t⁻³ + t⁻¹  is  {-4: -1, -3: 1, -1: 1}

@dataclass(frozen=True)
class LaurentPoly:
    coeffs: frozenset[tuple[int,int]]  # frozenset of (exponent, coefficient) pairs
    
    @staticmethod
    def from_dict(d: dict[int,int]) -> 'LaurentPoly':
        return LaurentPoly(frozenset((e,c) for e,c in d.items() if c != 0))
    
    def __add__(self, other: 'LaurentPoly') -> 'LaurentPoly':
        d = dict(self.coeffs)
        for (e, c) in other.coeffs:
            d[e] = d.get(e, 0) + c
        return LaurentPoly.from_dict(d)
    
    def __mul__(self, other: 'LaurentPoly') -> 'LaurentPoly':
        d = {}
        for (e1, c1) in self.coeffs:
            for (e2, c2) in other.coeffs:
                d[e1+e2] = d.get(e1+e2, 0) + c1*c2
        return LaurentPoly.from_dict(d)
    
    def evaluate(self, t: complex) -> complex:
        return sum(c * t**e for (e,c) in self.coeffs)
```

### 4.5 Modules and Free Modules

An **R-module** M is an abelian group (M, +) with a scalar multiplication
R × M → M satisfying the usual axioms (distributivity, associativity, unit action).
Modules generalize vector spaces to arbitrary rings (not just fields).

The critical module for Khovanov homology is the **free ℤ-module** on a finite
set B (the basis):
```
ℤ[B] = {Σ_{b∈B} nᵦ · b : nᵦ ∈ ℤ, almost all zero}
```

Represented as `dict[B, int]` (basis element → coefficient, omitting zeros).

```python
# A free module element: a ℤ-linear combination of basis elements
FreeElem = dict[str, int]   # basis label → coefficient

# Module operations
def add_free(a: FreeElem, b: FreeElem) -> FreeElem:
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return {k: v for k, v in result.items() if v != 0}  # drop zeros

def scale_free(n: int, a: FreeElem) -> FreeElem:
    return {k: n*v for k, v in a.items()} if n != 0 else {}
```

### 4.6 Linear Maps and Matrices

A **linear map** f: M → N between free ℤ-modules with bases B_M and B_N is
determined by a matrix with integer entries: f[b] = Σₖ Aₖᵦ · eₖ for each b ∈ B_M.

```python
@dataclass(frozen=True)
class LinearMap:
    domain_basis:   tuple[str, ...]         # basis of domain M
    codomain_basis: tuple[str, ...]         # basis of codomain N
    matrix:         tuple[tuple[int,...],...]  # A[i][j] = coefficient of eᵢ for f(bⱼ)
    
    def apply(self, v: FreeElem) -> FreeElem:
        result: FreeElem = {}
        for j, b in enumerate(self.domain_basis):
            c = v.get(b, 0)
            if c == 0: continue
            for i, e in enumerate(self.codomain_basis):
                result[e] = result.get(e, 0) + self.matrix[i][j] * c
        return {k: v for k, v in result.items() if v != 0}
    
    def compose(self, other: 'LinearMap') -> 'LinearMap':
        # self ∘ other: other domain → self codomain
        # matrix product
        A, B = self.matrix, other.matrix
        C = tuple(
            tuple(sum(A[i][k]*B[k][j] for k in range(len(B)))
                  for j in range(len(B[0])))
            for i in range(len(A))
        )
        return LinearMap(other.domain_basis, self.codomain_basis, C)
```

### 4.7 Chain Complexes and Homology

A **chain complex** (C*, ∂) over ℤ is a sequence of free ℤ-modules and linear maps:

```
... ──∂_{n+2}──▶ C_{n+1} ──∂_{n+1}──▶ C_n ──∂_n──▶ C_{n-1} ──∂_{n-1}──▶ ...
```

satisfying the **fundamental lemma**: ∂ₙ ∘ ∂ₙ₊₁ = 0 for all n.

This condition says: "the image of ∂_{n+1} is contained in the kernel of ∂_n."

The **homology groups** measure the failure of exactness:
```
Hₙ(C*) = ker(∂_n) / im(∂_{n+1})
```

- ker(∂_n) = "cycles at level n" (elements mapped to zero)
- im(∂_{n+1}) = "boundaries at level n" (elements that come from level n+1)
- Hₙ = "cycles that are not boundaries" = topological holes at level n

For Khovanov homology, this detects *which topological features of the knot
persist* (cycles) and which are consequences of the knot diagram choice (boundaries).

```python
@dataclass(frozen=True)
class ChainComplex:
    # C_n indexed by integer degree n
    modules:   dict[int, tuple[str,...]]     # degree → basis tuple
    boundary:  dict[int, LinearMap]          # ∂_n: C_n → C_{n-1}
    
    def verify(self) -> bool:
        """Check ∂ ∘ ∂ = 0 at every degree."""
        for n, dn in self.boundary.items():
            if n-1 in self.boundary:
                dn_minus = self.boundary[n-1]
                comp = dn_minus.compose(dn)  # ∂_{n-1} ∘ ∂_n
                # Every entry of the composition matrix must be zero
                if any(comp.matrix[i][j] != 0
                       for i in range(len(comp.matrix))
                       for j in range(len(comp.matrix[0]))):
                    return False
        return True
    
    def homology(self, n: int) -> dict:
        """Compute H_n = ker(∂_n) / im(∂_{n+1}) via Smith normal form."""
        # Returns: {'rank': int, 'torsion': list[int]}
        # rank = number of free ℤ summands
        # torsion = list of torsion coefficients (e.g., [2] means ℤ/2ℤ summand)
        return smith_normal_form_homology(
            self.boundary.get(n),
            self.boundary.get(n+1)
        )
```

**Smith Normal Form** (SNF) over ℤ: every integer matrix A can be written as
A = U D V where U,V are invertible integer matrices and D = diag(d₁,d₂,...,dₖ,0,...,0)
with d₁ | d₂ | ... | dₖ (divisibility chain). The homology is then:

```
H ≅ ℤ^{free rank} ⊕ ℤ/d₁ ⊕ ℤ/d₂ ⊕ ... ⊕ ℤ/dₖ   (excluding d_i = 1)
```

The SNF computation uses `sympy.Matrix.smith_normal_form()` for small matrices
(up to ~50×50, sufficient for knots with ≤ 5 crossings) and a custom
integer-arithmetic implementation for larger ones.

### 4.8 Bigraded Chain Complexes

Khovanov homology is **bigraded**: the chain complex has an additional grading j
(the "quantum grading" or "q-grading") that is preserved by the boundary maps.

```python
@dataclass(frozen=True)
class BidegreeChainComplex:
    # modules[i][j] = basis of C^{i,j}  (homological grade i, quantum grade j)
    modules:  dict[tuple[int,int], tuple[str,...]]
    boundary: dict[int, dict[int, LinearMap]]  # ∂^i_j: C^{i,j} → C^{i+1,j}
    # Note: boundary PRESERVES j (quantum grading), INCREASES i by 1
    
    def euler_characteristic(self) -> LaurentPoly:
        """Σ_{i,j} (-1)^i q^j rank(H^{i,j}) — should equal Jones polynomial."""
        result = {}
        for (i, j), basis in self.modules.items():
            rk = len(basis)
            result[j] = result.get(j, 0) + ((-1)**i) * rk
        return LaurentPoly.from_dict(result)
```

---

## 5. Category Theory Layer

### 5.1 Why Category Theory?

Category theory provides the language in which Khovanov's construction is most
naturally stated. The key objects — the cobordism category **Kob**, the Khovanov
functor F_Kh, and the natural transformations between them — are all categorical
constructions. Understanding them categorically clarifies:

1. **Why ∂² = 0** (it follows from the functor axioms applied to the cube of resolutions)
2. **Why the homology is a knot invariant** (it is the value of a functor on a morphism
   in the category of knots, hence invariant under knot isotopy = Reidemeister moves)
3. **What "categorification" means precisely** (it is a functor lifting a number-valued
   invariant to a vector-space-valued one, with the original invariant recovered as
   the Euler characteristic / Grothendieck group)

### 5.2 Definition of a Category

A **category** C consists of:
- A class Ob(C) of **objects**
- For each pair A, B ∈ Ob(C), a set Hom_C(A,B) of **morphisms** from A to B
- A **composition** law: ∘ : Hom(B,C) × Hom(A,B) → Hom(A,C), written g ∘ f
- An **identity morphism** id_A ∈ Hom(A,A) for each object A

satisfying:
- **Associativity**: (h ∘ g) ∘ f = h ∘ (g ∘ f)
- **Identity laws**: id_B ∘ f = f = f ∘ id_A

```python
from typing import TypeVar, Generic, Callable, FrozenSet
Obj = TypeVar('Obj')
Mor = TypeVar('Mor')

@dataclass(frozen=True)
class Category(Generic[Obj, Mor]):
    # Objects: implicit (any hashable type)
    hom:       Callable[[Obj, Obj], FrozenSet[Mor]]  # Hom(A,B)
    compose:   Callable[[Mor, Mor], Mor]              # g ∘ f (g second, f first)
    identity:  Callable[[Obj], Mor]                   # id_A
```

### 5.3 Key Categories in This Project

**Vect_ℤ** (free ℤ-modules and linear maps):
- Objects: free ℤ-modules of finite rank (equivalently: non-negative integers n,
  representing ℤⁿ)
- Morphisms: integer matrices (linear maps between free modules)
- Composition: matrix multiplication
- Identity: identity matrix

**Kob** (the cobordism category — Khovanov's construction lives here):
- Objects: **planar matchings** — collections of non-crossing arcs connecting 2n
  points on a line (equivalently: ways to smooth all crossings in a knot diagram)
- Morphisms from matching M to matching N: **cobordisms** — compact oriented surfaces
  embedded in [0,1] × ℝ² with boundary (M at bottom, N at top)
- Composition: stacking cobordisms vertically, gluing along the shared matching
- Identity: the "cylinder" cobordism M × [0,1]

The key morphisms in Kob are:
- **Cup** (birth): ∅ → circle (a disk — creating a circle from nothing)
- **Cap** (death): circle → ∅ (a disk — annihilating a circle)
- **Saddle** (merge/split): two circles → one circle, or one → two (a pair of pants
  or its mirror, depending on orientation)

**Chain(ℤ)** (chain complexes and chain maps):
- Objects: chain complexes of free ℤ-modules
- Morphisms: chain maps (degree-0 linear maps commuting with ∂)
- Composition: composition of chain maps
- Identity: identity chain map

**Knots** (knot diagrams and Reidemeister moves):
- Objects: oriented knot/link diagrams in the plane (up to planar isotopy)
- Morphisms: sequences of Reidemeister moves (R1, R2, R3)
- The Khovanov functor is invariant on this category (same object → same complex
  up to chain homotopy)

### 5.4 Functors

A **functor** F : C → D maps:
- Objects: F(A) ∈ Ob(D) for each A ∈ Ob(C)
- Morphisms: F(f) ∈ Hom_D(F(A), F(B)) for each f ∈ Hom_C(A,B)
preserving composition: F(g∘f) = F(g)∘F(f), and identities: F(id_A) = id_{F(A)}.

```python
@dataclass(frozen=True)
class Functor(Generic[Obj, Mor]):
    source: Category
    target: Category
    on_obj: Callable[[Any], Any]   # F on objects
    on_mor: Callable[[Any], Any]   # F on morphisms

    def verify(self, A, B, f, g) -> bool:
        # F(g ∘ f) = F(g) ∘ F(f)
        lhs = self.on_mor(self.source.compose(g, f))
        rhs = self.target.compose(self.on_mor(g), self.on_mor(f))
        return lhs == rhs
```

The **Khovanov functor** F_Kh : Kob → Vect_ℤ assigns:
- To each planar matching (circle arrangement): the tensor product V^{⊗ circles},
  where V = ℤ{v₊, v₋} is the 2-dimensional TQFT module
- To each cobordism: a linear map between tensor products

This functor is the TQFT (topological quantum field theory) underlying Khovanov homology.
Its value on the full cube of resolutions assembles into the Khovanov chain complex.

### 5.5 Natural Transformations

A **natural transformation** η: F ⟹ G between functors F, G: C → D assigns
to each object A ∈ C a morphism η_A: F(A) → G(A) in D, such that for every
morphism f: A → B in C the **naturality square** commutes:

```
F(A) ──F(f)──▶ F(B)
 │                │
η_A           η_B
 │                │
 ▼                ▼
G(A) ──G(f)──▶ G(B)
```

**In this project**: the relationship between the Jones polynomial and the Khovanov
homology is encoded as a natural transformation between two functors:
- F_Jones: Knots → ℤ[q,q⁻¹]  (Jones polynomial)
- F_Kh: Knots → Chain(ℤ)      (Khovanov complex)
- The "decategorification" map χ: F_Kh ⟹ F_Jones is the Euler characteristic
  natural transformation, with χ_K(CKh(K)) = V_K(q).

### 5.6 Monoidal Categories and the Frobenius Algebra

The TQFT functor F_Kh: Kob → Vect_ℤ is not just a functor but a **symmetric
monoidal functor** — it respects the monoidal structure (disjoint union of circles /
tensor product of modules).

The algebraic content is encoded in a **commutative Frobenius algebra** (A, m, η, Δ, ε):
- A = V = ℤ{v₊, v₋} (the TQFT module for one circle)
- m: V⊗V → V (multiplication, from the "pair of pants" cobordism — merge)
- η: ℤ → V (unit, from the "cup" cobordism — create circle)
- Δ: V → V⊗V (comultiplication, from the "copair of pants" — split)
- ε: V → ℤ (counit, from the "cap" cobordism — destroy circle)

The Frobenius axiom: (id⊗m) ∘ (Δ⊗id) = Δ ∘ m = (m⊗id) ∘ (id⊗Δ)

Explicitly, with basis {v₊, v₋} where deg(v₊) = +1, deg(v₋) = -1:

```
m: V⊗V → V          η: ℤ → V           Δ: V → V⊗V            ε: V → ℤ
m(v₊⊗v₊) = v₊       η(1) = v₊          Δ(v₊) = v₊⊗v₋+v₋⊗v₊  ε(v₊) = 0
m(v₊⊗v₋) = v₋       η(0) = 0           Δ(v₋) = v₋⊗v₋         ε(v₋) = 1
m(v₋⊗v₊) = v₋
m(v₋⊗v₋) = 0
```

Every assignment of m or Δ to each edge of the cube of resolutions
gives a chain complex with ∂² = 0 — **this follows automatically** from the
Frobenius axiom (it is a consequence of the functoriality of F_Kh, not an
independent verification).

---

## 6. Knot Theory of Protein Backbones

### 6.1 The Protein Backbone as a Mathematical Object

The Cα backbone is a sequence of N points {r₁, r₂, ..., rₙ} ∈ ℝ³ connected
by straight segments, forming a **piecewise-linear open curve** (a polygonal arc).

For knot theory to apply, we need a **closed curve** (an embedding S¹ → ℝ³).
The choice of closure method affects:
- Which knot type is detected
- How to interpret the result biologically

**Method 1 — Deterministic closure** (for Jones/Khovanov computation):
Connect rₙ to r₁ via an arc that passes through a point P_∞ far from the protein
(|P_∞| >> diameter of protein). The arc r_N → P_∞ → r_1 is chosen to minimize
additional crossings. The resulting closed curve has a well-defined knot type that
is independent of P_∞ direction for generic choices.

```python
def deterministic_closure(ca_coords: np.ndarray, 
                           closure_point_scale: float = 100.0) -> np.ndarray:
    """Close the Cα backbone into a loop via a distant point.
    
    The closure point P_∞ is placed at closure_point_scale * (bounding_box_size)
    along the direction that minimizes additional crossings.
    """
    center = ca_coords.mean(axis=0)
    radius = np.max(np.linalg.norm(ca_coords - center, axis=1))
    
    # Try 20 candidate closure directions, pick fewest crossings
    best_dir, best_crossings = None, np.inf
    for theta in np.linspace(0, np.pi, 20):
        for phi in np.linspace(0, 2*np.pi, 20):
            direction = np.array([np.sin(theta)*np.cos(phi),
                                   np.sin(theta)*np.sin(phi),
                                   np.cos(theta)])
            p_inf = center + closure_point_scale * radius * direction
            n_cross = count_closure_crossings(ca_coords, p_inf)
            if n_cross < best_crossings:
                best_crossings, best_dir = n_cross, direction
    
    p_inf = center + closure_point_scale * radius * best_dir
    # Insert p_inf between last and first Cα
    return np.vstack([ca_coords, p_inf[np.newaxis]])
```

**Method 2 — Probabilistic closure** (for knotting probability profile):
Sample M random closure directions uniformly on S². For each direction, close
the backbone and detect the knot type. The **knotting probability** at position i
is the fraction of closures that give a non-trivial knot when using only the
backbone segment from residue 1 to residue i.

```python
def knotting_probability_profile(ca_coords: np.ndarray, 
                                  n_samples: int = 100) -> np.ndarray:
    """Millett-Rawdon probabilistic closure.
    Returns array of shape (N,) with knotting probability at each residue.
    """
    N = len(ca_coords)
    probs = np.zeros(N)
    directions = fibonacci_sphere(n_samples)  # uniform S² sampling
    
    for i in range(3, N):
        knotted_count = 0
        for d in directions:
            closed = close_with_direction(ca_coords[:i+1], d)
            gauss = compute_gauss_code(closed)
            knotted_count += (jones_polynomial(gauss) != UNKNOT_JONES)
        probs[i] = knotted_count / n_samples
    
    return probs
```

### 6.2 Knot Diagrams — Formal Definition

A **knot diagram** D is a generic projection π: ℝ³ → ℝ² of a knot K, with:
- The projection is "generic" (no triple points, no tangencies, finitely many crossings)
- At each crossing, the **over/under** information is recorded (which strand has higher
  z-coordinate along the projection direction n̂)
- The diagram has a finite number of crossings c(D) ≥ 0

The **crossing number** c(K) of a knot K is the minimum c(D) over all diagrams D.
```
unknot:       c = 0  (trefoil is the simplest non-trivial knot)
trefoil 3₁:   c = 3  (only knot with 3 crossings)
figure-8 4₁:  c = 4  (only knot with 4 crossings)
```

### 6.3 Gauss Code — Algorithm and Data Structure

The **Gauss code** of a knot diagram encodes the crossing sequence encountered
when traveling along the knot:

```
GaussCode = list of signed crossing labels
  positive label +k: pass OVER crossing k (this strand is on top)
  negative label -k: pass UNDER crossing k (this strand is below)
```

For an oriented knot, we also record the **sign** of each crossing:
```
ε(crossing k) = +1 if: positive crossing (right-over-left by right-hand rule)
              = -1 if: negative crossing (left-over-right)
```

**Algorithm: Cα coordinates → Gauss code**

```python
def compute_gauss_code(curve: np.ndarray, 
                        proj_dir: np.ndarray = np.array([0,0,1])
                       ) -> list[tuple[int,int,int]]:
    """
    curve:    (N+1, 3) array, closed curve (last point = first point)
    proj_dir: projection direction n̂ (z-axis by default)
    Returns:  list of (crossing_id, over_under, sign)
              over_under: +1 = over, -1 = under
              sign: +1 = positive crossing, -1 = negative crossing
    """
    # Build orthonormal frame (u, v, n̂) for projection
    n = proj_dir / np.linalg.norm(proj_dir)
    # Project curve onto (u,v) plane
    coords_2d = project_onto_plane(curve, n)
    
    # Find all crossing pairs: segments (i, i+1) and (j, j+1) with i+2 < j
    crossings = []
    N = len(curve) - 1  # number of segments
    for i in range(N):
        for j in range(i+2, N):
            if i == 0 and j == N-1: continue  # adjacent at closure
            pt = segment_intersection_2d(coords_2d[i], coords_2d[i+1],
                                          coords_2d[j], coords_2d[j+1])
            if pt is not None:
                # Determine over/under from z-coordinates at intersection
                t_i = intersection_parameter(coords_2d[i], coords_2d[i+1], pt)
                z_i = lerp_z(curve[i], curve[i+1], t_i, n)
                t_j = intersection_parameter(coords_2d[j], coords_2d[j+1], pt)
                z_j = lerp_z(curve[j], curve[j+1], t_j, n)
                crossing_sign = compute_crossing_sign(
                    curve[i+1]-curve[i], curve[j+1]-curve[j], n)
                crossings.append({
                    'id': len(crossings),
                    'seg_i': i, 'seg_j': j,
                    'z_i': z_i, 'z_j': z_j,
                    'sign': crossing_sign
                })
    
    # Walk along the curve, recording crossings in order
    return build_gauss_code_from_crossings(crossings, N)
```

### 6.4 Reidemeister Moves — Detailed

The three Reidemeister moves are local changes to a knot diagram that do not
change the underlying knot type:

**R1 (Twist)**: Insert or remove a single loop — a strand makes a single curl.
```
Before: a straight strand
After:  a strand with a small loop (creating one crossing with itself)
Effect on writhe: w → w ± 1 (writhe changes!)
Effect on Jones: V_K unchanged (requires the writhe correction in the bracket)
```

**R2 (Poke)**: Two strands pass over/under each other with a double crossing.
```
Before: two strands crossing twice (one over+under sequence)
After:  two parallel non-crossing strands
Effect on writhe: w unchanged (one +1 and one -1 crossing cancel)
Effect on Kauffman bracket: ⟨K⟩ unchanged under R2 (can verify directly)
```

**R3 (Slide)**: A strand passes over/under a crossing between two other strands.
```
Before: one strand passing over a crossing between two others
After:  same strand, same crossing, but from the other side
Effect: no change to writhe, no change to knot type
```

Any two diagrams of the same knot are connected by a finite sequence of R1, R2, R3.
**This is the definition of knot equivalence** (Reidemeister 1926).

### 6.5 The Writhe — Gauss Integral Formulation

The **writhe** of a smooth closed curve γ: S¹ → ℝ³ is the **Gauss writhe**:

```
Wr(γ) = (1/4π) ∬_{S¹×S¹} [γ'(s) × γ'(t)] · (γ(s)-γ(t)) / |γ(s)-γ(t)|³  ds dt
```

For a polygonal curve (our piecewise-linear Cα backbone), this integral reduces
to the discrete sum over crossings: Wr = Σᵢ εᵢ (same as before).

The Gauss integral formulation makes the geometry clear:
- Wr measures the "average signed crossing number" over all projection directions
- It is related to the **self-linking number** of the curve
- It appears in the formula for the **twist** Tw and the **Calugareanu-White-Fuller
  theorem**: Lk = Tw + Wr, where Lk is the linking number of γ with a pushed-off copy

For the online writhe scale over the MC trajectory:
```python
def online_writhe_series(snapshots: list, ca_indices: list[int],
                          closure_method='deterministic') -> np.ndarray:
    """Compute writhe at each MC snapshot. Returns array of shape (N_snaps,)."""
    writhes = []
    for particles in snapshots:
        ca_coords = extract_ca_coords(particles, ca_indices)
        closed = close_backbone(ca_coords, method=closure_method)
        gauss = compute_gauss_code(closed)
        wr = sum(sign for (_, _, sign) in gauss) / 2  # each crossing appears twice
        writhes.append(wr)
    return np.array(writhes)
```

The **writhe power spectrum** S_w(f) = |FFT(w(t))|² distinguishes:
- Ordered proteins: Lorentzian spectrum (S_w(f) ∝ 1/(f² + f_c²)), single correlation time
- IDPs: power-law spectrum S_w(f) ∝ f^{-α}, scale-free = 1/f-like noise

### 6.6 The Alexander Polynomial — Bridge to Homology

Before Jones and Khovanov, the **Alexander polynomial** Δ_K(t) ∈ ℤ[t, t⁻¹] was
the first polynomial knot invariant (Alexander 1928). It is weaker than the Jones
polynomial but easier to compute:

**Via the Seifert matrix**: 
1. Choose a **Seifert surface** Σ for K (an oriented surface in ℝ³ with ∂Σ = K)
2. Compute the **Seifert matrix** V: Vᵢⱼ = lk(aᵢ, aⱼ⁺) where {aᵢ} are generators
   of H₁(Σ) and aⱼ⁺ is the positive push-off of aⱼ
3. Δ_K(t) = det(t^{1/2} V - t^{-1/2} Vᵀ)

The Alexander polynomial detects:
- Whether K is the unknot (Δ = 1 → possibly unknot; Δ ≠ 1 → definitely knotted)
- It cannot distinguish mirror images (Δ_K(t) = Δ_{mirror K}(t))

**The Jones polynomial is strictly stronger**: V_K(t) ≠ 1 implies non-trivial knot,
AND V_K(t) can distinguish mirror images (V_K(t) ≠ V_{mirror K}(t) for many knots).

**Khovanov homology is strictly stronger than Jones**: Kh^{i,j} distinguishes knots
with the same Jones polynomial (e.g., Conway knot vs Kinoshita-Terasaka knot).

The hierarchy: Khovanov ⊃ Jones ⊃ HOMFLY ⊃ Alexander ⊃ (unknotting number)

---

## 7. Jones Polynomial via Skein Relations

### 7.1 The Skein Triple and Recursive Definition

The Jones polynomial V_K(t) ∈ ℤ[t^{1/2}, t^{-1/2}] is defined by:

**(S0)** Normalization: V(unknot) = 1

**(S1)** Skein relation: for any three diagrams L₊, L₋, L₀ that are identical
outside a small disk and differ inside as follows:
```
L₊: positive crossing  (right strand over left)   ╲╱
                                                    ╱╲
L₋: negative crossing  (left strand over right)   ╲╱ (mirror)
                                                    ╱╲
L₀: 0-smoothing       (horizontal connection)     ╲  ╱
                                                    ╲╱
```

the relation is:
```
t⁻¹ V(L₊) - t V(L₋) + (t^{-1/2} - t^{1/2}) V(L₀) = 0
```

This allows **recursive computation**: given a diagram with n crossings, choose
any crossing, apply S1 to reduce to two diagrams each with n-1 crossings, recurse
until reaching the unknot (or disjoint unknots).

**Base cases**:
```
V(unknot) = 1
V(n disjoint unknots) = (-t^{1/2} - t^{-1/2})^{n-1}
```

### 7.2 Kauffman Bracket — Derivation from First Principles

The Jones polynomial is most efficiently computed via the **Kauffman bracket**
⟨·⟩ ∈ ℤ[A, A⁻¹] (Kauffman 1987), related to Jones by A = t^{-1/4}.

**Bracket rules**:
```
(B0) ⟨unknot⟩ = 1
(B1) ⟨D ⊔ unknot⟩ = δ ⟨D⟩    where δ = -A² - A⁻²  (loop value)
(B2) ⟨L_cross⟩ = A ⟨L_0⟩ + A⁻¹ ⟨L_∞⟩
```

where L_0 and L_∞ are the two **smoothings** of a crossing:
```
L_0 (A-smoothing):   connect top-left to bottom-left, top-right to bottom-right
                     )||( (vertical strands reconnected horizontally)

L_∞ (A⁻¹-smoothing): connect top-left to top-right, bottom-left to bottom-right
                     =(= (horizontal strands)
```

**Why this works**: rule (B2) expands any crossing into two smoothings recursively.
After n applications, we have 2ⁿ complete smoothings — each a disjoint union of
circles. Rule (B1) evaluates each such configuration as δ^{circles-1}.
The total is a polynomial in A.

**Invariance under R2 and R3** (not R1):

Under R2: a double crossing contributes A·A⁻¹ + A⁻¹·A = 1 + 1 = ... wait, let us
compute properly. The R2 move adds two crossings to a region where strands pass:
```
⟨R2 diagram⟩ = A(A ⟨...⟩ + A⁻¹ ⟨...⟩) + A⁻¹(A ⟨...⟩ + A⁻¹ ⟨...⟩)
             = (A² + 1 + 1 + A⁻²) ⟨...⟩ ... 
```
After careful tracking of circle counts and signs, one shows ⟨R2⟩ = ⟨pre-R2⟩.
Under R1: ⟨R1 diagram⟩ = (-A^{±3}) ⟨pre-R1⟩ (the bracket picks up a monomial factor).

**Writhe correction**: define the **regular isotopy invariant**:
```
f_K(A) = (-A³)^{-w(K)} ⟨K⟩
```
where w(K) = Σᵢ εᵢ is the writhe. The factor (-A³)^{-w} exactly compensates
the R1 change. The Jones polynomial is:
```
V_K(t) = f_K(t^{-1/4})
```

### 7.3 Worked Example — Trefoil Knot 3₁

The left-handed trefoil has 3 negative crossings, writhe w = -3.

Expanding the bracket by (B2) three times gives 2³ = 8 complete smoothings:

| Vertex (0,1) assignments | Circles | Contribution |
|---|---|---|
| (0,0,0) | 2 | A³ · δ = A³(-A²-A⁻²) |
| (1,0,0),(0,1,0),(0,0,1) | 1 | A² · A⁻¹ · 1 = A (×3) |
| (1,1,0),(1,0,1),(0,1,1) | 2 | A · A⁻² · δ (×3) |
| (1,1,1) | 3 | A⁻³ · δ² |

After collecting terms:
```
⟨3₁⟩ = -A⁻⁴ - A⁻¹² + A⁻¹⁶    (left-handed trefoil)
```

Writhe correction: f_{3₁} = (-A³)^{-(-3)} ⟨3₁⟩ = (-A³)³ ⟨3₁⟩

Substituting A = t^{-1/4}:
```
V_{3₁}(t) = -t⁻⁴ + t⁻³ + t⁻¹    (left-handed trefoil)
```

This is the **ground truth** for our Jones polynomial implementation test.

### 7.4 Memoization and State Representation

The recursive Kauffman bracket has exponential branching but massive subproblem
reuse — many different expansions lead to the same diagram. Effective memoization
requires a canonical **diagram state representation**:

```python
from functools import lru_cache
from typing import FrozenSet

# A diagram state: a frozenset of circles (each circle = frozenset of arc indices)
DiagramState = frozenset[frozenset[int]]

@lru_cache(maxsize=None)
def bracket(state: DiagramState, 
            crossings_remaining: tuple,
            ring_A: LaurentPoly,
            ring_A_inv: LaurentPoly,
            delta: LaurentPoly) -> LaurentPoly:
    """Kauffman bracket via memoized recursion over crossing states."""
    if not crossings_remaining:
        # Base case: count circles
        n_circles = len(state)
        return delta ** (n_circles - 1)
    
    c, *rest = crossings_remaining
    rest = tuple(rest)
    
    # A-smoothing: apply 0-resolution to crossing c
    state_0 = apply_smoothing(state, c, '0')
    # A⁻¹-smoothing: apply ∞-resolution to crossing c
    state_inf = apply_smoothing(state, c, 'inf')
    
    return (ring_A * bracket(state_0, rest, ring_A, ring_A_inv, delta) +
            ring_A_inv * bracket(state_inf, rest, ring_A, ring_A_inv, delta))
```

### 7.5 Online Writhe Scale — Implementation Design

The **online writhe scale** is the time series of writhe values across MC snapshots,
displayed as a live-updating plot in the ALMA disorder panel.

```
writhe_series[t] = Wr(backbone at snapshot t)
```

Visual display:
- X-axis: MC snapshot index (time)
- Y-axis: writhe value (integer for polygonal curves)
- Color: writhe value mapped to colormap (blue = negative, red = positive)
- Overlay: running mean ± σ band (shows trend vs fluctuation)
- Vertical dashed lines at identified basin transitions (from LandscapeWorker communities)

Statistical analysis of the writhe series:
```python
def writhe_statistics(wr_series: np.ndarray) -> dict:
    return {
        'mean':     wr_series.mean(),
        'variance': wr_series.var(),
        'autocorr_time': autocorrelation_time(wr_series),
        'lyapunov': benettin_1d(wr_series),     # 1D Lyapunov of the writhe signal
        'spectral_exponent': power_law_fit(power_spectrum(wr_series)),
    }
```

The `spectral_exponent` α in S(f) ∝ f^{-α}:
- α ≈ 0: white noise (random, uncorrelated writhe) → extreme IDP
- α ≈ 1: 1/f noise (long-range correlations) → marginal / partially disordered
- α ≈ 2: Brownian noise (integrated random walk) → intermediate
- α >> 2: Lorentzian (single correlation time) → ordered protein

---

## 8. Khovanov Homology

### 8.1 Categorification — The Fundamental Idea

The Jones polynomial V_K(t) is a **number** (a polynomial) attached to each knot.
Categorification replaces this number with a **chain complex** — a richer algebraic
object — such that the original number is recovered as the Euler characteristic.

**The basic idea of categorification**:
- Replace an integer n by a vector space V with dim(V) = n
- Replace a sum Σ nᵢ by a graded vector space ⊕ᵢ Vᵢ with dim(Vᵢ) = nᵢ
- Replace an alternating sum Σ (-1)ⁱ nᵢ by the Euler characteristic χ = Σ (-1)ⁱ dim(Hᵢ)
  of a chain complex with H_i = i-th homology

The **Grothendieck group** K₀(C) of an abelian category C is the universal
group receiving Euler characteristics: [M] - [N] for exact sequences 0→N→M→P→0.
Categorification is the process of "lifting" a quantity in K₀(C) to an actual
object of C.

For Khovanov: 
- V_K(t) ∈ ℤ[t,t⁻¹] is a class in K₀(graded ℤ-modules)
- CKh(K) is the actual chain complex — an object of Chain(ℤ-Mod)
- χ(CKh(K)) = V_K(t) in K₀

**Why strictly stronger**: two different chain complexes can have the same Euler
characteristic. Khovanov homology Kh(K) = H*(CKh(K)) distinguishes them.

### 8.2 The Cube of Resolutions — Complete Construction

Given a knot diagram K with n crossings labeled 1,...,n.

**Step 1: The state cube**. Each vertex v = (v₁,...,vₙ) ∈ {0,1}ⁿ specifies a
**complete smoothing**: for each crossing i, choose the 0-smoothing (vᵢ=0) or
1-smoothing (vᵢ=1). The homological degree of vertex v is |v| = Σvᵢ.

**Step 2: Circle counts**. For each vertex v, count the number of circles circles(v)
in the complete smoothing. (This requires tracking how the smoothings connect the strands.)

**Step 3: Chain groups**. The chain group at homological degree i is:
```
CKh^i(K) = ⊕_{v : |v|=i} V^{⊗ circles(v)}
```
where V = ℤ{v₊, v₋} with |v₊| = +1, |v₋| = -1 (quantum grading).

The total chain group (before grading) has rank:
```
rank CKh^i(K) = Σ_{v : |v|=i} 2^{circles(v)}
```

**Step 4: Quantum grading shift**. The actual bigrading uses shifted degrees.
For a diagram with n₊ positive crossings and n₋ negative crossings:
- Homological shift: i → i - n₋
- Quantum shift: j → j + n₊ - 2n₋

This shift ensures the homology is an invariant (not just of the diagram but of
the knot itself) — it corrects for the choice of diagram orientation.

**Step 5: Edge maps (boundary operators)**.
Each edge of the cube corresponds to flipping one bit: v → v' where v'_k = v_k + 1
for exactly one position k, and v'_j = v_j for j ≠ k.

This edge corresponds to a local cobordism (a "zip" or "unzip") at crossing k:
- If crossing k changes from 0 to 1, and this causes two circles to merge: apply m
- If crossing k changes from 0 to 1, and this causes one circle to split: apply Δ

**Sign assignment**: the edge from v to v' gets a sign (-1)^{s(v,k)} where:
```
s(v, k) = Σ_{j < k} v_j    (number of 1-bits before position k)
```

This is the **Koszul sign convention**, ensuring ∂² = 0. Explicitly:
```
∂(generator at vertex v) = Σ_{k : v_k=0} (-1)^{s(v,k)} · edge_map_{v→v+eₖ}(generator)
```

### 8.3 The Frobenius Algebra — Complete Specification

The algebra (V, m, η, Δ, ε) with V = ℤ{v₊, v₋}:

```python
# All maps written in terms of the basis {v₊, v₋}
# using dict notation: {basis_element: coefficient}

def m(a: str, b: str) -> dict[str, int]:
    """Multiplication: V ⊗ V → V  (merge two circles into one)"""
    table = {
        ('vp', 'vp'): {'vp': 1},
        ('vp', 'vm'): {'vm': 1},
        ('vm', 'vp'): {'vm': 1},
        ('vm', 'vm'): {},           # = 0
    }
    return table[(a, b)]

def eta() -> dict[str, int]:
    """Unit: ℤ → V  (create a circle from nothing — cup cobordism)"""
    return {'vp': 1}

def delta(a: str) -> dict[str, int]:
    """Comultiplication: V → V ⊗ V  (split one circle into two)"""
    table = {
        'vp': {('vp','vm'): 1, ('vm','vp'): 1},   # v₊ ↦ v₊⊗v₋ + v₋⊗v₊
        'vm': {('vm','vm'): 1},                     # v₋ ↦ v₋⊗v₋
    }
    return table[a]

def epsilon(a: str) -> dict[str, int]:
    """Counit: V → ℤ  (destroy a circle — cap cobordism)"""
    return {(): 1} if a == 'vm' else {}
```

**Quantum grading** of the basis elements: deg_q(v₊) = +1, deg_q(v₋) = -1.
The maps m, Δ change quantum degree by:
```
m:   deg_q(output) = deg_q(a) + deg_q(b) - 1  (degree shift -1)
Δ:   deg_q(outputs) = deg_q(input) + 1 each    (degree shift +1)
```
These shifts ensure the boundary map preserves the quantum grading j.

### 8.4 Worked Example — Unknot and Hopf Link

**Unknot** (0 crossings, 1 circle):
```
CKh^0 = V = ℤ{v₊, v₋}      (one vertex (empty), one circle)
All other CKh^i = 0
∂ = 0 (no edges in the cube)
```

Homology: Kh^{0,1}(unknot) = ℤ, Kh^{0,-1}(unknot) = ℤ, all others = 0.
Jones: V(q) = q + q⁻¹ (= q^1·1 + q^{-1}·1). ✓

**Trefoil 3₁** (3 crossings, all negative):

n = 3, n₋ = 3, n₊ = 0.
There are 2³ = 8 vertices. The circle counts are:
```
v = (0,0,0): 2 circles → CKh^0 contributes V^⊗2 = ℤ{v₊⊗v₊, v₊⊗v₋, v₋⊗v₊, v₋⊗v₋}
v = (1,0,0), (0,1,0), (0,0,1): 1 circle → each contributes V = ℤ{v₊, v₋}
v = (1,1,0), (1,0,1), (0,1,1): 2 circles → each contributes V^⊗2
v = (1,1,1): 3 circles → contributes V^⊗3
```

After computing all boundary maps and homology (by SNF over ℤ):
```
Kh^{-3,-9}(3₁) = ℤ
Kh^{-2,-5}(3₁) = ℤ
Kh^{0,-1}(3₁) = ℤ
Kh^{0,1}(3₁) = ℤ
All others = 0.
```

Euler characteristic check:
```
Σ (-1)^i q^j rank(Kh^{i,j}) 
= (-1)^{-3} q^{-9} + (-1)^{-2} q^{-5} + q^{-1} + q^1
= -q^{-9} + q^{-5} + q^{-1} + q
```

Under the substitution q = -t^{-1/2} - t^{1/2} → Jones: V(t) = -t⁻⁴ + t⁻³ + t⁻¹. ✓

### 8.5 Invariance Under Reidemeister Moves

**R1 invariance** (degenerate complex cancellation):
Adding an R1 curl at crossing k introduces a new vertex where one smoothing
gives a disjoint circle. The resulting chain complex is chain homotopy equivalent
to the original — the new generator is either a cycle that bounds, or a boundary.
Algebraically: the new complex = old complex ⊕ acyclic complex.

**R2 invariance** (delooping):
Adding two crossings for R2 gives a chain complex where there is a degree-1
chain map (a homotopy equivalence) between the new complex and the old one.
Key step: the "delooping" lemma — if a chain complex C has a contractible
direct summand, H*(C) ≅ H*(C / contractible summand).

**R3 invariance**:
Requires showing that two different ways to perform R3 (via different sequences
of R1, R2) give chain-homotopy-equivalent complexes. Proved by Bar-Natan (2002)
using the "movie moves" formalism.

The invariance is at the level of **chain homotopy equivalence** (not just homology),
making Khovanov homology an invariant of the *chain homotopy type* — an even finer
invariant than the homology groups alone.

### 8.6 Protein-Specific Khovanov Data

For the proteins in ALMA's `data/` directory:

| Protein | UniProt | Known fold | Expected knot type | Expected Jones |
|---|---|---|---|---|
| Insulin (P01308) | P01308 | 2 chains, 3 disulfide bonds | unknotted | V = 1 |
| p53 (P04637) | P04637 | tetramer, Zn-binding | unknotted | V = 1 |
| APP (P05067) | P05067 | type-1 membrane protein | unknotted | V = 1 |
| Tau (P10636) | P10636 | IDP, no fixed fold | unknotted | V = 1 |
| α-Synuclein (P37840) | P37840 | IDP, no fixed fold | unknotted | V = 1 |

None of the current ALMA test proteins are knotted (the known knotted proteins
MJ0366, UCH-L1 are not in the current dataset). This means:
1. The Jones/Khovanov pipeline should return V = 1 for all current proteins.
2. Non-trivial writhe (even for unknotted proteins) is still informative: it
   measures the *amount of twisting* under projection, not whether the knot is
   non-trivial.
3. Adding MJ0366 (trefoil, available from RCSB as PDB: 1J85) as a test case
   is required for validating the Jones/Khovanov computation.

---

## 9. Chern-Simons Field Theory

### 9.1 The Chern-Simons Action — Motivation and Definition

The **Chern-Simons (CS) functional** is the unique gauge-invariant, diffeomorphism-
invariant functional on connections over a 3-manifold that is of degree 3 in the
connection 1-form and does not require a metric. These properties make it
**topological** — its critical points and partition function encode topological,
not geometric, information.

For a 3-manifold M with gauge group G = SU(2), with connection A (a 1-form on M
valued in the Lie algebra su(2)):

```
S_CS[A] = (k/4π) ∫_M Tr(A ∧ dA + (2/3) A ∧ A ∧ A)
```

- k ∈ ℤ is the **level** — quantized because of gauge invariance under large gauge
  transformations (which shift S_CS by 8π²k · winding number)
- Tr is the trace in the fundamental representation of SU(2) (2×2 matrices)
- A ∧ dA is a 3-form: the gauge-field kinetic term
- A ∧ A ∧ A is the Chern-Simons cubic interaction

**Equations of motion**: δS_CS/δA = 0 gives F_A = dA + A∧A = 0 (flat connection).

**Physical interpretation**: unlike Yang-Mills theory (S_YM = ∫ Tr(F∧*F), which
requires a metric for the Hodge dual *), CS theory has no metric dependence.
The theory has no local (propagating) degrees of freedom — it is a **topological
quantum field theory** (TQFT). All observables are topological invariants.

### 9.2 Wilson Loops — Observables in CS Theory

The physical observables in any gauge theory are **gauge-invariant** quantities.
For CS theory, the natural gauge-invariant observables are **Wilson loops**:

```
W_R(K) = Tr_R[P exp(∮_K A)]
```

where:
- K ⊂ M is a knot (closed curve in the 3-manifold)
- R is a representation of G = SU(2) (e.g., the spin-j representation)
- P exp denotes the path-ordered exponential (a matrix-valued line integral)
- Tr_R denotes the trace in representation R

Physical interpretation: W_R(K) is the **holonomy** of the connection A around
the loop K, measured in representation R. In gauge theory language, it measures
the phase accumulated by a particle in representation R moving around K.

The **expectation value** in CS theory:
```
⟨W_R(K)⟩ = (1/Z) ∫ DA  W_R(K)  exp(i S_CS[A])
```
is the path integral over all connections A, weighted by the CS action.

### 9.3 Witten's Theorem — The Exact Connection

**Theorem** (Witten, Commun. Math. Phys. 121, 1989):

For G = SU(2), 3-manifold M = S³ (or ℝ³, which has the same effect for knots
in a ball), level k, and the spin-1/2 representation R = 2 (the fundamental):

```
⟨W_{1/2}(K)⟩_{CS, k} = V_K(q)    where q = e^{2πi/(k+2)}
```

The Jones polynomial V_K(q) is the expectation value of the fundamental Wilson
loop in SU(2) Chern-Simons theory at level k, evaluated at the root of unity
q = exp(2πi/(k+2)).

**Why this matters for proteins**:
1. The Jones polynomial is now a *physical quantity* — a quantum mechanical
   observable in a 3D gauge theory whose "space" is the protein's ambient ℝ³.
2. The protein backbone K acts as the Wilson loop — a probe of the CS gauge field.
3. The "CS gauge field" is an abstract mathematical object encoding the topology
   of ℝ³ as "seen by" the curve K.
4. Different protein backbone topologies → different Wilson loop expectation values
   → different Jones polynomials.

For practical computation, we do NOT need to simulate the CS path integral.
The theorem says: **computing V_K(q) by skein recursion IS the same as computing
⟨W_{1/2}(K)⟩**. The skein algorithm IS the CS computation.

### 9.4 Relation to KAM Theory — The Arnold-Maslov Index

This is the deepest theoretical connection in this project. The bridge:

**Arnold-Liouville theorem** (§3.1): KAM tori are Lagrangian submanifolds L ⊂ T*M
of the 2n-dimensional phase space (here: the protein's conformational space).

**Maslov index** of a Lagrangian submanifold L: an integer μ(L) ∈ ℤ that measures
how many times L "crosses" the "Maslov cycle" (a codimension-1 stratum of the
Lagrangian Grassmannian) as one traverses a loop in L. It controls WKB quantization:
the quantum energy levels are E_n ∼ ℏ(I + μ/4) (Maslov quantization).

**Connection to Chern-Simons**: the CS action evaluated on a flat connection A*
over a 3-manifold with boundary Σ (a Seifert surface) equals:
```
S_CS[A*] = π · ρ(Σ, A*)
```
where ρ is the **Rho invariant** (Atiyah-Patodi-Singer), which equals the
**Maslov index** of the associated Lagrangian submanifold (defined by the flat
connection on the boundary).

**The protein interpretation**:
- The KAM tori (stable conformational basins) are Lagrangian submanifolds
- Their Maslov index μ measures the topological "winding" of the basin
- Via the CS/Maslov connection, μ appears in the CS expansion coefficient
- A protein with high Maslov index has both topologically complex KAM tori
  AND a non-trivial CS/Jones invariant

**Practical consequence**: we expect a correlation between:
- High KAM stability score (intact tori, ordered protein)
- Non-trivial Maslov index μ(L) ≠ 0
- Non-trivial CS invariant / Jones polynomial

And conversely:
- IDP (broken tori, chaotic) → trivial Maslov index (no coherent Lagrangian structure)
- Trivial Jones polynomial (unknotted backbone)

This correlation is the *KAM-Khovanov bridge* — a theoretical prediction to be
tested by running the ALMA pipeline on the known knotted/ordered proteins.

### 9.5 The Reshetikhin-Turaev Construction — Algebraic Reformulation

For computational purposes, Witten's path integral can be made rigorous via the
**Reshetikhin-Turaev** (RT) construction (1990-1991), which does not use path
integrals at all but instead uses the **quantum group** U_q(sl₂):

```
⟨W_{1/2}(K)⟩_CS = RT invariant of K using U_q(sl₂) in the fundamental representation
                 = Jones polynomial V_K(q)
```

The quantum group U_q(sl₂) is a deformation of the universal enveloping algebra
U(sl₂) parametrized by q ∈ ℂ. Its representation theory (for generic q) mirrors
that of sl₂ but for q a root of unity, the theory is "truncated" (only finitely
many representations survive) — this truncation corresponds to the CS level k.

**For our implementation**: the skein algorithm already implicitly uses the
U_q(sl₂) representation theory (the skein relation IS the defining relation of the
Hecke algebra, which is the endomorphism algebra of tensor representations of U_q(sl₂)).
We do not need to implement quantum groups explicitly.

### 9.6 Topological Quantum Field Theory Structure

The CS theory is an example of a **(2+1)D TQFT** in the Atiyah-Segal sense:
a symmetric monoidal functor Z: Cob₂₊₁ → Vect assigning:
- To each closed 2-manifold Σ: a vector space Z(Σ) (the "Hilbert space")
- To each 3-manifold M with ∂M = Σ_out ⊔ Σ_in: a linear map Z(M): Z(Σ_in) → Z(Σ_out)

This is exactly the categorical structure of the Khovanov functor (§5.4).
The Khovanov complex CKh(K) is the "state space" of K viewed as a TQFT observable.

**Explicitly for SU(2) CS at level k**:
- Z(S¹) = V_k = ℂ^{k+1} (the space of SU(2) representations up to level k)
- Z(disk) ∈ V_k (a specific vector)
- Z(pair of pants) : V_k ⊗ V_k → V_k (the quantum group multiplication)

The Khovanov TQFT (§5.6) is the k→∞ (generic q) limit of this, with V_∞ = V = ℤ{v₊,v₋}.


## 10. Connecting Chaos to Topology — IDP Hypothesis

### 10.1 The Full Classification Framework

The joint (dynamical, topological) classification can be visualized as a 2D plane:

```
            Lyapunov λ₁
               │
  CHAOTIC      │      IDP ZONE
  KNOTTED      │    (λ₁>0, V=1)
  (rare)       │
               │──────────────── λ₁ = 0
               │
  ORDERED      │    ORDERED UNKNOTTED
  KNOTTED      │    (λ₁<0, V=1)
  (λ₁<0, V≠1) │
               └────────────────────── Jones V_K(t)
               V=1             V≠1
```

**Quadrant interpretation**:

| λ₁ | V_K | Classification | Example |
|---|---|---|---|
| < 0 | ≠ 1 | Ordered knotted | MJ0366 (trefoil), UCH-L1 (4₁) |
| < 0 | = 1 | Ordered unknotted | Most globular proteins (typical) |
| > 0 | = 1 | IDP | α-Synuclein (P37840), Tau (P10636) |
| > 0 | ≠ 1 | Knotted IDP (?) | Theoretically possible, not observed |

### 10.2 Multi-Scale Disorder Indicators

The new topology module provides disorder indicators at multiple scales, complementing
ALMA's existing IUPred (sequence-based) and RMSF (local, per-residue) measures:

| Scale | Indicator | Source | Meaning |
|---|---|---|---|
| Sequence | IUPred score per residue | `iupred.py` (existing) | Local residue disorder propensity |
| Local structural | RMSF per residue (Å) | `LandscapeWorker` (existing) | Per-residue fluctuation amplitude |
| Global dynamical | Lyapunov exponent λ₁ | `topology/lyapunov.py` (new) | Exponential divergence rate of trajectories |
| Basin structure | KAM score K_protein | `topology/kam.py` (new) | Thermal noise vs. barrier height |
| Topological, diagram | Online writhe w(t) | `topology/knot.py` (new) | Backbone twist under projection |
| Topological, invariant | Jones polynomial V_K(t) | `topology/jones.py` (new) | Knot type (projection-independent) |
| Topological, categorified | Khovanov homology Kh^{i,j} | `topology/khovanov.py` (new) | Full topological fingerprint |

### 10.3 Formal Classification Criteria

**ORDERED** (high confidence) — ALL of:
```
λ₁ < -0.01          (converging dynamics, stable basin)
K_protein < 0.5      (thermal energy << barriers, KAM regime)
σ²_writhe < 1.0      (stable backbone topology under projection)
```

**ORDERED** (moderate confidence) — ALL of:
```
-0.05 < λ₁ < 0.01   (near-zero, marginally converging)
K_protein < 0.8
σ²_writhe < 2.0
```

**POSSIBLY DISORDERED** — ANY of:
```
0 ≤ λ₁ < 0.05       (marginally positive chaos)
0.8 ≤ K_protein < 1.3  (near Chirikov threshold)
σ²_writhe ≥ 2.0 AND λ₁ ≥ 0   (topological instability + marginal dynamics)
```

**IDP** (high confidence) — ANY of:
```
λ₁ > 0.05           (clearly positive Lyapunov exponent)
K_protein > 1.3      (past KAM breakdown)
σ²_writhe ≥ 4.0 AND λ₁ > 0   (both dynamical and topological chaos)
```

The thresholds (0.05, 0.8, 1.3, etc.) are **initial estimates** to be calibrated
against the ALMA test proteins (insulin, p53, APP = ordered; α-synuclein, tau = IDP).

### 10.4 The Writhe as Bridge Observable

The online writhe w(t) simultaneously:
1. Is a **topological quantity**: counts signed crossings of the backbone under projection
2. Is **dynamically computed**: evaluated at every MC snapshot (a dynamical trajectory)
3. Has **both** a static value (knot type contribution) and dynamic fluctuations (topology chaos)

The writhe decomposes into two components:
```
w(t) = w₀ + δw(t)
```
where w₀ = ⟨w⟩ (time-averaged writhe, related to persistent fold topology) and
δw(t) = w(t) - w₀ (fluctuation, measuring topological chaos).

For an ordered protein: w₀ ≠ 0 (persistent helical structure contributes writhe),
δw(t) small (stable topology → small fluctuations).

For an IDP: w₀ ≈ 0 (no persistent fold), δw(t) large (random chain writhe fluctuations).

### 10.5 Expected Results for ALMA's Test Proteins

From the literature and the physics:

**Insulin (P01308)**: disulfide-bonded, highly ordered two-chain structure.
Expected: λ₁ << 0, K << 1, w₀ ≠ 0 (α-helical structure), V_K = 1 (unknotted).

**p53 (P04637)**: tetramer with ordered DNA-binding domain + disordered N/C-terminal regions.
Expected: λ₁ near 0 (mixed ordered/disordered), K ≈ 0.8, V_K = 1.
This is an interesting test case for the "POSSIBLY DISORDERED" category.

**APP (P05067)**: type-1 membrane protein, ordered ectodomain.
Expected: λ₁ < 0 (ordered), V_K = 1.

**α-Synuclein (P37840)**: canonical IDP, Parkinson's disease protein.
Expected: λ₁ > 0 (chaotic), K > 1 (past KAM breakdown), σ²_writhe large, V_K = 1.
This is the primary IDP positive control.

**Tau (P10636)**: canonical IDP, Alzheimer's disease protein (longest: ~758 residues).
Expected: same as α-synuclein but larger system. Good stress test for scaling.

---

## 11. Implementation Roadmap

### Phase 0 — Branch Setup ✓
```
✓ branch: feature/topological-mapping
  Create:  python/topology/__init__.py
           tests/test_topology.py (stubs for all phases)
```

### Phase 1 — Dynamical Analysis (Weeks 1-2)

Implements the Lyapunov and KAM analysis. These have no dependencies on Phase 2-4
and provide immediate value (better IDP classifier).

```
P1.1  python/topology/lyapunov.py
        Classes:
          BenettinLLE — largest Lyapunov exponent via paired trajectories
          LyapunovSpectrum — first k exponents via QR/Gram-Schmidt
          FiniteTimeLyapunov — FTLE for each snapshot (visualization)
          KaplanYorkeDim — D_KY from spectrum
        Functions:
          benettin_lle(engine, topo, init_atoms, n_steps, eps, T) → float
          lyapunov_spectrum(engine, topo, init_atoms, k, n_steps) → np.ndarray
          ftle_series(engine, topo, snapshots, eps) → np.ndarray
          kaplan_yorke_dim(spectrum: np.ndarray) → float
        Key design:
          Shadow trajectory uses same RNG sequence as reference (same seed)
          Perturbation is applied ONLY to Cα coordinates (not sidechain atoms)
          Renormalization uses ||δx|| in 3N-dimensional Cα space

P1.2  python/topology/kam.py
        Classes:
          BasinGraph — builds basin adjacency from PCA + community data
          KAMAnalyzer — computes KAM score, winding numbers, rotation number
        Functions:
          estimate_barriers(energies, communities, layout) → dict[pair, float]
          kam_score(T, barriers) → float
          rotation_number(layout, basin_center, n_steps=50) → float
          winding_numbers(layout, communities, energies) → np.ndarray
        Input:  LandscapeWorker.result dict (energies, layout, communities)
        Output: KAMResult{score, barriers, rotation_numbers, kam_regime}

P1.3  Integration into gui_main.py
        - Add topology result dict to LandscapeWorker.result
        - New sidebar metric: "LYAPUNOV λ₁" (replaces or augments IDP label)
        - FTLE values color-coded on landscape PCA nodes
        - KAM score displayed alongside basin count
        - Updated IDP classification decision tree using λ₁ + K
```

### Phase 2 — Algebraic Infrastructure (Weeks 3-4)

Pure Python, no dependencies on Phase 1 (can develop in parallel).

```
P2.1  python/topology/algebra.py
        Classes:
          LaurentPoly — ℤ[t, t⁻¹] arithmetic (add, mul, evaluate, __repr__)
          FreeModule  — free ℤ-module as dict[str, int] with operations
          LinearMap   — integer matrix between free modules (apply, compose)
          ChainComplex — list of FreeModules + LinearMaps with verify() + homology()
          BidegreeChainComplex — bigraded chain complex for Khovanov
        Functions:
          smith_normal_form(matrix) → (U, D, V) over ℤ (via sympy fallback)
          compute_homology_snf(boundary_in, boundary_out) → HomologyGroup
          HomologyGroup: {rank: int, torsion: list[int]}

P2.2  python/topology/category.py
        Classes:
          Functor     — (on_obj, on_mor) with verification
          KhovanovFunctor — Functor[Kob, Chain(ℤ)] (the central construction)
          FrobeniusAlgebra — (V, m, eta, delta, epsilon) with axiom verification
        Instances:
          FROBENIUS_Z — the standard Khovanov Frobenius algebra over ℤ
          FROBENIUS_F2 — Khovanov over 𝔽₂ (no sign issues, for quick tests)
```

### Phase 3 — Knot Theory (Weeks 5-6)

Depends on Phase 2 (LaurentPoly). Phase 1 not required.

```
P3.1  python/topology/knot.py
        Classes:
          KnotDiagram   — crossings + Gauss code + writhe
          BackboneClosure — deterministic and probabilistic closure methods
        Functions:
          close_backbone(ca_coords, method='deterministic') → np.ndarray
          project_to_diagram(closed_curve, proj_dir) → KnotDiagram
          compute_gauss_code(diagram) → list[tuple[int,int,int]]
          writhe(gauss_code) → int
          online_writhe_series(snapshots, ca_indices) → np.ndarray
          knotting_probability_profile(ca_coords, n_samples=100) → np.ndarray
          minimize_crossings_projection(closed_curve) → (KnotDiagram, np.ndarray)
          identify_knot_type(n_crossings, jones_poly) → str  # table lookup

P3.2  python/topology/jones.py
        Functions:
          kauffman_bracket(gauss_code) → LaurentPoly      (memoized recursion)
          jones_polynomial(gauss_code) → LaurentPoly      (bracket + writhe correction)
          jones_series(snapshots, ca_indices) → list[LaurentPoly]
          jones_distance(p1: LaurentPoly, p2: LaurentPoly) → float
              # Euclidean distance in coefficient space (for comparing snapshots)
        Constants:
          JONES_UNKNOT  = LaurentPoly({0: 1})          # V = 1
          JONES_TREFOIL_L = LaurentPoly({-4:-1,-3:1,-1:1})  # left-handed 3₁
          JONES_TREFOIL_R = LaurentPoly({4:-1, 3:1, 1:1})   # right-handed 3₁
          JONES_FIGURE8   = LaurentPoly({-2:1,-1:-1,0:1,1:-1,2:1}) # 4₁
```

### Phase 4 — Khovanov Homology (Weeks 7-9)

Depends on Phases 2 and 3.

```
P4.1  python/topology/khovanov.py
        Classes:
          Smoothing      — a complete smoothing: frozenset of circles
          CubeVertex     — vertex of the resolution cube: (bit_string, Smoothing)
          KhovanovComplex — full bigraded chain complex CKh(K)
        Functions:
          all_smoothings(gauss_code) → dict[tuple, Smoothing]
              # 2^n vertices of the cube
          chain_group(vertex, frobenius) → dict[tuple[int,int], FreeModule]
              # V^⊗circles at that vertex, with bigrading
          edge_map(v_from, v_to, frobenius) → LinearMap
              # m or Δ depending on whether circles merge or split
          boundary_map(i, all_vertices, frobenius) → LinearMap
              # sum of signed edge maps at homological degree i
          khovanov_complex(gauss_code, frobenius=FROBENIUS_Z) → KhovanovComplex
          khovanov_homology(kc: KhovanovComplex) → dict[tuple[int,int], HomologyGroup]
          khovanov_polynomial(homology) → LaurentPoly  # in two variables q, t
          verify_decategorification(homology, jones_poly) → bool
              # checks Euler characteristic = Jones

P4.2  Visualization in gui_main.py
        - Khovanov table (i-rows, j-columns, rank/torsion in cells) in disorder panel
        - Togglable: "TOPOLOGY" button → shows Khovanov table for best MC conformation
        - Color-coded: free ℤ summands in blue, ℤ/nℤ torsion in orange
```

### Phase 5 — Chern-Simons (Research Phase, Weeks 10+)

```
P5.1  python/topology/chern_simons.py
        Functions:
          jones_at_root_of_unity(jones_poly, k) → complex
              # evaluate V_K at q = exp(2πi/(k+2)) — the CS level-k invariant
          cs_invariant_series(snapshots, ca_indices, k=5) → np.ndarray
              # CS invariant over the trajectory
          maslov_index_estimate(landscape_result) → int
              # estimates Maslov index from basin winding numbers
          cs_power_spectrum(cs_series) → (np.ndarray, np.ndarray)
              # (frequencies, spectral density) — Lorentzian vs 1/f test
```

---

## 12. Module Architecture & Data Flow

### 12.1 Directory Layout

```
python/
├── gui_main.py              (existing — modified in P1.3 and P4.2)
├── amber_params.py          (existing — unchanged)
├── iupred.py                (existing — unchanged)
└── topology/
    ├── __init__.py           public API exports
    ├── lyapunov.py           § Phase 1
    ├── kam.py                § Phase 1
    ├── algebra.py            § Phase 2
    ├── category.py           § Phase 2
    ├── knot.py               § Phase 3
    ├── jones.py              § Phase 3
    ├── khovanov.py           § Phase 4
    └── chern_simons.py       § Phase 5

tests/
├── bridge_test.py           (existing — unchanged)
└── test_topology.py
    ├── TestLyapunov
    │     test_lle_ordered_basin      # mock stable basin → λ₁ < 0
    │     test_lle_random_walk        # mock random walk → λ₁ ≈ 0
    │     test_kaplan_yorke           # known spectrum → verify D_KY
    ├── TestKAM
    │     test_kam_score_deep_well    # large barrier → K < 0.5
    │     test_kam_score_flat         # tiny barrier → K > 1.3
    │     test_rotation_number        # circular PCA trajectory → ρ = 1
    ├── TestAlgebra
    │     test_laurent_poly_add_mul   # arithmetic identities
    │     test_chain_complex_verify   # ∂² = 0 for random complex
    │     test_homology_unknot        # H* of unknot complex = ℤ⊕ℤ
    │     test_snf_torsion            # matrix with Z/2 torsion
    ├── TestKnot
    │     test_writhe_trefoil         # w(trefoil) = ±3
    │     test_writhe_unknot          # w(unknot) = 0
    │     test_gauss_code_round_trip  # encode → decode → same knot
    │     test_projection_independence # same knot, diff proj → same Jones
    ├── TestJones
    │     test_jones_unknot           # V(unknot) = 1
    │     test_jones_trefoil_L        # V(3₁_L) = -t⁻⁴+t⁻³+t⁻¹
    │     test_jones_trefoil_R        # V(3₁_R) = -t⁴+t³+t  (mirror)
    │     test_jones_figure8          # V(4₁) known value
    │     test_reidemeister_r2        # apply R2 move → same Jones
    │     test_reidemeister_r3        # apply R3 move → same Jones
    ├── TestKhovanov
    │     test_kh_unknot              # Kh^{0,±1} = ℤ, rest 0
    │     test_kh_trefoil             # known Khovanov table
    │     test_decategorification     # χ(Kh) = Jones for 3₁
    │     test_boundary_squared       # ∂² = 0 for trefoil complex
    └── TestCS
          test_jones_at_root_of_unity  # V_{3₁}(e^{2πi/5}) known value
          test_maslov_estimate         # non-negative integer output
```

### 12.2 Data Flow — Complete Integration

```
gui_main.py
│
├── PipelineWorker.run()
│   │
│   ├── _parse_pdb() → atoms, ca_indices, ca_map, topo, iupred_scores, ca_residues
│   ├── generate_ensemble() → ensemble (list of Particle lists)
│   └── PipelineWorker.finished.emit(ensemble, energies, path,
│                                    ca_indices, ca_map, atoms, topo, extra)
│       │
│       └── on_pipeline_finished():
│           │
│           ├── [existing] render 3D structure
│           ├── [existing] update metric widgets
│           │
│           └── [NEW Phase 3-4] extract best conformation CA coords
│               → topology.knot.close_backbone(best_ca)
│               → topology.knot.compute_gauss_code(closed)
│               → topology.jones.jones_polynomial(gauss_code)  → V_K
│               → topology.khovanov.khovanov_complex(gauss_code)
│               → topology.khovanov.khovanov_homology(kc)       → Kh^{i,j}
│               → update disorder panel: show Jones + Khovanov table
│
└── LandscapeWorker.run()
    │
    ├── [existing] 120×80 MC chain → snapshots, energies
    ├── [existing] PCA → layout
    ├── [existing] NetworkX community detection → communities
    │
    └── [NEW Phase 1] topology analysis:
        ├── topology.lyapunov.benettin_lle(engine, topo, init_atoms)
        │     → λ₁ (float)
        ├── topology.lyapunov.ftle_series(engine, topo, snapshots)
        │     → ftle (np.ndarray, shape: [120])  — one per snapshot
        ├── topology.kam.KAMAnalyzer(energies, communities, layout)
        │     → KAMResult{score, barriers, rotation_numbers}
        ├── topology.knot.online_writhe_series(snapshots, ca_indices)
        │     → writhe_t (np.ndarray, shape: [120])
        └── emit updated result dict:
              {
                # existing keys
                'snapshots':   ...,
                'energies':    ...,
                'layout':      ...,
                'communities': ...,
                'node_comm':   ...,
                'var_exp':     ...,
                'n_sig':       ...,
                'funnel':      ...,
                'e_spread':    ...,
                # new keys
                'lyapunov':    λ₁,
                'ftle':        ftle_array,
                'kam':         KAMResult,
                'writhe_t':    writhe_array,
                'idp_label':   updated_label_using_λ₁_and_K,
              }
```

### 12.3 GUI Changes Summary

**Sidebar metric panel** (adds 2 new metric widgets):
```
┌─────────────┬─────────────┐
│  ATOMS      │  THREADS    │
├─────────────┼─────────────┤
│  BEST ENERGY│  CANDIDATES │
├─────────────┼─────────────┤
│  LYAPUNOV λ │  KAM SCORE  │  ← NEW (Phase 1)
└─────────────┴─────────────┘
```

**Landscape panel** — FTLE overlay:
- Landscape scatter nodes colored by FTLE value (cool-to-warm colormap)
- Low FTLE (blue) = stable basin region
- High FTLE (red) = chaotic region

**Disorder panel** — new "TOPOLOGY" sub-tab:
```
┌─ RESIDUE FLEXIBILITY PROFILE ──────────────────────────────────────┐
│  [IUPRED] [RMSF] [WRITHE] [TOPOLOGY]                               │
│                                                                     │
│  [WRITHE tab]: online writhe time series + power spectrum           │
│  [TOPOLOGY tab]: Jones polynomial + Khovanov table                  │
│                                                                     │
│  Jones polynomial: V_K(t) = -t⁻⁴ + t⁻³ + t⁻¹  [trefoil example] │
│                                                                     │
│  Khovanov homology:                                                 │
│  ┌────┬────┬────┬────┬────┐                                         │
│  │ j╲i│ -3 │ -2 │ -1 │  0 │                                         │
│  ├────┼────┼────┼────┼────┤                                         │
│  │ -9 │ ℤ  │    │    │    │                                         │
│  │ -5 │    │ ℤ  │    │    │                                         │
│  │ -1 │    │    │    │ ℤ  │                                         │
│  │  1 │    │    │    │ ℤ  │                                         │
│  └────┴────┴────┴────┴────┘                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 13. Open Questions & Research Directions

### 13.1 Immediate Empirical Questions (Testable with ALMA)

1. **Lyapunov-IUPred correlation**: does λ₁ (computed from MC) correlate with the
   IUPred per-residue disorder scores already computed in `iupred.py`? Both measure
   disorder but from completely different inputs (dynamics vs. sequence). A strong
   correlation would validate both methods; a discrepancy would reveal cases where
   sequence-predicted disorder disagrees with dynamical disorder.

2. **KAM score vs. RMSF**: does K_protein correlate with the mean RMSF? Both measure
   how "floppy" the protein is, but K_protein is a global thermodynamic ratio while
   RMSF is a structural fluctuation average. Comparing them on ALMA's 8 test proteins
   (P01308, P04637, P05067, P10636, P37840, P21543, Q92793, 1XQ8) would be
   a 1-2 day experiment.

3. **Writhe stability in ordered proteins**: the insulin structure (P01308) has 3
   disulfide bonds and 2 α-helices. α-helices contribute large positive writhe
   under standard projection. Does w₀ ≠ 0 for insulin, and is σ²_writhe << σ²_writhe
   for α-synuclein? This would validate writhe as an order parameter.

4. **Jones polynomial of ALMA's test proteins**: all current proteins are expected
   to give V_K = 1. If any gives V_K ≠ 1, that is a surprising result worth
   investigating (possible numerical artifact or genuine transient knotting in an
   MC snapshot).

### 13.2 The KAM-Khovanov Bridge Conjecture

**Conjecture**: For proteins in ALMA's dataset, the KAM stability score K and the
total rank of Khovanov homology Σ_{i,j} rank(Kh^{i,j}(K)) are negatively correlated:

```
K_protein ↑   ↔   Σ rank(Kh^{i,j}) ↓   (approaching unknot homology rank = 2)
```

More precisely: ordered proteins (low K) tend to have more complex backbone topology
(higher Khovanov rank) because topological complexity (knotting) requires a stable
fold that maintains the topology over time.

This would be a novel result connecting Hamiltonian dynamical systems theory (KAM)
to knot homology (Khovanov). It is **not** known in the mathematical literature
(to the author's knowledge) and would constitute an original contribution.

**How to test**: run the pipeline on:
- α-Synuclein (IDP, high K predicted): expected Kh = unknot homology (rank 2)
- Insulin (ordered, low K predicted): expected Kh = unknot homology (rank 2) — SAME
- MJ0366 (ordered, knotted): expected Kh = trefoil homology (rank 4)

Note: even if the conjecture is false in general, MJ0366 vs α-synuclein provides
the cleanest test (same expected Kh rank for unknotted proteins regardless of K).
The conjecture becomes non-trivial only when comparing knotted vs unknotted proteins.

### 13.3 The Categorified Lyapunov Exponent

Can the Lyapunov exponent be categorified? I.e., is there a chain complex C*(K, T)
(depending on the protein K and temperature T) such that:

```
χ(H*(C*(K,T))) = λ₁(K, T)   ?
```

This would be the most ambitious result: a "Khovanov-like" lift of the dynamical
Lyapunov exponent to a homological invariant. The chain groups would presumably be
related to the "stable and unstable manifolds" of the chaotic attractor (the Morse
complex of the free energy landscape).

**Partial evidence**: in Morse theory, the chain complex built from critical points
and gradient flow lines has Euler characteristic = the Euler characteristic of the
manifold. The free energy landscape has a Morse-like structure (critical points =
conformational basins and transition states). The Lyapunov exponent is related to
the curvature of the landscape near critical points. A Morse complex of the
free energy landscape might categorify λ₁.

### 13.4 Chern-Simons Time Series Analysis

The CS invariant time series:
```
cs_t = V_K^{(t)}(e^{2πi/(k+2)})    for each MC snapshot t
```
is a complex-valued time series derived from the backbone topology. Its properties:

- For an ordered protein: cs_t ≈ constant (stable fold → stable CS invariant)
- For an IDP: cs_t fluctuates, with cs_t = 1 most of the time (unknotted backbone)
  but occasional transient complex values if the backbone transiently self-crosses

The power spectrum of cs_t (or its real/imaginary parts) might show 1/f noise for
IDPs (scale-free conformational fluctuations) vs. white noise for ordered proteins
(stable, non-fluctuating topology). This connects to the broader question of whether
protein dynamics is generically 1/f (fractal, long-memory) or Markovian (memoryless).

### 13.5 Circuit Topology as Complement to Knot Theory

**Circuit topology** (Reidl & Mashaghi, 2015) is a protein-specific topological
framework that classifies the arrangement of contacts (hydrogen bonds, disulfide
bridges, hydrophobic contacts) along the chain into three basic circuit arrangements:
- **Series (S)**: two contacts sharing one end
- **Parallel (P)**: two contacts completely nested
- **Cross (X)**: two contacts interleaved (= topological crossing)

Circuit topology is **not the same as knot theory** — it captures the *contact pattern*
topology rather than the backbone curve topology. But it is:
1. Directly computable from the AMBER force field contact definitions already in ALMA
2. Richer for unknotted proteins than knot theory (which trivially gives V=1 for most proteins)
3. Experimentally validated as a predictor of mechanical properties and folding pathways

A future extension could implement circuit topology analysis alongside the Khovanov
pipeline, giving a complete topological picture at both the backbone (knot) and
contact (circuit) levels.

### 13.6 Computational Complexity Notes

| Computation | Complexity | N=50 residues | N=500 residues |
|---|---|---|---|
| Lyapunov LLE (1000 steps) | O(N · steps) | ~50ms | ~500ms |
| KAM analysis | O(basins² + N) | ~1ms | ~10ms |
| Backbone closure + projection | O(N²) | ~1ms | ~100ms |
| Gauss code (c crossings) | O(N²) | 0-3 crossings | 0-5 crossings |
| Jones polynomial (c crossings) | O(2^c) | 2^3=8 terms | 2^5=32 terms |
| Khovanov complex (c crossings) | O(2^c · c) | fast | fast |
| Smith normal form (m×m matrix) | O(m³) | ~1ms | ~100ms |
| Knotting probability (100 closures) | O(100 · N²) | ~0.1s | ~10s |

All computations except knotting probability profile are fast enough for interactive
use. The knotting probability profile should be run in a background thread (as a
new `TopologyWorker` QThread in `gui_main.py`).

---

## 14. References

### Dynamical Systems & Chaos
- Oseledets, V.I. (1968). "A multiplicative ergodic theorem." *Trudy MMO* 19, 179-210.
- Benettin, G., Galgani, L., Giorgilli, A., Strelcyn, J.-M. (1980). "Lyapunov
  characteristic exponents for smooth dynamical systems." *Meccanica* 15, 9-30.
- Kaplan, J.L. & Yorke, J.A. (1979). "Chaotic behavior of multidimensional difference
  equations." *Lecture Notes in Mathematics* 730, 204-227.
- Chirikov, B.V. (1979). "A universal instability of many-dimensional oscillator
  systems." *Physics Reports* 52(5), 263-379.

### KAM Theory
- Kolmogorov, A.N. (1954). "On conservation of conditionally periodic motions."
  *Dokl. Akad. Nauk USSR* 98, 527-530.
- Arnold, V.I. (1963). "Proof of A.N. Kolmogorov's theorem on the preservation of
  quasi-periodic motions." *Russian Math. Surveys* 18(5), 9-36.
- Moser, J. (1962). "On invariant curves of area-preserving mappings of an annulus."
  *Nachr. Akad. Wiss. Göttingen* 1-20.
- Arnold, V.I. (1964). "Instability of dynamical systems with many degrees of freedom."
  *Sov. Math. Dokl.* 5, 581-585.  [Arnold diffusion]

### Knot Theory
- Reidemeister, K. (1926). "Elementare Begründung der Knotentheorie."
  *Abh. Math. Sem. Hamburg* 5, 24-32.
- Alexander, J.W. (1928). "Topological invariants of knots and links."
  *Trans. AMS* 30, 275-306.
- Jones, V.F.R. (1985). "A polynomial invariant for knots via von Neumann algebras."
  *Bull. AMS* 12, 103-111.
- Kauffman, L.H. (1987). "State models and the Jones polynomial."
  *Topology* 26(3), 395-407.
- Freyd, P., Yetter, D., Hoste, J., Lickorish, W.B.R., Millett, K., Ocneanu, A. (1985).
  "A new polynomial invariant of knots and links." *Bull. AMS* 12, 239-246. [HOMFLY-PT]

### Khovanov Homology
- Khovanov, M. (2000). "A categorification of the Jones polynomial."
  *Duke Math. J.* 101(3), 359-426.
- Bar-Natan, D. (2002). "On Khovanov's categorification of the Jones polynomial."
  *Algebr. Geom. Topol.* 2, 337-370.
- Bar-Natan, D. (2005). "Khovanov's homology for tangles and cobordisms."
  *Geom. Topol.* 9, 1443-1499.
- Turner, P. (2017). "A hitchhiker's guide to Khovanov homology."
  arXiv:1409.6442 (expository — recommended entry point).

### Chern-Simons Theory & TQFT
- Witten, E. (1989). "Quantum field theory and the Jones polynomial."
  *Commun. Math. Phys.* 121, 351-399.
- Reshetikhin, N. & Turaev, V. (1991). "Invariants of 3-manifolds via link polynomials
  and quantum groups." *Invent. Math.* 103, 547-597.
- Atiyah, M. (1988). "Topological quantum field theory." *Publ. IHES* 68, 175-186.

### Category Theory & Algebra
- Mac Lane, S. (1971). *Categories for the Working Mathematician*. Springer.
- Kock, J. (2004). *Frobenius Algebras and 2D Topological Quantum Field Theories*.
  Cambridge University Press.

### Protein Topology
- Taylor, W.R. (2000). "A deeply knotted protein structure and how it might fold."
  *Nature* 406, 916-919.  [First report of knotted protein]
- Millett, K., Dobay, A., Stasiak, A. (2005). "Linear random knots and their scaling
  behavior." *Macromolecules* 38, 601-606.  [Probabilistic closure]
- Sulkowska, J.I., Rawdon, E.J., Millett, K.C., Onuchic, J.N., Stasiak, A. (2012).
  "Conservation of complex knotting and slipknotting patterns in proteins."
  *PNAS* 109(26), E1715-E1723.
- Reidl, M. & Mashaghi, A. (2015). "Circuit topology of proteins and nucleic acids."
  *Structure* 22, 1227-1237.  [Circuit topology framework]

### Protein Dynamics & Chaos
- Hayward, S. & Go, N. (1995). "Collective variable description of native protein
  dynamics." *Annu. Rev. Phys. Chem.* 46, 223-250.
- Frauenfelder, H., Sligar, S.G., Wolynes, P.G. (1991). "The energy landscapes and
  motions of proteins." *Science* 254, 1598-1603.
- Wales, D.J. (2003). *Energy Landscapes*. Cambridge University Press.

### Maslov Index & Arnold-CS Connection
- Arnold, V.I. (1967). "Characteristic class entering in quantization conditions."
  *Funct. Anal. Appl.* 1(1), 1-13.
- Kirk, P. & Klassen, E. (1993). "Chern-Simons invariants of 3-manifolds decomposed
  along tori." *Math. Ann.* 287, 343-367.
