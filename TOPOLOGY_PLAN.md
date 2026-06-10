# Topological Mapping of Protein Conformational Space
## Theoretical Background & Implementation Plan

> Branch: `feature/topological-mapping`
> Project: ALMA — Atomistic Local Motion Analyzer

---

## Table of Contents

1. [Motivation & Core Thesis](#1-motivation--core-thesis)
2. [Dynamical Chaos & Lyapunov Exponents](#2-dynamical-chaos--lyapunov-exponents)
3. [KAM Theory & Conformational Stability](#3-kam-theory--conformational-stability)
4. [Algebraic Foundations (Lambda-style)](#4-algebraic-foundations-lambda-style)
5. [Category Theory Layer](#5-category-theory-layer)
6. [Knot Theory of Protein Backbones](#6-knot-theory-of-protein-backbones)
7. [Jones Polynomial via Skein Relations](#7-jones-polynomial-via-skein-relations)
8. [Khovanov Homology](#8-khovanov-homology)
9. [Chern-Simons Field Theory](#9-chern-simons-field-theory)
10. [Connecting Chaos to Topology (IDP Hypothesis)](#10-connecting-chaos-to-topology-idp-hypothesis)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Module Architecture](#12-module-architecture)
13. [Open Questions & Research Directions](#13-open-questions--research-directions)

---

## 1. Motivation & Core Thesis

### The Problem with Current IDP Classification

ALMA's current IDP classifier counts metastable basins from a 9,600-step MC trajectory
and applies ad-hoc thresholds. This has two fundamental weaknesses:

1. **Dynamical**: the trajectory is too short; basin counts are noise-dominated.
2. **Topological**: two proteins can have identical basin statistics but fundamentally
   different fold topology — one knotted, one not. A flat energy landscape and a
   topologically trivial backbone are independent facts that both signal IDP-ness.

### The Thesis

> An intrinsically disordered protein (IDP) is characterized by **both**:
> (a) a chaotic conformational dynamics (positive Lyapunov exponent, KAM breakdown), **and**
> (b) a topologically trivial backbone (Jones polynomial = 1, Khovanov homology = ℤ).
>
> Ordered proteins with persistent folds exhibit (a) near-zero or negative Lyapunov
> exponents in the native basin, and (b) non-trivial knot/link invariants when their
> backbone forms crossings under projection.

These two independent invariants — one dynamical, one topological — together form
a more robust classification scheme than either alone.

### The Deep Connection

The bridge between (a) and (b) runs through:

```
Lyapunov exponents (dynamical)
    ↕  (KAM theory: when do invariant tori survive perturbation?)
KAM invariant tori
    ↕  (Arnol'd: tori in phase space ↔ topological invariants of orbits)
Topological invariants of curves in ℝ³
    ↕  (Witten 1989: Jones polynomial = CS partition function)
Chern-Simons TQFT
    ↕  (Khovanov 2000: categorification)
Khovanov homology
```

This is not merely analogy. The Chern-Simons → Jones polynomial connection is exact
(Witten, Commun. Math. Phys. 121, 1989). The KAM → topology connection goes through
the Arnold-Liouville theorem and the topological structure of integrable systems.

---

## 2. Dynamical Chaos & Lyapunov Exponents

### 2.1 Sensitivity to Initial Conditions

A dynamical system `ẋ = f(x)` is chaotic if nearby trajectories diverge exponentially:

```
||δx(t)|| ≈ ||δx(0)|| · e^{λ₁ t}
```

where `λ₁` is the **largest Lyapunov exponent** (LLE). The full **Lyapunov spectrum**
`{λ₁ ≥ λ₂ ≥ ... ≥ λ_n}` describes expansion/contraction in all phase-space directions.

- `λ₁ > 0`: chaotic (exponential divergence)
- `λ₁ = 0`: marginally stable (power-law divergence, quasiperiodic)
- `λ₁ < 0`: stable attractor (trajectories converge)

For a double pendulum at high energy: `λ₁ ≈ 1-3 s⁻¹` (strongly chaotic).
For a double pendulum at low energy: `λ₁ ≈ 0` (integrable, KAM regime).

### 2.2 Benettin Algorithm (Discrete Systems)

For the MC trajectory (a discrete map `x_{n+1} = Φ(x_n)`):

```
Algorithm (Benettin et al. 1980):
  1. Initialize x₀, δx₀ = ε · v̂  (v̂ random unit vector, ε = 1e-6)
  2. For n = 1 to N:
       x_n     = Φ(x_{n-1})         advance reference trajectory
       x_n'    = Φ(x_{n-1} + δx_{n-1})   advance perturbed trajectory
       δx_n    = x_n' - x_n         actual separation
       λ_acc  += ln(||δx_n|| / ε)   accumulate log-stretching
       δx_n    = ε · δx_n / ||δx_n||  renormalize (prevent overflow)
  3. λ₁ = λ_acc / N
```

The renormalization step is critical: without it, `||δx_n||` overflows for chaotic systems.

### 2.3 Lyapunov Spectrum via QR Decomposition (Gram-Schmidt)

For the full spectrum (needed for KAM analysis):

```
Evolve a set of n orthonormal vectors {v₁,...,vₙ} under the tangent map Df.
After each step, apply QR decomposition (Gram-Schmidt orthonormalization).
The diagonal elements of R give the local expansion rates.
λᵢ = lim_{T→∞} (1/T) Σₜ ln Rᵢᵢ(t)
```

For protein MC in 3N-dimensional Cα space (N residues), the tangent map is the
Jacobian of the torsion-angle update function. The first few exponents suffice
for IDP classification.

### 2.4 Kaplan-Yorke Dimension

The fractal dimension of the attractor, estimated from the Lyapunov spectrum:

```
D_KY = k + (λ₁ + λ₂ + ... + λ_k) / |λ_{k+1}|
```

where k is the largest index such that `Σᵢ₌₁ᵏ λᵢ ≥ 0`.

- Ordered protein (fixed fold): `D_KY ≈ 0` (point attractor)
- IDP: `D_KY >> 0` (high-dimensional strange attractor)

This gives a **single scalar measure of conformational chaos** that can be displayed
in the ALMA sidebar.

### 2.5 Application to MC Trajectory

The ALMA MC trajectory is a Markov chain, not a Hamiltonian flow. The Lyapunov
analysis still applies — we are measuring sensitivity of the chain to perturbations
in the initial particle coordinates. Two chains started from slightly different
conformations: do they stay together (ordered protein, funnel landscape) or diverge
(IDP, flat landscape)?

**Key insight**: for an ordered protein with a deep funnel, both chains relax to the
same native basin → `λ₁ ≤ 0`. For an IDP with no funnel, chains wander randomly
in conformational space → `λ₁ > 0`.

---

## 3. KAM Theory & Conformational Stability

### 3.1 The Arnold-Liouville Theorem

An integrable Hamiltonian system with n degrees of freedom has n independent
conserved quantities `{I₁,...,Iₙ}` (action variables). The phase space is foliated
by **invariant tori** `𝕋ⁿ` parametrized by the actions. Motion on each torus is
quasi-periodic with frequencies:

```
ωᵢ = ∂H/∂Iᵢ
```

The key object is the **frequency ratio** (winding number) `ω₁/ω₂`. 
- Rational ratio → periodic orbit (resonance)
- Irrational ratio → dense quasi-periodic orbit (fills the torus)

### 3.2 The KAM Theorem

**Statement** (Kolmogorov 1954, Arnold 1963, Moser 1962):

For a nearly-integrable Hamiltonian `H = H₀(I) + ε H₁(I,θ)`, most invariant
tori (those with sufficiently irrational frequency ratios) **survive** for small
perturbation ε. "Sufficiently irrational" means the **Diophantine condition**:

```
|ω · k| ≥ γ / |k|^τ    ∀k ∈ ℤⁿ \ {0}
```

for some γ > 0, τ > n-1. Tori violating this (resonant tori) are destroyed and
replaced by chaotic layers (Chirikov resonance overlap).

### 3.3 KAM Breakdown → Chaos Transition

As ε increases, tori are destroyed in order of their "rationality":
1. Highly resonant tori (ω₁/ω₂ = p/q, small p,q) destroyed first
2. Noble numbers (related to golden ratio) are most robust — last KAM tori to survive
3. Beyond the critical perturbation ε_c (Chirikov overlap criterion): **global chaos**

The **Chirikov overlap criterion**:
```
K = ε · (∂²H₀/∂I²)⁻¹ · (∂²H₁/∂θ²) ≈ 1   →   onset of global chaos
```

### 3.4 KAM Analogy for Proteins

Map the protein conformational space to a Hamiltonian system:
- **Action variables I**: amplitude of each normal mode (quasi-harmonic basin)
- **Angle variables θ**: phase of oscillation within a basin
- **ε**: thermal noise amplitude = kT / barrier height

```
Ordered protein:   kT << ΔE_barrier  →  ε << ε_c  →  KAM tori intact
                   protein stays near native basin, Lyapunov ≈ 0

IDP:               kT ≈ ΔE_barrier   →  ε ≈ ε_c   →  KAM breakdown
                   all tori destroyed, global conformational chaos
```

**KAM stability score** for a protein:
```
K_protein = kT / min_basin_barrier_height
K < 1: ordered (KAM regime)
K ≈ 1: partially disordered (KAM breakdown threshold)  
K > 1: IDP (global chaos regime)
```

This can be estimated from the landscape MC trajectory by measuring the smallest
energy barrier between the observed conformational basins.

### 3.5 Winding Numbers in Conformational Space

From the landscape PCA projection (PC1, PC2), define an angular coordinate around
each basin center. The **winding number** of a trajectory segment around a basin:

```
w = (1/2π) ∮ dθ
```

- Integer w: trajectory loops around basin (ordered, periodic orbit)
- Non-integer/irrational w: quasi-periodic KAM orbit
- Undefined (infinite variation): chaotic

This can be computed directly from the ALMA landscape trajectory.

---

## 4. Algebraic Foundations (Lambda-style)

### 4.1 Design Philosophy

The mathematical structures here (groups, rings, modules, chain complexes) are
most naturally expressed as **algebraic types with pure operations** — the
functional programming / lambda calculus style. In Python:

- **Types** as `dataclass` or `Protocol`
- **Operations** as pure functions (no mutation)
- **Composition** via higher-order functions
- **Exactness** via symbolic arithmetic (`sympy.Integer`, `Fraction`)

This mirrors the mathematical definition directly:

```python
# Category theory: a morphism is just a function between objects
Morphism = Callable[[A], B]

# Functor: structure-preserving map between categories
Functor = lambda obj_map, mor_map: (obj_map, mor_map)

# Natural transformation: a family of morphisms
NatTrans = lambda component: component   # indexed family
```

### 4.2 Group

A **group** (G, ·) satisfies:
- Closure: ∀a,b ∈ G: a·b ∈ G
- Associativity: (a·b)·c = a·(b·c)
- Identity: ∃e: e·a = a·e = a
- Inverses: ∀a ∃a⁻¹: a·a⁻¹ = e

```python
@dataclass(frozen=True)
class Group(Generic[G]):
    elements:   FrozenSet[G]
    mul:        Callable[[G, G], G]      # binary operation
    identity:   G
    inv:        Callable[[G], G]         # inverse

# The symmetric group S_n (permutations) — relevant to Khovanov
S3 = Group(
    elements = frozenset(permutations(range(3))),
    mul      = lambda p, q: tuple(p[q[i]] for i in range(3)),
    identity = (0, 1, 2),
    inv      = lambda p: tuple(sorted(range(3), key=lambda i: p[i]))
)
```

Key groups in this project:
- **ℤ**: integers under addition (coefficients in chain complexes)
- **ℤ[t, t⁻¹]**: Laurent polynomials (Jones polynomial ring)
- **ℤ/2ℤ**: integers mod 2 (Khovanov over 𝔽₂, simpler computations)
- **Braid group Bₙ**: generators σᵢ, relations σᵢσⱼ = σⱼσᵢ (|i-j|≥2), σᵢσᵢ₊₁σᵢ = σᵢ₊₁σᵢσᵢ₊₁

### 4.3 Ring & Module

A **ring** (R, +, ·): abelian group under +, monoid under ·, distributive.
A **module** M over ring R: abelian group with scalar multiplication R × M → M.

```python
@dataclass(frozen=True)
class Module(Generic[R, M]):
    group:      Group[M]                       # underlying abelian group
    scalar_mul: Callable[[R, M], M]            # R-action
    ring:       Ring[R]

# Free module ℤ^n — used for chain groups in Khovanov
def free_module(ring: Ring[R], basis: FrozenSet[str]) -> Module:
    # Elements are formal ℤ-linear combinations of basis vectors
    # Represented as dict[basis_element, coefficient]
    ...
```

### 4.4 Chain Complex

A **chain complex** is a sequence of modules with boundary maps:

```
... → C_{n+1} →^{∂_{n+1}} C_n →^{∂_n} C_{n-1} → ...
```

satisfying the fundamental condition: `∂ ∘ ∂ = 0` (boundary of boundary is zero).

The **homology** groups measure "how much" the complex fails to be exact:

```
H_n = ker(∂_n) / im(∂_{n+1})
```

```python
@dataclass(frozen=True)
class ChainComplex(Generic[R]):
    modules:   dict[int, Module]          # C_n for each degree n
    boundary:  dict[int, LinearMap]       # ∂_n : C_n → C_{n-1}

    def verify(self) -> bool:
        # Check ∂ ∘ ∂ = 0 for all degrees
        return all(
            compose(self.boundary[n], self.boundary[n+1]) == zero_map
            for n in self.modules
        )

    def homology(self, n: int) -> Module:
        ker = kernel(self.boundary[n])
        im  = image(self.boundary[n+1])
        return quotient(ker, im)          # H_n = ker/im
```

### 4.5 Graded Structures (for Khovanov)

Khovanov homology is **bigraded**: chain groups are indexed by two integers (i, j).

```python
@dataclass(frozen=True)
class BidegreeModule(Generic[R]):
    components: dict[tuple[int,int], Module]  # C^{i,j}
    
# Graded boundary map: only connects (i,j) → (i+1, j)
# The j-grading is preserved by ∂ (quantum grading conservation)
```

---

## 5. Category Theory Layer

### 5.1 Categories

A **category** C consists of:
- **Objects**: Ob(C)
- **Morphisms**: Hom(A,B) for each pair of objects A, B
- **Composition**: `∘ : Hom(B,C) × Hom(A,B) → Hom(A,C)`
- **Identity**: `id_A ∈ Hom(A,A)` for each object

satisfying associativity and identity laws.

```python
@dataclass(frozen=True)
class Category(Generic[Obj, Mor]):
    objects:   FrozenSet[Obj]
    hom:       Callable[[Obj, Obj], FrozenSet[Mor]]
    compose:   Callable[[Mor, Mor], Mor]          # g ∘ f
    identity:  Callable[[Obj], Mor]
```

Key categories in this project:
- **Vect_ℤ**: vector spaces (actually free ℤ-modules) and linear maps
- **Kob**: the cobordism category (objects = planar matchings, morphisms = cobordisms)
  - This is the "target" category for the Khovanov functor
- **Chain(R)**: chain complexes over R and chain maps
- **Knots**: knot diagrams and Reidemeister moves (as morphisms)

### 5.2 Functors

A **functor** F: C → D maps:
- Objects: F(A) ∈ Ob(D) for each A ∈ Ob(C)
- Morphisms: F(f): F(A) → F(B) for each f: A → B
- Preserving: F(g ∘ f) = F(g) ∘ F(f), F(id_A) = id_{F(A)}

```python
@dataclass(frozen=True)
class Functor(Generic[C, D]):
    source:   Category[C]
    target:   Category[D]
    obj_map:  Callable[[C.Obj], D.Obj]
    mor_map:  Callable[[C.Mor], D.Mor]
```

The **Khovanov functor** is:
```
F_Kh : Knots → Chain(ℤ-Mod)
```
It sends a knot diagram K to a chain complex CKh(K) such that the homology
H(CKh(K)) is a knot invariant (independent of choice of diagram for the knot).

### 5.3 Natural Transformations

A **natural transformation** η: F ⟹ G between functors F,G: C → D assigns
to each object A ∈ C a morphism ηA: F(A) → G(A), such that for every f: A→B:

```
G(f) ∘ ηA = ηB ∘ F(f)    (naturality square commutes)
```

Relevant here: the **categorification map** from Jones polynomial to Khovanov
is a natural transformation between the "decategorified" Jones functor and the
Euler characteristic of the Khovanov chain complex.

---

## 6. Knot Theory of Protein Backbones

### 6.1 Proteins as Curves in ℝ³

The Cα backbone trace is an **open curve** in ℝ³ (a path, not a closed loop).
To apply classical knot theory (which requires closed curves), we have two options:

1. **Closure**: connect the N-terminus to the C-terminus by a long arc through
   "infinity" (off to one side of the protein). The result is a closed curve.
   The knot type is well-defined if the closure arc is chosen consistently.

2. **Open knotting** (Millett-Rawdon): use a probabilistic closure — average over
   all closure directions. A residue i is "in the knotted region" if most closures
   produce a non-trivial knot.

We will implement both, using method 1 for Jones/Khovanov (needs exact closure)
and method 2 for the knotting probability profile.

### 6.2 Knot Diagrams and Gauss Codes

A **knot diagram** is a projection of the curve onto a plane (ℝ²) with crossing
information (over/under) recorded.

**Algorithm: backbone → Gauss code**
```
1. Choose projection direction n̂ (try several, pick fewest crossings)
2. Project all Cα + closure arc onto plane perpendicular to n̂
3. Find all crossing points (pairs of segments that intersect in projection)
4. For each crossing: record which segment is over (higher z-value along n̂)
5. Walk along the curve: record signed crossings in encounter order
   → Gauss code: sequence of signed crossing labels
```

**Signed crossing convention** (right-hand rule):
- Positive crossing (+): right strand passes over left strand
- Negative crossing (−): left strand passes over right strand

**Writhe** = sum of signs of all crossings:
```
w(K) = Σᵢ εᵢ    where εᵢ ∈ {+1, -1}
```

The writhe is **not** a topological invariant (it depends on the diagram choice)
but it measures the "online twist" of the backbone under projection.
For the online writhe scale, we compute w(K) at each snapshot of the MC trajectory.

### 6.3 Reidemeister Moves

Two knot diagrams represent the same knot iff they are connected by a sequence
of three local moves:
- **R1**: add/remove a single crossing (loop twist)
- **R2**: add/remove two crossings (strand passing over/under)  
- **R3**: slide a strand over a crossing (triangle move)

Any knot invariant must be invariant under all three moves.

### 6.4 Knot Invariants Relevant to Proteins

| Invariant | Complexity | Information |
|---|---|---|
| Crossing number | O(2^n) | minimal n crossings |
| Writhe | O(n²) | diagram-dependent, not invariant |
| Alexander polynomial Δ(t) | O(n³) | first homological invariant |
| Jones polynomial V(t) | O(exp(n)) via skein | stronger than Alexander |
| HOMFLY-PT P(v,z) | O(exp(n)) | generalizes Jones+Alexander |
| Khovanov homology Kh^{i,j} | O(exp(n)) | categorifies Jones |

Known knotted proteins (as of 2024): trefoil knot (3₁) in MJ0366, figure-eight
(4₁) in UCH-L1, 5₂ knot in ubiquitin C-terminal hydrolase. These are rare but
real. Most proteins are unknotted (Jones = 1).

---

## 7. Jones Polynomial via Skein Relations

### 7.1 Definition via Skein Relation

The **Jones polynomial** V_K(t) ∈ ℤ[t^{1/2}, t^{-1/2}] is the unique invariant of
oriented knots/links satisfying:

```
(S0) V(unknot) = 1
(S1) t⁻¹ V(L₊) − t V(L₋) + (t^{-1/2} − t^{1/2}) V(L₀) = 0
```

where L₊, L₋, L₀ are three diagrams identical outside a small disk where they
differ by a positive crossing, negative crossing, or smoothing respectively.

This skein relation gives a recursive algorithm:
```
V(K) via S1: reduce any crossing to two simpler diagrams
             recurse until all components are unknots
             V(unknot ∪ unknot ∪ ...) = (-t^{1/2} - t^{-1/2})^{n-1}
```

### 7.2 Bracket Polynomial (Kauffman)

More convenient for computation: the **Kauffman bracket** ⟨K⟩ ∈ ℤ[A, A⁻¹]:

```
(B0) ⟨unknot⟩ = 1
(B1) ⟨L with loop⟩ = δ ⟨L⟩   where δ = -A² - A⁻²
(B2) ⟨L_cross⟩ = A ⟨L_0⟩ + A⁻¹ ⟨L_∞⟩
```

where L_0, L_∞ are the two smoothings of a crossing (0-smoothing and ∞-smoothing).

The bracket is invariant under R2, R3 but not R1 (it picks up a factor of
`-A^{±3}` under R1). To make it R1-invariant:

```
X(K) = (-A)^{-3w(K)} ⟨K⟩    (writhe-normalized bracket)
```

The Jones polynomial is then `V_K(t) = X_K(t^{-1/4})` (with `A = t^{-1/4}`).

### 7.3 Computational Implementation

The bracket expansion produces a binary tree of depth n (number of crossings).
Each leaf is a collection of disjoint circles → contributes `δ^{circles-1}`.
Total: 2^n terms. For proteins: typical backbone knots have 0-5 crossings, so
this is entirely feasible (2^5 = 32 terms).

```python
def bracket(gauss_code: GaussCode, ring: LaurentRing) -> LaurentPoly:
    if len(gauss_code) == 0:
        return ring.one  # unknot
    crossing = gauss_code[0]
    # Two smoothings: A-smoothing and A⁻¹-smoothing
    L0 = smooth(gauss_code, crossing, '0')
    L_inf = smooth(gauss_code, crossing, 'inf')
    return ring.A * bracket(L0, ring) + ring.A_inv * bracket(L_inf, ring)
```

### 7.4 Online Writhe Scale

For the ALMA trajectory, compute the **writhe time series** `w(t)` across
MC snapshots. This gives a dynamical signal:

```
w(t) = Σ_{crossings of snapshot t} ε_i(t)
```

Large fluctuations in `w(t)` → backbone topology is dynamically unstable → IDP.
Small fluctuations → persistent fold topology → ordered protein.

The **writhe autocorrelation**:
```
C_w(τ) = ⟨w(t)w(t+τ)⟩ - ⟨w⟩²
```
decays exponentially for ordered proteins (correlation time τ_w) and as a power
law for IDPs (no characteristic time scale, scale-free fluctuations).

---

## 8. Khovanov Homology

### 8.1 Categorification Philosophy

The Jones polynomial is a number (polynomial) associated to a knot.
Khovanov homology is a **chain complex** (bigraded) whose **Euler characteristic**
recovers the Jones polynomial:

```
V_K(q) = Σ_{i,j} (-1)^i q^j · dim(Kh^{i,j}(K))
```

where `q = -t^{1/2} - t^{-1/2}` (the variable substitution).

This is **categorification**: replacing a number by a vector space (or chain complex)
whose Euler characteristic gives back the number. The chain complex contains
strictly more information than the polynomial.

**Why stronger?** The Jones polynomial of the trefoil equals that of no other
simple knot, but Khovanov homology can distinguish knots with the same Jones
polynomial (e.g., the Kinoshita-Terasaka and Conway knots — same Jones, different Kh).

### 8.2 The Cube of Resolutions

Given a knot diagram K with n crossings, label them 1,...,n.

**Resolution**: for each crossing, choose either a **0-smoothing** or **1-smoothing**:
```
0-smoothing: connect strands horizontally (oriented smoothing)
1-smoothing: connect strands vertically (twisted smoothing)
```

Each **vertex** of the n-dimensional cube {0,1}^n is a complete smoothing:
a disjoint union of circles in the plane.

**Chain groups**:
```
CKh^i(K) = ⊕_{v ∈ {0,1}^n, |v|=i} V^{⊗ circles(v)}
```
where `|v| = Σ vₖ` (number of 1-smoothings), `circles(v)` = number of circles in
that smoothing, and `V = ℤ{v₊, v₋}` is a 2-dimensional free ℤ-module (the
"TQFT vector space" associated to a circle, quantum grading: v₊ in degree +1,
v₋ in degree −1).

### 8.3 Boundary Maps

The boundary map `∂: CKh^i → CKh^{i+1}` counts edge maps in the cube (edges
where exactly one bit flips from 0 to 1). Each edge changes the number of circles:

- **Merge** (2 circles → 1): uses the **multiplication** map `m: V⊗V → V`
  ```
  m(v₊ ⊗ v₊) = v₊
  m(v₊ ⊗ v₋) = v₋
  m(v₋ ⊗ v₊) = v₋
  m(v₋ ⊗ v₋) = 0
  ```
- **Split** (1 circle → 2): uses the **comultiplication** map `Δ: V → V⊗V`
  ```
  Δ(v₊) = v₊ ⊗ v₋ + v₋ ⊗ v₊
  Δ(v₋) = v₋ ⊗ v₋
  ```

This (V, m, Δ) is a **Frobenius algebra** — the algebraic core of Khovanov homology.

**Sign convention**: each edge (u,v) in the cube gets a sign `(-1)^{s(u,v)}`
where `s(u,v) = Σ_{k < position of flipped bit} u_k`. This ensures `∂² = 0`.

### 8.4 Quantum Grading

The **quantum (j) grading** is preserved by the boundary map and shifts under
the TQFT operations:
```
j-grade(v₊ in V(circle c)) = +1
j-grade(v₋ in V(circle c)) = -1
```
The overall j-grading of a basis element in the tensor product is the sum of
component gradings, plus the **degree shift** from the diagram:
```
j(v) = homological_shift + Σ (grades of v-components)
```

### 8.5 Khovanov Polynomial

From the bigraded homology groups:
```
Kh(K; q, t) = Σ_{i,j} q^j t^i · rank(Kh^{i,j}(K))
```

Setting `t = -1` recovers the Jones polynomial (decategorification):
```
V_K(q) = Kh(K; q, -1)    (up to normalization)
```

### 8.6 Invariance Proof Sketch

Khovanov homology is invariant under Reidemeister moves because:
- R1: the chain complex changes by a chain homotopy equivalence
- R2: the complex changes by a direct sum with an acyclic complex
- R3: more subtle, requires the "delooping" lemma

The invariance is proved at the level of **chain complexes** (not just homology),
making it a stronger statement than Jones polynomial invariance.

### 8.7 Protein-Specific Interpretation

| Khovanov group | Protein meaning |
|---|---|
| Kh^{0,j}(K) = ℤ, all j | trivially unknotted backbone (IDP indicator) |
| Kh^{i,j} ≠ 0, i ≠ 0 | non-trivial knotting of backbone fold |
| Large total rank Σ rank(Kh^{i,j}) | complex fold topology |
| rank(Kh^{0,j}) − rank(Kh^{1,j}) + ... | recovers Jones polynomial |

---

## 9. Chern-Simons Field Theory

### 9.1 The Chern-Simons Action

On a 3-manifold M (for us: ℝ³, the protein's ambient space), with gauge group
G = SU(2), the **Chern-Simons action** is:

```
S_CS[A] = (k/4π) ∫_M Tr(A ∧ dA + (2/3) A ∧ A ∧ A)
```

where `A` is a connection 1-form on a principal G-bundle over M, and
`k ∈ ℤ` is the **level** (quantization condition).

Key properties:
- S_CS is **topological**: it does not depend on a metric on M
- It is invariant under gauge transformations of A (up to an integer multiple of 2πk)
- The equations of motion `F = dA + A∧A = 0` say the curvature vanishes (flat connection)

### 9.2 Wilson Loops and Knot Invariants

The **Wilson loop** of a representation R along a curve K ⊂ M:
```
W_R(K) = Tr_R [ P exp(∮_K A) ]
```
where P denotes path-ordering.

**Witten's theorem** (1989): the partition function of SU(2) Chern-Simons theory
with a Wilson loop in the spin-j representation along K computes:
```
Z(M, K; j) = ⟨ W_j(K) ⟩_CS = (Jones polynomial of K evaluated at q = e^{2πi/(k+2)})
```

Specifically, for the spin-1/2 representation and `q = e^{2πi/(k+2)}`:
```
⟨ W_{1/2}(K) ⟩ = V_K(q)
```

This is the **exact** connection between topology and field theory. The Jones
polynomial is a quantum observable in a 3D topological quantum field theory.

### 9.3 The Chern-Simons Path Integral as Topological Invariant

The CS path integral is **metric-independent**: it depends only on the topology
of M and the isotopy class of K. This is why it gives a knot invariant.

The **stationary phase approximation** (large k → classical limit):
```
Z ≈ Σ_{flat connections A*} exp(ik S_CS[A*]) · (det correction)
```
For ℝ³, the only flat connection is A* = 0, so the leading term gives the
Alexander polynomial. Higher-order terms give the full Jones polynomial.

### 9.4 Relation to KAM Theory

This is the deepest connection in this project. The bridge:

1. **KAM invariant tori** are Lagrangian submanifolds of phase space
2. **Chern-Simons theory** is defined on the space of connections on a 3-manifold
3. The **Arnold-Maslov index** of a Lagrangian submanifold controls quantum corrections
   to semiclassical WKB approximations — the same mathematics that underlies both
   KAM theory and the perturbative expansion of the CS path integral

More concretely: the Chern-Simons invariant `s(M) = S_CS[A*] / 4π²` measures the
topological complexity of a 3-manifold in a way that parallels how KAM theory
measures the "irrationality" (topological complexity) of invariant tori.

For our purposes: **the Chern-Simons-derived Jones polynomial is the topological
analog of the Lyapunov exponent** — one measures dynamical complexity, the other
topological complexity. Together they classify protein conformational behavior.

### 9.5 Practical Computation

We will not numerically integrate the CS path integral (lattice gauge theory).
Instead, we exploit the exact result: **the Jones polynomial IS the CS expectation
value**, so computing V_K(t) via skein relations IS computing the CS quantity.
The CS field theory provides the theoretical foundation; the skein algorithm
provides the practical computation.

---

## 10. Connecting Chaos to Topology (IDP Hypothesis)

### 10.1 The Full Picture

```
                    PROTEIN
                       │
          ┌────────────┴────────────┐
     DYNAMICS                  TOPOLOGY
          │                         │
   Lyapunov λ₁              Jones V_K(t)
   KAM score K           Khovanov Kh^{i,j}
          │                         │
   λ₁ > 0, K > 1 ──→ IDP ←── V_K = 1, Kh = ℤ
   λ₁ ≤ 0, K < 1 ──→ ORDERED ←── V_K ≠ 1, Kh complex
```

### 10.2 Formal IDP Classification Criteria

A protein is classified as:

**ORDERED** if ALL of:
- λ₁ < 0 (stable attractor in MC dynamics)
- K_protein < 0.8 (KAM tori intact, barriers >> kT)
- Writhe variance σ²_w < threshold (stable fold topology)

**POSSIBLY DISORDERED** if ANY of:
- 0 ≤ λ₁ < 0.1 (marginally stable)
- 0.8 ≤ K_protein < 1.2 (near KAM threshold)
- Moderate writhe variance

**IDP** if ANY of:
- λ₁ > 0.1 (positively chaotic)
- K_protein > 1.2 (past KAM breakdown)
- High writhe variance AND V_K(t) = 1 (dynamically and topologically trivial)

### 10.3 The Knotted IDP Question

Can an IDP have a knotted backbone? In principle yes — a disordered protein could
transiently form knots. But knotted proteins tend to have deep topological
constraints (the knot must be threaded during folding), which correlates with
ordered, slow-folding structures. The joint measurement:

```
(λ₁, V_K(t)) classification:
  (λ₁ < 0, V_K ≠ 1):  ordered knotted protein (most topologically complex)
  (λ₁ < 0, V_K = 1):  ordered unknotted (typical globular protein)
  (λ₁ > 0, V_K = 1):  IDP (dynamically and topologically trivial)
  (λ₁ > 0, V_K ≠ 1):  rare/transient knotted disorder (research question)
```

### 10.4 Writhe as a Dynamical Topological Bridge

The **online writhe** `w(t)` is computed at every MC snapshot and lives in
the intersection of dynamical and topological analysis:
- It is a topological quantity (counts crossings) computed from dynamics (trajectory)
- Its time series is a 1D signal amenable to Lyapunov and spectral analysis
- Large `λ_writhe` (Lyapunov of the writhe signal) directly measures topological chaos

This makes writhe the natural **bridge observable** between the two pillars.

---

## 11. Implementation Roadmap

### Phase 0 — Setup (Branch already created)
```
✓ branch: feature/topological-mapping
  python/topology/__init__.py
  tests/test_topology.py (stub)
```

### Phase 1 — Dynamical Analysis
```
P1.1  python/topology/lyapunov.py
        - BenettinLyapunov class
        - Largest Lyapunov exponent from MC chain pairs
        - Full spectrum via QR/Gram-Schmidt
        - Kaplan-Yorke dimension
        - Online writhe time series λ_writhe

P1.2  python/topology/kam.py
        - Basin curvature estimator (Hessian from landscape energies)
        - KAM stability score K_protein
        - Winding numbers from PCA trajectory (basin angular coordinate)
        - KAM breakdown threshold predictor

P1.3  Integration into gui_main.py
        - Lyapunov exponent widget in sidebar metrics
        - KAM score in landscape panel
        - Replace heuristic IDP classifier with λ₁ + K decision tree
```

### Phase 2 — Algebraic Infrastructure
```
P2.1  python/topology/algebra.py
        - Group[G]: frozen dataclass + operations
        - Ring[R]: Group + multiplication
        - Module[R,M]: R-module over a ring
        - ChainComplex[R]: graded modules + boundary maps
        - verify_boundary(): checks ∂² = 0
        - compute_homology(): Smith normal form over ℤ (via sympy)
        - LaurentPolynomial: ℤ[t, t⁻¹] arithmetic

P2.2  python/topology/category.py
        - Category, Functor, NaturalTransformation
        - Vect_Z: category of free ℤ-modules
        - Kob: cobordism category (planar matchings + cobordisms)
        - ChainCat: chain complexes and chain maps
```

### Phase 3 — Knot Theory
```
P3.1  python/topology/knot.py
        - Backbone → closed curve (two closure methods)
        - Projection → knot diagram (minimal crossings heuristic)
        - Gauss code computation
        - Writhe calculation
        - Online writhe series over trajectory
        - Knot type identification (crossing number ≤ 8 via table lookup)
        - Knotting probability profile (Millett-Rawdon probabilistic closure)

P3.2  python/topology/jones.py
        - Kauffman bracket ⟨K⟩ via skein recursion
        - Writhe normalization → Jones polynomial V_K(t)
        - LaurentPolynomial arithmetic (exact, over ℤ)
        - HOMFLY-PT polynomial P(v,z) (optional extension)
        - Caching for recursive subproblems (memoization)
```

### Phase 4 — Khovanov Homology
```
P4.1  python/topology/khovanov.py
        - Cube of resolutions: all 2^n smoothings
        - TQFT module V = ℤ{v₊, v₋} with grading
        - Tensor products of V for each vertex
        - Frobenius algebra maps: m (merge), Δ (split)
        - Boundary map ∂ construction (sign convention)
        - Verify ∂² = 0
        - Compute bigraded homology Kh^{i,j}(K)
        - Khovanov polynomial
        - Euler characteristic check: recovers Jones polynomial

P4.2  Visualization
        - Khovanov table display in disorder panel
        - Comparison of Kh across MC snapshots
        - Khovanov "distance" between conformations
```

### Phase 5 — Chern-Simons (Research Phase)
```
P5.1  python/topology/chern_simons.py
        - CS level k and SU(2) gauge group parameters
        - Verify CS → Jones correspondence for test knots
        - CS invariant as alternative Jones computation (via q = e^{2πi/(k+2)})
        - Connection to KAM: Maslov index of backbone path

P5.2  Field theory observables
        - Wilson loop expectation value (exact via Jones)
        - CS invariant time series over MC trajectory
        - Topological charge density along backbone
```

---

## 12. Module Architecture

```
python/
└── topology/
    ├── __init__.py             # public API: LyapunovResult, KAMScore, KnotInvariant
    ├── lyapunov.py             # BenettinLyapunov, LyapunovSpectrum, KaplanYorkeDim
    ├── kam.py                  # KAMStabilityScore, WindingNumbers, BasinCurvature  
    ├── algebra.py              # Group, Ring, Module, ChainComplex, LaurentPoly
    ├── category.py             # Category, Functor, NaturalTransformation, Vect_Z
    ├── knot.py                 # BackboneClosure, KnotDiagram, GaussCode, Writhe
    ├── jones.py                # KauffmanBracket, JonesPolynomial
    ├── khovanov.py             # CubeOfResolutions, KhovanovComplex, KhovanovHomology
    └── chern_simons.py         # CSAction, WilsonLoop, CSInvariant

tests/
└── test_topology.py
    ├── test_lyapunov_*
    ├── test_kam_*
    ├── test_algebra_chain_complex
    ├── test_jones_trefoil          # V_{3₁}(t) = -t⁻⁴ + t⁻³ + t⁻¹ (known)
    ├── test_jones_unknot           # V(t) = 1
    ├── test_khovanov_unknot        # Kh^{0,1} = Kh^{0,-1} = ℤ
    ├── test_khovanov_trefoil       # known Kh table
    └── test_reidemeister_invariance
```

### Data Flow Integration

```
gui_main.py
    │
    ├── LandscapeWorker.result
    │       │
    │       ├──→ topology.lyapunov.BenettinLyapunov(snapshots, ca_indices)
    │       │         └──→ LyapunovResult{λ₁, spectrum, D_KY}
    │       │
    │       ├──→ topology.kam.KAMStabilityScore(energies, basins)
    │       │         └──→ KAMResult{K_protein, winding_numbers}
    │       │
    │       └──→ topology.knot.OnlineWrithe(snapshots, ca_indices)
    │                 └──→ np.ndarray (writhe time series)
    │
    └── PipelineWorker.finished
            │
            └──→ topology.knot.BackboneClosure(best_ensemble_ca)
                      └──→ GaussCode
                                ├──→ topology.jones.JonesPolynomial(gauss_code)
                                └──→ topology.khovanov.KhovanovHomology(gauss_code)
```

---

## 13. Open Questions & Research Directions

### 13.1 Immediate Research Questions
1. Do IDP benchmark proteins (α-synuclein P37840, tau P10636 in ALMA's data set)
   show consistently positive Lyapunov exponents in the MC trajectory?
2. Does the KAM score K_protein correlate with experimentally measured disorder
   scores (PONDR, IUPred2A)?
3. Are the backbone Jones polynomials of the structured proteins in ALMA's data
   set non-trivial, while IDPs give V_K = 1?

### 13.2 Theoretical Open Questions
1. **KAM-Khovanov bridge**: is there a precise statement relating the KAM
   stability index to the rank of Khovanov homology? (Speculative but potentially
   a new result if true.)
2. **Chern-Simons time series**: the CS invariant computed at each MC snapshot
   gives a topological time series. What is its power spectrum? Does it show
   1/f noise for IDPs (power-law, scale-free) vs. Lorentzian for ordered proteins?
3. **Categorification of Lyapunov**: is there a "Khovanov-like" categorification
   of the Lyapunov exponent — a chain complex whose Euler characteristic gives λ₁?

### 13.3 Computational Challenges
1. **Khovanov for large knots**: the 2^n scaling limits us to ≤ 20 crossings
   practically. Most protein backbone knots have ≤ 5 crossings, so this is fine.
   For larger proteins, use Bar-Natan's fast Khovanov algorithm (canopoly).
2. **Smith normal form over ℤ**: computing homology of chain complexes over ℤ
   (not 𝔽₂) requires SNF, which can be slow for large matrices. Use `sympy`
   for small cases, `sage` or custom SNF for larger ones.
3. **Projection choice for Gauss code**: the knot type is projection-independent,
   but the number of crossings in the diagram is not. We should minimize crossings
   (= choose "best" projection direction) for efficiency.

### 13.4 Validation Strategy
1. **Known knotted proteins**: run the pipeline on MJ0366 (trefoil, 3₁) and
   UCH-L1 (figure-eight, 4₁) and verify the correct Jones polynomial is computed.
2. **Known unknotted proteins**: all others should give V_K = 1.
3. **IDP benchmarks**: α-synuclein, tau should give positive Lyapunov exponents.
4. **Double-check via decategorification**: always verify
   `Σ (-1)^i rank(Kh^{i,j}) = coefficient of q^j in V_K(q)`.

---

## References

### Dynamical Systems
- Benettin, G. et al. (1980). "Lyapunov characteristic exponents for smooth
  dynamical systems." *Meccanica* 15, 9-20.
- Chirikov, B.V. (1979). "A universal instability of many-dimensional oscillator
  systems." *Physics Reports* 52, 263-379.
- Arnold, V.I. (1963). "Proof of A.N. Kolmogorov's theorem on the preservation
  of quasi-periodic motions." *Russian Math. Surveys* 18, 9-36.

### Knot Theory
- Kauffman, L.H. (1987). "State models and the Jones polynomial."
  *Topology* 26, 395-407.
- Jones, V.F.R. (1985). "A polynomial invariant for knots via von Neumann
  algebras." *Bull. AMS* 12, 103-111.

### Khovanov Homology
- Khovanov, M. (2000). "A categorification of the Jones polynomial."
  *Duke Math. J.* 101, 359-426.
- Bar-Natan, D. (2002). "On Khovanov's categorification of the Jones polynomial."
  *Algebraic & Geometric Topology* 2, 337-370.

### Chern-Simons Theory
- Witten, E. (1989). "Quantum field theory and the Jones polynomial."
  *Commun. Math. Phys.* 121, 351-399.

### Protein Topology
- Millett, K. et al. (2013). "Identifying knots in proteins."
  *Biochem. Soc. Trans.* 41, 533-537.
- Sulkowska, J.I. et al. (2012). "Knotted proteins: a tangled tale of
  structural biology." *Biochem. Soc. Trans.* 41, 523-527.

### Protein Dynamics & Chaos
- Hayward, S. & Go, N. (1995). "Collective variable description of native
  protein dynamics." *Annu. Rev. Phys. Chem.* 46, 223-250.
