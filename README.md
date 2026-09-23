# FRI — Facet Region Interpreter

A graph-theoretic **region finder** for multiscale Feynman integrals. Given a graph (topology + external momenta) in the context of **massless scattering**, which we classify into the **wide-angle** and **collinear** types based on their kinematics, FRI constructs **all facet regions** of the asymptotic expansion directly from the graph — the majority, and mostly the entire list of regions. The construction is based on understandings of the all-order region structures in momentum space, without any Feynman-polynomial computation or polytope geometry.

For the wide-angle kinematics, the region-structure understanding is from the paper *"All-order prescription for facet regions in massless wide-angle scattering"* ([arXiv:2601.22144](https://arxiv.org/abs/2601.22144)). FRI is supposed to enumerate the whole list of facet regions for any Feynman graph under the following conditions.
1. The external momenta are within the following three types:
    (1) the "on-shell momenta" $p_i$, each close to a lightcone ($p_i^2$ small or 0);
    (2) the "off-shell momenta" $q_j$, each with $q_j^2 \sim 1$ (not small);
    (3) the "soft momenta" $l_k$, each has all components being small (but can approach zero at distinct speed).
2. No two external momenta are close to the same lightcone.
3. There are no massive propagators in the graph.

For the collinear kinematics, the region-structure understanding is from the author's knowledge which has not yet been published. At this moment, only two kinematics are available in FRI:
1. Five-point scattering with two particles collinear to each other (which we denote by partons 2 and 3, either timelike- or spacelike-collinear).
2. Four-point scattering in the Regge limit ($1+2\to 3+4$, with 1 and 3 spacelike-collinear while 2 and 4 spacelike-collinear).
It is worth noting that in the second case above, Glauber-mode propagators can emerge.

Pure Python (>= 3.9), standard library only. pySecDec is used *only* for off-line cross-validation, which is not part of this repository. The optional region visualisation renders through the Wolfram Engine — see *Quick start* below.

---

## Main function

The center function of FRI is to identify the entire list of (facet) regions and present them in momentum space, where the **mode** of each edge is manifested.

### Notations for the momentum modes

In the context of wide-angle kinematics, the possible modes are:
- $H$ — hard (all the components $\sim 1$),
- $C_i^n$ — collinear in direction *i* with virtuality $\sim\lambda^n$ ($n$ up to $\infty$ for being precisely lightlike),
- $S^m$ — soft with virtuality $\sim\lambda^{2m}$,
- $S^m C_i^n$ — soft-collinear in direction *i* with virtuality $\sim\lambda^{2m+n}$ ($n$ up to $\infty$).
For more detail, see section 3.2 of [arXiv:2601.22144](https://arxiv.org/abs/2601.22144).

In collinear kinematics, on top of these modes above, we also have:
- $sH$ — semihard, with all the components $\sim\sqrt{\lambda}$,
- $G$ — Glauber (in the current version of FRI, the only Glauber mode involved is ($\lambda, \lambda, \sqrt{\lambda}$)),
- $C_i^nC_{ij}$ — collinear in direction *i* with virtuality $\sim\lambda^{n+1}$ ($n$ up to $\infty$ for being precisely lightlike), this notation is for parton *i* when partons *i* and *j* have momenta collinear to each other.
- $S^mC_i^nC_{ij}$ — soft-collinear in direction *i* with virtuality $\sim\lambda^{2m+n+1}$.

### Usage

Run `python3 facet_regions_interactive.py` and choose a kinematics class; the browser takes the graph (edge list + external attachments) as the only input — the cut formalism stays internal.  It enumerates all regions and lists them together with their momentum-space mode assignment: one entry per internal edge, in input order — the same scaling vector that the region finders of pySecDec report, as compared in the cross-checks below.
From there the browser can: inspect individual regions (per-mode subgraphs and loop numbers, plus a concrete independent-loop-momentum basis), translate them to the Lee-Pomeransky parametric representation ($x_e \sim \lambda^{v_e}$ with $v_e = -V$, edge order) — the language natural to integration-by-regions tools — classify the list by the characteristic (softest) mode of each region, or render regions as figures: a single PDF atlas (default; A4 pages of rows, each with the region figure, `R{n}  v = (...)`, and the per-mode lines) or one PNG per region, written to `<module>/fri_out/regions_<timestamp>/` together with the rendering script and a preview page.  The renderer checks font fidelity first, so a broken font encoding fails loudly instead of producing silent glyph errors.  See *Quick start* for the commands.

---

## Byproducts

### Choosing independent loop momenta

For a given region, a basis of loop momenta adapted to its mode hierarchy is what makes the region's power counting manifest: `indep_loops.py` (wide-angle; ported to the Regge kinematics as `regge_indep_loops.py`) constructs one.  For every mode present, its contracted subgraph (the vertices carrying that join mode, plus one auxiliary vertex that absorbs all remaining endpoints) has a cycle rank equal to the number of independent loop momenta of that mode; the module returns a concrete basis (per one-vertex-irreducible block), with an option to force lines into the basis and a feasibility check (`forced_basis` / `forced_feasible`).  The interactive enumerators expose this under option **1)**; standalone demo: `python3 wide_angle/tests/indep_loops_demo.py`.
Whether the choice matters beyond bookkeeping is theory-dependent (e.g. power counting of operators in an EFT expansion) — FRI reports the bases, but performs no power counting itself.

### Scaleless diagnosis

The counterpart of the region browser: given an assignment of momentum modes to the edges that is NOT a region, the expanded integral is scaleless — and `scaleless_diagnosis.py` explains why.  It re-runs the region checks in their fixed order (momentum conservation → jet connectivity → contracted 1VI components → mojetic → First Connectivity → IR compatibility) and stops at the first failure, naming the responsible subgraph: a vertex, a jet, a single-vertex-attached component, a hard-jet interaction, or the union of subgraphs failing to confirm.
Input: one mode per edge (vertex modes are derived); the built-in example reproduces 4pt3loop region 8.
Usage: `python3 wide_angle/scaleless_diagnosis.py`.

### Relevant modes at each loop order

Modes switch on in a fixed order as the loop count grows — each mode has a definite first-appearance level.  `mode_levels.py` encodes this ladder for the two spacelike-collinear frameworks: Regge (`spacelike_collinear/regge/mode_levels.py`, k0–k5) and five-point 2→3 (`spacelike_collinear/2to3/mode_levels.py`, k0–k4) — where the same data also derives, per graph, the cut-chain refinement levels used by the skeleton enumerators ($L = E - V + 1$).  Each framework ships an interactive stepper (`mode_level_interactive.py`; the 2→3 one is also reachable from the unified entry): input the external-momentum modes, then walk up the loop orders — every step lists the newly available modes with their mechanism of origin (a messenger tower, two confirmed components meeting, and so on). Both modules are self-checking against the tabulated ladders (`--check`).

## Implementations

Two kinematics classes are implemented — **wide-angle** and **collinear** (timelike or spacelike) — sharing the same philosophy: a region is constructed purely from the graph, by enumerating cut configurations combinatorially and filtering them, and is then judged by the per-component subgraph requirements (the IR-compatibility fixpoint).  No Feynman-parameter computation, polytope geometry or integral evaluation enters at any point.

### Enumeration (cuts, overlay, pruning)

- **Cut construction.**  Every external leg of a refinement-chain type ($C_i^n$, $C_i^nC_{ij}$) whose root is not inside the hard subgraph *H* receives a route to *H* (routes of different legs are kept disjoint); cuts are connected vertex sets containing the leg root, built on nested levels that are confined to their allowed vertex regions and kept off the other legs' routes.  At the lowest tier (k0) the construction reduces to a union form with "≥ 2" leftover rules and corner pruning.
- **Overlay.**  A configuration of cuts is overlaid on the graph: every vertex takes the meet of the modes of the cuts covering it (*H* if none), every edge the meet of its endpoints — yielding the momentum-space mode assignment (the scaling vector) of the candidate region.
- **Filtering.**  Every subject cut must share vertices with partner cuts of other directions under the kinematics-specific overlap condition (in the layered refinements: two partner directions of matching total power); route exclusivity keeps every cut off the other legs' paths.
- **Pruning and fast paths** (all designed to be behaviour-preserving; validated by old-vs-new A/B comparisons plus the cross-checks below):
  - *bitmask fast path* (2026-09-22): cut sets are additionally carried as bitmasks, so dedup keys become fixed-slot integer arrays and the overlap/route tests become bit intersections — masks are cached per graph and computed once along a chain;
  - *per-graph refinement levels* (2026-09-21): the cut-chain levels that can contribute are read off the mode first-appearance ladder ($L = E - V + 1$), so needless refinement levels are never enumerated;
  - *layer compression* (wide-angle): only the usable cut layers are enumerated; higher towers collapse to their lower equivalent counts;
  - *chain-level early kills* (Regge skeleton): most candidates are rejected by count/level conditions before the full overlap test (validated against the overlap condition in shadow mode — zero mis-kills);
  - *deduplication*: candidates are deduplicated by cut set and by vertex-mode assignment; the verdict depends only on the vertex modes, so repeats are skipped cheaply.

In code: the pruned skeleton enumerators live in `wide_angle/skeleton.py` (the canonical implementation; soft externals included), `spacelike_collinear/regge/skeleton.py` and `spacelike_collinear/2to3/skeleton23.py`.

### Checks (subgraph requirements)

Every surviving configuration is judged by the same chain of subgraph requirements: momentum conservation at every vertex, jet connectivity (Coleman–Norton), one-vertex-irreducibility of the contracted mode components, the mojetic (hard–jet) condition, First Connectivity, and the per-component IR-compatibility fixpoint that removes the residual
non-regions.
The requirements are derived in [arXiv:2601.22144](https://arxiv.org/abs/2601.22144) for the wide-angle class; the collinear versions (Regge and the five-point 2→3 kinematics) follow the same cycle of subgraph conditions with the collinear mode algebra and will be presented in a forthcoming work.
A non-region fails at a definite first check — that diagnosis is exactly what `scaleless_diagnosis.py` reports (wide-angle).
The check implementations live in `wide_angle/region_checker.py`, `spacelike_collinear/regge/regge_core.py` and `spacelike_collinear/2to3/fri23.py`.

---

## Quick start

```bash
# unified entry: [1] wide-angle / [2] spacelike-collinear
#   ([2]: regge 2->2 / five-point 2->3 / mode stepper)
python3 facet_regions_interactive.py

# wide-angle: interactive browser for a new graph (direct entry)
python3 wide_angle/facet_regions_interactive.py

# wide-angle: built-in demonstrations on four graphs (full pipeline live)
python3 wide_angle/fri_demo.py          # or: python3 wide_angle/fri_demo.py 1

# wide-angle: why a mode assignment is NOT a region (scaleless diagnosis)
python3 wide_angle/scaleless_diagnosis.py

# Regge 2->2: interactive enumerator for a new graph (kinematics k0..k5)
python3 spacelike_collinear/regge/fri_interactive.py

# Regge 2->2: stepper for the mode first-appearance ladder
python3 spacelike_collinear/regge/mode_level_interactive.py

# five-point 2->3: interactive enumerator for a new graph (kinematics k0..k4);
# the built-in five-point example is the edge list 1-3,1-5,2-3,2-5,3-4,4-5
python3 spacelike_collinear/2to3/fri23_interactive.py

# five-point 2->3: stepper for the mode first-appearance ladder
python3 spacelike_collinear/2to3/mode_level_interactive.py

# self-checks and small demos (plain scripts, no pytest needed)
python3 wide_angle/tests/unit_test_messenger_sc.py
python3 wide_angle/tests/unit_test_massive.py
python3 wide_angle/tests/indep_loops_demo.py
python3 spacelike_collinear/regge/mode_levels.py --check
python3 spacelike_collinear/2to3/mode_levels.py --check
```

Region finding needs no installation: everything runs on a stock Python 3
(standard library only).  The optional region visualisation — option **4)**
in the interactive enumerators (`[a]` single PDF atlas (default) / `[p]`
one PNG per region) — additionally uses `wolframscript` (Wolfram Engine)
for rendering, `gs` (ghostscript) or `pdfunite` (poppler-utils) for
merging atlas pages, and `pdftotext` (poppler-utils) for the atlas font
check.

---

## Layout

```
FRI-project/
├── facet_regions_interactive.py       # unified entry: [1] wide-angle / [2] spacelike-collinear
├── wide_angle/                        # class 1: wide-angle scattering (canonical)
│   ├── fri_demo.py                    #   built-in demonstrations
│   ├── region_checker.py              #   mode algebra, components, mojetic (1VI) checks, IR compat, messengers
│   ├── skeleton.py                    #   pruned skeleton cut enumerator (p_i,q_j externals; n-leg)
│   ├── usable_modes.py                #   IR-compat mode closure (compression)
│   ├── primitives.py                  #   graph primitives + shared checks (incl. contracted-1VI)
│   ├── read_graph.py                  #   input parsing
│   ├── indep_loops.py                 #   independent loop momenta per region
│   ├── facet_regions_interactive.py   #   interactive browser (enumerate + menu)
│   ├── region_plot_wa.py              #   region figures + PDF atlas
│   ├── scaleless_diagnosis.py         #   why a non-region is scaleless
│   └── tests/                         #   fast unit tests + demos
└── spacelike_collinear/               # class 2: spacelike-collinear kinematics
    ├── regge/                         #   part 1: Regge limit of 2->2
    │   ├── regge_core.py              #     engine: mode lattice, cuts, pipeline, IR fixpoint
    │   ├── regge_modes.py             #     S^m C_i^n C_ij mode algebra (meet/join)
    │   ├── regge_indep_loops.py       #     independent loop momenta (semihard fix)
    │   ├── regge_graphs.py            #     example graph library + k0..k5 kinematics
    │   ├── skeleton.py                #     skeleton cut enumerator (k0–k5; fast path)
    │   ├── mode_levels.py             #     mode first-appearance ladder
    │   ├── mode_level_interactive.py  #     interactive stepper for the mode ladder
    │   ├── fri_interactive.py         #     interactive enumerator for a new graph
    │   └── region_plot.py             #     region figures + PDF atlas
    └── 2to3/                          #   part 2: five-point 2->3
        ├── skeleton23.py              #     skeleton cut enumerators (k0–k4; per-graph cut-chain levels)
        ├── mode_levels.py             #     mode first-appearance ladder + cut-chain level derivation
        ├── mode_level_interactive.py  #     interactive stepper for the mode ladder
        ├── fri23.py                   #     mode algebra + region checks (check chain shared with skeleton23)
        ├── fri23_interactive.py       #     interactive enumerator for a new graph
        ├── kin23.py                   #     kinematics table (k0..k4 presets; general virtuality patterns)
        └── region_plot23.py           #     region figures + PDF atlas
```

Dependency ladder (each layer uses only the ones below):

```text
wide_angle/
  skeleton.py           skeleton enumerator (all domains incl. soft externals)
  primitives.py         graph primitives + shared checks (incl. contracted-1VI)
  region_checker.py     mode algebra + region checks (base; no local deps)

spacelike_collinear/regge/
  skeleton.py           skeleton enumerator (fast path; k0-k5)
  mode_levels.py        mode ladder + first-appearance levels
  regge_graphs.py       graph library + k0..k5 kinematics
  regge_core.py         engine: cuts, pipeline, checks (base)

spacelike_collinear/2to3/
  skeleton23.py         skeleton enumerators (k0-k4; levels via mode_levels)
  mode_levels.py        mode ladder + cut-chain levels
  kin23.py              kinematics table
  fri23.py              mode algebra + region checks (base)
```

---

## Validation

Every implementation has been cross-checked against the region finder of [pySecDec](https://github.com/gudrunhe/secdec) (`find_regions`): the two region sets are compared as **sets of scaling vectors** (one entry per internal line, plus the smallness parameter).  All comparisons agree exactly.  Each kinematics class is checked on **two complementary sets of graphs** — (1) hand-built diagrams that target specific region structures, and (2) random batches of 1000+ graphs that probe for unexpected ones — the two catch different kinds of mistakes.

- **wide_angle**: ~830 configurations covering 45+ topologies (2→2 / 2→3 / 1→3, 3–5 loops, planar and nonplanar, including soft-emission families).
  The skeleton enumerator matches pySecDec on all 255 lightlike 2→2 configurations, and on random samples of 1,200 four-leg + 600 five-leg cases (on top of ~19,000 earlier random cases).
- **spacelike_collinear — regge**: the region files of 56 graphs in the six kinematics k0–k5 (lightlike and off-shell external legs, $\lambda^2$-suppressed virtualities); random batches of 1000 + 1000 (3-loop) and 100 + 200 (4-loop) graphs over k0–k5 — all matching.
- **spacelike_collinear — 2to3**: the 11-graph Frog family and 394 variants obtained by attaching a fifth external leg to the 2→2 topologies; random batches of 1000 3-loop, 500 4-loop, and 100 5-loop graphs in the five kinematics k0–k4 — all matching.

---

## Status

Research-grade, version 0.1.0.  License: TBD.

**Open items — general**

Cleared for now.

**Open items — wide-angle** 

1. **More checks are needed for the soft emission branch.**  This part is not well examined compared with the other branches. 1000+ further random diagrams should be included.

**Open items — spacelike-collinear 2→3 (`fri23`).**

Cleared for now.

**Open items — regge**

Cleared for now.
