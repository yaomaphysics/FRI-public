# FRI — Facet Region Interpreter

A graph-theoretic **region finder** for multiscale Feynman integrals. Given a graph (topology + external momenta) in the context of **massless scattering**, which we classify into the **wide-angle** and **collinear** types based on their kinematics, FRI constructs the momentum representation of **all facet regions** of the asymptotic expansion directly from the graph — the majority, and mostly the entire list of regions. The construction is based on understandings of the all-order region structures in momentum space, without any Feynman-polynomial computation or polytope geometry.

For the wide-angle kinematics, the region-structure understanding is from the paper *"All-order prescription for facet regions in massless wide-angle scattering"* ([arXiv:2601.22144](https://arxiv.org/abs/2601.22144)). The kinematics scope satisfy the following conditions.
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

Pure Python (>= 3.9), standard library only — region finding needs no installation. pySecDec is used *only* for off-line cross-validation, which is not part of this repository. The optional region visualisation additionally uses `wolframscript` (Wolfram Engine) for rendering, `gs` (ghostscript) or `pdfunite` (poppler-utils) for merging atlas pages, and `pdftotext` (poppler-utils) for the atlas font check.

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

Run `python3 facet_regions_interactive.py`; the entry first asks for the kinematics class (`[1] wide-angle` / `[2] collinear`, and under collinear `[1] 2->3` / `[2] 2->2`), then offers what that class can do: **obtain regions from a single graph**, or **mode prediction** (the collinear mode ladder, or the wide-angle usable-mode closure).  The region browser takes the graph (edge list + external attachments) as the only input — the cut formalism stays internal.  It enumerates all regions and lists them together with their momentum-space mode assignment: one entry per internal edge, in input order — the same scaling vector that the region finders of pySecDec report, as compared in the cross-checks below.
From there the browser can: inspect individual regions (per-mode subgraphs and loop numbers, plus a concrete independent-loop-momentum basis), translate them to the Lee-Pomeransky parametric representation ($x_e \sim \lambda^{v_e}$ with $v_e = -V$, edge order), classify the list by the characteristic (softest) mode of each region or by power counting (optionally with a numerator polynomial in the edge momenta; a per-region derivation of the power is available as a PDF report), or render regions as figures: a single PDF atlas (default; A4 pages of rows, each with the region figure, `R{n}  v = (...)`, and the per-mode lines) or one PNG per region, written to `<module>/fri_out/regions_<timestamp>/` together with the rendering script and a preview page.  The renderer checks font fidelity first, so a broken font encoding fails loudly instead of producing silent glyph errors.

---

## Byproducts

### Choosing independent loop momenta

For a given region, a basis of loop momenta adapted to its mode hierarchy is what makes the region's power counting manifest: `indep_loops.py` (one per framework — wide-angle, five-point 2->3, and Regge) constructs one.  For every mode present, its contracted subgraph (the vertices carrying that join mode, plus one auxiliary vertex that absorbs all remaining endpoints) has a cycle rank equal to the number of independent loop momenta of that mode; the module returns a concrete basis (per one-vertex-irreducible block), with an option to force lines into the basis and a feasibility check.  The interactive enumerators expose this under option **1)**.
Whether the choice matters beyond bookkeeping is theory-dependent (e.g. power counting of operators in an EFT expansion) — FRI reports the bases.

### Power counting

For a scalar integral, a region's contribution scales as $\lambda^{A-B}$: the loop measure contributes $A = (2-\epsilon)\sum_X r_X V(X)$ (one factor $(\lambda^{V(X)})^{2-\epsilon}$ per independent loop momentum), the propagators $B = \sum_e V_e$ (the virtuality degrees of the edge modes).  The browser groups all regions by this power and can emit a per-region derivation of it as a PDF report.  A numerator polynomial in the edge momenta $K_i$ (input edge order) is supported as well: it is expanded per region into the independent loop momenta, the scalar product of two modes $S^{m_1}C_i^{n_1}$ and $S^{m_2}C_j^{n_2}$ scales as $m_1+m_2+\min(n_1,n_2)$ for the same direction and as $m_1+m_2$ for distinct directions, with sums taking the minimum and products adding, and the resulting integer $N$ enters as $A - B + N$.  Implemented for wide-angle; the collinear frameworks are to follow.

### Scaleless diagnosis

The counterpart of the region browser: given an assignment of momentum modes to the edges that is NOT a region, the expanded integral is scaleless — and `scaleless_diagnosis.py` explains why.  It re-runs the region checks in their fixed order (momentum conservation → jet connectivity → mojetic → First Connectivity → IR compatibility) and stops at the first failure, naming the responsible subgraph: a vertex, a jet, a hard-jet interaction, or the union of subgraphs failing to confirm.
Input: one mode per edge (vertex modes are derived); the built-in example reproduces 4pt3loop region 8.
Usage: `python3 wide_angle/scaleless_diagnosis.py`.

### Relevant modes at each loop order

In the two collinear frameworks: Regge (`collinear/regge/mode_levels.py`) and five-point 2→3 (`collinear/2to3/mode_levels.py`), momentum modes are switched on in a fixed order as the loop count grows, and possibly end up with a cascade of modes. It is natural to ask the following (pre-graph) question: what modes are involved at the amplitude level, and for each mode, what is its first-appearance loop level? `mode_levels.py` encodes this through an interactive stepper (built into its `facet_regions_interactive.py`, run with `--ladder`, and reachable from the unified entry): input the external-momentum modes, then walk up the loop orders — every step lists the newly available modes with their origins explained.

For the wide-angle kinematics, it has been proved by corollary 5.4.2 of [2601.22144](https://arxiv.org/abs/2601.22144) that with the external kinematics given, the amplitude-level momentum modes are finite (no cascading modes). `wide_angle/usable_modes.py` clarifies the precise list of modes, using the notion of ``infrared compatibility'' developed in §5.2 of [2601.22144](https://arxiv.org/abs/2601.22144).

Both views sit behind the same second-level menu (class → `[2]`), and both can output derivations in a PDF report (`r` in the stepper, shared renderer `report_pdf.py`).

## Implementations

Two kinematics classes are implemented — **wide-angle** and **collinear** (timelike or spacelike) — sharing the same philosophy: a region is constructed purely from the graph, by enumerating cut configurations combinatorially and filtering them, and is then judged by the subgraph requirements (based on Coleman--Norton interpretation, connectivity, and infrared compatibility). No Feynman-parameter computation, polytope geometry or integral evaluation enters at any point.

### Enumeration (cuts, overlay, pruning)

- **Cut construction.**  Every external leg of a refinement-chain type ($C_i^n$, $C_i^nC_{ij}$) whose root is not inside the hard subgraph *H* receives a route to *H* (routes of different legs are kept disjoint); cuts are connected vertex sets containing the leg root, built on nested levels that are confined to their allowed vertex regions and kept off the other legs' routes.
- **Pruning and fast paths**:
  - *overlap constraint*: for any $C_i^m$-mode external momentum, any $C_i^n$-mode cut around it (with $n < m$) must share a vertex with cuts from other two directions; this requirement is from the infrared compatibility.
  - *bitmask fast path*: cut sets are additionally carried as bitmasks, so dedup keys become fixed-slot integer arrays and the overlap/route tests become bit intersections — masks are cached per graph and computed once along a chain;
  - *memoisation*: the graph kernels used by both enumeration and judgment — connectivity, 1VI, connected components and 1VI-block finding — are cached by content and cleared between cases;
  - *per-graph refinement levels*: the cut-chain levels that can contribute are read off the mode first-appearance ladder ($L = E - V + 1$), so needless refinement levels are never enumerated;
  - *layer compression*: only the usable cut layers are enumerated; higher towers collapse to their lower equivalent counts;
  - *chain-level early kills*: most candidates are rejected by count/level conditions before the full overlap test (validated against the overlap condition in shadow mode — zero mis-kills);
  - *deduplication*: candidates are deduplicated by cut set and by vertex-mode assignment; the verdict depends only on the vertex modes, so repeats are skipped cheaply.
- **Overlay.**  A configuration of cuts is overlaid on the graph: every vertex takes the meet of the modes of the cuts covering it (*H* if none), every edge the meet of its endpoints — yielding the momentum-space mode assignment (the scaling vector) of the candidate region.

In code: the enumerators live in `wide_angle/enumerator.py`, `collinear/2to3/enumerator.py`, and `collinear/regge/enumerator.py`.

### Filtration (subgraph requirements)

Every surviving configuration is judged by the same chain of subgraph requirements: momentum conservation at every vertex, Coleman--Norton interpretation, connectivity requirement, and infrared compatibility requirement that remove the residual non-regions.
These requirements are derived in [arXiv:2601.22144](https://arxiv.org/abs/2601.22144) for the wide-angle class; the collinear versions (Regge and the five-point 2→3 kinematics) follow the same cycle of subgraph conditions with the collinear mode algebra and will be presented in a forthcoming work.
A non-region fails at a definite first check — that diagnosis is exactly what `scaleless_diagnosis.py` reports (wide-angle).

In code: the filters live in `wide_angle/region_filter.py`, `collinear/2to3/region_filter.py`, and `collinear/regge/region_filter.py`.

---

## Layout

```
FRI-project/
├── facet_regions_interactive.py       # unified entry: class -> side -> [1] regions / [2] mode prediction
├── report_pdf.py                      # shared monospace report renderer (PDF) for the two mode views
├── wide_angle/                        # class 1: wide-angle scattering (canonical)
│   ├── primitives.py                  #   shared base: mode algebra + graph machinery
│   ├── enumerator.py                  #   pruned skeleton cut enumerator (enumeration core)
│   ├── region_filter.py              #   region judgment based on subgraph requirements
│   ├── indep_loops.py                 #   independent loop momenta per region
│   ├── power_counting.py              #   scalar + numerator power counting for the region browser
│   ├── power_report.py                #   per-region derivation report (PDF) for the power counting
│   ├── usable_modes.py                #   IR-compatible mode closure (compression; optional derivation trace)
│   ├── mode_report.py                 #   usable-mode derivation report (PDF)
│   ├── facet_regions_interactive.py   #   interactive browser (enumerate + menu)
│   ├── region_plot.py                 #   region figures + PDF atlas
│   └── scaleless_diagnosis.py         #   why a non-region is scaleless
├── collinear/                         # class 2: collinear kinematics
│   ├── 2to3/                          #   part 1: five-point 2->3
│   │   ├── primitives.py               #     mode algebra + graph tools + the external-kinematics API (zero-judgment base layer)
│   │   ├── enumerator.py               #     skeleton cut enumerators (k0--k4; per-graph cut-chain levels)
│   │   ├── region_filter.py           #     region judgment based on subgraph requirements
│   │   ├── indep_loops.py               #     independent loop momenta + line-momentum parameterization
│   │   ├── mode_levels.py              #     mode first-appearance ladder + cut-chain level derivation
│   │   ├── mode_report.py              #     per-level derivation report (PDF)
│   │   ├── facet_regions_interactive.py #     interactive enumerator for a new graph (built-in example: 1-3,1-5,2-3,2-5,3-4,4-5)
│   │   └── region_plot.py              #     region figures + PDF atlas
│   └── regge/                         #   part 2: Regge limit of 2->2
│       ├── primitives.py              #     mode algebra + mode strings + graph tools (zero-judgment base layer)
│       ├── enumerator.py              #     skeleton cut enumerators (k0--k5; the all-lightlike k1 case by its own engine)
│       ├── region_filter.py          #     region judgment based on subgraph requirements
│       ├── indep_loops.py             #     independent loop momenta (semihard fix)
│       ├── mode_levels.py             #     mode first-appearance ladder
│       ├── mode_report.py             #     per-level derivation report (PDF)
│       ├── facet_regions_interactive.py #     interactive enumerator for a new graph
│       └── region_plot.py             #     region figures + PDF atlas
└── validation/                        # cross-checks vs pySecDec (graphs, records, examples; see validation/README.md)
```

---

## Validation

Every implementation has been cross-checked against the region finder of [pySecDec](https://github.com/gudrunhe/secdec) (`find_regions`): the two region sets are compared as **sets of scaling vectors** (one entry per internal line, plus the smallness parameter).  All comparisons agree exactly.  Each kinematics class is checked on **two complementary sets of graphs** — (1) hand-built diagrams that target specific region structures, and (2) random batches of 1000+ graphs that probe for unexpected ones — the two catch different kinds of mistakes.  The complete records of these comparisons — the exact graph lists, the case-by-case verdicts, worked examples, and a ready-to-run recomputation script — are collected in [`validation/`](validation/).

<!-- raw HTML because markdown pipe-tables cannot merge cells; renders on GitHub, VS Code, and the preview tooling -->

<table>
<tr><th colspan="2">class</th><th>hand-built</th><th>random</th></tr>
<tr><td rowspan="5"><b>wide-angle</b></td><td><b>4 legs</b><br><span style="font-size: 90%; white-space: nowrap;">(with&nbsp;6&nbsp;kinematics)</span></td><td>52 topologies (3–5 loops)</td><td>1,200 (4-loop) + 200 (5-loop) graphs (on top of ~19,000 earlier random cases)</td></tr>
<tr><td><b>5 legs</b><br><span style="font-size: 90%; white-space: nowrap;">(with&nbsp;4&nbsp;kinematics)</span></td><td>397 graphs (a 5th leg attached to some 4-leg topologies)</td><td>1,000 (3-loop) + 700 (4-loop) + 300 (5-loop)</td></tr>
<tr><td><b>1 additional soft emission</b><br><span style="font-size: 90%; white-space: nowrap;">(with&nbsp;10&nbsp;kinematics)</span></td><td>470 configurations: the soft-emission families (Crown*ASE, v5-ASE, Fish soft-leg) plus CheesePizza</td><td>1,000 (3-loop) + 700 (4-loop) + 300 (5-loop)</td></tr>
<tr><td><b>2 additional soft emissions</b><br><span style="font-size: 90%; white-space: nowrap;">(with&nbsp;2&nbsp;kinematics)</span></td><td>—</td><td>1,000 (3-loop) + 500 (4-loop)</td></tr>
<tr><td><b>other topologies</b></td><td>Some test configurations</td><td>—</td></tr>
<tr><td colspan="2"><b>collinear: 2→3</b> (p2 ∥ p3)<br><span style="font-size: 90%; white-space: nowrap;">(with&nbsp;5&nbsp;kinematics)</span></td><td>397 graphs (a 5th leg attached to some 4-leg topologies)</td><td>1,000 (3-loop) + 700 (4-loop) + 300 (5-loop) graphs</td></tr>
<tr><td colspan="2"><b>collinear: 2→2 (Regge)</b> (p1 ∥ p3, p2 ∥ p4)<br><span style="font-size: 90%; white-space: nowrap;">(with&nbsp;6&nbsp;kinematics)</span></td><td>region files of 56 graphs</td><td>1,000 (3-loop) + 700 (4-loop) + 300 (5-loop) graphs</td></tr>
</table>

### Runtimes

As an example, the table below compares the average wall-clock time per case of FRI and of pySecDec's `find_regions` for the 5-leg class (averages over the 5-leg random batches and their four kinematics):

| loops | cases | FRI | pySecDec | speed-up |
|------:|------:|----:|---------:|---------:|
| 3 loops | 4,000 | 0.02 s | 0.02 s | ≈ 1× |
| 4 loops | 2,800 | 0.11 s | 0.83 s | ≈ 7× |
| 5 loops | 1,200 | 1.1 s | 160 s | ≈ 150× |

The advantage of FRI grows rapidly with the number of loops: the two tools are comparable on the smallest (3-loop) samples, while in the heaviest (5-loop) cases FRI is over two orders of magnitude faster — a single `find_regions` call takes minutes on average there.  The full per-group tables are in [`validation/runtimes_summary.txt`](validation/runtimes_summary.txt).

---

## Status

Research-grade, version 0.1.0.  License: MIT (code); CC BY 4.0 (validation data).  © 2026 ETH Zurich; created by Yao Ma. Below list some open items, which are functions supposed to be realized in the near future.

**Open items — general**

Interactive part -- power-counting module: added for wide-angle (scalar integrals and polynomial numerators); the collinear frameworks to follow.

**Open items — wide-angle** 

Cleared for now.

**Open items — collinear 2→3**

Cleared for now.

**Open items — regge**

Cleared for now.

---

## Citation

If you use FRI in your work, please cite the accompanying paper:

Y. Ma, *All-order prescription for facet regions in massless wide-angle scattering*, [arXiv:2601.22144](https://arxiv.org/abs/2601.22144).

```bibtex
@article{Ma:2026facet,
  author        = {Ma, Yao},
  title         = {All-order prescription for facet regions in massless wide-angle scattering},
  year          = {2026},
  eprint        = {2601.22144},
  archivePrefix = {arXiv},
  primaryClass  = {hep-ph}
}
```
