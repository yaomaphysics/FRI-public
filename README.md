# FRI — Facet Region Interpreter

A graph-theoretic **region finder** for multiscale Feynman integrals. Given a graph (topology + external momenta) in the context of **massless scattering**, which we classify into the **wide-angle** and **collinear** types based on their kinematics, FRI constructs **all facet regions** of the asymptotic expansion directly from the graph — the majority, and mostly the entire list of regions. The construction is based on understandings of the all-order region structures in momentum space, without any Feynman-polynomial computation or polytope geometry.

For the wide-angle kinematics, the region-structure understanding is from the paper *"All-order prescription for facet regions in massless wide-angle scattering"* (arXiv:2601.22144). FRI is supposed to enumerate the whole list of facet regions for any Feynman graph under the following conditions.
1. The external momenta are within the following three types:
    (1) the "on-shell momenta" p_i, each close to a lightcone (p_i^2 small or 0);
    (2) the "off-shell momenta" q_j, each with q_j^2 ~ 1 (not small);
    (3) the "soft momenta" l_k, each has all components being small (but can approach zero at distinct speed).
2. No two external momenta are close to the same lightcone.
3. There are no massive propagators in the graph.

For the collinear kinematics, the region-structure understanding is from the author's knowledge which has not yet been published. At this moment, only two kinematics are available in FRI:
1. Five-point scattering with two particles collinear to each other (which we denote by partons 2 and 3, either timelike- or spacelike-collinear).
2. Four-point scattering in the Regge limit (1+2\to 3+4, with 1 and 3 spacelike-collinear while 2 and 4 spacelike-collinear).
It is worth noting that in the second case above, Glauber-mode propagators can emerge.

Pure Python (>= 3.9), standard library only. pySecDec is used *only* for off-line cross-validation, which is not part of this repository. The optional region visualisation renders through the Wolfram Engine — see *Quick start* below.

---

## Implementations

Two kinematics classes are implemented, sharing the same philosophy
(graph-only region construction + per-component IR-compatibility fixpoint):

### `wide_angle/` — wide-angle scattering (canonical implementation)

Arbitrary external modes (H / S^m / C_i^n / SC_i) and external-leg
multiplicity; cross-validated against pySecDec on ~830 configurations
covering 45+ topologies (3–5 loops, planar and nonplanar, including
soft-emission families); interactive browser with region visualisation
(figures or a single-PDF atlas). Graphs with p_i/q_j externals only are
enumerated by the pruned *skeleton* cut enumerator (`skeleton.py`), which
supports general n-leg graphs with all external virtualities scaling as a
single small parameter (the 4-leg and 5-leg wide-angle classes are
validated); soft externals fall back to the layered enumerator.

### `spacelike_collinear/` — spacelike-collinear kinematics

Kinematics with a pair of spacelike-collinear external momenta, in two
variants:

- **`regge/`** — Regge limit of 2→2 scattering (small t-channel invariant): Glauber and semihard modes. Interactive enumerator for an arbitrary 2→2 graph, an example graph library with the six kinematics k0–k5, and region visualisation (figures or a single-PDF atlas).
- **`2to3/`** — five-point 2→3 scattering with a small spacelike pair invariant (s23 ~ λ). Enumerator with a built-in example graph, an interactive browser, and region visualisation (five kinematics k0–k4); cut enumeration is pruned by a skeleton construction, extended in 2026-09 with a strengthened overlap condition (two partner directions of matching total C-power).

---

## Quick start

```bash
# unified entry: [1] wide-angle / [2] spacelike-collinear (regge 2->2 / fri23 2->3)
python3 facet_regions_interactive.py

# wide-angle: built-in demonstrations on four graphs (full pipeline live)
python3 wide_angle/fri_demo.py          # or: python3 wide_angle/fri_demo.py 1

# Regge limit: interactive enumerator for a 2->2 graph (kinematics k0..k5)
python3 spacelike_collinear/regge/fri_interactive.py

# spacelike-collinear 2->3: built-in five-point example (kinematics k0..k4)
python3 spacelike_collinear/2to3/fri23.py           # default k1
python3 spacelike_collinear/2to3/fri23.py k4 -v     # with per-region details

# spacelike-collinear 2->3: interactive enumerator for a new graph
python3 spacelike_collinear/2to3/fri23_interactive.py

# unit tests and small demos (plain scripts, no pytest needed)
python3 wide_angle/tests/unit_test_messenger_sc.py
python3 wide_angle/tests/unit_test_massive.py
python3 wide_angle/tests/indep_loops_demo.py
```

Region finding needs no installation: everything runs on a stock Python 3
(standard library only).  The optional region visualisation — option **4)**
in the interactive enumerators (`[a]` single PDF atlas (default) / `[p]`
one PNG per region) — additionally uses `wolframscript` (Wolfram Engine)
for rendering, `gs` (ghostscript) or `pdfunite` (poppler-utils) for
merging atlas pages, and `pdftotext` (poppler-utils) for the atlas font
check.

---

## What a "region" looks like

A region assigns a **mode** to every edge of the graph:

- `H` — hard, `S^m` — soft, `C_i^n` — collinear in direction *i*
  (`n` up to `inf` for lightlike directions), `SC_i`, and the refined
  pair-collinear families `S^m C_i^n C_ij`.
- A region output is a scaling vector `v = (v_1 ... v_E, 1)`, one entry
  per internal line (`x_e ~ lambda^{v_e}`), plus the edge-mode assignment.
- Meet/join of modes is computed on the `S^m C_i^n C_ij` lattice
  (`wide_angle/region_checker.py`,
  `spacelike_collinear/regge/regge_modes.py`).

For the exact definitions see section 3.2 of arXiv:2601.22144.

---

## Region visualisation

The interactive enumerators (wide-angle, Regge 2→2, fri23 2→3) offer
option **4)**: render the selected regions with the per-mode colour
scheme (one style spec per kinematics class, recorded in
`region_plot_wa.py` / `region_plot.py` / `region_plot23.py`):

- **[a] single PDF atlas** (default): A4 pages holding 5–7 region rows
  each (auto, by the drawn content's aspect; wide graphs get more rows
  so the whitespace between figures stays small).  Each row shows the
  region figure on the left and, on the right, `R{n}  v = (...)`, plus
  one line per mode (`<mode>: v{...} e{...}`) with the mode names typeset
  LaTeX-style (e.g. `S^1C_1^2`, `C_13`, `C_2^2C_24`).
- **[p] individual PNG figures**, one file per region.

Results are written to `<module>/fri_out/regions_<timestamp>/` along with
the rendering script and a page-1 preview.  The atlas runs a font-fidelity
self-check before rendering, so a broken font encoding fails loudly
instead of producing silent glyph errors.

---

## Layout

```
FRI-project/
├── facet_regions_interactive.py       # unified entry: [1] wide-angle / [2] spacelike-collinear
├── wide_angle/                        # class 1: wide-angle scattering (canonical)
│   ├── fri_demo.py                    #   built-in demonstrations
│   ├── region_checker.py              #   mode algebra, components, mojetic (1VI) checks, IR compat, messengers
│   ├── truncation_check.py            #   layered enumerator (C/H in layer 0; incl. layer prefilter)
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
    │   ├── fri_interactive.py         #     interactive enumerator for a new graph
    │   └── region_plot.py             #     region figures + PDF atlas
    └── 2to3/                          #   part 2: five-point 2->3
        ├── fri23.py                   #     enumerator (+ built-in example graph)
        ├── fri23_interactive.py       #     interactive enumerator for a new graph
        ├── skeleton23.py              #     skeleton cut enumerators (k0–k4; strengthened overlap)
        ├── kin23.py                   #     k0..k4 external-virtuality kinematics table
        └── region_plot23.py           #     region figures + PDF atlas
```

Dependency ladder (each layer uses only the ones below):

```text
wide_angle/
  skeleton.py           skeleton enumerator (fast path; p_i/q_j domain)
  truncation_check.py   layered enumerator (all domains; soft-external fallback)
  primitives.py         graph primitives + shared checks (incl. contracted-1VI)
  region_checker.py     mode algebra + region checks (base; no local deps)

spacelike_collinear/2to3/
  skeleton23.py         skeleton enumerators (k0-k4)
  fri23.py              cut construction + check chain
  kin23.py              kinematics table
```

---

## Validation

Every implementation has been cross-checked against the region finder of
[pySecDec](https://github.com/gudrunhe/secdec) (`find_regions`): the two
region sets are compared as **sets of scaling vectors** (one entry per
internal line, plus the smallness parameter).  All comparisons agree
exactly.

- **wide_angle**: ~830 configurations covering 45+ topologies (2→2 / 2→3 /
  1→3, 3–5 loops, planar and nonplanar, including soft-emission families).
  The pruned skeleton enumerator (p_i/q_j externals) reproduces the layered
  region sets on all 255 lightlike 2→2 configurations (~30× faster overall),
  and matches pySecDec on the five-point (2→3) wide-angle class at k0 (the
  Frog family and a 500-graph random 4-loop batch).
- **spacelike_collinear — regge**: the region files of 50 graphs in the six
  kinematics k0–k5 (lightlike and off-shell external legs, λ²-suppressed
  virtualities).
- **spacelike_collinear — 2to3**: ~1500 randomly generated 3-/4-loop graphs
  in the five kinematics k0–k4, plus the 11-graph Frog family.

---

## Status

Research-grade, version 0.1.0.  License: TBD.

**Open items — general**

1. **The cut structure should be derived.** In most of these kinematics, the cuts are imposed at the beginning. Actually they should be derived: given the external kinematics and loop number, I should have given a way to output all the possible modes at this level, based on which the cuts are natural to see. This would need a major upgrade of usable_modes.py which is currently in the wide-angle branch.

**Open items — wide-angle.** 

1. **Extend the skeleton cut enumerator to soft emission.**  Currently the skeleton cut enumerator is applied to those graphs with only p_i, q_j externals. 

**Open items — spacelike-collinear 2→3 (`fri23`).**

1. **5-loop graphs remain untested for k2--k4** Under the current enumeration method, it may be time-consuming. I will do it once more optimizations are made.

**Open items — regge.** 

1. **Skeleton pruning method not yet implemented here.**

2. **The treatments for k2--k4 should be more unified so that users can understand better.**
