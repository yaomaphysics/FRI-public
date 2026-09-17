# FRI — Facet Region Interpreter

A graph-theoretic **region finder** for multiscale Feynman integrals.
Given a graph (topology + external momenta), FRI constructs **all regions**
of the asymptotic expansion directly from the graph — no Feynman-polynomial
computation, no polytope geometry.  The construction implements the region
conditions of *"All-order prescription for facet regions in massless
wide-angle scattering"* (arXiv:2601.22144): fundamental pattern,
connectivity, and infrared compatibility.

Pure Python (>= 3.9), standard library only.  pySecDec is used *only* for
off-line cross-validation, which is not part of this repository.  The
optional region visualisation renders through the Wolfram Engine — see
*Quick start* below.

---

## Implementations

Two kinematics classes are implemented, sharing the same philosophy
(graph-only region construction + per-component IR-compatibility fixpoint):

### `wide_angle/` — wide-angle scattering (canonical implementation)

Arbitrary external modes (H / S^m / C_i^n / SC_i) and external-leg
multiplicity; cross-validated against pySecDec on ~830 configurations
covering 45+ topologies (3–5 loops, planar and nonplanar, including
soft-emission families); interactive browser with region visualisation
(figures or a single-PDF atlas).

### `spacelike_collinear/` — spacelike-collinear kinematics

Kinematics with a pair of spacelike-collinear external momenta, in two
variants:

- **`regge/`** — Regge limit of 2→2 scattering (small t-channel
  invariant): Glauber and semihard modes.  Interactive enumerator for an
  arbitrary 2→2 graph, an example graph library with the six kinematics
  k0–k5, and region visualisation (figures or a single-PDF atlas).
- **`2to3/`** — five-point 2→3 scattering with a small spacelike pair
  invariant (s23 ~ λ).  Enumerator with a built-in example graph, an
  interactive browser, and region visualisation (five kinematics k0–k4).

---

## Quick start

```bash
# unified entry: [1] wide-angle / [2] spacelike-collinear (regge 2->2 / fri23 2->3)
python3 facet_regions_interactive.py

# wide-angle: interactive region browser
python3 wide_angle/fri.py

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
│   ├── fri.py                         #   interactive region browser (entry)
│   ├── fri_demo.py                    #   built-in demonstrations
│   ├── region_checker.py              #   mode algebra, components, IR compat, messengers
│   ├── truncation_check.py            #   layered enumerator (C/H in layer 0)
│   ├── mojetic_check.py               #   H∪J∖J_i mojetic (1VI) check
│   ├── contracted_1vi.py              #   contracted-mode-component 1VI check
│   ├── usable_modes.py                #   IR-compat mode closure (compression)
│   ├── primitives.py                  #   Step1 / first-connectivity / IR-primitive checks
│   ├── read_graph.py                  #   input parsing
│   ├── indep_loops.py                 #   independent loop momenta per region
│   ├── shared_prefilter.py            #   shared-vertex prefilter
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
        ├── kin23.py                   #     k0..k4 external-virtuality kinematics table
        └── region_plot23.py           #     region figures + PDF atlas
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
- **spacelike_collinear — regge**: the region files of 50 graphs in the six
  kinematics k0–k5 (lightlike and off-shell external legs, λ²-suppressed
  virtualities).
- **spacelike_collinear — 2to3**: ~1500 randomly generated 3-/4-loop graphs
  in the five kinematics k0–k4, plus the 11-graph Frog family.

---

## Status

Research-grade, version 0.1.0.  License: TBD.

**Open items — spacelike-collinear 2→3 (`fri23`), 2026-09-17.**  The recently
enabled 4-loop scans exposed two work items:

1. **Complete the generated cut structure.**  Some regions (found so far in
   `k2`) require refinement levels of the cut chains (e.g. `C4R1`/`C5R1`-type
   levels) that the current chain construction does not generate.  The gap
   lies in the cut generation itself and is *not* caused by the recent pruning
   optimizations — it became visible only now that the 4-loop scans run to
   completion.  Extend the cut families for the affected kinematics, together
   with additional strong restrictions to keep the combinatorial growth in
   check.
2. **Per-kinematics pruning for `k3`/`k4`.**  Enumerations in `k3`/`k4`
   remain slow — some graphs do not finish within ~10 minutes — and need
   further pruning conditions matched to each individual kinematics.

