# FRI — Facet Region Interpreter

A graph-theoretic **region finder** for multiscale Feynman integrals.
Given a graph (topology + external momenta), FRI constructs **all regions**
of the asymptotic expansion directly from the graph — no Feynman-polynomial
computation, no polytope geometry.  The construction implements the region
conditions of *"All-order prescription for facet regions in massless
wide-angle scattering"* (arXiv:2601.22144): fundamental pattern,
connectivity, and infrared compatibility.

Pure Python (>= 3.9), standard library only.  pySecDec is used *only* for
off-line cross-validation, which is not part of this repository.

---

## Implementations

Three kinematics regimes are implemented, sharing the same philosophy
(graph-only region construction + per-component IR-compatibility fixpoint):

| Directory | Regime | Status |
|---|---|---|
| `wide_angle/` | wide-angle scattering (canonical implementation) | cross-validated against pySecDec: ~830 configurations, 45+ topologies (3–5 loops, planar and nonplanar) |
| `regge_limit/` | Regge limit of 2→2 scattering (Glauber exchange / semihard) | validated against pySecDec: 50 graphs × 6 kinematics (k0–k5) |
| `spacelike_collinear/` | spacelike-collinear 2→3 scattering (five-point) | validated against pySecDec: 1000+ randomized 3-/4-loop graphs × k0–k4 |

---

## Quick start

```bash
# wide-angle: interactive region browser (enumerate + inspect + classify)
python3 fri.py

# wide-angle: built-in demonstrations on four graphs (full pipeline live)
python3 fri_demo.py          # or: python3 fri_demo.py 1

# Regge limit: interactive enumerator for a 2->2 graph (kinematics k0..k5)
python3 regge_limit/fri_interactive.py

# spacelike-collinear: built-in 2->3 example (kinematics k0..k4)
python3 spacelike_collinear/fri23.py            # default k1
python3 spacelike_collinear/fri23.py k4 -v      # with per-region details

# unit tests and small demos (plain scripts, no pytest needed)
python3 wide_angle/tests/unit_test_messenger_sc.py
python3 wide_angle/tests/unit_test_massive.py
python3 wide_angle/tests/indep_loops_demo.py
```

No installation is needed: everything runs on a stock Python 3
(standard library only).

---

## What a "region" looks like

A region assigns a **mode** to every edge of the graph.  Modes are the
large-momentum / small-momentum scalings of the corresponding line:

- `H` — hard, `S^m` — soft, `C_i^n` — collinear in direction *i*
  (`n` up to `inf` for lightlike directions), `SC_i`, and the refined
  families `S^m C_i^n C_ij` (Regge) and pair-family refinements
  (spacelike-collinear).
- A region output is a scaling vector `v = (v_1 ... v_E, 1)`, one entry
  per internal line (`x_e ~ lambda^{v_e}`), plus the edge-mode assignment.
- Meet/join of modes is computed on the `S^m C_i^n C_ij` lattice
  (`wide_angle/region_checker.py`, `regge_limit/regge_modes.py`).

For the exact definitions see section 3.2 of arXiv:2601.22144.

---

## Layout

```
FRI-project/
├── fri.py                        # wide-angle entry: interactive region browser
├── fri_demo.py                   # wide-angle: built-in demonstrations
├── wide_angle/                   # canonical implementation
│   ├── region_checker.py         # mode algebra, components, IR compat, messengers
│   ├── truncation_check.py       # layered enumerator (C/H in layer 0)
│   ├── mojetic_check.py          # H∪J∖J_i mojetic (1VI) check
│   ├── contracted_1vi.py         # contracted-mode-component 1VI check
│   ├── usable_modes.py           # IR-compat mode closure (compression)
│   ├── primitives.py             # Step1 / first-connectivity / IR-primitive checks
│   ├── read_graph.py             # input parsing
│   ├── indep_loops.py            # independent loop momenta per region
│   ├── shared_prefilter.py       # shared-vertex prefilter
│   ├── facet_regions_interactive.py   # interactive browser (enumerate + menu)
│   ├── scaleless_diagnosis.py    # why a non-region is scaleless
│   └── tests/                    # fast unit tests + demos (no dependencies)
├── regge_limit/                  # Regge limit of 2->2 scattering
│   ├── regge_core.py             # engine: mode lattice, cuts, pipeline, IR fixpoint
│   ├── regge_modes.py            # S^m C_i^n C_ij mode algebra (meet/join)
│   ├── regge_indep_loops.py      # independent loop momenta (semihard fix)
│   ├── regge_graphs.py           # example graph library + k0..k5 kinematics
│   └── fri_interactive.py        # interactive enumerator for a new graph
└── spacelike_collinear/          # spacelike-collinear 2->3 scattering
    ├── fri23.py                  # enumerator (+ built-in example graph)
    └── kin23.py                  # k0..k4 external-virtuality kinematics table
```

---

## Validation

Every regime has been cross-checked against the region finder of
[pySecDec](https://github.com/gudrunhe/secdec) (`find_regions`): the two
region sets are compared as **sets of scaling vectors** (one entry per
internal line, plus the smallness parameter).  All comparisons agree
exactly.

- **wide-angle**: ~830 configurations covering 45+ topologies
  (2→2 / 2→3 / 1→3, 3–5 loops, planar and nonplanar, including
  soft-emission families).
- **Regge limit**: the region files of 50 graphs in the six kinematics
  k0–k5 (all-lightlike legs, lightlike + off-shell legs, and
  λ²-suppressed virtualities).
- **spacelike-collinear**: 1000+ randomly generated 3-/4-loop graphs in
  the five kinematics k0–k4, plus the 11-graph Frog family.

---

## Status

Research-grade, version 0.1.0.  License: TBD.
