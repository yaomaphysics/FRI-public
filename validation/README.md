# Validation of FRI against pySecDec

Companion records for section 8 of *"All-order prescription for facet regions in massless wide-angle scattering"* ([arXiv:2601.22144](https://arxiv.org/abs/2601.22144)).

[FRI](https://github.com/yaomaphysics/FRI-public) is the program that implements the region prescription of the paper. For the graph families listed below, the complete lists of facet regions obtained from FRI were compared against the region finder of [pySecDec](https://github.com/gudrunhe/secdec). After translating the FRI regions from momentum space to the Lee--Pomeransky parameter space, the two lists of region vectors were compared **as sets**: in every case they agree exactly --- no missing regions, no extra regions.

## Scope

| Class | hand-built | random |
|-------|------------|--------|
| 4 legs (in 6 chosen kinematics) | 52 topologies (3--5 loops) | 1,200 (4-loop) + 200 (5-loop) graphs |
| 5 legs (in 4 chosen kinematics) | 397 graphs (a 5th leg attached to some 4-leg topologies) | 1,000 (3-loop) + 700 (4-loop) + 300 (5-loop) graphs |
| soft emission | 470 configurations (the soft-emission families) | --- |

<!-- At posting: fill in the FRI commit hash for v0.1.0 below. -->

## Contents

- [`graphs/`](graphs/) --- the exact graph lists for every family and batch in the scope table.
- [`results/`](results/) --- the case-by-case comparison records: per-case verdicts and region counts.
- [`examples/`](examples/) --- one or two worked examples with complete inputs and both sides' full region lists, for inspection without running anything.
- [`run_fri.py`](run_fri.py) --- a small runner that recomputes FRI's region lists for all cases (no pySecDec needed).
- [`runtimes_summary.txt`](runtimes_summary.txt) --- the FRI vs pySecDec runtime comparison (per batch and kinematics).

## Versions

- FRI: v0.1.0 (commit `...`)  <!-- fill in at posting time -->
- pySecDec: 1.6.6

## Reproducing -- for the user's curiosity

A ready-made runner is included --- `python3 run_fri.py --all` recomputes FRI's region lists for every validated case and reports, per case, the FRI runtime and whether the region count matches the stored record in `results/` (only the FRI code from the main repository is used; pySecDec is not needed).
The output should be like:

```text
FRI region runner (validation package) -- uses the FRI code from the main repository; pySecDec is not needed.
Each line: <case> <kin>: <N> regions  (<time>s, matched) -- "matched" = the count equals the stored record in results/.

== rand1000_4l_v8-10_no2v.json (1000 graphs, seed 2026091823) ==
  0000 v=9 k0: 309 regions  (0.14s, matched)
  0000 v=9 k1: 62 regions  (0.04s, matched)
  0000 v=9 k2: 76 regions  (0.12s, matched)
  ...
```

See `python3 run_fri.py --help` for per-batch and per-case modes, and `--full` to print the complete region lists.

Alternatively, run FRI on any graph from `graphs/` (see the main repository for usage), translate the regions to parameter space as described in the paper, and compare with the output of pySecDec's region finder.
The records in `results/` contain the expected verdicts and counts.

## Note

The collinear kinematic classes implemented in FRI are validated separately, which will be documented in a forthcoming publication.

## License

The graph lists and comparison records in this directory are released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/); the FRI source code is released under the MIT License (© 2026 ETH Zurich; created by Yao Ma) --- see the main repository.
