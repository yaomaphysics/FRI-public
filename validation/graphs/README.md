# Graph lists

The exact graphs used in the validation, grouped as:

- `random/` --- the random batches, one set per class:
  - [`4_legs/`](random/4_legs/) --- four-point (2→2) wide-angle graphs, six kinematics
  - [`5_legs/`](random/5_legs/) --- five-point (2→3) wide-angle graphs, four kinematics; also the two soft-emission families (batches marked `*_1soft_no2v` and `*_2soft_no2v`)

  Each batch is provided as JSON (machine-readable) and TXT (one line per graph), together with its seed and loop order.
- `hand_built/` --- the hand-built configurations: the 4-leg topologies, the 5-leg families (a fifth leg attached to the 4-leg topologies), and the soft-emission configurations; one spec book per class (same format as the `examples/` inputs).
