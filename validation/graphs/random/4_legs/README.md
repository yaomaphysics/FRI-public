# Wide-angle, 4 legs --- random batches

Four-point (2→2) wide-angle graphs with four lightlike external momenta, generated without 2-valent vertices. Every graph was checked in six kinematic configurations.

| batch | graphs | loops | seed |
|-------|-------:|------:|------|
| `rand1000_4l_v8-10_no2v` | 1000 | 4 | 2026091823 |
| `rand200_4l_no2v_v8-10`   | 200  | 4 | 20260917   |
| `rand200_5l_v10-12_no2v`  | 200  | 5 | 2026091824 |

Totals: 1,200 (4-loop) + 200 (5-loop) graphs.

File format: `*.json` --- `seed`, `nlegs`, `loops`, `n`, `graphs`; each graph carries `v` (vertices), `edges`, and `ext` (external attachments).
`*.txt` --- one line per graph with the same data.
