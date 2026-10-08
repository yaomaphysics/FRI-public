# Wide-angle, 5 legs --- random batches

Five-point (2→3) wide-angle graphs with five lightlike external momenta, generated without 2-valent vertices. Every graph was checked in four kinematic configurations.

| batch | graphs used | loops | seed |
|-------|------------:|------:|------|
| `rand1000_3l_no2v` | 1000 | 3 | 2026091202 |
| `rand700_4l_no2v`  | 700  | 4 | 2026091825+2026091203+2026100601 |
| `rand100_5l_no2v`  | 100  | 5 | 2026091205 |
| `rand500_5l_no2v`  | first 200 of 500 | 5 | 20260924 |

Totals: 1,000 (3-loop) + 700 (4-loop) + 300 (5-loop) graphs.

File format: `*.json` --- `seed`, `nlegs`, `loops`, `graphs`; each graph carries `name`, `V`, `E`, `edges`, and `ext_attach` (external attachments).
`*.txt` --- one line per graph with the same data.
