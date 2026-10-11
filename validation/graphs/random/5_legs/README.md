# Wide-angle, 5 legs --- random batches

Five-point (2→3) wide-angle graphs with five lightlike external momenta, generated without 2-valent vertices. Every graph of the first four batches was checked in four kinematic configurations; the soft-emission batches are checked in ten (one additional soft emission) and two (two additional soft emissions) kinematic configurations respectively.

| batch | graphs used | loops | seed |
|-------|------------:|------:|------|
| `rand1000_3l_no2v` | 1000 | 3 | 2026091202 |
| `rand700_4l_no2v`  | 700  | 4 | 2026091825+2026091203+2026100601 |
| `rand100_5l_no2v`  | 100  | 5 | 2026091205 |
| `rand500_5l_no2v`  | first 200 of 500 | 5 | 20260924 |
| `rand1000_3l_1soft_no2v` | 1000 | 3 | 20261010 |
| `rand700_4l_1soft_no2v`  | 700  | 4 | 20261010 |
| `rand300_5l_1soft_no2v`  | 300  | 5 | 20261010 |
| `rand1000_3l_2soft_no2v` | 1000 | 3 | 20261011 |
| `rand500_4l_2soft_no2v`  | 500  | 4 | 20261011 |

Totals: 1,000 (3-loop) + 700 (4-loop) + 300 (5-loop) graphs with five hard lightlike legs, plus the soft-emission batches: 1,000 (3-loop) + 700 (4-loop) + 300 (5-loop) graphs with one additional soft emission (legs p1,p2,p3,l1,q1; ten kinematics) and 1,000 (3-loop) + 500 (4-loop) graphs with two additional soft emissions (legs p1,p2,l1,l2,q1; two kinematics so far).

File format: `*.json` --- `seed`, `nlegs`, `loops`, `graphs`; each graph carries `name`, `V`, `E`, `edges`, and `ext_attach` (external attachments).
`*.txt` --- one line per graph with the same data.
