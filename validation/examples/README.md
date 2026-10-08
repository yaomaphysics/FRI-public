# Worked examples

Two representative cases are provided in full --- the graph input, pySecDec's
region list, and FRI's region list:

- [`section 7.1 example/`](section%207.1%20example/) --- `CrownST_k2`, the
  three-loop four-leg example (81 regions), discussed in section 7.1 of the
  paper.
- [`section 7.2 example/`](section%207.2%20example/) --- `CheesePizza_k1`, the
  six-loop 1→3 decay plus soft emission example (199 regions), discussed in
  section 7.2 of the paper.

Each directory contains:

- `pysecdec_input_and_regions.txt` --- the pySecDec input (graph + kinematics)
  and the region vectors returned by `find_regions`.
- `fri_regions.txt` --- FRI's regions in momentum-space form, with the
  translated scaling vector for each region.

For each case the two region sets agree exactly (as sets of scaling vectors;
last entry = the smallness parameter).
