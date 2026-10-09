# Comparison records

Per-case records of the FRI--pySecDec comparisons:

- [`summary.txt`](summary.txt) --- one line per random batch: cases, agreements, scaleless cases, mismatches.
- `<batch>_report.txt` --- one line per case for the random batches in [`../graphs/random/`](../graphs/random/): graph, kinematics, verdict, region count.
- [`handbuilt_summary.txt`](handbuilt_summary.txt) --- one line per hand-built case (configurations in [`../graphs/hand_built/`](../graphs/hand_built/)): verdict, region count, date.

Totals: 18,770 cases (16,400 random + 2,370 hand-built), 0 mismatches. Some random-batch cases are scaleless (both tools find no regions); these are marked "scaleless".

In every case the FRI and pySecDec region lists agree exactly as sets of scaling vectors (not merely in the number of regions).
