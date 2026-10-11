# Comparison records

Per-case records of the FRI--pySecDec comparisons:

- [`summary.txt`](summary.txt) --- one line per random batch: cases, agreements, scaleless cases, mismatches.
- `<batch>_report.txt` --- one line per case for the random batches in [`../graphs/random/`](../graphs/random/): graph, kinematics, verdict, region count.
- [`handbuilt_summary.txt`](handbuilt_summary.txt) --- one line per hand-built case (configurations in [`../graphs/hand_built/`](../graphs/hand_built/)): verdict, region count, date.

Totals: 40,963 recorded cases (38,593 random + 2,370 hand-built), 0 mismatches (the 5-loop 1-soft random batch is partially recorded: 2,193 of its 3,000 cases; the rest are pending). Some random-batch cases are scaleless (both tools find no regions); these are marked "scaleless".

In every case the FRI and pySecDec region lists agree exactly as sets of scaling vectors (not merely in the number of regions).
