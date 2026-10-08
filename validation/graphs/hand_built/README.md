# Hand-built configurations

The hand-built graphs used in the validation, one spec book per class (internal class names):

| file | cases | content |
|------|------:|---------|
| `lightlike_2to2.txt` | 261 | four-point (2→2) topologies, k0–k4 |
| `decay_1to3.txt` | 51 | the same topologies in the decay kinematics |
| `lightlike_2to3.txt` | 44 | five-point (2→3) Frog families, k0–k3 |
| `lightlike_2to2_1soft.txt` | 409 | soft-emission variants of the 4-leg topologies (soft + hard external legs) |
| `decay_1to3_1soft.txt` | 61 | soft-emission variants in the decay kinematics |

Each book contains one spec block per case: the graph (internal lines), the external lines, and the kinematics (replacement rules), in the same format as the pySecDec inputs in
[`../examples/`](../examples/).

Per-case records (verdict, region count, date) are in
[`../../results/handbuilt_summary.txt`](../../results/handbuilt_summary.txt).
