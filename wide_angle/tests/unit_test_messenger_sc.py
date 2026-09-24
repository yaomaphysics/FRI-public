#!/usr/bin/env python3
"""unit_test_messenger_sc.py — prove the corrected messenger direction-counting:
targets with n_i >= 1 are collected via the matching Gamma member S^m C_i^{n_i}
(not only via the kernel).

Scenario: degree-2 messenger Gamma = {S^2 kernel, S^2 C_1} (adjacent).
  gamma_1 = C_1^3   (m_i=2, n_i=1) -> needs S^2C_1 relevant (kernel CANNOT reach it)
  gamma_2 = C_2^2   (m_i=2, n_i=0) -> needs kernel relevant
  gamma_3 = C_3^2   (m_i=2, n_i=0) -> needs kernel relevant
Graph: chain 1-2-3-4-5-6 plus branch 2-3 with the SC edge:
  e1=(1,2) S^2 (V=4)      kernel edge
  e2=(2,3) S^2C_1 (V=5)   SC edge, branches off e1
  e3=(2,4) S^2            kernel-mode branch (vertex 2,4 are S^2 so the
                          kernel's monotone path stays in S^2 — a C-mode
                          intermediate would make the step C1^2 -> C2^2 an
                          overlapping/direction change, forbidden by the
                          2026-08-11 relevance monotonicity fix)
  e4=(4,5) C_2^2          gamma_2 edge
  e5=(4,6) C_3^2          gamma_3 edge
  e6=(3,7) C_1^3          gamma_1 edge (only reachable through e2)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import region_checker as rc
from primitives import marginal_softer

H = (0, 0, 0)
S2 = (2, 0, 0)
S2C1 = (2, 1, 1)
C13 = (0, 3, 1)
C22 = (0, 2, 2)
C32 = (0, 2, 3)

# vertices 2 and 4 are S^2 so the kernel (S^2) reaches C_2^2/C_3^2 through
# a monotone non-softer S^2 path; the SC branch (2,3) is the only route to C_1^3
verts = {1: H, 2: S2, 3: C13, 4: S2, 5: H, 6: H, 7: H}
edges = [
    (1, 2, S2), (2, 3, S2C1), (2, 4, S2),
    (4, 5, C22), (4, 6, C32), (3, 7, C13),
]

kernel = {'mode': S2, 'V': {1}, 'E': {(1, 2)}}
sc = {'mode': S2C1, 'V': {2, 3}, 'E': {(2, 3)}}
g1 = {'mode': C13, 'V': {7}, 'E': {(3, 7)}}
g2 = {'mode': C22, 'V': {5}, 'E': {(4, 5)}}
g3 = {'mode': C32, 'V': {6}, 'E': {(4, 6)}}
all_comps = [kernel, sc, g1, g2, g3]
attach = {}
extmode = {}

# sanity: the relevance facts the test depends on
assert marginal_softer(S2C1, C13), "S^2C_1 should be marginally softer than C_1^3"
assert not marginal_softer(S2, C13), "S^2 should NOT be marginally softer than C_1^3"
assert rc.relevant(sc, g1, verts, edges, all_comps), "S^2C_1 should reach C_1^3"
assert not rc.relevant(kernel, g1, verts, edges, all_comps), "kernel should NOT reach C_1^3"
assert rc.relevant(kernel, g2, verts, edges, all_comps), "kernel should reach C_2^2"
assert rc.relevant(kernel, g3, verts, edges, all_comps), "kernel should reach C_3^2"
print('precondition checks passed')

res = rc.find_messenger(kernel, all_comps, verts, edges, attach, extmode)
assert res is not None, "messenger should exist (3 directions: 1,2,3)"
G, special, _kb = res  # find_messenger returns (G, has_special, kernel_blocks) since 2026-09-07
assert special is False, "no attached S^m C_i^{n'} external — type-1 messenger only"
gids = {id(c) for c in G}
assert id(kernel) in gids and id(sc) in gids, "Gamma should contain kernel + S^2C_1"
print(f'PASS: find_messenger returned Gamma of {len(G)} members: '
      f'{sorted((m[0], m[1], m[2]) for m in [c["mode"] for c in G])}')
print('PASS: direction 1 collected via S^2C_1 member (new rule); 2,3 via kernel.')
print()
print('Now the negative control: kernel-only messenger (no SC member) should NOT')
print('collect C_1^3, so with only 2 kernel-relevant directions it must fail:')
kernel2 = {'mode': S2, 'V': {1}, 'E': {(1, 2)}}
all2 = [kernel2, g1, g2, g3]
res2 = rc.find_messenger(kernel2, all2, verts, edges, attach, extmode)
assert res2 is None, "with only 2 directions reachable, no messenger should exist"
print('PASS: kernel-only candidate correctly rejected (only 2 directions).')
print()
print('ALL MESSENGER SC TESTS PASSED')
