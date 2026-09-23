#!/usr/bin/env python3
"""unit_test_massive.py — massive-prescription unit tests (2026-08-15).

Prescription: with all internal masses either 0 or O(1) and the small
variables still the virtualities, the regions are precisely those of the
massless graph, constrained by big-massive propagators staying HARD.

Tests:
  1. massive_h_ok unit behaviour (empty / H / non-H);
  2. Crown k0 with a single massive line (1,5): FRI gives 20 regions —
     the massless 73 restricted to configurations where (1,5) is H.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import region_checker as rc

PASS = 0


def check(cond, msg):
    global PASS
    assert cond, msg
    PASS += 1
    print(f'  PASS: {msg}')


# ---- 1. massive_h_ok unit behaviour ----
H = (0, 0, 0)
m15 = frozenset({1, 5})
check(rc.massive_h_ok([], [], frozenset()) is True, 'empty: True')
check(rc.massive_h_ok([(1, 5)], [H], frozenset({m15})) is True,
      'massive line in H: True')
check(rc.massive_h_ok([(1, 5)], [(0, 1, 1)], frozenset({m15})) is False,
      'massive line in C: False')
check(rc.massive_h_ok([(1, 5)], [(1, 0, 0)], frozenset({m15})) is False,
      'massive line in S: False')
check(rc.massive_h_ok([(1, 5)], [(1, 1, 3)], frozenset({m15})) is False,
      'massive line in SC: False')
check(rc.massive_h_ok([(1, 5), (2, 5)], [H, (0, 1, 2)],
                      frozenset({m15})) is True,
      'massless line may be soft: True')

# ---- 2. Crown k0 with one massive line (1,5): 20 regions ----
import skeleton as SG

verts = [1, 2, 3, 4, 5, 6]
edges = [(1, 5), (2, 5), (3, 5), (4, 5), (1, 6), (2, 6), (3, 6), (4, 6)]
attach = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4}
modes = {'p1': (0, 1, 1), 'p2': (0, 1, 2), 'p3': (0, 1, 3), 'p4': (0, 1, 4)}
massive = {m15}

regs, _nc, _dt = SG.run(verts, edges, attach, modes, verbose=False,
                        overlap_strong=True, massive=massive)
total = len(regs)
check(total == 20, f'Crown k0 + massive (1,5): 20 regions (got {total})')

# every region must have edge (1,5) hard
edges_t = [tuple(sorted(e)) for e in edges]
bad = 0
for vm, em in regs:
    idx = edges_t.index((1, 5))
    if em[idx] != H:
        bad += 1
check(bad == 0, 'all regions keep massive line (1,5) hard')

print(f'\nALL MASSIVE TESTS PASSED ({PASS} checks)')
