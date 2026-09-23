#!/usr/bin/env python3
"""indep_loops_demo.py — demonstrate indep_loops on wide-angle examples.

For each example: run the FRI region pipeline (single layered
enumerator, C/H in layer 0), then for every region compute the independent
loop momenta per mode and check Σ r_X = L.

Usage:
    python3 indep_loops_demo.py 4pt3loop
    python3 indep_loops_demo.py hypercrown k1
    python3 indep_loops_demo.py            # all built-ins
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from read_graph import mode_str
from skeleton import run as skeleton_run
import indep_loops as il

H = (0, 0, 0)

CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


@case
def fourpt3loop():
    """paper §7.1: 8 vertices / 10 edges, p1=C1, p2=C2^2, p3=C3^inf,
    p4=C4^inf — 81 regions (16 C/H + 65 other)."""
    verts = [1, 2, 3, 4, 5, 6, 7, 8]
    edges = [(1, 5), (1, 8), (2, 5), (2, 7), (3, 6), (3, 8), (4, 6), (4, 7),
             (5, 6), (7, 8)]
    ext_attach = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4}
    ext_mode = {'p1': (0, 1, 1), 'p2': (0, 2, 2), 'p3': (0, 100, 3),
                'p4': (0, 100, 4)}
    return verts, edges, ext_attach, ext_mode, '4pt3loop (paper §7.1, 81 regions)'


@case
def hypercrown_k1():
    """12-propagator graph; k1: p1^2~t1, p2^2~t1, p3^2=0, p4^2=0 — 71 regions."""
    verts = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    edges = [(1, 5), (1, 7), (2, 5), (2, 6), (3, 5), (3, 8), (4, 5), (4, 9),
             (6, 7), (7, 8), (8, 9), (9, 6)]
    ext_attach = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4}
    ext_mode = {'p1': (0, 1, 1), 'p2': (0, 1, 2), 'p3': (0, 100, 3),
                'p4': (0, 100, 4)}
    return verts, edges, ext_attach, ext_mode, 'hypercrown k1 (71 regions)'


@case
def mtest1_k0():
    """messenger testbed MTest1 k0: 5 edges / 4 vertices,
    p1=C1^2@1, p2=C2^3@2, l1=SC3^inf@3, q1=H@4 — type-2 messenger
    (m_i=0) special case activated; 16 regions."""
    verts = [1, 2, 3, 4]
    edges = [(1, 4), (2, 4), (3, 4), (1, 3), (2, 3)]
    ext_attach = {'p1': 1, 'p2': 2, 'l1': 3, 'q1': 4}
    ext_mode = {'p1': (0, 2, 1), 'p2': (0, 3, 2),
                'l1': (1, 100, 3), 'q1': (0, 0, 0)}
    return verts, edges, ext_attach, ext_mode, 'MTest1 k0 (16 regions, type-2 messenger)'


@case
def mtest1_k1():
    """messenger testbed MTest1 k1: p1=C1@1, p2=C2^2@2,
    l1=SC3^inf@3, q1=H@4 — control: special case not activated; 12 regions."""
    verts = [1, 2, 3, 4]
    edges = [(1, 4), (2, 4), (3, 4), (1, 3), (2, 3)]
    ext_attach = {'p1': 1, 'p2': 2, 'l1': 3, 'q1': 4}
    ext_mode = {'p1': (0, 1, 1), 'p2': (0, 2, 2),
                'l1': (1, 100, 3), 'q1': (0, 0, 0)}
    return verts, edges, ext_attach, ext_mode, 'MTest1 k1 (12 regions, control)'


def regions_of(verts, edges, ext_attach, ext_mode):
    regs = {}
    rlist, _nc, _dt = skeleton_run(verts, edges, ext_attach, ext_mode,
                                   verbose=False, overlap_strong=True)
    for vm, em in rlist:
        key = (tuple(tuple(m) for m in em))
        regs.setdefault(key, (vm, em))
    return list(regs.values())


def fmt_mode(m):
    return mode_str(m) if m is not None else 'H'


def show_region(i, edges, vm, em, results, total, L):
    print(f'--- region {i}: em = {[fmt_mode(m) for m in em]}')
    for r in results:
        X = r['mode']
        print(f'    mode {fmt_mode(X)}: {r["rank"]} independent '
              f'loop momentum{"s" if r["rank"] != 1 else ""}')
        for b in r['blocks']:
            if not b['basis']:
                continue
            desc = ', '.join(f'#{j} {edges[j]}' for j in b['basis'])
            print(f'      basis: {desc}   '
                  f'(block verts {b["verts"]})')
    print(f'    Σ rank = {total}  vs  L = {L}  '
          f'{"✓" if total == L else "✗ MISMATCH"}')


def run_case(name):
    verts, edges, ext_attach, ext_mode, note = CASES[name]()
    print(f'== {name}: {note}')
    t0 = time.time()
    regs = regions_of(verts, edges, ext_attach, ext_mode)
    dt = time.time() - t0
    print(f'   {len(regs)} regions ({dt:.1f}s); L = '
          f'{len(edges) - len(verts) + 1}')
    bad = 0
    for i, (vm, em) in enumerate(regs, 1):
        results, total, L = il.indep_loops(edges, em, vm)
        if total != L:
            bad += 1
            print(f'   !! region {i}: Σ rank {total} != L {L}')
    print(f'   Σ r_X = L check: {len(regs) - bad}/{len(regs)} regions OK')
    # show representative regions: first with SC, first with S, first with
    # S^2, first pure C/H
    wanted = {}
    for i, (vm, em) in enumerate(regs, 1):
        if i in wanted.values():
            continue
        has_sc = any(m is not None and m[0] >= 1 and m[1] >= 1 for m in em)
        has_s = any(m is not None and m[0] == 1 and m[1] == 0 for m in em)
        has_s2 = any(m is not None and m[0] >= 2 for m in em)
        pure_ch = all(m is None or m == (0, 0, 0) or (m[0] == 0 and m[1] >= 1)
                      for m in em)
        if 'SC' not in wanted and has_sc:
            wanted['SC'] = i
        if 'S' not in wanted and has_s and 'SC' not in em and 'S' not in wanted:
            wanted['S'] = i
        if 'S^2' not in wanted and has_s2:
            wanted['S^2'] = i
        if 'C/H' not in wanted and pure_ch:
            wanted['C/H'] = i
    for label, i in wanted.items():
        vm, em = regs[i - 1]
        results, total, L = il.indep_loops(edges, em, vm)
        print(f'  [representative: {label}]')
        show_region(i, edges, vm, em, results, total, L)
    print()


def main():
    # quick self-test of forced-line feasibility on 4pt3loop region 17
    # (located by content: region order changed when C/H merged into
    # layer 0 on 2026-09-09 — previously construct-C/H regions came first)
    verts, edges, ext_attach, ext_mode, _ = CASES['fourpt3loop']()
    regs = regions_of(verts, edges, ext_attach, ext_mode)
    idx = next(i for i, (vm, em) in enumerate(regs)
               if any(m is not None and m[0] == 1 and m[1] == 1
                      for m in em))
    vm, em = regs[idx]          # SC2^1 self-loop case
    print('forced-feasibility self-test (region %d, em ='
          % (idx + 1), [fmt_mode(m) for m in em], ')')
    for F, expect in (([2], True),      # SC2^1 self-loop: always in basis
                      ([0], True),      # C1 line in the C1 block: extendable
                      ([3], False),     # C2^2 line: C2^2 block is a tree
                      ([0, 8], False),  # both C1 lines of v5 deleted: v5
                                        # left uncovered by the C1 tree
                      ([8, 9], True),   # both C1 basis lines forced
                      ([0, 1, 8, 9], False)):  # over-force: block tree broken
        ok, fails = il.forced_feasible(edges, em, vm, F)
        print(f'   F={F}: feasible={ok} (expect {expect})'
              + ('' if ok == expect else '   <-- MISMATCH!'))
    names = sys.argv[1:] or list(CASES)
    for n in names:
        if n in CASES:
            run_case(n)
        else:
            print(f'unknown case {n}; have {list(CASES)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
