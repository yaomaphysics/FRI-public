#!/usr/bin/env python3
"""fri_demo.py — built-in demonstrations of the Facet Region Interpreter
(FRI tool, 2026-08-15).

Each demo runs the FULL pipeline live (no cached results) on a built-in
graph and explains what you are seeing:

  demo 1: 4-point 3-loop (paper §7.1)      — a textbook case: 81 regions,
           grouped by type; watch the layer table.
  demo 2: HyperCrown k1                     — a graph where naive enumeration
           explodes; the layered construction + shared-vertex prefilter
           keeps it fast and exact (pySecDec 71 = 71).
  demo 3: messenger testbed (MTest1)        — IR-compatibility fixed point:
           soft components confirmed via cond1/cond2/cond3 in rounds.
  demo 4: 5-loop soft-emission K33A5 (k6)   — the newest kinematics family
           (l1 = pure soft S), all regions with scaling vectors.

Usage:
    python3 fri_demo.py [1|2|3|4]     # run one demo (default: all)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'wide_angle'))
from read_graph import mode_str
from truncation_check import kappa_of, run_layered
from facet_regions_interactive import (group_by_type,
                                  type_order, sc_short)


def scaling_of(md):
    return 0 if md is None else -(2 * md[0] + md[1])


def fmt_region(vm, em, edges):
    """Compact one-line region: mode counts + scaling vector."""
    edges_t = [tuple(sorted(e, key=str)) for e in edges]
    from collections import Counter
    cnt = Counter(mode_str(md) for md in em if md is not None)
    parts = ', '.join(f'{m}×{n}' for m, n in sorted(cnt.items()))
    vec = [scaling_of(md) for md in em]
    return f'  modes: {parts or "all H"}   vector=({", ".join(map(str, vec))}, 1)'


def run_graph(title, blurb, verts, edges, ext_attach, ext_mode, max_show=12):
    print('=' * 70)
    print(title)
    print('-' * 70)
    print(blurb)
    print(f'  vertices: {sorted(verts, key=str)}')
    print(f'  edges ({len(edges)}): '
          f'{[tuple(sorted(e, key=str)) for e in edges]}')
    print(f'  externals: '
          f'{", ".join(f"{n}@{ext_attach[n]}={mode_str(ext_mode[n])}" for n in ext_mode)}')
    kappa = kappa_of(ext_mode)
    print(f'  kappa (softest S-power) = {kappa}')
    print()

    t0 = time.time()
    layers = run_layered(verts, edges, ext_attach, ext_mode, brief=True,
                         use_compression=True)
    t_all = time.time() - t0
    allr = [r for L in layers for r in L['regions']]

    print(f'  regions     : {len(allr)} total (layered '
          f'{[f"k={L['k']}:{L['ir']}" for L in layers]}; {t_all:.1f}s)')
    print(f'  TOTAL       : {len(allr)} regions')
    print()

    groups = group_by_type(allr, kappa)
    order = type_order(kappa)
    labels = [sc_short(*mn) for mn in order] + ['C/H']
    shown = 0
    for lab in labels:
        if lab not in groups:
            continue
        regs = groups[lab]
        print(f'  [{lab}] {len(regs)} regions:')
        for vm, em in regs[:max_show]:
            print(fmt_region(vm, em, edges))
        if len(regs) > max_show:
            print(f'      … and {len(regs) - max_show} more')
        shown += len(regs)
    print()


# ---------------------------------------------------------------- demos
def demo1():
    from truncation_check import case_4pt3loop
    v, e, ea, em = case_4pt3loop()
    run_graph(
        'Demo 1 — 4-point 3-loop (paper §7.1 example)',
        'A textbook 8-vertex / 10-edge graph with p1=C1, p2=C2^2, p3=C3^inf, '
        'p4=C4^inf.  Known result (paper + pySecDec): 81 regions = '
        '16 C/H + 65 other (S^2, SC, S types); layer 0 holds the 31 '
        'no-shared-vertex regions (C/H merged into layer 0, 2026-09-09). ',
        v, e, ea, em)


def demo2():
    from truncation_check import case_hypercrown
    v, e, ea, em = case_hypercrown('k1')
    run_graph(
        'Demo 2 — HyperCrown k1',
        'A 9-vertex ring graph (central vertex 5 attached to all 4 externals, '
        'ring 6-7-8-9).  Naive enumeration of cut combinations is huge, but '
        'the layered construction + shared-vertex prefilter + mode '
        'compression keeps it exact: pySecDec 71 = 71 regions.',
        v, e, ea, em, max_show=8)


def demo3():
    # MTest1: 4 vertices / 5 edges, messenger type-2 testbed
    verts = [1, 2, 3, 4]
    edges = [(1, 4), (2, 4), (3, 4), (1, 3), (2, 3)]
    ext_attach = {'p1': 1, 'p2': 2, 'l1': 3, 'q1': 4}
    ext_mode = {'p1': (0, 2, 1), 'p2': (0, 3, 2), 'l1': (1, 100, 3),
                'q1': (0, 0, 0)}
    run_graph(
        'Demo 3 — Messenger testbed (MTest1)',
        'A 2-loop graph with p1=C1^2, p2=C2^3, l1=SC3^inf, q1=H.  The SC3 '
        'soft component needs the messenger mechanism (cond2) to be '
        'confirmed in the IR-compat fixed point.  pySecDec: 16 regions.',
        verts, edges, ext_attach, ext_mode)


def demo4():
    # K33A5 k6 from the soft-emission family (l1 = S pure soft)
    verts = [1, 2, 3, 4, 5, 6]
    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 5), (3, 6),
             (4, 5), (4, 6), (5, 6)]
    ext_attach = {'p1': 1, 'p2': 2, 'p3': 3, 'l1': 4, 'q1': 5}
    ext_mode = {'p1': (0, 1, 1), 'p2': (0, 2, 2), 'p3': (0, 100, 3),
                'l1': (1, 0, 0), 'q1': (0, 0, 0)}
    run_graph(
        'Demo 4 — 5-loop soft-emission (K33A5, k6)',
        'A nonplanar 5-loop graph with p1=C1, p2=C2^2, p3=C3^inf, '
        'l1 = S (PURE soft), q1 = H — the newest kinematics family '
        '(2026-08-14).  pySecDec: 45 regions.  Scaling vectors show the '
        'full soft/collinear structure.',
        verts, edges, ext_attach, ext_mode)


def main():
    demos = {'1': demo1, '2': demo2, '3': demo3, '4': demo4}
    sel = sys.argv[1] if len(sys.argv) > 1 else None
    if sel:
        if sel not in demos:
            print(f'unknown demo {sel!r}; choose 1..4')
            sys.exit(1)
        demos[sel]()
    else:
        for d in demos.values():
            d()
    print('=' * 70)
    print('All demos done.  Run a single demo:  python3 fri_demo.py <1..4>')


if __name__ == '__main__':
    main()
