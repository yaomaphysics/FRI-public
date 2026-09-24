#!/usr/bin/env python3
"""Interactive Regge-limit FRI region enumerator.

Usage:
    python3 fri_interactive.py

At startup you pick one of the 6 Regge 2->2 kinematics (k0..k5, default
k1).  The kinematics sets the external-momentum modes (ext_mode) fed into
the FRI pipeline (cuts -> overlay -> Glauber adjustment -> subgraph
requirements -> IR compatibility); the definitions are shared with
regge_graphs.py.  Enumeration runs on the pruned skeleton enumerator
(skeleton.py); k1 uses its all-lightlike engine.

  k0: p_i^2 ~ lambda for all i                -> ext modes C13/C24
  k1: all p_i lightlike (original case)       -> ext modes C1^oo C13, ...
  k2: p1^2 ~ lambda, others lightlike
  k3: p1^2, p3^2 ~ lambda
  k4: p1^2, p3^2 ~ lambda^2
  k5: p1^2, p2^2 ~ lambda, p3^2 = p4^2 = 0

Type an edge list to enumerate (external momenta p1..p4 attach at
vertices 1, 2, 3, 4).  Commands:  'kin kX' switches kinematics on the
fly, 'q' quits.  After enumeration a menu offers:
    1) Inspect specific regions   — per-mode subgraphs + loop numbers
       (Σ r_X vs L), then optionally a concrete independent-loop-momentum
       basis (default or forced lines) and the physical line momenta
       (k1..kL + externals, momentum-conservation check),
    2) Lee-Pomeransky parametric representation (v_e = -V, edge order),
    3) Classification by characteristic (softest) mode.
    4) Visualization: one PDF atlas (default) or PNG figures with the
       per-mode colour scheme (region_plot.py; saved under fri_out/regions_*/).
Semihard loops use the γ_sH ∪ γ_G rule (2026-09-01).  Results are
saved to fri_out/<timestamp>.txt.

Edge-list input formats (all accepted):
    "1-7,1-12,2-6,...,11-12"
    "[1,7],[1,12],[2,6],..."
    "1 7; 1 12; 2 6"
"""
import sys, os, time, ast, re

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from regge_core import to_scaling, mode_components
from skeleton import skel_regions, skel_regions_k1   # pruned skeleton enumerators
from regge_modes import to_mode          # mode strings are plain literals
                                         # ('C13', 'sH', ...) everywhere
from regge_indep_loops import show_basis     # indep loops + edge momenta
# The 6 Regge kinematics (ext_mode + momentum-invariant scalings + notes),
# shared with the graph/kinematics library.
from regge_graphs import KIN
from collections import defaultdict

KIN_CHOICES = ['k0', 'k1', 'k2', 'k3', 'k4', 'k5']


def parse_edges(s):
    """Parse user input into a list of (int, int) edges."""
    s = s.strip()
    if not s:
        return []
    if s.startswith('['):
        lst = ast.literal_eval(s)
        return [tuple(int(x) for x in e) for e in lst]
    out = []
    for part in re.split(r'[;,]\s*', s):
        part = part.strip()
        if not part:
            continue
        m = re.match(r'^(\d+)\s*[-,\s]\s*(\d+)$', part)
        if not m:
            raise ValueError(f"cannot parse edge: '{part}'")
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def fmt_scaling(sc):
    return '(' + ', '.join(str(x) for x in sc) + ')'


def parse_region_select(s, n):
    """'3, 8--10' -> [3, 4, 8, 10] (1-based).  Supports ',' separators and
    '--' / '-' ranges.  Returns None if any number is out of range."""
    out = set()
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '--' in part:
            a, b = part.split('--', 1)
        elif '-' in part:
            a, b = part.split('-', 1)
        else:
            out.add(int(part))
            continue
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        out.update(range(a, b + 1))
    bad = sorted(i for i in out if not (1 <= i <= n))
    if bad:
        print(f'  ! region numbers out of range (1..{n}): {bad}')
        return None
    return sorted(out)


def _components(verts, edges):
    """Number of connected components of the (multi)graph."""
    adj = {v: [] for v in verts}
    for (a, b) in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen = set()
    n = 0
    for v in verts:
        if v in seen:
            continue
        n += 1
        stack = [v]
        while stack:
            x = stack.pop()
            for w in adj.get(x, ()):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
    return n


def show_mode_subgraphs(edges, em, vm):
    """Per-mode subgraphs {{V_X}, {E_X}} with loop number r_X, plus the
    sum check vs L = |E| - |V| + 1.

    r_X uses the standard contracted-subgraph rule (1VI blocks of gamma~_X
    with the aux vertex; r = |E| - |V| + 1 per block) for all lattice
    modes.  sH is an OVERLAY product (not a lattice mode): its edges
    collapse onto the aux vertex and the standard rule overcounts
    (necklace: 2 parallel sH edges -> two contracted self-loops -> r = 2,
    but they are ONE bubble, r = 1).  Fix (2026-09-01): the semihard
    loops are the loops of the subgraph gamma_{sH} U gamma_{G} computed on
    the ORIGINAL graph (no aux).  The G subgraph is the necklace string
    (acyclic), so it has no loops by itself and the union's loops are
    exactly the semihard ones.
    """
    verts = sorted({v for e in edges for v in e})
    L = len(edges) - len(verts) + 1
    modes = sorted(set(em) | set(vm.values()))
    tot = 0
    for mode in modes:
        E_idx = [i for i, m in enumerate(em) if m == mode]
        V = sorted(v for v in verts if vm.get(v) == mode)
        Es = '{' + ', '.join(f'{i}:{edges[i]}' for i in E_idx) + '}'
        Vs = '{' + ', '.join(str(v) for v in V) + '}'
        if mode == 'sH':
            # semihard loops = rank of gamma_{sH} U gamma_{G} (plain
            # subgraph of the original graph, no aux contraction)
            idx = [i for i, m in enumerate(em) if m in ('sH', 'G')]
            sub_edges = [edges[i] for i in idx]
            sub_verts = sorted({v for e in sub_edges for v in e})
            r = (len(sub_edges) - len(sub_verts)
                 + _components(sub_verts, sub_edges))
            note = '  [semihard loops = loops of gamma_sH U gamma_G]'
        else:
            blocks = mode_components(mode, vm, em, edges, verts)
            r = sum(len(idxs) - len(bv) + 1 for (bv, be, idxs) in blocks)
            note = ''
        tot += r
        print(f'  mode {mode}: {{vertices: {Vs}, edges: {Es}}}   '
              f'loop number = {r}{note}')
    if tot == L:
        flag = '✓'
    else:
        flag = '✗ MISMATCH'
    print(f'  sum r_X = {tot}  vs  L = {L}  {flag}')


def parse_forced_lines(edges, line):
    """'[2,5],[7,8]' -> sorted edge indices (order-free endpoints).
    Returns None if nothing parses or an edge is unknown.  NB: for
    parallel edges, a (a,b) pair forces ALL copies (wide-angle behavior)."""
    pairs = re.findall(r'[\[(]\s*(\d+)\s*,\s*(\d+)\s*[\])]', line)
    if not pairs:
        print('  ! no (x,y) pairs found — use e.g. (2,5),(7,8) '
              '(square brackets ok too)')
        return None
    idx = {}
    for i, (a, b) in enumerate(edges):
        idx.setdefault((a, b), []).append(i)
        idx.setdefault((b, a), []).append(i)
    F, unknown = [], []
    for a, b in pairs:
        got = idx.get((int(a), int(b)))
        if not got:
            unknown.append((int(a), int(b)))
        else:
            F.extend(got)
    if unknown:
        print('  ! unknown edges: ' +
              ', '.join(f'[{a},{b}]' for a, b in unknown))
        return None
    return sorted(set(F))


def _softness_key(mode):
    """Softness order for classification, softest first: lattice modes
    with m >= 1 by virtuality V desc (tie: m desc, n desc, name), then sH
    (chain rank 1), then G (chain rank 2).  Chain (regge_modes):
    H >- G >- sH >- every family mode."""
    if mode == 'sH':
        return (1, 0, 0, 0, mode)
    if mode == 'G':
        return (2, 0, 0, 0, mode)
    try:
        M = to_mode(mode)
        return (0, -M.V, -M.m, -M.n, mode)
    except Exception:
        return (0, -99, 0, 0, mode)


def classify(vm, em):
    """Softest characteristic mode of a region (exclusion-style): the
    softest lattice mode with m >= 1 present in em/vm; if none, sH; if
    none, G; else 'C/H'."""
    cands = set(em) | set(vm.values())
    soft = [m for m in cands
            if m not in ('sH', 'G') and to_mode(m).m >= 1]
    if soft:
        return min(soft, key=_softness_key)
    if 'sH' in cands:
        return 'sH'
    if 'G' in cands:
        return 'G' 
    return 'C/H'


def show_parametric(regs):
    """Option 2: Lee-Pomeransky parametric representation per region:
    x_e ~ λ^{v_e} with v_e = -V(𝒳(e)), edge order, trailing 1 = λ power
    of the single expansion scale (pySecDec style, same as the verified
    region files)."""
    print('  Lee-Pomeransky parametric representation '
          '(x_e ~ λ^{v_e}, v_e = -V, edge order):')
    for i, (cut13, cut24, cut1, cut3, cut2, cut4, vm, em) in enumerate(regs, 1):
        print(f'    R{i}: v = {to_scaling(em)}')


def show_classify(regs, edges):
    """Option 3: classify regions by their characteristic (softest) mode,
    grouped by type (softest first)."""
    groups = defaultdict(list)
    for r in regs:
        cut13, cut24, cut1, cut3, cut2, cut4, vm, em = r
        groups[classify(vm, em)].append(r)
    order = sorted(groups, key=lambda lab:
                   (99,) if lab == 'C/H' else _softness_key(lab))
    total = 0
    for lab in order:
        g = groups[lab]
        total += len(g)
        print()
        print(f'There are {len(g)} {lab} type regions:')
        for i, (cut13, cut24, cut1, cut3, cut2, cut4, vm, em) in enumerate(g, 1):
            print(f'  --- {lab} region {i} ---')
            print(f'    em = {em}')
            print(f'    scaling = {to_scaling(em)}')
    print(f'\nTOTAL: {total} regions')


def inspect_regions(edges, regs, ext_attach):
    """Option 1: pick regions -> per-mode subgraphs with loop numbers,
    then optionally a concrete basis / forced lines / line momenta."""
    n = len(regs)
    line = input('Region numbers (e.g. "3, 8--10" represents regions '
                 '3, 8, 9, and 10; empty = all) > ').strip()
    sel = parse_region_select(line, n) if line else list(range(1, n + 1))
    if sel is None:
        sel = list(range(1, n + 1))
    for i in sel:
        cut13, cut24, cut1, cut3, cut2, cut4, vm, em = regs[i - 1]
        print(f'  --- region {i}:')
        show_mode_subgraphs(edges, em, vm)
    if input('Select a set of line momenta as independent loop '
             'momenta? (y/n) [n] > ').strip().lower() == 'y':
        asked = False
        while True:
            prompt = ('Force lines into the basis? ((x,y) pairs; '
                      'empty = show default basis) > ' if not asked
                      else 'Force more lines? ((x,y) pairs; '
                           'empty = done) > ')
            line = input(prompt).strip()
            if not line:
                if not asked:
                    for i in sel:
                        cut13, cut24, cut1, cut3, cut2, cut4, vm, em = regs[i - 1]
                        print(f'  --- region {i}:')
                        show_basis(edges, em, vm, ext_attach=ext_attach)
                break
            asked = True
            F = parse_forced_lines(edges, line)
            if F is None:
                continue
            for i in sel:
                cut13, cut24, cut1, cut3, cut2, cut4, vm, em = regs[i - 1]
                print(f'  --- region {i}:')
                show_basis(edges, em, vm, F, ext_attach)


def visualize_regions(edges, regs, ext_mode):
    """Option 4: visualize the selected regions — PDF atlas or PNG files."""
    n = len(regs)
    line = input('Region numbers (e.g. "3, 8--10" represents regions '
                 '3, 8, 9, and 10; empty = all) > ').strip()
    sel = parse_region_select(line, n) if line else list(range(1, n + 1))
    if sel is None:
        return
    fmt = input('Output: [a] single PDF atlas (default) / '
                '[p] individual PNG files > ').strip().lower()
    try:
        import region_plot
    except Exception as e:
        print(f'  ! visualization module unavailable: {e}')
        return
    outdir = os.path.join(BASE, 'fri_out',
                          'regions_' + time.strftime('%Y%m%d-%H%M%S'))
    if fmt in ('p', 'png', 'files'):
        print(f'  rendering {len(sel)} region figure(s) via wolframscript ...')
        try:
            paths = region_plot.render_regions(
                edges, sorted({v for e in edges for v in e}),
                [(i, regs[i - 1]) for i in sel],
                ext_mode=ext_mode, outdir=outdir)
        except Exception as e:
            print(f'  ! region rendering failed: {e}')
            return
        print(f'  saved {len(paths)} region PNG(s) to: {outdir}')
    else:
        print(f'  building the PDF atlas for {len(sel)} region(s) '
              f'via wolframscript ...')
        try:
            path = region_plot.render_atlas(
                edges, sorted({v for e in edges for v in e}),
                [(i, regs[i - 1]) for i in sel],
                ext_mode=ext_mode, outdir=outdir)
        except Exception as e:
            print(f'  ! atlas rendering failed: {e}')
            return
        print(f'  saved atlas PDF to: {path}')


def print_kin_menu():
    print('available Regge 2->2 kinematics:')
    for k in KIN_CHOICES:
        em = KIN[k]['ext_mode']
        modes = ', '.join(f'{n}={em[n]}' for n in ('p1', 'p2', 'p3', 'p4'))
        print(f'  {k}: {KIN[k]["note"]}')
        print(f'      ext modes: {modes}')


def choose_kinematics():
    """Ask for a kinematics label at startup; default k1."""
    print_kin_menu()
    while True:
        try:
            s = input('kinematics [default k1]> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            return 'k1'
        if not s:
            return 'k1'
        if s in KIN_CHOICES:
            return s
        print(f'  [error] unknown kinematics "{s}" — choose from '
              f'{", ".join(KIN_CHOICES)}')


def run_graph(edges_raw, kin_name):
    kin = KIN[kin_name]
    edges = [tuple(e) for e in edges_raw]
    verts = sorted({v for e in edges for v in e})
    missing = [v for v in (1, 2, 3, 4) if v not in verts]
    if missing:
        print(f'  [warning] external-leg vertices {missing} not in the '
              f'graph!  Regge 2->2 needs vertices 1,2,3,4.')
        return
    ext_attach = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4}
    ext_mode = kin['ext_mode']
    L = len(edges) - len(verts) + 1
    print(f'  kinematics: {kin_name} ({kin["note"]})')
    print(f'  graph: {len(edges)} edges, {len(verts)} vertices, '
          f'L = {L} loops')
    print(f'  edges: {edges}')
    if L >= 5:
        print('  [warning] L >= 5 may be slow')
    t0 = time.time()
    try:
        if kin_name == 'k1':
            regs, _cnt, _depth = skel_regions_k1(edges, verts, ext_attach,
                                                 ext_mode)
        else:
            regs, _cnt, _depth = skel_regions(edges, verts, ext_attach,
                                              ext_mode)
    except (Exception, SystemExit) as e:
        print(f'  [error] skeleton engine could not handle this input: {e}')
        return
    dt = time.time() - t0
    # deterministic display order (scaling, then em); same convention as the
    # 2->3 browser after its skeleton switch (2026-09-21).
    regs = sorted(regs, key=lambda r: (to_scaling(r[7]),
                                       tuple(str(m) for m in r[7])))
    print(f'  FRI regions: {len(regs)}  ({dt:.1f}s)')
    scal = set()
    for i, (cut13, cut24, cut1, cut3, cut2, cut4, vm, em) in enumerate(regs, 1):
        sc = to_scaling(em)
        scal.add(sc)
        ref = ''
        if cut1: ref += f'  C1C13={sorted(cut1)}'
        if cut3: ref += f'  C3C13={sorted(cut3)}'
        if cut2: ref += f'  C2C24={sorted(cut2)}'
        if cut4: ref += f'  C4C24={sorted(cut4)}'
        print(f'  R{i:3d}: scaling {fmt_scaling(sc)}')
        print(f'        cuts: C13={sorted(cut13)}  C24={sorted(cut24)}{ref}')
        print(f'        em: {em}')
    print(f'  unique scalings: {len(scal)}')
    while True:
        print('  Options:')
        print('    1) Inspect specific regions')
        print('    2) Show Lee-Pomeransky parametric representation')
        print('    3) Classify these regions based on their '
              'characteristic modes')
        print('    4) Visualize these regions (PDF atlas / PNGs)')
        opt = input('  (1/2/3/4; empty or q = done with this graph) > ')\
            .strip().lower()
        if opt in ('', 'q', 'quit'):
            break
        if opt == '1':
            inspect_regions(edges, regs, ext_attach)
        elif opt == '2':
            show_parametric(regs)
        elif opt == '3':
            show_classify(regs, edges)
        elif opt == '4':
            visualize_regions(edges, regs, ext_mode)
        else:
            print('  ! enter 1, 2, 3, 4, or q')
    # save
    fdir = os.path.join(BASE, 'fri_out')
    os.makedirs(fdir, exist_ok=True)
    fname = os.path.join(fdir, time.strftime('%Y%m%d-%H%M%S') + '.txt')
    with open(fname, 'w') as f:
        f.write(f'# Regge FRI regions - {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# kinematics: {kin_name} ({kin["note"]})\n')
        f.write(f'# edges: {edges}  ({len(edges)} edges, {len(verts)} '
                f'verts, L={L})\n')
        f.write(f'# FRI regions: {len(regs)} ({dt:.1f}s)\n\n')
        for i, (cut13, cut24, cut1, cut3, cut2, cut4, vm, em) in enumerate(regs, 1):
            ref = ''
            if cut1: ref += f'  C1C13={sorted(cut1)}'
            if cut3: ref += f'  C3C13={sorted(cut3)}'
            if cut2: ref += f'  C2C24={sorted(cut2)}'
            if cut4: ref += f'  C4C24={sorted(cut4)}'
            f.write(f'R{i:3d}: scaling {fmt_scaling(to_scaling(em))}\n')
            f.write(f'      cuts: C13={sorted(cut13)}  C24={sorted(cut24)}{ref}\n')
            f.write(f'      em: {em}\n')
            vmtxt = ', '.join(f'{v}: {vm[v]}' for v in sorted(vm))
            f.write(f'      vm: {{{vmtxt}}}\n')
    print(f'  saved to: {fname}')


def main():
    print('=' * 70)
    print('Regge-limit FRI region enumerator (6 kinematics, k0..k5)')
    print('external momenta: p1@1, p2@2, p3@3, p4@4')
    print('type an edge list, e.g. 1-7,1-12,2-6,...,11-12')
    print("commands: 'kin kX' switch kinematics, 'q' quit")
    print('=' * 70)
    kin_name = choose_kinematics()
    print(f'-> using kinematics {kin_name} '
          f'({KIN[kin_name]["note"]})')
    while True:
        try:
            s = input(f'\nedge list [{kin_name}]> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nbye!')
            break
        if not s:
            continue
        if s.lower() in ('q', 'quit', 'exit'):
            print('bye!')
            break
        low = s.lower()
        if low.startswith('kin '):
            k = low.split()[1]
            if k in KIN_CHOICES:
                kin_name = k
                print(f'-> kinematics switched to {kin_name} '
                      f'({KIN[kin_name]["note"]})')
            else:
                print(f'  [error] unknown kinematics "{k}" — choose from '
                      f'{", ".join(KIN_CHOICES)}')
            continue
        try:
            edges = parse_edges(s)
        except Exception as e:
            print(f'  [error] {e}')
            continue
        if not edges:
            print('  [error] empty input')
            continue
        t0 = time.time()
        run_graph(edges, kin_name)
        print(f'  (this graph took {time.time() - t0:.1f}s total)')


if __name__ == '__main__':
    main()
