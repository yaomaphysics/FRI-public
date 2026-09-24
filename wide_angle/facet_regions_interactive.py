#!/usr/bin/env python3
"""facet_regions_interactive.py — interactive facet-region browser.

Input: graph topology + external kinematics ONLY (the cut formalism is internal — the user never touches cuts).

Script:
  1. enumerates ALL regions of the graph (C/H family via nested cuts + compression on by default),
  2. lists them with numbers (edge-mode sequence per region, in input edge order), then offers a menu:
       1) Inspect specific regions  — pick a subset ("e.g., 3, 8--10" = regions 3, 4, 8, 10; empty = all);
           for every picked region, output the mode subgraphs:
           "X: {vertices: {...}, edges: {...}}   loop number = ...."
          Optionally, the user can select a set of line momenta as independent loop momenta (a basis) with optional forced lines.
       2) Show Lee-Pomeransky parametric representation — per region the scaling vector v_e (x_e ~ λ^{v_e} with λ the expansion parameter, v_e = -(2m+n), edge order),
       3) Classify these regions based on their characteristic modes — per region its softest-mode class + by-type counts,
       4) Visualize the selected regions — per-mode colours; can choose a single PDF atlas (default) or one PNG file per region.
  This loops until the user quits (empty or q).

Usage: python3 facet_regions_interactive.py
  Each input is one line of Python literal; empty line = built-in example, which is the example in Sec. 7.1 of arXiv:2601.22144.
  internal_lines = [[1,5],[1,8],[2,5],[2,7],[3,6],[3,8],[4,6],[4,7],[5,6],[7,8]]
  externals      = {'p1':[1,'C1'], 'p2':[2,'C2^2'], 'p3':[3,'C3^inf'], 'p4':[4,'C4^inf']}
  (81 regions in total)

Mode syntax: H / S / S^m / C_i / C_i^n / C_i^inf / C_i^\\infty / SC_i / SC_i^n / S^mC_i^n  (C_i without n means n=1; SC_i without n means n=1; i is the direction index from 1).
"""
import sys, re, os, time, warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_graph import (mode_str, INF, parse_mode, sc_short, type_order, group_by_type)
from primitives import scaling_of, spanning_tree
from indep_loops import indep_loops
from skeleton import run as skeleton_run, kappa_of
H = (0, 0, 0)


# One mode tuple -> display string ('∅' when an edge mode is missing).
def ms(md):
    return mode_str(md) if md is not None else '∅'

# ---------------------------------------------------------------- input helpers
# Prompt for one line of Python input; empty -> default.
def ask(label, default=None):
    print(f'{label}:' + (f'  (default: {default})' if default else ''))
    line = input('> ').strip()
    if not line and default: line = default
    try:
        return eval(line, {'inf': INF, '∞': INF, 'INF': INF})
    except Exception as e:
        print(f'  ! parse error: {e}')
        return ask(label, default)

# ---------------------------------------------------------------- enumeration
# enumerates all regions of the graphs; return list of (vm, em).
def enumerate_regions(verts, edges, ext_attach, ext_mode):
    regs = {}
    regions, _nc, _dt = skeleton_run(verts, edges, ext_attach, ext_mode, verbose=False, overlap_strong=True)
    for vm, em in regions:
        regs.setdefault(tuple(tuple(m) for m in em), (vm, em))
    return list(regs.values())

# Edge modes as one display string, in terms of the input edge order.
def em_sequence(em):
    return ', '.join(ms(m) for m in em)


# Format a vertex/edge set as '{a, b, c}' ('empty' for the empty set).
def _fmt_set(items):
    if not items:
        return 'empty'
    parts = []
    for x in items:
        if isinstance(x, tuple) and len(x) == 2:
            parts.append(f'[{x[0]},{x[1]}]')
        else:
            parts.append(str(x))
    return '{' + ', '.join(parts) + '}'


# Print a region as one line per mode: <mode> vertex = {...}; <mode> edge = {...}. Empty vertex/edge sets print 'empty'.
def print_region_modes(vm, em, edges, indent='    '):
    edges_t = [tuple(sorted(e, key=str)) for e in edges]
    v_by_mode = {}
    for v in sorted(vm, key=str):
        v_by_mode.setdefault(vm[v], []).append(v)
    e_by_mode = {}
    for (a, b), md in zip(edges_t, em):
        e_by_mode.setdefault(md, []).append((a, b))
    all_modes = list(v_by_mode) + [m for m in e_by_mode if m not in v_by_mode]
    for md in all_modes:
        s = mode_str(md)
        print(f'{indent}{s} vertex = {_fmt_set(v_by_mode.get(md, []))}; {s} edge = {_fmt_set(e_by_mode.get(md, []))}.')


# print_region_modes followed by the corresponding region vector.
def print_region_with_vector(vm, em, edges, indent='    '):
    print_region_modes(vm, em, edges, indent=indent)
    vec = [scaling_of(md) for md in em]
    print(f'{indent}scaling vector = ({', '.join(map(str, vec))}, 1)')


# ---------------------------------------------------------------- display
# Split a region into per-mode subgraphs {mode: (vertex set, edge list)}.
def mode_subgraphs(edges, em, vm):
    subs = {}
    for i, m in enumerate(em):
        m = m if m is not None else H
        subs.setdefault(m, [set(), []])[1].append(edges[i])
    for v, m in vm.items():
        m = m if m is not None else H
        subs.setdefault(m, [set(), []])[0].add(v)
    return subs

# Print per-mode subgraphs {{V_X}, {E_X}} with their independent loop numbers.
def show_mode_subgraphs(edges, em, vm):
    results, total, L = indep_loops(edges, em, vm)
    rank = {r['mode']: r['rank'] for r in results}
    for X in sorted(mode_subgraphs(edges, em, vm),
                    key=lambda m: (m[0], m[1], m[2])):
        V, E = mode_subgraphs(edges, em, vm)[X]
        Vs = '{' + ', '.join(str(v) for v in sorted(V)) + '}'
        Es = '{' + ', '.join(f'({u},{v})' for u, v in sorted(E)) + '}'
        print(f'  mode {ms(X)}: {{vertices: {Vs}, edges: {Es}}}   loop number = {rank.get(X, 0)}')
    print(f'  Σ loop numbers = {total}  vs  L = {L}  {"✓" if total == L else "✗ MISMATCH"}')


# Express line momenta as linear combinations of the loop and external momenta.
# More details: carrier lines are the chords (forced lines first in the k numbering,
# remaining carriers chosen freely); the other |V|-1 lines form a spanning tree on
# the ORIGINAL graph.  A carrier's loop momentum flows through every tree line on its
# fundamental cycle — a "self-loop" of a contracted mode subgraph is NOT inert: it is
# an ordinary line of the original graph and its momentum couples to the other lines.
# Lines are oriented (a,b) with a<b; external momenta enter at their attachment
# vertices (net-inflow convention: in - out = p_v).
# Returns (True, order, momenta, verts), momenta[ei] = {term: coeff} (order = carrier
# edge indices in k-numbering order), or (False, reason, None, None): |carriers| > L,
# or the complement of the carriers is disconnected (a forced line is a bridge).
def edge_momenta(edges, em, vm, carriers, ext_attach):
    verts = sorted({v for e in edges for v in e})
    E = len(edges)
    V = len(verts)
    L = E - V + 1
    F = set(carriers)
    if len(F) > L:
        return False, f'{len(F)} forced lines exceed L = {L}', None, None
    tree = spanning_tree(verts, edges, F)
    if tree is None:
        return False, ('the remaining lines do not connect (a forced line is a bridge and cannot carry a loop momentum)'), \
            None, None
    chords = [i for i in range(E) if i not in tree]
    # k-numbering: forced lines first (input order), then remaining chords in edge order
    order = sorted(F) + [i for i in chords if i not in F]
    kname = {ei: f'k{order.index(ei) + 1}' for ei in order}
    orient = {i: (a, b) if a < b else (b, a) for i, (a, b) in enumerate(edges)}
    # tree adjacency (for the downstream-side computation)
    adj = {v: [] for v in verts}
    for ti in tree:
        a, b = orient[ti]
        adj[a].append((b, ti))
        adj[b].append((a, ti))
    def component_without(removed, root):
        seen = {root}
        stack = [root]
        while stack:
            v = stack.pop()
            for (w, ti) in adj[v]:
                if ti == removed or w in seen:
                    continue
                seen.add(w)
                stack.append(w)
        return seen
    momenta = {}
    for ti in tree:
        x, y = orient[ti]
        Y = component_without(ti, y)
        terms = {}
        for nm, v in ext_attach.items():
            if v in Y:
                terms[nm] = terms.get(nm, 0) + 1
        for ei in chords:
            a, b = orient[ei]
            if a in Y and b not in Y:
                terms[kname[ei]] = terms.get(kname[ei], 0) + 1
            elif b in Y and a not in Y:
                terms[kname[ei]] = terms.get(kname[ei], 0) - 1
        momenta[ti] = terms
    for ei in chords:
        momenta[ei] = {kname[ei]: 1}
    return True, order, momenta, verts


# Check momentum conservation at each vertex.
def _check_conservation(edges, verts, momenta, ext_attach):
    for v in verts:
        flow = {}
        for ei, (a, b) in enumerate(edges):
            da, db = (a, b) if a < b else (b, a)
            terms = momenta[ei]
            if da == v:                       # outflow at v
                for t, c in terms.items():
                    flow[t] = flow.get(t, 0) - c
            if db == v:                       # inflow at v
                for t, c in terms.items():
                    flow[t] = flow.get(t, 0) + c
        for nm, w in ext_attach.items():
            if w == v:
                flow[nm] = flow.get(nm, 0) - 1
        kvals = [c for t, c in flow.items() if t.startswith('k')]
        if any(c != 0 for c in kvals):
            return False
        pvals = [c for t, c in flow.items() if not t.startswith('k')]
        if pvals and any(c != pvals[0] for c in pvals):
            return False
    return True


# Sort key for momentum terms: k's first (numeric), then externals (alphabetical).
def _term_key(t):
    return (0, int(t[1:]), '') if t.startswith('k') else (1, 0, t)


# {term: coeff} -> readable string such as 'k1 - k2 + p2 + p3'.
def fmt_expr(terms):
    if not terms:
        return '0'
    items = sorted(terms.items(), key=lambda kv: _term_key(kv[0]))
    out = []
    for t, c in items:
        if c == 1:
            out.append(t)
        elif c == -1:
            out.append(f'-{t}')
        else:
            out.append(f'{c}*{t}')
    s = out[0]
    for x in out[1:]:
        s += (' + ' + x) if not x.startswith('-') else (' - ' + x[1:])
    return s


# Print the line-momentum basis and every line's momentum, together with a conservation check.
def show_edge_momenta(edges, em, vm, carriers, ext_attach, forced=False):
    ok, payload, momenta, verts = edge_momenta(edges, em, vm, carriers, ext_attach)
    if not ok:
        reason = payload
        if forced:
            print('  ✗ unfeasible: these line momenta do not form a basis for the loop momenta')
        else:
            print(f'  ! default basis cannot serve as loop-momentum carriers: {reason}')
        return
    order = payload
    print('  line momenta: ' + ', '.join(f'k{i+1} ↦ {edges[ei]} along {edges[ei][0]}→{edges[ei][1]}' for i, ei in enumerate(order)))
    print('  edge momenta (flow along the displayed direction; p_i = external):')
    for ei in range(len(edges)):
        a, b = edges[ei]
        print(f'    ({a},{b}) {a}→{b}: {fmt_expr(momenta[ei])}')
    if _check_conservation(edges, verts, momenta, ext_attach):
        print('  ✓ momentum conservation at every vertex')
    else:
        print('  ✗ momentum conservation FAILED (bug!)')
    print('  These line momenta can form a loop-momentum basis.')

# ---------------------------------------------------------------- parsing
# Parse a region selection like '3, 8--10' -> [3, 4, 8, 10]; sorted 1-based indices (None if invalid).
def parse_region_select(s, n):
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

# Parse forced lines like '(2,5),(7,8)'; edge indices (None if invalid).
def parse_forced_lines(edges, line):
    pairs = re.findall(r'[\[(]\s*(\d+)\s*,\s*(\d+)\s*[\])]', line)
    if not pairs:
        print('  ! no (x,y) pairs found — use e.g. (2,5),(7,8) (square brackets ok too)')
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
        print('  ! unknown edges: ' + ', '.join(f'[{a},{b}]' for a, b in unknown))
        return None
    return sorted(set(F))

# Show a loop-momentum basis (forced lines or the default per-mode basis) and the momenta.
def show_basis(edges, em, vm, F=None, ext_attach=None):
    if F:
        show_edge_momenta(edges, em, vm, F, ext_attach, forced=True)
    else:
        results, total, L = indep_loops(edges, em, vm)
        print('  independent loop momenta (a concrete basis):')
        for r in results:
            X = r['mode']
            for b in r['blocks']:
                if not b['basis']:
                    continue
                desc = []
                for j in b['basis']:
                    u, v = edges[j]
                    s = f'#{j} ({u},{v})'
                    if u == v:
                        s += ' [self-loop]'
                    desc.append(s)
                print(f'    mode {ms(X)}: basis = {", ".join(desc)}')
        basis_all = [j for r in results for b in r['blocks'] for j in b['basis']]
        show_edge_momenta(edges, em, vm, basis_all, ext_attach)


# ---------------------------------------------------------------- option handlers (menu 1-4)
# Option 1: pick regions -> mode subgraphs -> optional loop-momentum basis.
def inspect_regions(edges, regs, ext_attach):
    n = len(regs)
    line = input('Region numbers (e.g. "3, 8--10" represents regions 3, 8, 9, and 10; empty = all) > ').strip()
    sel = parse_region_select(line, n) if line else list(range(1, n + 1))
    if sel is None:
        sel = list(range(1, n + 1))
    for i in sel:
        vm, em = regs[i - 1]
        print(f'  --- region {i}:')
        show_mode_subgraphs(edges, em, vm)
    if input('Select a set of line momenta as independent loop momenta? (y/n) [n] > ').strip().lower() == 'y':
        asked = False
        while True:
            prompt = ('Force lines into the basis? ((x,y) pairs; empty = show default basis) > ' if not asked
                      else 'Force more lines? ((x,y) pairs; empty = done) > ')
            line = input(prompt).strip()
            if not line:
                if not asked:
                    for i in sel:
                        vm, em = regs[i - 1]
                        print(f'  --- region {i}:')
                        show_basis(edges, em, vm, ext_attach=ext_attach)
                break
            asked = True
            F = parse_forced_lines(edges, line)
            if F is None:
                continue
            for i in sel:
                vm, em = regs[i - 1]
                print(f'  --- region {i}:')
                show_basis(edges, em, vm, F, ext_attach)


# Option 2: Lee-Pomeransky parametric representation per region: x_e ~ λ^{v_e}, v_e = -(2m+n), in input edge order.
def show_parametric(regs):
    print('  Lee-Pomeransky parametric representation (x_e ~ λ^{v_e} with λ the expansion parameter, edge order):')
    for i, (vm, em) in enumerate(regs, 1):
        print(f'    R{i}: v = {tuple(scaling_of(m) for m in em)}')


# Option 3: group the regions by their softest-mode class and print each group.
def show_classify(regs, edges, kappa):
    groups = group_by_type(regs, kappa)
    order = type_order(kappa)
    labels = [sc_short(*mn) for mn in order] + ['C/H']
    total = 0
    for lab in labels:
        if lab not in groups:
            continue
        g = groups[lab]
        total += len(g)
        print()
        print(f'There are {len(g)} {lab} type regions:')
        for i, (vm, em) in enumerate(g, 1):
            print(f'  --- {lab} region {i} ---')
            print_region_with_vector(vm, em, edges, indent='    ')
    print()
    print(f'TOTAL: {total} regions')


# Option 4: visualize the selected regions via wolframscript (can output PDF atlas or PNGs).
def visualize_regions(edges, regs, extmode, ext_attach):
    n = len(regs)
    line = input('Region numbers (e.g. "3, 8--10" represents regions 3, 8, 9, and 10; empty = all) > ').strip()
    sel = parse_region_select(line, n) if line else list(range(1, n + 1))
    if sel is None:
        return
    fmt = input('Output: [a] single PDF atlas (default) / [p] individual PNG files > ').strip().lower()
    try:
        import region_plot_wa
    except Exception as e:
        print(f'  ! visualization module unavailable: {e}')
        return
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fri_out', 'regions_' + time.strftime('%Y%m%d-%H%M%S'))
    if fmt in ('p', 'png', 'files'):
        print(f'  rendering {len(sel)} region figure(s) via wolframscript ...')
        try:
            paths = region_plot_wa.render_regions(edges, sorted({v for e in edges for v in e}), [(i, regs[i - 1]) for i in sel], ext_mode=extmode, ext_attach=ext_attach, outdir=outdir)
        except Exception as e:
            print(f'  ! region rendering failed: {e}')
            return
        print(f'  saved {len(paths)} region PNG(s) to: {outdir}')
    else:
        print(f'  building the PDF atlas for {len(sel)} region(s) via wolframscript ...')
        try:
            path = region_plot_wa.render_atlas(
                edges, sorted({v for e in edges for v in e}),
                [(i, regs[i - 1]) for i in sel],
                ext_mode=extmode, ext_attach=ext_attach, outdir=outdir)
        except Exception as e:
            print(f'  ! atlas rendering failed: {e}')
            return
        print(f'  saved atlas PDF to: {path}')


# ---------------------------------------------------------------- main loop
DEFAULT_EDGES = '[(1,5),(1,8),(2,5),(2,7),(3,6),(3,8),(4,6),(4,7),(5,6),(7,8)]'
DEFAULT_EXTS = ("{'p1':[1,'C1'],'p2':[2,'C2^2'],'p3':[3,'C3^inf'],'p4':[4,'C4^inf']}")

# Interactive loop: read a graph, enumerate its regions, serve the menu; save results to fri_out/.
def main():
    print('=== facet-region browser ===')
    print('Input: graph topology + external momenta only (Python literals).')
    while True:
        internal_lines = ask('internal_lines (topology, edge list)', DEFAULT_EDGES)
        externals = ask("externals ({name: [vertex, mode_str]})", DEFAULT_EXTS)
        try:
            verts = sorted({v for (a, b) in internal_lines for v in (a, b)} | {vv[0] for vv in externals.values()})
            edges = [tuple(sorted((a, b))) for (a, b) in internal_lines]
            ext_attach = {n: vv[0] for n, vv in externals.items()}
            extmode = {n: parse_mode(s) for n, (v, s) in externals.items()}
        except ValueError as e:
            print(f'  ! {e}')
            continue
        print(f'\n  enumerating all regions ({len(edges)} edges, {len(verts)} vertices) ...')
        t0 = time.time()
        regs = enumerate_regions(verts, edges, ext_attach, extmode)
        dt = time.time() - t0
        edge_seq = ', '.join(f'({u},{v})' for u, v in edges)
        print(f'  {len(regs)} regions in {dt:.1f}s (presented in terms of the edge modes {edge_seq}):')
        for i, (vm, em) in enumerate(regs, 1):
            print(f'    R{i}: {em_sequence(em)}')
        kappa = kappa_of(extmode)
        while True:
            print('  Options:')
            print('    1) Inspect specific regions')
            print('    2) Show Lee-Pomeransky parametric representation')
            print('    3) Classify these regions based on their characteristic modes')
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
                show_classify(regs, edges, kappa)
            elif opt == '4':
                visualize_regions(edges, regs, extmode, ext_attach)
            else:
                print('  ! enter 1, 2, 3, 4, or q')
        # save
        fdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fri_out')
        os.makedirs(fdir, exist_ok=True)
        fname = os.path.join(fdir, time.strftime('%Y%m%d-%H%M%S') + '.txt')
        with open(fname, 'w') as f:
            f.write(f'# wide-angle FRI regions - {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'# internal lines: {internal_lines}\n')
            f.write(f'# externals: {externals}\n')
            f.write(f'# regions: {len(regs)} ({dt:.1f}s)\n\n')
            for i, (vm, em) in enumerate(regs, 1):
                sc = ', '.join(map(str, [scaling_of(m) for m in em]))
                f.write(f'R{i:3d}: scaling ({sc}, 1)\n')
                f.write(f'      em: {[ms(m) for m in em]}\n')
                vmtxt = ', '.join(f'{v}: {ms(vm[v])}' for v in sorted(vm))
                f.write(f'      vm: {{{vmtxt}}}\n')
        print(f'  saved to: {fname}')
        print('=' * 72)
        if input('Another graph? (y/n) [n] > ').strip().lower() != 'y':
            break

if __name__ == '__main__':
    main()
