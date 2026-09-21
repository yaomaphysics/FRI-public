#!/usr/bin/env python3
"""fri23_interactive.py — interactive region browser for the spacelike-collinear 2->3 FRI enumerator (fri23).

Input: graph topology + external kinematics ONLY (the cut formalism is
internal — the user never touches cuts).  External momenta p1..p5 attach
at vertices 1,2,3,4,5.  The script:

  1. enumerates ALL regions of the graph (skeleton23: k0 union /
     k1 engine / k2-k4 chain; the fri enumerator was retired 2026-09-21),
  2. lists them with numbers (scaling vector + non-empty cuts + edge-mode
     sequence), then offers a menu:
       1) Inspect specific regions — pick a subset ("3, 8--10" = regions
          3, 4, 8, 10; empty = all); for every picked region:
          MODE SUBGRAPHS
            mode X: {vertices: {V_X}, edges: {E_X}}   loop number of X
          (r_X = sum of |E| - |V| + 1 over the 1VI blocks of the contracted
          mode subgraph, computed with fri23's own mode_components; Σ r_X vs
          L is checked), then optionally a concrete set of independent loop
          momenta (default basis) with optional FORCED lines ((x,y) endpoint
          pairs, square brackets also accepted),
       2) Show Lee-Pomeransky parametric representation — per region the
          scaling vector v_e = -V (edge order, trailing 1 = expansion scale,
          pySecDec style, same as the verified region files),
       3) Classify these regions based on their characteristic modes —
          per region its softest-mode class + by-type counts,
       4) Visualize the selected regions: one PDF atlas (default) or PNG
          figures (per-mode colours; saved under fri_out/regions_*/),
   and loops until the user quits (empty or q).

Kinematics: k0..k4 (defined in kin23.py; default k1).  Commands inside
the edge prompt: 'kin kX' switches kinematics, 'v' toggles the
enumeration statistics, 'q' quits.  Start with -v to have the statistics
on from the beginning.
Results are saved to fri_out/<timestamp>.txt after each graph.

Usage: python3 fri23_interactive.py [-v]
  Example graph (the built-in Frog): 1-3,1-5,2-3,2-5,3-4,4-5
"""
import ast
import os
import re
import sys
import time
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fri23
import kin23
import skeleton23
from fri23 import INF, V, m_of, name, n_of

EXT_ATTACH = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4, 'p5': 5}
KIN_CHOICES = list(kin23.KIN_ORDER)
KIN_NOTES = {
    'k0': 'all p_i^2 ~ t1',
    'k1': 'all p_i^2 = 0 (lightlike)',
    'k2': 'p_1^2 ~ t1; p_i^2 = 0 otherwise',
    'k3': 'p_1^2 ~ t1, p_2^2 ~ t1^2; p_i^2 = 0 otherwise',
    'k4': 'p_2^2 ~ p_3^2 ~ t1^2; p_i^2 = 0 otherwise',
}


def _disp_key(m):
    """Display order of modes: H first, then S, then carriers by direction
    (1, 4, 5, then the 23 pair family)."""
    if m[0] == 'H':
        return (0, 0, 0, 0, 0)
    if m[0] == 'S':
        return (1, 0, 0, m[1], 0)
    _, d, n, mem, sm = m
    return (2, d, 10**9 if n == INF else (n if n is not None else 0),
            -1 if mem is None else mem, sm)


# ---------------------------------------------------------------- input helpers
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


def fmt_cuts(cuts):
    """Non-empty cuts as a dict-of-lists string (native fri23 style)."""
    return str({k: sorted(v) for k, v in cuts.items() if v})


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


def parse_forced_lines(edges, line):
    """'[2,5],[7,8]' -> sorted edge indices (order-free endpoints).
    Returns None if nothing parses or an edge is unknown.  NB: for
    parallel edges, a (a,b) pair forces ALL copies."""
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


# ---------------------------------------------------------------- mode display
def show_mode_subgraphs(edges, em, vm):
    """Per-mode subgraphs {{V_X}, {E_X}} with loop numbers r_X, plus the
    sum check vs L = |E| - |V| + 1.

    r_X uses the contracted-subgraph rule (1VI blocks of gamma~_X with the
    aux vertex; r = |E| - |V| + 1 per block) — same rule as the 2->2
    browser, computed with fri23's own mode_components."""
    verts = sorted({v for e in edges for v in e})
    results, total, L = indep_loops(edges, em, vm)
    rank = {r['mode']: r['rank'] for r in results}
    for mode in sorted(rank, key=_disp_key):
        E_idx = [i for i, m in enumerate(em) if m == mode]
        Vset = [v for v in verts if vm.get(v) == mode]
        Es = '{' + ', '.join(f'{i}:{edges[i]}' for i in E_idx) + '}'
        Vs = '{' + ', '.join(str(v) for v in Vset) + '}'
        print(f'  mode {name(mode)}: {{vertices: {Vs}, edges: {Es}}}   '
              f'loop number = {rank[mode]}')
    flag = '✓' if total == L else '✗ MISMATCH'
    print(f'  Σ loop numbers = {total}  vs  L = {L}  {flag}')


# ---------------------------------------------------------------- basis machinery
# (ported from the 2->2 browser's regge_indep_loops, minus the sH/G overlay:
#  every fri23 mode is a lattice mode, so each mode is its own unit)

def _basis_of_block(block_verts, block_edges, edges2):
    """Spanning tree of a connected block (aux included); returns
    (tree_edge_indices, basis_edge_indices) — basis = deleted edges."""
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for v in block_verts:
        parent[v] = v
    tree, basis = [], []
    for i in block_edges:
        a, b = edges2[i]
        if a == b:                      # self-loop: never a tree edge
            basis.append(i)
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            tree.append(i)
        else:
            basis.append(i)
    return tree, basis


def _rank_of(blocks):
    return sum(len(idxs) - len(bv) + 1 for (bv, be, idxs) in blocks)


def _iter_units(em, vm, edges, verts):
    """Yield (unit_label, blocks) in canonical order — every mode
    appearing in em/vm is one unit."""
    for mode in sorted(set(em) | set(vm.values()), key=_disp_key):
        blocks = []
        for (bv, be, idxs) in fri23.mode_components(mode, vm, em, edges,
                                                    verts):
            blocks.append((bv, list(be), list(idxs)))
        yield mode, blocks


def indep_loops(edges, em, vm):
    """Per-unit independent loop momenta of a region.

    Returns (results, total_rank, L):
      results = [{'mode': unit, 'rank': r_X,
                  'blocks': [{'verts': ..., 'tree': [...], 'basis': [...]}]}]
    and validates Σ r_X = L."""
    verts = sorted({v for e in edges for v in e})
    L = len(edges) - len(verts) + 1
    results = []
    total = 0
    for unit, blocks in _iter_units(em, vm, edges, verts):
        r = _rank_of(blocks)
        total += r
        out_blocks = []
        for (bv, be, idxs) in blocks:
            tree, basis = _basis_of_block(bv, list(range(len(be))), be)
            out_blocks.append({'verts': sorted(bv, key=str),
                               'tree': [idxs[j] for j in tree],
                               'basis': [idxs[j] for j in basis]})
        results.append({'mode': unit, 'rank': r, 'blocks': out_blocks})
    return results, total, L


# ------------------------------------------- physical momentum parameterization
def _spanning_tree(verts, edge_list, skip):
    """Spanning tree of (verts, edge_list) avoiding the skip set; None if
    no such tree exists (skip set contains a bridge / disconnects)."""
    parent = {v: v for v in verts}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    tree = []
    for i, (a, b) in enumerate(edge_list):
        if i in skip or a == b:
            continue
        if find(a) != find(b):
            union(a, b)
            tree.append(i)
    if len(tree) != len(verts) - 1:
        return None
    root = find(verts[0])
    return tree if all(find(v) == root for v in verts) else None


def edge_momenta(edges, carriers, ext_attach):
    """Momentum of every line as a linear combination of the loop momenta
    k1..kL and the external momenta, self-consistent at every vertex.

    Parameterization on the ORIGINAL graph (tree + chords): the carrier
    lines are the chords (loop-momentum carriers; forced lines first in
    the k numbering, remaining carriers chosen freely), the other |V|-1
    lines form a spanning tree.  A carrier's loop momentum flows through
    every tree line on its fundamental cycle.

    Lines are oriented (a,b) with a<b; the reported momentum is the flow
    along that direction.  External momenta enter at their attachment
    vertices (net-inflow convention: in - out = p_v).

    Returns (True, order, momenta, verts) with momenta[ei] = {term: coeff}
    (order = carrier edge indices in k-numbering order), or
    (False, reason, None, None) if no such parameterization exists:
    |carriers| > L, or the complement of the carriers is disconnected."""
    verts = sorted({v for e in edges for v in e})
    E = len(edges)
    V = len(verts)
    L = E - V + 1
    F = set(carriers)
    if len(F) > L:
        return False, f'{len(F)} forced lines exceed L = {L}', None, None
    tree = _spanning_tree(verts, edges, F)
    if tree is None:
        return False, ('the remaining lines do not connect (a forced line '
                       'is a bridge and cannot carry a loop momentum)'), \
            None, None
    chords = [i for i in range(E) if i not in tree]
    # k-numbering: forced lines first (input order), then remaining chords
    # in edge order
    order = sorted(F) + [i for i in chords if i not in F]
    kname = {ei: f'k{order.index(ei) + 1}' for ei in order}
    orient = {i: (a, b) if a < b else (b, a) for i, (a, b) in enumerate(edges)}
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


def _check_conservation(edges, verts, momenta, ext_attach):
    """in - out = p_v at every vertex, up to Σ_v p_v = 0: k-terms must
    vanish individually, p-terms must all have equal coefficients."""
    for v in verts:
        flow = {}
        for ei, (a, b) in enumerate(edges):
            da, db = (a, b) if a < b else (b, a)
            terms = momenta[ei]
            if da == v:
                for t, c in terms.items():
                    flow[t] = flow.get(t, 0) - c
            if db == v:
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


def _term_key(t):
    return (0, int(t[1:]), '') if t.startswith('k') else (1, 0, t)


def fmt_expr(terms):
    """'k1 - k2 + p2 + p3' from {term: coeff} (0 if empty)."""
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


def show_edge_momenta(edges, carriers, ext_attach, forced=False):
    """Print line momenta (k1..kL carriers) and every line's momentum."""
    ok, payload, momenta, verts = edge_momenta(edges, carriers, ext_attach)
    if not ok:
        reason = payload
        if forced:
            print('  ✗ unfeasible: these line momenta do not form a basis '
                  'for the loop momenta')
        else:
            print(f'  ! default basis cannot serve as loop-momentum '
                  f'carriers: {reason}')
        return
    order = payload
    print('  line momenta: ' + ', '.join(
        f'{f"k{i + 1}"} ↦ {edges[ei]} along {edges[ei][0]}→{edges[ei][1]}'
        for i, ei in enumerate(order)))
    print('  edge momenta (flow along the displayed direction; '
          'p_i = external):')
    for ei in range(len(edges)):
        a, b = edges[ei]
        print(f'    ({a},{b}) {a}→{b}: {fmt_expr(momenta[ei])}')
    if _check_conservation(edges, verts, momenta, ext_attach):
        print('  ✓ momentum conservation at every vertex')
    else:
        print('  ✗ momentum conservation FAILED (bug!)')
    print('  These line momenta can form a loop-momentum basis.')


def show_basis(edges, em, vm, F=None, ext_attach=None):
    """A concrete basis: default (F=None, per-mode 1VI-block basis) or
    forced lines F (physical tree+chords parameterization), then every
    line's momentum in terms of k1..kL and the external momenta."""
    if F:
        show_edge_momenta(edges, F, ext_attach, forced=True)
    else:
        results, total, L = indep_loops(edges, em, vm)
        print('  independent loop momenta (a concrete basis):')
        for r in results:
            mode = r['mode']
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
                print(f'    mode {name(mode)}: basis = {", ".join(desc)}')
        basis_all = [j for r in results for b in r['blocks']
                     for j in b['basis']]
        show_edge_momenta(edges, basis_all, ext_attach)


# ---------------------------------------------------------------- menus
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
        vec, cuts, em, vm = regs[i - 1]
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
                        vec, cuts, em, vm = regs[i - 1]
                        print(f'  --- region {i}:')
                        show_basis(edges, em, vm, ext_attach=ext_attach)
                break
            asked = True
            F = parse_forced_lines(edges, line)
            if F is None:
                continue
            for i in sel:
                vec, cuts, em, vm = regs[i - 1]
                print(f'  --- region {i}:')
                show_basis(edges, em, vm, F, ext_attach)


def show_parametric(regs):
    """Option 2: Lee-Pomeransky parametric representation per region:
    x_e ~ λ^{v_e}, v_e = -V (edge order, trailing 1 = expansion scale)."""
    print('  Lee-Pomeransky parametric representation '
          '(x_e ~ λ^{v_e}, v_e = -V, edge order):')
    for i, (vec, cuts, em, vm) in enumerate(regs, 1):
        print(f'    R{i}: v = {fmt_scaling(vec)}')


def _softness_key(m):
    """Softness order for classification, softest first: modes with a soft
    prefactor m >= 1 by virtuality V (desc); tie m desc, n desc, name."""
    return (0, -V(m), -m_of(m),
            -(n_of(m) if n_of(m) is not None else 0), name(m))


def classify(vm, em):
    """Softest characteristic mode of a region (exclusion-style): the
    softest mode with m >= 1 present in em/vm; if none, the bare
    carrier/hard class 'C/H'."""
    cands = set(em) | set(vm.values())
    soft = [m for m in cands if m[0] != 'H' and m_of(m) >= 1]
    if soft:
        return min(soft, key=_softness_key)
    return 'C/H'


def show_classify(regs):
    """Option 3: classify regions by their characteristic (softest) mode,
    grouped by type (softest first)."""
    groups = defaultdict(list)
    for r in regs:
        vec, cuts, em, vm = r
        groups[classify(vm, em)].append(r)
    order = sorted(groups, key=lambda lab:
                   (99,) if lab == 'C/H' else _softness_key(lab))
    total = 0
    for lab in order:
        g = groups[lab]
        total += len(g)
        label = lab if lab == 'C/H' else name(lab)
        print()
        print(f'There are {len(g)} {label} type regions:')
        for i, (vec, cuts, em, vm) in enumerate(g, 1):
            print(f'  --- {label} region {i} ---')
            print(f'    em = {[name(m) for m in em]}')
            print(f'    scaling = {fmt_scaling(vec)}')
    print(f'\nTOTAL: {total} regions')


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
        import region_plot23
    except Exception as e:
        print(f'  ! visualization module unavailable: {e}')
        return
    outdir = os.path.join(BASE, 'fri_out',
                          'regions_' + time.strftime('%Y%m%d-%H%M%S'))
    if fmt in ('p', 'png', 'files'):
        print(f'  rendering {len(sel)} region figure(s) via wolframscript ...')
        try:
            paths = region_plot23.render_regions(
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
            path = region_plot23.render_atlas(
                edges, sorted({v for e in edges for v in e}),
                [(i, regs[i - 1]) for i in sel],
                ext_mode=ext_mode, outdir=outdir)
        except Exception as e:
            print(f'  ! atlas rendering failed: {e}')
            return
        print(f'  saved atlas PDF to: {path}')


def print_kin_menu():
    print('available 2->3 kinematics:')
    for k in KIN_CHOICES:
        print(f'  {k}: {KIN_NOTES[k]}')


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


# ---------------------------------------------------------------- driver
def run_graph(edges_raw, kin_name, verbose=False):
    edges = [tuple(e) for e in edges_raw]
    verts = sorted({v for e in edges for v in e})
    missing = [v for v in (1, 2, 3, 4, 5) if v not in verts]
    if missing:
        print(f'  [warning] external-leg vertices {missing} not in the '
              f'graph!  The 2->3 kinematics needs vertices 1,2,3,4,5.')
        return
    ext_mode = kin23.ext_modes(kin_name)
    L = len(edges) - len(verts) + 1
    print(f'  kinematics: {kin_name} ({KIN_NOTES[kin_name]})')
    print(f'  graph: {len(edges)} edges, {len(verts)} vertices, '
          f'L = {L} loops')
    print(f'  edges: {edges}')
    if L >= 5:
        print('  [warning] L >= 5 may be slow')
    t0 = time.time()
    try:
        _vecs, total, stats = skeleton23.enumerate_surv(edges, verts,
                                                        EXT_ATTACH, kin_name)
        surv = stats['survivors']
    except Exception as e:
        print(f'  [error] skeleton engine could not handle this input: {e}')
        return
    dt = time.time() - t0
    regs = sorted(surv, key=lambda s: (tuple(float(x) for x in s[0]),
                                       tuple(name(m) for m in s[2])))
    print(f'  FRI regions: {len(regs)}  ({dt:.1f}s)')
    if verbose:
        rej = ', '.join(f'{k}={v}' for k, v in sorted(stats.items())
                        if k != 'survivors')
        print(f'  [stats] combos={total}  kept={len(regs)}  '
              f'counters: {rej or "none"}')
    scal = set()
    for i, (vec, cuts, em, vm) in enumerate(regs, 1):
        scal.add(vec)
        print(f'  R{i:3d}: scaling {fmt_scaling(vec)}')
        print(f'        cuts: {fmt_cuts(cuts)}')
        print(f'        em: {[name(m) for m in em]}')
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
            inspect_regions(edges, regs, EXT_ATTACH)
        elif opt == '2':
            show_parametric(regs)
        elif opt == '3':
            show_classify(regs)
        elif opt == '4':
            visualize_regions(edges, regs, ext_mode)
        else:
            print('  ! enter 1, 2, 3, 4, or q')
    # save
    fdir = os.path.join(BASE, 'fri_out')
    os.makedirs(fdir, exist_ok=True)
    fname = os.path.join(fdir, time.strftime('%Y%m%d-%H%M%S') + '.txt')
    with open(fname, 'w') as f:
        f.write(f'# fri23 regions - {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# kinematics: {kin_name}\n')
        f.write(f'# edges: {edges}  ({len(edges)} edges, {len(verts)} '
                f'verts, L={L})\n')
        f.write(f'# FRI regions: {len(regs)} ({dt:.1f}s)\n\n')
        for i, (vec, cuts, em, vm) in enumerate(regs, 1):
            f.write(f'R{i:3d}: scaling {fmt_scaling(vec)}\n')
            f.write(f'      cuts: {fmt_cuts(cuts)}\n')
            f.write(f'      em: {[name(m) for m in em]}\n')
            vmtxt = ', '.join(f'{v}: {name(vm[v])}' for v in sorted(vm))
            f.write(f'      vm: {{{vmtxt}}}\n')
    print(f'  saved to: {fname}')


def main():
    verbose = any(a in ('-v', '--verbose') for a in sys.argv[1:])
    print('=' * 72)
    print('Spacelike-collinear 2->3 FRI region enumerator (k0..k4)')
    print('external momenta: p1@1, p2@2, p3@3, p4@4, p5@5')
    print('type an edge list, e.g. 1-3,1-5,2-3,2-5,3-4,4-5 (the Frog)')
    print("commands: 'kin kX' switch kinematics, 'v' toggle stats, 'q' quit")
    print('=' * 72)
    kin_name = choose_kinematics()
    print(f'-> using kinematics {kin_name}')
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
                      f'({KIN_NOTES[kin_name]})')
            else:
                print(f'  [error] unknown kinematics "{k}" — choose from '
                      f'{", ".join(KIN_CHOICES)}')
            continue
        if low in ('v', '-v', 'stats', 'verbose'):
            verbose = not verbose
            print(f'-> enumeration statistics {"ON" if verbose else "OFF"}')
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
        run_graph(edges, kin_name, verbose)
        print(f'  (this graph took {time.time() - t0:.1f}s total)')


if __name__ == '__main__':
    main()
