#!/usr/bin/env python3
"""facet_regions_interactive.py — interactive facet-region browser (2026-08-20).

Input: graph topology + external kinematics ONLY (the cut formalism is
internal — the user never touches cuts).  The script:

  1. enumerates ALL regions of the graph (C/H family via nested cuts +
     layered enumeration for the overlapping family; mode-analysis
     compression on by default),
  2. lists them with numbers (edge-mode sequence per region, in input
     edge order), then offers a menu:
       1) Inspect specific regions  — pick a subset ("3, 8--10" =
          regions 3, 4, 8, 10; empty = all); for every picked region:
          MODE SUBGRAPHS
            mode X: {vertices: {V_X}, edges: {E_X}}   loop number of X
          (V_X = join-mode-X vertices, E_X = X-mode edges, r_X = cycle
          rank of the contracted X-subgraph gamma~_X), then optionally a
          concrete set of independent loop momenta (a basis) with
          optional FORCED lines ((x,y) endpoint pairs, square brackets
          also accepted; empty = default basis),
       2) Show Lee-Pomeransky parametric representation — per region the
          scaling vector v_e (x_e ~ λ^{v_e} with λ the expansion parameter,
          v_e = -(2m+n), edge order),
       3) Classify these regions based on their characteristic modes —
          per region its softest-mode class + by-type counts,
       4) Visualize the selected regions — per-mode colours; choose
          [a] a single PDF atlas (default) or [p] one PNG per region
          (region_plot_wa.py; saved under fri_out/regions_*/),
   and loops until the user quits (empty or q).

Usage: python3 facet_regions_interactive.py
  Each input is one line of Python literal; empty line = built-in example.

  internal_lines = [[1,5],[1,8],[2,5],[2,7],[3,6],[3,8],[4,6],[4,7],[5,6],[7,8]]
  externals      = {'p1':[1,'C1'], 'p2':[2,'C2^2'], 'p3':[3,'C3^inf'], 'p4':[4,'C4^inf']}
  (built-in example: 4pt3loop, paper §7.1, 81 regions)

Mode syntax: H / S / S^m / C_i / C_i^n / C_i^inf / C_i^\\infty / SC_i /
SC_i^n / S^mC_i^n  (C_i without n means n=1; SC_i without n means n=1;
i is the direction index from 1).
"""
import sys, re, os, time, warnings
from collections import defaultdict
warnings.filterwarnings('ignore', category=SyntaxWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_graph import mode_str, INF
from indep_loops import indep_loops
from truncation_check import all_regions, kappa_of
from skeleton import run as skeleton_run, has_soft_externals
H = (0, 0, 0)

# ---------------------------------------------------------------- mode parsing
# order matters: SC_i before C_i, S^mC_i^n before SC_i
def parse_mode(s):
    s = s.strip().replace(' ', '')
    if s == 'H': return (0, 0, 0)
    if s == 'S': return (1, 0, 0)
    m = re.fullmatch(r'S\^(\d+)', s)
    if m: return (int(m.group(1)), 0, 0)
    m = re.fullmatch(r'S\^?(\d+)?C_?(\d+)\^?(\d+|inf|infty|∞|\\infty)?', s)
    if m:
        ms, i, ns = m.group(1), int(m.group(2)), m.group(3)
        mval = int(ms) if ms else 1
        nval = INF if ns in ('inf', 'infty', '∞', '\\infty') else (int(ns) if ns else 1)
        return (mval, nval, i)
    m = re.fullmatch(r'SC_?(\d+)\^?(\d+|inf|infty|∞|\\infty)?', s)
    if m:
        i, ns = int(m.group(1)), m.group(2)
        nval = INF if ns in ('inf', 'infty', '∞', '\\infty') else (int(ns) if ns else 1)
        return (1, nval, i)
    m = re.fullmatch(r'C_?(\d+)\^?(\d+|inf|infty|∞|\\infty)?', s)
    if m:
        i, ns = int(m.group(1)), m.group(2)
        nval = INF if ns in ('inf', 'infty', '∞', '\\infty') else (int(ns) if ns else 1)
        return (0, nval, i)
    raise ValueError(f"cannot parse mode: {s!r} (use e.g. C2^inf, C2^\\infty, SC4, S^2, H)")

def ms(md):
    return mode_str(md) if md is not None else '∅'

# ---------------------------------------------------------------- input helpers
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
def enumerate_regions(verts, edges, ext_attach, ext_mode):
    """All regions of the graph.  On the validated domain (externals all
    p_i q_j-type) this uses the skeleton enumerator (pruned; verified
    set-equal on the corpus, 2026-09-17/18).  If any external is soft
    (S^mC^n / S^m), it falls back to the layered enumerator (小马
    2026-09-18: soft externals excluded until revisited).
    Returns list of (vm, em)."""
    regs = {}
    if not has_soft_externals(ext_mode):
        regions, _nc, _dt = skeleton_run(verts, edges, ext_attach, ext_mode,
                                         verbose=False, overlap_strong=True)
        for vm, em in regions:
            regs.setdefault(tuple(tuple(m) for m in em), (vm, em))
        return list(regs.values())
    for vm, em in all_regions(verts, edges, ext_attach, ext_mode,
                              use_compression=True):
        regs.setdefault(tuple(tuple(m) for m in em), (vm, em))
    return list(regs.values())

def em_sequence(em):
    """Edge-mode sequence in input edge order: 'C_1, C_1, S^1C_2^1, ...'"""
    return ', '.join(ms(m) for m in em)

# ---------------------------------------------------------------- classification
# (from the former facet_interpreter.py — canonical main archived 2026-08-20)
def sc_short(m, n):
    """Short label: (1,1)->SC, (1,2)->SC^2, (2,1)->S^2C, (m,0)->S^m."""
    if n == 0:
        return 'S' if m == 1 else f'S^{m}'
    if m == 1:
        return 'SC' if n == 1 else f'SC^{n}'
    if n == 1:
        return f'S^{m}C'
    return f'S^{m}C^{n}'

def type_order(kappa):
    """All (m,n) with m>=1, n>=0, m+n<=kappa, softest-first:
    (sigma=m+n desc, then m desc)."""
    lst = []
    for m in range(1, kappa + 1):
        for n in range(0, kappa + 1 - m):
            lst.append((m, n))
    lst.sort(key=lambda mn: (-(mn[0] + mn[1]), -mn[0]))
    return lst

def classify(vm, em, kappa):
    """Classify a region by its softest mode present (exclusion-style order).
    Considers ALL mode components: both vertex modes and edge modes (a soft
    component such as SC can live on a propagator)."""
    order = type_order(kappa)
    present = set()
    for v, md in vm.items():
        m, n, i = md
        if m >= 1:
            present.add((m, n))
    for md in em:
        m, n, i = md
        if m >= 1:
            present.add((m, n))
    for mn in order:
        if mn in present:
            return sc_short(*mn)
    return 'C/H'

def group_by_type(results, kappa):
    groups = defaultdict(list)
    for vm, em in results:
        groups[classify(vm, em, kappa)].append((vm, em))
    return groups

def scaling_of(md):
    """v_e = -(2m + n)  (x_e ~ λ^{v_e} with λ the expansion parameter);  H -> 0."""
    if md is None:
        return 0
    return -(2 * md[0] + md[1])

def show_parametric(regs):
    """Option 2: Lee-Pomeransky parametric representation per region:
    x_e ~ λ^{v_e}, v_e = -(2m+n), in input edge order."""
    print('  Lee-Pomeransky parametric representation '
          '(x_e ~ λ^{v_e} with λ the expansion parameter, edge order):')
    for i, (vm, em) in enumerate(regs, 1):
        print(f'    R{i}: v = {tuple(scaling_of(m) for m in em)}')


def _fmt_set(items):
    """Format a set/list as '{a, b, c}' or 'empty'. Edges print as [a,b]."""
    if not items:
        return 'empty'
    parts = []
    for x in items:
        if isinstance(x, tuple) and len(x) == 2:
            parts.append(f'[{x[0]},{x[1]}]')
        else:
            parts.append(str(x))
    return '{' + ', '.join(parts) + '}'


def print_region_modes(vm, em, edges, indent='    '):
    """Print a region as one line per mode:
        <mode> vertex = {...}; <mode> edge = {...}.
    Empty vertex/edge sets print 'empty'. Modes in stable order."""
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
        print(f'{indent}{s} vertex = {_fmt_set(v_by_mode.get(md, []))}; '
              f'{s} edge = {_fmt_set(e_by_mode.get(md, []))}.')


def print_region_with_vector(vm, em, edges, indent='    '):
    """Region as mode assignment + scaling vector (trailing 1 = the
    t1/λ power of the single expansion scale, pySecDec style)."""
    print_region_modes(vm, em, edges, indent=indent)
    vec = [scaling_of(md) for md in em]
    print(f'{indent}scaling vector = ({', '.join(map(str, vec))}, 1)')


def show_classify(regs, edges, kappa):
    """Option 3: classify regions by their characteristic (softest) modes,
    grouped by type (softest first), each with mode assignment and
    scaling vector."""
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

# ---------------------------------------------------------------- display
def mode_subgraphs(edges, em, vm):
    """{mode: (V_X, E_X)} — V_X by join-mode, E_X by edge mode (H default)."""
    subs = {}
    for i, m in enumerate(em):
        m = m if m is not None else H
        subs.setdefault(m, [set(), []])[1].append(edges[i])
    for v, m in vm.items():
        m = m if m is not None else H
        subs.setdefault(m, [set(), []])[0].add(v)
    return subs

def show_mode_subgraphs(edges, em, vm):
    """Per-mode subgraphs {{V_X}, {E_X}} with independent loop number."""
    results, total, L = indep_loops(edges, em, vm)
    rank = {r['mode']: r['rank'] for r in results}
    for X in sorted(mode_subgraphs(edges, em, vm),
                    key=lambda m: (m[0], m[1], m[2])):
        V, E = mode_subgraphs(edges, em, vm)[X]
        Vs = '{' + ', '.join(str(v) for v in sorted(V)) + '}'
        Es = '{' + ', '.join(f'({u},{v})' for u, v in sorted(E)) + '}'
        print(f'  mode {ms(X)}: {{vertices: {Vs}, edges: {Es}}}   '
              f'loop number = {rank.get(X, 0)}')
    print(f'  Σ loop numbers = {total}  vs  L = {L}  '
          f'{"✓" if total == L else "✗ MISMATCH"}')

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


def edge_momenta(edges, em, vm, carriers, ext_attach):
    """Momentum of every line as a linear combination of the loop momenta
    k1..kL and the external momenta, self-consistent at every vertex.

    Parameterization on the ORIGINAL graph (tree + chords): the carrier
    lines are the chords (loop-momentum carriers; forced lines first in
    the k numbering, remaining carriers chosen freely), the other |V|-1
    lines form a spanning tree.  A carrier's loop momentum flows through
    every tree line on its fundamental cycle — a "self-loop" of a
    contracted mode subgraph is NOT inert: it is an ordinary line of the
    original graph and its momentum couples to the other lines.

    Lines are oriented (a,b) with a<b; the reported momentum is the flow
    along that direction.  External momenta enter at their attachment
    vertices (net-inflow convention: in - out = p_v).

    Returns (True, order, momenta, verts) with momenta[ei] = {term: coeff}
    (order = carrier edge indices in k-numbering order), or
    (False, reason, None, None) if no such parameterization exists:
    |carriers| > L, or the complement of the carriers is disconnected
    (a carrier would be a bridge — it cannot carry a loop momentum)."""
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


def _check_conservation(edges, verts, momenta, ext_attach):
    """in - out = p_v at every vertex, up to the external-momentum
    identity Σ_v p_v = 0 (momentum conservation of the kinematics):
    k-terms must vanish individually, p-terms must all have equal
    coefficients."""
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


def show_edge_momenta(edges, em, vm, carriers, ext_attach, forced=False):
    """Print line momenta (k1..kL carriers) and every line's momentum."""
    ok, payload, momenta, verts = edge_momenta(edges, em, vm, carriers,
                                                ext_attach)
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

# ---------------------------------------------------------------- parsing
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
    Returns None if nothing parses or an edge is unknown."""
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

def show_basis(edges, em, vm, F=None, ext_attach=None):
    """A concrete basis: default (F=None, per-mode algebraic basis) or
    forced lines F (physical tree+chords parameterization), then every
    line's momentum in terms of k1..kL and the external momenta."""
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
        basis_all = [j for r in results for b in r['blocks']
                     for j in b['basis']]
        show_edge_momenta(edges, em, vm, basis_all, ext_attach)


def inspect_regions(edges, regs, ext_attach):
    """Option 1: pick regions -> mode subgraphs -> optional basis/force."""
    n = len(regs)
    line = input('Region numbers (e.g. "3, 8--10" represents regions '
                 '3, 8, 9, and 10; empty = all) > ').strip()
    sel = parse_region_select(line, n) if line else list(range(1, n + 1))
    if sel is None:
        sel = list(range(1, n + 1))
    for i in sel:
        vm, em = regs[i - 1]
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


def visualize_regions(edges, regs, extmode, ext_attach):
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
        import region_plot_wa
    except Exception as e:
        print(f'  ! visualization module unavailable: {e}')
        return
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'fri_out',
                          'regions_' + time.strftime('%Y%m%d-%H%M%S'))
    if fmt in ('p', 'png', 'files'):
        print(f'  rendering {len(sel)} region figure(s) via wolframscript ...')
        try:
            paths = region_plot_wa.render_regions(
                edges, sorted({v for e in edges for v in e}),
                [(i, regs[i - 1]) for i in sel],
                ext_mode=extmode, ext_attach=ext_attach, outdir=outdir)
        except Exception as e:
            print(f'  ! region rendering failed: {e}')
            return
        print(f'  saved {len(paths)} region PNG(s) to: {outdir}')
    else:
        print(f'  building the PDF atlas for {len(sel)} region(s) '
              f'via wolframscript ...')
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
DEFAULT_EXTS = ("{'p1':[1,'C1'],'p2':[2,'C2^2'],'p3':[3,'C3^inf'],"
                "'p4':[4,'C4^inf']}")

def main():
    print('=== facet-region browser ===')
    print('Input: graph topology + external momenta only (Python literals).')
    while True:
        internal_lines = ask('internal_lines (topology, edge list)',
                             DEFAULT_EDGES)
        externals = ask("externals ({name: [vertex, mode_str]})", DEFAULT_EXTS)
        try:
            verts = sorted({v for (a, b) in internal_lines for v in (a, b)}
                           | {vv[0] for vv in externals.values()})
            edges = [tuple(sorted((a, b))) for (a, b) in internal_lines]
            ext_attach = {n: vv[0] for n, vv in externals.items()}
            extmode = {n: parse_mode(s) for n, (v, s) in externals.items()}
        except ValueError as e:
            print(f'  ! {e}')
            continue
        print(f'\n  enumerating all regions ({len(edges)} edges, '
              f'{len(verts)} vertices) ...')
        t0 = time.time()
        regs = enumerate_regions(verts, edges, ext_attach, extmode)
        dt = time.time() - t0
        edge_seq = ', '.join(f'({u},{v})' for u, v in edges)
        print(f'  {len(regs)} regions in {dt:.1f}s (presented in terms of '
              f'the edge modes {edge_seq}):')
        for i, (vm, em) in enumerate(regs, 1):
            print(f'    R{i}: {em_sequence(em)}')
        kappa = kappa_of(extmode)
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
                show_classify(regs, edges, kappa)
            elif opt == '4':
                visualize_regions(edges, regs, extmode, ext_attach)
            else:
                print('  ! enter 1, 2, 3, 4, or q')
        # save
        fdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'fri_out')
        os.makedirs(fdir, exist_ok=True)
        fname = os.path.join(fdir, time.strftime('%Y%m%d-%H%M%S') + '.txt')
        with open(fname, 'w') as f:
            f.write(f'# wide-angle FRI regions - '
                    f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n')
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
