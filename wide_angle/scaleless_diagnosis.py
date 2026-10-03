#!/usr/bin/env python3
"""scaleless_diagnosis.py — why is a momentum-mode assignment not a region?

This is the counterpart of the region browser: instead of telling MORE
about a region, it explains why a NON-region is scaleless.  The user
inputs an assignment of momentum modes to the edges of a graph (vertex
modes are derived: 𝒳(v) = ∨ of the incident edge modes ∪ attached
externals).  FRI then runs its region checks in order:

  1. momentum conservation at every vertex,
  2. jet connectivity (Coleman--Norton interpretation),
  3. mojetic (H∪J∖J_i),
  4. First Connectivity,
  5. IR compatibility (fixed-point confirmation flow),

and stops at the FIRST failure.  If the assignment is a region, the
answer is simply that.  If not, the expanded integral is scaleless, and
the failing check identifies the responsible subgraph together with the
physical mechanism:

  - momentum violation        -> the specific vertex,
  - jet disconnected          -> the specific jet (Coleman--Norton),
  - mojetic                   -> hard-jet interaction,
  - First Connectivity        -> the non-H subgraph harder than its
                                 neighbours,
  - IR-compat deadlock        -> the union of the non-confirmed
                                 subgraphs.

Usage: python3 scaleless_diagnosis.py
  internal_lines: edge list, e.g. "1-5,1-8,..." (Python lists like [[1,5],[1,8],...] work too)
  externals: prompted one by one (vertex, name, mode) — the example uses p1 = [1,'C1'], p2 = [2,'C2^2'],
  p3 = [3,'C3^inf'], p4 = [4,'C4^inf']
  edge_modes     = ['C1','C1','C1','H','H','C1','H','H','C1','C1']
  (one mode per edge, in input edge order; built-in example = 4pt3loop
  region 8, a genuine region)

Mode syntax: H / S / S^m / C_i / C_i^n / C_i^inf / C_i^\\infty / SC_i /
SC_i^n / S^mC_i^n  (C_i without n means n=1; i is the direction index).
"""
import sys, os, itertools, re, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_graph import parse_mode, INF
from primitives import Graph, vertex_mode, vee, eq, V
from region_checker import (hard_jet_mojetic_ok, confirm_all, mode_components)

H = (0, 0, 0)


# ---------------------------------------------------------------- diagnostics
# First vertex where momentum conservation fails (same logic as primitives.momentum_ok, but reports the vertex).
def momentum_fail_vertex(g, em, extmode):
    for v in g.vertices:
        inc = []
        for ei in g.incident.get(v, []):
            if em[ei] is not None:
                inc.append(em[ei])
        for name, vv in g.ext.items():
            if vv == v:
                inc.append(extmode[name])
        n = len(inc)
        if n < 2:
            if n == 1 and inc and inc[0] != H:
                return v
            continue
        if n == 2:
            if not eq(inc[0], inc[1]):
                return v
            continue
        ok = False
        for r in range(1, n):
            for A in itertools.combinations(range(n), r):
                B = [i for i in range(n) if i not in A]
                if not B:
                    continue
                if eq(vee([inc[i] for i in A]), vee([inc[i] for i in B])):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            return v
    return None


# Direction of the first disconnected jet (same logic as region_checker.jet_connected_ok, but reports the direction).
def disconnected_jet(vm, em, edges_in):
    dirs = set()
    for v, md in vm.items():
        m, n, i = md
        if m == 0 and n >= 1 and i != 0:
            dirs.add(i)
    for md in em:
        m, n, i = md
        if m == 0 and n >= 1 and i != 0:
            dirs.add(i)
    for i in sorted(dirs):
        V = {v for v, md in vm.items() if md[0] == 0 and md[1] >= 1 and md[2] == i}
        if not V:
            continue
        adj = {v: set() for v in V}
        for (a, b), md in zip(edges_in, em):
            if md[0] == 0 and md[1] >= 1 and md[2] == i and a in V and b in V:
                adj[a].add(b)
                adj[b].add(a)
        seen = {next(iter(V))}
        st = list(seen)
        while st:
            x = st.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    st.append(y)
        if len(seen) != len(V):
            return i
    return None


# First First-Connectivity failure: the isolated component (verts + edges) of ∪_{𝒱≤n} Γ_X — the "non-H subgraph harder than its neighbours"; returns (verts, edges) or None.
def fc_fail_subgraph(g, em, extmode):
    eVs = [V(m) if m is not None else INF for m in em]
    vVs = []
    for v in g.vertices:
        md = vertex_mode(g, em, v, extmode)
        vVs.append(V(md) if md else INF)
    thresh = sorted({v for v in eVs if v < INF} | {v for v in vVs if v < INF})
    for n in thresh:
        sub = [ei for ei, vv in enumerate(eVs) if vv <= n]
        nodes = set()
        for ei in sub:
            nodes.add(g.vidx[g.edges[ei][0]])
            nodes.add(g.vidx[g.edges[ei][1]])
        for i, vv in enumerate(vVs):
            if vv <= n:
                nodes.add(i)
        if not nodes:
            continue
        par = {i: i for i in nodes}
        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                par[ra] = rb
        for ei in sub:
            union(g.vidx[g.edges[ei][0]], g.vidx[g.edges[ei][1]])
        roots = {find(i) for i in nodes}
        if len(roots) > 1:
            # the isolated component: smallest root set (first found)
            comp = [i for i in nodes if find(i) == min(roots)]
            verts = sorted(g.vertices[i] for i in comp)
            eset = set()
            for ei in sub:
                a = g.vidx[g.edges[ei][0]]
                b = g.vidx[g.edges[ei][1]]
                if a in comp and b in comp:
                    eset.add(tuple(sorted((g.vertices[a], g.vertices[b]))))
            return verts, sorted(eset)
    return None


# IR-compatibility fixed-point flow (2026-08-21: components are 1VI blocks from mode_components, not connected components; 2026-09-26: shared confirm_all core); returns (ok, stuck_components).
def ir_compat_fail(vm, em, edges_in, ext_attach, extmode):
    e3 = [(a, b, md) for (a, b), md in zip(edges_in, em) if md is not None]
    all_comps = mode_components(edges_in, [md for (_, _, md) in e3], ext_attach, extmode)
    ok, _order, _confirmed, stuck = confirm_all(all_comps, vm, e3, ext_attach, extmode)
    return ok, stuck


# '{a, b, c}' or 'empty'; tuples of two ints print as [a,b].
def _fmt_set(items):
    if not items:
        return 'empty'
    parts = []
    for x in sorted(items, key=str):
        if isinstance(x, tuple) and len(x) == 2 and not isinstance(x[0], tuple):
            parts.append(f'[{x[0]},{x[1]}]')
        else:
            parts.append(str(x))
    return '{' + ', '.join(parts) + '}'


# ---------------------------------------------------------------- the chain
# Run the FRI check chain; stop at the first failure; returns (is_region, message).
def diagnose(edges, em, ext_attach, extmode):
    verts = sorted({v for e in edges for v in e})
    g = Graph(verts, edges, ext_attach)
    vm = {v: vertex_mode(g, em, v, extmode) for v in verts}

    # 1. momentum conservation
    v = momentum_fail_vertex(g, em, extmode)
    if v is not None:
        return False, f'momentum conservation is violated at vertex {v}'

    # 2. jet connectivity (Coleman--Norton)
    j = disconnected_jet(vm, em, edges)
    if j is not None:
        return False, (f'the Coleman--Norton interpretation is violated because jet C_{j} is disconnected')

    # (contracted-1VI check removed — superseded by third_port; mode components are 1VI blocks from mode_components — see ir_ok.)

    # 3. mojetic (H∪J∖J_i)
    ok, _ = hard_jet_mojetic_ok(edges, em, ext_attach, extmode)
    if not ok:
        return False, ('momentum conservation is violated at the hard-jet interaction')

    # 4. First Connectivity
    fc = fc_fail_subgraph(g, em, extmode)
    if fc is not None:
        V, E = fc
        return False, (f'integrating over the following subgraph is scaleless: {{{_fmt_set(V)}, {_fmt_set(E)}}} (the non-H subgraph harder than its neighbours)')

    # 5. IR compatibility
    ok, stuck = ir_compat_fail(vm, em, edges, ext_attach, extmode)
    if not ok:
        V, E = set(), set()
        for c in stuck:
            V |= c['V']
            E |= c['E']
        return False, (f'integrating over the following subgraph is scaleless: {{{_fmt_set(V)}, {_fmt_set(E)}}} (the union of the non-confirmed subgraphs)')

    return True, 'this mode assignment IS a region'


# ---------------------------------------------------------------- interactive
DEFAULT_EDGES = '[(1,5),(1,8),(2,5),(2,7),(3,6),(3,8),(4,6),(4,7),(5,6),(7,8)]'
DEFAULT_MODES = "['C1','C1','C1','H','H','C1','H','H','C1','C1']"


# Ask a y/n question (empty = n); any other input re-asks.
def ask_yn(prompt):
    while True:
        ans = input(prompt).strip().lower() or 'n'
        if ans in ('y', 'n'):
            return ans == 'y'
        print('  ! answer y or n (empty = n)')


def ask(label, default=None):
    print(f'{label}:' + (f'  (default: {default})' if default else ''))
    line = input('> ').strip()
    if line.lower() in ('q', 'quit', 'exit', 'b', 'back'):
        return None
    if not line and default:
        line = default
    try:
        return eval(line, {'inf': INF, '∞': INF, 'INF': INF})
    except Exception as e:
        print(f'  ! parse error: {e}')
        return ask(label, default)

# Parse an edge list leniently: accepts '1-5,1-8,...', '(1,5),(1,8),...' or '[(1,5),(1,8),...]'.
def parse_edge_list(s):
    t = re.sub(r'[\[\](){}]', ' ', s).replace(';', ',')
    dp = re.findall(r'(\d+)\s*-\s*(\d+)', t)
    dn = re.findall(r'\d+', t)
    if not dn:
        if t.strip():
            raise ValueError(f"cannot parse edge list: '{s.strip()}'")
        return []
    if dp and 2 * len(dp) == len(dn):
        return [(int(a), int(b)) for a, b in dp]
    if len(dn) % 2:
        raise ValueError(f"cannot parse edge list: '{s.strip()}'")
    return [(int(dn[k]), int(dn[k + 1])) for k in range(0, len(dn), 2)]

# Ask for the edge list; empty input = default; 'q' quits; re-prompts on parse errors.
def ask_edges(label, default):
    print(f'{label}:' + (f'  (default: {default})' if default else ''))
    while True:
        line = input('> ').strip()
        if line.lower() in ('q', 'quit', 'exit', 'b', 'back'):
            return None
        line = line or default
        try:
            return parse_edge_list(line)
        except ValueError as e:
            print(f'  ! {e}')

# Ask for the external momenta one by one (vertex, name, mode); asks whether to add another.
def ask_externals():
    print('Externals: add them one by one (vertex, name, mode; q/b = quit).')
    out = {}
    while True:
        i = len(out) + 1
        while True:
            s = input(f'  external {i}: attach to which vertex? (integer) > ').strip()
            if s.lower() in ('q', 'quit', 'exit', 'b', 'back'):
                return None
            try:
                vtx = int(s); break
            except ValueError:
                print('  ! enter an integer vertex label, e.g. 1')
        while True:
            nm = input(f'  external {i}: name? (e.g. p{i}) > ').strip()
            if nm.lower() in ('q', 'quit', 'exit', 'b', 'back'):
                return None
            if nm and nm not in out:
                break
            print('  ! the name must be non-empty and not yet used: ' + (', '.join(out) if out else '(no names so far)'))
        while True:
            s = input(f'  external {i}: mode? (form S^mC_i^n, e.g. S^2C1^3; also allowed: H, S^m, C_i^n) > ').strip()
            if s.lower() in ('q', 'quit', 'exit', 'b', 'back'):
                return None
            try:
                parse_mode(s); break
            except ValueError as e:
                print(f'  ! {e}')
        out[nm] = [vtx, s]
        print(f'  added: {nm} = [vertex {vtx}, mode {s}]')
        if not ask_yn('  Add another external? (y/n) [n] > '):
            break
    print(f'  externals = {out}')
    return out


def main():
    print('=== scaleless diagnosis ===')
    print('Input: graph + external kinematics + an edge-mode assignment.')
    print('Vertex modes are derived: 𝒳(v) = ∨(incident edge modes ∪ externals).')
    while True:
        internal_lines = ask_edges('internal_lines (topology, edge list; e.g. 1-5,1-8,...)', DEFAULT_EDGES)
        if internal_lines is None:
            print('bye!')
            return
        externals = ask_externals()
        if externals is None:
            print('bye!')
            return
        mode_strs = ask('edge_modes (one mode per edge, input edge order; q/b = quit)', DEFAULT_MODES)
        if mode_strs is None:
            print('bye!')
            return
        try:
            edges = [tuple(sorted((a, b))) for (a, b) in internal_lines]
            ext_attach = {n: vv[0] for n, vv in externals.items()}
            extmode = {n: parse_mode(s) for n, (v, s) in externals.items()}
            em = [parse_mode(s) for s in mode_strs]
            if len(em) != len(edges):
                print(f'  ! {len(em)} edge modes for {len(edges)} edges')
                continue
        except ValueError as e:
            print(f'  ! {e}')
            continue
        is_reg, msg = diagnose(edges, em, ext_attach, extmode)
        print('  ' + msg)
        if not ask_yn('Another assignment? (y/n) [n] > '):
            break


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.default_int_handler)  # keep Ctrl+C working even if the launcher had ignored SIGINT
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nbye!')
