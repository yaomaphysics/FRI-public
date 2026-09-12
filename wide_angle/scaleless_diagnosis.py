#!/usr/bin/env python3
"""scaleless_diagnosis.py — why is a momentum-mode assignment not a region?

This is the counterpart of the region browser: instead of telling MORE
about a region, it explains why a NON-region is scaleless.  The user
inputs an assignment of momentum modes to the edges of a graph (vertex
modes are derived: 𝒳(v) = ∨ of the incident edge modes ∪ attached
externals).  FRI then runs its region checks in order:

  1. momentum conservation at every vertex,
  2. jet connectivity (Coleman--Norton interpretation),
  3. contracted-mode-component 1VI (S^m C^n with n >= 1),
  4. mojetic (H∪J∖J_i),
  5. First Connectivity,
  6. IR compatibility (fixed-point confirmation flow),

and stops at the FIRST failure.  If the assignment is a region, the
answer is simply that.  If not, the expanded integral is scaleless, and
the failing check identifies the responsible subgraph together with the
physical mechanism:

  - momentum violation        -> the specific vertex,
  - jet disconnected          -> the specific jet (Coleman--Norton),
  - contracted component      -> the subgraph scaleless (single-vertex
                                 attachment),
  - mojetic                   -> hard-jet interaction,
  - First Connectivity        -> the non-H subgraph harder than its
                                 neighbours,
  - IR-compat deadlock        -> the union of the non-confirmed
                                 subgraphs.

Usage: python3 scaleless_diagnosis.py
  internal_lines = [[1,5],[1,8],[2,5],[2,7],[3,6],[3,8],[4,6],[4,7],[5,6],[7,8]]
  externals      = {'p1':[1,'C1'], 'p2':[2,'C2^2'], 'p3':[3,'C3^inf'], 'p4':[4,'C4^inf']}
  edge_modes     = ['C1','C1','C1','H','H','C1','H','H','C1','C1']
  (one mode per edge, in input edge order; built-in example = 4pt3loop
  region 8, a genuine region)

Mode syntax: H / S / S^m / C_i / C_i^n / C_i^inf / C_i^\\infty / SC_i /
SC_i^n / S^mC_i^n  (C_i without n means n=1; i is the direction index).
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import region_checker as rc
from read_graph import parse_mode, mode_str, INF
from primitives import Graph, vertex_mode, vee
from mojetic_check import jet_connected_ok, cond1_ok
from contracted_1vi import mode_components_wa

H = (0, 0, 0)


# ---------------------------------------------------------------- diagnostics
def momentum_fail_vertex(g, em, extmode):
    """First vertex where momentum conservation fails (same logic as
    primitives.momentum_ok, but reports the vertex)."""
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
            if not rc.eq(inc[0], inc[1]):
                return v
            continue
        ok = False
        for r in range(1, n):
            for A in itertools.combinations(range(n), r):
                B = [i for i in range(n) if i not in A]
                if not B:
                    continue
                if rc.eq(vee([inc[i] for i in A]), vee([inc[i] for i in B])):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            return v
    return None


def disconnected_jet(vm, em, edges_in):
    """Direction of the first disconnected jet (same logic as
    mojetic_check.jet_connected_ok, but reports the direction)."""
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
        V = {v for v, md in vm.items()
             if md[0] == 0 and md[1] >= 1 and md[2] == i}
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


def fc_fail_subgraph(g, em, extmode):
    """First First-Connectivity failure: the threshold n and the isolated
    component (vertices + edges) of ∪_{𝒱≤n} Γ_X — the "non-H subgraph
    harder than its neighbours".  Returns (verts, edges) or None."""
    eVs = [rc.V(m) if m is not None else INF for m in em]
    vVs = []
    for v in g.vertices:
        md = vertex_mode(g, em, v, extmode)
        vVs.append(rc.V(md) if md else INF)
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


def ir_compat_fail(vm, em, edges_in, ext_attach, extmode):
    """IR-compatibility fixed-point flow (2026-08-21: components are 1VI
    blocks from mode_components_wa, not connected components).
    Returns (ok, stuck_components)."""
    e3 = [(a, b, md) for (a, b), md in zip(edges_in, em) if md is not None]
    all_comps = mode_components_wa(edges_in, [md for (_, _, md) in e3],
                                   ext_attach, extmode, rc)
    confirmed = []
    while True:
        changed = False
        for comp in all_comps:
            if comp in confirmed:
                continue
            tag = rc.check_conditions(comp, confirmed, vm, e3, all_comps,
                                      ext_attach, extmode)
            if tag is not None:
                confirmed.append(comp)
                changed = True
        if not changed:
            break
    stuck = [c for c in all_comps if c not in confirmed]
    return len(stuck) == 0, stuck


def _fmt_set(items):
    """'{a, b, c}' or 'empty'; tuples of two ints print as [a,b]."""
    if not items:
        return 'empty'
    parts = []
    for x in sorted(items, key=str):
        if isinstance(x, tuple) and len(x) == 2 \
                and not isinstance(x[0], tuple):
            parts.append(f'[{x[0]},{x[1]}]')
        else:
            parts.append(str(x))
    return '{' + ', '.join(parts) + '}'


# ---------------------------------------------------------------- the chain
def diagnose(edges, em, ext_attach, extmode):
    """Run the FRI check chain; stop at the first failure.
    Returns (is_region, message)."""
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
        return False, ('the Coleman--Norton interpretation is violated '
                       f'because jet C_{j} is disconnected')

    # 3. (removed 2026-08-21: mode components are 1VI blocks from
    #    mode_components_wa — the biconnected decomposition replaces the
    #    old contracted-1VI filter; see ir_ok_blocks.)

    # 4. mojetic (H∪J∖J_i)
    ok, _ = cond1_ok(edges, em, ext_attach, extmode)
    if not ok:
        return False, ('momentum conservation is violated at the '
                       'hard-jet interaction')

    # 5. First Connectivity
    fc = fc_fail_subgraph(g, em, extmode)
    if fc is not None:
        V, E = fc
        return False, ('integrating over the following subgraph is '
                       f'scaleless: {{{_fmt_set(V)}, {_fmt_set(E)}}} '
                       '(the non-H subgraph harder than its neighbours)')

    # 6. IR compatibility
    ok, stuck = ir_compat_fail(vm, em, edges, ext_attach, extmode)
    if not ok:
        V, E = set(), set()
        for c in stuck:
            V |= c['V']
            E |= c['E']
        return False, ('integrating over the following subgraph is '
                       f'scaleless: {{{_fmt_set(V)}, {_fmt_set(E)}}} '
                       '(the union of the non-confirmed subgraphs)')

    return True, 'this mode assignment IS a region'


# ---------------------------------------------------------------- interactive
DEFAULT_EDGES = '[(1,5),(1,8),(2,5),(2,7),(3,6),(3,8),(4,6),(4,7),(5,6),(7,8)]'
DEFAULT_EXTS = ("{'p1':[1,'C1'],'p2':[2,'C2^2'],'p3':[3,'C3^inf'],"
                "'p4':[4,'C4^inf']}")
DEFAULT_MODES = "['C1','C1','C1','H','H','C1','H','H','C1','C1']"


def ask(label, default=None):
    print(f'{label}:' + (f'  (default: {default})' if default else ''))
    line = input('> ').strip()
    if not line and default:
        line = default
    try:
        return eval(line, {'inf': INF, '∞': INF, 'INF': INF})
    except Exception as e:
        print(f'  ! parse error: {e}')
        return ask(label, default)


def main():
    print('=== scaleless diagnosis ===')
    print('Input: graph + external kinematics + an edge-mode assignment.')
    print('Vertex modes are derived: 𝒳(v) = ∨(incident edge modes ∪ externals).')
    while True:
        internal_lines = ask('internal_lines (topology, edge list)',
                             DEFAULT_EDGES)
        externals = ask("externals ({name: [vertex, mode_str]})", DEFAULT_EXTS)
        mode_strs = ask('edge_modes (one mode per edge, input edge order)',
                        DEFAULT_MODES)
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
        if input('Another assignment? (y/n) [n] > ').strip().lower() != 'y':
            break


if __name__ == '__main__':
    main()
