"""primitives.py — the shared base layer of the wide-angle FRI code: mode
algebra and graph machinery (no region criteria — those live in
region_checker.py).

Contents:
  - mode algebra on (m, n, i) tuples [S^m C_i^n; H = (0,0,0)]:
      V, norm, eq, harder_or_eq, join, meet, marginal_softer;
  - mode scaling: scaling_of (v_e = -V);
  - mode-string parsing: parse_mode (single implementation in
      read_graph.py; re-exported);
  - graph representation + helpers: Graph (vertex_mode, vee, softer_v,
      is_sc_type, allowed);
  - component machinery: build_components;
  - graph utilities: biconnected_blocks (Tarjan; used by the 1VI checks),
      spanning_tree.

Used by: skeleton.py (enumeration), region_checker.py (judgment),
usable_modes.py, scaleless_diagnosis.py, indep_loops.py.

(2026-09-24 “two cores” split: the base parts of the old region_checker.py
were merged in here; the Step-1 / First-Connectivity / IR check functions
moved to region_checker.py; the standalone 5pt6loop enumerator lives in
private/wide_angle_dev/dev_5pt6loop_enum.py.)
"""
from collections import defaultdict

from read_graph import parse_mode  # re-export; single implementation (2026-09-24)

INF = 100   # sentinel for C_i^inf; one value for the whole WA tree (2026-09-24)
H = (0, 0, 0)


# ============================== mode algebra ==============================

def V(m): return 2*m[0] + m[1]

def norm(m):
    m0, n, i = m
    if n > INF//2: n = INF
    if n == 0: i = 0
    return (m0, n, i)

def eq(X, Y):
    X, Y = norm(X), norm(Y)
    return X[0] == Y[0] and X[1] == Y[1] and (X[2] == Y[2] or X[1] == 0 or Y[1] == 0)

def harder_or_eq(X, Y):
    X, Y = norm(X), norm(Y)
    if eq(X, Y): return True
    if X[2] == Y[2]:
        return X[0] <= Y[0] and X[0] + X[1] <= Y[0] + Y[1]
    return X[0] + X[1] <= Y[0]

def join(X, Y): return _join_meet(X, Y)[0]

def meet(X, Y): return _join_meet(X, Y)[1]

def _join_meet(X, Y):
    X, Y = norm(X), norm(Y)
    if X == (0, 0, 0): return (X, Y)
    if Y == (0, 0, 0): return (Y, X)
    if eq(X, Y): return (X, X)
    iX = X[2] if X[1] != 0 else (Y[2] if Y[1] != 0 else 0)
    iY = Y[2] if Y[1] != 0 else iX
    if iX == iY:
        A, B = (X[0], X[1], iX), (Y[0], Y[1], iX)
        for (P, Q) in ((A, B), (B, A)):
            m1, n1, _ = P; m2, n2, _ = Q
            if m2 < m1 <= m1 + n1 < m2 + n2:
                return (norm((m2, m1 + n1 - m2, iX)), norm((m1, m2 + n2 - m1, iX)))
        if harder_or_eq(A, B) and not harder_or_eq(B, A): return (norm(A), norm(B))
        if harder_or_eq(B, A) and not harder_or_eq(A, B): return (norm(B), norm(A))
        return (norm(A), norm(B))
    else:
        # different directions: first check comparability (meet/join of two
        # comparable modes = the softer/harder one, regardless of direction)
        if harder_or_eq(X, Y) and not harder_or_eq(Y, X):
            return (norm(X), norm(Y))      # X harder: join=X, meet=Y
        if harder_or_eq(Y, X) and not harder_or_eq(X, Y):
            return (norm(Y), norm(X))      # Y harder: join=Y, meet=X
        if X[0] + X[1] <= Y[0] + Y[1]: P, Q = X, Y
        else: P, Q = Y, X
        m1, n1, i = P; m2, n2, j = Q
        if m1 <= m2: jn = norm((m1, m2 - m1, i))
        else: jn = norm((m2, m1 - m2, j))
        mt = norm((m1 + n1, m2 + n2 - m1 - n1, j))
        return (jn, mt)

def marginal_softer(X, Y):
    X, Y = norm(X), norm(Y)
    if eq(X, Y): return False
    if not harder_or_eq(Y, X): return False
    if harder_or_eq(X, Y): return False
    return X[0] <= Y[0] + Y[1]


def scaling_of(md):
    """Scaling exponent v_e = -(2m + n) = -V(md) of edge mode md
    (x_e ~ lambda^{v_e}, lambda = expansion parameter); None -> 0."""
    return 0 if md is None else -V(md)


# =========================== mode-string parsing ==========================
# (parse_mode is re-exported from read_graph — see the import at the top.)


# =========================== graph representation =========================

class Graph:
    def __init__(self, vertices, edges, ext):
        self.vertices = list(vertices)
        self.edges = list(edges)
        self.ext = dict(ext)
        self.vidx = {v: i for i, v in enumerate(self.vertices)}
        self.incident = defaultdict(list)
        for ei, (u, v) in enumerate(self.edges):
            self.incident[u].append(ei)
            self.incident[v].append(ei)
    def edge_other(self, ei, v):
        u, w = self.edges[ei]
        return w if u == v else u

def vertex_mode(g, edge_modes, v, EXTMODE):
    accs = []
    for ei in g.incident.get(v, []):
        if edge_modes[ei] is not None:
            accs.append(edge_modes[ei])
    for name, vv in g.ext.items():
        if vv == v: accs.append(EXTMODE[name])
    if not accs: return None
    acc = accs[0]
    for m in accs[1:]: acc = join(acc, m)
    return norm(acc)

def vee(modes):
    if not modes: return None
    acc = modes[0]
    for m in modes[1:]: acc = join(acc, m)
    return norm(acc)

def softer_v(va, vb):
    if eq(va, vb): return va
    return va if V(va) > V(vb) else vb

def is_sc_type(md):
    return md[0] >= 1 and md[1] >= 1

def allowed(va, vb):
    if eq(va, vb):
        return [va]
    if harder_or_eq(va, vb):
        return [vb]
    if harder_or_eq(vb, va):
        return [va]
    sv = softer_v(va, vb)
    if is_sc_type(va) or is_sc_type(vb):
        return [sv]
    opts = []
    if V(va) != V(vb):
        opts.append(sv)
    mt = meet(va, vb)
    if mt not in opts: opts.append(mt)
    return opts

# =========================== component machinery =========================

def build_components(verts, edges):
    by_mode = defaultdict(lambda: {'V': set(), 'E': set()})
    for v, md in verts.items(): by_mode[md]['V'].add(v)
    for (a, b, md) in edges: by_mode[md]['E'].add((a, b))
    comps = []
    for md, g in by_mode.items():
        nodes = list(g['V']) + list(g['E'])
        idx = {nd: k for k, nd in enumerate(nodes)}
        par = list(range(len(nodes)))
        def find(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: par[ra] = rb
        for k, e in enumerate(g['E']):
            a, b = e
            for v in (a, b):
                if v in g['V']: union(idx[e], idx[v])
        groups = defaultdict(lambda: {'V': set(), 'E': set()})
        for k, nd in enumerate(nodes):
            r = find(k)
            if nd in g['V']: groups[r]['V'].add(nd)
            else: groups[r]['E'].add(nd)
        for gr in groups.values():
            comps.append({'mode': md, 'V': gr['V'], 'E': gr['E']})
    return comps

# ======================== graph utilities ========================

def spanning_tree(verts, edge_list, skip):
    """Spanning tree of (verts, edge_list) avoiding the skip set (edge
    indices); None if no such tree exists (the skip set contains a bridge
    / disconnects the graph).  (Moved from the wide-angle interactive
    browser, 2026-09-24.)"""
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


def biconnected_blocks(verts, edges):
    """Biconnected components (blocks) of (verts, edges); bridges appear as
    single-edge blocks; self-loops as single-vertex blocks; isolated vertices
    as single-vertex blocks.  (Tarjan; identical algorithm to
    spacelike_collinear/regge/regge_core.mode_components — 2026-08-21: the
    Regge methodology decomposes γ̃_X into 1VI blocks instead of filtering.)"""
    loops = [(a, b) for (a, b) in edges if a == b]
    other = [(a, b) for (a, b) in edges if a != b]
    out = []
    for (a, b) in loops:
        out.append(({a}, [(a, b)]))
    adj = {v: [] for v in verts}
    for i, (a, b) in enumerate(other):
        adj[a].append((b, i)); adj[b].append((a, i))
    disc = {}; low = {}; t = 0; stack = []; blocks = []
    def dfs(u, pe):
        nonlocal t
        disc[u] = low[u] = t; t += 1
        for (w, ei) in adj[u]:
            if ei == pe: continue
            if w not in disc:
                stack.append(ei)
                dfs(w, ei)
                low[u] = min(low[u], low[w])
                if low[w] >= disc[u]:
                    blk = set()
                    while True:
                        e = stack.pop(); blk.add(e)
                        if e == ei: break
                    blocks.append(blk)
            elif disc[w] < disc[u]:
                stack.append(ei)
                low[u] = min(low[u], disc[w])
    for v in verts:
        if v not in disc:
            dfs(v, -1)
            if stack:
                blocks.append(set(stack)); stack.clear()
    for blk in blocks:
        be = [other[i] for i in blk]
        out.append(({v for e in be for v in e}, be))
    used = {v for _, be in out for e in be for v in e}
    for v in verts:
        if v not in used:
            out.append(({v}, []))
    return out
