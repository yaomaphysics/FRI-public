"""primitives.py — the shared base layer of the wide-angle FRI code: mode
algebra and graph machinery (no region criteria — those live in
region_checker.py).

Contents:
  - mode algebra on (m, n, i) tuples [S^m C_i^n; H = (0,0,0)]: V, norm, eq, harder_or_eq, join, meet, marginally_softer, vee;
  - mode scaling: scaling_of (v_e = -V);
  - mode-string parsing: parse_mode, mode_str;
  - graph helpers: Graph (vertex_mode); spanning_tree; find_1vi_blocks (Tarjan; used by the 1VI checks).

Used by: skeleton.py (enumeration), region_checker.py (judgment), usable_modes.py, scaleless_diagnosis.py, indep_loops.py, read_graph.py (input).
"""

import re
from collections import defaultdict
from functools import lru_cache # lru: least recently used (cache eviction policy)

INF = 100   # a large value representing \infty.
H = (0, 0, 0)


# ============================== MODE ALGEBRA ==============================

# lru: hot micro layer -- tiny key space, call counts in the millions per case.
@lru_cache(maxsize=None)
def V(m): return 2*m[0] + m[1]

@lru_cache(maxsize=None)
def norm(m):
    m0, n, i = m
    if n > INF//2: n = INF
    if n == 0: i = 0
    return (m0, n, i)

@lru_cache(maxsize=None)
def eq(X, Y):
    X, Y = norm(X), norm(Y)
    return X[0] == Y[0] and X[1] == Y[1] and (X[2] == Y[2] or X[1] == 0 or Y[1] == 0)

@lru_cache(maxsize=None)
def harder_or_eq(X, Y):
    X, Y = norm(X), norm(Y)
    if eq(X, Y): return True
    if X[2] == Y[2]:
        return X[0] <= Y[0] and X[0] + X[1] <= Y[0] + Y[1]
    return X[0] + X[1] <= Y[0]

def join(X, Y): return _join_meet(X, Y)[0]

def meet(X, Y): return _join_meet(X, Y)[1]

# Join (vee) and meet (wedge) of two modes X and Y, returned as the pair (join, meet); memoised because the same (X, Y) pairs recur heavily in the enumerators.
@lru_cache(maxsize=None) # lru cache: least-recently-used eviction
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
        # Different directions: first check comparability (meet/join of two comparable modes = the softer/harder one, regardless of direction).
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

# Check whether X is marginally softer than Y
@lru_cache(maxsize=None)
def marginally_softer(X, Y):
    X, Y = norm(X), norm(Y)
    if eq(X, Y): return False
    if not harder_or_eq(Y, X): return False
    if harder_or_eq(X, Y): return False
    return X[0] <= Y[0] + Y[1]


# Join of a list of modes -- the softest mode that is harder than (or equal to) all of them.
def vee(modes):
    modes = [m for m in modes if m is not None]
    if not modes:
        return H
    acc = modes[0]
    for m in modes[1:]:
        acc = join(acc, m)
    return norm(acc)


# Scaling exponent v_e = -(2m + n) = -V(md) of edge mode md (x_e ~ lambda^{v_e}, lambda -- the expansion parameter).
def scaling_of(md):
    return 0 if md is None else -V(md)


# =========================== MODE-STRING PARSING ==========================

# Parse a mode string into (m, n, i) [S^m C_i^n; H = (0,0,0); C_i^inf -> n = INF]; accepts 'inf'/'infty'/'∞'/'\\infty' and the SC_i shorthand.
def parse_mode(s):
    s = s.strip().replace(' ', '')
    if s in ('H', 'h'):
        return (0, 0, 0)
    if s == 'S':
        return (1, 0, 0)
    m = re.fullmatch(r'S\^(\d+)', s)
    if m:
        return (int(m.group(1)), 0, 0)
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


# Internal tuple back to a readable string (for verification echo).
def mode_str(md):
    m, n, i = md
    if m == 0 and n == 0:
        return 'H'
    if n == 0:
        return 'S' if m == 1 else f'S^{m}'
    if m == 0:
        if n >= INF:
            return f'C_{i}^∞'
        return f'C_{i}' if n == 1 else f'C_{i}^{n}'
    return f'S^{m}C_{i}^{n}'


# ============================= GRAPH HELPERS ==============================

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


# Connectivity of the subgraph induced on verts (adj = full adjacency; edges leaving verts are ignored).
def is_connected(verts, adj):
    if not verts:
        return True
    verts = set(verts)
    seen = {next(iter(verts))}
    st = list(seen)
    while st:
        u = st.pop()
        for w in adj.get(u, ()):
            if w in verts and w not in seen:
                seen.add(w)
                st.append(w)
    return len(seen) == len(verts)


# Vertex mode: join of the vertex's non-None incident edge modes and its attached external modes; None if there is nothing to join.
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


# Spanning tree of (verts, edge_list) avoiding the skip set (edge indices); None if no such tree exists.
def spanning_tree(verts, edge_list, skip):
    parent = {v: v for v in verts}
    # union-find: find, with path compression.
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    # union-find: merge two components.
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    # Kruskal algorithm: add every non-skipped edge that joins two different components.
    tree = []
    for i, (a, b) in enumerate(edge_list):
        if i in skip or a == b:
            continue
        if find(a) != find(b):
            union(a, b)
            tree.append(i)
    # a spanning tree needs exactly |V| - 1 edges and must reach every vertex.
    if len(tree) != len(verts) - 1:
        return None
    root = find(verts[0])
    return tree if all(find(v) == root for v in verts) else None


# 1VI components of (verts, edges): bridges = single-edge blocks, self-loops / isolated vertices = single-vertex blocks (Tarjan).
def find_1vi_blocks(verts, edges):
    # self-loops are single-vertex blocks.
    loops = [(a, b) for (a, b) in edges if a == b]
    other = [(a, b) for (a, b) in edges if a != b]
    out = []
    for (a, b) in loops:
        out.append(({a}, [(a, b)]))
    # adjacency with edge indices (a block is a set of edge indices).
    adj = {v: [] for v in verts}
    for i, (a, b) in enumerate(other):
        adj[a].append((b, i)); adj[b].append((a, i))
    disc = {}; low = {}; t = 0; stack = []; blocks = []
    # Tarjan: pop an edge block whenever a child's low >= the current disc.
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
    # disconnected pieces: flush whatever remains on the stack.
    for v in verts:
        if v not in disc:
            dfs(v, -1)
            if stack:
                blocks.append(set(stack)); stack.clear()
    # back from edge indices to (vertex set, edge list) blocks.
    for blk in blocks:
        be = [other[i] for i in blk]
        out.append(({v for e in be for v in e}, be))
    # vertices touched by no block: keep as single-vertex blocks.
    used = {v for _, be in out for e in be for v in e}
    for v in verts:
        if v not in used:
            out.append(({v}, []))
    return out
