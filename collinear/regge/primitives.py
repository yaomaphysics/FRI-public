#!/usr/bin/env python3
"""primitives.py — zero-judgment base layer for the Regge 2->2 framework.

Mode-string layer over regge_modes (relations, virtuality, meet/join/vee),
mode blocks and scaling, external-momentum helpers, and graph tools.
The region checks (judgment) live in regge_core.py; the enumerators live in
skeleton.py.
"""

from functools import lru_cache
from itertools import combinations

import regge_modes

# ============================ mode string layer ============================

# x marginally softer than y (formula: regge_modes.marginal_softer); the G/sH sentinels and the leg-blind
# legacy symbols have no marginal reading.
@lru_cache(maxsize=None)
def marginally_softer(x1, x2):
    if x1 in ('G', 'sH') or x2 in ('G', 'sH') or x1 in regge_modes.LEG_BLIND or x2 in regge_modes.LEG_BLIND:
        return False
    return regge_modes.marginal_softer(regge_modes.to_mode(x1), regge_modes.to_mode(x2))

# Virtuality of a mode string, from the name alone: 𝒱(S^m C_i^n C_ij) = 2m+n+1 (n=∞ → ∞); G/sH sentinels carry 𝒱 = 1.
@lru_cache(maxsize=None)
def V(key):
    return 1 if key in regge_modes.SPECIAL else regge_modes.to_mode(key).V

@lru_cache(maxsize=None)
def meet(a, b):
    return regge_modes.old_meet(a, b)

@lru_cache(maxsize=None)
def join(a, b):
    return regge_modes.old_join(a, b)

def vee(modes):
    acc = modes[0]
    for m in modes[1:]: acc = join(acc, m)
    return acc

# ============================ mode blocks / scaling ============================

# 1VI blocks of a MULTIGRAPH: self-loops become single-vertex blocks; parallel edges are distinguished by index.
# Returns list of (vertex_set, edge_index_list) pairs.
@lru_cache(maxsize=None)
def _find_1vi_blocks(verts, edges):
    loops = [(i, e) for i, e in enumerate(edges) if e[0] == e[1]]
    other = [(i, e) for i, e in enumerate(edges) if e[0] != e[1]]
    other_map = dict(other)
    out = []
    for i, (a, b) in loops:
        out.append(({a}, [i]))
    adj = {v: [] for v in verts}
    for i, (a, b) in other:
        adj[a].append((b, i))
        adj[b].append((a, i))
    disc = {}
    low = {}
    t = 0
    stack = []
    blocks = []

    def dfs(u, pe):
        nonlocal t
        disc[u] = low[u] = t
        t += 1
        for (w, ei) in adj[u]:
            if ei == pe:
                continue
            if w not in disc:
                stack.append(ei)
                dfs(w, ei)
                low[u] = min(low[u], low[w])
                if low[w] >= disc[u]:
                    blk = set()
                    while True:
                        e = stack.pop()
                        blk.add(e)
                        if e == ei:
                            break
                    blocks.append(blk)
            elif disc[w] < disc[u]:
                stack.append(ei)
                low[u] = min(low[u], disc[w])

    for v in verts:
        if v not in disc:
            dfs(v, -1)
            if stack:
                blocks.append(set(stack))
                stack.clear()
    for blk in blocks:
        be = [other_map[i] for i in blk]
        out.append(({v for e in be for v in e}, blk))
    used = {v for bv, _ in out for v in bv}
    for v in verts:
        if v not in used:
            out.append(({v}, []))
    return out


def find_1vi_blocks(verts, edges):
    # content-keyed memo wrapper: normalise arguments, delegate to the cached impl
    return _find_1vi_blocks(verts if isinstance(verts, frozenset) else frozenset(verts), edges if isinstance(edges, tuple) else tuple(map(tuple, edges)))

# 1VI blocks of the contracted X-subgraph: X edges + X vertices (join-mode), non-X endpoints absorbed into aux.
# Blocks sharing a cut vertex stay SEPARATE — not connected components.  Each block is (vertex_set, contracted_edges, original_edge_indices).
# The ORIGINAL indices are needed because distinct self-loops (1,3) and (7,8) both contract to (aux,aux) and are otherwise indistinguishable.
def mode_components(mode, vm, em, edges, verts):
    gv = {v for v in verts if vm.get(v) == mode}
    ge_idx = [i for i, m in enumerate(em) if m == mode]
    if not gv and not ge_idx:
        return []
    verts2 = set(gv) | {'aux'}
    edges2 = []
    for i in ge_idx:
        a, b = edges[i]
        edges2.append(('aux' if a not in gv else a, 'aux' if b not in gv else b))
    out = []
    for (bv, eidx2) in find_1vi_blocks(verts2, edges2):
        out.append((bv, [edges2[j] for j in eidx2], [ge_idx[j] for j in eidx2]))
    return out

def to_scaling(em):
    return tuple(-V(m) for m in em) + (1,)

# ============================ external momenta ============================

def ext_mode_for(ext_attach, ext_mode, n):
    if ext_mode is not None and n in ext_mode:
        return ext_mode[n]
    return {'p1': 'C1∞C13', 'p2': 'C2∞C24', 'p3': 'C3∞C13', 'p4': 'C4∞C24'}[n]

def ext_m(mode): # m_i from an external-mode string: p_i^2 ~ λ^{m_i}; the strict lightlike case (p_i^2 = 0) is denoted by "None".
    if mode in ('C13', 'C24'):
        return 1
    if mode in ('C1C13', 'C3C13', 'C2C24', 'C4C24'):
        return 2
    if mode in ('C1^2C13', 'C3^2C13', 'C2^2C24', 'C4^2C24'):
        return 3
    return None

def possibly_softest(m1, m2, m3, m4): # analyze the possibly softest mode in the expansion, given the external kinematics p1=C_1^{m1}C13, p2=C_2^{m2}C24, p3=C_3^{m3}C13, p4=C_4^{m4}C24.
    def _min(*xs):
        xs = [x for x in xs if x is not None]
        return min(xs) if xs else None
    def _max(*xs):
        return max(xs) if all(x is not None for x in xs) else None

    finite = [i for i, m in enumerate((m1, m2, m3, m4)) if m is not None]
    if not finite: # if all the external momenta are precisely on shell, p_1^2=p_2^2=p_3^2=p_4^2=0 (kinematics 1), then no softest mode, a cascade of modes appears.
        return []
    if len(finite) == 1: # if there is a unique off-shell external momentum, identify the corresponding m, and the possibly softest mode is of type S^{m + 1}C.
        i, m = finite[0], (m1, m2, m3, m4)[finite[0]]
        return [f'S^{m + 1}C24' if i in (0, 2) else f'S^{m + 1}C13']
    M1 = _min(m1, m3, None if _max(m2, m4) is None else _max(m2, m4) + 1) # in the presence of multiple off-shell external momenta, evaluate the possibly softest mode directly.
    M2 = _min(m2, m4, None if _max(m1, m3) is None else _max(m1, m3) + 1)
    if M1 == M2: # in this case there are two possibly softest modes, otherwise there is precisely one possibly softest mode.
        return [f'S^{M1}C13', f'S^{M2}C24']
    return [f'S^{M1}C13' if M1 > M2 else f'S^{M2}C24']

# ============================ graph helpers ============================

@lru_cache(maxsize=None)
def _connected(vs, es): # checks whether the induced graph (including only edges whose endpoints both lie in vs) is connected; vs: vertex set (may include the auxiliary vertex "aux"); es: edge list.
    if len(vs) <= 1: return True
    adj = {v: set() for v in vs}
    for (a, b) in es:
        if a in vs and b in vs: # goes through those edges whose endpoints both lie in vs.
            adj[a].add(b); adj[b].add(a)
    seen = {next(iter(vs))}; stack = [next(iter(vs))]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in seen: seen.add(u); stack.append(u)
    return seen == set(vs)


def connected(vs, es):
    # content-keyed memo wrapper: normalise arguments, delegate to the cached impl
    return _connected(vs if isinstance(vs, frozenset) else frozenset(vs), es if isinstance(es, tuple) else tuple(map(tuple, es)))

@lru_cache(maxsize=None)
def _is_1vi(vs, es): # checks whether the graph is one-vertex irreducible (deleting any single vertex leaves the graph connected).
    if len(vs) <= 1: return True
    for v in vs:
        rest = vs - {v}
        if not _connected(rest, es): return False
    return True


def is_1vi(vs, es):
    # content-keyed memo wrapper: normalise arguments, delegate to the cached impl
    return _is_1vi(vs if isinstance(vs, frozenset) else frozenset(vs), es if isinstance(es, tuple) else tuple(map(tuple, es)))

@lru_cache(maxsize=None)
def _components(es, verts): # returns a mapping {vertex: component_root} via Union-Find (disjoint set union).
    parent = {v: v for v in verts}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for (a, b) in es:
        if a in parent and b in parent: union(a, b)
    return {v: find(v) for v in verts}


def components(es, verts):
    # content-keyed memo wrapper: normalise arguments, delegate to the cached impl
    return _components(es if isinstance(es, tuple) else tuple(map(tuple, es)), verts if isinstance(verts, frozenset) else frozenset(verts))

# Refined (single-external) cut candidates: connected subsets containing root, inside S_base, avoiding forbidden vertices; empty allowed.
def refined_opts(edges, root, S_base, forbid):
    allowed = {v for v in S_base if v not in forbid}
    out = [frozenset()]
    if root not in allowed:
        return out
    rest = list(allowed - {root})
    for r in range(len(rest) + 1):
        for sub in combinations(rest, r):
            S = frozenset({root} | set(sub))
            if connected(S, edges): out.append(S)
    return out

# Connected supersets of req inside dom.
def conn_sets(req, dom, edges):
    req = set(req); dd = set(dom)
    if not req <= dd:
        return []
    extras = sorted(dd - req); out = []
    for r in range(len(extras) + 1):
        for add in combinations(extras, r):
            C = req | set(add)
            if connected(C, edges):
                out.append(frozenset(C))
    return out

# Supersets of req inside dom; every connected component contains va or vb.
def comp_sets(req, dom, va, vb, edges):
    req = set(req); dd = set(dom)
    if not req <= dd:
        return []
    extras = sorted(dd - req); out = []
    for r in range(len(extras) + 1):
        for add in combinations(extras, r):
            S = req | set(add)
            comp = components(edges, S)
            groups = {}
            for v in S:
                groups.setdefault(comp[v], set()).add(v)
            if all(va in g or vb in g for g in groups.values()):
                out.append(frozenset(S))
    return out


def clear_graph_caches():
    # drop the per-case memo tables (called at case start by the skeleton drivers)
    _connected.cache_clear()
    _is_1vi.cache_clear()
    _components.cache_clear()
    _find_1vi_blocks.cache_clear()
