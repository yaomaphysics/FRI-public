#!/usr/bin/env python3
"""
This script is the core of the module of choosing independent loop momenta of a given region (in wide-angle kinematics). It is based on §3.2 of 2601.22144.

Given a region, for every mode X:

  * Γ_X: the X subgraph, consisting of edges and vertices of the X mode.
  
  * Contracted X-subgraph Γ̃_X (§3.2 of 2601.22144).
    Definition: to obtain from Γ_X, identify ALL its adjacent vertices from other (necessarily harder) mode subgraphs with an auxiliary vertex (shorthanded as "aux").
    Γ̃_X is connected.

  * The number of independent X-loop momenta = the loop number of Γ̃_X = |E(Γ̃_X)| − |V(Γ̃_X)| + 1.

  * Basis lines: decompose Γ̃_X into its 1VI components.
    In each component, delete edges until it becomes a spanning tree (aux included); the DELETED edges' line momenta form a basis of the independent X-loop momenta in this region.
    Note:
      - self-loops are always in the basis;
      - parallel edges (those with the same endpoints): at most one survives in the tree;
      - the choice of which edges to delete is otherwise free.

  * Forced lines F: user can require certain lines to be in the basis.

For sanity check: Σ_X r_X = L = |E| − |V| + 1 for every region.
"""
from collections import defaultdict


# 1VI components of a subgraph.
def find_1vi_blocks(verts, edges):
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


# spanning tree of a connected block (aux included); returns (tree_edge_indices, basis_edge_indices). That is, basis = deleted edges.
def _basis_of_block(block_verts, block_edges, edges2):
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


def _connected_spanning(verts, edge_indices, edges2):
    if not verts:
        return True
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
    for i in edge_indices:
        a, b = edges2[i]
        if a in parent and b in parent and a != b:
            union(a, b)
    root = find(next(iter(verts)))
    return all(find(v) == root for v in verts) # true iff (verts, edges2[i] for i in edge_indices) is connected and spans all of verts.


# Per-mode independent loop momenta of a region: rank r_X and basis blocks per mode.
# Inputs: edges = (u, v) lines (order = index space); em = edge modes (None/(0,0,0) = H); vm = {vertex: join-mode tuple}.
# Returns (results, total_rank, L); results = per-mode {'mode', 'rank', 'blocks'} entries.
def indep_loops(edges, em, vm):
    from primitives import eq
    H = (0, 0, 0)
    em2 = [m if m is not None else H for m in em]

    # group edges by mode
    emodes = defaultdict(list)          # mode -> [edge indices]
    for i, m in enumerate(em2):
        emodes[m].append(i)

    results = []
    total_rank = 0
    for X in sorted(emodes, key=lambda m: (m[0], m[1], m[2])):
        ex = emodes[X]
        vx = {v for v, jm in vm.items() if eq(jm, X)}
        verts2 = set(vx) | {'aux'}
        edges2 = [('aux' if a not in vx else a, 'aux' if b not in vx else b) for (a, b) in edges]
        ge2 = [edges2[i] for i in ex]

        # connected components of γ̃_X (V_X ∪ aux, X-edges)
        parent = {v: v for v in verts2}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for i in ex:
            a, b = edges2[i]
            union(a, b)
        comps = defaultdict(set)
        for v in verts2:
            comps[find(v)].add(v)
        c = len(comps)
        rank = len(ex) - len(verts2) + c
        total_rank += rank

        # 1VI blocks of γ̃_X
        blocks = []
        for (bv, be) in find_1vi_blocks(list(verts2), ge2):
            tree, basis = _basis_of_block(bv, be, ge2)
            blocks.append({'verts': sorted(bv, key=str), 'tree': [ex[j] for j in tree], 'basis': [ex[j] for j in basis]})
        results.append({'mode': X, 'rank': rank, 'blocks': blocks})

    # Sanity check: Σ_X r_X = L for every region.
    L = len(edges) - len({v for e in edges for v in e}) + 1
    return results, total_rank, L


# A concrete basis containing the forced lines F (edge indices); returns (ok, failures, results).
# ok=False iff removing F leaves a block unspanned (failures lists those blocks); results=None if not ok, else as in indep_loops().
def forced_basis(edges, em, vm, F):
    from primitives import eq
    H = (0, 0, 0)
    em2 = [m if m is not None else H for m in em]
    F = set(F)
    emodes = defaultdict(list)
    for i, m in enumerate(em2):
        emodes[m].append(i)

    results = []
    failures = []
    total_rank = 0
    for X in sorted(emodes, key=lambda m: (m[0], m[1], m[2])):
        ex = emodes[X]
        vx = {v for v, jm in vm.items() if eq(jm, X)}
        verts2 = set(vx) | {'aux'}
        edges2 = [('aux' if a not in vx else a, 'aux' if b not in vx else b) for (a, b) in edges]
        ge2 = [edges2[i] for i in ex]

        # connected components of gamma~_X -> rank
        parent = {v: v for v in verts2}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for i in ex:
            a, b = edges2[i]
            union(a, b)
        comps = defaultdict(set)
        for v in verts2:
            comps[find(v)].add(v)
        c = len(comps)
        rank = len(ex) - len(verts2) + c
        total_rank += rank

        blocks = []
        for (bv, be) in find_1vi_blocks(list(verts2), ge2):
            # forced lines enter the basis first; the remaining edges must still span the block
            forced_in = [j for j in be if ex[j] in F]
            rem = [j for j in be if ex[j] not in F]
            if not _connected_spanning(bv, rem, ge2):
                failures.append((X, sorted(bv, key=str), [ex[j] for j in forced_in]))
                continue
            tree, extra = _basis_of_block(bv, rem, ge2)
            # basis = forced lines + non-tree remaining edges; self-loops are always basis lines.
            basis = forced_in + extra       # ge2 indices
            blocks.append({'verts': sorted(bv, key=str), 'tree': [ex[j] for j in tree], 'basis': [ex[j] for j in basis]})
        results.append({'mode': X, 'rank': rank, 'blocks': blocks})

    if failures:
        return False, failures, None
    L = len(edges) - len({v for e in edges for v in e}) + 1
    return True, [], (results, total_rank, L)


# Feasibility of forcing lines F (edge indices) into the independent-loop-momentum basis.
def forced_feasible(edges, em, vm, F):
    ok, failures, _ = forced_basis(edges, em, vm, F)
    return ok, failures
