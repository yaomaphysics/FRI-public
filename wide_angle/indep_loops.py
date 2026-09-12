#!/usr/bin/env python3
"""indep_loops.py — independent loop momenta per region (wide-angle).

For a region (edge-mode assignment em + vertex join-modes vm), for every
mode X present in the graph:

  * contracted X-subgraph γ̃_X (paper §3.2):  V_X ∪ {aux},
    where V_X = vertices with join mode X, and the aux vertex absorbs ALL
    vertices of other (necessarily harder) mode subgraphs adjacent to Γ_X.
    X-edges with an endpoint outside V_X map to aux; X-edges with both
    endpoints outside V_X become self-loops (edge count preserved).

  * The number of independent X-loop momenta = the cycle rank of γ̃_X
    (2026-08-20: "the loop number of the contracted X subgraph,
    including the auxiliary vertex"):
        r_X = |E(γ̃_X)| − |V(γ̃_X)| + c(γ̃_X),   c = #connected components.

  * Basis lines: decompose γ̃_X into its 1VI (biconnected) components
    (blocks; blocks share cut vertices, e.g. two blocks sharing aux).
    In each block, delete edges until the block becomes a spanning tree
    (aux included); the DELETED edges' line momenta form a basis of the
    independent X-loop momenta in this region.
      - self-loops are never tree edges -> always in the basis;
      - parallel edges: at most one survives in the tree;
      - the choice of which edges to delete is otherwise free (per-block
        spanning tree, any one).

  * Forced lines F (user requires certain lines to be in the basis):
    feasible iff for every block γ (of the mode of each forced line),
    γ ∖ (F ∩ γ) is connected (contains a spanning tree).  Self-loops are
    forced automatically.

Sanity: Σ_X r_X = L = |E| − |V| + 1 for every region (checked).

Independent module: does not modify any region-analysis code.
"""
from collections import defaultdict


def biconnected_blocks(verts, edges):
    """Biconnected components (blocks) of a MULTIGRAPH (verts, edges) where
    edges is a list of (a, b) pairs; self-loops become single-vertex blocks,
    parallel edges are distinguished by index.  Returns list of
    (vertex_set, edge_index_list)."""
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


def _connected_spanning(verts, edge_indices, edges2):
    """True iff (verts, edges2[i] for i in edge_indices) is connected and
    spans all of verts."""
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
    return all(find(v) == root for v in verts)


def indep_loops(edges, em, vm, rc=None):
    """Per-mode independent loop momenta of a region.

    edges: list of (u, v) internal lines (order = index space).
    em:    list of edge mode tuples (m, n, i); None or (0,0,0) = H.
    vm:    {vertex: join-mode tuple} (may omit isolated vertices).

    Returns list of mode entries:
        {'mode': X, 'rank': r_X,
         'blocks': [{'verts': ..., 'tree': [...], 'basis': [...]}]}
    and validates Σ r_X = L.
    """
    if rc is None:
        import region_checker as rc
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
        vx = {v for v, jm in vm.items() if rc.eq(jm, X)}
        verts2 = set(vx) | {'aux'}
        edges2 = [('aux' if a not in vx else a,
                   'aux' if b not in vx else b) for (a, b) in edges]
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

        # 1VI (biconnected) blocks of γ̃_X
        blocks = []
        for (bv, be) in biconnected_blocks(list(verts2), ge2):
            tree, basis = _basis_of_block(bv, be, ge2)
            blocks.append({'verts': sorted(bv, key=str),
                           'tree': [ex[j] for j in tree],
                           'basis': [ex[j] for j in basis]})
        results.append({'mode': X, 'rank': rank, 'blocks': blocks})

    L = len(edges) - len({v for e in edges for v in e}) + 1
    return results, total_rank, L


def forced_basis(edges, em, vm, F, rc=None):
    """A concrete basis containing the forced lines F (edge indices).

    Returns (ok, failures, results):
      ok       — True iff every block of every mode still admits a spanning
                  tree after its forced lines are removed.
      failures — list of (mode, block_verts, offending_forced_lines) for
                  each block whose remaining edges do not span it.
      results  — None if not ok; otherwise the same structure as
                  indep_loops() (per-mode rank + blocks with tree/basis),
                  with every line in F guaranteed to be in some basis list.

    Construction: in each 1VI block, forced lines are placed in the basis
    first; a spanning tree is then chosen among the REMAINING edges
    (feasibility = such a tree exists); every non-tree remaining edge is
    also a basis line.  Self-loops are always basis lines."""
    if rc is None:
        import region_checker as rc
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
        vx = {v for v, jm in vm.items() if rc.eq(jm, X)}
        verts2 = set(vx) | {'aux'}
        edges2 = [('aux' if a not in vx else a,
                   'aux' if b not in vx else b) for (a, b) in edges]
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
        for (bv, be) in biconnected_blocks(list(verts2), ge2):
            forced_in = [j for j in be if ex[j] in F]
            rem = [j for j in be if ex[j] not in F]
            if not _connected_spanning(bv, rem, ge2):
                failures.append((X, sorted(bv, key=str),
                                 [ex[j] for j in forced_in]))
                continue
            tree, extra = _basis_of_block(bv, rem, ge2)
            basis = forced_in + extra       # ge2 indices
            blocks.append({'verts': sorted(bv, key=str),
                           'tree': [ex[j] for j in tree],
                           'basis': [ex[j] for j in basis]})
        results.append({'mode': X, 'rank': rank, 'blocks': blocks})

    if failures:
        return False, failures, None
    L = len(edges) - len({v for e in edges for v in e}) + 1
    return True, [], (results, total_rank, L)


def forced_feasible(edges, em, vm, F, rc=None):
    """Feasibility of forcing lines F (edge indices) into the independent-
    loop-momentum basis.  Returns (ok, failures) where failures lists
    (mode, block_verts, offending_forced_lines) for each block whose
    remaining edges do not span it."""
    ok, failures, _ = forced_basis(edges, em, vm, F, rc)
    return ok, failures
