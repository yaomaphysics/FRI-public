#!/usr/bin/env python3
"""Shared-vertex prefilter for layered enumeration (2026-08-10).

For each layer k >= 3, before enumerating cuts:
  - candidate shared-vertex sets S = k-subsets of V_verts (vertices NOT
    attached by large-component externals, i.e. m==0 modes; SC/S externals
    shareable);
  - keep only S such that G - S (remove S and their incident edges) is
    still CONNECTED.  Physical meaning: shared vertices carry no large
    momentum (vmode = meet of >=2 different-direction cuts = SC/S), so the
    large-momentum flow (H∪C skeleton + external points) must remain
    connected after removing them.  This is a NECESSARY condition — safe.
  - If layer k has no surviving S, layers k+1, k+2, ... are provably empty
    (removing more vertices cannot reconnect a disconnected graph), so the
    layered enumeration terminates.

Usage (library): shared_sets_for_k(verts, adj, V_verts, k) -> list of
frozensets.  Also exposes is_connected_after_removal().
"""
import itertools


def is_connected(verts, adj, alive):
    """True if the induced subgraph on `alive` (vertex set) is connected.
    Single-vertex / empty sets count as connected (trivially)."""
    if len(alive) <= 1:
        return True
    start = next(iter(alive))
    seen = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for w in adj.get(v, ()):
            if w in alive and w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == len(alive)


def is_connected_after_removal(verts, adj, S):
    """G - S connected?  Remove vertices in S and all their incident edges,
    then check connectivity of the remaining induced subgraph (NO aux vertex
    — 2026-08-10: an isolated external point means its large momentum
    cannot flow into the graph, momentum conservation already violated)."""
    alive = set(verts) - set(S)
    return is_connected(verts, adj, alive)


def shared_sets_for_k(verts, adj, V_verts, k):
    """All k-subsets S ⊆ V_verts with G-S connected.  V_verts = vertices
    NOT attached by large-component externals (m==0); SC/S-external
    attachment points are shareable and therefore included in V_verts."""
    if k == 0:
        return [frozenset()] if is_connected_after_removal(verts, adj, ()) else []
    out = []
    for S in itertools.combinations(sorted(V_verts), k):
        if is_connected_after_removal(verts, adj, S):
            out.append(frozenset(S))
    return out


if __name__ == '__main__':
    # quick self-test on DoubleMoth: V_verts = {5,6,7,8,9}
    edges = [[1,6],[1,9],[2,5],[2,8],[3,7],[3,9],[4,5],[4,8],
             [5,6],[5,7],[6,7],[7,8],[8,9]]
    verts = list(range(1, 10))
    adj = {v: set() for v in verts}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    Vv = [5, 6, 7, 8, 9]
    for k in range(0, 6):
        ss = shared_sets_for_k(verts, adj, Vv, k)
        print(f'k={k}: {len(ss)} surviving sets')
        if ss:
            print('   e.g.', sorted(ss)[:4])
    print()
    print('expect: k=0..3 nonempty (k=3 only {5,6,7}), k=4 empty, k=5 empty')
