#!/usr/bin/env python3
"""
indep_loops.py — independent loop momenta + line-momentum parameterization (five-point 2->3).

For every region: the contracted mode subgraphs are split into their 1VI blocks (region_checker.mode_components);
a unit's loop number is the sum of its block cycle ranks, and a concrete basis is the complement of a spanning
tree inside each block (contracted self-loops are always basis lines).  The physical line momenta are then
parameterized by tree+chords on the original graph, with optional forced lines.
"""
import os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import region_checker
from primitives import INF, name


# Display order of modes: H first, then S, then carriers by direction (1, 4, 5, then the 23 pair family).
def _disp_key(m):
    if m[0] == 'H':
        return (0, 0, 0, 0, 0)
    if m[0] == 'S':
        return (1, 0, 0, m[1], 0)
    _, d, n, mem, sm = m
    return (2, d, 10**9 if n == INF else (n if n is not None else 0), -1 if mem is None else mem, sm)


# ---------------------------------------------------------------- basis machinery
# (ported from the 2->2 side's indep_loops, minus the sH/G overlay:
#  every region_checker mode is a lattice mode, so each mode is its own unit)

# Spanning tree of a connected block (aux included); returns (tree_edge_indices, basis_edge_indices) — basis =
# deleted edges.
def _basis_of_block(block_verts, block_edges, edges2, skip=None):
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
        if a == b or (skip and i in skip):  # self-loop / pinned: never a tree edge
            basis.append(i)
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            tree.append(i)
        else:
            basis.append(i)
    if skip:                            # with the pinned lines removed the block must still span
        root = find(next(iter(block_verts)))
        if any(find(v) != root for v in block_verts):
            return None, None
    return tree, basis


def _rank_of(blocks):
    return sum(len(idxs) - len(bv) + 1 for (bv, be, idxs) in blocks)


# Yield (unit_label, blocks) in canonical order — every mode appearing in em/vm is one unit.
def _iter_units(em, vm, edges, verts):
    for mode in sorted(set(em) | set(vm.values()), key=_disp_key):
        blocks = []
        for (bv, be, idxs) in region_checker.mode_components(mode, vm, em, edges, verts):
            blocks.append((bv, list(be), list(idxs)))
        yield mode, blocks


# Per-unit independent loop momenta of a region. Returns (results, total_rank, L): results = [{'mode': unit, 'rank':
# r_X, 'blocks': [{'verts': ..., 'tree': [...], 'basis': [...]}]}] and validates Σ r_X = L.
def indep_loops(edges, em, vm):
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
            out_blocks.append({'verts': sorted(bv, key=str), 'tree': [idxs[j] for j in tree], 'basis': [idxs[j] for j in basis]})
        results.append({'mode': unit, 'rank': r, 'blocks': out_blocks})
    return results, total, L


# Forced-line feasibility over the per-unit 1VI blocks: every forced line must be able to sit in its own block's
# basis (the block must still span without it); completion picks each block's remaining basis lines as usual, with
# the forced lines pinned.  Returns (basis, reason) — basis = original edge indices; reason = None when feasible.
def forced_basis(edges, em, vm, F):
    verts = sorted({v for e in edges for v in e})
    Fset = set(F)
    basis_all = []
    for unit, blocks in _iter_units(em, vm, edges, verts):
        for (bv, be, idxs) in blocks:
            skip = {j for j, ei in enumerate(idxs) if ei in Fset}
            tree, basis = _basis_of_block(bv, list(range(len(be))), be, skip or None)
            if tree is None:
                r = len(idxs) - len(bv) + 1
                u, v = edges[idxs[min(skip)]]
                if r == 0:
                    return None, (f'forced line ({u}, {v}) cannot carry a loop momentum: '
                                  f'its {name(unit)} block has no loop (r = 0)')
                return None, (f'the forced lines in the {name(unit)} block cannot all carry a loop momentum '
                              f'(block r = {r})')
            basis_all += [idxs[j] for j in basis]
    return basis_all, None

# ------------------------------------------- physical momentum parameterization
# Spanning tree of (verts, edge_list) avoiding the skip set; None if no such tree exists (skip set contains a bridge
# / disconnects).
def _spanning_tree(verts, edge_list, skip):
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


# Momentum of every line as a combination of the loop momenta k1..kL and the externals; tree+chords parameterization on
# the ORIGINAL graph: carriers = chords (forced lines first in the k numbering), the other |V|-1 lines form a spanning
# tree (flow along a<b, in - out = p_v). Returns (True, order, momenta, verts) with momenta[ei] = {term: coeff}, or
# (False, reason, None, None) if unfeasible: |carriers| > L, a forced line cannot sit in its mode block's basis
# (see forced_basis), or the complement of the carriers is disconnected.
def edge_momenta(edges, em, vm, carriers, ext_attach):
    verts = sorted({v for e in edges for v in e})
    E = len(edges)
    V = len(verts)
    L = E - V + 1
    F = set(carriers)
    if len(F) > L:
        return False, f'{len(F)} forced lines exceed L = {L}', None, None
    # per-block step: pin the forced lines inside their mode blocks; the rest of the basis is completed per block
    basis_all, reason = forced_basis(edges, em, vm, F)
    if reason is not None:
        return False, reason, None, None
    tree = _spanning_tree(verts, edges, basis_all)
    if tree is None:
        return False, ('the remaining lines do not connect (a forced line is a bridge and cannot carry a loop momentum)'), None, None
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


# in - out = p_v at every vertex, up to Σ_v p_v = 0: k-terms must vanish individually, p-terms must all have equal
# coefficients.
def _check_conservation(edges, verts, momenta, ext_attach):
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


# 'k1 - k2 + p2 + p3' from {term: coeff} (0 if empty).
def fmt_expr(terms):
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


# Print line momenta (k1..kL carriers) and every line's momentum.
def show_edge_momenta(edges, em, vm, carriers, ext_attach, forced=False):
    ok, payload, momenta, verts = edge_momenta(edges, em, vm, carriers, ext_attach)
    if not ok:
        reason = payload
        if forced:
            print(f'  ✗ unfeasible: {reason}')
        else:
            print(f'  ! default basis cannot serve as loop-momentum carriers: {reason}')
        return
    order = payload
    print('  line momenta: ' + ', '.join(f'{f"k{i + 1}"} ↦ {edges[ei]} along {edges[ei][0]}→{edges[ei][1]}' for i, ei in enumerate(order)))
    print('  edge momenta (flow along the displayed direction; p_i = external):')
    for ei in range(len(edges)):
        a, b = edges[ei]
        print(f'    ({a},{b}) {a}→{b}: {fmt_expr(momenta[ei])}')
    if _check_conservation(edges, verts, momenta, ext_attach):
        print('  ✓ momentum conservation at every vertex')
    else:
        print('  ✗ momentum conservation FAILED (bug!)')
    print('  These line momenta can form a loop-momentum basis.')


# A concrete basis: default (F=None, per-mode 1VI-block basis) or forced lines F (physical tree+chords
# parameterization), then every line's momentum in terms of k1..kL and the external momenta.
def show_basis(edges, em, vm, F=None, ext_attach=None):
    if F:
        show_edge_momenta(edges, em, vm, F, ext_attach, forced=True)
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
        basis_all = [j for r in results for b in r['blocks'] for j in b['basis']]
        show_edge_momenta(edges, em, vm, basis_all, ext_attach)
