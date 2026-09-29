#!/usr/bin/env python3
"""regge_indep_loops.py — independent loop momenta per region (Regge).

Port of the wide-angle indep_loops machinery (2026-08-20) to the Regge
framework, with the semihard fix
(2026-09-01):

  * lattice modes (H, C13/C24, refinements, S^m, SC, S^2C, ...): the
    contracted X-subgraph gamma~_X (V_X ∪ aux, all non-X endpoints -> aux);
    r_X = cycle rank of gamma~_X = sum over its 1VI blocks of
    (|E| - |V| + 1); basis = per block, complement of a spanning tree
    (contracted self-loops are always basis lines).  The 1VI blocks are
    exactly primitives.mode_components().

  * semihard (sH): sH is an OVERLAY product, not a lattice mode.  The
    contracted-rank rule overcounts (two parallel sH edges between two G
    vertices both contract to (aux,aux) self-loops -> r = 2, but they are
    ONE bubble).  Fix: the semihard loops are the loops of the subgraph
    gamma_{sH} U gamma_{G} computed on the ORIGINAL graph (no aux).  The
    G subgraph is the necklace string (acyclic), so it has no loops by
    itself and the union's loops are exactly the semihard ones.  One
    combined entry 'sH∪G' is reported; G is not reported separately when
    sH is present (its edges live inside the union).

Sanity: Σ_X r_X = L = |E| − |V| + 1 for every region (checked).

Forced lines F (edge indices): for the forced path the PHYSICAL
parameterization (edge_momenta) is used: |F| <= L and the complement of
F must be connected (a bridge cannot carry a loop momentum), with forced
lines first in the k-numbering.

Independent module: does not modify regge_core.
"""
import os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from primitives import mode_components, find_1vi_blocks

UNION = 'sH∪G'        # combined semihard unit: gamma_sH ∪ gamma_G


# ---------------------------------------------------------------- graph tools


# Spanning tree of a connected block (aux included); returns (tree_edge_indices, basis_edge_indices) — basis = deleted edges.
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


# ------------------------------------------------------------- per-unit blocks
# 1VI blocks of gamma~_mode as (bv, contracted_edges, original_idxs).
def _lattice_blocks(mode, vm, em, edges, verts):
    out = []
    for (bv, be, idxs) in mode_components(mode, vm, em, edges, verts):
        out.append((bv, list(be), list(idxs)))
    return out


# 1VI blocks of gamma_sH ∪ gamma_G on the ORIGINAL graph (no aux).
def _union_blocks(edges, em, verts):
    idx = [i for i, m in enumerate(em) if m in ('sH', 'G')]
    sub = [edges[i] for i in idx]
    sv = sorted({v for e in sub for v in e})
    out = []
    for (bv, be_idx) in find_1vi_blocks(sv, sub):
        out.append((bv, [sub[j] for j in be_idx], [idx[j] for j in be_idx]))
    return out


def _rank_of(blocks):
    return sum(len(idxs) - len(bv) + 1 for (bv, be, idxs) in blocks)


# Yield (unit_label, blocks) in canonical order.  If sH is present, ONE unit 'sH∪G' is yielded (G not repeated);
# otherwise every mode appearing in em/vm is a unit (G included, standard rule).
def _iter_units(em, vm, edges, verts):
    modes = sorted(set(em) | set(vm.values()))
    if any(m == 'sH' for m in modes):
        yield UNION, _union_blocks(edges, em, verts)
        for mode in modes:
            if mode not in ('sH', 'G'):
                yield mode, _lattice_blocks(mode, vm, em, edges, verts)
    else:
        for mode in modes:
            yield mode, _lattice_blocks(mode, vm, em, edges, verts)


# ------------------------------------------------------------------ the count
# Per-unit independent loop momenta of a region: rank r_X and basis blocks per unit.
# Inputs: edges = (u, v) lines (order = index space); em = edge-mode strings; vm = {vertex: mode string}.
# Returns (results, total, L); results = [{'mode', 'rank', 'blocks': [{'verts', 'tree', 'basis'}]}] entries.
def indep_loops(edges, em, vm):
    verts = sorted({v for e in edges for v in e})
    L = len(edges) - len(verts) + 1
    # sanity: Σ r_X = L for every region
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


# ------------------------------------------- physical momentum parameterization
# Spanning tree of (verts, edge_list) avoiding the skip set; None if no such tree exists (bridge / disconnection).
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


# Line momenta as {term: coeff} combinations of the loop momenta k1..kL and the externals; tree+chords parameterization
# on the ORIGINAL graph: carriers = chords (forced lines first in the k numbering), the other |V|-1 lines form a spanning tree.
# A contracted "self-loop" is NOT inert — it is an ordinary line of the original graph and couples to the other lines.
# Returns (True, order, momenta, verts), or (False, reason, None, None) if |carriers| > L or the complement is disconnected.
def edge_momenta(edges, em, vm, carriers, ext_attach):
    verts = sorted({v for e in edges for v in e})
    E = len(edges)
    V = len(verts)
    L = E - V + 1
    F = set(carriers)
    if len(F) > L:
        return False, f'{len(F)} forced lines exceed L = {L}', None, None
    tree = _spanning_tree(verts, edges, F)
    if tree is None:
        return False, ('the remaining lines do not connect (a forced line is a bridge and cannot carry a loop momentum)'), None, None
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


# in - out = p_v at every vertex, up to Σ_v p_v = 0 (momentum conservation of the kinematics):
# k-terms must vanish individually, p-terms must all have equal coefficients.
def _check_conservation(edges, verts, momenta, ext_attach):
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
            print('  ✗ unfeasible: these line momenta do not form a basis for the loop momenta')
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


# ------------------------------------------------------------------ display
# A concrete basis: default per-unit 1VI-block algebraic basis, or forced lines F (physical tree+chords parameterization);
# then every line's momentum in terms of k1..kL and the externals.
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
                print(f'    mode {mode}: basis = {", ".join(desc)}')
        print(f'  Σ |basis| = {total}  vs  L = {L}  {"✓" if total == L else "✗ MISMATCH"}')
        basis_all = [j for r in results for b in r['blocks'] for j in b['basis']]
        show_edge_momenta(edges, em, vm, basis_all, ext_attach)


if __name__ == '__main__':
    # quick self-test on the necklace (sH fix) and the box (k0, G edge)
    from skeleton import fri_regions_full
    cases = [
        ('box k1', [(1, 3), (2, 4), (1, 2), (3, 4)],
         {'p1': 'C1∞C13', 'p2': 'C2∞C24', 'p3': 'C3∞C13', 'p4': 'C4∞C24'}),
        ('box k0', [(1, 3), (2, 4), (1, 2), (3, 4)],
         {'p1': 'C13', 'p2': 'C24', 'p3': 'C13', 'p4': 'C24'}),
        ('necklace k1',
         [(1, 3), (1, 5), (3, 5), (2, 4), (2, 6), (4, 6), (5, 7), (6, 8),
          (7, 8), (7, 8)],
         {'p1': 'C1∞C13', 'p2': 'C2∞C24', 'p3': 'C3∞C13', 'p4': 'C4∞C24'}),
    ]
    ext_attach = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4}
    for name, edges, ext_mode in cases:
        verts = sorted({v for e in edges for v in e})
        regs = fri_regions_full(edges, verts, ext_attach, ext_mode)
        print(f'== {name}: {len(regs)} regions ==')
        for i, (cut13, cut24, cut1, cut3, cut2, cut4, vm, em) in enumerate(regs, 1):
            results, total, L = indep_loops(edges, em, vm)
            ok = '✓' if total == L else '✗ MISMATCH'
            units = ', '.join(f"{r['mode']}:{r['rank']}" for r in results)
            print(f'  R{i}: Σ r_X = {total} vs L = {L} {ok}  [{units}]')
            if total != L:
                for r in results:
                    print(f'      {r["mode"]}: {r["blocks"]}')
