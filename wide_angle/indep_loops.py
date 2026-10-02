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

For sanity check: Σ_X r_X = L = |E| − |V| + 1 for every region.

Additionally provides the physical line-momentum parameterization (tree+chords, with the option
to force lines into the basis) and its display helpers, used by the interactive browser.
"""
from collections import defaultdict

from primitives import spanning_tree
from read_graph import mode_str


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

# Forced-line feasibility over the per-mode 1VI blocks: every forced line must be able to sit in its own block's
# basis (the block must still span without it); completion picks each block's remaining basis lines as usual, with
# the forced lines pinned.  Returns (basis, reason) — basis = original edge indices; reason = None when feasible.
def forced_basis(edges, em, vm, F):
    from primitives import eq
    H = (0, 0, 0)
    em2 = [m if m is not None else H for m in em]
    Fset = set(F)
    emodes = defaultdict(list)
    for i, m in enumerate(em2):
        emodes[m].append(i)
    basis_all = []
    for X in sorted(emodes, key=lambda m: (m[0], m[1], m[2])):
        ex = emodes[X]
        vx = {v for v, jm in vm.items() if eq(jm, X)}
        verts2 = set(vx) | {'aux'}
        edges2 = [('aux' if a not in vx else a, 'aux' if b not in vx else b) for (a, b) in edges]
        ge2 = [edges2[i] for i in ex]
        for (bv, be) in find_1vi_blocks(list(verts2), ge2):
            skip = {j for j in be if ex[j] in Fset}
            tree, basis = _basis_of_block(bv, be, ge2, skip or None)
            if tree is None:
                r = len(be) - len(bv) + 1
                u, v = edges[ex[min(skip)]]
                if r == 0:
                    return None, (f'forced line ({u}, {v}) cannot carry a loop momentum: '
                                  f'its {_ms(X)} block has no loop (r = 0)')
                return None, (f'the forced lines in the {_ms(X)} block cannot all carry a loop momentum '
                              f'(block r = {r})')
            basis_all += [ex[j] for j in basis]
    return basis_all, None

# ---------------------------------------------------------------- line-momentum parameterization

# Express line momenta as linear combinations of the loop and external momenta.
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
    tree = spanning_tree(verts, edges, basis_all)
    if tree is None:
        return False, ('the remaining lines do not connect (a forced line is a bridge and cannot carry a loop momentum)'), None, None
    chords = [i for i in range(E) if i not in tree]
    # k-numbering: forced lines first (input order), then remaining chords in edge order
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


# Check momentum conservation at each vertex.
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


# Sort key for momentum terms: k's first (numeric), then externals (alphabetical).
def _term_key(t):
    return (0, int(t[1:]), '') if t.startswith('k') else (1, 0, t)


# {term: coeff} -> readable string such as 'k1 - k2 + p2 + p3'.
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


# Print the line-momentum basis and every line's momentum, together with a conservation check.
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
    print('  line momenta: ' + ', '.join(f'k{i+1} ↦ {edges[ei]} along {edges[ei][0]}→{edges[ei][1]}' for i, ei in enumerate(order)))
    print('  edge momenta (flow along the displayed direction; p_i = external):')
    for ei in range(len(edges)):
        a, b = edges[ei]
        print(f'    ({a},{b}) {a}→{b}: {fmt_expr(momenta[ei])}')
    if _check_conservation(edges, verts, momenta, ext_attach):
        print('  ✓ momentum conservation at every vertex')
    else:
        print('  ✗ momentum conservation FAILED (bug!)')
    print('  These line momenta can form a loop-momentum basis.')


# ---------------------------------------------------------------- basis display

# Display name for a mode tuple ('∅' when missing).
def _ms(md):
    return mode_str(md) if md is not None else '∅'


# Show a loop-momentum basis (forced lines or the default per-mode basis) and the momenta.
def show_basis(edges, em, vm, F=None, ext_attach=None):
    if F:
        show_edge_momenta(edges, em, vm, F, ext_attach, forced=True)
    else:
        results, total, L = indep_loops(edges, em, vm)
        print('  independent loop momenta (a concrete basis):')
        for r in results:
            X = r['mode']
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
                print(f'    mode {_ms(X)}: basis = {", ".join(desc)}')
        basis_all = [j for r in results for b in r['blocks'] for j in b['basis']]
        show_edge_momenta(edges, em, vm, basis_all, ext_attach)
