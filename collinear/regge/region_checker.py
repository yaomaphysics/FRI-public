'''
This script constructs the "judgment module" (one of the two cores of the main code), which filters regions from those cut-overlaying results.
Similar to the wide-angle case, the filter is built on the following subgraph requirements:
    (A) fundamental pattern (momentum conservation, Coleman--Norton interpretation),
    (B) connectivity requirement (First Connectivity Theorem),
    (C) infrared-compatibility requirement.
These requirements as a whole, can be seen as the necessary and sufficient condition for facet region.
We note that these requirements are realized differently from the wide-angle case.

Another key difference from the wide-angle kinematics is that, the Glauber and semihard subgraphs are not from overlaying the cuts; they are identified immediately after the overlay of cuts.
'''

from itertools import combinations
from functools import lru_cache
import os
from primitives import (V, marginally_softer, harder_or_eq, meet, join, vee, connected, is_1vi, components, mode_components, find_1vi_blocks, ext_mode_for)

J13_FAM = {'C13', 'C1C13', 'C3C13', 'C1^2C13', 'C3^2C13'}
J24_FAM = {'C24', 'C2C24', 'C4C24', 'C2^2C24', 'C4^2C24'}

# S-power of a mode string ('S' -> 1, 'S^m...' -> m, everything else -> 0).
@lru_cache(maxsize=None)
def _m_of(md):
    if md == 'S':
        return 1
    if isinstance(md, str) and md.startswith('S^'):
        num = ''
        for ch in md[2:]:
            if ch.isdigit():
                num += ch
            else:
                break
        return int(num) if num else 1
    return 0

# ================= GLAUBER / SEMIHARD IDENTIFICATION ==================
# Identifying the Glauber and semihard subgraphs:
# glauber_adjust finds the Glauber edges and cut vertices; _glauber_semihard turns the components attached to no external leg into semihard.

# One-pass helpers for the Glauber search (used by glauber_adjust / _glauber_cut_vertices).
# Both are content-keyed caches; dropped per case via clear_adjust_caches().

# Bridges of the multigraph (verts, edge_list): {edge position: frozenset(vertex side)}.
def bridges_with_side(verts, edge_list):
    adj = {v: [] for v in verts}
    for k, (a, b) in enumerate(edge_list):
        adj.setdefault(a, []).append((b, k))
        adj.setdefault(b, []).append((a, k))
    disc = {}
    low = {}
    t = [0]
    out = {}

    def dfs(u, pe):
        disc[u] = low[u] = t[0]
        t[0] += 1
        sub = {u}
        for (w, k) in adj.get(u, ()):
            if k == pe:
                continue
            if w not in disc:
                csub = dfs(w, k)
                sub |= csub
                if low[w] < low[u]:
                    low[u] = low[w]
                if low[w] > disc[u]:
                    out[k] = frozenset(csub)
            else:
                if disc[w] < low[u]:
                    low[u] = disc[w]
        return sub

    for v in sorted(verts, key=str):
        if v not in disc:
            dfs(v, None)
    return out


@lru_cache(maxsize=None)
def _bridges_cached(big_verts, big_edges):
    return bridges_with_side(big_verts, list(big_edges))


# One DFS giving, per vertex: subtree tag-mask and separator children (child, child subtree mask).
@lru_cache(maxsize=None)
def _sep_info_cached(big_verts, big_edges, tag_key):
    vset = set(big_verts)
    ptags = {}
    for v, b in tag_key:
        ptags[v] = ptags.get(v, 0) | b
    adj = {v: [] for v in vset}
    for k, (a, b) in enumerate(big_edges):
        adj.setdefault(a, []).append((b, k))
        adj.setdefault(b, []).append((a, k))
    disc = {}
    low = {}
    t = [0]
    sub_mask = {}
    sep_children = {}

    def dfs(u, pe):
        disc[u] = low[u] = t[0]
        t[0] += 1
        m = ptags.get(u, 0)
        sc = []
        for (w, k) in adj.get(u, ()):
            if k == pe:
                continue
            if w not in disc:
                cm = dfs(w, k)
                m |= cm
                if low[w] < low[u]:
                    low[u] = low[w]
                if low[w] >= disc[u]:
                    sc.append((w, cm))
            else:
                if disc[w] < low[u]:
                    low[u] = disc[w]
        sub_mask[u] = m
        sep_children[u] = tuple(sc)
        return m

    for v in sorted(vset, key=str):
        if v not in disc:
            dfs(v, None)
    return sub_mask, sep_children


# Drop the per-case memo tables of the two helpers above (called next to clear_graph_caches()).
def clear_adjust_caches():
    _bridges_cached.cache_clear()
    _sep_info_cached.cache_clear()


# Identify the Glauber and semihard subgraphs; returns the adjusted (em, vm), or None if the region is discarded.
def glauber_adjust(edges, em, vm, ext_attach, ext_mode=None):
    em = list(em)
    verts = frozenset({v for e in edges for v in e} | set(ext_attach.values())) # frozenset: cache-key form for the memoised graph kernels
    # small-momentum subgraph: every S^m ... mode with m >= 1
    small_edges = {e for e, m in zip(edges, em) if _m_of(m) >= 1}
    small_verts = {v for v in verts if _m_of(vm.get(v)) >= 1}
    big_edges = tuple(e for e in edges if e not in small_edges) # tuple: cache-key form for the memoised kernels
    big_verts = verts - small_verts
    # the large-momentum subgraph (hard + jets + Glauber + semihard) must be connected
    if not connected(big_verts, big_edges):
        return None
    p1, p3 = ext_attach['p1'], ext_attach['p3']
    p2, p4 = ext_attach['p2'], ext_attach['p4']
    big_pairs = [(j, e) for j, e in enumerate(edges) if e not in small_edges]
    # one bridges pass: separating hard edges are the Glauber propagators
    big_idx = [j for j, _ in big_pairs]
    brid = _bridges_cached(frozenset(big_verts), tuple(big_edges))
    g_edges = []
    for k in sorted(brid):
        S = set(brid[k])
        T = set(big_verts) - S
        if not ((p1 in S and p3 in S) or (p1 in T and p3 in T)):
            continue
        if not ((p2 in S and p4 in S) or (p2 in T and p4 in T)):
            continue
        side1 = S if p1 in S else T
        if p2 in side1 or p4 in side1:
            continue
        i = big_idx[k]
        if em[i] != 'H':  # a non-hard separating edge discards the region
            return None
        g_edges.append(i)
    for i in g_edges:
        em[i] = 'G'
    vm = dict(vm)
    # one scan fills the per-vertex momentum lists
    acc_map = {v: [] for v in verts}
    for i, (a, b) in enumerate(edges):
        acc_map[a].append(em[i])
        if b != a: acc_map[b].append(em[i])
    for v in verts:
        accs = acc_map[v]
        for n, vv in ext_attach.items():
            if vv == v: accs.append(ext_mode_for(ext_mode, n))
        vm[v] = vee(accs) if accs else 'H'
    # separating cut vertices become G; then the semihard assignment
    vm2 = _glauber_cut_vertices(edges, em, vm, ext_attach, big_edges, big_verts, verts)
    if vm2 is not None:
        vm = vm2
        res = _glauber_semihard(edges, em, vm, [j for j, _ in big_pairs], big_verts, verts, ext_attach, ext_mode)
        if res is None:
            return None
        em, vm = res
    return em, vm

# Turn the (large-momentum minus Glauber) components attached to no external leg into sH; every sH component must touch exactly 2 Glauber vertices.  Returns the adjusted (em, vm), or None if the check fails.
def _glauber_semihard(edges, em, vm, big_idx, big_verts, verts, ext_attach, ext_mode):
    em = list(em)
    Gset = {v for v in big_verts if vm.get(v) == 'G'}
    if len(Gset) < 2:
        return em, vm
    changed = False
    # In (large-momentum subgraph minus Glauber), vertices and edges are independent: an edge stays unless it is a Glauber edge itself.
    # Components attached to none of the external legs (including edges with both endpoints removed) become sH.
    rest = big_verts - Gset
    # vertex components: connected pieces of the rest-vertex graph
    adj_rest = {v: [] for v in rest}
    for i in big_idx:
        a, b = edges[i]
        if a in rest and b in rest:
            adj_rest[a].append(b)
            adj_rest[b].append(a)
    seen = set()
    for v0 in rest:
        if v0 in seen:
            continue
        comp = {v0}
        stack = [v0]
        while stack:
            x = stack.pop()
            for w in adj_rest.get(x, ()):
                if w not in comp:
                    comp.add(w)
                    stack.append(w)
        seen |= comp
        # external legs attached to this component (endpoint inside it)
        att = [nm for nm, vv in ext_attach.items() if vv in comp]
        if not att:
            for i in big_idx:
                a, b = edges[i]
                if (a in comp or b in comp) and em[i] != 'sH':
                    em[i] = 'sH'
                    changed = True
    # edges with both endpoints removed: they attach to no external leg
    for i in big_idx:
        if em[i] == 'G':
            continue
        a, b = edges[i]
        if a not in rest and b not in rest and em[i] != 'sH':
            em[i] = 'sH'
            changed = True
    # Check: every sH component must be adjacent to exactly 2 Glauber vertices.
    # sH components are the 1VI blocks of the contracted semihard subgraph: the Glauber vertices are identified with one auxiliary vertex.
    sH_idx = [i for i in big_idx if em[i] == 'sH']
    if sH_idx:
        verts2 = {'aux'}
        edges2 = []
        for i in sH_idx:
            a, b = edges[i]
            if a not in Gset:
                verts2.add(a)
            else:
                a = 'aux'
            if b not in Gset:
                verts2.add(b)
            else:
                b = 'aux'
            edges2.append((a, b))
        for (_bv, eidx2) in find_1vi_blocks(verts2, tuple(edges2)):
            adjG = set()
            for j in eidx2:
                a, b = edges[sH_idx[j]]
                if a in Gset:
                    adjG.add(a)
                if b in Gset:
                    adjG.add(b)
            if len(adjG) != 2:
                return None
    if not changed:
        return em, vm
    vm2 = {}
    # one scan fills the per-vertex momentum lists
    acc_map = {v: [] for v in verts}
    for i, (a, b) in enumerate(edges):
        acc_map[a].append(em[i])
        if b != a: acc_map[b].append(em[i])
    for v in verts:
        if v in Gset:
            vm2[v] = 'G'
            continue
        accs = acc_map[v]
        for n, vv in ext_attach.items():
            if vv == v: accs.append(ext_mode_for(ext_mode, n))
        vm2[v] = vee(accs) if accs else 'H'
    return em, vm2

# Lift cut vertices that separate p1,p3 from p2,p4 to Glauber
def _glauber_cut_vertices(edges, em, vm, ext_attach, big_edges, big_verts, verts):
    p1, p3 = ext_attach['p1'], ext_attach['p3']
    p2, p4 = ext_attach['p2'], ext_attach['p4']
    biggest = set(big_verts)
    tag_key = []
    for p, b in ((p1, 1), (p3, 2), (p2, 4), (p4, 8)):
        if p in biggest:
            tag_key.append((p, b))
    _sub_mask, sep_children = _sep_info_cached(frozenset(big_verts), tuple(big_edges), tuple(tag_key))
    ptags = {}
    for v, b in tag_key:
        ptags[v] = ptags.get(v, 0) | b
    all_mask = 0
    for v, b in tag_key:
        all_mask |= b
    B13 = 0
    B24 = 0
    for p, b in ((p1, 1), (p3, 2), (p2, 4), (p4, 8)):
        if b & 3:
            B13 |= b if p in biggest else 0
        else:
            B24 |= b if p in biggest else 0
    changed = False
    vm2 = dict(vm)
    for v in big_verts:
        s13 = [p for p in (p1, p3) if p != v]
        s24 = [p for p in (p2, p4) if p != v]
        if not s13 or not s24:
            if (v == p1 == p3 or v == p2 == p4):  # p1 and p3 (or, p2 and p4) enter the same vertex -- must be Glauber
                inc = [em[i] for i, (a, b) in enumerate(edges) if (a == v or b == v) and ((a, b) in big_edges or (b, a) in big_edges)]
                if inc:
                    vm2[v] = 'G'
                    changed = True
            continue
        vmask = ptags.get(v, 0)
        total = all_mask & (0b1111 ^ vmask)
        subs = sep_children[v]
        sum_sub = 0
        for (_w, sm) in subs:
            sum_sub |= sm
        rem = total & ~sum_sub
        ok = True
        for (_w, sm) in subs:
            if (sm & B13) and (sm & B24):
                ok = False
                break
        if ok and (rem & B13) and (rem & B24):
            ok = False
        if ok:
            vm2[v] = 'G'
            changed = True
    return vm2 if changed else None

# =========================== FUNDAMENTAL PATTERN ===========================


# Momentum conservation per vertex: incident momenta split in two nonempty groups of equal ∨-join.
def momentum_ok(edges, em, vm, ext_attach, ext_mode=None):
    verts = set(v for e in edges for v in e) | set(ext_attach.values())
    # one scan builds the per-vertex incident modes (externals grouped separately below)
    inc_map = {v: [] for v in verts}
    for i, (a, b) in enumerate(edges):
        inc_map[a].append(em[i])
        if b != a: inc_map[b].append(em[i])
    ext_map = {}
    for n, vv in ext_attach.items():
        ext_map.setdefault(vv, []).append(ext_mode_for(ext_mode, n))
    for v in verts:
        inc = inc_map[v] + ext_map.get(v, [])
        if vm.get(v) in ('G', 'sH'):  # G/sH vertex: skip (transverse)
            continue
        if any(m in ('G', 'sH') for m in inc):  # touches a G/sH line: skip (transverse)
            continue
        n = len(inc)
        if n < 2:
            return False
        if n == 2:  # equality of the two line modes
            a, b = inc
            if vee([a]) == vee([b]):
                continue
            return False
        ok = False
        for r in range(1, n):
            for A in combinations(range(n), r):
                B = [i for i in range(n) if i not in A]
                if vee([inc[i] for i in A]) == vee([inc[i] for i in B]):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            return False
    return True


# Every connected component of the jet graph (vertices+edges) contains at least one of its external vertices
def jet_components_ok(jv, je, ext_verts):
    if not jv:
        return True
    comp = components(list(je), jv)
    groups = {}
    for v in jv:
        groups.setdefault(comp[v], set()).add(v)
    for g in groups.values():
        if not (g & ext_verts):
            return False
    return True


# k0 mojetic (two parts): each contracted jet is 1VI, and no H∪J13 / H∪J24 cut vertex separates its family.
def mojetic_k0_ok(edges, em, vm, fam13, fam24, ext_attach):
    # part 1: contract the jet — H/G adjacency becomes aux1, the two externals aux2 — and require 1VI
    def jet_ok(fam, ext_names):
        fv = {v for v in vm if vm[v] in fam}
        fe = [e for e, m in zip(edges, em) if m in fam]
        # H/G vertices adjacent to the jet: jet-edge endpoint, or touching a jet vertex via any edge
        adj = set()
        for v in vm:
            if vm[v] in ('H', 'G') and v not in fv:
                if any(v == a or v == b for (a, b) in fe) or any((v == a or v == b) and (a in fv or b in fv) for (a, b) in edges):
                    adj.add(v)
        vs = set(fv) | {'aux1', 'aux2'}
        es = set()
        for (a, b) in fe:
            es.add(('aux1' if a in adj or a not in vs else a, 'aux1' if b in adj or b not in vs else b))
        for n in ext_names:
            vv = ext_attach[n]
            vv2 = 'aux2' if vv in adj or vv not in vs else vv
            es.add((vv2, 'aux2'))
        es.add(('aux1', 'aux2'))
        return is_1vi(vs, es)

    if not jet_ok(fam13, ('p1', 'p3')):
        return False
    if not jet_ok(fam24, ('p2', 'p4')):
        return False
    # part 2: no H∪J cut vertex may separate the family from the rest
    hv = {v for v in vm if vm[v] == 'H'}
    he = [(a, b) for (a, b), m in zip(edges, em) if m == 'H']
    if hv:
        def fam_ok(fam, ext1, ext2):
            fv = {v for v in vm if vm[v] in fam}
            fe = [(a, b) for (a, b), m in zip(edges, em) if m in fam]
            gv = hv | fv
            ge = he + fe
            pa, pb = ext_attach[ext1], ext_attach[ext2]
            for v in gv:
                rest = [(a, b) for (a, b) in ge if a != v and b != v]
                comp = components(rest, gv)
                c_fam = {comp[p] for p in (pa, pb) if p in gv}
                if not c_fam:
                    continue
                # separated iff some vertex lies outside every family-survivor component (family's own vertices count as survivors)
                for x in gv:
                    if x == v or x == pa or x == pb:
                        continue
                    if comp[x] not in c_fam:
                        return False
            return True
        if not fam_ok(fam13, 'p1', 'p3') or not fam_ok(fam24, 'p2', 'p4'):
            return False
    return True


# Mojetic check: contract the component's externals and all adjacent outside vertices to one aux vertex; the result must be 1VI.
def mojetic_ok(hv, he, jv, je, gv, edges, ext_attach, ext_names):
    vs = set(hv) | set(jv) | {'aux'}
    es = set(he) | set(je)
    adj = {v for v in gv if v not in vs and any((v == a and b in jv) or (v == b and a in jv) for (a, b) in edges)}
    if adj:
        es = {('aux' if a in adj else a, 'aux' if b in adj else b) for (a, b) in es}
    for n in ext_names:
        vv = ext_attach[n]
        if vv in vs:
            es.add((vv, 'aux'))
    return is_1vi(vs, es)


# ============================ CONNECTIVITY ============================
# This part constructs the subgraph requirements subject to the First Connectivity Theorem (certain unions of mode subgraphs must be connected, see theorem 5.1 of 2601.22144 for the wide-angle scenario).

# no C13/C24 scaleless island: a C13/C24 component whose ADJACENT modes are all softer (𝒱 > 𝒱(C13) = 1) is rejected (First Connectivity Theorem).
# Components = connected components of the exact-mode subgraph; adjacency = the modes of the edges leaving the component.
def c13_c24_island_ok(edges, em, vm, verts):
    for mode in ('C13', 'C24'):
        gv = frozenset(v for v in verts if vm.get(v) == mode)
        if not gv:
            continue
        cmap = components(tuple((a, b) for i, (a, b) in enumerate(edges) if em[i] == mode), gv)
        for root in sorted(set(cmap.values())):
            comp = frozenset(v for v in gv if cmap[v] == root)
            adjacent = [em[i] for i, (a, b) in enumerate(edges) if (a in comp) != (b in comp)]
            if adjacent and all(V(m) > V(mode) for m in adjacent):
                return False
    return True


# ============ INFRARED COMPATIBILITY: RELEVANCE AND RULES =============


# Meet-of-two: relevant to two confirmed components whose modes meet (∧) to the block's mode.
def meet_of_two_confirms(blk, mode, comps, confirmed, edges, em, vm, _lc=None):
    for mode1, lst1 in comps.items():
        for i1, comp1 in enumerate(lst1):
            if not confirmed.get((mode1, i1)):
                continue
            if not _relevant_lc(_lc, blk, comp1, mode1, edges, em, vm, mode):
                continue
            for mode2, lst2 in comps.items():
                for i2, comp2 in enumerate(lst2):
                    if not confirmed.get((mode2, i2)):
                        continue
                    if not _relevant_lc(_lc, blk, comp2, mode2, edges, em, vm, mode):
                        continue
                    if meet(mode1, mode2) == mode:
                        return True
    return False


# Local-cache helpers for one ir_ok call: (edges, em, vm, comps) are fixed within a call, so
# pure sub-results can be reused across the fixpoint rounds via an id-keyed dict.
_LC_MISS = object()


def _relevant_lc(_lc, blk, comp, dst_mode, edges, em, vm, sc_mode):
    if _lc is None:
        return relevant(blk, comp, dst_mode, edges, em, vm, sc_mode)
    key = ('rel', id(blk), id(comp), dst_mode, sc_mode)
    v = _lc.get(key, _LC_MISS)
    if v is _LC_MISS:
        v = relevant(blk, comp, dst_mode, edges, em, vm, sc_mode)
        _lc[key] = v
    return v


# Messenger (SC/S²C condition): relevant to all of `own` and (if given) at least one of `other`, with >=1 confirmed target.
def messenger_confirms(blk, mode, own, other, comps, confirmed, edges, em, vm, _lc=None):
    adj = {}
    for m in own + other:
        for i, comp in enumerate(comps[m]):
            if _relevant_lc(_lc, blk, comp, m, edges, em, vm, mode):
                adj.setdefault(m, set()).add(i)
    if not all(m in adj for m in own):
        return False
    if other and not any(m in adj for m in other):
        return False
    for m in own + other:
        for i in adj.get(m, ()):
            if confirmed.get((m, i)):
                return True
    return False


# Relevance of an SC/S block to a component: marginally softer mode plus a non-increasing path — not geometric adjacency.
def relevant(blk, comp, dst_mode, edges, em, vm, sc_mode):
    if not marginally_softer(sc_mode, dst_mode):
        return False
    bv, be, *rest = blk
    sidx = rest[0] if rest else None
    cv, _, *_ = comp
    src = {v for v in bv if v != 'aux'}
    tgt = {v for v in cv if v != 'aux'}
    if not tgt:
        return False
    # direct contact: a γ1 edge incident with a γ2 vertex is a length-1 path
    # membership by original edge index (self-loops both contract to (aux,aux))
    for ei, (a, b) in enumerate(edges):
        if em[ei] != sc_mode:
            continue
        if sidx is not None:
            if ei not in sidx:
                continue
        else:
            a2 = a if vm.get(a) == sc_mode else 'aux'
            b2 = b if vm.get(b) == sc_mode else 'aux'
            if (a2, b2) not in be and (b2, a2) not in be:
                continue
        if a in tgt or b in tgt:
            return True
    # external-momentum component: contact = the external vertex itself in the target
    if src and len(be) == 1 and 'aux' in be[0]:
        v_ext = next(iter(src))
        if v_ext in tgt:
            return True
    if not src:
        # pure self-loop block: its edge endpoints are the path entry points
        for ei, (a, b) in enumerate(edges):
            if em[ei] != sc_mode:
                continue
            if sidx is not None:
                if ei not in sidx:
                    continue
            else:
                a2 = a if vm.get(a) == sc_mode else 'aux'
                b2 = b if vm.get(b) == sc_mode else 'aux'
                if (a2, b2) not in be and (b2, a2) not in be:
                    continue
            if a != 'aux': src.add(a)
            if b != 'aux': src.add(b)
    if not src:
        return False
    adj = {v: [] for v in vm}
    for i, (a, b) in enumerate(edges):
        adj[a].append((b, i)); adj[b].append((a, i))
    # monotone walk, state = (vertex, last V): V never increases, starting from the vertex's own mode (not V(sc_mode))
    # G/H/sH cannot conduct IR info — not even as a path start
    src = {v for v in src if vm.get(v) not in ('G', 'H', 'sH')}
    if not src:
        return False
    seen = set(src)
    stack = [(v, V(vm.get(v, 'H'))) for v in src]
    while stack:
        v, last_V = stack.pop()
        for w, ei in adj[v]:
            if w in seen:
                continue
            if vm.get(w) in ('G', 'H', 'sH'):  # no passing through G/H/sH: no longitudinal info / off-shell absorber / necklace-internal
                continue
            e_V = V(em[ei])
            if e_V > last_V:  # vertex -> edge step: V non-increasing (edge not softer than the vertex it leaves)
                continue
            w_V = V(vm.get(w, 'H'))
            if w_V > e_V:  # edge -> vertex step: arrived vertex must not be softer than the edge it arrives through
                continue
            if w in tgt:
                return True
            seen.add(w)
            stack.append((w, w_V))
    return False


# SC hidden path (Regge-only): an SC relevant to two distinct pair-mode components and one same-side refined component conducts confirmation between them (the SC itself is not confirmed).
def sc_hidden_path_confirms(blk, i, comps, confirmed, edges, em, vm, ext_attach, ext_mode, sc_mode, pair_mode, ext_a, ext_b, fam_modes, ext_a_fam, ext_b_fam, _lc=None):
    va = ext_attach[ext_a]
    vb = ext_attach[ext_b]
    ma = ext_mode_for(ext_mode, ext_a)
    mb = ext_mode_for(ext_mode, ext_b)
    # C_a^{m_a}·pair / C_b^{m_b}·pair with m >= 0 (incl. ∞ = lightlike).
    if ma not in ext_a_fam or mb not in ext_b_fam:
        return False
    # the receiver must still pass the port requirement (conduction is not an exemption)
    if not _hp_receiver_port_ok(blk, pair_mode, edges, em, vm, ext_attach, ext_mode, confirmed, comps, _lc=_lc):
        return False
    cv = {v for v in blk[0] if v != 'aux'}
    has_a = va in cv
    has_b = vb in cv
    if has_a == has_b:
        # must be attached by exactly one of ext_a, ext_b (distinct comps)
        return False
    partner = None
    for j, cj in enumerate(comps[pair_mode]):
        if j == i:
            continue
        cjv = {v for v in cj[0] if v != 'aux'}
        if has_a and vb in cjv:
            partner = j
            break
        if has_b and va in cjv:
            partner = j
            break
    if partner is None:
        return False
    if not confirmed.get((pair_mode, partner)):
        return False
    # one SC component relevant to blk, to the partner, AND to at least one fam_modes component
    for sblk in comps[sc_mode]:
        if not _relevant_lc(_lc, sblk, blk, pair_mode, edges, em, vm, sc_mode):
            continue
        if not _relevant_lc(_lc, sblk, comps[pair_mode][partner], pair_mode, edges, em, vm, sc_mode):
            continue
        ok_fam = False
        for fm in fam_modes:
            for cblk in comps[fm]:
                if _relevant_lc(_lc, sblk, cblk, fm, edges, em, vm, sc_mode):
                    ok_fam = True
                    break
            if ok_fam:
                break
        if ok_fam:
            return True
    return False


# Relevance to an H block: like relevant(), but H vertices are traversable and the path may end at the H target.
def h_relevant(comp, tgt, edges, em, vm, extvs, sc_mode, start_V=None, cross_idx=(), allow_ext_mid=False):
    bv, be, *rest = comp
    sidx = rest[0] if rest else None
    src = {v for v in bv if v != 'aux'}
    if not src:
        for ei, (a, b) in enumerate(edges):
            if em[ei] != sc_mode:
                continue
            if sidx is not None:
                if ei not in sidx:
                    continue
            else:
                a2 = a if vm.get(a) == sc_mode else 'aux'
                b2 = b if vm.get(b) == sc_mode else 'aux'
                if (a2, b2) not in be and (b2, a2) not in be:
                    continue
            if a != 'aux': src.add(a)
            if b != 'aux': src.add(b)
    if not src:
        return False
    if any(v in tgt for v in src):
        return True
    # direct contact: an edge of the source block incident with tgt (crossing edges excluded)
    for ei, (a, b) in enumerate(edges):
        if ei in cross_idx:
            continue
        if em[ei] != sc_mode:
            continue
        if sidx is not None:
            if ei not in sidx:
                continue
        else:
            a2 = a if vm.get(a) == sc_mode else 'aux'
            b2 = b if vm.get(b) == sc_mode else 'aux'
            if (a2, b2) not in be and (b2, a2) not in be:
                continue
        if a in tgt or b in tgt:
            return True
    adj = {v: [] for v in vm}
    for i, (a, b) in enumerate(edges):
        adj[a].append((b, i)); adj[b].append((a, i))
    # BFS (crossing edges excluded; G/sH blocked; externals are sinks unless H-mode / allow_ext_mid)
    seen = set(src)
    stack = [(v, start_V if start_V is not None else V(vm.get(v, 'H'))) for v in src]
    while stack:
        v, last_V = stack.pop()
        for w, ei in adj[v]:
            if ei in cross_idx:
                continue
            if w in seen:
                continue
            e_V = V(em[ei])
            if e_V > last_V:
                continue
            w_V = V(vm.get(w, 'H'))
            if w_V > e_V:
                continue
            if w in tgt:
                return True
            # the pass-through restrictions apply only to intermediate vertices — the path may END at an external or H vertex of the target
            if vm.get(w) in ('G', 'sH'):
                continue
            if w in extvs:
                # external-momentum vertices are sources/sinks mid-path — except H-mode ones (the hard blob conducts) and allow_ext_mid flows
                if not allow_ext_mid and vm.get(w) != 'H':
                    continue
            # H vertices ARE traversable mid-path: hard momentum flows freely inside the hard blob (G/sH remain blocked)
            seen.add(w)
            stack.append((w, w_V))
    return False


# H component IR compatibility: total inflow — externals in/reaching the block + line momenta of confirmed components — must ∨ to H.
def h_comp_confirmed(blk, comps, confirmed, edges, em, vm, ext_attach, ext_mode):
    verts = [v for v in blk[0] if v != 'aux']
    if not verts:
        return False
    if len(verts) == 1:  # single-vertex H component: trivially compatible
        return True
    piece = set(verts)
    extvs = set(ext_attach.values())
    inflows = []
    for e in ext_attach:
        vv = ext_attach[e]
        m_ext = ext_mode_for(ext_mode, e)
        if vv in piece:
            inflows.append(m_ext)
        else:
            ext_comp = ({vv, 'aux'}, [(vv, 'aux')])
            # starts at the external vertex with its own mode
            if h_relevant(ext_comp, piece, edges, em, vm, extvs, m_ext, start_V=V(vm.get(vv, 'H'))):
                inflows.append(m_ext)
    # line momentum of a confirmed component: enters the blob and flows within it
    for j, (a, b) in enumerate(edges):
        m = em[j]
        if m in ('H', 'G', 'sH'):
            continue
        for ci, (sv, se, *sidx) in enumerate(comps.get(m, ())):
            if not confirmed.get((m, ci)):
                continue
            if sidx and j in sidx[0]:
                if h_relevant((sv, se, *sidx), piece, edges, em, vm, extvs, m, allow_ext_mid=True):
                    inflows.append(m)
                break
    if not inflows:
        return False
    acc = inflows[0]
    for m2 in inflows[1:]:
        acc = join(acc, m2)
    return acc == 'H'


# Condition 1: inflow momenta (external or from confirmed components) enter the block with join == mode, and a third port remains.
def cond1_confirms(blk, mode, comps, confirmed, edges, em, vm, ext_attach, ext_mode, _lc=None):
    # entries: strict monotone entry walk (edge modes checked, first-touch stop); a single entry of the mode is allowed too
    realV = {v for v in blk[0] if v != 'aux'}
    if not realV:
        return False
    # which real vertices a starting momentum first touches (monotone walk); cached per ir_ok call
    def _ew_impl(start_vs):
        src = {v for v in start_vs if vm.get(v) not in ('G', 'H', 'sH')}
        if not src:
            return set()
        adj = {v: [] for v in vm}
        for i, (a, b) in enumerate(edges):
            adj[a].append((b, i)); adj[b].append((a, i))
        seen = set(src)
        stack = [(v, V(vm.get(v, 'H'))) for v in src]
        touch = set()
        while stack:
            v, last = stack.pop()
            if v in realV:
                touch.add(v); continue
            for w, ei in adj[v]:
                if w in seen:
                    continue
                if vm.get(w) in ('G', 'H', 'sH'):
                    continue
                eV = V(em[ei])
                if eV > last:
                    continue
                wV = V(vm.get(w, 'H'))
                if wV > eV:
                    continue
                seen.add(w); stack.append((w, wV))
        return touch

    def entry_walk(start_vs):
        if _lc is None:
            return _ew_impl(start_vs)
        key = ('ew', id(blk), tuple(sorted(map(str, start_vs))))
        v = _lc.get(key, _LC_MISS)
        if v is _LC_MISS:
            v = _ew_impl(start_vs)
            _lc[key] = v
        return v
    # candidates: externals (in the block, or entering by a walk) and line momenta of confirmed components
    cands = []          # (tag, md, entry_points)
    for extn, v0 in ext_attach.items():
        md = ext_mode_for(ext_mode, extn)
        if v0 in realV:
            cands.append((extn, md, {v0}))
            continue
        if md == mode or marginally_softer(md, mode):
            t = entry_walk([v0])
            if t:
                cands.append((extn, md, t))
    for (cm, ci) in list(confirmed.keys()):
        if cm != mode and not marginally_softer(cm, mode):
            continue
        sblk = comps[cm][ci]
        sidx = sblk[2] if len(sblk) > 2 and sblk[2] else []
        for ei in sidx:
            a, b = edges[ei]
            if cm == mode:
                ent = {w for w in (a, b) if w in realV}
                if ent:
                    cands.append(('%s#%d' % (cm, ci), cm, ent))
            else:
                t = entry_walk([a, b])
                if t:
                    cands.append(('%s#%d' % (cm, ci), cm, t))
    if not cands:
        return False
    for (_tag, md, ent) in cands:
        if md == mode:
            for a in sorted(ent):
                if third_port(blk, mode, {a}, comps.get(mode, []), vm, edges):
                    return True
    # pairs: join == mode plus a third port
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            m1, e1 = cands[i][1], cands[i][2]
            m2, e2 = cands[j][1], cands[j][2]
            try:
                if join(m1, m2) != mode:
                    continue
            except Exception:
                continue
            for a in e1:
                for b in e2:
                    if third_port(blk, mode, {a, b}, comps.get(mode, []), vm, edges):
                        return True
    return False

# ========== INFRARED COMPATIBILITY: WALKS, PORTS AND DRIVER ===========

# ---- Third-port requirement (part of condition 1) ----
# A block whose scale is assembled from entries at {va, vb} needs a THIRD PORT w outside them — either
# (A) a same-mode vertex shared with another same-mode component, or (B) a harder-mode vertex reached
# by one of the block's own edges; a scale with no exit port is scaleless, H components are exempt.
USE_THIRD_PORT = True


# All target elements reachable from the start elements by a non-softer flow; returns the entered real vertices.
def _collect_entries(blk, start_vs, start_es, cur_mode, edges, em, vm, _lc=None):
    key = None
    if _lc is not None:
        key = ('ce', id(blk), tuple(sorted(map(str, start_vs))),
               tuple(sorted(map(str, start_es or ()))), cur_mode)
        v = _lc.get(key, _LC_MISS)
        if v is not _LC_MISS:
            return v
    realV = {v for v in blk[0] if v != 'aux'}
    tgt = {('v', v) for v in realV}
    tgt |= {('e', ei) for ei in (blk[2] if len(blk) > 2 and blk[2] else [])}

    def mode_of(el):
        return vm.get(el[1], 'H') if el[0] == 'v' else em[el[1]]

    def neighbors(el):
        out = []
        if el[0] == 'v':
            v = el[1]
            for ei, (a, b) in enumerate(edges):
                if a == v:
                    out.append(('e', ei)); out.append(('v', b))
                elif b == v:
                    out.append(('e', ei)); out.append(('v', a))
        else:
            a, b = edges[el[1]]
            out.append(('v', a)); out.append(('v', b))
        return out

    # BFS over (vertex | edge) elements, never through G/sH, only onto non-softer modes
    start = [('v', v) for v in start_vs] + [('e', ei) for ei in start_es]
    visited = set(start)
    queue = [(el, cur_mode) for el in start]
    hits = set()
    while queue:
        el, cur = queue.pop(0)
        if el in tgt:
            hits.add(el); continue
        for nb in neighbors(el):
            if nb in visited:
                continue
            m_nb = mode_of(nb)
            if m_nb in ('G', 'sH'):
                continue
            if harder_or_eq(m_nb, cur):
                visited.add(nb); queue.append((nb, m_nb))
    ev = set()
    for h in hits:
        if h[0] == 'v':
            if h[1] in realV:
                ev.add(h[1])
        else:
            for w in edges[h[1]]:
                if w in realV:
                    ev.add(w)
    if key is not None:
        _lc[key] = ev
    return ev


# The third port: (A) a same-mode vertex shared with another same-mode component, or (B) a harder-mode exit endpoint.
def third_port(blk, X, entry_vs, comps_same, vm, edges):
    realV = {v for v in blk[0] if v != 'aux'}
    # (A) same-mode shared vertex
    for w in sorted(realV):
        if w in entry_vs:
            continue
        if vm.get(w) != X:
            continue
        for c in comps_same:
            if c is blk:
                continue
            if w in {v for v in c[0] if v != 'aux'}:
                return ('A', w)
    # (B) harder endpoint of one of the block's own edges
    idxs = blk[2] if len(blk) > 2 and blk[2] else []
    for ei in idxs:
        a, b = edges[ei]
        for w in (a, b):
            if w in realV or w in entry_vs:
                continue
            mw = vm.get(w)
            if mw is not None and mw != X and harder_or_eq(mw, X):
                return ('B', w)
    return None


# Condition-1 sources: externals (inside the block, or entering by a walk) and confirmed components entering it.
def _port_sources(blk, X, edges, em, vm, ext_attach, ext_mode, confirmed, comps, _lc=None):
    srcs = []
    realV = {v for v in blk[0] if v != 'aux'}
    for extn, v0 in ext_attach.items():
        md = ext_mode_for(ext_mode, extn)
        if v0 in realV:
            srcs.append((extn, md, {v0}, {v0}))
        else:
            if md == X or marginally_softer(md, X):
                c = _collect_entries(blk, [v0], [], vm.get(v0, 'H'), edges, em, vm, _lc=_lc)
                if c:
                    srcs.append((extn, md, c, set()))
    for (cm, ci) in list(confirmed.keys()):
        if not marginally_softer(cm, X):
            continue
        sblk = comps[cm][ci]
        sv = {v for v in sblk[0] if v != 'aux'}
        se = sblk[2] if len(sblk) > 2 and sblk[2] else []
        c = _collect_entries(blk, sv, se, cm, edges, em, vm, _lc=_lc)
        if c:
            srcs.append(('%s#%d' % (cm, ci), cm, c, set()))
    return srcs


# Hidden-path receiver port check: exclude the union of source entries, then require a third port.
def _hp_receiver_port_ok(blk, X, edges, em, vm, ext_attach, ext_mode, confirmed, comps, _lc=None):
    if not USE_THIRD_PORT:
        return True
    srcs = _port_sources(blk, X, edges, em, vm, ext_attach, ext_mode, confirmed, comps, _lc=_lc)
    entry = set()
    for (_tag, md, cands, att) in srcs:
        entry |= set(cands)
    return third_port(blk, X, entry, comps.get(X, []), vm, edges) is not None


# Regge IR compatibility: fixpoint over the 1VI components, each mode confirmed by its own channels; False if any stays unconfirmed.
def ir_ok(edges, em, vm, ext_attach, ext_mode=None):
    verts = set(v for e in edges for v in e) | set(ext_attach.values())
    comps = {}
    # collect the 1VI blocks of every mode
    for m in ('H', 'C13', 'C24', 'C1C13', 'C3C13', 'C2C24', 'C4C24', 'S^1C13', 'S^1C24', 'S', 'S^2', 'C1^2C13', 'C3^2C13', 'C2^2C24', 'C4^2C24', 'S^2C13', 'S^2C24', 'S^1C1C13', 'S^1C3C13', 'S^1C2C24', 'S^1C4C24'):
        comps[m] = mode_components(m, vm, em, edges, verts)
    # H blocks: each needs its own confirming cut; aux-only blocks are contraction artifacts
    comps['H'] = [b for b in mode_components('H', vm, em, edges, verts) if any(v != 'aux' for v in b[0])]

    # ---- tadpole gate: soft blocks without a softer adjacent component ----
    def _is_pure_soft(md):
        return _m_of(md) >= 1 and 'C' not in md

    def _tadp(md, blk):
        if _m_of(md) < 1:
            return False
        realV = {v for v in blk[0] if v != 'aux'}
        idxs = blk[2] if len(blk) > 2 and blk[2] else []
        ends = set(realV)
        for ei in idxs:
            ends |= set(edges[ei])
        for w in ends:
            if vm.get(w) == 'H':
                return False
        def _adj(a_idxs, a_v, b_idxs, b_v):
            for ei in a_idxs:
                for w in edges[ei]:
                    if w in b_v:
                        return True
            for ei in b_idxs:
                for w in edges[ei]:
                    if w in a_v:
                        return True
            return False
        for m2, lst2 in comps.items():
            if V(m2) >= V(md):
                continue
            for c2 in lst2:
                c2v = {v for v in c2[0] if v != 'aux'}
                c2i = c2[2] if len(c2) > 2 and c2[2] else []
                if _adj(idxs, realV, c2i, c2v):
                    return False
        return True

    def _vee_ok(md, blk):
        realV = {v for v in blk[0] if v != 'aux'}
        idxs = blk[2] if len(blk) > 2 and blk[2] else []
        ends = set(realV)
        for ei in idxs:
            ends |= set(edges[ei])
        pool = []
        for i2 in range(len(edges)):
            u, w = edges[i2]
            if (u in ends or w in ends) and 'C' in em[i2]:
                pool.append(em[i2])
        for nm, v in ext_attach.items():
            if v in ends:
                pool.append(ext_mode_for(ext_mode, nm))
        if len(pool) < 2:
            return False
        for ai in range(len(pool)):
            for bi in range(ai + 1, len(pool)):
                if join(pool[ai], pool[bi]) == md:
                    return True
        return False

    _tad_cache, _vee_cache = {}, {}

    def _cond_allowed(md, blk):
        t = _tad_cache.get(id(blk))
        if t is None:
            t = _tadp(md, blk)
            _tad_cache[id(blk)] = t
        if not t:
            return True
        if not _is_pure_soft(md):
            return False
        ok = _vee_cache.get(id(blk))
        if ok is None:
            ok = _vee_ok(md, blk)
            _vee_cache[id(blk)] = ok
        return ok
    # (pure-aux SC carriers fail the relevance rules automatically — no separate check)
    confirmed = {}
    _lc = {}   # per-call local cache for pure sub-results (reused across fixpoint rounds)
    # ---- per-mode confirmation channels: f(blk, i) -> bool; every non-H mode gets cond1 (entry + vee + port) ----
    channels = {}

    def add(mode, fn):
        channels.setdefault(mode, []).append(fn)

    def cond1_ch(mode):
        return lambda blk, i: cond1_confirms(blk, mode, comps, confirmed, edges, em, vm, ext_attach, ext_mode, _lc=_lc)

    for m in comps:
        if m != 'H':
            add(m, cond1_ch(m))
    # refined collinear & C²: direct p-attach
    for mode, ext in (('C1C13', 'p1'), ('C3C13', 'p3'), ('C2C24', 'p2'), ('C4C24', 'p4'),
                      ('C1^2C13', 'p1'), ('C3^2C13', 'p3'), ('C2^2C24', 'p2'), ('C4^2C24', 'p4')):
        vext = ext_attach[ext]
        m_ext = ext_mode_for(ext_mode, ext)
        add(mode, lambda blk, i, vext=vext, m_ext=m_ext, mode=mode: vext in blk[0] and m_ext == mode)
    # S-family: messenger channels (simultaneously relevant to the target set with >=1 confirmed)
    for mode, own, other in (('S^1C13', ('C1C13', 'C3C13'), ('C24',)), ('S^1C24', ('C2C24', 'C4C24'), ('C13',))):
        add(mode, lambda blk, i, mode=mode, own=own, other=other: _cond_allowed(mode, blk) and messenger_confirms(blk, mode, own, other, comps, confirmed, edges, em, vm, _lc=_lc))
    for mode, own, other in (('S^2C13', ('C1^2C13', 'C3^2C13'), ('S^1C24', 'C2C24', 'C4C24')), ('S^2C24', ('C2^2C24', 'C4^2C24'), ('S^1C13', 'C1C13', 'C3C13'))):
        add(mode, lambda blk, i, mode=mode, own=own, other=other: _cond_allowed(mode, blk) and messenger_confirms(blk, mode, own, other, comps, confirmed, edges, em, vm, _lc=_lc))
    # S-family: meet-of-two channels (two confirmed relevant components whose modes meet to the block's mode)
    for mode in ('S', 'S^2', 'S^1C13', 'S^1C24', 'S^2C13', 'S^2C24', 'S^1C1C13', 'S^1C3C13', 'S^1C2C24', 'S^1C4C24'):
        add(mode, lambda blk, i, mode=mode: _cond_allowed(mode, blk) and meet_of_two_confirms(blk, mode, comps, confirmed, edges, em, vm, _lc=_lc))
    # SC hidden path: an SC bridges two distinct pair-mode components (or S between two C13s / C24s); one confirmed pair conducts to the other
    for sc_mode, pair_mode, ext_a, ext_b, fam_modes, fam_a, fam_b in (
            ('S^1C24', 'C13', 'p1', 'p3', ('C2C24', 'C4C24'), ('C13', 'C1C13', 'C1^2C13', 'C1∞C13'), ('C13', 'C3C13', 'C3^2C13', 'C3∞C13')),
            ('S^1C13', 'C24', 'p2', 'p4', ('C1C13', 'C3C13'), ('C24', 'C2C24', 'C2^2C24', 'C2∞C24'), ('C24', 'C4C24', 'C4^2C24', 'C4∞C24')),
            ('S', 'C13', 'p1', 'p3', ('C24',), ('C13', 'C1C13', 'C1^2C13', 'C1∞C13'), ('C13', 'C3C13', 'C3^2C13', 'C3∞C13')),
            ('S', 'C24', 'p2', 'p4', ('C13',), ('C24', 'C2C24', 'C2^2C24', 'C2∞C24'), ('C24', 'C4C24', 'C4^2C24', 'C4∞C24'))):
        add(pair_mode, lambda blk, i, sc_mode=sc_mode, pair_mode=pair_mode, ext_a=ext_a, ext_b=ext_b, fam_modes=fam_modes, fam_a=fam_a, fam_b=fam_b: sc_hidden_path_confirms(blk, i, comps, confirmed, edges, em, vm, ext_attach, ext_mode, sc_mode, pair_mode, ext_a, ext_b, fam_modes, fam_a, fam_b, _lc=_lc))
    # H: total inflow ∨ == H
    add('H', lambda blk, i: h_comp_confirmed(blk, comps, confirmed, edges, em, vm, ext_attach, ext_mode))

    # fixpoint: sweep unconfirmed blocks until no new confirmation
    for _ in range(30):
        changed = False
        for mode, fns in channels.items():
            for i, blk in enumerate(comps[mode]):
                if (mode, i) in confirmed:
                    continue
                if any(f(blk, i) for f in fns):
                    confirmed[(mode, i)] = True
                    changed = True
        if not changed:
            break
    # every block must be confirmed
    for m, lst in comps.items():
        for i in range(len(lst)):
            if (m, i) not in confirmed:
                return False
    return True

# ================== REGION REQUIREMENTS AND BUILDER ===================


# Build a region from its cuts: overlay, Glauber/semihard adjustment, then all requirements; returns (vm, em) or None.
def _build_region(edges, verts, ext_attach, ext_mode, cut13, cut24, cut1, cut3, cut2, cut4, cut1sq=None, cut3sq=None, cut2sq=None, cut4sq=None):
    # nonempty cuts as (mode, vertex set) pairs
    cuts = []
    if cut13: cuts.append(('C13', cut13))
    if cut24: cuts.append(('C24', cut24))
    if cut1: cuts.append(('C1C13', cut1))
    if cut3: cuts.append(('C3C13', cut3))
    if cut2: cuts.append(('C2C24', cut2))
    if cut4: cuts.append(('C4C24', cut4))
    if cut1sq: cuts.append(('C1^2C13', cut1sq))
    if cut3sq: cuts.append(('C3^2C13', cut3sq))
    if cut2sq: cuts.append(('C2^2C24', cut2sq))
    if cut4sq: cuts.append(('C4^2C24', cut4sq))

    # subgraph not in any cut: nonempty and connected (depends only on the cuts)
    covered_v = set().union(*[S for (m, S) in cuts]) if cuts else set()
    hv = set(verts) - covered_v
    if not hv: return None
    he = {e for e in edges if e[0] not in covered_v and e[1] not in covered_v}
    if not connected(hv, he): return None
    vm = {}
    for v in verts:
        acc = None
        for (m, S) in cuts:
            if v in S:
                acc = m if acc is None else meet(acc, m)
        vm[v] = acc if acc is not None else 'H'
    em = [meet(vm[a], vm[b]) for (a, b) in edges]
    # edge-mode self-consistency: em[e] = vm[u] ∧ vm[v] (before the Glauber adjustment)
    for (a, b), m in zip(edges, em):
        if meet(vm[a], vm[b]) != m:
            return None
    # Glauber adjustment: t-channel propagators of the large-momentum flow become G
    adj = glauber_adjust(edges, em, vm, ext_attach, ext_mode)
    if adj is None:
        return None
    em, vm = adj
    # [Fundamental pattern] momentum conservation at every vertex
    if not momentum_ok(edges, em, vm, ext_attach, ext_mode):
        return None
    # [Fundamental pattern] jet components: every component contains its external vertices
    je13 = {e for e, m in zip(edges, em) if m in J13_FAM}
    jv13 = ({v for v in verts if vm[v] in J13_FAM} | {v for e in je13 for v in e})
    je24 = {e for e, m in zip(edges, em) if m in J24_FAM}
    jv24 = ({v for v in verts if vm[v] in J24_FAM} | {v for e in je24 for v in e})
    if not jet_components_ok(jv13, je13, {ext_attach['p1'], ext_attach['p3']}):
        return None
    if not jet_components_ok(jv24, je24, {ext_attach['p2'], ext_attach['p4']}):
        return None
    # (3) mojetic (REGGE_NO_MOJETIC=1 disables, for tests)
    if os.environ.get('REGGE_NO_MOJETIC') != '1':
        # k0 (all externals finite C13/C24): each contracted jet 1VI, each H component needs jet edges of BOTH families adjacent
        k0 = all(ext_mode_for(ext_mode, n) in ('C13', 'C24') for n in ext_attach)
        if k0:
            if not mojetic_k0_ok(edges, em, vm, J13_FAM, J24_FAM, ext_attach):
                return None
        else:
            gv = {v for v in verts if vm[v] == 'G'}
            hv_m = {v for v in verts if vm[v] == 'H'}
            he_m = {e for e, m in zip(edges, em) if m == 'H'}
            for jv, je, ext_names in ((jv13, je13, ('p1', 'p3')), (jv24, je24, ('p2', 'p4'))):
                if not mojetic_ok(hv_m, he_m, jv, je, gv, edges, ext_attach, ext_names):
                    return None
    # [Connectivity] no C13/C24 scaleless island (see c13_c24_island_ok)
    if not c13_c24_island_ok(edges, em, vm, verts):
        return None
    # [Infrared compatibility]
    if not ir_ok(edges, em, vm, ext_attach, ext_mode):
        return None
    return (vm, em)
