#!/usr/bin/env python3
"""enumerator.py — skeleton-based cut enumeration for the collinear 2->3 kinematics (general external virtualities).

A region is built from a skeleton: a connected uncovered vertex set H, one leg path P_i per external momentum, and per chain a cut expanded as a connected superset of P_i inside its domain.
One enumeration engine covers the kinematics: enumerate_skelg, for a general ms = (m1, ..., m5); the
all-on-shell case is simply ms = (INF,)*5.  It deduplicates candidates and runs the shared region_filter check chain.

Skeleton and cuts.
  * H connected; 5 paths P1, ..., P5 (P2/P3 form a Y and may pass through v2/v3);
  * for i = 1,4,5, the C_i cut ⊇ the whole P_i, avoiding other P, and the refined cuts are nested, i.e., C_i^1 ⊇ C_i^2 ⊇ ...;
  * the C23 cut ⊇ P2 ∪ P3 (connected or not; every component contains v2 or v3) and touches neither P1, P4 nor P5;
  * refined cuts in the 23 channel are nested: C23 ⊇ C2C23 ⊇ C2^2C23 ⊇ ..., and C23 ⊇ C3C23 ⊇ C3^2C23 ⊇ ...;
  * the C2C23 cut overlaps with the v2 branch (P2 - P3) into a connected subpath with v2 as one endpoint; it avoids P1, P3, P4, P5; the C3C23 cut is similar;

Pruning -- overlap condition:
  * for i = 1,4,5, the C_i^n cut must share a vertex with two cuts from two distinct other directions j1, j2 of total C-power n (C_j^n for wide j; C_j^{n-1}C23 for pair j).
  * for i = 2,3 the same holds for every C_i^n C23 cut.
"""
import os, sys
from itertools import combinations, product
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import region_filter as F
import mode_levels as ML
from primitives import INF, W, P, meet, H, ext_modes
from region_filter import (momentum_ok, jets_ok, uncovered_ok, mojetic_all_ok, island_ok, ir_ok)

# ==================== OVERLAY ====================
CUT_MODES = {'C23': P(0, 0), 'C1': W(1, 1), 'C4': W(4, 1), 'C5': W(5, 1), 'C1R1': W(1, 2), 'C4R1': W(4, 2), 'C5R1': W(5, 2), 'C2R1': P(1, 2), 'C3R1': P(1, 3), 'C2R2': P(2, 2), 'C3R2': P(2, 3)}

# Build the overlay (em, vm) from a cut assignment {name: frozenset}; cut_modes overrides the default name -> mode table.
def build_overlay(edges, verts, ext_attach, ext_mode, cuts, vm_seen=None, cut_modes=None):
    # vm(v) = meet of the covering cuts' modes, H when no cut covers v.
    acc_map = {}
    _cm = CUT_MODES if cut_modes is None else cut_modes
    _meet = meet
    for nm in _cm:
        S = cuts.get(nm)
        if not S:
            continue
        m = _cm[nm]
        for v in S:
            acc = acc_map.get(v)
            if acc is None:
                acc_map[v] = m
            else:
                try:
                    acc_map[v] = _meet(acc, m)
                except ArithmeticError as e:
                    return None, f'meet:{e}'
    vm = {}
    _H = H
    for v in verts:
        acc = acc_map.get(v)
        vm[v] = acc if acc is not None else _H()
    if vm_seen is not None:
        # Dedup key: the modes in fixed (sorted) vertex order; callers pass the sorted vertex list.
        vkey = tuple(map(vm.__getitem__, verts))
        if vkey in vm_seen:
            return None, 'vm-dup'
        vm_seen.add(vkey)
    em = []
    _ap = em.append
    for (a, b) in edges:
        try:
            _ap(_meet(vm[a], vm[b]))
        except ArithmeticError as e:
            return None, f'edge-meet:{e}'
    return (em, vm), 'ok'

# ==================== SHARED HELPERS ====================

# True iff the vertex subset S induces a connected subgraph.
def _connected_set(S, adj):
    S = set(S)
    if not S:
        return False
    seen = {next(iter(S))}
    st = list(seen)
    while st:
        u = st.pop()
        for w in adj[u]:
            if w in S and w not in seen:
                seen.add(w)
                st.append(w)
    return seen == S


# All connected supersets of req inside dom.
def _conn_sets(req, dom, adj):
    req = set(req); dd = set(dom)
    if not req <= dd:
        return []
    extras = sorted(dd - req); out = []
    for r in range(len(extras) + 1):
        for add in combinations(extras, r):
            C = req | set(add)
            if _connected_set(C, adj):
                out.append(frozenset(C))
    return out


# All vertex paths from start to H that avoid other external vertices.
def _enumerate_seqs(start, Hset, adj, Vset, extv, friends=()):
    allowed = [w for w in Vset if w not in Hset and (w == start or w in friends or w not in extv)]
    seqs = []
    st = [(start, {start}, (start,))]
    while st:
        cur, vis, path = st.pop()
        if adj[cur] & Hset:
            seqs.append(path)
        for w in adj[cur]:
            if w in allowed and w not in vis:
                st.append((w, vis | {w}, path + (w,)))
    return seqs


# True iff the two p2/p3 leg paths share only a common prefix (the Y stem).
def _compat(seq2, seq3):
    r2 = seq2[::-1]; r3 = seq3[::-1]
    k = 0
    while k < min(len(r2), len(r3)) and r2[k] == r3[k]:
        k += 1
    return not (set(r2[k:]) & set(r3[k:]))


# ==================== MASK & PRUNING HELPERS ====================

# [Optimization] Return mk(frozenset) -> int bitmask (bit i <-> verts[i]), cached per enumeration.
def _make_mk(verts):
    bidx = {v: 1 << i for i, v in enumerate(verts)}
    cache = {}

    # [Optimization] cached bitmask of a vertex subset.
    def mk(S):
        r = cache.get(S)
        if r is None:
            r = 0
            for v in S:
                r |= bidx[v]
            cache[S] = r
        return r

    return mk


# Shared per-enumeration context: bitmask cache, adjacency basis and connected-set list.
def _slot_ctx(V, E, ext_attach):
    mk = _make_mk(V)                  # bitmask cache
    adj = {v: set() for v in V}
    for a, b in E:
        adj[a].add(b); adj[b].add(a)
    Vset = set(V)
    extv = set(ext_attach.values())
    roots = {leg: ext_attach[leg] for leg in ('p1', 'p2', 'p3', 'p4', 'p5')}
    Hs = [frozenset(S) for r in range(1, len(V) + 1) for S in combinations(V, r) if _connected_set(S, adj)]
    return mk, adj, Vset, extv, roots, Hs


# [Optimization] Fixed slot order for the cut-dedup keys.
_CUT_SLOTS = ('C23', 'C1', 'C4', 'C5', 'C1R1', 'C4R1', 'C5R1', 'C2R1', 'C3R1', 'C2R2', 'C3R2')


# Dynamic cut slots (name, mode, leg, power) of a chain assignment: one slot per appearing carrier power PW[leg].
def _deep_slots(PW):
    out = [('C23', P(0, 0), 23, 0)]
    for i in (1, 4, 5):
        for n in PW[i]:
            out.append(('C%d' % i if n == 1 else 'C%dR%d' % (i, n - 1), W(i, n), i, n))
    for i in (2, 3):
        for k in PW[i]:
            out.append(('C%dR%d' % (i, k), P(k, i), i, k))
    return out


# True iff no two other directions can supply cuts of total C-power n sharing a vertex with this subject (pools: leg -> {power: mask}).
def _prune_dead(msx, i, n, pools):
    for j1 in (1, 2, 3, 4, 5):
        if j1 == i:
            continue
        U1 = pools[j1].get(n, 0)
        if not U1:
            continue
        for j2 in (1, 2, 3, 4, 5):
            if j2 == i or j2 == j1:
                continue
            U2 = pools[j2].get(n, 0)
            if U2 and (msx & U1 & U2):
                return False
    return True


# ==================== CUT-SET CONSTRUCTION HELPERS ====================

# one refined cut, one level deeper: the empty set plus connected supersets of `root` inside `base`, avoiding `forbid`.
def _refined_opts(root, base, forbid, adj):
    allowed = {v for v in base if v not in forbid}
    out = [frozenset()]
    if root not in allowed:
        return out
    rest = sorted(allowed - {root})
    for r in range(len(rest) + 1):
        for sub in combinations(rest, r):
            S = frozenset({root} | set(sub))
            if _connected_set(S, adj):
                out.append(S)
    return out


# Nested refined cuts: level 1 from lvl1_opts, each deeper level nested inside the previous one.
def _tower(root, lvl1_opts, depth, forbid, adj):
    if depth <= 0:
        return [()]
    t = [(s,) for s in lvl1_opts]
    for _ in range(depth - 1):
        t = [ch + (s2,) for ch in t for s2 in _refined_opts(root, ch[-1], forbid, adj)]
    return t


# Supersets of req inside dom whose every connected component contains v2 or v3.
def _c23_opts(req, dom, v2, v3, adj):
    extras = sorted(set(dom) - set(req))
    out = []
    for r in range(len(extras) + 1):
        for sub in combinations(extras, r):
            Sm = frozenset(set(req) | set(sub))
            seen = set()
            ok = True
            for start in Sm:
                if start in seen:
                    continue
                stack = [start]
                comp = set()
                while stack:
                    u = stack.pop()
                    if u in comp:
                        continue
                    comp.add(u)
                    for w in adj[u]:
                        if w in Sm and w not in comp:
                            stack.append(w)
                seen |= comp
                if not (v2 in comp or v3 in comp):
                    ok = False
                    break
            if ok:
                out.append(Sm)
    return out


# ==================== DEEP CHAIN ENUMERATION ====================

# Chain construction for the deep regime: one nested refined cut per appearing carrier power (wide legs and the pair channel).
def _enumerate_skelg_deep(edges, verts, ext_attach, ms, ext_mode, PWS, use_overlap, collect):
    PW = {i: list(PWS['p%d' % i]) for i in (1, 2, 3, 4, 5)}
    mof = dict(zip((1, 2, 3, 4, 5), ms))
    edges = [tuple(e) for e in edges]
    V = sorted(verts)
    mk, adj, Vset, extv, roots, Hs = _slot_ctx(V, edges, ext_attach)
    v2r, v3r = roots['p2'], roots['p3']
    slots = _deep_slots(PW)
    # mode table of the dynamic cut names, handed to the overlay builder
    cut_modes = {nm: md for nm, md, _lg, _lv in slots}
    total_cand = 0
    dup_cuts = 0
    skip_emvm = 0
    n_vmdup = 0
    n_overlap = 0
    cuts_seen = set()
    emvm_seen = set()
    vm_seen = set()
    found = {}
    SURV = {}

    # Partner mask of direction j at total C-power c (wide leg C_j^c, pair leg C_j^{c-1} C23); missing powers are absent cuts.
    def pmask(plist, cm, j, c):
        if j in (1, 4, 5):
            return plist[j].get(c, 0)
        k = c - 1
        if k == 0:
            return cm['C23']
        return plist[j].get(k, 0)

    # Region checks for one chain assignment (cm: name -> mask, plist: leg -> masks indexed by level).
    def run_checks(cutmap, cm, plist):
        nonlocal total_cand, dup_cuts, skip_emvm, n_overlap, n_vmdup
        total_cand += 1
        ck = tuple(cm.get(nm, 0) for nm, _md, _lg, _lv in slots)
        if ck in cuts_seen:
            dup_cuts += 1
            return
        cuts_seen.add(ck)
        if use_overlap:
            # the C_i^n cut must share a vertex with two cuts from two distinct other directions of total C-power n
            for i in (1, 4, 5, 2, 3):
                for c in PW[i]:
                    if i in (1, 4, 5):
                        if not (c < mof[i]):
                            continue
                    else:
                        if not (c < mof[i] - 1):
                            continue
                    subj = plist[i].get(c, 0)
                    if not subj:
                        continue
                    # A shallower cut that coincides with a deeper cut of the same leg is exempt.
                    if any(plist[i].get(c2) == subj for c2 in PW[i] if c2 > c):
                        continue
                    dirs = [j for j in (1, 4, 5, 2, 3) if j != i]
                    ok = False
                    for a in range(len(dirs)):
                        for b in range(a + 1, len(dirs)):
                            if pmask(plist, cm, dirs[a], c) & pmask(plist, cm, dirs[b], c) & subj:
                                ok = True
                                break
                        if ok:
                            break
                    if not ok:
                        n_overlap += 1
                        return
        if not uncovered_ok(edges, V, cutmap):
            return
        res, why = build_overlay(edges, V, ext_attach, ext_mode, cutmap, vm_seen=vm_seen, cut_modes=cut_modes)
        if res is None:
            if why == 'vm-dup':
                n_vmdup += 1
            return
        em, vm = res
        ek = tuple((i, F.name(m)) for i, m in enumerate(em))
        vk = tuple(F.name(vm[w]) for w in V)
        if (ek, vk) in emvm_seen:
            skip_emvm += 1
            return
        emvm_seen.add((ek, vk))
        if not momentum_ok(edges, V, ext_attach, ext_mode, em, vm):
            return
        okinfo = jets_ok(edges, V, vm, em, ext_attach)
        if not okinfo[0]:
            return
        if not mojetic_all_ok(edges, V, vm, em, ext_attach, okinfo[1]):
            return
        if not island_ok(edges, V, em, vm):
            return
        if not ir_ok(edges, V, em, vm, ext_attach, ext_mode, {}):
            return
        vec = tuple(-F.V(m) if F.V(m) != F.INF else 'inf' for m in em) + (1,)
        found[(ek, vk)] = vec
        if collect:
            SURV[(ek, vk)] = (dict((kk, set(vv)) for kk, vv in cutmap.items()), em, vm)

    for H in Hs:
        Hset = set(H)
        outs = [leg for leg in ('p1', 'p2', 'p3', 'p4', 'p5') if roots[leg] not in Hset]
        fam = {}
        ok = True
        for leg in ('p1', 'p4', 'p5'):
            if leg not in outs:
                fam[leg] = None
                continue
            start = roots[leg]
            sets = set()
            allowed = [w for w in Vset if w not in Hset and (w == start or w not in extv)]
            st = [(start, {start}, (start,))]
            while st:
                cur, vis, path = st.pop()
                if adj[cur] & Hset:
                    sets.add(frozenset(path))
                for w in adj[cur]:
                    if w in allowed and w not in vis:
                        st.append((w, vis | {w}, path + (w,)))
            if not sets:
                ok = False
                break
            fam[leg] = list(sets)
        if not ok:
            continue
        pair_opts = []
        both_H = (roots['p2'] in Hset) and (roots['p3'] in Hset)
        if both_H:
            pair_opts = [(None, None, None, None)]
        else:
            fr = {roots['p2'], roots['p3']}
            seq2 = (_enumerate_seqs(roots['p2'], Hset, adj, Vset, extv, fr) if roots['p2'] not in Hset else [])
            seq3 = (_enumerate_seqs(roots['p3'], Hset, adj, Vset, extv, fr) if roots['p3'] not in Hset else [])
            if not seq2 and not seq3:
                continue
            seen_po = set()
            if seq2 and seq3:
                for a in seq2:
                    ra = a[::-1]
                    for b in seq3:
                        if not _compat(a, b):
                            continue
                        rb = b[::-1]
                        k = 0
                        while k < min(len(ra), len(rb)) and ra[k] == rb[k]:
                            k += 1
                        key = (frozenset(a), frozenset(b), ra[k:], rb[k:])
                        if key in seen_po:
                            continue
                        seen_po.add(key)
                        pair_opts.append((frozenset(a), frozenset(b), ra[k:], rb[k:]))
            elif seq2:
                for a in seq2:
                    key = (frozenset(a), frozenset(), a[::-1], None)
                    if key in seen_po:
                        continue
                    seen_po.add(key)
                    pair_opts.append((frozenset(a), frozenset(), a[::-1], None))
            else:
                for b in seq3:
                    key = (frozenset(), frozenset(b), None, b[::-1])
                    if key in seen_po:
                        continue
                    seen_po.add(key)
                    pair_opts.append((frozenset(), frozenset(b), None, b[::-1]))
            if not pair_opts:
                continue
        l1 = fam['p1'] if fam['p1'] is not None else [None]
        l4 = fam['p4'] if fam['p4'] is not None else [None]
        l5 = fam['p5'] if fam['p5'] is not None else [None]
        f1 = extv - {roots['p1']}
        f4 = extv - {roots['p4']}
        f5 = extv - {roots['p5']}
        f2 = extv - {roots['p2']}
        f3 = extv - {roots['p3']}
        for S1, S4, S5 in product(l1, l4, l5):
            u = set()
            skip = False
            for S in (S1, S4, S5):
                if S is not None:
                    if S & u:
                        skip = True
                        break
                    u |= S
            if skip:
                continue
            for S2, S3, br2, br3 in pair_opts:
                if (S2 is not None or S3 is not None) and (((S2 or frozenset()) | (S3 or frozenset())) & u):
                    continue
                a1 = S1 or frozenset()
                a4 = S4 or frozenset()
                a5 = S5 or frozenset()
                a2 = S2 or frozenset()
                a3 = S3 or frozenset()
                C1s = (_conn_sets(a1, Vset - Hset - a2 - a3 - a4 - a5, adj) if S1 is not None else [frozenset()])
                if not C1s:
                    continue
                C4s = (_conn_sets(a4, Vset - Hset - a1 - a2 - a3 - a5, adj) if S4 is not None else [frozenset()])
                if not C4s:
                    continue
                C5s = (_conn_sets(a5, Vset - Hset - a1 - a2 - a3 - a4, adj) if S5 is not None else [frozenset()])
                if not C5s:
                    continue
                T1 = _tower(roots['p1'], C1s, len(PW[1]), f1, adj)
                T4 = _tower(roots['p4'], C4s, len(PW[4]), f4, adj)
                T5 = _tower(roots['p5'], C5s, len(PW[5]), f5, adj)
                if S2 is not None or S3 is not None:
                    for C23 in _c23_opts(a2 | a3, Vset - Hset - a1 - a4 - a5, v2r, v3r, adj):
                        D = frozenset(set(C23) - a2 - a3)
                        M2 = {frozenset()}
                        # the C2C23 cut is trivial for m_2 = 1 (C_2^0 C23 = C23)
                        if br2 and mof[2] > 1:
                            for j in range(len(br2)):
                                T = frozenset(br2[j:])
                                M2.update(_conn_sets(T, T | D, adj))
                        M3 = {frozenset()}
                        # the C3C23 cut is trivial for m_3 = 1 (C_3^0 C23 = C23)
                        if br3 and mof[3] > 1:
                            for j in range(len(br3)):
                                T = frozenset(br3[j:])
                                M3.update(_conn_sets(T, T | D, adj))
                        T2 = _tower(roots['p2'], M2, len(PW[2]), f2, adj)
                        T3 = _tower(roots['p3'], M3, len(PW[3]), f3, adj)
                        mC23 = mk(C23)
                        for ch1, ch4, ch5, ch2, ch3 in product(T1, T4, T5, T2, T3):
                            cutmap = {'C23': C23}
                            cm = {'C23': mC23}
                            plist = {1: {}, 4: {}, 5: {}, 2: {}, 3: {}}
                            for i, ch in ((1, ch1), (4, ch4), (5, ch5)):
                                for n, S in zip(PW[i], ch):
                                    nm = 'C%d' % i if n == 1 else 'C%dR%d' % (i, n - 1)
                                    msk = mk(S)
                                    cutmap[nm] = S
                                    cm[nm] = msk
                                    plist[i][n] = msk
                            for i, ch in ((2, ch2), (3, ch3)):
                                for k, S in zip(PW[i], ch):
                                    nm = 'C%dR%d' % (i, k)
                                    msk = mk(S)
                                    cutmap[nm] = S
                                    cm[nm] = msk
                                    plist[i][k] = msk
                            run_checks(cutmap, cm, plist)
                else:
                    for ch1, ch4, ch5 in product(T1, T4, T5):
                        cutmap = {'C23': frozenset()}
                        cm = {'C23': 0}
                        plist = {1: {}, 4: {}, 5: {}, 2: {}, 3: {}}
                        for i, ch in ((1, ch1), (4, ch4), (5, ch5)):
                            for n, S in zip(PW[i], ch):
                                nm = 'C%d' % i if n == 1 else 'C%dR%d' % (i, n - 1)
                                msk = mk(S)
                                cutmap[nm] = S
                                cm[nm] = msk
                                plist[i][n] = msk
                        run_checks(cutmap, cm, plist)

    info = {'dup_cuts': dup_cuts, 'skip_emvm': skip_emvm, 'overlap_kill': n_overlap, 'vm_dup': n_vmdup}
    if collect:
        info['survivors'] = [(vv,) + sv for vv, sv in sorted(((vv, SURV[k]) for k, vv in found.items()), key=lambda t: t[0])]
    return sorted(found.values()), total_cand, info


# ==================== MAIN ENUMERATION (GENERAL CHAIN) ====================

# True when every leg's appearing powers fit the shallow fixed-slot chain (wide {1} or {1,2}, pair {1}; none fits too).
# A gap, a power above 2, or a wide chain that starts above the base forces the general (deep) engine.
def _shallow_chain_ok(levels):
    for leg in ('p1', 'p4', 'p5'):
        if levels[leg] not in ([1], [1, 2]):
            return False
    for leg in ('p2', 'p3'):
        if levels[leg] not in ([], [1]):
            return False
    return True


# Cuts and refined levels are built by the chain construction for a general ms = (m1, ..., m5).
def enumerate_skelg(edges, verts, ext_attach, kin, use_overlap=True, collect=False):
    # kin is the raw (m1, ..., m5) virtuality tuple (general kinematics).
    ms = tuple(kin)
    ext_mode = ext_modes(ms)
    m1, m2, m3, m4, m5 = ms
    # carrier powers of each chain from the loop count L = E - V + 1
    L = len(edges) - len(verts) + 1
    PWS = ML.cut_chain_levels(ext_mode, L)
    # the shallow fixed-slot engine consumes one refinement level per leg (wide: max n - 1, pair: max k); the deep engine needs the powers.
    LEVELS = ML.cut_chain_depth(PWS)
    if max(LEVELS['p1'], LEVELS['p4'], LEVELS['p5'], LEVELS['p2'], LEVELS['p3']) > 1 or not _shallow_chain_ok(PWS):
        return _enumerate_skelg_deep(edges, verts, ext_attach, ms, ext_mode, PWS,
                                     use_overlap, collect)
    edges = [tuple(e) for e in edges]
    V = sorted(verts)
    mk, adj, Vset, extv, roots, Hs = _slot_ctx(V, edges, ext_attach)
    v2r, v3r = roots['p2'], roots['p3']

    total_cand = 0
    dup_cuts = 0
    skip_emvm = 0
    n_vmdup = 0
    n_overlap = 0
    _use_prune = bool(use_overlap)   # early pruning mirrors the overlap condition
    _RT = _use_prune and not os.environ.get('FRI_NO_RT')       # later-stage (re-)tests
    _TAIL = ((1 < m2 - 1) or (1 < m3 - 1)) and _use_prune and not os.environ.get('FRI_NO_TAIL')
    n_prune = 0          # subtrees skipped by the overlap pruning
    n_prune_split = {'11': 0, '12': 0, '41': 0, '42': 0, '51': 0, '52': 0, '21': 0, '31': 0,
                   '11r': 0, '12r': 0, '41r': 0, '42r': 0}
    skip_combos = 0       # complete combos skipped (accounting vs total_cand)
    cuts_seen = set()
    emvm_seen = set()
    vm_seen = set()   # vm-level dedup
    found = {}
    SURV = {}   # collect=True: (ek, vk) -> (cuts, em, vm)

    # Region checks for one cut assignment (cuts dict -> overlay -> filter chain).
    def run_checks(C1, C1R1, C4, C4R1, C5, C5R1, C2, C3, C23, M=None):
        nonlocal total_cand, dup_cuts, skip_emvm, n_overlap, n_vmdup
        total_cand += 1
        # [Optimization] per-cut bitmasks for the fixed cut slots (callers may pass the precomputed tuple M)
        if M is not None:
            mC23, mC1, mC4, mC5, mC1R1, mC4R1, mC5R1, mC2, mC3 = M
        else:
            mC23 = mk(C23); mC1 = mk(C1); mC4 = mk(C4); mC5 = mk(C5)
            mC1R1 = mk(C1R1); mC4R1 = mk(C4R1); mC5R1 = mk(C5R1)
            mC2 = mk(C2); mC3 = mk(C3)
        # [Optimization] fixed-slot mask tuple as the cut-dedup key
        mslot = (mC23, mC1, mC4, mC5, mC1R1, mC4R1, mC5R1, mC2, mC3, 0, 0)
        ck = mslot
        if ck in cuts_seen:
            dup_cuts += 1
            return
        cuts_seen.add(ck)
        if use_overlap:
            # overlap: the C_i^n cut must share a vertex with two cuts from two distinct other directions of total C-power n
            if C1 and 1 < m1 and not (mC1 == mC1R1) and not (mC1 & ((mC4 & mC5) | mC23)):
                n_overlap += 1
                return
            if C4 and 1 < m4 and not (mC4 == mC4R1) and not (mC4 & ((mC1 & mC5) | mC23)):
                n_overlap += 1
                return
            if C5 and 1 < m5 and not (mC5 == mC5R1) and not (mC5 & ((mC1 & mC4) | mC23)):
                n_overlap += 1
                return
            if C1R1 and not (mC1R1 & ((mC4R1 & mC5R1) | ((mC4R1 | mC5R1) & (mC2 | mC3)) | (mC2 & mC3))):
                n_overlap += 1
                return
            if C4R1 and 2 < m4 and not (mC4R1 & ((mC1R1 & mC5R1) | ((mC1R1 | mC5R1) & (mC2 | mC3)) | (mC2 & mC3))):
                n_overlap += 1
                return
            if C5R1 and 2 < m5 and not (mC5R1 & ((mC1R1 & mC4R1) | ((mC1R1 | mC4R1) & (mC2 | mC3)) | (mC2 & mC3))):
                n_overlap += 1
                return
            if C2 and 1 < m2 - 1 and not (mC2 & ((mC1 & mC4) | (mC1 & mC5) | (mC4 & mC5) | (mC23 & (mC1 | mC4 | mC5)))):
                n_overlap += 1
                return
            if C3 and 1 < m3 - 1 and not (mC3 & ((mC1 & mC4) | (mC1 & mC5) | (mC4 & mC5) | (mC23 & (mC1 | mC4 | mC5)))):
                n_overlap += 1
                return
        # cuts dict built only for the overlap survivors
        cuts = {'C23': C23, 'C1': C1, 'C4': C4, 'C5': C5, 'C1R1': C1R1, 'C4R1': C4R1, 'C5R1': C5R1, 'C2R1': C2, 'C3R1': C3, 'C2R2': frozenset(), 'C3R2': frozenset()}
        if not uncovered_ok(edges, V, cuts):
            return
        res, why = build_overlay(edges, V, ext_attach, ext_mode, cuts, vm_seen=vm_seen)
        if res is None:
            if why == 'vm-dup':
                n_vmdup += 1
            return
        em, vm = res
        ek = tuple((i, F.name(m)) for i, m in enumerate(em))
        # fixed vertex-order key (V is sorted)
        vk = tuple(F.name(vm[w]) for w in V)
        if (ek, vk) in emvm_seen:
            skip_emvm += 1
            return
        emvm_seen.add((ek, vk))
        if not momentum_ok(edges, V, ext_attach, ext_mode, em, vm):
            return
        okinfo = jets_ok(edges, V, vm, em, ext_attach)
        if not okinfo[0]:
            return
        if not mojetic_all_ok(edges, V, vm, em, ext_attach, okinfo[1]):
            return
        if not island_ok(edges, V, em, vm):
            return
        if not ir_ok(edges, V, em, vm, ext_attach, ext_mode, {}):
            return
        vec = tuple(-F.V(m) if F.V(m) != F.INF else 'inf' for m in em) + (1,)
        found[(ek, vk)] = vec
        if collect:
            SURV[(ek, vk)] = (dict((kk, set(vv)) for kk, vv in cuts.items()), em, vm)

    for H in Hs:
        Hset = set(H)
        outs = [leg for leg in ('p1', 'p2', 'p3', 'p4', 'p5') if roots[leg] not in Hset]
        fam = {}
        ok = True
        for leg in ('p1', 'p4', 'p5'):
            if leg not in outs:
                fam[leg] = None
                continue
            start = roots[leg]
            sets = set()
            allowed = [w for w in Vset if w not in Hset and (w == start or w not in extv)]
            st = [(start, {start}, (start,))]
            while st:
                cur, vis, path = st.pop()
                if adj[cur] & Hset:
                    sets.add(frozenset(path))
                for w in adj[cur]:
                    if w in allowed and w not in vis:
                        st.append((w, vis | {w}, path + (w,)))
            if not sets:
                ok = False
                break
            fam[leg] = list(sets)
        if not ok:
            continue
        pair_opts = []
        both_H = (roots['p2'] in Hset) and (roots['p3'] in Hset)
        if both_H:
            pair_opts = [(None, None, None, None)]
        else:
            fr = {roots['p2'], roots['p3']}
            seq2 = (_enumerate_seqs(roots['p2'], Hset, adj, Vset, extv, fr) if roots['p2'] not in Hset else [])
            seq3 = (_enumerate_seqs(roots['p3'], Hset, adj, Vset, extv, fr) if roots['p3'] not in Hset else [])
            if not seq2 and not seq3:
                continue
            seen_po = set()
            if seq2 and seq3:
                for a in seq2:
                    ra = a[::-1]
                    for b in seq3:
                        if not _compat(a, b):
                            continue
                        rb = b[::-1]
                        k = 0
                        while k < min(len(ra), len(rb)) and ra[k] == rb[k]:
                            k += 1
                        key = (frozenset(a), frozenset(b), ra[k:], rb[k:])
                        if key in seen_po:
                            continue
                        seen_po.add(key)
                        pair_opts.append((frozenset(a), frozenset(b), ra[k:], rb[k:]))
            elif seq2:
                for a in seq2:
                    key = (frozenset(a), frozenset(), a[::-1], None)
                    if key in seen_po:
                        continue
                    seen_po.add(key)
                    pair_opts.append((frozenset(a), frozenset(), a[::-1], None))
            else:
                for b in seq3:
                    key = (frozenset(), frozenset(b), None, b[::-1])
                    if key in seen_po:
                        continue
                    seen_po.add(key)
                    pair_opts.append((frozenset(), frozenset(b), None, b[::-1]))
            if not pair_opts:
                continue
        l1 = fam['p1'] if fam['p1'] is not None else [None]
        l4 = fam['p4'] if fam['p4'] is not None else [None]
        l5 = fam['p5'] if fam['p5'] is not None else [None]
        for S1, S4, S5 in product(l1, l4, l5):
            u = set()
            skip = False
            for S in (S1, S4, S5):
                if S is not None:
                    if S & u:
                        skip = True
                        break
                    u |= S
            if skip:
                continue
            for S2, S3, br2, br3 in pair_opts:
                if (S2 is not None or S3 is not None) and (((S2 or frozenset()) | (S3 or frozenset())) & u):
                    continue
                a1 = S1 or frozenset(); a4 = S4 or frozenset()
                a5 = S5 or frozenset(); a2 = S2 or frozenset()
                a3 = S3 or frozenset()
                C1s = (_conn_sets(a1, Vset - Hset - a2 - a3 - a4 - a5, adj) if S1 is not None else [frozenset()])
                if not C1s:
                    continue
                C4s = (_conn_sets(a4, Vset - Hset - a1 - a2 - a3 - a5, adj) if S4 is not None else [frozenset()])
                if not C4s:
                    continue
                C5s = (_conn_sets(a5, Vset - Hset - a1 - a2 - a3 - a4, adj) if S5 is not None else [frozenset()])
                if not C5s:
                    continue
                # [Optimization] refined levels depend only on the cut; precompute the maps once
                L1map = ({C1: _refined_opts(roots['p1'], C1, extv - {roots['p1']}, adj) for C1 in C1s} if LEVELS['p1'] else {C1: [frozenset()] for C1 in C1s})
                L4map = ({C4: _refined_opts(roots['p4'], C4, extv - {roots['p4']}, adj) for C4 in C4s} if LEVELS['p4'] else {C4: [frozenset()] for C4 in C4s})
                L5map = ({C5: _refined_opts(roots['p5'], C5, extv - {roots['p5']}, adj) for C5 in C5s} if LEVELS['p5'] else {C5: [frozenset()] for C5 in C5s})
                # [Optimization] mask unions over the still-possible cuts for legs 4/5 (fixed picks override them)
                U4a = U4b = U5a = U5b = 0
                for _c4 in C4s:
                    U4a |= mk(_c4)
                    for _l4 in L4map[_c4]:
                        U4b |= mk(_l4)
                for _c5 in C5s:
                    U5a |= mk(_c5)
                    for _l5 in L5map[_c5]:
                        U5b |= mk(_l5)
                N4 = sum(len(L4map[_c4]) for _c4 in C4s)
                N5 = sum(len(L5map[_c5]) for _c5 in C5s)
                if S2 is not None or S3 is not None:
                    for C23 in _c23_opts(a2 | a3, Vset - Hset - a1 - a4 - a5, v2r, v3r, adj):
                        D = frozenset(set(C23) - a2 - a3)
                        C2s = {frozenset()}
                        # the C2C23 cut is trivial for m_2 = 1 (C_2^0 C23 = C23)
                        if br2 and m2 > 1:
                            for j in range(len(br2)):
                                T = frozenset(br2[j:])
                                C2s.update(_conn_sets(T, T | D, adj))
                        C3s = {frozenset()}
                        # the C3C23 cut is trivial for m_3 = 1 (C_3^0 C23 = C23)
                        if br3 and m3 > 1:
                            for j in range(len(br3)):
                                T = frozenset(br3[j:])
                                C3s.update(_conn_sets(T, T | D, adj))
                        # [Optimization] mask unions for the pair legs (C23 fixed; C2/C3 open)
                        mC23r = mk(C23)
                        U2b = 0
                        for _c2 in C2s:
                            U2b |= mk(_c2)
                        U3b = 0
                        for _c3 in C3s:
                            U3b |= mk(_c3)
                        n_c2 = len(C2s); n_c3 = len(C3s)
                        S2x = U2b | U3b
                        P23x = U2b & U3b
                        T11 = (U4a & U5a) | mC23r
                        T12 = (U4b & U5b) | ((U4b | U5b) & S2x) | P23x
                        for C1 in C1s:
                            mC1x = mk(C1)
                            if _use_prune and C1 and 1 < m1 and C1 not in L1map[C1] and not (mC1x & T11):
                                n_prune += 1; n_prune_split['11'] += 1
                                skip_combos += len(L1map[C1]) * N4 * N5 * n_c2 * n_c3
                                continue
                            T41 = (mC1x & U5a) | mC23r
                            for L1 in L1map[C1]:
                                mL1x = mk(L1)
                                if _use_prune and L1 and not (mL1x & T12):
                                    n_prune += 1; n_prune_split['12'] += 1
                                    skip_combos += N4 * N5 * n_c2 * n_c3
                                    continue
                                T42 = (mL1x & U5b) | ((mL1x | U5b) & S2x) | P23x
                                for C4 in C4s:
                                    mC4x = mk(C4)
                                    if _use_prune and C4 and 1 < m4 and C4 not in L4map[C4] and not (mC4x & T41):
                                        n_prune += 1; n_prune_split['41'] += 1
                                        skip_combos += len(L4map[C4]) * N5 * n_c2 * n_c3
                                        continue
                                    for L4 in L4map[C4]:
                                        mL4x = mk(L4)
                                        if _use_prune and L4 and 2 < m4 and not (mL4x & T42):
                                            n_prune += 1; n_prune_split['42'] += 1
                                            skip_combos += N5 * n_c2 * n_c3
                                            continue
                                        if _RT and C1 and 1 < m1 and not (L1 and mL1x == mC1x) and not (mC1x & ((mC4x & U5a) | mC23r)):
                                            n_prune += 1; n_prune_split['11r'] += 1
                                            skip_combos += N5 * n_c2 * n_c3
                                            continue
                                        if _RT and L1 and not (mL1x & ((mL4x & U5b) | ((mL4x | U5b) & S2x) | P23x)):
                                            n_prune += 1; n_prune_split['12r'] += 1
                                            skip_combos += N5 * n_c2 * n_c3
                                            continue
                                        T51 = (mC1x & mC4x) | mC23r
                                        T52 = (mL1x & mL4x) | ((mL1x | mL4x) & S2x) | P23x
                                        W12 = mL4x | S2x; Z12 = (mL4x & S2x) | P23x
                                        W42 = mL1x | S2x; Z42 = (mL1x & S2x) | P23x
                                        for C5 in C5s:
                                            mC5x = mk(C5)
                                            if _TAIL:
                                                T21c = ((mC1x & mC4x) | (mC1x & mC5x) | (mC4x & mC5x)
                                                        | (mC23r & (mC1x | mC4x | mC5x)))
                                            else:
                                                T21c = 0
                                            if _use_prune and C5 and 1 < m5 and C5 not in L5map[C5] and not (mC5x & T51):
                                                n_prune += 1; n_prune_split['51'] += 1
                                                skip_combos += len(L5map[C5]) * n_c2 * n_c3
                                                continue
                                            if _RT and C1 and 1 < m1 and not (L1 and mL1x == mC1x) and not (mC1x & ((mC5x & mC4x) | mC23r)):
                                                n_prune += 1; n_prune_split['11r'] += 1
                                                skip_combos += len(L5map[C5]) * n_c2 * n_c3
                                                continue
                                            if _RT and C4 and 1 < m4 and not (L4 and mL4x == mC4x) and not (mC4x & ((mC5x & mC1x) | mC23r)):
                                                n_prune += 1; n_prune_split['41r'] += 1
                                                skip_combos += len(L5map[C5]) * n_c2 * n_c3
                                                continue
                                            for L5 in L5map[C5]:
                                                mL5x = mk(L5)
                                                if _use_prune and L5 and 2 < m5 and not (mL5x & T52):
                                                    n_prune += 1; n_prune_split['52'] += 1
                                                    skip_combos += n_c2 * n_c3
                                                    continue
                                                if _RT and L1 and not (mL1x & ((mL5x & W12) | Z12)):
                                                    n_prune += 1; n_prune_split['12r'] += 1
                                                    skip_combos += n_c2 * n_c3
                                                    continue
                                                if _RT and L4 and 2 < m4 and not (mL4x & ((mL5x & W42) | Z42)):
                                                    n_prune += 1; n_prune_split['42r'] += 1
                                                    skip_combos += n_c2 * n_c3
                                                    continue
                                                for C2 in C2s:
                                                    if _TAIL and C2 and 1 < m2 - 1 and not (mk(C2) & T21c):
                                                        n_prune += 1; n_prune_split['21'] += 1
                                                        skip_combos += n_c3
                                                        continue
                                                    for C3 in C3s:
                                                        if _TAIL and C3 and 1 < m3 - 1 and not (mk(C3) & T21c):
                                                            n_prune += 1; n_prune_split['31'] += 1
                                                            skip_combos += 1
                                                            continue
                                                        run_checks(C1, L1, C4, L4, C5, L5, C2, C3, C23,
                                                                   (mC23r, mC1x, mC4x, mC5x, mL1x, mL4x, mL5x, mk(C2), mk(C3)))
                else:
                    mC23r = 0; U2b = U3b = 0; n_c2 = n_c3 = 1
                    S2x = 0; P23x = 0
                    T11 = U4a & U5a
                    T12 = U4b & U5b
                    for C1 in C1s:
                        mC1x = mk(C1)
                        if _use_prune and C1 and 1 < m1 and not (mC1x & T11):
                            n_prune += 1; n_prune_split['11'] += 1
                            skip_combos += len(L1map[C1]) * N4 * N5
                            continue
                        T41 = (mC1x & U5a) | mC23r
                        for L1 in L1map[C1]:
                            mL1x = mk(L1)
                            if _use_prune and L1 and not (mL1x & T12):
                                n_prune += 1; n_prune_split['12'] += 1
                                skip_combos += N4 * N5
                                continue
                            T42 = (mL1x & U5b) | ((mL1x | U5b) & S2x) | P23x
                            for C4 in C4s:
                                mC4x = mk(C4)
                                if _use_prune and C4 and 1 < m4 and not (mC4x & T41):
                                    n_prune += 1; n_prune_split['41'] += 1
                                    skip_combos += len(L4map[C4]) * N5
                                    continue
                                for L4 in L4map[C4]:
                                    mL4x = mk(L4)
                                    if _use_prune and L4 and 2 < m4 and not (mL4x & T42):
                                        n_prune += 1; n_prune_split['42'] += 1
                                        skip_combos += N5
                                        continue
                                    if _RT and C1 and 1 < m1 and not (mC1x & ((mC4x & U5a) | mC23r)):
                                        n_prune += 1; n_prune_split['11r'] += 1
                                        skip_combos += N5
                                        continue
                                    if _RT and L1 and not (mL1x & ((mL4x & U5b) | ((mL4x | U5b) & S2x) | P23x)):
                                        n_prune += 1; n_prune_split['12r'] += 1
                                        skip_combos += N5
                                        continue
                                    T51 = (mC1x & mC4x) | mC23r
                                    T52 = (mL1x & mL4x) | ((mL1x | mL4x) & S2x) | P23x
                                    W12 = mL4x | S2x; Z12 = (mL4x & S2x) | P23x
                                    W42 = mL1x | S2x; Z42 = (mL1x & S2x) | P23x
                                    for C5 in C5s:
                                        mC5x = mk(C5)
                                        if _use_prune and C5 and 1 < m5 and not (mC5x & T51):
                                            n_prune += 1; n_prune_split['51'] += 1
                                            skip_combos += len(L5map[C5])
                                            continue
                                        if _RT and C1 and 1 < m1 and not (mC1x & ((mC5x & mC4x) | mC23r)):
                                            n_prune += 1; n_prune_split['11r'] += 1
                                            skip_combos += len(L5map[C5])
                                            continue
                                        if _RT and C4 and 1 < m4 and not (mC4x & ((mC5x & mC1x) | mC23r)):
                                            n_prune += 1; n_prune_split['41r'] += 1
                                            skip_combos += len(L5map[C5])
                                            continue
                                        for L5 in L5map[C5]:
                                            mL5x = mk(L5)
                                            if _use_prune and L5 and 2 < m5 and not (mL5x & T52):
                                                n_prune += 1; n_prune_split['52'] += 1
                                                skip_combos += 1
                                                continue
                                            if _RT and L1 and not (mL1x & ((mL5x & W12) | Z12)):
                                                n_prune += 1; n_prune_split['12r'] += 1
                                                skip_combos += 1
                                                continue
                                            if _RT and L4 and 2 < m4 and not (mL4x & ((mL5x & W42) | Z42)):
                                                n_prune += 1; n_prune_split['42r'] += 1
                                                skip_combos += 1
                                                continue
                                            run_checks(C1, L1, C4, L4, C5, L5, frozenset(), frozenset(), frozenset(),
                                                       (0, mC1x, mC4x, mC5x, mL1x, mL4x, mL5x, 0, 0))

    info = {'dup_cuts': dup_cuts, 'skip_emvm': skip_emvm, 'overlap_kill': n_overlap, 'vm_dup': n_vmdup,
            'n_prune': n_prune, 'prune_split': dict(n_prune_split), 'skip_combos': skip_combos}
    if collect:
        info['survivors'] = [(vv,) + sv for vv, sv in sorted(((vv, SURV[k]) for k, vv in found.items()), key=lambda t: t[0])]
    return sorted(found.values()), total_cand, info


# ==================== UNIFIED ENTRY ====================

# Unified entry returning full records: info['survivors'] = [(vec, cuts, em, vm), ...] sorted by vec.
def enumerate_surv(edges, verts, ext_attach, kin):
    return enumerate_skelg(edges, verts, ext_attach, tuple(kin), collect=True)
