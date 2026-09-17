#!/usr/bin/env python3
"""
skel23.py — skeleton-based region enumeration for spacelike-collinear 2->3
(k1; the engine behind run_random.py --engine skel).

Rules:
  * member-cut rule: a nonempty C2C23 / C3C23 takes from the paths only a
    connected subpath of its own branch (p2 branch / p3 branch) with the leg
    vertex as one endpoint, excluding the shared subpath and the other
    branch; vertices on neither path may be included freely.
    C23 still contains the whole P2 and P3.
  * acceptance filter: keep a candidate if the wide-leg share condition (M)
    holds — for i in 1,4,5 with nonempty P_i, some vertex shared by C_i, C23
    and another wide cut C_j (j != i); otherwise keep it only via one of the
    three symmetric pathways (all five P_i nonempty; with s_ab = C_a∩C_b∩C23
    exactly one of s14/s15/s45 nonempty and s145 = C1∩C4∩C5 nonempty):
      PA: s14 ;  PB: s15 ;  PC: s45.
  Validated against fri23 / pySecDec on the random-batch series of 2026-09
  (spot checks plus full batches via run_random.py --engine skel).

Idea (skeleton construction): don't sweep the whole cut-superset space.
Build each region from a skeleton —
    H       : the uncovered vertex set,
    P1..P5  : the leg "jet" paths from each external vertex to H
              (P2/P3 form a Y: a shared prefix out of H, then split; they
               may pass through v2/v3, which are both roots of C23),
then expand each leg's cut supersets inside its domain and run fri23's
standard check chain.  Canonical de-duplication skips repeated
representations: identical cut tuples are skipped outright; after the
coverage check, candidates with identical (em, vm) are skipped too — every
check downstream of `uncovered_ok` is a function of (em, vm) alone, so the
skip is safe.

Public API:
    enumerate_skel(edges, verts, ext_attach, kin='k1')
        -> (vecs, n_cand, info)
    vecs   : sorted scaling vectors (same convention as fri23)
    n_cand : raw candidate count (pre-dedup)
    info   : {'dup_cuts': int, 'skip_emvm': int, 'wide_kill': int, 'pathway': int}

Batch use: run_random.py --engine skel.
"""
import os, sys
from itertools import combinations, product

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fri23 as F
import kin23 as K
from fri23 import (build_overlay, momentum_ok, jets_ok, uncovered_ok,
                   mojetic_all_ok, island_ok, ir_ok)


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


def _enumerate_seqs(start, Hset, adj, Vset, extv, friends=()):
    allowed = [w for w in Vset if w not in Hset and
               (w == start or w in friends or w not in extv)]
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


def _compat(seq2, seq3):
    """Two leg-paths for p2/p3 are compatible iff, seen from H backwards,
    they share only a common prefix (the Y stem) and never meet again."""
    r2 = seq2[::-1]; r3 = seq3[::-1]
    k = 0
    while k < min(len(r2), len(r3)) and r2[k] == r3[k]:
        k += 1
    return not (set(r2[k:]) & set(r3[k:]))


def enumerate_skel(edges, verts, ext_attach, kin='k1'):
    if kin != 'k1':
        raise NotImplementedError('skel23 supports k1 only (for now)')
    edges = [tuple(e) for e in edges]
    V = sorted(verts)
    adj = {v: set() for v in V}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    Vset = set(V)
    extv = set(ext_attach.values())
    roots = {leg: ext_attach[leg] for leg in ('p1', 'p2', 'p3', 'p4', 'p5')}
    ext_mode = K.ext_modes(kin)
    Hs = [frozenset(S) for r in range(1, len(V) + 1)
          for S in combinations(V, r) if _connected_set(S, adj)]

    total_cand = 0
    dup_cuts = 0
    skip_emvm = 0
    n_wide = 0
    n_path = 0
    cuts_seen = set()
    emvm_seen = set()
    found = {}

    def run_checks(C1, C2, C3, C4, C5, C23, legreq):
        nonlocal total_cand, dup_cuts, skip_emvm, n_wide, n_path
        total_cand += 1
        # acceptance filter + three symmetric pathways (2026-09-16):
        # (M) every i in 1,4,5 with nonempty P_i shares a vertex of
        #     C_i, C23 and another wide cut C_j; or if M fails, one of
        # (PA) all five P_i nonempty; exists v in C1∩C4∩C23; none in
        #     C1∩C5∩C23; none in C4∩C5∩C23; exists v in C1∩C4∩C5;
        # (PB) all five; exists v in C1∩C5∩C23; none in C1∩C4∩C23;
        #     none in C4∩C5∩C23; exists v in C1∩C4∩C5;
        # (PC) all five; exists v in C4∩C5∩C23; none in C1∩C4∩C23;
        #     none in C1∩C5∩C23; exists v in C1∩C4∩C5.
        _W = ((1, C1), (4, C4), (5, C5))
        _ok_main = True
        for _t, (_i, _Ci) in enumerate(_W):
            if not legreq[_t]:
                continue
            if not any((_Ci & C23 & _Cj) for _j, _Cj in _W if _j != _i):
                _ok_main = False
                break
        if not _ok_main:
            _all5 = legreq[0] and legreq[1] and legreq[2] and legreq[3] and legreq[4]
            _s14 = C1 & C4 & C23
            _s15 = C1 & C5 & C23
            _s45 = C4 & C5 & C23
            _s145 = C1 & C4 & C5
            _ok_path = _all5 and _s145 and (
                (_s14 and not _s15 and not _s45) or
                (_s15 and not _s14 and not _s45) or
                (_s45 and not _s14 and not _s15))
            if not _ok_path:
                n_wide += 1
                return
            n_path += 1
        cuts = {'C23': set(C23), 'C1': set(C1), 'C4': set(C4), 'C5': set(C5),
                'C1R1': set(), 'C4R1': set(), 'C5R1': set(),
                'C2R1': set(C2), 'C3R1': set(C3), 'C2R2': set(), 'C3R2': set()}
        ck = tuple(sorted((k, tuple(sorted(v))) for k, v in cuts.items()))
        if ck in cuts_seen:
            dup_cuts += 1
            return
        cuts_seen.add(ck)
        res, why = build_overlay(edges, V, ext_attach, ext_mode, cuts)
        if res is None:
            return
        em, vm = res
        if not uncovered_ok(edges, V, cuts):
            return
        ek = tuple((i, F.name(m)) for i, m in enumerate(em))
        vk = tuple(sorted((w, F.name(m)) for w, m in vm.items()))
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

    for H in Hs:
        Hset = set(H)
        outs = [leg for leg in ('p1', 'p2', 'p3', 'p4', 'p5')
                if roots[leg] not in Hset]
        fam = {}
        ok = True
        for leg in ('p1', 'p4', 'p5'):
            if leg not in outs:
                fam[leg] = None
                continue
            start = roots[leg]
            sets = set()
            allowed = [w for w in Vset if w not in Hset and
                       (w == start or w not in extv)]
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
        if roots['p2'] in Hset:
            if roots['p3'] not in Hset:
                continue
            pair_opts = [(None, None, None, None)]
        else:
            fr = {roots['p2'], roots['p3']}
            seq2 = _enumerate_seqs(roots['p2'], Hset, adj, Vset, extv, fr)
            seq3 = _enumerate_seqs(roots['p3'], Hset, adj, Vset, extv, fr)
            if not seq2 or not seq3:
                continue
            # 2026-09-16 (agreed rule): keep each compatible realization, and
            # split it at the shared prefix: br2/br3 = the branch sequences
            # (from the split down to v2 / v3, i.e. reversed path after the
            # shared part).  Member cuts may take any connected subpath of
            # their own branch with the leg vertex as one endpoint.
            pair_opts = []
            seen_po = set()
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
            if not pair_opts:
                continue
        l1 = fam['p1'] if fam['p1'] is not None else [None]
        l4 = fam['p4'] if fam['p4'] is not None else [None]
        l5 = fam['p5'] if fam['p5'] is not None else [None]
        for S1, S4, S5 in product(l1, l4, l5):
            u = set(); skip = False
            for S in (S1, S4, S5):
                if S is not None:
                    if S & u:
                        skip = True
                        break
                    u |= S
            if skip:
                continue
            for S2, S3, br2, br3 in pair_opts:
                if S2 is not None and ((S2 | S3) & u):
                    continue
                legreq = (S1 is not None, S4 is not None, S5 is not None,
                          S2 is not None, S3 is not None)
                a1 = S1 or frozenset(); a4 = S4 or frozenset()
                a5 = S5 or frozenset(); a2 = S2 or frozenset()
                a3 = S3 or frozenset()
                C1s = (_conn_sets(a1, Vset - Hset - a2 - a3 - a4 - a5, adj)
                       if S1 is not None else [frozenset()])
                if not C1s:
                    continue
                C4s = (_conn_sets(a4, Vset - Hset - a1 - a2 - a3 - a5, adj)
                       if S4 is not None else [frozenset()])
                if not C4s:
                    continue
                C5s = (_conn_sets(a5, Vset - Hset - a1 - a2 - a3 - a4, adj)
                       if S5 is not None else [frozenset()])
                if not C5s:
                    continue
                if S2 is not None:
                    for C23 in _conn_sets(a2 | a3, Vset - Hset - a1 - a4 - a5, adj):
                        # agreed rule (2026-09-16): a nonempty C2C23/C3C23
                        # takes from the paths only a connected subpath of its
                        # own branch with the leg vertex as one endpoint
                        # (shared subpath and the other branch excluded);
                        # vertices on neither path may be included freely.
                        D = frozenset(set(C23) - a2 - a3)
                        C2s = {frozenset()}
                        for j in range(len(br2)):
                            T = frozenset(br2[j:])
                            C2s.update(_conn_sets(T, T | D, adj))
                        C3s = {frozenset()}
                        for j in range(len(br3)):
                            T = frozenset(br3[j:])
                            C3s.update(_conn_sets(T, T | D, adj))
                        for C2 in C2s:
                            for C3 in C3s:
                                for C1 in C1s:
                                    for C4 in C4s:
                                        for C5 in C5s:
                                            run_checks(C1, C2, C3, C4, C5, C23, legreq)
                else:
                    for C1 in C1s:
                        for C4 in C4s:
                            for C5 in C5s:
                                run_checks(C1, frozenset(), frozenset(),
                                           C4, C5, frozenset(), legreq)

    info = {'dup_cuts': dup_cuts, 'skip_emvm': skip_emvm, 'wide_kill': n_wide, 'pathway': n_path}
    return sorted(found.values()), total_cand, info
