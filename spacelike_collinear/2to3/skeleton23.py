#!/usr/bin/env python3
"""skeleton23.py — skeleton-based cut enumeration for the spacelike-collinear
2->3 kinematics k2-k4 (k1 lives in `skel23.py`; k0 runs through the union
construction, merged from `dev_skel0` on 2026-09-20).  Promoted from the
private prototype
`dev_skelg.py` (2026-09-19).

Model (agreed with 小马, 2026-09-16):
  * H connected; 5 paths P1..P5 (P2/P3 a Y; may pass through v2/v3);
  * C23 ⊇ P2∪P3 (connected or not; every component contains v2 or v3);
    cannot touch P1/P4/P5;
  * wide chains i=1,4,5: base C_i ⊇ the whole P_i; ONE refined level
    (our k's have M_i-1 ≤ 1): refined_opts23 shape = connected, contains
    the leg root, nested in the base, avoiding other external legs;
  * pair chains 2/3: first level = the k1 member rule (overlap with the
    branch = a connected subpath with the leg vertex as one endpoint;
    stem and other branch excluded; free non-path vertices);
  * partial chains (levels may be absent);
  * overlap conditions:
      - literal bookkeeping (小马 2026-09-16): every existing level of every
        chain (incl. bases) must have a nonempty overlap with some cut of
        another direction; set overlap_strong=False to use this form
        (shared vertex, or an edge with endpoints in the two cuts;
        overlap_strict=True removes the edge branch).
      - strengthened overlap (小马 2026-09-19; DEFAULT since promotion):
        for i=1,4,5, every C_i^n cut (n<m; unless it equals C_i^m) must have
        a vertex shared with TWO cuts from two distinct other directions
        j1,j2 (i, j1, j2 all distinct), both of total C-power n (C_j^n for
        wide j; C_j^{n-1}C23 for pair j).  For i=2,3, every C_i^n C23 cut
        (n<m; unless it equals C_i^m C23) must have a vertex shared with two
        distinct-direction cuts of total C-power n.  Overlap form = shared
        vertex only.  (The "unless it equals C_i^m" exemption is vacuous
        for the current k-ladder - subjects only occur for m=INF.)

Use:  enumerate_skelg(edges, verts, ext_attach, kin, use_overlap=True,
                      overlap_strict=False, overlap_strong=True)
"""
import os, sys
from itertools import combinations, product
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fri23 as F
import kin23 as K
from fri23 import (build_overlay, momentum_ok, jets_ok, uncovered_ok,
                   mojetic_all_ok, island_ok, ir_ok, INF)
import skel23 as SKL

_connected_set = SKL._connected_set
_conn_sets = SKL._conn_sets
_enumerate_seqs = SKL._enumerate_seqs
_compat = SKL._compat


def _chain_tops(ext_mode):
    """fri23's M_i / tower levels (non-k1 branch)."""
    def _mf(leg):
        x = ext_mode[leg][2]
        if x == INF:
            return None
        return x + 1 if leg in ('p2', 'p3') else x
    f = [v for v in (_mf(l) for l in ('p1', 'p2', 'p3', 'p4', 'p5')) if v is not None]
    msf = max(f)

    def _top(leg):
        x = ext_mode[leg][2]
        if x == INF:
            return msf
        mi = x + 1 if leg in ('p2', 'p3') else x
        return min(mi, msf)
    M = {l: _top(l) for l in ('p1', 'p2', 'p3', 'p4', 'p5')}
    levels = {'p1': M['p1'] - 1, 'p4': M['p4'] - 1, 'p5': M['p5'] - 1,
              'p2': max(M['p2'] - 1, 1), 'p3': max(M['p3'] - 1, 1)}
    # 小马 2026-09-19: open wide-leg refinement level 1 — k2 gets C1^2/C4^2/C5^2,
    # k3 gets C1^2 (aligning with k4; k0 untouched).
    if (ext_mode['p1'][2] == 1 and ext_mode['p2'][2] == INF
            and ext_mode['p3'][2] == INF and ext_mode['p4'][2] == INF
            and ext_mode['p5'][2] == INF):          # k2 ladder
        levels['p1'] = levels['p4'] = levels['p5'] = 1
    if (ext_mode['p1'][2] == 1 and ext_mode['p2'][2] == 1
            and ext_mode['p3'][2] == INF and ext_mode['p4'][2] == INF
            and ext_mode['p5'][2] == INF):          # k3 ladder
        levels['p1'] = 1
    return M, levels


def _refined_opts(root, base, forbid, adj):
    """refined_opts23 shape: empty + connected supersets of `root` inside
    base, avoiding `forbid`."""
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


def _c23_opts(req, dom, v2, v3, adj):
    """Supersets of `req` inside dom; every connected component contains
    v2 or v3 (connected form or the S4-1 split form; nothing else)."""
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


def _overlap_pair(A, B, adj, strict=False):
    """Overlap: share a vertex, or an edge with endpoints in A / B resp.
    strict=True: shared vertex only (edge branch removed)."""
    if A & B:
        return True
    if strict:
        return False
    for u in A:
        if adj[u] & B:
            return True
    return False


def enumerate_skelg(edges, verts, ext_attach, kin, use_overlap=True,
                    overlap_strict=False, overlap_strong=True):
    if kin == 'k1':
        raise NotImplementedError('k1 lives in skel23')
    if kin == 'k0':
        return _k0_union(edges, verts, ext_attach)
    ext_mode = K.ext_modes(kin)
    m1, m2, m3, m4, m5 = K.KIN[kin]['ms']
    M, LEVELS = _chain_tops(ext_mode)
    if max(LEVELS['p1'], LEVELS['p4'], LEVELS['p5'], LEVELS['p2'], LEVELS['p3']) > 1:
        raise NotImplementedError('deeper chains not implemented yet (levels > 1)')
    edges = [tuple(e) for e in edges]
    V = sorted(verts)
    adj = {v: set() for v in V}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    Vset = set(V)
    extv = set(ext_attach.values())
    roots = {leg: ext_attach[leg] for leg in ('p1', 'p2', 'p3', 'p4', 'p5')}
    v2r, v3r = roots['p2'], roots['p3']
    Hs = [frozenset(S) for r in range(1, len(V) + 1)
          for S in combinations(V, r) if _connected_set(S, adj)]

    total_cand = 0
    dup_cuts = 0
    skip_emvm = 0
    n_vmdup = 0
    n_overlap = 0
    cuts_seen = set()
    emvm_seen = set()
    vm_seen = set()   # vm-level dedup (2026-09-18)
    found = {}

    def run_checks(C1, C1R1, C4, C4R1, C5, C5R1, C2, C3, C23):
        nonlocal total_cand, dup_cuts, skip_emvm, n_overlap, n_vmdup
        total_cand += 1
        cuts = {'C23': set(C23), 'C1': set(C1), 'C4': set(C4), 'C5': set(C5),
                'C1R1': set(C1R1), 'C4R1': set(C4R1), 'C5R1': set(C5R1),
                'C2R1': set(C2), 'C3R1': set(C3), 'C2R2': set(), 'C3R2': set()}
        ck = tuple(sorted((k, tuple(sorted(v))) for k, v in cuts.items()))
        if ck in cuts_seen:
            dup_cuts += 1
            return
        cuts_seen.add(ck)
        if use_overlap:
            # subjects: C_i^n / C_i^nC23 cuts with n < m
            # wide: m = m_i (n=1 base, n=2 first refinement)
            # pair: m = m_i - 1 (first pair level n=1)
            subs = []
            if C1 and 1 < m1:
                subs.append((1, 1, C1))
            if C4 and 1 < m4:
                subs.append((4, 1, C4))
            if C5 and 1 < m5:
                subs.append((5, 1, C5))
            if C1R1:  # 小马 2026-09-19: C1^2 also in the overlap list (三等价 with C4^2/C5^2)
                subs.append((1, 2, C1R1))
            if C4R1 and 2 < m4:
                subs.append((4, 2, C4R1))
            if C5R1 and 2 < m5:
                subs.append((5, 2, C5R1))
            if C2 and 1 < m2 - 1:
                subs.append((2, 1, C2))
            if C3 and 1 < m3 - 1:
                subs.append((3, 1, C3))

            def _lvl_cut(j, p):
                """Direction-j cut with total C-power p: C_j^p for wide
                j=1,4,5; C_j^{p-1}C23 for pair j=2,3 (p=1 -> C23)."""
                d = {1: (C1, C1R1), 4: (C4, C4R1), 5: (C5, C5R1),
                     2: (C23, C2), 3: (C23, C3)}[j]
                return d[p - 1] if 1 <= p <= 2 else None

            for i, n, Sx in subs:
                if overlap_strong:
                    # strengthened (小马 2026-09-19): a vertex shared by
                    # this cut and two cuts from two distinct other
                    # directions, both of total C-power p = n (wide) /
                    # n+1 (pair).  Overlap = shared vertex only.
                    p = n  # partners at total C-power n (小马 2026-09-19: the n+1 was a slip)
                    ok = False
                    for j1 in (1, 2, 3, 4, 5):
                        if j1 == i:
                            continue
                        T1 = _lvl_cut(j1, p)
                        if not T1:
                            continue
                        for j2 in (1, 2, 3, 4, 5):
                            if j2 == i or j2 == j1:
                                continue
                            T2 = _lvl_cut(j2, p)
                            if not T2:
                                continue
                            if any(v in T1 and v in T2 for v in Sx):
                                ok = True
                                break
                        if ok:
                            break
                    if not ok:
                        n_overlap += 1
                        return
                else:
                    # literal bookkeeping (小马 2026-09-16): every C_i^n /
                    # C_i^nC23 cut with n < m must overlap some existing
                    # cut of another direction; targets: other directions'
                    # cuts (bases included) plus C23; C23 never a subject.
                    others = []
                    for dn, cts in ((1, (C1, C1R1)), (4, (C4, C4R1)),
                                    (5, (C5, C5R1)), (2, (C2,)),
                                    (3, (C3,))):
                        if dn == i:
                            continue
                        others += [x for x in cts if x]
                    if C23:
                        others.append(C23)
                    if not any(_overlap_pair(Sx, T, adj,
                                             strict=overlap_strict)
                               for T in others):
                        n_overlap += 1
                        return
        if not uncovered_ok(edges, V, cuts):
            return
        res, why = build_overlay(edges, V, ext_attach, ext_mode, cuts, vm_seen=vm_seen)
        if res is None:
            if why == 'vm-dup':
                n_vmdup += 1
            return
        em, vm = res
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
        pair_opts = []
        both_H = (roots['p2'] in Hset) and (roots['p3'] in Hset)
        if both_H:
            pair_opts = [(None, None, None, None)]
        else:
            fr = {roots['p2'], roots['p3']}
            seq2 = (_enumerate_seqs(roots['p2'], Hset, adj, Vset, extv, fr)
                    if roots['p2'] not in Hset else [])
            seq3 = (_enumerate_seqs(roots['p3'], Hset, adj, Vset, extv, fr)
                    if roots['p3'] not in Hset else [])
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
                if (S2 is not None or S3 is not None) and \
                        (((S2 or frozenset()) | (S3 or frozenset())) & u):
                    continue
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
                if S2 is not None or S3 is not None:
                    for C23 in _c23_opts(a2 | a3, Vset - Hset - a1 - a4 - a5,
                                         v2r, v3r, adj):
                        D = frozenset(set(C23) - a2 - a3)
                        C2s = {frozenset()}
                        if br2:
                            for j in range(len(br2)):
                                T = frozenset(br2[j:])
                                C2s.update(_conn_sets(T, T | D, adj))
                        C3s = {frozenset()}
                        if br3:
                            for j in range(len(br3)):
                                T = frozenset(br3[j:])
                                C3s.update(_conn_sets(T, T | D, adj))
                        for C1 in C1s:
                            L1o = (_refined_opts(roots['p1'], C1,
                                                 extv - {roots['p1']}, adj)
                                   if LEVELS['p1'] else [frozenset()])
                            for L1 in L1o:
                                for C4 in C4s:
                                    L4o = (_refined_opts(roots['p4'], C4,
                                                         extv - {roots['p4']}, adj)
                                           if LEVELS['p4'] else [frozenset()])
                                    for L4 in L4o:
                                        for C5 in C5s:
                                            L5o = (_refined_opts(roots['p5'], C5,
                                                                 extv - {roots['p5']}, adj)
                                                   if LEVELS['p5'] else [frozenset()])
                                            for L5 in L5o:
                                                for C2 in C2s:
                                                    for C3 in C3s:
                                                        run_checks(C1, L1, C4, L4,
                                                                   C5, L5, C2, C3, C23)
                else:
                    for C1 in C1s:
                        L1o = (_refined_opts(roots['p1'], C1,
                                             extv - {roots['p1']}, adj)
                               if LEVELS['p1'] else [frozenset()])
                        for L1 in L1o:
                            for C4 in C4s:
                                L4o = (_refined_opts(roots['p4'], C4,
                                                     extv - {roots['p4']}, adj)
                                       if LEVELS['p4'] else [frozenset()])
                                for L4 in L4o:
                                    for C5 in C5s:
                                        L5o = (_refined_opts(roots['p5'], C5,
                                                             extv - {roots['p5']}, adj)
                                               if LEVELS['p5'] else [frozenset()])
                                        for L5 in L5o:
                                            run_checks(C1, L1, C4, L4, C5, L5,
                                                       frozenset(), frozenset(),
                                                       frozenset())

    info = {'dup_cuts': dup_cuts, 'skip_emvm': skip_emvm, 'overlap_kill': n_overlap, 'vm_dup': n_vmdup}
    return sorted(found.values()), total_cand, info


# ---------------------------------------------------------------------------
# k0 union construction (merged from dev_skel0, 2026-09-20 — 小马)
#
# Scheme (小马, 2026-09-18; validated vs pySecDec on rand100_4l_no2v,
# rand200_4l_no2v, rand100_5l_no2v — all green; the wide-angle skeleton uses
# the same construction):
#   * H connected (as before);
#   * per leg i in 1,4,5:  P_i = connected set ⊆ V∖H containing root_i and
#     touching H  (think: union of several possibly-overlapping root→H paths);
#   * P_23 = Q2 ∪ Q3, each Qk = connected set ⊆ V∖H containing root_k and
#     touching H (Q2/Q3 may overlap each other; Qk = ∅ iff root_k ∈ H);
#   * P_1, P_23, P_4, P_5 pairwise disjoint;
#   * leftover = V − (P_1∪P_23∪P_4∪P_5).
#     cond3 (comp_adj2): every component of the leftover subgraph must be
#       adjacent to ≥2 of the four groups;
#     cond5 variants: 'vert_adj2' (every leftover vertex adjacent to ≥2
#       groups) / 'cut_mem2' (every leftover vertex contained in ≥2 cuts).
#   * cuts: C_g connected ⊇ P_g inside (V∖H) minus the other groups' P's;
#     C_g = ∅ iff P_g = ∅; every component of C23 contains v2 or v3.
#
# _k0_union returns (sorted vecs, total_cand, info) — same contract as
# enumerate_skelg.
# ---------------------------------------------------------------------------

def _comps(sub, adj):
    sub = set(sub); seen = set(); out = []
    for s in sorted(sub):
        if s in seen:
            continue
        stack = [s]; comp = set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            for w in adj[u]:
                if w in sub and w not in comp:
                    stack.append(w)
        seen |= comp
        out.append(comp)
    return out


def _c23_sets(req, dom, v2, v3, adj):
    req = set(req); dd = set(dom)
    if not req <= dd:
        return []
    extras = sorted(dd - req)
    out = []
    for r in range(len(extras) + 1):
        for add in combinations(extras, r):
            S = req | set(add)
            ok = True
            for K_ in _comps(S, adj):
                if not (v2 in K_ or v3 in K_):
                    ok = False
                    break
            if ok:
                out.append(frozenset(S))
    return out


def _k0_union(edges, verts, ext_attach,
                    comp_adj2=True, vert_adj2=False, cut_mem2=True,
                    excl_h=True, skip_if_lt2=True, kill_exact1_only=False,
                    count_startpoints=True, keepP=False, cap=2000000,
                    corner_prune=True):
    # defaults = the blessed k0 config (2026-09-18): comp_adj2 + excl_h +
    # skip_if_lt2 + startpoints + corner_prune; dedup key includes H.
    edges = [tuple(e) for e in edges]
    V = sorted(verts)
    adj = {v: set() for v in V}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    Vset = set(V)
    extv = set(ext_attach.values())
    roots = {l: ext_attach[l] for l in ('p1', 'p2', 'p3', 'p4', 'p5')}
    ext_mode = K.ext_modes('k0')
    Hs_all = [frozenset(S) for r in range(1, len(V) + 1)
              for S in combinations(V, r) if _connected_set(S, adj)]
    n_corner_skip = 0
    if corner_prune:
        # H is the hardest mode: an H-edge incident with a vertex forces that
        # vertex to be H as well (小马, 2026-09-18).  Operational rule: skip H
        # if some NON-root v∉H has all its edges into H (N(v) ⊆ H) — such v can
        # never sit on a path or in a cut, so every config of this H would
        # need v absorbed anyway.  Roots exempt (a root can be covered by its
        # leg / can appear in a cut).  See dev_skel0_corner_summary.txt.
        Hs = []
        for H in Hs_all:
            bad = False
            for v in V:
                if v in H:
                    continue
                if v in extv:
                    continue
                if adj[v] and adj[v] <= H:
                    bad = True
                    break
            if bad:
                n_corner_skip += 1
            else:
                Hs.append(H)
    else:
        Hs = Hs_all
    total_cand = 0
    dup_cuts = 0
    skip_emvm = 0
    n_vmdup = 0
    n_filt = 0
    n_combo = 0
    n_filtF = 0
    cuts_seen = set()
    emvm_seen = set()
    vm_seen = set()   # vm-level dedup (2026-09-18)
    found = {}

    def run_checks(C1, C4, C5, C23, leftover, H, P1, a23, P4, P5):
        nonlocal total_cand, dup_cuts, skip_emvm, n_filt, n_vmdup
        total_cand += 1
        if cap and total_cand > cap:
            raise RuntimeError('cap')
        if cut_mem2 and leftover:
            # NOTE: must run BEFORE the cut-set dedup — the same cut-set can be
            # reached under different (H, P) configs whose leftover sets differ;
            # a violating config must not block a clean one via cuts_seen
            # (2026-09-18 fix, same class as the H-key dedup bug).  Default ON
            # since 2026-09-18 evening (小马): every vertex of V∖(H∪P's) must
            # lie in >=2 cuts.
            csets = (set(C1), set(C23), set(C4), set(C5))
            for z in (leftover - set(H)):
                if sum(1 for cs in csets if z in cs) < 2:
                    n_filt += 1
                    return
        cuts = {'C23': set(C23), 'C1': set(C1), 'C4': set(C4), 'C5': set(C5),
                'C1R1': set(), 'C4R1': set(), 'C5R1': set(),
                'C2R1': set(), 'C3R1': set(), 'C2R2': set(), 'C3R2': set()}
        ck = tuple(sorted((k, tuple(sorted(v))) for k, v in cuts.items()))
        if ck in cuts_seen:
            dup_cuts += 1
            return
        cuts_seen.add(ck)
        if not uncovered_ok(edges, V, cuts):
            return
        res, why = build_overlay(edges, V, ext_attach, ext_mode, cuts, vm_seen=vm_seen)
        if res is None:
            if why == 'vm-dup':
                n_vmdup += 1
            return
        em, vm = res
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
        if keepP:
            found[(ek, vk)] = (vec, cuts, em, vm, (H, P1, a23, P4, P5))
        else:
            found[(ek, vk)] = (vec, cuts, em, vm)

    seenP = set()
    for H in Hs:
        Hset = set(H)
        dom_base = Vset - Hset
        opt = {}
        bad = False
        for leg in ('p1', 'p4', 'p5'):
            r = roots[leg]
            if r in Hset:
                opt[leg] = [frozenset()]
                continue
            allowed = {w for w in Vset if w not in Hset and
                       (w == r or w not in extv)}
            os_ = [S for S in _conn_sets({r}, allowed, adj)
                   if any(adj[u] & Hset for u in S)]
            if not os_:
                bad = True
                break
            opt[leg] = os_
        if bad:
            continue
        allowed23 = {w for w in Vset if w not in Hset and
                     (w in (roots['p2'], roots['p3']) or w not in extv)}
        if roots['p2'] in Hset:
            q2 = [frozenset()]
        else:
            q2 = [S for S in _conn_sets({roots['p2']}, allowed23, adj)
                  if any(adj[u] & Hset for u in S)]
            if not q2:
                continue
        if roots['p3'] in Hset:
            q3 = [frozenset()]
        else:
            q3 = [S for S in _conn_sets({roots['p3']}, allowed23, adj)
                  if any(adj[u] & Hset for u in S)]
            if not q3:
                continue
        for P1 in opt['p1']:
            for P4 in opt['p4']:
                if P1 & P4:
                    continue
                for P5 in opt['p5']:
                    if P1 & P5 or P4 & P5:
                        continue
                    u = P1 | P4 | P5
                    for b2 in q2:
                        for b3 in q3:
                            a23 = b2 | b3
                            if a23 & u:
                                continue
                            # NOTE (2026-09-18 fix): the filter depends on H (leftover =
                            # V - H - P's), so the same P-quad with a different H can give
                            # a different verdict; H must enter the dedup key.  The old
                            # cross-H dedup skipped larger-H configs entirely -> false
                            # losses of regions that a larger-H config keeps.
                            keyP = (H, P1, a23, P4, P5)
                            if keyP in seenP:
                                continue
                            seenP.add(keyP)
                            n_combo += 1
                            leftover = Vset - P1 - a23 - P4 - P5
                            lf = (leftover - Hset) if excl_h else leftover
                            groups = [g for g in (P1, a23, P4, P5) if g]
                            do_filter = (comp_adj2 or vert_adj2)
                            if do_filter and skip_if_lt2 and len(groups) < 2:
                                do_filter = False
                            if do_filter:
                                good = True
                                gg = []
                                for g in groups:
                                    if count_startpoints:
                                        gg.append(g | {h for h in Hset
                                                       if adj[h] & g})
                                    else:
                                        gg.append(g)
                                if comp_adj2:
                                    for K_ in _comps(lf, adj):
                                        cnt = 0
                                        for g in gg:
                                            if any((adj[uu] & g) for uu in K_):
                                                cnt += 1
                                        if kill_exact1_only:
                                            if cnt == 1:
                                                good = False
                                                break
                                        elif cnt < 2:
                                            good = False
                                            break
                                if good and vert_adj2:
                                    for z in lf:
                                        cnt = sum(1 for g in gg
                                                  if (adj[z] & g))
                                        if cnt < 2:
                                            good = False
                                            break
                                if not good:
                                    n_filtF += 1
                                    continue
                            C1s = (_conn_sets(P1, dom_base - a23 - P4 - P5, adj)
                                   if P1 else [frozenset()])
                            C4s = (_conn_sets(P4, dom_base - P1 - a23 - P5, adj)
                                   if P4 else [frozenset()])
                            C5s = (_conn_sets(P5, dom_base - P1 - a23 - P4, adj)
                                   if P5 else [frozenset()])
                            if not C1s or not C4s or not C5s:
                                continue
                            if a23:
                                C23s = _c23_sets(a23, dom_base - P1 - P4 - P5,
                                                 roots['p2'], roots['p3'], adj)
                            else:
                                C23s = [frozenset()]
                            if not C23s:
                                continue
                            for C1 in C1s:
                                for C4 in C4s:
                                    for C5 in C5s:
                                        for C23 in C23s:
                                            run_checks(C1, C4, C5, C23,
                                                       leftover, H,
                                                       P1, a23, P4, P5)
    info = {'dup_cuts': dup_cuts, 'skip_emvm': skip_emvm,
            'filt_kill': n_filt, 'n_combo': n_combo, 'filtF_kill': n_filtF,
            'corner_skip': n_corner_skip, 'vm_dup': n_vmdup}
    reps = list(found.values())
    return sorted(v[0] for v in reps), total_cand, info
