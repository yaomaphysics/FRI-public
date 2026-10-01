#!/usr/bin/env python3
"""skeleton.py — region enumerators for the Regge 2->2 FRI: the skeleton engine (skel_regions, skel_regions_k1) and the FRI enumerators (fri_regions_full, fri_regions_onshell, fri_regions_offshell).

Promoted from the private prototype `dev_regge_skel.py` (2026-09-22).

Spec (2026-09-20):
  §1 Γ = H∪G first: enumerate all nonempty connected Γ (induced); pair legs 1/3
     get the Y (shared stem out of Γ, split at a junction into br1 → v1, br3 → v3);
     pair legs 2/4 the same (br2 → v2, br4 → v4); _compat: seen from Γ backwards
     only a common prefix, branches never meet again; the two pairs must not overlap.
  §2 Chains: C13 ⊇ P1∪P3 (components each contain v1 or v3; domain V∖Γ∖Y24);
     C24 ⊇ P2∪P4 symmetric. Member cuts C1C13/C3C13 = connected supersets of a
     branch subpath with the leg vertex as one endpoint (suffix br[j:]), stem and
     other branch excluded, free vertices of the family cut may join; C2C24/C4C24
     likewise. Higher levels nest (refined_opts shape inside the level above).
     Partial chains; both roots in Γ -> empty chain.
  §3 Pruning: coverage AT GENERATION (every vertex outside Γ in some cut; enforced
     as C24 ⊇ Y24 ∪ (V∖Γ)∖C13 given C13, so C13∪C24 ⊇ V∖Γ); identical cuts
     skipped; strong overlap (mate at level n, cross at n−1, "two of three";
     exemption: coincides with a deeper same-leg cut n'>n); em/vm built; repeated
     (em,vm) skipped before the remaining checks; checks unchanged.
     Fast path (2026-09-22): cuts carry bitmasks; legs-2/4 level-1 chain filter
     + level-2 counting quick-kill replicate overlap's first checks (gated by
     l2quick; A/B set-equal on the case suite).

Depth caps follow the mode-level framework (k0/k5 (0,0), k2/k3 (0,1),
k4 (1,2); k1 is loop-count driven).  Checks: region_checker._build_region
(stock, unchanged).  k0-k5 are all covered; k1 goes through
skel_regions_k1 (L-gated refinements + I0/I1/I2 + Cond-1/2/3).

Usage:
    python3 skeleton.py                 # default small-graph sweep
    python3 skeleton.py box k5 hexagon k4    # name kin [name kin ...]
"""
import sys, os, time
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import region_checker as R                                        # noqa: E402
from regge_graphs import GRAPHS, KIN, EXT_ATTACHES            # noqa: E402
from regge_graphs import DEFAULT_EXT_ATTACH as EXT_ATTACH     # noqa: E402
from primitives import (conn_sets, comp_sets, components, connected, refined_opts, ext_m,
                        possibly_softest, meet, to_scaling, clear_graph_caches)  # noqa: E402

FS = frozenset


# ---------------------------- helpers ----------------------------


def enumerate_seqs(start, Gset, adj, Vset, extv, friends):
    allowed = [w for w in Vset if w not in Gset and (w == start or w in friends or w not in extv)]
    seqs = []
    st = [(start, {start}, (start,))]
    while st:
        cur, vis, path = st.pop()
        if adj[cur] & Gset:
            seqs.append(path)
        for w in adj[cur]:
            if w in allowed and w not in vis:
                st.append((w, vis | {w}, path + (w,)))
    return seqs


def compat(a, b):
    ra, rb = a[::-1], b[::-1]
    k = 0
    while k < min(len(ra), len(rb)) and ra[k] == rb[k]:
        k += 1
    return not (set(ra[k:]) & set(rb[k:]))


# Y realizations for one pair; (None,None,None,None) when both roots in Γ.
def pair_opts_for(va, vb, Gset, adj, Vset, extv):
    if va in Gset and vb in Gset:
        return [(None, None, None, None)]
    fr = {va, vb}
    seqA = enumerate_seqs(va, Gset, adj, Vset, extv, fr) if va not in Gset else []
    seqB = enumerate_seqs(vb, Gset, adj, Vset, extv, fr) if vb not in Gset else []
    if not seqA and not seqB:
        return []
    opts = []; seen = set()
    if seqA and seqB:
        for a in seqA:
            ra = a[::-1]
            for b in seqB:
                if not compat(a, b):
                    continue
                rb = b[::-1]
                k = 0
                while k < min(len(ra), len(rb)) and ra[k] == rb[k]:
                    k += 1
                key = (FS(a), FS(b), ra[k:], rb[k:])
                if key in seen:
                    continue
                seen.add(key)
                opts.append(key)
    elif seqA:
        for a in seqA:
            key = (FS(a), FS(), a[::-1], None)
            if key in seen:
                continue
            seen.add(key); opts.append(key)
    else:
        for b in seqB:
            key = (FS(), FS(b), None, b[::-1])
            if key in seen:
                continue
            seen.add(key); opts.append(key)
    return opts


# First level of a leg chain: connected supersets of a branch suffix T = br[j:] (leg vertex as one endpoint) inside T
# ∪ D.
def member_opts(br, Dcut, edges):
    out = {FS()}
    if br:
        for j in range(len(br)):
            T = FS(br[j:])
            out.update(conn_sets(T, set(T) | set(Dcut), edges))
    return list(out)


# Nested tower options: tuples of levels (empty at any level forces deeper empty).
def towers(root, lvl1_opts, depth, forbid, edges):
    if depth <= 0:
        return [()]
    if depth == 1:
        return [(s,) for s in lvl1_opts]
    t = [(s,) for s in lvl1_opts]
    for _ in range(depth - 1):
        t = [chain + (s2,) for chain in t for s2 in refined_opts(edges, root, chain[-1], forbid)]
    return t


_MATE = {1: 3, 3: 1, 2: 4, 4: 2}
_CROSS = {1: (2, 4), 3: (2, 4), 2: (1, 3), 4: (1, 3)}

# diagnostic flag: True -> L2 quick-kill does not skip, only counts
_SHADOW_L2 = False


def _cut_at(f13, f24, leg, lvl):
    base, lv = f13 if leg in (1, 3) else f24
    if lvl == 0:
        return base
    return lv[leg][lvl - 1] if lvl - 1 < len(lv[leg]) else None


# Level-1 necessary condition of overlap_ok for legs 2/4, at chain level: subject C_iC_fam needs >=2 of {mate at n=1,
# cross-base, cross-base} hits (= cut & base); skipped when mi cuts it off, the cut is empty, or it coincides with
# its deeper twin (exemption). chain: (frozenset, mask).
def _l1_keep(chain, mbase, mi):
    if mi is not None and not (1 < mi):
        return True
    if len(chain) == 0:
        return True
    Sxm = chain[0][1]
    if not Sxm:
        return True
    if len(chain) >= 2 and chain[1][1] == Sxm:
        return True
    return bool(Sxm & mbase)


# Strong overlap, the regge form (2026-09-20); MASK version (2026-09-22): fam structures carry int bitmasks (0 =
# empty) instead of frozensets. subject C_i^n C_fam (n<m): need a vertex shared with two of the three partners {mate
# at level n, cross legs at level n−1}; exemption: coincides with a deeper same-leg cut (n'>n).
def overlap_ok(fam13, fam24, m):
    for leg in (1, 2, 3, 4):
        lv = (fam13[1] if leg in (1, 3) else fam24[1])[leg]
        for n0, Sx in enumerate(lv, start=1):
            if not Sx:
                continue
            mi = m[leg]
            if mi is not None and not n0 < mi:
                continue
            # exemption (the original text): coincides with a deeper same-leg cut n' > n
            lv_rest = lv[n0:]
            if any(t == Sx for t in lv_rest):
                continue
            cr = _CROSS[leg]
            T1 = _cut_at(fam13, fam24, _MATE[leg], n0)
            T2 = _cut_at(fam13, fam24, cr[0], n0 - 1)
            T3 = _cut_at(fam13, fam24, cr[1], n0 - 1)
            cnt2 = (1 if (T1 and (Sx & T1)) else 0) + (1 if (T2 and (Sx & T2)) else 0) + (1 if (T3 and (Sx & T3)) else 0)
            if cnt2 < 2:
                return False
    return True


# Bitmask helper for vertex sets (shared by both enumerators).
def _make_mk(mcache, bidx):
    def mk(S):
        r = mcache.get(S)
        if r is None:
            r = 0
            for v in S:
                r |= bidx[v]
            mcache[S] = r
        return r
    return mk


# ---------------------------- main enumerator ----------------------------
def skel_regions(edges, verts, ext_attach, ext_mode):
    clear_graph_caches() # start each case with empty graph-kernel memo tables
    verts = sorted(verts)
    E = [tuple(e) for e in edges]
    adj = {v: set() for v in verts}
    for a, b in E:
        adj[a].add(b); adj[b].add(a)
    Vset = set(verts)
    extv = set(ext_attach.values())
    v1, v2, v3, v4 = (ext_attach['p1'], ext_attach['p2'], ext_attach['p3'], ext_attach['p4'])
    bidx = {v: 1 << i for i, v in enumerate(verts)}
    mcache = {}

    mk = _make_mk(mcache, bidx)

    def wrap(chs):
        return [tuple((S, mk(S)) for S in ch) for ch in chs]

    # depths (mirror fri_regions_offshell)
    ms = {n: ext_m(ext_mode.get(n)) for n in ext_attach}
    ms_m = {n: (None if v is None else v - 1) for n, v in ms.items()}
    m_int = {1: ms_m['p1'], 2: ms_m['p2'], 3: ms_m['p3'], 4: ms_m['p4']}
    ps = possibly_softest(ms_m['p1'], ms_m['p2'], ms_m['p3'], ms_m['p4'])
    if not ps:
        raise SystemExit('all-lightlike (k1): k1 engine not in this prototype')
    if len(ps) == 2:
        d13 = d24 = int(ps[0].split('^')[1].split('C')[0])
    else:
        Mside = int(ps[0].split('^')[1].split('C')[0])
        if ps[0].endswith('C13'):
            d13, d24 = Mside, Mside - 1
        else:
            d13, d24 = Mside - 1, Mside
    if max(d13, d24) > 2:
        raise SystemExit('depth > 2 not implemented')
    # L2 quick-kill gate: residual overlap checks are exactly legs 2/4 at
    # level 2; legs 1/3 must be silenced here (mi <= 1) and 2/4 unchecked.
    l2quick = (d24 == 2 and m_int[1] is not None and m_int[1] <= 1 and m_int[3] is not None and m_int[3] <= 1 and m_int[2] is None and m_int[4] is None)

    Gammas = [FS(S) for r in range(1, len(verts) + 1) for S in combinations(verts, r) if connected(set(S), E)]

    cnt = dict(gamma=len(Gammas), combos=0, cov_kill=0, pair_overlap_kill=0, dup_cuts=0, ov_kill=0, skip_pre=0, build_none=0, out=0, l2_kill=0, shadow_bad=0, shadow_inv=0)
    found = {}
    seen_cuts = set()
    seen_pre = set()

    for Gset in Gammas:
        opts13 = pair_opts_for(v1, v3, Gset, adj, Vset, extv)
        if not opts13:
            continue
        opts24 = pair_opts_for(v2, v4, Gset, adj, Vset, extv)
        if not opts24:
            continue
        forbid1 = {v2, v3, v4}
        forbid3 = {v1, v2, v4}
        forbid2 = {v1, v3, v4}
        forbid4 = {v1, v2, v3}
        for o13 in opts13:
            P1, P3 = (o13[0] or FS()), (o13[1] or FS())
            for o24 in opts24:
                P2, P4 = (o24[0] or FS()), (o24[1] or FS())
                P12 = set(P1) | set(P3)
                P24 = set(P2) | set(P4)
                if P12 & P24:
                    cnt['pair_overlap_kill'] += 1
                    continue
                dom13 = Vset - Gset - P24
                dom24 = Vset - Gset - P12
                C13opts = comp_sets(P12, dom13, v1, v3, E) if P12 else [FS()]
                if not C13opts:
                    cnt['cov_kill'] += 1
                    continue
                for C13 in C13opts:
                    mC13 = mk(C13)
                    rem = Vset - Gset - set(C13)
                    if P24:
                        C24opts = comp_sets(P24 | rem, dom24, v2, v4, E)
                    else:
                        C24opts = [FS()] if not rem else []
                    if not C24opts:
                        cnt['cov_kill'] += 1
                        continue
                    for C24 in C24opts:
                        mC24 = mk(C24)
                        # ---- chains ----
                        lv13 = {1: [], 3: []}; lv24 = {2: [], 4: []}
                        if P12:
                            D13 = set(C13) - P12
                            if d13 >= 1:
                                T1 = towers(v1, member_opts(o13[2], D13, E), d13, forbid1, E)
                                T3 = towers(v3, member_opts(o13[3], D13, E), d13, forbid3, E)
                                lv1_full, lv3_full = T1, T3
                                lv13[1] = [c[0] for c in T1]
                                lv13[3] = [c[0] for c in T3]
                            else:
                                lv1_full = [()]; lv3_full = [()]
                        else:
                            lv1_full = [()]; lv3_full = [()]
                        if P24:
                            D24 = set(C24) - P24
                            if d24 >= 1:
                                T2 = towers(v2, member_opts(o24[2], D24, E), d24, forbid2, E)
                                T4 = towers(v4, member_opts(o24[3], D24, E), d24, forbid4, E)
                                lv2_full, lv4_full = T2, T4
                                lv24[2] = [c[0] for c in T2]
                                lv24[4] = [c[0] for c in T4]
                            else:
                                lv2_full = [()]; lv4_full = [()]
                        else:
                            lv2_full = [()]; lv4_full = [()]
                        lv1_full = wrap(lv1_full)
                        lv3_full = wrap(lv3_full)
                        lv2_full = wrap(lv2_full)
                        lv4_full = wrap(lv4_full)
                        lv2_full = [c for c in lv2_full if _l1_keep(c, mC13, m_int[2])]
                        lv4_full = [c for c in lv4_full if _l1_keep(c, mC13, m_int[4])]
                        for t1 in lv1_full:
                            for t3 in lv3_full:
                                for t2 in lv2_full:
                                    for t4 in lv4_full:
                                        cut1 = t1[0][0] if len(t1) >= 1 else FS()
                                        cut1sq = t1[1][0] if len(t1) >= 2 else FS()
                                        cut3 = t3[0][0] if len(t3) >= 1 else FS()
                                        cut3sq = t3[1][0] if len(t3) >= 2 else FS()
                                        cut2 = t2[0][0] if len(t2) >= 1 else FS()
                                        cut2sq = t2[1][0] if len(t2) >= 2 else FS()
                                        cut4 = t4[0][0] if len(t4) >= 1 else FS()
                                        cut4sq = t4[1][0] if len(t4) >= 2 else FS()
                                        cnt['combos'] += 1
                                        mc1 = t1[0][1] if len(t1) >= 1 else 0
                                        mc1s = t1[1][1] if len(t1) >= 2 else 0
                                        mc3 = t3[0][1] if len(t3) >= 1 else 0
                                        mc3s = t3[1][1] if len(t3) >= 2 else 0
                                        mc2 = t2[0][1] if len(t2) >= 1 else 0
                                        mc2s = t2[1][1] if len(t2) >= 2 else 0
                                        mc4 = t4[0][1] if len(t4) >= 1 else 0
                                        mc4s = t4[1][1] if len(t4) >= 2 else 0
                                        pred_bad = False
                                        if l2quick:
                                            ok2 = (mc2s == 0) or (((mc2s & mc4s) != 0) + ((mc2s & mc1) != 0) + ((mc2s & mc3) != 0) >= 2)
                                            ok4 = (mc4s == 0) or (((mc4s & mc2s) != 0) + ((mc4s & mc1) != 0) + ((mc4s & mc3) != 0) >= 2)
                                            pred_bad = not (ok2 and ok4)
                                            if pred_bad:
                                                if _SHADOW_L2:
                                                    cnt['shadow_bad'] += 1
                                                else:
                                                    cnt['l2_kill'] += 1
                                                    continue
                                        ck = (mC13, mC24, mc1, mc3, mc2, mc4, mc1s, mc3s, mc2s, mc4s)
                                        if ck in seen_cuts:
                                            cnt['dup_cuts'] += 1
                                            continue
                                        seen_cuts.add(ck)
                                        # overlap (strengthened), mask version
                                        fam13m = (mC13, {1: [mc1, mc1s] if P12 else [], 3: [mc3, mc3s] if P12 else []})
                                        fam24m = (mC24, {2: [mc2, mc2s] if P24 else [], 4: [mc4, mc4s] if P24 else []})
                                        ov = overlap_ok(fam13m, fam24m, m_int)
                                        if pred_bad and _SHADOW_L2 and ov:
                                            cnt['shadow_inv'] += 1
                                        if not ov:
                                            cnt['ov_kill'] += 1
                                            continue
                                        cuts = (('C13', C13), ('C24', C24),
                                                ('C1C13', cut1), ('C3C13', cut3),
                                                ('C2C24', cut2), ('C4C24', cut4),
                                                ('C1^2C13', cut1sq), ('C3^2C13', cut3sq),
                                                ('C2^2C24', cut2sq), ('C4^2C24', cut4sq))
                                        # pre-(em,vm) skip
                                        vm = {}
                                        for v in verts:
                                            acc = None
                                            for mstr, S in cuts:
                                                if v in S:
                                                    acc = mstr if acc is None else meet(acc, mstr)
                                            vm[v] = acc if acc is not None else 'H'
                                        em = [meet(vm[a], vm[b]) for (a, b) in E]
                                        pre = (tuple(em), tuple(vm.values()))
                                        if pre in seen_pre:
                                            cnt['skip_pre'] += 1
                                            continue
                                        seen_pre.add(pre)
                                        reg = R._build_region(E, verts, ext_attach, ext_mode, C13, C24, cut1, cut3, cut2, cut4, cut1sq, cut3sq, cut2sq, cut4sq)
                                        if reg is None:
                                            cnt['build_none'] += 1
                                            continue
                                        vmk, emk = reg
                                        outkey = (tuple(emk), tuple(sorted(vmk.items())))
                                        if outkey not in found:
                                            cnt['out'] += 1
                                            found[outkey] = (set(C13), set(C24), set(cut1), set(cut3), set(cut2), set(cut4), vmk, emk)
    regs = list(found.values())
    return regs, cnt, (d13, d24)


# ---------------------------- k1 enumerator ----------------------------
# Prototype for the k1 (all-lightlike) regime: same construction with m_i = ∞, plus a structural mirror of the current
# k1 enumerator: L = E - V + 1; L<3: main cuts only; 3<=L<5: first-power refinements, one side at a time (13 or 24);
# L>=5: first-power both sides + S^2C branches with the coincidences (squares := level 1), one side per build.
# Prune mirror [2026-09-20 late]: has_sc_vertex (3<=L<5), I0/I1/I2-family gates, reg3, Cond-1/2/3 branch gates carry over.
def skel_regions_k1(edges, verts, ext_attach, ext_mode, use_overlap=True):
    clear_graph_caches() # start each case with empty graph-kernel memo tables
    verts = sorted(verts)
    E = [tuple(e) for e in edges]
    adj = {v: set() for v in verts}
    for a, b in E:
        adj[a].add(b); adj[b].add(a)
    Vset = set(verts); extv = set(ext_attach.values())
    v1, v2, v3, v4 = (ext_attach['p1'], ext_attach['p2'], ext_attach['p3'], ext_attach['p4'])
    L = len(E) - len(verts) + 1
    bidx = {v: 1 << i for i, v in enumerate(verts)}
    mcache = {}

    mk = _make_mk(mcache, bidx)
    Gammas = [FS(S) for r in range(1, len(verts) + 1) for S in combinations(verts, r) if connected(set(S), E)]
    cnt = dict(gamma=len(Gammas), combos=0, cov_kill=0, pair_overlap_kill=0, dup_cuts=0, ov_kill=0, skip_pre=0, build_none=0, out=0, hsc_kill=0, i0_kill=0, i2_kill=0, gate_kill=0)
    found = {}; seen_cuts = set(); seen_pre = set()
    m_inf = {1: None, 2: None, 3: None, 4: None}
    inner = Vset - extv
    e_set = set(E) | {(b, a) for (a, b) in E}
    reg3 = all(sum(1 for (a, b) in E if a == v or b == v) + sum(1 for _n, vv in ext_attach.items() if vv == v) == 3 for v in verts)

    def has_common_vertex(sets):
        nonempty = [s for s in sets if s]
        if not nonempty:
            return False
        inter = set(nonempty[0])
        for s in nonempty[1:]:
            inter &= set(s)
        return bool(inter)

    def run_combo(C13, C24, cut1, cut3, cut2, cut4, cut1sq, cut3sq, cut2sq, cut4sq):
        cnt['combos'] += 1
        mC13 = mk(C13); mC24 = mk(C24)
        mc1 = mk(cut1); mc3 = mk(cut3)
        mc2 = mk(cut2); mc4 = mk(cut4)
        mc1s = mk(cut1sq); mc3s = mk(cut3sq)
        mc2s = mk(cut2sq); mc4s = mk(cut4sq)
        ck = (mC13, mC24, mc1, mc3, mc2, mc4, mc1s, mc3s, mc2s, mc4s)
        if ck in seen_cuts:
            cnt['dup_cuts'] += 1
            return
        seen_cuts.add(ck)
        fam13m = (mC13, {1: [mc1, mc1s], 3: [mc3, mc3s]})
        fam24m = (mC24, {2: [mc2, mc2s], 4: [mc4, mc4s]})
        if use_overlap and not overlap_ok(fam13m, fam24m, m_inf):
            cnt['ov_kill'] += 1
            return
        cuts = (('C13', C13), ('C24', C24), ('C1C13', cut1), ('C3C13', cut3), ('C2C24', cut2), ('C4C24', cut4), ('C1^2C13', cut1sq), ('C3^2C13', cut3sq), ('C2^2C24', cut2sq), ('C4^2C24', cut4sq))
        vm = {}
        for v in verts:
            acc = None
            for mstr, S in cuts:
                if v in S:
                    acc = mstr if acc is None else meet(acc, mstr)
            vm[v] = acc if acc is not None else 'H'
        em = [meet(vm[a], vm[b]) for (a, b) in E]
        pre = (tuple(em), tuple(vm.values()))
        if pre in seen_pre:
            cnt['skip_pre'] += 1
            return
        seen_pre.add(pre)
        reg = R._build_region(E, verts, ext_attach, ext_mode, C13, C24, cut1, cut3, cut2, cut4, cut1sq, cut3sq, cut2sq, cut4sq)
        if reg is None:
            cnt['build_none'] += 1
            return
        vmk, emk = reg
        outkey = (tuple(emk), tuple(sorted(vmk.items())))
        if outkey not in found:
            cnt['out'] += 1
            found[outkey] = (set(C13), set(C24), set(cut1), set(cut3), set(cut2), set(cut4), vmk, emk)

    Z = FS()
    for Gset in Gammas:
        opts13 = pair_opts_for(v1, v3, Gset, adj, Vset, extv)
        if not opts13:
            continue
        opts24 = pair_opts_for(v2, v4, Gset, adj, Vset, extv)
        if not opts24:
            continue
        for o13 in opts13:
            P1, P3 = (o13[0] or FS()), (o13[1] or FS())
            for o24 in opts24:
                P2, P4 = (o24[0] or FS()), (o24[1] or FS())
                P12 = set(P1) | set(P3)
                P24 = set(P2) | set(P4)
                if P12 & P24:
                    cnt['pair_overlap_kill'] += 1
                    continue
                dom13 = Vset - Gset - P24
                dom24 = Vset - Gset - P12
                C13opts = conn_sets(P12, dom13, E) if P12 else [FS()]
                if not C13opts:
                    cnt['cov_kill'] += 1
                    continue
                for C13 in C13opts:
                    rem = Vset - Gset - set(C13)
                    if P24:
                        C24opts = conn_sets(P24 | rem, dom24, E)
                    else:
                        C24opts = [FS()] if not rem else []
                    if not C24opts:
                        cnt['cov_kill'] += 1
                        continue
                    for C24 in C24opts:
                        D13 = set(C13) - P12
                        D24 = set(C24) - P24
                        mo1 = member_opts(o13[2], D13, E) if P12 else [FS()]
                        mo3 = member_opts(o13[3], D13, E) if P12 else [FS()]
                        mo2 = member_opts(o24[2], D24, E) if P24 else [FS()]
                        mo4 = member_opts(o24[3], D24, E) if P24 else [FS()]
                        if L < 3:
                            run_combo(C13, C24, Z, Z, Z, Z, Z, Z, Z, Z)
                            continue
                        if L < 5:
                            run_combo(C13, C24, Z, Z, Z, Z, Z, Z, Z, Z)
                            for cut1 in mo1:
                                for cut3 in mo3:
                                    if not (cut1 or cut3):
                                        continue
                                    if not has_common_vertex([C13, C24, cut1, cut3]):
                                        cnt['hsc_kill'] += 1
                                        continue
                                    run_combo(C13, C24, cut1, cut3, Z, Z, Z, Z, Z, Z)
                            for cut2 in mo2:
                                for cut4 in mo4:
                                    if not (cut2 or cut4):
                                        continue
                                    if not has_common_vertex([C13, C24, cut2, cut4]):
                                        cnt['hsc_kill'] += 1
                                        continue
                                    run_combo(C13, C24, Z, Z, cut2, cut4, Z, Z, Z, Z)
                            continue
                        # L >= 5  (I0/I1/I2-family gates + Cond-1/2/3 + reg3)
                        run_combo(C13, C24, Z, Z, Z, Z, Z, Z, Z, Z)
                        I0 = (set(C13) if C13 else Vset) & (set(C24) if C24 else Vset)
                        if not I0:
                            cnt['i0_kill'] += 1
                            continue
                        for cut1 in mo1:
                            for cut3 in mo3:
                                I1 = I0 & (set(cut1) if cut1 else Vset) & (set(cut3) if cut3 else Vset)
                                if not I1:
                                    continue
                                for cut2 in mo2:
                                    for cut4 in mo4:
                                        if not (cut1 or cut3 or cut2 or cut4):
                                            continue
                                        I2 = I1 & (set(cut2) if cut2 else Vset) & (set(cut4) if cut4 else Vset)
                                        I2x4 = I1 & (set(cut2) if cut2 else Vset)
                                        I2x2 = I1 & (set(cut4) if cut4 else Vset)
                                        I2x3 = I0 & (set(cut1) if cut1 else Vset) & (set(cut2) if cut2 else Vset) & (set(cut4) if cut4 else Vset)
                                        I2x1 = I0 & (set(cut3) if cut3 else Vset) & (set(cut2) if cut2 else Vset) & (set(cut4) if cut4 else Vset)
                                        if not (I2 or I2x4 or I2x2 or I2x3 or I2x1):
                                            cnt['i2_kill'] += 1
                                            continue
                                        run_combo(C13, C24, cut1, cut3, cut2, cut4, Z, Z, Z, Z)
                                        if cut1 or cut3:
                                            ok13 = False
                                            u13 = [u for u in inner if u in cut2 and u in cut4 and u in C13 and u not in cut1 and u not in cut3]
                                            if u13:
                                                hv13 = Vset - (set(C13) | set(C24) | set(cut1) | set(cut3) | set(cut2) | set(cut4))
                                                if (not reg3) or any(a in hv13 and b in hv13 for (a, b) in E):
                                                    ok13 = any((u, v) in e_set for u in u13 for v in I2)
                                                    if not ok13:
                                                        ok13 = bool(I2x4) and any((w, v) in e_set for w in (set(cut2) - set(C13)) for v in I2x4)
                                                    if not ok13:
                                                        ok13 = bool(I2x2) and any((w, v) in e_set for w in (set(cut4) - set(C13)) for v in I2x2)
                                            if ok13:
                                                run_combo(C13, C24, cut1, cut3, cut2, cut4, cut1, cut3, Z, Z)
                                            else:
                                                cnt['gate_kill'] += 1
                                        if cut2 or cut4:
                                            ok24 = False
                                            u24 = [u for u in inner if u in cut1 and u in cut3 and u in C24 and u not in cut2 and u not in cut4]
                                            if u24:
                                                hv24 = Vset - (set(C13) | set(C24) | set(cut1) | set(cut3) | set(cut2) | set(cut4))
                                                if (not reg3) or any(a in hv24 and b in hv24 for (a, b) in E):
                                                    ok24 = any((u, v) in e_set for u in u24 for v in I2)
                                                    if not ok24:
                                                        ok24 = bool(I2x3) and any((w, v) in e_set for w in (set(cut1) - set(C24)) for v in I2x3)
                                                    if not ok24:
                                                        ok24 = bool(I2x1) and any((w, v) in e_set for w in (set(cut3) - set(C24)) for v in I2x1)
                                            if ok24:
                                                run_combo(C13, C24, cut1, cut3, cut2, cut4, Z, Z, cut2, cut4)
                                            else:
                                                cnt['gate_kill'] += 1
    regs = list(found.values())
    return regs, cnt, ('L=%d' % L,)


# ------------------------- FRI enumerators -------------------------

# Cut representations are NOT unique: any SC region also has a representative with an SC vertex in ALL nonempty cuts
# (same em/scaling), so combinations without such a vertex are pruned — a lifted combination gives the same region.

# Whether some vertex lies in all nonempty cuts (the SC vertex; needed when refinements are present).
def has_sc_vertex(cut_sets):
    nonempty = [s for s in cut_sets.values() if s]
    if not nonempty:
        return False
    inter = nonempty[0]
    for s in nonempty[1:]:
        inter = inter & s
    return bool(inter)

# Enumerate all regions of the Regge 2->2 kinematics; dispatches to the on/off-shell enumerator by the external modes.
def fri_regions_full(edges, verts, ext_attach, ext_mode, M=1):
    # (M: kept for backward compatibility, unused)
    ms = {n: ext_m(ext_mode.get(n)) for n in ext_attach}
    # all lightlike -> on-shell enumerator (refinement depth by loop count)
    if all(m is None for m in ms.values()):
        return fri_regions_onshell(edges, verts, ext_attach, ext_mode)
    # off-shell: ext_m gives 𝒱 = m+1 (C13 -> 1), possibly_softest needs m — convert back
    ms_m = {n: (None if v is None else v - 1) for n, v in ms.items()}
    return fri_regions_offshell(edges, verts, ext_attach, ext_mode, ms_m)

# k1 enumerator (all externals lightlike): refinement depth decided by the LOOP COUNT instead of possibly_softest.
def fri_regions_onshell(edges, verts, ext_attach, ext_mode):
    v1, v3 = ext_attach['p1'], ext_attach['p3']
    v2, v4 = ext_attach['p2'], ext_attach['p4']
    forbid13 = {v2, v4}; forbid24 = {v1, v3}
    inner = set(verts) - set(ext_attach.values())
    L = len(edges) - len(verts) + 1
    # depth: first-power refined cuts from L >= 3; both sides + S²C branches from L >= 5
    use_refined = L >= 3
    # C_ij cut candidates (lightlike kinematics): connected sets containing BOTH va and vb; empty set = no cut.
    def opts(va, vb, forbid):
        base = ({va, vb} | inner) - forbid
        out = [frozenset()]
        rest = list(base - {va, vb})
        for r in range(len(rest) + 1):
            for sub in combinations(rest, r):
                S = frozenset({va, vb} | set(sub))
                if connected(S, edges):
                    out.append(S)
        return out

    O13 = opts(v1, v3, forbid13)
    O24 = opts(v2, v4, forbid24)
    regs = []
    seen = set()
    for cut13 in O13:
        for cut24 in O24:
            # H-vertex pruning: every region has a vertex outside all cuts
            if cut13 | cut24 >= set(verts):
                continue
            # L >= 5: full two-sided refinement with S²C branches (below: the L < 5 loop)
            if use_refined and L >= 5:
                r1 = refined_opts(edges, v1, cut13, {v2, v3, v4})
                r3 = refined_opts(edges, v3, cut13, {v1, v2, v4})
                r2 = refined_opts(edges, v2, cut24, {v1, v3, v4})
                r4 = refined_opts(edges, v4, cut24, {v1, v2, v3})
                inner = set(verts) - set(ext_attach.values())
                verts_set = set(verts)
                e_set = set(edges) | set((b, a) for (a, b) in edges)
                # 3-regular check: degree-3 vertices (external legs counted) tighten the Cond-1/2/3 H requirement to >=2 edge-connected H vertices
                reg3 = all(sum(1 for (a, b) in edges if a == v or b == v) + sum(1 for n, vv in ext_attach.items() if vv == v) == 3 for v in verts)
                # layered pruning: an SC/S²C vertex must lie in ALL nonempty cuts; I0 = main-cut intersection (empty cut = full set)
                I0 = (cut13 if cut13 else verts_set) & (cut24 if cut24 else verts_set)
                # pure main-cut combos (all refinements empty) always run — the source of the non-SC regions
                reg = R._build_region(edges, verts, ext_attach, ext_mode, cut13, cut24, frozenset(), frozenset(), frozenset(), frozenset())
                if reg is not None:
                    key = (tuple(reg[1]), tuple(sorted(reg[0].items())))
                    if key not in seen:
                        seen.add(key)
                        regs.append((set(cut13), set(cut24), frozenset(), frozenset(), frozenset(), frozenset(), reg[0], reg[1]))
                if not I0:
                    # no possible SC/S²C vertex: prune the refinement subtree (k1-only pruning)
                    continue
                for cut1 in r1:
                    for cut3 in r3:
                        # I1: intersection after the 13-side refinements; empty ⟹ skip cut2/cut4 entirely
                        I1 = I0 & (cut1 if cut1 else verts_set) & (cut3 if cut3 else verts_set)
                        if not I1:
                            continue
                        for cut2 in r2:
                            for cut4 in r4:
                                # all-empty refinement = pure main cut (already run above)
                                if not (cut1 or cut3 or cut2 or cut4):
                                    continue
                                # I2 = intersection of ALL nonempty cuts (Cond-1 v); I2x4/I2x2 = minus C4C24/C2C24 (Cond-2/3 v); I2x3/I2x1 = branch-24 Cond-2/3 (minus C3C13/C1C13)
                                I2 = I1 & (cut2 if cut2 else verts_set) & (cut4 if cut4 else verts_set)
                                I2x4 = I1 & (cut2 if cut2 else verts_set)
                                I2x2 = I1 & (cut4 if cut4 else verts_set)
                                I2x3 = I0 & (cut1 if cut1 else verts_set) & (cut2 if cut2 else verts_set) & (cut4 if cut4 else verts_set)
                                I2x1 = I0 & (cut3 if cut3 else verts_set) & (cut2 if cut2 else verts_set) & (cut4 if cut4 else verts_set)
                                if not (I2 or I2x4 or I2x2 or I2x3 or I2x1):
                                    continue
                                # H-vertex pruning after refinements
                                if cut13 | cut24 | cut1 | cut3 | cut2 | cut4 >= verts_set:
                                    continue
                                # branch 0: no second-power refinement (first-power SC regions)
                                reg = R._build_region(edges, verts, ext_attach, ext_mode, cut13, cut24, cut1, cut3, cut2, cut4)
                                if reg is not None:
                                    key = (tuple(reg[1]), tuple(sorted(reg[0].items())))
                                    if key not in seen:
                                        seen.add(key)
                                        regs.append((set(cut13), set(cut24), set(cut1), set(cut3), set(cut2), set(cut4), reg[0], reg[1]))
                                # branch 13: S²C13 target — any one of Cond 1/2/3 suffices:
                                #   Cond 1: v ∈ I2 adjacent to an SC24 vertex;  Cond 2: v ∈ I2x4 adjacent to C2C24\C13;
                                #   Cond 3: v ∈ I2x2 adjacent to C4C24\C13.  C1²C13 := C1C13, C3²C13 := C3C13;
                                #   under 3-regularity the H part needs >=2 edge-connected vertices.
                                if (cut1 or cut3):
                                    u13 = [u for u in inner if u in cut2 and u in cut4 and u in cut13 and u not in cut1 and u not in cut3]
                                    if u13:
                                        hv13 = verts_set - (cut13 | cut24 | cut1 | cut3 | cut2 | cut4)
                                        h_ok = (not reg3) or any(a in hv13 and b in hv13 for (a, b) in edges)
                                        if h_ok:
                                            ok13 = any((u, v) in e_set for u in u13 for v in I2)
                                            if not ok13:
                                                # Cond 2: v in all but C4C24
                                                ok13 = bool(I2x4) and any((w, v) in e_set for w in (cut2 - cut13) for v in I2x4)
                                            if not ok13:
                                                # Cond 3: v in all but C2C24
                                                ok13 = bool(I2x2) and any((w, v) in e_set for w in (cut4 - cut13) for v in I2x2)
                                            if ok13:
                                                reg = R._build_region(edges, verts, ext_attach, ext_mode, cut13, cut24, cut1, cut3, cut2, cut4, cut1, cut3)
                                                if reg is not None:
                                                    key = (tuple(reg[1]), tuple(sorted(reg[0].items())))
                                                    if key not in seen:
                                                        seen.add(key)
                                                        regs.append((set(cut13), set(cut24), set(cut1), set(cut3), set(cut2), set(cut4), reg[0], reg[1]))
                                # branch 24: S²C24 target, symmetric:
                                #   Cond 1: v ∈ I2 adjacent to an SC13 vertex;  Cond 2: v ∈ I2x3 adjacent to C1C13\C24;
                                #   Cond 3: v ∈ I2x1 adjacent to C3C13\C24.  C2²C24 := C2C24, C4²C24 := C4C24.
                                if (cut2 or cut4):
                                    u24 = [u for u in inner if u in cut1 and u in cut3 and u in cut24 and u not in cut2 and u not in cut4]
                                    if u24:
                                        hv24 = verts_set - (cut13 | cut24 | cut1 | cut3 | cut2 | cut4)
                                        h_ok = (not reg3) or any(a in hv24 and b in hv24 for (a, b) in edges)
                                        if h_ok:
                                            ok24 = any((u, v) in e_set for u in u24 for v in I2)
                                            if not ok24:
                                                # Cond 2: v in all but C3C13
                                                ok24 = bool(I2x3) and any((w, v) in e_set for w in (cut1 - cut24) for v in I2x3)
                                            if not ok24:
                                                # Cond 3: v in all but C1C13
                                                ok24 = bool(I2x1) and any((w, v) in e_set for w in (cut3 - cut24) for v in I2x1)
                                            if ok24:
                                                reg = R._build_region(edges, verts, ext_attach, ext_mode, cut13, cut24, cut1, cut3, cut2, cut4, None, None, cut2, cut4)
                                                if reg is not None:
                                                    key = (tuple(reg[1]), tuple(sorted(reg[0].items())))
                                                    if key not in seen:
                                                        seen.add(key)
                                                        regs.append((set(cut13), set(cut24), set(cut1), set(cut3), set(cut2), set(cut4), reg[0], reg[1]))
                continue
            # 3 <= L < 5: 13-side and 24-side refinements, exclusive
            for refine in ((0, 1, 2) if use_refined else (0,)):
                r1 = r3 = r2 = r4 = [frozenset()]
                if refine == 1 and use_refined:
                    r1 = refined_opts(edges, v1, cut13, {v2, v3, v4})
                    r3 = refined_opts(edges, v3, cut13, {v1, v2, v4})
                elif refine == 2 and use_refined:
                    r2 = refined_opts(edges, v2, cut24, {v1, v3, v4})
                    r4 = refined_opts(edges, v4, cut24, {v1, v2, v3})
                for cut1 in r1:
                    for cut3 in r3:
                        for cut2 in r2:
                            for cut4 in r4:
                                # all-empty refinement ≡ refine=0 (pure main cut, already built there)
                                if refine != 0 and not (cut1 or cut3 or cut2 or cut4):
                                    continue
                                if refine != 0 and not has_sc_vertex({'C13': cut13, 'C24': cut24, 'C1C13': cut1, 'C3C13': cut3, 'C2C24': cut2, 'C4C24': cut4}):
                                    continue
                                if cut13 | cut24 | cut1 | cut3 | cut2 | cut4 >= set(verts):
                                    continue
                                reg = R._build_region(edges, verts, ext_attach, ext_mode, cut13, cut24, cut1, cut3, cut2, cut4)
                                if reg is not None:
                                    key = (tuple(reg[1]), tuple(sorted(reg[0].items())))
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    regs.append((set(cut13), set(cut24), set(cut1), set(cut3), set(cut2), set(cut4), reg[0], reg[1]))
    return regs


# Nested refinement towers for one external: S_1 ⊇ S_2 ⊇ ... with each level a connected subset containing root; empty stops deeper levels.
def _refine_towers(edges, root, base, forbid, depth):
    if depth <= 0:
        yield ()
        return
    for cut1 in refined_opts(edges, root, base, forbid):
        for rest in _refine_towers(edges, root, cut1, forbid, depth - 1):
            yield (cut1,) + rest


# Non-k1 enumerator: main cuts may be disconnected; refinement depth from possibly_softest (no loop-count gate).
def fri_regions_offshell(edges, verts, ext_attach, ext_mode, ms):
    v1, v3 = ext_attach['p1'], ext_attach['p3']
    v2, v4 = ext_attach['p2'], ext_attach['p4']
    forbid13 = {v2, v4}
    forbid24 = {v1, v3}
    inner = set(verts) - set(ext_attach.values())
    verts_set = set(verts)
    m1, m2, m3, m4 = ms['p1'], ms['p2'], ms['p3'], ms['p4']

    # depth limits: possibly_softest(m_i) decides d13/d24 (both sides -> M each; single side -> M and M-1)
    ps = possibly_softest(m1, m2, m3, m4)
    if len(ps) == 2:                    # S^M C13 + S^M C24 simultaneously
        d13 = d24 = int(ps[0].split('^')[1].split('C')[0])
    else:                               # single candidate
        Mside = int(ps[0].split('^')[1].split('C')[0])
        if ps[0].endswith('C13'):
            d13, d24 = Mside, Mside - 1
        else:
            d13, d24 = Mside - 1, Mside
    if d13 > 2 or d24 > 2:
        raise NotImplementedError(f"fri_regions_offshell: depth {max(d13, d24)} > 2 — meet/vee tables stop at C_i²C_ij (possibly_softest={ps}); extend the mode tables first.")

    # C_ij cut candidates (off-shell): may be disconnected, but every connected component must contain va or vb —
    # hard/Glauber vertices may stay outside all cuts (requirement 2 then holds automatically).
    def opts(va, vb, forbid):
        base = ({va, vb} | inner) - forbid
        out = [frozenset()]
        rest = list(base)
        for r in range(len(rest) + 1):
            for sub in combinations(rest, r):
                S = frozenset(sub)
                if not S:
                    continue
                es = [(a, b) for (a, b) in edges if a in S and b in S]
                comp = components(es, S)
                groups = {}
                for v in S:
                    groups.setdefault(comp[v], set()).add(v)
                if all(va in c or vb in c for c in groups.values()):
                    out.append(S)
        return out

    O13 = opts(v1, v3, forbid13)
    O24 = opts(v2, v4, forbid24)

    regs = []
    seen = set()
    for cut13 in O13:
        for cut24 in O24:
            # H-vertex prune (kinematics-independent): every region has a vertex in no cut
            if cut13 | cut24 >= verts_set:
                continue
            T1 = list(_refine_towers(edges, v1, cut13, {v2, v3, v4}, d13))
            T3 = list(_refine_towers(edges, v3, cut13, {v1, v2, v4}, d13))
            T2 = list(_refine_towers(edges, v2, cut24, {v1, v3, v4}, d24))
            T4 = list(_refine_towers(edges, v4, cut24, {v1, v2, v3}, d24))
            for t1 in T1:
                for t3 in T3:
                    for t2 in T2:
                        for t4 in T4:
                            cut1 = t1[0] if len(t1) >= 1 else frozenset()
                            cut1sq = t1[1] if len(t1) >= 2 else frozenset()
                            cut3 = t3[0] if len(t3) >= 1 else frozenset()
                            cut3sq = t3[1] if len(t3) >= 2 else frozenset()
                            cut2 = t2[0] if len(t2) >= 1 else frozenset()
                            cut2sq = t2[1] if len(t2) >= 2 else frozenset()
                            cut4 = t4[0] if len(t4) >= 1 else frozenset()
                            cut4sq = t4[1] if len(t4) >= 2 else frozenset()
                            if (cut13 | cut24 | cut1 | cut3 | cut2 | cut4 | cut1sq | cut3sq | cut2sq | cut4sq) >= verts_set:
                                continue
                            reg = R._build_region(edges, verts, ext_attach, ext_mode, cut13, cut24, cut1, cut3, cut2, cut4, cut1sq, cut3sq, cut2sq, cut4sq)
                            if reg is None:
                                continue
                            key = (tuple(reg[1]), tuple(sorted(reg[0].items())))
                            if key in seen:
                                continue
                            seen.add(key)
                            regs.append((set(cut13), set(cut24), set(cut1), set(cut3), set(cut2), set(cut4), reg[0], reg[1]))
    return regs


# ---------------------------- harness ----------------------------
def run_case(name, kin, quiet=False):
    edges = [tuple(e) for e in GRAPHS[name]]
    verts = sorted({v for e in edges for v in e})
    ext = EXT_ATTACHES.get(name, EXT_ATTACH)
    emk = KIN[kin]['ext_mode']
    t0 = time.perf_counter()
    ref = fri_regions_full(edges, verts, ext, emk)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    if kin == 'k1':
        sk, cnt, depth = skel_regions_k1(edges, verts, ext, emk)
    else:
        sk, cnt, depth = skel_regions(edges, verts, ext, emk)
    t_sk = time.perf_counter() - t0
    ref_set = {to_scaling(r[7]) for r in ref}
    sk_set = {to_scaling(r[7]) for r in sk}
    miss = ref_set - sk_set
    extra = sk_set - ref_set
    ok = not miss and not extra
    print('%-12s %s depth=%s | ref=%3d skel=%3d | %s  (ref %.2fs, skel %.2fs)%s'
          % (name, kin, depth, len(ref_set), len(sk_set),
             'OK ' if ok else 'DIFF', t_ref, t_sk,
             '' if quiet or ok else '  miss=%d extra=%d' % (len(miss), len(extra))),
          flush=True)
    if not ok and not quiet:
        print('   counters:', cnt, flush=True)
        for v in sorted(miss)[:3]:
            for r in ref:
                if to_scaling(r[7]) == v:
                    print('   MISS ref cfg: C13=%s C24=%s C1=%s C3=%s C2=%s C4=%s' % (sorted(r[0]), sorted(r[1]), sorted(r[2]), sorted(r[3]), sorted(r[4]), sorted(r[5])), flush=True)
                    break
            else:
                print('   MISS', v, flush=True)
        for v in sorted(extra)[:3]:
            for r in sk:
                if to_scaling(r[7]) == v:
                    print('   EXTRA skel cfg: C13=%s C24=%s C1=%s C3=%s C2=%s C4=%s' % (sorted(r[0]), sorted(r[1]), sorted(r[2]), sorted(r[3]), sorted(r[4]), sorted(r[5])), flush=True)
                    break
    return ok, miss, extra, t_ref, t_sk


DEFAULT = [('box', 'k0'), ('box', 'k5'),
           ('hexagon', 'k0'), ('hexagon', 'k2'), ('hexagon', 'k3'),
           ('hexagon', 'k4'), ('hexagon', 'k5'),
           ('Necklace1', 'k0'), ('Necklace1', 'k4'),
           ('CrossLadder1', 'k4'), ('Crown', 'k4'), ('KidCrown', 'k4'),
           ('K33A', 'k4'), ('ladder', 'k4')]


def main():
    args = sys.argv[1:]
    cases = []
    if args:
        for i in range(0, len(args) - 1, 2):
            cases.append((args[i], args[i + 1]))
    else:
        cases = DEFAULT
    n_ok = 0
    for name, kin in cases:
        ok, *_ = run_case(name, kin)
        n_ok += ok
    print('--- %d/%d OK' % (n_ok, len(cases)))


if __name__ == '__main__':
    main()
