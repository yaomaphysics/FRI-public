#!/usr/bin/env python3
r"""
enumerator.py — skeleton-based cut enumeration & pruning for the Regge 2->2 kinematics (general external virtualities).

Start from a skeleton: a connected uncovered vertex set Γ (the union of H, sH and G), one shared "stem" per leg pair (1/3 and 2/4) that leaves Γ and splits into the two "branches".
Cuts are constructed upon each such skeleton.
Two enumerators cover the kinematics: skel_regions for a general ext_mode, and skel_regions_k1 for the kinematics where all p_i^2 = 0 (which we call k1).

Skeleton and cuts.
  * Γ connected; each leg pair gets a stem path that leaves Γ and splits into the two branch paths, keeping only the common prefix when read backwards from Γ, and the two pairs may not overlap;
  * the C13 and C24 cuts contain their pair paths P1\cup P3 and P2\cup P4 respectively;
  * the cuts are nested as follows -- C13 ⊇ C1C13 ⊇ C1^2C13 ⊇ ..., etc.;
  * the C1C13 cut overlaps with the v1 branch (P1 - P3) into a connected subpath with v1 as one endpoint, while avoiding other paths; similar for other cuts.

Pruning.
  * coverage: every vertex outside the block lies in some cut; duplicated cut assignments are skipped;
  * overlap: a cut at level n and the two cross partners at level n-1 (two of three), with the deeper-cut exemption;
  * k1 additionally prunes with the SC-vertex condition (a vertex common to all nonempty cuts), applied per level as I0/I1/I2, plus the gate that decides the second level.

Key constraints: the hard block and the two leg-pair stems must stay consistent, since the checks downstream read the resulting (em, vm); the refinement depth is capped at 2 (deeper towers raise an error).
"""
import sys, os
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import region_filter as R                                        # noqa: E402
import mode_levels as ML                                         # noqa: E402
from primitives import (conn_sets, comp_sets, connected, refined_opts, Mode, to_mode, CC, H, mode_meet, possibly_softest, clear_graph_caches)  # noqa: E402

# ==================== HELPERS ====================
FS = frozenset

# External-mode labels of the kinematics, as Mode objects, with their virtuality (ext_m).
_EXT_M = {to_mode('C13'): 1, to_mode('C24'): 1,
          to_mode('C1C13'): 2, to_mode('C3C13'): 2, to_mode('C2C24'): 2, to_mode('C4C24'): 2,
          to_mode('C1^2C13'): 3, to_mode('C3^2C13'): 3, to_mode('C2^2C24'): 3, to_mode('C4^2C24'): 3}

# Boundary converter: legacy external-mode strings -> Mode objects (Mode values pass through).
def _conv_ext(ext_mode):
    return None if ext_mode is None else {k: (v if isinstance(v, Mode) else to_mode(v)) for k, v in ext_mode.items()}

# Virtuality (m+1) of an external mode; None for the lightlike ones.
def _ext_m(md):
    return _EXT_M.get(md)

# Refinement-mode label from possibly_softest ('S^kC13'/'S^kC24'), as a Mode.


# All vertex sequences from a start vertex that reach Γ, within the allowed set.
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

# Compatible pair traversal: read backwards from Γ the two branches share only a common prefix and never rejoin.
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


# First level of a leg chain: connected supersets of a branch suffix T = br[j:] (leg vertex as one endpoint) inside T ∪ D.
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

# One cut of a leg at a given level: level 0 is the family base cut, level lvl the lvl-th nested cut.
def _cut_at(f13, f24, leg, lvl):
    base, lv = f13 if leg in (1, 3) else f24
    if lvl == 0:
        return base
    return lv[leg][lvl - 1] if lvl - 1 < len(lv[leg]) else None


# Necessary condition at level 1 for legs 2/4: the cut must meet the family base cut unless it is empty, mi cuts it off, or the deeper-cut exemption applies.
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


# Overlap: a cut at level n needs a vertex shared with two of the three partners {the same-leg cut at level n, the two cross-leg cuts at level n-1}, unless it coincides with a deeper same-leg cut.
def overlap_ok(fam13, fam24, m):
    for leg in (1, 2, 3, 4):
        lv = (fam13[1] if leg in (1, 3) else fam24[1])[leg]
        for n0, Sx in enumerate(lv, start=1):
            if not Sx:
                continue
            mi = m[leg]
            if mi is not None and not n0 < mi:
                continue
            # deeper-cut exemption: coincides with a deeper same-leg cut n' > n
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


# [Optimization] Bitmask helper for vertex sets (shared by both enumerators).
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


# ==================== GENERAL ENUMERATOR ====================

# General skeleton enumerator: builds the skeleton (Γ, the pair stems and branches) and the nested cuts, prunes, then runs the check chain; all p_i^2 = 0 (k1) goes to skel_regions_k1.
def skel_regions(edges, verts, ext_attach, ext_mode):
    clear_graph_caches() # [Optimization] start each case with empty graph-kernel memo tables
    R.clear_adjust_caches()
    ext_mode = _conv_ext(ext_mode)
    verts = sorted(verts)
    E = [tuple(e) for e in edges]
    adj = {v: set() for v in verts}
    for a, b in E:
        adj[a].add(b); adj[b].add(a)
    Vset = set(verts)
    extv = set(ext_attach.values())
    v1, v2, v3, v4 = (ext_attach['p1'], ext_attach['p2'], ext_attach['p3'], ext_attach['p4'])
    bidx = {v: 1 << i for i, v in enumerate(verts)}
    mcache = {}   # [Optimization] per-case vertex-set mask cache

    mk = _make_mk(mcache, bidx)
    vbits = [(v, bidx[v]) for v in verts]   # mask-bit per vertex (vm build below)

    def wrap(chs):
        return [tuple((S, mk(S)) for S in ch) for ch in chs]

    # refinement depths per side
    ms = {n: _ext_m(ext_mode.get(n)) for n in ext_attach}
    ms_m = {n: (None if v is None else v - 1) for n, v in ms.items()}
    m_int = {1: ms_m['p1'], 2: ms_m['p2'], 3: ms_m['p3'], 4: ms_m['p4']}
    ps = [to_mode(p) for p in possibly_softest(ms_m['p1'], ms_m['p2'], ms_m['p3'], ms_m['p4'])]
    # all p_i^2 = 0 (k1): skel_regions_k1 builds it
    if not ps:
        return skel_regions_k1(edges, verts, ext_attach, ext_mode)
    if len(ps) == 2:
        d13 = d24 = ps[0].m
    else:
        Mside = ps[0].m
        if ps[0].fam == 13:
            d13, d24 = Mside, Mside - 1
        else:
            d13, d24 = Mside - 1, Mside
    if max(d13, d24) > 2:
        raise SystemExit('depth > 2 not implemented')
    # L2 quick-kill gate: residual overlap checks are exactly legs 2/4 at level 2; legs 1/3 must be silenced here (mi <= 1) and 2/4 unchecked.
    l2quick = (d24 == 2 and m_int[1] is not None and m_int[1] <= 1 and m_int[3] is not None and m_int[3] <= 1 and m_int[2] is None and m_int[4] is None)

    Gammas = [FS(S) for r in range(1, len(verts) + 1) for S in combinations(verts, r) if connected(set(S), E)]

    cnt = dict(gamma=len(Gammas), combos=0, cov_kill=0, pair_overlap_kill=0, dup_cuts=0, ov_kill=0, skip_pre=0, build_none=0, out=0, l2_kill=0)
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
                        # ==================== CHAINS ====================
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
                        # flatten the towers once: (mask lvl1, mask lvl2, cut lvl1, cut lvl2) — removes the per-combo len()/indexing guards from the inner loop
                        flat1 = [(t[0][1] if len(t) >= 1 else 0, t[1][1] if len(t) >= 2 else 0,
                                  t[0][0] if len(t) >= 1 else FS(), t[1][0] if len(t) >= 2 else FS()) for t in lv1_full]
                        flat3 = [(t[0][1] if len(t) >= 1 else 0, t[1][1] if len(t) >= 2 else 0,
                                  t[0][0] if len(t) >= 1 else FS(), t[1][0] if len(t) >= 2 else FS()) for t in lv3_full]
                        flat2 = [(t[0][1] if len(t) >= 1 else 0, t[1][1] if len(t) >= 2 else 0,
                                  t[0][0] if len(t) >= 1 else FS(), t[1][0] if len(t) >= 2 else FS()) for t in lv2_full]
                        flat4 = [(t[0][1] if len(t) >= 1 else 0, t[1][1] if len(t) >= 2 else 0,
                                  t[0][0] if len(t) >= 1 else FS(), t[1][0] if len(t) >= 2 else FS()) for t in lv4_full]
                        for mc1, mc1s, cut1, cut1sq in flat1:
                            for mc3, mc3s, cut3, cut3sq in flat3:
                                for mc2, mc2s, cut2, cut2sq in flat2:
                                    for mc4, mc4s, cut4, cut4sq in flat4:
                                        cnt['combos'] += 1
                                        pred_bad = False
                                        if l2quick:
                                            ok2 = (mc2s == 0) or (((mc2s & mc4s) != 0) + ((mc2s & mc1) != 0) + ((mc2s & mc3) != 0) >= 2)
                                            ok4 = (mc4s == 0) or (((mc4s & mc2s) != 0) + ((mc4s & mc1) != 0) + ((mc4s & mc3) != 0) >= 2)
                                            pred_bad = not (ok2 and ok4)
                                            if pred_bad:
                                                cnt['l2_kill'] += 1
                                                continue
                                        ck = (mC13, mC24, mc1, mc3, mc2, mc4, mc1s, mc3s, mc2s, mc4s)
                                        if ck in seen_cuts:
                                            cnt['dup_cuts'] += 1
                                            continue
                                        seen_cuts.add(ck)
                                        # overlap, inlined: a cut at level n needs two of the three partners {the same-leg cut at level n, the two cross-leg cuts at level n-1}; at level 1 it reduces to a single AND.
                                        ov = True
                                        if mc1 and (m_int[1] is None or 1 < m_int[1]) and mc1s != mc1 and not (mc1 & mC24):
                                            ov = False
                                        elif mc1s and (m_int[1] is None or 2 < m_int[1]) and ((mc1s & mc3s) != 0) + ((mc1s & mc2) != 0) + ((mc1s & mc4) != 0) < 2:
                                            ov = False
                                        elif mc2 and (m_int[2] is None or 1 < m_int[2]) and mc2s != mc2 and not (mc2 & mC13):
                                            ov = False
                                        elif mc2s and (m_int[2] is None or 2 < m_int[2]) and ((mc2s & mc4s) != 0) + ((mc2s & mc1) != 0) + ((mc2s & mc3) != 0) < 2:
                                            ov = False
                                        elif mc3 and (m_int[3] is None or 1 < m_int[3]) and mc3s != mc3 and not (mc3 & mC24):
                                            ov = False
                                        elif mc3s and (m_int[3] is None or 2 < m_int[3]) and ((mc3s & mc1s) != 0) + ((mc3s & mc2) != 0) + ((mc3s & mc4) != 0) < 2:
                                            ov = False
                                        elif mc4 and (m_int[4] is None or 1 < m_int[4]) and mc4s != mc4 and not (mc4 & mC13):
                                            ov = False
                                        elif mc4s and (m_int[4] is None or 2 < m_int[4]) and ((mc4s & mc2s) != 0) + ((mc4s & mc1) != 0) + ((mc4s & mc3) != 0) < 2:
                                            ov = False
                                        if not ov:
                                            cnt['ov_kill'] += 1
                                            continue
                                        # pre-(em,vm) skip (mask-based cut membership: msk & bit == "vertex in cut", faster than a set lookup)
                                        cut_masks = ((R.C13, mC13), (R.C24, mC24),
                                                     (R.C1C13, mc1), (R.C3C13, mc3),
                                                     (R.C2C24, mc2), (R.C4C24, mc4),
                                                     (R.C1SQ13, mc1s), (R.C3SQ13, mc3s),
                                                     (R.C2SQ24, mc2s), (R.C4SQ24, mc4s))
                                        vm = {}
                                        for v, bv in vbits:
                                            acc = None
                                            for md, msk in cut_masks:
                                                if msk & bv:
                                                    acc = md if acc is None else mode_meet(acc, md)
                                            vm[v] = acc if acc is not None else H
                                        # vm-only dedup key — em is a deterministic function of vm, so the dedup classes are identical (em is not needed for anything else here).
                                        vkey = tuple(vm.values())
                                        if vkey in seen_pre:
                                            cnt['skip_pre'] += 1
                                            continue
                                        seen_pre.add(vkey)
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


# ==================== K1 ENUMERATOR ====================

# skel_regions_k1: the same construction for all p_i^2 = 0 (k1), with m_i infinite; the refinement depth comes from the mode ladder at the loop count L = |E| - |V| + 1.
def skel_regions_k1(edges, verts, ext_attach, ext_mode, use_overlap=True):
    clear_graph_caches() # [Optimization] start each case with empty graph-kernel memo tables
    R.clear_adjust_caches()
    ext_mode = _conv_ext(ext_mode)
    verts = sorted(verts)
    E = [tuple(e) for e in edges]
    adj = {v: set() for v in verts}
    for a, b in E:
        adj[a].add(b); adj[b].add(a)
    Vset = set(verts); extv = set(ext_attach.values())
    v1, v2, v3, v4 = (ext_attach['p1'], ext_attach['p2'], ext_attach['p3'], ext_attach['p4'])
    L = len(E) - len(verts) + 1
    # member-chain powers per leg from the mode ladder: C_i^nC_ij appears at its first level, and the construction covers powers 1 and 2.
    PWS = ML.cut_chain_levels(ext_mode, L)
    n_pow = max((max(v) for v in PWS.values() if v), default=0)
    if n_pow > 2:
        raise SystemExit('depth > 2 not implemented')
    bidx = {v: 1 << i for i, v in enumerate(verts)}
    mcache = {}   # [Optimization] per-case vertex-set mask cache

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

    Z = FS()

    # The eight member-cut modes of the chain, in slot order (power ascending, legs 1/3/2/4).
    MC1, MC3, MC2, MC4 = CC(1, 1), CC(3, 1), CC(2, 1), CC(4, 1)
    MC1S, MC3S, MC2S, MC4S = CC(1, 2), CC(3, 2), CC(2, 2), CC(4, 2)

    # Region checks for one cut assignment: the two base cuts plus the eight member slots (Z = a slot the ladder does not reach).
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
        cuts = ((R.C13, C13), (R.C24, C24), (MC1, cut1), (MC3, cut3), (MC2, cut2), (MC4, cut4),
                (MC1S, cut1sq), (MC3S, cut3sq), (MC2S, cut2sq), (MC4S, cut4sq))
        vm = {}
        for v in verts:
            acc = None
            for md, S in cuts:
                if v in S:
                    acc = md if acc is None else mode_meet(acc, md)
            vm[v] = acc if acc is not None else H
        em = [mode_meet(vm[a], vm[b]) for (a, b) in E]
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
                        if n_pow == 0:                    # L <= 2: the two base cuts alone
                            run_combo(C13, C24, Z, Z, Z, Z, Z, Z, Z, Z)
                            continue
                        if n_pow == 1:                    # L = 3/4: a level-1 member cut on either family
                            # overlap, baked into the domain: a cut at level 1 must meet the other family's base cut
                            # (in this tier the second level is empty, so the deeper-cut exemption never applies).
                            mo1 = [c for c in mo1 if not c or (set(c) & set(C24))]
                            mo3 = [c for c in mo3 if not c or (set(c) & set(C24))]
                            mo2 = [c for c in mo2 if not c or (set(c) & set(C13))]
                            mo4 = [c for c in mo4 if not c or (set(c) & set(C13))]
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
                        # L >= 5  (I0/I1/I2 per level, the gate that decides the second level, Cond-1/2/3, reg3)
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
