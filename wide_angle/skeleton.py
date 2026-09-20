#!/usr/bin/env python3
"""skeleton.py — skeleton-based cut enumeration & pruning for wide-angle FRI
(formalized 2026-09-18; development history + intermediate rules in
private/skeleton_rules_history.md).

Conditions (agreed with 小马):
  * H connected (enumerated explicitly);
  * for every C_i^m-type external (mode (0,n,i), n>=1) whose root is not in H:
    a path P_i from the root that touches H (inside V\\H, avoiding other
    p/q attachment vertices; soft (l, m>=1) attachment vertices are NOT
    restricted — 小马 2026-09-20); paths of different externals pairwise
    disjoint;
  * base cut (largest chain level, paired with the least-soft cut) ⊇ P_i;
    chain levels S_1 ⊇ S_2 ⊇ ... : each connected containing root, within its
    own per-level allowed set, inside V\\H; partial chains (prefix nonempty);
  * combination -> (vm, em) via cut meets; same check chain as the enumerator.

* route-exclusivity: no leg's cut layers may touch another leg's route
  (S_k^(i) ∩ P_j = ∅ for i≠j) — default ON;
* overlap (C_i^m legs, layer n<m): either option (1) a vertex shared by
  this cut and the same-sigma (sigma = n) S^{n'}C_j^{n-n'} cuts (n'=0..n;
  n'=0 reduces to C_j^n) of >=2 other directions j, or (2) soft support —
  an external of soft power exactly n (S^nC_j^k / S^n) whose incident
  vertex is contained in this layer — unless the layer coincides with a
  deeper same-leg C_i^N cut (N > n) [exemption relaxed 2026-09-20].
  Default ON (overlap_strong=True since 2026-09-18 evening; pass
  overlap_strong=False for the old shared-vertex-only form, e.g. A/B
  runs).  [(1) relaxed + exemption: 小马 2026-09-20]
* soft externals (S^mC^n / S^m, m>=1; 小马 2026-09-20): nested cut chains
  S^mC_i^1..S^mC_i^N (N = n for finite n, κ−m for n=∞; a single S^m cut
  when n=0), confined to V∖H and avoiding ALL route paths P_j; no root->H
  path requirement (2026-09-17).
* k0 (all externals C_i^1 or H): enumerated via the union construction —
  H + P_i = connected touch-H sets; every vertex of V∖(H∪P's) must lie
  in >=2 cuts; corner closure.  Region-identical to the old path
  (validated 2026-09-18).

Validated domain: p_i q_j externals (2026-09-18).  Soft externals
(S^mC^n / S^m, m>=1): the 2026-09-20 extended spec is implemented but
still under validation — refused by default; pass allow_soft=True to run.

(v1/v2/v3/v3.2 development history preserved in
private/skeleton_rules_history.md.)
"""
import sys, os, time, itertools
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import truncation_check as TC
import region_checker as rc
from primitives import Graph, vee, check_fc, momentum_ok, ir_ok_blocks
from region_checker import jet_connected_ok, cond1_ok
from usable_modes import derive_usable_modes, usable_layers

H = (0, 0, 0)


def _connected(S, adj):
    if not S:
        return True
    S = set(S)
    st = [next(iter(S))]
    seen = set()
    while st:
        u = st.pop()
        if u in seen:
            continue
        seen.add(u)
        for w in adj[u]:
            if w in S and w not in seen:
                st.append(w)
    return seen == S


def _conn_sets(root, allowed, adj):
    """connected subsets containing root, within allowed (incl. {root})."""
    if root not in allowed:
        return []
    rest = sorted(set(allowed) - {root})
    out = []
    for r in range(len(rest) + 1):
        for sub in itertools.combinations(rest, r):
            S = frozenset({root} | set(sub))
            if _connected(S, adj):
                out.append(S)
    return out


def _conn_supersets(base, allowed, adj):
    """connected supersets of base within allowed."""
    rest = sorted(set(allowed) - set(base))
    out = []
    for r in range(len(rest) + 1):
        for sub in itertools.combinations(rest, r):
            S = frozenset(set(base) | set(sub))
            if _connected(S, adj):
                out.append(S)
    return out


def _paths_to_H(root, Hset, allowed, adj):
    sets = set()
    st = [(root, frozenset({root}))]
    while st:
        cur, vis = st.pop()
        if adj[cur] & Hset:
            sets.add(vis)
        for w in adj[cur]:
            if w in allowed and w not in vis:
                st.append((w, vis | {w}))
    return list(sets)


def _check_combo(verts, edges_t, g, ext_attach, ext_mode, combo, seen_vm=None):
    """copy of the enumerator's check chain (Step1 / FC / jet / mojetic / IR).

    seen_vm: optional set for vm-level dedup (2026-09-18).  em = meet(vm[u],
    vm[v]) and every check depends only on (vm, em): the outcome is a
    function of vm alone, so a repeated vm is skipped outright (the region
    key is vm-based anyway).  Behaviour-preserving; big speedup.
    """
    vm = {}
    for v in verts:
        incuts = [cut.mode for (cut, S) in combo if v in S]
        if not incuts:
            vm[v] = H
        else:
            acc = incuts[0]
            for md in incuts[1:]:
                acc = rc.meet(acc, md)
            vm[v] = rc.norm(acc)
    if seen_vm is not None:
        key_m = frozenset((v, vm[v]) for v in verts)
        if key_m in seen_vm:
            return None
        seen_vm.add(key_m)
    em = [rc.meet(vm[u], vm[v]) for (u, v) in edges_t]
    for v in verts:
        accs = []
        for ei in g.incident.get(v, []):
            accs.append(em[ei])
        for name, vv in ext_attach.items():
            if vv == v:
                accs.append(ext_mode[name])
        if accs:
            if not rc.eq(vee(accs), vm[v]):
                return None
    if not rc.massive_h_ok(edges_t, em, frozenset()):
        return None
    if not momentum_ok(g, em, ext_mode):
        return None
    if not jet_connected_ok(vm, em, edges_t):
        return None
    if not check_fc(g, em, ext_mode):
        return None
    ok_mj, _ = cond1_ok(edges_t, em, ext_attach, ext_mode)
    if not ok_mj:
        return None
    if not ir_ok_blocks(g, em, ext_mode):
        return None
    return vm, em


def has_soft_externals(ext_mode):
    """True iff any external has softness m >= 1 (S^mC^n / S^m).
    p_i q_j is the fully validated domain (小马 2026-09-18); the soft-domain
    spec (2026-09-20) is implemented but still pending validation."""
    return any(md[0] != 0 for md in ext_mode.values())


def _comps(sub, adj):
    """connected components of the induced subgraph on `sub`."""
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


def _k0_union_domain(ext_mode):
    """k0: every external is H or C_i^1 (no soft, no refined C_i^m)."""
    for md in ext_mode.values():
        if md[0] != 0:
            return False
        if md != H and md[1] != 1:
            return False
    return True


def _run_k0_union(verts, edges, ext_attach, ext_mode, verbose=True, vm_dedup=True):
    """k0 enumeration — union construction + the ">=2 cuts" rule (2026-09-18).

    Port of private/wide_angle_dev/dev_wa_skel0.py (validated region-identical
    to the old path on the full k0 corpus + random graphs):
      * H connected;  corner closure (no NON-root v∉H with all edges into H);
      * per leg: P_i = connected set ⊆ V∖H, containing root_i, touching H
        (= union of possibly-overlapping paths);  P_i = ∅ iff root_i ∈ H;
      * P_i pairwise disjoint;
      * remaining = V∖(H∪P's): every component adjacent to >=2 P's
        (startpoints counted as part of their P);
      * C_i = connected ⊇ P_i within (V∖H) ∩ allowed − (other P's);
      * every vertex of the remaining set must lie in >=2 cuts;
      * combo -> (vm, em) -> the unchanged check chain (with vm-level dedup).
    Returns None when the input is outside this sub-domain (old path handles it).
    """
    if not _k0_union_domain(ext_mode):
        return None
    V = sorted(verts); Vset = set(V)
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    edges_t = [tuple(sorted(e, key=str)) for e in edges]
    extv = set(ext_attach.values())
    g = Graph(V, edges_t, ext_attach)
    kappa = TC.kappa_of(ext_mode)
    usable = derive_usable_modes(ext_mode, kappa)
    layers_by_ext = usable_layers(ext_mode, kappa, usable)
    ext_cuts = {}
    allowed_by_cut = {}
    for name in ext_mode:
        md = ext_mode[name]
        if md == H:
            continue
        ext_cuts[name] = TC.cuts_for_external(name, ext_attach[name], md, kappa,
                                              layers_by_ext.get(name))
        allowed_by_cut[name] = [TC.cut_allowed_vertices(verts, edges, ext_attach,
                                                        ext_mode, name, c.mode)
                                for c in ext_cuts[name]]
    legs = sorted(ext_cuts.keys())
    for n in legs:
        if len(ext_cuts[n]) != 1:
            return None          # not plain single-level C_i^1 -> old path
    Hs = [frozenset(S) for r in range(1, len(V) + 1)
          for S in itertools.combinations(V, r) if _connected(S, adj)]
    n_corner = 0
    Hs2 = []
    for H0 in Hs:
        bad = False
        for v in V:
            if v in H0 or v in extv:
                continue
            if adj[v] and adj[v] <= set(H0):
                bad = True
                break
        if bad:
            n_corner += 1
        else:
            Hs2.append(H0)
    Hs = Hs2
    regions = {}
    n_cand = n_dup = n_left_kill = n_rule_kill = 0
    seen_ck = set()
    seen_vm = set() if vm_dedup else None
    t0 = time.time()
    for H0 in Hs:
        Hset = set(H0)
        alw = {n: set(allowed_by_cut[n][0]) & (Vset - Hset) for n in legs}
        pops = {}
        okH = True
        for n in legs:
            root = ext_attach[n]
            if root in Hset:
                pops[n] = [frozenset()]
                continue
            allowed = {w for w in Vset if w not in Hset and
                       (w == root or w not in extv)}
            opts = [S for S in _conn_sets(root, allowed, adj)
                    if any(adj[u] & Hset for u in S) and set(S) <= alw[n]]
            if not opts:
                okH = False
                break
            pops[n] = opts
        if not okH:
            continue
        for Ps in itertools.product(*[pops[n] for n in legs]):
            u = set()
            ok = True
            for S in Ps:
                if S & u:
                    ok = False
                    break
                u |= S
            if not ok:
                continue
            Pmap = dict(zip(legs, Ps))
            lf = (Vset - u) - Hset
            groups = [(n, Pmap[n]) for n in legs if Pmap[n]]
            if len(groups) >= 2:
                gg = []
                for n, P in groups:
                    gg.append(P | {h for h in Hset if adj[h] & P})
                good = True
                for K in _comps(lf, adj):
                    cnt = 0
                    for gset in gg:
                        if any(adj[uu] & gset for uu in K):
                            cnt += 1
                    if cnt < 2:
                        good = False
                        break
                if not good:
                    n_left_kill += 1
                    continue
            Csets = {}
            okc = True
            for n in legs:
                P = Pmap[n]
                if not P:
                    Csets[n] = [frozenset()]
                    continue
                dom = alw[n] - (u - set(P))
                opts = _conn_supersets(P, dom, adj)
                if not opts:
                    okc = False
                    break
                Csets[n] = opts
            if not okc:
                continue
            for Ccomb in itertools.product(*[Csets[n] for n in legs]):
                if lf:
                    viol = False
                    for v in lf:
                        cntc = sum(1 for Cc in Ccomb if v in Cc)
                        if cntc < 2:
                            viol = True
                            break
                    if viol:
                        n_rule_kill += 1
                        continue
                assign = [(ext_cuts[n][0], Cc) for n, Cc in zip(legs, Ccomb) if Cc]
                ck = tuple(sorted((c.name, tuple(sorted(S))) for c, S in assign))
                if ck in seen_ck:
                    n_dup += 1
                    continue
                seen_ck.add(ck)
                n_cand += 1
                r = _check_combo(V, edges_t, g, ext_attach, ext_mode, assign,
                                 seen_vm)
                if r is None:
                    continue
                vm, em = r
                key = frozenset((v, vm[v]) for v in V)
                if key not in regions:
                    regions[key] = (vm, em)
    dt = time.time() - t0
    if verbose:
        print('skeleton k0-union: candidates %d (dup %d, leftover-filter %d, '
              '>=2-cut killed %d, corner-skipped H %d, vm-unique %d), regions %d, %.1fs'
              % (n_cand, n_dup, n_left_kill, n_rule_kill, n_corner,
                 len(seen_vm) if seen_vm is not None else 0, len(regions), dt))
    return list(regions.values()), n_cand, dt


def run(verts, edges, ext_attach, ext_mode, verbose=True, use_overlap=True,
        use_route=True, overlap_strict=True, overlap_strong=True,
        overlap_level=True, cfg_out=None, allow_soft=False, vm_dedup=True):
    t0 = time.time()
    if has_soft_externals(ext_mode) and not allow_soft:
        raise ValueError(
            'skeleton: soft externals (S^mC^n / S^m) present; the p_i q_j '
            'domain is fully validated, and soft support (小马 2026-09-20 '
            'spec) is implemented but still under validation. Pass '
            'allow_soft=True to run it (or use the layered enumerator).')
    # k0: always the union construction (single-path route removed 2026-09-20).
    r = _run_k0_union(verts, edges, ext_attach, ext_mode,
                      verbose=verbose, vm_dedup=vm_dedup)
    if r is not None:
        return r
    kappa = TC.kappa_of(ext_mode)
    V = sorted(verts)
    Vset = set(V)
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    edges_t = [tuple(sorted(e, key=str)) for e in edges]
    extv = set(ext_attach.values())
    # 小马 2026-09-20: routes avoid OTHER p/q attachment vertices (m=0);
    # soft (l, m>=1) attachment vertices are NOT restricted.
    hard_extv = {ext_attach[n] for n in ext_mode if ext_mode[n][0] == 0}
    g = Graph(V, edges_t, ext_attach)

    def _overlap(S1, S2):
        if S1 & S2:
            return True
        if overlap_strict:
            # 小马 2026-09-17: strengthened semantics — require a shared
            # vertex; the edge-overlap branch (edge with endpoints in the
            # two cuts) is dropped.  Now the default (193-case sweep,
            # zero mis-kills; pass overlap_strict=False for the old form).
            return False
        for u in S1:
            if adj[u] & S2:
                return True
        return False

    # Layer compression (usable_modes, the 2026-08-12 module; wired in
    # 2026-09-17): enumerate cuts only on the usable layers — dead layers
    # below n0 are dropped; k4/k5-type towers collapse to k2 chain counts
    # (e.g. DivingBeetle k4: 73/273/273/308 -> 13/73/73/88).
    usable = derive_usable_modes(ext_mode, kappa)
    layers_by_ext = usable_layers(ext_mode, kappa, usable)
    ext_cuts = {}
    for name in ext_mode:
        md = ext_mode[name]
        if md == H:
            continue
        ext_cuts[name] = TC.cuts_for_external(name, ext_attach[name], md,
                                              kappa, layers_by_ext.get(name))
    allowed_by_cut = {}
    for name, cs in ext_cuts.items():
        allowed_by_cut[name] = [TC.cut_allowed_vertices(verts, edges, ext_attach,
                                                        ext_mode, name, c.mode)
                                for c in cs]
    # 小马 2026-09-20: a C_i^a cut may not contain the incident vertex of a
    # soft leg (S^mC_j^n / S^m, m>=1) when a > m (paths may; cuts may not).
    # [A/B escape: FRI_NO_SOFTCUT_RESTRICTION=1]
    if not os.environ.get('FRI_NO_SOFTCUT_RESTRICTION'):
        for name in allowed_by_cut:
            md0 = ext_mode[name]
            if md0[0] != 0:
                continue
            for idx, cut in enumerate(ext_cuts[name]):
                a = cut.mode[1]
                for ln_ in ext_mode:
                    md2 = ext_mode[ln_]
                    if ln_ == name or md2[0] < 1:
                        continue
                    if a > md2[0]:
                        allowed_by_cut[name][idx].discard(ext_attach[ln_])
    ctype = [n for n in ext_cuts if ext_mode[n][0] == 0]
    stype = [n for n in ext_cuts if ext_mode[n][0] != 0]   # S^mC^n / S^m, m>=1
    if verbose:
        print('kappa=%d externals: %s | C-type: %s | soft: %s'
              % (kappa, {n: TC.mode_str(ext_mode[n]) for n in ext_mode},
                 ctype, stype))

    Hs = [frozenset(S) for r in range(1, len(V) + 1)
          for S in itertools.combinations(V, r) if _connected(S, adj)]
    if verbose:
        print('connected H blocks: %d' % len(Hs))

    regions = {}
    n_cand = 0
    n_skip = 0
    n_ov_kill = 0
    n_route_kill = 0
    seen_ck = set()
    seen_vm = set()
    tH0 = time.time()
    for iH, H0 in enumerate(Hs):
        Hset = set(H0)
        need = [n for n in ctype if ext_attach[n] not in Hset]
        legpaths = {}
        ok = True
        for n in need:
            root = ext_attach[n]
            allowed = {w for w in Vset if w not in Hset and
                       (w == root or w not in hard_extv)}
            ps = _paths_to_H(root, Hset, allowed, adj)
            if not ps:
                ok = False
                break
            legpaths[n] = ps
        if not ok:
            continue
        need_sorted = sorted(need)

        def rec_paths(i, used, chosen):
            if i == len(need_sorted):
                yield dict(chosen)
                return
            n = need_sorted[i]
            for p in legpaths[n]:
                if not (p & used):
                    chosen[n] = p
                    yield from rec_paths(i + 1, used | p, chosen)
                    del chosen[n]

        for path_assign in rec_paths(0, frozenset(), {}):
            chain_opts = {}
            okc = True
            for n in list(ext_cuts.keys()):
                root = ext_attach[n]
                cs = ext_cuts[n]
                if not cs:
                    # no usable cuts (SC/S softer than S^kappa): external
                    # momentum only — no cut contributes.
                    chain_opts[n] = [()]
                    continue
                if root in Hset:
                    chain_opts[n] = [()]
                    continue
                alw = [set(a) & (Vset - Hset) for a in allowed_by_cut[n]]
                if ext_mode[n][0] != 0:
                    # S^mC^n / S^m (m>=1): nested cut chains confined to
                    # V\H and avoiding ALL routes P_j (小马 2026-09-20) —
                    # the soft cuts touch neither H nor any P-path vertex.
                    # No root->H path requirement (小马 2026-09-17).
                    blocked = set()
                    for P in path_assign.values():
                        blocked |= P
                    alw = [a - blocked for a in alw]
                    chain_opts[n] = TC.nested_chains(verts, edges, root,
                                                     len(cs), alw)
                    continue
                # C_i^m-type: base cut must contain a root->H path.
                P = path_assign[n]
                N = len(cs)
                if not (P <= alw[0]):
                    okc = False
                    break
                S1s = _conn_supersets(P, alw[0], adj)
                if not S1s:
                    okc = False
                    break
                chains = []
                for S1 in S1s:
                    def rec2(j, prev, acc):
                        chains.append(acc + tuple(frozenset() for _ in range(N - len(acc))))
                        if j >= N:
                            return
                        allowed_j = alw[j] & set(prev)
                        for S in _conn_sets(root, allowed_j, adj):
                            rec2(j + 1, S, acc + (S,))
                    rec2(1, S1, (S1,))
                if not chains:
                    okc = False
                    break
                chain_opts[n] = chains
            if not okc:
                continue
            leglist = sorted(chain_opts.keys())
            for combo_chains in itertools.product(*[chain_opts[n] for n in leglist]):
                assign = []
                for n, chain in zip(leglist, combo_chains):
                    cs = ext_cuts[n]
                    for cut, S in zip(cs, chain):
                        if S:
                            assign.append((cut, S))
                ck = tuple(sorted((c.name, tuple(sorted(S))) for c, S in assign))
                if use_overlap:
                    # 小马 2026-09-17（WA 版, rev. 01:55）: every C_i^n cut
                    # with n < m must have nonempty overlap with some cut
                    # from another direction j -- UNLESS it coincides (as a
                    # vertex set) with the C_i^m cut of the same leg (which
                    # cannot happen when m = inf).  overlap = shared vertex,
                    # or an edge with endpoints in the two cuts respectively.
                    ok_ov = True
                    for cut, S in assign:
                        nm = cut.ext
                        md = ext_mode[nm]
                        if md[0] != 0:
                            continue          # only C_i^m-type externals
                        if cut.mode[1] < md[1]:  # n < m
                            if any(c2.ext == nm and c2.mode[1] > cut.mode[1]
                                   and S2 == S for c2, S2 in assign):
                                continue      # coincides with a deeper
                                # same-leg C_i^N cut (N > n) — 小马
                                # 2026-09-20 (former rule: only the m-level
                                # cut, impossible for m = inf)
                            if overlap_strong:
                                # 小马 2026-09-17 pm: "strong overlap" — the
                                # layer must contain a vertex v shared with
                                # cuts from at least TWO other directions
                                # (triple-sharing; takes precedence over the
                                # shared-vertex rule when enabled).
                                # 2026-09-18: overlap_level tightens to
                                # "two other C^n cuts" — the other
                                # directions' cuts must have the SAME level
                                # n as this layer.  Default ON since
                                # 2026-09-18 (pass False for the older
                                # any-level strong overlap).
                                # (1) 小马 2026-09-20 relaxed: shares a vertex
                                # with same-level partners of the form
                                # S^{n'}C_j^{n-n'} (sigma = n; n'=0 gives
                                # C_j^n) from >= 2 other directions j.
                                sig = cut.mode[0] + cut.mode[1]
                                by_dir = {}
                                for c2, S2 in assign:
                                    if c2.ext == nm:
                                        continue
                                    j2 = c2.mode[2]
                                    if j2 == 0 or j2 == cut.mode[2]:
                                        continue
                                    if overlap_level and (c2.mode[0] + c2.mode[1]) != sig:
                                        continue
                                    by_dir.setdefault(j2, set()).update(S2)
                                ok_st = False
                                for v in S:
                                    if sum(1 for vs in by_dir.values() if v in vs) >= 2:
                                        ok_st = True
                                        break
                                if not ok_st:
                                    # (2) 小马 2026-09-20 (amended 14:52):
                                    # soft support — an external of soft
                                    # power exactly n (S^nC_j^k or S^n)
                                    # whose incident vertex lies in this
                                    # layer.
                                    n_lv = cut.mode[1]
                                    for ln_ in ext_mode:
                                        md2 = ext_mode[ln_]
                                        if ln_ == nm or md2[0] != n_lv:
                                            continue
                                        if ext_attach[ln_] in S:
                                            ok_st = True
                                            break
                                if not ok_st:
                                    ok_ov = False
                                    if os.environ.get('FRI_DEBUG_OVL'):
                                        print('OVL-STRONG-KILL %s %s S=%s assign=%s' % (
                                              nm, cut.name, sorted(S),
                                              [(c.name, sorted(ss)) for c, ss in assign]),
                                              flush=True)
                                    break
                            else:
                                others = [S2 for (c2, S2) in assign if c2.ext != nm]
                                if not any(_overlap(S, S2) for S2 in others):
                                    ok_ov = False
                                    break
                    if not ok_ov:
                        n_ov_kill += 1
                        continue
                if use_route:
                    # 小马 2026-09-17 18:02: S_k^(i) ∩ P_j = ∅ for i != j —
                    # no leg's cut layers may touch another leg's route P_j.
                    ok_rt = True
                    for cut, S in assign:
                        for jn, P in path_assign.items():
                            if jn != cut.ext and (S & P):
                                ok_rt = False
                                break
                        if not ok_rt:
                            break
                    if not ok_rt:
                        n_route_kill += 1
                        if os.environ.get('FRI_DEBUG_ROUTE'):
                            print('ROUTE-KILL H=%s assign=%s paths=%s' % (
                                  sorted(Hset),
                                  [(c.name, sorted(S)) for c, S in assign],
                                  {n: sorted(P) for n, P in path_assign.items()}),
                                  flush=True)
                        continue
                # dedup AFTER the filters: the same cut-assignment can be
                # reached under several path choices, and passing the route
                # check may depend on that choice — dedup must not block the
                # good-path variant (2026-09-17 fix).
                if ck in seen_ck:
                    n_skip += 1
                    continue
                seen_ck.add(ck)
                n_cand += 1
                r = _check_combo(V, edges_t, g, ext_attach, ext_mode, assign,
                                 seen_vm if vm_dedup else None)
                if r is None:
                    continue
                vm, em = r
                key = frozenset((v, vm[v]) for v in V)
                if cfg_out is not None:
                    got = cfg_out.setdefault(key, [])
                    if len(got) < 200:
                        got.append(list(assign))
                if key not in regions:
                    regions[key] = (vm, em)
    dt = time.time() - t0
    if verbose:
        print('skeleton: candidates %d (dup-skipped %d, vm-unique %d, overlap-killed %d, route-killed %d), regions %d, %.1fs'
              % (n_cand, n_skip, len(seen_vm), n_ov_kill, n_route_kill, len(regions), dt))
    return list(regions.values()), n_cand, dt
