#!/usr/bin/env python3
"""skeleton.py — skeleton-based cut enumeration & pruning for wide-angle FRI
(formalized 2026-09-18; development history + intermediate rules in
private/skeleton_rules_history.md).

Conditions (agreed with 小马):
  * H connected (enumerated explicitly);
  * for every C_i^m-type external (mode (0,n,i), n>=1) whose root is not in H:
    a path P_i from the root that touches H (inside V\\H, avoiding other
    external vertices); paths of different externals pairwise disjoint;
  * base cut (largest chain level, paired with the least-soft cut) ⊇ P_i;
    chain levels S_1 ⊇ S_2 ⊇ ... : each connected containing root, within its
    own per-level allowed set, inside V\\H; partial chains (prefix nonempty);
  * combination -> (vm, em) via cut meets; same check chain as the enumerator.

* route-exclusivity: no leg's cut layers may touch another leg's route
  (S_k^(i) ∩ P_j = ∅ for i≠j) — default ON;
* overlap (C_i^m legs, layer n<m): a vertex shared by this cut and >=2 other
  directions' same-level C^n cuts, unless the layer coincides with the leg's
  own C_i^m cut (m=∞: no exemption) — default ON (overlap_strong=True since 2026-09-18 evening;
  pass overlap_strong=False for the old shared-vertex-only form, e.g. A/B runs).

Validated domain (小马 2026-09-18): p_i q_j externals only.  Soft externals
(S^mC^n / S^m, m>=1) are refused by default (allow_soft=True opts into the
experimental, unvalidated path; revisit later).

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
from mojetic_check import jet_connected_ok, cond1_ok
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
    Skeleton's validated domain is p_i q_j only (小马 2026-09-18)."""
    return any(md[0] != 0 for md in ext_mode.values())


def run(verts, edges, ext_attach, ext_mode, verbose=True, use_overlap=True,
        use_route=True, overlap_strict=True, overlap_strong=True,
        overlap_level=True, cfg_out=None, allow_soft=False, vm_dedup=True):
    t0 = time.time()
    if has_soft_externals(ext_mode) and not allow_soft:
        raise ValueError(
            'skeleton: soft externals (S^mC^n / S^m) present; the validated '
            'domain is p_i q_j only (小马 2026-09-18) — soft-domain support '
            'is deferred. Use the layered enumerator, or pass '
            'allow_soft=True for the experimental (unvalidated) path.')
    kappa = TC.kappa_of(ext_mode)
    V = sorted(verts)
    Vset = set(V)
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    edges_t = [tuple(sorted(e, key=str)) for e in edges]
    extv = set(ext_attach.values())
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
            allowed = {w for w in Vset if w not in Hset and (w == root or w not in extv)}
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
                    # V\H — the cut may not contain any H point (小马
                    # 2026-09-17); no root->H path requirement.
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
                            if any(c2.ext == nm and c2.mode[1] == md[1]
                                   and S2 == S for c2, S2 in assign):
                                continue      # coincides with the C_i^m cut
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
                                by_dir = {}
                                for c2, S2 in assign:
                                    if c2.ext != nm:
                                        if overlap_level and c2.mode[1] != cut.mode[1]:
                                            continue
                                        by_dir.setdefault(c2.ext, set()).update(S2)
                                ok_st = False
                                for v in S:
                                    if sum(1 for vs in by_dir.values() if v in vs) >= 2:
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
