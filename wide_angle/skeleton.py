#!/usr/bin/env python3
r"""
skeleton.py — skeleton-based cut enumeration & pruning for wide-angle FRI.

Constructing the skeleton.
  * Enumerated explicitly a set of vertices whose induced subgraph is connected; this is H (no cuts touch it).
  * For every C_i^m-type external momentum that does not directly enter H: a path P_i from the root that touches H, such that
      it avoids vertices where other large-component momenta (p_j with j\neq i or q) enter; paths of different externals pairwise disjoint.

Constructing cuts associated with each P_i. They are a nested set of cuts in the p_i channel, such that:
    the modes of the cuts are C_i, C_i^2, ..., C_i^{n_i}, with n_i = m if m is finite, otherwise n_i = \kappa;
    the C_i cut contains the C_i^2 cut, which contains the C_i^3 cut, etc.;
    the C_i cut ⊇ P_i;
    the C_i, C_i^2, ..., C_i^{n_i} cuts do not contain vertices in H or other paths P_j.

Soft cuts. Around each S^mC^n-mode soft external momentum (m>=1), nested cuts S^mC_i^1, ..., S^mC_i^N (N = n for finite n, otherwise N = \kappa − m) or a single S^m cut when n=0, such that
    the S^mC_i^1 cut contains the S^mC_i^2 cut, which contains the S^mC_i^3 cut, etc.;
    these cuts avoid all the vertices in the aforementioned paths P_i or H.

In the absence of soft momenta (all external modes are H or C): every vertex of V∖(H∪P's) must lie in >=2 cuts.

Further pruning: overlap constraint.
* For each C_i^n-mode cut with n<m, either
    (1) a vertex shared by this cut and the same-sigma (sigma = n) S^{n'}C_j^{n-n'} cuts (n'=0..n; n'=0 reduces to C_j^n) of >=2 other directions j;
    (2) soft support — an external of soft power exactly n (S^nC_j^k / S^n) whose incident vertex is contained in this cut.
* Exemption from the requirements above: the C_i^n cut coincides with a C_i^N cut with N > n.

Tips for acceleration.
  * mask: cut sets carry bitmasks alongside the vertex sets; dedup keys and the route/overlap tests are fixed-slot bit operations.
  * chain-completion memoisation: the nested-chain superset recursion is memoised per (layer, innermost set), turning the recursion tree into a DAG (directed acyclic graph).
  * layer compression: cuts are enumerated only on the usable layers — dead layers are dropped, higher towers collapse to lower chain counts.
  * cut-set / vm dedup: repeated cut assignments are skipped by fixed-slot keys; repeated vertex modes are skipped outright (every check depends only on vm).
  * DFS (depth-first search) early pruning: a subtree is pruned as soon as some overlap requirement (or route exclusion) is provably unsatisfiable for all of its completions.
"""
import sys, os, time, itertools
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from primitives import Graph, vee, eq, harder_or_eq, meet, norm, is_connected, mode_str, INF
from region_checker import (jet_connected_ok, hard_jet_mojetic_ok, check_fc, momentum_ok, ir_ok)
from usable_modes import derive_usable_modes, usable_layers

H = (0, 0, 0)


# ===================== CUT & CHAIN CONSTRUCTION =====================

# One unitarity cut: a connected vertex set with a MODE; for external p_i with maximal degree n there are cuts C_i..C_i^n.
class Cut:
    def __init__(self, name, root, mode, ext=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.ext = ext  # owning external name


# Softest-mode S-power (paper corollary: no cascading modes): kappa = max{m+n} over the modes of ALL partial sums of the externals with nonzero virtuality.
def kappa_of(ext_mode):
    best = 0
    names = list(ext_mode)
    for r in range(1, len(names) + 1):
        for sub in itertools.combinations(names, r):
            acc = vee([ext_mode[n] for n in sub])
            m, n, i = acc
            if acc == H or n >= INF:
                continue  # zero virtuality or massless (n = +\infty)
            s = m + n
            if s > best:
                best = s
    return best if best > 0 else 1


# All connected subsets containing root, within allowed.
def connected_sets(root, allowed, adj):
    out = set()
    def dfs(cur, frontier):
        out.add(frozenset(cur))
        for w in list(frontier):
            dfs(cur | {w}, (frontier | (adj[w] & allowed)) - (cur | {w}))
    dfs({root}, set(adj[root]) & allowed)
    return out


# All connected supersets of the connected base, within allowed.
def connected_supersets(base, allowed, adj):
    out = set()
    base = frozenset(base)
    frontier = {w for v in base for w in adj[v] if w not in base and w in allowed}
    def dfs(cur, fr):
        out.add(frozenset(cur))
        for w in list(fr):
            dfs(cur | {w}, (fr | (adj[w] & allowed)) - (cur | {w}))
    dfs(set(base), frontier)
    return out


# Vertices allowed in a cut of leg `k_name`: unattached vertices, the leg's own root, and vertices carrying an external no harder than the cut's mode.
def cut_allowed_vertices(verts, edges, ext_attach, ext_mode, k_name, k_md):
    allowed = set()
    for v in verts:
        exts_at_v = [n for n, vv in ext_attach.items() if vv == v]
        if not exts_at_v:
            allowed.add(v)
            continue
        for n in exts_at_v:
            md = ext_mode[n]
            if n == k_name:
                allowed.add(v)
            elif eq(md, k_md) or harder_or_eq(k_md, md):
                allowed.add(v)
    return allowed


# The nested cuts of one external: S^mC_i, S^mC_i^2, ..., S^mC_i^N (N = md[1] if finite, else kappa).
def cuts_for_external(name, root, md, kappa, layers=None):
    m, n, i = md
    # m = soft power kept as a prefix (an SC/S external's cut inherits its soft feature; m = 0 reduces to C_i..C_i^N)
    if n == 0 and m >= 1:
        # Pure soft external S^m: one S^m cut (mode (m,0,0)) — the soft blob itself, no collinear direction.
        return [Cut(f'{name}_S{m}', root, (m, 0, 0), ext=name)]
    if layers is not None:
        # only the compressed reachable layers; sigma = m+k <= kappa caps k <= kappa - m (empty when kappa - m = 0)
        n_max = (kappa - m) if n >= INF else (n if n >= 1 else 1)
        ns = [k for k in layers if 1 <= k <= n_max]
        if not ns:
            # (compressed layers empty — fall back to the full range)
            ns = list(range(1, n_max + 1))
        return [Cut(f'{name}_S{m}C{i}^{k}', root, (m, k, i), ext=name) for k in ns]
    if n >= INF:
        n_max = kappa - m   # sigma(m,k) = m+k <= kappa
    else:
        n_max = n if n >= 1 else 1
    return [Cut(f'{name}_S{m}C{i}^{k}', root, (m, k, i), ext=name) for k in range(1, n_max + 1)]


# All nested chains (S_1 ⊇ ... ⊇ S_n) of one external, with per-layer allowed sets; partial chains allowed.
def nested_chains(root, n, allowed_list, adj):
    chains = []
    if n < 1:
        chains.append(())
        return chains
    # memoised completions per (layer, innermost set) — the recursion becomes a DAG traversal
    memo = {}
    # all nested supersets (S_j, ..., S_1) with S_k = S (innermost given)
    def completions(k, S):
        key = (k, frozenset(S))
        if key in memo:
            return memo[key]
        if k == 1:
            memo[key] = [(frozenset(S),)]
            return memo[key]
        out = []
        # the next-outer layer S_{k-1} must be within ITS OWN allowed set
        for S_up in connected_supersets(S, allowed_list[k-2], adj):
            for tail in completions(k - 1, S_up):
                out.append(tail + (frozenset(S),))
        memo[key] = out
        return out
    for k in range(0, n + 1):
        tail_empty = n - k
        if k == 0:
            chains.append(tuple(frozenset() for _ in range(n)))
            continue
        # the innermost nonempty layer S_k lies in its OWN layer's allowed set (essential for partial chains)
        for S_k in connected_sets(root, allowed_list[k-1], adj):
            for comp in completions(k, S_k):
                # store outer-first + trailing empty layers (S_1 outermost/largest; the pairing order matters)
                chain = tuple(comp) + tuple(frozenset() for _ in range(tail_empty))
                chains.append(chain)
    return chains


# ======================== GRAPH & MASK HELPERS ========================

# Bitmask helper: frozenset -> int mask, cached per graph (bit i <-> verts[i], injective; equal masks == equal sets, & == shared vertices).
def _make_mk(verts):
    bidx = {v: 1 << i for i, v in enumerate(verts)}
    cache = {}

    def mk(S):
        r = cache.get(S)
        if r is None:
            r = 0
            for v in S:
                r |= bidx[v]
            cache[S] = r
        return r

    return mk


# All simple root->H paths within allowed, each as its vertex set.
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


# Connected components of the subgraph induced on `sub`.
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


# ============================ CHECK CHAIN ============================

# Run the check chain (Step1 / FC / jet / mojetic / IR) on one cut assignment; returns (vm, em) or None.
def _check_combo(verts, edges_t, g, ext_attach, ext_mode, combo, seen_vm=None):
    vm = {}
    for v in verts:
        # combo entries: (cut, set, mask); bare (cut, set) entries are tolerated
        incuts = [cut.mode for cut, S, *_rest in combo if v in S]
        if not incuts:
            vm[v] = H
        else:
            acc = incuts[0]
            for md in incuts[1:]:
                acc = meet(acc, md)
            vm[v] = norm(acc)
    if seen_vm is not None:
        # vm-level dedup: every check depends only on vm; key = the vm mapping in fixed vertex order
        key_m = tuple([vm[v] for v in verts])
        if key_m in seen_vm:
            return None
        seen_vm.add(key_m)
    em = [meet(vm[u], vm[v]) for (u, v) in edges_t]
    for v in verts:
        accs = []
        for ei in g.incident.get(v, []):
            accs.append(em[ei])
        for name, vv in ext_attach.items():
            if vv == v:
                accs.append(ext_mode[name])
        if accs:
            if not eq(vee(accs), vm[v]):
                return None
    if not momentum_ok(g, em, ext_mode):
        return None
    if not jet_connected_ok(vm, em, edges_t):
        return None
    if not check_fc(g, em, ext_mode):
        return None
    ok_mj, _ = hard_jet_mojetic_ok(edges_t, em, ext_attach, ext_mode)
    if not ok_mj:
        return None
    if not ir_ok(g, em, ext_mode):
        return None
    return vm, em


# ================== ONLY H AND C: OVERLAP RESTRICTION ==================

# True when every external mode is H or C_i^1 — the domain of the union construction.
def _all_H_or_C(ext_mode):
    for md in ext_mode.values():
        if md[0] != 0:
            return False
        if md != H and md[1] != 1:
            return False
    return True


# Union construction for the only-H-and-C case + the ">=2 cuts" rule; returns None outside this domain.
def _run_union_construction(verts, edges, ext_attach, ext_mode, verbose=True, vm_dedup=True):
    if not _all_H_or_C(ext_mode):
        return None
    V = sorted(verts); Vset = set(V)
    mk = _make_mk(V)                  # (set, mask) pairs built once, below
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    edges_t = [tuple(sorted(e, key=str)) for e in edges]
    extv = set(ext_attach.values())
    g = Graph(V, edges_t, ext_attach)
    kappa = kappa_of(ext_mode)
    usable = derive_usable_modes(ext_mode, kappa)
    layers_by_ext = usable_layers(ext_mode, kappa, usable)
    ext_cuts = {}
    allowed_by_cut = {}
    for name in ext_mode:
        md = ext_mode[name]
        if md == H:
            continue
        ext_cuts[name] = cuts_for_external(name, ext_attach[name], md, kappa, layers_by_ext.get(name))
        allowed_by_cut[name] = [cut_allowed_vertices(verts, edges, ext_attach, ext_mode, name, c.mode) for c in ext_cuts[name]]
    legs = sorted(ext_cuts.keys())
    for n in legs:
        if len(ext_cuts[n]) != 1:
            return None          # not plain single-level C_i^1 -> general path
    SLOTS0 = sorted(ext_cuts[n][0].name for n in legs)
    SIDX0 = {nm: k for k, nm in enumerate(SLOTS0)}
    Hs = [frozenset(S) for r in range(1, len(V) + 1) for S in itertools.combinations(V, r) if is_connected(S, adj)]
    n_corner = 0
    Hs2 = []
    # corner closure: drop H containing a non-root v ∉ H whose edges all go into H
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
        # P_i: connected set ⊆ V∖H containing root_i and touching H (pairwise disjointness checked below)
        for n in legs:
            root = ext_attach[n]
            if root in Hset:
                pops[n] = [frozenset()]
                continue
            allowed = {w for w in Vset if w not in Hset and (w == root or w not in extv)}
            opts = [S for S in connected_sets(root, allowed, adj) if any(adj[u] & Hset for u in S) and set(S) <= alw[n]]
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
            # every component of V∖(H∪P's) must be adjacent to >= 2 paths (startpoints count in their own path)
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
                    Csets[n] = [(frozenset(), 0)]      # empty set -> mask 0
                    continue
                dom = alw[n] - (u - set(P))
                # C_i: connected superset of P_i within (V∖H) ∩ allowed minus the other paths
                opts = connected_supersets(P, dom, adj)
                if not opts:
                    okc = False
                    break
                # carry (set, mask) with each cut; the inner loops use masks
                Csets[n] = [(S, mk(S)) for S in opts]
            if not okc:
                continue
            for Ccomb in itertools.product(*[Csets[n] for n in legs]):
                # every remaining vertex must lie in >= 2 cuts
                if lf:
                    viol = False
                    for v in lf:
                        cntc = sum(1 for (S, _m) in Ccomb if v in S)
                        if cntc < 2:
                            viol = True
                            break
                    if viol:
                        n_rule_kill += 1
                        continue
                assign = [(ext_cuts[n][0], S, m) for n, (S, m) in zip(legs, Ccomb) if S]
                # Fixed-slot dedup key: arr[cut slot] = cut mask; -1 = cut absent (masks are >= 0 -> the sentinel is safe).
                arr = [-1] * len(SLOTS0)
                for c, _S, m in assign:
                    arr[SIDX0[c.name]] = m
                ck = tuple(arr)
                if ck in seen_ck:
                    n_dup += 1
                    continue
                seen_ck.add(ck)
                n_cand += 1
                r = _check_combo(V, edges_t, g, ext_attach, ext_mode, assign, seen_vm)
                if r is None:
                    continue
                vm, em = r
                # fixed-order tuple key (same dedup semantics)
                key = tuple([vm[v] for v in V])
                if key not in regions:
                    regions[key] = (vm, em)
    dt = time.time() - t0
    if verbose:
        print('skeleton union (only H and C): candidates %d (dup %d, leftover-filter %d, >=2-cut killed %d, corner-skipped H %d, vm-unique %d), regions %d, %.1fs' % (n_cand, n_dup, n_left_kill, n_rule_kill, n_corner, len(seen_vm) if seen_vm is not None else 0, len(regions), dt))
    return list(regions.values()), n_cand, dt


# ========================= MAIN ENUMERATION =========================

# Main entry: enumerate skeleton candidates for all H blocks; returns (regions, candidates, dt).
def run(verts, edges, ext_attach, ext_mode, verbose=True, use_overlap=True, use_route=True, overlap_strict=True, overlap_strong=True, overlap_level=True, cfg_out=None, allow_soft=True, vm_dedup=True):
    t0 = time.time()
    _DBG_OVL = bool(os.environ.get('FRI_DEBUG_OVL'))
    _DBG_ROUTE = bool(os.environ.get('FRI_DEBUG_ROUTE'))
    _no_prune = bool(os.environ.get('FRI_NO_PRUNE'))   # [A/B escape: disable the early pruning]
    # soft externals supported (validated against the soft corpora; allow_soft is a compatibility no-op); the only-H-and-C case takes the union construction first
    r = _run_union_construction(verts, edges, ext_attach, ext_mode, verbose=verbose, vm_dedup=vm_dedup)
    if r is not None:
        return r
    kappa = kappa_of(ext_mode)
    V = sorted(verts)
    Vset = set(V)
    mk = _make_mk(V)                  # per-graph bitmask helper
    # external-attachment masks (soft-support test below =  m & emask[ln_])
    emask = {ln_: mk(frozenset({ext_attach[ln_]})) for ln_ in ext_mode}
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    edges_t = [tuple(sorted(e, key=str)) for e in edges]
    extv = set(ext_attach.values())
    # Routes avoid OTHER p/q attachment vertices (m=0); soft (l, m>=1) attachment vertices are NOT restricted.
    hard_extv = {ext_attach[n] for n in ext_mode if ext_mode[n][0] == 0}
    g = Graph(V, edges_t, ext_attach)

    def _overlap(S1, S2):
        # shared vertex; in non-strict mode an edge between the two cuts also counts
        if S1 & S2:
            return True
        if overlap_strict:
            return False
        for u in S1:
            if adj[u] & S2:
                return True
        return False

    # layer compression: only the usable layers are enumerated (towers collapse, e.g. DivingBeetle k4: 73/273/273/308 -> 13/73/73/88)
    usable = derive_usable_modes(ext_mode, kappa)
    layers_by_ext = usable_layers(ext_mode, kappa, usable)
    ext_cuts = {}
    for name in ext_mode:
        md = ext_mode[name]
        if md == H:
            continue
        ext_cuts[name] = cuts_for_external(name, ext_attach[name], md, kappa, layers_by_ext.get(name))
    allowed_by_cut = {}
    for name, cs in ext_cuts.items():
        allowed_by_cut[name] = [cut_allowed_vertices(verts, edges, ext_attach, ext_mode, name, c.mode) for c in cs]
    # Fixed slot order for cut-dedup keys: a candidate's key is the int array arr[SIDX[cut.name]] = cut mask (-1 = cut absent).
    SLOTS = sorted({c.name for n in ext_cuts for c in ext_cuts[n]})
    SIDX = {nm: k for k, nm in enumerate(SLOTS)}
    # a C_i^a cut may not use the incident vertex of a soft leg when a > m (paths may; cuts may not)  [escape: FRI_NO_SOFTCUT_RESTRICTION=1]
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
    # check plan for early pruning: per checked cut — partner slot pairs (same sigma, other directions; SC slots included) + soft-support slots
    _legnames = sorted(ext_cuts.keys())
    _legpos = {_n: _i for _i, _n in enumerate(_legnames)}
    _nslots = [len(ext_cuts.get(_n, [])) for _n in _legnames]
    _checks = []
    for _n in _legnames:
        _md = ext_mode[_n]
        if _md[0] != 0:
            continue
        for _ci, _cut in enumerate(ext_cuts[_n]):
            _lv = _cut.mode[1]
            if not (_lv < _md[1]):
                continue
            _i = _cut.mode[2]
            _bydir = {}
            for _n2 in _legnames:
                if _n2 == _n:
                    continue
                for _ci2, _c2 in enumerate(ext_cuts[_n2]):
                    _j2 = _c2.mode[2]
                    if _j2 == 0 or _j2 == _i:
                        continue
                    if _c2.mode[0] + _c2.mode[1] != _lv:
                        continue
                    _bydir.setdefault(_j2, []).append((_n2, _ci2))
            _dirs = list(_bydir.keys())
            _pairs = [(_bydir[_dirs[_a]], _bydir[_dirs[_b]])
                      for _a in range(len(_dirs)) for _b in range(_a + 1, len(_dirs))]
            _soft = [(_n2, emask[_n2]) for _n2 in ext_mode
                     if _n2 != _n and ext_mode[_n2][0] == _lv]
            _checks.append({'owner': _legpos[_n], 'own': (_n, _ci), 'lv': _lv,
                            'pairs': _pairs, 'soft': _soft})

    if verbose:
        print('kappa=%d externals: %s | C-type: %s | soft: %s' % (kappa, {n: mode_str(ext_mode[n]) for n in ext_mode}, ctype, stype))

    Hs = [frozenset(S) for r in range(1, len(V) + 1) for S in itertools.combinations(V, r) if is_connected(S, adj)]
    if verbose:
        print('connected H blocks: %d' % len(Hs))

    regions = {}
    n_cand = 0
    n_skip = 0
    n_ov_kill = 0
    n_route_kill = 0
    n_prune = 0
    n_prune_rt = 0
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
            allowed = {w for w in Vset if w not in Hset and (w == root or w not in hard_extv)}
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
                    # No usable cuts (SC/S softer than S^kappa): external momentum only — no cut contributes.
                    chain_opts[n] = [()]
                    continue
                if root in Hset:
                    chain_opts[n] = [()]
                    continue
                alw = [set(a) & (Vset - Hset) for a in allowed_by_cut[n]]
                if ext_mode[n][0] != 0:
                    # S^mC^n / S^m: nested cut chains confined to V\H, avoiding all routes P_j (no root->H path requirement)
                    blocked = set()
                    for P in path_assign.values():
                        blocked |= P
                    alw = [a - blocked for a in alw]
                    # chain elements carry (set, mask) — built once, reused
                    chain_opts[n] = [tuple((S, mk(S)) for S in ch) for ch in nested_chains(root, len(cs), alw, adj)]
                    continue
                # C_i^m-type: base cut must contain a root->H path.
                P = path_assign[n]
                N = len(cs)
                if not (P <= alw[0]):
                    okc = False
                    break
                S1s = connected_supersets(P, alw[0], adj)
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
                        for S in connected_sets(root, allowed_j, adj):
                            rec2(j + 1, S, acc + (S,))
                    rec2(1, S1, (S1,))
                if not chains:
                    okc = False
                    break
                # chain elements carry (set, mask) — inner loops stay mask-only
                chain_opts[n] = [tuple((S, mk(S)) for S in ch) for ch in chains]
            if not okc:
                continue
            leglist = sorted(chain_opts.keys())
            # per-slot possibility masks (union over this context's chain options), used by the early pruning
            _slotposs = {}
            for _n in leglist:
                for _ci in range(len(ext_cuts.get(_n, []))):
                    _pm = 0
                    for _ch in chain_opts[_n]:
                        if len(_ch) > _ci:
                            _pm |= _ch[_ci][1]
                    _slotposs[(_n, _ci)] = _pm
            _use_prune = bool(use_overlap and overlap_strong and overlap_level and _checks
                               and _legnames == leglist)
            # route masks: P_j as a bitmask (route test:  m & pmask[jn])
            pmask = {jn: mk(P) for jn, P in path_assign.items()}
            # depth-first over legs; a subtree is pruned once the overlap requirements (or route exclusion) are provably unsatisfiable for all completions
            _opts = [chain_opts[n] for n in leglist]
            _nleg = len(leglist)
            # visit legs owning overlap checks first (they unlock early pruning), the rest after
            _haschk = [False] * len(leglist)
            for _ck in _checks:
                _haschk[_ck['owner']] = True
            _dord = ([_i for _i in range(len(leglist)) if _haschk[_i]] +
                     [_i for _i in range(len(leglist)) if not _haschk[_i]])
            _cur = [None] * len(leglist)
            _chc = [None] * len(leglist)
            _curall = [0] * len(leglist)
            # route early-prune: for each leg, mask of vertices forbidden by OTHER legs' paths P_j
            _forb = []
            for _i in range(len(leglist)):
                _f = 0
                for _j, _m in pmask.items():
                    if _j != leglist[_i]:
                        _f |= _m
                _forb.append(_f)

            def _slot_val(_n, _ci):
                # mask of a cut slot: actual if its leg is assigned, else a possibility upper bound
                _v = _cur[_legpos[_n]]
                if _v is not None:
                    return _v[_ci]
                return _slotposs.get((_n, _ci), 0)

            def _prune_dead():
                # True if some overlap check is provably failed for every completion of this prefix
                for _ck in _checks:
                    _v = _cur[_ck['owner']]
                    if _v is None:
                        continue
                    _ci = _ck['own'][1]
                    _m = _v[_ci]
                    if _m == 0:
                        continue
                    _exempt = False
                    for _ci2 in range(_ci + 1, _nslots[_ck['owner']]):
                        if _v[_ci2] == _m:
                            _exempt = True
                            break
                    if _exempt:
                        continue
                    _live = False
                    for (_sa, _sb) in _ck['pairs']:
                        _ma = 0
                        for (_ln, _lc) in _sa:
                            _ma |= _slot_val(_ln, _lc)
                        _mb = 0
                        for (_ln, _lc) in _sb:
                            _mb |= _slot_val(_ln, _lc)
                        if _m & _ma & _mb:
                            _live = True
                            break
                    if not _live and _ck['soft']:
                        for (_ln, _eb) in _ck['soft']:
                            if _m & _eb:
                                _live = True
                                break
                    if not _live:
                        return True
                return False

            _stack = [(0, iter(_opts[_dord[0]]))]
            while _stack:
                _d, _it = _stack[-1]
                _nx = next(_it, None)
                if _nx is None:
                    _stack.pop()
                    _cur[_dord[_d]] = None
                    _chc[_dord[_d]] = None
                    _curall[_dord[_d]] = 0
                    continue
                _li = _dord[_d]
                _cur[_li] = tuple(_nx[_c][1] if _c < len(_nx) else 0
                                  for _c in range(_nslots[_li]))
                _chc[_li] = _nx
                _allm = 0
                for _mm in _cur[_li]:
                    _allm |= _mm
                _curall[_li] = _allm
                if not _no_prune and (_allm & _forb[_li]):
                    _sk = 1
                    for _dd in range(_d + 1, _nleg):
                        _sk *= len(_opts[_dord[_dd]])
                    n_prune_rt += _sk
                    continue
                if _d < _nleg - 1:
                    if _use_prune and not _no_prune and _prune_dead():
                        _sk = 1
                        for _dd in range(_d + 1, _nleg):
                            _sk *= len(_opts[_dord[_dd]])
                        n_prune += _sk
                        continue
                    _stack.append((_d + 1, iter(_opts[_dord[_d + 1]])))
                    continue
                combo_chains = tuple(_chc)
                assign = []
                for n, chain in zip(leglist, combo_chains):
                    cs = ext_cuts[n]
                    for cut, SM in zip(cs, chain):
                        S, m = SM
                        if S:
                            assign.append((cut, S, m))
                arr = [-1] * len(SLOTS)
                for c, _S, m in assign:
                    arr[SIDX[c.name]] = m
                ck = tuple(arr)
                if use_overlap:
                    # overlap requirement: every C_i^n cut (n < m) needs a partner from another direction — unless it coincides with a deeper same-leg cut
                    ok_ov = True
                    _deep = {}
                    for _c2, _S2, _m2 in assign:
                        _k2 = (_c2.ext, _m2)
                        if _c2.mode[1] > _deep.get(_k2, -1):
                            _deep[_k2] = _c2.mode[1]
                    for cut, S, m in assign:
                        nm = cut.ext
                        md = ext_mode[nm]
                        if md[0] != 0:
                            continue          # only C_i^m-type externals
                        if cut.mode[1] < md[1]:  # n < m
                            if _deep.get((nm, m), -1) > cut.mode[1]:
                                continue      # coincides with a deeper
                                # Same-leg C_i^N cut (N > n) — for m = inf the m-level cut does not exist.
                            if overlap_strong:
                                # strong overlap: a vertex shared with same-sigma partners (S^{n'}C_j^{n-n'}) from >= 2 other directions (overlap_level: partners must have the same level n)
                                sig = cut.mode[0] + cut.mode[1]
                                by_dir = {}
                                for c2, S2, m2 in assign:
                                    if c2.ext == nm:
                                        continue
                                    j2 = c2.mode[2]
                                    if j2 == 0 or j2 == cut.mode[2]:
                                        continue
                                    if overlap_level and (c2.mode[0] + c2.mode[1]) != sig:
                                        continue
                                    by_dir[j2] = by_dir.get(j2, 0) | m2
                                # Masks: pair_or ORs the pairwise ANDs; a bit survives iff shared with >= 2 directions.
                                ok_st = False
                                dirs = list(by_dir.values())
                                pair_or = 0
                                _L = len(dirs)
                                for a_i in range(_L):
                                    for b_i in range(a_i + 1, _L):
                                        pair_or |= dirs[a_i] & dirs[b_i]
                                if m & pair_or:
                                    ok_st = True
                                if not ok_st:
                                    # (2) soft support — an external of soft power exactly n (S^nC_j^k or S^n) whose incident vertex lies in this layer.
                                    n_lv = cut.mode[1]
                                    for ln_ in ext_mode:
                                        md2 = ext_mode[ln_]
                                        if ln_ == nm or md2[0] != n_lv:
                                            continue
                                        if m & emask[ln_]:
                                            ok_st = True
                                            break
                                if not ok_st:
                                    ok_ov = False
                                    if _DBG_OVL:
                                        print('OVL-STRONG-KILL %s %s S=%s assign=%s' % (nm, cut.name, sorted(S), [(c.name, sorted(ss)) for c, ss, _m2 in assign]), flush=True)
                                    break
                            else:
                                others = [S2 for (c2, S2, _m2) in assign if c2.ext != nm]
                                if not any(_overlap(S, S2) for S2 in others):
                                    ok_ov = False
                                    break
                    if not ok_ov:
                        n_ov_kill += 1
                        continue
                if use_route:
                    # route exclusivity: no cut layer may touch another leg's route P_j (mask test: m & pmask[jn] == 0)
                    ok_rt = True
                    for cut, S, m in assign:
                        for jn in path_assign:
                            if jn != cut.ext and (m & pmask[jn]):
                                ok_rt = False
                                break
                        if not ok_rt:
                            break
                    if not ok_rt:
                        n_route_kill += 1
                        if _DBG_ROUTE:
                            print('ROUTE-KILL H=%s assign=%s paths=%s' % (sorted(Hset), [(c.name, sorted(S)) for c, S, _m in assign], {n: sorted(P) for n, P in path_assign.items()}), flush=True)
                        continue
                # dedup after the filters: the same cut-assignment can be reached under several path choices (dedup must not block the good-path variant)
                if ck in seen_ck:
                    n_skip += 1
                    continue
                seen_ck.add(ck)
                n_cand += 1
                r = _check_combo(V, edges_t, g, ext_attach, ext_mode, assign, seen_vm if vm_dedup else None)
                if r is None:
                    continue
                vm, em = r
                # fixed-order tuple key (same dedup semantics)
                key = tuple([vm[v] for v in V])
                if cfg_out is not None:
                    got = cfg_out.setdefault(key, [])
                    if len(got) < 200:
                        got.append([(c, S) for (c, S, _m) in assign])
                if key not in regions:
                    regions[key] = (vm, em)
    dt = time.time() - t0
    if verbose:
        print('skeleton: candidates %d (dup-skipped %d, vm-unique %d, early-pruned %d, overlap-killed %d, route-killed %d), regions %d, %.1fs' % (n_cand, n_skip, len(seen_vm), n_prune + n_prune_rt, n_ov_kill, n_route_kill, len(regions), dt))
    return list(regions.values()), n_cand, dt
