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

Validated domain: p_i q_j externals (2026-09-18); soft externals
(S^mC^n / S^m, m>=1) via the 2026-09-20 extended spec — validated against
the soft corpora (464/464 files; see verify_soft_full.py).
* 2026-09-22 fast path (ported from the regge skeleton): cut sets carry
  bitmasks alongside the sets; cut-dedup keys, route and overlap checks are
  mask-based (per-graph memoization; fixed-slot int arrays).  Semantically
  identical — verified by old-vs-new A/B (region sets + per-case counters
  byte-identical, 2026-09-22).

(v1/v2/v3/v3.2 development history preserved in
private/skeleton_rules_history.md.)
"""
import sys, os, time, itertools
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import region_checker as rc
from primitives import Graph, vee, check_fc, momentum_ok, ir_ok_blocks
from read_graph import mode_str, INF
from region_checker import jet_connected_ok, cond1_ok
from usable_modes import derive_usable_modes, usable_layers

H = (0, 0, 0)


# ---------------- cut basics (moved from truncation_check.py, 2026-09-23) ----------------
class Cut:
    """One unitarity cut: a connected vertex set with a MODE.
    For external p_i with maximal degree n there are cuts C_i..C_i^n."""
    def __init__(self, name, root, mode, ext=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.ext = ext  # owning external name


def kappa_of(ext_mode):
    """Softest-mode S-power (paper corollary: no cascading modes).

    Per draft-v16 eq:partial_sum_external_momenta_mode: consider the modes of
    ALL partial sums (one or more momenta) of the external momenta with
    nonzero virtuality (mode != H, n != +inf);  kappa = max{m+n} over them.
    (2026-08-13: fixed from 'single external only' — caught it; e.g. a
    purely massless kinematics {C_i^inf, SC^inf} has partial-sum modes
    {H, C_i} -> kappa = 1, not a hardcoded fallback.)"""
    best = 0
    names = list(ext_mode)
    for r in range(1, len(names) + 1):
        for sub in itertools.combinations(names, r):
            acc = vee([ext_mode[n] for n in sub])
            m, n, i = acc
            if acc == H or n >= INF:
                continue  # zero virtuality or massless (n = +inf)
            s = m + n
            if s > best:
                best = s
    return best if best > 0 else 1


def connected_sets(verts, edges, root, allowed=None, maxsize=None):
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    if allowed is None:
        allowed = set(verts)
    if maxsize is None:
        maxsize = len(verts)
    out = set()
    def dfs(cur, frontier):
        out.add(frozenset(cur))
        if len(cur) >= maxsize:
            return
        for w in list(frontier):
            dfs(cur | {w}, (frontier | (adj[w] & allowed)) - (cur | {w}))
    dfs({root}, set(adj[root]) & allowed)
    return out


def connected_supersets(verts, edges, base, allowed=None, maxsize=None):
    """All connected vertex sets containing base (base connected, nonempty)."""
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    if allowed is None:
        allowed = set(verts)
    if maxsize is None:
        maxsize = len(verts)
    out = set()
    base = frozenset(base)
    frontier = {w for v in base for w in adj[v] if w not in base and w in allowed}
    def dfs(cur, fr):
        out.add(frozenset(cur))
        if len(cur) >= maxsize:
            return
        for w in list(fr):
            dfs(cur | {w}, (fr | (adj[w] & allowed)) - (cur | {w}))
    dfs(set(base), frontier)
    return out


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
            elif rc.eq(md, k_md) or rc.harder_or_eq(k_md, md):
                allowed.add(v)
    return allowed


def cuts_for_external(name, root, md, kappa, layers=None):
    """The nested cuts for external `name`: S^mC_i, S^mC_i^2, ..., S^mC_i^N.

    m = soft power of the external mode (kept as a prefix: an SC/S external's
    cut inherits its soft feature); N = md[1] if finite, N = kappa if C_i^infty
    (rule (2)); for a pure collinear external (m=0) this reduces to the
    classical C_i..C_i^N chain (2026-08-11: conjecture, to be verified
    against pySecDec for SC externals).
    If `layers` is given (compressed reachable layers, ascending), only those
    n values are used (skipped layers collapse -> fewer cuts, same regions).
    """
    m, n, i = md
    if n == 0 and m >= 1:
        # pure soft external S^m: one S^m cut (mode (m,0,0)) — the soft
        # blob itself, no collinear direction.  2026-08-12.
        return [Cut(f'{name}_S{m}', root, (m, 0, 0), ext=name)]
    if layers is not None:
        # 2026-08-13 : the SC cut S^mC_i^k has sigma = m+k, and
        # no-cascading bounds every region mode by S^kappa (sigma <= kappa)
        # -> k <= kappa - m.  (kappa-m=0: SC external already softer than
        # S^kappa — no usable cuts, legitimately empty.)
        n_max = (kappa - m) if n >= INF else (n if n >= 1 else 1)
        ns = [k for k in layers if 1 <= k <= n_max]
        if not ns:
            # compressed layers empty — fall back to the full 1..n_max range
            ns = list(range(1, n_max + 1))
        return [Cut(f'{name}_S{m}C{i}^{k}', root, (m, k, i), ext=name) for k in ns]
    if n >= INF:
        n_max = kappa - m   # sigma(m,k) = m+k <= kappa (2026-08-13)
    else:
        n_max = n if n >= 1 else 1
    return [Cut(f'{name}_S{m}C{i}^{k}', root, (m, k, i), ext=name)
            for k in range(1, n_max + 1)]


def nested_chains(verts, edges, root, n, allowed_list):
    """All nested chains (S_1, ..., S_n) with S_1 ⊇ S_2 ⊇ ... ⊇ S_n,
    each S_k connected containing root, OR empty.
    Partial chains allowed: the first k >= 0 entries may be nonempty (nested
    supersets), the remaining n-k entries empty.  (A jet may use only some of
    its softer layers, e.g. only C_i without C_i^2.)

    allowed_list: PER-LAYER allowed vertex sets (allowed_list[k-1] for layer
    k).  Each layer S_k must lie inside the allowed set of ITS OWN cut mode —
    an outer (harder) cut may contain vertices with softer externals that the
    innermost cut cannot (e.g. an SC4 external vertex is allowed in the C1 cut
    but not the C1^2 cut, since SC4 <= C1 but SC4 and C1^2 overlap).  Using a
    single innermost-only allowed set for all layers was a bug (2026-08-11):
    it silently dropped regions like the paper's 5pt6loop Ciregion1, where
    l1=SC4 attaches inside the C1 jet.

    Optimisation (2026-08-09): the DFS over connected_supersets can explode
    on large graphs (e.g. hypercrown: 10^7+ chains).  We keep the enumeration
    but memoise per (j, S) the set of completions; this turns the nested
    superset recursion into a DAG traversal.  Identical output, much faster.
    """
    chains = []
    if n < 1:
        chains.append(())
        return chains
    # memo: (k, frozenset S_j) -> list of completions (tuples of frozensets)
    memo = {}
    def completions(k, S):
        """All nested supersets (S_{j}, ..., S_1) with S_k=S (innermost given)."""
        key = (k, frozenset(S))
        if key in memo:
            return memo[key]
        if k == 1:
            memo[key] = [(frozenset(S),)]
            return memo[key]
        out = []
        # the next-outer layer S_{k-1} must be within ITS OWN allowed set
        for S_up in connected_supersets(verts, edges, S, allowed_list[k-2]):
            for tail in completions(k - 1, S_up):
                out.append(tail + (frozenset(S),))
        memo[key] = out
        return out
    for k in range(0, n + 1):
        tail_empty = n - k
        if k == 0:
            chains.append(tuple(frozenset() for _ in range(n)))
            continue
        # the innermost NONEMPTY layer is S_k — it must lie in the k-th
        # layer's OWN allowed set (for k < n this is NOT the innermost cut's
        # allowed set; 2026-08-11 partial-chain fix)
        for S_k in connected_sets(verts, edges, root, allowed_list[k-1]):
            for comp in completions(k, S_k):
                # comp = (S_1, ..., S_k) outer-first (S_1 outermost/largest);
                # store outer-first + trailing empty layers (docstring semantics:
                # nested chain S_1 ⊇ ... ⊇ S_k, nonempty layers first).
                # (2026-08-09 fix: the previous `reversed(comp)` swapped the
                # layer order, pairing the innermost soft cut with the LARGEST
                # vertex set -> wrong vertex modes; 4pt3loop lost 2 regions,
                # 81 -> 79.)
                chain = tuple(comp) + tuple(frozenset() for _ in range(tail_empty))
                chains.append(chain)
    return chains


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


def _make_mk(verts):
    """Bitmask helper (2026-09-22 fast path, ported from the regge skeleton):
    frozenset(vertices) -> int mask, cached per graph.  Injective on subsets
    of `verts` (bit i <-> verts[i]): mask equality == vertex-set equality
    (coincidence/exemption tests) and mask & mask == shared vertices
    (overlap/route checks).  Empty set -> 0; masks are never negative, so
    the value -1 in dedup keys can safely mean "cut absent"."""
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


def _check_combo(verts, edges_t, g, ext_attach, ext_mode, combo,
                 seen_vm=None, massive=frozenset()):
    """copy of the enumerator's check chain (Step1 / FC / jet / mojetic / IR).

    seen_vm: optional set for vm-level dedup (2026-09-18).  em = meet(vm[u],
    vm[v]) and every check depends only on (vm, em): the outcome is a
    function of vm alone, so a repeated vm is skipped outright (the region
    key is vm-based anyway).  Behaviour-preserving; big speedup.
    """
    vm = {}
    for v in verts:
        # combo entries: (cut, set, mask); tolerant unpack keeps older dev
        # callers (cut, set) working
        incuts = [cut.mode for cut, S, *_rest in combo if v in S]
        if not incuts:
            vm[v] = H
        else:
            acc = incuts[0]
            for md in incuts[1:]:
                acc = rc.meet(acc, md)
            vm[v] = rc.norm(acc)
    if seen_vm is not None:
        # key = vm mapping in fixed vertex order (tuple builds/hashes cheaper
        # than a frozenset of pairs; 2026-09-22)
        key_m = tuple(vm[v] for v in verts)
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
    if not rc.massive_h_ok(edges_t, em, massive):
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
    spec (2026-09-20) is validated on the soft corpora (464/464)."""
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


def _run_k0_union(verts, edges, ext_attach, ext_mode, verbose=True,
                  vm_dedup=True, massive=frozenset()):
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
        ext_cuts[name] = cuts_for_external(name, ext_attach[name], md, kappa,
                                              layers_by_ext.get(name))
        allowed_by_cut[name] = [cut_allowed_vertices(verts, edges, ext_attach,
                                                        ext_mode, name, c.mode)
                                for c in ext_cuts[name]]
    legs = sorted(ext_cuts.keys())
    for n in legs:
        if len(ext_cuts[n]) != 1:
            return None          # not plain single-level C_i^1 -> old path
    SLOTS0 = sorted(ext_cuts[n][0].name for n in legs)
    SIDX0 = {nm: k for k, nm in enumerate(SLOTS0)}
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
                    Csets[n] = [(frozenset(), 0)]      # empty set -> mask 0
                    continue
                dom = alw[n] - (u - set(P))
                opts = _conn_supersets(P, dom, adj)
                if not opts:
                    okc = False
                    break
                # carry (set, mask) with each cut; the inner loops use masks
                Csets[n] = [(S, mk(S)) for S in opts]
            if not okc:
                continue
            for Ccomb in itertools.product(*[Csets[n] for n in legs]):
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
                # fixed-slot dedup key: arr[cut slot] = cut mask,
                # -1 = cut absent (masks are >= 0 -> sentinel is safe)
                arr = [-1] * len(SLOTS0)
                for c, _S, m in assign:
                    arr[SIDX0[c.name]] = m
                ck = tuple(arr)
                if ck in seen_ck:
                    n_dup += 1
                    continue
                seen_ck.add(ck)
                n_cand += 1
                r = _check_combo(V, edges_t, g, ext_attach, ext_mode, assign,
                                 seen_vm, massive)
                if r is None:
                    continue
                vm, em = r
                # fixed-order tuple key (2026-09-22; same dedup semantics)
                key = tuple(vm[v] for v in V)
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
        overlap_level=True, cfg_out=None, allow_soft=True, vm_dedup=True,
        massive=frozenset()):
    t0 = time.time()
    # soft externals: supported since the 2026-09-20 spec; validated against
    # the soft corpora (464/464).  allow_soft kept for backward compatibility
    # (no-op).
    # k0: always the union construction (single-path route removed 2026-09-20).
    r = _run_k0_union(verts, edges, ext_attach, ext_mode,
                      verbose=verbose, vm_dedup=vm_dedup, massive=massive)
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
        ext_cuts[name] = cuts_for_external(name, ext_attach[name], md,
                                              kappa, layers_by_ext.get(name))
    allowed_by_cut = {}
    for name, cs in ext_cuts.items():
        allowed_by_cut[name] = [cut_allowed_vertices(verts, edges, ext_attach,
                                                        ext_mode, name, c.mode)
                                for c in cs]
    # fixed slot order for cut-dedup keys: a candidate's key is the int
    # array  arr[SIDX[cut.name]] = cut mask  (-1 = cut absent)
    SLOTS = sorted({c.name for n in ext_cuts for c in ext_cuts[n]})
    SIDX = {nm: k for k, nm in enumerate(SLOTS)}
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
              % (kappa, {n: mode_str(ext_mode[n]) for n in ext_mode},
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
                    # chain elements carry (set, mask) — built once, reused
                    chain_opts[n] = [
                        tuple((S, mk(S)) for S in ch)
                        for ch in nested_chains(verts, edges, root, len(cs), alw)]
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
                # chain elements carry (set, mask) — inner loops stay mask-only
                chain_opts[n] = [tuple((S, mk(S)) for S in ch)
                                 for ch in chains]
            if not okc:
                continue
            leglist = sorted(chain_opts.keys())
            # route masks: P_j as a bitmask (route test:  m & pmask[jn])
            pmask = {jn: mk(P) for jn, P in path_assign.items()}
            for combo_chains in itertools.product(*[chain_opts[n] for n in leglist]):
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
                    # 小马 2026-09-17（WA 版, rev. 01:55）: every C_i^n cut
                    # with n < m must have nonempty overlap with some cut
                    # from another direction j -- UNLESS it coincides (as a
                    # vertex set) with the C_i^m cut of the same leg (which
                    # cannot happen when m = inf).  overlap = shared vertex,
                    # or an edge with endpoints in the two cuts respectively.
                    ok_ov = True
                    for cut, S, m in assign:
                        nm = cut.ext
                        md = ext_mode[nm]
                        if md[0] != 0:
                            continue          # only C_i^m-type externals
                        if cut.mode[1] < md[1]:  # n < m
                            if any(c2.ext == nm and c2.mode[1] > cut.mode[1]
                                   and m2 == m for c2, S2, m2 in assign):
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
                                for c2, S2, m2 in assign:
                                    if c2.ext == nm:
                                        continue
                                    j2 = c2.mode[2]
                                    if j2 == 0 or j2 == cut.mode[2]:
                                        continue
                                    if overlap_level and (c2.mode[0] + c2.mode[1]) != sig:
                                        continue
                                    by_dir[j2] = by_dir.get(j2, 0) | m2
                                # masks: pair_or ORs the pairwise ANDs; a bit
                                # survives iff shared with >= 2 directions
                                ok_st = False
                                dirs = list(by_dir.values())
                                pair_or = 0
                                for a_i in range(len(dirs)):
                                    for b_i in range(a_i + 1, len(dirs)):
                                        pair_or |= dirs[a_i] & dirs[b_i]
                                if m & pair_or:
                                    ok_st = True
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
                                        if m & emask[ln_]:
                                            ok_st = True
                                            break
                                if not ok_st:
                                    ok_ov = False
                                    if os.environ.get('FRI_DEBUG_OVL'):
                                        print('OVL-STRONG-KILL %s %s S=%s assign=%s' % (
                                              nm, cut.name, sorted(S),
                                              [(c.name, sorted(ss)) for c, ss, _m2 in assign]),
                                              flush=True)
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
                    # 小马 2026-09-17 18:02: S_k^(i) ∩ P_j = ∅ for i != j —
                    # no leg's cut layers may touch another leg's route P_j.
                    ok_rt = True
                    # mask test: route-exclusive cuts have  m & pmask[jn] == 0
                    for cut, S, m in assign:
                        for jn in path_assign:
                            if jn != cut.ext and (m & pmask[jn]):
                                ok_rt = False
                                break
                        if not ok_rt:
                            break
                    if not ok_rt:
                        n_route_kill += 1
                        if os.environ.get('FRI_DEBUG_ROUTE'):
                            print('ROUTE-KILL H=%s assign=%s paths=%s' % (
                                  sorted(Hset),
                                  [(c.name, sorted(S)) for c, S, _m in assign],
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
                                 seen_vm if vm_dedup else None, massive)
                if r is None:
                    continue
                vm, em = r
                # fixed-order tuple key (2026-09-22; same dedup semantics)
                key = tuple(vm[v] for v in V)
                if cfg_out is not None:
                    got = cfg_out.setdefault(key, [])
                    if len(got) < 200:
                        got.append([(c, S) for (c, S, _m) in assign])
                if key not in regions:
                    regions[key] = (vm, em)
    dt = time.time() - t0
    if verbose:
        print('skeleton: candidates %d (dup-skipped %d, vm-unique %d, overlap-killed %d, route-killed %d), regions %d, %.1fs'
              % (n_cand, n_skip, len(seen_vm), n_ov_kill, n_route_kill, len(regions), dt))
    return list(regions.values()), n_cand, dt
