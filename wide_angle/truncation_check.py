#!/usr/bin/env python3
"""truncation_check.py — the FRI region enumerator: layered DFS over
nested unitarity cuts (C/H family merged into layer 0, 2026-09-09).

Single enumerator: it builds the per-external nested cut chains and runs a
layered DFS over cut-chain combinations.  Layer 0 owns ALL regions with zero
shared vertices — the former C/H constructor (construct_CH_regions.py,
removed 2026-09-09) produced exactly the no-cross-edge half of layer 0, so
this is a pure union of two complementary subfamilies with an identical
check chain (Step1/FC/Mojetic/IR).

Layering (definition confirmed 2026-08-09):
  - a vertex is SHARED iff it is covered by cuts of >= 2 DIFFERENT externals;
    nested cuts C_i ⊇ C_i^2 ⊇ ... of the SAME external count only once.
  - layer k = regions with exactly k shared vertices (layer 0 = C/H family
    plus the k=0 overlapping regions; cross-edge overlap does not count
    toward k).

Truncation conjecture (: if layer k produces no regions, then all
deeper layers k+1..inf are empty.  Since |V| <= 12 here, we enumerate every
layer 0..|V| and check the conjecture EXACTLY on the instance (never rely on
the conjecture itself to stop).

Usage:
  python3 truncation_check.py 4pt3loop [--maxk N] [--brief]
  python3 truncation_check.py hypercrown k1|k2|k3|k4|k5 [--maxk N] [--brief]
"""
import sys, os, time, itertools
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_graph import mode_str, INF
from primitives import Graph, vee, check_fc, momentum_ok, ir_ok_blocks
import region_checker as rc

H = (0, 0, 0)

# ---------------- cut object ----------------
class Cut:
    """One unitarity cut: a connected vertex set with a MODE.
    For external p_i with maximal degree n there are cuts C_i..C_i^n."""
    def __init__(self, name, root, mode, ext=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.ext = ext  # owning external name

# ---------------- helpers ----------------
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

# ---------------- nested cuts per external ----------------
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

# ---------------- test cases ----------------
def case_4pt3loop():
    """7.1 example: 8 vertices / 10 edges, p1=C1, p2=C2^2, p3=C3^inf, p4=C4^inf.
    Known: 81 regions total = 16 C/H + 65 other."""
    verts = [1, 2, 3, 4, 5, 6, 7, 8]
    edges = [(1, 5), (1, 8), (2, 5), (2, 7), (3, 6), (3, 8), (4, 6), (4, 7),
             (5, 6), (7, 8)]
    ext_attach = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4}
    ext_mode = {'p1': (0, 1, 1), 'p2': (0, 2, 2), 'p3': (0, INF, 3),
                'p4': (0, INF, 4)}
    return verts, edges, ext_attach, ext_mode

def case_hypercrown(k):
    """12-propagator graph: central vertex 5 attached to all 4 externals,
    ring 6-7-8-9-6 with one external each (6-p2, 7-p1, 8-p3, 9-p4).
    Kinematics from verification/gen_hypercrown.py:
      k1: p1^2~t1,   p2^2~t1,   p3^2=0, p4^2=0   (pySecDec 71, FRI verified)
      k2: p1^2~t1,   p2^2~t1^2, p3^2=0, p4^2=0   (pySecDec 130, FRI explosion)
      k3: p1^2~t1,   p2^2~t1^2, p3^2=m3s, p4^2=0 (pySecDec 82, FRI verified)
      k4: p1^2~t1,   p2^2~t1^2, p3^2~t1, p4^2=0  (FRI explosion)
      k5: p1^2~t1^2, p2^2~t1^3, p3^2=0, p4^2=0   (FRI explosion)"""
    verts = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    edges = [(1, 5), (1, 7), (2, 5), (2, 6), (3, 5), (3, 8), (4, 5), (4, 9),
             (6, 7), (7, 8), (8, 9), (9, 6)]
    ext_attach = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4}
    if k == 'k1':
        ext_mode = {'p1': (0, 1, 1), 'p2': (0, 1, 2), 'p3': (0, INF, 3), 'p4': (0, INF, 4)}
    elif k == 'k2':
        ext_mode = {'p1': (0, 1, 1), 'p2': (0, 2, 2), 'p3': (0, INF, 3), 'p4': (0, INF, 4)}
    elif k == 'k3':
        ext_mode = {'p1': (0, 1, 1), 'p2': (0, 2, 2), 'p3': H, 'p4': (0, INF, 4)}
    elif k == 'k4':
        ext_mode = {'p1': (0, 1, 1), 'p2': (0, 2, 2), 'p3': (0, 1, 3), 'p4': (0, INF, 4)}
    elif k == 'k5':
        ext_mode = {'p1': (0, 2, 1), 'p2': (0, 3, 2), 'p3': (0, INF, 3), 'p4': (0, INF, 4)}
    else:
        raise SystemExit(f'unknown hypercrown case {k}')
    return verts, edges, ext_attach, ext_mode

# ---------------- layered enumeration ----------------
def run_layered(verts, edges, ext_attach, ext_mode, maxk=None, brief=False,
                no_ext_attach=False, use_compression=False, on_region=None,
                massive=frozenset()):
    """on_region: optional callback(k, vm, em) called the FIRST time a
    region is found (global dedup across layers — a region reachable via
    several chain combos / in several layers is reported once, at the layer
    where it is first discovered)."""
    kappa = kappa_of(ext_mode)
    if not brief:
        print(f'kappa (S-power of softest mode) = {kappa}')

    # optional layer compression: only usable C_i^n layers
    # kept (dead layers below n0 / anchorless S^m are dropped — 2026-08-12
    # usable_modes replaces the old over-approximating mode_analysis closure)
    comp_layers = None
    if use_compression:
        from usable_modes import derive_usable_modes, usable_layers
        usable = derive_usable_modes(ext_mode, kappa)
        comp_layers = usable_layers(ext_mode, kappa, usable)
        if not brief:
            print('mode analysis: usable layers per external:')
            for name, lv in comp_layers.items():
                print(f'  {name}: C_i^n with n in {lv}')

    # cuts per external
    ext_cuts = {}
    for name in ext_mode:
        md = ext_mode[name]
        if md == H:
            continue
        layers = comp_layers.get(name) if comp_layers else None
        ext_cuts[name] = cuts_for_external(name, ext_attach[name], md, kappa,
                                           layers)
    if not brief:
        for name, cs in ext_cuts.items():
            print(f'  {name} (mode {mode_str(ext_mode[name])}): cuts '
                  f'{[mode_str(c.mode) for c in cs]}')
    if not ext_cuts:
        # all externals hard (H): no cut structure -> no region (same as the
        # former C/H constructor's early return; layer 0 now owns C/H too, so
        # this guard must stay explicit — 2026-09-09 merge).
        return []

    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    edges_t = [tuple(sorted(e, key=str)) for e in edges]

    # per-external allowed sets (needed for lower-bound pruning)
    allowed_by_ext = {}
    # chains per external
    chains_by_ext = {}
    for name, cs in ext_cuts.items():
        if not cs:
            # SC/S external softer than S^kappa: no usable cuts — it still
            # attaches as an external momentum (ext_attach), confirmed via
            # cond 1, but contributes no cut chain (2026-08-13).
            continue
        innermost = cs[-1]
        # per-layer allowed sets (2026-08-11): an outer/harder cut may contain
        # vertices with softer externals that the innermost cut cannot; see
        # nested_chains docstring.
        allowed_list = [cut_allowed_vertices(verts, edges, ext_attach, ext_mode,
                                             name, c.mode) for c in cs]
        t0 = time.time()
        # UNION of all per-layer allowed sets: max_new_shareable needs a safe
        # UPPER bound, and an outer/harder cut may contain vertices that the
        # innermost cut cannot (e.g. MTest1 v3=S: the p2 outer cut C_2 covers
        # v3 while the innermost C_2^3 cannot — S and C_2^3 overlap).  The
        # old innermost-only bound under-estimated and wrongly pruned
        # k=1 layers (2026-08-12).
        allowed_by_ext[name] = set().union(*allowed_list)
        chains_by_ext[name] = nested_chains(verts, edges, innermost.root,
                                            len(cs), allowed_list)
        if not brief:
            print(f'  {name}: {len(chains_by_ext[name])} nested chains '
                  f'({time.time() - t0:.1f}s)')
        else:
            print(f'  {name}: {len(chains_by_ext[name])} nested chains '
                  f'({time.time() - t0:.1f}s)', flush=True)

    names = list(chains_by_ext.keys())
    ext_chains = [chains_by_ext[n] for n in names]
    ext_cut_lists = [ext_cuts[n] for n in names]
    g = Graph(verts, edges_t, ext_attach)
    if maxk is None:
        # Layer upper bound: shared vertices are a subset of vertices NOT
        # attached by any LARGE-component external momentum (H and C^n, i.e.
        # mode with m == 0). SC/S externals (m > 0) are shareable, so their
        # attachment vertices count toward V.  max_k = V (NOT V-1: all
        # shareable vertices can be shared — UserGraph k=6 region, where the
        # whole S blob {v4,v6..v10} incl. the l1 vertex is shared; the old
        # V-1 silently dropped it, layered 198 vs 199, 2026-08-11).
        V = sum(1 for v in verts
                if not any(vv == v and ext_mode[nm][0] == 0
                           for nm, vv in ext_attach.items()))
        maxk = V

    # constraint (2026-08-09): the cuts around p_i must NOT include
    # vertices attached to other externals p_j (j != i), else momentum
    # conservation is violated at that vertex.  excluded[i] = attachment
    # vertices of all other externals.
    excluded = {}
    if no_ext_attach:
        for i, nm in enumerate(names):
            excluded[i] = {ext_attach[n2] for n2 in ext_attach if n2 != nm}

    def check_combo(combo):
        """Step1 + FC + IR on a finished cut assignment; returns (vm, em) or None."""
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
        # Massive prescription (2026-08-15 : big-massive lines forced H.
        if not rc.massive_h_ok(edges_t, em, massive):
            return None
        if not momentum_ok(g, em, ext_mode):
            return None
        # Fundamental pattern: each jet connected (before FC/IR compat;
        # 2026-08-11 — same check as region_checker).
        from region_checker import jet_connected_ok
        if not jet_connected_ok(vm, em, edges_t):
            return None
        # IR compatibility (2026-08-21 Regge-style rewrite): mode
        # components are 1VI blocks (mode_components_wa) — no separate
        # contracted-1VI filter; external momenta may propagate along
        # their own mode (2026-08-21).
        if not check_fc(g, em, ext_mode):
            return None
        # Mojetic (Theorem 3 cond1): H∪J∖J_i must be 1VI after joining
        # attached externals to an aux vertex (component-level momentum
        # conservation). Same check as region_checker
        # — MUST be here too (layered path; missing it allowed 2 extra
        # regions on DoubleMothT k2, 2026-08-10).
        from region_checker import cond1_ok
        ok_mj, _ = cond1_ok(edges_t, em, ext_attach, ext_mode)
        if not ok_mj:
            return None
        if not ir_ok_blocks(g, em, ext_mode):
            if os.environ.get('FRI_DEBUG_IR'):
                print('IR-REJECT', [tuple(md if md is not None else (0,0,0)) for md in em], flush=True)
            return None
        return vm, em

    def max_new_shareable(cur_i, cover):
        """Upper bound on how many MORE shared vertices the remaining
        externals (indices >= cur_i) can create:
          - single-covered vertex v: needs 1 more remaining external whose
            allowed set contains v;
          - uncovered vertex v: needs 2 remaining externals whose allowed
            sets contain v (pairwise intersection).
        Over-counting is safe (prune only when even the upper bound fails)."""
        rem_allowed = [allowed_by_ext[names[j]]
                       for j in range(cur_i, len(names))]
        if not rem_allowed:
            return 0
        union = set().union(*rem_allowed)
        pair_union = set()
        for a in range(len(rem_allowed)):
            for b in range(a + 1, len(rem_allowed)):
                pair_union |= rem_allowed[a] & rem_allowed[b]
        n = 0
        for v in verts:
            b = cover.get(v, 0)
            if b.bit_count() >= 2:
                continue  # already shared
            if b.bit_count() == 1:
                if v in union:
                    n += 1
            else:
                if v in pair_union:
                    n += 1
        return n

    def walk(i, assign, cover, target_k, stats, count_only=False):
        """DFS over externals. cover[v] = bitmask of externals whose cuts
        cover v (nested cuts of one external OR the same bit)."""
        stats['nodes'] += 1
        cur_shared = sum(1 for b in cover.values() if b.bit_count() >= 2)
        if cur_shared + max_new_shareable(i, cover) < target_k:
            return  # can never reach exactly target_k shared vertices
        if target_k >= 2:
            # shared-vertex prefilter: the set of currently-shared
            # vertices must be a subset of SOME surviving removal set, else
            # no valid region can complete from here.
            shared_now = frozenset(v for v, b in cover.items()
                                   if b.bit_count() >= 2)
            if not any(shared_now <= S for S in prefilter[target_k]):
                return
        if i == len(names):
            cur_shared = sum(1 for b in cover.values() if b.bit_count() >= 2)
            if cur_shared != target_k:
                return
            stats['kept'] += 1
            if count_only:
                return  # lightweight pre-count: same pruning, no combo check
            r = check_combo(assign)
            if r is None:
                stats['step1_fail'] += 1
                return
            stats['step1'] += 1
            vm, em = r
            key = frozenset((v, vm[v]) for v in verts)
            if key in global_seen:
                stats.setdefault('dup', 0)
                stats['dup'] += 1
                return   # already reported at an earlier layer / combo
            global_seen.add(key)
            stats['regions'].append((vm, em))
            if on_region is not None:
                on_region(target_k, vm, em)
            return
        name = names[i]
        cs = ext_cut_lists[i]
        bit = 1 << i
        for chain in ext_chains[i]:
            if excluded and any(S & excluded[i] for S in chain if S):
                continue
            # incremental cover update for this chain
            touched = []
            appended = []
            ok = True
            for cut, S in zip(cs, chain):
                if not S:
                    continue
                for v in S:
                    old = cover.get(v, 0)
                    cover[v] = old | bit
                    touched.append((v, old))
                # layer bound: shared vertices cannot exceed target_k
                if sum(1 for b in cover.values() if b.bit_count() >= 2) > target_k:
                    ok = False
                    break
                assign.append((cut, S))
                appended.append((cut, S))
            if ok:
                walk(i + 1, assign, cover, target_k, stats, count_only)
            # backtrack: restore cover (LIFO) and drop appended cuts
            for v, old in reversed(touched):
                if old == 0:
                    del cover[v]
                else:
                    cover[v] = old
            if appended:
                del assign[len(assign) - len(appended):]
        return

    # ---- shared-vertex prefilter (2026-08-10) ----
    # V_verts = vertices NOT attached by any large-component external
    # (m==0: H and C^n).  SC/S externals (m>0) are shareable, so their
    # attachment vertices count toward V_verts.
    V_verts = [v for v in verts
               if not any(vv == v and ext_mode[nm][0] == 0
                          for nm, vv in ext_attach.items())]
    from shared_prefilter import shared_sets_for_k
    global_seen = set()   # cross-layer dedup: region keys already reported
    prefilter = {}   # k -> list of surviving shared-vertex sets (k >= 2)
    layers = []
    for target_k in range(0, maxk + 1):
        if target_k >= 2:
            cand = shared_sets_for_k(verts, adj, V_verts, target_k)
            prefilter[target_k] = cand
            if not cand:
                # removing k vertices already disconnects the graph, so
                # removing k+1, k+2, ... cannot reconnect it: all deeper
                # layers are provably empty — terminate.
                print(f'  layer k={target_k}: no connected removal set — '
                      f'terminating (deeper layers provably empty)', flush=True)
                break
        t0 = time.time()
        # 2026-08-13 : show the (exact) cut-config count BEFORE the layer
        # runs — a lightweight count-only DFS with identical pruning gives it
        # up front, then the real walk appends scaleful/new-regions on the
        # same line.  (+3-5% total time, ~proportional-to-time progress info.)
        pre_stats = {'nodes': 0, 'kept': 0}
        walk(0, [], {}, target_k, pre_stats, count_only=True)
        print(f'  layer k={target_k}: cut configs {pre_stats["kept"]:>8} ',
              end='', flush=True)
        stats = {'nodes': 0, 'kept': 0, 'step1_fail': 0, 'step1': 0,
                 'regions': []}
        walk(0, [], {}, target_k, stats)
        # dedup (same region reachable via several chain combos)
        seen = set()
        uniq = []
        for vm, em in stats['regions']:
            key = frozenset((v, vm[v]) for v in verts)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((vm, em))
        layers.append({'k': target_k, 'nodes': stats['nodes'],
                       'kept': stats['kept'],
                       'step1': stats['step1'],
                       'dup': stats.get('dup', 0),
                       'ir': len(uniq), 'secs': time.time() - t0,
                       'keys': [frozenset((v, vm[v]) for v in verts)
                                for vm, em in uniq],
                       'regions': list(uniq)})
        print(f'| scaleful {stats["step1"]:>5} | new regions {len(uniq):>4} '
              f'({time.time() - t0:.1f}s)', flush=True)
        # if this layer is empty we still continue: conjecture checked at the end
    return layers


def all_regions(verts, edges, ext_attach, ext_mode, use_compression=False,
                massive=frozenset()):
    """All regions of the graph (C/H family included in layer 0), as a flat
    list of (vm, em).  Replaces the former construct_regions(allow_overlap=
    False) + run_layered pair (2026-09-09 merge)."""
    layers = run_layered(verts, edges, ext_attach, ext_mode, brief=True,
                         use_compression=use_compression, massive=massive)
    return [r for L in layers for r in L['regions']]


# ---------------- main ----------------
def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    brief = '--brief' in sys.argv
    no_ext_attach = '--no-ext-attach' in sys.argv
    maxk = None
    for a in sys.argv:
        if a.startswith('--maxk='):
            maxk = int(a.split('=')[1])
    if not args:
        print(__doc__)
        return
    case = args[0]
    if case == '4pt3loop':
        verts, edges, ext_attach, ext_mode = case_4pt3loop()
        label = '4pt3loop (7.1 example)'
    elif case == 'hypercrown':
        k = args[1] if len(args) > 1 else 'k2'
        verts, edges, ext_attach, ext_mode = case_hypercrown(k)
        label = f'HyperCrown {k}'
    else:
        raise SystemExit(f'unknown case {case}')

    print(f'=== {label} ===')
    print(f'vertices {len(verts)}, edges {len(edges)}, externals {list(ext_attach)}')
    t0 = time.time()
    layers = run_layered(verts, edges, ext_attach, ext_mode, maxk, brief,
                         no_ext_attach)
    total = sum(L['ir'] for L in layers)
    print(f'\n=== layer table ===')
    print(f'{"k":>2} | {"cut configs":>14} | {"scaleful":>6} | {"new regions":>7} | {"time":>6}')
    for L in layers:
        print(f'{L["k"]:>2} | {L["kept"]:>14} | {L["step1"]:>6} | {L["ir"]:>7} | {L["secs"]:>5.1f}s')
    print(f'\ntotal regions across layers: {total}  ({time.time() - t0:.1f}s)')

    # truncation conjecture: first empty layer; are all deeper layers empty?
    nonempty = [L['k'] for L in layers if L['ir'] > 0]
    if nonempty:
        first_empty = None
        for L in layers:
            if L['ir'] == 0 and L['k'] > 0:
                first_empty = L['k']
                break
        deeper_all_empty = True
        if first_empty is not None:
            for L in layers:
                if L['k'] > first_empty and L['ir'] > 0:
                    deeper_all_empty = False
                    break
        else:
            deeper_all_empty = True  # no empty layer at all -> vacuous
        if first_empty is None:
            verdict = 'NO EMPTY LAYER (conjecture vacuous on this instance)'
        elif deeper_all_empty:
            verdict = (f'CONJECTURE HOLDS on this instance: first empty layer '
                       f'k={first_empty}, all deeper layers empty')
        else:
            verdict = (f'CONJECTURE FAILS on this instance: layer k={first_empty} '
                       f'empty but deeper layers nonempty')
        print(f'truncation: {verdict}')
        print(f'nonempty layers: {nonempty}')
    else:
        print('truncation: NO regions at all in any layer?!')

if __name__ == '__main__':
    main()
