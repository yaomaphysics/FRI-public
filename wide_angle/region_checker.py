"""
This script constructs the "judgment module" (one of the two cores of the main code), which filters regions from those cut-overlaying results.
The filter is built on the following subgraph requirements (§4 and §5 of 2601.22144):
    (A) fundamental pattern (momentum conservation, Coleman--Norton interpretation),
    (B) connectivity requirement (First Connectivity Theorem),
    (C) infrared-compatibility requirement.
These requirements as a whole, can be seen as the necessary and sufficient condition for facet region. See §6 of 2601.22144 for rigorous proofs.

To realize them, the following notions are developed:
    mojetic graph -- Definition: a graph is mojetic if it is 1VI after contracting all its external momenta to an auxiliary vertex.
                     Requiring certain subgraphs to be mojetic helps eliminate those configurations with pathological tadpoles.
    mode component -- fundamental building blocks of a region
    messenger -- certain subgraphs which help "deliver the infrared message".
    infrared compatibility -- this notion is defined recursively to check the "delivery of correct infrared scaling" to each mode component of a region.
                              For each region, it is required that all its mode components are infrared compatible (the infrared-compatibility requirement).
For more details, it is recommended to read the relevant sections of 2601.22144.

The pipeline checks used by the enumerators (check_fc / momentum_ok / ir_ok_blocks) live here next to the conditions they implement;
the shared base layer (mode algebra, graph and component machinery) is in primitives.py.
"""
import os, re
from collections import defaultdict, deque
from itertools import combinations
from primitives import (V, eq, harder_or_eq, join, meet, _join, marginal_softer, parse_mode, find_1vi_blocks, vee, vertex_mode, H)

INF = 100   # a large value representing the positive infinity.


# ================== basic graph & mode constructions ==================

# Classify edges into (J, H_edges, S_edges) with J = {i: [C_i^n edges]} (by direction).
def classify(em, edges_in):
    J = {}
    H = []
    S = []
    for e, md in zip(edges_in, em):
        if md is None:
            H.append(e)
            continue
        a, b, c = md
        if a == 0 and b >= 1:
            J.setdefault(c, []).append(e)
        elif a == 0 and b == 0:
            H.append(e)
        else:
            S.append(e)
    return J, H, S


# Connected, with aux_conn vertices glued through the aux vertex.
def _connected_aux(verts, edges, aux_conn=None):
    if not verts:
        return True
    aux_conn = (aux_conn or set()) & set(verts)
    adj = {v: set() for v in verts}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)
    # aux vertex glues all aux_conn vertices together
    if len(aux_conn) > 1:
        ac = list(aux_conn)
        for i in range(1, len(ac)):
            adj[ac[0]].add(ac[i])
            adj[ac[i]].add(ac[0])
    start = next(iter(verts))
    seen = {start}
    st = [start]
    while st:
        x = st.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                st.append(y)
    return len(seen) == len(verts)


# Mode components: decompose each contracted γ̃_X into its 1VI blocks (shared cut vertex ⇒ separate) and expand back to original {'mode','V','E'} elements.
def mode_components_wa(edges_in, em, ext_attach, ext_mode):
    from collections import defaultdict
    inc = defaultdict(list)
    for e, md in zip(edges_in, em):
        m = md if md is not None else (0, 0, 0)
        u, v = e
        inc[u].append(m); inc[v].append(m)
    if ext_mode:
        for name, v in ext_attach.items():
            if name in ext_mode:
                inc[v].append(ext_mode[name])
    vjoin = {v: _join(ms) for v, ms in inc.items()}
    emodes = defaultdict(list)
    for e, md in zip(edges_in, em):
        m = md if md is not None else (0, 0, 0)
        emodes[m].append(e)
    aux = ('aux',)
    comps = []
    for X, ex in emodes.items():
        vx = {v for v, jm in vjoin.items() if eq(jm, X)}
        # No X vertices but X edges existing: each edge is its own component (isolated cross edges whose endpoints have harder join modes).
        if not vx and ex:
            for e in ex:
                comps.append({'mode': X, 'V': set(), 'E': {e}})
            continue
        if not vx:
            continue
        verts2 = set(vx) | {aux}
        aux_verts = set()
        emap = {e: m for e, m in zip(edges_in, em) if m is not None}
        for e2 in edges_in:
            a, b = e2
            a_in, b_in = a in vx, b in vx
            if a_in == b_in:
                continue
            Y = emap.get(e2)
            if Y is None:
                continue
            if eq(Y, X):
                w = b if a_in else a
                aux_verts.add(w)
        edges2 = []
        orig2 = []
        for e2 in edges_in:
            a, b = e2
            Y = emap.get(e2)
            if Y is None:
                continue
            if eq(Y, X):
                # X edges always belong to γ̃_X; adjacent vertices of other (harder) modes are absorbed into aux.
                a2 = a if a in vx else aux
                b2 = b if b in vx else aux
                if a2 is not None and b2 is not None:
                    edges2.append((a2, b2))
                    orig2.append(e2)
        raw = find_1vi_blocks(verts2, edges2)
        used = set()
        for (bv, be) in raw:
            V = {v for v in bv if v != aux}
            E = set()
            for ce in be:
                for j, c2 in enumerate(edges2):
                    if j in used or c2 != ce:
                        continue
                    used.add(j)
                    oe = orig2[j]
                    if isinstance(oe[0], str) and oe[0].startswith('ext:'):
                        continue
                    Y = emap.get(oe)
                    if Y is not None and eq(Y, X):
                        E.add(oe)
                    break
            if not V and not E:
                continue
            comps.append({'mode': X, 'V': V, 'E': E})
    return comps


def elements(c): return c['V'] | c['E']


def neighbors(elem, verts, edges):
    if elem in verts:
        return {(a, b) for (a, b, md) in edges if elem in (a, b)}
    return {elem[0], elem[1]}


def adjacent(g1, g2):
    for e in g1['E']:
        for v in g2['V']:
            if v in e: return True
    for e in g2['E']:
        for v in g1['V']:
            if v in e: return True
    return False


# True iff g1 is relevant to g2 — thin wrapper over relevant_hit (all_comps kept for call-site compatibility).
def relevant(g1, g2, verts, edges, all_comps):
    return relevant_hit(g1, g2, verts, edges) is not None


# First element of g2 reached by g1's monotone flow (the cond-1 third-port entry, 2026-09-05); None if g1 is not relevant to g2.
def relevant_hit(g1, g2, verts, edges):
    if not marginal_softer(g1['mode'], g2['mode']):
        return None
    emode = {v: md for v, md in verts.items()}
    for (a, b, md) in edges:
        emode[(a, b)] = md; emode[(b, a)] = md
    start = elements(g1)
    target = elements(g2)
    visited = set(start)
    dq = deque((e, g1['mode']) for e in start)
    while dq:
        e, cur = dq.popleft()
        if e in target:
            return e
        for nb in neighbors(e, verts, edges):
            if nb in visited: continue
            nm = emode[nb]
            # Steps must be monotone non-softer ("softer-or-equal" per element pair); OVERLAPPING steps are forbidden.
            if harder_or_eq(nm, cur):
                visited.add(nb); dq.append((nb, nm))
    return None


# ======================== fundamental pattern =========================

# Momentum conservation at each vertex (§5.1): incident momenta can be splitted into two nonempty sets with equal ∨ee values (necessary, not sufficient).
def momentum_ok(g, edge_modes, EXTMODE):
    for v in g.vertices:
        inc = []
        for ei in g.incident.get(v, []):
            if edge_modes[ei] is not None: inc.append(edge_modes[ei])
        for name, vv in g.ext.items():
            if vv == v: inc.append(EXTMODE[name])
        n = len(inc)
        if n < 2:
            if n == 1 and inc and inc[0] != H:
                return False
            continue
        if n == 2:
            if not eq(inc[0], inc[1]): return False
            continue
        ok = False
        for r in range(1, n):
            for A in combinations(range(n), r):
                B = [i for i in range(n) if i not in A]
                if not B: continue
                if eq(vee([inc[i] for i in A]), vee([inc[i] for i in B])):
                    ok = True; break
            if ok: break
        if not ok: return False
    return True


# Fundamental pattern: each jet J_i (union of all C_i^n subgraphs, n >= 1) must be connected (if nonempty).
def jet_connected_ok(vm, em, edges_in):
    dirs = set()
    for v, md in vm.items():
        m, n, i = md
        if m == 0 and n >= 1 and i != 0:
            dirs.add(i)
    for md in em:
        m, n, i = md
        if m == 0 and n >= 1 and i != 0:
            dirs.add(i)
    for i in dirs:
        V = {v for v, md in vm.items() if md[0] == 0 and md[1] >= 1 and md[2] == i}
        if not V:
            continue
        adj = {v: set() for v in V}
        for (a, b), md in zip(edges_in, em):
            if md[0] == 0 and md[1] >= 1 and md[2] == i and a in V and b in V:
                adj[a].add(b)
                adj[b].add(a)
        seen = {next(iter(V))}
        st = list(seen)
        while st:
            x = st.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    st.append(y)
        if len(seen) != len(V):
            return False
    return True


# Mojetic: 1VI after joining all external momenta to an auxiliary vertex.
def mojetic(vertset, edges_, aux_conn):
    if not _connected_aux(vertset, edges_, aux_conn):
        return False
    for v in list(vertset):
        vs = vertset - {v}
        es = [(u, w) for (u, w) in edges_ if u != v and w != v]
        ac = aux_conn - {v}
        if not _connected_aux(vs, es, ac):
            return False
    return True


# Mojetic check (Theorem 3 ① of 2211.14845): for each external direction i, H∪J∖J_i must be mojetic — it becomes 1VI once all externals attached to it are joined to an aux vertex.
# Classification here: J_i = union of all C_i^n edges (n >= 1); H = hard edges (mode (0,0,0)); S = everything else (S^m, S^mC_i^n).
# Physical meaning: mojetic failure ⟺ some H component sees large momentum flow from at most one jet direction (which violates either momentum conservation or the Coleman--Norton interpretation).
def hard_jet_mojetic_ok(edges_in, em, ext_attach, ext_mode=None):
    J, H, S = classify(em, edges_in)
    J_all = set(H)
    for es in J.values():
        J_all |= set(es)
    ext_names = list(ext_attach)
    failures = []
    # Note that the index i in H∪J∖J_i runs over ALL external-momentum directions — including directions with J_i EMPTY.
    # (Then ∖J_i removes no edges, and only the external momentum p_i is excluded.)
    if ext_mode is None:
        dirs = sorted({ext_attach[n] for n in ext_names})
    else:
        dirs = sorted({ext_mode[n][2] for n in ext_names if ext_mode[n][0] == 0 and ext_mode[n][2] != 0})
    for i in dirs:
        Ji = set(J.get(i, ()))     # may be empty: remove nothing; only drop p_i
        Jm = J_all - Ji
        es = list(Jm)
        vs = set()
        for u, w in es:
            vs.add(u)
            vs.add(w)
        # Soft externals (m >= 1) are never joined to the aux vertex — an O(λ) momentum cannot balance O(1) flows (2026-08-11).
        # Without ext_mode, fall back to joining all externals (old behaviour).
        if ext_mode is None:
            ac = {ext_attach[n] for n in ext_names if ext_attach[n] in vs}
        else:
            ac = {ext_attach[n] for n in ext_names
                  if ext_attach[n] in vs
                  and ext_mode[n][0] == 0          # hard/collinear only
                  and ext_mode[n][2] != i}         # NOT direction i's momentum
        if not mojetic(vs, es, ac):
            failures.append(i)
    return len(failures) == 0, failures


# ============================ connectivity ============================

# First Connectivity (§5.1): ∪_{𝒱≤n} Γ_X must be connected for every threshold n (isolated mode-vertices count as nodes).
def check_fc(g, edge_modes, EXTMODE):
    eVs = [V(m) if m is not None else INF for m in edge_modes]
    vVs = [V(vertex_mode(g, edge_modes, v, EXTMODE)) if vertex_mode(g, edge_modes, v, EXTMODE) else INF for v in g.vertices]
    thresh = sorted({v for v in eVs if v < INF} | {v for v in vVs if v < INF})
    for n in thresh:
        sub = [ei for ei, vv in enumerate(eVs) if vv <= n]
        nodes = set()
        for ei in sub:
            nodes.add(g.vidx[g.edges[ei][0]])
            nodes.add(g.vidx[g.edges[ei][1]])
        for i, vv in enumerate(vVs):
            if vv <= n: nodes.add(i)
        if not nodes: continue
        par = {i: i for i in nodes}
        def find(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: par[ra] = rb
        for ei in sub:
            union(g.vidx[g.edges[ei][0]], g.vidx[g.edges[ei][1]])
        if len({find(i) for i in nodes}) > 1:
            return False
    return True


# ======================= infrared compatibility =======================
# Recursive per-component requirement — "delivery of correct infrared scaling" to every mode component γ, through one of three conditions:
# cond 1 (mode = the external momenta and already infrared-confirmed line momenta entering γ),
# cond 2 (messenger mechanism),
# cond 3 (mode = the wedge of the harder modes which γ is relevant to)
# The whole procedure may require several iterations; region only if all components are confirmed.

# Condition 1: ∨-join of the momenta entering γ — externals attached to it or flowing in via monotone paths, plus confirmed components relevant to γ.
def partial_sum_mode(g, confirmed, verts, edges, all_comps, attach, extmode):
    modes = []
    for name, v in attach.items():
        em = extmode[name]
        # attach point in γ, or X relevant to γ (marginal + path from attach vertex)
        if v in g['V']:
            modes.append(em)
        else:
            if eq(em, g['mode']) or marginal_softer(em, g['mode']):
                # Path from attach vertex to γ through monotonically non-softer elements ("softer than or equal" per element pair, paper's relevance def; overlapping steps forbidden).
                emode = {vv: md for vv, md in verts.items()}
                for (a, b, md) in edges:
                    emode[(a, b)] = md; emode[(b, a)] = md
                start = {v}
                target = elements(g)
                visited = set(start)
                # Start with the ATTACH VERTEX's own mode — the flow first meets the vertex it is attached to.
                dq = deque((e, emode[v]) for e in start)
                while dq:
                    e, cur = dq.popleft()
                    if e in target: modes.append(em); break
                    for nb in neighbors(e, verts, edges):
                        if nb in visited: continue
                        nm = emode[nb]
                        if harder_or_eq(nm, cur):
                            visited.add(nb); dq.append((nb, nm))
    for g2 in confirmed:
        if g2 is g: continue
        if relevant(g2, g, verts, edges, all_comps):
            modes.append(g2['mode'])
    if not modes: return None
    acc = modes[0]
    for m in modes[1:]: acc = join(acc, m)
    return acc

# In Condition 1, to confirm that the momentum flows in (rather than simply touches) the mode component γ, we need to make sure that it has an "inlet" and an "outlet" -- this induces the following.
# Third-port requirement: besides the cond-1 source entry vertices va, vb, γ must touch a vertex w that is either
# (A) a same-mode cut vertex shared with another 𝒳(γ)-component, or (B) a harder vertex reached by γ's own edge; returns ('A'|'B', w) or None.
def third_port(g, entry_vs, all_comps, verts):
    X = g['mode']
    for w in sorted(g['V']):
        if w in entry_vs:
            continue
        mw = verts.get(w)
        if mw is None or not eq(mw, X):
            continue
        for c in all_comps:
            if c is g or not eq(c['mode'], X):
                continue
            if w in c['V']:
                return 'A', w
    for e in sorted(g['E']):
        for w in e:
            if w in g['V'] or w in entry_vs:
                continue
            mw = verts.get(w)
            if mw is not None and not eq(mw, X) and harder_or_eq(mw, X):
                return 'B', w
    return None


# Cond-1 sources: every reachable entry per source is enumerated.
def cond1_sources(g, confirmed, verts, edges, all_comps, attach, extmode):
    modes = []
    for name, v in attach.items():
        em = extmode[name]
        if v in g['V']:
            modes.append((em, {v}))
        else:
            if eq(em, g['mode']) or marginal_softer(em, g['mode']):
                hs = _collect_entries({v}, verts.get(v, (0, 0, 0)), elements(g), verts, edges)
                ev = _entry_set(g, hs)
                if ev:
                    modes.append((em, ev))
    for g2 in confirmed:
        if g2 is g:
            continue
        if not marginal_softer(g2['mode'], g['mode']):
            continue
        hs = _collect_entries(elements(g2), g2['mode'], elements(g), verts, edges)
        ev = _entry_set(g, hs)
        if ev:
            modes.append((g2['mode'], ev))
    return modes


# All target elements reachable from start_elems by monotone non-softer flow (BFS, no early break).
def _collect_entries(start_elems, start_mode, target, verts, edges):
    emode = {v: md for v, md in verts.items()}
    for (a, b, md) in edges:
        emode[(a, b)] = md; emode[(b, a)] = md
    visited = set(start_elems)
    hits = set()
    dq = deque((e, start_mode) for e in start_elems)
    while dq:
        e, cur = dq.popleft()
        if e in target:
            hits.add(e)
            continue          # do not propagate inside target: entry = first contact only
        for nb in neighbors(e, verts, edges):
            if nb in visited:
                continue
            nm = emode[nb]
            if harder_or_eq(nm, cur):
                visited.add(nb); dq.append((nb, nm))
    return hits


# Vertices of γ at which a flow with these hit elements enters.
def _entry_set(g, hit_elems):
    vs = set()
    for h in hit_elems:
        if h in g['V']:
            vs.add(h)
        elif isinstance(h, tuple):
            for w in h:
                if w in g['V']:
                    vs.add(w)
    return vs


# Condition 1 (entry-choice enumeration): each source may enter γ at any reachable vertex (so va may equal vb; the third port may then sit at the bridge's far end).
def cond1_confirms(g, confirmed, verts, edges, all_comps, attach, extmode):
    if g['mode'] == (0, 0, 0):
        ps = partial_sum_mode(g, confirmed, verts, edges, all_comps, attach, extmode)
        return ps is not None and eq(ps, g['mode'])
    srcs = cond1_sources(g, confirmed, verts, edges, all_comps, attach, extmode)
    for md, cands in srcs:
        if eq(md, g['mode']):
            for a in cands:
                if third_port(g, {a}, all_comps, verts):
                    return True
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            md1, c1 = srcs[i]
            md2, c2 = srcs[j]
            if not eq(join(md1, md2), g['mode']):
                continue
            for a in c1:
                for b in c2:
                    if third_port(g, {a, b}, all_comps, verts):
                        return True
    return False


# IR-compat cond 2 (messenger, v16): for mode(g) = S^m, build the connected closure Γ^[m] of S^m C_i^n components around g.
# Require >=3 targets in pairwise-distinct directions, each matched by a relevant Γ member (n_i = 0: kernel; n_i >= 1: S^m C_i^{n_i} member).
# Returns (Γ^[m] member list, special, kernel_blocks) or None.
def find_messenger(g, all_comps, verts, edges, attach, extmode, confirmed=(), kernel=None, kernel_blocks=None):
    m = g['mode'][0]
    if g['mode'][1] != 0 or g['mode'][2] != 0: return None
    # Default kernel: the single block g (2026-08-14 connected-kernel semantics); a passed kernel is the whole connected S^m cloud (rule 1, 2026-09-04).
    if kernel is None:
        kernel = {'mode': g['mode'], 'V': set(g['V']), 'E': set(g['E'])}
        kernel_blocks = [g]
    cands = [c for c in all_comps if c['mode'][0] == m and c['mode'][1] != INF]
    ids = {id(c): c for c in cands}
    memb = {id(c): False for c in cands}
    for kb in kernel_blocks:
        memb[id(kb)] = True
    changed = True
    while changed:
        changed = False
        for c in cands:
            if memb[id(c)]: continue
            if any(memb[id(d)] and adjacent(c, d) for d in cands):
                memb[id(c)] = True; changed = True
    G = [c for c in cands if memb[id(c)]]
    # Connected-kernel semantics (2026-08-14): a degree-m messenger's kernel is the CONNECTED S^m component being checked (g) — always connected by construction.
    # The 2026-08-13 "n_s > 1 -> FAIL" rule was too coarse: a Γ^[m] closure may legitimately contain a second S^m component bridged via a shared confirmed SC member.
    # (K33P02 k4: S{7,8} and S{6,9} both touch the confirmed SC4 component — each an independent valid degree-1 messenger.)
    # What must be prevented instead is BORROWING: the other S component's external must not feed this kernel (CheesePizza k2: l1 sat on the OTHER S component). So:
    #   - special (a): the feeding external must sit on the kernel's own vertices or on a Γ member ADJACENT to the kernel;
    #   - n_i = 0 kernel matches use ONLY g (not any other S^m in Γ).
    n_s = sum(1 for c in G if c['mode'][0] == m and c['mode'][1] == 0 and c['mode'][2] == 0)
    if n_s > 1:
        # Still proceed: kernel g is connected; other S^m members are not part of g's verdict (guarded by the two rules below).
        pass
    gids = {id(c) for c in G}
    # A target γ_i = S^{m-m_i} C_i^{m_i+n_i} (m_i = m - m', n_i = n' - m_i) is counted iff the Γ member S^m C_i^{n_i} is relevant to it.
    # For n_i = 0 that member is the KERNEL S^m (mode (m,0,0)); for n_i >= 1 it is an S^m C_i^{n_i} component of the messenger.
    # (v16: no "at least one n_i = 0" requirement — the author dropped it; verified harmless on 55=55, 81/81, 118/118.)
    by_mode = {}
    for c in G:
        by_mode.setdefault(c['mode'], []).append(c)
    R = set()
    for d in all_comps:
        if id(d) in gids:
            continue
        md = d['mode']
        mi = m - md[0]                      # m_i in 1..m
        if not (1 <= mi <= m):
            continue
        n_i = md[1] - mi                    # n_i >= 0
        if n_i < 0:
            continue
        need = (m, 0, 0) if n_i == 0 else (m, n_i, md[2])
        pool = by_mode.get(need, ())
        if need == (m, 0, 0):
            # Kernel match: ONLY the checked connected kernel — the whole S^m cloud (2026-09-04 rule 1) — may serve as the (m,0,0) member for this target.
            # Another S^m component in the closure is NOT part of the messenger (2026-08-14 connected-kernel semantics).
            pool = [kernel] if kernel['mode'] == (m, 0, 0) else []
        for c in pool:
            # Connected kernel: the matching Γ member must be ADJACENT to the kernel — a leg on the far side of a bridged S^m component is not part of g's messenger.
            # (K33P02 k4: leaf S{9} borrowed SC1(1,7)/SC3(3,8) from S{7,8} through the SC4 bridge — its own direction is only SC4, <3 → reject.)
            if adjacent(c, kernel) and relevant(c, d, verts, edges, all_comps):
                R.add(id(d))
                break
    # ---- type-2 / m_i=0 special target (2026-08-12; confirmed, always on) ----
    # The kernel S^m itself can be a target when it carries an attached external of mode S^m C_i^{n'} (same m, n' >= 1, i != 0).
    # The S^m part of that external IS the kernel's own scale source — no relevance to confirmed subgraphs is needed.
    # (CrownSS k5 region: S kernel {v5} fed by l1 = SC5^inf; targets C1^2 / C2^3 plus the kernel itself, dir 5.)
    special_dir = None
    # (a) Attached external of mode S^m C_i^{n'} feeding the messenger; it must sit on the kernel's OWN vertices or on a Γ member ADJACENT to the kernel (2026-08-14).
    # An external on the OTHER (bridged) S^m component must NOT feed this kernel (CheesePizza k2).
    # In K33P02 k4, l1 sits on the confirmed SC4 component adjacent to the kernel via line (4,7) — allowed.
    # The SC4 line momentum itself is the source (branch (b)); the external sitting there is just its avatar.
    kernel_verts = set(kernel['V'])
    for c in G:
        if adjacent(c, kernel):
            kernel_verts |= set(c['V'])
    for nm, v in attach.items():
        if v in kernel_verts:
            e = extmode[nm]
            if e[0] == m and e[1] >= 1 and e[2] != 0:
                special_dir = e[2]
                break
    # (b) A CONFIRMED component of mode S^m C_i^{n'} (n' >= 1, i != 0) touching the messenger feeds it exactly like an attached external (2026-08-12).
    # Any line momentum of a confirmed component X has the status of an X-mode external; Γ members are NOT excluded.
    if special_dir is None:
        for d in confirmed:
            md = d['mode']
            if md[0] == m and md[1] >= 1 and md[2] != 0 and adjacent(d, kernel):
                special_dir = md[2]
                break
    ntargets = len(R) + (1 if special_dir is not None else 0)
    if ntargets < 3: return None
    # The messenger must be relevant to >=3 subgraphs from DIFFERENT directions (definition precedes IR compatibility; count distinct i).
    dirs = set()
    for rid in R:
        d = next(x for x in all_comps if id(x) == rid)
        if d['mode'][2] != 0:
            dirs.add(d['mode'][2])
    if special_dir is not None:
        dirs.add(special_dir)
    if len(dirs) < 3: return None
    # Check each γ_i: mode = S^{m-m_i} C_i^{m_i+n_i}, m_i in 1..m, n_i >= 0; every external entering γ_i must be compatible.
    for rid in R:
        d = next(x for x in all_comps if id(x) == rid)
        md = d['mode']
        mi = m - md[0]
        if not (1 <= mi <= m): return None
        if md[1] < mi: return None
        ext = [extmode[nm] for nm, v in attach.items() if v in d['V']]
        for e in ext:
            # Form-compatible external: S^{m-m_i} C_i^{m_i+n_i+n'_i} (same direction, same m, at least as deep).
            if e[0] == md[0] and e[2] == md[2] and e[1] >= md[1]:
                continue
            # Absorbable softer external (e.g. an SC5^inf soft emission at a C2 component: C2^2 ∨ SC5^inf = C2) is swallowed by the target's own mode and must not break the verdict.
            # (2026-08-11: CrownSS k2 region — S messenger relevant to C2/C3/C4; the old check killed it on the l1 direction mismatch.)
            if harder_or_eq(md, e):
                continue
            return None
    return G, special_dir is not None, kernel_blocks


# IR compatibility, unified core (2026-09-26): one shared fixpoint implementation replaces the former inline
# copy in ir_ok_blocks and the old simpler check_conditions.  Consumers: skeleton (ir_ok_blocks), scaleless_diagnosis.

# S^mC^n tadpole (2026-09-03, case #66 k4): a soft-family component (m >= 1) adjacent to NO harder-mode component is already a 1VI block of its own mode subgraph.
# It can only be confirmed by cond 1 (partial sum) — unless it is a pure S^m block carrying two momenta whose vee is exactly its mode (rule 3, _vee_ok).
# In that case cond 2/3 are allowed again.
# Hard (m=0) and jet modes are exempt.
def _is_tadpole(comp, verts, all_comps):
    if comp['mode'][0] < 1:
        return False
    # Adjacency to a HARD vertex (vm = H) counts — the hard subgraph includes isolated hard vertices with no H edge.
    # (MTest1 k0: S{3}+(3,4) touches v4 = H via the q1 external — NOT a tadpole; 2026-09-03. Component elements: its vertices plus the endpoints of its edges.)
    for v in comp['V']:
        if verts.get(v) is not None and eq(verts[v], (0, 0, 0)):
            return False
    for (u, w) in comp['E']:
        for v in (u, w):
            if verts.get(v) is not None and eq(verts[v], (0, 0, 0)):
                return False
    for other in all_comps:
        if other is comp:
            continue
        if V(other['mode']) < V(comp['mode']) and adjacent(comp, other):
            return False
    return True

# 2026-09-04 (rules 1+2+3): the kernel of a messenger Γ^[m] is the whole CONNECTED component of the pure-soft S^m subgraph (rule 1).
# Several 1VI blocks joined through shared REAL S^m vertices count as ONE kernel; blocks touching only through non-S^m (aux) vertices do not.
# A messenger confirmation covers every confirmable S^m block of that kernel cloud (rule 2).
# A pure-soft S^m tadpole may use cond 2/3 iff two momenta attached to it have vee exactly equal to its mode (rule 3).
# The 09-03 blanket ban stays for SC (n>=1) tadpoles; cond 1 stays open to everyone.
def _pure_soft(comp):
    md = comp['mode']
    return md[0] >= 1 and md[1] == 0

# Precompute tadpole flags, kernel clouds and the vee cache for one fixpoint call.
def _ir_prepare(all_comps, verts, edges, attach, extmode):
    tadpole = {id(c): _is_tadpole(c, verts, all_comps) for c in all_comps}
    # Kernel clouds: blocks of the same pure-soft mode X are one cloud iff transitively connected through shared real X-mode vertices.
    cloud_of = {}
    pure_by_mode = {}
    for comp in all_comps:
        if _pure_soft(comp):
            pure_by_mode.setdefault(comp['mode'], []).append(comp)
    for X, blocks in pure_by_mode.items():
        if len(blocks) == 1:
            cloud_of[id(blocks[0])] = blocks
            continue
        par = {id(b): id(b) for b in blocks}
        def find(i):
            while par[i] != i:
                par[i] = par[par[i]]
                i = par[i]
            return i
        vowner = {}
        for b in blocks:
            for v in b['V']:
                if v in vowner:
                    ra, rb = find(id(b)), find(vowner[v])
                    if ra != rb:
                        par[ra] = rb
                else:
                    vowner[v] = id(b)
        roots = {}
        for b in blocks:
            roots.setdefault(find(id(b)), []).append(b)
        for lst in roots.values():
            for b in lst:
                cloud_of[id(b)] = lst
    return {'tadpole': tadpole, 'cloud_of': cloud_of, 'vee_cache': {},
            'edges': edges, 'attach': attach, 'extmode': extmode}

# Rule 3 (2026-09-04): a tadpole X must carry two momenta k1, k2 attached to it — line momenta with a collinear part (n >= 1) at the component, or externals.
# Their vee must be exactly X.
# Pure-soft lines (n = 0) are excluded as scale sources.
def _vee_ok(comp, edges, attach, extmode):
    vs = set(comp['V'])
    for (u, w) in comp['E']:
        vs.add(u); vs.add(w)
    pool = []
    for (u, w, md) in edges:
        if md[1] < 1:
            continue
        if u in vs or w in vs:
            pool.append(md)
    for nm, v in attach.items():
        if v in vs:
            pool.append(extmode[nm])
    if len(pool) < 2:
        return False
    for a in range(len(pool)):
        for b in range(a + 1, len(pool)):
            if eq(join(pool[a], pool[b]), comp['mode']):
                return True
    return False

# Cond 2/3 allowance: non-tadpoles always; pure-soft tadpoles only via rule 3 (cached); SC (n>=1) tadpoles never.
def _cond_allowed(comp, state):
    if not state['tadpole'].get(id(comp)):
        return True
    if not _pure_soft(comp):
        return False        # SC tadpoles keep the 09-03 cond 2/3 ban
    ok = state['vee_cache'].get(id(comp))
    if ok is None:
        ok = _vee_ok(comp, state['edges'], state['attach'], state['extmode'])
        state['vee_cache'][id(comp)] = ok
    return ok

# Merged kernel dict {'mode','V','E'} of comp's cloud (rule 1); (None, None) when the cloud is the single block itself.
def _cloud_kernel(comp, state):
    blocks = state['cloud_of'].get(id(comp))
    if blocks is None or len(blocks) == 1:
        return None, None
    kern = {'mode': comp['mode'], 'V': set(), 'E': set()}
    for b in blocks:
        kern['V'] |= set(b['V'])
        kern['E'] |= set(b['E'])
    return kern, tuple(blocks)

# Three recursive IR-compat conditions for one component: cond 1 (partial-sum ∨ via cond1_confirms, entry-choice; 2026-09-05/07),
# cond 3 (meet of two relevant confirmed), cond 2 (messenger, cloud kernel); through the tadpole allowance (2026-09-03/04).
# Returns (condition number 1|2|3, covered blocks) or (None, ()); covered blocks carry rule 2 for the caller's fixpoint.
def check_conditions(g, confirmed, verts, edges, all_comps, attach, extmode, state=None):
    if state is None:
        state = _ir_prepare(all_comps, verts, edges, attach, extmode)
    if cond1_confirms(g, confirmed, verts, edges, all_comps, attach, extmode):
        return 1, ()
    if not _cond_allowed(g, state):
        return None, ()
    for g1, g2 in combinations(confirmed, 2):
        if relevant(g, g1, verts, edges, all_comps) and relevant(g, g2, verts, edges, all_comps):
            if eq(meet(g1['mode'], g2['mode']), g['mode']): return 3, ()
    kern, kbs = _cloud_kernel(g, state)
    res = find_messenger(g, all_comps, verts, edges, attach, extmode, confirmed, kernel=kern, kernel_blocks=kbs or None)
    if res is not None:
        G, special, kbs = res
        # Type-2 special (m_i=0): the kernel is fed by its own attached S^m C_i^{n'} external — no relevance to a confirmed subgraph required (2026-08-12).
        if special:
            return 2, kbs
        for g0 in confirmed:
            if relevant(g, g0, verts, edges, all_comps): return 2, kbs
    return None, ()

# Shared fixpoint: confirm components until no change; a cond-2 confirmation also covers the other
# confirmable blocks of its messenger kernel cloud (rule 2).  Returns (ok, order, confirmed, stuck).
def confirm_all(all_comps, verts, edges, attach, extmode):
    state = _ir_prepare(all_comps, verts, edges, attach, extmode)
    confirmed = []
    order = []
    changed = True
    while changed:
        changed = False
        for comp in all_comps:
            if comp in confirmed:
                continue
            cc, kbs = check_conditions(comp, confirmed, verts, edges, all_comps, attach, extmode, state=state)
            if cc is not None:
                confirmed.append(comp); order.append((comp, cc)); changed = True
                if cc == 2:
                    # Rule 2: a messenger confirmation covers every confirmable S^m block of the kernel cloud (vee-failing tadpoles stay out, rule 3).
                    for kb in kbs:
                        if kb is not comp and kb not in confirmed and _cond_allowed(kb, state):
                            confirmed.append(kb); changed = True
    ok = len(confirmed) == len(all_comps)
    stuck = [c for c in all_comps if c not in confirmed]
    return ok, order, confirmed, stuck


# IR compatibility requirement: each mode component confirmed independently by the 3-condition fixpoint.
def ir_ok_blocks(g, edge_modes, EXTMODE):
    verts = {}
    for v in g.vertices:
        md = vertex_mode(g, edge_modes, v, EXTMODE)
        if md is not None:
            verts[v] = md
    edges = [(u, v, md) for (u, v), md in zip(g.edges, edge_modes) if md is not None]
    attach = {name: vv for name, vv in g.ext.items()}
    extmode = {name: EXTMODE[name] for name in g.ext}
    edges_in = [(u, v) for (u, v, _) in edges]
    em = [md for (_, _, md) in edges]
    all_comps = mode_components_wa(edges_in, em, attach, extmode)

    # 2026-09-26: fixpoint extracted to the shared confirm_all (see above; also used by scaleless_diagnosis).
    ok, _order, _confirmed, _stuck = confirm_all(all_comps, verts, edges, attach, extmode)
    return ok


# The tikz-figure entry (parse / infer_orange_comps / is_region) was retired here 2026-09-26;
# archived with usage notes in private/dev_backups/fig_entry_retired_20260926/.
