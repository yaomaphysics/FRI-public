"""region_checker.py — the FRI region checker (judgment core): the subgraph
requirements + IR-compatibility machinery that decides whether a graph
configuration is a facet region.

Given a graph configuration (vertex/edge mode assignment), decide whether it
is a facet region, per the paper's criteria:
  (A) fundamental pattern (momentum conservation, jet connectivity,
      contracted-mode-component 1VI, mojetic),
  (B) First Connectivity,
  (C) IR compatibility (recursive 3-condition fixpoint).

The pipeline checks used by the enumerators (check_fc / momentum_ok / ir_ok /
ir_ok_blocks) live here next to the conditions they implement; the shared
base layer (mode algebra, graph and component machinery) is in primitives.py.
Accepts tikz-figure input (nodes with fill=Col, \\path edge [Col], label
nodes) as well as raw mode arrays.

(2026-09-24 “two cores” split: base parts moved to primitives.py; skeleton.py
(the enumerator) and this module are the two cores.  Formerly region_checker5.py.)
"""
import os, re
from collections import defaultdict, deque
from itertools import combinations
from primitives import (V, norm, eq, harder_or_eq, join, meet, marginal_softer,
                        parse_mode, build_components, biconnected_blocks,
                        vee, vertex_mode, H)

INF = 10**9

# 2026-08-12 TENTATIVE (: type-2 / m_i=0 special messenger target.
# When True, an S^m kernel carrying an attached S^m C_i^{n'} external counts
# as its own target (direction i from the external) and needs no relevance
# to a confirmed subgraph.  Set to False to emulate the pre-special-case
# behaviour for cross-checks (CrownSS k5: 80 vs 81).
ALLOW_SPECIAL_TARGET = True


# ======================= fundamental-pattern checks =======================

def classify(em, edges_in):
    """em: list of (m, n, i) or None (H); edges_in: parallel edge list.
    Returns (J, H_edges, S_edges) with J = {i: [edges...]}."""
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

def _connected(vertset, edges_, aux_conn):
    if not vertset:
        return True
    adj = {v: set() for v in vertset}
    for u, v in edges_:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)
    start = next(iter(vertset))
    seen = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
        if x in aux_conn:
            for y in aux_conn:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
    return len(seen) == len(vertset)

def mojetic(vertset, edges_, aux_conn):
    """1VI after joining all external momenta to an auxiliary vertex."""
    if not _connected(vertset, edges_, aux_conn):
        return False
    for v in list(vertset):
        vs = vertset - {v}
        es = [(u, w) for (u, w) in edges_ if u != v and w != v]
        ac = aux_conn - {v}
        if not _connected(vs, es, ac):
            return False
    return True

def cond1_ok(edges_in, em, ext_attach, ext_mode=None):
    """Check H∪J∖J_i mojetic for every jet direction i.
    edges_in: list of (u,v) internal edges (canonical order = em order).
    em: list of edge mode tuples (m,n,i) or None.
    ext_attach: {external-name: vertex} for ALL externals (p1.., l1, q1...).
    ext_mode: {external-name: mode tuple (m,n,i)}; if given, SOFT external
        momenta (m >= 1, i.e. S^m and S^m C_i^n) are NOT joined to the
        auxiliary vertex — an O(λ) momentum cannot balance O(1) flows, so
        it must not act as the aux rescue for a dangling 1VI component.
        (2026-08-11 : the on-shell paper's mojetic definition only had
        hard/collinear externals; with soft externals one should not connect
        them to the aux vertex.)  If ext_mode is None (e.g. pure-C corpora
        where it is unavailable), all externals are joined (old behaviour;
        identical for kinematics without soft externals).
    Returns (ok, failed_directions)."""
    J, H, S = classify(em, edges_in)
    J_all = set(H)
    for es in J.values():
        J_all |= set(es)
    ext_names = list(ext_attach)
    failures = []
    # 2026-08-13 : the index i in H∪J∖J_i runs over ALL external-momentum
    # directions — including directions with J_i EMPTY.  For those, ∖J_i
    # removes no edges, but the direction's own momentum p_i is still
    # excluded from the auxiliary vertex (it belongs to J_i as a whole, and
    # a jetless C-type momentum carries no O(1) flow that could rescue a
    # dangling 1VI component).  Example: K33a5_46 k0 region 315 —
    # H∪J∖J_1 (J_1 = ∅) is NOT mojetic: after deleting vertex 6, vertex 1
    # (p1 + C_2 line (1,2) + H line (1,6) + S line) is isolated and p1 is
    # not on aux, so the check fails and the region is killed (551 -> 550).
    if ext_mode is None:
        dirs = sorted({ext_attach[n] for n in ext_names})
    else:
        dirs = sorted({ext_mode[n][2] for n in ext_names
                       if ext_mode[n][0] == 0 and ext_mode[n][2] != 0})
    for i in dirs:
        Ji = set(J.get(i, ()))     # may be empty: remove nothing, drop p_i
        Jm = J_all - Ji
        es = list(Jm)
        vs = set()
        for u, w in es:
            vs.add(u)
            vs.add(w)
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

def jet_connected_ok(vm, em, edges_in):
    """Fundamental pattern check: each jet J_i (union of all C_i^n mode
    subgraphs, n >= 1) must be connected (or empty).

    A disconnected jet (e.g. C1 = {v1} ∪ {v9,v10} attached to H at two points)
    violates the fundamental pattern directly — its orphaned components can
    only acquire a scale via non-monotone relevance paths, which the paper's
    IR-compat definition forbids.  Checking it here (before FC / IR compat)
    is the pattern-level statement: jets are connected by construction
    (2026-08-11; killed 36 false positives on UserGraph, FRI 235 -> 199).

    vm: {vertex: mode tuple}; em: list of edge modes (order = edges_in).
    """
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

# ================= pipeline: Step 1 / First Connectivity ==================

def check_fc(g, edge_modes, EXTMODE):
    """First Connectivity Theorem (§5.1): for every threshold n, the union of
    components with 𝒱 <= n must be connected (∪_{𝒱≤n} Γ_X connected ∀n).
    Thresholds are the distinct 𝒱 values present; isolated mode-vertices are
    included as nodes (Γ_X contains mode-X vertices)."""
    eVs = [V(m) if m is not None else INF for m in edge_modes]
    vVs = [V(vertex_mode(g, edge_modes, v, EXTMODE)) if vertex_mode(g, edge_modes, v, EXTMODE) else INF
           for v in g.vertices]
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

def momentum_ok(g, edge_modes, EXTMODE):
    """Momentum conservation at each vertex (§5.1): the incident momenta can be
    split into two nonempty sets whose ∨-joins are equal (∨(in)=∨(out) at mode
    level; necessary, not sufficient). Single-momentum vertices are allowed
    only if that momentum is H (balance with nothing)."""
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

# =============== pipeline: contracted components (1VI) ====================

def _join(modes):
    """Join of a list of modes (hardest common mode)."""
    modes = [m for m in modes if m is not None]
    if not modes:
        return (0, 0, 0)
    acc = modes[0]
    for m in modes[1:]:
        acc = join(acc, m)
    return norm(acc)

def _is_1vi(verts, edges, aux_conn=None):
    """1VI: connected and remains connected after deleting any single vertex.
    Vertices in aux_conn are joined to the auxiliary vertex (X-mode external
    momenta identified with the aux vertex).  Vertices with no incident edge
    are dropped (not part of the component)."""
    used = {u for u, v in edges} | {v for u, v in edges}
    verts = set(verts) & used
    if len(verts) <= 1:
        return True
    aux_conn = (aux_conn or set()) & verts
    if not _connected_aux(verts, edges, aux_conn):
        return False
    for w in list(verts):
        vs = verts - {w}
        es = [(u, v) for (u, v) in edges if u != w and v != w]
        ac = aux_conn - {w}
        if not _connected_aux(vs, es, ac):
            return False
    return True

def _connected_aux(verts, edges, aux_conn=None):
    """Connected, with aux_conn vertices glued through the aux vertex."""
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

def _components(verts, edges):
    """Vertex sets of the connected components (isolated real vertices kept;
    the aux vertex without edges is dropped by _is_1vi anyway)."""
    adj = {v: set() for v in verts}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)
    seen = set()
    comps = []
    for v in verts:
        if v in seen:
            continue
        stack, c = [v], set()
        while stack:
            x = stack.pop()
            if x in c:
                continue
            c.add(x)
            stack.extend(adj[x] - c)
        comps.append(c)
        seen |= c
    return comps

def mode_components_wa(edges_in, em, ext_attach, ext_mode):
    """Regge-style mode components for the wide-angle framework
    (2026-08-21 rewrite).

    For each mode X, decompose the contracted X-subgraph γ̃_X into
    BICONNECTED (1VI) blocks — the paper's mode-component definition —
    instead of taking connected components and filtering out non-1VI ones.
    Blocks sharing a cut vertex (e.g. two SC self-loops joined through an
    aux-absorbed vertex) are separate components, each confirmed
    independently by the IR-compat fixpoint (Regge 08-17 rule).

    Blocks are expanded back to ORIGINAL graph elements ({'mode','V','E'})
    so the existing 3-condition fixpoint (partial_sum / messenger / meet of
    two confirmed) runs unchanged.

    Every mode X (including pure-soft S^m and H — the 2026-08-14
    connected-component exemption was WRONG: it merged 1VI-distinct S
    blocks into one component and let a messenger confirm a soft cloud it
    does not feed; random-graph MISMATCH 2026-09-03 #66 k4) goes through
    the same contracted biconnected decomposition.

    Returns list of comp dicts {'mode', 'V', 'E'} (V/E original elements)."""
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
        # no X vertices but X edges exist: each edge is its own component
        # (build_components behaviour — isolated cross edges like an SC
        # edge whose endpoints both have harder join modes)
        if not vx and ex:
            for e in ex:
                comps.append({'mode': X, 'V': set(), 'E': {e}})
            continue
        if not vx:
            continue
        # all modes (incl. pure-soft S^m and H): biconnected decomposition
        # of the CONTRACTED γ̃_X.  Contracted γ̃_X = Γ_X + one auxiliary
        # vertex, identified with the vertices of other mode subgraphs that
        # are ADJACENT to Γ_X through an X-mode edge (the far endpoint w of
        # an X edge reaching outside V_X — automatically a harder-mode
        # vertex, since edge mode = meet of endpoint join-modes = X).
        # (2026-09-06: cross-mode edges of other modes do NOT enter
        # γ̃_X, and external momenta are NOT joined to this aux — joining
        # externals to an auxiliary vertex belongs to the MOJETIC check,
        # not to the contracted-subgraph construction.  The former
        # implementation absorbed whole V_Y groups via cross edges, which
        # welded e.g. G11 k1's C3 pendant (5,8) onto its triangle block
        # through the aux (S-neighbour group {v2,v4,v9} + p3 line) and hid
        # the cut vertex v8.)
        verts2 = set(vx) | {aux}
        # aux_verts: far endpoints of X edges reaching outside V_X
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
                # X edges are always part of γ̃_X: endpoints outside V_X
                # are absorbed into aux unconditionally; BOTH outside
                # becomes a (aux,aux) self-loop — an isolated X edge, a
                # real component (build_components parity; 2026-08-22
                # K33P02 k2 lost the SC3 edge (3,6) without this).
                a2 = a if a in vx else aux
                b2 = b if b in vx else aux
                if a2 is not None and b2 is not None:
                    edges2.append((a2, b2))
                    orig2.append(e2)
        raw = biconnected_blocks(verts2, edges2)
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
                    # only mode-X edges belong to the component's edge set
                    # (cross edges shape the block but are not γ_X edges)
                    Y = emap.get(oe)
                    if Y is not None and eq(Y, X):
                        E.add(oe)
                    break
            if not V and not E:
                continue
            # pure-aux blocks (V empty) are kept: they are the isolated
            # cross edges (e.g. an SC edge with no SC vertex), which are
            # real components that must be confirmed (build_components
            # gives them V={} E={edge} — 2026-08-21 fix).
            comps.append({'mode': X, 'V': V, 'E': E})
    return comps

def contracted_1vi_ok(edges_in, em, ext_attach, ext_mode):
    """Check that every contracted mode component is 1VI.

    edges_in: list of (u,v) internal edges (order matches em).
    em: list of edge mode tuples (m,n,i) or None (= H).
    ext_attach: {external-name: vertex}.
    ext_mode: {external-name: mode tuple} (may be None → no externals joined).
    Returns (ok, failures) where failures = list of (mode, comp) for
    non-1VI contracted components."""

    # ---- join modes of all vertices (paper: 𝒳(v) = ∨ incident ∪ external) ----
    inc = defaultdict(list)
    for e, md in zip(edges_in, em):
        u, v = e
        m = md if md is not None else (0, 0, 0)
        inc[u].append(m)
        inc[v].append(m)
    if ext_mode:
        for name, v in ext_attach.items():
            if name in ext_mode:
                inc[v].append(ext_mode[name])
    vjoin = {v: _join(ms) for v, ms in inc.items()}

    # ---- group edges by mode; adjacency of the full graph ----
    emodes = defaultdict(list)
    emap = {}
    for e, md in zip(edges_in, em):
        m = md if md is not None else (0, 0, 0)
        emodes[m].append(e)
        emap[e] = m

    failures = []
    for X, ex in emodes.items():
        # 2026-08-14 : the contracted-1VI requirement applies ONLY to
        # X = S^m C^n with n >= 1 (collinear component present).  Pure soft
        # modes X = S^m (n = 0) and H are exempt (e.g. region A of
        # K33a3_2_nn kinematics2 has a pure-soft S^1 component {4,6,7}
        # whose contracted graph is not 1VI, yet the region is a genuine
        # facet — it must NOT be killed).
        if X[1] == 0:
            continue
        # V_X = vertices whose join mode is X
        vx = {v for v, jm in vjoin.items() if eq(jm, X)}
        if not vx:
            continue
        # connected components of Γ_X (X-edges with both endpoints in V_X)
        parent = {v: v for v in vx}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for u, v in ex:
            if u in vx and v in vx:
                union(u, v)
        comps = defaultdict(set)
        for v in vx:
            comps[find(v)].add(v)
        for root, V_C in comps.items():
            # aux_verts: vertices of the other mode subgraphs adjacent to
            # this component.  Γ_Y is adjacent to Γ_X iff some edge e ∈ E_Y
            # (i.e. mode(e) == Y) has an endpoint in V_X.  So: for every
            # edge e=(a,b) with exactly one endpoint in V_C:
            #   - if mode(e) == X: the other endpoint b belongs to Γ_{join(b)}
            #     (a vertex of a harder subgraph joined by an X-edge) → aux
            #     absorbs all of V_{join(b)};
            #   - if mode(e) == Y ≠ X: e ∈ E_Y touches V_X → aux absorbs V_Y.
            # (2026-08-22 : aux's identity IS "the vertices from other,
            # harder subgraphs" — it has nothing to do with whether an
            # external momentum sits there.  This is broader than the paper's
            # mojetic aux (which joins only attached externals) and is the
            # reason K33P02's valid SC components survive: their vertices are
            # rescued by harder/equal neighbours absorbed into aux.  Naive
            # external-only aux kills 475-558/856 K33P02 regions.)
            aux_verts = set()
            for e2 in edges_in:
                a, b = e2
                a_in, b_in = a in V_C, b in V_C
                if a_in == b_in:
                    continue
                Y = emap[e2]
                if eq(Y, X):
                    w = b if a_in else a
                    Yw = vjoin.get(w)
                    if Yw is not None:
                        for z, jm in vjoin.items():
                            if eq(jm, Yw):
                                aux_verts.add(z)
                else:
                    for z, jm in vjoin.items():
                        if eq(jm, Y):
                            aux_verts.add(z)
            # contracted graph: induced subgraph of G on V_C ∪ aux_verts,
            # with aux_verts contracted to a single aux vertex.
            cverts = set(V_C) | {('aux', X)}
            cedges = []
            for e2 in edges_in:
                a, b = e2
                a2 = a if a in V_C else (('aux', X) if a in aux_verts else None)
                b2 = b if b in V_C else (('aux', X) if b in aux_verts else None)
                if a2 is not None and b2 is not None and a2 != b2:
                    cedges.append((a2, b2))
            # X-mode external momenta attached to the component are
            # identified with the same aux vertex (2026-08-14).
            # The external-momentum mode may be S^m C^{n+n'} with
            # n' >= 0: same softness m, same direction, collinear index
            # >= n (i.e. equal or harder within the same jet family).
            if ext_mode:
                mX, nX, iX = X
                for name, v in ext_attach.items():
                    if v in V_C and name in ext_mode:
                        mY, nY, iY = norm(ext_mode[name])
                        if mY == mX and iY == iX and nY >= nX:
                            cedges.append((v, ('aux', X)))
            for comp in _components(cverts, cedges):
                if not _is_1vi(comp, cedges):
                    failures.append((X, comp))
    return len(failures) == 0, failures

# ================== condition machinery & IR fixpoint =====================

def relevant(g1, g2, verts, edges, all_comps):
    if not marginal_softer(g1['mode'], g2['mode']): return False
    emode = {v: md for v, md in verts.items()}
    for (a, b, md) in edges:
        emode[(a, b)] = md; emode[(b, a)] = md
    start = elements(g1); target = elements(g2)
    visited = set(start)
    dq = deque((e, g1['mode']) for e in start)
    while dq:
        e, cur = dq.popleft()
        if e in target: return True
        for nb in neighbors(e, verts, edges):
            if nb in visited: continue
            nm = emode[nb]
            # paper's relevance (def): along the path away from g1 the modes
            # must be monotonically non-softer — "softer than or equal" per
            # element pair; OVERLAPPING steps are forbidden.  The old V-based
            # check (V(emode[nb]) <= V(cur)) allowed overlapping steps (e.g.
            # C2 -> C1 with equal V), which let disconnected jet pieces get
            # fake scale sources through non-monotone paths (2026-08-11: 36
            # false positives on UserGraph; pySecDec 199 vs FRI 235).
            if harder_or_eq(nm, cur):
                visited.add(nb); dq.append((nb, nm))
    return False

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

def partial_sum_mode(g, confirmed, verts, edges, all_comps, attach, extmode):
    """Cond 1: partial sum over momenta entering γ.
    External momentum X enters γ if X attaches to γ, or X's momentum flows into γ
    (X marginally softer than γ and a monotone path exists from X's attach vertex).
    Also include flows from confirmed subgraphs relevant to γ."""
    modes = []
    for name, v in attach.items():
        em = extmode[name]
        # attach point in γ, or X relevant to γ (marginal + path from attach vertex)
        if v in g['V']:
            modes.append(em)
        else:
            # 2026-08-21 : an external momentum may propagate ALONG ITS
            # OWN MODE — equal mode counts (not just marginal_softer), e.g.
            # p2 (C2²) entering a C2² block {7}+(4,7) through the shared
            # cut vertex 7 via the C2² edge (2,7).  (k+k)² ~ 4k² still
            # depends on k, so the cross-term condition of marginal
            # softness holds trivially for equal modes.
            if eq(em, g['mode']) or marginal_softer(em, g['mode']):
                # path from attach vertex to γ through monotonically-non-softer
                # elements (paper's relevance def: "softer than or equal" per
                # element pair; OVERLAPPING steps forbidden — the old V-based
                # check allowed overlapping steps, 2026-08-11).
                emode = {vv: md for vv, md in verts.items()}
                for (a, b, md) in edges:
                    emode[(a, b)] = md; emode[(b, a)] = md
                start = {v}
                target = elements(g)
                visited = set(start)
                # start with the ATTACH VERTEX's own mode — the external's
                # flow first meets the vertex it is attached to, so that is
                # the first element of the monotone path.  (2026-08-11 :
                # starting with the external's mode wrongly let e.g. an
                # SC5^inf at a C2 vertex flow C2 -> S -> C3, a softening
                # step, and "confirmed" C3 components that have no real
                # scale source; CrownASE k1-k5 false positives.)
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

def relevant_hit(g1, g2, verts, edges):
    """First element of g2 reached by g1's monotone flow, or None.
    Same walk as relevant() but reports WHERE the flow enters g2 — the
    entry element for cond1_strong's third-port requirement (2026-09-05/06)."""
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
            if harder_or_eq(nm, cur):
                visited.add(nb); dq.append((nb, nm))
    return None

def _entry_of(g, hit):
    """Entry vertices of a flow whose monotone path hit element `hit` of g.
    A vertex hit is itself; an edge hit contributes its endpoints that lie
    in g.V (the other endpoint is aux-absorbed, e.g. a pendant edge)."""
    if hit in g['V']:
        return {hit}
    if isinstance(hit, tuple):
        return {v for v in hit if v in g['V']}
    return set()

def cond1_sources(g, confirmed, verts, edges, all_comps, attach, extmode):
    """Cond-1 partial-sum sources with their ENTRY vertices into γ.
    Returns [(mode, entry_set), ...]: external momenta attached to γ
    (entry = attach vertex) or entering via a monotone path (entry = hit
    element's γ-vertices), plus confirmed components relevant to γ.
    Sources whose entry cannot be pinned to a γ-vertex are dropped (they
    cannot serve as the va/vb of the third-port requirement)."""
    modes = []
    for name, v in attach.items():
        em = extmode[name]
        if v in g['V']:
            modes.append((em, {v}))
        else:
            if eq(em, g['mode']) or marginal_softer(em, g['mode']):
                emode = {vv: md for vv, md in verts.items()}
                for (a, b, md) in edges:
                    emode[(a, b)] = md; emode[(b, a)] = md
                start = {v}
                target = elements(g)
                visited = set(start)
                dq = deque((e, emode[v]) for e in start)
                hit = None
                while dq:
                    e, cur = dq.popleft()
                    if e in target:
                        hit = e
                        break
                    for nb in neighbors(e, verts, edges):
                        if nb in visited: continue
                        nm = emode[nb]
                        if harder_or_eq(nm, cur):
                            visited.add(nb); dq.append((nb, nm))
                if hit is not None:
                    ev = _entry_of(g, hit)
                    if ev:
                        modes.append((em, ev))
    for g2 in confirmed:
        if g2 is g:
            continue
        hit = relevant_hit(g2, g, verts, edges)
        if hit is not None:
            ev = _entry_of(g, hit)
            if ev:
                modes.append((g2['mode'], ev))
    return modes

def third_port(g, entry_vs, all_comps, verts):
    """Third-port requirement (2026-09-05): apart from the entry
    vertices va, vb of the cond-1 momentum sources, γ must touch a vertex w
    that is either
      (A) an 𝒳(γ)-mode vertex of γ shared with another 𝒳(γ)-mode component
          (cut vertex to a same-mode component — γ continues along the jet),
      (B) a harder-mode vertex reached by one of γ's OWN edges (the
          pendant/exit endpoint, aux-absorbed in the representation).
    Returns ('A'|'B', w) or None."""
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

def cond1_strong(g, confirmed, verts, edges, all_comps, attach, extmode):
    """Strengthened cond 1 (2026-09-05).  Two momenta k1, k2 (external
    momenta or confirmed line momenta) are relevant to γ — monotone paths,
    possibly trivial, entering at va, vb ∈ γ (va may equal vb) — with
    𝒳(k1) ∨ 𝒳(k2) = 𝒳(γ); the =1 case (single source whose mode equals
    𝒳(γ)) is allowed; >2 sources reduce to 2 (paper Thm 3.2).
    ADDITIONALLY γ must have a THIRD PORT w outside {va, vb}: a same-mode
    cut vertex to another 𝒳(γ)-component (A) or a harder vertex reached by
    γ's own edge (B).  A block whose scale must be assembled from two
    sources (e.g. a lightlike C^∞ external + S cutting the depth back to
    C^1) but has no exit port is scaleless (G11 k1 pseudo-region: the C3
    triangle).  H components are momentum sinks — exempt (no harder mode
    exists).

    Returns True/False."""
    if g['mode'] == (0, 0, 0):
        ps = partial_sum_mode(g, confirmed, verts, edges, all_comps,
                              attach, extmode)
        return ps is not None and eq(ps, g['mode'])
    srcs = cond1_sources(g, confirmed, verts, edges, all_comps, attach,
                         extmode)
    # single source whose mode equals 𝒳(γ)
    for md, ev in srcs:
        if eq(md, g['mode']) and third_port(g, ev, all_comps, verts):
            return True
    # pair of sources whose join equals 𝒳(γ)
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            md1, ev1 = srcs[i]
            md2, ev2 = srcs[j]
            if eq(join(md1, md2), g['mode']) \
                    and third_port(g, ev1 | ev2, all_comps, verts):
                return True
    return False

def cond1_sources_ae(g, confirmed, verts, edges, all_comps, attach, extmode):
    """cond1_sources with ALL entry candidates per source (vertex sets)."""
    modes = []
    for name, v in attach.items():
        em = extmode[name]
        if v in g['V']:
            modes.append((em, {v}))
        else:
            if eq(em, g['mode']) or marginal_softer(em, g['mode']):
                hs = _collect_entries({v}, verts.get(v, (0, 0, 0)),
                                      elements(g), verts, edges)
                ev = _entry_set(g, hs)
                if ev:
                    modes.append((em, ev))
    for g2 in confirmed:
        if g2 is g:
            continue
        if not marginal_softer(g2['mode'], g['mode']):
            continue
        hs = _collect_entries(elements(g2), g2['mode'], elements(g),
                              verts, edges)
        ev = _entry_set(g, hs)
        if ev:
            modes.append((g2['mode'], ev))
    return modes

def _collect_entries(start_elems, start_mode, target, verts, edges):
    """All elements of `target` reachable from start_elems by monotone
    non-softer flow (BFS without early break)."""
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
            continue          # do NOT propagate inside target: entry = first
                              # contact only (2026-09-07, G11 case)
        for nb in neighbors(e, verts, edges):
            if nb in visited:
                continue
            nm = emode[nb]
            if harder_or_eq(nm, cur):
                visited.add(nb); dq.append((nb, nm))
    return hits

def _entry_set(g, hit_elems):
    """Vertices of γ at which a flow (with these hit elements) enters."""
    vs = set()
    for h in hit_elems:
        if h in g['V']:
            vs.add(h)
        elif isinstance(h, tuple):
            for w in h:
                if w in g['V']:
                    vs.add(w)
    return vs

def cond1_strong_ae(g, confirmed, verts, edges, all_comps, attach, extmode):
    """cond1_strong with entry-choice enumeration: each momentum source may
    enter γ at any of its reachable vertices, so va may equal vb even when
    the first-hit entries differ (third port then sits at the other end of
    the two-port bridge)."""
    if g['mode'] == (0, 0, 0):
        ps = partial_sum_mode(g, confirmed, verts, edges, all_comps,
                              attach, extmode)
        return ps is not None and eq(ps, g['mode'])
    srcs = cond1_sources_ae(g, confirmed, verts, edges, all_comps, attach,
                            extmode)
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


# ==================== mojetic / 1VI checks ====================
# (merged from mojetic_check.py, 2026-09-19)
# Mojetic check (Theorem 3 cond1 of the on-shell paper): for each external
# direction i, the subgraph H∪J∖J_i must be mojetic — i.e. it becomes 1VI
# (connected after deleting any single vertex) once all external momenta
# attached to it are joined to an auxiliary vertex.
#
# Classification (general definition):
#   J_i  = union of all edges whose mode is C_i^n (n >= 1)
#   H    = hard edges (mode (0,0,0))
#   S    = everything else (S^m, S^m C_i^n)
#
# Physical meaning: mojetic failure ⟺ some 1VI component of H∪J is attached
# only via a single vertex / only by S lines / only by one jet direction —
# all of which violate component-level momentum conservation (O(1) momenta
# cannot be balanced by O(λ) flows).
# Tested 2026-08-10: zero false kills on 7179 wide-angle regions; exactly
# the 4 known hypercrown k0 extras fail.

def find_messenger(g, all_comps, verts, edges, attach, extmode, confirmed=(),
                  kernel=None, kernel_blocks=None):
    """IR-compat cond 2 support (v16 definition). If mode(g)=S^m, build
    Γ^[m] = connected closure of S^m C_i^{n_i} components containing g and
    check: >=3 targets γ_i in pairwise-distinct directions, each matched to
    a Γ member relevant to it (n_i=0 → kernel; n_i>=1 → S^m C_{j_i}^{n_i}).
    Returns (Γ^[m] member list, special, kernel blocks) if valid, else None.

    2026-09-04 (rule 1): the kernel of Γ^[m] is the whole CONNECTED
    component of the S^m subgraph — possibly several 1VI blocks joined
    through shared real S^m vertices (kernel/kernel_blocks).  The kernel
    match, the member-adjacency and the special-external positions are
    evaluated against the whole cloud.  Default (no kernel given) = the
    single block g (2026-08-14 connected-kernel semantics)."""
    m = g['mode'][0]
    if g['mode'][1] != 0 or g['mode'][2] != 0: return None
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
    # 2026-08-14 (connected kernel): a degree-m messenger's kernel is the
    # CONNECTED S^m component being checked (g) — always connected by
    # construction.  The 2026-08-13 rule "n_s > 1 -> FAIL" was too coarse:
    # a Gamma^[m] closure may legitimately contain a SECOND S^m component
    # bridged through a shared confirmed SC member (e.g. K33P02 k4: S{7,8}
    # and S{6,9} both touch the confirmed SC4 component via lines (4,7)/(4,9)
    # — each is an independent valid degree-1 messenger with directions
    # 1 (SC1->C1^2), 3 (SC3->C3^2) and 4 (confirmed SC4 line).  The old rule
    # killed both.  What must be prevented instead is BORROWING: the other S
    # component's external momentum must not feed this kernel (that was the
    # CheesePizza k2 case: l1 sat on the OTHER S component).  So:
    #   - special (a): the feeding external must sit on the kernel's own
    #     vertices or on a Gamma member ADJACENT to the kernel;
    #   - n_i = 0 kernel matches use ONLY g (not any other S^m in Gamma).
    n_s = sum(1 for c in G if c['mode'][0] == m and c['mode'][1] == 0 and c['mode'][2] == 0)
    if n_s > 1:
        # still proceed: kernel g is connected; other S^m members are simply
        # not part of g's verdict (guarded by the two rules below)
        pass
    gids = {id(c) for c in G}
    # A target gamma_i = S^{m-m_i} C_i^{m_i+n_i} (m_i = m - m', n_i = n' - m_i)
    # is counted iff the Gamma member S^m C_i^{n_i} is relevant to it: for
    # n_i = 0 that member is the KERNEL S^m (direction-free, mode (m,0,0));
    # for n_i >= 1 it is an S^m C_i^{n_i} component of the messenger
    # (v16: no "at least one n_i = 0" requirement — author dropped it,
    #  verified harmless on 55=55, 81/81, 118/118).
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
            # kernel match: ONLY the checked connected kernel — the whole
            # S^m cloud (2026-09-04 rule 1) — may serve as the (m,0,0)
            # member for this target; another S^m component in the closure
            # is NOT part of the messenger (2026-08-14 connected kernel).
            pool = [kernel] if kernel['mode'] == (m, 0, 0) else []
        for c in pool:
            # connected kernel: the matching Gamma member must be ADJACENT to
            # the kernel — a leg on the far side of a bridged S^m component is
            # not part of g's messenger (2026-08-14; K33P02 k4: the leaf S{9}
            # borrowed SC1(1,7)/SC3(3,8) from the other S{7,8} through the
            # SC4 bridge — its only OWN direction is SC4, <3 -> reject).
            if adjacent(c, kernel) and relevant(c, d, verts, edges, all_comps):
                R.add(id(d))
                break
    # ---- 2026-08-12 TENTATIVE (: type-2 / m_i=0 special target ----
    # The kernel S^m itself can be a target when it carries an attached
    # external momentum of mode S^m C_i^{n'} (same m, n' >= 1, i != 0): the
    # S^m part of that external IS the kernel's own scale source, so no
    # relevance to confirmed subgraphs is needed for it (CrownSS k5 region
    # (-3,-4,-2,0,0,-2,-3,0,0,0,1): S kernel {v5} fed by l1 = SC5^inf,
    # targets C1^2 / C2^3 (strict relevance) + kernel-itself (dir 5)).
    special_dir = None
    if ALLOW_SPECIAL_TARGET:
        # (a) attached external momentum of mode S^m C_i^{n'} feeding the
        # messenger.  2026-08-14 (connected kernel): the external must
        # sit on the kernel's OWN vertices or on a Gamma member ADJACENT to
        # the kernel — an external on the OTHER S^m component (bridged) must
        # NOT feed this kernel (CheesePizza k2 case).  In the K33P02 k4 case
        # l1 sits on the confirmed SC4 component which IS adjacent to the
        # kernel via line (4,7) — allowed; more precisely the SC4 line
        # momentum itself is the source (branch (b)), the external sitting
        # there is just its avatar.
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
        # (b) 2026-08-12 : a CONFIRMED component of mode S^m C_i^{n'}
        # (n' >= 1, i != 0) touching the messenger feeds it exactly like an
        # attached external — any line momentum of a confirmed component X
        # has the same status as an X-mode external momentum.  Gamma members
        # are NOT excluded (a confirmed member's line momentum feeds the
        # kernel through the shared messenger).
        if special_dir is None:
            for d in confirmed:
                md = d['mode']
                if md[0] == m and md[1] >= 1 and md[2] != 0 \
                   and adjacent(d, kernel):
                    special_dir = md[2]
                    break
    ntargets = len(R) + (1 if special_dir is not None else 0)
    if ntargets < 3: return None
    # a messenger is relevant to >=3 subgraphs from DIFFERENT directions
    # (definition precedes IR compatibility; count distinct directions i)
    dirs = set()
    for rid in R:
        d = next(x for x in all_comps if id(x) == rid)
        if d['mode'][2] != 0:
            dirs.add(d['mode'][2])
    if special_dir is not None:
        dirs.add(special_dir)
    if len(dirs) < 3: return None
    # check each γ_i: mode = S^{m-m_i} C_i^{m_i+n_i}, m_i in 1..m, n_i >= 0,
    # and any external momentum entering γ_i is compatible.
    # NOTE (v16): no "at least one n_i = 0" requirement — author dropped it
    # (verified: no effect on 55=55, 81/81, 118/118).
    for rid in R:
        d = next(x for x in all_comps if id(x) == rid)
        md = d['mode']
        mi = m - md[0]
        if not (1 <= mi <= m): return None
        if md[1] < mi: return None
        ext = [extmode[nm] for nm, v in attach.items() if v in d['V']]
        for e in ext:
            # form-compatible external: S^{m-m_i} C_i^{m_i+n_i+n'_i} (same
            # direction, same m, at least as deep)
            if e[0] == md[0] and e[2] == md[2] and e[1] >= md[1]:
                continue
            # absorbable softer external (e.g. an SC5^inf soft emission at a
            # C2 component: C2^2 ∨ SC5^inf = C2) — it is swallowed by the
            # target's own mode and must not break the messenger verdict
            # (2026-08-11 : CrownSS k2 region, S messenger relevant to
            # C2/C3/C4; the old check killed it on the l1 direction mismatch).
            if harder_or_eq(md, e):
                continue
            return None
    return G, special_dir is not None, kernel_blocks

def check_conditions(g, confirmed, verts, edges, all_comps, attach, extmode):
    """Three recursive IR-compat conditions (same as region_checker.py).
    cond 1: partial sum of incoming momenta ∨-joins to 𝒳(γ) AND the
            strengthened third-port requirement (cond1_strong, 2026-09-05);
    cond 3: 𝒳(γ) = 𝒳(γ₁)∧𝒳(γ₂), γ relevant to two confirmed;
    cond 2: γ = S^m in a degree-m messenger, relevant to confirmed γ₀."""
    if cond1_strong(g, confirmed, verts, edges, all_comps, attach, extmode):
        return 1
    for g1, g2 in combinations(confirmed, 2):
        if relevant(g, g1, verts, edges, all_comps) and relevant(g, g2, verts, edges, all_comps):
            if eq(meet(g1['mode'], g2['mode']), g['mode']): return 3
    res = find_messenger(g, all_comps, verts, edges, attach, extmode, confirmed)
    if res is not None:
        G, special, _ = res
        # type-2 special (m_i=0): the kernel is fed by its own attached
        # S^m C_i^{n'} external — no relevance to a confirmed subgraph
        # is required (2026-08-12 , tentative).
        if special:
            return 2
        for g0 in confirmed:
            if relevant(g, g0, verts, edges, all_comps): return 2
    return None

def ir_ok(g, edge_modes, EXTMODE):
    """IR compatibility (§5.2): run the recursive 3-condition fixpoint over all
    components (cond 1 partial-sum ∨; cond 3 meet of two confirmed; cond 2
    messenger). Region iff every component becomes confirmed.
    Inline copy of region_checker.check_conditions logic (kept separate so
    the zero-knowledge enumerator has no dependency on paper figures)."""
    verts = {}
    for v in g.vertices:
        md = vertex_mode(g, edge_modes, v, EXTMODE)
        if md is not None: verts[v] = md
    edges = [(u, v, md) for (u, v), md in zip(g.edges, edge_modes) if md is not None]
    attach = {name: vv for name, vv in g.ext.items()}
    extmode = {name: EXTMODE[name] for name in g.ext}
    all_comps = build_components(verts, edges)
    confirmed = []
    changed = True
    while changed:
        changed = False
        for comp in all_comps:
            if comp in confirmed: continue
            cc = None
            # cond 1 — MUST use the real partial_sum_mode (region_checker):
            # the old inline version only counted externals attached directly
            # to the component, missing marginally-softer externals entering
            # via a monotone path (e.g. l1=SC4 flowing into a C1 component
            # through an S line).  That drift silently dropped every S-mode
            # region of the paper's 5pt6loop example (SCiregion66 etc.),
            # 21 vs 40 S regions (2026-08-11).
            ps = partial_sum_mode(comp, confirmed, verts, edges, all_comps,
                                     attach, extmode)
            if ps is not None and eq(ps, comp['mode']):
                cc = 1
            if cc is None:
                for g1, g2 in combinations(confirmed, 2):
                    if relevant(comp, g1, verts, edges, all_comps) and relevant(comp, g2, verts, edges, all_comps):
                        if eq(meet(g1['mode'], g2['mode']), comp['mode']): cc = 3; break
            if cc is None:
                res = find_messenger(comp, all_comps, verts, edges, attach, extmode, confirmed)
                if res is not None:
                    G, special, _ = res
                    # type-2 special (m_i=0): kernel fed by its own attached
                    # S^m C_i^{n'} external — no confirmed relevance needed
                    # (2026-08-12 , tentative).
                    if special:
                        cc = 2
                    else:
                        for g0 in confirmed:
                            if relevant(comp, g0, verts, edges, all_comps): cc = 2; break
            if cc is not None:
                confirmed.append(comp); changed = True
    return len(confirmed) == len(all_comps)


# =========================================================================
# Contracted mode component check (paper §3.2) — moved here from
# contracted_1vi.py, 2026-09-20 (that file was removed; this check belongs
# with the IR-compatibility machinery that consumes it).
#
# Requirement: EACH contracted mode component is 1VI.
#
# Definitions (paper §3.2):
#   - mode X: one (m,n,i) tuple.
#   - mode subgraph Γ_X = (V_X, E_X):
#       V_X = {v : 𝒳(v) = X}   (vertices by JOIN mode: 𝒳(v) = ∨ of incident
#              edge modes and attached external modes)
#       E_X = {e : 𝒳(e) = X}   (edges by mode)
#   - contracted X subgraph ~Γ_X: Γ_X + one auxiliary vertex, identifying the
#     aux vertex with ALL vertices of other mode subgraphs adjacent to Γ_X.
#     Implemented as the INDUCED subgraph of the full graph G on
#     V_X ∪ aux_verts, with aux_verts contracted to a single aux vertex,
#     where aux_verts = ∪{V_Y : Γ_Y adjacent to Γ_X} (adjacent = some edge
#     of G joins a vertex of Γ_X to a vertex of Γ_Y).  Cross-mode edges
#     between a component vertex and an aux vertex are kept (they carry the
#     momentum flow into the harder background).
#   - X-mode component: a connected component of ~Γ_X that is 1VI.
#
# The check: for EVERY mode X present, EVERY connected component of ~Γ_X
# (ignoring self-loops) must be 1VI (deleting any single vertex leaves it
# connected).
#
# Physical meaning: a mode component that is not 1VI after contraction has a
# cut vertex — some part attaches to the rest only through one vertex, its
# loop momenta are not pinned by any spanning structure, the leading F
# polynomial factorises through the cut, and the region is scaleless.  This
# is the graph-theoretic (polynomial-free) form of the "leading F depends on
# x8,x9,x10 only through x8+x9+x10" obstruction found 2026-08-14 on
# K33a3_2_nn kinematics2 (FRI 135 vs pySecDec 134).
# =========================================================================

def ir_ok_blocks(g, edge_modes, EXTMODE):
    """IR compatibility, Regge-style (2026-08-21 rewrite).

    Mode components = BICONNECTED (1VI) blocks of the contracted γ̃_X
    (mode_components_wa), each block confirmed independently by the
    recursive 3-condition fixpoint (cond 1 partial-sum ∨; cond 3 meet of
    two confirmed; cond 2 messenger).  Region iff every block becomes
    confirmed.  Replaces the old connected-components + 1VI-filter logic
    (contracted_1vi_ok + ir_ok): a non-1VI connected piece is now
    DECOMPOSED into its 1VI blocks instead of rejected."""
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
    # 2026-08-22 : re-instate the contracted-1VI filter for modes with a
    # collinear component (C_i^n and S^m C_i^n, n >= 1).  The 08-21 rewrite
    # dropped it in favour of the biconnected decomposition, which lets the
    # fixpoint confirm every block of a NON-1VI SC component independently:
    # K33a3_2_nn k2/k4 — the S^1C_4^1 component {4,5,6,7} with pendant (4,5)
    # splits into triangle + pendant, both confirm via l1 (cond1), and the
    # four isolated S^2 edges then confirm via cond3 meet(C_1^2, S^1C_4^1)=S^2
    # — a region pySD rejects (scaleless).  contracted_1vi_ok kills exactly
    # that: 0/98439 ground-truth regions, 0/856 K33P02 k2,
    # spurious K33a3_2_nn k2/k4 region killed.
    ok1vi, _ = contracted_1vi_ok(edges_in, em, attach, extmode)
    if not ok1vi:
        return False
    # 2026-09-06: mode_components_wa builds γ̃_X with the correct aux (far
    # endpoints of X edges only) — its components ARE the paper's 1VI
    # blocks (pendant vs loop split); cond1_strong's third-port requirement
    # runs on them.
    all_comps = mode_components_wa(edges_in, em, attach, extmode)

    # S^mC^n tadpole (2026-09-03, random-graph case #66 k4): a
    # soft-family component (m >= 1) adjacent to NO harder-mode component.
    # Such a component is already a 1VI block of its OWN mode subgraph
    # before any contraction (contraction has nothing to attach to), so it
    # can only be confirmed by cond 1 (partial sum) — unless it is a pure
    # S^m block carrying two attached momenta whose vee is exactly its mode
    # (2026-09-04 rule 3, _vee_ok below), in which case cond 2/3 are
    # allowed again.  Hard (m=0) and jet modes are exempt.
    def _is_tadpole(comp):
        if comp['mode'][0] < 1:
            return False
        # adjacency to a HARD vertex (vm = H) counts: the hard subgraph
        # includes isolated hard vertices with no H edge (MTest1 k0:
        # S{3}+(3,4) touches v4 = H via the q1 external — NOT a tadpole;
        # 2026-09-03).  Component elements: its vertices plus the
        # endpoints of its edges.
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
            if V(other['mode']) < V(comp['mode']) \
                    and adjacent(comp, other):
                return False
        return True

    tadpole = {id(c): _is_tadpole(c) for c in all_comps}

    # 2026-09-04 (rules 1+2+3): the kernel of a messenger Γ^[m] is
    # the whole CONNECTED component of the pure-soft S^m subgraph (rule 1)
    # — several 1VI blocks joined through shared REAL S^m vertices count as
    # ONE kernel; blocks that only touch through non-S^m (aux) vertices do
    # not.  A messenger confirmation then covers every confirmable S^m
    # block of that kernel cloud (rule 2).  And a pure-soft S^m tadpole
    # block may use cond 2/3 when — and only when — two momenta attached to
    # it have vee exactly equal to its mode (rule 3).  The 09-03 blanket
    # ban stays for SC (n>=1) tadpoles; cond 1 stays open to everyone.
    def _pure_soft(comp):
        md = comp['mode']
        return md[0] >= 1 and md[1] == 0

    # kernel clouds: blocks of the same pure-soft mode X are in one cloud
    # iff transitively connected through shared real X-mode vertices.
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

    def _vee_ok(comp):
        """Rule 3 (2026-09-04): a tadpole X must carry two momenta k1,
        k2 attached to it — line momenta of modes with a collinear part
        (n >= 1) incident at the component, or external momenta — whose vee
        is precisely X.  Pure-soft lines (n=0) cannot be the scale source
        of a soft blob and are excluded."""
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
    _vee_cache = {}

    def _cond_allowed(comp):
        if not tadpole.get(id(comp)):
            return True
        if not _pure_soft(comp):
            return False        # SC tadpoles keep the 09-03 cond 2/3 ban
        ok = _vee_cache.get(id(comp))
        if ok is None:
            ok = _vee_ok(comp)
            _vee_cache[id(comp)] = ok
        return ok

    def _cloud_kernel(comp):
        """Merged kernel dict {'mode','V','E'} of comp's cloud (rule 1),
        or None when the cloud is the single block itself."""
        blocks = cloud_of.get(id(comp))
        if blocks is None or len(blocks) == 1:
            return None, None
        kern = {'mode': comp['mode'], 'V': set(), 'E': set()}
        for b in blocks:
            kern['V'] |= set(b['V'])
            kern['E'] |= set(b['E'])
        return kern, tuple(blocks)

    confirmed = []
    changed = True
    while changed:
        changed = False
        for comp in all_comps:
            if comp in confirmed:
                continue
            cc = None
            if cond1_strong_ae(comp, confirmed, verts, edges, all_comps,
                                  attach, extmode):
                cc = 1
            if cc is None and _cond_allowed(comp):
                for g1, g2 in combinations(confirmed, 2):
                    if relevant(comp, g1, verts, edges, all_comps) and relevant(comp, g2, verts, edges, all_comps):
                        if eq(meet(g1['mode'], g2['mode']), comp['mode']): cc = 3; break
            kbs = ()
            if cc is None and _cond_allowed(comp):
                kern, kbs = _cloud_kernel(comp)
                res = find_messenger(comp, all_comps, verts, edges, attach,
                                        extmode, confirmed, kernel=kern,
                                        kernel_blocks=kbs or None)
                if res is not None:
                    G, special, kbs = res
                    if special:
                        cc = 2
                    else:
                        for g0 in confirmed:
                            if relevant(comp, g0, verts, edges, all_comps): cc = 2; break
            if cc is not None:
                confirmed.append(comp); changed = True
                if cc == 2:
                    # rule 2: a messenger confirmation covers every
                    # confirmable S^m block of the kernel cloud (vee-
                    # failing tadpoles stay out, rule 3).
                    for kb in kbs:
                        if kb is not comp and kb not in confirmed \
                                and _cond_allowed(kb):
                            confirmed.append(kb); changed = True
    return len(confirmed) == len(all_comps)

# ================== tikz-figure input & figure entry ======================

def _nearest(pt, verts):
    b, bd = None, 1e9
    for v in verts:
        d = (v[0]-pt[0])**2 + (v[1]-pt[1])**2
        if d < bd: b, bd = v, d
    return b, bd

def parse(fig, colormap, extmap):
    colmode = {c: parse_mode(m) for c, m in colormap.items()}
    extmode = {e: parse_mode(m) for e, m in extmap.items()}
    src = open(fig).read()
    # vertices: \node [draw, Col, circle, ..., fill=Col, ...] () at (x,y) {};
    verts = {}
    for m in re.finditer(r'\\node\s*\[([^\]]*)\]\s*\(\)\s*at\s*\(([-\d.]+),([-\d.]+)\)', src):
        col, x, y = m.group(1), float(m.group(2)), float(m.group(3))
        for c, md in colmode.items():
            if re.search(r'\b' + c + r'\b', col): verts[(x, y)] = md; break
    # edges: \\path (a) edge [..., Col, ...] (b) {};   and \draw [..., Col, ...] (a) -- (b);
    raw = []
    for m in re.finditer(r'\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)\s*edge\s*\[([^]]*)\]\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', src):
        x1, y1, col, x2, y2 = m.group(1), float(m.group(2)), m.group(3), float(m.group(4)), float(m.group(5))
        for c, md in colmode.items():
            if re.search(r'\b' + c + r'\b', col):
                raw.append(((float(x1), float(y1)), (float(x2), float(y2)), md)); break
    # draw colored lines: (v1) -- (near-label point); map label point to nearest vertex.
    # If the label-side endpoint's nearest vertex has a DIFFERENT mode than the line color,
    # the line is an external-momentum line (e.g. q1's blue lines to (5.7,5.9)/(5.9,5.7),
    # whose nearest vertex (6,7) is a soft-mode vertex) -> skip (attach handled by labels).
    # If the nearest vertex matches the line color, it is an internal edge (e.g. H blue edge
    # (5,5)-(5,2) with (5,2) a Blue vertex).
    for m in re.finditer(r'\\draw\s*\[([^]]*)\]\s*\(([-\d.]+),([-\d.]+)\)\s*--\s*\(([-\d.]+),([-\d.]+)\)', src):
        col, x1, y1, x2, y2 = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
        for c, md in colmode.items():
            if re.search(r'\b' + c + r'\b', col):
                e1, d1 = _nearest((float(x1), float(y1)), verts)
                e2, d2 = _nearest((float(x2), float(y2)), verts)
                if e1 is None or e2 is None or d1 >= 4.0 or d2 >= 4.0 or e1 == e2:
                    break
                # label-side vertex mode must match the line's mode (internal edge)
                if verts[e2] != md:
                    break
                raw.append((e1, e2, md))
                break
    def isv(p): return any((p[0]-v[0])**2 + (p[1]-v[1])**2 < 0.4 for v in verts)
    edges = [e for e in raw if e[2] is not None and isv(e[0]) and isv(e[1])]
    # external momenta: \node at (x,y) {\huge $l_1$}; etc.
    # attach to nearest vertex whose mode is COMPATIBLE with the external mode
    # (e.g. q1=H attaches to the H vertex, not to a nearby soft vertex)
    attach = {}
    for m in re.finditer(r'\\node\s*at\s*\(([-\d.]+),([-\d.]+)\)\s*\{\\huge\s*\$([\w]+)\$', src):
        lx, ly, name = float(m.group(1)), float(m.group(2)), m.group(3).replace('_', '')
        if not re.match(r'^(p|q|l)\d+$', name): continue
        em = extmode.get(name)
        best, bd = None, 1e9
        for v, md in verts.items():
            if em is not None:
                # compatible iff vertex mode is harder-or-equal external mode
                # (q1=H -> H vertex; p1=C1^inf -> C1^2 vertex; l1=SC4 -> SC4 vertex)
                if not harder_or_eq(md, em): continue
            d = (v[0]-lx)**2 + (v[1]-ly)**2
            if d < bd: best, bd = v, d
        if best is not None and bd < 4.0:
            attach[name] = best
    return verts, edges, attach, extmode

def infer_orange_comps(all_comps, verts, edges):
    """For each Orange (SC placeholder) component whose mode is (1,1,4) but which is NOT
    fed by l1 (no SC4 external attach) and sits between two C-mode components,
    infer SC_i = meet of the two adjacent C modes. Returns dict id(comp)->new mode."""
    relabel = {}
    for c in all_comps:
        if c['mode'] != (1, 1, 4): continue
        # adjacent C-type component modes (harder, direction != 0)
        cm = set()
        for d in all_comps:
            if d is c: continue
            if adjacent(c, d) and d['mode'][1] >= 1 and d['mode'][0] <= 1 and d['mode'][2] != 0:
                cm.add(d['mode'])
        if len(cm) == 2:
            cms = sorted(cm, key=lambda md: (md[0], md[1]))
            m = meet(cms[0], cms[1])
            if m[0] == 1 and m[1] >= 1:
                relabel[id(c)] = m
    return relabel

def is_region(fig, colormap, extmap, verbose=True):
    verts, edges, attach, extmode = parse(fig, colormap, extmap)
    all_comps = build_components(verts, edges)
    # relabel orange components whose SC_i is structure-determined (not SC4)
    relabel = infer_orange_comps(all_comps, verts, edges)
    for c in all_comps:
        if id(c) in relabel:
            c['mode'] = relabel[id(c)]
            if verbose:
                print(f"  [inferred] orange comp at {sorted(c['V'])} -> SC_{relabel[id(c)][2]}")
    if verbose:
        print(f"== {os.path.basename(fig)} ==")
        print(f"vertices: {len(verts)}, edges: {len(edges)}, components: {len(all_comps)}")
        for i, c in enumerate(all_comps):
            print(f"  γ{i}: mode={c['mode']} V={sorted(c['V'])} E={sorted(c['E'])}")
        print(f"attach: {attach}")
        for v, md in verts.items():
            inc = [e[2] for e in edges if v in (e[0], e[1])]
            inc += [em for nm, em in extmode.items() if attach.get(nm) == v]
            if inc:
                acc = inc[0]
                for mm in inc[1:]: acc = join(acc, mm)
                if not eq(acc, md):
                    print(f"  WARNING: vertex {v} color-mode {md} != join(incident) {acc}")
    confirmed = []
    order = []
    changed = True
    # Fundamental pattern: each jet J_i must be connected (or empty) — a
    # disconnected jet violates the pattern directly, before FC/IR compat
    # (2026-08-11).
    if not jet_connected_ok(verts, [md for (_, _, md) in edges],
                            [(u, v) for (u, v, _) in edges]):
        if verbose:
            print('  NOT FUNDAMENTAL PATTERN: some jet is disconnected')
        return False, []
    # (contracted-1VI filter removed 2026-08-21: mode components are 1VI
    # blocks from mode_components_wa — see ir_ok_blocks.  is_region keeps
    # build_components above for the orange relabel.)
    # Mojetic (Theorem 3 cond1): H∪J∖J_i must be 1VI after joining attached
    # external momenta to an aux vertex, for every jet direction i.
    mj_ok, mj_fail = cond1_ok([(u, v) for (u, v, _) in edges],
                              [md for (_, _, md) in edges], attach, extmode)
    if not mj_ok:
        if verbose:
            print(f"  NOT MOJETIC for i={mj_fail}: H∪J∖J_i not 1VI after aux connection")
        return False, []
    while changed:
        changed = False
        for g in all_comps:
            cc = check_conditions(g, confirmed, verts, edges, all_comps, attach, extmode)
            if cc is not None:
                confirmed.append(g); order.append((g, cc)); changed = True
                if verbose:
                    print(f"  confirm γ(mode={g['mode']}, V={sorted(g['V'])}, E={sorted(g['E'])}) via cond {cc}")
    ok = len(confirmed) == len(all_comps)
    if verbose:
        if ok:
            print("  -> ALL COMPONENTS IR COMPATIBLE (region candidate)")
        else:
            missing = [c for c in all_comps if c not in confirmed]
            print(f"  -> {len(missing)} component(s) NOT confirmed:")
            for c in missing:
                print(f"     mode={c['mode']} V={sorted(c['V'])} E={sorted(c['E'])}")
    order_simple = [(c['mode'], cc, tuple(sorted(c['V']))) for c, cc in order]
    return ok, order_simple

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('fig')
    ap.add_argument('--colors', required=True)
    ap.add_argument('--ext', required=True)
    args = ap.parse_args()
    colormap = dict(kv.split(':') for kv in args.colors.split())
    extmap = dict(kv.split(':') for kv in args.ext.split())
    ok, _ = is_region(args.fig, colormap, extmap)
    print("REGION" if ok else "NOT REGION")


# ---- all-entries cond1 (2026-09-07): a source may enter γ at several
# vertices; cond1_sources reported only the first BFS hit, which locked e.g.
# the S^1C_2^1 line momentum out of v4 and made cond1_strong reject genuine
# two-port C_i^n bridges (DivingBeetle kin1/3 etc.).
# cond1_strong_ae enumerates each source's reachable entries (va may equal
# vb, as the paper's cond ① allows).
