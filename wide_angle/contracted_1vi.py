#!/usr/bin/env python3
"""Contracted mode component check (paper §3.2, 2026-08-14 .

Requirement: EACH contracted mode component is 1VI.

Definitions (paper §3.2):
  - mode X: one (m,n,i) tuple.
  - mode subgraph Γ_X = (V_X, E_X):
      V_X = {v : 𝒳(v) = X}   (vertices by JOIN mode: 𝒳(v) = ∨ of incident
             edge modes and attached external modes)
      E_X = {e : 𝒳(e) = X}   (edges by mode)
  - contracted X subgraph ~Γ_X: Γ_X + one auxiliary vertex, identifying the
    aux vertex with ALL vertices of other mode subgraphs adjacent to Γ_X.
    Implemented as the INDUCED subgraph of the full graph G on
    V_X ∪ aux_verts, with aux_verts contracted to a single aux vertex,
    where aux_verts = ∪{V_Y : Γ_Y adjacent to Γ_X} (adjacent = some edge
    of G joins a vertex of Γ_X to a vertex of Γ_Y).  Cross-mode edges
    between a component vertex and an aux vertex are kept (they carry the
    momentum flow into the harder background).
  - X-mode component: a connected component of ~Γ_X that is 1VI.

The check: for EVERY mode X present, EVERY connected component of ~Γ_X
(ignoring self-loops) must be 1VI (deleting any single vertex leaves it
connected).

Physical meaning: a mode component that is not 1VI after contraction has a
cut vertex — some part attaches to the rest only through one vertex, its
loop momenta are not pinned by any spanning structure, the leading F
polynomial factorises through the cut, and the region is scaleless.  This
is the graph-theoretic (polynomial-free) form of the "leading F depends on
x8,x9,x10 only through x8+x9+x10" obstruction found 2026-08-14 on
K33a3_2_nn kinematics2 (FRI 135 vs pySecDec 134).
"""
from collections import defaultdict


def _join(rc, modes):
    """Join of a list of modes (hardest common mode)."""
    modes = [m for m in modes if m is not None]
    if not modes:
        return (0, 0, 0)
    acc = modes[0]
    for m in modes[1:]:
        acc = rc.join(acc, m)
    return rc.norm(acc)


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
    if not _connected(verts, edges, aux_conn):
        return False
    for w in list(verts):
        vs = verts - {w}
        es = [(u, v) for (u, v) in edges if u != w and v != w]
        ac = aux_conn - {w}
        if not _connected(vs, es, ac):
            return False
    return True


def _connected(verts, edges, aux_conn=None):
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


def biconnected_blocks(verts, edges):
    """Biconnected components (blocks) of (verts, edges); bridges appear as
    single-edge blocks; self-loops as single-vertex blocks; isolated vertices
    as single-vertex blocks.  (Tarjan; identical algorithm to
    regge_limit/regge_core.mode_components — 2026-08-21: the Regge
    methodology decomposes γ̃_X into 1VI blocks instead of filtering.)"""
    loops = [(a, b) for (a, b) in edges if a == b]
    other = [(a, b) for (a, b) in edges if a != b]
    out = []
    for (a, b) in loops:
        out.append(({a}, [(a, b)]))
    adj = {v: [] for v in verts}
    for i, (a, b) in enumerate(other):
        adj[a].append((b, i)); adj[b].append((a, i))
    disc = {}; low = {}; t = 0; stack = []; blocks = []
    def dfs(u, pe):
        nonlocal t
        disc[u] = low[u] = t; t += 1
        for (w, ei) in adj[u]:
            if ei == pe: continue
            if w not in disc:
                stack.append(ei)
                dfs(w, ei)
                low[u] = min(low[u], low[w])
                if low[w] >= disc[u]:
                    blk = set()
                    while True:
                        e = stack.pop(); blk.add(e)
                        if e == ei: break
                    blocks.append(blk)
            elif disc[w] < disc[u]:
                stack.append(ei)
                low[u] = min(low[u], disc[w])
    for v in verts:
        if v not in disc:
            dfs(v, -1)
            if stack:
                blocks.append(set(stack)); stack.clear()
    for blk in blocks:
        be = [other[i] for i in blk]
        out.append(({v for e in be for v in e}, be))
    used = {v for _, be in out for e in be for v in e}
    for v in verts:
        if v not in used:
            out.append(({v}, []))
    return out


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


def mode_components_wa(edges_in, em, ext_attach, ext_mode, rc=None):
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
    if rc is None:
        import region_checker as rc
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
    vjoin = {v: _join(rc, ms) for v, ms in inc.items()}
    emodes = defaultdict(list)
    for e, md in zip(edges_in, em):
        m = md if md is not None else (0, 0, 0)
        emodes[m].append(e)
    aux = ('aux',)
    comps = []
    for X, ex in emodes.items():
        vx = {v for v, jm in vjoin.items() if rc.eq(jm, X)}
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
            if rc.eq(Y, X):
                w = b if a_in else a
                aux_verts.add(w)
        edges2 = []
        orig2 = []
        for e2 in edges_in:
            a, b = e2
            Y = emap.get(e2)
            if Y is None:
                continue
            if rc.eq(Y, X):
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
                    if Y is not None and rc.eq(Y, X):
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


def contracted_1vi_ok(edges_in, em, ext_attach, ext_mode, rc=None):
    """Check that every contracted mode component is 1VI.

    edges_in: list of (u,v) internal edges (order matches em).
    em: list of edge mode tuples (m,n,i) or None (= H).
    ext_attach: {external-name: vertex}.
    ext_mode: {external-name: mode tuple} (may be None → no externals joined).
    rc: region_checker module (injected for tests); imported lazily.
    Returns (ok, failures) where failures = list of (mode, comp) for
    non-1VI contracted components."""
    if rc is None:
        import region_checker as rc

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
    vjoin = {v: _join(rc, ms) for v, ms in inc.items()}

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
        vx = {v for v, jm in vjoin.items() if rc.eq(jm, X)}
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
                if rc.eq(Y, X):
                    w = b if a_in else a
                    Yw = vjoin.get(w)
                    if Yw is not None:
                        for z, jm in vjoin.items():
                            if rc.eq(jm, Yw):
                                aux_verts.add(z)
                else:
                    for z, jm in vjoin.items():
                        if rc.eq(jm, Y):
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
                        mY, nY, iY = rc.norm(ext_mode[name])
                        if mY == mX and iY == iX and nY >= nX:
                            cedges.append((v, ('aux', X)))
            for comp in _components(cverts, cedges):
                if not _is_1vi(comp, cedges):
                    failures.append((X, comp))
    return len(failures) == 0, failures
