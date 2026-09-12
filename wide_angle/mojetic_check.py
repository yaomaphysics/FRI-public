#!/usr/bin/env python3
"""Mojetic check (Theorem 3 cond1 of the on-shell paper): for each external
direction i, the subgraph H∪J∖J_i must be mojetic — i.e. it becomes 1VI
(connected after deleting any single vertex) once all external momenta
attached to it are joined to an auxiliary vertex.

Classification (general definition):
  J_i  = union of all edges whose mode is C_i^n (n >= 1)
  H    = hard edges (mode (0,0,0))
  S    = everything else (S^m, S^m C_i^n)

Physical meaning: mojetic failure ⟺ some 1VI component of H∪J is attached
only via a single vertex / only by S lines / only by one jet direction —
all of which violate component-level momentum conservation (O(1) momenta
cannot be balanced by O(λ) flows). This is the graph-theoretic form of the
component-level momentum-conservation check that `momentum_ok` (vertex-level,
∨(in)=∨(out) only) misses.

Tested 2026-08-10: zero false kills on 7179 wide-angle regions;
exactly the 4 known hypercrown k0 extras fail.
"""
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


def cond2_ok(em, edges_in):
    """Theorem 3 cond2: every connected component of S connects >= 2 jets.
    NOTE (2026-08-10): NOT a valid general check — 23/118 of the legitimate
    5pt6loop regions violate it (SC/messenger structures legitimately touch a
    single jet). Kept for completeness/reference only; do NOT wire into the
    constructor."""
    J, H, S = classify(em, edges_in)
    if not S:
        return True, []
    jet_of = {}
    for i, es in J.items():
        for e in es:
            for v in e:
                jet_of.setdefault(v, set()).add(i)
    adj = {}
    for e in S:
        u, w = e
        adj.setdefault(u, set()).add(w)
        adj.setdefault(w, set()).add(u)
    seen = set()
    viol = []
    for v0 in list(adj):
        if v0 in seen:
            continue
        comp = set()
        stack = [v0]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            stack.extend(adj.get(x, ()))
        jets_touched = set()
        for v in comp:
            jets_touched |= jet_of.get(v, set())
        if len(jets_touched) < 2:
            viol.append((tuple(sorted(comp)), sorted(jets_touched)))
    return len(viol) == 0, viol
