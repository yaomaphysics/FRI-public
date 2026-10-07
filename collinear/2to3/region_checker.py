#!/usr/bin/env python3
# region_checker.py — collinear 2->3 FRI enumerator.
#
# Pipeline: cuts
#   -> overlay (em/vm)
#   -> fundamental pattern (momentum conservation, jets, H-region, mojetic, island)
#   -> IR compatibility (fixpoint over three conditions:
#        condition 1 = partial-sum vee of inflows;
#        condition 2 = messenger (incl. the SC23 special messenger);
#        condition 3 = meet-of-two).
#
# Mode algebra and graph tools: see primitives.py (zero-judgment base layer).
# Run:  python3 region_checker.py [k0..k4] [-v]   (built-in example; default k1)
from itertools import combinations

_DBG = False
CONDUCT23 = True  # S^mC23 hidden-path conduction (general m; approved 2026-09-19 15:15; supersedes the 03:30 bridge prototype).  Set False to disable.

from primitives import (H, S, W, P, isS, isC, m_of, d_of, n_of, mem_of, V, name, parse,
    meet, join, connected, is_1vi, components, find_1vi_blocks,
    marginally_softer, harder_or_eq, INF)

def mode_components(mode, vm, em, edges, verts):
    # 1VI blocks of the contracted mode subgraph.
    gv = {v for v in verts if vm.get(v) == mode}
    ge = [e for e, m in zip(edges, em) if m == mode]
    if not gv and not ge:
        return []
    ge_idx = [i for i, m in enumerate(em) if m == mode]
    verts2 = set(gv) | {'aux'}
    edges2 = []
    for (a, b) in ge:
        edges2.append(('aux' if a not in gv else a, 'aux' if b not in gv else b))
    raw = find_1vi_blocks(verts2, edges2)
    out = []
    used = set()
    for (bv, be) in raw:
        idxs = []
        for ce in be:
            for j, c2 in enumerate(edges2):
                if j in used or c2 != ce: continue
                used.add(j); idxs.append(ge_idx[j]); break
        out.append((bv, be, idxs))
    return out

# ============================ overlay ============================
CUT_MODES = {'C23': P(0, 0), 'C1': W(1, 1), 'C4': W(4, 1), 'C5': W(5, 1), 'C1R1': W(1, 2), 'C4R1': W(4, 2), 'C5R1': W(5, 2), 'C2R1': P(1, 2), 'C3R1': P(1, 3), 'C2R2': P(2, 2), 'C3R2': P(2, 3)}
_CUT_NAMES = tuple(CUT_MODES)   # fixed iteration order for build_overlay

def build_overlay(edges, verts, ext_attach, ext_mode, cuts, vm_seen=None):
    # cuts: {name: frozenset}. Returns (em, vm) or (None, reason).
    # Vertex-first construction: vm(v) = meet of the modes of the nonempty cuts covering v (no covering cut -> H); then em(e) = vm(u) ∧ vm(w).
    # vm_seen (optional, 2026-09-18): vm-level dedup.  em is a function of vm and
    # every downstream check depends only on (em, vm) => the outcome is a
    # function of vm alone; a repeated vm can return early (before the edge
    # meets and all checks).  Callers pass a per-enumeration set.
    # per-vertex meets built in one pass over the cuts; the vertex loop
    # below just reads them off
    acc_map = {}
    _cm = CUT_MODES
    _meet = meet
    for nm in _CUT_NAMES:
        S = cuts.get(nm)
        if not S:
            continue
        m = _cm[nm]
        for v in S:
            acc = acc_map.get(v)
            if acc is None:
                acc_map[v] = m
            else:
                try:
                    acc_map[v] = _meet(acc, m)
                except ArithmeticError as e:
                    return None, f'meet:{e}'
    vm = {}
    _H = H
    for v in verts:
        acc = acc_map.get(v)
        vm[v] = acc if acc is not None else _H()
    if vm_seen is not None:
        # fast path (2026-09-22): fixed vertex order (callers pass the
        # sorted V); same equality classes as the old frozenset-of-pairs key
        vkey = tuple(map(vm.__getitem__, verts))
        if vkey in vm_seen:
            return None, 'vm-dup'
        vm_seen.add(vkey)
    em = []
    _ap = em.append
    for (a, b) in edges:
        try:
            _ap(_meet(vm[a], vm[b]))
        except ArithmeticError as e:
            return None, f'edge-meet:{e}'
    return (em, vm), 'ok'

# ============================ fundamental pattern ============================
# jet family tag: J1/J4/J5 (wide legs), J23 (pair), S, H.
def fam_tag(x):
    if x[0] == 'H': return 'H'
    if x[0] == 'S': return 'S'
    _, d, n, mem, m = x
    if d in (1, 4, 5):
        return f'J{d}'
    return 'J23'

FAM_EXT = {'J1': ['p1'], 'J4': ['p4'], 'J5': ['p5'], 'J23': ['p2', 'p3']}

# Momentum conservation per vertex: exists a partition of the incident momenta with ∨(A) == ∨(B).
def momentum_ok(edges, verts, ext_attach, ext_mode, em, vm):
    for v in sorted(verts):
        inc = [em[i] for i, (a, b) in enumerate(edges) if a == v or b == v]
        for nm, vv in ext_attach.items():
            if vv == v: inc.append(ext_mode[nm])
        n = len(inc)
        if n < 2: return False
        # Momentum conservation at each vertex follows one single rule:
        # there exists a partition with ∨(A) == ∨(B) (for two incident momenta this reduces to equality).
        ok = False
        for r in range(1, n):
            for A in combinations(range(n), r):
                B = [i for i in range(n) if i not in A]
                try:
                    vA = acc_join([inc[i] for i in A])
                    vB = acc_join([inc[i] for i in B])
                except ArithmeticError:
                    continue
                if vA == vB:
                    ok = True; break
            if ok: break
        if not ok:
            return False
    return True

# left-fold join over a list of modes.
def acc_join(modes):
    acc = modes[0]
    for m2 in modes[1:]:
        acc = join(acc, m2)
    return acc

# tightened connectivity for jet pieces (2026-09-19): pieces connect only
# through a vertex that is itself of the same family; every resulting component
# must touch the family's external leg(s).  (Replaces the looser any-endpoint
# connectivity, which merged pieces through far/ H vertices — k3 trio fix.)
def _tight_jet_ok(je, vm, tag, extvs):
    if not je: return True
    n = len(je); adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            shared = set(je[i]) & set(je[j])
            if any(fam_tag(vm[v]) == tag for v in shared):
                adj[i].add(j); adj[j].add(i)
    seen = set()
    for i in range(n):
        if i in seen: continue
        comp = [i]; st = [i]; seen.add(i)
        while st:
            u = st.pop()
            for w in adj[u]:
                if w not in seen:
                    seen.add(w); comp.append(w); st.append(w)
        vs = set()
        for k in comp: vs |= set(je[k])
        if not (vs & extvs): return False
    return True

# jet structure per family; returns (ok, jvje).
def jets_ok(edges, verts, vm, em, ext_attach):
    jvje = {}
    for tag in ('J1', 'J4', 'J5', 'J23'):
        je = [e for e, m in zip(edges, em) if fam_tag(m) == tag]
        jv = {v for v in verts if fam_tag(vm[v]) == tag} | {v for e in je for v in e}
        jvje[tag] = (jv, je)
        names = FAM_EXT[tag]
        extvs = {ext_attach[n] for n in names}
        if not _tight_jet_ok(je, vm, tag, extvs):
            return False, (tag, 'jet')
    return True, jvje

# 2026-09-23: the subgraph H ∪ C23 must be connected ("C23" = the whole
# C23 mode subgraph, not the cut): the H- and C23-mode elements (vertices
# plus their mode edges) must form one connected component.
def h_c23_connected_ok(edges, verts, em, vm):
    hv = {v for v in verts if vm.get(v) == H()}
    cv = {v for v in verts if vm.get(v) == P(0, 0)}
    vs = hv | cv
    es = []
    for e, m in zip(edges, em):
        if m == H() or m == P(0, 0):
            es.append(e)
            vs.update(e)
    return connected(vs, es)

# the subgraph outside all cuts (H) must be nonempty and connected.
def uncovered_ok(edges, verts, cuts):
    # one fewer temp set; inline BFS for the pre-filtered (hv, he)
    # (he endpoints are a subset of hv by construction — no re-check needed).
    covered_v = set()
    for S in cuts.values():
        if S:
            covered_v.update(S)
    hv = set(verts)
    hv.difference_update(covered_v)
    if not hv:
        return False
    he = [(a, b) for (a, b) in edges if a not in covered_v and b not in covered_v]
    if len(hv) <= 1:
        return True
    adj = {}
    for (a, b) in he:
        if a in adj:
            adj[a].append(b)
        else:
            adj[a] = [b]
        if b in adj:
            adj[b].append(a)
        else:
            adj[b] = [a]
    seen = {next(iter(hv))}
    st = list(seen)
    while st:
        v = st.pop()
        for u in adj.get(v, ()):
            if u not in seen:
                seen.add(u)
                st.append(u)
    return len(seen) == len(hv)

# 1VI test on the contracted (outside + jets + aux) graph.
def mojetic_ok(hv, he, jv, je, edges, ext_attach, ext_names):
    vs = set(hv) | set(jv) | {'aux'}
    es = set(he) | set(je)
    for n in ext_names:
        vv = ext_attach[n]
        if vv in vs:
            es.add((vv, 'aux'))
    return is_1vi(vs, es)

# mojetic test for every jet-family skip.
def mojetic_all_ok(edges, verts, vm, em, ext_attach, jvje):
    moded = {}
    for e, m in zip(edges, em):
        moded[e] = m
    jvje2 = {}
    for tag, (jv, je) in jvje.items():
        je2 = [e for e in je if m_of(moded[e]) == 0]
        jv2 = {v for e in je2 for v in e}
        for v in verts:
            if fam_tag(vm[v]) == tag and m_of(vm[v]) == 0:
                jv2.add(v)
        jvje2[tag] = (jv2, je2)
    hv_m = {v for v in verts if vm[v] == H()}
    he_m = [e for e, m in zip(edges, em) if m == H()]
    for skip in ('J1', 'J4', 'J5', 'J23'):
        jv = set(); je = []
        names = []
        for tag in ('J1', 'J4', 'J5', 'J23'):
            if tag == skip: continue
            a, b = jvje2[tag]
            jv |= a; je += list(b); names += FAM_EXT[tag]
        if not mojetic_ok(hv_m, he_m, jv, je, edges, ext_attach, names):
            return False
    return True

# C23-mode "islands" whose incident modes are all at higher levels are rejected.
# Components = connected components of the exact-mode subgraph; adjacency = the modes of the edges leaving the component.
def island_ok(edges, verts, em, vm):
    mode = P(0, 0)
    gv = frozenset(v for v in verts if vm.get(v) == mode)
    if not gv:
        return True
    cmap = components(tuple((a, b) for i, (a, b) in enumerate(edges) if em[i] == mode), gv)
    for root in sorted(set(cmap.values())):
        comp = frozenset(v for v in gv if cmap[v] == root)
        adjacent = [em[i] for i, (a, b) in enumerate(edges) if (a in comp) != (b in comp)]
        if adjacent and all(V(m) > V(mode) for m in adjacent):
            return False
    return True

# ---- switches (all set to the validated settings) ----
# Enforce the condition 1 third-port check (see the third-port block below).
USE_THIRD_PORT = True
# C23 cut: every component must contain v2 or v3; for all-INF kinematics this reduces to a connected cut containing both incident roots.
# ============================ IR compatibility ============================
# relevance (path version): marginally softer + monotone path (V non-increasing), no pass-through H.
def relevant(src_blk, dst_blk, dst_mode, edges, em, vm, src_mode):
    if not marginally_softer(src_mode, dst_mode): return False
    bv, be, *rest = src_blk
    sidx = rest[0] if rest else None
    cv = dst_blk[0]
    src = {v for v in bv if v != 'aux'}
    tgt = {v for v in cv if v != 'aux'}
    if not tgt: return False
    # Direct touch = degenerate relevant path: the source's OWN carrier (its vertices, or its own lines)
    # touching tgt — no global scan.
    # (Relevance is required — not mere adjacency.)
    if src & tgt: return True
    _ends = []
    if sidx:
        for ei in sidx:
            _ends += list(edges[ei])
    if not _ends:
        for (a, b) in be:
            _ends += [a, b]
    for x in _ends:
        if x in tgt: return True
    # external-momentum component (single edge (v,aux))
    if len(be) == 1 and be[0][1] == 'aux' and len(src) == 1:
        v_ext = next(iter(src))
        if v_ext in tgt: return True
    if not src:
        for ei, (a, b) in enumerate(edges):
            if em[ei] != src_mode: continue
            if sidx and ei not in sidx: continue
            if a != 'aux': src.add(a)
            if b != 'aux': src.add(b)
    if not src: return False
    # monotone BFS: V non-increasing along the path; no pass-through H
    adj = {}
    for v in vm: adj[v] = []
    for i, (a, b) in enumerate(edges):
        adj[a].append((b, i)); adj[b].append((a, i))
    seen = set(src)
    stack = [(v, V(vm.get(v, H()))) for v in src]
    while stack:
        v, last_V = stack.pop()
        for w, ei in adj[v]:
            if w in seen: continue
            if vm.get(w) == H(): continue
            e_V = V(em[ei])
            if e_V > last_V: continue
            w_V = V(vm.get(w, H()))
            if w_V > e_V: continue
            if w in tgt: return True
            seen.add(w); stack.append((w, w_V))
    return False

def cond1_confirms(blk, mode, comps, confirmed, edges, em, vm, ext_attach, ext_mode, dbg=None):
    # condition 1: exists a cut of the block such that vee(inflows) == mode.
    bv, be, *rest = blk
    verts = [v for v in bv if v != 'aux']
    n = len(verts)
    if n == 0: return False
    # reachability inside the block along the cut mode.
    def _reachable(v, A):
        if vm.get(v) == H(): return False
        seen = {v}; st = [v]
        while st:
            x = st.pop()
            if x in A: return True
            for ei, (a, b) in enumerate(edges):
                if em[ei] != mode: continue
                w = b if a == x else (a if b == x else None)
                if w is None or w in seen: continue
                if vm.get(w) == H(): continue
                seen.add(w); st.append(w)
        return False
    # inflows of a piece (externals + confirmed-component flows).
    def inflows_of(A):
        inflow = []
        for extn, vv in ext_attach.items():
            m_ext = ext_mode[extn]
            if vv in A:
                inflow.append(m_ext); continue
            if m_ext == mode:
                if _reachable(vv, A): inflow.append(m_ext)
                continue
            ext_comp = ({vv, 'aux'}, [(vv, 'aux')], [])
            tgt = ({v for v in A}, [])
            if relevant(ext_comp, tgt, mode, edges, em, vm, m_ext):
                inflow.append(m_ext)
        for ei, (a, b) in enumerate(edges):
            m = em[ei]
            if m == H(): continue
            for i, comp in enumerate(comps.get(m, ())):
                if not confirmed.get((m, i)): continue
                idxs = comp[2] if len(comp) > 2 else None
                if idxs and ei not in idxs: continue
                if m == mode:
                    if a in A or b in A: inflow.append(m)
                else:
                    line_comp = ({a, b}, [(a, b)], [])
                    tgt = ({v for v in A}, [])
                    if relevant(line_comp, tgt, mode, edges, em, vm, m):
                        inflow.append(m)
                break
        return inflow
    # the third-port requirement is part of condition 1: two sources need an exit port for the outgoing flow.
    def _port_ok():
        return port_ok_pairs(blk, mode, edges, em, vm, ext_attach, ext_mode, confirmed, comps, dbg)
    if n == 1 or not any((a in set(verts)) != (b in set(verts)) for (a, b) in be):
        inflow = inflows_of(set(verts))
        if not inflow: return False
        acc = inflow[0]
        for m2 in inflow[1:]:
            try: acc = join(acc, m2)
            except ArithmeticError: return False
        return acc == mode and _port_ok()
    for mask in range(1, 1 << n):
        A = {verts[i] for i in range(n) if (mask >> i) & 1}
        if not A: continue
        if not any((a in A) != (b in A) for (a, b) in be): continue
        inflow = inflows_of(A)
        if not inflow: continue
        acc = inflow[0]
        for m2 in inflow[1:]:
            try: acc = join(acc, m2)
            except ArithmeticError: return False
        if acc == mode: return _port_ok()
    return False

# Note a subtlety in applying condition 1: we need to make sure that the momenta "..." in vee(...) must be flowing into the mode component.
# To realize this, we use the "third-port" check: besides the two vertices for the two incoming momenta (which can coincide), there must exist a third vertex to for the outgoing momentum flow.
#   (A) as a same-mode vertex shared with another same-mode component, or
#   (B) as a harder-mode vertex reached by one of the block's OWN edges.
# A scale assembled from two sources with no exit port is scaleless.
def _collect_entries(blk, start_vs, start_es, cur_mode, edges, em, vm):
    # Element-level first-ENTRY collection: vertices AND edges are elements; walk with monotone harder_or_eq steps;
    # when an element of the block is reached it is a HIT and the walk does not propagate inside the block.
    # Entries = hit vertices / endpoints of hit edges that belong to the block
    # (the entry is the connection point where the monotone path enters the component).
    realV = {v for v in blk[0] if v != 'aux'}
    tgt = {('v', v) for v in realV}
    tgt |= {('e', ei) for ei in (blk[2] if len(blk) > 2 else [])}
    # mode of a walk element (vertex or edge).
    def mode_of(el):
        return vm.get(el[1], H()) if el[0] == 'v' else em[el[1]]
    # walk neighbours of an element (edge <-> its endpoints).
    def neighbors(el):
        out = []
        if el[0] == 'v':
            v = el[1]
            for ei, (a, b) in enumerate(edges):
                if a == v:
                    out.append(('e', ei)); out.append(('v', b))
                elif b == v:
                    out.append(('e', ei)); out.append(('v', a))
        else:
            a, b = edges[el[1]]
            out.append(('v', a)); out.append(('v', b))
        return out
    start = [('v', v) for v in start_vs] + [('e', ei) for ei in start_es]
    visited = set(start)
    queue = [(el, cur_mode) for el in start]
    hits = set()
    while queue:
        el, cur = queue.pop(0)
        if el in tgt:
            hits.add(el); continue
        for nb in neighbors(el):
            if nb in visited: continue
            if harder_or_eq(mode_of(nb), cur):
                visited.add(nb); queue.append((nb, mode_of(nb)))
    ev = set()
    for h in hits:
        if h[0] == 'v':
            if h[1] in realV: ev.add(h[1])
        else:
            for w in edges[h[1]]:
                if w in realV: ev.add(w)
    return ev

# third-port test: (A) same-mode vertex shared with another component, (B) harder vertex via the block's own edge;
# returns (tag, w) or None.
def third_port(blk, X, entry_vs, comps_same, vm, edges):
    realV = {v for v in blk[0] if v != 'aux'}
    for w in sorted(realV):
        if w in entry_vs: continue
        if vm.get(w) != X: continue
        for c in comps_same:
            if c is blk: continue
            if w in {v for v in c[0] if v != 'aux'}:
                return ('A', w)
    idxs = blk[2] if len(blk) > 2 else []
    for ei in idxs:
        a, b = edges[ei]
        for w in (a, b):
            if w in realV or w in entry_vs: continue
            mw = vm.get(w)
            if mw is not None and mw != X and harder_or_eq(mw, X):
                return ('B', w)
    return None

def _outside_contacts(blk, X, v0, edges, em, vm):
    # Entry candidates of an outside source: element walk from its attach
    # vertex with the vertex's own mode as the starting mode.
    return _collect_entries(blk, [v0], [], vm.get(v0, H()), edges, em, vm)

def _port_sources(blk, X, edges, em, vm, ext_attach, ext_mode, confirmed, comps):
    # Contact-based sources for the third-port check: externals (always) + confirmed components
    # with a marginally softer mode; entries are contacts only.
    srcs = []
    realV = {v for v in blk[0] if v != 'aux'}
    for extn, v0 in ext_attach.items():
        md = ext_mode[extn]
        if v0 in realV:
            srcs.append((extn, md, {v0}))
        else:
            if md == X or marginally_softer(md, X):
                c = _outside_contacts(blk, X, v0, edges, em, vm)
                if c:
                    srcs.append((extn, md, c))
    for (cm, ci) in confirmed:
        if not marginally_softer(cm, X):
            continue
        sblk = comps[cm][ci]
        sv = {v for v in sblk[0] if v != 'aux'}
        se = sblk[2] if len(sblk) > 2 else []
        c = _collect_entries(blk, sv, se, cm, edges, em, vm)
        if c:
            srcs.append(('%s#%d' % (name(cm), ci), cm, c))
    return srcs

def port_ok_pairs(blk, X, edges, em, vm, ext_attach, ext_mode, confirmed, comps, dbg=None):
    # Third-port check over contact-based source pairs (join == X).
    if not USE_THIRD_PORT:
        return True
    srcs = _port_sources(blk, X, edges, em, vm, ext_attach, ext_mode, confirmed, comps)
    comps_same = comps.get(X, [])
    for (_tag, md, cands) in srcs:
        if md == X:
            for a in cands:
                if third_port(blk, X, {a}, comps_same, vm, edges):
                    return True
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            md1, c1 = srcs[i][1], srcs[i][2]
            md2, c2 = srcs[j][1], srcs[j][2]
            try:
                if join(md1, md2) != X:
                    continue
            except ArithmeticError:
                continue
            for a in c1:
                for b in c2:
                    if third_port(blk, X, {a, b}, comps_same, vm, edges):
                        return True
    if dbg is not None:
        dbg.setdefault('port_fail', []).append((name(X), [(t, sorted(c)) for (t, _m, c) in srcs]))
    return False

# depth index used by the messenger check.
def depth_of(x):
    if x[0] != 'C': return 0
    _, d, n, mem, m = x
    if d in (1, 4, 5):
        return INF if n == INF else n
    return INF if n == INF else (n + 1)

def _real_verts(blk):
    return {v for v in blk[0] if v != 'aux'}

def _comp_adjacent(a_blk, b_blk, edges):
    # vertex-contact adjacency (WA `adjacent` semantics): an edge of one block has an endpoint
    # that is an OWN (real) vertex of the other block.  This is what blocks BORROWING (e.g. a
    # member hanging on a third component's cut vertex does not attach).
    bv = _real_verts(b_blk)
    if bv:
        for ei in (a_blk[2] if len(a_blk) > 2 else []):
            u, w = edges[ei]
            if u in bv or w in bv: return True
    av = _real_verts(a_blk)
    if av:
        for ei in (b_blk[2] if len(b_blk) > 2 else []):
            u, w = edges[ei]
            if u in av or w in av: return True
    return False

def messenger_targets(blk, mode, comps, edges, em, vm, ext_attach, ext_mode, debug=None, kernel=None, kernel_blocks=None):
    # Collect (dirs, targets) for the messenger check of the S^m block blk — FULL version
    # (2026-09-19, WA-aligned; kernel clouds added 2026-09-23 per the 2026-09-04 rules 1+2).
    #   Gamma^[m] = connected closure (vertex-contact) of all soft-power-m components around
    #   the kernel (S^m and S^m C^n members alike; kernel = the whole connected pure-S^m cloud
    #   when given — several 1VI blocks joined through shared real S^m vertices).
    #   A target gamma_i counts iff the n_i-matched Gamma member is relevant to it:
    #     n_i = 0  -> the KERNEL (the whole cloud may serve; no borrowing from other S blocks);
    #     n_i >= 1 -> an S^m C_i^{n_i} member ADJACENT to the kernel (cloud).
    m = mode[1]
    _kern = kernel if kernel is not None else blk
    pool = []
    for md, lst in comps.items():
        if m_of(md) != m: continue
        for i, b in enumerate(lst):
            pool.append((md, i, b))
    gmemb = [False] * len(pool)
    _seeds = kernel_blocks if kernel_blocks else (blk,)
    for j in range(len(pool)):
        if any(pool[j][2] is _kb for _kb in _seeds): gmemb[j] = True
    changed = True
    while changed:
        changed = False
        for j in range(len(pool)):
            if gmemb[j]: continue
            for k in range(len(pool)):
                if gmemb[k] and _comp_adjacent(pool[j][2], pool[k][2], edges):
                    gmemb[j] = True; changed = True; break
    gids = {id(pool[j][2]) for j in range(len(pool)) if gmemb[j]}
    dirs = set()
    targets = []
    for md, lst in comps.items():
        for i, d in enumerate(lst):
            if id(d) in gids: continue
            mi = m - m_of(md)
            if not (1 <= mi <= m): continue
            depth = depth_of(md)
            if depth == INF: continue
            n_i = depth - mi
            if n_i < 0: continue
            via = None
            if n_i == 0:
                if relevant(_kern, d, md, edges, em, vm, mode):
                    via = 'kernel'
            else:
                for j in range(len(pool)):
                    if not gmemb[j]: continue
                    md2, i2, b2 = pool[j]
                    if depth_of(md2) != n_i: continue
                    if d_of(md2) != d_of(md): continue
                    if not _comp_adjacent(b2, _kern, edges): continue
                    if relevant(b2, d, md, edges, em, vm, md2):
                        via = name(md2)
                        break
            if via is None: continue
            ok_ext = True
            for extn, vv in ext_attach.items():
                if vv in {v for v in d[0] if v != 'aux'}:
                    e = ext_mode[extn]
                    if d_of(e) is not None and d_of(e) == d_of(md) and depth_of(e) >= depth_of(md):
                        continue
                    try:
                        if V(e) <= V(md):
                            continue
                    except Exception:
                        pass
                    ok_ext = False; break
            if not ok_ext: continue
            targets.append((md, i))
            if d_of(md) is not None:
                dirs.add(d_of(md))
            if debug is not None:
                debug.append((name(md), i, n_i, via))
    return dirs, targets

def messenger_ok(blk, mode, comps, confirmed, edges, em, vm, ext_attach, ext_mode, info=None, kernel=None, kernel_blocks=None):
    # S^m messenger check — full Gamma^[m] form (WA-aligned; kernel + n_i-matched members,
    # connected closure; >=3 distinct directions; >=1 confirmed-relevant).
    dirs, targets = messenger_targets(blk, mode, comps, edges, em, vm, ext_attach, ext_mode, kernel=kernel, kernel_blocks=kernel_blocks)
    if info is not None:
        info.setdefault('messenger', []).append((name(mode), sorted(dirs)))
    if len(dirs) < 3: return False
    # need >=1 confirmed relevant comp
    for (cm, ci) in list(confirmed):
        cblk = comps[cm][ci]
        if relevant(blk, cblk, cm, edges, em, vm, mode):
            return True
    return False

# IR compatibility — recursive fixpoint over three conditions; every non-H component must be confirmed:
#   [condition 1] the partial sum of the momenta entering the component (externals + confirmed components) is of its mode (third-port included);
#   [condition 2] the component is a messenger (S^m; or the SC23 special messenger of this kinematics);
#   [condition 3] the component mode is the meet of two confirmed components relevant to it.
def ir_ok(edges, verts, em, vm, ext_attach, ext_mode, dbg=None):
    comps = {}
    modes_present = sorted(set(em) | set(vm.values()))
    for m in modes_present:
        comps[m] = mode_components(m, vm, em, edges, verts)
    # drop empty
    comps = {m: c for m, c in comps.items() if c}
    # 2026-09-04 rules 1+2 (ported 2026-09-23; WA primitives): kernel clouds —
    # blocks of the same pure-soft S^m mode joined (transitively) through shared
    # REAL S^m vertices form ONE kernel; a messenger confirmation covers every
    # confirmable block of that cloud (rule 2).
    _cloud_of = {}
    _pure_by = {}
    for _md, _lst in comps.items():
        if isS(_md):
            _pure_by[_md] = _lst
    for _md, _blocks in _pure_by.items():
        if len(_blocks) == 1:
            _cloud_of[id(_blocks[0])] = _blocks
            continue
        _par = {id(b): id(b) for b in _blocks}

        def _uf(i, _par=_par):
            while _par[i] != i:
                _par[i] = _par[_par[i]]
                i = _par[i]
            return i

        _vown = {}
        for _b in _blocks:
            for _v in _b[0]:
                if _v == 'aux': continue
                if _v in _vown:
                    _ra, _rb = _uf(id(_b)), _uf(_vown[_v])
                    if _ra != _rb: _par[_ra] = _rb
                else:
                    _vown[_v] = id(_b)
        _roots = {}
        for _b in _blocks:
            _roots.setdefault(_uf(id(_b)), []).append(_b)
        for _lst2 in _roots.values():
            for _b in _lst2:
                _cloud_of[id(_b)] = _lst2

    def _make_kern(blk):
        _lst = _cloud_of.get(id(blk))
        if not _lst or len(_lst) == 1:
            return None, None
        _V = set()
        _I = []
        for _b in _lst:
            _V |= {v for v in _b[0] if v != 'aux'}
            if len(_b) > 2:
                _I += list(_b[2])
        return (_V, [tuple(edges[_i]) for _i in _I], _I), _lst
    confirmed = {}
    # tadpole gate (2026-09-12, ported from wide-angle primitives.py).
    # A soft-containing component (m >= 1) adjacent to NO harder-mode component is already a 1VI block of its own mode subgraph;
    #   it can then only be confirmed by cond 1 — unless it is pure soft carrying two attached momenta whose join is exactly its mode (rule 3), in which case cond 2/3 are allowed again.
    # Hard and jet modes are exempt.
    def _tadp(md, blk):
        if m_of(md) < 1:
            return False
        realV = {v for v in blk[0] if v != 'aux'}
        idxs = blk[2] if len(blk) > 2 else []
        ends = set(realV)
        for ei in idxs:
            ends |= set(edges[ei])
        for w in ends:
            if vm.get(w) == H():
                return False
        def _adj(a_idxs, a_v, b_idxs, b_v):
            for ei in a_idxs:
                for w in edges[ei]:
                    if w in b_v:
                        return True
            for ei in b_idxs:
                for w in edges[ei]:
                    if w in a_v:
                        return True
            return False
        for md2, lst2 in comps.items():
            if V(md2) >= V(md):
                continue
            for c2 in lst2:
                c2v = {v for v in c2[0] if v != 'aux'}
                c2i = c2[2] if len(c2) > 2 else []
                if _adj(idxs, realV, c2i, c2v):
                    return False
        return True

    def _vee_ok(md, blk):
        # Rule 3: two momenta attached to the component — carrier line
        # modes (collinear part) or external momenta — with join equal to
        # its mode.  Pure-soft lines cannot be the scale source of a soft
        # blob and are excluded.
        realV = {v for v in blk[0] if v != 'aux'}
        idxs = blk[2] if len(blk) > 2 else []
        ends = set(realV)
        for ei in idxs:
            ends |= set(edges[ei])
        pool = []
        for i2 in range(len(edges)):
            u, w = edges[i2]
            if (u in ends or w in ends) and isC(em[i2]):
                pool.append(em[i2])
        for nm, v in ext_attach.items():
            if v in ends:
                pool.append(ext_mode[nm])
        if len(pool) < 2:
            return False
        for a in range(len(pool)):
            for b in range(a + 1, len(pool)):
                if join(pool[a], pool[b]) == md:
                    return True
        return False

    _tad_cache = {}
    _vee_cache = {}

    def _cond_allowed(md, blk):
        t = _tad_cache.get(id(blk))
        if t is None:
            t = _tadp(md, blk)
            _tad_cache[id(blk)] = t
        if not t:
            return True
        if not isS(md):
            return False        # SC tadpoles keep the cond 2/3 ban
        ok = _vee_cache.get(id(blk))
        if ok is None:
            ok = _vee_ok(md, blk)
            _vee_cache[id(blk)] = ok
        return ok

    # record a fresh confirmation.
    def try_confirm(mode, i):
        if (mode, i) in confirmed: return False
        confirmed[(mode, i)] = True
        if _DBG: print('   CONFIRM %s#%d' % (name(mode), i))
        return True
    if _DBG: print('-- initial cond1 --')
    # initial: condition 1 for every block (vee of inflows == mode; the C23 blocks also require third-port).
    for md, lst in comps.items():
        if md[0] == 'H': continue
        for i, blk in enumerate(lst):
            if not cond1_confirms(blk, md, comps, confirmed, edges, em, vm, ext_attach, ext_mode, dbg):
                continue
            try_confirm(md, i)
    changed = True
    guard = 0
    while changed and guard < 40:
        changed = False; guard += 1
        if _DBG: print('== iter %d ==' % guard)
        if _DBG: print('  -- cond2 S-messenger --')
        # [condition 2] S messengers.
        for sm, lst in comps.items():
            if sm[0] != 'S': continue
            for i, blk in enumerate(lst):
                if (sm, i) in confirmed: continue
                if not _cond_allowed(sm, blk): continue
                _kern, _kbs = _make_kern(blk)
                if messenger_ok(blk, sm, comps, confirmed, edges, em, vm, ext_attach, ext_mode, dbg, kernel=_kern, kernel_blocks=_kbs):
                    if try_confirm(sm, i):
                        changed = True
                        # rule 2 (2026-09-04): a messenger confirmation covers every
                        # confirmable S^m block of the kernel cloud.
                        if _kbs:
                            _kids = {id(b) for b in _kbs}
                            for j2, kb in enumerate(lst):
                                if id(kb) in _kids and (sm, j2) not in confirmed and _cond_allowed(sm, kb):
                                    changed |= try_confirm(sm, j2)
        if _DBG: print('  -- cond2 special-messenger --')
        # [condition 2] special messenger for the 23-collinear kinematics: simultaneously relevant to a C2C23 component, a C3C23 component, and a C_i component (i in 1,4,5), with >=1 of those confirmed.
        # [hidden path — approved 2026-09-19 15:15; (二) restated 17:06 (v-E)]
        # S^m C_i conduction (m >= 1):
        #   conductor X = S^m C_i, i in {1,4,5,23};
        #   - SC23-type (i = 23): X relevant to >=1 C2^m C23 or C3^m C23 + two wide C_j^m targets
        #     [2026-09-19: anchor extended to C2/C3 (fixes the R442-class k2 misses)]
        #     (distinct directions); any one confirmed conducts the other wide(s).  [unchanged form]
        #   - wide-type (i in 1,4,5): (1) X relevant to one C_i^2 target (own direction);
        #     (2) X relevant to one C_j^1 & one C_k^1 target (i,j,k pairwise distinct; j,k in {1,23,4,5});
        #     conduction within the (2)-pair: one confirmed => the other confirmed.
        #   X itself is NOT confirmed by this rule.
        def _is_cond_mode(md):
            if md[0] != 'C' or md[4] < 1: return False
            if md[1] == 23: return md[2] == 0 and md[3] == 0
            if md[1] in (1, 4, 5): return md[2] == 1 and md[3] is None
            return False
        for _scm in sorted((md for md in comps if _is_cond_mode(md)), key=lambda x: (x[1], x[4])):
            _cm = _scm[4]
            for _i, _blk in enumerate(comps.get(_scm, [])):
                if not CONDUCT23: continue
                if _scm[1] == 23:
                    # SC23-type (form unchanged): anchor >=1 C2^m C23; two wide C_j^m targets
                    # in distinct directions; conduct among the wide targets.
                    _anchor = False
                    for _am in (P(_cm, 2), P(_cm, 3)):
                        for _j2, _comp in enumerate(comps.get(_am, [])):
                            if relevant(_blk, _comp, _am, edges, em, vm, _scm):
                                _anchor = True; break
                        if _anchor: break
                    if not _anchor:
                        continue
                    _dirs = set(); _cand = []
                    for _md in (W(1, _cm), W(4, _cm), W(5, _cm)):
                        for _j2, _comp in enumerate(comps.get(_md, [])):
                            if relevant(_blk, _comp, _md, edges, em, vm, _scm):
                                _cand.append((_md, _j2)); _dirs.add(d_of(_md))
                    if len(_dirs) < 2:
                        continue
                    if not any(confirmed.get(t) for t in _cand):
                        continue
                    for (_md, _j2) in _cand:
                        if (_md, _j2) not in confirmed:
                            if _DBG: print('   conduct23-fire: %s#%d -> conduct to %s#%d' % (name(_scm), _i, name(_md), _j2))
                            changed |= try_confirm(_md, _j2)
                else:
                    # wide-type (2026-09-19 17:06 form / v-E):
                    # (1) X relevant to one C_i^2 target (own direction i);
                    # (2) X relevant to one C_j^1 & one C_k^1 target (i,j,k pairwise distinct; j,k in {1,23,4,5});
                    # conduction within the (2)-pair: one confirmed => the other confirmed.
                    _A = []
                    for _j2, _comp in enumerate(comps.get(W(_scm[1], 2), [])):
                        if relevant(_blk, _comp, W(_scm[1], 2), edges, em, vm, _scm):
                            _A.append((W(_scm[1], 2), _j2))
                    if not _A:
                        continue
                    _D = {}
                    for _dd in (1, 23, 4, 5):
                        if _dd == _scm[1]: continue
                        _md1 = P(0, 0, 0) if _dd == 23 else W(_dd, 1)
                        _lst = []
                        for _j2, _comp in enumerate(comps.get(_md1, [])):
                            if relevant(_blk, _comp, _md1, edges, em, vm, _scm):
                                _lst.append((_md1, _j2))
                        if _lst:
                            _D[_dd] = _lst
                    _dks = sorted(_D.keys())
                    for _a1 in range(len(_dks)):
                        for _a2 in range(_a1 + 1, len(_dks)):
                            for (_am, _aj) in _D[_dks[_a1]]:
                                for (_bm, _bj) in _D[_dks[_a2]]:
                                    if confirmed.get((_am, _aj)) and (_bm, _bj) not in confirmed:
                                        if _DBG: print('   conduct23-fire: %s#%d -> conduct to %s#%d' % (name(_scm), _i, name(_bm), _bj))
                                        changed |= try_confirm(_bm, _bj)
                                    elif confirmed.get((_bm, _bj)) and (_am, _aj) not in confirmed:
                                        if _DBG: print('   conduct23-fire: %s#%d -> conduct to %s#%d' % (name(_scm), _i, name(_am), _aj))
                                        changed |= try_confirm(_am, _aj)
        # [generalized 2026-09-19] S^m C23 (m >= 1): relevant to a C2^m C23 component, a C3^m C23
        # component, and a C_i^m component (i in 1,4,5) — each category nonempty, >=1 confirmed overall
        # -> the S^m C23 component is confirmed itself.  (Only m = 1 occurs on the current corpus.)
        for scm_m in sorted((md for md in comps if md[0] == 'C' and md[1] == 23 and md[2] == 0 and md[3] == 0 and md[4] >= 1), key=lambda x: x[4]):
            _m = scm_m[4]
            sc_cats = ((P(_m, 2),), (P(_m, 3),), (W(1, _m), W(4, _m), W(5, _m)))
            for i, blk in enumerate(comps.get(scm_m, [])):
                if (scm_m, i) in confirmed: continue
                if not _cond_allowed(scm_m, blk): continue
                adj = []
                ok_all = True
                for cat in sc_cats:
                    found = False
                    for md in cat:
                        for j, comp in enumerate(comps.get(md, [])):
                            if relevant(blk, comp, md, edges, em, vm, scm_m):
                                found = True
                                adj.append((md, j))
                    if not found:
                        ok_all = False
                        break
                if ok_all and any(confirmed.get((md, j)) for (md, j) in adj):
                    changed |= try_confirm(scm_m, i)
        if _DBG: print('  -- cond3 --')
        # [condition 3] meet of two confirmed relevant components.
        for md, lst in comps.items():
            for i, blk in enumerate(lst):
                if (md, i) in confirmed: continue
                if not _cond_allowed(md, blk): continue
                hit = []
                for (cm, ci) in list(confirmed):
                    if relevant(blk, comps[cm][ci], cm, edges, em, vm, md):
                        hit.append(cm)
                done = False
                for ia in range(len(hit)):
                    for ib in range(ia + 1, len(hit)):
                        try:
                            if meet(hit[ia], hit[ib]) == md:
                                done = True; break
                        except ArithmeticError:
                            continue
                    if done: break
                if done:
                    changed |= try_confirm(md, i)
        if _DBG: print('  -- cond1 --')
        # [condition 1] for every unconfirmed block, we need (1) vee of inflows == mode (2) the C23 blocks also require third-port.
        for md, lst in comps.items():
            if md[0] == 'H': continue
            for i, blk in enumerate(lst):
                if (md, i) in confirmed: continue
                if not cond1_confirms(blk, md, comps, confirmed, edges, em, vm, ext_attach, ext_mode, dbg):
                    continue
                changed |= try_confirm(md, i)
    # all components (except H) must be confirmed; H automatically okay in this scenario.
    unconf = []
    for md, lst in comps.items():
        if md[0] == 'H': continue
        for i in range(len(lst)):
            if (md, i) not in confirmed:
                unconf.append(f'{name(md)}#{i}')
    if unconf:
        if dbg is not None: dbg['unconfirmed'] = unconf
        return False
    return True

