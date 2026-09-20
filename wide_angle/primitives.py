"""primitives.py — graph primitives + the Step-1 / First-Connectivity /
IR-compatibility check functions shared by the construction pipeline.

Contents:
  - Graph: internal-line graph representation + mode helpers (vertex_mode,
    vee, allowed);
  - check_fc:   First Connectivity;
  - momentum_ok: vertex-level momentum conservation (Step 1);
  - ir_ok:      IR-compatibility fixpoint (used by the region checks);
  - parse_master: 5pt6loop master-graph figure parser (demo/__main__ only).

No paper knowledge needed: the mode set is derived from the external modes
by meet/join closure, bounded by the softest external mode (no-cascading);
vertex candidates are constrained only by their external legs (momentum
balance); DFS assigns vertex modes with incremental pruning; edge modes are
enumerated per edge (allowed); then the full checks (vertex consistency,
First Connectivity, momentum conservation, IR compat) run.

(Formerly find_regions_5pt6_v2.py, a pure enumerator written for the
5pt6loop master graph — the name was a historical artifact.)

Usage (standalone 5pt6loop enumerator):
    python3 primitives.py [--brief] [--maxv N] [--probe]
"""
import sys, os, re, time, itertools
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import region_checker as rc
from region_checker import cond1_ok

INF = 10**9
H = (0, 0, 0)
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'draft-v16', 'figs', '5pt6loop')
MASTER = os.path.join(BASE, 'six_loop_example_1to3_decay_plus_soft_emission.tex')

# ---------------- master graph parsing ----------------
def parse_master(path):
    src = open(path).read()
    verts = set()
    for m in re.finditer(r'\\node\s*\[[^\]]*circle[^\]]*\]\s*\(\)\s*at\s*\(([-\d.]+),([-\d.]+)\)', src):
        verts.add((float(m.group(1)), float(m.group(2))))
    edges = []
    for m in re.finditer(r'\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)\s*edge\s*\[([^\]]*)\]\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', src):
        a, b = (float(m.group(1)), float(m.group(2))), (float(m.group(4)), float(m.group(5)))
        if a in verts and b in verts and (a, b) not in edges and (b, a) not in edges:
            edges.append((a, b))
    for m in re.finditer(r'\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)\s*--\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', src):
        a, b = (float(m.group(1)), float(m.group(2))), (float(m.group(3)), float(m.group(4)))
        if a in verts and b in verts and (a, b) not in edges and (b, a) not in edges:
            edges.append((a, b))
    ext = {}
    for m in re.finditer(r'\\draw \[ultra thick, color=(\w+)\]\s*\(([-\d.]+),([-\d.]+)\)\s*--\s*\(([-\d.]+),([-\d.]+)\)', src):
        col = m.group(1)
        e1 = (float(m.group(2)), float(m.group(3)))
        e2 = (float(m.group(4)), float(m.group(5)))
        inner = e1 if e1 in verts else (e2 if e2 in verts else None)
        if inner is None: continue
        best, bd = None, 1e9
        for lm in re.finditer(r'\\node\s*at\s*\(([-\d.]+),([-\d.]+)\)\s*\{\\LARGE\s*\$([\w_]+)\$', src):
            lx, ly, nm = float(lm.group(1)), float(lm.group(2)), lm.group(3).replace('_', '')
            d = (lx - (e2 if inner == e1 else e1)[0])**2 + (ly - (e2 if inner == e1 else e1)[1])**2
            if d < bd: best, bd = nm, d
        if best is not None:
            ext[best] = inner
    return sorted(verts), edges, ext

# ---------------- mode derivation (from external modes only) ----------------
def derive_modes(EXTMODE):
    seeds = list(EXTMODE.values()) + [H]
    softest = max((m+n for (m, n, i) in EXTMODE.values() if n < INF), default=0)
    modes = set(seeds)
    for m in range(1, softest + 1):
        modes.add((m, 0, 0))
    changed = True
    while changed:
        changed = False
        lst = list(modes)
        for a in lst:
            for b in lst:
                for r in (rc.meet(a, b), rc.join(a, b)):
                    if r not in modes:
                        modes.add(r); changed = True
    modes = {m for m in modes if m[1] < INF}
    vmax = max(rc.V(m) for m in EXTMODE.values() if m != H)
    modes = {m for m in modes if rc.V(m) <= vmax}
    return sorted(modes, key=rc.V)

# ---------------- machinery ----------------
class Graph:
    def __init__(self, vertices, edges, ext):
        self.vertices = list(vertices)
        self.edges = list(edges)
        self.ext = dict(ext)
        self.vidx = {v: i for i, v in enumerate(self.vertices)}
        self.incident = defaultdict(list)
        for ei, (u, v) in enumerate(self.edges):
            self.incident[u].append(ei)
            self.incident[v].append(ei)
    def edge_other(self, ei, v):
        u, w = self.edges[ei]
        return w if u == v else u

def vertex_mode(g, edge_modes, v, EXTMODE):
    accs = []
    for ei in g.incident.get(v, []):
        if edge_modes[ei] is not None:
            accs.append(edge_modes[ei])
    for name, vv in g.ext.items():
        if vv == v: accs.append(EXTMODE[name])
    if not accs: return None
    acc = accs[0]
    for m in accs[1:]: acc = rc.join(acc, m)
    return rc.norm(acc)

def vee(modes):
    if not modes: return None
    acc = modes[0]
    for m in modes[1:]: acc = rc.join(acc, m)
    return rc.norm(acc)

def softer_v(va, vb):
    if rc.eq(va, vb): return va
    return va if rc.V(va) > rc.V(vb) else vb

def is_sc_type(md):
    return md[0] >= 1 and md[1] >= 1

def allowed(va, vb):
    if rc.eq(va, vb):
        return [va]
    if rc.harder_or_eq(va, vb):
        return [vb]
    if rc.harder_or_eq(vb, va):
        return [va]
    sv = softer_v(va, vb)
    if is_sc_type(va) or is_sc_type(vb):
        return [sv]
    opts = []
    if rc.V(va) != rc.V(vb):
        opts.append(sv)
    mt = rc.meet(va, vb)
    if mt not in opts: opts.append(mt)
    return opts

def check_fc(g, edge_modes, EXTMODE):
    """First Connectivity Theorem (§5.1): for every threshold n, the union of
    components with 𝒱 <= n must be connected (∪_{𝒱≤n} Γ_X connected ∀n).
    Thresholds are the distinct 𝒱 values present; isolated mode-vertices are
    included as nodes (Γ_X contains mode-X vertices)."""
    eVs = [rc.V(m) if m is not None else INF for m in edge_modes]
    vVs = [rc.V(vertex_mode(g, edge_modes, v, EXTMODE)) if vertex_mode(g, edge_modes, v, EXTMODE) else INF
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
            if not rc.eq(inc[0], inc[1]): return False
            continue
        ok = False
        for r in range(1, n):
            for A in itertools.combinations(range(n), r):
                B = [i for i in range(n) if i not in A]
                if not B: continue
                if rc.eq(vee([inc[i] for i in A]), vee([inc[i] for i in B])):
                    ok = True; break
            if ok: break
        if not ok: return False
    return True

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
    all_comps = rc.build_components(verts, edges)
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
            ps = rc.partial_sum_mode(comp, confirmed, verts, edges, all_comps,
                                     attach, extmode)
            if ps is not None and rc.eq(ps, comp['mode']):
                cc = 1
            if cc is None:
                for g1, g2 in itertools.combinations(confirmed, 2):
                    if rc.relevant(comp, g1, verts, edges, all_comps) and rc.relevant(comp, g2, verts, edges, all_comps):
                        if rc.eq(rc.meet(g1['mode'], g2['mode']), comp['mode']): cc = 3; break
            if cc is None:
                res = rc.find_messenger(comp, all_comps, verts, edges, attach, extmode, confirmed)
                if res is not None:
                    G, special, _ = res
                    # type-2 special (m_i=0): kernel fed by its own attached
                    # S^m C_i^{n'} external — no confirmed relevance needed
                    # (2026-08-12 , tentative).
                    if special:
                        cc = 2
                    else:
                        for g0 in confirmed:
                            if rc.relevant(comp, g0, verts, edges, all_comps): cc = 2; break
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
    spacelike_collinear/regge/regge_core.mode_components — 2026-08-21: the
    Regge methodology decomposes γ̃_X into 1VI blocks instead of filtering.)"""
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
    ok1vi, _ = contracted_1vi_ok(edges_in, em, attach, extmode, rc)
    if not ok1vi:
        return False
    # 2026-09-06: mode_components_wa builds γ̃_X with the correct aux (far
    # endpoints of X edges only) — its components ARE the paper's 1VI
    # blocks (pendant vs loop split); cond1_strong's third-port requirement
    # runs on them.
    all_comps = mode_components_wa(edges_in, em, attach, extmode, rc)

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
            if verts.get(v) is not None and rc.eq(verts[v], (0, 0, 0)):
                return False
        for (u, w) in comp['E']:
            for v in (u, w):
                if verts.get(v) is not None and rc.eq(verts[v], (0, 0, 0)):
                    return False
        for other in all_comps:
            if other is comp:
                continue
            if rc.V(other['mode']) < rc.V(comp['mode']) \
                    and rc.adjacent(comp, other):
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
                if rc.eq(rc.join(pool[a], pool[b]), comp['mode']):
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
            if rc.cond1_strong_ae(comp, confirmed, verts, edges, all_comps,
                                  attach, extmode):
                cc = 1
            if cc is None and _cond_allowed(comp):
                for g1, g2 in itertools.combinations(confirmed, 2):
                    if rc.relevant(comp, g1, verts, edges, all_comps) and rc.relevant(comp, g2, verts, edges, all_comps):
                        if rc.eq(rc.meet(g1['mode'], g2['mode']), comp['mode']): cc = 3; break
            kbs = ()
            if cc is None and _cond_allowed(comp):
                kern, kbs = _cloud_kernel(comp)
                res = rc.find_messenger(comp, all_comps, verts, edges, attach,
                                        extmode, confirmed, kernel=kern,
                                        kernel_blocks=kbs or None)
                if res is not None:
                    G, special, kbs = res
                    if special:
                        cc = 2
                    else:
                        for g0 in confirmed:
                            if rc.relevant(comp, g0, verts, edges, all_comps): cc = 2; break
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

def main():
    brief = '--brief' in sys.argv
    maxv = None
    if '--maxv' in sys.argv:
        maxv = int(sys.argv[sys.argv.index('--maxv') + 1])
    t0 = time.time()
    verts, edges, ext = parse_master(MASTER)
    print(f'master graph: {len(verts)} vertices, {len(edges)} edges')
    print(f'external attachments: {ext}')
    EXTMODE = {'p1': (0, INF, 1), 'p2': (0, 1, 2), 'p3': (0, INF, 3), 'l1': (1, 1, 4), 'q1': H}
    print(f'external modes: {EXTMODE}')
    modes = derive_modes(EXTMODE)
    print(f'derived internal mode set: {len(modes)} modes: {[str(m) for m in modes]}')

    g = Graph(verts, edges, ext)
    deg = defaultdict(int)
    for a, b in edges:
        deg[a] += 1; deg[b] += 1
    Vorder = sorted(g.vertices, key=lambda v: -deg[v])
    if maxv: Vorder = Vorder[:maxv]

    hard = {}
    for v in g.vertices:
        exts = [n for n, vv in ext.items() if vv == v]
        if 'q1' in exts: hard[v] = [H]
        elif 'p1' in exts: hard[v] = [H, (0, 1, 1)]
        elif 'p2' in exts: hard[v] = [H, (0, 1, 2)]
        elif 'p3' in exts: hard[v] = [H, (0, 1, 3)]
        elif 'l1' in exts: hard[v] = [H, (1, 0, 0), (1, 1, 4), (2, 0, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3)]
        else: hard[v] = list(modes)

    # incremental DFS: assign vertex modes; when both endpoints of an edge are
    # known, the edge mode is forced to allowed(va, vb); prune a vertex as soon
    # as its mode is not reproducible from its incident (known) edges+externals.
    edges_by_vertex = defaultdict(list)
    for ei, (a, b) in enumerate(edges):
        edges_by_vertex[a].append(ei)
        edges_by_vertex[b].append(ei)

    solutions = []
    vm = {}
    def dfs(idx):
        if idx == len(Vorder):
            solutions.append(dict(vm)); return
        v = Vorder[idx]
        for md in hard[v]:
            vm[v] = md
            # prune: for each neighbor w already assigned, edge (v,w) must be
            # allowed; and w's mode must be reproducible from its known edges
            ok = True
            for ei in edges_by_vertex[v]:
                o = g.edge_other(ei, v)
                if o not in vm: continue
                if not allowed(vm[v], vm[o]):
                    ok = False; break
            if ok:
                # vertex-consistency prune: only when ALL incident edges are
                # known (all neighbors assigned) is the check sound.
                known = [ei for ei in edges_by_vertex[v] if g.edge_other(ei, v) in vm]
                if len(known) == len(edges_by_vertex[v]):
                    al_list = [allowed(vm[v], vm[g.edge_other(ei, v)]) for ei in known]
                    ext_modes = [EXTMODE[n] for n, vv in ext.items() if vv == v]
                    reachable = False
                    if not al_list and not ext_modes:
                        reachable = (md is None)
                    else:
                        for combo in itertools.product(*al_list) if al_list else [()]:
                            accs = list(combo) + ext_modes
                            if accs and rc.eq(vee(accs), md):
                                reachable = True; break
                    if reachable:
                        dfs(idx + 1)
                else:
                    dfs(idx + 1)
            del vm[v]
    dfs(0)
    print(f'vertex configs: {len(solutions)} ({time.time()-t0:.1f}s)')

    # edge-mode completion + full checks
    regs = set()
    n_vc = n_fc = n_mc = n_mj = 0
    for sol in solutions:
        al = [allowed(sol[u], sol[v]) for (u, v) in edges]
        if any(len(a) == 0 for a in al): continue
        for combo in itertools.product(*al):
            em = list(combo)
            ok = True
            for w in Vorder:
                accs = []
                for ei in g.incident.get(w, []):
                    accs.append(em[ei])
                for n, vv in ext.items():
                    if vv == w: accs.append(EXTMODE[n])
                if not accs: ok = False; break
                if not rc.eq(vee(accs), vm_mode(sol, w)): ok = False; break
            if not ok: continue
            n_vc += 1
            if not check_fc(g, em, EXTMODE): continue
            n_fc += 1
            if not momentum_ok(g, em, EXTMODE): continue
            n_mc += 1
            # Mojetic (Theorem 3 cond1): H∪J∖J_i 1VI after aux connection.
            # Component-level momentum conservation; zero false kills on all
            # 7179 verified pySecDec FRI regions (2026-08-10). Also catches
            # vertex-inconsistent tex errors like SCiregion71 (v16).
            ok_mj, _ = cond1_ok(list(g.edges), list(em), ext, EXTMODE)
            if not ok_mj: continue
            n_mj += 1
            if ir_ok(g, em, EXTMODE):
                regs.add(tuple(em))
    print(f'VC: {n_vc}, +FC: {n_fc}, +MC: {n_mc}, +Mojetic: {n_mj}, +IR: {len(regs)} ({time.time()-t0:.1f}s)')
    print(f'\nTOTAL REGIONS FOUND: {len(regs)}')
    from collections import Counter
    cls = Counter()
    for em in regs:
        vms = [vertex_mode(g, list(em), v, EXTMODE) for v in g.vertices]
        has_s2 = any(m and m[0] >= 2 for m in vms)
        has_sc = any(m and m[0] >= 1 and m[1] >= 1 for m in vms)
        has_s = any(m and m[0] >= 1 and m[1] == 0 for m in vms)
        if has_s2: cls['S^2'] += 1
        elif has_sc: cls['SC'] += 1
        elif has_s: cls['S'] += 1
        else: cls['C/H'] += 1
    print('by softest-mode class:', dict(cls))

def vm_mode(sol, w):
    return sol.get(w)

if __name__ == '__main__':
    main()
