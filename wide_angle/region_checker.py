"""region_checker.py — the FRI region checker: mode algebra, region
components, IR compatibility, messengers.

Given a graph configuration (vertex/edge mode assignment), decide whether it
is a facet region, per the paper's criteria:
  (A) fundamental pattern (momentum conservation, jet connectivity,
      contracted-mode-component 1VI, mojetic),
  (B) First Connectivity,
  (C) IR compatibility (recursive 3-condition fixpoint).

Mode tuples: (m, n, i) for S^m C_i^n (H = (0,0,0)).  Accepts tikz-figure
input (nodes with fill=Col, \\path edge [Col], label nodes) as well as raw
mode arrays.  (Formerly region_checker5.py — the "5" was a 5pt6loop-era
artifact, the successor of the figure-based region_checker.py.)
"""
import re, os
from collections import defaultdict, deque
from itertools import combinations
from mojetic_check import cond1_ok

INF = 10**9

def V(m): return 2*m[0] + m[1]
def massive_h_ok(edges, em, massive=frozenset()):
    """Big-massive propagators (m_i = O(1)) must stay in the HARD mode.

    Physics (massive prescription, 2026-08-15): with all internal
    masses either 0 or O(1) and the small variables still the virtualities,
    the regions are precisely those of the massless graph, constrained by
    the requirement that every big-massive propagator is in H.  For a
    soft/collinear line k^2 ~ lambda^n << m_i^2, the propagator
    1/(k^2 - m_i^2) -> -1/m_i^2 + O(lambda^n): the line's scaling collapses
    to an O(1) constant, so the mode is not self-consistent (the Landau
    surface k^2 = m^2 meets neither the soft nor the collinear face).
    Since the edge mode is the join of its endpoint modes, a massive edge
    is H iff both endpoints are H.
    """
    if not massive:
        return True
    for e, md in zip(edges, em):
        if md != (0, 0, 0) and frozenset(e) in massive:
            return False
    return True


def norm(m):
    m0, n, i = m
    if n > INF//2: n = INF
    if n == 0: i = 0
    return (m0, n, i)

# 2026-08-12 TENTATIVE (: type-2 / m_i=0 special messenger target.
# When True, an S^m kernel carrying an attached S^m C_i^{n'} external counts
# as its own target (direction i from the external) and needs no relevance
# to a confirmed subgraph.  Set to False to emulate the pre-special-case
# behaviour for cross-checks (CrownSS k5: 80 vs 81).
ALLOW_SPECIAL_TARGET = True
def eq(X, Y):
    X, Y = norm(X), norm(Y)
    return X[0] == Y[0] and X[1] == Y[1] and (X[2] == Y[2] or X[1] == 0 or Y[1] == 0)
def harder_or_eq(X, Y):
    X, Y = norm(X), norm(Y)
    if eq(X, Y): return True
    if X[2] == Y[2]:
        return X[0] <= Y[0] and X[0] + X[1] <= Y[0] + Y[1]
    return X[0] + X[1] <= Y[0]
def join(X, Y): return _join_meet(X, Y)[0]
def meet(X, Y): return _join_meet(X, Y)[1]
def _join_meet(X, Y):
    X, Y = norm(X), norm(Y)
    if X == (0, 0, 0): return (X, Y)
    if Y == (0, 0, 0): return (Y, X)
    if eq(X, Y): return (X, X)
    iX = X[2] if X[1] != 0 else (Y[2] if Y[1] != 0 else 0)
    iY = Y[2] if Y[1] != 0 else iX
    if iX == iY:
        A, B = (X[0], X[1], iX), (Y[0], Y[1], iX)
        for (P, Q) in ((A, B), (B, A)):
            m1, n1, _ = P; m2, n2, _ = Q
            if m2 < m1 <= m1 + n1 < m2 + n2:
                return (norm((m2, m1 + n1 - m2, iX)), norm((m1, m2 + n2 - m1, iX)))
        if harder_or_eq(A, B) and not harder_or_eq(B, A): return (norm(A), norm(B))
        if harder_or_eq(B, A) and not harder_or_eq(A, B): return (norm(B), norm(A))
        return (norm(A), norm(B))
    else:
        # different directions: first check comparability (meet/join of two
        # comparable modes = the softer/harder one, regardless of direction)
        if harder_or_eq(X, Y) and not harder_or_eq(Y, X):
            return (norm(X), norm(Y))      # X harder: join=X, meet=Y
        if harder_or_eq(Y, X) and not harder_or_eq(X, Y):
            return (norm(Y), norm(X))      # Y harder: join=Y, meet=X
        if X[0] + X[1] <= Y[0] + Y[1]: P, Q = X, Y
        else: P, Q = Y, X
        m1, n1, i = P; m2, n2, j = Q
        if m1 <= m2: jn = norm((m1, m2 - m1, i))
        else: jn = norm((m2, m1 - m2, j))
        mt = norm((m1 + n1, m2 + n2 - m1 - n1, j))
        return (jn, mt)
def marginal_softer(X, Y):
    X, Y = norm(X), norm(Y)
    if eq(X, Y): return False
    if not harder_or_eq(Y, X): return False
    if harder_or_eq(X, Y): return False
    return X[0] <= Y[0] + Y[1]

def parse_mode(s):
    if s == 'H': return (0, 0, 0)
    m = n = 0; i = 0
    mm = re.match(r'S(?:\^(\d+))?', s)
    if mm: m = int(mm.group(1)) if mm.group(1) else 1
    cm = re.search(r'C(\d+)(?:\^(\d+)|(inf))?', s)
    if cm:
        i = int(cm.group(1))
        if cm.group(3) == 'inf': n = INF
        elif cm.group(2): n = int(cm.group(2))
        else: n = 1
    return (m, n, i)

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

def build_components(verts, edges):
    by_mode = defaultdict(lambda: {'V': set(), 'E': set()})
    for v, md in verts.items(): by_mode[md]['V'].add(v)
    for (a, b, md) in edges: by_mode[md]['E'].add((a, b))
    comps = []
    for md, g in by_mode.items():
        nodes = list(g['V']) + list(g['E'])
        idx = {nd: k for k, nd in enumerate(nodes)}
        par = list(range(len(nodes)))
        def find(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: par[ra] = rb
        for k, e in enumerate(g['E']):
            a, b = e
            for v in (a, b):
                if v in g['V']: union(idx[e], idx[v])
        groups = defaultdict(lambda: {'V': set(), 'E': set()})
        for k, nd in enumerate(nodes):
            r = find(k)
            if nd in g['V']: groups[r]['V'].add(nd)
            else: groups[r]['E'].add(nd)
        for gr in groups.values():
            comps.append({'mode': md, 'V': gr['V'], 'E': gr['E']})
    return comps

def elements(c): return c['V'] | c['E']
def neighbors(elem, verts, edges):
    if elem in verts:
        return {(a, b) for (a, b, md) in edges if elem in (a, b)}
    return {elem[0], elem[1]}

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
    from mojetic_check import jet_connected_ok
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
