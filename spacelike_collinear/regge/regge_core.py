#!/usr/bin/env python3
# Regge-limit region enumerator core.
#
# OVERVIEW
# Mode space, two classes:
#   (1) lattice modes S^m C_i^n C_ij — H (hard), C13, C24, S, SC, S²C, refinements; algebra in regge_modes.py;
#   (2) G (Glauber) and sH (semihard), representing t-channel Glauber-momentum transfer.
# Lattice modes come from overlaying cuts; G and sH are assigned afterwards by glauber_adjust and/or necklace_detect.
# Cut rules: C13 surrounds p1&p3 (nonempty => contains both, induced-subgraph connected), may be empty, never contains p2/p4 vertices; C24 symmetric.
# Region requirements (see _build_region for the pipeline):
#   (1) every connected component of each jet J13/J24 contains at least one of its external vertices (jet_components_ok);
#   (2) the subgraph not included in any cut is nonempty and connected;
#   (3) mojetic — k0 (all externals C13/C24): mojetic_k0_ok; non-k0: mojetic_ok;
#   (4) no C13/C24 component whose adjacent modes are all softer than it (c13_c24_island_ok);
#   (5) IR compatibility (ir_ok_region).

from itertools import combinations
from functools import lru_cache
import os

import regge_modes

# Mode STRINGS: em/vm values throughout this file are plain literals — the serialization of the general S^m C_i^n C_ij modes defined in regge_modes.py.
# Canonical names (with virtuality in parentheses):
#   H (0) · G (1) · sH (1, semihard overlay)
#   pure soft: S (2), S^2 (4), S^3 (6)
#   pair-collinear: C13 (1), C24 (1) · S^1C13 (3), S^1C24 (3)
#   S^2C13 (5), S^2C24 (5)
#   leg refinements C_i^n C_ij: C1C13 (2), C3C13 (2), C2C24 (2), C4C24 (2);
#   C1^2C13 (3), C3^2C13 (3), C2^2C24 (3), C4^2C24 (3);
#   S^1C1C13 (4), S^1C3C13 (4), S^1C2C24 (4), S^1C4C24 (4)
#   lightlike externals C_i^∞ C_ij (m=0): C1∞C13 (3), C3∞C13 (3),
#   C2∞C24 (3), C4∞C24 (3)
MARGINAL_SOFTER = {
    ('C1C13','C13'): True, ('C3C13','C13'): True, ('C2C24','C24'): True, ('C4C24','C24'): True,
    ('S^1C13','C24'): True, ('S^1C13','C1C13'): True, ('S^1C13','C3C13'): True, ('S^1C13','S'): True,
    ('S^1C24','C13'): True, ('S^1C24','C2C24'): True, ('S^1C24','C4C24'): True, ('S^1C24','S'): True,
    ('S','C13'): True, ('S','C24'): True,
    ('S^2','C1C13'): True, ('S^2','C3C13'): True, ('S^2','C2C24'): True, ('S^2','C4C24'): True, ('S^2','S^1C13'): True, ('S^2','S^1C24'): True,
    ('C1^2C13','C13'): True, ('C1^2C13','C1C13'): True, ('C1^2C13','C3C13'): True,
    ('C3^2C13','C13'): True, ('C3^2C13','C1C13'): True, ('C3^2C13','C3C13'): True,
    ('C2^2C24','C24'): True, ('C2^2C24','C2C24'): True, ('C2^2C24','C4C24'): True,
    ('C4^2C24','C24'): True, ('C4^2C24','C2C24'): True, ('C4^2C24','C4C24'): True,
    ('S^2C13','C1^2C13'): True, ('S^2C13','C3^2C13'): True, ('S^2C13','C2C24'): True, ('S^2C13','C4C24'): True, ('S^2C13','S^2'): True, ('S^2C13','S^1C24'): True,
    ('S^2C24','C1C13'): True, ('S^2C24','C3C13'): True, ('S^2C24','C2^2C24'): True, ('S^2C24','C4^2C24'): True, ('S^2C24','S^2'): True, ('S^2C24','S^1C13'): True,
    ('C1∞C13','C13'): True, ('C1∞C13','C1C13'): True, ('C1∞C13','C3C13'): True, ('C3∞C13','C13'): True, ('C3∞C13','C1C13'): True, ('C3∞C13','C3C13'): True,
    ('C2∞C24','C24'): True, ('C2∞C24','C2C24'): True, ('C2∞C24','C4C24'): True, ('C4∞C24','C24'): True, ('C4∞C24','C2C24'): True, ('C4∞C24','C4C24'): True,
    ('C1∞C13','C1^2C13'): True, ('C3∞C13','C3^2C13'): True,
    ('C2∞C24','C2^2C24'): True, ('C4∞C24','C4^2C24'): True,
    ('S^1C1C13','C24'): True, ('S^1C1C13','S'): True, ('S^1C1C13','S^1C13'): True, ('S^1C1C13','C1^2C13'): True,
    ('S^1C3C13','C24'): True, ('S^1C3C13','S'): True, ('S^1C3C13','S^1C13'): True, ('S^1C3C13','C3^2C13'): True,
    ('S^1C2C24','C13'): True, ('S^1C2C24','S'): True, ('S^1C2C24','S^1C24'): True, ('S^1C2C24','C2^2C24'): True,
    ('S^1C4C24','C13'): True, ('S^1C4C24','S'): True, ('S^1C4C24','S^1C24'): True, ('S^1C4C24','C4^2C24'): True,
    ('S^3','S^1C1C13'): True, ('S^3','S^1C3C13'): True, ('S^3','S^1C2C24'): True, ('S^3','S^1C4C24'): True,
    ('S^2C13','S^1C1C13'): True, ('S^2C13','S^1C3C13'): True, ('S^2C24','S^1C2C24'): True, ('S^2C24','S^1C4C24'): True,
    ('C13','H'): True, ('C24','H'): True,
    ('C1C13','H'): True, ('C3C13','H'): True, ('C2C24','H'): True, ('C4C24','H'): True,
    ('C1^2C13','H'): True, ('C3^2C13','H'): True, ('C2^2C24','H'): True, ('C4^2C24','H'): True,
    ('C1∞C13','H'): True, ('C3∞C13','H'): True, ('C2∞C24','H'): True, ('C4∞C24','H'): True,
}

def marginally_softer(x1, x2):
    return MARGINAL_SOFTER.get((x1, x2), False)

V = {'H': 0, 'C13': 1, 'C24': 1, 'S': 2, 'G': 1, 'sH': 1,
     'C1C13': 2, 'C3C13': 2, 'C2C24': 2, 'C4C24': 2,
     'S^1C13': 3, 'S^1C24': 3, 'S^2': 4, 'S^3': 6,
     # 5-loop: C^3 (Ci^2Cij) 𝒱=3, S^2C 𝒱=5
     'C1^2C13': 3, 'C3^2C13': 3, 'C2^2C24': 3, 'C4^2C24': 3,
     'S^2C13': 5, 'S^2C24': 5,
     # external-momentum modes C1∞C13: marginally softer than C1C13 (C1^{n1}C13 softer than C1^{n2}C13 iff n1 > n2; 2026-08-17).
     'C1∞C13': 3, 'C3∞C13': 3, 'C2∞C24': 3, 'C4∞C24': 3}
# leg-explicit SC^2 names (mode algebra, 2026-08-29): S^1 C_i^1 C_ij, 𝒱=4.
V['S^1C1C13'] = V['S^1C3C13'] = V['S^1C2C24'] = V['S^1C4C24'] = 4
J13_FAM = {'C13', 'C1C13', 'C3C13', 'C1^2C13', 'C3^2C13'}
J24_FAM = {'C24', 'C2C24', 'C4C24', 'C2^2C24', 'C4^2C24'}

@lru_cache(maxsize=None)
def meet(a, b):
    return regge_modes.old_meet(a, b)


@lru_cache(maxsize=None)
def join(a, b):
    return regge_modes.old_join(a, b)

def vee(modes):
    acc = modes[0]
    for m in modes[1:]: acc = join(acc, m)
    return acc

def connected(vs, es): # checks whether the induced graph (including only edges whose endpoints both lie in vs) is connected; vs: vertex set (may include the auxiliary vertex "aux"); es: edge list.
    if len(vs) <= 1: return True
    adj = {v: set() for v in vs}
    for (a, b) in es:
        if a in vs and b in vs: # goes through those edges whose endpoints both lie in vs.
            adj[a].add(b); adj[b].add(a)
    seen = {next(iter(vs))}; stack = [next(iter(vs))]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in seen: seen.add(u); stack.append(u)
    return seen == set(vs)

def is_1vi(vs, es): # checks whether the graph is one-vertex irreducible (deleting any single vertex leaves the graph connected).
    if len(vs) <= 1: return True
    for v in vs:
        rest = vs - {v}
        if not connected(rest, es): return False
    return True

def components(es, verts): # returns a mapping {vertex: component_root} via Union-Find (disjoint set union).
    parent = {v: v for v in verts}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for (a, b) in es:
        if a in parent and b in parent: union(a, b)
    return {v: find(v) for v in verts}

def ext_mode_for(ext_attach, ext_mode, n):
    if ext_mode is not None and n in ext_mode:
        return ext_mode[n]
    return {'p1': 'C1∞C13', 'p2': 'C2∞C24', 'p3': 'C3∞C13', 'p4': 'C4∞C24'}[n]

def ext_m(mode): # m_i from an external-mode string: p_i^2 ~ λ^{m_i}; the strict lightlike case (p_i^2 = 0) is denoted by "None".
    if mode in ('C13', 'C24'):
        return 1
    if mode in ('C1C13', 'C3C13', 'C2C24', 'C4C24'):
        return 2
    if mode in ('C1^2C13', 'C3^2C13', 'C2^2C24', 'C4^2C24'):
        return 3
    return None

def possibly_softest(m1, m2, m3, m4): # analyze the possibly softest mode in the expansion, given the external kinematics p1=C_1^{m1}C13, p2=C_2^{m2}C24, p3=C_3^{m3}C13, p4=C_4^{m4}C24.
    def _min(*xs):
        xs = [x for x in xs if x is not None]
        return min(xs) if xs else None
    def _max(*xs):
        return max(xs) if all(x is not None for x in xs) else None

    finite = [i for i, m in enumerate((m1, m2, m3, m4)) if m is not None]
    if not finite: # if all the external momenta are precisely on shell, p_1^2=p_2^2=p_3^2=p_4^2=0 (kinematics 1), then no softest mode, a cascade of modes appears.
        return []
    if len(finite) == 1: # if there is a unique off-shell external momentum, identify the corresponding m, and the possibly softest mode is of type S^{m + 1}C.
        i, m = finite[0], (m1, m2, m3, m4)[finite[0]]
        return [f'S^{m + 1}C24' if i in (0, 2) else f'S^{m + 1}C13']
    M1 = _min(m1, m3, None if _max(m2, m4) is None else _max(m2, m4) + 1) # in the presence of multiple off-shell external momenta, evaluate the possibly softest mode directly.
    M2 = _min(m2, m4, None if _max(m1, m3) is None else _max(m1, m3) + 1)
    if M1 == M2: # in this case there are two possibly softest modes, otherwise there is precisely one possibly softest mode.
        return [f'S^{M1}C13', f'S^{M2}C24']
    return [f'S^{M1}C13' if M1 > M2 else f'S^{M2}C24']

# The next few functions describe the routine of identifying Glauber and semihard subgraphs.
# This is done after overlaying the cuts; the obtained configuration is called the preliminary configuration.
# In the preliminary configuration, some hard edges/vertices should have been identified as Glauber or semihard.
# In glauber_adjust we focus on the large-momentum subgraph of the preliminary configuration, which is required to be connected, otherwise momentum conservation is violated and we discard the configuration.
# Next, identify any H edge whose removal separates p1,p3 from p2,p4; such edges carry momentum p1-p3 (or p2-p4) and are changed to Glauber edges.
# If we find any non-H (jet) edge with such a property, discard the configuration.
# Similarly, in _glauber_cut_vertices, cut vertices separating p1,p3 from p2,p4 are changed to Glauber vertices.
# Then we identify the semihard edges and vertices: in _glauber_semihard we examine subgraphs between two Glauber vertices.
# There are two possibilities:
#   (1) a single propagator connecting them -> it must have been identified as a Glauber;
#   (2) a blob of several propagators -> the WHOLE blob becomes semihard (parallel edges count as a blob too).

def glauber_adjust(edges, em, vm, ext_attach, ext_mode=None): # identifies the Glauber and semihard subgraphs.
    em = list(em)
    verts = set(v for e in edges for v in e) | set(ext_attach.values())
    _SMALL = ('S', 'S^1C13', 'S^1C24', 'S^2', 'S^2C13', 'S^2C24', 'S^1C1C13', 'S^1C3C13', 'S^1C2C24', 'S^1C4C24')
    small_edges = {e for e, m in zip(edges, em) if m in _SMALL} # define the small-momentum subgraph
    small_verts = {v for v in verts if vm.get(v) in _SMALL}
    big_edges = [e for e in edges if e not in small_edges]
    big_verts = verts - small_verts
    if not connected(big_verts, big_edges): # the large-momentum subgraph (defined as hard + jets + Glauber + semihard) must be connected.
        return None
    p1, p3 = ext_attach['p1'], ext_attach['p3']
    p2, p4 = ext_attach['p2'], ext_attach['p4']
    big_pairs = [(j, e) for j, e in enumerate(edges) if e not in small_edges]
    g_edges = []
    for i, e in enumerate(edges):
        rest = [x for j, x in big_pairs if j != i]
        comp = components(rest, verts)
        if not (comp[p1] == comp[p3] and comp[p2] == comp[p4] and comp[p1] != comp[p2]):  # large-momentum edges separating p1,p3 from p2,p4
            continue
        if em[i] != 'H': # if we find a jet edge with the property above, then discard the region.
            return None
        g_edges.append(i) # such an off-shell edge is identified as a Glauber propagator
    for i in g_edges:
        em[i] = 'G'
    vm = dict(vm)
    for v in verts:
        accs = []
        for i, (a, b) in enumerate(edges):
            if a == v or b == v: accs.append(em[i])
        for n, vv in ext_attach.items():
            if vv == v: accs.append(ext_mode_for(ext_attach, ext_mode, n))
        vm[v] = vee(accs) if accs else 'H'
    vm2 = _glauber_cut_vertices(edges, em, vm, ext_attach, ext_mode, big_edges, big_verts, verts)
    if vm2 is not None:
        vm = vm2
        em, vm = _glauber_semihard(edges, em, vm, [j for j, _ in big_pairs], big_verts, verts, ext_attach, ext_mode)
    return em, vm

def _glauber_semihard(edges, em, vm, big_idx, big_verts, verts, ext_attach, ext_mode): # identifies the semihard subgraph (if nonempty).
    em = list(em)
    big_edges = [edges[i] for i in big_idx]
    Gset = {v for v in big_verts if vm.get(v) == 'G'}
    if len(Gset) < 2:
        return em, vm
    changed = False
    # (a) direct edges between a G-G pair: exactly one -> G; parallel (>=2) -> sH
    pairs = {}
    for i in big_idx:
        a, b = edges[i]
        if a in Gset and b in Gset:
            pairs.setdefault(frozenset((a, b)), []).append(i)
    for idxs in pairs.values():
        if len(idxs) == 1:
            if em[idxs[0]] != 'G':
                em[idxs[0]] = 'G'
                changed = True
        else:
            for i in idxs:
                if em[i] != 'sH':
                    em[i] = 'sH'
                    changed = True
    # (b) components of big_verts - Gset adjacent to >=2 Glauber vertices -> the whole blob (internal + attachment edges) becomes sH
    rest = big_verts - Gset
    if rest:
        adj_rest = {v: [] for v in rest}
        for i in big_idx:
            a, b = edges[i]
            if a in rest and b in rest:
                adj_rest[a].append(b)
                adj_rest[b].append(a)
        seen = set()
        for v0 in rest:
            if v0 in seen:
                continue
            comp = {v0}
            stack = [v0]
            while stack:
                x = stack.pop()
                for w in adj_rest.get(x, ()):
                    if w not in comp:
                        comp.add(w)
                        stack.append(w)
            seen |= comp
            att = {g for g in Gset if any(
                (g == a and b in comp) or (g == b and a in comp)
                for i in big_idx for (a, b) in (edges[i],))}
            if len(att) >= 2:
                for i in big_idx:
                    a, b = edges[i]
                    if (a in comp or b in comp) and em[i] != 'sH':
                        em[i] = 'sH'
                        changed = True
    if not changed:
        return em, vm
    vm2 = {}
    for v in verts:
        if v in Gset:
            vm2[v] = 'G'
            continue
        accs = []
        for i, (a, b) in enumerate(edges):
            if a == v or b == v:
                accs.append(em[i])
        for n, vv in ext_attach.items():
            if vv == v:
                accs.append(ext_mode_for(ext_attach, ext_mode, n))
        vm2[v] = vee(accs) if accs else 'H'
    return em, vm2

def _glauber_cut_vertices(edges, em, vm, ext_attach, ext_mode, big_edges, big_verts, verts):
    p1, p3 = ext_attach['p1'], ext_attach['p3']
    p2, p4 = ext_attach['p2'], ext_attach['p4']
    changed = False
    vm2 = dict(vm)
    for v in big_verts:
        rest = [(a, b) for (a, b) in big_edges if a != v and b != v]  # no `if not rest: continue` — deleting v may isolate every other vertex, which still fully separates the families
        comp = components(rest, verts)
        s13 = [p for p in (p1, p3) if p != v]
        s24 = [p for p in (p2, p4) if p != v]
        if not s13 or not s24:
            if (v == p1 == p3 or v == p2 == p4): # if p1 and p3 (or p2 and p4) are incident with the same vertex, then it must be a Glauber vertex.
                inc = [em[i] for i, (a, b) in enumerate(edges)
                       if (a == v or b == v) and ((a, b) in big_edges or (b, a) in big_edges)]
                if inc:
                    vm2[v] = 'G'
                    changed = True
            continue
        c13 = {comp[p] for p in s13}
        if any(comp[p] in c13 for p in s24):  # separating p1,p3 FROM p2,p4: 13/24-side survivors must share NO component (each side may split freely)
            continue
        vm2[v] = 'G'
        changed = True
    return vm2 if changed else None


def necklace_detect(edges, em, vm, ext_attach):
    em = list(em)
    verts = set(v for e in edges for v in e) | set(ext_attach.values())
    H_edges = [i for i, m in enumerate(em) if m == 'H']
    # connected components of the H-edge subgraph (vertex sets)
    adj = {v: set() for v in verts}
    for i in H_edges:
        a, b = edges[i]
        adj[a].add(b); adj[b].add(a)
    seen = set()
    comps = []
    for i in H_edges:
        a, b = edges[i]
        if a in seen:
            continue
        comp = {a}; stack = [a]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if w not in comp:
                    comp.add(w); stack.append(w)
        seen |= comp
        comps.append(comp)
    for comp in comps:
        comp_edges = [i for i in H_edges
                      if edges[i][0] in comp and edges[i][1] in comp]
        # NOT a necklace if it carries an external momentum: a self-energy loop strung between two Glauber lines; the jets' boundary G edge is the bridge
        if comp & set(ext_attach.values()):
            continue
        if len(comp_edges) < len(comp):  # (a) needs a cycle: a connected component has |E| >= |V| (parallel edges count separately)
            continue
        # (b) all boundary edges (one endpoint inside, one outside) are G
        bnd = [i for i, (a, b) in enumerate(edges)
               if (a in comp) != (b in comp)]
        if not bnd:
            continue
        if not all(em[i] == 'G' for i in bnd):
            continue
        for i in comp_edges:
            em[i] = 'sH'
    return em

def _recompute_vm(edges, em, ext_attach, ext_mode):
    verts = set(v for e in edges for v in e) | set(ext_attach.values())
    vm = {}
    for v in verts:
        accs = []
        for i, (a, b) in enumerate(edges):
            if a == v or b == v: accs.append(em[i])
        for n, vv in ext_attach.items():
            if vv == v: accs.append(ext_mode_for(ext_attach, ext_mode, n))
        vm[v] = vee(accs) if accs else 'H'
    return vm

# k0 mojetic, two parts, both required:
#   (1) each contracted jet is 1VI — H/G vertices adjacent to the jet identified with aux1; p1,p3 (resp. p2,p4) identified with aux2 (two extra legs); plus an edge (aux1,aux2);
#   (2) if H is nonempty, no cut vertex of H∪J13 (resp. H∪J24) separates p1,p3 (resp. p2,p4) from the rest.
def mojetic_k0_ok(edges, em, vm, fam13, fam24, ext_attach, ext_mode, verts):
    def jet_ok(fam, ext_names):
        fv = {v for v in vm if vm[v] in fam}
        fe = [e for e, m in zip(edges, em) if m in fam]
        # H/G vertices adjacent to the jet: jet-edge endpoint, or touching a jet vertex via any edge
        adj = set()
        for v in vm:
            if vm[v] in ('H', 'G') and v not in fv:
                if any(v == a or v == b for (a, b) in fe) or any(
                        (v == a or v == b) and (a in fv or b in fv)
                        for (a, b) in edges):
                    adj.add(v)
        vs = set(fv) | {'aux1', 'aux2'}
        es = set()
        for (a, b) in fe:
            es.add(('aux1' if a in adj or a not in vs else a,
                    'aux1' if b in adj or b not in vs else b))
        for n in ext_names:
            vv = ext_attach[n]
            vv2 = 'aux2' if vv in adj or vv not in vs else vv
            es.add((vv2, 'aux2'))
        es.add(('aux1', 'aux2'))
        return is_1vi(vs, es)

    if not jet_ok(fam13, ('p1', 'p3')):
        return False
    if not jet_ok(fam24, ('p2', 'p4')):
        return False
    # Part 2: if H is nonempty, no cut vertex of H∪J13 (resp. H∪J24) separates p1,p3 (resp. p2,p4) from the rest
    hv = {v for v in vm if vm[v] == 'H'}
    he = [(a, b) for (a, b), m in zip(edges, em) if m == 'H']
    if hv:
        def fam_ok(fam, ext1, ext2):
            fv = {v for v in vm if vm[v] in fam}
            fe = [(a, b) for (a, b), m in zip(edges, em) if m in fam]
            gv = hv | fv
            ge = he + fe
            pa, pb = ext_attach[ext1], ext_attach[ext2]
            for v in gv:
                rest = [(a, b) for (a, b) in ge if a != v and b != v]
                comp = components(rest, gv)
                c_fam = {comp[p] for p in (pa, pb) if p in gv}
                if not c_fam:
                    continue
                # separated iff some vertex lies outside every family-survivor component (family's own vertices count as survivors)
                for x in gv:
                    if x == v or x == pa or x == pb:
                        continue
                    if comp[x] not in c_fam:
                        return False
            return True
        if not fam_ok(fam13, 'p1', 'p3') or not fam_ok(fam24, 'p2', 'p4'):
            return False
    return True


# H∪J mojetic: identify the component's external momenta AND all its adjacent vertices (from other mode subgraphs) with a single aux vertex; each external momentum adds ONE edge (its leg ending on aux); the result must be 1VI.
def mojetic_ok(hv, he, jv, je, gv, edges, ext_attach, ext_names):
    vs = set(hv) | set(jv) | {'aux'}
    es = set(he) | set(je)
    adj = {v for v in gv if v not in vs and any(
        (v == a and b in jv) or (v == b and a in jv) for (a, b) in edges)}
    if adj:
        es = {('aux' if a in adj else a, 'aux' if b in adj else b)
              for (a, b) in es}
    for n in ext_names:
        vv = ext_attach[n]
        if vv in vs:
            es.add((vv, 'aux'))
    return is_1vi(vs, es)



# Biconnected components (blocks) of (verts, edges): bridges appear as single-edge blocks; self-loops and isolated vertices as single-vertex blocks.
def biconnected_blocks(verts, edges):
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

# 1VI blocks (biconnected components) of the contracted X-subgraph γ̃_X: X edges + X vertices (join-mode), non-X endpoints absorbed into aux.
# Blocks sharing a cut vertex stay SEPARATE — NOT connected components; IR compatibility is per-component (each block confirmed separately).
# Each block is (vertex_set, contracted_edge_list, original_edge_indices): the ORIGINAL indices are needed because distinct self-loops (1,3) and (7,8) both contract to (aux,aux) and are otherwise indistinguishable.
def mode_components(mode, vm, em, edges, verts):
    gv = {v for v in verts if vm.get(v) == mode}
    ge = [e for e, m in zip(edges, em) if m == mode]
    if not gv and not ge:
        return []
    ge_idx = [i for i, (e, m) in enumerate(zip(edges, em)) if m == mode]
    verts2 = set(gv) | {'aux'}
    edges2 = []
    for (a, b) in ge:
        edges2.append(('aux' if a not in gv else a, 'aux' if b not in gv else b))
    raw = biconnected_blocks(verts2, edges2)
    out = []
    used = set()
    for (bv, be) in raw:
        # map contracted edges back to original indices via position in edges2 (each position belongs to exactly ONE block)
        idxs = []
        for ce in be:
            for j, c2 in enumerate(edges2):
                if j in used or c2 != ce:
                    continue
                used.add(j)
                idxs.append(ge_idx[j])
                break
        out.append((bv, be, idxs))
    return out


# External momentum reaches the block ALONG ITS OWN MODE: BFS from v_ext through vm==mode vertices and em==mode edges to the block's real vertices.
# The source must not be G/H/sH (same source filter as relevant(); intermediates are already restricted to vm==mode).
def same_mode_reaches(vext, blk, mode, edges, em, vm):
    if vm.get(vext) in ('G', 'H', 'sH'):
        return False
    tgt = {v for v in blk[0] if v != 'aux'}
    if not tgt:
        return False
    seen = {vext}
    stack = [vext]
    while stack:
        v = stack.pop()
        if v in tgt:
            return True
        for i, (a, b) in enumerate(edges):
            if em[i] != mode:
                continue
            w = b if a == v else (a if b == v else None)
            if w is None or w in seen:
                continue
            if vm.get(w) != mode and w not in tgt:
                continue
            seen.add(w)
            stack.append(w)
    return False


# S^2 IR compatibility: relevant to a confirmed X1 component and a confirmed X2 component with S^2 = X1 ∧ X2 (13-family × 24-family covers every way to reach S^2 = X1∧X2).
def s2_comp_confirmed(blk, comps, confirmed, edges, em, vm):
    for mode1 in J13_FAM | {'S^1C13', 'S^2C13'}:
        for i1, comp1 in enumerate(comps[mode1]):
            if not confirmed.get((mode1, i1)):
                continue
            if not relevant(blk, comp1, mode1, edges, em, vm, 'S^2'):
                continue
            for mode2 in J24_FAM | {'S^1C24', 'S^2C24'}:
                for i2, comp2 in enumerate(comps[mode2]):
                    if not confirmed.get((mode2, i2)):
                        continue
                    if not relevant(blk, comp2, mode2, edges, em, vm, 'S^2'):
                        continue
                    if meet(mode1, mode2) == 'S^2':
                        return True


# Generic meet-of-two confirmation for the S^m C_i^n C_ij family (m,n >= 1): a component of mode `mode` is IR compatible if it is relevant to two already-confirmed components X1, X2 with X1 ∧ X2 = mode.
# Same rule as s2_comp_confirmed (S^2) without the 13×24 restriction — any confirmed pair whose meet hits the mode counts.
def sc2_gen_confirmed(blk, comps, confirmed, edges, em, vm, mode):
    for mode1, lst1 in comps.items():
        for i1, comp1 in enumerate(lst1):
            if not confirmed.get((mode1, i1)):
                continue
            if not relevant(blk, comp1, mode1, edges, em, vm, mode):
                continue
            for mode2, lst2 in comps.items():
                for i2, comp2 in enumerate(lst2):
                    if not confirmed.get((mode2, i2)):
                        continue
                    if not relevant(blk, comp2, mode2, edges, em, vm, mode):
                        continue
                    if meet(mode1, mode2) == mode:
                        return True
    return False


# S IR compatibility: relevant to >=1 confirmed 13-family component and >=1 confirmed 24-family component (S = C13 ∧ C24 automatically).
def s_comp_confirmed(blk, comps, confirmed, edges, em, vm):
    hit13 = hit24 = False
    for mode in ('C13', 'C1C13', 'C3C13'):
        for i, comp in enumerate(comps[mode]):
            if confirmed.get((mode, i)) and relevant(blk, comp, mode, edges, em, vm, 'S'):
                hit13 = True
    for mode in ('C24', 'C2C24', 'C4C24'):
        for i, comp in enumerate(comps[mode]):
            if confirmed.get((mode, i)) and relevant(blk, comp, mode, edges, em, vm, 'S'):
                hit24 = True
    return hit13 and hit24


# Component confirmed by momentum flow of a CONFIRMED SC component: an SC edge of the confirmed block touches vertex v in blk (or the SC component is relevant to blk, path-based) and join(sc_mode, vm[v]) == vm[v] (vee rule keeps the mode).
# For C13/C24 (ext_names given) an external momentum must also be inside or relevant to the block, and the vee over the relevant external modes must keep the mode too.
def sc_flow_confirms(mode, sc_mode, blk, comps, confirmed, edges, em, vm,
                     ext_attach=None, ext_names=None, ext_mode=None):
    blk_verts = blk[0]
    if ext_names is not None:
        if not any(ext_attach[ext] in blk_verts for ext in ext_names):
            # the external momentum counts if RELEVANT to the block (path-based, G/sH excluded)
            ok = False
            if ext_mode is not None:
                for ext in ext_names:
                    vext = ext_attach[ext]
                    ext_comp = ({vext, 'aux'}, [(vext, 'aux')])
                    if relevant(ext_comp, blk, mode, edges, em, vm,
                                ext_mode_for(ext_attach, ext_mode, ext)):
                        ok = True
                        break
            if not ok:
                return False
    # C13/C24 vee-keeping: vee(sc_mode, relevant ext modes) must equal mode, else the block collapses
    ext_modes = []
    if ext_names is not None and ext_mode is not None:
        for n in ext_names:
            vext = ext_attach[n]
            em_ext = ext_mode_for(ext_attach, ext_mode, n)
            if vext in blk_verts:
                ext_modes.append(em_ext)
            else:
                ext_comp = ({vext, 'aux'}, [(vext, 'aux')])
                if relevant(ext_comp, blk, mode, edges, em, vm, em_ext):
                    ext_modes.append(em_ext)
    def _vee_ok(sc_mode):
        acc = sc_mode
        for m2 in ext_modes:
            acc = join(acc, m2)
        return acc == mode
    for i, (sv, se, *sidx) in enumerate(comps[sc_mode]):
        if not confirmed.get((sc_mode, i)):
            continue
        if relevant((sv, se, *sidx), blk, mode, edges, em, vm, sc_mode):  # SC flow = path-based relevance, not only direct contact
            for v in blk_verts:
                if v != 'aux' and join(sc_mode, vm.get(v)) == vm.get(v):  # vee rule: needs a real v whose mode is kept (join == vm[v])
                    # vee inputs confirmed: vm[v] is circular for refined modes (C13/C24 get identity from ext_names)
                    if ext_names is None and not _vee_identity_ok(
                            v, vm.get(v), ext_attach, ext_mode, comps,
                            confirmed):
                        continue
                    if ext_names is not None and not _vee_ok(sc_mode):
                        continue
                    if _scflow_gate_port_ok(blk, mode, v, sv, se, sidx,
                                            edges, em, vm, comps):
                        return True
                    continue
        # legacy direct-contact fallback (relevant() already covers it; kept explicit for clarity)
        for ei, (a, b) in enumerate(edges):
            if em[ei] != sc_mode:
                continue
            if sidx:
                if ei not in sidx[0]:
                    continue
            else:
                a2 = a if vm.get(a) == sc_mode else 'aux'
                b2 = b if vm.get(b) == sc_mode else 'aux'
                if (a2, b2) not in se and (b2, a2) not in se:
                    continue
            for v in (a, b):
                if v in blk_verts and v != 'aux':
                    if join(sc_mode, vm.get(v)) != vm.get(v):
                        continue
                    # vee inputs confirmed: vm[v] is circular for refined modes (C13/C24 get identity from ext_names)
                    if ext_names is None and not _vee_identity_ok(
                            v, vm.get(v), ext_attach, ext_mode, comps,
                            confirmed):
                        continue
                    if ext_names is not None and not _vee_ok(sc_mode):
                        continue
                    if _scflow_gate_port_ok(blk, mode, v, sv, se, sidx,
                                            edges, em, vm, comps):
                        return True
                    continue
    return False


# vee(X1,X2)=X needs both inputs confirmed.  vm[v] alone is the block's own (possibly unconfirmed) identity — circular for refined-mode components (C1C13/C3C13 etc.) without an ext_names gate.
# Identity source: v is an external vertex whose input mode == vmode, or v belongs to an already-confirmed component block.
def _vee_identity_ok(v, vmode, ext_attach, ext_mode, comps, confirmed):
    if ext_attach is not None and ext_mode is not None:
        for n, vv in ext_attach.items():
            if vv == v and ext_mode_for(ext_attach, ext_mode, n) == vmode:
                return True
    for m2, lst2 in comps.items():
        for i2, blk2 in enumerate(lst2):
            if not confirmed.get((m2, i2)):
                continue
            if v in blk2[0]:
                return True
    return False

# Glauber propagator compatibility: a cut splits its endpoints with p1,p3 in one connected component and p2,p4 in the other.
def ir_glauber_ok(e, verts, edges, p1, p2, p3, p4):
    u, v = e
    vs = list(verts)
    n = len(vs)
    for mask in range(1, 1 << n):
        A = {vs[i] for i in range(n) if (mask >> i) & 1}
        if p1 not in A or p3 not in A: continue
        B = verts - A
        if p2 not in B or p4 not in B: continue
        if not ((u in A and v in B) or (v in A and u in B)): continue
        if not connected(A, edges): continue
        if not connected(B, edges): continue
        return True
    return False


# γ1 (SC/S block) relevant to γ2 (C-mode component):
#   (i) 𝒳(γ1) marginally softer than 𝒳(γ2): V[sc_mode] > V[dst_mode];
#   (ii) a path from γ1 to γ2 whose modes are non-increasing at EVERY step (vertices AND edges) — not geometric adjacency.
def relevant(blk, comp, dst_mode, edges, em, vm, sc_mode):
    if not marginally_softer(sc_mode, dst_mode):
        return False
    bv, be, *rest = blk
    sidx = rest[0] if rest else None
    cv, _, *_ = comp
    src = {v for v in bv if v != 'aux'}
    tgt = {v for v in cv if v != 'aux'}
    if not tgt:
        return False
    # Direct contact: one edge of γ1 (em == sc_mode, in this block) incident with a vertex of γ2 ⟹ path of length 1 ⟹ relevant.
    # Membership by ORIGINAL edge index (distinct self-loops both contract to (aux,aux) and are indistinguishable otherwise).
    for ei, (a, b) in enumerate(edges):
        if em[ei] != sc_mode:
            continue
        if sidx is not None:
            if ei not in sidx:
                continue
        else:
            a2 = a if vm.get(a) == sc_mode else 'aux'
            b2 = b if vm.get(b) == sc_mode else 'aux'
            if (a2, b2) not in be and (b2, a2) not in be:
                continue
        if a in tgt or b in tgt:
            return True
    # external-momentum component (only edge (v_ext, aux)): direct contact = the external vertex itself lies in the target block
    if src and len(be) == 1 and 'aux' in be[0]:
        v_ext = next(iter(src))
        if v_ext in tgt:
            return True
    if not src:
        # a block with no real vertex (pure self-loop): its edge ENDPOINTS are the path entry points
        for ei, (a, b) in enumerate(edges):
            if em[ei] != sc_mode:
                continue
            if sidx is not None:
                if ei not in sidx:
                    continue
            else:
                a2 = a if vm.get(a) == sc_mode else 'aux'
                b2 = b if vm.get(b) == sc_mode else 'aux'
                if (a2, b2) not in be and (b2, a2) not in be:
                    continue
            if a != 'aux': src.add(a)
            if b != 'aux': src.add(b)
    if not src:
        return False
    adj = {v: [] for v in vm}
    for i, (a, b) in enumerate(edges):
        adj[a].append((b, i)); adj[b].append((a, i))
    # state = (vertex, last V); V non-increasing at every step; the first edge needs V[e] <= V[vm[v]] (the vertex's own mode, NOT V[sc_mode] — the src may be an endpoint of a harder mode)
    # G/H/sH cannot conduct IR info — not even as path start.
    src = {v for v in src if vm.get(v) not in ('G', 'H', 'sH')}
    if not src:
        return False
    seen = set(src)
    stack = [(v, V[vm.get(v, 'H')]) for v in src]
    while stack:
        v, last_V = stack.pop()
        for w, ei in adj[v]:
            if w in seen:
                continue
            if vm.get(w) in ('G', 'H', 'sH'):  # no passing through G/H/sH: no longitudinal info / off-shell absorber / necklace-internal
                continue
            e_V = V[em[ei]]
            if e_V > last_V:  # vertex -> edge step: V non-increasing (edge not softer than the vertex it leaves)
                continue
            w_V = V[vm.get(w, 'H')]
            if w_V > e_V:  # edge -> vertex step: arrived vertex must not be softer than the edge it arrives through
                continue
            if w in tgt:
                return True
            seen.add(w)
            stack.append((w, w_V))
    return False


def sc_hidden_path_confirms(blk, i, comps, confirmed, edges, em, vm,
                             ext_attach, ext_mode, sc_mode, pair_mode,
                             ext_a, ext_b, fam_modes, ext_a_fam, ext_b_fam):
    # Generic SC "hidden path", Regge-only: if an SC component (sc_mode) is relevant to TWO DISTINCT pair_mode components — one attached by ext_a's C_a^{m_a}·pair external momentum (mode in ext_a_fam), the other by ext_b's C_b^{m_b}·pair (in ext_b_fam), m>=0 incl. ∞ — and to at least one component of a fam_modes mode, then ONE pair component confirmed ⟹ the OTHER is confirmed too.
    # The SC itself is NOT confirmed by this rule: the path only conducts confirmation between the two pair components.  Called for an unconfirmed pair block i; returns True iff a qualifying SC exists and the partner pair component is confirmed.
    va = ext_attach[ext_a]
    vb = ext_attach[ext_b]
    ma = ext_mode_for(ext_attach, ext_mode, ext_a)
    mb = ext_mode_for(ext_attach, ext_mode, ext_b)
    # C_a^{m_a}·pair / C_b^{m_b}·pair with m >= 0 (incl. ∞ = lightlike).
    if ma not in ext_a_fam or mb not in ext_b_fam:
        return False
    # receiver must still pass the port requirement (conduction is not an exemption); excludes the union of all source entries  (R068_v8 k3, 2026-09-13)
    if not _hp_receiver_port_ok(blk, pair_mode, edges, em, vm, ext_attach, ext_mode,
                                confirmed, comps):
        return False
    cv = {v for v in blk[0] if v != 'aux'}
    has_a = va in cv
    has_b = vb in cv
    if has_a == has_b:
        # must be attached by exactly one of ext_a, ext_b (distinct comps)
        return False
    partner = None
    for j, cj in enumerate(comps[pair_mode]):
        if j == i:
            continue
        cjv = {v for v in cj[0] if v != 'aux'}
        if has_a and vb in cjv:
            partner = j
            break
        if has_b and va in cjv:
            partner = j
            break
    if partner is None:
        return False
    if not confirmed.get((pair_mode, partner)):
        return False
    # one SC component relevant to blk, to the partner, AND to at least one fam_modes component
    for sblk in comps[sc_mode]:
        if not relevant(sblk, blk, pair_mode, edges, em, vm, sc_mode):
            continue
        if not relevant(sblk, comps[pair_mode][partner], pair_mode, edges,
                        em, vm, sc_mode):
            continue
        ok_fam = False
        for fm in fam_modes:
            for cblk in comps[fm]:
                if relevant(sblk, cblk, fm, edges, em, vm, sc_mode):
                    ok_fam = True
                    break
            if ok_fam:
                break
        if ok_fam:
            return True
    return False


# SC block adjacent to a C-mode component: share a vertex, or an SC edge of this block (em == sc_mode) touches a component vertex.
# Self-loop SC edges (both ends absorbed into aux) are matched via the contracted edge list.
def adjacent(blk, comp, edges, em, vm, sc_mode):
    bv, be = blk
    cv, _, *_ = comp
    if {v for v in bv if v != 'aux'} & cv:
        return True
    for (a, b), m in zip(edges, em):
        if m != sc_mode: continue
        a2 = a if vm.get(a) == sc_mode else 'aux'
        b2 = b if vm.get(b) == sc_mode else 'aux'
        if (a2, b2) in be or (b2, a2) in be:
            if a in cv or b in cv:
                return True
    return False

# SC13/SC24 cond-1 — momentum-flow confirmation: EXISTS a cut of the SC component into two pieces such that for ONE piece, the line momenta of CONFIRMED components crossing into it (edge belongs to a confirmed component, relevant to the piece) have ∨(inflows) == mode.
# Single-real-vertex blocks take the degenerate cut (whole block one piece, aux alone).  External momenta do not participate: an external alone can never ∨ to an SC mode.
def sc_cond1_confirms(blk, mode, comps, confirmed, edges, em, vm,
                      ext_attach, ext_mode):
    bv, be, *rest = blk
    sidx = rest[0] if rest else []
    real = [v for v in bv if v != 'aux']
    n = len(real)
    if n == 0:
        return False
    # the third-port requirement is part of condition 1 (same as cond1_confirms; ported from wide-angle / fri23, 2026-09-13)
    def _port_ok():
        return _port_ok_regge(blk, mode, edges, em, vm, ext_attach, ext_mode, confirmed, comps)
    def inflows_of(A):
        inflow = []
        tgt = (set(A), [])
        for ei, (a, b) in enumerate(edges):
            m = em[ei]
            if m in ('H', 'G', 'sH'):
                continue
            for i, (sv, se, *sidx2) in enumerate(comps.get(m, ())):
                if not confirmed.get((m, i)):
                    continue
                if sidx2 and ei in sidx2[0]:
                    if m == mode:
                        # same-mode line of a confirmed block: direct contact with A suffices (as in cond1_confirms)
                        if a in A or b in A:
                            inflow.append(m)
                    else:
                        # line momentum of a confirmed component CROSSING INTO the piece: the edge touches A (direct contact) or reaches A via a monotone path.
                        # (Direct contact flows along the line regardless of path monotonicity — relevant() would reject a line softer than its own endpoints.)
                        if a in A or b in A:
                            inflow.append(m)
                        else:
                            line_comp = ({a, b}, [(a, b)])
                            if relevant(line_comp, tgt, mode, edges, em,
                                        vm, m):
                                inflow.append(m)
                    break
        return inflow
    if n == 1 or not any((a in set(real)) != (b in set(real))
                         for (a, b) in be):
        inflow = inflows_of(set(real))
        if not inflow:
            return False
        acc = inflow[0]
        for m2 in inflow[1:]:
            acc = join(acc, m2)
        return acc == mode and _port_ok()
    for mask in range(1, 1 << n):
        A = {real[i] for i in range(n) if (mask >> i) & 1}
        if not A:
            continue
        # the cut must actually separate the block: some block edge crosses
        if not any((a in A) != (b in A) for (a, b) in be):
            continue
        inflow = inflows_of(A)
        if not inflow:
            continue
        acc = inflow[0]
        for m2 in inflow[1:]:
            acc = join(acc, m2)
        if acc == mode:
            return _port_ok()
    return False


# SC13/SC24 component confirmation: (1) relevant to all of {C24, C1C13, C3C13} (resp. {C13, C2C24, C4C24}) with >=1 confirmed; (2) relevant to two of them, both confirmed.  Relevance is path-based (mode monotone), not geometric adjacency.
def sc_confirmed_rule(blk, rel_modes, comps, confirmed, edges, em, vm,
                      sc_mode):
    adj = {}
    for mode in rel_modes:
        for i, comp in enumerate(comps[mode]):
            if relevant(blk, comp, mode, edges, em, vm, sc_mode):
                adj.setdefault(mode, set()).add(i)
    if all(mode in adj for mode in rel_modes):
        if any(confirmed.get((mode, i)) for mode in rel_modes
               for i in adj[mode]):
            return True
    for (m1, m2) in ((rel_modes[0], rel_modes[1]), (rel_modes[0], rel_modes[2]),
                     (rel_modes[1], rel_modes[2])):
        if m1 in adj and m2 in adj:
            # any confirmed block per target mode suffices — the other (unconfirmed) blocks it touches need not be confirmed first
            if any(confirmed.get((m1, i)) for i in adj[m1]) and \
               any(confirmed.get((m2, i)) for i in adj[m2]):
                return True
    return False


# S²C13/S²C24 component confirmation:
# S²C13 (req=C1²C13,C3²C13, fam=SC24,C2C24,C4C24): (1) relevant to all req_modes AND at least one fam_mode, with >=1 confirmed among all targets; (2) relevant to two confirmed target modes from req+fam, NOT both from fam (at least one from req).
# S²C24 symmetric (req=C2²C24,C4²C24, fam=SC13,C1C13,C3C13).
# The 24-side flow need not be SC24 itself: relevant to C2C24/C4C24 via a monotone path is equally valid.
def s2c_confirmed_rule(blk, fam_modes, req_modes, comps, confirmed,
                       edges, em, vm, sc_mode):
    adj = {}
    for mode in fam_modes + req_modes:
        for i, comp in enumerate(comps[mode]):
            if relevant(blk, comp, mode, edges, em, vm, sc_mode):
                adj.setdefault(mode, set()).add(i)
    # (1) all required modes + at least one family mode
    if all(mode in adj for mode in req_modes) and \
       any(mode in adj for mode in fam_modes):
        targets = [m for m in fam_modes + req_modes if m in adj]
        if any(confirmed.get((m, i)) for m in targets for i in adj[m]):
            return True
    # (2) two target modes with >=1 confirmed relevant block each (per-family exists), not both from fam_modes
    all_modes = fam_modes + req_modes
    for i1 in range(len(all_modes)):
        for i2 in range(i1 + 1, len(all_modes)):
            m1, m2 = all_modes[i1], all_modes[i2]
            if m1 in fam_modes and m2 in fam_modes:
                continue
            if m1 in adj and m2 in adj:
                if any(confirmed.get((m1, i)) for i in adj[m1]) and \
                   any(confirmed.get((m2, i)) for i in adj[m2]):
                    return True
    return False


# Relevance to an H block: like relevant(), but the path may START at an H/G vertex of the source (an external-momentum vertex can be H-mode) and may END at the H target.
# G/sH still cannot be traversed mid-path (G carries no longitudinal info; sH lives inside a necklace); H vertices ARE traversable (hard momentum flows inside the blob).
# The path may not pass through EXTERNAL-MOMENTUM vertices (sources/sinks — except H-mode externals and allow_ext_mid), nor use the cut's crossing edges (inflows enter without crossing the cut).
def h_relevant(comp, tgt, edges, em, vm, extvs, sc_mode, start_V=None,
               cross_idx=(), allow_ext_mid=False):
    bv, be, *rest = comp
    sidx = rest[0] if rest else None
    src = {v for v in bv if v != 'aux'}
    if not src:
        for ei, (a, b) in enumerate(edges):
            if em[ei] != sc_mode:
                continue
            if sidx is not None:
                if ei not in sidx:
                    continue
            else:
                a2 = a if vm.get(a) == sc_mode else 'aux'
                b2 = b if vm.get(b) == sc_mode else 'aux'
                if (a2, b2) not in be and (b2, a2) not in be:
                    continue
            if a != 'aux': src.add(a)
            if b != 'aux': src.add(b)
    if not src:
        return False
    if any(v in tgt for v in src):
        return True
    # direct contact: an edge of the source block incident with tgt
    for ei, (a, b) in enumerate(edges):
        if ei in cross_idx:
            continue
        if em[ei] != sc_mode:
            continue
        if sidx is not None:
            if ei not in sidx:
                continue
        else:
            a2 = a if vm.get(a) == sc_mode else 'aux'
            b2 = b if vm.get(b) == sc_mode else 'aux'
            if (a2, b2) not in be and (b2, a2) not in be:
                continue
        if a in tgt or b in tgt:
            return True
    adj = {v: [] for v in vm}
    for i, (a, b) in enumerate(edges):
        adj[a].append((b, i)); adj[b].append((a, i))
    seen = set(src)
    stack = [(v, start_V if start_V is not None else V[vm.get(v, 'H')])
             for v in src]
    while stack:
        v, last_V = stack.pop()
        for w, ei in adj[v]:
            if ei in cross_idx:
                continue
            if w in seen:
                continue
            e_V = V[em[ei]]
            if e_V > last_V:
                continue
            w_V = V[vm.get(w, 'H')]
            if w_V > e_V:
                continue
            if w in tgt:
                return True
            # the pass-through restrictions apply only to intermediate vertices — the path may END at an external or H vertex of the target
            if vm.get(w) in ('G', 'sH'):
                continue
            if w in extvs:
                # external-momentum vertices are sources/sinks mid-path — except H-mode ones (the hard blob conducts) and allow_ext_mid flows
                if not allow_ext_mid and vm.get(w) != 'H':
                    continue
            # H vertices ARE traversable mid-path: hard momentum flows freely inside the hard blob (G/sH remain blocked)
            seen.add(w)
            stack.append((w, w_V))
    return False


# H-component IR compatibility — FULL cond1, the two conditions together:
#   (1) there EXISTS a cut of the component (a proper bipartition with crossing component edges) that does NOT cut the whole diagram (removing the crossing edges keeps the whole graph connected);
#   (2) for one piece, the inflows — external momenta sitting in the piece + line momenta of CONFIRMED components with a raw endpoint in the piece (G/sH lines don't conduct) — have ∨(inflows) == H.
# The hard blob confirms as a whole; the cut is only a test device.
def h_comp_confirmed(blk, comps, confirmed, edges, em, vm, ext_attach,
                    ext_mode, all_verts):
    bv, be, *rest = blk
    hidxs = rest[0] if rest else []
    verts = [v for v in bv if v != 'aux']
    n = len(verts)
    if n == 0:
        return False
    vset = set(verts)
    extvs = set(ext_attach.values())

    # the search: a cut of the WHOLE DIAGRAM into connected A, B with one piece carrying exactly one of {p1,p3} and one of {p2,p4}, splitting the H component (H∩A, H∩B both nonempty); each piece's inflows must ∨ to H (piece_ok).  A cut isolating a single external is invalid.
    def relevant_ext(extn, A, cross_idx):
        vv = ext_attach[extn]
        m_ext = ext_mode_for(ext_attach, ext_mode, extn)
        if vv in A:
            return True
        ext_comp = ({vv, 'aux'}, [(vv, 'aux')])
        return h_relevant(ext_comp, A, edges, em, vm, extvs, m_ext,
                          start_V=V[vm.get(vv, 'H')],  # path starts at the external vertex with ITS OWN mode (H-mode external steps only on H edges)
                          cross_idx=cross_idx)

    if n == 1:  # degenerate: a single-vertex H component is a plain hard junction, trivially IR compatible
        return True

    allv = set(all_verts)
    Vlist = sorted(allv)
    Vn = len(Vlist)
    for mask in range(1, (1 << Vn) - 1):
        A = {Vlist[i] for i in range(Vn) if (mask >> i) & 1}
        B = allv - A
        if not connected(A, edges) or not connected(B, edges):
            continue
        n13 = len(A & {ext_attach['p1'], ext_attach['p3']})
        n24 = len(A & {ext_attach['p2'], ext_attach['p4']})
        if not (n13 == 1 and n24 == 1):
            continue
        HA = vset & A
        HB = vset & B
        if not HA or not HB:
            continue
        def h_connected(Sv):  # the cut must split H into exactly TWO connected pieces, else 3+ H components remain
            if len(Sv) <= 1:
                return True
            seen = set()
            stack = [next(iter(Sv))]
            seen.add(stack[0])
            while stack:
                x = stack.pop()
                for j in hidxs:
                    a, b = edges[j]
                    if a == x and b in Sv and b not in seen:
                        seen.add(b); stack.append(b)
                    elif b == x and a in Sv and a not in seen:
                        seen.add(a); stack.append(a)
            return seen == Sv
        if not h_connected(HA) or not h_connected(HB):
            continue
        cross = [j for j, (a, b) in enumerate(edges)
                 if (a in A) != (b in A)]
        exA = [e for e in ext_attach if ext_attach[e] in A]
        exB = [e for e in ext_attach if ext_attach[e] in B]
        # per-piece: ∨(inflows) == H — externals sitting/relevant in the piece, plus line momenta of CONFIRMED non-H components (G/sH don't conduct)
        def piece_ok(piece, exts):
            inflows = []
            for e in exts:
                if relevant_ext(e, piece, cross):
                    inflows.append(ext_mode_for(ext_attach, ext_mode, e))
            # line momentum of a CONFIRMED non-H component: enters the blob at an H vertex (h_relevant) and flows WITHIN it to the piece
            for j, (a, b) in enumerate(edges):
                m = em[j]
                if m in ('H', 'G', 'sH'):
                    continue
                for ci, (sv, se, *sidx) in enumerate(comps.get(m, ())):
                    if not confirmed.get((m, ci)):
                        continue
                    if sidx and j in sidx[0]:
                        # line momentum conducts within the hard blob regardless of the cut (crossing edges block only EXTERNAL paths)
                        if h_relevant((sv, se, *sidx), piece, edges, em, vm,
                                      extvs, m, allow_ext_mid=True):
                            inflows.append(m)
                        break
            if not inflows:
                return False
            acc = inflows[0]
            for m2 in inflows[1:]:
                acc = join(acc, m2)
            return acc == 'H'
        if not piece_ok(HA, exA):
            continue
        if not piece_ok(HB, exB):
            continue
        return True
    return False


# IR compatibility cond1 for C13/C24 (replaces the older separate cut-check / external-relevance / flow rules): an X component is confirmed if there EXISTS a cut of this component (not cutting the whole diagram) into two pieces, such that for one piece, some external momenta or some line momenta of already-confirmed mode components are RELEVANT to it, and the vee of these momentum modes is precisely X.
# Implementation: enumerate bipartitions of the block's real vertices; one side A gathers inflows = external momenta sitting in A (or relevant to A) + line momenta (edges of CONFIRMED components crossing into A, with the edge's far endpoint in the confirmed component).
# The cut edges of X itself are not inflows.  ∨(inflows) == mode ⟹ confirmed.
def cond1_confirms(blk, mode, comps, confirmed, edges, em, vm,
                   ext_attach, ext_mode):
    # NEW FORM v3 (2026-09-13 晚, 小马): no "cut".  Two momenta — external or
    # already-confirmed line momenta — must ENTER the component (strict
    # monotone entry walk with edge modes checked, first-touch stop), their
    # join == mode; a third port must remain for ONE admissible choice of
    # entry points (any combination).  Single momentum already of the mode:
    # allowed, port per entry.
    realV = {v for v in blk[0] if v != 'aux'}
    if not realV:
        return False
    def entry_walk(start_vs):
        src = {v for v in start_vs if vm.get(v) not in ('G', 'H', 'sH')}
        if not src:
            return set()
        adj = {v: [] for v in vm}
        for i, (a, b) in enumerate(edges):
            adj[a].append((b, i)); adj[b].append((a, i))
        seen = set(src)
        stack = [(v, V[vm.get(v, 'H')]) for v in src]
        touch = set()
        while stack:
            v, last = stack.pop()
            if v in realV:
                touch.add(v); continue
            for w, ei in adj[v]:
                if w in seen:
                    continue
                if vm.get(w) in ('G', 'H', 'sH'):
                    continue
                eV = V[em[ei]]
                if eV > last:
                    continue
                wV = V[vm.get(w, 'H')]
                if wV > eV:
                    continue
                seen.add(w); stack.append((w, wV))
        return touch
    cands = []          # (tag, md, entry_points)
    for extn, v0 in ext_attach.items():
        md = ext_mode_for(ext_attach, ext_mode, extn)
        if v0 in realV:
            cands.append((extn, md, {v0}))
            continue
        if md == mode or marginally_softer(md, mode):
            t = entry_walk([v0])
            if t:
                cands.append((extn, md, t))
    for (cm, ci) in list(confirmed.keys()):
        if cm != mode and not marginally_softer(cm, mode):
            continue
        sblk = comps[cm][ci]
        sidx = sblk[2] if len(sblk) > 2 and sblk[2] else []
        for ei in sidx:
            a, b = edges[ei]
            if cm == mode:
                ent = {w for w in (a, b) if w in realV}
                if ent:
                    cands.append(('%s#%d' % (cm, ci), cm, ent))
            else:
                t = entry_walk([a, b])
                if t:
                    cands.append(('%s#%d' % (cm, ci), cm, t))
    if not cands:
        return False
    for (_tag, md, ent) in cands:
        if md == mode:
            for a in sorted(ent):
                if _third_port_regge(blk, mode, {a}, comps.get(mode, []), vm, edges):
                    return True
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            m1, e1 = cands[i][1], cands[i][2]
            m2, e2 = cands[j][1], cands[j][2]
            try:
                if join(m1, m2) != mode:
                    continue
            except Exception:
                continue
            for a in e1:
                for b in e2:
                    if _third_port_regge(blk, mode, {a, b}, comps.get(mode, []), vm, edges):
                        return True
    return False


# ---- Third-port requirement (part of condition 1) ----
# Ported from the wide-angle cond1_strong and the 2->3 enumerator (2026-09-13): a component whose scale is
# assembled from momenta entering at {va, vb} must have a THIRD PORT w outside {va, vb} — either
#   (A) a same-mode vertex of the block shared with another same-mode component, or
#   (B) a harder-mode vertex reached by one of the block's own edges (the exit endpoint; G counts).
# A scale assembled from two sources with no exit port is scaleless; H components are exempt (momentum sinks).
USE_THIRD_PORT = True


def _harder_or_eq_regge(mw, X):
    if mw == X:
        return True
    if X == 'H':
        return False
    if mw == 'H':
        return True
    if mw == 'G':
        return True
    if mw == 'sH' or X in ('G', 'sH'):
        return False
    return V.get(mw, 99) <= V.get(X, 0)


def _collect_entries_regge(blk, start_vs, start_es, cur_mode, edges, em, vm):
    realV = {v for v in blk[0] if v != 'aux'}
    tgt = {('v', v) for v in realV}
    tgt |= {('e', ei) for ei in (blk[2] if len(blk) > 2 and blk[2] else [])}

    def mode_of(el):
        return vm.get(el[1], 'H') if el[0] == 'v' else em[el[1]]

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
            if nb in visited:
                continue
            m_nb = mode_of(nb)
            if m_nb in ('G', 'sH'):
                continue
            if _harder_or_eq_regge(m_nb, cur):
                visited.add(nb); queue.append((nb, m_nb))
    ev = set()
    for h in hits:
        if h[0] == 'v':
            if h[1] in realV:
                ev.add(h[1])
        else:
            for w in edges[h[1]]:
                if w in realV:
                    ev.add(w)
    return ev


def _third_port_regge(blk, X, entry_vs, comps_same, vm, edges):
    realV = {v for v in blk[0] if v != 'aux'}
    for w in sorted(realV):
        if w in entry_vs:
            continue
        if vm.get(w) != X:
            continue
        for c in comps_same:
            if c is blk:
                continue
            if w in {v for v in c[0] if v != 'aux'}:
                return ('A', w)
    idxs = blk[2] if len(blk) > 2 and blk[2] else []
    for ei in idxs:
        a, b = edges[ei]
        for w in (a, b):
            if w in realV or w in entry_vs:
                continue
            mw = vm.get(w)
            if mw is not None and mw != X and _harder_or_eq_regge(mw, X):
                return ('B', w)
    return None


def _rule3_gate_entry_walk(blk, start_vs, edges, em, vm):
    # strict monotone entry walk (edge modes checked at every step; first-touch stop)
    realV = {v for v in blk[0] if v != 'aux'}
    src = {v for v in start_vs if vm.get(v) not in ('G', 'H', 'sH')}
    if not src:
        return set()
    adj = {v: [] for v in vm}
    for i, (a, b) in enumerate(edges):
        adj[a].append((b, i)); adj[b].append((a, i))
    seen = set(src)
    stack = [(v, V[vm.get(v, 'H')]) for v in src]
    touch = set()
    while stack:
        v, last = stack.pop()
        if v in realV:
            touch.add(v); continue
        for w, ei in adj[v]:
            if w in seen:
                continue
            if vm.get(w) in ('G', 'H', 'sH'):
                continue
            eV = V[em[ei]]
            if eV > last:
                continue
            wV = V[vm.get(w, 'H')]
            if wV > eV:
                continue
            seen.add(w); stack.append((w, wV))
    return touch


def _refined_flow2_confirms(mode, blk, comps, confirmed, edges, em, vm,
                            ext_attach, ext_mode, ext):
    # cond1-skeleton flow for refined components (2026-09-14): the momenta that may
    # enter the block are the associated external and/or lines of already-confirmed
    # components (SC lines included, per-line not per-component), entering via strict
    # monotone first-touch walks; join == mode; ONE admissible combo must leave a
    # third port.  Replaces the old component-level Rule 3 (p-ext + relevant SC).
    realV = {v for v in blk[0] if v != 'aux'}
    cands = []
    v0 = ext_attach[ext]
    md0 = ext_mode_for(ext_attach, ext_mode, ext)
    if v0 in realV:
        cands.append((ext, md0, {v0}))
    elif md0 == mode or marginally_softer(md0, mode):
        t = _rule3_gate_entry_walk(blk, [v0], edges, em, vm)
        if t:
            cands.append((ext, md0, t))
    for (cm, ci) in list(confirmed.keys()):
        if cm != mode and not marginally_softer(cm, mode):
            continue
        sblk = comps[cm][ci]
        se = sblk[2] if len(sblk) > 2 and sblk[2] else []
        for ei in se:
            a, b = edges[ei]
            t = _rule3_gate_entry_walk(blk, [a, b], edges, em, vm)
            if t:
                cands.append(('%s#%d' % (cm, ci), em[ei], t))
    if not cands:
        return False
    comps_same = comps.get(mode, [])
    for (_tag, md, ent) in cands:
        if md == mode:
            for a2 in sorted(ent):
                if _third_port_regge(blk, mode, {a2}, comps_same, vm, edges):
                    return True
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            m1, e1 = cands[i][1], cands[i][2]
            m2, e2 = cands[j][1], cands[j][2]
            try:
                if join(m1, m2) != mode:
                    continue
            except Exception:
                continue
            for a2 in e1:
                for b2 in e2:
                    if _third_port_regge(blk, mode, {a2, b2}, comps_same, vm, edges):
                        return True
    return False


def _scflow_gate_port_ok(blk, mode, v, sv, se, sidx, edges, em, vm, comps):
    # port test for the SC-flow channel: exclude the identity vertex v and the
    # SC's first-touch entry points; require a third port
    ent = {v}
    idx_list = sidx[0] if sidx else []
    for ei in idx_list:
        a, b = edges[ei]
        ent |= _rule3_gate_entry_walk(blk, [a, b], edges, em, vm)
    return _third_port_regge(blk, mode, ent, comps.get(mode, []), vm, edges) is not None


def _port_sources_regge(blk, X, edges, em, vm, ext_attach, ext_mode, confirmed, comps):
    srcs = []
    realV = {v for v in blk[0] if v != 'aux'}
    for extn, v0 in ext_attach.items():
        md = ext_mode_for(ext_attach, ext_mode, extn)
        if v0 in realV:
            srcs.append((extn, md, {v0}, {v0}))
        else:
            if md == X or marginally_softer(md, X):
                c = _collect_entries_regge(blk, [v0], [], vm.get(v0, 'H'), edges, em, vm)
                if c:
                    srcs.append((extn, md, c, set()))
    for (cm, ci) in list(confirmed.keys()):
        if not marginally_softer(cm, X):
            continue
        sblk = comps[cm][ci]
        sv = {v for v in sblk[0] if v != 'aux'}
        se = sblk[2] if len(sblk) > 2 and sblk[2] else []
        c = _collect_entries_regge(blk, sv, se, cm, edges, em, vm)
        if c:
            srcs.append(('%s#%d' % (cm, ci), cm, c, set()))
    return srcs


def _port_ok_regge(blk, X, edges, em, vm, ext_attach, ext_mode, confirmed, comps):
    if not USE_THIRD_PORT:
        return True
    srcs = _port_sources_regge(blk, X, edges, em, vm, ext_attach, ext_mode, confirmed, comps)
    comps_same = comps.get(X, [])
    for (_tag, md, cands, att) in srcs:
        if md == X:
            for a in cands:
                if _third_port_regge(blk, X, {a}, comps_same, vm, edges):
                    return True
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            md1, c1, a1 = srcs[i][1], srcs[i][2], srcs[i][3]
            md2, c2, a2 = srcs[j][1], srcs[j][2], srcs[j][3]
            try:
                if join(md1, md2) != X:
                    continue
            except Exception:
                continue
            for a in c1:
                for b in c2:
                    if _third_port_regge(blk, X, {a, b}, comps_same, vm, edges):
                        return True
    return False


# Hidden-path receiver port check: exclude the UNION of all source entries, then require a third port.  (cond1's _port_ok_regge only triggers the single-source test for FULL-mode sources; here a single marginally-softer source like C3∞C13 must also trigger it.  R068_v8 k3 / K33a3_1_pi, 2026-09-13)
def _hp_receiver_port_ok(blk, X, edges, em, vm, ext_attach, ext_mode, confirmed, comps):
    if not USE_THIRD_PORT:
        return True
    srcs = _port_sources_regge(blk, X, edges, em, vm, ext_attach, ext_mode, confirmed, comps)
    entry = set()
    for (_tag, md, cands, att) in srcs:
        entry |= set(cands)
    return _third_port_regge(blk, X, entry, comps.get(X, []), vm, edges) is not None


# Regge IR compatibility — fixpoint over 1VI components, each mode confirmed by its own rule:
#   C13/C24: cond1 (cut + vee of inflows);
#   refined collinear (C1C13 ... C4C24): direct p-attach (equal-mode external), same-mode reach, SC/S²C momentum flow, or p-ext + confirmed SC/S²C relevant;
#   C1²C13 etc.: same with SC -> S²C;
#   SC13/SC24: relevance rule or cond1; S²C13/S²C24: relevance rule;
#   S²: meet-of-two; SC² family (S^m C_i^n C_ij): generic meet-of-two;
#   S: 13-family + 24-family relevance; H: full cond1;
#   G: its own cut check;
#   SC hidden path: an SC relevant to two distinct p1/p3-attached C13 (resp. p2/p4-attached C24) + one same-side refined component conducts confirmation from the confirmed pair component to the other (the SC itself is not confirmed).
# Any unconfirmed component at the end => False.
def ir_ok_region(edges, em, vm, ext_attach, ext_mode=None, sum_degrees=None):
    verts = set(v for e in edges for v in e) | set(ext_attach.values())
    p1, p2 = ext_attach['p1'], ext_attach['p2']
    p3, p4 = ext_attach['p3'], ext_attach['p4']
    comps = {}
    for m in ('H', 'C13', 'C24', 'C1C13', 'C3C13', 'C2C24', 'C4C24', 'S^1C13', 'S^1C24', 'S', 'S^2',
              'C1^2C13', 'C3^2C13', 'C2^2C24', 'C4^2C24', 'S^2C13', 'S^2C24',
              # leg-explicit SC² family (S^m C_i^n C_ij, m,n >= 1): confirmed by the generic meet-of-two rule
              'S^1C1C13', 'S^1C3C13', 'S^1C2C24', 'S^1C4C24'):
        # an X-mode component = a 1VI block of the contracted X-subgraph Γ̃_X (NOT a connected component)
        comps[m] = mode_components(m, vm, em, edges, verts)
    # H components: 1VI blocks like every other mode — each block needs its OWN confirming cut (a cut that confirms one block does not confirm another sharing a cut vertex with it).  aux-only blocks are contraction artifacts.
    comps['H'] = [b for b in mode_components('H', vm, em, edges, verts)
                if any(v != 'aux' for v in b[0])]

    # ---- tadpole gate (PROTOTYPE 2026-09-13; translated from fri23 _tadp / wide-angle primitives.py) ----
    def _m_of(md):
        if md == 'S':
            return 1
        if md.startswith('S^'):
            num = ''
            for ch in md[2:]:
                if ch.isdigit():
                    num += ch
                else:
                    break
            return int(num) if num else 1
        return 0

    def _is_pure_soft(md):
        return _m_of(md) >= 1 and 'C' not in md

    def _tadp(md, blk):
        if _m_of(md) < 1:
            return False
        realV = {v for v in blk[0] if v != 'aux'}
        idxs = blk[2] if len(blk) > 2 and blk[2] else []
        ends = set(realV)
        for ei in idxs:
            ends |= set(edges[ei])
        for w in ends:
            if vm.get(w) == 'H':
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
        for m2, lst2 in comps.items():
            if V.get(m2, 99) >= V.get(md, 0):
                continue
            for c2 in lst2:
                c2v = {v for v in c2[0] if v != 'aux'}
                c2i = c2[2] if len(c2) > 2 and c2[2] else []
                if _adj(idxs, realV, c2i, c2v):
                    return False
        return True

    def _vee_ok(md, blk):
        realV = {v for v in blk[0] if v != 'aux'}
        idxs = blk[2] if len(blk) > 2 and blk[2] else []
        ends = set(realV)
        for ei in idxs:
            ends |= set(edges[ei])
        pool = []
        for i2 in range(len(edges)):
            u, w = edges[i2]
            if (u in ends or w in ends) and 'C' in em[i2]:
                pool.append(em[i2])
        for nm, v in ext_attach.items():
            if v in ends:
                pool.append(ext_mode_for(ext_attach, ext_mode, nm))
        if len(pool) < 2:
            return False
        for ai in range(len(pool)):
            for bi in range(ai + 1, len(pool)):
                if join(pool[ai], pool[bi]) == md:
                    return True
        return False

    _tad_cache, _vee_cache = {}, {}

    def _cond_allowed(md, blk):
        t = _tad_cache.get(id(blk))
        if t is None:
            t = _tadp(md, blk)
            _tad_cache[id(blk)] = t
        if not t:
            return True
        if not _is_pure_soft(md):
            return False
        ok = _vee_cache.get(id(blk))
        if ok is None:
            ok = _vee_ok(md, blk)
            _vee_cache[id(blk)] = ok
        return ok
    # (the "pure-aux carrier" check is a CONSEQUENCE of IR compatibility, not an independent constraint — a pure-aux SC component fails the relevance rules automatically; removing it changes nothing)
    confirmed = {}
    # initial round: C13/C24 by cond1 (cut-check).
    for m in ('C13', 'C24'):
        for i in range(len(comps[m])):
            if cond1_confirms(comps[m][i], m, comps, confirmed, edges, em, vm,
                              ext_attach, ext_mode):
                confirmed[(m, i)] = True
    for e, m in zip(edges, em):
        if m == 'G':
            if not ir_glauber_ok(e, verts, edges, p1, p2, p3, p4):
                return False
    for _ in range(30):
        changed = False
        # refined-collinear components: direct p-attach, same-mode reach, SC momentum flow, or p-ext + confirmed SC relevant
        for mode, ext, sc_mode in (('C1C13', 'p1', 'S^1C13'), ('C3C13', 'p3', 'S^1C13'),
                                   ('C2C24', 'p2', 'S^1C24'), ('C4C24', 'p4', 'S^1C24')):
            for i, blk in enumerate(comps[mode]):
                if (mode, i) in confirmed: continue
                ok = False
                # direct p-attach: an external whose mode EQUALS the block mode (p_i itself C1C13 etc., p_i² ~ λ²) confirms it without SC inflow
                vext = ext_attach[ext]
                if vext in blk[0] and \
                        ext_mode_for(ext_attach, ext_mode, ext) == mode:
                    ok = True
                # Rule 4: same-mode external propagation — monotone path from v_ext along its own mode confirms the block
                if not ok and \
                        ext_mode_for(ext_attach, ext_mode, ext) == mode and \
                        same_mode_reaches(vext, blk, mode, edges, em, vm):
                    ok = True
                # Rule 2: SC momentum flow keeping the refined mode (∨(SC, vm[v]) = vm[v]); Rule 1 (p-attach + SC line) deleted — rule 3 covers it
                if not ok and sc_flow_confirms(mode, sc_mode, blk, comps, confirmed,
                                      edges, em, vm):
                    ok = True
                if not ok:
                    # Rule 3 (cond1 skeleton, per-line; 2026-09-14): associated p-ext + a
                    # line of an already-confirmed component (SC lines included), entries =
                    # strict first-touch walks, third port required for one admissible combo.
                    ok = _refined_flow2_confirms(mode, blk, comps, confirmed, edges,
                                                 em, vm, ext_attach, ext_mode, ext)
                if ok:
                    confirmed[(mode, i)] = True
                    changed = True
        # refined-collinear C³ components (C1²C13 etc.): same rules as C² with SC -> S²C
        for mode, ext, s2c_mode in (('C1^2C13', 'p1', 'S^2C13'),
                                    ('C3^2C13', 'p3', 'S^2C13'),
                                    ('C2^2C24', 'p2', 'S^2C24'),
                                    ('C4^2C24', 'p4', 'S^2C24')):
            for i, blk in enumerate(comps[mode]):
                if (mode, i) in confirmed: continue
                ok = False
                # direct p-attach for m_i-finite externals whose mode equals the block mode
                vext = ext_attach[ext]
                if vext in blk[0] and \
                        ext_mode_for(ext_attach, ext_mode, ext) == mode:
                    ok = True
                # Rule 4: same-mode external propagation (as above)
                if not ok and \
                        ext_mode_for(ext_attach, ext_mode, ext) == mode and \
                        same_mode_reaches(vext, blk, mode, edges, em, vm):
                    ok = True
                if not ok and sc_flow_confirms(mode, s2c_mode, blk, comps, confirmed,
                                    edges, em, vm):
                    ok = True
                if not ok:
                    # Rule 3 (cond1 skeleton, per-line; 2026-09-14) — as above, for C³ blocks.
                    ok = _refined_flow2_confirms(mode, blk, comps, confirmed, edges,
                                                 em, vm, ext_attach, ext_mode, ext)
                if ok:
                    confirmed[(mode, i)] = True
                    changed = True
        # SC components via relevance
        for mode, rel_modes in (('S^1C13', ('C24', 'C1C13', 'C3C13')),
                                ('S^1C24', ('C13', 'C2C24', 'C4C24'))):
            for i, blk in enumerate(comps[mode]):
                if (mode, i) in confirmed: continue
                if _cond_allowed(mode, blk) and sc_confirmed_rule(blk, rel_modes, comps, confirmed,
                                     edges, em, vm, mode):
                    confirmed[(mode, i)] = True
                    changed = True
                elif sc_cond1_confirms(blk, mode, comps, confirmed,
                                       edges, em, vm, ext_attach, ext_mode):
                    # SC cond-1: momentum flow of confirmed components, ∨(inflows) == mode
                    confirmed[(mode, i)] = True
                    changed = True
        # S²C components via relevance (see s2c_confirmed_rule).
        for mode, req_modes, fam_modes in (
                ('S^2C13', ('C1^2C13', 'C3^2C13'), ('S^1C24', 'C2C24', 'C4C24')),
                ('S^2C24', ('C2^2C24', 'C4^2C24'), ('S^1C13', 'C1C13', 'C3C13'))):
            for i, blk in enumerate(comps[mode]):
                if (mode, i) in confirmed: continue
                if _cond_allowed(mode, blk) and s2c_confirmed_rule(
                        blk, fam_modes, req_modes, comps,
                        confirmed, edges, em, vm, mode):
                    confirmed[(mode, i)] = True
                    changed = True
        # S² components: relevant to confirmed X1 (13-family) and X2 (24-family) with S² = X1 ∧ X2
        for i, blk in enumerate(comps['S^2']):
            if ('S^2', i) in confirmed: continue
            if _cond_allowed('S^2', blk) and s2_comp_confirmed(
                    blk, comps, confirmed, edges, em, vm):
                confirmed[('S^2', i)] = True
                changed = True
        # SC²-family components (S^m C_i^n C_ij, m,n>=1): generic meet-of-two rule
        for mode in ('S^1C1C13', 'S^1C3C13', 'S^1C2C24', 'S^1C4C24'):
            for i, blk in enumerate(comps[mode]):
                if (mode, i) in confirmed: continue
                if _cond_allowed(mode, blk) and sc2_gen_confirmed(
                        blk, comps, confirmed, edges, em, vm, mode):
                    confirmed[(mode, i)] = True
                    changed = True
        # S components: relevant to confirmed 13-family + 24-family
        for i, blk in enumerate(comps['S']):
            if ('S', i) in confirmed: continue
            if _cond_allowed('S', blk) and s_comp_confirmed(blk, comps, confirmed, edges, em, vm):
                confirmed[('S', i)] = True
                changed = True
        # H components: full cond1 — a cut of the component (not cutting the diagram) with a piece whose inflows ∨ to H
        for i, blk in enumerate(comps['H']):
            if ('H', i) in confirmed: continue
            if h_comp_confirmed(blk, comps, confirmed, edges, em, vm,
                                ext_attach, ext_mode, verts):
                confirmed[('H', i)] = True
                changed = True
        # C13/C24 again: cond1 with the full fixpoint state (SC/S flow + refined-collinear inflow when externals were refined away)
        for mode in ('C13', 'C24'):
            for i, blk in enumerate(comps[mode]):
                if (mode, i) in confirmed: continue
                if cond1_confirms(blk, mode, comps, confirmed, edges, em, vm,
                                  ext_attach, ext_mode):
                    confirmed[(mode, i)] = True
                    changed = True
        # SC hidden path: an SC bridges two distinct pair-mode components (p1/p3-attached C13 for SC24, p2/p4-attached C24 for SC13; also S between two distinct C13s/C24s) + one same-side refined component; one confirmed ⟹ the other too.  Regge-only.
        for sc_mode, pair_mode, ext_a, ext_b, fam_modes, fam_a, fam_b in (
                ('S^1C24', 'C13', 'p1', 'p3', ('C2C24', 'C4C24'),
                 ('C13', 'C1C13', 'C1^2C13', 'C1∞C13'),
                 ('C13', 'C3C13', 'C3^2C13', 'C3∞C13')),
                ('S^1C13', 'C24', 'p2', 'p4', ('C1C13', 'C3C13'),
                 ('C24', 'C2C24', 'C2^2C24', 'C2∞C24'),
                 ('C24', 'C4C24', 'C4^2C24', 'C4∞C24')),
                # S conducts between two distinct C13s (+ a C24), resp. two distinct C24s (+ a C13)
                ('S', 'C13', 'p1', 'p3', ('C24',),
                 ('C13', 'C1C13', 'C1^2C13', 'C1∞C13'),
                 ('C13', 'C3C13', 'C3^2C13', 'C3∞C13')),
                ('S', 'C24', 'p2', 'p4', ('C13',),
                 ('C24', 'C2C24', 'C2^2C24', 'C2∞C24'),
                 ('C24', 'C4C24', 'C4^2C24', 'C4∞C24'))):
            for i, blk in enumerate(comps[pair_mode]):
                if (pair_mode, i) in confirmed: continue
                if sc_hidden_path_confirms(blk, i, comps, confirmed, edges,
                                           em, vm, ext_attach, ext_mode,
                                           sc_mode, pair_mode, ext_a, ext_b,
                                           fam_modes, fam_a, fam_b):
                    confirmed[(pair_mode, i)] = True
                    changed = True
        if not changed:
            break
    for m, lst in comps.items():
        for i in range(len(lst)):
            if (m, i) not in confirmed:
                return False
    return True

def ind_connected(S, edges):  # connected within S (induced subgraph)
    if len(S) <= 1: return True
    adj = {v: set() for v in S}
    for (a, b) in edges:
        if a in S and b in S:
            adj[a].add(b); adj[b].add(a)
    seen = {next(iter(S))}; stack = [next(iter(S))]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in seen: seen.add(u); stack.append(u)
    return seen == set(S)

# Refined (single-external) cut candidates: connected subsets containing root, inside S_base, avoiding forbidden vertices; empty allowed.
def refined_opts(edges, root, S_base, forbid):
    allowed = {v for v in S_base if v not in forbid}
    out = [frozenset()]
    if root not in allowed:
        return out
    rest = list(allowed - {root})
    for r in range(len(rest) + 1):
        for sub in combinations(rest, r):
            S = frozenset({root} | set(sub))
            if ind_connected(S, edges): out.append(S)
    return out


# Cut representations are NOT unique: any SC region can be represented with an SC vertex v ∈ ALL nonempty cuts (lifting v into every cut gives the same em/scaling).
# Hence refinement combinations WITHOUT such a vertex are replaceable and pruned — the region they generate is also generated by a lifted combination.

def has_sc_vertex(cut_sets):
    # ∃ a vertex in ALL nonempty cuts (the SC vertex); necessary for any combination with a nonempty refinement
    nonempty = [s for s in cut_sets.values() if s]
    if not nonempty:
        return False
    inter = nonempty[0]
    for s in nonempty[1:]:
        inter = inter & s
    return bool(inter)

# Enumerate regions under the cut rules + IR compatibility; from 3-loop level, refined single-external cuts are allowed (C1C13/C3C13 inside C13, C2C24/C4C24 inside C24).
# M: (p_i+p_j)² ~ λ^M (default 1).
def fri_regions(edges, verts, ext_attach, ext_mode, M=1, verbose=False):
    return [(cut13, cut24, vm, em) for (cut13, cut24, cut1, cut3, cut2, cut4, vm, em)
            in fri_regions_full(edges, verts, ext_attach, ext_mode, M=M)]


# Enumerate all regions of the Regge 2->2 kinematics given by ext_mode; each region is (cut13, cut24, cut1, cut3, cut2, cut4, vm, em).
#
# Dispatcher between the two enumerators, selected by the external momenta:
#   * all lightlike (k1: p_i² = 0 for all i) -> fri_regions_onshell.  No single softest mode (possibly_softest returns []), so the refinement depth is decided by the LOOP COUNT: first-power cuts (C1C13 etc.) from L >= 3, second-power (C1²C13 etc., the S²C branches) from L >= 5.
#   * otherwise (k2-k5: some p_i² ~ λ^{m_i}) -> fri_regions_offshell.  Refinement depth from possibly_softest(m_i), NO L gate: k2/k3 depth (0,1) at any loop order, k4 depth (1,2), k5 depth (0,0).
#
# M: (p_i+p_j)² ~ λ^M (default 1); kept for backward compatibility, currently unused (the per-external thresholds are kinematics-dependent and handled inside each branch).
def fri_regions_full(edges, verts, ext_attach, ext_mode, M=1, verbose=False):
    ms = {n: ext_m(ext_mode.get(n)) for n in ext_attach}
    if all(m is None for m in ms.values()):
        return fri_regions_onshell(edges, verts, ext_attach, ext_mode,
                                   verbose=verbose)
    # off-shell: possibly_softest depth limits + nested refine towers; ext_m returns 𝒱=m+1 (C13->1), possibly_softest needs m (C13->0) — convert back
    ms_m = {n: (None if v is None else v - 1)
            for n, v in ms.items()}
    return fri_regions_offshell(edges, verts, ext_attach, ext_mode, ms_m,
                                verbose=verbose)

# k1 enumerator: all external momenta lightlike (p_i² = 0).  Each region is (cut13, cut24, cut1, cut3, cut2, cut4, vm, em) — the C13/C24 cuts plus the refined single-external cuts C1C13/C3C13/C2C24/C4C24.
#
# All-lightlike kinematics have NO single softest mode (possibly_softest returns []), so the refinement depth is decided by the LOOP COUNT:
#   L < 3:         main cuts only;
#   3 <= L < 5:    first-power refined cuts, 13-side and 24-side EXCLUSIVE (no S²C regions);
#   L >= 5:        both sides may coexist, and the second-power S²C branches run with the cut coincidences C1²C13 := C1C13, C3²C13 := C3C13, ...
# The has_sc_vertex pruning applies only here — for finite m_i it over-prunes, so the off-shell enumerator does not use it.
def fri_regions_onshell(edges, verts, ext_attach, ext_mode, verbose=False):
    v1, v3 = ext_attach['p1'], ext_attach['p3']
    v2, v4 = ext_attach['p2'], ext_attach['p4']
    forbid13 = {v2, v4}; forbid24 = {v1, v3}
    inner = set(verts) - set(ext_attach.values())
    L = len(edges) - len(verts) + 1
    use_refined = L >= 3
    # C_ij cut candidates (lightlike kinematics): connected sets containing BOTH va and vb; empty set = no cut.
    def opts(va, vb, forbid):
        base = ({va, vb} | inner) - forbid
        out = [frozenset()]
        rest = list(base - {va, vb})
        for r in range(len(rest) + 1):
            for sub in combinations(rest, r):
                S = frozenset({va, vb} | set(sub))
                if ind_connected(S, edges):
                    out.append(S)
        return out

    O13 = opts(v1, v3, forbid13)
    O24 = opts(v2, v4, forbid24)
    regs = []
    seen = set()
    for cut13 in O13:
        for cut24 in O24:
            # H-vertex pruning: every region has a vertex in no cut; full coverage here kills all refinements too (they only ADD coverage)
            if cut13 | cut24 >= set(verts):
                continue
            # L >= 5: full two-sided refinement with S²C branches (the exclusive <5-loop case is handled at the bottom loop)
            if use_refined and L >= 5:
                r1 = refined_opts(edges, v1, cut13, {v2, v3, v4})
                r3 = refined_opts(edges, v3, cut13, {v1, v2, v4})
                r2 = refined_opts(edges, v2, cut24, {v1, v3, v4})
                r4 = refined_opts(edges, v4, cut24, {v1, v2, v3})
                inner = set(verts) - set(ext_attach.values())
                verts_set = set(verts)
                e_set = set(edges) | set((b, a) for (a, b) in edges)
                # 3-regular check: degree-3 vertices (external legs counted) tighten the Cond-1/2/3 H requirement to >=2 edge-connected H vertices
                reg3 = all(
                    sum(1 for (a, b) in edges if a == v or b == v)
                    + sum(1 for n, vv in ext_attach.items() if vv == v) == 3
                    for v in verts)
                # layered pruning: an SC/S²C vertex must lie in ALL nonempty cuts; I0 = main-cut intersection (empty cut = full set)
                I0 = (cut13 if cut13 else verts_set) & (cut24 if cut24 else verts_set)
                # pure main-cut combos (all refinements empty) always run — the source of the non-SC regions
                reg = _build_region(edges, verts, ext_attach, ext_mode,
                                    cut13, cut24, frozenset(), frozenset(),
                                    frozenset(), frozenset())
                if reg is not None:
                    key = (tuple(reg[1]), tuple(sorted(reg[0].items())))
                    if key not in seen:
                        seen.add(key)
                        regs.append((set(cut13), set(cut24),
                                     frozenset(), frozenset(),
                                     frozenset(), frozenset(),
                                     reg[0], reg[1]))
                if not I0:
                    # no possible SC/S²C vertex: prune the whole refinement subtree (k1 only — finite m_i need no common SC vertex)
                    continue
                for cut1 in r1:
                    for cut3 in r3:
                        # I1: intersection after the 13-side refinements; empty ⟹ skip cut2/cut4 entirely
                        I1 = I0 & (cut1 if cut1 else verts_set) \
                               & (cut3 if cut3 else verts_set)
                        if not I1:
                            continue
                        for cut2 in r2:
                            for cut4 in r4:
                                # all-empty refinement = pure main cut (already run above)
                                if not (cut1 or cut3 or cut2 or cut4):
                                    continue
                                # I2 = intersection of ALL nonempty cuts (Cond-1 v); I2x4/I2x2 = minus C4C24/C2C24 (Cond-2/3 v); I2x3/I2x1 = branch-24 Cond-2/3 (minus C3C13/C1C13)
                                I2 = I1 & (cut2 if cut2 else verts_set) \
                                       & (cut4 if cut4 else verts_set)
                                I2x4 = I1 & (cut2 if cut2 else verts_set)
                                I2x2 = I1 & (cut4 if cut4 else verts_set)
                                I2x3 = I0 & (cut1 if cut1 else verts_set) \
                                           & (cut2 if cut2 else verts_set) \
                                           & (cut4 if cut4 else verts_set)
                                I2x1 = I0 & (cut3 if cut3 else verts_set) \
                                           & (cut2 if cut2 else verts_set) \
                                           & (cut4 if cut4 else verts_set)
                                if not (I2 or I2x4 or I2x2 or I2x3 or I2x1):
                                    continue
                                # H-vertex pruning after refinements
                                if cut13 | cut24 | cut1 | cut3 | cut2 | cut4 >= verts_set:
                                    continue
                                # branch 0: no second-power refinement (first-power SC regions)
                                reg = _build_region(edges, verts, ext_attach,
                                                    ext_mode, cut13, cut24, cut1, cut3,
                                                    cut2, cut4)
                                if reg is not None:
                                    key = (tuple(reg[1]),
                                           tuple(sorted(reg[0].items())))
                                    if key not in seen:
                                        seen.add(key)
                                        regs.append((set(cut13), set(cut24),
                                                     set(cut1), set(cut3),
                                                     set(cut2), set(cut4),
                                                     reg[0], reg[1]))
                                # branch 13: S²C13 target — Cond 1/2/3 (any one suffices):
                                #   Cond 1 (relevant to SC24): v ∈ all nonempty cuts (I2), u an SC24 vertex with edge u-v;
                                #   Cond 2 (relevant to C2C24): v ∈ I2x4 (all but C4C24), w ∈ C2C24\C13 adjacent (S²C13 inflow entry);
                                #   Cond 3 (relevant to C4C24): symmetric — v ∈ I2x2, w ∈ C4C24\C13.
                                # C1²C13 := C1C13, C3²C13 := C3C13; H vertices: >=2 edge-connected under 3-regularity
                                if (cut1 or cut3):
                                    u13 = [u for u in inner
                                           if u in cut2 and u in cut4 and u in cut13
                                           and u not in cut1 and u not in cut3]
                                    if u13:
                                        hv13 = verts_set - (cut13 | cut24 | cut1
                                                             | cut3 | cut2 | cut4)
                                        h_ok = (not reg3) or any(
                                            a in hv13 and b in hv13
                                            for (a, b) in edges)
                                        if h_ok:
                                            ok13 = any(
                                                (u, v) in e_set
                                                for u in u13 for v in I2)
                                            if not ok13:
                                                ok13 = bool(I2x4) and any(  # Cond 2: v in all but C4C24
                                                    (w, v) in e_set
                                                    for w in (cut2 - cut13)
                                                    for v in I2x4)
                                            if not ok13:
                                                ok13 = bool(I2x2) and any(  # Cond 3: v in all but C2C24
                                                    (w, v) in e_set
                                                    for w in (cut4 - cut13)
                                                    for v in I2x2)
                                            if ok13:
                                                reg = _build_region(
                                                    edges, verts, ext_attach,
                                                    ext_mode, cut13, cut24, cut1,
                                                    cut3, cut2, cut4, cut1, cut3)
                                                if reg is not None:
                                                    key = (tuple(reg[1]),
                                                           tuple(sorted(
                                                               reg[0].items())))
                                                    if key not in seen:
                                                        seen.add(key)
                                                        regs.append(
                                                            (set(cut13),
                                                             set(cut24),
                                                             set(cut1),
                                                             set(cut3),
                                                             set(cut2),
                                                             set(cut4),
                                                             reg[0],
                                                             reg[1]))
                                # branch 24: S²C24 target, symmetric:
                                #   Cond 1: v ∈ all nonempty cuts (I2) + SC13 vertex u adjacent;
                                #   Cond 2: v ∈ I2x3 (all but C3C13), w ∈ C1C13\C24 adjacent;
                                #   Cond 3: v ∈ I2x1 (all but C1C13), w ∈ C3C13\C24 adjacent;
                                #   C2²C24 := C2C24, C4²C24 := C4C24.
                                if (cut2 or cut4):
                                    u24 = [u for u in inner
                                           if u in cut1 and u in cut3 and u in cut24
                                           and u not in cut2 and u not in cut4]
                                    if u24:
                                        hv24 = verts_set - (cut13 | cut24 | cut1
                                                             | cut3 | cut2 | cut4)
                                        h_ok = (not reg3) or any(
                                            a in hv24 and b in hv24
                                            for (a, b) in edges)
                                        if h_ok:
                                            ok24 = any(
                                                (u, v) in e_set
                                                for u in u24 for v in I2)
                                            if not ok24:
                                                ok24 = bool(I2x3) and any(  # Cond 2: v in all but C3C13
                                                    (w, v) in e_set
                                                    for w in (cut1 - cut24)
                                                    for v in I2x3)
                                            if not ok24:
                                                ok24 = bool(I2x1) and any(  # Cond 3: v in all but C1C13
                                                    (w, v) in e_set
                                                    for w in (cut3 - cut24)
                                                    for v in I2x1)
                                            if ok24:
                                                reg = _build_region(
                                                    edges, verts, ext_attach,
                                                    ext_mode, cut13, cut24, cut1,
                                                    cut3, cut2, cut4, None,
                                                    None, cut2, cut4)
                                                if reg is not None:
                                                    key = (tuple(reg[1]),
                                                           tuple(sorted(
                                                               reg[0].items())))
                                                    if key not in seen:
                                                        seen.add(key)
                                                        regs.append(
                                                            (set(cut13),
                                                             set(cut24),
                                                             set(cut1),
                                                             set(cut3),
                                                             set(cut2),
                                                             set(cut4),
                                                             reg[0],
                                                             reg[1]))
                continue
            for refine in ((0, 1, 2) if use_refined else (0,)):
                r1 = r3 = r2 = r4 = [frozenset()]
                if refine == 1 and use_refined:
                    r1 = refined_opts(edges, v1, cut13, {v2, v3, v4})
                    r3 = refined_opts(edges, v3, cut13, {v1, v2, v4})
                elif refine == 2 and use_refined:
                    r2 = refined_opts(edges, v2, cut24, {v1, v3, v4})
                    r4 = refined_opts(edges, v4, cut24, {v1, v2, v3})
                for cut1 in r1:
                    for cut3 in r3:
                        for cut2 in r2:
                            for cut4 in r4:
                                # all-empty refinement ≡ refine=0 (pure main cut, already built there)
                                if refine != 0 and not (cut1 or cut3 or cut2 or cut4):
                                    continue
                                if refine != 0 and not has_sc_vertex(
                                        {'C13': cut13, 'C24': cut24,
                                         'C1C13': cut1, 'C3C13': cut3,
                                         'C2C24': cut2, 'C4C24': cut4}):
                                    continue
                                if cut13 | cut24 | cut1 | cut3 | cut2 | cut4 >= set(verts):
                                    continue
                                reg = _build_region(edges, verts, ext_attach,
                                                    ext_mode, cut13, cut24, cut1, cut3,
                                                    cut2, cut4)
                                if reg is not None:
                                    key = (tuple(reg[1]),
                                           tuple(sorted(reg[0].items())))
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    regs.append((set(cut13), set(cut24),
                                                 set(cut1), set(cut3),
                                                 set(cut2), set(cut4),
                                                 reg[0], reg[1]))
    return regs



# Nested refinement towers for one external: yields (S_1, ..., S_d) with S_n ⊆ S_{n-1} (S_0 = base), each level a connected subset containing root (or empty).
# Empty at any level forces all deeper levels empty.
def _refine_towers(edges, root, base, forbid, depth):
    if depth <= 0:
        yield ()
        return
    for cut1 in refined_opts(edges, root, base, forbid):
        for rest in _refine_towers(edges, root, cut1, forbid, depth - 1):
            yield (cut1,) + rest



# Non-k1 enumerator.  ms = {p1..p4: m_i} with None = ∞ (p_i² = 0), m_i the C_i^{m_i} power (C13 → 0).
# Depth limits from possibly_softest:
#   - S^M C13 + S^M C24 both: n_1..n_4 ∈ 0..M
#   - S^M C13 alone:          n_1,n_3 ∈ 0..M, n_2,n_4 ∈ 0..M−1
#   - S^M C24 alone:          n_1,n_3 ∈ 0..M−1, n_2,n_4 ∈ 0..M
# Each level is enumerated independently as connected subsets, nested C_i^{n+1} ⊆ C_i^n; depth ≤ 2 for now (meet/vee tables stop at C_i²C_ij).
# No refinement pruning (has_sc_vertex etc. are k1-only); only the kinematics-independent H-vertex (full-coverage) prune.
def fri_regions_offshell(edges, verts, ext_attach, ext_mode, ms, verbose=False):
    v1, v3 = ext_attach['p1'], ext_attach['p3']
    v2, v4 = ext_attach['p2'], ext_attach['p4']
    forbid13 = {v2, v4}
    forbid24 = {v1, v3}
    inner = set(verts) - set(ext_attach.values())
    verts_set = set(verts)
    m1, m2, m3, m4 = ms['p1'], ms['p2'], ms['p3'], ms['p4']

    ps = possibly_softest(m1, m2, m3, m4)
    if len(ps) == 2:                    # S^M C13 + S^M C24 simultaneously
        d13 = d24 = int(ps[0].split('^')[1].split('C')[0])
    else:                               # single candidate
        Mside = int(ps[0].split('^')[1].split('C')[0])
        if ps[0].endswith('C13'):
            d13, d24 = Mside, Mside - 1
        else:
            d13, d24 = Mside - 1, Mside
    if d13 > 2 or d24 > 2:
        raise NotImplementedError(
            f"fri_regions_offshell: depth {max(d13, d24)} > 2 — meet/vee "
            f"tables stop at C_i²C_ij (possibly_softest={ps}); extend the "
            f"mode tables first.")

    # C_ij cut candidates (off-shell): may be DISCONNECTED, with at most two connected components — one around va, one around vb — and every connected component of S must contain va or vb.
    # This lets Glauber/hard vertices stay OUTSIDE all cuts, so requirement 2 (nonempty off-shell subgraph) is satisfied automatically.
    def opts(va, vb, forbid):
        base = ({va, vb} | inner) - forbid
        out = [frozenset()]
        rest = list(base)
        for r in range(len(rest) + 1):
            for sub in combinations(rest, r):
                S = frozenset(sub)
                if not S:
                    continue
                es = [(a, b) for (a, b) in edges if a in S and b in S]
                comp = components(es, S)
                groups = {}
                for v in S:
                    groups.setdefault(comp[v], set()).add(v)
                if all(va in c or vb in c for c in groups.values()):
                    out.append(S)
        return out

    # non-k1: the main cuts MAY be disconnected (not a per-side requirement); every component must contain va or vb
    O13 = opts(v1, v3, forbid13)
    O24 = opts(v2, v4, forbid24)

    regs = []
    seen = set()
    for cut13 in O13:
        for cut24 in O24:
            # H-vertex prune (kinematics-independent): every region has a vertex in no cut
            if cut13 | cut24 >= verts_set:
                continue
            T1 = list(_refine_towers(edges, v1, cut13, {v2, v3, v4}, d13))
            T3 = list(_refine_towers(edges, v3, cut13, {v1, v2, v4}, d13))
            T2 = list(_refine_towers(edges, v2, cut24, {v1, v3, v4}, d24))
            T4 = list(_refine_towers(edges, v4, cut24, {v1, v2, v3}, d24))
            for t1 in T1:
                for t3 in T3:
                    for t2 in T2:
                        for t4 in T4:
                            cut1 = t1[0] if len(t1) >= 1 else frozenset()
                            cut1sq = t1[1] if len(t1) >= 2 else frozenset()
                            cut3 = t3[0] if len(t3) >= 1 else frozenset()
                            cut3sq = t3[1] if len(t3) >= 2 else frozenset()
                            cut2 = t2[0] if len(t2) >= 1 else frozenset()
                            cut2sq = t2[1] if len(t2) >= 2 else frozenset()
                            cut4 = t4[0] if len(t4) >= 1 else frozenset()
                            cut4sq = t4[1] if len(t4) >= 2 else frozenset()
                            if (cut13 | cut24 | cut1 | cut3 | cut2 | cut4
                                    | cut1sq | cut3sq | cut2sq | cut4sq) >= verts_set:
                                continue
                            reg = _build_region(edges, verts, ext_attach,
                                                ext_mode, cut13, cut24, cut1, cut3,
                                                cut2, cut4, cut1sq, cut3sq, cut2sq,
                                                cut4sq)
                            if reg is None:
                                continue
                            key = (tuple(reg[1]),
                                   tuple(sorted(reg[0].items())))
                            if key in seen:
                                continue
                            seen.add(key)
                            regs.append((set(cut13), set(cut24), set(cut1),
                                         set(cut3), set(cut2), set(cut4),
                                         reg[0], reg[1]))
    return regs


# Glauber/semihard-vertex momentum rule: at a G/sH vertex all components are small, so >=2 small line momenta + exactly one large (O(1)) one is unbalanced.
# Large = H + collinear family (incl. refined and lightlike-external modes); everything else is small.
_GV_LARGE = {'H', 'C13', 'C24', 'C1C13', 'C3C13', 'C2C24', 'C4C24',
             'C1^2C13', 'C3^2C13', 'C2^2C24', 'C4^2C24',
             'C1∞C13', 'C3∞C13', 'C2∞C24', 'C4∞C24'}


def glauber_vertex_mom_ok(inc):
    n_large = sum(1 for m in inc if m in _GV_LARGE)
    n_small = len(inc) - n_large
    return not (n_small >= 2 and n_large == 1)


# Momentum conservation at each vertex (Regge version): incident momenta (edge modes + attached external) split into two nonempty groups with equal ∨-joins.
# Adaptations: G/sH vertices and vertices touching a G/sH edge are skipped (transverse, does not mix with longitudinal); H is NOT skipped (an O(1) flow that must balance).
# Both internal and external-momentum vertices are checked: n>=2 via ∨-partition (for n=2 this reduces to equality of the two line modes).  vee() is authoritative (regge_modes lattice).
def momentum_ok_regge(edges, em, vm, ext_attach, ext_mode=None):
    verts = set(v for e in edges for v in e) | set(ext_attach.values())
    for v in verts:
        inc = []
        for i, (a, b) in enumerate(edges):
            if a == v or b == v:
                inc.append(em[i])
        ext = [ext_mode_for(ext_attach, ext_mode, n)
               for n, vv in ext_attach.items() if vv == v]
        inc = inc + ext
        if vm.get(v) in ('G', 'sH'):
            if not glauber_vertex_mom_ok(inc):  # G/sH: >=2 small lines + exactly one large (O(1)) line is forbidden — it cannot cancel
                return False
            continue
        if any(m in ('G', 'sH') for m in inc):
            continue
        n = len(inc)
        if n < 2:
            return False
        if n == 2:
            a, b = inc
            if vee([a]) == vee([b]):
                continue
            return False
        ok = False
        for r in range(1, n):
            for A in combinations(range(n), r):
                B = [i for i in range(n) if i not in A]
                if vee([inc[i] for i in A]) == vee([inc[i] for i in B]):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            return False
    return True


# Every cut vertex separating p1,p3 from p2,p4 must be of G mode.
def family_cut_vertex_ok(edges, verts, vm, ext_attach):
    p1, p2, p3, p4 = (ext_attach['p1'], ext_attach['p2'],
                       ext_attach['p3'], ext_attach['p4'])
    extvs = set(ext_attach.values())
    for v in verts:
        if v in extvs:
            continue
        if vm.get(v) == 'G':
            continue
        rem_v = [vv for vv in verts if vv != v]
        rem_e = [e for e in edges if v not in e]
        comp = components(rem_e, rem_v)
        if (comp[p1] == comp[p3] and comp[p2] == comp[p4]
                and comp[p1] != comp[p2]):
            return False
    return True


# Every connected component of the jet graph (vertices+edges) contains at least one of its external vertices
def jet_components_ok(jv, je, ext_verts):
    if not jv:
        return True
    comp = components(list(je), jv)
    groups = {}
    for v in jv:
        groups.setdefault(comp[v], set()).add(v)
    for g in groups.values():
        if not (g & ext_verts):
            return False
    return True

def _build_region(edges, verts, ext_attach, ext_mode, cut13, cut24, cut1, cut3,
                  cut2, cut4, cut1sq=None, cut3sq=None, cut2sq=None, cut4sq=None):
    cuts = []
    if cut13: cuts.append(('C13', cut13))
    if cut24: cuts.append(('C24', cut24))
    if cut1: cuts.append(('C1C13', cut1))
    if cut3: cuts.append(('C3C13', cut3))
    if cut2: cuts.append(('C2C24', cut2))
    if cut4: cuts.append(('C4C24', cut4))
    # second-power refinements: C1²C13 inside C1C13, etc.
    if cut1sq: cuts.append(('C1^2C13', cut1sq))
    if cut3sq: cuts.append(('C3^2C13', cut3sq))
    if cut2sq: cuts.append(('C2^2C24', cut2sq))
    if cut4sq: cuts.append(('C4^2C24', cut4sq))

    # subgraph not in any cut: nonempty and connected (hoisted 2026-09-15 — depends only on
    # the cuts; was checked after glauber/momentum/jets: pure reorder, survivors unchanged)
    covered_v = set().union(*[S for (m, S) in cuts]) if cuts else set()
    hv = set(verts) - covered_v
    if not hv: return None
    he = {e for e in edges if e[0] not in covered_v and e[1] not in covered_v}
    if not connected(hv, he): return None
    # VM-FIRST EXPERIMENT (fri23-style): vm = ∧ of cuts containing v; then em = vm_u ∧ vm_v
    vm = {}
    for v in verts:
        acc = None
        for (m, S) in cuts:
            if v in S:
                acc = m if acc is None else meet(acc, m)
        vm[v] = acc if acc is not None else 'H'
    em = [meet(vm[a], vm[b]) for (a, b) in edges]
    # edge-mode self-consistency: em[e] must equal vm[u] ∧ vm[v] (checked BEFORE glauber_adjust — G is an adjustment product)
    for (a, b), m in zip(edges, em):
        if meet(vm[a], vm[b]) != m:
            return None
    # large-momentum-flow adjustment
    adj = glauber_adjust(edges, em, vm, ext_attach, ext_mode)
    if adj is None:
        return None
    em, vm = adj
    # Glauber-necklace detection: H self-energy loops between Glauber edges become sH (runs BEFORE requirements)
    em2 = necklace_detect(edges, em, vm, ext_attach)
    if em2 != em:
        em = em2
        vm = _recompute_vm(edges, em, ext_attach, ext_mode)
    # momentum conservation at every vertex; H absorbs (skip), G/sH dropped, externals checked via ∨-partition
    if not momentum_ok_regge(edges, em, vm, ext_attach, ext_mode):
        return None
    # (0) family-separating cut vertices must be G (else hard momentum would be forced through a jet line)
    if not family_cut_vertex_ok(edges, verts, vm, ext_attach):
        return None
    # (1) jets: every connected component contains its external vertices (legs may be bridged by soft·collinear lines)
    je13 = {e for e, m in zip(edges, em) if m in J13_FAM}
    jv13 = ({v for v in verts if vm[v] in J13_FAM}
            | {v for e in je13 for v in e})
    je24 = {e for e, m in zip(edges, em) if m in J24_FAM}
    jv24 = ({v for v in verts if vm[v] in J24_FAM}
            | {v for e in je24 for v in e})
    if not jet_components_ok(jv13, je13, {ext_attach['p1'], ext_attach['p3']}):
        return None
    if not jet_components_ok(jv24, je24, {ext_attach['p2'], ext_attach['p4']}):
        return None
    # (3) mojetic (REGGE_NO_MOJETIC=1 disables, for tests)
    if os.environ.get('REGGE_NO_MOJETIC') != '1':
        # k0 (all externals finite C13/C24): each contracted jet 1VI, each H component needs jet edges of BOTH families adjacent
        k0 = all(ext_mode_for(ext_attach, ext_mode, n) in ('C13', 'C24')
                 for n in ext_attach)
        if k0:
            if not mojetic_k0_ok(edges, em, vm, J13_FAM, J24_FAM,
                                 ext_attach, ext_mode, verts):
                return None
        else:
            gv = {v for v in verts if vm[v] == 'G'}
            hv_m = {v for v in verts if vm[v] == 'H'}
            he_m = {e for e, m in zip(edges, em) if m == 'H'}
            for jv, je, ext_names in ((jv13, je13, ('p1', 'p3')),
                                      (jv24, je24, ('p2', 'p4'))):
                if not mojetic_ok(hv_m, he_m, jv, je, gv, edges, ext_attach,
                                  ext_names):
                    return None
    # no C13/C24 scaleless-island check (see c13_c24_island_ok).
    if not c13_c24_island_ok(edges, em, vm, verts):
        return None
    # IR compatibility
    if not ir_ok_region(edges, em, vm, ext_attach, ext_mode):
        return None
    return (vm, em)

# no C13/C24 scaleless island: a C13/C24 component whose ADJACENT modes are all softer (𝒱 > 𝒱(C13) = 1) is rejected — a 𝒱=1 island separated from the hard core by 𝒱>=2 edges carries no scale (First Connectivity Theorem).
# H/G/sH/C13/C24 and own-mode edges are never softer, so any block carrying its own C13/C24 line passes.
def c13_c24_island_ok(edges, em, vm, verts):
    for mode in ('C13', 'C24'):
        for (bv, be, *rest) in mode_components(mode, vm, em, edges, verts):
            real = {v for v in bv if v != 'aux'}
            if not real:
                continue
            incident = [em[i] for i, (a, b) in enumerate(edges)
                        if a in real or b in real]
            if incident and all(V[m] > V[mode] for m in incident):
                return False
    return True


def to_scaling(em):
    return tuple(-V[m] for m in em) + (1,)
