#!/usr/bin/env python3
"""usable_modes.py — derive the usable internal modes from external modes
+ IR compatibility, as an iterative closure (fixpoint).  2026-08-12.

This step runs BEFORE any region analysis: given only the external momentum
modes and the IR-compatibility conditions (§5.2 of the paper), compute the
set of internal modes that can possibly appear in a valid region of ANY
graph with this kinematics.  It is a monotone fixpoint; the result is closed
under the iteration.

Mode-level IR-compatibility rules (each justified by the region-level
conditions):
  R1 seeds: every external momentum confirms its own mode.
  R2 cond 1 (partial sum): C_i^N + S^m flow -> C_i^m  (m < N strictly;
     the lattice join join(C_i^N, S^m) = C_i^m, m < N).
  R3 cond 3 (meet): the lattice meet of any two confirmed modes is confirmed
     (covers SC_i^n = meet(C_j^a, C_i^b) and S^m = meet(C_i^m, C_j^m)).
  R4 cond 2 (messenger): S^m is confirmable iff
       (i)  m >= n0, where n0 = min level over the external modes
            (level(C_i^n)=n, level(C_i^inf)=kappa, level(S^m C_i^n)=m) —
            the kernel's level-m collinear targets C_i^m exist as
            components only if a scale source at level m exists, and the
            seeds are the only bootstrap; and
       (ii) >= 3 distinct directions host collinear structures of depth
            >= m (the messenger needs pairwise-distinct target directions).

The dead-layer law is a THEOREM of these rules, not an input:
levels below n0 are never generated (S^m with m < n0 fails R4, so R2 can
never produce C_i^m with m < n0, so no level-<n0 mode is ever confirmed).

The result is the CONFIRMABLE set: a superset of what any specific graph
realizes (realization is graph-dependent), and a subset of all algebraically
possible modes.  It is exactly what layer compression should use.
"""
from primitives import norm, eq, harder_or_eq, join, meet
from read_graph import INF  # pipeline INF = 100 (C^inf/SC^inf marker)


# 'Level' of an external mode: the minimal softness index it anchors (level(C_i^n)=n, C_i^inf=kappa, S^m C_i^n=m, S^m=m; H -> None).
def external_level(md, kappa):
    m, n, i = md
    if m == 0 and n == 0:
        return None
    if n == 0:
        return m                       # pure soft S^m
    if m == 0:
        return kappa if n >= INF else n
    return m                           # SC: the S-part is the soft anchor


# Deepest collinear level hosted by direction i (kappa for C^inf).
def depth_of(external_modes, kappa, i):
    d = 0
    for md in external_modes.values():
        m, n, ii = md
        if ii == i and n != 0:
            d = max(d, kappa if n >= INF else n)
    return d


# Iterative closure: external modes + IR compatibility -> usable modes.
# ext_mode: {name: (m, n, i)}; kappa: max softness power; returns a set (finite n only, H in).
def derive_usable_modes(ext_mode, kappa):
    H = (0, 0, 0)
    C = set()
    # R1: seeds
    for md in ext_mode.values():
        if md != H:
            C.add(norm(md))
    # n0 = min external level  (dead-layer law input to R4)
    levels = [external_level(md, kappa) for md in ext_mode.values()]
    n0 = min(l for l in levels if l is not None)
    dirs = {md[2] for md in ext_mode.values() if md[2] != 0}

    changed = True
    it = 0
    while changed:
        changed = False
        it += 1
        L = list(C)
        # R2: collinear levels from soft flows  (join(C_i^N, S^m) = C_i^m, m<N)
        for md in L:
            m, n, i = md
            if n != 0 or m == 0:
                continue               # S^m only
            for other in L:
                o_m, o_n, o_i = other
                if o_n == 0:
                    continue           # need a collinear C_i^N
                if m < o_n:            # strict: m < N (lattice handles dir)
                    jn = join(other, md)
                    if jn not in C:
                        C.add(jn); changed = True
        # R3: meets of confirmed pairs (cond 3)
        for a in range(len(L)):
            for b in range(a + 1, len(L)):
                mt = meet(L[a], L[b])
                if mt not in C:
                    C.add(mt); changed = True
        # R4: S^m messengers (cond 2)
        for m in range(1, kappa + 1):
            if (m, 0, 0) in C:
                continue
            # Anchor: an external mode whose own level is EXACTLY m — a level-m collinear target must have an independent source (an external C_i^m, or an S^m/S^mC_i^n feeding the kernel).
            # Without it, the messenger's targets would all depend on S^m itself: pure circularity, no scale source.
            # (2026-08-12: case3 S^3 has no level-3 external -> dead.)
            anchored = any(external_level(md, kappa) == m for md in ext_mode.values())
            if not anchored:
                continue
            # Target directions, either:
            #   (a) collinear depth >= m  (kernel/member targets C_i^{n'}), or
            #   (b) an SC/S external S^{m'}C_i^{n'} with m' < m <= m' + n' — its own mode is a messenger target (m_i = m - m', n_i = n' - m_i >= 0).
            #       Example: CheesePizza k1 S^2 uses l1 = S^1C_4^1 as its 3rd direction (2026-08-12).
            nd = 0
            for i in dirs:
                if depth_of(ext_mode, kappa, i) >= m:
                    nd += 1
                    continue
                for md in ext_mode.values():
                    m2, n2, i2 = md
                    if i2 == i and 1 <= m2 < m and (n2 >= INF or m <= m2 + n2):
                        nd += 1
                        break
            if nd >= 3:
                C.add((m, 0, 0)); changed = True
    # filter: no infinite parts, nothing beyond the kinematics' softness cap
    out = {m for m in C if m[0] <= kappa and m[1] <= kappa}
    out.add(H)
    return out


# Per external: the cut-tower layers to keep (compression input).
# Pure-C: {n : C_i^n usable} — dead levels below n0 are excluded by construction.
# SC external (m>=1): its own tower S^mC_i^k, k=1..n_max, is always kept (containers hosting soft-attachment vertices, e.g. CrownASE l1=SC5^inf).
def usable_layers(ext_mode, kappa, usable):
    layers = {}
    for name, md in ext_mode.items():
        if md == (0, 0, 0):
            continue
        m, n, i = md
        n_max = kappa if n >= INF else (n if n >= 1 else 1)
        if m >= 1:
            kept = list(range(1, n_max + 1))
        else:
            kept = [k for k in range(1, n_max + 1) if (0, k, i) in usable]
        layers[name] = kept
    return layers
