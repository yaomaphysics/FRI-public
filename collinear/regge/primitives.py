#!/usr/bin/env python3
"""primitives.py — zero-judgment base layer for the Regge 2->2 framework.

Mode algebra (the lattice of momentum modes S^m C_i^n C_ij, the G/sH overlay
products, relations, virtuality, mode_meet/mode_join/vee) + the mode-string layer
(meet/join/vee/marginally_softer on plain strings) + mode blocks and scaling +
external-momentum helpers + graph tools.  The region checks (judgment) live in
region_checker.py; the enumerators live in skeleton.py.
"""

from __future__ import annotations

import itertools
import time
from functools import lru_cache
from itertools import combinations
from typing import Optional

# ============================ mode algebra ============================
#
# Regge-limit momentum-mode algebra.
# 
# 2026-08-29.  Every momentum mode is one of three types:
# 
#     S^m C_i^n C_ij  |  G (Glauber)  |  sH (semihard)
# 
# The parametrised family S^m C_i^n C_ij:
# 
#     m >= 0 integer
#     n in {-1, 0, 1, 2, ...} U {INF}
#     family (pair) ij in {13, 24};  leg i in {1,3} (fam 13) or {2,4} (fam 24)
# 
#     n = -1 : the trailing C_ij is cancelled by C_i^{-1}  ->  pure S^m
#              (m = 0 as well  ->  the hard mode H).  No leg, no family.
#     n =  0 : S^m C_ij  (m = 0 -> the plain C13 / C24 cut).  Leg label absent.
#     n = INF: the four lightlike external momenta C_i^INF C_ij (m = 0 forced).
# 
# Scaling, in the light-cone basis of leg i (beta_i lightlike, see the draft):
# 
#     k^mu = (k.betabar_i, k.beta_i, k.beta_iperp) ~ lambda^m (1, l^{n+1}, l^{(n+1)/2})
# 
# Virtuality (derived, not postulated: k^2 = 2(k.bb)(k.b) - k_perp^2 and both
# terms are the same order, so there is no cancellation):
# 
#     V(S^m C_i^n C_ij) = 2m + n + 1          (n = -1 degenerates to 2m)
# 
# Order (2026-08-29): drop the trailing C_ij and compare the S^m C_i^n parts
# by the *wide-angle* rules.  With sigma = m + n:
# 
#     same leg       : X1 softer than X2  <=>  m1 >= m2  and  sigma1 >= sigma2
#     different leg  : X1 softer than X2  <=>  m1 >= sigma2
#     different fam  : X1 softer than X2  <=>  m1 >= sigma2 + 1
# 
# Modes with n <= 0 carry no leg label, so inside their family they compare by
# the same-leg rule (they are isotropic in the leg degree of freedom).  The +1
# in the last row is the back-to-back projection: across families the minus
# component of X2 is what shows up as the plus component in X1's frame.
# 
# Wedge / vee, same family (2026-08-29 19:16): strip the trailing C_ij, do
# the wide-angle operation on S^m C_i^n (three cases), put the
# C_ij back.  Verified equal to the glb/lub of the order above on 1872/1872
# same-family pairs (m<=3, n<=3) -- EXCEPT when n = -1 is involved, where all
# 112 disagreements have the strip-and-wide-angle answer failing to be a bound
# at all.  Reason: C_i^{-1} and C_ij cancel *as a pair* (so C_ij cannot be
# stripped alone: Regge's S^m has sigma = m-1 while wide-angle's pure soft
# S^m = S^m C^0 has sigma = m), and more fundamentally wide-angle compares a
# direction-free S^m to C_i^n by the DIFFERENT-direction criterion (that is
# where  S wedge C_i^2 = SC_i  comes from) whereas the Regge S^m ~ l^m (1,1,1) is
# genuinely isotropic -- nothing to project, so the comparison is componentwise.
# 
#     PATCH: pure S^m (n = -1, including H) is isotropic and
#     combines with any same-family mode by the wide-angle SAME-direction rule,
#     i.e. the product order on (m, sigma).  n >= 0 uses strip-and-wide-angle.
# 
# With that clause: 1984/1984 agreement on same-family pairs.  Counterexample it
# fixes: C1C13 /\ S -> S (unpatched) is not a lower bound, since S ~ l(1,1,1) is
# LARGER than C1C13 ~ (1,l^2,l) in the minus component; glb = S^1C13 (V=3),
# which is also what the old table has.
# 
# Cross-family wedge/vee: still to come.  The glb/lub of the order
# above predicts  mode_meet = (min sigma + 1, max sigma) with the label of the larger
# sigma,  mode_join = (min m, max m - 1) with the label of the smaller m.
# 
# mode_meet / mode_join are computed as genuine glb / lub of this order -- no tables.
# Consequences verified by self_test():
#   * the order is a partial order (transitive, antisymmetric)
#   * glb / lub exist and are unique  =>  it is a lattice
#   * idempotence, absorption, associativity, and the virtuality identity
#     V(X1) + V(X2) = V(X1 mode_meet X2) + V(X1 mode_join X2)  all hold identically
#   * the 2026-08-18 hand rule "mode_meet keeps the deeper leg" is now a theorem
# 
# G and sH (2026-08-29)
# --------------------------
# Order:  H  >-  G  >-  sH  >-  every S^m C_i^n C_ij mode
#         (">-" = harder than).  G is the necklace string, sH the pearl between
#         two G's, so an sH vertex only ever sees sH and G neighbours.
# 
# They are NOT members of the lattice L = {S^m C_i^n C_ij}; they are overlay
# products, and the separation is enforced by *ordering in time*, not by fiat:
# 
#   region_checker._build_region does
#      (1) overlay construction  vm from the cuts, em[e] = mode_meet(vm[u], vm[v])  <- L only
#      (2) glauber_adjust  assigns G and sH, then recomputes vm with vee   <- sees G/sH
# 
#   => mode_meet NEVER sees G or sH          (dead table entries; the correction
#                                        MEET(G,C13)=C13 is pure cleanup)
#   => mode_join sees G/sH only inside step (2)'s vm recomputation, but by then an sH vertex
#      has only sH/G neighbours, so mode_join(sH, family mode) is vacuous too.
# 
# This is why mode_join(C13, C24) = H survives: that mode_join happens in step (1), before
# sH exists.  As a *poset* L u {G, sH} is fine, but L is not a join-sublattice
# of it (the lub of C13 and C24 in the bigger poset is sH), so the two must never
# be mixed in one call.  assert_family() below guards the overlay-construction code paths.

INF = float('inf')
FAM_LEGS = {13: (1, 3), 24: (2, 4)}
LEG_FAM = {1: 13, 3: 13, 2: 24, 4: 24}


# Canonical mode. Stored as (m, n, leg, fam); leg/fam are None when the corresponding label is meaningless (n <= 0
# kills leg, n = -1 kills fam).
class Mode(tuple):
    __slots__ = ()

    def __new__(cls, m: int, n, leg: Optional[int] = None, fam: Optional[int] = None):
        if n != INF:
            n = int(n)
            if n < -1:
                raise ValueError(f'n must be >= -1 or INF, got {n}')
        if m < 0:
            raise ValueError(f'm must be >= 0, got {m}')
        if n == INF and m != 0:
            raise ValueError('n = INF requires m = 0 (external momentum)')
        if n == -1:
            leg = fam = None                      # isotropic: no direction
        elif n == 0:
            if leg is not None and fam is None:
                fam = LEG_FAM[leg]
            leg = None                            # collinear to the pair only
        else:
            if leg is None:
                raise ValueError('n >= 1 requires a leg index')
            if fam is None:
                fam = LEG_FAM[leg]
            if leg not in FAM_LEGS[fam]:
                raise ValueError(f'leg {leg} not in family {fam}')
        if n >= 0 and fam is None:
            raise ValueError('n >= 0 requires a family')
        return super().__new__(cls, (m, n, leg, fam))

    m = property(lambda self: self[0])
    n = property(lambda self: self[1])
    leg = property(lambda self: self[2])
    fam = property(lambda self: self[3])
    sigma = property(lambda self: self[0] + self[1])

    # Virtuality 2m + n + 1 = m + sigma + 1.
    @property
    def V(self):
        return INF if self.n == INF else 2 * self.m + self.n + 1

    # Exponents of (k.betabar_i, k.beta_i, k.beta_iperp) in units of lambda.
    def scaling(self):
        if self.n == INF:
            return (self.m, INF, INF)
        return (self.m, self.m + self.n + 1, self.m + (self.n + 1) / 2)

    def __str__(self):
        m, n, leg, fam = self
        if n == -1:
            # 'S' (not 'S^1') to stay byte-compatible with the old table
            return 'H' if m == 0 else ('S' if m == 1 else f'S^{m}')
        pre = '' if m == 0 else f'S^{m}'
        if n == 0:
            return f'{pre}C{fam}'
        exp = '' if n == 1 else ('^INF' if n == INF else f'^{n}')
        return f'{pre}C{leg}{exp}C{fam}'

    __repr__ = __str__


# ----------------------------------------------------------------- constructors
H = Mode(0, -1)
S = lambda m=1: Mode(m, -1)                       # noqa: E731
C = lambda fam, m=0: Mode(m, 0, None, fam)        # noqa: E731
CC = lambda leg, n=1, m=0: Mode(m, n, leg)        # noqa: E731
EXT = lambda leg: Mode(0, INF, leg)               # noqa: E731

# ---------------------------------------------------------------- old-name map
_OLD = {
    'H': H, 'S': S(1), 'S^2': S(2), 'S^3': S(3), 'S^4': S(4),
    'C13': C(13), 'C24': C(24),
    'S^1C13': C(13, 1), 'S^1C24': C(24, 1),
    'S^2C13': C(13, 2), 'S^2C24': C(24, 2),
    'C1C13': CC(1), 'C3C13': CC(3), 'C2C24': CC(2), 'C4C24': CC(4),
    'C1^2C13': CC(1, 2), 'C3^2C13': CC(3, 2),
    'C2^2C24': CC(2, 2), 'C4^2C24': CC(4, 2),
    'C1∞C13': EXT(1), 'C3∞C13': EXT(3), 'C2∞C24': EXT(2), 'C4∞C24': EXT(4),
}
# reverse map for O(1) name lookup (to_old hot path)
_OLD_REV = {v: k for k, v in _OLD.items()}
# The old leg-blind V=4 symbol decodes to (m,n) = (1,1) with an UNKNOWN leg.
LEG_BLIND = {'S^1C13^2': 13, 'S^1C24^2': 24}


# Old region_checker string -> Mode. Leg-blind symbols return a tuple of the two possible readings (the information was
# not in the old symbol).
def from_old(name: str):
    if name in _OLD:
        return _OLD[name]
    if name in LEG_BLIND:
        fam = LEG_BLIND[name]
        return tuple(Mode(1, 1, leg, fam) for leg in FAM_LEGS[fam])
    p = _parse(name)
    if p is not None:
        return p
    raise KeyError(name)


# Parse a leg-explicit mode string like 'S^1C4C24', 'C1^2C13', 'C3∞C13', 'S^2C24', 'C24', 'S^3', 'H' into a Mode, or
# None if it does not parse (e.g. the old leg-blind 'S^1C13^2').
def _parse(name: str):
    import re
    if name == 'H':
        return Mode(0, -1)
    m = 0
    rest = name
    sm = re.match(r'S\^(\d+)', rest)
    if sm:
        m = int(sm.group(1))
        rest = rest[sm.end():]
    elif rest.startswith('S'):
        return None
    if not rest:
        return Mode(m, -1)
    cm = re.match(r'C(\d+)(?:\^(\d+)|(∞|inf))?', rest)
    if not cm:
        return None
    leg = int(cm.group(1))
    n = cm.group(2)
    inf = cm.group(3)
    rest2 = rest[cm.end():]
    if not rest2.startswith('C'):
        return None
    fam = int(rest2[1:])
    if fam not in (13, 24):
        return None
    if inf:
        if m != 0:
            return None
        return Mode(0, INF, leg, fam)
    nv = 1 if n is None else int(n)
    return Mode(m, nv, leg, fam)


def to_old(x: Mode) -> str:
    if x == G:
        return 'G'
    if x == SH:
        return 'sH'
    k = _OLD_REV.get(x)
    if k is not None:
        return k
    if (x.m, x.n) == (1, 1):
        # 2026-08-29: keep the leg -- the leg-blind 'S^1C_ij^2' loses
        # information on the string round-trip (loop4b k4 A: the (4,8) edge
        # must stay S^1C4C24, not fall back to the leg-2 reading).
        return f'S^1C{x.leg}C{x.fam}'
    return str(x)


# Old region_checker mode string -> Mode, including the G / sH sentinels.
def to_mode(name: str) -> Mode:
    if name in ('G', 'sH'):
        return G if name == 'G' else SH
    x = from_old(name)
    if isinstance(x, Mode):
        return x
    return x[0]                                    # leg-blind: first reading


# region_checker-facing mode_meet (string in, string out). mode_meet must NEVER see G/sH in the region_checker pipeline (they are
# assigned only afterwards, by glauber_adjust), so this is the enforcement point.
def old_meet(a: str, b: str) -> str:
    if a in SPECIAL or b in SPECIAL:
        raise AssertionError(
            f'mode_meet called with G/sH: ({a}, {b}). G/sH are overlay products '
            'and must not reach the lattice-only code paths '
            '(the overlay construction runs before glauber_adjust).')
    return to_old(mode_meet(to_mode(a), to_mode(b)))


# region_checker-facing mode_join (string in, string out). G/sH are legal here: glauber_adjust recomputes
# vertex modes with vee.
def old_join(a: str, b: str) -> str:
    return to_old(mode_join(to_mode(a), to_mode(b)))


# ----------------------------------------------------------------------- order
# 'same' (same leg / label-free), 'xleg', or 'xfam'.
def _relation(a: Mode, b: Mode) -> str:
    if a.n == -1 or b.n == -1:
        return 'same'                             # isotropic
    if a.fam != b.fam:
        return 'xfam'
    if a.leg is None or b.leg is None:
        return 'same'                             # n <= 0 has no leg
    return 'same' if a.leg == b.leg else 'xleg'


# a softer-or-equal than b (2026-08-29 wide-angle-inherited rules).
def softer(a: Mode, b: Mode) -> bool:
    if a == b:
        return True
    r = _relation(a, b)
    if r == 'same':
        return a.m >= b.m and a.sigma >= b.sigma
    if r == 'xleg':
        return a.m >= b.sigma
    return a.m >= b.sigma + 1


def harder(a: Mode, b: Mode) -> bool:
    return softer(b, a)


# Wide-angle reduction form: x strictly softer than y and x.m <= y.sigma.
def _wa_reduction(x: Mode, y: Mode) -> bool:
    return softer(x, y) and not softer(y, x) and x.m <= y.sigma


# x marginally softer than y: the sigma/V "strictly between" rule (matches the wide-angle and 2to3 forms).
@lru_cache(maxsize=None)
def mode_marginally_softer(x: Mode, y: Mode) -> bool:
    if x == y:
        return False
    if y.n == -1 and y.m == 0:                    # y = H: only bare collinear sources
        return x.m == 0 and x.n >= 0
    if x.n == -1 and x.m == 0:                    # x = H
        return False
    if x.n == -1:                                 # x = S^m: sigma equality
        return x.sigma == y.sigma
    if x.m == 0:                                  # bare collinear source: C_fam / C_i^n C_ij / C_i∞ C_ij
        if y.fam != x.fam:
            return False
        if y.n == 0:                              # -> C_fam: only the bare carrier, strictly harder
            return y.m == 0 and x.V > y.V
        if y.n != INF and y.n >= 1 and y.m == 0:  # -> C_i^n C_ij: wide-angle reduction (∞ carrier keeps the old form)
            if x.n == INF:
                if x.leg == y.leg:
                    return x.V > y.V
                return x.V > y.V and y.n <= 1
            return _wa_reduction(x, y)
        return False
    # soft-carrier source: S^m C_fam (m >= 1) or S^m C_i C_ij
    if y.n == -1:                                 # -> S^m': soft powers equal
        return y.m == x.m
    if y.n == 0:                                  # -> C_fam': same family keeps m, cross family lowers by one
        return y.m == x.m if y.fam == x.fam else y.m == x.m - 1
    if y.n != INF and y.n >= 1:
        if y.m == 0:                              # -> C_i^n C_ij: n' = sigma (same leg) or n' = m (cross leg)
            if y.fam == x.fam:
                if x.n == 0:
                    return y.n == x.m
                if x.leg == y.leg:
                    return y.n == x.sigma
                return y.n == x.m
            return x.n == 0 and y.n + 1 == x.m
        # -> S^m' C_i C_ij: same leg keeps m or sigma; cross leg aligns source m with target sigma
        if x.n == 0:
            return y.fam == x.fam and y.sigma == x.m
        if y.fam == x.fam:
            if x.leg == y.leg:
                return (x.m == y.m and x.sigma > y.sigma) or (x.sigma == y.sigma and x.m > y.m)
            return y.sigma == x.m and x.sigma > y.m
        return False
    return False


# ------------------------------------------------------- G / sH (overlay chain)
G = 'G'                     # Glauber: the necklace string
SH = 'sH'                   # semihard: the pearl between two G's
SPECIAL = (G, SH)
# hardness chain, hardest first:  H  >-  G  >-  sH  >-  everything in L
_CHAIN = {G: 1, SH: 2}      # rank; H = 0, any family mode = 3


def _rank(x):
    if x in SPECIAL:
        return _CHAIN[x]
    return 0 if x == H else 3


# Softer of the two under the chain H >- G >- sH >- L. NB: never invoked by region_checker -- mode_meet runs before G/sH are
# assigned (see module docstring).
def _chain_meet(a, b):
    if a == b:
        return a
    ra, rb = _rank(a), _rank(b)
    if ra != rb:
        return a if ra > rb else b            # larger rank = softer
    return a if a in SPECIAL else b


# Harder of the two under the chain (reachable: the vm recomputation in glauber_adjust).
def _chain_join(a, b):
    if a == b:
        return a
    ra, rb = _rank(a), _rank(b)
    if ra != rb:
        return a if ra < rb else b            # smaller rank = harder
    return a if a in SPECIAL else b


# Guard for the overlay-construction code paths (vertex-mode joins and edge-mode meets): G/sH must not appear
# there, otherwise the lub of C13 and C24 would collapse to sH instead of H.
def assert_family(*modes):
    bad = [m for m in modes if m in SPECIAL]
    if bad:
        raise AssertionError(f'G/sH reached a lattice-only code path: {bad}. Vertex-mode joins must run before glauber_adjust.')


# ------------------------------------------------------------- mode_meet / mode_join
_DIRS = ([('iso', None, None)] + [('lf', fam, None) for fam in (13, 24)] + [('leg', fam, leg) for fam, legs in FAM_LEGS.items() for leg in legs])


# Closed-form glb, O(1), ONLY for unconditional cases: comparable pairs, and same-direction pairs (product order glb =
# componentwise max, label from the labelled side). Cross-leg / cross-family overlap -> None; the caller falls back to
# the full direction scan. Boundaries are subtle (sigma ties kill the leg label; leg-free inputs can give leg-free
# cross-family results; comparability first -- mode_meet(C13, S^2C24) = S^2C24 since C13 is harder).
def _meet_fast(a: Mode, b: Mode):
    if a == b:
        return a
    if softer(a, b):
        return a
    if softer(b, a):
        return b
    if _relation(a, b) != 'same':
        return None                                  # -> full scan
    m = max(a.m, b.m); s = max(a.sigma, b.sigma)
    leg = a.leg if a.leg is not None else b.leg
    fam = a.fam if a.fam is not None else b.fam
    n = s - m
    if n == -1:
        leg = fam = None
    elif n == 0:
        leg = None
    return Mode(m, n, leg, fam)


def _join_fast(a: Mode, b: Mode):
    if a == b:
        return a
    if harder(a, b):
        return a
    if harder(b, a):
        return b
    r = _relation(a, b)
    if r == 'same':
        m = min(a.m, b.m); s = min(a.sigma, b.sigma)
        leg = a.leg if a.leg is not None else b.leg
        fam = a.fam if a.fam is not None else b.fam
        n = s - m
        if n == -1:
            leg = fam = None
        elif n == 0:
            leg = None
        return Mode(m, n, leg, fam)
    if r == 'xleg':
        # (min m, max m): works for n=INF too (m stays finite).
        m = min(a.m, b.m); s = max(a.m, b.m)
        src = a if a.m <= b.m else b
        leg, fam = src.leg, src.fam
    else:                                          # xfam
        # (min m, max m - 1): INF-safe (m finite).
        m = min(a.m, b.m); s = max(a.m, b.m) - 1
        src = a if a.m <= b.m else b
        leg, fam = src.leg, src.fam
    n = s - m
    if n < -1:
        raise ArithmeticError(f'mode_join({a},{b}) produced n={n} < -1')
    if n == -1:
        leg = fam = None
    elif n == 0:
        leg = None
    return Mode(m, n, leg, fam)


def _rel_dir(kind, fam, leg, z: Mode) -> str:
    if kind == 'iso' or z.n == -1:
        return 'same'
    if fam != z.fam:
        return 'xfam'
    if kind == 'lf' or z.leg is None:
        return 'same'
    return 'same' if leg == z.leg else 'xleg'


def _candidates(a: Mode, b: Mode, mode: str):
    out = []
    for kind, fam, leg in _DIRS:
        if mode == 'glb':
            lo_m = lo_s = -1
            for z in (a, b):
                r = _rel_dir(kind, fam, leg, z)
                if r == 'same':
                    lo_m, lo_s = max(lo_m, z.m), max(lo_s, z.sigma)
                elif r == 'xleg':
                    lo_m = max(lo_m, z.sigma)
                else:
                    lo_m = max(lo_m, z.sigma + 1)
            lo_m = max(lo_m, 0)
            if lo_m == INF or lo_s == INF:
                continue                          # would need an infinite power
            if kind == 'iso':
                m = max(lo_m, lo_s + 1); cand = Mode(m, -1)
            elif kind == 'lf':
                m = max(lo_m, lo_s);     cand = Mode(m, 0, None, fam)
            else:
                m = lo_m
                cand = Mode(m, max(lo_s, m + 1) - m, leg, fam)
        else:
            hi_m = hi_s = INF
            for z in (a, b):
                r = _rel_dir(kind, fam, leg, z)
                if r == 'same':
                    hi_m, hi_s = min(hi_m, z.m), min(hi_s, z.sigma)
                elif r == 'xleg':
                    hi_s = min(hi_s, z.m)
                else:
                    hi_s = min(hi_s, z.m - 1)
            if hi_m == INF or hi_s == INF:
                continue
            if kind == 'iso':
                m = min(hi_m, hi_s + 1)
                if m < 0:
                    continue
                cand = Mode(m, -1)
            elif kind == 'lf':
                m = min(hi_m, hi_s)
                if m < 0:
                    continue
                cand = Mode(m, 0, None, fam)
            else:
                s = hi_s; m = min(hi_m, s - 1)
                if m < 0 or s < m + 1:
                    continue
                cand = Mode(m, s - m, leg, fam)
        out.append(cand)
    uniq = set(out)
    if mode == 'glb':
        return sorted({x for x in uniq if not any(y != x and harder(y, x) for y in uniq)})
    return sorted({x for x in uniq if not any(y != x and softer(y, x) for y in uniq)})


# Greatest lower bound (softest common ... hardest common softer mode).
@lru_cache(maxsize=None)
def mode_meet(a: Mode, b: Mode) -> Mode:
    if a in SPECIAL or b in SPECIAL:
        return _chain_meet(a, b)
    r = _meet_fast(a, b)
    if r is not None:
        return r
    c = _candidates(a, b, 'glb')
    if len(c) != 1:
        raise ArithmeticError(f'glb({a},{b}) not unique: {c}')
    return c[0]


@lru_cache(maxsize=None)
def mode_join(a: Mode, b: Mode) -> Mode:
    if a in SPECIAL or b in SPECIAL:
        return _chain_join(a, b)
    r = _join_fast(a, b)
    if r is not None:
        return r
    c = _candidates(a, b, 'lub')
    if len(c) != 1:
        raise ArithmeticError(f'lub({a},{b}) not unique: {c}')
    return c[0]


# --------------------------------------------- precomputed table (fast path 2)
# 2026-08-29: precompute the full mode_meet/mode_join table for the common mode
# range so that repeated calls are O(1) dict hits -- exactly what the old
# table's _expand_formulas does, but for the lattice.  The table is built
# at import time (a few seconds) and lives inside the lru_cache of mode_meet/join.
_PRECOMP_M = 6                       # m <= 6
_PRECOMP_N = 5                       # n <= 5 (per leg)


def _precomp_set():
    out = [Mode(m, -1) for m in range(_PRECOMP_M + 1)]
    for fam, legs in FAM_LEGS.items():
        for m in range(_PRECOMP_M + 1):
            out.append(Mode(m, 0, None, fam))
            for leg in legs:
                for n in range(1, _PRECOMP_N + 1):
                    out.append(Mode(m, n, leg, fam))
    return sorted(set(out))


# Fill the mode_meet/mode_join caches over the common range, closed under mode_meet/mode_join (m, sigma bounded, so the closure
# terminates). Subsequent calls anywhere inside the closure are O(1) lru_cache hits; anything outside falls back to
# the fast path / direction scan.
def _warm_cache(verbose=False):
    M = _precomp_set()
    seen = set(M)
    frontier = list(M)
    while frontier:
        a = frontier.pop()
        for b in list(seen):
            for op in (mode_meet, mode_join):
                try:
                    r = op(a, b)
                except (ArithmeticError, ValueError):
                    continue
                if r not in seen:
                    seen.add(r)
                    frontier.append(r)
    if verbose:
        print(f'[primitives] precomputed mode_meet/mode_join over {len(seen)} modes ({len(seen) * len(seen)} pairs), mode_meet cache {mode_meet.cache_info()}, mode_join cache {mode_join.cache_info()}')
    return len(seen)


# ------------------------------------------------------------------ self test
def _bounded_set(mmax=4, nmax=3):
    out = [Mode(m, -1) for m in range(mmax + 1)]
    for fam, legs in FAM_LEGS.items():
        for m in range(mmax + 1):
            out.append(Mode(m, 0, None, fam))
            for leg in legs:
                for n in range(1, nmax + 1):
                    out.append(Mode(m, n, leg, fam))
    return sorted(set(out))


def self_test(mmax=4, nmax=3, verbose=True):
    M = _bounded_set(mmax, nmax)
    res = {}
    res['modes'] = len(M)
    res['V_formula'] = all(x.V == 2 * x.m + x.n + 1 for x in M)
    res['scaling_gives_V'] = all(x.scaling()[0] + x.scaling()[1] == x.V and 2 * x.scaling()[2] == x.V for x in M)
    res['antisymmetry'] = sum(1 for a, b in itertools.combinations(M, 2) if softer(a, b) and softer(b, a))
    res['transitivity'] = sum(1 for a, b, c in itertools.permutations(M, 3) if softer(a, b) and softer(b, c) and not softer(a, c))
    nonuniq = 0
    for a, b in itertools.combinations_with_replacement(M, 2):
        for op in (mode_meet, mode_join):
            try:
                op(a, b)
            except ArithmeticError:
                nonuniq += 1
    res['non_unique_glb_lub'] = nonuniq
    # cross-check: closed-form fast path vs the full direction scan
    bad_fast = 0
    for a, b in itertools.combinations_with_replacement(M, 2):
        for op in ('mode_meet', 'mode_join'):
            try:
                scan = (_candidates(a, b, 'glb') if op == 'mode_meet' else _candidates(a, b, 'lub'))
                fast = (_meet_fast(a, b) if op == 'mode_meet' else _join_fast(a, b))
            except (ArithmeticError, ValueError):
                continue
            if fast is not None and (len(scan) != 1 or scan[0] != fast):
                bad_fast += 1
    res['fastpath_mismatch'] = bad_fast
    res['idempotence'] = sum(1 for a in M if mode_meet(a, a) != a or mode_join(a, a) != a)
    res['V_identity'] = sum(1 for a, b in itertools.combinations(M, 2) if a.V + b.V != mode_meet(a, b).V + mode_join(a, b).V)
    res['absorption'] = sum(1 for a, b in itertools.permutations(M, 2) if mode_meet(a, mode_join(a, b)) != a or mode_join(a, mode_meet(a, b)) != a)
    res['meet_assoc'] = sum(1 for a, b, c in itertools.permutations(M, 3) if mode_meet(mode_meet(a, b), c) != mode_meet(a, mode_meet(b, c)))
    res['join_assoc'] = sum(1 for a, b, c in itertools.permutations(M, 3) if mode_join(mode_join(a, b), c) != mode_join(a, mode_join(b, c)))
    if verbose:
        print(f'self_test: {res["modes"]} modes (m<={mmax}, n<={nmax})')
        for k in ('V_formula', 'scaling_gives_V'):
            print(f'  {k:20} {res[k]}')
        for k in ('antisymmetry', 'transitivity', 'non_unique_glb_lub', 'idempotence', 'V_identity', 'absorption', 'meet_assoc', 'join_assoc', 'fastpath_mismatch'):
            flag = 'ok' if res[k] == 0 else '*** FAIL'
            print(f'  {k:20} {res[k]:>6}  {flag}')
    return res


PYSD_CASES = [
    # (label, a, b, expected edge scaling)   -- both were required simultaneously
    ('loop4b k4 A/B  (same leg 4)', Mode(1, 1, 4), Mode(0, 2, 4), -4),
    ('CrownST k4     (leg 4 vs 2)', Mode(1, 1, 4), Mode(0, 2, 2), -5),
]
BASICS = [
    (C(13), C(24), 'S', 'H'),
    (CC(1), CC(3), 'S^1C13', 'C13'),
    (CC(1, 2), CC(3, 2), 'S^2C13', 'C13'),
    (CC(1), CC(3, 2), 'S^1C3C13', 'C13'),
    (CC(1), CC(2), 'S^2', 'H'),
    (S(1), CC(1, 2), 'S^1C1C13', 'C13'),
    (C(13, 1), CC(1, 2), 'S^1C1C13', 'C1C13'),
    (C(13), C(24, 2), 'S^2C24', 'C13'),
]


def check_physics(verbose=True):
    ok = True
    if verbose:
        print('\nG / sH chain (order: H >- G >- sH >- L):')
    chain_cases = [
        ('mode_meet', G, C(13), 'C13'), ('mode_meet', SH, C(13), 'C13'),
        ('mode_meet', G, H, 'G'), ('mode_meet', G, SH, 'sH'),
        ('mode_join', G, C(13), 'G'), ('mode_join', SH, C(13), 'sH'),
        ('mode_join', G, SH, 'G'), ('mode_join', G, H, 'H'), ('mode_join', SH, SH, 'sH'),
    ]
    for op, a, b, want in chain_cases:
        got = str((mode_meet if op == 'mode_meet' else mode_join)(a, b))
        good = got == want
        ok &= good
        sym = '/\\' if op == 'mode_meet' else '\\/'
        note = '' if op == 'mode_join' else '   (dead path in region_checker)'
        if verbose:
            print(f'  {str(a):4} {sym} {str(b):6} = {got:5} (want {want:5}) {"ok" if good else "*** FAIL"}{note}')
    try:
        assert_family(C(13), G)
        ok = False
        if verbose:
            print('  assert_family did NOT fire  *** FAIL')
    except AssertionError:
        if verbose:
            print('  assert_family fires on (C13, G)  ok')
    if verbose:
        print('\npySD constraints that were mutually contradictory in the old table:')
    for lbl, a, b, want in PYSD_CASES:
        r = mode_meet(a, b)
        good = -r.V == want
        ok &= good
        if verbose:
            print(f'  {lbl:30} {a} /\\ {b} = {str(r):11} -> {-r.V:>3}   want {want:>3}   {"ok" if good else "*** FAIL"}')
    if verbose:
        print('\nbasic entries (old table / established):')
    for a, b, wm, wj in BASICS:
        gm, gj = str(mode_meet(a, b)), str(mode_join(a, b))
        good = (gm == wm and gj == wj)
        ok &= good
        if verbose:
            print(f'  {str(a):9} /\\ {str(b):9} = {gm:11} (want {wm:11})  \\/ = {gj:8} (want {wj:6}) {"ok" if good else "*** FAIL"}')
    return ok


_WARM_SIZE = _warm_cache(verbose=False)


# ============================ mode string layer ============================

# x marginally softer than y (formula: mode_marginally_softer); the G/sH sentinels and the leg-blind
# legacy symbols have no marginal reading.
@lru_cache(maxsize=None)
def marginally_softer(x1, x2):
    if x1 in ('G', 'sH') or x2 in ('G', 'sH') or x1 in LEG_BLIND or x2 in LEG_BLIND:
        return False
    return mode_marginally_softer(to_mode(x1), to_mode(x2))

# Port-walk hardness test: mw >= X, with G hard, sH never, H only against non-H; else V(mw) <= V(X).
def harder_or_eq(mw, X):
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
    return V(mw) <= V(X)

# Virtuality of a mode string, from the name alone: 𝒱(S^m C_i^n C_ij) = 2m+n+1 (n=∞ → ∞); G/sH sentinels carry 𝒱 = 1.
@lru_cache(maxsize=None)
def V(key):
    return 1 if key in SPECIAL else to_mode(key).V

@lru_cache(maxsize=None)
def meet(a, b):
    return old_meet(a, b)

@lru_cache(maxsize=None)
def join(a, b):
    return old_join(a, b)

def vee(modes):
    acc = modes[0]
    for m in modes[1:]: acc = join(acc, m)
    return acc

# ============================ mode blocks / scaling ============================

# 1VI blocks of a MULTIGRAPH: self-loops become single-vertex blocks; parallel edges are distinguished by index.
# Returns list of (vertex_set, edge_index_list) pairs.
@lru_cache(maxsize=None)
def _find_1vi_blocks(verts, edges):
    loops = [(i, e) for i, e in enumerate(edges) if e[0] == e[1]]
    other = [(i, e) for i, e in enumerate(edges) if e[0] != e[1]]
    other_map = dict(other)
    out = []
    for i, (a, b) in loops:
        out.append(({a}, [i]))
    adj = {v: [] for v in verts}
    for i, (a, b) in other:
        adj[a].append((b, i))
        adj[b].append((a, i))
    disc = {}
    low = {}
    t = 0
    stack = []
    blocks = []

    def dfs(u, pe):
        nonlocal t
        disc[u] = low[u] = t
        t += 1
        for (w, ei) in adj[u]:
            if ei == pe:
                continue
            if w not in disc:
                stack.append(ei)
                dfs(w, ei)
                low[u] = min(low[u], low[w])
                if low[w] >= disc[u]:
                    blk = set()
                    while True:
                        e = stack.pop()
                        blk.add(e)
                        if e == ei:
                            break
                    blocks.append(blk)
            elif disc[w] < disc[u]:
                stack.append(ei)
                low[u] = min(low[u], disc[w])

    for v in verts:
        if v not in disc:
            dfs(v, -1)
            if stack:
                blocks.append(set(stack))
                stack.clear()
    for blk in blocks:
        be = [other_map[i] for i in blk]
        out.append(({v for e in be for v in e}, blk))
    used = {v for bv, _ in out for v in bv}
    for v in verts:
        if v not in used:
            out.append(({v}, []))
    return out


def find_1vi_blocks(verts, edges):
    # content-keyed memo wrapper: normalise arguments, delegate to the cached impl
    return _find_1vi_blocks(verts if isinstance(verts, frozenset) else frozenset(verts), edges if isinstance(edges, tuple) else tuple(map(tuple, edges)))

# 1VI blocks of the contracted X-subgraph: X edges + X vertices (join-mode), non-X endpoints absorbed into aux.
# Blocks sharing a cut vertex stay SEPARATE — not connected components.  Each block is (vertex_set, contracted_edges, original_edge_indices).
# The ORIGINAL indices are needed because distinct self-loops (1,3) and (7,8) both contract to (aux,aux) and are otherwise indistinguishable.
def mode_components(mode, vm, em, edges, verts):
    gv = {v for v in verts if vm.get(v) == mode}
    ge_idx = [i for i, m in enumerate(em) if m == mode]
    if not gv and not ge_idx:
        return []
    verts2 = set(gv) | {'aux'}
    edges2 = []
    for i in ge_idx:
        a, b = edges[i]
        edges2.append(('aux' if a not in gv else a, 'aux' if b not in gv else b))
    out = []
    for (bv, eidx2) in find_1vi_blocks(verts2, edges2):
        out.append((bv, [edges2[j] for j in eidx2], [ge_idx[j] for j in eidx2]))
    return out

def to_scaling(em):
    return tuple(-V(m) for m in em) + (1,)

# ============================ external momenta ============================

def ext_mode_for(ext_mode, n):
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

# ============================ graph helpers ============================

@lru_cache(maxsize=None)
def _connected(vs, es): # checks whether the induced graph (including only edges whose endpoints both lie in vs) is connected; vs: vertex set (may include the auxiliary vertex "aux"); es: edge list.
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


def connected(vs, es):
    # content-keyed memo wrapper: normalise arguments, delegate to the cached impl
    return _connected(vs if isinstance(vs, frozenset) else frozenset(vs), es if isinstance(es, tuple) else tuple(map(tuple, es)))

@lru_cache(maxsize=None)
def _is_1vi(vs, es): # checks whether the graph is one-vertex irreducible (deleting any single vertex leaves the graph connected).
    if len(vs) <= 1: return True
    for v in vs:
        rest = vs - {v}
        if not _connected(rest, es): return False
    return True


def is_1vi(vs, es):
    # content-keyed memo wrapper: normalise arguments, delegate to the cached impl
    return _is_1vi(vs if isinstance(vs, frozenset) else frozenset(vs), es if isinstance(es, tuple) else tuple(map(tuple, es)))

@lru_cache(maxsize=None)
def _components(es, verts): # returns a mapping {vertex: component_root} via Union-Find (disjoint set union).
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


def components(es, verts):
    # content-keyed memo wrapper: normalise arguments, delegate to the cached impl
    return _components(es if isinstance(es, tuple) else tuple(map(tuple, es)), verts if isinstance(verts, frozenset) else frozenset(verts))

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
            if connected(S, edges): out.append(S)
    return out

# Connected supersets of req inside dom.
def conn_sets(req, dom, edges):
    req = set(req); dd = set(dom)
    if not req <= dd:
        return []
    extras = sorted(dd - req); out = []
    for r in range(len(extras) + 1):
        for add in combinations(extras, r):
            C = req | set(add)
            if connected(C, edges):
                out.append(frozenset(C))
    return out

# Supersets of req inside dom; every connected component contains va or vb.
def comp_sets(req, dom, va, vb, edges):
    req = set(req); dd = set(dom)
    if not req <= dd:
        return []
    extras = sorted(dd - req); out = []
    for r in range(len(extras) + 1):
        for add in combinations(extras, r):
            S = req | set(add)
            comp = components(edges, S)
            groups = {}
            for v in S:
                groups.setdefault(comp[v], set()).add(v)
            if all(va in g or vb in g for g in groups.values()):
                out.append(frozenset(S))
    return out


def clear_graph_caches():
    # drop the per-case memo tables (called at case start by the skeleton drivers)
    _connected.cache_clear()
    _is_1vi.cache_clear()
    _components.cache_clear()
    _find_1vi_blocks.cache_clear()


if __name__ == '__main__':
    t0 = time.perf_counter()
    r = self_test()
    good = check_physics()
    # careful: isinstance(True, int) is True in Python -- separate the flags
    bad = [k for k, v in r.items() if v is False]
    bad += [k for k, v in r.items() if not isinstance(v, bool) and k != 'modes' and v]
    print('\nRESULT:', 'ALL GREEN' if not bad and good else f'FAILURES: {bad}')
