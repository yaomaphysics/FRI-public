#!/usr/bin/env python3
r"""regge_modes.py -- Regge-limit momentum-mode algebra.

2026-08-29.  Every momentum mode is one of three types:

    S^m C_i^n C_ij  |  G (Glauber)  |  sH (semihard)

The parametrised family S^m C_i^n C_ij:

    m >= 0 integer
    n in {-1, 0, 1, 2, ...} U {INF}
    family (pair) ij in {13, 24};  leg i in {1,3} (fam 13) or {2,4} (fam 24)

    n = -1 : the trailing C_ij is cancelled by C_i^{-1}  ->  pure S^m
             (m = 0 as well  ->  the hard mode H).  No leg, no family.
    n =  0 : S^m C_ij  (m = 0 -> the plain C13 / C24 cut).  Leg label absent.
    n = INF: the four lightlike external momenta C_i^INF C_ij (m = 0 forced).

Scaling, in the light-cone basis of leg i (beta_i lightlike, see the draft):

    k^mu = (k.betabar_i, k.beta_i, k.beta_iperp) ~ lambda^m (1, l^{n+1}, l^{(n+1)/2})

Virtuality (derived, not postulated: k^2 = 2(k.bb)(k.b) - k_perp^2 and both
terms are the same order, so there is no cancellation):

    V(S^m C_i^n C_ij) = 2m + n + 1          (n = -1 degenerates to 2m)

Order (2026-08-29): drop the trailing C_ij and compare the S^m C_i^n parts
by the *wide-angle* rules.  With sigma = m + n:

    same leg       : X1 softer than X2  <=>  m1 >= m2  and  sigma1 >= sigma2
    different leg  : X1 softer than X2  <=>  m1 >= sigma2
    different fam  : X1 softer than X2  <=>  m1 >= sigma2 + 1

Modes with n <= 0 carry no leg label, so inside their family they compare by
the same-leg rule (they are isotropic in the leg degree of freedom).  The +1
in the last row is the back-to-back projection: across families the minus
component of X2 is what shows up as the plus component in X1's frame.

Wedge / vee, same family (2026-08-29 19:16): strip the trailing C_ij, do
the wide-angle operation on S^m C_i^n (three cases), put the
C_ij back.  Verified equal to the glb/lub of the order above on 1872/1872
same-family pairs (m<=3, n<=3) -- EXCEPT when n = -1 is involved, where all
112 disagreements have the strip-and-wide-angle answer failing to be a bound
at all.  Reason: C_i^{-1} and C_ij cancel *as a pair* (so C_ij cannot be
stripped alone: Regge's S^m has sigma = m-1 while wide-angle's pure soft
S^m = S^m C^0 has sigma = m), and more fundamentally wide-angle compares a
direction-free S^m to C_i^n by the DIFFERENT-direction criterion (that is
where  S wedge C_i^2 = SC_i  comes from) whereas the Regge S^m ~ l^m (1,1,1) is
genuinely isotropic -- nothing to project, so the comparison is componentwise.

    PATCH: pure S^m (n = -1, including H) is isotropic and
    combines with any same-family mode by the wide-angle SAME-direction rule,
    i.e. the product order on (m, sigma).  n >= 0 uses strip-and-wide-angle.

With that clause: 1984/1984 agreement on same-family pairs.  Counterexample it
fixes: C1C13 /\ S -> S (unpatched) is not a lower bound, since S ~ l(1,1,1) is
LARGER than C1C13 ~ (1,l^2,l) in the minus component; glb = S^1C13 (V=3),
which is also what the old table has.

Cross-family wedge/vee: still to come.  The glb/lub of the order
above predicts  meet = (min sigma + 1, max sigma) with the label of the larger
sigma,  join = (min m, max m - 1) with the label of the smaller m.

meet / join are computed as genuine glb / lub of this order -- no tables.
Consequences verified by self_test():
  * the order is a partial order (transitive, antisymmetric)
  * glb / lub exist and are unique  =>  it is a lattice
  * idempotence, absorption, associativity, and the virtuality identity
    V(X1) + V(X2) = V(X1 meet X2) + V(X1 join X2)  all hold identically
  * the 2026-08-18 hand rule "meet keeps the deeper leg" is now a theorem

G and sH (2026-08-29)
--------------------------
Order:  H  >-  G  >-  sH  >-  every S^m C_i^n C_ij mode
        (">-" = harder than).  G is the necklace string, sH the pearl between
        two G's, so an sH vertex only ever sees sH and G neighbours.

They are NOT members of the lattice L = {S^m C_i^n C_ij}; they are overlay
products, and the separation is enforced by *ordering in time*, not by fiat:

  regge_core._build_region does
     (1) vertex modes  vm[v] = vee(incident em)         <- L only
     (2) self-consistency  em[e] == meet(vm[u], vm[v])  <- L only, and the
         comment at line 2576 says it explicitly: "Checked BEFORE
         glauber_adjust (G is an adjustment product, not an overlay mode)"
     (3) glauber_adjust  assigns G, then recomputes vm with vee   <- sees G
     (4) necklace_detect assigns sH, then _recompute_vm            <- sees sH

  => meet NEVER sees G or sH          (dead table entries; the correction
                                       MEET(G,C13)=C13 is pure cleanup)
  => join sees G in step 3 and sH in step 4, but by then an sH vertex has only
     sH/G neighbours, so join(sH, family mode) is vacuous too.

This is why join(C13, C24) = H survives: that join happens in step 1, before
sH exists.  As a *poset* L u {G, sH} is fine, but L is not a join-sublattice
of it (the lub of C13 and C24 in the bigger poset is sH), so the two must never
be mixed in one call.  assert_family() below guards the step-1/2 code paths.
"""
from __future__ import annotations

import itertools
import time
from functools import lru_cache
from typing import Optional, Tuple

INF = float('inf')
FAM_LEGS = {13: (1, 3), 24: (2, 4)}
LEG_FAM = {1: 13, 3: 13, 2: 24, 4: 24}


class Mode(tuple):
    """Canonical mode.  Stored as (m, n, leg, fam); leg/fam are None when the
    corresponding label is meaningless (n <= 0 kills leg, n = -1 kills fam)."""
    __slots__ = ()

    def __new__(cls, m: int, n, leg: Optional[int] = None,
                fam: Optional[int] = None):
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

    @property
    def V(self):
        """Virtuality 2m + n + 1 = m + sigma + 1."""
        return INF if self.n == INF else 2 * self.m + self.n + 1

    def scaling(self):
        """Exponents of (k.betabar_i, k.beta_i, k.beta_iperp) in units of lambda."""
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


def from_old(name: str):
    """Old regge_core string -> Mode.  Leg-blind symbols return a tuple of the
    two possible readings (the information was not in the old symbol)."""
    if name in _OLD:
        return _OLD[name]
    if name in LEG_BLIND:
        fam = LEG_BLIND[name]
        return tuple(Mode(1, 1, leg, fam) for leg in FAM_LEGS[fam])
    p = _parse(name)
    if p is not None:
        return p
    raise KeyError(name)


def _parse(name: str):
    """Parse a leg-explicit mode string like 'S^1C4C24', 'C1^2C13',
    'C3∞C13', 'S^2C24', 'C24', 'S^3', 'H' into a Mode, or None if it does
    not parse (e.g. the old leg-blind 'S^1C13^2')."""
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


def to_mode(name: str) -> Mode:
    """Old regge_core mode string -> Mode, including the G / sH sentinels."""
    if name in ('G', 'sH'):
        return G if name == 'G' else SH
    x = from_old(name)
    if isinstance(x, Mode):
        return x
    return x[0]                                    # leg-blind: first reading


def old_meet(a: str, b: str) -> str:
    """regge_core-facing meet (string in, string out).  meet must NEVER
    see G/sH in the regge_core pipeline (they are assigned only after the
    self-consistency check), so this is the enforcement point."""
    if a in SPECIAL or b in SPECIAL:
        raise AssertionError(
            f'meet called with G/sH: ({a}, {b}). G/sH are overlay products '
            'and must not reach the lattice-only self-consistency check '
            '(regge_core line ~2576: checked BEFORE glauber_adjust).')
    return to_old(meet(to_mode(a), to_mode(b)))


def old_join(a: str, b: str) -> str:
    """regge_core-facing join (string in, string out).  G/sH are legal
    here: glauber_adjust / necklace_detect recompute vertex modes with vee."""
    return to_old(join(to_mode(a), to_mode(b)))


# ----------------------------------------------------------------------- order
def _relation(a: Mode, b: Mode) -> str:
    """'same' (same leg / label-free), 'xleg', or 'xfam'."""
    if a.n == -1 or b.n == -1:
        return 'same'                             # isotropic
    if a.fam != b.fam:
        return 'xfam'
    if a.leg is None or b.leg is None:
        return 'same'                             # n <= 0 has no leg
    return 'same' if a.leg == b.leg else 'xleg'


def softer(a: Mode, b: Mode) -> bool:
    """a softer-or-equal than b (2026-08-29 wide-angle-inherited rules)."""
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


def comparable(a: Mode, b: Mode) -> bool:
    return softer(a, b) or softer(b, a)


def overlapping(a: Mode, b: Mode) -> bool:
    return not comparable(a, b)


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


def _chain_meet(a, b):
    """Softer of the two under the chain H >- G >- sH >- L.  NB: never invoked
    by regge_core -- meet runs before G/sH are assigned (see module docstring)."""
    if a == b:
        return a
    ra, rb = _rank(a), _rank(b)
    if ra != rb:
        return a if ra > rb else b            # larger rank = softer
    return a if a in SPECIAL else b


def _chain_join(a, b):
    """Harder of the two under the chain (reachable: step 3/4 vm recomputation)."""
    if a == b:
        return a
    ra, rb = _rank(a), _rank(b)
    if ra != rb:
        return a if ra < rb else b            # smaller rank = harder
    return a if a in SPECIAL else b


def assert_family(*modes):
    """Guard for the step-1/2 code paths (vertex modes and the em == vm/\\vm
    self-consistency check): G/sH must not appear there, otherwise the lub of
    C13 and C24 would collapse to sH instead of H."""
    bad = [m for m in modes if m in SPECIAL]
    if bad:
        raise AssertionError(
            f'G/sH reached a lattice-only code path: {bad}. '
            'Vertex-mode joins and the em==meet(vm,vm) check must run before '
            'glauber_adjust / necklace_detect.')


# ------------------------------------------------------------- meet / join
_DIRS = ([('iso', None, None)]
         + [('lf', fam, None) for fam in (13, 24)]
         + [('leg', fam, leg) for fam, legs in FAM_LEGS.items() for leg in legs])


def _meet_fast(a: Mode, b: Mode):
    """Closed-form glb, O(1), but ONLY for the cases where the formula is
    unconditional: comparable pairs, and same-direction pairs (product order
    glb = componentwise max, label from the labelled side).  Cross-leg and
    cross-family overlap return None -> the caller falls back to the full
    direction scan.  (The three-case formulas are subtle at the boundaries:
    sigma ties kill the leg label, leg-free inputs can make the result
    leg-free even in cross-family overlap, and the comparability check must
    come first -- meet(C13, S^2C24) is S^2C24 itself since C13 is harder.)"""
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
        raise ArithmeticError(f'join({a},{b}) produced n={n} < -1')
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
        return sorted({x for x in uniq
                       if not any(y != x and harder(y, x) for y in uniq)})
    return sorted({x for x in uniq
                   if not any(y != x and softer(y, x) for y in uniq)})


@lru_cache(maxsize=None)
def meet(a: Mode, b: Mode) -> Mode:
    """Greatest lower bound (softest common ... hardest common softer mode)."""
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
def join(a: Mode, b: Mode) -> Mode:
    if a in SPECIAL or b in SPECIAL:
        return _chain_join(a, b)
    r = _join_fast(a, b)
    if r is not None:
        return r
    c = _candidates(a, b, 'lub')
    if len(c) != 1:
        raise ArithmeticError(f'lub({a},{b}) not unique: {c}')
    return c[0]


def closure(seed):
    """Mode set generated from `seed` by meet and join."""
    cur = set(seed)
    while True:
        new = set()
        for a, b in itertools.combinations_with_replacement(sorted(cur), 2):
            for op in (meet, join):
                try:
                    new.add(op(a, b))
                except ArithmeticError:
                    pass
        if new <= cur:
            return sorted(cur)
        cur |= new


# --------------------------------------------- precomputed table (fast path 2)
# 2026-08-29: precompute the full meet/join table for the common mode
# range so that repeated calls are O(1) dict hits -- exactly what the old
# table's _expand_formulas does, but for the lattice.  The table is built
# at import time (a few seconds) and lives inside the lru_cache of meet/join.
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


def _warm_cache(verbose=False):
    """Fill the meet/join caches over the common range, closed under meet/join
    (m, sigma bounded, so the closure terminates).  Subsequent calls anywhere
    inside the closure are O(1) lru_cache hits; anything outside falls back to
    the fast path / direction scan."""
    M = _precomp_set()
    seen = set(M)
    frontier = list(M)
    while frontier:
        a = frontier.pop()
        for b in list(seen):
            for op in (meet, join):
                try:
                    r = op(a, b)
                except (ArithmeticError, ValueError):
                    continue
                if r not in seen:
                    seen.add(r)
                    frontier.append(r)
    if verbose:
        print(f'[regge_modes] precomputed meet/join over {len(seen)} modes '
              f'({len(seen) * len(seen)} pairs), '
              f'meet cache {meet.cache_info()}, join cache {join.cache_info()}')
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
    res['scaling_gives_V'] = all(
        x.scaling()[0] + x.scaling()[1] == x.V and 2 * x.scaling()[2] == x.V
        for x in M)
    res['antisymmetry'] = sum(1 for a, b in itertools.combinations(M, 2)
                              if softer(a, b) and softer(b, a))
    res['transitivity'] = sum(1 for a, b, c in itertools.permutations(M, 3)
                              if softer(a, b) and softer(b, c) and not softer(a, c))
    nonuniq = 0
    for a, b in itertools.combinations_with_replacement(M, 2):
        for op in (meet, join):
            try:
                op(a, b)
            except ArithmeticError:
                nonuniq += 1
    res['non_unique_glb_lub'] = nonuniq
    # cross-check: closed-form fast path vs the full direction scan
    bad_fast = 0
    for a, b in itertools.combinations_with_replacement(M, 2):
        for op in ('meet', 'join'):
            try:
                scan = (_candidates(a, b, 'glb') if op == 'meet'
                        else _candidates(a, b, 'lub'))
                fast = (_meet_fast(a, b) if op == 'meet'
                        else _join_fast(a, b))
            except (ArithmeticError, ValueError):
                continue
            if fast is not None and (len(scan) != 1 or scan[0] != fast):
                bad_fast += 1
    res['fastpath_mismatch'] = bad_fast
    res['idempotence'] = sum(1 for a in M if meet(a, a) != a or join(a, a) != a)
    res['V_identity'] = sum(1 for a, b in itertools.combinations(M, 2)
                            if a.V + b.V != meet(a, b).V + join(a, b).V)
    res['absorption'] = sum(1 for a, b in itertools.permutations(M, 2)
                            if meet(a, join(a, b)) != a or join(a, meet(a, b)) != a)
    res['meet_assoc'] = sum(1 for a, b, c in itertools.permutations(M, 3)
                            if meet(meet(a, b), c) != meet(a, meet(b, c)))
    res['join_assoc'] = sum(1 for a, b, c in itertools.permutations(M, 3)
                            if join(join(a, b), c) != join(a, join(b, c)))
    if verbose:
        print(f'self_test: {res["modes"]} modes (m<={mmax}, n<={nmax})')
        for k in ('V_formula', 'scaling_gives_V'):
            print(f'  {k:20} {res[k]}')
        for k in ('antisymmetry', 'transitivity', 'non_unique_glb_lub',
                  'idempotence', 'V_identity', 'absorption',
                  'meet_assoc', 'join_assoc', 'fastpath_mismatch'):
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
        ('meet', G, C(13), 'C13'), ('meet', SH, C(13), 'C13'),
        ('meet', G, H, 'G'), ('meet', G, SH, 'sH'),
        ('join', G, C(13), 'G'), ('join', SH, C(13), 'sH'),
        ('join', G, SH, 'G'), ('join', G, H, 'H'), ('join', SH, SH, 'sH'),
    ]
    for op, a, b, want in chain_cases:
        got = str((meet if op == 'meet' else join)(a, b))
        good = got == want
        ok &= good
        sym = '/\\' if op == 'meet' else '\\/'
        note = '' if op == 'join' else '   (dead path in regge_core)'
        if verbose:
            print(f'  {str(a):4} {sym} {str(b):6} = {got:5} (want {want:5})'
                  f' {"ok" if good else "*** FAIL"}{note}')
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
        r = meet(a, b)
        good = -r.V == want
        ok &= good
        if verbose:
            print(f'  {lbl:30} {a} /\\ {b} = {str(r):11} -> {-r.V:>3}'
                  f'   want {want:>3}   {"ok" if good else "*** FAIL"}')
    if verbose:
        print('\nbasic entries (old table / established):')
    for a, b, wm, wj in BASICS:
        gm, gj = str(meet(a, b)), str(join(a, b))
        good = (gm == wm and gj == wj)
        ok &= good
        if verbose:
            print(f'  {str(a):9} /\\ {str(b):9} = {gm:11} (want {wm:11})'
                  f'  \\/ = {gj:8} (want {wj:6}) {"ok" if good else "*** FAIL"}')
    return ok


_WARM_SIZE = _warm_cache(verbose=False)


if __name__ == '__main__':
    t0 = time.perf_counter()
    r = self_test()
    good = check_physics()
    # careful: isinstance(True, int) is True in Python -- separate the flags
    bad = [k for k, v in r.items() if v is False]
    bad += [k for k, v in r.items()
            if not isinstance(v, bool) and k != 'modes' and v]
    print('\nRESULT:', 'ALL GREEN' if not bad and good else f'FAILURES: {bad}')
