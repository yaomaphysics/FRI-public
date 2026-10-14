#!/usr/bin/env python3
"""
primitives.py — fundamental tools.

External kinematics (2->3): for a raw virtuality tuple ms = (m1, ..., m5) — m = 1, 2, ... for
p_i^2 ~ t1**m and INF for exactly on-shell — the external modes follow the established formulas
    p_i = C_i^{m_i} (i = 1, 4, 5),  p_2 = C_2^{m_2-1} C_23,  p_3 = C_3^{m_3-1} C_23
(m = 1 degenerates the pair member: C_2^0 C_23 = C23, member marker dropped).  ext_modes() and
mass_exprs() implement them for an arbitrary ms.
"""

import re
from functools import lru_cache

INF = float('inf')


# ==================== MODE ALGEBRA ====================

# Hard mode constructor.
def H():  return ('H',)
# pure-soft mode S^m (S^1 by default).
def S(m=1): return ('S', m)
# wide-leg carrier mode S^m C_i^n (n >= 1 or INF; m = soft prefactor).
def W(i, n, m=0): return ('C', i, n, None, m)
# pair-family mode S^m C_mem^k C23 (k = refinement index; mem = member leg).
def P(k, mem=0, m=0): return ('C', 23, k, mem, m)

# True for pure-soft modes.
def isS(x): return x[0] == 'S'
# True for carrier modes (wide and pair).
def isC(x): return x[0] == 'C'
# soft prefactor m of a mode.
def m_of(x): return x[1] if x[0] == 'S' else (x[4] if x[0] == 'C' else 0)
# carrier label (1/4/5 = wide leg, 23 = pair family).
def d_of(x): return None if x[0] != 'C' else x[1]
# depth index (n for wide legs, k for pair modes).
def n_of(x): return x[2] if x[0] == 'C' else None
# pair member leg (None for wide legs / base C23).
def mem_of(x): return x[3] if x[0] == 'C' else None

# level index of a mode (edge scalings are -V).
def V(x):
    if x[0] == 'H': return 0
    if x[0] == 'S': return 2 * x[1]
    _, d, n, mem, m = x
    if d in (1, 4, 5):
        return INF if n == INF else 2 * m + n
    return INF if n == INF else 2 * m + (n + 1)

# pretty name of a mode (e.g. C1, S^1C5, C2C23, H).
def name(x):
    if x[0] == 'H': return 'H'
    if x[0] == 'S': return 'S' if x[1] == 1 else f'S^{x[1]}'
    _, d, n, mem, m = x
    pre = '' if m == 0 else (f'S^{m}')
    if d in (1, 4, 5):
        lift = '∞' if n == INF else ('' if n == 1 else f'^{n}')
        return f'{pre}C{d}{lift}'
    if d == 23:
        if n == 0:
            base = 'C23'
        elif n == INF:
            base = f'C{mem}∞C23'
        else:
            base = f'C{mem}C23' if n == 1 else f'C{mem}^{n}C23'
        return f'{pre}{base}'
    raise ValueError(x)

# parse a mode-name string back into a canonical tuple.
def parse(s):
    s = s.strip()
    if s == 'H': return H()
    m = re.match(r'^S\^(\d+)$', s)
    if m: return S(int(m.group(1)))
    if s == 'S': return S(1)
    m = re.match(r'^(?:S\^(\d+))?C(\d+)(?:\^(\d+)|(∞))?(C23)?$', s)
    if not m: raise ValueError(f'cannot parse mode {s!r}')
    sm = int(m.group(1)) if m.group(1) else 0
    d = int(m.group(2))
    if m.group(4) == '∞':
        n = INF
    elif m.group(3):
        n = int(m.group(3))
    else:
        n = None  # bare
    pair = m.group(5) is not None
    if d in (1, 4, 5):
        return W(d, 1 if n is None else n, sm)
    if d == 23 or pair:
        if pair:
            # C{mem}C23 / C{mem}^kC23 / C{mem}∞C23
            mem = d
            k = 1 if n is None else (INF if n == INF else n)
            return P(k, mem, sm)
        return P(0, 0, sm)  # bare C23
    raise ValueError(s)

# ==================== DIRECTION HELPERS ====================

# True when both modes sit on the same carrier direction (leg).
def same_dir(a, b):
    return d_of(a) is not None and d_of(a) == d_of(b)
# True for wide-leg carrier modes (legs 1, 4, 5).
def wide_like(x): return x[0] == 'C' and x[1] in (1, 4, 5)
# True for pair-family modes (C23 family).
def pair_like(x): return x[0] == 'C' and x[1] == 23


# True when mode m is admissible as the momentum mode of leg i.
def leg_check(i, m):
    d = d_of(m)
    if i in (1, 4, 5):
        return d == i
    if i in (2, 3):
        return d == 23 and mem_of(m) in (0, i)
    return False

# Decode a (m, sigma) pair into a wide-leg mode; degenerate cases fall back to S or H.
def _decode_wide(m, sig, leg):
    # n = sig - m; n <= 0 gives a pure S^m.
    n = sig - m
    if n <= 0:
        if n == 0:
            return S(m) if m >= 1 else S(1)  # degenerate, shouldn't happen
        return S(m) if m >= 1 else H()
    if leg is None:
        return S(m) if m >= 1 else H()
    return W(leg, n, m)


# Wide×wide carrier vee/wedge for finite carrier powers; coordinates (m, n, i) = (soft index, carrier power, direction).
def _wide_join_meet(a, b):
    # Normalise a coordinate: a zero carrier power carries no direction.
    def _norm(t):
        m, n, i = t
        if n == 0:
            i = 0
        return (m, n, i)

    # Equality of two coordinates, ignoring the direction when a carrier power is zero.
    def _eq(X, Y):
        X, Y = _norm(X), _norm(Y)
        return X[0] == Y[0] and X[1] == Y[1] and (X[2] == Y[2] or X[1] == 0 or Y[1] == 0)

    # Harder-or-equal order on the coordinates.
    def _harder_or_eq(X, Y):
        X, Y = _norm(X), _norm(Y)
        if _eq(X, Y):
            return True
        if X[2] == Y[2]:
            return X[0] <= Y[0] and X[0] + X[1] <= Y[0] + Y[1]
        return X[0] + X[1] <= Y[0]

    X, Y = _norm((a[4], a[2], a[1])), _norm((b[4], b[2], b[1]))
    if _eq(X, Y):
        vt = wt = X
    else:
        iX = X[2] if X[1] != 0 else (Y[2] if Y[1] != 0 else 0)
        iY = Y[2] if Y[1] != 0 else iX
        if iX == iY:
            A, B = _norm((X[0], X[1], iX)), _norm((Y[0], Y[1], iX))
            vt = wt = None
            for (P, Q) in ((A, B), (B, A)):
                m1, n1, _ = P
                m2, n2, _ = Q
                if m2 < m1 <= m1 + n1 < m2 + n2:
                    vt = _norm((m2, m1 + n1 - m2, iX))
                    wt = _norm((m1, m2 + n2 - m1, iX))
                    break
            if vt is None:
                if _harder_or_eq(A, B) and not _harder_or_eq(B, A):
                    vt, wt = A, B
                elif _harder_or_eq(B, A) and not _harder_or_eq(A, B):
                    vt, wt = B, A
                else:
                    vt, wt = A, B
        else:
            if _harder_or_eq(X, Y) and not _harder_or_eq(Y, X):
                vt, wt = X, Y
            elif _harder_or_eq(Y, X) and not _harder_or_eq(X, Y):
                vt, wt = Y, X
            else:
                P, Q = (X, Y) if X[0] + X[1] <= Y[0] + Y[1] else (Y, X)
                m1, n1, i = P
                m2, n2, j = Q
                if m1 <= m2:
                    jn = _norm((m1, m2 - m1, i))
                else:
                    jn = _norm((m2, m1 - m2, j))
                mt = _norm((m1 + n1, m2 + n2 - m1 - n1, j))
                vt, wt = jn, mt

    # Convert a coordinate back into a mode (n <= 0 -> S or H).
    def _back(t):
        m, n, i = t
        if n <= 0:
            return S(m) if m >= 1 else H()
        return W(i, n, m)

    return _back(vt), _back(wt)

# ==================== PAIR-FAMILY ALGEBRA ====================

# Pair mode -> (m, k, leg) coordinates used by the pair algebra.
def _pair_parts(x):
    return (x[4], x[2], (x[3] if x[3] else None))

# sigma = m + n of a pair coordinate (INF-safe).
def _pair_sig(z):
    m, n, leg = z
    return INF if n == INF else m + n

# pair coordinates -> mode (n = -1 -> S/H; n = 0 -> base C23).
def _pair_from(m, n, leg):
    if n == -1:
        return H() if m == 0 else S(m)
    if n == INF and m != 0:
        raise ArithmeticError('n = INF requires m = 0 (external momentum)')
    if n == 0:
        return P(0, 0, m)
    return P(n, leg if leg else 0, m)

# same-relation test for pair coordinates (legless counts as same).
def _pair_same_rel(u, v):
    if u[1] == -1 or v[1] == -1:
        return True
    if u[2] is None or v[2] is None:
        return True
    return u[2] == v[2]

# pair softness order (same-leg / cross-leg cases).
def _pair_softer(u, v):
    if _pair_same_rel(u, v):
        return u[0] >= v[0] and _pair_sig(u) >= _pair_sig(v)
    return u[0] >= _pair_sig(v)

# inverse relation of _pair_softer.
def _pair_harder(a, b):
    return _pair_softer(b, a)

# relation class of a coordinate against a candidate direction.
def _pair_rel_dir(kind, z):
    if kind == 'iso' or z[1] == -1:
        return 'same'
    if kind == 'lf' or z[2] is None:
        return 'same'
    return 'same' if (2 if kind == 'leg2' else 3) == z[2] else 'xleg'

# glb/lub candidate search over the finite direction set.
def _pair_candidates(u, v, mode):
    out = []
    for kind in ('iso', 'lf', 'leg2', 'leg3'):
        if mode == 'glb':
            lo_m = lo_s = -1
            for z in (u, v):
                r = _pair_rel_dir(kind, z)
                if r == 'same':
                    lo_m = max(lo_m, z[0]); lo_s = max(lo_s, _pair_sig(z))
                elif r == 'xleg':
                    lo_m = max(lo_m, _pair_sig(z))
                else:
                    lo_m = max(lo_m, _pair_sig(z) + 1)
            lo_m = max(lo_m, 0)
            if lo_m == INF or lo_s == INF:
                continue
            if kind == 'iso':
                out.append((max(lo_m, lo_s + 1), -1, None))
            elif kind == 'lf':
                out.append((max(lo_m, lo_s), 0, None))
            else:
                m = lo_m
                out.append((m, max(lo_s, m + 1) - m, 2 if kind == 'leg2' else 3))
        else:
            hi_m = hi_s = INF
            for z in (u, v):
                r = _pair_rel_dir(kind, z)
                if r == 'same':
                    hi_m = min(hi_m, z[0]); hi_s = min(hi_s, _pair_sig(z))
                elif r == 'xleg':
                    hi_s = min(hi_s, z[0])
                else:
                    hi_s = min(hi_s, z[0] - 1)
            if hi_m == INF or hi_s == INF:
                continue
            if kind == 'iso':
                m = min(hi_m, hi_s + 1)
                if m < 0: continue
                out.append((m, -1, None))
            elif kind == 'lf':
                m = min(hi_m, hi_s)
                if m < 0: continue
                out.append((m, 0, None))
            else:
                s = hi_s; m = min(hi_m, s - 1)
                if m < 0 or s < m + 1: continue
                out.append((m, s - m, 2 if kind == 'leg2' else 3))
    uniq = set(out)
    if mode == 'glb':
        keep = {x for x in uniq if not any(y != x and _pair_harder(y, x) for y in uniq)}
    else:
        keep = {x for x in uniq if not any(y != x and _pair_softer(y, x) for y in uniq)}
    return keep

# pair-family meet (greatest lower bound).
def _pair_meet(a, b):
    u, v = _pair_parts(a), _pair_parts(b)
    if u == v:
        return a
    if _pair_softer(u, v):
        return _pair_from(*u)
    if _pair_softer(v, u):
        return _pair_from(*v)
    if _pair_same_rel(u, v):
        m = max(u[0], v[0]); s = max(_pair_sig(u), _pair_sig(v))
        leg = u[2] if u[2] is not None else v[2]
        n = s - m
        if n == -1 or n == 0:
            leg = None
        return _pair_from(m, n, leg)
    keep = _pair_candidates(u, v, 'glb')
    if len(keep) != 1:
        raise ArithmeticError('glb23(%s,%s) not unique: %s' % (name(a), name(b), sorted(name(_pair_from(*x)) for x in keep)))
    return _pair_from(*next(iter(keep)))

# pair-family join (least upper bound).
def _pair_join(a, b):
    u, v = _pair_parts(a), _pair_parts(b)
    if u == v:
        return a
    if _pair_harder(u, v):
        return _pair_from(*u)
    if _pair_harder(v, u):
        return _pair_from(*v)
    if _pair_same_rel(u, v):
        m = min(u[0], v[0]); s = min(_pair_sig(u), _pair_sig(v))
        leg = u[2] if u[2] is not None else v[2]
        n = s - m
        if n == -1 or n == 0:
            leg = None
        return _pair_from(m, n, leg)
    # cross-leg closed form: (min m, max m), label from the smaller-m side; INF-safe (m stays finite).
    m = min(u[0], v[0]); s = max(u[0], v[0])
    src = u if u[0] <= v[0] else v
    leg = src[2]
    n = s - m
    if n == -1 or n == 0:
        leg = None
    return _pair_from(m, n, leg)

# ==================== CROSS-FAMILY ALGEBRA (WIDE x PAIR) ====================

# Wide/pair mode -> (m, n', leg, family) coordinates (n' = n - 1 for a wide leg).
def _wp_parts(x):
    if wide_like(x):
        n = x[2]
        return (x[4], INF if n == INF else n - 1, x[1], 'W')
    return (x[4], x[2], (x[3] if x[3] else None), 'P')

# sigma of a cross-family coordinate (INF-safe).
def _wp_sig(z):
    return INF if z[1] == INF else z[0] + z[1]

# cross-family coordinates -> mode.
def _wp_from(m, n, leg, fam):
    if n == -1:
        return H() if m == 0 else S(m)
    if fam == 'W':
        if n == INF:
            return W(leg, INF, m)
        return W(leg, 1 if n == 0 else n + 1, m)
    if n == INF:
        if m != 0:
            raise ArithmeticError('n = INF requires m = 0 (external momentum)')
        return P(INF, leg if leg else 0, m)
    if n == 0:
        return P(0, 0, m)
    return P(n, leg if leg else 0, m)

# relation class (family/leg) of a coordinate against a candidate direction.
def _wp_rel2(z, kind, fam, leg):
    if kind == 'iso' or z[1] == -1:
        return 'same'
    if z[3] != fam:
        return 'xfam'
    if kind == 'lf' or z[2] is None:
        return 'same'
    return 'same' if leg == z[2] else 'xleg'

# full softness order for the cross-family coordinates.
def _wp_softer_full(u, v):
    if u[1] == -1 or v[1] == -1:
        return u[0] >= v[0] and _wp_sig(u) >= _wp_sig(v)
    if u[3] != v[3]:
        return u[0] >= _wp_sig(v) + 1
    if u[2] is None or v[2] is None:
        return u[0] >= v[0] and _wp_sig(u) >= _wp_sig(v)
    if u[2] == v[2]:
        return u[0] >= v[0] and _wp_sig(u) >= _wp_sig(v)
    return u[0] >= _wp_sig(v)

# cross-family meet (wide x pair).
def _wide_pair_meet(a, b):
    u, v = _wp_parts(a), _wp_parts(b)
    wleg = u[2] if u[3] == 'W' else v[2]

    # convert cross-family coordinates back to a mode (leg fallback via wleg).
    def conv(x):
        m, n, leg, fam = x
        if fam == 'W' and leg is None:
            leg = wleg
        return _wp_from(m, n, leg, fam)

    if u == v:
        return conv(u)
    if _wp_softer_full(u, v):
        return conv(u)
    if _wp_softer_full(v, u):
        return conv(v)
    cands = []
    for kind, fam, leg in (('iso', None, None), ('lf', 'W', None), ('leg', 'W', wleg), ('lf', 'P', None), ('leg', 'P', 2), ('leg', 'P', 3)):
        lo_m = lo_s = -1
        for z in (u, v):
            r = _wp_rel2(z, kind, fam, leg)
            if r == 'same':
                lo_m = max(lo_m, z[0]); lo_s = max(lo_s, _wp_sig(z))
            elif r == 'xleg':
                lo_m = max(lo_m, _wp_sig(z))
            else:
                lo_m = max(lo_m, _wp_sig(z) + 1)
        lo_m = max(lo_m, 0)
        if lo_m == INF or lo_s == INF:
            continue
        if kind == 'iso':
            cands.append((max(lo_m, lo_s + 1), -1, None, None))
        elif kind == 'lf':
            cands.append((max(lo_m, lo_s), 0, None, fam))
        else:
            m = lo_m
            cands.append((m, max(lo_s, m + 1) - m, leg, fam))
    uniq = set(cands)
    keep = {x for x in uniq if not any(y != x and _wp_softer_full(x, y) for y in uniq)}
    if len(keep) != 1:
        raise ArithmeticError('glb23(%s,%s) not unique: %s' % (name(a), name(b), sorted(name(conv(x)) for x in keep)))
    return conv(next(iter(keep)))

# cross-family join (wide x pair).
def _wide_pair_join(a, b):
    u, v = _wp_parts(a), _wp_parts(b)
    if u == v:
        return _wp_from(*u)
    if _wp_softer_full(u, v):
        return _wp_from(*v)
    if _wp_softer_full(v, u):
        return _wp_from(*u)
    # cross-family closed form: (min m, max m - 1), label from the smaller-m side (leg kept: our base wide is leg-tagged).
    m = min(u[0], v[0]); s = max(u[0], v[0]) - 1
    if s < -1:
        raise ArithmeticError('join(%s,%s) produced s < -1' % (name(a), name(b)))
    src = u if u[0] <= v[0] else v
    n = s - m
    if n == -1:
        return H() if m == 0 else S(m)
    leg = src[2]
    return _wp_from(m, n, leg, src[3])

@lru_cache(maxsize=None)   # [Optimization] memoised
# Mode meet (AND): softest common refinement; raises ArithmeticError when undefined. Memoised.
def meet(a, b):
    # Keep the key ordered: tie cases return the first argument, so meet(a, b) may differ from meet(b, a).
    if a == b: return a
    if a == H(): return b
    if b == H(): return a
    # S x S
    if isS(a) and isS(b):
        return S(max(a[1], b[1]))
    # S x C: softest common refinement (nu = n - 1 for wide legs, nu = k for pair modes).
    if isS(a) or isS(b):
        s_, c_ = (a, b) if isS(a) else (b, a)
        m1 = s_[1]
        ms = m_of(c_)
        mm = max(m1, ms)
        if wide_like(c_):
            nu = n_of(c_)
            if nu == INF:
                raise ArithmeticError(f'meet soft x inf: ({name(a)},{name(b)})')
            nu = nu - 1
        else:
            nu = n_of(c_)
            if nu == INF:
                raise ArithmeticError(f'meet soft x inf: ({name(a)},{name(b)})')
        sig = max(m1 - 1, ms + nu)
        nr = sig - mm
        if wide_like(c_):
            if nr >= 0:
                return W(d_of(c_), nr + 1, mm)
            return S(mm)
        if nr >= 1:
            return P(nr, mem_of(c_), mm)
        if nr == 0:
            return P(0, 0, mm)
        return S(mm)
    # C x C
    if wide_like(a) and wide_like(b):
        if n_of(a) != INF and n_of(b) != INF:
            # Carrier wedge — same rules as wide_angle/primitives._join_meet.
            return _wide_join_meet(a, b)[1]
        # n = INF takes the INF-specific branch.
        if same_dir(a, b):
            d_ = d_of(a)
            # Same-direction special case: S^1C_i ∧ C_i^2 = S^1C_i  (SC5)
            if {a, b} == {W(d_, 1, 1), W(d_, 2, 0)}:
                return W(d_, 1, 1)
            m = max(m_of(a), m_of(b))
            sig = max(V(a), V(b))
            return _decode_wide(m, sig, d_of(a))
        sig1, sig2 = V(a), V(b)
        m = min(sig1, sig2)
        sig = max(sig1, sig2)
        leg = d_of(a) if sig1 > sig2 else (d_of(b) if sig2 > sig1 else None)
        return _decode_wide(m, sig, leg)
    if pair_like(a) and pair_like(b):
        return _pair_meet(a, b)
    # wide x pair
    if (wide_like(a) and pair_like(b)) or (pair_like(a) and wide_like(b)):
        return _wide_pair_meet(a, b)
    # hybrid wide x pair
    if m_of(a) == 0 and m_of(b) == 0:
        w = a if wide_like(a) else b
        p = b if wide_like(a) else a
        an, k = n_of(w), n_of(p)
        if an == INF or k == INF:
            raise ArithmeticError(f'meet hybrid inf: ({name(a)},{name(b)})')
        i, j = d_of(w), mem_of(p)
        if an <= k:
            return P(k - an, j, an) if k - an >= 1 else P(0, 0, an)
        elif an == k + 1:
            return S(k + 1)
        else:
            return W(i, an - k - 1, k + 1)
    raise ArithmeticError(f'meet unhandled: ({name(a)},{name(b)})')

@lru_cache(maxsize=None)   # [Optimization] memoised
# Mode join (OR): combined mode of two momenta at a vertex. Memoised.
def join(a, b):
    if a == b: return a
    if a == H() or b == H(): return H()
    if isS(a) and isS(b):
        return S(min(a[1], b[1]))
    if isS(a) or isS(b):
        s_ = a if isS(a) else b
        c_ = b if isS(a) else a
        # soft x carrier join: (min m, min sigma - min m) arithmetic on the indices
        # (wide legs: our index n = nu + 1; pair: our k = nu); soft prefix kept.
        ms = c_[4]
        mm = min(s_[1], ms)
        if wide_like(c_):
            nu = n_of(c_)
            nu = INF if nu == INF else nu - 1
        else:
            nu = n_of(c_)
        sig2 = nu if nu == INF else ms + nu
        sig = min(s_[1] - 1, sig2)
        nu2 = sig - mm
        if nu2 < 0:
            return S(mm) if mm >= 1 else H()
        if wide_like(c_):
            return W(d_of(c_), 1 if nu2 == 0 else nu2 + 1, mm)
        if nu2 == 0:
            return P(0, 0, mm)
        return P(nu2, mem_of(c_), mm)
    if wide_like(a) and wide_like(b):
        if n_of(a) != INF and n_of(b) != INF:
            # Carrier vee — same rules as wide_angle/primitives._join_meet.
            return _wide_join_meet(a, b)[0]
        # n = INF takes the INF-specific branch.
        if same_dir(a, b):
            # S^1C_i^n ∨ C_i^∞ = C_i^{n+1}
            for x, y in ((a, b), (b, a)):
                if x[4] == 1 and y[4] == 0 and n_of(y) == INF and n_of(x) != INF:
                    return W(d_of(x), n_of(x) + 1, 0)
            m = min(m_of(a), m_of(b))
            sig = min(V(a), V(b))
            return _decode_wide(m, sig, d_of(a))
        # Soft-wide S^m C_i joined with an INF-mode C_j (j != i) gives C_j^m.
        for x, y in ((a, b), (b, a)):
            if x[4] >= 1 and y[4] == 0 and n_of(y) == INF:
                return W(d_of(y), x[4], 0)
        return H()
    if pair_like(a) and pair_like(b):
        return _pair_join(a, b)
    # wide x pair
    if (wide_like(a) and pair_like(b)) or (pair_like(a) and wide_like(b)):
        return _wide_pair_join(a, b)
    # wide x pair
    return H()


# ==================== GRAPH HELPERS ====================
# connectivity of a vertex subset via the induced edges.
def connected(vs, es):
    vs = set(vs)
    if len(vs) <= 1: return True
    adj = {v: set() for v in vs}
    for (a, b) in es:
        if a in vs and b in vs:
            adj[a].add(b); adj[b].add(a)
    seen = {next(iter(vs))}; st = list(seen)
    while st:
        v = st.pop()
        for u in adj[v]:
            if u not in seen: seen.add(u); st.append(u)
    return seen == vs

# True when deleting any single vertex keeps the graph connected (1-vertex irreducible).
def is_1vi(vs, es):
    vs = set(vs)
    if len(vs) <= 1: return True
    for v in vs:
        if not connected(vs - {v}, es): return False
    return True

# Union-Find root of x in parent (path halving).
def uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


# Merge the trees of a and b in parent.
def uf_union(parent, a, b):
    ra, rb = uf_find(parent, a), uf_find(parent, b)
    if ra != rb:
        parent[ra] = rb


@lru_cache(maxsize=None)   # [Optimization] memoised
# Connected components of the induced subgraph: {vertex: component_root} via Union-Find.
def _components(es, verts):
    parent = {v: v for v in verts}
    for (a, b) in es:
        if a in parent and b in parent:
            uf_union(parent, a, b)
    return {v: uf_find(parent, v) for v in verts}

# [Optimization] Content-keyed memo wrapper: normalise the arguments and delegate to the cached implementation.
def components(es, verts):
    return _components(es if isinstance(es, tuple) else tuple(map(tuple, es)), verts if isinstance(verts, frozenset) else frozenset(verts))

# 1VI blocks of (verts, edges): self-loops and isolated vertices become single-vertex blocks.
def find_1vi_blocks(verts, edges):
    loops = [(a, b) for (a, b) in edges if a == b]
    other = [(a, b) for (a, b) in edges if a != b]
    out = [({a}, [(a, b)]) for (a, b) in loops]
    adj = {v: [] for v in verts}
    for i, (a, b) in enumerate(other):
        adj[a].append((b, i)); adj[b].append((a, i))
    disc = {}; low = {}; t = 0; stack = []; blocks = []
    # Tarjan DFS collecting the blocks.
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


# ==================== MODE RELATIONS ====================
# The marginally-softer test and the harder-or-equal order used by the monotone walks.
def marginally_softer(src, dst):
    if src == dst: return False
    if isS(src) and isC(dst):
        # Marginal means equal sigma: sigma(S^m) = m - 1, sigma(C_i^n) = m + n - 1 (wide), sigma(C_j^kC23) = m + k.
        if wide_like(dst):
            sg = m_of(dst) + n_of(dst) - 1
        elif pair_like(dst):
            sg = m_of(dst) + n_of(dst)
        else:
            return False
        return sg == src[1] - 1
    if isC(src) and isC(dst):
        if same_dir(src, dst):
            return V(src) > V(dst)
        m = m_of(src)
        if m < 1:
            return False
        if pair_like(src) and wide_like(dst) and m_of(dst) == 0:
            nd = n_of(dst)          # wide level = n-1; cross -> m == level+1 = n
            return nd != INF and m == nd
        if wide_like(src) and pair_like(dst) and m_of(dst) == 0:
            kd = n_of(dst)          # pair level = k; cross -> m == k + 1
            return kd != INF and m == kd + 1
        if wide_like(src) and wide_like(dst) and m_of(dst) == 0:
            # wide soft-carrier -> wide target, level L = n-1; m == L+1 = n (e.g. SC5 is marginally softer than C1)
            nd = n_of(dst)
            return nd != INF and m == nd
        return False
    if isC(src) and isS(dst):
        # Soft carrier -> pure soft: the carrier soft power must equal the target soft power.
        return m_of(src) == dst[1]
    return False


# Harder-or-equal order on the 2->3 modes, used by the monotone walks and the port (B) clause.
def harder_or_eq(mw, X):
    if mw == X: return True
    if mw == H(): return True
    if X == H(): return False
    # mode -> (m, n, i) coordinates for the order test.
    def rc(x):
        if isS(x): return (x[1], 0, 0)
        if wide_like(x):
            n = x[2]
            zr = INF if n == INF else n - 1
            return (x[4], zr, 0 if zr == 0 else x[1])
        if pair_like(x):
            k = x[2]
            leg = 2 if x[3] == 2 else (4 if x[3] == 3 else 0)
            return (x[4], INF if k == INF else k, 0 if k == 0 else leg)
        return (0, -1, 0)
    m1, n1, i1 = rc(mw); m2, n2, i2 = rc(X)
    s1 = INF if n1 == INF else m1 + n1
    s2 = INF if n2 == INF else m2 + n2
    if i1 == i2:
        return m1 <= m2 and s1 <= s2
    return s1 <= m2


# ==================== EXTERNAL KINEMATICS ====================

# The five m_i**2 expressions handed to pySecDec for a raw (m1, ..., m5) virtuality tuple.
def mass_exprs(ms):
    return ['0' if m == INF else ('t1*m%dsq' % (i + 1) if m == 1 else 't1**%d*m%dsq' % (m, i + 1)) for i, m in enumerate(ms)]


# the ext_mode dict {'p1'..'p5'} for a raw (m1, ..., m5) virtuality tuple (general kinematics).
def ext_modes(ms):
    m1, m2, m3, m4, m5 = tuple(ms)

    # Pair member mode: C_mem^{m-1} C23 (m = 1 degenerates to a bare C23).
    def pair(m, mem):
        k = m - 1 # note that INF - 1 = INF
        return P(k, mem if k >= 1 else 0)

    return {'p1': W(1, m1), 'p2': pair(m2, 2), 'p3': pair(m3, 3), 'p4': W(4, m4), 'p5': W(5, m5)}
