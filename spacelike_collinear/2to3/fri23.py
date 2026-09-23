#!/usr/bin/env python3
# fri23.py — Spacelike-collinear 2->3 FRI enumerator.
#
# Pipeline: cuts
#   -> overlay (em/vm)
#   -> fundamental pattern (momentum conservation, jets, H-region, mojetic, island)
#   -> IR compatibility (fixpoint over three conditions:
#        condition 1 = partial-sum vee of inflows;
#        condition 2 = messenger (incl. the SC23 special messenger);
#        condition 3 = meet-of-two).
#
# Run:  python3 fri23.py [k0..k4] [-v]   (built-in example; default k1)
import re
from itertools import combinations

INF = float('inf')
_DBG = False
CONDUCT23 = True  # S^mC23 hidden-path conduction (general m; approved 2026-09-19 15:15; supersedes the 03:30 bridge prototype).  Set False to disable.

# ============================ mode algebra ============================
# canonical mode tuple:
#   ('H',)                        hard
#   ('S', m)                      pure soft S^m (m >= 1)
#   ('C', d, n, mem, m)           carrier: S^m C-carrier
#     d in {1,4,5}: wide leg;  n >= 1 or INF (depth); mem = None
#     d == 23: pair family;    n >= 0 (refinement index k); mem in {0,2,3}
#     m >= 0 soft prefactor
# hard mode constructor.
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

# --- direction helpers
# True when both modes sit on the same carrier direction (leg).
def same_dir(a, b):
    return d_of(a) is not None and d_of(a) == d_of(b)
# True for wide-leg carrier modes (legs 1, 4, 5).
def wide_like(x): return x[0] == 'C' and x[1] in (1, 4, 5)
# True for pair-family modes (C23 family).
def pair_like(x): return x[0] == 'C' and x[1] == 23

def _decode_wide(m, sig, leg):
    # (m, sigma) -> wide mode.  n = sig - m; n<0 or 0 -> pure S^m!
    n = sig - m
    if n <= 0:
        if n == 0:
            return S(m) if m >= 1 else S(1)  # degenerate, shouldn't happen
        return S(m) if m >= 1 else H()
    if leg is None:
        return S(m) if m >= 1 else H()
    return W(leg, n, m)


def _wide23_join_meet(a, b):
    # wide×wide carrier vee/wedge for finite carrier powers — translated from
    # wide_angle/region_checker._join_meet (aligned 2026-09-16; keep the two in sync).
    # tuple view: (m, n, i) = (soft index, carrier power, direction).
    def _norm(t):
        m, n, i = t
        if n == 0:
            i = 0
        return (m, n, i)

    def _eq(X, Y):
        X, Y = _norm(X), _norm(Y)
        return X[0] == Y[0] and X[1] == Y[1] and (X[2] == Y[2] or X[1] == 0 or Y[1] == 0)

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

    def _back(t):
        m, n, i = t
        if n <= 0:
            return S(m) if m >= 1 else H()
        return W(i, n, m)

    return _back(vt), _back(wt)

# ---------- pair-family algebra (meet/join in the C23 family) ----------
# Our pair tuple P(k, mem, m): soft index m = x[4]; refinement level n = x[2] (INF allowed);
# member leg = x[3] when meaningful (None for base C23 / n = 0).
# The pair algebra (member legs 2/3) includes the raise cases.
# pair mode -> (m, k, leg) coordinates used by the pair algebra.
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
        keep = {x for x in uniq
                if not any(y != x and _pair_harder(y, x) for y in uniq)}
    else:
        keep = {x for x in uniq
                if not any(y != x and _pair_softer(y, x) for y in uniq)}
    return keep

# pair-family meet (greatest lower bound).
def _pair_meet23(a, b):
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
        raise ArithmeticError('glb23(%s,%s) not unique: %s'
                              % (name(a), name(b),
                                 sorted(name(_pair_from(*x)) for x in keep)))
    return _pair_from(*next(iter(keep)))

# pair-family join (least upper bound).
def _pair_join23(a, b):
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

# ---------- cross-family algebra for wide-single x pair ----------
# Coordinates: wide W(i, n_ours, m) <-> (m, n_ours-1, leg i, fam 'W');
# pair P(k, mem, m) <-> (m, k, mem, fam 'P').
# wide/pair mode -> (m, n', leg, family) coordinates (n' = n-1 for wide).
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
def _wide_pair_meet23(a, b):
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
    for kind, fam, leg in (('iso', None, None), ('lf', 'W', None),
                           ('leg', 'W', wleg), ('lf', 'P', None),
                           ('leg', 'P', 2), ('leg', 'P', 3)):
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
    keep = {x for x in uniq
            if not any(y != x and _wp_softer_full(x, y) for y in uniq)}
    if len(keep) != 1:
        raise ArithmeticError('glb23(%s,%s) not unique: %s'
                              % (name(a), name(b),
                                 sorted(name(conv(x)) for x in keep)))
    return conv(next(iter(keep)))

# cross-family join (wide x pair).
def _wide_pair_join23(a, b):
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
        raise ArithmeticError('join23(%s,%s) produced s < -1' % (name(a), name(b)))
    src = u if u[0] <= v[0] else v
    n = s - m
    if n == -1:
        return H() if m == 0 else S(m)
    leg = src[2]
    return _wp_from(m, n, leg, src[3])

# mode meet (AND): softest common refinement; ArithmeticError when undefined.
def meet23(a, b):
    if a == b: return a
    if a == H(): return b
    if b == H(): return a
    # S x S
    if isS(a) and isS(b):
        return S(max(a[1], b[1]))
    # S x C  (same-branch arithmetic):
    #   m' = max(m_S, m_soft); sig' = max(m_S - 1, m_soft + nu); nu' = sig' - m'
    #   (nu = n - 1 for i=1,4,5; nu = k for pair).
    if isS(a) or isS(b):
        s_, c_ = (a, b) if isS(a) else (b, a)
        m1 = s_[1]
        ms = m_of(c_)
        mm = max(m1, ms)
        if wide_like(c_):
            nu = n_of(c_)
            if nu == INF:
                raise ArithmeticError(f'meet23 soft x inf: ({name(a)},{name(b)})')
            nu = nu - 1
        else:
            nu = n_of(c_)
            if nu == INF:
                raise ArithmeticError(f'meet23 soft x inf: ({name(a)},{name(b)})')
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
            # carrier wedge — same rules as wide_angle/_join_meet (aligned 2026-09-16).
            return _wide23_join_meet(a, b)[1]
        # n = INF keeps the previous handling:
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
        return _pair_meet23(a, b)
    # wide x pair
    if (wide_like(a) and pair_like(b)) or (pair_like(a) and wide_like(b)):
        return _wide_pair_meet23(a, b)
    # hybrid wide x pair
    if m_of(a) == 0 and m_of(b) == 0:
        w = a if wide_like(a) else b
        p = b if wide_like(a) else a
        an, k = n_of(w), n_of(p)
        if an == INF or k == INF:
            raise ArithmeticError(f'meet23 hybrid inf: ({name(a)},{name(b)})')
        i, j = d_of(w), mem_of(p)
        if an <= k:
            return P(k - an, j, an) if k - an >= 1 else P(0, 0, an)
        elif an == k + 1:
            return S(k + 1)
        else:
            return W(i, an - k - 1, k + 1)
    raise ArithmeticError(f'meet23 unhandled: ({name(a)},{name(b)})')

# mode join (OR): combined mode of two momenta at a vertex.
def join23(a, b):
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
            # carrier vee — same rules as wide_angle/_join_meet (aligned 2026-09-16).
            return _wide23_join_meet(a, b)[0]
        # n = INF keeps the previous handling:
        if same_dir(a, b):
            # S^1C_i^n ∨ C_i^∞ = C_i^{n+1}
            for x, y in ((a, b), (b, a)):
                if x[4] == 1 and y[4] == 0 and n_of(y) == INF and n_of(x) != INF:
                    return W(d_of(x), n_of(x) + 1, 0)
            m = min(m_of(a), m_of(b))
            sig = min(V(a), V(b))
            return _decode_wide(m, sig, d_of(a))
        # 小马 2026-09-19: soft-wide (S^mC_i, m>=1) ∨ C_j^∞ (j != i) = C_j^m
        # (aligned with the pair-type route; fixes the R046-class v11 block).
        for x, y in ((a, b), (b, a)):
            if x[4] >= 1 and y[4] == 0 and n_of(y) == INF:
                return W(d_of(y), x[4], 0)
        return H()
    if pair_like(a) and pair_like(b):
        return _pair_join23(a, b)
    # wide x pair
    if (wide_like(a) and pair_like(b)) or (pair_like(a) and wide_like(b)):
        return _wide_pair_join23(a, b)
    # wide x pair
    return H()

# ============================ graph helpers ============================
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

# union-find connected components; returns {vertex: root}.
def components(es, verts):
    parent = {v: v for v in verts}
    # union-find find (path compression).
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    # union-find merge.
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for (a, b) in es:
        if a in parent and b in parent: union(a, b)
    return {v: find(v) for v in verts}

# True when deleting any single vertex keeps the graph connected (1-vertex irreducible).
def is_1vi(vs, es):
    vs = set(vs)
    if len(vs) <= 1: return True
    for v in vs:
        if not connected(vs - {v}, es): return False
    return True

def biconnected_blocks(verts, edges):
    # Biconnected blocks (self-loops / isolated vertices as single-vertex blocks).
    loops = [(a, b) for (a, b) in edges if a == b]
    other = [(a, b) for (a, b) in edges if a != b]
    out = [({a}, [(a, b)]) for (a, b) in loops]
    adj = {v: [] for v in verts}
    for i, (a, b) in enumerate(other):
        adj[a].append((b, i)); adj[b].append((a, i))
    disc = {}; low = {}; t = 0; stack = []; blocks = []
    # Tarjan DFS collecting biconnected blocks.
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
    raw = biconnected_blocks(verts2, edges2)
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
CUT_MODES = {'C23': P(0, 0), 'C1': W(1, 1), 'C4': W(4, 1), 'C5': W(5, 1),
             'C1R1': W(1, 2), 'C4R1': W(4, 2), 'C5R1': W(5, 2),
             'C2R1': P(1, 2), 'C3R1': P(1, 3), 'C2R2': P(2, 2), 'C3R2': P(2, 3)}

def build_overlay(edges, verts, ext_attach, ext_mode, cuts, vm_seen=None):
    # cuts: {name: frozenset}. Returns (em, vm) or (None, reason).
    # Vertex-first construction: vm(v) = meet of the modes of the nonempty cuts covering v (no covering cut -> H); then em(e) = vm(u) ∧ vm(w).
    # vm_seen (optional, 2026-09-18): vm-level dedup.  em is a function of vm and
    # every downstream check depends only on (em, vm) => the outcome is a
    # function of vm alone; a repeated vm can return early (before the edge
    # meets and all checks).  Callers pass a per-enumeration set.
    vm = {}
    for v in verts:
        acc = None
        for nm, S in cuts.items():
            if S and v in S:
                m = CUT_MODES[nm]
                if acc is None:
                    acc = m
                else:
                    try:
                        acc = meet23(acc, m)
                    except ArithmeticError as e:
                        return None, f'meet23:{e}'
        vm[v] = acc if acc is not None else H()
    if vm_seen is not None:
        # fast path (2026-09-22): fixed vertex order (callers pass the
        # sorted V); same equality classes as the old frozenset-of-pairs key
        vkey = tuple(vm[v] for v in verts)
        if vkey in vm_seen:
            return None, 'vm-dup'
        vm_seen.add(vkey)
    em = []
    for (a, b) in edges:
        try:
            em.append(meet23(vm[a], vm[b]))
        except ArithmeticError as e:
            return None, f'edge-meet23:{e}'
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
        acc = join23(acc, m2)
    return acc

# every jet component must touch its external leg(s).
def jet_components_ok(jv, je, ext_verts):
    if not jv: return True
    comp = components(list(je), jv)
    groups = {}
    for v in jv:
        groups.setdefault(comp[v], set()).add(v)
    for g in groups.values():
        if not (g & ext_verts): return False
    return True

# tightened connectivity for jet pieces (小马 2026-09-19): pieces connect only
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

# the subgraph outside all cuts (H) must be nonempty and connected.
def uncovered_ok(edges, verts, cuts):
    covered_v = set()
    for nm, S in cuts.items():
        if S: covered_v |= set(S)
    hv = set(verts) - covered_v
    if not hv: return False
    he = [(a, b) for (a, b) in edges if a not in covered_v and b not in covered_v]
    return connected(hv, he)

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
def island_ok(edges, verts, em, vm):
    for (bv, be, idxs) in mode_components(P(0, 0), vm, em, edges, verts):
        real = {v for v in bv if v != 'aux'}
        if not real: continue
        incident = [em[i] for i, (a, b) in enumerate(edges) if a in real or b in real]
        if incident and all(V(m) > V(P(0, 0)) for m in incident):
            return False
    return True

# ---- switches (all set to the validated settings) ----
# Enforce the condition 1 third-port check (see the third-port block below).
USE_THIRD_PORT = True
# C23 cut: every component must contain v2 or v3; for all-INF kinematics this reduces to a connected cut containing both incident roots.
# ============================ IR compatibility ============================
def marginally_softer23(src, dst):
    # src marginally softer than dst.
    # Cross-family soft-carrier entries follow the level rule: S^m-carrier -> other-side target
    # of level L iff m == L + 1; same-side targets keep the V-based rule.
    if src == dst: return False
    if isS(src) and isC(dst):
        # marginal ⟺ sigma equal:
        # sigma(S^m) = m-1; sigma(C_i^n) = m+n-1 (i=1,4,5); sigma(C_j^kC23) = m+k.
        # So S^2 is marginal to C_i^2, S^1C_i, C3C23; NOT to C23 / C_i^1 / C24 level.
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
    return False

# relevance (path version): marginally softer + monotone path (V non-increasing), no pass-through H.
def relevant23(src_blk, dst_blk, dst_mode, edges, em, vm, src_mode):
    if not marginally_softer23(src_mode, dst_mode): return False
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
            if relevant23(ext_comp, tgt, mode, edges, em, vm, m_ext):
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
                    if relevant23(line_comp, tgt, mode, edges, em, vm, m):
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
            try: acc = join23(acc, m2)
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
            try: acc = join23(acc, m2)
            except ArithmeticError: return False
        if acc == mode: return _port_ok()
    return False

# Note a subtlety in applying condition 1: we need to make sure that the momenta "..." in vee(...) must be flowing into the mode component.
# To realize this, we use the "third-port" check: besides the two vertices for the two incoming momenta (which can coincide), there must exist a third vertex to for the outgoing momentum flow.
#   (A) as a same-mode vertex shared with another same-mode component, or
#   (B) as a harder-mode vertex reached by one of the block's OWN edges.
# A scale assembled from two sources with no exit port is scaleless.
def harder_or_eq23(mw, X):
    # Order test on the 2->3 modes (S / C_i^n / C_j^kC_23 / H), used by the monotone walks and the port (B) clause.
    if mw == X: return True
    if mw == H(): return True
    if X == H(): return False
    # mode -> (m, n, i) coordinates for the order test.
    def rc23(x):
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
    m1, n1, i1 = rc23(mw); m2, n2, i2 = rc23(X)
    s1 = INF if n1 == INF else m1 + n1
    s2 = INF if n2 == INF else m2 + n2
    if i1 == i2:
        return m1 <= m2 and s1 <= s2
    return s1 <= m2

def _collect_entries23(blk, start_vs, start_es, cur_mode, edges, em, vm):
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
            if harder_or_eq23(mode_of(nb), cur):
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
def third_port23(blk, X, entry_vs, comps_same, vm, edges):
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
            if mw is not None and mw != X and harder_or_eq23(mw, X):
                return ('B', w)
    return None

def _outside_contacts(blk, X, v0, edges, em, vm):
    # Entry candidates of an outside source: element walk from its attach
    # vertex with the vertex's own mode as the starting mode.
    return _collect_entries23(blk, [v0], [], vm.get(v0, H()), edges, em, vm)

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
            if md == X or marginally_softer23(md, X):
                c = _outside_contacts(blk, X, v0, edges, em, vm)
                if c:
                    srcs.append((extn, md, c))
    for (cm, ci) in confirmed:
        if not marginally_softer23(cm, X):
            continue
        sblk = comps[cm][ci]
        sv = {v for v in sblk[0] if v != 'aux'}
        se = sblk[2] if len(sblk) > 2 else []
        c = _collect_entries23(blk, sv, se, cm, edges, em, vm)
        if c:
            srcs.append(('%s#%d' % (name(cm), ci), cm, c))
    return srcs

def port_ok_pairs(blk, X, edges, em, vm, ext_attach, ext_mode, confirmed, comps, dbg=None):
    # Third-port check over contact-based source pairs (join == X).
    if not USE_THIRD_PORT:
        return True
    srcs = _port_sources(blk, X, edges, em, vm, ext_attach, ext_mode,
                         confirmed, comps)
    comps_same = comps.get(X, [])
    for (_tag, md, cands) in srcs:
        if md == X:
            for a in cands:
                if third_port23(blk, X, {a}, comps_same, vm, edges):
                    return True
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            md1, c1 = srcs[i][1], srcs[i][2]
            md2, c2 = srcs[j][1], srcs[j][2]
            try:
                if join23(md1, md2) != X:
                    continue
            except ArithmeticError:
                continue
            for a in c1:
                for b in c2:
                    if third_port23(blk, X, {a, b}, comps_same, vm, edges):
                        return True
    if dbg is not None:
        dbg.setdefault('port_fail', []).append(
            (name(X), [(t, sorted(c)) for (t, _m, c) in srcs]))
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

def messenger_targets(blk, mode, comps, edges, em, vm, ext_attach, ext_mode, debug=None):
    # Collect (dirs, targets) for the messenger check of the S^m block blk — FULL version
    # (2026-09-19, WA-aligned).
    #   Gamma^[m] = connected closure (vertex-contact) of all soft-power-m components around
    #   the kernel block blk (S^m and S^m C^n members alike).
    #   A target gamma_i counts iff the n_i-matched Gamma member is relevant to it:
    #     n_i = 0  -> the KERNEL itself (only the checked block; no borrowing from other S blocks);
    #     n_i >= 1 -> an S^m C_i^{n_i} member ADJACENT to the kernel.
    m = mode[1]
    pool = []
    for md, lst in comps.items():
        if m_of(md) != m: continue
        for i, b in enumerate(lst):
            pool.append((md, i, b))
    gmemb = [False] * len(pool)
    for j in range(len(pool)):
        if pool[j][2] is blk: gmemb[j] = True
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
                if relevant23(blk, d, md, edges, em, vm, mode):
                    via = 'kernel'
            else:
                for j in range(len(pool)):
                    if not gmemb[j]: continue
                    md2, i2, b2 = pool[j]
                    if depth_of(md2) != n_i: continue
                    if d_of(md2) != d_of(md): continue
                    if not _comp_adjacent(b2, blk, edges): continue
                    if relevant23(b2, d, md, edges, em, vm, md2):
                        via = name(md2)
                        break
            if via is None: continue
            ok_ext = True
            for extn, vv in ext_attach.items():
                if vv in {v for v in d[0] if v != 'aux'}:
                    e = ext_mode[extn]
                    if d_of(e) is not None and d_of(e) == d_of(md) and \
                       depth_of(e) >= depth_of(md):
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

def messenger_ok(blk, mode, comps, confirmed, edges, em, vm, ext_attach, ext_mode, info=None):
    # S^m messenger check — full Gamma^[m] form (WA-aligned; kernel + n_i-matched members,
    # connected closure; >=3 distinct directions; >=1 confirmed-relevant).
    dirs, targets = messenger_targets(blk, mode, comps, edges, em, vm, ext_attach, ext_mode)
    if info is not None:
        info.setdefault('messenger', []).append((name(mode), sorted(dirs)))
    if len(dirs) < 3: return False
    # need >=1 confirmed relevant comp
    for (cm, ci) in list(confirmed):
        cblk = comps[cm][ci]
        if relevant23(blk, cblk, cm, edges, em, vm, mode):
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
        # modes (collinear part) or external momenta — with join23 equal to
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
                if join23(pool[a], pool[b]) == md:
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
                if messenger_ok(blk, sm, comps, confirmed, edges, em, vm, ext_attach, ext_mode, dbg):
                    changed |= try_confirm(sm, i)
        if _DBG: print('  -- cond2 special-messenger --')
        # [condition 2] special messenger for the 23-collinear kinematics: simultaneously relevant to a C2C23 component, a C3C23 component, and a C_i component (i in 1,4,5), with >=1 of those confirmed.
        # [hidden path — approved 2026-09-19 15:15; (二) restated 17:06 (v-E)]
        # S^m C_i conduction (m >= 1):
        #   conductor X = S^m C_i, i in {1,4,5,23};
        #   - SC23-type (i = 23): X relevant to >=1 C2^m C23 or C3^m C23 + two wide C_j^m targets
        #     [小马 2026-09-19: anchor extended to C2/C3 (fixes the R442-class k2 misses)]
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
        for _scm in sorted((md for md in comps if _is_cond_mode(md)),
                           key=lambda x: (x[1], x[4])):
            _cm = _scm[4]
            for _i, _blk in enumerate(comps.get(_scm, [])):
                if not CONDUCT23: continue
                if _scm[1] == 23:
                    # SC23-type (form unchanged): anchor >=1 C2^m C23; two wide C_j^m targets
                    # in distinct directions; conduct among the wide targets.
                    _anchor = False
                    for _am in (P(_cm, 2), P(_cm, 3)):
                        for _j2, _comp in enumerate(comps.get(_am, [])):
                            if relevant23(_blk, _comp, _am, edges, em, vm, _scm):
                                _anchor = True; break
                        if _anchor: break
                    if not _anchor:
                        continue
                    _dirs = set(); _cand = []
                    for _md in (W(1, _cm), W(4, _cm), W(5, _cm)):
                        for _j2, _comp in enumerate(comps.get(_md, [])):
                            if relevant23(_blk, _comp, _md, edges, em, vm, _scm):
                                _cand.append((_md, _j2)); _dirs.add(d_of(_md))
                    if len(_dirs) < 2:
                        continue
                    if not any(confirmed.get(t) for t in _cand):
                        continue
                    for (_md, _j2) in _cand:
                        if (_md, _j2) not in confirmed:
                            if _DBG: print('   conduct23-fire: %s#%d -> conduct to %s#%d'
                                  % (name(_scm), _i, name(_md), _j2))
                            changed |= try_confirm(_md, _j2)
                else:
                    # wide-type (2026-09-19 17:06 form / v-E):
                    # (1) X relevant to one C_i^2 target (own direction i);
                    # (2) X relevant to one C_j^1 & one C_k^1 target (i,j,k pairwise distinct; j,k in {1,23,4,5});
                    # conduction within the (2)-pair: one confirmed => the other confirmed.
                    _A = []
                    for _j2, _comp in enumerate(comps.get(W(_scm[1], 2), [])):
                        if relevant23(_blk, _comp, W(_scm[1], 2), edges, em, vm, _scm):
                            _A.append((W(_scm[1], 2), _j2))
                    if not _A:
                        continue
                    _D = {}
                    for _dd in (1, 23, 4, 5):
                        if _dd == _scm[1]: continue
                        _md1 = P(0, 0, 0) if _dd == 23 else W(_dd, 1)
                        _lst = []
                        for _j2, _comp in enumerate(comps.get(_md1, [])):
                            if relevant23(_blk, _comp, _md1, edges, em, vm, _scm):
                                _lst.append((_md1, _j2))
                        if _lst:
                            _D[_dd] = _lst
                    _dks = sorted(_D.keys())
                    for _a1 in range(len(_dks)):
                        for _a2 in range(_a1 + 1, len(_dks)):
                            for (_am, _aj) in _D[_dks[_a1]]:
                                for (_bm, _bj) in _D[_dks[_a2]]:
                                    if confirmed.get((_am, _aj)) and (_bm, _bj) not in confirmed:
                                        if _DBG: print('   conduct23-fire: %s#%d -> conduct to %s#%d'
                                              % (name(_scm), _i, name(_bm), _bj))
                                        changed |= try_confirm(_bm, _bj)
                                    elif confirmed.get((_bm, _bj)) and (_am, _aj) not in confirmed:
                                        if _DBG: print('   conduct23-fire: %s#%d -> conduct to %s#%d'
                                              % (name(_scm), _i, name(_am), _aj))
                                        changed |= try_confirm(_am, _aj)
        # [generalized 2026-09-19] S^m C23 (m >= 1): relevant to a C2^m C23 component, a C3^m C23
        # component, and a C_i^m component (i in 1,4,5) — each category nonempty, >=1 confirmed overall
        # -> the S^m C23 component is confirmed itself.  (Only m = 1 occurs on the current corpus.)
        for scm_m in sorted((md for md in comps if md[0] == 'C' and md[1] == 23
                             and md[2] == 0 and md[3] == 0 and md[4] >= 1),
                            key=lambda x: x[4]):
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
                            if relevant23(blk, comp, md, edges, em, vm, scm_m):
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
                    if relevant23(blk, comps[cm][ci], cm, edges, em, vm, md):
                        hit.append(cm)
                done = False
                for ia in range(len(hit)):
                    for ib in range(ia + 1, len(hit)):
                        try:
                            if meet23(hit[ia], hit[ib]) == md:
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

