#!/usr/bin/env python3
"""mode_levels.py -- first-appearance (loop-level) prediction for the
regge mode ladder, for a given kinematics.

Rules (settled with 小马, 2026-09-21/22):

  seeds
    L0: G, H, and every external mode without INF
    L1: sH, and C13 / C24 if they are not external
    C13 / C24 cost: 0 if external, else 1 (internal base)

  costs  (every mode takes the MIN over all routes that produce it)
    vee  (join A,B):   cost = max(cost A, cost B)
                       -- free ONLY as member extension: exactly one operand
                       is an INF carrier (e.g. S^mC24 v C2∞C24 = C2^mC24).
                       carrier-carrier or structure-structure joins are not
                       free (C1C13 v C3C13 = C13, S^2C13 v S^2C24 = S^2).
    wedge (meet A,B):  cost = cost A + cost B + 1   [union of loops + 1]
                       meets with an INF carrier are skipped.
    messenger tower:   S^mC13 = cost(C24) + 2m,  S^mC24 = cost(C13) + 2m
    messenger at L=2:  S^mC24 @ 2 when m = 1 + min(n over the FINITE 13-legs)
                       (k2/k3: S^1C24; k4: S^2C24), gated by [3]; relevant
                       to {C2^mC24, C4^mC24, one finite 13-leg structure}
                       (小马 2026-09-22).  (symmetric for S^mC13 over the
                       24-legs.)

  [3] gate for S^mC_fam (m >= 1):
      - the two legs of fam:            n(p_i) >= m
      - the two legs of the other fam:  min(n(p_i), n(p_i')) >= m - 1
    with n(C13) = 0, n(C1C13) = 1, n(C1^2C13) = 2, n(INF carrier) = INF.

Acceptance: --check reproduces the k0..k5 first-appearance tables extracted
from region_files (see EXPECT below).  Status: ALL PASS (2026-09-22).

Usage:
  python3 mode_levels.py --check        run the k0..k5 self-check
  python3 mode_levels.py --kin k4       print the ladder for one kin
"""
import os
import sys
import itertools

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from regge_modes import (Mode, meet, join, to_mode, to_old, C, H, INF)  # noqa: E402

LMAX_DEFAULT = 8
BUDGET_SLACK = 4


class Predictor:
    def __init__(self, ext_modes, Lmax=LMAX_DEFAULT):
        # ext_modes: {1: 'C13', 2: 'C24', 3: 'C13', 4: 'C24'} (or mode strings)
        self.ext = {int(k): to_mode(v) for k, v in ext_modes.items()}
        self.Lmax = Lmax
        self.mmax = min(6, Lmax // 2 + 1)
        self.cost = {}          # Mode -> best loop count
        self.route = {}         # Mode -> producing mechanism (stepper UI)

    # ---------------------------------------------------------------- helpers
    def n(self, leg):
        return self.ext[leg].n          # 0,1,2,... or INF

    def gate13(self, m):
        """[3] gate for producing S^mC13."""
        n1, n2, n3, n4 = (self.n(l) for l in (1, 2, 3, 4))
        return (n1 >= m) and (n3 >= m) and (min(n2, n4) >= (m - 1))

    def gate24(self, m):
        """[3] gate for producing S^mC24."""
        n1, n2, n3, n4 = (self.n(l) for l in (1, 2, 3, 4))
        return (n2 >= m) and (n4 >= m) and (min(n1, n3) >= (m - 1))

    def upd(self, x, c, label=None):
        if not isinstance(x, Mode):
            return False
        if c > self.Lmax + BUDGET_SLACK:
            return False
        if x not in self.cost or c < self.cost[x]:
            self.cost[x] = c
            if label is not None:
                self.route[x] = label
            return True
        return False

    # ----------------------------------------------------------------- rules
    def seed(self):
        c = self.cost
        ext_old = {to_old(v) for v in self.ext.values()}
        c[C(13)] = 0 if 'C13' in ext_old else 1
        c[C(24)] = 0 if 'C24' in ext_old else 1
        for v in self.ext.values():
            c[v] = 0                 # finite modes; carriers kept for joins only
        c[H] = 0
        for x in c:
            self.route[x] = (('ext',) if to_old(x) in ext_old else
                             ('base',) if x in (C(13), C(24)) else ('hard',))

    def towers(self):
        ch = False
        for m in range(1, self.mmax + 1):
            if self.gate13(m):
                ch |= self.upd(C(13, m), self.cost[C(24)] + 2 * m,
                               ('messenger', m, 'C24', self.cost[C(24)]))
            if self.gate24(m):
                ch |= self.upd(C(24, m), self.cost[C(13)] + 2 * m,
                               ('messenger', m, 'C13', self.cost[C(13)]))
        return ch

    def seats(self):
        """Messenger born at L=2 (k2/k3 S^1C24@2, k4 S^2C24@2): degree
        m = 1 + min(n over the FINITE legs of the other family); relevant
        to {C2^mC24, C4^mC24, one finite 13-leg structure} (小马 2026-09-22)."""
        ch = False
        fin13 = [v for v in (self.n(1), self.n(3)) if v != INF]
        if fin13:
            m = 1 + int(min(fin13))
            if self.gate24(m):
                ch |= self.upd(C(24, m), 2, ('messenger2', m, '24'))
        fin24 = [v for v in (self.n(2), self.n(4)) if v != INF]
        if fin24:
            m = 1 + int(min(fin24))
            if self.gate13(m):
                ch |= self.upd(C(13, m), 2, ('messenger2', m, '13'))
        return ch

    def run(self):
        self.seed()
        for _ in range(80):
            ch = False
            ch |= self.towers()
            ch |= self.seats()
            items = list(self.cost.items())
            for (a, ca), (b, cb) in itertools.combinations_with_replacement(items, 2):
                if max(ca, cb) <= self.Lmax + BUDGET_SLACK:
                    n_carrier = sum(1 for x in (a, b) if x.n == INF)
                    if n_carrier == 1:
                        # vee is free only as member extension: one side is an
                        # INF carrier (e.g. S^mC24 v C2∞C24 = C2^mC24).
                        # carrier-carrier / structure-structure joins are NOT
                        # free (C1C13 v C3C13 = C13, S^2C13 v S^2C24 = S^2).
                        try:
                            ch |= self.upd(join(a, b), max(ca, cb),
                                           ('vee', a, ca, b, cb))
                        except Exception:
                            pass
                if (a.n != INF and b.n != INF
                        and ca + cb + 1 <= self.Lmax + BUDGET_SLACK):
                    try:
                        ch |= self.upd(meet(a, b), ca + cb + 1,
                                       ('wedge', a, ca, b, cb))
                    except Exception:
                        pass
            if not ch:
                break

    # ---------------------------------------------------------------- output
    def levels(self):
        lv = {}
        for x, c in self.cost.items():
            if x.n == INF or c > self.Lmax:
                continue
            lv.setdefault(c, set()).add(x)
        lv.setdefault(0, set()).add(H)
        out = {}
        for L in sorted(lv):
            arr = sorted(lv[L], key=_skey)
            names = [to_old(x) for x in arr]
            if L == 0:
                names = ['G', 'H'] + [t for t in names if t not in ('G', 'H')]
            if L == 1:
                names = ['sH'] + [t for t in names if t != 'sH']
            out[L] = names
        return out


def _skey(x):
    if x.n == -1:
        return (0, 0, x.m, 0, 0)
    if x.n == 0 and x.m == 0:
        return (1, x.fam, 0, 0, 0)
    if x.n == 0:
        return (2, x.fam, x.m, 0, 0)
    if x.m == 0:
        return (3, x.fam, 0, x.n, x.leg)
    return (4, x.fam, x.m, x.n, x.leg)


# ---------------------------------------------------------------- kinematics
KIN_EXT = {
    'k0': {1: 'C13', 2: 'C24', 3: 'C13', 4: 'C24'},
    'k1': {1: 'C1∞C13', 2: 'C2∞C24', 3: 'C3∞C13', 4: 'C4∞C24'},
    'k2': {1: 'C13', 2: 'C2∞C24', 3: 'C3∞C13', 4: 'C4∞C24'},
    'k3': {1: 'C13', 2: 'C2∞C24', 3: 'C13', 4: 'C4∞C24'},
    'k4': {1: 'C1C13', 2: 'C2∞C24', 3: 'C3C13', 4: 'C4∞C24'},
    'k5': {1: 'C13', 2: 'C24', 3: 'C3∞C13', 4: 'C4∞C24'},
}

# first-appearance tables (from region_files, settled 2026-09-21/22)
EXPECT = {
    'k0': {0: ['G', 'H', 'C13', 'C24'],
           1: ['sH', 'S']},
    'k1': {0: ['G', 'H'],
           1: ['sH', 'C13', 'C24'],
           3: ['S', 'S^1C13', 'S^1C24', 'C1C13', 'C3C13', 'C2C24', 'C4C24'],
           5: ['S^2C13', 'S^2C24', 'C1^2C13', 'C3^2C13', 'C2^2C24', 'C4^2C24'],
           7: ['S^2', 'S^3C13', 'S^3C24',
               'C1^3C13', 'C3^3C13', 'C2^3C24', 'C4^3C24',
               'S^1C1C13', 'S^1C3C13', 'S^1C2C24', 'S^1C4C24']},
    'k2': {0: ['G', 'H', 'C13'],
           1: ['sH', 'C24'],
           2: ['S', 'S^1C24', 'C2C24', 'C4C24']},
    'k3': {0: ['G', 'H', 'C13'],
           1: ['sH', 'C24'],
           2: ['S', 'S^1C24', 'C2C24', 'C4C24']},
    'k4': {0: ['G', 'H', 'C1C13', 'C3C13'],
           1: ['sH', 'C13', 'C24', 'S^1C13'],
           2: ['S^2C24', 'C2^2C24', 'C4^2C24'],
           3: ['S', 'S^1C24', 'C2C24', 'C4C24'],
           4: ['S^2', 'S^1C2C24', 'S^1C4C24']},
    'k5': {0: ['G', 'H', 'C13', 'C24'],
           1: ['sH', 'S']},
}


def ladder(kin, Lmax=LMAX_DEFAULT):
    p = Predictor(KIN_EXT[kin], Lmax=Lmax)
    p.run()
    return p.levels()


def show(kin, Lmax=LMAX_DEFAULT):
    ext = KIN_EXT[kin]
    desc = ', '.join(f'p{i}={ext[i]}' for i in (1, 2, 3, 4))
    print(f'{kin}  ({desc})')
    lv = ladder(kin, Lmax)
    for L in range(0, Lmax + 1):
        items = lv.get(L, [])
        if items:
            print(f'  L{L}: ' + ', '.join(items))
    print()


def check(verbose=True):
    ok_all = True
    for kin in KIN_EXT:
        lv = ladder(kin, 8)
        for L in range(0, 9):
            got = set(lv.get(L, []))
            want = set(EXPECT.get(kin, {}).get(L, []))
            if got != want:
                ok_all = False
                if verbose:
                    print(f'{kin} L{L}: MISMATCH')
                    print(f'   extra  : {sorted(got - want)}')
                    print(f'   missing: {sorted(want - got)}')
    if verbose:
        print('ALL PASS' if ok_all else '*** FAIL ***')
    return ok_all


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args or args[0] == '--check':
        check()
    elif args[0] == '--kin':
        show(args[1] if len(args) > 1 else 'k1')
    else:
        print(__doc__)
