#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mode_level_interactive.py — interactive stepper for the mode first-appearance ladder
(mode_levels module; the public regge 2->2 toolkit).

Ask for the four external-momentum modes (p1..p4; plain Enter = the regge k1
defaults), show the L = 0 modes, then advance one loop level per key press:

    SPACE : next loop level (+1)
    ENTER : jump to the next level that has new modes
    A     : show the cumulative table so far
    Q/ESC : quit

The display is capped at L = 10.

Run:  python3 mode_level_interactive.py
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from regge_modes import Mode, INF, to_mode, to_old
from mode_levels import Predictor

LMAX = 10

BANNER = '''\
==============================================================
  mode first-appearance stepper   (regge)
--------------------------------------------------------------
  Enter the four external-momentum modes (Enter = k1 defaults),
  then:
    SPACE = step +1 loop      ENTER = jump to next non-empty loop
    A     = cumulative table  Q     = quit
  (levels are capped at L = %d)
==============================================================''' % LMAX

DEFAULT_MOMENTA = ['C1∞C13', 'C2∞C24', 'C3∞C13', 'C4∞C24']

def _fmt_cost(c):
    return 'tree' if c == 0 else ('1 loop' if c == 1 else '%d loops' % c)


def leg_check(i, m):
    """True when mode m is admissible as the momentum mode of leg i."""
    if not isinstance(m, Mode):
        return False
    fam = 13 if i in (1, 3) else 24
    if m.n == INF:
        return m.fam == fam and m.leg == i
    return m.fam == fam and (m.leg is None or m.leg == i)


def ask_momenta():
    print('Enter the four external-momentum modes '
          '(Enter = regge k1 defaults; examples: C13 / C1C13 / C2∞C24 / S^1C13):')
    out = {}
    for i, dflt in enumerate(DEFAULT_MOMENTA, 1):
        while True:
            try:
                s = input('  p%d mode [%s]: ' % (i, dflt)).strip()
            except EOFError:
                s = ''
            if not s:
                s = dflt
            try:
                m = to_mode(s)
            except Exception as e:
                print('  ↑ parse failed (%s), try again' % str(e)[:60])
                continue
            if not leg_check(i, m):
                fam = 13 if i in (1, 3) else 24
                print('  ↑ p%d expects a %d-family mode '
                      '(C%d, C%dC%d, C%d^2C%d, C%d∞C%d, ...) — try again'
                      % (i, fam, fam, i, fam, i, fam, i, fam))
                continue
            out[i] = s
            break
    return out


def read_key():
    """single keypress on a tty; line-based fallback otherwise ('' line = space)."""
    if not sys.stdin.isatty():
        try:
            s = input('▶ (SPACE=advance, J=jump, A=all, Q=quit): ')
        except EOFError:
            return 'q'
        s = s.strip()
        if s == '':
            return ' '
        return s[0]
    import termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


class Stepper:
    def __init__(self, ext_modes):
        self.ext = dict(ext_modes)
        self.pred = None
        self.levels = {}
        self.by_name = {}
        self.computed = -1

    def ensure(self, k):
        k = min(k, LMAX)
        if k <= self.computed:
            return
        self.pred = Predictor(self.ext, Lmax=k)
        self.pred.run()
        self.levels = self.pred.levels()
        self.by_name = {to_old(x): x for x in self.pred.cost if x.n != INF}
        self.computed = k

    def modes_at(self, k):
        return self.levels.get(k, [])

    def source_text(self, name):
        """One-line origin of mode `name` (route records set by mode_levels)."""
        if name == 'G':
            return 'tree level (Glauber)'
        if name == 'sH':
            return '1 loop (semihard)'
        x = self.by_name.get(name)
        if x is None:
            return 'tree level (hard)' if name == 'H' else '—'
        r = self.pred.route.get(x)
        if r is None:
            return '—'
        kind = r[0]
        if kind == 'ext':
            return 'external mode'
        if kind == 'hard':
            return 'tree level (hard)'
        if kind == 'base':
            return 'base structure (1 loop)'
        if kind == 'messenger':
            _, m, stock, sc = r
            return 'messenger: %s (%s) + %d' % (stock, _fmt_cost(sc), 2 * m)
        if kind == 'messenger2':
            _, m, branch = r
            third = self._other_legs(branch)
            A, B = ('C2', 'C4') if branch == '24' else ('C1', 'C3')
            p = '' if m == 1 else '^%d' % m
            A, B = A + p + 'C' + branch, B + p + 'C' + branch
            return ('messenger: relevant to %s, %s, and %s simultaneously, '
                    'with %s confirmed IR-compatible' % (A, B, third, third))
        if kind == 'vee':
            _, a, ca, b, cb = r
            if a.n == INF or b.n == INF:
                nc, cc = (b, a) if a.n == INF else (a, b)
                return 'vee (member extension): %s ∨ %s' % (to_old(nc), to_old(cc))
            return 'vee: %s (%s) ∨ %s (%s)' % (to_old(a), _fmt_cost(ca),
                                               to_old(b), _fmt_cost(cb))
        if kind == 'wedge':
            _, a, ca, b, cb = r
            if (cb, to_old(b)) < (ca, to_old(a)):
                a, ca, b, cb = b, cb, a, ca
            return ('wedge: %s (%s) + %s (%s) + an additional loop'
                    % (to_old(a), _fmt_cost(ca), to_old(b), _fmt_cost(cb)))
        return '—'

    def _other_legs(self, branch):
        legs = (1, 3) if branch == '24' else (2, 4)
        names = []
        for i in legs:
            m = to_mode(self.ext[i])
            if m.n != INF:
                nm = to_old(m)
                if nm not in names:
                    names.append(nm)
        if len(names) == 1:
            return names[0]
        return '%s (or %s)' % (names[0], ' or '.join(names[1:]))


def main():
    print(BANNER)
    ext = ask_momenta()
    print('momenta: ' + ', '.join('p%d=%s' % (i, ext[i]) for i in sorted(ext)))
    st = Stepper(ext)
    st.ensure(0)
    print('L=0 seeds: %s' % ', '.join(st.modes_at(0)))

    cur = 0

    def show(k):
        st.ensure(k)
        names = st.modes_at(k)
        print()
        if names:
            n = len(names)
            print('L=%d — %d new mode%s:' % (k, n, 's' if n != 1 else ''))
            w = min(22, max(len(nm) for nm in names))
            for nm in names:
                print('   %-*s — %s' % (w, nm, st.source_text(nm)))
        else:
            print('L=%d — (no new modes)' % k)
        n_le = sum(len(st.modes_at(lv)) for lv in st.levels if lv <= k)
        print('   cumulative: %d modes (≤L=%d) | SPACE=+1, ENTER=next non-empty, A=all, Q=quit'
              % (n_le, k))

    show(cur)
    while True:
        k = read_key()
        if k in ('q', 'Q', '\x1b', '\x03'):
            print('\nbye 👋')
            break
        if k == ' ':
            if cur < LMAX:
                cur += 1
                show(cur)
            else:
                print('  (display capped at L=%d)' % LMAX)
        elif k in ('\r', '\n', 'j'):
            j0, hit = cur, None
            while j0 < LMAX:
                hi = min(j0 + 6, LMAX)
                st.ensure(hi)
                for k2 in range(j0 + 1, hi + 1):
                    if st.modes_at(k2):
                        hit = k2
                        break
                if hit is not None:
                    break
                j0 += 6
            if hit is None:
                print('  (no new modes up to L=%d)' % LMAX)
            else:
                cur = hit
                show(cur)
        elif k in ('a', 'A'):
            st.ensure(cur)
            print()
            for lv in sorted(set(st.levels)):
                if lv > cur:
                    break
                print('L=%-3d: %s' % (lv, ', '.join(st.modes_at(lv))))
        else:
            print('  (SPACE=advance, ENTER=jump, A=all, Q=quit)')


if __name__ == '__main__':
    main()
