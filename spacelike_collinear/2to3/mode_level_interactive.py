#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mode_level_interactive.py — interactive stepper for the mode first-appearance ladder
(mode_levels module; the public 2->3 toolkit).

Ask for the five external-momentum modes (p1..p5; plain Enter = the fri23 k1
defaults), show the L = 0 modes, then advance one loop level per key press:

    space : next loop level (+1)
    enter : jump to the next level that has new modes
    a     : show the cumulative table so far
    q/esc : quit

The display is capped at L = 10.

Run:  python3 mode_level_interactive.py

(Also reachable from the unified entry facet_regions_interactive.py,
[2] -> [3].)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fri23 as F
from mode_levels import predict, derive_external_modes

LMAX = 10

BANNER = '''\
==============================================================
  mode first-appearance stepper   (fri23)
--------------------------------------------------------------
  Enter the five external-momentum modes (Enter = k1 defaults),
  then:
    space = step +1 loop      enter = jump to next non-empty loop
    a     = cumulative table  q     = quit
  (levels are capped at L = %d)
==============================================================''' % LMAX

DEFAULT_MOMENTA = ['C1∞', 'C2∞C23', 'C3∞C23', 'C4∞', 'C5∞']


def leg_check(i, m):
    """True when mode m is admissible as the momentum mode of leg i."""
    d = F.d_of(m)
    if i in (1, 4, 5):
        return d == i
    if i in (2, 3):
        return d == 23 and F.mem_of(m) in (0, i)
    return False


def ask_momenta():
    print('Enter the five external-momentum modes '
          '(Enter = fri23 k1 defaults; examples: C1∞ / C2∞C23 / C4 / S^1C4):')
    out = []
    for i, dflt in enumerate(DEFAULT_MOMENTA, 1):
        while True:
            try:
                s = input('  p%d mode [%s]: ' % (i, dflt)).strip()
            except EOFError:
                s = ''
            if not s:
                s = dflt
            try:
                m = F.parse(s)
            except Exception as e:
                print('  ↑ parse failed (%s), try again' % str(e)[:60])
                continue
            if not leg_check(i, m):
                if i in (1, 4, 5):
                    print('  ↑ p%d expects a direction-%d mode '
                          '(C%d, C%d^2, C%d∞, S^1C%d, ...) — try again'
                          % (i, i, i, i, i, i))
                else:
                    print('  ↑ p%d expects a 23-family mode '
                          '(C23, C%dC23, C%d^2C23, C%d∞C23, ...) — try again'
                          % (i, i, i, i))
                continue
            out.append(m)
            break
    return out


def read_key():
    """single keypress on a tty; line-based fallback otherwise ('' line = space)."""
    if not sys.stdin.isatty():
        try:
            s = input('▶ (blank=advance, j=jump, a=all, q=quit): ')
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
    def __init__(self, ext, momenta):
        self.ext, self.momenta = ext, momenta
        self.levels = {}
        self.m_log, self.w_log = [], []
        self.computed = -1

    def ensure(self, k):
        k = min(k, LMAX)
        if k <= self.computed:
            return
        self.levels, log = predict(self.ext, k, momenta=self.momenta)
        self.m_log, self.w_log = log['messenger_log'], log['wedge_log']
        self.computed = k

    def modes_at(self, k):
        return [F.name(m) for m in self.levels if self.levels[m] == k]

    def source_line(self, k):
        for lv, nm in self.m_log:
            if lv == k:
                return 'messenger round: %s (+2 anchored)' % nm
        for lv, names in self.w_log:
            if lv == k:
                return 'wedge round (+1): ' + ', '.join(names)
        if k == 0:
            return 'seeds (external modes)'
        return 'vee closure'


def main():
    print(BANNER)
    momenta = ask_momenta()
    mnames = [F.name(m) for m in momenta]
    print('momenta: ' + ', '.join('p%d=%s' % (i + 1, nm) for i, nm in enumerate(mnames)))
    ext = derive_external_modes(momenta)
    print('external modes (L=0 seeds): %s' % ', '.join(ext))

    st = Stepper(ext, momenta)
    cur = 0

    def show(k):
        st.ensure(k)
        names = st.modes_at(k)
        print()
        if names:
            n = len(names)
            print('L=%d — %d new mode%s:' % (k, n, 's' if n != 1 else ''))
            print('   ' + ', '.join(names))
        else:
            print('L=%d — (no new modes)' % k)
        print('   [%s]' % st.source_line(k))
        n_le = sum(1 for lv in st.levels.values() if lv <= k)
        print('   cumulative: %d modes (≤L=%d) | space=+1, enter=next non-empty, a=all, q=quit'
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
            for lv in sorted(set(st.levels.values())):
                if lv > cur:
                    break
                print('L=%-3d: %s' % (lv, ', '.join(st.modes_at(lv))))
        else:
            print('  (space=advance, enter=jump, a=all, q=quit)')


if __name__ == '__main__':
    main()
