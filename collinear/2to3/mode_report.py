#!/usr/bin/env python3
"""
mode_report.py -- per-level derivation report for the collinear 2->3 mode ladder.

For one kinematics, restate the ladder over L = 0 .. Lcur as a checkable derivation: for every mode,
the level at which it first occurs and the round that produces it, following mode_levels.py
(seeds / vee closure / wedge batch / messenger tower).

The result is rendered as a multi-page monospace PDF.

Called from the ladder stepper at any level ("r" in facet_regions_interactive.py --ladder).
"""
import os
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.dirname(HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import region_filter as F
from report_pdf import render_pdf, emit_level, cost as _cost, lev as _lev, art as _art


# ================================ WORDING ================================

# p<i> of the external momentum carrying mode name nm ('' when it is not one of the five).
def _leg_of(st, nm):
    for i, m in enumerate(st.momenta, 1):
        if F.name(m) == nm:
            return 'p%d' % i
    return ''


# =============================== EXPLANATIONS ===============================

# The "Explanation:" line for one mode, built from its recorded route.
def explain(st, name):
    m = next((x for x in st.levels if F.name(x) == name), None)
    if m is None:
        return '%s: first occurs at tree level.' % name
    r = st.routes.get(m)
    head = '%s: first occurs at %s; ' % (name, _lev(st.levels[m]))
    kind = r[0] if r else None
    if kind == 'seed':
        leg = _leg_of(st, name)
        if leg:
            return head + 'it is the external-momentum mode of %s.' % leg
        if name == 'C23':
            return head + 'p2+p3 (or p2-p3) is of the C23 scaling'
        if name == 'H':
            return head + 'p1+p2, p1-p4, etc., are of the hard scaling.'
        return head + 'it is a free vee closure of the external-momentum modes (a vee join does not raise the level).'
    if kind == 'vee':
        _, a, b = r
        la, lb = st.levels.get(a, 0), st.levels.get(b, 0)
        extra = 0 if any((st.routes.get(x) or ('',))[0] == 'messenger' for x in (a, b)) else 1
        pa, pb = _leg_of(st, F.name(a)), _leg_of(st, F.name(b))
        if pa or pb:
            nc, cc, car, leg = (b, lb, a, pa) if pa else (a, la, b, pb)
            return head + ('the simplest structure needs the %s component (%s), which joins the %s '
                           'external-momentum mode %s (+%d loop).' % (F.name(nc), _cost(cc), F.name(car), leg, extra))
        return head + ('the simplest structure joins %s %s component (%s) with %s %s component (%s) (%s).' % (
            _art(F.name(a)), F.name(a), _cost(la), _art(F.name(b)), F.name(b), _cost(lb),
            'free: one of them is a messenger' if extra == 0 else 'one extra loop'))
    if kind == 'wedge':
        _, a, b = r
        return head + ('the simplest structure needs %s %s component (%s), %s %s component (%s), and %s %s '
                       'propagator adjacent to them both (+1 loop).' % (
                           _art(F.name(a)), F.name(a), _cost(st.levels.get(a, 0)),
                           _art(F.name(b)), F.name(b), _cost(st.levels.get(b, 0)), _art(name), name))
    if kind == 'messenger':
        _, lvl, comp = r
        part = ('the tree-level external modes (tree-level)' if comp is None
                else 'the %s component (%s)' % (F.name(comp), _cost(st.levels.get(comp, 0))))
        return head + 'the simplest structure needs %s and %s %s messenger (allowed at this level; +2 loops).' % (
            part, _art(name), name)
    return head[:-2] + '.'


# =============================== REPORT / PDF ===============================

# Build the report for the levels L = 0 .. upto of the stepper's kinematics and render it; returns (pdf, lines).
def build_report(st, upto, outdir=None):
    st.ensure(upto)
    lines = []
    lines.append('kinematics: ' + ', '.join('p%d = %s' % (i, F.name(m)) for i, m in enumerate(st.momenta, 1)))
    lines.append('report covers L = 0 .. L = %d' % upto)
    lines.append('generated: ' + time.strftime('%Y-%m-%d %H:%M:%S'))
    lines.append('')
    for k in range(upto + 1):
        names = st.modes_at(k)
        cum = sum(1 for lv in st.levels.values() if lv <= k)
        emit_level(lines, k, names, 'mode', cum, lambda nm: explain(st, nm))
    fdir = outdir or os.path.join(HERE, 'fri_out')
    os.makedirs(fdir, exist_ok=True)
    pdf = os.path.join(fdir, 'mode_ladder_%s.pdf' % time.strftime('%Y%m%d-%H%M%S'))
    render_pdf(lines, pdf, 'FRI mode-ladder derivation report',
               'collinear 2->3 kinematics — first appearance of each mode, L = 0 .. L = %d' % upto)
    return pdf, lines
