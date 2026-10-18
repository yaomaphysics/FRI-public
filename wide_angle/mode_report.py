#!/usr/bin/env python3
"""
mode_report.py -- derivation report for the wide-angle usable-mode closure.

For one choice of external modes, list every usable mode with the rule that produced it, in
derivation order (seeds, then the vee / wedge / messenger rounds).

The result is rendered as a multi-page monospace PDF (shared renderer, report_pdf.py).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.dirname(HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from primitives import mode_str
from usable_modes import derive_usable_modes, usable_layers
from report_pdf import render_pdf, SEP


# One line for a mode: the rule that produced it.
def explain(md, route, ext_mode):
    r = route.get(md)
    if r is None:
        return '—'
    kind = r[0]
    if kind == 'seed':
        names = [nm for nm, m in ext_mode.items() if m == md]
        who = names[0] if len(names) == 1 else ', '.join(names)
        noun = 'momentum' if len(names) == 1 else 'momenta'
        return 'from the external %s %s directly' % (noun, who)
    if kind == 'vee':
        x, y = mode_str(r[1]), mode_str(r[2])
        return 'from the "joint" mode of %s and %s (X = %s ∨ %s)' % (x, y, x, y)
    if kind == 'wedge':
        x, y = mode_str(r[1]), mode_str(r[2])
        return 'from the "intermediating" mode between %s and %s (X = %s ∧ %s)' % (x, y, x, y)
    if kind == 'messenger':
        return 'messenger S^%d, %d eligible directions: %s' % (r[1], r[2], ', '.join(str(i) for i in r[3]))
    return kind


# Build the report for one choice of external modes and render it; returns (pdf, lines).
def build_report(ext_mode, kappa, verts=None, outdir=None):
    usable, order, route = derive_usable_modes(ext_mode, kappa, trace=True)
    layers = usable_layers(ext_mode, kappa, usable)
    head = []
    for nm, md in ext_mode.items():
        v = (verts or {}).get(nm)
        head.append('%s = %s%s' % (nm, mode_str(md), '' if v is None else ' (vertex %s)' % v))
    lines = ['external modes: ' + ', '.join(head),
             'kappa = %d   usable modes = %d' % (kappa, len(usable)),
             'generated: ' + time.strftime('%Y-%m-%d %H:%M:%S'),
             '', "(wedge and vee algebra: see section 3.2 of 2601.22144)",
             '', 'Modes at the amplitude level:']
    for md in order:
        lines.append('   %s — %s' % (mode_str(md), explain(md, route, ext_mode)))
    lines += ['', SEP, 'cuts modes (used in the enumeration step):']
    cut_modes = []
    for nm, md in ext_mode.items():
        m, _n, i = md
        for k in layers.get(nm) or []:
            cut_modes.append((m, k, i) if m >= 1 else (0, k, i))
    lines.append('   ' + ', '.join(mode_str(x) for x in cut_modes))
    lines.append('')
    fdir = outdir or os.path.join(HERE, 'fri_out')
    os.makedirs(fdir, exist_ok=True)
    pdf = os.path.join(fdir, 'usable_modes_%s.pdf' % time.strftime('%Y%m%d-%H%M%S'))
    render_pdf(lines, pdf, 'FRI usable-mode report', 'wide-angle kinematics — infrared-compatible modes and their cut layers')
    return pdf, lines
