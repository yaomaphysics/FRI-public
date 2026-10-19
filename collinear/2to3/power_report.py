#!/usr/bin/env python3
r"""
power_report.py -- per-region derivation report for the five-point 2->3 power counting.

For the selected regions, expand lambda^{A-B} (and A-B+N when a numerator is given) into a
checkable step-by-step derivation: the independent loop momenta per mode block, the measure
A = (2-eps) * sum_X r_X V(X), the denominator B = sum_e V(em_e), and the numerator's power N.

The result is rendered as a multi-page monospace PDF (shared renderer, report_pdf.py).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from primitives import name
from indep_loops import indep_loops, edge_momenta
from power_counting import (measure_power, integrand_power, fmt_power, region_context,
                            term_breakdown, _power_V)
from report_pdf import render_pdf, SEP


# Display name of a mode.
def _ms(md):
    return name(md)


# One region's derivation lines.
def region_lines(edges, vm, em, ext_attach, extmode, i, numerator=None):
    out = []
    results, _total, _L = indep_loops(edges, em, vm)
    a0, a1 = measure_power(results)
    b = integrand_power(em)
    pairs, n = None, 0
    if numerator is not None:
        ctx = region_context(edges, em, vm, ext_attach, extmode)
        pairs, n = term_breakdown(numerator[0], ctx)
    power = fmt_power(a0 - b + n, a1) if numerator is not None else fmt_power(a0 - b, a1)
    out.append('R%d  —  power = %s' % (i, power))
    out.append('  edge modes (input order): %s' % ', '.join(_ms(m) for m in em))
    out.append('  vertex modes: {%s}' % ', '.join('%s: %s' % (v, _ms(vm[v])) for v in sorted(vm, key=str)))
    out.append(SEP)
    out.append('  independent loop momenta:')
    basis_all = []
    for r in results:
        if r['rank'] == 0:
            continue
        basis_lines = [j for blk in r['blocks'] for j in blk['basis']]
        basis_all += basis_lines
        desc = ', '.join('#%d (%s,%s)' % (j, edges[j][0], edges[j][1]) for j in basis_lines)
        out.append('    %s: rank %d — basis lines %s' % (_ms(r['mode']), r['rank'], desc))
    ok, payload, _mom, _verts = edge_momenta(edges, em, vm, basis_all, ext_attach)
    if ok and payload:
        out.append('    ' + ', '.join('k%d ↦ (%s,%s)' % (m + 1, edges[ei][0], edges[ei][1])
                                      for m, ei in enumerate(payload)))
    elif not ok:
        out.append('    (line-momentum map unavailable: %s)' % payload)
    out.append(SEP)
    out.append(r'  integration measure ~ \lambda^A:')
    out.append(r'    A = (2-\epsilon) * Σ_X r_X·V(X)')
    terms = []
    for r in results:
        if r['rank'] == 0:
            continue
        v = _power_V(r['mode'])
        terms.append((_ms(r['mode']), r['rank'], v, r['rank'] * v))
    for nm, rr, v, rv in terms:
        out.append('      %s: r = %d, V = %d  →  r*V = %d' % (nm, rr, v, rv))
    tot = sum(t[3] for t in terms)
    out.append('      Σ r_X·V(X) = %s = %d' % (' + '.join(str(t[3]) for t in terms), tot))
    out.append(r'    A = (2-\epsilon)*%d = %s' % (tot, fmt_power(a0, a1)))
    out.append(SEP)
    out.append(r'  denominator ~ \lambda^B:')
    vs = [0 if m is None else _power_V(m) for m in em]
    out.append('    B = Σ V(em) = %s = %d' % ('+'.join(map(str, vs)), b))
    if numerator is not None:
        out.append(SEP)
        out.append(r'  numerator ~ \lambda^N:')
        for j, (ttext, tp) in enumerate(pairs, 1):
            out.append('    term %d: %s   ->  %s' % (j, ttext, fmt_power(tp, 0)))
        out.append('    N = min = %s' % fmt_power(n, 0))
        out.append('*' * 76)
        out.append('  power = A - B + N = %s' % fmt_power(a0 - b + n, a1))
    else:
        out.append('*' * 76)
        out.append('  power = A - B = %s' % fmt_power(a0 - b, a1))
    return out


# Build the report for the selected regions (1-based indices) and render it; returns (pdf, lines).
def build_report(edges, regs, sel, ext_attach, extmode, numerator=None, outdir=None):
    lines = ['internal lines: %s' % (edges,), 'externals: %s' % (extmode,)]
    if numerator is not None:
        lines.append('numerator = %s' % numerator[1])
    lines.append(SEP)
    lines.append('generated: %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    lines.append('regions in this report: ' + ', '.join('R%d' % i for i in sel))
    lines.append('')
    for i in sel:
        vec, cuts, em, vm = regs[i - 1]
        lines.append('=' * 78)
        lines += region_lines(edges, vm, em, ext_attach, extmode, i, numerator=numerator)
        lines.append('')
    fdir = outdir or os.path.join(HERE, 'fri_out')
    os.makedirs(fdir, exist_ok=True)
    pdf = os.path.join(fdir, 'power_derivation_%s.pdf' % time.strftime('%Y%m%d-%H%M%S'))
    render_pdf(lines, pdf, 'FRI power-counting derivation report',
               r'five-point 2->3 — per-region derivation of \lambda^{A-B}')
    return pdf, lines
