#!/usr/bin/env python3
r"""power_report.py — per-region derivation report for the scalar power counting.

For a chosen list of regions, expand \lambda^{A-B} into a checkable step-by-step derivation: the independent loop momenta (per mode block), the integration-measure power
    A = (2-\epsilon) * sum_X r_X * V(X),
the denominator power B = sum V(em), and (optionally) a numerator polynomial in the edge momenta K_i: its power N is the min over terms of the sum of their factor powers, so power = A - B + N.
The result is rendered as a multi-page monospace PDF.

Called from the interactive browser after "Group these regions by their power".
"""

import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.dirname(HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from power_counting import (measure_power, integrand_power, fmt_power)
from numerator_power import region_context, exact_power, numerator_details, edge_routing_check
from indep_loops import indep_loops, edge_momenta, fmt_expr
from primitives import mode_str, V
from report_pdf import render_pdf

# ================================ HELPERS ================================

# Display name for a mode tuple ('∅' when missing).
def _ms(md):
    return mode_str(md) if md is not None else '∅'

# 'a0 + a1\epsilon' / 'a0 - a1\epsilon' / 'a0': exponent body without the \lambda^.
def _exp_body(a0, a1):
    if a1 == 0:
        return f'{a0}'
    if a1 > 0:
        return f'{a0} + {a1}\\epsilon'
    return f'{a0} - {-a1}\\epsilon'

# ============================ DERIVATION TEXT ============================

# One region's derivation block, as a list of lines.
def region_lines(edges, vm, em, ext_attach, extmode, i, numerator=None, precomputed=None):
    out = []
    results, _total, _L = indep_loops(edges, em, vm)
    a0, a1 = measure_power(results)
    b = integrand_power(em)
    num = None
    n = 0
    if numerator is not None and precomputed is not None:
        ctx, eliminated, terms, n = precomputed
        num = ('no-basis', None, None) if ctx is None else (ctx, eliminated, terms)
    elif numerator is not None:
        ctx = region_context(edges, em, vm, ext_attach, extmode)
        if ctx is None:
            num = ('no-basis', None, None)
        else:
            n, eliminated, terms = numerator_details(numerator[1], ctx, len(edges), list(ext_attach))
            num = (ctx, eliminated, terms)
    out.append(f'R{i}  —  power = {fmt_power(a0 - b + n, a1) if numerator is not None else fmt_power(a0 - b, a1)}')
    out.append(f"  edge modes (input order): {', '.join(_ms(m) for m in em)}")
    vmtxt = ', '.join(f'{v}: {_ms(vm[v])}' for v in sorted(vm, key=str))
    out.append(f'  vertex modes: {{{vmtxt}}}')
    # Independent loop momenta, per mode block (+ the k_i -> line map).
    out.append('-' * 76)
    out.append('  independent loop momenta:')
    basis_all = []
    for r in results:
        if r['rank'] == 0:
            continue
        basis_lines = [j for blk in r['blocks'] for j in blk['basis']]
        basis_all += basis_lines
        desc = ', '.join(f'#{j} ({edges[j][0]},{edges[j][1]})' for j in basis_lines)
        out.append(f'    {_ms(r["mode"])}: rank {r["rank"]} — basis lines {desc}')
    ok, payload, _momenta, _verts = edge_momenta(edges, em, vm, basis_all, ext_attach)
    if ok and payload:
        order = payload
        kmap = ', '.join(f'k{m + 1} ↦ ({edges[ei][0]},{edges[ei][1]})' for m, ei in enumerate(order))
        out.append(f'    {kmap}')
    elif not ok:
        out.append(f'    (line-momentum map unavailable: {payload})')
    # Integration measure: A = (2-eps) * sum_X r_X * V(X), term by term.
    out.append('-' * 76)
    out.append('  integration measure ~ \\lambda^A:')
    out.append('    A = (2-\\epsilon) * Σ_X r_X·V(X)')
    terms = []
    for r in results:
        if r['rank'] == 0:
            continue
        mm, nn = r['mode'][0], r['mode'][1]
        v = 2 * mm + nn
        terms.append((_ms(r['mode']), r['rank'], mm, nn, v, r['rank'] * v))
    for name, rr, mm, nn, v, rv in terms:
        out.append(f'      {name}: r = {rr}, V = 2*{mm}+{nn} = {v}  →  r*V = {rv}')
    tot = sum(t[5] for t in terms)
    out.append(f'      Σ r_X·V(X) = ' + ' + '.join(str(t[5]) for t in terms) + f' = {tot}')
    out.append(f'    A = (2-\\epsilon)*{tot} = {_exp_body(a0, a1)}')
    # Denominator: B = sum V(em), edge by edge.
    out.append('-' * 76)
    out.append('  denominator ~ \\lambda^B:')
    vs = [0 if m is None else 2 * m[0] + m[1] for m in em]
    out.append(f'    B = Σ V(em) = ' + '+'.join(map(str, vs)) + f' = {b}')
    # Numerator (optional): its power N, term by term, then the total power.
    if numerator is not None:
        out.append('-' * 76)
        out.append('  numerator ~ \\lambda^N:')
        if num[0] == 'no-basis':
            out.append('    skipped: no independent loop-momentum basis for this region')
            out.append('*' * 76)
            out.append(f'  power = A - B = {_exp_body(a0 - b, a1)}  (numerator not included)')
        else:
            ctx, eliminated, terms = num
            if eliminated:
                out.append(f'    external eliminated with sum(p)+sum(q)=0: {eliminated}')
            else:
                out.append('    externals rewritten through their own vertex; external part canonicalised with sum(p)=0')
            out.append("    K_i in the basis of this region:")
            for i in sorted(ctx['K_expr']):
                out.append('      K%d = %s' % (i, fmt_expr(ctx['K_expr'][i])))
            bad_route = edge_routing_check(ctx, em, len(edges), list(ext_attach))
            if not bad_route:
                out.append('    routing check (K_i^2 = V(edge mode)): OK for all edges')
            else:
                out.append('    routing check FAILED: ' + '; '.join(
                    'K%d: got %s, expected %s' % (i, g, w) for i, g, w in bad_route))
            out.append(f'    the numerator is a sum of {len(terms)} term(s); each term is a product of factors:')
            for ti, (items, tp) in enumerate(terms, 1):
                out.append(f'      term {ti}   (product of {len(items)} factors)   ->  {fmt_power(tp, 0)}')
                for ttext, tpow in items:
                    note = '   (a sum of terms)' if _is_sum(ttext) else ''
                    out.append(f'        factor: {ttext}   ->  {fmt_power(tpow, 0)}{note}')
            out.append(f'    N = min over the expanded polynomial = {fmt_power(n, 0)}')
            out.append('*' * 76)
            out.append(f'  power = A - B + N = {_exp_body(a0 - b + n, a1)}')
    else:
        out.append('*' * 76)
        out.append(f'  power = A - B = {_exp_body(a0 - b, a1)}')
    return out


# True when the text is (after stripping one pair of outer parentheses) a sum of terms.
def _is_sum(txt):
    t = txt.strip()
    # strip one outer pair only if it really encloses the whole text
    if t.startswith('('):
        depth = 0
        for i, ch in enumerate(t):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    if i == len(t) - 1:
                        t = t[1:-1]
                    break
    depth = 0
    for i, ch in enumerate(t):
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif ch in '+-' and depth == 0 and i > 0:
            return True
    return False


# =============================== TOP LEVEL ===============================

# Build the report for the selected regions (1-based indices) and render it; returns (pdf, preview, lines).
def build_report(edges, regs, sel, ext_attach, internal_lines, externals, extmode=None, numerator=None, outdir=None, details=None):
    lines = []
    lines.append(f'internal lines: {internal_lines}')
    lines.append(f'externals: {externals}')
    if numerator is not None:
        lines.append(f'numerator = {numerator[1]}')
    lines.append('-' * 76)
    if numerator is not None:
        lines.append('numerator handling')
        lines.append('  K_i follow the input edge order and flow from the smaller to the larger vertex:')
        items = []
        for i, (a, b) in enumerate(edges, 1):
            lo, hi = (a, b) if a < b else (b, a)
            items.append('K%d: %d->%d' % (i, lo, hi))
        for k in range(0, len(items), 6):
            lines.append('    ' + '   '.join(items[k:k + 6]))
        lines.append('  every edge momentum flows from the lower- to the higher-numbered vertex and')
        lines.append('  every external momentum is outgoing, so that at each vertex')
        lines.append('      sum(K out) - sum(K in) + p_ext = 0.')
        lines.append('  The polynomial is expanded exactly: every K_i is replaced by its signed')
        lines.append('  combination of k_i and p_j, one hard external is eliminated with')
        lines.append('  sum(p) + sum(q) = 0, and the result is collected in the atoms a.b')
        lines.append('  (a, b single momenta) with integer coefficients.  The power of a monomial')
        lines.append('  is the sum of its atom powers; N is the smallest power with a nonzero')
        lines.append('  coefficient, so cancellations are taken into account.  The per-factor')
        lines.append('  values below are the same quantity for each top-level bracket.')
        lines.append('-' * 76)
    lines.append(f'generated: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('regions in this report: ' + ', '.join(f'R{i}' for i in sel))
    lines.append('')
    live = bool(sys.stdout.isatty() and len(sel) >= 20)
    total_sel = len(sel)
    for k, i in enumerate(sel, 1):
        vm, em = regs[i - 1]
        lines.append('=' * 78)
        lines += region_lines(edges, vm, em, ext_attach, extmode, i, numerator=numerator, precomputed=(details or {}).get(i))
        lines.append('')
        if live:
            print(f'\r    {k}/{total_sel} regions', end='', flush=True)
        elif k % 25 == 0:
            print(f'    {k}/{total_sel} regions', flush=True)
    if live:
        print(f'\r    {total_sel}/{total_sel} regions', flush=True)
    # Output: <outdir>/power_derivation_<timestamp>.pdf (+ first-page PNG preview); outdir=None -> fri_out/.
    fdir = outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fri_out')
    os.makedirs(fdir, exist_ok=True)
    ts = time.strftime('%Y%m%d-%H%M%S')
    pdf = os.path.join(fdir, f'power_derivation_{ts}.pdf')
    prev = pdf.replace('.pdf', '_p1.png')
    render_pdf(lines, pdf, 'FRI power-counting derivation report', 'scalar integrals — per-region derivation of \\lambda^{A-B}', preview=prev)
    with open(pdf.replace('.pdf', '.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return pdf, prev, lines
