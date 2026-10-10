#!/usr/bin/env python3
r"""power_report.py — per-region derivation report for the scalar power counting.

For a chosen list of regions, expand \lambda^{A-B} into a checkable step-by-step derivation: the independent loop momenta (per mode block), the integration-measure power
    A = (2-\epsilon) * sum_X r_X * V(X),
the denominator power B = sum V(em), and (optionally) a numerator polynomial in the edge momenta K_i: its power N is the min over terms of the sum of their factor powers, so power = A - B + N.
The result is rendered as a multi-page monospace PDF.

Called from the interactive browser after "Group these regions by their power".
"""

import os, time
from PIL import Image, ImageDraw, ImageFont

from power_counting import (measure_power, integrand_power, fmt_power, region_context, term_breakdown)
from indep_loops import indep_loops, edge_momenta
from primitives import mode_str

# Page geometry: A4 at 150 dpi.
W, H = 1240, 1754
MARGIN = 64
LH = 21
BG, FG, GRAY = (255, 255, 255), (17, 17, 17), (140, 140, 140)


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
def region_lines(edges, vm, em, ext_attach, extmode, i, numerator=None):
    out = []
    results, _total, _L = indep_loops(edges, em, vm)
    a0, a1 = measure_power(results)
    b = integrand_power(em)
    out.append(f'R{i}  —  power = {fmt_power(a0 - b, a1)}')
    out.append(f"  edge modes (input order): {', '.join(_ms(m) for m in em)}")
    vmtxt = ', '.join(f'{v}: {_ms(vm[v])}' for v in sorted(vm, key=str))
    out.append(f'  vertex modes: {{{vmtxt}}}')
    # Independent loop momenta, per mode block (+ the k_i -> line map).
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
    out.append('  denominator ~ \\lambda^B:')
    vs = [0 if m is None else 2 * m[0] + m[1] for m in em]
    out.append(f'    B = Σ V(em) = ' + '+'.join(map(str, vs)) + f' = {b}')
    # Numerator (optional): its power N, term by term, then the total power.
    n = 0
    if numerator is not None:
        ast = numerator[0]
        ctx = region_context(edges, em, vm, ext_attach, extmode)
        pairs, n = term_breakdown(ast, ctx)
        out.append('  numerator ~ \\lambda^N:')
        for j, (ttext, tp) in enumerate(pairs, 1):
            out.append(f'    term {j}: {ttext}   ->  {fmt_power(tp, 0)}')
        out.append(f'    N = min = {fmt_power(n, 0)}')
        out.append(f'  power = A - B + N = {_exp_body(a0 - b + n, a1)}')
    else:
        out.append(f'  power = A - B = {_exp_body(a0 - b, a1)}')
    return out

# =============================== RENDERER ================================

# Render a list of text lines as a multi-page monospace PDF (+ first-page PNG).
def render_pdf(lines, outpath, title, subtitle, preview=None):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 15)
    font_b = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 15)
    font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 11)
    adv = font.getlength('M')
    cols = int((W - 2 * MARGIN) / adv)
    rows = int((H - 2 * MARGIN - 90) / LH) - 2
    # wrap long lines
    wrapped = []
    for ln in lines:
        s = ln
        while len(s) > cols:
            wrapped.append(s[:cols])
            s = s[cols:]
        wrapped.append(s)
    pages = [wrapped[i:i + rows] for i in range(0, len(wrapped), rows)] or [[]]
    imgs = []
    total = len(pages)
    for pi, plines in enumerate(pages, 1):
        img = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(img)
        y = MARGIN
        if pi == 1:
            d.text((MARGIN, y), title, font=font_b, fill=FG)
            y += LH + 6
            d.text((MARGIN, y), subtitle, font=font_s, fill=GRAY)
            y += LH + 12
            d.line([(MARGIN, y), (W - MARGIN, y)], fill=(210, 210, 210), width=1)
            y += 14
        for ln in plines:
            d.text((MARGIN, y), ln, font=font, fill=FG)
            y += LH
        d.text((W - MARGIN - 60, H - MARGIN + 10), f'{pi} / {total}', font=font_s, fill=GRAY)
        imgs.append(img)
    # Save the first page as a PNG preview (optional), then all pages into one PDF.
    if preview:
        imgs[0].save(preview)
    imgs[0].save(outpath, 'PDF', save_all=True, append_images=imgs[1:], resolution=150.0)
    return outpath

# =============================== TOP LEVEL ===============================

# Build the report for the selected regions (1-based indices) and render it; returns (pdf, preview, lines).
def build_report(edges, regs, sel, ext_attach, internal_lines, externals, extmode=None, numerator=None, outdir=None):
    lines = []
    lines.append(f'internal lines: {internal_lines}')
    lines.append(f'externals: {externals}')
    if numerator is not None:
        lines.append(f'numerator: {numerator[1]}')
    lines.append(f'generated: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('regions in this report: ' + ', '.join(f'R{i}' for i in sel))
    lines.append('')
    for i in sel:
        vm, em = regs[i - 1]
        lines += region_lines(edges, vm, em, ext_attach, extmode, i, numerator=numerator)
        lines.append('')
    # Output: <outdir>/power_derivation_<timestamp>.pdf (+ first-page PNG preview); outdir=None -> fri_out/.
    fdir = outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fri_out')
    os.makedirs(fdir, exist_ok=True)
    ts = time.strftime('%Y%m%d-%H%M%S')
    pdf = os.path.join(fdir, f'power_derivation_{ts}.pdf')
    prev = os.path.join(fdir, f'power_derivation_{ts}_page1.png')
    render_pdf(lines, pdf, 'FRI power-counting derivation report',
               'scalar integrals — per-region derivation of \\lambda^{A-B}', preview=prev)
    return pdf, prev, lines
