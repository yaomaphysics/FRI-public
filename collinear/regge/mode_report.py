#!/usr/bin/env python3
"""
mode_report.py -- per-level derivation report for the regge mode ladder.

For one kinematics, restate the ladder over L = 0 .. Lcur as a checkable derivation: for every
edge scaling, the level at which it first occurs and the simplest structure that produces it,
following the mechanisms of mode_levels.py (base / vee / wedge / messenger tower / messenger seat).

The result is rendered as a multi-page monospace PDF.

Called from the ladder stepper at any level ("R" in facet_regions_interactive.py --ladder).
"""
import os
import time
from PIL import Image, ImageDraw, ImageFont

from primitives import to_old, to_mode, FAM_LEGS, INF

# Page geometry: A4 at 150 dpi.
W, H = 1240, 1754
MARGIN = 64
LH = 21
BG, FG, GRAY = (255, 255, 255), (17, 17, 17), (140, 140, 140)

SEP = '-' * 66
BAR = '=' * 66


# ================================= WORDING =================================

# Cost in words.
def _cost(c):
    return 'tree-level' if c == 0 else ('1 loop' if c == 1 else '%d loops' % c)


# Level in words for the "first occurs at ..." clause.
def _lev(k):
    return 'tree level' if k == 0 else ('one-loop level' if k == 1 else '%d-loop level' % k)


# Indefinite article for a mode name (an 'S...' mode is read "ess", so it takes "an").
def _art(name):
    return 'an' if name[:1] == 'S' else 'a'


# p1 / p1 and p3 for a leg list.
def _legs_str(legs):
    return ' and '.join('p%d' % i for i in legs)


# Members of a mode set that reached the cheapest cost, in ladder order.
def _picked(keep):
    return [to_old(y) for y in keep]


# The "needs <targets> component(s) (<cost>) and an <mode> messenger (allowed at this level; +2 loops)" clause.
def _messenger_clause(st, x):
    keep, cbest = st.messenger_targets(x)
    if not keep:
        return 'the simplest structure is the messenger step recorded by the ladder (allowed at this level; +2 loops).'
    tgt = '/'.join(_picked(keep))
    part = ('a %s component (%s)' % (tgt, _cost(cbest)) if len(keep) == 1
            else 'the %s components (%s)' % (tgt, _cost(cbest)))
    name = to_old(x)
    return 'the simplest structure needs %s and %s %s messenger (allowed at this level; +2 loops).' % (
        part, _art(name), name)


# =============================== EXPLANATIONS ===============================

# The "Explanation:" line for one edge scaling, built from its recorded route.
def explain(st, name):
    if name == 'G':
        return 'G: first occurs at tree level; p1-p3 = p4-p2 is of the Glauber mode.'
    if name == 'sH':
        return 'sH: first occurs at one-loop level; the simplest structure is a loop between two Glauber vertices.'
    x = st.by_name.get(name)
    if x is None:
        return '%s: first occurs at tree level.' % name
    r = st.pred.route.get(x)
    head = '%s: first occurs at %s; ' % (name, _lev(st.pred.cost[x]))
    kind = r[0] if r else None
    if kind == 'hard':
        return head + 'p1+p2 = p3+p4 is of the hard mode.'
    if kind == 'ext':
        legs = [x.leg] if x.leg is not None else list(FAM_LEGS[x.fam])
        return head + 'it is the external-momentum mode of %s.' % _legs_str(legs)
    if kind == 'base':
        a, b = FAM_LEGS[x.fam]
        return head + 'the simplest structure is a loop which both p%d and p%d are attached to.' % (a, b)
    if kind == 'vee':
        _, a, ca, b, cb = r
        if a.n == INF or b.n == INF:
            nc, cc, car = (b, cb, a) if a.n == INF else (a, ca, b)
            return head + ('the simplest structure needs the %s component (%s), which joins the %s '
                           'external-momentum mode p%d (+0 loop).' % (to_old(nc), _cost(cc), to_old(car), car.leg))
        return head + 'the simplest structure joins %s %s component (%s) with %s %s component (%s).' % (
            _art(to_old(a)), to_old(a), _cost(ca), _art(to_old(b)), to_old(b), _cost(cb))
    if kind == 'wedge':
        _, a, ca, b, cb = r
        return head + ('the simplest structure needs %s %s component (%s), %s %s component (%s), and %s %s '
                       'propagator adjacent to them both (+1 loop).' % (_art(to_old(a)), to_old(a), _cost(ca),
                                                                        _art(to_old(b)), to_old(b), _cost(cb),
                                                                        _art(name), name))
    if kind in ('messenger', 'messenger2'):
        return head + _messenger_clause(st, x)
    return head[:-2] + '.'


# =============================== REPORT / PDF ===============================

# Render the report lines into a multi-page monospace PDF (A4 at 150 dpi).
def render_pdf(lines, outpath, title, subtitle):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 15)
    font_b = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 15)
    font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 11)
    cols = int((W - 2 * MARGIN) / font.getlength('M'))
    rows = int((H - 2 * MARGIN - 90) / LH) - 2
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
        d.text((W - MARGIN - 60, H - MARGIN + 10), '%d / %d' % (pi, total), font=font_s, fill=GRAY)
        imgs.append(img)
    imgs[0].save(outpath, 'PDF', save_all=True, append_images=imgs[1:], resolution=150.0)
    return outpath


# Build the report for the levels L = 0 .. upto of the stepper's kinematics and render it; returns (pdf, lines).
def build_report(st, upto, outdir=None):
    st.ensure(upto)
    lines = []
    lines.append('kinematics: ' + ', '.join('p%d = %s' % (i, to_old(to_mode(st.ext[i]))) for i in sorted(st.ext)))
    lines.append('report covers L = 0 .. L = %d' % upto)
    lines.append('generated: ' + time.strftime('%Y-%m-%d %H:%M:%S'))
    lines.append('')
    for k in range(upto + 1):
        names = st.modes_at(k)
        if names:
            n = len(names)
            lines.append('L=%d — %d new edge scaling%s:' % (k, n, 's' if n != 1 else ''))
            for nm in names:
                lines.append('   %s' % nm)
        else:
            lines.append('L=%d — (no new edge scalings)' % k)
        lines.append('   cumulative: %d edge scalings (≤L=%d)' % (sum(len(st.modes_at(lv)) for lv in st.levels if lv <= k), k))
        lines.append(SEP)
        if names:
            lines.append('Explanation:')
            for nm in names:
                lines.append('   ' + explain(st, nm))
        lines.append('')
        lines.append(BAR)
        lines.append('')
    fdir = outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fri_out')
    os.makedirs(fdir, exist_ok=True)
    pdf = os.path.join(fdir, 'mode_ladder_%s.pdf' % time.strftime('%Y%m%d-%H%M%S'))
    render_pdf(lines, pdf, 'FRI mode-ladder derivation report',
               'regge kinematics — first appearance of each edge scaling, L = 0 .. L = %d' % upto)
    return pdf, lines
