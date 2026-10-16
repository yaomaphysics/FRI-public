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
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import region_filter as F

# Page geometry: A4 at 150 dpi.
W, H = 1240, 1754
MARGIN = 64
LH = 21
BG, FG, GRAY = (255, 255, 255), (17, 17, 17), (140, 140, 140)

SEP = '-' * 66
BAR = '=' * 66


# ================================ WORDING ================================

# Cost in words.
def _cost(c):
    return 'tree' if c == 0 else ('1 loop' if c == 1 else '%d loops' % c)


# Level in words for the "first occurs at ..." clause.
def _lev(k):
    return 'tree level' if k == 0 else ('one-loop level' if k == 1 else '%d-loop level' % k)


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
        if name == 'H':
            return head + 'it is the hard mode.'
        return head + 'it is a finite vee closure of the external-momentum modes, at tree level.'
    if kind == 'vee':
        _, a, la, b, lb = r
        return head + ('the join of the %s mode (%s) with the %s mode (%s) is free '
                       '(a vee join does not raise the level).' % (F.name(a), _cost(la), F.name(b), _cost(lb)))
    if kind == 'wedge':
        _, a, b = r
        return head + ('the meet of the %s mode (%s) and the %s mode (%s) forms the next wedge batch, '
                       'one level above the current top (+1 loop).' % (F.name(a), _cost(st.levels.get(a, 0)),
                                                                      F.name(b), _cost(st.levels.get(b, 0))))
    if kind == 'messenger':
        _, lvl, prev_mess, cur, anchored = r
        if anchored is not None:
            return head + ('a tree-level target anchors the messenger at L = %d + 2 = %d, earlier than the '
                           'tower rule max(%d + 2, %d + 1) = %d.' % (anchored, lvl, prev_mess, cur,
                                                                     max(prev_mess + 2, cur + 1)))
        return head + ('the tower step lands at max(previous messenger + 2, current top level + 1) '
                       '= max(%d + 2, %d + 1) = %d loops.' % (prev_mess, cur, lvl))
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
    lines.append('kinematics: ' + ', '.join('p%d = %s' % (i, F.name(m)) for i, m in enumerate(st.momenta, 1)))
    lines.append('report covers L = 0 .. L = %d' % upto)
    lines.append('generated: ' + time.strftime('%Y-%m-%d %H:%M:%S'))
    lines.append('')
    for k in range(upto + 1):
        names = st.modes_at(k)
        if names:
            n = len(names)
            lines.append('L=%d — %d new mode%s:' % (k, n, 's' if n != 1 else ''))
            lines.append('   ' + ', '.join(names))
        else:
            lines.append('L=%d — (no new modes)' % k)
        lines.append('   [%s]' % st.source_line(k))
        lines.append('   cumulative: %d modes (≤L=%d)' % (sum(1 for lv in st.levels.values() if lv <= k), k))
        lines.append(SEP)
        if names:
            lines.append('Explanation:')
            for nm in names:
                lines.append('   ' + explain(st, nm))
        lines.append('')
        lines.append(BAR)
        lines.append('')
    fdir = outdir or os.path.join(HERE, 'fri_out')
    os.makedirs(fdir, exist_ok=True)
    pdf = os.path.join(fdir, 'mode_ladder_%s.pdf' % time.strftime('%Y%m%d-%H%M%S'))
    render_pdf(lines, pdf, 'FRI mode-ladder derivation report',
               'collinear 2->3 kinematics — first appearance of each mode, L = 0 .. L = %d' % upto)
    return pdf, lines
