#!/usr/bin/env python3
"""
report_pdf.py -- shared monospace PDF renderer for the mode and power-counting views.

Both sides render the same A4 monospace PDF and the same per-level block; only the wording and the
per-mode explanation differ, so those stay in the two mode_report.py modules.
"""
import os
from PIL import Image, ImageDraw, ImageFont

# Page geometry: A4 at 150 dpi.
W, H = 1240, 1754
MARGIN = 64
LH = 21
BG, FG, GRAY = (255, 255, 255), (17, 17, 17), (140, 140, 140)

SEP = '-' * 66
BAR = '=' * 66


# Cost in words.
def cost(c):
    return 'tree-level' if c == 0 else ('1 loop' if c == 1 else '%d loops' % c)


# Level in words for the "first occurs at ..." clause.
def lev(k):
    return 'tree level' if k == 0 else ('one-loop level' if k == 1 else '%d-loop level' % k)


# Indefinite article for a mode name (an 'S...' mode is read "ess", so it takes "an").
def art(name):
    return 'an' if name[:1] == 'S' else 'a'


# One level block: header, the new names, the cumulative line, then the explanations.
def emit_level(lines, k, names, noun, cum, explain):
    n = len(names)
    if names:
        lines.append('L=%d — %d new %s%s:' % (k, n, noun, 's' if n != 1 else ''))
        for nm in names:
            lines.append('   %s' % nm)
    else:
        lines.append('L=%d — (no new %ss)' % (k, noun))
    lines.append('   cumulative: %d %s%s (≤L=%d)' % (cum, noun, 's' if cum != 1 else '', k))
    lines.append(SEP)
    if names:
        lines.append('Explanation:')
        for nm in names:
            lines.append('   ' + explain(nm))
    lines.append('')
    lines.append(BAR)
    lines.append('')


# Render the report lines into a multi-page monospace PDF (A4 at 150 dpi).
def render_pdf(lines, outpath, title, subtitle, preview=None):
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
    if preview:
        imgs[0].save(preview)
    imgs[0].save(outpath, 'PDF', save_all=True, append_images=imgs[1:], resolution=150.0)
    return outpath
