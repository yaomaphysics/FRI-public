#!/usr/bin/env python3
"""region_plot_wa.py — render wide-angle FRI regions as PNG figures with per-mode colours.

Style spec (final, 2026-09-15) — wide-angle edition:

  colours  (directions: C_1 -> 1, C_2 -> 2, C_3 -> 3, C_4 -> 4)
      H                                           Blue
      direction 1    (C_1, C_1^2, ...)            Green
      direction 2    (C_2, C_2^2, ...)            DarkGreen
      direction 3    (C_3, C_3^2, ...)            D12 = Hue[11/24, 0.92, 0.60]
      direction 4    (C_4, C_4^2, ...)            D6  = Hue[ 5/24, 0.92, 0.60]
      S              Magenta
      S^2            Red
      S^k x carrier  (S^k C_i^n, k>=1, n>=1)      Orange (k = 1) / Pink (k >= 2)
                                                  — direction-blind (by design)
  (there is no G / sH in the wide-angle case)

  geometry & captions: identical to the regge edition (region_plot.py):
      solid edge thickness 0.0055; vertex radius 0.010, no outlines; vertex
      numbers grey (FontSize 11) offset 0.026 outward; external legs
      (p1..p4) stub + label in the external-mode colour; caption
      "R{label}:  v = (...)" + one colour->modes line per colour (modes
      displayed via read_graph.mode_str(); merged modes listed together).

Rendering runs through `wolframscript` (must be on PATH).

API:
    render_regions(edges, verts, items, ext_mode=None, ext_attach=None,
                   outdir=None, image_size=720) -> [png paths]
        items = [(label, region), ...] with region = (vm, em) — a wide-angle
        region as returned by the wide-angle browser's enumerate_regions();
        em/vm entries are mode tuples (m, n, i) of read_graph.

    render_atlas(edges, verts, items, ext_mode=None, ext_attach=None,
                 outdir=None, nrows=5, font=None, title='wide-angle')
            -> atlas PDF path
        one A4 PDF: each row = region figure (left) + "R{n} v = (...)" and
        the per-mode structure (right); mode names / the vector v are
        typeset (italic serif letters, upright digits, sub/superscripts)
        with ATLAS_FONT (= a LaTeX-like rendering inside the Wolfram
        pipeline; no TeX needed).  The text block is centred on the
        figure's content centre line.  Runs check_font_fidelity() first
        (guards against broken font encodings — e.g. the 'Nimbus Roman'
        -34 shift: H -> &, S -> 1, C -> !).
"""
import os
import shutil
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from read_graph import mode_str as mode_name

EDGE_T = '0.0055'
R_VERT = '0.010'
DEFAULT_EXT = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4}
EXT_LEG_BLUE = 'RGBColor[0.05, 0.35, 0.8]'
EXT_LABEL_BLUE = 'RGBColor[0.02, 0.25, 0.7]'

COLOR_ORDER = ['Blue', 'Green', 'DarkGreen', 'D12', 'D6', 'Magenta',
               'Orange', 'Red', 'Pink', 'DarkYellow', 'DarkRed']


def mode_color(x):
    """(color-name, wl-directive) for one wide-angle mode tuple (m, n, i)."""
    if x is None:
        x = (0, 0, 0)
    m, n, i = x
    if m == 0 and n == 0:          # H
        return 'Blue', 'Blue'
    if m >= 1 and n == 0:          # pure soft family S^k
        return ('Magenta', 'Magenta') if m == 1 else ('Red', 'Red')
    if m >= 1:                     # S^k x carrier — direction-blind
        return ('Orange', 'Orange') if m == 1 else ('Pink', 'Pink')
    # C_i^n — coloured by direction
    if i == 1:
        return 'Green', 'Green'
    if i == 2:
        return 'DarkGreen', 'DarkGreen'
    if i == 3:
        return 'D12', 'Hue[11/24., 0.92, 0.60]'
    if i == 4:
        return 'D6', 'Hue[5/24., 0.92, 0.60]'
    return 'Black', 'Black'


def edge_directive(x):
    """(style, thickness) for one edge — all wide-angle edges are solid."""
    return mode_color(x)[1], EDGE_T


def vertex_directive(x):
    """(fill, edgeform) for one vertex — plain fill, no outline."""
    return mode_color(x)[1], 'None'


def make_caption_lines(em, vm):
    """Caption block: one line per colour, listing all modes it covers."""
    def _nm(m):
        return mode_name(m if m is not None else (0, 0, 0))
    by = {}
    for m in sorted(set(em) | set(vm.values()), key=_nm):
        c, _ = mode_color(m)
        by.setdefault(c, []).append(m)
    parts = []
    for c in COLOR_ORDER:
        if c in by:
            parts.append('%s: %s' % (c, ', '.join(_nm(m) for m in by[c])))
    lines, cur = [], ''
    for p in parts:
        if not cur:
            cur = p
        elif len(cur) + len(p) + 3 <= 78:
            cur = cur + ' | ' + p
        else:
            lines.append(cur)
            cur = p
    if cur:
        lines.append(cur)
    return lines


def _esc(s):
    """Escape a literal string for inclusion in WL source.  Backslashes and
    quotes are escaped; non-ASCII characters are converted to WL unicode
    escapes (\\:hhhh) — raw non-ASCII bytes in .wls files are mis-decoded
    by this pipeline (render mojibake, e.g. the 'â' artifacts)."""
    s = s.replace('\\', '\\\\').replace('"', '\\"')
    return ''.join(ch if ord(ch) < 128 else '\\:%04x' % ord(ch) for ch in s)


def scaling_of(md):
    """v_e = -(2m + n)  (x_e ~ λ^{v_e} with λ the expansion parameter);
    H -> 0 (same convention as the wide-angle browser)."""
    if md is None:
        md = (0, 0, 0)
    return -(2 * md[0] + md[1])


def split_region(r):
    """(vec, em, vm) from a wide-angle region (vm, em); vec = scaling
    vector with the trailing 1 (t1 power, pySecDec style)."""
    vm, em = r
    vec = tuple(scaling_of(m) for m in em) + (1,)
    return vec, em, vm


def wls_preamble(edges, verts, ext_attach):
    """Wolfram-Language lines: seeded layout + the stub[] helper."""
    nv = max(verts)
    edge_str = ', '.join('{%d, %d}' % e for e in edges)
    leg_str = ', '.join('{"%s", %d}' % (k, v)
                        for k, v in sorted(ext_attach.items()))
    wl = []
    wl.append('SeedRandom[42];')
    wl.append('edgesL = {%s};' % edge_str)
    wl.append('extL = {%s};' % leg_str)
    wl.append('nv = %d;' % nv)
    wl.append('rvert = %s;' % R_VERT)
    wl.append('legV = extL[[All, 2]];')
    wl.append('legA = extL[[All, 1]];')
    wl.append('allV = Join[Range[nv], legV];')
    wl.append('edgePairs = Join[edgesL, Table[{legV[[i]], legA[[i]]}, '
              '{i, Length[extL]}]];')
    wl.append('g = Graph[allV, UndirectedEdge @@@ edgePairs, '
              'GraphLayout -> "SpringElectricalEmbedding"];')
    wl.append('vl = VertexList[g];')
    wl.append('emb = GraphEmbedding[g];')
    wl.append('assoc = AssociationThread[vl -> emb];')
    wl.append('xs = emb[[All, 1]]; ys = emb[[All, 2]];')
    wl.append('minx = Min[xs]; maxx = Max[xs]; miny = Min[ys]; maxy = Max[ys];')
    wl.append('sc0 = 0.60/Max[maxx - minx, maxy - miny];')
    wl.append('ncoord = AssociationMap[{0.185 + (assoc[#][[1]] - minx) sc0, '
              '0.235 + (assoc[#][[2]] - miny) sc0}&, vl];')
    wl.append('cx = Mean[Table[ncoord[v][[1]], {v, allV}]];')
    wl.append('cy = Mean[Table[ncoord[v][[2]], {v, allV}]];')
    wl.append('stub[v_] := ncoord[v] + Normalize[ncoord[v] - {cx, cy} + '
              '{1.*^-6, 0}] (rvert + 0.095);')
    return wl


def panel_lines(pos, label, vec, em, vm, edges, outpath, ext_attach,
                ext_mode=None, image_size=720):
    """Wolfram-Language lines for one region panel plus its Export."""
    verts = sorted({v for e in edges for v in e})
    items = []
    for i, (a, b) in enumerate(edges):
        style, th = edge_directive(em[i])
        items.append('{%s, Thickness[%s], Line[{ncoord[%d], ncoord[%d]}]}'
                     % (style, th, a, b))
    for legname, vertex in sorted(ext_attach.items()):
        if ext_mode and legname in ext_mode:
            _, wlcol = mode_color(ext_mode[legname])
        else:
            wlcol = EXT_LEG_BLUE
        items.append('{%s, Thickness[0.0045], Line[{ncoord[%d], stub[%d]}]}'
                     % (wlcol, vertex, vertex))
    for v in verts:
        face, ef = vertex_directive(vm[v])
        items.append('{EdgeForm[%s], %s, Disk[ncoord[%d], rvert]}'
                     % (ef, face, v))
    for v in verts:
        items.append('{GrayLevel[0.1], Text[Style[ToString[%d], '
                     'FontSize -> 11], ncoord[%d] + Normalize[ncoord[%d] - '
                     '{cx, cy} + {1.*^-6, 0}] 0.026]}' % (v, v, v))
    for legname, vertex in sorted(ext_attach.items()):
        if ext_mode and legname in ext_mode:
            _, wlcol = mode_color(ext_mode[legname])
        else:
            wlcol = EXT_LABEL_BLUE
        items.append('{%s, Text[Style["%s", Bold, FontSize -> 12], stub[%d] + '
                     'Normalize[stub[%d] - ncoord[%d]] 0.030]}'
                     % (wlcol, legname, vertex, vertex, vertex))
    cap1 = 'R%d:  v = (%s)' % (label, ', '.join(str(x) for x in vec))
    caplines = make_caption_lines(em, vm)
    items.append('{GrayLevel[0.1], Text[Style["%s", FontSize -> 11.5], '
                 '{0.020, 0.105}, {-1, 0}]}' % _esc(cap1))
    for j, ln in enumerate(caplines[:3]):
        items.append('{GrayLevel[0.3], Text[Style["%s", FontSize -> 9.5], '
                     '{0.020, %s}, {-1, 0}]}' % (_esc(ln), 0.070 - j * 0.032))
    return ['panel%d = Graphics[{%s}, PlotRange -> {{0, 1}, {0, 1}}, '
            'AspectRatio -> 1, ImageSize -> %d];'
            % (pos, ', '.join(items), image_size),
            'Export["%s", panel%d];' % (outpath, pos)]


def render_regions(edges, verts, items, ext_mode=None, ext_attach=None,
                   outdir=None, image_size=720, verbose=False):
    """Render [(label, region), ...] to PNG files; returns the paths."""
    if ext_attach is None:
        ext_attach = dict(DEFAULT_EXT)
    if outdir is None:
        outdir = os.path.join(BASE, 'fri_out',
                              time.strftime('regions_%Y%m%d-%H%M%S'))
    os.makedirs(outdir, exist_ok=True)
    wl = wls_preamble(edges, verts, ext_attach)
    paths = []
    for pos, (label, region) in enumerate(items, 1):
        vec, em, vm = split_region(region)
        outpath = os.path.join(outdir, 'r%02d.png' % label)
        wl += panel_lines(pos, label, vec, em, vm, edges, outpath,
                          ext_attach, ext_mode, image_size)
        paths.append(outpath)
    if not paths:
        return []
    wls_path = os.path.join(outdir, '_region_plot.wls')
    with open(wls_path, 'w') as f:
        f.write('\n'.join(wl) + '\n')
    r = subprocess.run(['wolframscript', '-file', wls_path],
                       capture_output=True, text=True, timeout=900)
    missing = [p for p in paths if not os.path.exists(p)]
    if r.returncode != 0 or missing:
        raise RuntimeError('wolframscript rendering failed (wls: %s)\n%s%s'
                           % (wls_path, r.stdout[-400:], r.stderr[-400:]))
    if verbose:
        print(r.stdout[-600:])
    return paths


# ---------------------------------------------------------------- PDF atlas
# Region atlas: A4 portrait, `nrows` rows/page; each row shows the region
# figure on the left and, on the right, "R{n} v = (...)" plus one line per
# mode ("<typeset mode>: v{...} e{...}").  Pages are exported one by one
# and merged with ghostscript (fallback: pdfunite).

ATLAS_FONT = 'Utopia'
ATLAS_ROWS = 5
ATLAS_PAGE = (595, 842)          # A4 in points
ATLAS_MARG = 36
ATLAS_TOP = 812                  # y of the top of the first row
ATLAS_BOTTOM = 22                # bottom margin
ATLAS_VFILL = 146                # drawn-content height (pt) — fills the row
ATLAS_WMAX = 210                 # drawn-content width cap (pt)
ATLAS_LEAD = 12.6
ATLAS_FZ = 9.0
ATLAS_FOLD = 70
MINI = {'edge_t': '0.0072', 'leg_t': '0.0066', 'rv': '0.0130',
        'fz_v': 7.0, 'fz_leg': 7.5, 'voff': '0.032', 'loff': '0.034'}


def ts(s, font=None, italic=False, bold=False):
    """WL Style[] for a literal string in ATLAS_FONT."""
    font = font or ATLAS_FONT
    opts = 'FontFamily -> "%s"' % font
    if italic:
        opts += ', FontSlant -> Italic'
    if bold:
        opts += ', FontWeight -> Bold'
    return 'Style["%s", %s]' % (_esc(s), opts)


def ts_mode(md, font=None):
    """WL typeset expression for a wide-angle mode tuple (LaTeX-style)."""
    font = font or ATLAS_FONT
    if md is None:
        md = (0, 0, 0)
    m, n, i = md

    def L(s):
        return ts(s, font, italic=True)

    def U(s):
        return ts(s, font)
    if (m, n, i) == (0, 0, 0):
        return L('H')
    if n == 0:                      # pure soft family S^m
        if m == 1:
            return L('S')
        return 'Superscript[%s, %s]' % (L('S'), U(str(m)))
    nstr = '\u221e' if n >= 100 else str(n)
    if m == 0:                      # C_i^n
        if n == 1:
            return 'Subscript[%s, %s]' % (L('C'), U(str(i)))
        return 'Subsuperscript[%s, %s, %s]' % (L('C'), U(str(i)), U(nstr))
    mpart = 'Superscript[%s, %s]' % (L('S'), U(str(m)))
    if n == 1:
        cpart = 'Subscript[%s, %s]' % (L('C'), U(str(i)))
    else:
        cpart = 'Subsuperscript[%s, %s, %s]' % (L('C'), U(str(i)), U(nstr))
    return 'Row[{%s, %s}]' % (mpart, cpart)


def fold_lines(s, width=ATLAS_FOLD):
    out, cur = [], ''
    for piece in s.split(', '):
        cand = piece if not cur else cur + ', ' + piece
        if len(cand) <= width:
            cur = cand
        else:
            if cur:
                out.append(cur)
            cur = piece
            while len(cur) > width:
                out.append(cur[:width])
                cur = cur[width:]
    if cur:
        out.append(cur)
    return out


def mode_key(m):
    m0, n, i = m if m is not None else (0, 0, 0)
    if (m0, n, i) == (0, 0, 0):
        return (2,)
    if m0 == 0:
        return (0, n, i)
    return (1, m0, n, i)


def mini_fig_expr(vm, em, edges, ext_attach, ext_mode):
    """WL Graphics[...] for one row's mini figure (all fonts absolute pt)."""
    items = []
    for i, (a, b) in enumerate(edges):
        style, th = edge_directive(em[i])
        items.append('{%s, Thickness[%s], Line[{ncoord[%d], ncoord[%d]}]}'
                     % (style, MINI['edge_t'], a, b))
    for legname, v in sorted(ext_attach.items()):
        if ext_mode and legname in ext_mode:
            _, col = mode_color(ext_mode[legname])
        else:
            col = EXT_LEG_BLUE
        items.append('{%s, Thickness[%s], Line[{ncoord[%d], stub[%d]}]}'
                     % (col, MINI['leg_t'], v, v))
    for v in sorted(vm):
        face, ef = vertex_directive(vm[v])
        items.append('{EdgeForm[%s], %s, Disk[ncoord[%d], %s]}'
                     % (ef, face, v, MINI['rv']))
    for v in sorted(vm):
        items.append('{GrayLevel[0.1], Text[Style[ToString[%d], '
                     'FontSize -> %s], ncoord[%d] + Normalize[ncoord[%d] - '
                     '{cx, cy} + {1.*^-6, 0}] %s]}'
                     % (v, MINI['fz_v'], v, v, MINI['voff']))
    for legname, v in sorted(ext_attach.items()):
        if ext_mode and legname in ext_mode:
            _, col = mode_color(ext_mode[legname])
        else:
            col = EXT_LABEL_BLUE
        items.append('{%s, Text[Style["%s", Bold, FontSize -> %s], stub[%d] + '
                     'Normalize[stub[%d] - ncoord[%d]] %s]}'
                     % (col, legname, MINI['fz_leg'], v, v, v, MINI['loff']))
    return 'Graphics[{%s}, PlotRange -> {{0, 1}, {0, 1}}]' % ', '.join(items)


def atlas_row_exprs(k, r, edges, ext_attach, font=None):
    """WL expressions for the right-hand text lines of one row."""
    vec, em, vm = split_region(r)
    vecstr = ', '.join(str(x) for x in vec)
    out = ['Row[{"R%d   ", %s, " = (%s)"}]'
           % (k, ts('v', font, italic=True, bold=True), _esc(vecstr))]
    per = {}
    for i, (a, b) in enumerate(edges):
        per.setdefault(em[i], {'v': [], 'e': []})['e'].append((a, b))
    for v, md in vm.items():
        per.setdefault(md, {'v': [], 'e': []})['v'].append(v)

    def nm(m):
        return mode_name(m) if m is not None else 'H'

    for md in sorted(per, key=mode_key):
        d = per[md]
        vs = 'v{' + ','.join(str(x) for x in sorted(d['v'])) + '}' \
            if d['v'] else ''
        es = 'e{' + ','.join('[%d,%d]' % e for e in d['e']) + '}' \
            if d['e'] else ''
        tail = ': ' + (vs + ' ' + es).strip()
        for j, piece in enumerate(fold_lines(tail)):
            if j == 0:
                # Colour each mode name exactly like the mode is coloured
                # in the figure (small change requested 2026-09-18).
                pre = 'Style[%s, FontColor -> %s]' % (
                    ts_mode(md, font), mode_color(md)[1])
            else:
                pre = '"' + ' ' * 4 + '"'
            out.append('Row[{%s, "%s"}]' % (pre, _esc(piece)))
    return out


def atlas_page_expr(page_no, npages, rows, first, last, edges, ext_attach,
                    ext_mode, nrows, font, title):
    """WL Graphics[...] for one atlas page (A4).

    The mini figure is placed by its content bounding box (computed in WL
    as bcx/bcy/bbw/bbh/scF): the drawn content is scaled by scF and its
    box centre aligned to the row centre, so only the drawn region (not
    the empty canvas margins) defines the visual spacing."""
    W, H = ATLAS_PAGE
    marg, top = ATLAS_MARG, ATLAS_TOP
    rowh = (top - ATLAS_BOTTOM) / nrows
    tx = '(%d + scF*bbw + 16)' % marg
    it = ['Text[Style["%s - region atlas - page %d/%d  '
          '(regions %d-%d)", 8.5], {%d, %d}, {-1, 0}]'
          % (title, page_no, npages, first, last, marg, H - 16)]
    for ridx, (k, r) in enumerate(rows):
        ytop = top - ridx * rowh
        yc = round(ytop - rowh / 2, 1)
        vec, em, vm = split_region(r)
        it.append('Inset[%s, {%d + scF*bbw/2.0, %s}, {bcx, bcy}, '
                  '{scF, scF}]'
                  % (mini_fig_expr(vm, em, edges, ext_attach, ext_mode),
                     marg, yc))
        lines = atlas_row_exprs(k, r, edges, ext_attach, font)
        n = len(lines)
        for j, expr in enumerate(lines):
            y = round(yc + (n - 1) * ATLAS_LEAD / 2 - j * ATLAS_LEAD, 1)
            it.append('Text[Style[%s, FontSize -> %s], {%s, %s}, {-1, 0}]'
                      % (expr, ATLAS_FZ, tx, y))
        if ridx < len(rows) - 1:
            it.append('{GrayLevel[0.9], AbsoluteThickness[0.6], Line[{{%d, %s}, '
                      '{%d, %s}}]}' % (marg, round(ytop - rowh, 1),
                                       W - marg, round(ytop - rowh, 1)))
    return ('Graphics[{%s}, PlotRange -> {{0, %d}, {0, %d}}, ImageSize -> 826, '
            'AspectRatio -> %d/%d]' % (', '.join(it), W, H, H, W))


def check_font_fidelity(font=None, workdir=None):
    """Guard against broken font encodings: render a small probe in the
    fonts/styles the atlas uses and verify that pdftotext extracts exactly
    the input characters.  (Broken example: 'Nimbus Roman' renders with a
    constant -34 shift — H -> &, S -> 1, C -> !.)  Raises RuntimeError on
    mismatch; returns the extracted text."""
    font = font or ATLAS_FONT
    if workdir is None:
        workdir = os.path.join(BASE, 'fri_out', 'font_fidelity')
    os.makedirs(workdir, exist_ok=True)
    probe = [
        'Text[Style["HSCe", FontSize -> 30], {40, 700}, {-1, 0}]',
        'Text[Style["0123456789", FontSize -> 30], {40, 640}, {-1, 0}]',
        'Text[Style["&", FontSize -> 30], {40, 580}, {-1, 0}]',
        'Text[Style["HSCv", FontSize -> 30, FontFamily -> "%s", '
        'FontSlant -> Italic], {40, 480}, {-1, 0}]' % font,
        'Text[Style["vv", FontSize -> 30, FontFamily -> "%s", '
        'FontSlant -> Italic, FontWeight -> Bold], {40, 420}, {-1, 0}]'
        % font,
        'Text[Style["9876543210", FontSize -> 30, FontFamily -> "%s"], '
        '{40, 360}, {-1, 0}]' % font,
        'Text[Style["%s", FontSize -> 30, FontFamily -> "%s"], '
        '{40, 300}, {-1, 0}]' % (_esc('\u221e'), font),
    ]
    page = ('Graphics[{%s}, PlotRange -> {{0, 595}, {0, 842}}, '
            'ImageSize -> 826, AspectRatio -> 842/595]'
            % ', '.join(probe))
    pdf = os.path.join(workdir, 'font_check.pdf')
    wls = os.path.join(workdir, '_font_check.wls')
    with open(wls, 'w') as f:
        f.write('Export["%s", %s];\n' % (pdf, page))
    r = subprocess.run(['wolframscript', '-file', wls], capture_output=True,
                       text=True, timeout=300)
    if r.returncode != 0 or not os.path.exists(pdf):
        raise RuntimeError('font fidelity probe failed to render: %s%s'
                           % (r.stdout[-300:], r.stderr[-300:]))
    ext = subprocess.run(['pdftotext', pdf, '-'], capture_output=True,
                         text=True).stdout
    ext_n = ' '.join(ext.split())
    missing = [t for t in ['HSCe', '0123456789', '&', 'HSCv', 'vv',
                           '9876543210', '\u221e'] if t not in ext_n]
    if missing:
        raise RuntimeError(
            'font fidelity check FAILED for %r: missing %r in extraction %r '
            '(broken font encoding?)' % (font, missing, ext_n[:200]))
    return ext_n


def render_atlas(edges, verts, items, ext_mode=None, ext_attach=None,
                 outdir=None, nrows=ATLAS_ROWS, font=None, title='wide-angle',
                 verbose=False):
    """Render [(label, region), ...] as a single PDF atlas (A4, nrows/page);
    returns the merged PDF path.  Runs a font fidelity check first."""
    font = font or ATLAS_FONT
    if ext_attach is None:
        ext_attach = dict(DEFAULT_EXT)
    if outdir is None:
        outdir = os.path.join(BASE, 'fri_out',
                              time.strftime('regions_%Y%m%d-%H%M%S'))
    os.makedirs(outdir, exist_ok=True)
    check_font_fidelity(font, workdir=os.path.join(outdir, '_font_check'))
    npages = (len(items) + nrows - 1) // nrows
    wl = wls_preamble(edges, verts, ext_attach)
    wl.append('padG = 0.022;')
    wl.append('pts = Flatten[Table[{ncoord[v], stub[v], stub[v] + '
              'Normalize[stub[v] - ncoord[v]]*0.030}, {v, allV}], 1];')
    wl.append('bx0 = Min[pts[[All, 1]]] - padG; bx1 = Max[pts[[All, 1]]] + padG;')
    wl.append('by0 = Min[pts[[All, 2]]] - padG; by1 = Max[pts[[All, 2]]] + padG;')
    wl.append('bbw = bx1 - bx0; bbh = by1 - by0;')
    wl.append('scF = Min[%d/bbh, %d/bbw];' % (ATLAS_VFILL, ATLAS_WMAX))
    wl.append('bcx = (bx0 + bx1)/2; bcy = (by0 + by1)/2;')
    for p in range(npages):
        chunk = items[p * nrows:(p + 1) * nrows]
        wl.append('Export["%s/page_%02d.pdf", %s];'
                  % (outdir, p + 1,
                     atlas_page_expr(p + 1, npages, chunk, chunk[0][0],
                                     chunk[-1][0], edges, ext_attach,
                                     ext_mode, nrows, font, title)))
    wl.append('Export["%s/preview_p01.png", %s];'
              % (outdir, atlas_page_expr(1, npages, items[:nrows],
                                        items[0][0],
                                        items[min(nrows,
                                                  len(items)) - 1][0],
                                        edges, ext_attach, ext_mode,
                                        nrows, font, title)))
    wls_path = os.path.join(outdir, '_atlas.wls')
    with open(wls_path, 'w') as f:
        f.write('\n'.join(wl) + '\n')
    r = subprocess.run(['wolframscript', '-file', wls_path],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError('atlas rendering failed (wls: %s)\n%s%s'
                           % (wls_path, r.stdout[-400:], r.stderr[-400:]))
    pages = sorted(f for f in os.listdir(outdir) if f.startswith('page_')
                   and f.endswith('.pdf'))
    page_files = [os.path.join(outdir, f) for f in pages]
    atlas = os.path.join(outdir, 'atlas.pdf')
    if shutil.which('gs'):
        subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH',
                        '-sDEVICE=pdfwrite', '-sOutputFile=' + atlas]
                       + page_files, check=True)
    else:
        subprocess.run(['pdfunite'] + page_files + [atlas], check=True)
    for f in page_files:
        os.remove(f)
    if verbose:
        print(r.stdout[-600:])
    return atlas
