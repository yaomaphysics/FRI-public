#!/usr/bin/env python3
"""region_plot23.py — render fri23 (spacelike-collinear 2->3) regions as PNG figures with per-mode colours.

Style spec (final, 2026-09-15) — fri23 edition:

  colours  (directions: p1 -> 1, p4 -> 4, p5 -> 5, the p2/p3 pair -> 23)
      H                                          Blue
      direction 1    (C1, C1^2, ...)             Green
      direction 4    (C4, C4^2, ...)             D12 = Hue[11/24, 0.92, 0.60]
      direction 5    (C5, C5^2, ...)             D6  = Hue[ 5/24, 0.92, 0.60]
      direction 23   (C23, C2C23, C3C23, ...)    DarkGreen
      S              Magenta
      S^2            Red
      S^k x carrier  (SC23, ...)                 Orange (k = 1) / Pink (k >= 2)
                                                 — direction-blind (by design)
  (there is no G / sH in the 2->3 case)

  geometry & captions: identical to the regge edition (region_plot.py):
      solid edge thickness 0.0055; vertex radius 0.010, no outlines; vertex
      numbers grey (FontSize 11) offset 0.026 outward; external legs
      (p1..p5 @ vertices 1..5) stub + label in the external-mode colour;
      caption "R{label}:  v = (...)" + one colour->modes line per colour
      (modes displayed via fri23.name(); merged modes listed together).

Rendering runs through `wolframscript` (must be on PATH).

API:
    render_regions(edges, verts, items, ext_mode=None, ext_attach=None,
                   outdir=None, image_size=720) -> [png paths]
        items = [(label, region), ...] with region = (vec, cuts, em, vm)
        (the fri23.enumerate_regions survivor format); em/vm entries are
        fri23 mode tuples.
"""
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fri23
from fri23 import name as mode_name

EDGE_T = '0.0055'
R_VERT = '0.010'
DEFAULT_EXT = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4, 'p5': 5}
EXT_LEG_BLUE = 'RGBColor[0.05, 0.35, 0.8]'
EXT_LABEL_BLUE = 'RGBColor[0.02, 0.25, 0.7]'

COLOR_ORDER = ['Blue', 'Green', 'DarkGreen', 'D12', 'D6', 'Magenta',
               'Orange', 'Red', 'Pink', 'DarkYellow', 'DarkRed']


def mode_color(x):
    """(color-name, wl-directive) for one fri23 mode tuple."""
    if x[0] == 'H':
        return 'Blue', 'Blue'
    if x[0] == 'S':
        return ('Magenta', 'Magenta') if x[1] == 1 else ('Red', 'Red')
    d, m = x[1], x[4]              # ('C', d, n, mem, m)
    if m >= 1:                     # S^k x carrier — direction-blind
        return ('Orange', 'Orange') if m == 1 else ('Pink', 'Pink')
    if d == 23:
        return 'DarkGreen', 'DarkGreen'
    if d == 1:
        return 'Green', 'Green'
    if d == 4:
        return 'D12', 'Hue[11/24., 0.92, 0.60]'
    if d == 5:
        return 'D6', 'Hue[5/24., 0.92, 0.60]'
    return 'Black', 'Black'


def edge_directive(x):
    """(style, thickness) for one edge — all fri23 edges are solid."""
    return mode_color(x)[1], EDGE_T


def vertex_directive(x):
    """(fill, edgeform) for one vertex — plain fill, no outline."""
    return mode_color(x)[1], 'None'


def make_caption_lines(em, vm):
    """Caption block: one line per colour, listing all modes it covers."""
    by = {}
    for m in sorted(set(em) | set(vm.values()), key=mode_name):
        c, _ = mode_color(m)
        by.setdefault(c, []).append(m)
    parts = []
    for c in COLOR_ORDER:
        if c in by:
            parts.append('%s: %s'
                         % (c, ', '.join(mode_name(m) for m in by[c])))
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
    return s.replace('\\', '\\\\').replace('"', '\\"')


def split_region(r):
    """(vec, em, vm) from a (vec, cuts, em, vm) survivor tuple."""
    vec, _cuts, em, vm = r
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
