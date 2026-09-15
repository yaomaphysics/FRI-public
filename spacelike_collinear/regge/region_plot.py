#!/usr/bin/env python3
"""region_plot.py — render FRI regions of a graph as PNG figures with per-mode colours.

Style spec (final, 2026-09-15):

  colours
      H                                         Blue
      direction 1   (C13, C1C13, C3C13, ...)    Green
      direction 2   (C24, C2C24, C4C24, ...)    DarkGreen
      direction 3   (reserved)                  D12  = Hue[11/24, 0.92, 0.60]
      direction 4   (reserved)                  D6   = Hue[ 5/24, 0.92, 0.60]
      S        Magenta
      SC       Orange       (S^k x carrier, k = 1)
      S^2      Red
      S^2C     Pink         (S^k x carrier, k >= 2)
      G        DarkYellow   |   sH   DarkRed

  G (Glauber) specials
      vertices: black ring, Thickness 0.0045, DarkYellow fill
      edges:    Dashing[{0.007, 0.0125}]   ->  dot ~6.7 px, gap ~12 px at 960 px
                (Dashing lengths are fractions of the canvas width)

  geometry (canvas 1 x 1; ImageSize 720 -> 960 px files)
      solid edge thickness 0.0055;  vertex radius 0.010, no outline on any
      non-G vertex;  vertex numbers grey (FontSize 11), offset 0.026 outward;
      external legs: stub + label in the leg's external-mode colour
      (a refinement C_i^n inherits the C_i direction colour).

  captions (below each figure)
      "R{label}:  v = (...)" plus one line per colour used, listing all modes
      it covers, e.g. "Green: C13, C1C13 | DarkGreen: C24 | Orange: S^1C13"
      (modes sharing one colour are listed together).

Rendering runs through `wolframscript` (must be on PATH).

API:
    render_regions(edges, verts, items, ext_mode=None, ext_attach=None,
                   outdir=None, image_size=720, verbose=False) -> [png paths]
        items = [(label, region), ...]; a region is either the 8-tuple from
        fri_regions_full()  (cut13, cut24, cut1, cut3, cut2, cut4, vm, em)
        or (vec, cuts, em, vm).  `label` names the figure ("R{label}") and the
        file (r{label:02d}.png).
        ext_mode = {leg: mode_str} colours the external legs (None -> blue).
"""
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from regge_core import to_scaling

EDGE_T = '0.0055'
R_VERT = '0.010'
DEFAULT_EXT = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4}
EXT_LEG_BLUE = 'RGBColor[0.05, 0.35, 0.8]'
EXT_LABEL_BLUE = 'RGBColor[0.02, 0.25, 0.7]'

COLOR_ORDER = ['Blue', 'Green', 'DarkGreen', 'D12', 'D6', 'Magenta',
               'Orange', 'Red', 'Pink', 'DarkYellow', 'DarkRed']


def mode_color(m):
    """(color-name, wl-directive) for one mode string (regge vocabulary)."""
    if m == 'H':
        return 'Blue', 'Blue'
    if m == 'G':
        return 'DarkYellow', 'DarkYellow'
    if m == 'sH':
        return 'DarkRed', 'DarkRed'
    if m == 'S':
        return 'Magenta', 'Magenta'
    mS = re.match(r'^S\^(\d+)$', m)
    if mS:
        k = int(mS.group(1))
        return ('Magenta', 'Magenta') if k == 1 else ('Red', 'Red')
    mSC = re.match(r'^S\^(\d+)C', m)
    if mSC:
        k = int(mSC.group(1))
        return ('Orange', 'Orange') if k == 1 else ('Pink', 'Pink')
    if 'C13' in m:
        return 'Green', 'Green'
    if 'C24' in m:
        return 'DarkGreen', 'DarkGreen'
    return 'Black', 'Black'


def edge_directive(m):
    """(style, thickness) for one edge; G = heavy dots with wide spacing."""
    if m == 'G':
        return 'Dashing[{0.007, 0.0125}], DarkYellow', EDGE_T
    return mode_color(m)[1], EDGE_T


def vertex_directive(m):
    """(fill, edgeform): G gets a black ring, all others no outline."""
    if m == 'G':
        return 'DarkYellow', '{Black, Thickness[0.0045]}'
    return mode_color(m)[1], 'None'


def make_caption_lines(em, vm):
    """Caption block: one line per colour, listing all modes it covers."""
    by = {}
    for m in sorted(set(em) | set(vm.values())):
        c, _ = mode_color(m)
        by.setdefault(c, []).append(m)
    parts = []
    for c in COLOR_ORDER:
        if c in by:
            parts.append('%s: %s' % (c, ', '.join(by[c])))
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
    """(vec, em, vm) from either the 8-tuple or (vec, cuts, em, vm)."""
    if len(r) == 8:
        vm, em = r[6], r[7]
        return to_scaling(em), em, vm
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
