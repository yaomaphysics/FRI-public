#!/usr/bin/env python3
"""facet_regions_interactive.py — interactive region browser for the collinear 2->3 FRI enumerator (region_checker).

Input: graph topology + external kinematics ONLY (the cut formalism is
internal — the user never touches cuts).  External momenta p1..p5 attach
at vertices 1,2,3,4,5.  The script:

  1. enumerates ALL regions of the graph (skeleton: k0 union / k1 engine / k2-k4 chain),
  2. lists them with numbers (scaling vector + non-empty cuts + edge-mode sequence), then offers a menu:
       1) Inspect specific regions — pick a subset ("3, 8--10" = regions 3, 4, 8, 10; empty = all); for every picked region:
          MODE SUBGRAPHS
            mode X: {vertices: {V_X}, edges: {E_X}}   loop number of X (r_X = sum of |E| - |V| + 1 over the 1VI blocks of the contracted
          mode subgraph, computed with region_checker's own mode_components; Σ r_X vs L is checked), then optionally a concrete set of independent loop momenta (default basis) with optional FORCED lines ((x,y) endpoint pairs, square brackets also accepted),
       2) Show Lee-Pomeransky parametric representation — per region the scaling vector v_e = -V (edge order, trailing 1 = expansion scale,
          pySecDec style, same as the verified region files),
       3) Classify these regions based on their characteristic modes — per region its softest-mode class + by-type counts,
       4) Visualize the selected regions: one PDF atlas (default) or PNG figures (per-mode colours; saved under fri_out/regions_*/),
   and loops until the user quits (empty or q).

Kinematics: k0..k4 (defined in kin23.py; default k1).  Commands inside
the edge prompt: 'kin kX' switches kinematics, 'v' toggles the
enumeration statistics, 'q' quits.  Start with -v to have the statistics
on from the beginning.
Results are saved to fri_out/<timestamp>.txt after each graph.

Usage: python3 facet_regions_interactive.py [-v]
  Example graph (the built-in Frog): 1-3,1-5,2-3,2-5,3-4,4-5
"""
import os
import re
import signal
import sys
import time
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import region_checker
import kin23
import skeleton
from indep_loops import indep_loops, show_basis, _disp_key   # independent loop momenta + line-momentum parameterization
from primitives import V, m_of, name, n_of

EXT_ATTACH = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4, 'p5': 5}
KIN_CHOICES = list(kin23.KIN_ORDER)
KIN_NOTES = {
    'k0': 'all p_i^2 ~ t1',
    'k1': 'all p_i^2 = 0 (lightlike)',
    'k2': 'p_1^2 ~ t1; p_i^2 = 0 otherwise',
    'k3': 'p_1^2 ~ t1, p_2^2 ~ t1^2; p_i^2 = 0 otherwise',
    'k4': 'p_2^2 ~ p_3^2 ~ t1^2; p_i^2 = 0 otherwise',
}


# ---------------------------------------------------------------- input helpers
# Ask a y/n question (empty = n); any other input re-asks.
def ask_yn(prompt):
    while True:
        ans = input(prompt).strip().lower() or 'n'
        if ans in ('y', 'n'):
            return ans == 'y'
        print('  ! answer y or n (empty = n)')

# Parse an edge list leniently: accepts '1-3,1-5,...', '(1,3),(1,5),...' or '[(1,3),(1,5),...]'.
def parse_edges(s):
    t = re.sub(r'[\[\](){}]', ' ', s).replace(';', ',')
    dp = re.findall(r'(\d+)\s*-\s*(\d+)', t)
    dn = re.findall(r'\d+', t)
    if not dn:
        if t.strip():
            raise ValueError(f"cannot parse edge list: '{s.strip()}'")
        return []
    if dp and 2 * len(dp) == len(dn):
        return [(int(a), int(b)) for a, b in dp]
    if len(dn) % 2:
        raise ValueError(f"cannot parse edge list: '{s.strip()}'")
    return [(int(dn[k]), int(dn[k + 1])) for k in range(0, len(dn), 2)]


def fmt_scaling(sc):
    return '(' + ', '.join(str(x) for x in sc) + ')'


# Non-empty cuts as a dict-of-lists string (native 2->3 style).
def fmt_cuts(cuts):
    return str({k: sorted(v) for k, v in cuts.items() if v})


# '3, 8--10' -> [3, 4, 8, 10] (1-based). Supports ',' separators and '--' / '-' ranges. Returns None if any number is
# out of range.
def parse_region_select(s, n):
    out = set()
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '--' in part:
            a, b = part.split('--', 1)
        elif '-' in part:
            a, b = part.split('-', 1)
        else:
            out.add(int(part))
            continue
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        out.update(range(a, b + 1))
    bad = sorted(i for i in out if not (1 <= i <= n))
    if bad:
        print(f'  ! region numbers out of range (1..{n}): {bad}')
        return None
    return sorted(out)


# '[2,5],[7,8]' -> sorted edge indices (order-free endpoints). Returns None if nothing parses or an edge is unknown.
# NB: for parallel edges, a (a,b) pair forces ALL copies.
def parse_forced_lines(edges, line):
    pairs = re.findall(r'[\[(]\s*(\d+)\s*,\s*(\d+)\s*[\])]', line)
    if not pairs:
        print('  ! no (x,y) pairs found — use e.g. (2,5),(7,8) (square brackets ok too)')
        return None
    idx = {}
    for i, (a, b) in enumerate(edges):
        idx.setdefault((a, b), []).append(i)
        idx.setdefault((b, a), []).append(i)
    F, unknown = [], []
    for a, b in pairs:
        got = idx.get((int(a), int(b)))
        if not got:
            unknown.append((int(a), int(b)))
        else:
            F.extend(got)
    if unknown:
        print('  ! unknown edges: ' + ', '.join(f'[{a},{b}]' for a, b in unknown))
        return None
    return sorted(set(F))


# ---------------------------------------------------------------- mode display
# Per-mode subgraphs {{V_X}, {E_X}} with loop numbers r_X, plus the sum check vs L = |E| - |V| + 1. r_X uses the
# contracted-subgraph rule (1VI blocks of gamma~_X with the aux vertex; r = |E| - |V| + 1 per block) — same rule as
# the 2->2 browser, computed with region_checker's own mode_components.
def show_mode_subgraphs(edges, em, vm):
    verts = sorted({v for e in edges for v in e})
    results, total, L = indep_loops(edges, em, vm)
    rank = {r['mode']: r['rank'] for r in results}
    for mode in sorted(rank, key=_disp_key):
        E_idx = [i for i, m in enumerate(em) if m == mode]
        Vset = [v for v in verts if vm.get(v) == mode]
        Es = '{' + ', '.join(f'{i}:{edges[i]}' for i in E_idx) + '}'
        Vs = '{' + ', '.join(str(v) for v in Vset) + '}'
        print(f'  mode {name(mode)}: {{vertices: {Vs}, edges: {Es}}}   loop number = {rank[mode]}')
    flag = '✓' if total == L else '✗ MISMATCH'
    print(f'  Σ loop numbers = {total}  vs  L = {L}  {flag}')


# ---------------------------------------------------------------- menus
# Option 1: pick regions -> per-mode subgraphs with loop numbers, then optionally a concrete basis / forced lines /
# line momenta.
def inspect_regions(edges, regs, ext_attach):
    n = len(regs)
    line = input('Region numbers (e.g. "3, 8--10" represents regions 3, 8, 9, and 10; empty = all; q/b = back) > ').strip()
    if line.lower() in ('q', 'quit', 'b', 'back'):
        return
    sel = parse_region_select(line, n) if line else list(range(1, n + 1))
    if sel is None:
        sel = list(range(1, n + 1))
    for i in sel:
        vec, cuts, em, vm = regs[i - 1]
        print(f'  --- region {i}:')
        show_mode_subgraphs(edges, em, vm)
    if ask_yn('Select a set of line momenta as independent loop momenta? (y/n) [n] > '):
        asked = False
        while True:
            prompt = ('Force lines into the basis? ((x,y) pairs; empty = show default basis) > ' if not asked else 'Force lines (new input replaces the previous set)? ((x,y) pairs; empty = done) > ')
            line = input(prompt).strip()
            if line.lower() in ('q', 'quit', 'b', 'back'):
                break
            if not line:
                if not asked:
                    for i in sel:
                        vec, cuts, em, vm = regs[i - 1]
                        print(f'  --- region {i}:')
                        show_basis(edges, em, vm, ext_attach=ext_attach)
                break
            asked = True
            F = parse_forced_lines(edges, line)
            if F is None:
                continue
            for i in sel:
                vec, cuts, em, vm = regs[i - 1]
                print(f'  --- region {i}:')
                show_basis(edges, em, vm, F, ext_attach)


# Option 2: Lee-Pomeransky parametric representation per region: x_i ~ \lambda^{v_i}, v_i = -V (edge order, trailing 1 =
# expansion scale).
def show_parametric(regs):
    print('  Lee-Pomeransky parametric representation (v_i = -V, input edge order; trailing entry = expansion scale):')
    for i, (vec, cuts, em, vm) in enumerate(regs, 1):
        print(f'    R{i}: x_i ~ \\lambda^{{v_i}} for i = 1,2,...,{len(vec)}, with v = {fmt_scaling(vec)}')


# Softness order for classification, softest first: modes with a soft prefactor m >= 1 by virtuality V (desc); tie m
# desc, n desc, name.
def _softness_key(m):
    return (0, -V(m), -m_of(m), -(n_of(m) if n_of(m) is not None else 0), name(m))


# Softest characteristic mode of a region (exclusion-style): the softest mode with m >= 1 present in em/vm; if none,
# the bare carrier/hard class 'C/H'.
def classify(vm, em):
    cands = set(em) | set(vm.values())
    soft = [m for m in cands if m[0] != 'H' and m_of(m) >= 1]
    if soft:
        return min(soft, key=_softness_key)
    return 'C/H'


# Option 3: classify regions by their characteristic (softest) mode, grouped by type (softest first).
def show_classify(regs):
    groups = defaultdict(list)
    for r in regs:
        vec, cuts, em, vm = r
        groups[classify(vm, em)].append(r)
    order = sorted(groups, key=lambda lab: (99,) if lab == 'C/H' else _softness_key(lab))
    total = 0
    for lab in order:
        g = groups[lab]
        total += len(g)
        label = lab if lab == 'C/H' else name(lab)
        print()
        print(f'There are {len(g)} {label} type regions:')
        for i, (vec, cuts, em, vm) in enumerate(g, 1):
            print(f'  --- {label} region {i} ---')
            print(f'    em = {[name(m) for m in em]}')
            print(f'    scaling = {fmt_scaling(vec)}')
    print(f'\nTOTAL: {total} regions')


# Option 4: visualize the selected regions — PDF atlas or PNG files.
def visualize_regions(edges, regs, ext_mode):
    n = len(regs)
    line = input('Region numbers (e.g. "3, 8--10" represents regions 3, 8, 9, and 10; empty = all; q/b = back) > ').strip()
    if line.lower() in ('q', 'quit', 'b', 'back'):
        return
    sel = parse_region_select(line, n) if line else list(range(1, n + 1))
    if sel is None:
        return
    fmt = input('Output: [a] single PDF atlas (default) / [p] individual PNG files > ').strip().lower()
    if fmt in ('q', 'quit', 'b', 'back'):
        return
    try:
        import region_plot
    except Exception as e:
        print(f'  ! visualization module unavailable: {e}')
        return
    outdir = os.path.join(BASE, 'fri_out', 'regions_' + time.strftime('%Y%m%d-%H%M%S'))
    if fmt in ('p', 'png', 'files'):
        print(f'  rendering {len(sel)} region figure(s) via wolframscript ...')
        try:
            paths = region_plot.render_individual_pngs(edges, sorted({v for e in edges for v in e}), [(i, regs[i - 1]) for i in sel], ext_mode=ext_mode, outdir=outdir)
        except Exception as e:
            print(f'  ! region rendering failed: {e}')
            return
        print(f'  saved {len(paths)} region PNG(s) to: {outdir}')
    else:
        print(f'  building the PDF atlas for {len(sel)} region(s) via wolframscript ...')
        try:
            path = region_plot.render_combined_pdf(edges, sorted({v for e in edges for v in e}), [(i, regs[i - 1]) for i in sel], ext_mode=ext_mode, outdir=outdir)
        except Exception as e:
            print(f'  ! atlas rendering failed: {e}')
            return
        print(f'  saved atlas PDF to: {path}')


def print_kin_menu():
    print('available 2->3 kinematics:')
    for k in KIN_CHOICES:
        print(f'  {k}: {KIN_NOTES[k]}')


# Ask for a kinematics label at startup; default k1.
def choose_kinematics():
    print_kin_menu()
    while True:
        try:
            s = input('kinematics [default k1; q/b = back]> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if not s:
            return 'k1'
        if s in ('q', 'quit', 'exit', 'b', 'back'):
            return None
        if s in KIN_CHOICES:
            return s
        print(f'  [error] unknown kinematics "{s}" — choose from {", ".join(KIN_CHOICES)} (q/b = back)')


# ---------------------------------------------------------------- driver
def run_graph(edges_raw, kin_name, verbose=False):
    edges = [tuple(e) for e in edges_raw]
    verts = sorted({v for e in edges for v in e})
    missing = [v for v in (1, 2, 3, 4, 5) if v not in verts]
    if missing:
        print(f'  [warning] external-leg vertices {missing} not in the graph!  The 2->3 kinematics needs vertices 1,2,3,4,5.')
        return
    ext_mode = kin23.ext_modes(kin_name)
    L = len(edges) - len(verts) + 1
    print(f'  kinematics: {kin_name} ({KIN_NOTES[kin_name]})')
    print(f'  graph: {len(edges)} edges, {len(verts)} vertices, L = {L} loops')
    print(f'  edges: {edges}')
    if L >= 5:
        print('  [warning] L >= 5 may be slow')
    t0 = time.time()
    try:
        _vecs, total, stats = skeleton.enumerate_surv(edges, verts, EXT_ATTACH, kin_name)
        surv = stats['survivors']
    except Exception as e:
        print(f'  [error] skeleton engine could not handle this input: {e}')
        return
    dt = time.time() - t0
    regs = sorted(surv, key=lambda s: (tuple(float(x) for x in s[0]), tuple(name(m) for m in s[2])))
    print(f'  FRI regions: {len(regs)}  ({dt:.1f}s)')
    if verbose:
        rej = ', '.join(f'{k}={v}' for k, v in sorted(stats.items()) if k != 'survivors')
        print(f'  [stats] combos={total}  kept={len(regs)}  counters: {rej or "none"}')
    scal = set()
    for i, (vec, cuts, em, vm) in enumerate(regs, 1):
        scal.add(vec)
        print(f'  R{i:3d}: scaling {fmt_scaling(vec)}')
        print(f'        cuts: {fmt_cuts(cuts)}')
        print(f'        em: {[name(m) for m in em]}')
    print(f'  unique scalings: {len(scal)}')
    while True:
        print('  Options:')
        print('    1) Inspect specific regions')
        print('    2) Show Lee-Pomeransky parametric representation')
        print('    3) Classify these regions based on their characteristic modes')
        print('    4) Visualize these regions (PDF atlas / PNGs)')
        opt = input('  (1/2/3/4; empty/q/b = done with this graph) > ') .strip().lower()
        if opt in ('', 'q', 'quit', 'b', 'back'):
            break
        if opt == '1':
            inspect_regions(edges, regs, EXT_ATTACH)
        elif opt == '2':
            show_parametric(regs)
        elif opt == '3':
            show_classify(regs)
        elif opt == '4':
            visualize_regions(edges, regs, ext_mode)
        else:
            print('  ! enter 1, 2, 3, 4, or q')
    # save
    fdir = os.path.join(BASE, 'fri_out')
    os.makedirs(fdir, exist_ok=True)
    fname = os.path.join(fdir, time.strftime('%Y%m%d-%H%M%S') + '.txt')
    with open(fname, 'w') as f:
        f.write(f'# collinear 2->3 FRI regions - {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# kinematics: {kin_name}\n')
        f.write(f'# edges: {edges}  ({len(edges)} edges, {len(verts)} verts, L={L})\n')
        f.write(f'# FRI regions: {len(regs)} ({dt:.1f}s)\n\n')
        for i, (vec, cuts, em, vm) in enumerate(regs, 1):
            f.write(f'R{i:3d}: scaling {fmt_scaling(vec)}\n')
            f.write(f'      cuts: {fmt_cuts(cuts)}\n')
            f.write(f'      em: {[name(m) for m in em]}\n')
            vmtxt = ', '.join(f'{v}: {name(vm[v])}' for v in sorted(vm))
            f.write(f'      vm: {{{vmtxt}}}\n')
    print(f'  saved to: {fname}')


def main():
    verbose = any(a in ('-v', '--verbose') for a in sys.argv[1:])
    print('=' * 72)
    print('collinear 2->3 FRI region enumerator (k0..k4)')
    print('external momenta: p1@1, p2@2, p3@3, p4@4, p5@5')
    print('type an edge list, e.g. 1-3,1-5,2-3,2-5,3-4,4-5 (the Frog)')
    print("commands: 'kin kX' switch kinematics, 'v' toggle stats, 'q'/'b' quit")
    print('=' * 72)
    kin_name = choose_kinematics()
    if kin_name is None:
        print('bye!')
        return
    print(f'-> using kinematics {kin_name}')
    while True:
        try:
            s = input(f'\nedge list (e.g. 1-3,1-5,2-3,2-5,3-4,4-5) [{kin_name}]> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nbye!')
            break
        if not s:
            continue
        if s.lower() in ('q', 'quit', 'exit', 'b', 'back'):
            print('bye!')
            break
        low = s.lower()
        if low.startswith('kin '):
            k = low.split()[1]
            if k in KIN_CHOICES:
                kin_name = k
                print(f'-> kinematics switched to {kin_name} ({KIN_NOTES[kin_name]})')
            else:
                print(f'  [error] unknown kinematics "{k}" — choose from {", ".join(KIN_CHOICES)}')
            continue
        if low in ('v', '-v', 'stats', 'verbose'):
            verbose = not verbose
            print(f'-> enumeration statistics {"ON" if verbose else "OFF"}')
            continue
        try:
            edges = parse_edges(s)
        except Exception as e:
            print(f'  [error] {e}')
            continue
        if not edges:
            print('  [error] empty input')
            continue
        t0 = time.time()
        run_graph(edges, kin_name, verbose)
        print(f'  (this graph took {time.time() - t0:.1f}s total)')


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.default_int_handler)  # keep Ctrl+C working even if the launcher had ignored SIGINT
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nbye!')
