#!/usr/bin/env python3
"""facet_regions_interactive.py — interactive region browser for the collinear 2->3 FRI enumerator (region_filter).

Input: graph topology + external kinematics ONLY (the cut formalism is
internal — the user never touches cuts).  External momenta p1..p5 attach
at vertices 1,2,3,4,5.  The script:

  1. enumerates ALL regions of the graph (skeleton: all-on-shell engine / general chain),
  2. lists them with numbers (scaling vector + non-empty cuts + edge-mode sequence), then offers a menu:
       1) Inspect specific regions — pick a subset ("3, 8--10" = regions 3, 4, 8, 10; empty = all); for every picked region:
          MODE SUBGRAPHS
            mode X: {vertices: {V_X}, edges: {E_X}}   loop number of X (r_X = sum of |E| - |V| + 1 over the 1VI (one-vertex-irreducible) blocks of the contracted
          mode subgraph, computed with region_filter's own mode_components; Σ r_X vs L is checked), then optionally a concrete set of independent loop momenta (default basis) with optional FORCED lines ((x,y) endpoint pairs, square brackets also accepted),
       2) Show Lee-Pomeransky parametric representation — per region the scaling vector v_e = -V (edge order, trailing 1 = expansion scale,
          pySecDec style, same as the verified region files),
       3) Classify these regions based on their characteristic modes — per region its softest-mode class + by-type counts,
       4) Group these regions by their power — a numerator polynomial in the edge momenta K_i (input order) and the externals p_j may be entered first ("1" = scalar),
       5) Visualize the selected regions: one PDF atlas (default) or PNG figures (per-mode colours; saved under fri_out/regions_*/),
   and loops until the user quits (empty or q).

Kinematics: k0..k2 are shown as local examples (default k1); type kx for a custom p_i^2 pattern.
At the edge prompt: 'kin kx' (or a k0..k2 label) switches kinematics, 'v' toggles the enumeration statistics, 'q'/'b' quits.

Usage: python3 facet_regions_interactive.py [-v]
  Example graph (the built-in Fish): 1-3,1-5,2-3,2-5,3-4,4-5
"""
import os
import re
import signal
import sys
import time
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import region_filter
import enumerator
from indep_loops import indep_loops, show_basis, _disp_key   # independent loop momenta + line-momentum parameterization
from primitives import V, m_of, name, n_of, INF, leg_check, ext_modes
from mode_levels import predict, derive_external_modes
from power_counting import (show_power_counting, measure_power, integrand_power, fmt_power,
                           parse_numerator, fmt_ast, region_context, numerator_power)

EXT_ATTACH = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4, 'p5': 5}
KIN_EXAMPLES = ['k0', 'k1', 'k2']
# Local illustration of the three example kinematics: label text + raw (m1..m5) virtuality tuple.
KIN_NOTES = {
    'k0': ('all p_i^2 ~ t1', (1, 1, 1, 1, 1)),
    'k1': ('all p_i^2 = 0 (lightlike)', (INF, INF, INF, INF, INF)),
    'k2': ('p_1^2 ~ t1; p_i^2 = 0 otherwise', (1, INF, INF, INF, INF)),
}

# Per external leg: (label, mode prefix, mode suffix, lowest accepted power).
LEG_MODES = [('p1', 'C1^', '', 1), ('p2', 'C2^', 'C23', 0), ('p3', 'C3^', 'C23', 0), ('p4', 'C4^', '', 1), ('p5', 'C5^', '', 1)]


# ==================== INPUT HELPERS ====================

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


# Format a scaling tuple as a parenthesised comma-separated string.
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


# ==================== MODE DISPLAY ====================

# Per-mode subgraphs {V_X}, {E_X} with loop numbers r_X (1VI blocks of the contracted mode subgraph), plus the sum check against L = |E| - |V| + 1.
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
    show_power_counting(results, em)


# ==================== MENU ACTIONS ====================

# Ask for the numerator polynomial (in K_i / externals; "1" or empty = scalar); q/b cancels.
def ask_numerator(nedge, pnames):
    while True:
        print('Input the numerator')
        print('(e.g. (p1\\cdot K_1)(K_2\\cdot K_3) + ..., where K_i is the momentum of i-th propagator in the input; "1" or empty = scalar; q/b = cancel) ')
        line = input('> ').strip()
        if line.lower() in ('q', 'quit', 'b', 'back'):
            return None, False
        if not line or line == '1':
            print('  numerator = 1 (scalar)')
            return None, True
        try:
            ast, warns = parse_numerator(line, nedge, pnames)
        except ValueError as e:
            print('  ! %s' % e)
            continue
        for w in warns:
            print('  ! note: %s' % w)
        if ast is None:
            print('  numerator = 1 (scalar)')
            return None, True
        text = fmt_ast(ast)
        show = text if len(text) <= 160 else text[:157] + '...'
        print('  numerator = %s' % show)
        return (ast, text), True


# Option 4: group the regions by their power lambda^{A-B+N} (a numerator is asked first).
def show_group_by_power(regs, edges, ext_attach, extmode, numerator):
    groups = defaultdict(list)
    for i, (vec, cuts, em, vm) in enumerate(regs, 1):
        res, _t, _L = indep_loops(edges, em, vm)
        a0, a1 = measure_power(res)
        n = 0
        if numerator is not None:
            ctx = region_context(edges, em, vm, ext_attach, extmode)
            n = numerator_power(numerator[0], ctx) if ctx is not None else 0
        groups[(a0 - integrand_power(em) + n, a1)].append(i)
    for (p0, p1), rs in sorted(groups.items()):
        print()
        print('Region(s) with power = %s:' % fmt_power(p0, p1))
        for i in rs:
            print('  R%d: %s' % (i, ', '.join(name(m) for m in regs[i - 1][2])))


# Option 1: pick regions, show their per-mode subgraphs, then optionally a concrete basis / forced lines / line momenta.
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


# Power of a mode as text ('inf' for the on-shell case).
def _pow_str(q):
    return 'inf' if q == INF else str(q)


# The virtuality tuple (m1..m5) that reproduces a custom power tuple (the pair power is m - 1).
def custom_ms(q):
    q1, q2, q3, q4, q5 = q
    return (q1, INF if q2 == INF else q2 + 1, INF if q3 == INF else q3 + 1, q4, q5)


# Short label for a kinematics: a preset name, or the custom powers.
def kin_label(kin):
    return kin if isinstance(kin, str) else 'custom ' + '/'.join(_pow_str(q) for q in kin)


# Display note for a kinematics: a preset name, or the custom mode literals.
def kin_note(kin):
    if isinstance(kin, str):
        return KIN_NOTES[kin][0]
    q1, q2, q3, q4, q5 = kin
    return ('C1^' + _pow_str(q1) + ', C2^' + _pow_str(q2) + 'C23, C3^' + _pow_str(q3) + 'C23, C4^' + _pow_str(q4) + ', C5^' + _pow_str(q5))


# Read a custom kinematics: the power of each external mode (integer >= 0, or inf for on-shell).
def ask_momenta():
    print('Input the external modes (only the powers, selected from {0, 1, ..., ' + chr(92) + 'infty})')
    q = []
    for i, (leg, pre, post, lo) in enumerate(LEG_MODES, 1):
        while True:
            try:
                s = input('  %s mode [%s?%s] (only input the power) > ' % (leg, pre, post)).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if s in ('q', 'quit', 'exit', 'b', 'back'):
                return None
            if s in ('inf', 'infty', 'os', 'on-shell', chr(92) + 'infty'):
                p = '∞'
            elif s.isdigit() and int(s) >= lo:
                p = s
            else:
                print('    [error] enter an integer >= %d, or ' % lo + chr(92) + 'infty')
                continue
            if not leg_check(i, region_filter.parse((pre[:-1] + p + post) if p == '∞' else (pre + p + post))):
                print('    [error] not allowed for %s' % leg)
                continue
            q.append(INF if p == '∞' else int(p))
            break
    return tuple(q)


# Print the example kinematics; kx enters a custom pattern.
def print_kin_menu():
    print('Input kinematics:')
    print('For example,')
    for k in KIN_EXAMPLES:
        print(f'  {k}: {KIN_NOTES[k][0]};')


# Ask for a kinematics label at startup; default k1.
def choose_kinematics():
    print_kin_menu()
    while True:
        try:
            s = input('kinematics [k0, k1, k2 for above; kx for others; q/b = back]> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if s in ('q', 'quit', 'exit', 'b', 'back'):
            return None
        if s in KIN_EXAMPLES:
            return s
        if s == 'kx':
            kin = ask_momenta()
            if kin is not None:
                return kin
            print_kin_menu()
            continue
        print(f'  [error] unknown kinematics "{s}" — choose from {", ".join(KIN_EXAMPLES)}, or kx (q/b = back)')


# ==================== DRIVER ====================

# Enumerate one graph's regions and offer the inspection menu; writes a run file under fri_out/.
def run_graph(edges_raw, kin_name, verbose=False):
    edges = [tuple(e) for e in edges_raw]
    verts = sorted({v for e in edges for v in e})
    missing = [v for v in (1, 2, 3, 4, 5) if v not in verts]
    if missing:
        print(f'  [warning] external-leg vertices {missing} not in the graph!  The 2->3 kinematics needs vertices 1,2,3,4,5.')
        return
    kin_ms = custom_ms(kin_name) if isinstance(kin_name, tuple) else KIN_NOTES[kin_name][1]
    ext_mode = ext_modes(kin_ms)
    L = len(edges) - len(verts) + 1
    print(f'  kinematics: {kin_label(kin_name)} ({kin_note(kin_name)})')
    print(f'  graph: {len(edges)} edges, {len(verts)} vertices, L = {L} loops')
    print(f'  edges: {edges}')
    if L >= 5:
        print('  [warning] L >= 5 may be slow')
    t0 = time.time()
    try:
        _vecs, total, stats = skeleton.enumerate_surv(edges, verts, EXT_ATTACH, kin_ms)
        surv = stats['survivors']
    except Exception as e:
        print(f'  [error] skeleton engine could not handle this input: {e}')
        return
    dt = time.time() - t0
    regs = sorted(surv, key=lambda s: (tuple(float(x) for x in s[0]), tuple(name(m) for m in s[2])))
    print(f'  FRI regions: {len(regs)}  ({dt:.1f}s)')
    # Enumeration statistics (hidden debug switch: 'v' at the edge prompt / -v at startup).
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
        print('    4) Group these regions by their power')
        print('    5) Visualize these regions (PDF atlas / PNGs)')
        opt = input('  (1/2/3/4/5; empty/q/b = done with this graph) > ') .strip().lower()
        if opt in ('', 'q', 'quit', 'b', 'back'):
            break
        if opt == '1':
            inspect_regions(edges, regs, EXT_ATTACH)
        elif opt == '2':
            show_parametric(regs)
        elif opt == '3':
            show_classify(regs)
        elif opt == '4':
            num, ok = ask_numerator(len(edges), ['p1', 'p2', 'p3', 'p4', 'p5'])
            if ok:
                show_group_by_power(regs, edges, EXT_ATTACH, ext_mode, num)
                ans = input('  Detailed derivation report (PDF) for some regions? ([y]/n) > ').strip().lower()
                if ans in ('', 'y', 'yes'):
                    line = input('  Region numbers (empty = all; q/b = none) > ').strip()
                    if line.lower() not in ('q', 'quit', 'b', 'back'):
                        sel = parse_region_select(line, len(regs)) if line else list(range(1, len(regs) + 1))
                        if sel:
                            import power_report
                            pdf, _lines = power_report.build_report(edges, regs, sel, EXT_ATTACH, ext_mode, numerator=num)
                            print('  saved: %s' % pdf)
        elif opt == '5':
            visualize_regions(edges, regs, ext_mode)
        else:
            print('  ! enter 1, 2, 3, 4, 5, or q')
    # save
    fdir = os.path.join(BASE, 'fri_out')
    os.makedirs(fdir, exist_ok=True)
    fname = os.path.join(fdir, time.strftime('%Y%m%d-%H%M%S') + '.txt')
    with open(fname, 'w') as f:
        f.write(f'# collinear 2->3 FRI regions - {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# kinematics: {kin_label(kin_name)}\n')
        f.write(f'# edges: {edges}  ({len(edges)} edges, {len(verts)} verts, L={L})\n')
        f.write(f'# FRI regions: {len(regs)} ({dt:.1f}s)\n\n')
        for i, (vec, cuts, em, vm) in enumerate(regs, 1):
            f.write(f'R{i:3d}: scaling {fmt_scaling(vec)}\n')
            f.write(f'      cuts: {fmt_cuts(cuts)}\n')
            f.write(f'      em: {[name(m) for m in em]}\n')
            vmtxt = ', '.join(f'{v}: {name(vm[v])}' for v in sorted(vm))
            f.write(f'      vm: {{{vmtxt}}}\n')
    print(f'  saved to: {fname}')


LMAX = 10

BANNER = '''\
==============================================================
  What modes appear at which loop level?   (2->3)
--------------------------------------------------------------
  Enter the five external-momentum modes,
  then:
    space = step +1 loop      enter = jump to next non-empty loop
    a     = cumulative table  r     = report (L=0 .. current level)
    q/b   = quit
  (levels are capped at L = %d)
==============================================================''' % LMAX

# single keypress on a tty; line-based fallback otherwise ('' line = space).
def read_key():
    if not sys.stdin.isatty():
        try:
            s = input('▶ (blank=advance, j=jump, a=all, r=report, q/b=quit): ')
        except EOFError:
            return 'q'
        s = s.strip()
        if s == '':
            return ' '
        return s[0]
    import termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


# Display state for one kinematics: extends the ladder on demand and keeps the per-mode routes.
class Stepper:
    # Store the entered external modes and the momentum modes; the ladder is built lazily by ensure().
    def __init__(self, ext, momenta):
        self.ext, self.momenta = ext, momenta
        self.levels = {}
        self.routes = {}
        self.computed = -1

    # [Optimization] Extend the cached ladder to level k (recompute with the larger cap when the display advances).
    def ensure(self, k):
        k = min(k, LMAX)
        if k <= self.computed:
            return
        self.levels, log = predict(self.ext, k, momenta=self.momenta)
        self.routes = log['routes']
        self.computed = k

    # Mode names first computed at exactly level k.
    def modes_at(self, k):
        return [region_filter.name(m) for m in self.levels if self.levels[m] == k]

# Entry point: interactive loop over the mode ladder (the -l/--ladder mode).
def run_ladder():
    print(BANNER)
    q = ask_momenta()
    if q is None:
        print('bye 👋')
        return
    momenta = [region_filter.parse((pre[:-1] + '∞' + post) if p == INF else (pre + str(p) + post)) for (leg, pre, post, lo), p in zip(LEG_MODES, q)]
    mnames = [region_filter.name(m) for m in momenta]
    print('momenta: ' + ', '.join('p%d=%s' % (i + 1, nm) for i, nm in enumerate(mnames)))
    ext = derive_external_modes(momenta)
    print('external modes (L=0 seeds): %s' % ', '.join(ext))

    st = Stepper(ext, momenta)
    cur = 0

    # Print level k's new modes and how they arose.
    def show(k):
        st.ensure(k)
        names = st.modes_at(k)
        print()
        if names:
            n = len(names)
            print('L=%d — %d new mode%s:' % (k, n, 's' if n != 1 else ''))
            for nm in names:
                print('   %s' % nm)
        else:
            print('L=%d — (no new modes)' % k)
        n_le = sum(1 for lv in st.levels.values() if lv <= k)
        print('   cumulative: %d modes (≤L=%d) | space=+1, enter=next non-empty, a=all, r=report, q/b=quit' % (n_le, k))

    show(cur)
    while True:
        k = read_key()
        if k in ('q', 'Q', 'b', 'B', '\x1b', '\x03'):
            print('\nbye 👋')
            break
        if k == ' ':
            if cur < LMAX:
                cur += 1
                show(cur)
            else:
                print('  (display capped at L=%d)' % LMAX)
        elif k in ('\r', '\n', 'j'):
            j0, hit = cur, None
            while j0 < LMAX:
                hi = min(j0 + 6, LMAX)
                st.ensure(hi)
                for k2 in range(j0 + 1, hi + 1):
                    if st.modes_at(k2):
                        hit = k2
                        break
                if hit is not None:
                    break
                j0 += 6
            if hit is None:
                print('  (no new modes up to L=%d)' % LMAX)
            else:
                cur = hit
                show(cur)
        elif k in ('a', 'A'):
            st.ensure(cur)
            print()
            for lv in sorted(set(st.levels.values())):
                if lv > cur:
                    break
                print('L=%-3d: %s' % (lv, ', '.join(st.modes_at(lv))))
        elif k in ('r', 'R'):
            import mode_report
            pdf, _ = mode_report.build_report(st, cur)
            print('  saved mode-ladder report to: %s' % pdf)
        else:
            print('  (space=advance, enter=jump, a=all, r=report, q/b=quit)')

# Entry point: interactive loop over graphs.
def main():
    # hidden mode: -l/--ladder runs the mode-ladder stepper instead of the graph browser
    if any(a in ('-l', '--ladder') for a in sys.argv[1:]):
        run_ladder()
        return
    # Hidden debug switch (not shown to the user): -v/--verbose starts with the enumeration statistics on.
    verbose = any(a in ('-v', '--verbose') for a in sys.argv[1:])
    print('=' * 72)
    print('collinear 2->3 FRI region enumerator')
    print('external momenta: p1@1, p2@2, p3@3, p4@4, p5@5')
    print('=' * 72)
    kin_name = choose_kinematics()
    if kin_name is None:
        print('bye!')
        return
    print(f'-> using kinematics {kin_label(kin_name)} ({kin_note(kin_name)})')
    while True:
        try:
            s = input(f'\nedge list for {kin_label(kin_name)} (e.g., the Fish, 1-3,1-5,2-3,2-5,3-4,4-5)> ').strip()
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
            if k in KIN_EXAMPLES:
                kin_name = k
                print(f'-> kinematics switched to {kin_label(kin_name)} ({kin_note(kin_name)})')
            elif k == 'kx':
                kin = ask_momenta()
                if kin is not None:
                    kin_name = kin
                    print(f'-> kinematics switched to {kin_label(kin_name)} ({kin_note(kin_name)})')
            else:
                print(f'  [error] unknown kinematics "{k}" — choose from {", ".join(KIN_EXAMPLES)}, or kx')
            continue
        # Hidden debug switch: 'v' toggles the enumeration statistics.
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
