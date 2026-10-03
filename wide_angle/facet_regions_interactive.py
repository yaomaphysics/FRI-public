#!/usr/bin/env python3
"""
facet_regions_interactive.py — interactive facet-region browser.

Input: graph topology + external kinematics ONLY (the cut formalism is internal — the user never touches cuts).

Script:
  1. enumerates ALL regions of the graph (C/H family via nested cuts + compression on by default),
  2. lists them with numbers (edge-mode sequence per region, in input edge order), then offers a menu:
       1) Inspect specific regions  — pick a subset ("e.g., 3, 8--10" = regions 3, 4, 8, 10; empty = all);
           for every picked region, output the mode subgraphs:
           "X: {vertices: {...}, edges: {...}}   loop number = ...."
           plus the scalar power counting (integration measure / integrand / power, as \\lambda^{...}).
          Optionally, the user can select a set of line momenta as independent loop momenta (a basis) with optional forced lines.
       2) Show Lee-Pomeransky parametric representation — per region the scaling vector v_i (x_i ~ \\lambda^{v_i}, v_i = -(2m+n), input edge order),
       3) Group these regions by their characteristic mode — per region its softest-mode class + by-type counts,
       4) Group these regions by their power — a numerator may be entered first (a polynomial in the edge momenta K_i and the externals p_j; "1" = scalar);
          each region's power is then computed silently and grouped in ascending order (leading first),
          with an optional step-by-step derivation report (PDF) for a chosen subset.
       5) Visualize the selected regions — per-mode colours; can choose a single PDF atlas (default) or one PNG file per region.
  This loops until the user quits (empty or q).

Usage: python3 facet_regions_interactive.py
  internal_lines: edge list, e.g. "1-5,1-8,..." (a Python list also works); empty line = built-in example, which is the example in Sec. 7.1 of arXiv:2601.22144.
  externals: prompted one by one (vertex, name, mode) — the example uses p1 = [1,'C1'], p2 = [2,'C2^2'],
  p3 = [3,'C3^inf'], p4 = [4,'C4^inf'] (81 regions in total).

Mode syntax: H / S / S^m / C_i / C_i^n / C_i^inf / C_i^\\infty / SC_i / SC_i^n / S^mC_i^n  (C_i without n means n=1; SC_i without n means n=1; i is the direction index from 1).
"""
import sys, re, os, time, warnings, signal
warnings.filterwarnings('ignore', category=SyntaxWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_graph import (mode_str, parse_mode, sc_short, type_order, group_by_type)
from primitives import scaling_of
from indep_loops import indep_loops, show_basis
from power_counting import (show_power_counting, measure_power, integrand_power, fmt_power,
                            parse_numerator, fmt_ast, region_context, numerator_power)
from power_report import build_report
from skeleton import run as skeleton_run, kappa_of
H = (0, 0, 0)


# One mode tuple -> display string ('∅' when an edge mode is missing).
def ms(md):
    return mode_str(md) if md is not None else '∅'

# ---------------------------------------------------------------- input helpers
# Ask a y/n question (empty = n); any other input re-asks.
def ask_yn(prompt):
    while True:
        ans = input(prompt).strip().lower() or 'n'
        if ans in ('y', 'n'):
            return ans == 'y'
        print('  ! answer y or n (empty = n)')

# Parse an edge list leniently: accepts '1-5,1-8,...', '(1,5),(1,8),...' or '[(1,5),(1,8),...]'.
def parse_edge_list(s):
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

# Ask for the edge list; empty input = default; 'q' quits; re-prompts on parse errors.
def ask_edges(label, default):
    print(f'{label}:' + (f'  (default: {default})' if default else ''))
    while True:
        line = input('> ').strip()
        if line.lower() in ('q', 'quit', 'exit', 'b', 'back'):
            return None
        line = line or default
        try:
            return parse_edge_list(line)
        except ValueError as e:
            print(f'  ! {e}')

# Ask for the external momenta one by one (vertex, name, mode) until the user no longer wants to add another.
def ask_externals():
    print('Externals: add them one by one (vertex, name, mode; q/b = quit).')
    out = {}
    while True:
        i = len(out) + 1
        while True:
            s = input(f'  external {i}: attach to which vertex? (integer) > ').strip()
            if s.lower() in ('q', 'quit', 'exit', 'b', 'back'):
                return None
            try:
                vtx = int(s); break
            except ValueError:
                print('  ! enter an integer vertex label, e.g. 1')
        while True:
            nm = input(f'  external {i}: name? (e.g. p{i}) > ').strip()
            if nm.lower() in ('q', 'quit', 'exit', 'b', 'back'):
                return None
            if nm and nm not in out:
                break
            print('  ! the name must be non-empty and not yet used: ' + (', '.join(out) if out else '(no names so far)'))
        while True:
            s = input(f'  external {i}: mode? (form S^mC_i^n, e.g. S^2C1^3; also allowed: H, S^m, C_i^n) > ').strip()
            if s.lower() in ('q', 'quit', 'exit', 'b', 'back'):
                return None
            try:
                parse_mode(s); break
            except ValueError as e:
                print(f'  ! {e}')
        out[nm] = [vtx, s]
        print(f'  added: {nm} = [vertex {vtx}, mode {s}]')
        if not ask_yn('  Add another external? (y/n) [n] > '):
            break
    print(f'  externals = {out}')
    return out

# ---------------------------------------------------------------- enumeration
# Enumerate all regions of the graph; returns a list of (vm, em).
def enumerate_regions(verts, edges, ext_attach, ext_mode):
    regs = {}
    regions, _nc, _dt = skeleton_run(verts, edges, ext_attach, ext_mode, verbose=False, overlap_strong=True)
    for vm, em in regions:
        regs.setdefault(tuple(tuple(m) for m in em), (vm, em))
    return list(regs.values())

# Edge modes as one display string, in terms of the input edge order.
def em_sequence(em):
    return ', '.join(ms(m) for m in em)


# Format a vertex/edge set as '{a, b, c}' ('empty' for the empty set).
def _fmt_set(items):
    if not items:
        return 'empty'
    parts = []
    for x in items:
        if isinstance(x, tuple) and len(x) == 2:
            parts.append(f'[{x[0]},{x[1]}]')
        else:
            parts.append(str(x))
    return '{' + ', '.join(parts) + '}'


# Print a region as one line per mode: <mode> vertex = {...}; <mode> edge = {...}. Empty vertex/edge sets print 'empty'.
def print_region_modes(vm, em, edges, indent='    '):
    edges_t = [tuple(sorted(e, key=str)) for e in edges]
    v_by_mode = {}
    for v in sorted(vm, key=str):
        v_by_mode.setdefault(vm[v], []).append(v)
    e_by_mode = {}
    for (a, b), md in zip(edges_t, em):
        e_by_mode.setdefault(md, []).append((a, b))
    all_modes = list(v_by_mode) + [m for m in e_by_mode if m not in v_by_mode]
    for md in all_modes:
        s = mode_str(md)
        print(f'{indent}{s} vertex = {_fmt_set(v_by_mode.get(md, []))}; {s} edge = {_fmt_set(e_by_mode.get(md, []))}.')


# print_region_modes followed by the corresponding region vector.
def print_region_with_vector(vm, em, edges, indent='    '):
    print_region_modes(vm, em, edges, indent=indent)
    vec = [scaling_of(md) for md in em]
    print(f'{indent}scaling vector = ({', '.join(map(str, vec))}, 1)')


# ---------------------------------------------------------------- display
# Split a region into per-mode subgraphs {mode: (vertex set, edge list)}.
def mode_subgraphs(edges, em, vm):
    subs = {}
    for i, m in enumerate(em):
        m = m if m is not None else H
        subs.setdefault(m, [set(), []])[1].append(edges[i])
    for v, m in vm.items():
        m = m if m is not None else H
        subs.setdefault(m, [set(), []])[0].add(v)
    return subs

# Print per-mode subgraphs {{V_X}, {E_X}} with their independent loop numbers.
def show_mode_subgraphs(edges, em, vm):
    results, total, L = indep_loops(edges, em, vm)
    rank = {r['mode']: r['rank'] for r in results}
    for X in sorted(mode_subgraphs(edges, em, vm), key=lambda m: (m[0], m[1], m[2])):
        V, E = mode_subgraphs(edges, em, vm)[X]
        Vs = '{' + ', '.join(str(v) for v in sorted(V)) + '}'
        Es = '{' + ', '.join(f'({u},{v})' for u, v in sorted(E)) + '}'
        print(f'  mode {ms(X)}: {{vertices: {Vs}, edges: {Es}}}   loop number = {rank.get(X, 0)}')
    print(f'  Σ loop numbers = {total}  vs  L = {L}  {"✓" if total == L else "✗ MISMATCH"}')
    show_power_counting(results, em)


# ---------------------------------------------------------------- parsing
# Parse a region selection like '3, 8--10' -> [3, 4, 8, 10]; sorted 1-based indices (None if invalid).
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

# Parse forced lines like '(2,5),(7,8)'; edge indices (None if invalid).
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

# ---------------------------------------------------------------- option handlers (menu 1-4)
# Option 1: pick regions -> mode subgraphs -> optional loop-momentum basis.
def inspect_regions(edges, regs, ext_attach):
    n = len(regs)
    line = input('Region numbers (e.g. "3, 8--10" represents regions 3, 8, 9, and 10; empty = all; q/b = back) > ').strip()
    if line.lower() in ('q', 'quit', 'b', 'back'):
        return
    sel = parse_region_select(line, n) if line else list(range(1, n + 1))
    if sel is None:
        sel = list(range(1, n + 1))
    for i in sel:
        vm, em = regs[i - 1]
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
                        vm, em = regs[i - 1]
                        print(f'  --- region {i}:')
                        show_basis(edges, em, vm, ext_attach=ext_attach)
                break
            asked = True
            F = parse_forced_lines(edges, line)
            if F is None:
                continue
            for i in sel:
                vm, em = regs[i - 1]
                print(f'  --- region {i}:')
                show_basis(edges, em, vm, F, ext_attach)


# Option 2: Lee-Pomeransky parametric representation per region: x_i ~ \lambda^{v_i}, v_i = -(2m+n), in input edge order.
def show_parametric(regs, nedge):
    print('  Lee-Pomeransky parametric representation (v_i = -(2m+n), input edge order):')
    for i, (vm, em) in enumerate(regs, 1):
        print(f'    R{i}: x_i ~ \\lambda^{{v_i}} for i = 1,2,...,{nedge}, with v = {tuple(scaling_of(m) for m in em)}')


# Option 3: group the regions by their softest-mode class and print each group.
def show_classify(regs, edges, kappa):
    groups = group_by_type(regs, kappa)
    order = type_order(kappa)
    labels = [sc_short(*mn) for mn in order] + ['C/H']
    total = 0
    for lab in labels:
        if lab not in groups:
            continue
        g = groups[lab]
        total += len(g)
        print()
        print(f'There are {len(g)} {lab} type regions:')
        for i, (vm, em) in enumerate(g, 1):
            print(f'  --- {lab} region {i} ---')
            print_region_with_vector(vm, em, edges, indent='    ')
    print()
    print(f'TOTAL: {total} regions')


# Option 4: group the regions by their scalar power (plus the numerator power N if given); computed silently, ascending order.
def show_group_by_power(regs, edges, ext_attach, extmode, numerator):
    groups = {}
    for i, (vm, em) in enumerate(regs, 1):
        results, _total, _L = indep_loops(edges, em, vm)
        a0, a1 = measure_power(results)
        n = 0
        if numerator is not None:
            ctx = region_context(edges, em, vm, ext_attach, extmode)
            n = numerator_power(numerator[0], ctx)
        groups.setdefault((a0 - integrand_power(em) + n, a1), []).append(i)
    for (p0, p1), rs in sorted(groups.items()):
        print()
        print(f'Region(s) with power = {fmt_power(p0, p1)}:')
        for i in rs:
            print(f'  R{i}: {em_sequence(regs[i - 1][1])}')


# Ask for the numerator polynomial (in K_i / externals; '1' = scalar); empty keeps the previous one; q/b cancels.
def ask_numerator(nedge, pnames, current):
    keep = 'scalar' if current is None else 'previous'
    while True:
        line = input(f'Numerator ("1" = scalar; e.g. (p1\\cdot K_1)(K_2\\cdot K_3); empty = keep {keep}; q/b = cancel) > ').strip()
        if line.lower() in ('q', 'quit', 'b', 'back'):
            return None, False
        if not line:
            if current is None:
                print('  numerator = 1 (scalar)')
            else:
                show = current[1] if len(current[1]) <= 160 else current[1][:157] + '...'
                print(f'  numerator = {show}')
            return current, True
        try:
            ast, warns = parse_numerator(line, nedge, pnames)
        except ValueError as e:
            print(f'  ! {e}')
            continue
        for w in warns:
            print(f'  ! note: {w}')
        if ast is None:
            print('  numerator = 1 (scalar)')
            return None, True
        text = fmt_ast(ast)
        show = text if len(text) <= 160 else text[:157] + '...'
        print(f'  numerator = {show}')
        return (ast, text), True


# Option 4 (follow-up): render a step-by-step derivation PDF for chosen regions (empty = all).
def offer_derivation_report(edges, regs, ext_attach, internal_lines, externals, extmode, numerator):
    line = input('Region numbers for the report (e.g. "3, 8--10"; empty = all; q/b = back) > ').strip()
    if line.lower() in ('q', 'quit', 'b', 'back'):
        return
    sel = parse_region_select(line, len(regs)) if line else list(range(1, len(regs) + 1))
    if sel is None:
        return
    print(f'  building the derivation report for {len(sel)} region(s) ...')
    try:
        pdf, prev, _lines = build_report(edges, regs, sel, ext_attach, internal_lines, externals, extmode=extmode, numerator=numerator)
    except Exception as e:
        print(f'  ! report generation failed: {e}')
        return
    print(f'  saved derivation report to: {pdf}')
    print(f'  first page preview: {prev}')


# Option 5: visualize the selected regions via wolframscript (can output PDF atlas or PNGs).
def visualize_regions(edges, regs, extmode, ext_attach):
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
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fri_out', 'regions_' + time.strftime('%Y%m%d-%H%M%S'))
    if fmt in ('p', 'png', 'files'):
        print(f'  rendering {len(sel)} region figure(s) via wolframscript ...')
        try:
            paths = region_plot.render_individual_pngs(edges, sorted({v for e in edges for v in e}), [(i, regs[i - 1]) for i in sel], ext_mode=extmode, ext_attach=ext_attach, outdir=outdir)
        except Exception as e:
            print(f'  ! region rendering failed: {e}')
            return
        print(f'  saved {len(paths)} region PNG(s) to: {outdir}')
    else:
        print(f'  building the PDF atlas for {len(sel)} region(s) via wolframscript ...')
        try:
            path = region_plot.render_combined_pdf(edges, sorted({v for e in edges for v in e}), [(i, regs[i - 1]) for i in sel], ext_mode=extmode, ext_attach=ext_attach, outdir=outdir)
        except Exception as e:
            print(f'  ! atlas rendering failed: {e}')
            return
        print(f'  saved atlas PDF to: {path}')


# ---------------------------------------------------------------- main loop
DEFAULT_EDGES = '[(1,5),(1,8),(2,5),(2,7),(3,6),(3,8),(4,6),(4,7),(5,6),(7,8)]'

# Interactive loop: read a graph, enumerate its regions, serve the menu; save results to fri_out/.
def main():
    print('=== facet-region browser ===')
    print('Input: graph topology + external momenta only.')
    while True:
        internal_lines = ask_edges('internal_lines (topology, edge list; e.g. 1-5,1-8,...; q/b = back)', DEFAULT_EDGES)
        if internal_lines is None:
            print('bye!')
            return
        externals = ask_externals()
        if externals is None:
            print('bye!')
            return
        try:
            verts = sorted({v for (a, b) in internal_lines for v in (a, b)} | {vv[0] for vv in externals.values()})
            edges = [tuple(sorted((a, b))) for (a, b) in internal_lines]
            ext_attach = {n: vv[0] for n, vv in externals.items()}
            extmode = {n: parse_mode(s) for n, (v, s) in externals.items()}
        except ValueError as e:
            print(f'  ! {e}')
            continue
        print(f'\n  enumerating all regions ({len(edges)} edges, {len(verts)} vertices) ...')
        t0 = time.time()
        regs = enumerate_regions(verts, edges, ext_attach, extmode)
        dt = time.time() - t0
        edge_seq = ', '.join(f'({u},{v})' for u, v in edges)
        print(f'  {len(regs)} regions in {dt:.1f}s (presented in terms of the edge modes {edge_seq}):')
        for i, (vm, em) in enumerate(regs, 1):
            print(f'    R{i}: {em_sequence(em)}')
        kappa = kappa_of(extmode)
        numerator = None
        while True:
            print('  Options:')
            print('    1) Inspect specific regions')
            print('    2) Show Lee-Pomeransky parametric representation')
            print('    3) Group these regions by their characteristic mode')
            print('    4) Group these regions by their power')
            print('    5) Visualize these regions (PDF atlas / PNGs)')
            opt = input('  (1/2/3/4/5; empty/q/b = done with this graph) > ').strip().lower()
            if opt in ('', 'q', 'quit', 'b', 'back'):
                break
            if opt == '1':
                inspect_regions(edges, regs, ext_attach)
            elif opt == '2':
                show_parametric(regs, len(edges))
            elif opt == '3':
                show_classify(regs, edges, kappa)
            elif opt == '4':
                num, ok = ask_numerator(len(edges), set(extmode), numerator)
                if ok:
                    numerator = num
                    show_group_by_power(regs, edges, ext_attach, extmode, numerator)
                    if ask_yn('Show the detailed derivation as a PDF report? (y/n) [n] > '):
                        offer_derivation_report(edges, regs, ext_attach, internal_lines, externals, extmode, numerator)
            elif opt == '5':
                visualize_regions(edges, regs, extmode, ext_attach)
            else:
                print('  ! enter 1, 2, 3, 4, 5, or q')
        # save
        fdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fri_out')
        os.makedirs(fdir, exist_ok=True)
        fname = os.path.join(fdir, time.strftime('%Y%m%d-%H%M%S') + '.txt')
        with open(fname, 'w') as f:
            f.write(f'# wide-angle FRI regions - {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'# internal lines: {internal_lines}\n')
            f.write(f'# externals: {externals}\n')
            f.write(f'# regions: {len(regs)} ({dt:.1f}s)\n\n')
            for i, (vm, em) in enumerate(regs, 1):
                sc = ', '.join(map(str, [scaling_of(m) for m in em]))
                f.write(f'R{i:3d}: scaling ({sc}, 1)\n')
                f.write(f'      em: {[ms(m) for m in em]}\n')
                vmtxt = ', '.join(f'{v}: {ms(vm[v])}' for v in sorted(vm))
                f.write(f'      vm: {{{vmtxt}}}\n')
        print(f'  saved to: {fname}')
        print('=' * 72)
        if not ask_yn('Another graph? (y/n) [n] > '):
            break

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.default_int_handler)  # keep Ctrl+C working
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nbye!')
