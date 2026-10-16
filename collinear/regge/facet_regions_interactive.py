#!/usr/bin/env python3
"""
Interactive Regge-limit FRI region enumerator.

Usage: python3 facet_regions_interactive.py [-v]

At startup you choose a kinematics (k0..k2 are shown as local examples, default k1; kx enters the four external-momentum modes p1..p4 as powers); its ext_mode is fed into the FRI pipeline (cuts -> overlay -> Glauber adjustment -> subgraph requirements -> IR compatibility).
Enumeration runs on the pruned skeleton enumerator (enumerator.py).

Type an edge list to enumerate (external momenta p1..p4 attach at vertices 1, 2, 3, 4).
At the edge prompt: 'kin kx' (or a k0..k2 label) switches kinematics, 'v' toggles the enumeration statistics, 'q'/'b' quits.
Start with -v to have the statistics on from the beginning.
After enumeration a menu offers:
    1) Inspect specific regions — per-mode subgraphs + loop numbers (Σ r_X vs L), then optionally a concrete independent-loop-momentum basis (default or forced lines),
    2) Lee-Pomeransky parametric representation (v_i = -V, input edge order; trailing entry = expansion scale),
    3) Classification by characteristic (softest) mode.
    4) Visualization: one PDF atlas (default) or PNG figures with the per-mode colour scheme (region_plot.py; saved under fri_out/regions_*/).
Semihard loops use the γ_sH ∪ γ_G rule.
Results are saved to fri_out/<timestamp>.txt.

Edge-list input formats (all accepted):
    "1-5,3-5,5-6,2-6,4-6,1-7,3-7,7-8,2-8,4-8"
    "(1,5),(3,5),(5,6),..."
    "[1,5],[3,5],[5,6],..."
    "1 5; 3 5; 5 6; ..."
"""
import sys, os, time, re, signal

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from primitives import to_scaling, mode_components, INF, to_old, leg_check, C, CC, FAM_LEGS
from enumerator import skel_regions   # pruned skeleton enumerator
from primitives import to_mode          # mode strings are plain literals ('C13', 'sH', ...) everywhere
from indep_loops import show_basis     # indep loops + edge momenta
from mode_levels import Predictor, KIN_EXT, _skey
from collections import defaultdict

DEFAULT_MOMENTA = ['C1∞C13', 'C2∞C24', 'C3∞C13', 'C4∞C24']
KIN_EXAMPLES = ['k0', 'k1', 'k2']
# Local illustration of the three example kinematics (the external modes come from mode_levels.KIN_EXT).
KIN_NOTES = {
    'k0': 'p_i^2 ~ λ for all i',
    'k1': 'p_i^2 = 0 for all i (lightlike, original case)',
    'k2': 'p1^2 ~ λ, others lightlike',
}


# Ask a y/n question (empty = n); any other input re-asks.
def ask_yn(prompt):
    while True:
        ans = input(prompt).strip().lower() or 'n'
        if ans in ('y', 'n'):
            return ans == 'y'
        print('  ! answer y or n (empty = n)')


# Parse an edge list leniently: accepts '1-5,3-5,...', '(1,5),(3,5),...' or '[(1,5),(3,5),...]'.
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


# '3, 8--10' -> [3, 4, 8, 10] (1-based). Supports ',' separators and '--' / '-' ranges. Returns None if any number is out of range.
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


# Number of connected components of the (multi)graph.
def _components(verts, edges):
    adj = {v: [] for v in verts}
    for (a, b) in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen = set()
    n = 0
    for v in verts:
        if v in seen:
            continue
        n += 1
        stack = [v]
        while stack:
            x = stack.pop()
            for w in adj.get(x, ()):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
    return n


# Per-mode subgraphs {V_X}, {E_X} with loop number r_X from the 1VI (one-vertex-irreducible) blocks of the contracted X subgraph (sH uses the rank of gamma_sH U gamma_G instead), plus the sum check against L.
def show_mode_subgraphs(edges, em, vm):
    verts = sorted({v for e in edges for v in e})
    L = len(edges) - len(verts) + 1
    modes = sorted(set(em) | set(vm.values()))
    tot = 0
    for mode in modes:
        E_idx = [i for i, m in enumerate(em) if m == mode]
        V = sorted(v for v in verts if vm.get(v) == mode)
        Es = '{' + ', '.join(f'{i}:{edges[i]}' for i in E_idx) + '}'
        Vs = '{' + ', '.join(str(v) for v in V) + '}'
        if mode == 'sH':
            # semihard loops = rank of gamma_{sH} U gamma_{G} (plain subgraph of the original graph, no aux contraction)
            idx = [i for i, m in enumerate(em) if m in ('sH', 'G')]
            sub_edges = [edges[i] for i in idx]
            sub_verts = sorted({v for e in sub_edges for v in e})
            r = (len(sub_edges) - len(sub_verts) + _components(sub_verts, sub_edges))
            note = '  [semihard loops = loops of gamma_sH U gamma_G]'
        else:
            blocks = mode_components(mode, vm, em, edges, verts)
            r = sum(len(idxs) - len(bv) + 1 for (bv, be, idxs) in blocks)
            note = ''
        tot += r
        print(f'  mode {mode}: {{vertices: {Vs}, edges: {Es}}}   loop number = {r}{note}')
    if tot == L:
        flag = '✓'
    else:
        flag = '✗ MISMATCH'
    print(f'  sum r_X = {tot}  vs  L = {L}  {flag}')


# '[2,5],[7,8]' -> sorted edge indices (order-free endpoints); for parallel edges a pair forces all copies. Returns None if nothing parses or an edge is unknown.
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


# Softness order for classification, softest first: lattice modes with m >= 1 by decreasing virtuality V (then m, n, name), then sH, then G.
def _softness_key(mode):
    if mode == 'sH':
        return (1, 0, 0, 0, mode)
    if mode == 'G':
        return (2, 0, 0, 0, mode)
    try:
        M = to_mode(mode)
        return (0, -M.V, -M.m, -M.n, mode)
    except Exception:
        return (0, -99, 0, 0, mode)


# Softest characteristic mode of a region (exclusion-style): the softest lattice mode with m >= 1 present in em/vm; if none, sH; if none, G; else 'C/H'.
def classify(vm, em):
    cands = set(em) | set(vm.values())
    soft = [m for m in cands if m not in ('sH', 'G') and to_mode(m).m >= 1]
    if soft:
        return min(soft, key=_softness_key)
    if 'sH' in cands:
        return 'sH'
    if 'G' in cands:
        return 'G' 
    return 'C/H'


# Option 2: Lee-Pomeransky parametric representation per region: x_i ~ \lambda^{v_i} with v_i = -V(𝒳(e)), edge order.
def show_parametric(regs):
    print('  Lee-Pomeransky parametric representation (v_i = -V, input edge order; trailing entry = expansion scale):')
    for i, (cut13, cut24, cut1, cut3, cut2, cut4, vm, em) in enumerate(regs, 1):
        sc = to_scaling(em)
        print(f'    R{i}: x_i ~ \\lambda^{{v_i}} for i = 1,2,...,{len(sc)}, with v = {sc}')


# Option 3: classify regions by their characteristic (softest) mode, grouped by type (softest first).
def show_classify(regs, edges):
    groups = defaultdict(list)
    for r in regs:
        cut13, cut24, cut1, cut3, cut2, cut4, vm, em = r
        groups[classify(vm, em)].append(r)
    order = sorted(groups, key=lambda lab: (99,) if lab == 'C/H' else _softness_key(lab))
    total = 0
    for lab in order:
        g = groups[lab]
        total += len(g)
        print()
        print(f'There are {len(g)} {lab} type regions:')
        for i, (cut13, cut24, cut1, cut3, cut2, cut4, vm, em) in enumerate(g, 1):
            print(f'  --- {lab} region {i} ---')
            print(f'    em = {em}')
            print(f'    scaling = {to_scaling(em)}')
    print(f'\nTOTAL: {total} regions')


# Option 1: pick regions -> per-mode subgraphs with loop numbers, then optionally a concrete basis / forced lines / line momenta.
def inspect_regions(edges, regs, ext_attach):
    n = len(regs)
    line = input('Region numbers (e.g. "3, 8--10" represents regions 3, 8, 9, and 10; empty = all; q/b = back) > ').strip()
    if line.lower() in ('q', 'quit', 'b', 'back'):
        return
    sel = parse_region_select(line, n) if line else list(range(1, n + 1))
    if sel is None:
        sel = list(range(1, n + 1))
    for i in sel:
        cut13, cut24, cut1, cut3, cut2, cut4, vm, em = regs[i - 1]
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
                        cut13, cut24, cut1, cut3, cut2, cut4, vm, em = regs[i - 1]
                        print(f'  --- region {i}:')
                        show_basis(edges, em, vm, ext_attach=ext_attach)
                break
            asked = True
            F = parse_forced_lines(edges, line)
            if F is None:
                continue
            for i in sel:
                cut13, cut24, cut1, cut3, cut2, cut4, vm, em = regs[i - 1]
                print(f'  --- region {i}:')
                show_basis(edges, em, vm, F, ext_attach)


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

# Format the external modes as 'p1=C13, p2=C24, ...'.
def fmt_ext(ext_mode):
    return ', '.join(f'{n}={ext_mode[n]}' for n in ('p1', 'p2', 'p3', 'p4'))


# input the external modes as powers.
def ask_momenta():
    print('Input the external modes (only the powers, selected from {0, 1, ..., ' + chr(92) + 'infty})')
    out = []
    # the four legs of the 2->2 kinematics are collinear-pair legs: power 0 is C_ij itself
    lo = 0
    for i, dflt in enumerate(DEFAULT_MOMENTA, 1):
        tmpl = dflt.replace('∞', '^?')
        while True:
            try:
                s = input('  p%d mode [%s] (only input the power) > ' % (i, tmpl)).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if s in ('q', 'quit', 'exit', 'b', 'back'):
                return None
            if s in ('inf', 'infty', 'os', 'on-shell', chr(92) + 'infty'):
                mstr = tmpl.replace('^?', '∞').replace('^∞', '∞')
            elif s.isdigit() and int(s) >= lo:
                mstr = tmpl.replace('?', s)
            else:
                print('    [error] enter an integer >= %d, or ' % lo + chr(92) + 'infty')
                continue
            try:
                m = to_mode(mstr)
            except Exception:
                print('    [error] enter an integer >= %d, or ' % lo + chr(92) + 'infty')
                continue
            if not leg_check(i, m):
                print('  ↑ not allowed for p%d — try again' % i)
                continue
            out.append(mstr)
            break
    return out


# Short label for a kinematics: a preset name, or the custom external modes.
def kin_label(kin):
    return kin if isinstance(kin, str) else 'custom ' + fmt_ext(kin)


# Display note for a kinematics: the preset note, or the custom external modes.
def kin_note(kin):
    return KIN_NOTES[kin] if isinstance(kin, str) else fmt_ext(kin)


# The external-mode map of a kinematics: a preset name (mode_levels.KIN_EXT) or a custom map.
def kin_ext_mode(kin):
    if not isinstance(kin, str):
        return kin
    m = KIN_EXT[kin]
    return {'p%d' % i: m[i] for i in sorted(m)}




# Print the example kinematics; kx enters a custom pattern.
def print_kin_menu():
    print('Input kinematics:')
    print('For example,')
    for k in KIN_EXAMPLES:
        print(f'  {k}: {KIN_NOTES[k]};')


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
            lst = ask_momenta()
            if lst is not None:
                return {'p%d' % (i + 1): m for i, m in enumerate(lst)}
            print_kin_menu()
            continue
        print(f'  [error] unknown kinematics "{s}" — choose from {", ".join(KIN_EXAMPLES)}, or kx (q/b = back)')


# Enumerate the regions of one graph under the external modes and print them, then offer the inspection menu.
def run_graph(edges_raw, ext_mode, verbose=False):
    edges = [tuple(e) for e in edges_raw]
    verts = sorted({v for e in edges for v in e})
    missing = [v for v in (1, 2, 3, 4) if v not in verts]
    if missing:
        print(f'  [warning] external-leg vertices {missing} not in the graph!  Regge 2->2 needs vertices 1,2,3,4.')
        return
    ext_attach = {'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4}
    L = len(edges) - len(verts) + 1
    print(f'  kinematics: {fmt_ext(ext_mode)}')
    print(f'  graph: {len(edges)} edges, {len(verts)} vertices, L = {L} loops')
    print(f'  edges: {edges}')
    if L >= 5:
        print('  [warning] L >= 5 may be slow')
    t0 = time.time()
    try:
        regs, cnt, depth = skel_regions(edges, verts, ext_attach, ext_mode)
    except (Exception, SystemExit) as e:
        print(f'  [error] skeleton engine could not handle this input: {e}')
        return
    dt = time.time() - t0
    regs = sorted(regs, key=lambda r: (to_scaling(r[7]), tuple(str(m) for m in r[7])))
    print(f'  FRI regions: {len(regs)}  ({dt:.1f}s)')
    # Enumeration statistics (hidden debug switch: 'v' at the edge prompt / -v at startup).
    if verbose:
        rej = ', '.join(f'{k}={v}' for k, v in sorted(cnt.items()) if k != 'combos')
        print(f'  [stats] combos={cnt["combos"]}  kept={len(regs)}  depth={"/".join(str(x) for x in depth)}  counters: {rej or "none"}')
    scal = set()
    for i, (cut13, cut24, cut1, cut3, cut2, cut4, vm, em) in enumerate(regs, 1):
        sc = to_scaling(em)
        scal.add(sc)
        ref = ''
        if cut1: ref += f'  C1C13={sorted(cut1)}'
        if cut3: ref += f'  C3C13={sorted(cut3)}'
        if cut2: ref += f'  C2C24={sorted(cut2)}'
        if cut4: ref += f'  C4C24={sorted(cut4)}'
        print(f'  R{i:3d}: scaling {fmt_scaling(sc)}')
        print(f'        cuts: C13={sorted(cut13)}  C24={sorted(cut24)}{ref}')
        print(f'        em: {em}')
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
            inspect_regions(edges, regs, ext_attach)
        elif opt == '2':
            show_parametric(regs)
        elif opt == '3':
            show_classify(regs, edges)
        elif opt == '4':
            visualize_regions(edges, regs, ext_mode)
        else:
            print('  ! enter 1, 2, 3, 4, or q')
    fdir = os.path.join(BASE, 'fri_out')
    os.makedirs(fdir, exist_ok=True)
    fname = os.path.join(fdir, time.strftime('%Y%m%d-%H%M%S') + '.txt')
    with open(fname, 'w') as f:
        f.write(f'# Regge FRI regions - {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# kinematics: {fmt_ext(ext_mode)}\n')
        f.write(f'# edges: {edges}  ({len(edges)} edges, {len(verts)} verts, L={L})\n')
        f.write(f'# FRI regions: {len(regs)} ({dt:.1f}s)\n\n')
        for i, (cut13, cut24, cut1, cut3, cut2, cut4, vm, em) in enumerate(regs, 1):
            ref = ''
            if cut1: ref += f'  C1C13={sorted(cut1)}'
            if cut3: ref += f'  C3C13={sorted(cut3)}'
            if cut2: ref += f'  C2C24={sorted(cut2)}'
            if cut4: ref += f'  C4C24={sorted(cut4)}'
            f.write(f'R{i:3d}: scaling {fmt_scaling(to_scaling(em))}\n')
            f.write(f'      cuts: C13={sorted(cut13)}  C24={sorted(cut24)}{ref}\n')
            f.write(f'      em: {em}\n')
            vmtxt = ', '.join(f'{v}: {vm[v]}' for v in sorted(vm))
            f.write(f'      vm: {{{vmtxt}}}\n')
    print(f'  saved to: {fname}')

LMAX = 10 # Stepper display cap (levels are shown up to L = LMAX).

BANNER = '''\
==============================================================
  What modes appear at which loop level?   (regge)
--------------------------------------------------------------
  Enter the four external-momentum modes, then:
    SPACE = step +1 loop      ENTER = jump to next non-empty loop
    A     = cumulative table  R     = report (L=0 .. current level)
    Q/B   = quit
  (levels are capped at L = %d)
==============================================================''' % LMAX


# single keypress on a tty; line-based fallback otherwise ('' line = space).
def read_key():
    if not sys.stdin.isatty():
        try:
            s = input('▶ (SPACE=advance, J=jump, A=all, R=report, Q/B=quit): ')
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


# Display state for one kinematics: extends the ladder on demand and renders each new edge scaling's origin.
class Stepper:
    # Store the entered kinematics; pred/levels/by_name/computed are (re)built lazily by ensure().
    def __init__(self, ext_modes):
        self.ext = dict(ext_modes)
        self.pred = None
        self.levels = {}
        self.by_name = {}
        self.computed = -1

    # Extend the computed ladder to level k (recompute with the larger cap when the display has advanced).
    def ensure(self, k):
        k = min(k, LMAX)
        if k <= self.computed:
            return
        self.pred = Predictor(self.ext, Lmax=k)
        self.pred.run()
        self.levels = self.pred.levels()
        self.by_name = {to_old(x): x for x in self.pred.cost if x.n != INF}
        self.computed = k

    # New-mode names listed at level k ([] when the level is empty).
    def modes_at(self, k):
        return self.levels.get(k, [])

    # Cheapest other-family power-(m-1) targets of a messenger mode, in ladder order.
    def messenger_targets(self, x):
        other = 24 if x.fam == 13 else 13
        ms = x.m - 1
        seen, tset = set(), []
        for y in [C(other, ms)] + [CC(leg, ms) for leg in FAM_LEGS[other]]:
            if y in seen or y not in self.pred.cost:
                continue
            seen.add(y)
            tset.append(y)
        cbest = min((self.pred.cost[y] for y in tset), default=None)
        if cbest is None or cbest + 2 != self.pred.cost[x]:
            return [], None
        return sorted((y for y in tset if self.pred.cost[y] == cbest), key=_skey), cbest


# Ask for the kinematics, then run the stepper's key loop.
def run_ladder():
    print(BANNER)
    lst = ask_momenta()
    if lst is None:
        print('bye 👋')
        return
    ext = {i + 1: m for i, m in enumerate(lst)}
    print('momenta: ' + ', '.join('p%d=%s' % (i, ext[i]) for i in sorted(ext)))
    st = Stepper(ext)
    st.ensure(0)
    print('L=0 seeds: %s' % ', '.join(st.modes_at(0)))

    cur = 0

    # Print level k's new edge scalings with their origins, plus the cumulative count.
    def show(k):
        st.ensure(k)
        names = st.modes_at(k)
        print()
        if names:
            n = len(names)
            print('L=%d — %d new edge scaling%s:' % (k, n, 's' if n != 1 else ''))
            for nm in names:
                print('   %s' % nm)
        else:
            print('L=%d — (no new edge scalings)' % k)
        n_le = sum(len(st.modes_at(lv)) for lv in st.levels if lv <= k)
        print('   cumulative: %d edge scalings (≤L=%d) | SPACE=+1, ENTER=next non-empty, A=all, R=report, Q/B=quit' % (n_le, k))

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
                print('  (no new edge scalings up to L=%d)' % LMAX)
            else:
                cur = hit
                show(cur)
        elif k in ('a', 'A'):
            st.ensure(cur)
            print()
            for lv in sorted(set(st.levels)):
                if lv > cur:
                    break
                print('L=%-3d: %s' % (lv, ', '.join(st.modes_at(lv))))
        elif k in ('r', 'R'):
            import mode_report
            pdf, _ = mode_report.build_report(st, cur)
            print('  saved mode-ladder report to: %s' % pdf)
        else:
            print('  (SPACE=advance, ENTER=jump, A=all, R=report, Q/B=quit)')

# Entry point: interactive loop over graphs; -v/--verbose starts with the enumeration statistics on.
def main():
    # hidden mode: -l/--ladder runs the mode-ladder stepper instead of the graph browser
    if any(a in ('-l', '--ladder') for a in sys.argv[1:]):
        run_ladder()
        return
    # Hidden debug switch (not shown to the user): -v/--verbose starts with the enumeration statistics on.
    verbose = any(a in ('-v', '--verbose') for a in sys.argv[1:])
    print('=' * 72)
    print('Regge-limit FRI region enumerator')
    print('external momenta: p1@1, p2@2, p3@3, p4@4')
    print('=' * 72)
    kin = choose_kinematics()
    if kin is None:
        print('bye!')
        return
    print(f'-> using kinematics {kin_label(kin)} ({kin_note(kin)})')
    while True:
        try:
            s = input(f'\nedge list for {kin_label(kin)} (e.g., the CrownTT, 1-5,3-5,5-6,2-6,4-6,1-7,3-7,7-8,2-8,4-8)> ').strip()
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
                kin = k
                print(f'-> kinematics switched to {kin_label(kin)} ({kin_note(kin)})')
            elif k == 'kx':
                new = ask_momenta()
                if new is not None:
                    kin = new
                    print(f'-> kinematics switched to {kin_label(kin)} ({kin_note(kin)})')
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
        run_graph(edges, kin_ext_mode(kin), verbose)
        print(f'  (this graph took {time.time() - t0:.1f}s total)')


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.default_int_handler)  # keep Ctrl+C working even if the launcher had ignored SIGINT
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nbye!')
