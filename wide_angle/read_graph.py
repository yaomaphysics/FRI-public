#!/usr/bin/env python3
"""read_graph.py — interactive graph + external-momenta reader for the Facet Region Interpreter (unitarity-cut approach).

Step 1: ask for the edges.
    Format:  [a,b],[c,d],...     e.g. [1,2],[1,3],[2,3] for a triangle graph
    Each [x,y] is an undirected line between vertex x and vertex y.
    Order within a pair and order of pairs do not matter.
Step 2: ask for the external momenta.
    Format:  ['p1',1,C1],['p2',2,C2^2],['p3',3,C3^\\infty],['p4',4,C4^\\infty]
    Each entry: [name, vertex, mode].

The program echoes a canonical summary (vertices, sorted edges, degrees, external attachments, parsed modes) and asks for confirmation, so the user can verify that the graph was read correctly.

Usage: python3 read_graph.py
"""
import re
import sys
from collections import defaultdict

from primitives import INF, parse_mode, mode_str   # mode parsing lives in the base layer

# ---------------- mode classes & short labels ----------------
# Short label: (1,1)->SC, (1,2)->SC^2, (2,1)->S^2C, (m,0)->S^m.
def sc_short(m, n):
    if n == 0:
        return 'S' if m == 1 else f'S^{m}'
    if m == 1:
        return 'SC' if n == 1 else f'SC^{n}'
    if n == 1:
        return f'S^{m}C'
    return f'S^{m}C^{n}'


# All (m,n) with m>=1, n>=0, m+n<=kappa, softest-first: (sigma=m+n desc, then m desc).
def type_order(kappa):
    lst = []
    for m in range(1, kappa + 1):
        for n in range(0, kappa + 1 - m):
            lst.append((m, n))
    lst.sort(key=lambda mn: (-(mn[0] + mn[1]), -mn[0]))
    return lst


# Classify a region by its softest mode present (exclusion-style order); considers ALL mode components — vertex and edge modes.
# (A soft component such as SC can live on a propagator.)
def classify(vm, em, kappa):
    order = type_order(kappa)
    present = set()
    for v, md in vm.items():
        m, n, i = md
        if m >= 1:
            present.add((m, n))
    for md in em:
        m, n, i = md
        if m >= 1:
            present.add((m, n))
    for mn in order:
        if mn in present:
            return sc_short(*mn)
    return 'C/H'


# Group regions (vm, em) by their softest-mode class label (classify).
def group_by_type(results, kappa):
    groups = defaultdict(list)
    for vm, em in results:
        groups[classify(vm, em, kappa)].append((vm, em))
    return groups

# ---------------- parsing of user input ----------------
# Parse '[1,6],[1,5],...' into a sorted list of frozenset pairs.
def parse_edges(raw):
    pairs = re.findall(r'\[\s*([^,\]^]+?)\s*,\s*([^\]^]+?)\s*(?:,\s*(.*?)\s*)?\]', raw)
    edges = []
    for a, b, _mcol in pairs:
        a, b = a.strip(), b.strip()
        # try to interpret labels as ints when possible
        try:
            a = int(a)
        except ValueError:
            pass
        try:
            b = int(b)
        except ValueError:
            pass
        edges.append(frozenset((a, b)))
    # deduplicate (undirected, order-free)
    edges = list(dict.fromkeys(edges))
    return sorted(edges, key=lambda e: tuple(sorted(e, key=str)))

# Parse "['p1',1,C1],['p2',2,C2^2],..." into [(name, vertex, mode_tuple)]; tolerant (name may be quoted or bare, whitespace flexible).
# Returns (entries, n_seen, n_parsed) so the caller can warn when some entries failed to parse.
def parse_externals(raw):
    entries = re.findall(r"\[\s*(?:'([^']+)'|([^',\]]+))\s*,\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]", raw)
    out = []
    for qname, bname, v, md in entries:
        name = qname if qname else bname.strip()
        v = v.strip()
        try:
            v = int(v)
        except ValueError:
            pass
        out.append((name.strip(), v, parse_mode(md)))
    n_seen = len(re.findall(r"\[", raw)) if raw.strip() else 0
    return out, n_seen, len(out)


# Derive the possible-modes structure from the external momenta; returns (kappa, softest_mode, (c_modes, s_modes, sc_modes), types).
# Paper input (all derived, no region data):
#   (i) mode-resolution: internal modes are S^m C_i^n like the externals;
#   (ii) no-cascading: kappa = max{m_i + n_i}, S^kappa softest;
#   (iii) IR-compat: S^kappa must be a degree-kappa messenger (>=3 harder components); SC_i = meet of two harder C-modes (or SC external);
#         pure S^m (m<kappa) adjacent to >=2 jets; C/H has no soft component.
def analyze_kinematics(exts):
    # kappa = max over external modes of (m + n), finite n only
    kappas = [m + n for (_, _, (m, n, i)) in exts if n < INF and (m, n, i) != (0, 0, 0)]
    kappa = max(kappas) if kappas else 1
    softest = (kappa, 0, 0)   # S^kappa

    # directions present in the external momenta
    dirs = sorted({i for (_, _, (m, n, i)) in exts if i != 0})

    # ---- complete list of allowed internal modes ----
    # No-cascading corollary: every internal mode is harder-or-equal to the softest mode S^kappa, i.e. sigma = m + n <= kappa (also V = 2m+n <= 2*kappa).
    # Families (i runs over the external directions):
    #   C_i^n :  m = 0, 1 <= n <= kappa
    #   S^m   :  n = 0, 1 <= m <= kappa
    #   S^m C_i^n (SC family):  m >= 1, n >= 1, m + n <= kappa
    c_modes = [(0, n, i) for n in range(1, kappa + 1) for i in dirs]
    s_modes = [(m, 0, 0) for m in range(1, kappa + 1)]
    sc_modes = [(m, n, i) for m in range(1, kappa + 1) for n in range(1, kappa + 1 - m) for i in dirs]

    # SC seeds from externals
    sc_seeds = [(name, md) for (name, _, md) in exts if md[0] >= 1 and md[1] >= 1]

    # ---- classification: one class per softest-mode (m,n) combination ----
    # Order from softest to hardest: sigma = m+n descending, then m descending (for fixed sigma, larger m means larger V = 2m+n, i.e. softer).
    # Each (m,n) with m >= 1, n >= 0, m+n <= kappa gets its own class, defined by exclusion: "no softer class's mode, but contains (m,n)".
    # Pure C modes (m = 0) never define a class; they fall into C/H.
    mn_list = type_order(kappa)

    types = []
    for idx, (m, n) in enumerate(mn_list):
        label = sc_short(m, n) + '-type'
        if idx == 0:
            desc = f'regions containing {sc_short(m, n)}'
        else:
            softer = ', '.join(sc_short(mm, nn) for (mm, nn) in mn_list[:idx])
            desc = (f'regions containing no {softer}, but containing {sc_short(m, n)}')
        types.append((label, desc))
    # C/H class (no soft component at all)
    c_list = ', '.join(mode_str(md) for md in sorted(c_modes, key=lambda md: (md[1], md[2])))
    types.append(('C/H-type', f'regions containing none of the above modes, i.e. only C-type and H modes (C modes: {c_list}, i in {dirs})'))
    return kappa, softest, (c_modes, s_modes, sc_modes), types

# ---------------- interactive input (reusable) ----------------
# Interactive graph input: returns (verts, edges, ext_attach, ext_mode).
def input_graph():
    # Ensure arrow keys / line editing work (readline may not be active by default in some terminals / wrappers).
    try:
        import readline  # noqa: F401
    except ImportError:
        pass
    print('Step 1: please enter the edges of the graph.')
    print('Format:  [a,b],[c,d],...   (e.g. [1,2],[1,3],[2,3] for a triangle graph)')
    print('Each [x,y] is an undirected line between vertex x and vertex y.')
    raw = input('Edges> ').strip()
    if not raw:
        print('ERROR: no input.'); sys.exit(1)
    edges = parse_edges(raw)
    verts = set()
    for e in edges:
        verts |= set(e)
    if not edges:
        print('ERROR: could not parse any edge.'); sys.exit(1)

    print()
    print('Step 2: please enter the external momenta.')
    print("Format:  ['p1',1,C1],['p2',2,C2^2],...   (name, vertex, mode)")
    print('Supported modes: H, C_i, C_i^n, C_i^inf (or C_i^\\infty), S, S^m, S^m C_i^n.')
    raw2 = input('Externals> ').strip()
    if not raw2:
        print('ERROR: no input.'); sys.exit(1)
    exts, n_seen, n_parsed = parse_externals(raw2)
    if not exts:
        print('ERROR: could not parse any external momentum.'); sys.exit(1)
    if n_parsed < n_seen:
        print(f'WARNING: parsed {n_parsed} of {n_seen} external entries. Each must look like [name,vertex,mode] with the name quoted or bare, e.g. [p1,1,C1] or [\'p1\',1,C1]. Continuing with {n_parsed}.')
    ext_attach = {n: v for (n, v, md) in exts}
    ext_mode = {n: md for (n, v, md) in exts}
    return sorted(verts, key=str), edges, ext_attach, ext_mode

# ---------------- main interactive flow ----------------
def main():
    print('=' * 64)
    print('Facet Region Interpreter — graph reader')
    print('=' * 64)
    verts, edges, ext_attach, ext_mode = input_graph()
    exts = [(n, ext_attach[n], ext_mode[n]) for n in ext_attach]

    # ---- validation ----
    problems = []
    for name, v, md in exts:
        if v not in verts:
            problems.append(f'  external {name} attaches to vertex {v}, which is not in the graph')
    if problems:
        print()
        print('WARNING: inconsistencies found:')
        for p in problems:
            print(p)
        print('(continuing; fix the input if this is unexpected)')

    # ---- canonical echo for verification ----
    print()
    print('=' * 64)
    print('READBACK — please verify the graph was read correctly.')
    print('=' * 64)
    print(f'Number of vertices: {len(verts)}')
    print(f'Vertices (sorted):  {sorted(verts, key=str)}')
    print(f'Number of edges:    {len(edges)}')
    print('Edges (canonical, undirected, deduplicated):')
    for e in edges:
        a, b = sorted(e, key=str)
        print(f'   [{a},{b}]')
    # degrees
    deg = {v: 0 for v in verts}
    for e in edges:
        for v in e:
            deg[v] += 1
    print('Degrees:')
    for v in sorted(verts, key=str):
        print(f'   vertex {v}: degree {deg[v]}')
    # external momenta
    print(f'External momenta ({len(exts)}):')
    for name, v, md in exts:
        print(f"   {name}: attached at vertex {v}, mode {mode_str(md)}  (raw {md})")

    # ---- confirmation ----
    print()
    ans = input('Is this correct? [Y/n] ').strip().lower()
    if ans in ('', 'y', 'yes'):
        print()
        print('Graph accepted.')
        print()
        print('Summary for the algorithm stage:')
        print(f'  V = {sorted(verts, key=str)}')
        print(f'  E = {[tuple(sorted(e, key=str)) for e in edges]}')
        print(f"  EXT = {{name: (vertex, mode)}} = {{ {', '.join(f"'{n}': ({v}, {mode_str(md)})" for n, v, md in exts)} }}")

        # ---- possible-modes analysis ----
        kappa, softest, (c_modes, s_modes, sc_modes), types = analyze_kinematics(exts)
        print()
        print('=' * 64)
        print('POSSIBLE MODES ANALYSIS')
        print('=' * 64)
        print(f'The possibly softest mode in this expansion is {mode_str(softest)} (kappa = {kappa}).')
        print()
        print(f'All modes harder-or-equal to {mode_str(softest)} (softness exponent m+n <= {kappa}):')
        print(f'  H')
        if c_modes:
            print(f'  C-type: ' + ', '.join(mode_str(md) for md in sorted(c_modes, key=lambda md: (md[1], md[2]))))
        if s_modes:
            print(f'  S-type: ' + ', '.join(mode_str(md) for md in s_modes))
        if sc_modes:
            print(f'  SC-type (S^m C_i^n, m,n >= 1): ' + ', '.join(mode_str(md) for md in sorted(sc_modes, key=lambda md: (md[0], md[1]))))
        print()
        print('Therefore, we classify the regions into the following types:')
        for i, (label, desc) in enumerate(types, 1):
            print(f'  {i}. {label}: {desc}')
        print()
        ans2 = input('Continue? [Y/n] ').strip().lower()
        if ans2 in ('', 'y', 'yes'):
            print()
            print('OK — proceeding to the region-construction algorithm (next step).')
        else:
            print('Stopped.')
    else:
        print('Aborted — please re-run and re-enter the graph.')

if __name__ == '__main__':
    main()
