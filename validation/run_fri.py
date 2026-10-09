#!/usr/bin/env python3
"""run_fri.py — recompute FRI's region lists for the cases in this validation
package.  No external dependencies: only the FRI code from the main repository
(../wide_angle) and the Python standard library are used; pySecDec is NOT
needed.

Usage examples (run from this directory):

    python3 run_fri.py --batch graphs/random/5_legs/rand1000_3l_no2v.json
    python3 run_fri.py --batch graphs/random/5_legs/rand1000_3l_no2v.json --graph 0
    python3 run_fri.py --book graphs/hand_built/lightlike_2to2.txt --only CrownST_k2
    python3 run_fri.py --all

Options: --graph N (batch: a single graph), --kin 0,1 (subset of kinematics),
--only NAME (book: a single case), --full (print the complete region lists).
Region counts are checked against the records in results/ automatically.
"""
import argparse, ast, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'wide_angle'))

import skeleton as SG                      # noqa: E402
from primitives import mode_str            # noqa: E402

DATA = json.load(open(os.path.join(HERE, 'fri_runner_data.json')))

BATCHES = [
    'graphs/random/4_legs/rand1000_4l_v8-10_no2v.json',
    'graphs/random/4_legs/rand200_4l_no2v_v8-10.json',
    'graphs/random/4_legs/rand200_5l_v10-12_no2v.json',
    'graphs/random/5_legs/rand1000_3l_no2v.json',
    'graphs/random/5_legs/rand700_4l_no2v.json',
    'graphs/random/5_legs/rand100_5l_no2v.json',
    'graphs/random/5_legs/rand500_5l_no2v.json',
]
BOOKS = [
    'graphs/hand_built/lightlike_2to2.txt',
    'graphs/hand_built/decay_1to3.txt',
    'graphs/hand_built/lightlike_2to3.txt',
    'graphs/hand_built/lightlike_2to3_fifthleg.txt',
    'graphs/hand_built/lightlike_2to2_1soft.txt',
    'graphs/hand_built/decay_1to3_1soft.txt',
]


def to_scaling(em):
    return tuple(-(2 * md[0] + md[1]) if md is not None else 0 for md in em) + (1,)


def run_case(edges, ext_attach, ext_mode):
    verts = sorted({v for e in edges for v in e} | set(ext_attach.values()))
    edges_in = [tuple(sorted(e)) for e in edges]
    t0 = time.time()
    regs, _nc, _dt = SG.run(verts, edges_in, ext_attach, ext_mode,
                            verbose=False, overlap_strong=True)
    fri_map = {tuple(tuple(m) for m in em): to_scaling(em) for _vm, em in regs}
    return fri_map, time.time() - t0


def fmt_full(fri_map):
    out = []
    for i, (key, vec) in enumerate(sorted(fri_map.items(),
                                          key=lambda kv: (kv[1], kv[0])), 1):
        out.append('FRI region {:>2}: [{}]  ->  scaling vector ({})'.format(
            i, ', '.join(mode_str(md) for md in key), ', '.join(map(str, vec))))
    return out


def kin_order(keys):
    return sorted(keys, key=lambda k: int(k[1:]))


def resolve(p):
    return p if os.path.exists(p) else os.path.join(HERE, p)


# ---------------- random batches ----------------
def run_batch(path, only_graph=None, kin_filter=None, full=False):
    B = json.load(open(path))
    nlegs = B.get('nlegs', 5)
    kin = DATA['kinematics_4'] if nlegs == 4 else DATA['kinematics_5']
    ks = kin_order(kin.keys())
    if kin_filter is not None:
        ks = [k for k in ks if int(k[1:]) in kin_filter]
    graphs = B['graphs']
    idx = range(len(graphs)) if only_graph is None else [only_graph]
    rec = load_batch_records(path)
    results = {}
    n_tot = n_chk = n_mm = 0
    t00 = time.time()
    print('== %s (%d graphs, seed %s) ==' % (os.path.basename(path), len(graphs), B.get('seed')))
    for gi in idx:
        g = graphs[gi]
        edges = [tuple(e) for e in g['edges']]
        if 'ext' in g:
            ext_attach = {'p%d' % (i + 1): int(x) for i, x in enumerate(g['ext'])}
            name = g.get('name') or ('%04d v=%d' % (gi, g['v']))
        else:
            ext_attach = {k: int(v) for k, v in g['ext_attach'].items()}
            name = g.get('name') or ('%04d' % gi)
        for k in ks:
            em = {n: tuple(v) for n, v in kin[k].items()}
            fri_map, dt = run_case(edges, ext_attach, em)
            n = len(fri_map)
            results[(name, k)] = n
            n_tot += n
            note = ''
            if rec is not None and (name, k) in rec:
                n_chk += 1
                if rec[(name, k)] == n:
                    note = ', matched'
                else:
                    n_mm += 1
                    note = ', MISMATCH: record=%d' % rec[(name, k)]
            print('  %s %s: %d regions  (%.2fs%s)' % (name, k, n, dt, note))
            if full:
                for ln in fmt_full(fri_map):
                    print('      ' + ln)
                print()
    print('== %s: %d case(s), %d region(s) total, %.1fs' %
          (os.path.basename(path), len(results), n_tot, time.time() - t00))
    if n_chk:
        print('== compare: %d case(s) checked, %d mismatch(es)' % (n_chk, n_mm))
    return results


# ---------------- hand-built books ----------------
def parse_book(path):
    txt = open(path).read()
    parts = re.split(r'^# ==== (\S+) ====\n', txt, flags=re.M)
    cases = {}
    for i in range(1, len(parts), 2):
        fname, body = parts[i], parts[i + 1]
        name = re.search(r"^\s*name = '([^']+)'", body, re.M).group(1)
        internal = ast.literal_eval(
            re.search(r'^\s*internal_lines = (\[.*\])', body, re.M).group(1))
        external = ast.literal_eval(
            re.search(r'^\s*external_lines = (\[.*\])', body, re.M).group(1))
        cases[fname] = (name, internal, external)
    return cases


def run_book(path, only=None, full=False):
    cls = os.path.basename(path)[:-4]
    modes = DATA['handbuilt'][cls]
    cases = parse_book(path)
    rec = load_book_records(cls)
    results = {}
    n_tot = n_chk = n_mm = 0
    t00 = time.time()
    print('== %s (%d cases) ==' % (os.path.basename(path), len(cases)))
    for fname, (name, internal, external) in cases.items():
        if only and only not in (fname, name):
            continue
        edges = [tuple((a, b)) for _, (a, b) in internal]
        ext_attach = {n: v for n, v in external}
        em = {n: tuple(v) for n, v in modes[fname].items()}
        fri_map, dt = run_case(edges, ext_attach, em)
        n = len(fri_map)
        results[name] = n
        n_tot += n
        note = ''
        if rec is not None and name in rec:
            n_chk += 1
            if rec[name] == n:
                note = ', matched'
            else:
                n_mm += 1
                note = ', MISMATCH: record=%d' % rec[name]
        print('  %s: %d regions  (%.2fs%s)' % (name, n, dt, note))
        if full:
            for ln in fmt_full(fri_map):
                print('      ' + ln)
            print()
    print('== %s: %d case(s), %d region(s) total, %.1fs' %
          (os.path.basename(path), len(results), n_tot, time.time() - t00))
    if n_chk:
        print('== compare: %d case(s) checked, %d mismatch(es)' % (n_chk, n_mm))
    return results


# ---------------- comparison against the stored records ----------------
RECORD_FOR = {
    'rand1000_3l_no2v.json': 'rand1000_3l_report.txt',
    'rand700_4l_no2v.json': 'rand700_4l_report.txt',
    'rand100_5l_no2v.json': 'rand100_5l_report.txt',
    'rand500_5l_no2v.json': 'rand500_5l_first200_report.txt',
    'rand1000_4l_v8-10_no2v.json': 'rand1000_4l_report.txt',
    'rand200_4l_no2v_v8-10.json': 'rand200_4l_report.txt',
    'rand200_5l_v10-12_no2v.json': 'rand200_5l_report.txt',
}


def load_batch_records(path):
    """{(name, kin): regions} from the stored report of this batch, or None."""
    rec = RECORD_FOR.get(os.path.basename(path))
    if rec is None:
        return None
    recp = os.path.join(HERE, 'results', rec)
    if not os.path.exists(recp):
        return None
    out = {}
    for ln in open(recp):
        m = re.match(r'^(\S+)(?: (v=\d+))?  (k\d)  \S+  (\d+)$', ln.strip())
        if m:
            name = m.group(1) + ((' ' + m.group(2)) if m.group(2) else '')
            out[(name, m.group(3))] = int(m.group(4))
    return out


def load_book_records(cls):
    """{case: regions} from handbuilt_summary.txt for one class, or None."""
    recp = os.path.join(HERE, 'results', 'handbuilt_summary.txt')
    if not os.path.exists(recp):
        return None
    out = {}
    for ln in open(recp):
        m = re.match(r'^\s+(\S+)/(\S+)\.txt  OK  regions=(\d+)', ln)
        if m and m.group(1) == cls:
            out[m.group(2)] = int(m.group(3))
    return out


def main():
    ap = argparse.ArgumentParser(description='run FRI on the validation cases (no pySecDec needed)')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--batch', help='a random-batch JSON from graphs/random/')
    g.add_argument('--book', help='a spec book from graphs/hand_built/')
    g.add_argument('--all', action='store_true', help='run everything (all batches + all books)')
    ap.add_argument('--graph', type=int, help='batch: run a single graph (index)')
    ap.add_argument('--kin', help='batch: subset of kinematics, e.g. 0,1')
    ap.add_argument('--only', help='book: run a single case (file or name)')
    ap.add_argument('--full', action='store_true', help='print the complete region lists')
    args = ap.parse_args()

    print('FRI region runner (validation package) -- uses the FRI code from the main repository; pySecDec is not needed.')
    print('Each line: <case> <kin>: <N> regions  (<time>s, matched) -- "matched" = the count equals the stored record in results/.')
    print()

    kin_filter = [int(x) for x in args.kin.split(',')] if args.kin else None

    if args.batch:
        run_batch(resolve(args.batch), args.graph, kin_filter, args.full)
    elif args.book:
        run_book(resolve(args.book), args.only, args.full)
    else:
        for b in BATCHES:
            run_batch(resolve(b), None, None, args.full)
        for b in BOOKS:
            run_book(resolve(b), None, args.full)


if __name__ == '__main__':
    main()
