#!/usr/bin/env python3
r"""power_counting.py — scalar-integral power counting, with numerator polynomials.

Scalar part.  The contribution of a region to a scalar Feynman integral
(integrand 1/(product of propagators)) scales as \lambda^{A-B}:

  integration measure:  A = (2 - eps) * sum over the independent loop momenta
      of V(k).  A loop momentum of mode X = S^m C_i^n scales as
      (\lambda^m, \lambda^{m+n}, \lambda^{m+n/2}); its measure is
      (\lambda^{V})^(2-eps) with V = 2m + n = pr_V(X).
  integrand:  B = sum of V(edge mode) over the propagators (their
      virtualities), i.e. product of propagators ~ \lambda^B.
  power = A - B.

Numerator part.  A polynomial in the edge momenta K_i (i = input edge
order) and the externals p_j; coefficients and signs are irrelevant for
the power.  For each region, every K_i is expanded (via the line-momentum
decomposition of that region) into its independent loop momenta k_a and
the externals, and the lambda-power of the result is evaluated with

  atom (x.y):  lambda^{V(X_x vee X_y)};   sum -> min; product -> add; power -> times n.

`numerator_power` returns that integer N, so a region's power becomes A - B + N.
"""

import re

from primitives import V, join
from indep_loops import indep_loops, edge_momenta


# ============================== SCALAR PART ==============================

# Integration measure: A = (2 - eps) * sum of rank*V over the per-mode loop classes. Returned as (p0, p1) for \lambda^{p0 + p1*eps}.
def measure_power(results):
    p = sum(r['rank'] * V(r['mode']) for r in results)
    return 2 * p, -p


# Product of propagators: \lambda^B with B = sum of the edge-mode virtualities V.
def integrand_power(em):
    return sum(0 if m is None else V(m) for m in em)


# \lambda^{p0 + p1*eps} as a LaTeX-ready string ('- 7\\epsilon' style).
def fmt_power(p0, p1):
    if p1 == 0:
        exp = f'{p0}'
    elif p1 > 0:
        exp = f'{p0} + {p1}\\epsilon'
    else:
        exp = f'{p0} - {-p1}\\epsilon'
    return f'\\lambda^{{{exp}}}'


# Print the power counting (measure, integrand, total power) for one region.
def show_power_counting(results, em):
    a0, a1 = measure_power(results)
    b = integrand_power(em)
    print(f'  integration measure = {fmt_power(a0, a1)}')
    print(f'  integrand = {fmt_power(-b, 0)}')
    print(f'  power = {fmt_power(a0 - b, a1)}')


# ============================== NUMERATOR PART ==============================

# ---------------------------------------------------------------- preprocessing
# Drop LaTeX decoration: '$', \cdot -> '·', leftover backslashes -> spaces, subscripts 'K_{1}'/'K_1' -> 'K1'.
def normalize(text):
    s = text.replace('$', '')
    s = s.replace('\\cdot', '·')
    s = s.replace('\\', ' ')
    s = re.sub(r'_\s*\{(\d+)\}', r'\1', s)
    s = re.sub(r'_\s*(\d+)', r'\1', s)
    return s


# ---------------------------------------------------------------- tokenizer
# Tokens: ('num', n) | ('word', w) | ('op', c) with c in '+-^()·' ('*' reads as '·').
def tokenize(s):
    toks = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c.isdigit():
            j = i
            while j < n and s[j].isdigit():
                j += 1
            toks.append(('num', int(s[i:j])))
            i = j
        elif c.isalpha():
            j = i
            while j < n and s[j].isalnum():
                j += 1
            toks.append(('word', s[i:j]))
            i = j
        elif c in '+-^()':
            toks.append(('op', c))
            i += 1
        elif c in '·*':
            toks.append(('op', '·'))
            i += 1
        else:
            raise ValueError(f"unexpected character '{c}' (position {i})")
    return toks


# ---------------------------------------------------------------- parser
# Recursive-descent parser for the numerator polynomial (the grammar is spelled out in the per-method comments).
class _Parser:
    def __init__(self, toks, nedge, pnames):
        self.toks = toks
        self.pos = 0
        self.nedge = nedge
        self.pnames = set(pnames)
        self.warnings = []

    # Current token, or None at the end.
    def peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    # Consume and return the current token.
    def take(self):
        tok = self.toks[self.pos]
        self.pos += 1
        return tok

    # expr := term (('+'|'-') term)*   (signs are dropped; irrelevant for the power)
    def parse_expr(self):
        terms = [self.parse_term()]
        while True:
            tok = self.peek()
            if tok in (('op', '+'), ('op', '-')):
                self.take()
                terms.append(self.parse_term())
            else:
                break
        return terms[0] if len(terms) == 1 else ('add', terms)

    # term := ['+'|'-']* factor+   (leading signs are irrelevant for the power)
    def parse_term(self):
        while self.peek() in (('op', '+'), ('op', '-')):
            self.take()
        factors = []
        while True:
            tok = self.peek()
            if tok is None or tok in (('op', '+'), ('op', '-'), ('op', ')')):
                break
            if tok[0] == 'num':
                self.take()
                factors.append(('num', tok[1]))
                continue
            factors.append(self.parse_factor())
        if not factors:
            raise ValueError('empty term')
        return factors[0] if len(factors) == 1 else ('mul', factors)

    # factor := unit ('^' n)?   (a bare vector with ^2m means the m-th self-product power)
    def parse_factor(self):
        unit = self.parse_unit()
        if self.peek() == ('op', '^'):
            self.take()
            nt = self.peek()
            if nt is None or nt[0] != 'num':
                raise ValueError('expected an integer power after ^')
            self.take()
            pw = nt[1]
            if unit[0] == 'var':
                if pw % 2:
                    raise ValueError('an odd power of a vector is not defined')
                return ('vpow', unit[1], unit[2], pw)
            return ('spow', unit, pw)
        if unit[0] == 'var':
            name = f'K_{unit[2]}' if unit[1] == 'K' else unit[2]
            raise ValueError(f'vector {name} must appear in a dot product or with a power')
        return unit

    # unit := '(' expr ')' | var ('·' var)?
    def parse_unit(self):
        tok = self.peek()
        if tok == ('op', '('):
            self.take()
            inner = self.parse_expr()
            if self.peek() != ('op', ')'):
                raise ValueError('missing closing parenthesis')
            self.take()
            return inner
        if tok is not None and tok[0] == 'word':
            self.take()
            x = self.classify(tok[1])
            if self.peek() == ('op', '·'):
                self.take()
                t2 = self.peek()
                if t2 is None or t2[0] != 'word':
                    raise ValueError('expected a symbol after the dot product')
                self.take()
                y = self.classify(t2[1])
                return ('dot', x[1], x[2], y[1], y[2])
            return x
        raise ValueError('expected a symbol or parenthesis')

    # Word -> ('var', kind, key); kind 'K' (edge momentum) or 'p' (external name).
    def classify(self, w):
        m = re.fullmatch(r'([Kk])(\d+)', w)
        if m:
            idx = int(m.group(2))
            if not (1 <= idx <= self.nedge):
                raise ValueError(f'K_{idx} out of range (edges are 1..{self.nedge})')
            # lowercase k is a hand slip: read as K_{idx} and warn.
            if m.group(1) == 'k':
                self.warnings.append(f'lowercase "{w}" read as K_{idx} (lowercase k is reserved for loop momenta)')
            return ('var', 'K', idx)
        if w in self.pnames:
            return ('var', 'p', w)
        raise ValueError(f"unknown symbol '{w}' (use K_i for edge momenta or an external name)")


# Parse a numerator string -> (ast, warnings); ast = None means "1" (scalar).
def parse_numerator(text, nedge, pnames):
    s = normalize(text)
    if not s.strip() or s.strip() == '1':
        return None, []
    p = _Parser(tokenize(s), nedge, pnames)
    ast = p.parse_expr()
    if p.peek() is not None:
        raise ValueError(f'unexpected extra input starting at {p.peek()!r}')
    return ast, list(dict.fromkeys(p.warnings))


# ---------------------------------------------------------------- display
# Canonical rendering ('\cdot', '^'); coefficients kept, signs dropped.
def fmt_ast(node):
    kind = node[0]
    if kind == 'num':
        return str(node[1])
    if kind == 'var':
        return f'K_{node[2]}' if node[1] == 'K' else node[2]
    if kind == 'dot':
        return f'({fmt_ast(("var", node[1], node[2]))}\\cdot {fmt_ast(("var", node[3], node[4]))})'
    if kind == 'vpow':
        base = f'K_{node[2]}' if node[1] == 'K' else node[2]
        return f'{base}^{node[3]}'
    if kind == 'spow':
        inner = fmt_ast(node[1])
        if node[1][0] in ('add', 'mul'):
            inner = f'({inner})'
        return f'{inner}^{node[2]}'
    if kind == 'mul':
        parts = []
        for ch in node[1]:
            s = fmt_ast(ch)
            if ch[0] == 'add':
                s = f'({s})'
            parts.append(s)
        return ' '.join(parts)
    if kind == 'add':
        return ' + '.join(fmt_ast(ch) for ch in node[1])
    raise ValueError(f'bad node {node!r}')


# ---------------------------------------------------------------- per-region evaluation
# Context for one region: k_modes (a -> mode of k_a), p_modes (name -> mode), K_terms (i -> [(kind, key), ...]).
# The K_i expansion follows this region's own loop-momentum decomposition (indep_loops / edge_momenta).
def region_context(edges, em, vm, ext_attach, extmode):
    results, _total, _L = indep_loops(edges, em, vm)
    basis_all = [j for r in results for b in r['blocks'] for j in b['basis']]
    ok, order, momenta, _verts = edge_momenta(edges, em, vm, basis_all, ext_attach)
    if not ok:
        return None
    K_terms = {}
    for ei in range(len(edges)):
        toks = []
        for t, c in sorted(momenta[ei].items()):
            if c == 0:
                continue
            toks.append(('k', int(t[1:])) if t.startswith('k') else ('p', t))
        K_terms[ei + 1] = toks
    k_modes = {}
    for a, ei in enumerate(order, 1):
        for r in results:
            if any(ei in b['basis'] for b in r['blocks']):
                k_modes[a] = r['mode']
                break
    return {'k_modes': k_modes, 'p_modes': dict(extmode), 'K_terms': K_terms}


# The lambda-power (an integer) of the numerator for one region; scales like lambda^N.
def numerator_power(ast, ctx):
    # Expansion terms of an operand: k_a / p_j are single tokens; K_i expands into its region terms.
    def terms_of(v):
        return ctx['K_terms'][v[1]] if v[0] == 'K' else [v]

    def mode_of(t):
        return ctx['k_modes'][t[1]] if t[0] == 'k' else ctx['p_modes'][t[1]]

    # min over the expansion pairs (a.b), a from v1's terms, b from v2's terms
    def pair_pow(v1, v2):
        return min(V(join(mode_of(a), mode_of(b))) for a in terms_of(v1) for b in terms_of(v2))

    # Recursive evaluation: the docstring rules (sum -> min, product -> add, power -> times n); atoms go to pair_pow.
    def ev(node):
        kind = node[0]
        if kind == 'num':
            return 0
        if kind == 'vpow':
            v, n = (node[1], node[2]), node[3]
            return (n // 2) * pair_pow(v, v)
        if kind == 'dot':
            return pair_pow((node[1], node[2]), (node[3], node[4]))
        if kind == 'spow':
            return node[2] * ev(node[1])
        if kind == 'mul':
            return sum(ev(ch) for ch in node[1])
        if kind == 'add':
            return min(ev(ch) for ch in node[1])
        raise ValueError(f'bad node {node!r}')

    return ev(ast)


# Per-term breakdown for reports: [(text, power), ...] and the min over terms.
def term_breakdown(ast, ctx):
    terms = ast[1] if ast[0] == 'add' else [ast]
    powers = [(fmt_ast(t), numerator_power(t, ctx)) for t in terms]
    return powers, min(p for _t, p in powers)


# ---------------------------------------------------------------- snapshot CLI
# Quick check on the default example graph: parse a file path OR a literal expression, print N per region.
if __name__ == '__main__':
    import os, sys
    from facet_regions_interactive import enumerate_regions, DEFAULT_EDGES, parse_edge_list, parse_region_select
    from read_graph import parse_mode

    edges = [tuple(sorted((a, b))) for (a, b) in parse_edge_list(DEFAULT_EDGES)]
    exts = {'p1': (1, 'C1'), 'p2': (2, 'C2^2'), 'p3': (3, 'C3^inf'), 'p4': (4, 'C4^inf')}
    ext_attach = {n: v for n, (v, s) in exts.items()}
    extmode = {n: parse_mode(s) for n, (v, s) in exts.items()}
    verts = sorted({v for e in edges for v in e} | set(ext_attach.values()))
    regs = enumerate_regions(verts, edges, ext_attach, extmode)

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        text = open(arg).read() if os.path.exists(arg) else arg
    else:
        text = r'(p1\cdot K_1)(K_2\cdot K_3)'
    sel = parse_region_select(sys.argv[2], len(regs)) if len(sys.argv) > 2 else [1, 21, 81]
    ast, warns = parse_numerator(text, len(edges), set(extmode))
    for w in warns:
        print('! note:', w)
    print('numerator =', fmt_ast(ast) if ast else '1 (scalar)')
    for i in sel:
        vm, em = regs[i - 1]
        ctx = region_context(edges, em, vm, ext_attach, extmode)
        results, _t, _L = indep_loops(edges, em, vm)
        a0, a1 = measure_power(results)
        b = integrand_power(em)
        n = numerator_power(ast, ctx) if ast else 0
        print(f'R{i}: A-B = {fmt_power(a0 - b, a1)} | N = {n} | power = {fmt_power(a0 - b + n, a1)}')
