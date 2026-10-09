r"""
power_counting.py — compute the expansion power w.r.t. a given region in dimensional regularization.

The contribution of a region to a Feynman integral (integrand 1/(product of propagators)) scales as \lambda^{A-B+N}, where
    integration measure ~ \lambda^A,
    denominator ~ \lambda^B,
    numerator ~ \lambda^N.

To compute A. First consider the independent loop momenta of this region.
              A S^m C_i^n-mode loop momentum scales as (\lambda^m, \lambda^{m+n}, \lambda^{m+n/2}), contributing \lambda^{(2m+n) \cdot (2-eps)} to the integration measure.
              Note that 2m + n is also the virtuality degree of S^m C_i^n.

To compute B. B = sum of V(edge mode) over the propagators.

To compute N. First express the numerator (given by the user) in terms of the independent loop momenta k1, k2 ....
              Then it should be a sum of products of ki.kj or ki.pj.
              The power of ki.kj is then the virtuality degree of Xi\vee Xj (with Xi and Xj being modes of ki and kj, respectively). ki.pj similar.
              Namely, ki.kj ~ \lambda^{V(Xi\vee Xj)}.
"""

import re
from primitives import V, join
from indep_loops import indep_loops, edge_momenta


# ============================== INTEGRATION MEASURE & INTEGRAND ==============================

# A: each S^m C_i^n-mode loop momentum contributes \lambda^{(2m+n) \cdot (2-eps)} to the integration measure; returned as (a0, a1) for \lambda^{a0 + a1*eps}.
def measure_power(results):
    p = sum(r['rank'] * V(r['mode']) for r in results)
    return 2 * p, -p


# B: the product of propagators ~ \lambda^B, with B = sum of V(edge mode) over the propagators.
def integrand_power(em):
    return sum(0 if m is None else V(m) for m in em)


# \lambda^{a0 + a1*eps} as a LaTeX-ready string.
def fmt_power(a0, a1):
    if a1 == 0:
        exp = f'{a0}'
    elif a1 > 0:
        exp = f'{a0} + {a1}\\epsilon'
    else:
        exp = f'{a0} - {-a1}\\epsilon'
    return f'\\lambda^{{{exp}}}'


# Print the power counting (measure, integrand, total power) for one region.
def show_power_counting(results, em):
    a0, a1 = measure_power(results)
    b = integrand_power(em)
    print(f'  integration measure = {fmt_power(a0, a1)}')
    print(f'  integrand = {fmt_power(-b, 0)}')
    print(f'  power = {fmt_power(a0 - b, a1)}')


# ============================== NUMERATOR ==============================

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
# Per-region context for the numerator expansion: the loop-momentum and external modes, and each K_i expressed in k1, k2, ... / p_j.
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


# N: the numerator's expansion power for one region (an integer): each atom ki.kj contributes \lambda^{V(Xi \vee Xj)}.
def numerator_power(ast, ctx):
    # Expansion terms of an operand: k_a / p_j are single tokens; K_i expands into its region terms.
    def terms_of(v):
        return ctx['K_terms'][v[1]] if v[0] == 'K' else [v]

    def mode_of(t):
        return ctx['k_modes'][t[1]] if t[0] == 'k' else ctx['p_modes'][t[1]]

    # Power of one atom x.y: min over the expansion pairs a.b (a from v1, b from v2) of V(X_a \vee X_b).
    def pair_pow(v1, v2):
        return min(V(join(mode_of(a), mode_of(b))) for a in terms_of(v1) for b in terms_of(v2))

    # Recursive evaluation (rules in the module docstring): atom -> V(Xi \vee Xj); sum -> min; product -> add; power -> times n.
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
