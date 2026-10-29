r"""
numerator_power.py - power of a Feynman-rule numerator in a given region (BETA).

For a region and a numerator written in the edge momenta K_i (and optionally the external momenta p_j), compute N in  integrand ~ lambda^{A - B + N}.

Method.  Every K_i is first written as a signed combination of the independent loop momenta k_a and the externals p_j.
The numerator is expanded into atoms a.b (a, b single basis momenta) with integer coefficients.
The atom products are not independent: sum_j p_i.p_j = 0 for every external and sum_j p_j.k_a = 0 for every loop momentum.
Those relations are reduced to a row echelon form against a fixed global order and substituted into the polynomial, after which N is the smallest power of any surviving monomial.
The substitution keeps integer coefficients, so the answer does not depend on how a momentum is written: the edge-only and the p-form numerator agree in every region.

Implementation notes.
The polynomial is reduced after every factor multiplication and the monomials are bucketed by power, so that only the vanishing low levels are reduced.
Both are exact: substitution can only raise a power.

Layout of this file.

  1  reading the numerator string      text -> display AST and token stream
  2  region context                    K_i written in terms of k_a and p_j
  3  atom polynomials                  monomials are multisets of atoms a.b
  4  momentum-conservation relations   row echelon reduction and substitution
  5  power of a region                 N = smallest power of a surviving monomial

Known limits (this is why the module is marked BETA).
The relation set above is complete only when the externals satisfy a single linear relation, i.e. up to five externals in four dimensions.
With six or more externals (3->3 and beyond) a second independent relation is needed and this module does not know it.
Five-point k0 and soft externals have not been tested.
For a new class of kinematics, validate against a numerical evaluation first.
"""

import itertools
import re
from fractions import Fraction
from math import gcd as _gcd

from primitives import V, vee
from indep_loops import indep_loops, edge_momenta

BETA = True


# ============================== 1 · READING THE NUMERATOR STRING ==============================
# Text in, structure out: the AST used for display and the token stream used by the
# atom parser below.  No physics in this section.

# -------- preprocessing
# Drop LaTeX decoration: '$', \cdot -> '·', leftover backslashes -> spaces, subscripts 'K_{1}'/'K_1' -> 'K1'.
def normalize(text):
    s = text.replace('$', '')
    s = s.replace('\\cdot', '·')
    # subscripts first: K_{4} -> K4, before braces are turned into parentheses
    s = re.sub(r'_\s*\{(\d+)\}', r'\1', s)
    s = re.sub(r'_\s*(\d+)', r'\1', s)
    # LaTeX sizing commands (\big \Big \bigg \Bigg, with optional l/r/m) and \left/\right
    s = re.sub(r'\\[bB]igg?[lrm]?\s*', '', s)
    s = re.sub(r'\\(left|right)\s*', '', s)
    # braces and brackets all become round parentheses
    s = s.replace('\\{', '(').replace('\\}', ')')
    s = s.replace('\\lbrace', '(').replace('\\rbrace', ')')
    s = s.replace('\\lbrack', '(').replace('\\rbrack', ')')
    s = s.replace('{', '(').replace('}', ')')
    s = s.replace('[', '(').replace(']', ')')
    # any remaining backslash is decoration
    s = s.replace('\\', ' ')
    return s


# -------- tokenizer
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


# -------- recursive-descent parser (display AST)
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

    # unit := '(' expr ')' | '(' vexpr ')' '·' voperand | var ('·' voperand)?
    def parse_unit(self):
        tok = self.peek()
        if tok == ('op', '('):
            if self._group_followed_by_dot():
                self.take()
                v1 = self.parse_vexpr()
                self._close_paren()
                self.take()
                return ('vdot', v1, self.parse_voperand())
            self.take()
            inner = self.parse_expr()
            self._close_paren()
            return inner
        if tok is not None and tok[0] == 'word':
            self.take()
            x = self.classify(tok[1])
            if self.peek() == ('op', '·'):
                self.take()
                return ('vdot', ('vsum', [(1, ('var', x[1], x[2]))]), self.parse_voperand())
            return x
        raise ValueError('expected a symbol or parenthesis')

    def _close_paren(self):
        if self.peek() != ('op', ')'):
            raise ValueError('missing closing parenthesis')
        self.take()

    # True when the '(' at self.pos closes at a ')' that is followed by '·'.
    def _group_followed_by_dot(self):
        depth = 0
        for k in range(self.pos, len(self.toks)):
            t = self.toks[k]
            if t == ('op', '('):
                depth += 1
            elif t == ('op', ')'):
                depth -= 1
                if depth == 0:
                    return k + 1 < len(self.toks) and self.toks[k + 1] == ('op', '·')
        raise ValueError('missing closing parenthesis')

    # A vector operand after '·': a parenthesised vexpr or a single symbol, with optional leading signs.
    def parse_voperand(self):
        sign = 1
        while self.peek() in (('op', '+'), ('op', '-')):
            if self.take()[1] == '-':
                sign = -sign
        t = self.peek()
        if t == ('op', '('):
            self.take()
            v = self.parse_vexpr()
            self._close_paren()
            return v if sign == 1 else ('vsum', [(-c, w) for c, w in v[1]])
        if t is not None and t[0] == 'word':
            self.take()
            x = self.classify(t[1])
            return ('vsum', [(sign, ('var', x[1], x[2]))])
        raise ValueError('expected a vector expression after the dot product')

    # vexpr := vterm (('+'|'-') vterm)*   (linear combinations of p_i and K_i)
    def parse_vexpr(self):
        sign = 1
        while self.peek() in (('op', '+'), ('op', '-')):
            if self.take()[1] == '-':
                sign = -sign
        items = self.parse_vterm(sign)
        while self.peek() in (('op', '+'), ('op', '-')):
            sgn = 1 if self.take()[1] == '+' else -1
            items += self.parse_vterm(sgn)
        return self._vsum(items)

    # vterm := [num] (symbol | '(' vexpr ')')
    def parse_vterm(self, sign):
        coef = 1
        t = self.peek()
        if t is not None and t[0] == 'num':
            coef = self.take()[1]
        t = self.peek()
        if t == ('op', '('):
            self.take()
            inner = self.parse_vexpr()
            self._close_paren()
            return [(sign * coef * c, v) for c, v in inner[1]]
        if t is not None and t[0] == 'word':
            self.take()
            x = self.classify(t[1])
            return [(sign * coef, ('var', x[1], x[2]))]
        raise ValueError('expected a vector symbol or parenthesis')

    # Combine (coef, var) items, dropping symbols whose coefficients cancel.
    def _vsum(self, items):
        acc, order = {}, []
        for c, v in items:
            key = (v[1], v[2])
            if key not in acc:
                acc[key] = 0
                order.append(key)
            acc[key] += c
        return ('vsum', [(acc[k], ('var', k[0], k[1])) for k in order if acc[k] != 0])

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


# Parse a numerator string for validation/echo only (the power comes from exact_power).
# Returns (ast, warnings); ast = None means "1" (scalar).
def parse_numerator(text, nedge, pnames):
    s = normalize(text)
    if not s.strip() or s.strip() == '1':
        return None, []
    p = _Parser(tokenize(s), nedge, pnames)
    ast = p.parse_expr()
    if p.peek() is not None:
        raise ValueError(f'unexpected extra input starting at {p.peek()!r}')
    return ast, list(dict.fromkeys(p.warnings))


# -------- display
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
        if node[1][0] in ('add', 'mul', 'vdot', 'vsum'):
            inner = f'({inner})'
        return f'{inner}^{node[2]}'
    if kind == 'vsum':
        if not node[1]:
            return '0'
        parts = []
        for c, v in node[1]:
            name = f'K_{v[2]}' if v[1] == 'K' else v[2]
            if c == 1:
                parts.append(name)
            elif c == -1:
                parts.append('-' + name)
            elif c > 1:
                parts.append(f'{c}{name}')
            else:
                parts.append(f'-{abs(c)}{name}')
        out = parts[0]
        for t in parts[1:]:
            out += (' ' + t) if t.startswith('-') else (' + ' + t)
        return out
    if kind == 'vdot':
        v1, v2 = fmt_ast(node[1]), fmt_ast(node[2])
        if v1 == '0' or v2 == '0':
            return '0'
        return f'({v1})\\cdot ({v2})'
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


# ============================== 2 · REGION CONTEXT ==============================
# Each K_i written as a signed combination of the independent loop momenta k_a and the
# externals p_j, i.e. the routing that makes a region explicit.
# Per-region context for the numerator expansion: the loop-momentum and external modes, and each K_i expressed in k1, k2, ... / p_j.
def region_context(edges, em, vm, ext_attach, extmode):
    results, _total, _L = indep_loops(edges, em, vm)
    basis_all = [j for r in results for b in r['blocks'] for j in b['basis']]
    ok, order, momenta, _verts = edge_momenta(edges, em, vm, basis_all, ext_attach)
    if not ok:
        return None
    K_terms = {}
    K_expr = {}
    for ei in range(len(edges)):
        toks = []
        for t, c in sorted(momenta[ei].items()):
            if c == 0:
                continue
            toks.append(('k', int(t[1:])) if t.startswith('k') else ('p', t))
        K_terms[ei + 1] = toks
        K_expr[ei + 1] = dict(momenta[ei])
    k_modes = {}
    for a, ei in enumerate(order, 1):
        for r in results:
            if any(ei in b['basis'] for b in r['blocks']):
                k_modes[a] = r['mode']
                break
    p_edges = {}
    for name, v in ext_attach.items():
        coeffs = {}
        for ei, (a, b) in enumerate(edges):
            if a == v:
                coeffs['K%d' % (ei + 1)] = -1
            elif b == v:
                coeffs['K%d' % (ei + 1)] = 1
        if coeffs:
            p_edges[name] = coeffs
    return {'k_modes': k_modes, 'p_modes': dict(extmode), 'K_terms': K_terms, 'K_expr': K_expr,
            'p_edges': p_edges}


# ============================== 3 · ATOM POLYNOMIALS ==============================
# An atom is a.b (two single basis momenta); a monomial is a multiset of atoms; a
# polynomial maps monomials to integer coefficients.  This section holds that
# representation and the parser that produces it.

# Tokenizer that also records the character span of every token.
def _tokenize_pos(s):
    toks, spans = [], []
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
            spans.append((i, j))
            i = j
        elif c.isalpha():
            j = i
            while j < n and s[j].isalnum():
                j += 1
            toks.append(('word', s[i:j]))
            spans.append((i, j))
            i = j
        elif c in '+-^()':
            toks.append(('op', c))
            spans.append((i, i + 1))
            i += 1
        elif c in '·*':
            toks.append(('op', '·'))
            spans.append((i, i + 1))
            i += 1
        else:
            raise ValueError("unexpected character '%s' (position %d)" % (c, i))
    return toks, spans


# -------- integer polynomial arithmetic
# A monomial is a sorted tuple of atoms ('d', a, b); the empty tuple is 1.
def _add(p, q, sign=1):
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, 0) + sign * c
        if out[m] == 0:
            del out[m]
    return out


def _mul(p, q):
    out = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = tuple(sorted(m1 + m2))
            out[m] = out.get(m, 0) + c1 * c2
    return {m: c for m, c in out.items() if c}


def _pow(p, n):
    r = {(): 1}
    for _ in range(n):
        r = _mul(r, p)
    return r


class _PolyParser:
    def __init__(self, toks, spans, src, nedge, pnames, ksub):
        self.t = toks
        self.spans = spans
        self.src = src
        self.i = 0
        self.nedge = nedge
        self.pnames = set(pnames)
        self.ksub = ksub
        self.depth = 0
        self.terms = []
        self.ctx = None
        self.reduce_each = False
        self._cur = []
        self.eliminated = None

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        x = self.t[self.i]
        self.i += 1
        return x

    def expr(self):
        sign = 1
        while self.peek() in (('op', '+'), ('op', '-')):
            if self.take()[1] == '-':
                sign = -sign
        if self.depth == 0:
            self._cur = []
        p = self.term()
        if sign < 0:
            p = {m: -c for m, c in p.items()}
        if self.depth == 0:
            self.terms.append((self._cur, p))
        while self.peek() in (('op', '+'), ('op', '-')):
            s = 1 if self.take()[1] == '+' else -1
            if self.depth == 0:
                self._cur = []
            q = self.term()
            if self.depth == 0:
                self.terms.append((self._cur, q))
            p = _add(p, q, s)
        return p

    def term(self):
        p = {(): 1}
        while True:
            t = self.peek()
            if t is None or t in (('op', '+'), ('op', '-'), ('op', ')')):
                break
            if t[0] == 'num':
                p = _mul(p, {(): self.take()[1]})
                continue
            i0 = self.i
            f = self.factor()
            if self.depth == 0 and self.i > i0:
                txt = self.src[self.spans[i0][0]:self.spans[self.i - 1][1]]
                self._cur.append((f, txt))
            p = _mul(p, f)
            if self.reduce_each and self.depth == 0:
                p = reduce_poly(p, self.ctx, sorted(self.pnames))
        return p

    def factor(self):
        u = self.unit()
        if self.peek() == ('op', '^'):
            self.take()
            nt = self.take()
            if nt is None or nt[0] != 'num':
                raise ValueError('expected an integer power after ^')
            n = nt[1]
            if isinstance(u, tuple) and u[0] == 'bare':
                if n % 2:
                    raise ValueError('an odd power of a vector is not defined')
                v = {u[1]: 1}
                return _pow(self.dotpoly(v, v), n // 2)
            return _pow(u, n)
        if isinstance(u, tuple) and u[0] == 'bare':
            raise ValueError(f'vector {u[1]} must appear in a dot product or with a power')
        return u

    def unit(self):
        t = self.peek()
        if t == ('op', '('):
            if self._group_dot():
                self.take()
                v1 = self.vexpr()
                self._close()
                self.take()
                return self.dotpoly(v1, self.voperand())
            self.take()
            self.depth += 1
            p = self.expr()
            self.depth -= 1
            self._close()
            return p
        if t is not None and t[0] == 'word':
            self.take()
            x = self.classify(t[1])
            if self.peek() == ('op', '·'):
                self.take()
                return self.dotpoly({x: 1}, self.voperand())
            return ('bare', x)
        raise ValueError('expected a symbol or parenthesis')

    def _close(self):
        if self.peek() != ('op', ')'):
            raise ValueError('missing closing parenthesis')
        self.take()

    def _group_dot(self):
        depth = 0
        for k in range(self.i, len(self.t)):
            if self.t[k] == ('op', '('):
                depth += 1
            elif self.t[k] == ('op', ')'):
                depth -= 1
                if depth == 0:
                    return k + 1 < len(self.t) and self.t[k + 1] == ('op', '·')
        raise ValueError('missing closing parenthesis')

    def voperand(self):
        sign = 1
        while self.peek() in (('op', '+'), ('op', '-')):
            if self.take()[1] == '-':
                sign = -sign
        t = self.peek()
        if t == ('op', '('):
            self.take()
            v = self.vexpr()
            self._close()
            return {k: sign * c for k, c in v.items()}
        if t is not None and t[0] == 'word':
            self.take()
            return {self.classify(t[1]): sign}
        raise ValueError('expected a vector expression after the dot product')

    def vexpr(self):
        sign = 1
        while self.peek() in (('op', '+'), ('op', '-')):
            if self.take()[1] == '-':
                sign = -sign
        v = self.vterm(sign)
        while self.peek() in (('op', '+'), ('op', '-')):
            s = 1 if self.take()[1] == '+' else -1
            for k, c in self.vterm(s).items():
                v[k] = v.get(k, 0) + c
        return {k: c for k, c in v.items() if c}

    def vterm(self, sign):
        coef = 1
        t = self.peek()
        if t is not None and t[0] == 'num':
            coef = self.take()[1]
        t = self.peek()
        if t == ('op', '('):
            self.take()
            v = self.vexpr()
            self._close()
            return {k: sign * coef * c for k, c in v.items()}
        if t is not None and t[0] == 'word':
            self.take()
            return {self.classify(t[1]): sign * coef}
        raise ValueError('expected a vector symbol or parenthesis')

    def classify(self, w):
        m = re.fullmatch(r'([Kk])(\d+)', w)
        if m:
            idx = int(m.group(2))
            if not (1 <= idx <= self.nedge):
                raise ValueError(f'K_{idx} out of range (edges are 1..{self.nedge})')
            return 'K' + str(idx)
        if w not in self.pnames:
            raise ValueError(f'unknown symbol {w!r} (use K_i or an external name)')
        return w

    def expand(self, name):
        return dict(self.ksub[name]) if name in self.ksub else {name: 1}

    # Full expansion of a vector expression into the k/p basis.
    def expand_vec(self, v):
        out = {}
        for name, c in v.items():
            for x, cx in self.expand(name).items():
                out[x] = out.get(x, 0) + c * cx
        return {k: v for k, v in out.items() if v}

    def dotpoly(self, v1, v2):
        e1 = self.expand_vec(v1)
        e2 = self.expand_vec(v2)
        out = {}
        for x, cx in e1.items():
            for y, cy in e2.items():
                atom = ('d',) + tuple(sorted((x, y)))
                out[(atom,)] = out.get((atom,), 0) + cx * cy
        return {m: c for m, c in out.items() if c}


# ============================== 4 · MOMENTUM-CONSERVATION RELATIONS ==============================
# The atom products are not independent: sum_j p_i.p_j = 0 for every external and
# sum_j p_j.k_a = 0 for every loop momentum.  This section reduces those relations to row
# echelon form and substitutes them into a polynomial.

# Relations: sum_j p_i.p_j = 0 for every external i, and sum_j p_j.k = 0 for every basis loop momentum.
def _relations(ctx, pnames, loop_first=False):
    names = sorted(pnames)
    ext_rows = []
    for a in names:
        r = {}
        for b in names:
            at = ('d',) + tuple(sorted((a, b)))
            r[at] = r.get(at, 0) + 1
        ext_rows.append(r)
    loop_rows = []
    for k in sorted(ctx['k_modes']):
        kn = 'k%d' % k
        r = {}
        for b in names:
            at = ('d',) + tuple(sorted((kn, b)))
            r[at] = r.get(at, 0) + 1
        loop_rows.append(r)
    return loop_rows + ext_rows if loop_first else ext_rows + loop_rows

# Substitute one atom in a linear relation.
def _sub_rel(r, piv, expr):
    c = r.pop(piv, 0)
    if not c:
        return r
    for a, ce in expr.items():
        r[a] = r.get(a, 0) + c * ce
        if r[a] == 0:
            del r[a]
    return r

# Substitute one atom in a polynomial; coefficients stay integers, scaled by a nonzero constant.
def _sub_poly(poly, piv, expr, den):
    if not poly:
        return poly
    m_max = max(mon.count(piv) for mon in poly)
    if not m_max:
        return poly
    base = {(a,): c for a, c in expr.items()}
    pw = [ {(): 1} ]
    for _ in range(m_max):
        pw.append(_mul(pw[-1], base))
    out = {}
    for mon, coef in poly.items():
        m = mon.count(piv)
        rest = tuple(a for a in mon if a != piv) if m else mon
        c0 = coef * den ** (m_max - m)
        for m2, c2 in pw[m].items():
            key = tuple(sorted(rest + m2)) if m2 else rest
            out[key] = out.get(key, 0) + c0 * c2
    return {k: v for k, v in out.items() if v}


# Pivot list for this region in reduced row echelon form, cached on the context.
def _subs(ctx, pnames, loop_first=False):
    key = '_elim_lf' if loop_first else '_elim'
    cached = ctx.get(key)
    if cached is not None:
        return cached
    order = lambda a: (atom_power(a[1], a[2], ctx), a)
    rows = []
    for r in _relations(ctx, pnames, loop_first):
        rows.append({a: Fraction(c) for a, c in r.items() if c})
    rows = [r for r in rows if r]
    rows.sort(key=lambda r: order(min(r, key=order)))
    subs = []
    for i in range(len(rows)):
        r = {a: c for a, c in rows[i].items() if c}
        if not r:
            continue
        piv = min(r, key=order)
        coef = r[piv]
        r = {a: c / coef for a, c in r.items()}
        rows[i] = r
        for j in range(len(rows)):
            if j == i:
                continue
            c = rows[j].get(piv, 0)
            if not c:
                continue
            nr = dict(rows[j])
            for a, ca in r.items():
                nr[a] = nr.get(a, 0) - c * ca
                if nr[a] == 0:
                    del nr[a]
            nr.pop(piv, None)
            rows[j] = nr
        d = 1
        for c in r.values():
            d = d * c.denominator // _gcd(d, c.denominator)
        ri = {a: int(c * d) for a, c in r.items()}
        subs.append((piv, {a: -c for a, c in ri.items() if a != piv}, ri[piv]))
    ctx[key] = subs
    return subs


# Reduce a polynomial by those relations; returns it unchanged when no pivot atom occurs.
def reduce_poly(poly, ctx, pnames, loop_first=False):
    subs = _subs(ctx, pnames, loop_first)
    pivots = {piv for piv, _e, _d in subs}
    if not any(a in pivots for mon in poly for a in mon):
        return poly
    out = poly
    for piv, expr, den in subs:
        out = _sub_poly(out, piv, expr, den)
        if not out:
            break
    return out


# ============================== 5 · POWER OF A REGION ==============================
# With a reduced polynomial in hand, N is the smallest power of a surviving monomial.
def atom_power(a, b, ctx):
    def mode(name):
        if name.startswith('k'):
            return ctx['k_modes'][int(name[1:])]
        return ctx['p_modes'][name]
    m1, n1, i1 = mode(a)
    m2, n2, i2 = mode(b)
    if i1 == i2 or i1 == 0 or i2 == 0:
        return m1 + m2 + min(n1, n2)
    return m1 + m2


# Parse once; returns the parser (with its top-level factors) and the polynomial.
def _parse(text, ctx, nedge, pnames, loop_first=False, raw=False, reduce_each=False):
    ksub = {('K%d' % i): dict(v) for i, v in ctx['K_expr'].items()}
    eliminated = None
    norm = normalize(text)
    toks, spans = _tokenize_pos(norm)
    parser = _PolyParser(toks, spans, norm, nedge, pnames, ksub)
    parser.ctx = ctx
    parser.reduce_each = reduce_each
    parser.eliminated = eliminated
    poly = parser.expr()
    if raw:
        return parser, poly
    poly = reduce_poly(poly, ctx, pnames, loop_first)
    if not poly:
        raise ValueError('the numerator vanishes identically in this region')
    if parser.peek() is not None:
        raise ValueError(f'unexpected extra input starting at {parser.peek()!r}')
    if not poly:
        raise ValueError('the numerator vanishes identically in this region')
    return parser, poly


# Smallest power among the monomials with a nonzero coefficient.
def _min_power(poly, ctx):
    best = None
    for mon, coef in poly.items():
        if not coef:
            continue
        p = sum(atom_power(a[1], a[2], ctx) for a in mon)
        if best is None or p < best:
            best = p
    return best


# Minimum power by increasing power level: only the vanishing low levels are reduced.
def _bucketed_min_power(poly, ctx, pnames, loop_first=False):
    cache = {}

    def pw(atom):
        v = cache.get(atom)
        if v is None:
            v = atom_power(atom[1], atom[2], ctx)
            cache[atom] = v
        return v

    buckets = {}
    for mon, coef in poly.items():
        buckets.setdefault(sum(pw(a) for a in mon), {})[mon] = coef
    for level in sorted(buckets):
        bucket = {m: c for m, c in buckets[level].items() if c}
        if not bucket:
            continue
        red = reduce_poly(bucket, ctx, pnames, loop_first)
        if not red:
            continue
        low, high = {}, {}
        for mon, coef in red.items():
            q = sum(pw(a) for a in mon)
            if q == level:
                low[mon] = coef
            else:
                high.setdefault(q, {})[mon] = coef
        if low:
            return level
        for q, part in high.items():
            tgt = buckets.setdefault(q, {})
            for mon, coef in part.items():
                tgt[mon] = tgt.get(mon, 0) + coef
                if tgt[mon] == 0:
                    del tgt[mon]
    return None


# Exact numerator power for one region.
def exact_power(text, ctx, nedge, pnames, loop_first=False):
    _parser, poly = _parse(text, ctx, nedge, pnames, loop_first, raw=True, reduce_each=True)
    res = _bucketed_min_power(poly, ctx, pnames, loop_first)
    if res is None:
        raise ValueError('the numerator vanishes identically in this region')
    return res


# Exact power N, the eliminated external, and the per-term factor breakdown.
def numerator_details(text, ctx, nedge, pnames):
    parser, poly = _parse(text, ctx, nedge, pnames, reduce_each=True)
    terms = []
    for factors, term_poly in parser.terms:
        items = []
        for f, txt in factors:
            p = _min_power(reduce_poly(f, ctx, pnames), ctx)
            if p is not None:
                items.append((txt, p))
        terms.append((items, _min_power(reduce_poly(term_poly, ctx, pnames), ctx)))
    return _min_power(poly, ctx), parser.eliminated, terms


# Routing self-check: the power of K_i.K_i must equal V(edge mode) for every edge.
def edge_routing_check(ctx, em, nedge, pnames):
    bad = []
    for i, md in enumerate(em, 1):
        got = exact_power(r'K_%d\cdot K_%d' % (i, i), ctx, nedge, pnames)
        want = V(md)
        if got != want:
            bad.append((i, got, want))
    return bad
