r"""numerator_power.py - the mode algebra for this side, plus thin wrappers.

The parser, the region context and the recursive evaluation are shared and live in
collinear/numerator_core.py; only the scalar product of two modes differs between the
2->3 and the Regge side, and it is defined below.  The numerator power counting is
beta on every side.
"""

import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
COLLINEAR = os.path.dirname(BASE)
if COLLINEAR not in sys.path:
    sys.path.insert(0, COLLINEAR)


def _parts(x):
    if x[0] == 'H':
        return 0, 0, None, None
    if x[0] == 'S':
        return x[1], 0, None, None
    _, d, n, mem, m = x
    return m, n, d, mem


# Scalar product of two 2->3 modes: same wide direction -> m1+m2+min(n1,n2); same C23 pair ->
# m1+m2+1 (plus min(n1,n2) when both carry the same member leg); otherwise m1+m2.
def _atom_pow(X, Y):
    m1, n1, d1, k1 = _parts(X)
    m2, n2, d2, k2 = _parts(Y)
    if d1 in (1, 4, 5) and d2 in (1, 4, 5):
        return m1 + m2 + (min(n1, n2) if d1 == d2 else 0)
    if d1 == 23 and d2 == 23:
        same = (k1 is not None and k1 == k2)
        return m1 + m2 + 1 + (min(n1, n2) if same else 0)
    return m1 + m2


# ============================== NUMERATOR ==============================

# ---------------------------------------------------------------- preprocessing
# Drop LaTeX decoration: '$', \cdot -> '·', leftover backslashes -> spaces, subscripts 'K_{1}'/'K_1' -> 'K1'.


BETA = True


from numerator_core import (parse_numerator, fmt_ast, region_context,   # noqa: F401
                           numerator_power as _core_numerator_power,
                           term_breakdown as _core_term_breakdown)


# Thin wrappers: the shared core takes this side's atom rule as an argument.
def numerator_power(ast, ctx):
    return _core_numerator_power(ast, ctx, _atom_pow)


def term_breakdown(ast, ctx):
    return _core_term_breakdown(ast, ctx, _atom_pow)
