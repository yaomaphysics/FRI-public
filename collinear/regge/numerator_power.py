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

from primitives import G, SH


def _parts(x):
    if x == 'H' or x.n == -1:
        return 0, 0, None, None
    return x.m, x.n, x.fam, x.leg


# True for the semihard / Glauber units (the sH mode, G, or the combined 'sH∪G' unit).
def _is_semi(x):
    return x == SH or x == G or x == 'sH∪G'


# Scalar product of two Regge modes: same family -> m1+m2+1 (plus min(n1,n2) when both carry the
# same leg); otherwise m1+m2.  Among sH / G only K_sH . K_sH occurs (contributing 1).
def _atom_pow(X, Y):
    if _is_semi(X) or _is_semi(Y):
        if _is_semi(X) and _is_semi(Y):
            return 1
        raise ValueError('a numerator may only contain K_sH . K_sH among the semihard / Glauber modes')
    m1, n1, f1, l1 = _parts(X)
    m2, n2, f2, l2 = _parts(Y)
    if f1 is None or f2 is None:
        return m1 + m2
    if f1 == f2:
        same = (l1 is not None and l1 == l2)
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
