#!/usr/bin/env python3
r"""
power_counting.py — scalar power counting for the Regge region browser.

A region's contribution scales as lambda^{A-B}: the loop measure contributes
    A = (2-eps) * sum_X r_X * V(X)      (one factor (lambda^{V(X)})^{2-eps} per independent loop momentum),
and the propagators B = sum_e V(em_e).  V is Mode.V (2m+n+1); sH and G (and the combined 'sH∪G' unit) count 1 per loop.

Returned as (a0, a1) for lambda^{a0 + a1*eps}; the numerator is not handled here.
"""
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from indep_loops import indep_loops, edge_momenta


# V of one unit (sH / G and the combined semihard unit count 1 per loop).
def _power_V(u):
    return 1 if isinstance(u, str) else u.V


# A: the loop measure, (a0, a1) for lambda^{a0 + a1*eps}.
def measure_power(results):
    p = sum(r['rank'] * _power_V(r['mode']) for r in results)
    return 2 * p, -p


# B: the product of propagators ~ lambda^B (all internal lines).
def integrand_power(em):
    return sum(0 if m is None else _power_V(m) for m in em)


# lambda^{a0 + a1*eps} as a LaTeX-ready string.
def fmt_power(a0, a1):
    if a1 == 0:
        exp = '%d' % a0
    elif a1 == 1:
        exp = '%d + \\epsilon' % a0
    elif a1 == -1:
        exp = '%d - \\epsilon' % a0
    elif a1 > 0:
        exp = '%d + %d\\epsilon' % (a0, a1)
    else:
        exp = '%d - %d\\epsilon' % (a0, -a1)
    return '\\lambda^{' + exp + '}'


# Print the power counting (measure, integrand, numerator, total power) for one region.
def show_power_counting(results, em, N=None):
    a0, a1 = measure_power(results)
    b = integrand_power(em)
    print('  integration measure = %s' % fmt_power(a0, a1))
    print('  integrand = %s' % fmt_power(-b, 0))
    if N is None:
        print('  power = %s' % fmt_power(a0 - b, a1))
    else:
        print('  numerator = %s' % fmt_power(N, 0))
        print('  power = %s' % fmt_power(a0 - b + N, a1))


# (m, n, family, member leg) of a Regge mode; H / S^m have no family.
