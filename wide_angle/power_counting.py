r"""
power_counting.py - expansion power of a facet region.

The integrand of a region scales as lambda^{A - B}:
    A = (2 - eps) * sum_X rank_X * V(X)   (integration measure),
    B = sum_i V(edge mode_i)              (product of propagators),
with V(S^m C_i^n) = 2m + n.
The computation of numerator power N is not shown here (or, the integral is regarded as a scalar integral); see numerator_power.py (BETA).
"""

from primitives import V

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
    elif a1 == 1:
        exp = f'{a0} + \\epsilon'
    elif a1 == -1:
        exp = f'{a0} - \\epsilon'
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


