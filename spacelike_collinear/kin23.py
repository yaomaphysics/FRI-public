#!/usr/bin/env python3
"""kin23.py — external-virtuality kinematics table for the spacelike-collinear 2->3 scattering.

The small parameter is t1; s23 ~ t1 for every k.
Per kinematics:
  ms = (m1 ... m5) virtuality orders of the external legs: an integer m = the leg's virtuality
               scales as t1**m; INF (representing "infinity") = exactly on-shell.
  mass_exprs = the five m_i**2 expressions handed to pySecDec.

External modes follow the established formulas (here we always require s23)
    p_i = C_i^{m_i}  (i = 1,4,5),   p_2 = C_2^{m_2-1} C_23,   p_3 = C_3^{m_3-1} C_23.
    Note that m = 1 degenerates the pair member: C_2^0 C_23 = C23, member marker dropped.

Usage:  import kin23 as K
        K.ext_modes('k3')       -> fri23 ext_mode dict {'p1'..'p5'}
        K.mass_exprs('k3')      -> ['t1*m1sq', 't1*t1*m2sq', '0', '0', '0']
"""
from fri23 import W, P, INF

KIN = {
    'k0': dict(ms=(1, 1, 1, 1, 1),
               mass_exprs=['t1*m1sq', 't1*m2sq', 't1*m3sq', 't1*m4sq', 't1*m5sq']), # all the p_i^2 are of the same size (~t1)
    'k1': dict(ms=(INF, INF, INF, INF, INF),
               mass_exprs=['0', '0', '0', '0', '0']), # p_i^2 =0 for all i
    'k2': dict(ms=(1, INF, INF, INF, INF),
               mass_exprs=['t1*m1sq', '0', '0', '0', '0']), # p_1^2 ~ t1, p_i^2 = 0 for other i
    'k3': dict(ms=(1, 2, INF, INF, INF),
               mass_exprs=['t1*m1sq', 't1*t1*m2sq', '0', '0', '0']), # p_1^2 ~ t1, p_2^2 ~ t1^2, p_i^2 = 0 for other i
    'k4': dict(ms=(INF, 2, 2, INF, INF),
               mass_exprs=['0', 't1*t1*m2sq', 't1*t1*m3sq', '0', '0']), # p_2^2 ~ p_3^2 ~ t1^2, p_i^2 = 0 for other i
}
KIN_ORDER = ['k0', 'k1', 'k2', 'k3', 'k4']


def mass_exprs(kind):
    return list(KIN[kind]['mass_exprs'])


def ext_modes(kind):
    """fri23 ext_mode dict {'p1'..'p5'} for the given kinematics."""
    m1, m2, m3, m4, m5 = KIN[kind]['ms']

    def pair(m, mem):
        k = m - 1 # note that INF - 1 = INF
        return P(k, mem if k >= 1 else 0)

    return {'p1': W(1, m1), 'p2': pair(m2, 2), 'p3': pair(m3, 3),
            'p4': W(4, m4), 'p5': W(5, m5)}
