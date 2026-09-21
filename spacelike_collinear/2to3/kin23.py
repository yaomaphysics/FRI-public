#!/usr/bin/env python3
"""kin23.py — external-virtuality kinematics for the spacelike-collinear 2->3 scattering.

Kinematics are GENERAL: any virtuality pattern ms = (m1, ..., m5), where
    m = 1, 2, ... : the leg's virtuality scales as t1**m;
    INF           : the leg is exactly on-shell.
The small parameter is t1; s23 ~ t1 for every kinematics.  The external
modes follow the established formulas
    p_i = C_i^{m_i}  (i = 1, 4, 5),  p_2 = C_2^{m_2-1} C_23,  p_3 = C_3^{m_3-1} C_23
    (m = 1 degenerates the pair member: C_2^0 C_23 = C23, member marker dropped).
They are implemented once, for an arbitrary ms — pass a raw tuple to
ext_modes()/mass_exprs(); no table edit is needed to use a new kinematics.

Cut levels are NOT part of this file: the refinement depth of every cut
chain is derived per graph from the mode first-appearance table —
mode_levels.cut_chain_levels(ext_mode, L), L = E - V + 1 (see skeleton23).

The named shortcuts below are presets used by the test corpora (each batch
is swept over k0..k4) and as defaults in the interactive browser:
  * k0, k1 — the TWO SPECIAL CASES, each with a dedicated engine:
      k0: all p_i^2 ~ t1 (no hierarchy)   -> k0 union construction;
      k1: all p_i^2 = 0 (lightlike)       -> the k1 engine;
  * k2, k3, k4 — examples of general mixed kinematics (kept as named presets
    because the validation reports are organised per k0..k4; the engines give
    them no special treatment).

Usage:  import kin23 as K
        K.ext_modes('k3')                    -> fri23 ext_mode dict {'p1'..'p5'}
        K.ext_modes((1, 2, INF, INF, INF))   -> the same, built directly
        K.mass_exprs('k3')                   -> pySecDec m_i**2 expressions
"""
from fri23 import W, P, INF

KIN = {
    # ---- the two special cases (dedicated engines) ----
    'k0': dict(ms=(1, 1, 1, 1, 1),
               mass_exprs=['t1*m1sq', 't1*m2sq', 't1*m3sq', 't1*m4sq', 't1*m5sq']), # all the p_i^2 are of the same size (~t1)
    'k1': dict(ms=(INF, INF, INF, INF, INF),
               mass_exprs=['0', '0', '0', '0', '0']), # p_i^2 =0 for all i
    # ---- examples of general mixed kinematics (see the docstring) ----
    'k2': dict(ms=(1, INF, INF, INF, INF),
               mass_exprs=['t1*m1sq', '0', '0', '0', '0']), # p_1^2 ~ t1, p_i^2 = 0 for other i
    'k3': dict(ms=(1, 2, INF, INF, INF),
               mass_exprs=['t1*m1sq', 't1*t1*m2sq', '0', '0', '0']), # p_1^2 ~ t1, p_2^2 ~ t1^2, p_i^2 = 0 for other i
    'k4': dict(ms=(INF, 2, 2, INF, INF),
               mass_exprs=['0', 't1*t1*m2sq', 't1*t1*m3sq', '0', '0']), # p_2^2 ~ p_3^2 ~ t1^2, p_i^2 = 0 for other i
}
KIN_ORDER = ['k0', 'k1', 'k2', 'k3', 'k4']


def mass_exprs(kind_or_ms):
    """The five m_i**2 expressions handed to pySecDec.
    Accepts a preset name or a raw (m1, ..., m5) virtuality tuple."""
    if not isinstance(kind_or_ms, str):
        return ['0' if m == INF else
                ('t1*m%dsq' % (i + 1) if m == 1
                 else 't1**%d*m%dsq' % (m, i + 1))
                for i, m in enumerate(kind_or_ms)]
    return list(KIN[kind_or_ms]['mass_exprs'])


def ext_modes(kind_or_ms):
    """fri23 ext_mode dict {'p1'..'p5'} for a preset name or a raw
    (m1, ..., m5) virtuality tuple (general kinematics)."""
    ms = KIN[kind_or_ms]['ms'] if isinstance(kind_or_ms, str) else tuple(kind_or_ms)
    m1, m2, m3, m4, m5 = ms

    def pair(m, mem):
        k = m - 1 # note that INF - 1 = INF
        return P(k, mem if k >= 1 else 0)

    return {'p1': W(1, m1), 'p2': pair(m2, 2), 'p3': pair(m3, 3),
            'p4': W(4, m4), 'p5': W(5, m5)}
