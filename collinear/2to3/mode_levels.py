#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mode_levels.py — first-appearance loop levels of internal modes and the cut-chain levels that follow from them.

The mode ladder for the collinear 2->3 kinematics:
  * seeds: the external modes (L = 0) and the finite vee closure of the five external momenta;
  * vee closure: a join is free when one operand is a messenger, and costs one extra loop otherwise;
  * wedge: every meet costs ca + cb + 1;
  * messenger: a tree-level target gives t + 2, otherwise the component gives its level + 2
    (S^m C23 <- S^m, S^{m+1} <- S^m C23); gated by messenger_gate and by "the degree-m mode exists two loops earlier".

predict() returns {mode tuple: first-appearance level} together with a log of the messenger and wedge rounds.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import region_filter as F
from primitives import ext_modes


# ==================== MODE LADDER ====================

# Grow the mode ladder to level Lmax; returns ({mode tuple: level}, log). momenta overrides the five external-momentum modes.
def predict(ext_tokens, Lmax, momenta=None):
    ext_modes = [F.parse(t) for t in ext_tokens]
    ext_mom = list(momenta) if momenta is not None else list(ext_modes((F.INF,) * 5).values())

    levels = {}
    routes = {}
    exc_count = [0]

    # Record mode m at lvl when it is new or cheaper; the route travels with the level. True on a change.
    def add(m, lvl, route=None):
        old = levels.get(m)
        if old is None or lvl < old:
            levels[m] = lvl
            if route is not None:
                routes[m] = route
            return True
        return False

    # True when m was produced by the messenger (its +2 already pays for the extra edge of a join).
    def is_messenger(m):
        r = routes.get(m)
        return bool(r) and r[0] == 'messenger'

    for m0 in ext_modes:
        add(m0, 0, ('seed',))

    # One sweep over every pair of `items`; `visit` returns True when it changed something.
    def pair_sweep(items, visit):
        changed = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                try:
                    changed = visit(items[i], items[j]) or changed
                except Exception:
                    continue
        return changed

    # Fixpoint of `visit` over the mode pairs; `extra` adds (mode, 0) entries to the pool.
    def closure(visit, extra=()):
        any_change = False
        while True:
            items = [(m, levels[m]) for m in list(levels)] + list(extra)
            if not pair_sweep(items, visit):
                return any_change
            any_change = True

    # Vee: a join is free when one operand is a messenger, and costs +1 otherwise.
    def vee_closure():
        def visit(pa, pb):
            (a, ca), (b, cb) = pa, pb
            extra = 0 if (is_messenger(a) or is_messenger(b)) else 1
            return add(F.join(a, b), max(ca, cb) + extra, ('vee', a, b))
        return closure(visit, [(e, 0) for e in ext_mom])

    # Wedge: every meet costs ca + cb + 1.
    def wedge_closure():
        def visit(pa, pb):
            (a, ca), (b, cb) = pa, pb
            lvl = ca + cb + 1
            return lvl <= Lmax and add(F.meet(a, b), lvl, ('wedge', a, b))
        return closure(visit)

    # Mode of one direction at the messenger's degree: C_i^m (wide), C_mem^{m-1}C23 (A) or C_mem^mC23 (B).
    def target_mode(i, m, kind):
        if i in (1, 4, 5):
            return F.W(i, m)
        if kind == 'B':
            return F.P(m, i)
        return F.P(m - 1, i) if m > 1 else F.P(0, 0)

    # Tower predecessor of a candidate: S^m C23 <- S^m, S^{m+1} <- S^m C23; S^1 has none.
    def component_of(cand):
        m = F.m_of(cand)
        if cand[0] == 'S':
            return F.P(0, 0, m - 1) if m > 1 else None
        return F.S(m)

    # Candidate messengers in tower order: S^m, then S^m C23.
    def messenger_iter():
        m = 1
        while True:
            yield F.S(m)
            yield F.P(0, 0, m)
            m += 1

    mit = messenger_iter()

    # Next tower candidate that passes the degree-m acceptance gate, or (None, reason).
    def next_messenger():
        for cand in mit:
            m = F.m_of(cand)
            kind = 'A' if cand[0] == 'S' else 'B'
            ok, detail = messenger_gate(ext_mom, m, kind)
            if not ok:
                return None, ('messenger tower capped by item-[3]: degree-%d (%s) %s' % (m, kind, detail))
            return cand, None
        return None, 'no new messenger -- end'

    # Alternate the two closures with one messenger step until nothing more can be added.
    stop = None
    while True:
        ch = vee_closure()
        ch = wedge_closure() or ch
        cur = max(levels.values())
        if cur >= Lmax:
            stop = 'gate: level L=%d reached Lmax=%d' % (cur, Lmax)
            break
        # Take tower candidates in order until one lowers a level or is new; a mode already there may still be too late.
        added = False
        while True:
            nxt, why = next_messenger()
            if nxt is None:
                stop = why
                break
            # A tree-level target gives t + 2; otherwise the level is the component's level + 2.
            t = _tree_anchor_level(ext_mom, F.m_of(nxt))
            comp = component_of(nxt)
            if t is not None:
                lvl, rec = t + 2, None
            elif comp is None:
                lvl, rec = 2, None
            elif comp in levels:
                lvl, rec = levels[comp] + 2, comp
            else:
                stop = 'messenger %s: component %s is not in the ladder' % (F.name(nxt), F.name(comp))
                break
            # One eligible direction must already carry the degree-m mode two loops earlier.
            mdeg = F.m_of(nxt)
            kind = 'A' if nxt[0] == 'S' else 'B'
            if not any(levels.get(target_mode(i, mdeg, kind), F.INF) <= lvl - 2 for i in (1, 4, 5, 2, 3)):
                stop = 'messenger %s: no direction carries the degree-%d mode two loops earlier' % (F.name(nxt), mdeg)
                break
            if lvl > Lmax:
                stop = 'next messenger %s would land at L=%d > Lmax' % (F.name(nxt), lvl)
                break
            if add(nxt, lvl, ('messenger', lvl, rec)):
                added = True
                break
        ch = ch or added
        if not ch:
            break

    return levels, dict(routes=routes, stop=stop)


# External modes at L = 0: the finite members of the vee closure of the five external-momentum modes.
def derive_external_modes(momenta):
    seen = set(momenta)
    changed = True
    while changed:
        changed = False
        cur = list(seen)
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                try:
                    r = F.join(cur[i], cur[j])
                except Exception:
                    continue
                if r not in seen:
                    seen.add(r)
                    changed = True
    out = [F.name(m) for m in seen]
    out = [s for s in out if '∞' not in s]
    out.sort(key=lambda s: (s == 'H', s))
    return out


# ==================== MESSENGER ELIGIBILITY & GATE ====================

# Eligible external directions for a degree-m messenger of the given kind ('A' = S^m, 'B' = S^m C23).
def messenger_eligibility(momenta, m, kind):
    dirs = set()
    for e in momenta:
        d = F.d_of(e)
        if d is None:
            continue
        if d == 23:
            # m_2 and/or m_3 = 0 (bare C23): C23 counts as one direction.
            need = m + 1 if kind == 'B' else m
            if F.depth_of(e) >= need:
                dirs.add('pair%d' % (F.mem_of(e) or 0))
        else:
            if F.depth_of(e) >= m:
                dirs.add(d)
    return dirs


# Item-[3] acceptance gate for advancing the tower to degree m; returns (ok, detail) with a reason when not ok.
def messenger_gate(momenta, m, kind):
    dirs = messenger_eligibility(momenta, m, kind)
    if kind == 'A':
        ok = len(dirs) >= 3
        detail = ('has only %d eligible direction(s) %s' % (len(dirs), sorted(dirs, key=str)))
        return ok, detail
    pair_ok = ('pair2' in dirs) and ('pair3' in dirs)
    wide_ok = any(d in (1, 4, 5) for d in dirs)
    if pair_ok and wide_ok:
        return True, ''
    parts = []
    if not pair_ok:
        depths = ', '.join('p%d depth %s (need >= %d)' % (i, F.depth_of(e), m + 1) for i, e in enumerate(momenta, 1) if F.d_of(e) == 23)
        parts.append('pair side not attachable: %s' % (depths or 'no pair legs'))
    if not wide_ok:
        parts.append('no wide leg attachable (need depth >= %d)' % m)
    return False, '; '.join(parts)


# ==================== CUT-CHAIN LEVELS ====================

# Earliest level (0, or None) at which an S^m kernel can anchor on a tree-level n_i=0 target among the external legs.
def _tree_anchor_level(momenta, m):
    for e in momenta:
        if F.d_of(e) is None:
            continue
        dep = F.depth_of(e)
        if dep in (None, F.INF):
            continue
        mi = m - F.m_of(e)
        if 1 <= mi <= m and dep - mi == 0:
            return 0
    return None


# Per-graph cut-chain powers (ext_mode: {'p1'..'p5'}): wide leg = sorted powers n of C_i^n with first level <= L;
# the 2/3 pair = sorted powers k of C_mem^k C23 with first level <= L; an empty set means no cut in that channel.
def cut_chain_levels(ext_mode, L):
    mom = [ext_mode[k] for k in ('p1', 'p2', 'p3', 'p4', 'p5')]
    tokens = derive_external_modes(mom)
    # Run with headroom L + 2 and keep only first levels <= L (the "reached Lmax" gate can prune same-level additions).
    table, _log = predict(tokens, L + 2, momenta=mom)
    wide = {1: set(), 4: set(), 5: set()}
    pair = {2: set(), 3: set()}
    for m, mlv in table.items():
        if mlv > L:
            continue
        d = F.d_of(m)
        if d in (1, 4, 5):
            n = F.n_of(m)
            if n is not None and n != F.INF and n >= 1:
                wide[d].add(n)
        elif d == 23:
            k = F.n_of(m)
            if k is not None and k != F.INF and k >= 1:
                mem = F.mem_of(m)
                if mem in (2, 3):
                    pair[mem].add(k)
    return {'p1': sorted(wide[1]), 'p2': sorted(pair[2]), 'p3': sorted(pair[3]),
            'p4': sorted(wide[4]), 'p5': sorted(wide[5])}


# Scalar chain depth per leg (the legacy view of the ladder): wide legs max power - 1, the 2/3 pair max k; 0 when no power appears.
# The shallow fixed-slot path needs only this scalar (0/1); the strict-overlap dev engine keeps it.
def cut_chain_depth(levels):
    out = {}
    for leg in ('p1', 'p2', 'p3', 'p4', 'p5'):
        v = levels[leg]
        if not v:
            out[leg] = 0
        elif leg in ('p1', 'p4', 'p5'):
            out[leg] = max(v) - 1
        else:
            out[leg] = max(v)
    return out
