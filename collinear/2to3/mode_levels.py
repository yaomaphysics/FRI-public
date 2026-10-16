#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mode_levels.py — first-appearance loop levels of internal modes and the cut-chain levels that follow from them.

The mode ladder for the collinear 2->3 kinematics:
  * seeds: the external modes (L = 0) and the finite vee closure of the five external momenta;
  * vee closure: joins of modes and momenta are free (they do not raise the level);
  * wedge batches: meet products raise the level by one per complete batch;
  * messenger tower: a degree-m kernel starts at max(prev_messenger + 2, current + 1), gated by messenger_gate;
  * close-out: anything the last additions still force within the cap.

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
    messenger_log, wedge_log = [], []
    exc_count = [0]

    # Register mode m at level lvl once (no-op if already present), recording how it was produced.
    def add(m, lvl, route=None):
        if m not in levels:
            levels[m] = lvl
            if route is not None:
                routes[m] = route
            return True
        return False

    for m0 in ext_modes:
        add(m0, 0, ('seed',))

    # Vee closure: fixpoint over joins of modes and momenta (free, does not raise the level).
    def vee_closure():
        changed = True
        while changed:
            changed = False
            pool = [(m, levels[m]) for m in list(levels)] + [(e, 0) for e in ext_mom]
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    try:
                        r = F.join(pool[i][0], pool[j][0])
                    except Exception:
                        exc_count[0] += 1
                        continue
                    if r not in levels:
                        levels[r] = max(pool[i][1], pool[j][1])
                        routes[r] = ('vee', pool[i][0], pool[i][1], pool[j][0], pool[j][1])
                        changed = True

    # Meet products still missing from the current set (the next wedge batch).
    def wedge_missing():
        items = list(levels)
        miss = {}
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                try:
                    r = F.meet(items[i], items[j])
                except Exception:
                    exc_count[0] += 1
                    continue
                if r not in levels:
                    miss.setdefault(r, (items[i], items[j]))
        return list(miss.items())

    # The collinear 2->3 (2,3) pair always brings the C23 family in (via the momenta).
    pair_enabled = True

    # Candidate messengers in tower order: S^m, then S^m C23.
    def messenger_iter():
        m = 1
        while True:
            yield F.S(m)
            if pair_enabled:
                yield F.P(0, 0, m)
            m += 1

    mit = messenger_iter()

    # Next tower candidate that passes the degree-m acceptance gate, or (None, reason).
    def next_messenger():
        for cand in mit:
            if cand in levels:
                continue
            m = F.m_of(cand)
            kind = 'A' if cand[0] == 'S' else 'B'
            ok, detail = messenger_gate(ext_mom, m, kind)
            if not ok:
                return None, ('messenger tower capped by item-[3]: degree-%d (%s) %s' % (m, kind, detail))
            return cand, None
        return None, 'no new messenger -- end'

    # Alternate vee closure, one wedge batch, and one messenger step until the cap.
    prev_mess = 0
    stop = None
    while True:
        vee_closure()
        cur = max(levels.values())
        if cur >= Lmax:
            stop = 'gate: level L=%d reached Lmax=%d' % (cur, Lmax)
            break
        batch = wedge_missing()
        if batch:
            lvl = cur + 1
            if lvl > Lmax:
                stop = 'next wedge batch would land at L=%d > Lmax' % lvl
                break
            for r, pr in batch:
                add(r, lvl, ('wedge', pr[0], pr[1]))
            wedge_log.append((lvl, sorted(F.name(r) for r, _ in batch)))
            cur = lvl
        nxt, why = next_messenger()
        if nxt is None:
            stop = why
            break
        lvl = max(prev_mess + 2, cur + 1)
        anchored = None
        # Tree-anchored kernel: an S^m kernel with a tree-level n_i=0 target may appear as early as L = 2.
        if nxt[0] == 'S':
            t = _tree_anchor_level(ext_mom, F.m_of(nxt))
            if t is not None and t + 2 < lvl:
                lvl = t + 2
                anchored = t
        if lvl > Lmax:
            stop = 'next messenger %s would land at L=%d > Lmax' % (F.name(nxt), lvl)
            break
        add(nxt, lvl, ('messenger', lvl, prev_mess, cur, anchored))
        prev_mess = lvl
        messenger_log.append((lvl, F.name(nxt)))

    # Close-out: capture anything else the last additions force within Lmax.
    for _ in range(5):
        vee_closure()
        batch = wedge_missing()
        if not batch:
            break
        lvl = max(levels.values()) + 1
        if lvl > Lmax:
            break
        for r, pr in batch:
            add(r, lvl, ('wedge', pr[0], pr[1]))
        wedge_log.append((lvl, sorted(F.name(r) for r, _ in batch)))

    return levels, dict(messenger_log=messenger_log, wedge_log=wedge_log, routes=routes, stop=stop, exc=exc_count[0])


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
