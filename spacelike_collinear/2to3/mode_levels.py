#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mode_levels.py — first-appearance loop levels of internal modes, and the
cut-chain refinement levels that follow from them.

(Public core; graduated from the private mode_level_prediction.py on
2026-09-21 so skeleton23 can derive cut levels per graph.  The private file
re-exports these functions for the interactive tool.)

predict(ext_tokens, Lmax, momenta=None)
    The mode ladder: seeds (external modes at L=0) -> vee closure (free) ->
    wedge batches (+1, complete batch) -> messenger tower (+2 anchored:
    max(prev_mess + 2, cur + 1)) -> close-out.  Returns
    {mode tuple: first-appearance level} plus a log dict.
    Messenger tower (this class of inputs): S^1, S^1C23, S^2, S^2C23, ...
    (wide-type / pair-type interleaved; chain-checked 2026-09-20/21).

derive_external_modes(momenta)
    L=0 seeds: finite members of the vee closure of the five momentum modes.

messenger_eligibility / messenger_gate
    The item-[3] external acceptance gate.
    kind A (S^m): >= 3 eligible directions.
    kind B (S^mC23): BOTH pair legs attachable (depth >= m+1) AND >= 1 wide.

cut_chain_levels(ext_mode, L)
    Per-graph cut-chain refinement levels ("L enters the decision",
    wired into skeleton23 on 2026-09-21).  For each wide leg i the level is
    (max carrier power n among C_i^n / S^m C_i^n modes present within L) - 1;
    for the pair chains the level is the max k among C_mem^k C23 modes
    present within L (C2C23 = level 1).  k0: all zeros (no fine structure at
    any L).  Convention matches skeleton23's LEVELS.

Tree-anchored messengers (小马 2026-09-21): a degree-m S^m kernel may
anchor directly on a tree-level target -- an external leg whose mode is an
n_i=0 target of the kernel; it may then appear as early as L = 0 + 2 (k4:
S^2 relevant to the C2C23/C3C23 seeds -> S^2 at L = 2, verified against
pysd_cache region data).  Members may share a level (k4: S^2@2 with S@2).

Known caveat (pending): same-direction joins S^m C_i^n v C_i^inf (m >= 2)
differ between fri23 (C_i^{2m+n}) and wide_angle (sigma = m+n); affects
extrapolated high-L output only (can inflate derived levels for L >= 9 on
k2-like inputs -> those hit the engine's "deeper chains not implemented"
guard).  See fri23_vs_wa_inf_join_note_20260921.md (private).
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fri23 as F
import kin23


# ---------------------------------------------------------------- algorithm
def predict(ext_tokens, Lmax, kin='k1', momenta=None):
    """Returns (levels: {mode tuple -> level}, log: dict).

    momenta: optional override for the five external-momentum modes
             (list of fri23 mode tuples); default = kin23.ext_modes(kin).
    """
    ext_modes = [F.parse(t) for t in ext_tokens]
    ext_mom = list(momenta) if momenta is not None else list(kin23.ext_modes(kin).values())

    levels = {}
    messenger_log, wedge_log = [], []
    exc_count = [0]

    def add(m, lvl):
        if m not in levels:
            levels[m] = lvl
            return True
        return False

    for m0 in ext_modes:
        add(m0, 0)

    # ---- take vee closure: fixpoint over join23 (modes x modes, modes x momenta, momenta x momenta)
    def vee_closure():
        changed = True
        while changed:
            changed = False
            pool = [(m, levels[m]) for m in list(levels)] + [(e, 0) for e in ext_mom]
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    try:
                        r = F.join23(pool[i][0], pool[j][0])
                    except Exception:
                        exc_count[0] += 1
                        continue
                    if r not in levels:
                        levels[r] = max(pool[i][1], pool[j][1])
                        changed = True

    # ---- wedge check: missing pair meet products over the current set
    def wedge_missing():
        items = list(levels)
        miss = {}
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                try:
                    r = F.meet23(items[i], items[j])
                except Exception:
                    exc_count[0] += 1
                    continue
                if r not in levels:
                    miss[r] = None
        return list(miss)

    # ---- messenger tower (the single place to swap in the general rule)
    pair_enabled = True  # fri23: the collinear (2,3) pair always brings the C23 family in (via the momenta)
    def messenger_iter():
        m = 1
        while True:
            yield F.S(m)
            if pair_enabled:
                yield F.P(0, 0, m)
            m += 1

    mit = messenger_iter()

    def next_messenger():
        for cand in mit:
            if cand in levels:
                continue
            m = F.m_of(cand)
            kind = 'A' if cand[0] == 'S' else 'B'
            ok, detail = messenger_gate(ext_mom, m, kind)
            if not ok:
                return None, ('messenger tower capped by item-[3]: degree-%d (%s) %s'
                              % (m, kind, detail))
            return cand, None
        return None, 'no new messenger -- end'

    # ---- main loop
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
            for r in batch:
                add(r, lvl)
            wedge_log.append((lvl, sorted(F.name(r) for r in batch)))
            cur = lvl
        nxt, why = next_messenger()
        if nxt is None:
            stop = why
            break
        lvl = max(prev_mess + 2, cur + 1)
        # Tree-anchored shortcut (小马 2026-09-21): an S^m kernel with a
        # tree-level n_i=0 target may appear as early as 0 + 2 -- it does not
        # have to wait for the tower chain (k4: S^2@2 together with S@2).
        if nxt[0] == 'S':
            t = _tree_anchor_level(ext_mom, F.m_of(nxt))
            if t is not None and t + 2 < lvl:
                lvl = t + 2
        if lvl > Lmax:
            stop = 'next messenger %s would land at L=%d > Lmax' % (F.name(nxt), lvl)
            break
        add(nxt, lvl)
        prev_mess = lvl
        messenger_log.append((lvl, F.name(nxt)))

    # ---- close-out: capture anything else the last additions force within Lmax
    for _ in range(5):
        vee_closure()
        batch = wedge_missing()
        if not batch:
            break
        lvl = max(levels.values()) + 1
        if lvl > Lmax:
            break
        for r in batch:
            add(r, lvl)
        wedge_log.append((lvl, sorted(F.name(r) for r in batch)))

    return levels, dict(messenger_log=messenger_log, wedge_log=wedge_log,
                        stop=stop, exc=exc_count[0])


def derive_external_modes(momenta):
    """External modes at L = 0: the finite members of the vee closure of the
    five external-momentum modes (infinity-containing entries act only as
    closure intermediates and are not listed)."""
    seen = set(momenta)
    changed = True
    while changed:
        changed = False
        cur = list(seen)
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                try:
                    r = F.join23(cur[i], cur[j])
                except Exception:
                    continue
                if r not in seen:
                    seen.add(r)
                    changed = True
    out = [F.name(m) for m in seen]
    out = [s for s in out if '∞' not in s]
    out.sort(key=lambda s: (s == 'H', s))
    return out


# ---- item-[3] external eligibility  (messenger_completion_draft_20260921.md)
def messenger_eligibility(momenta, m, kind):
    """Eligible directions for a degree-m messenger from the external legs.

    kind 'A' (S^m):     wide legs need depth(e) >= m;  pair legs depth(e) >= m;
    kind 'B' (S^mC23):  wide legs need depth(e) >= m;  pair legs depth(e) >= m+1.
    (fri23 depth_of: wide C_i^a -> a;  pair C_mem^a C23 -> a+1.)
    Pair members are counted separately ("pair2", "pair3") -- change here if a
    single 23-direction count is intended.

    NOTE 2026-09-21: this returns the RAW eligible-direction set; the acceptance
    shape is applied in messenger_gate() (kind B requires both pair legs plus
    at least one wide leg).
    """
    dirs = set()
    for e in momenta:
        d = F.d_of(e)
        if d is None:
            continue
        if d == 23:
            # m_2 and/or m_3 = 0 (bare C23): C23 counts as ONE direction
            # (小马 2026-09-21); the current label/display stands.
            need = m + 1 if kind == 'B' else m
            if F.depth_of(e) >= need:
                dirs.add('pair%d' % (F.mem_of(e) or 0))
        else:
            if F.depth_of(e) >= m:
                dirs.add(d)
    return dirs


def messenger_gate(momenta, m, kind):
    """Acceptance gate (item [3]) for advancing the tower to degree m.

    kind 'A' (S^m):     >= 3 eligible directions suffice.
    kind 'B' (S^mC23):  targets are pinned at power m -- {C2^mC23, C3^mC23,
        wide C_i^m} -- so BOTH pair legs must be attachable (depth >= m+1)
        AND at least one wide leg (depth >= m).  [2026-09-21: e.g. bare-C23
        pair legs fail this and cap the tower -- matches the k0 result.]
    Returns (ok, detail) with a human-readable reason when not ok.
    """
    dirs = messenger_eligibility(momenta, m, kind)
    if kind == 'A':
        ok = len(dirs) >= 3
        detail = ('has only %d eligible direction(s) %s'
                  % (len(dirs), sorted(dirs, key=str)))
        return ok, detail
    pair_ok = ('pair2' in dirs) and ('pair3' in dirs)
    wide_ok = any(d in (1, 4, 5) for d in dirs)
    if pair_ok and wide_ok:
        return True, ''
    parts = []
    if not pair_ok:
        depths = ', '.join('p%d depth %s (need >= %d)' % (i, F.depth_of(e), m + 1)
                           for i, e in enumerate(momenta, 1) if F.d_of(e) == 23)
        parts.append('pair side not attachable: %s' % (depths or 'no pair legs'))
    if not wide_ok:
        parts.append('no wide leg attachable (need depth >= %d)' % m)
    return False, '; '.join(parts)


# ------------------------------------------------- cut-chain levels for engines
def _tree_anchor_level(momenta, m):
    """Earliest level of an n_i=0 target for the S^m kernel among the
    external legs (tree level -> 0; None when no leg qualifies).

    A leg e qualifies when depth(e) - (m - m_of(e)) == 0, i.e. the leg's
    mode is exactly a minimal target the kernel can be relevant to.
    (小马 2026-09-21: k4 -- C2C23/C3C23 at tree level enable S^2@2.)"""
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


def cut_chain_levels(ext_mode, L):
    """Cut-chain refinement levels needed by a graph of loop count L.

    ext_mode: dict {'p1'..'p5': mode tuple} (e.g. kin23.ext_modes(kin)).
    Returns {'p1','p2','p3','p4','p5'} of ints, skeleton23 LEVELS convention:
      wide i (1/4/5): number of refined levels = (max carrier power n among
        direction-i modes C_i^n / S^m C_i^n within L loops) - 1;
      pair 2/3:      max k among C_mem^k C23 modes within L (C2C23 = 1);
      k0:            all zeros (no fine structure at any loop count).

    NOTE: levels > 1 are not implemented in the engines yet (guard raises);
    currently only k2 can reach level 2+ (and then only at L >= 6/8, where
    no corpus graphs exist yet).

    k4 (fixed 2026-09-21, 小马): S^2 -- and its vee products C1^2/C4^2/C5^2
    -- now sit at L=2 (the degree-2 kernel anchors directly on the tree-level
    C2C23/C3C23 targets; see _tree_anchor_level).  Verified against the
    pysd_cache frog dumps; the earlier temporary floor is gone.
    """
    mom = [ext_mode[k] for k in ('p1', 'p2', 'p3', 'p4', 'p5')]
    tokens = derive_external_modes(mom)
    # Headroom note: predict's "reached Lmax" gate can prune additions that
    # would still land at <= L (same-level cases, e.g. k4's S^2@2 when L=2);
    # run with L + 2 headroom and filter explicitly to first-level <= L.
    table, _log = predict(tokens, L + 2, momenta=mom)
    lv = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for m, mlv in table.items():
        if mlv > L:
            continue
        d = F.d_of(m)
        if d in (1, 4, 5):
            n = F.n_of(m)
            if n is not None and n != F.INF and n >= 1:
                lv[d] = max(lv[d], n - 1)
        elif d == 23:
            k = F.n_of(m)
            if k is not None and k != F.INF and k >= 1:
                mem = F.mem_of(m)
                if mem in (2, 3):
                    lv[mem] = max(lv[mem], k)
    return {'p1': lv[1], 'p2': lv[2], 'p3': lv[3], 'p4': lv[4], 'p5': lv[5]}
