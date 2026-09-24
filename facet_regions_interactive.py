#!/usr/bin/env python3
"""facet_regions_interactive.py — unified Facet Region Interpreter entry point.

First choose a framework; the framework's own interactive browser then runs (all paths relative to this project root):

    [1] wide-angle           -> wide_angle/facet_regions_interactive.py
    [2] spacelike-collinear
          [1] 2->3           -> spacelike_collinear/2to3/fri23_interactive.py
          [2] 2->2 (Regge)   -> spacelike_collinear/regge/fri_interactive.py
          [3] mode stepper   -> 2->3: spacelike_collinear/2to3/mode_level_interactive.py
                                2->2: spacelike_collinear/regge/mode_level_interactive.py

Every browser shares the same workflow: give a graph (topology + external kinematics ONLY — the cut formalism stays internal), enumerate ALL its regions, list them, then
    1) inspect specific regions  — per-mode subgraphs + loop numbers (+ a concrete independent-loop-momentum basis with forced lines),
    2) Lee-Pomeransky parametric representation (scaling vectors),
    3) classification by characteristic (softest) mode.

([3] is the exception: a stepper for the mode first-appearance ladder — no graph needed, just the external-momentum modes.)

Usage: python3 facet_regions_interactive.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(path, name):
    """Load a backend module from a file path under a unique name (avoids any basename clashes between the framework directories)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ask(prompt, valid, default=None):
    while True:
        s = input(prompt).strip().lower()
        if not s and default is not None:
            return default
        if s in valid:
            return s
        print('  ! choose one of: %s' % '/'.join(valid))


def run_backend(path, name, label):
    print('=' * 72)
    print('launching %s interactive ...' % label)
    print('=' * 72)
    try:
        mod = _load(path, name)
        mod.main()
    except KeyboardInterrupt:
        print('\n  [interrupted — back to the framework menu]')


def run_wide_angle():
    run_backend(os.path.join(HERE, 'wide_angle', 'facet_regions_interactive.py'), 'fri_wide_angle_interactive', 'wide-angle')


def run_regge():
    run_backend(os.path.join(HERE, 'spacelike_collinear', 'regge', 'fri_interactive.py'), 'fri_regge_interactive', 'spacelike-collinear 2->2 (Regge)')


def run_fri23():
    run_backend(os.path.join(HERE, 'spacelike_collinear', '2to3', 'fri23_interactive.py'), 'fri_fri23_interactive', 'spacelike-collinear 2->3')


def run_mode_ladder():
    print()
    print('  mode first-appearance stepper:')
    print('    [1] 2->3')
    print('    [2] 2->2 (Regge)')
    print('    [b] back')
    s = _ask('stepper [1/2/b] > ', ('1', '2', 'b'))
    if s == '1':
        run_backend(os.path.join(HERE, 'spacelike_collinear', '2to3', 'mode_level_interactive.py'), 'fri_mode_ladder', 'mode first-appearance ladder')
    elif s == '2':
        run_backend(os.path.join(HERE, 'spacelike_collinear', 'regge', 'mode_level_interactive.py'), 'fri_mode_ladder_regge', 'mode first-appearance ladder (2->2 Regge)')


def main():
    print('=' * 72)
    print('Facet Region Interpreter (FRI)')
    print('=' * 72)
    while True:
        print()
        print('  frameworks:')
        print('    [1] wide-angle')
        print('    [2] spacelike-collinear')
        print('    [q] quit')
        f = _ask('framework [1/2/q] > ', ('1', '2', 'q'))
        if f == 'q':
            break
        if f == '1':
            run_wide_angle()
            continue
        while True:
            print()
            print('  spacelike-collinear:')
            print('    [1] 2->3 scattering (p2 collinear to p3)')
            print('    [2] 2->2 scattering (Regge limit, p1 collinear to p3 while p2 collinear to p4)')
            print('    [3] mode first-appearance stepper')
            print('    [b] back')
            s = _ask('sub-framework [1/2/3/b] > ', ('1', '2', '3', 'b'))
            if s == 'b':
                break
            if s == '1':
                run_fri23()
            elif s == '2':
                run_regge()
            elif s == '3':
                run_mode_ladder()
    print('bye!')


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nbye!')
