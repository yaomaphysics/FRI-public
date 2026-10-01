#!/usr/bin/env python3
"""facet_regions_interactive.py — unified Facet Region Interpreter entry point.

First choose a framework; the framework's own interactive browser then runs (all paths relative to this project root):

    [1] wide-angle           -> wide_angle/facet_regions_interactive.py
    [2] collinear
          [1] 2->3           -> collinear/2to3/facet_regions_interactive.py
          [2] 2->2 (Regge)   -> collinear/regge/facet_regions_interactive.py
          [3] mode stepper   -> 2->3: collinear/2to3/mode_level_interactive.py
                                2->2: collinear/regge/mode_level_interactive.py

Every browser shares the same workflow: give a graph (topology + external kinematics ONLY — the cut formalism stays internal), enumerate ALL its regions, list them, then
    1) inspect specific regions  — per-mode subgraphs + loop numbers (+ a concrete independent-loop-momentum basis with forced lines),
    2) Lee-Pomeransky parametric representation (scaling vectors),
    3) classification by characteristic (softest) mode.

([3] is the exception: a stepper for the mode first-appearance ladder — no graph needed, just the external-momentum modes.)

Usage: python3 facet_regions_interactive.py
"""
import os
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _ask(prompt, valid, default=None):
    while True:
        s = input(prompt).strip().lower()
        if not s and default is not None:
            return default
        if s in valid:
            return s
        print('  ! choose one of: %s' % '/'.join(valid))


def run_backend(path, label):
    print('=' * 72)
    print('launching %s interactive ...' % label)
    print('=' * 72)
    # Each framework runs in its own process: same-named modules (skeleton, primitives, ...) cannot clash.
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        subprocess.call([sys.executable, path])
    finally:
        signal.signal(signal.SIGINT, old)


def run_wide_angle():
    run_backend(os.path.join(HERE, 'wide_angle', 'facet_regions_interactive.py'), 'wide-angle')


def run_regge():
    run_backend(os.path.join(HERE, 'collinear', 'regge', 'facet_regions_interactive.py'), 'collinear 2->2 (Regge)')


def run_two_to_three():
    run_backend(os.path.join(HERE, 'collinear', '2to3', 'facet_regions_interactive.py'), 'collinear 2->3')


def run_mode_ladder():
    print()
    print('  mode first-appearance stepper:')
    print('    [1] 2->3')
    print('    [2] 2->2 (Regge)')
    print('    [b] back')
    s = _ask('stepper [1/2/b] > ', ('1', '2', 'b'))
    if s == '1':
        run_backend(os.path.join(HERE, 'collinear', '2to3', 'mode_level_interactive.py'), 'mode first-appearance ladder')
    elif s == '2':
        run_backend(os.path.join(HERE, 'collinear', 'regge', 'mode_level_interactive.py'), 'mode first-appearance ladder (2->2 Regge)')


def main():
    print('=' * 72)
    print('Facet Region Interpreter (FRI)')
    print('=' * 72)
    while True:
        print()
        print('  frameworks:')
        print('    [1] wide-angle')
        print('    [2] collinear')
        print('    [q] quit')
        f = _ask('framework [1/2/q] > ', ('1', '2', 'q'))
        if f == 'q':
            break
        if f == '1':
            run_wide_angle()
            continue
        while True:
            print()
            print('  collinear:')
            print('    [1] 2->3 scattering (p2 collinear to p3)')
            print('    [2] 2->2 scattering (Regge limit, p1 collinear to p3 while p2 collinear to p4)')
            print('    [3] mode first-appearance stepper')
            print('    [b] back')
            s = _ask('sub-framework [1/2/3/b] > ', ('1', '2', '3', 'b'))
            if s == 'b':
                break
            if s == '1':
                run_two_to_three()
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
