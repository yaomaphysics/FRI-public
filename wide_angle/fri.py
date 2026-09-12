#!/usr/bin/env python3
"""fri.py — Facet Region Interpreter: main program (FRI tool, 2026-08-15).

Given a graph (topology + external kinematics), find ALL its regions:

  1. enumerate all regions (C/H family + layered family, compressed),
  2. list them with numbers (edge-mode sequence in input edge order),
  3. menu: inspect specific regions (mode subgraphs + loop numbers +
     independent loop momenta / forced lines), Lee-Pomeransky parametric
     representation (scaling vectors), or classification by
     characteristic (softest) modes.

Usage:
    python3 fri.py

    internal_lines = [[1,2],[1,3],...]          # edge list
    externals      = {'p1':[1,'C1^2'], ...}     # {name: [vertex, mode_str]}

Mode syntax: H / S / S^m / C_i / C_i^n / C_i^inf / SC_i / SC_i^n / S^mC_i^n.

This file is a thin wrapper: the interactive browser lives in
facet_regions_interactive.py (same directory).
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facet_regions_interactive import main

if __name__ == '__main__':
    main()
