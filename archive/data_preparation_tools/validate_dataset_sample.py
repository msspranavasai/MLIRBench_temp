#!/usr/bin/env python3
"""Compatibility command: canonical shared-input audit, replacing legacy single-seed labels.

Use `python run.py validate --sample --output <new-report>.jsonl` for the full CLI.
"""
import argparse
from pathlib import Path
from artifact import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/processed/MLIRBench/master/master.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=30)
    args = parser.parse_args()
    raise SystemExit(main(['validate', '--sample', '--dataset', str(args.dataset), '--output', str(args.output), '--timeout', str(args.timeout)]))
