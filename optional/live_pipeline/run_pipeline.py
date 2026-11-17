"""Compatibility entry point for the maintained reviewer stage runner."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from mlirbench.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['pipeline',*sys.argv[1:]]))
