"""Convenience entry point; all implementation lives in src/mlirbench."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from mlirbench.cli import main

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:] or ['--help']))
