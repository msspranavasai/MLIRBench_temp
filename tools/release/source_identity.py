"""Compute the same stdlib-only source identity used in reviewer evidence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from mlirbench.ae import source_fingerprint
if __name__=='__main__': print(source_fingerprint(ROOT))
