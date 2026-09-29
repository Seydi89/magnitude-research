"""Compatibility entry point for the maintained token-to-completion diagnostics.

Usage: python t2c.py results/minilm_c6.json [--cache-dir cache]
Without a result argument, reads res_6.json as the older script did.
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("res_6.json")
    runpy.run_path(str(Path(__file__).with_name("detail.py")), run_name="__main__")
