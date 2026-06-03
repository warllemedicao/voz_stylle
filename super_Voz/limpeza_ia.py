#!/usr/bin/env python3
"""Compatibility entrypoint for the Kaggle audio cleaner.

Older Kaggle runners execute `python limpeza_ia.py` with cwd set to
`/kaggle/working/Super_voz/super_Voz`. The real Kaggle cleaner lives under
`super_Voz/kaglle`, so keep this shim to make old notebooks fail-safe.
"""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parent / "kaglle" / "limpeza_ia.py"
runpy.run_path(str(TARGET), run_name="__main__")
