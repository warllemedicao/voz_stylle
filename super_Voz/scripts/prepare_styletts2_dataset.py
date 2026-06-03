#!/usr/bin/env python3
"""Compatibility entrypoint for the Kaggle dataset preparer."""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parents[1] / "kaglle" / "scripts" / "prepare_styletts2_dataset.py"
runpy.run_path(str(TARGET), run_name="__main__")
