#!/usr/bin/env python3
"""Compatibility entrypoint for Kaggle runs started from the repo root."""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parents[1] / "super_Voz" / "kaglle" / "scripts" / "prepare_styletts2_dataset.py"
runpy.run_path(str(TARGET), run_name="__main__")
