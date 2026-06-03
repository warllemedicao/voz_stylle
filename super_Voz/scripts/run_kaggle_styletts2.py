#!/usr/bin/env python3
"""Compatibility entrypoint for the Kaggle StyleTTS2 runner."""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parents[1] / "kaglle" / "scripts" / "run_kaggle_styletts2.py"
runpy.run_path(str(TARGET), run_name="__main__")
