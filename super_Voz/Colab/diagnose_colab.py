#!/usr/bin/env python3
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}


def count_files(path: Path, audio_only: bool = False) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        if audio_only and item.suffix.lower() not in AUDIO_EXTS:
            continue
        total += 1
    return total


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERRO: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostico do ambiente Colab do super_Voz.")
    parser.add_argument("--config", default="styletts2_colab_config.yml")
    parser.add_argument("--output", default="logs/diagnostico_colab.json")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    config_path = project_dir / args.config
    output_path = project_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "project_dir": str(project_dir),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "files": {
            "run_colab_styletts2.py": (project_dir / "scripts" / "run_colab_styletts2.py").exists(),
            "prepare_styletts2_dataset.py": (project_dir / "scripts" / "prepare_styletts2_dataset.py").exists(),
            "limpeza_ia.py": (project_dir / "limpeza_ia.py").exists(),
        },
        "commands": {
            "git_version": command_output(["git", "--version"]),
            "ffmpeg_version": command_output(["ffmpeg", "-version"]),
        },
        "r2": {},
    }

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        report["config"] = cfg

        r2_cfg = cfg.get("cloudflare_r2", {})
        report["r2"]["configured"] = bool(r2_cfg and "INSERIR" not in r2_cfg.get("access_key_id", ""))
        report["r2"]["bucket"] = r2_cfg.get("bucket_name")
        report["r2"]["endpoint"] = r2_cfg.get("endpoint_url")

        local_raw = project_dir / "Audios_brutos"
        local_processed = project_dir / "Audios_processados"
        
        report["local"] = {
            "audios_brutos": {
                "exists": local_raw.exists(),
                "count": count_files(local_raw, audio_only=True)
            },
            "audios_processados": {
                "exists": local_processed.exists(),
                "count": count_files(local_processed, audio_only=True),
                "train_txt": (local_processed / "train.txt").exists()
            }
        }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nDiagnostico salvo em: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
