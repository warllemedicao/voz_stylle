#!/usr/bin/env python3
"""Valida um pacote de voz StyleTTS2 antes de usar os notebooks oficiais."""

from pathlib import Path


REQUIRED_FILES = [
    "model/best_model.pth",
    "model/config.yml",
    "data_reference/referencia_voz.wav",
    "data_reference/train_list.txt",
    "data_reference/val_list.txt",
    "data_reference/metadata.csv",
]


def validate_package(package_dir: str | Path) -> dict[str, Path]:
    root = Path(package_dir).resolve()
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError("Arquivos ausentes no pacote: " + ", ".join(missing))

    return {
        "package_dir": root,
        "checkpoint": root / "model" / "best_model.pth",
        "config": root / "model" / "config.yml",
        "reference_audio": root / "data_reference" / "referencia_voz.wav",
    }


if __name__ == "__main__":
    paths = validate_package(Path(__file__).resolve().parents[1])
    for name, path in paths.items():
        print(f"{name}={path}")
    print("Use estes caminhos no notebook oficial Inference_LibriTTS.ipynb do StyleTTS2.")
