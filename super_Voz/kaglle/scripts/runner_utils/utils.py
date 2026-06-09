#!/usr/bin/env python3
import os
import sys
import json
import shutil
import subprocess
import threading
import time
import zipfile
import re
import csv
import importlib
import importlib.metadata
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}
HF_HUB_COMPAT_PACKAGE = "huggingface_hub>=0.23.2,<1.0"
ML_RUNTIME_MODULES = {
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "accelerate": "accelerate",
}

RESEMBLE_COMPAT_PACKAGES = [
    "numpy==1.26.2",
    "scipy==1.11.4",
    "pandas==2.1.3",
    "matplotlib==3.8.1",
    "tabulate==0.8.10",
    "resampy==0.4.2",
]

RESEMBLE_COMPAT_VERSIONS = {
    "numpy": "1.26.2",
    "scipy": "1.11.4",
    "pandas": "2.1.3",
    "matplotlib": "3.8.1",
    "tabulate": "0.8.10",
    "resampy": "0.4.2",
}

def run(cmd, cwd=None, check=True, display_cmd=None):
    shown = display_cmd if display_cmd is not None else cmd
    print("\n$ " + " ".join(map(str, shown)))
    if cwd:
        print("cwd:", cwd)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)

def clone_or_pull(url: str, dest: Path) -> None:
    if dest.exists():
        # Garantir que estamos em uma branch antes de dar pull e resetar para evitar conflitos
        run(["git", "-C", str(dest), "fetch", "--all"], check=False)
        run(["git", "-C", str(dest), "checkout", "main"], check=False)
        run(["git", "-C", str(dest), "reset", "--hard", "origin/main"], check=False)
    else:
        run(["git", "clone", url, str(dest)])

def find_path_case_insensitive(path_str: str) -> Path | None:
    """Resolve um caminho de forma insensível a maiúsculas/minúsculas."""
    path = Path(path_str)
    if path.exists():
        return path

    parts = path.parts
    if not parts:
        return None

    if path_str.startswith("/"):
        current = Path("/")
        parts = parts[1:]
    else:
        current = Path(".")

    for part in parts:
        next_path = current / part
        if next_path.exists():
            current = next_path
            continue

        found = False
        if current.exists() and current.is_dir():
            try:
                for item in current.iterdir():
                    if item.name.lower() == part.lower():
                        current = item
                        found = True
                        break
            except (PermissionError, OSError):
                return None

        if not found:
            return None

    return current if current.exists() else None

def first_existing(paths: list[str]) -> Path | None:
    for item in paths:
        path = find_path_case_insensitive(item)
        if path:
            return path
    return None

def discover_kaggle_audio_dirs(root: Path = Path("/kaggle/input")) -> list[Path]:
    if not root.exists():
        return []

    found: dict[Path, int] = {}
    for item in root.rglob("*"):
        if not item.is_file() or item.suffix.lower() not in AUDIO_EXTS:
            continue
        parent = item.parent
        score = 0
        lowered_parts = {part.lower() for part in parent.parts}
        if any("audio" in part or "audios" in part for part in lowered_parts):
            score += 10
        if any("bruto" in part or "brutos" in part or "raw" in part for part in lowered_parts):
            score += 10
        found[parent] = max(found.get(parent, 0), score)

    return [path for path, _score in sorted(found.items(), key=lambda pair: (-pair[1], str(pair[0])))]

def copy_tree_files(src_dir: Path, dst_dir: Path, allowed=None) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(src_dir.rglob("*")):
        if not src.is_file():
            continue
        if allowed and not allowed(src):
            continue
        dst = dst_dir / src.relative_to(src_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied

def count_files(path: Path, allowed=None) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and (allowed is None or allowed(item)):
            total += 1
    return total

def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

def report_working_disk(label: str, working_dir: Path = Path("/kaggle/working")) -> None:
    target = working_dir if working_dir.exists() else Path.cwd()
    usage = shutil.disk_usage(target)
    percent = usage.used / usage.total * 100 if usage.total else 0
    print(
        f"[DISCO] {label}: usado {format_bytes(usage.used)} de {format_bytes(usage.total)} "
        f"({percent:.1f}%), livre {format_bytes(usage.free)} em {target}"
    )

def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total

def disk_size(paths: Path | list[Path]) -> int:
    items = [paths] if isinstance(paths, Path) else paths
    seen: set[tuple[int, int]] = set()
    total = 0

    for path in items:
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
        for item in candidates:
            stat = item.stat()
            key = (stat.st_dev, stat.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += getattr(stat, "st_blocks", 0) * 512 or stat.st_size
    return total

def removable_file_bytes(path: Path) -> int:
    stat = path.stat()
    if getattr(stat, "st_nlink", 1) > 1:
        return 0
    return getattr(stat, "st_blocks", 0) * 512 or stat.st_size

def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_2nd_(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else -1

def list_valid_finetune_checkpoints(style_dir: Path) -> list[Path]:
    ckpt_dir = style_dir / "Models" / "super_Voz"
    checkpoints = [path for path in ckpt_dir.glob("epoch_2nd_*.pth") if zipfile.is_zipfile(path)]
    return sorted(checkpoints, key=lambda path: (checkpoint_epoch(path), path.name))

def checkpoint_state(path: Path | None) -> tuple[str, int, int] | None:
    if not path or not path.exists():
        return None
    stat = path.stat()
    return (path.name, stat.st_size, stat.st_mtime_ns)

def is_stable_checkpoint(path: Path, min_age_seconds: int = 20) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    if time.time() - path.stat().st_mtime < min_age_seconds:
        return False
    return zipfile.is_zipfile(path)

def get_kaggle_secret(secret_label: str, required: bool = False) -> str:
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(secret_label)
    except Exception as exc:
        if required:
            print(f"Secret obrigatorio {secret_label} nao encontrado ou indisponivel ({exc}).")
        return ""

    return str(value).strip() if value else ""

def get_env_or_kaggle_secret(labels: list[str], required: bool = False) -> str:
    for label in labels:
        value = os.environ.get(label, "").strip()
        if value:
            return value

    for label in labels:
        value = get_kaggle_secret(label, required=required)
        if value:
            os.environ[label] = value
            print(f"Secret {label} carregado para variavel de ambiente.")
            return value

    return ""

def cleanup_intermediate_audio(cfg: dict, local_raw: Path, local_processed: Path) -> None:
    if not cfg.get("cleanup_intermediate_audio", True):
        print("[DISCO] Limpeza de audios intermediarios desativada por configuracao.")
        return

    for path in [local_raw, local_processed]:
        if not path.exists():
            continue
        shutil.rmtree(path)
        print(f"[DISCO] Pasta intermediaria removida apos preparar o pacote: {path}")

def run_training_with_progress(cmd, cwd=None) -> None:
    print("\n$ " + " ".join(map(str, cmd)))
    if cwd:
        print("cwd:", cwd)
    print("[TREINO] Iniciando. O console mostrara uma barra por epoca/passo; detalhes ficam em Models/super_Voz/train.log.")

    log_path = Path(cwd) / "Models" / "super_Voz" / "train.log" if cwd else None
    log_start_pos = log_path.stat().st_size if log_path and log_path.exists() else 0

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    train_re = re.compile(r"Epoch \[(\d+)/(\d+)\], Step \[(\d+)/(\d+)\], Loss: ([0-9.]+)")
    val_re = re.compile(r"Validation loss: ([0-9.]+), Dur loss: ([0-9.]+), F0 loss: ([0-9.]+)")
    important_re = re.compile(r"(error|traceback|cuda|out of memory|oom|killed|exception|saving|loading|checkpoint)", re.I)
    current_line = ""
    last_progress = None
    print_lock = threading.Lock()
    stop_log_tail = threading.Event()

    def handle_progress_line(line: str) -> bool:
        nonlocal current_line, last_progress

        train_match = train_re.search(line)
        if train_match:
            progress_key = ("train",) + train_match.groups()
            if progress_key == last_progress:
                return True
            last_progress = progress_key

            epoch, epochs, step, steps, loss = train_match.groups()
            step_i = int(step)
            steps_i = max(1, int(steps))
            filled = int(30 * step_i / steps_i)
            bar = "#" * filled + "-" * (30 - filled)
            current_line = f"[TREINO] Epoca {epoch}/{epochs} |{bar}| passo {step}/{steps} loss {loss}"
            print("\r" + current_line, end="", flush=True)
            return True

        val_match = val_re.search(line)
        if val_match:
            progress_key = ("val",) + val_match.groups()
            if progress_key == last_progress:
                return True
            last_progress = progress_key

            if current_line:
                print()
                current_line = ""
            val_loss, dur_loss, f0_loss = val_match.groups()
            print(f"[VALIDACAO] loss {val_loss} | dur {dur_loss} | f0 {f0_loss}")
            return True

        return False

    log_position = [log_start_pos]

    def drain_train_log() -> None:
        if not log_path:
            return

        if not log_path.exists():
            return

        with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
            log_file.seek(log_position[0])
            for line in log_file:
                with print_lock:
                    handle_progress_line(line.strip())
            log_position[0] = log_file.tell()

    def tail_train_log() -> None:
        while not stop_log_tail.is_set():
            drain_train_log()
            time.sleep(1)

    log_thread = threading.Thread(target=tail_train_log, daemon=True)
    log_thread.start()

    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue

        with print_lock:
            if handle_progress_line(line):
                continue

            if important_re.search(line):
                if current_line:
                    print()
                    current_line = ""
                print(line)

    stop_log_tail.set()
    log_thread.join(timeout=2)
    drain_train_log()

    with print_lock:
        if current_line:
            print()

    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)

def run_with_keepalive(cmd, cwd=None, keepalive_interval_seconds: int = 120) -> None:
    print("\n$ " + " ".join(map(str, cmd)))
    if cwd:
        print("cwd:", cwd)

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    stop_event = threading.Event()

    def keepalive() -> None:
        while not stop_event.wait(keepalive_interval_seconds):
            print("[KAGGLE][KEEPALIVE] Treino em andamento; aguardando novo log/checkpoint.", flush=True)

    thread = threading.Thread(target=keepalive, daemon=True)
    thread.start()
    try:
        for line in process.stdout:
            print(line.rstrip(), flush=True)
    finally:
        stop_event.set()
        thread.join(timeout=5)

    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)

def replace_with_hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_suffix(dst.suffix + ".tmp")
    temp.unlink(missing_ok=True)
    try:
        os.link(src, temp)
    except OSError:
        shutil.copy2(src, temp)
    temp.replace(dst)

def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if src.is_dir():
        if not dst.exists():
            shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True

def hardlink_tree(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        target = dst / path.relative_to(src)
        replace_with_hardlink_or_copy(path, target)
    return True

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

def is_stable_file(path: Path, min_age_seconds: int = 30) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    if time.time() - path.stat().st_mtime < min_age_seconds:
        return False
    first = path.stat()
    time.sleep(1)
    if not path.exists():
        return False
    second = path.stat()
    return first.st_size == second.st_size and first.st_mtime_ns == second.st_mtime_ns

def prune_uploaded_checkpoints(checkpoint_dir: Path, uploaded_checkpoint: Path) -> tuple[int, int]:
    checkpoints = sorted(
        [path for path in checkpoint_dir.glob("epoch_2nd_*.pth") if path.is_file()],
        key=lambda path: (checkpoint_epoch(path), path.name),
    )
    latest_local = checkpoints[-1] if checkpoints else None
    removed = 0
    recovered = 0
    uploaded_epoch = checkpoint_epoch(uploaded_checkpoint)

    for path in checkpoints:
        if path.resolve() == uploaded_checkpoint.resolve():
            continue
        if latest_local and path.resolve() == latest_local.resolve():
            continue
        if checkpoint_epoch(path) >= uploaded_epoch:
            continue
        size = path.stat().st_size
        freed = removable_file_bytes(path)
        path.unlink()
        removed += 1
        recovered += freed
        print(
            "[DISCO] Checkpoint local obsoleto removido apos upload confirmado: "
            f"{path.name} ({format_bytes(size)}, liberado {format_bytes(freed)})"
        )

    if removed:
        print(f"[DISCO] Limpeza de checkpoints recuperou {format_bytes(recovered)}.")
    return removed, recovered

def remove_pretrained_base_after_finetune_upload(style_dir: Path) -> int:
    base_checkpoint = style_dir / "Models" / "LibriTTS" / "epochs_2nd_00020.pth"
    if not base_checkpoint.exists():
        return 0
    size = base_checkpoint.stat().st_size
    freed = removable_file_bytes(base_checkpoint)
    base_checkpoint.unlink()
    print(
        "[DISCO] Checkpoint base removido apos persistir a voz treinada: "
        f"{base_checkpoint} ({format_bytes(size)}, liberado {format_bytes(freed)})"
    )
    return freed

def cleanup_training_artifacts(style_dir: Path, package_dir: Path) -> int:
    roots = [
        style_dir / "Models" / "super_Voz",
        package_dir,
    ]
    removable_names = {"__pycache__", ".ipynb_checkpoints"}
    removable_suffixes = {".tmp", ".part", ".partial", ".incomplete"}
    recovered = 0

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*"), reverse=True):
            try:
                if path.is_dir() and path.name in removable_names:
                    size = disk_size(path)
                    shutil.rmtree(path)
                    recovered += size
                    print(f"[DISCO] Temporario removido: {path} ({format_bytes(size)})")
                elif path.is_file() and path.suffix.lower() in removable_suffixes:
                    size = path.stat().st_size
                    freed = removable_file_bytes(path)
                    path.unlink()
                    recovered += freed
                    print(f"[DISCO] Temporario removido: {path} ({format_bytes(size)}, liberado {format_bytes(freed)})")
            except OSError as exc:
                print(f"[DISCO][AVISO] Nao foi possivel remover temporario {path}: {exc}")

    if recovered:
        print(f"[DISCO] Limpeza de temporarios recuperou {format_bytes(recovered)}.")
    return recovered

def write_voice_package_metadata(
    package_dir: Path,
    cfg: dict,
    latest: Path | None,
    sample_candidates: list[Path],
    missing_aux: list[str],
) -> None:
    language = "pt-BR"
    phonemizer_language = str(cfg.get("phonemizer_language", "pt-br"))
    sample_rate = int(cfg.get("sample_rate", 24000))
    model_name = "super_Voz StyleTTS2 pt-BR"
    latest_name = latest.name if latest else None
    has_best = (package_dir / "model" / "best_model.pth").exists()

    manifest = {
        "schema_version": 1,
        "package_name": "super_Voz",
        "model_name": model_name,
        "architecture": "StyleTTS2",
        "primary_language": language,
        "supported_languages": [language],
        "sample_rate": sample_rate,
        "inference": {
            "checkpoint": "model/best_model.pth",
            "config": "model/config.yml",
            "reference_audio": "data_reference/referencia_voz.wav",
            "requirements": "inference/requirements.txt",
        },
        "training_resume": {
            "enabled": True,
            "latest_checkpoint": "model/latest_checkpoint.pth",
            "source_checkpoint": latest_name,
        },
        "tokenizer": {
            "phonemize": bool(cfg.get("phonemize", True)),
            "phonemizer_language": phonemizer_language,
            "config": "tokenizer/phonemizer_config.txt",
        },
        "auxiliary_models": {
            "ASR": "model/Utils/ASR",
            "JDC": "model/Utils/JDC",
            "PLBERT": "model/Utils/PLBERT",
        },
        "status": {
            "has_best_model": has_best,
            "has_latest_checkpoint": latest is not None,
            "missing_auxiliary": missing_aux,
            "has_generated_sample": bool(sample_candidates),
        },
    }
    write_json(package_dir / "manifest.json", manifest)

    write_json(
        package_dir / "config.json",
        {
            "architectures": ["StyleTTS2"],
            "model_type": "styletts2",
            "name_or_path": "super_Voz",
            "language": language,
            "languages": [language],
            "sample_rate": sample_rate,
            "checkpoint_format": "pytorch_pth",
            "checkpoint": "model/best_model.pth",
            "styletts2_config": "model/config.yml",
            "reference_audio": "data_reference/referencia_voz.wav",
            "phonemizer_language": phonemizer_language,
            "transformers_compatible": False,
            "notes": "Metadados para empacotamento; este pacote nao carrega diretamente via transformers.pipeline.",
        },
    )

    write_json(
        package_dir / "tokenizer_config.json",
        {
            "tokenizer_class": "StyleTTS2Phonemizer",
            "language": language,
            "phonemizer_language": phonemizer_language,
            "phonemize": bool(cfg.get("phonemize", True)),
            "backend": "phonemizer",
            "plbert_path": "model/Utils/PLBERT",
            "transformers_tokenizer": False,
        },
    )

    write_json(
        package_dir / "api_config.json",
        {
            "api_version": 1,
            "default_language": language,
            "input_field": "text",
            "output_format": "wav",
            "sample_rate": sample_rate,
            "checkpoint": "model/best_model.pth",
            "config": "model/config.yml",
            "reference_audio": "data_reference/referencia_voz.wav",
        },
    )

    readme = f"""# super_Voz StyleTTS2 pt-BR

Pacote de voz treinado com StyleTTS2 para portugues do Brasil.

## Inferencia

Use `model/best_model.pth` com `model/config.yml` e mantenha os auxiliares em `model/Utils/`.
O audio de referencia fica em `data_reference/referencia_voz.wav`.

Este pacote nao e um modelo Transformers nativo. Os arquivos `config.json` e
`tokenizer_config.json` existem para documentar o pacote no Hugging Face e facilitar validacao.

## Retomada de treino

`model/latest_checkpoint.pth` e `model/latest_checkpoint.txt` sao usados para retomada. Para
comecar um treino do zero, remova checkpoints antigos antes de executar o Kaggle.

## Idioma

- Idioma principal: `pt-BR`
- Phonemizer: `{phonemizer_language}`
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")

    usage = """# Uso de inferencia

Arquivos essenciais:

- `model/best_model.pth`
- `model/config.yml`
- `model/Utils/ASR`
- `model/Utils/JDC`
- `model/Utils/PLBERT`
- `data_reference/referencia_voz.wav`
- `tokenizer/phonemizer_config.txt`

Use os notebooks oficiais do StyleTTS2 incluidos em `inference/` quando estiverem presentes.
O pacote foi marcado como portugues do Brasil (`pt-BR`).
"""
    (package_dir / "docs" / "uso_inferencia.md").write_text(usage, encoding="utf-8")

def read_validation_losses(style_dir: Path) -> dict[int, float]:
    log_path = style_dir / "Models" / "super_Voz" / "train.log"
    if not log_path.exists():
        return {}

    epoch_re = re.compile(r"Epoch \[(\d+)/\d+\]")
    val_re = re.compile(r"Validation loss: ([0-9.]+)")
    current_epoch = None
    losses: dict[int, float] = {}

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        epoch_match = epoch_re.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        val_match = val_re.search(line)
        if val_match and current_epoch is not None:
            losses[current_epoch] = float(val_match.group(1))

    return losses

def read_best_metric(package_dir: Path) -> float | None:
    metric_path = package_dir / "model" / "best_metric.txt"
    if not metric_path.exists():
        return None
    for line in metric_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("validation_loss="):
            try:
                return float(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None

def maybe_update_best_model(package_dir: Path, style_dir: Path, latest: Path) -> bool:
    model_dir = package_dir / "model"
    best_path = model_dir / "best_model.pth"
    best_metric_path = model_dir / "best_metric.txt"
    latest_epoch = checkpoint_epoch(latest)
    losses = read_validation_losses(style_dir)
    latest_loss = losses.get(latest_epoch)
    current_best = read_best_metric(package_dir)

    candidate = latest
    candidate_epoch = latest_epoch
    candidate_loss = latest_loss
    if not best_path.exists() and losses:
        checkpoints_by_epoch = {checkpoint_epoch(path): path for path in list_valid_finetune_checkpoints(style_dir)}
        available = [
            (loss, epoch, checkpoints_by_epoch[epoch])
            for epoch, loss in losses.items()
            if epoch in checkpoints_by_epoch
        ]
        if available:
            candidate_loss, candidate_epoch, candidate = min(available, key=lambda item: item[0])

    should_update = not best_path.exists()
    if candidate_loss is not None and (current_best is None or candidate_loss < current_best):
        should_update = True

    if not should_update:
        return False

    replace_with_hardlink_or_copy(candidate, best_path)
    metric_lines = [
        f"source_checkpoint={candidate.name}",
        f"epoch={candidate_epoch}",
    ]
    if candidate_loss is not None:
        metric_lines.append(f"validation_loss={candidate_loss}")
    else:
        metric_lines.append("validation_loss=unknown")
    best_metric_path.write_text("\n".join(metric_lines) + "\n", encoding="utf-8")
    print(f"[CHECKPOINT] best_model.pth atualizado a partir de {candidate.name}.")
    return True

def materialize_voice_package(
    package_dir: Path,
    style_dir: Path,
    dataset_dir: Path,
    processed_dir: Path,
    project_dir: Path,
    config_path: Path,
    cfg: dict,
) -> Path | None:
    latest = latest_finetune_checkpoint(style_dir)
    model_dir = package_dir / "model"
    data_dir = package_dir / "data_reference"
    docs_dir = package_dir / "docs"
    outputs_dir = package_dir / "outputs"
    for path in [model_dir, data_dir, package_dir / "tokenizer", package_dir / "inference", docs_dir, outputs_dir]:
        path.mkdir(parents=True, exist_ok=True)
    (model_dir / "vocoder").mkdir(parents=True, exist_ok=True)

    if latest:
        replace_with_hardlink_or_copy(latest, model_dir / "latest_checkpoint.pth")
        (model_dir / "latest_checkpoint.txt").write_text(latest.name + "\n", encoding="utf-8")
        maybe_update_best_model(package_dir, style_dir, latest)
    copy_if_exists(config_path, model_dir / "config.yml")
    (model_dir / "vocoder" / "README.txt").write_text(
        "O decoder/vocoder do StyleTTS2 esta incorporado em model/best_model.pth.\n",
        encoding="utf-8",
    )

    auxiliary = {
        style_dir / "Utils" / "ASR": model_dir / "Utils" / "ASR",
        style_dir / "Utils" / "JDC": model_dir / "Utils" / "JDC",
        style_dir / "Utils" / "PLBERT": model_dir / "Utils" / "PLBERT",
    }
    missing_aux = [
        str(src.relative_to(style_dir))
        for src, dst in auxiliary.items()
        if not hardlink_tree(src, dst)
    ]

    for name in ["train_list.txt", "val_list.txt", "all_list.txt"]:
        copy_if_exists(dataset_dir / "Data" / name, data_dir / name)
    package_wavs = data_dir / "wavs"
    if package_wavs.exists():
        shutil.rmtree(package_wavs)
    hardlink_tree(dataset_dir / "wavs", package_wavs)
    metadata = processed_dir / "train.txt"
    if not metadata.exists():
        metadata = processed_dir / "metadata.csv"
    copy_if_exists(metadata, data_dir / "metadata.csv")

    wavs = sorted((dataset_dir / "wavs").glob("*.wav"))
    if wavs:
        replace_with_hardlink_or_copy(wavs[0], data_dir / "referencia_voz.wav")

    (package_dir / "tokenizer" / "phonemizer_config.txt").write_text(
        f"phonemize={cfg.get('phonemize', True)}\n"
        f"language={cfg.get('phonemizer_language', 'pt-br')}\n"
        "PLBERT=model/Utils/PLBERT\n",
        encoding="utf-8",
    )
    copy_if_exists(style_dir / "requirements.txt", package_dir / "inference" / "requirements.txt")
    copy_if_exists(style_dir / "Models" / "super_Voz" / "train.log", docs_dir / "train.log")
    for name in ["Inference_LibriTTS.ipynb", "Inference_LJSpeech.ipynb"]:
        copy_if_exists(style_dir / "Demo" / name, package_dir / "inference" / name)
    for name in ["inference.py", "exemplo_uso.py"]:
        copy_if_exists(project_dir / "inference" / name, package_dir / "inference" / name)

    sample_candidates = sorted(Path(cfg.get("output_dir", "/kaggle/working/super_Voz_outputs")).rglob("*.wav"))
    if sample_candidates:
        copy_if_exists(sample_candidates[0], outputs_dir / "amostras_geradas.wav")
    write_voice_package_metadata(package_dir, cfg, latest, sample_candidates, missing_aux)

    report = dataset_dir / "prepare_report.txt"
    params = [
        f"{key}={value}"
        for key, value in sorted(cfg.items())
        if isinstance(value, (str, int, float, bool))
    ]
    if report.exists():
        params.extend(["", report.read_text(encoding="utf-8", errors="replace")])
    (docs_dir / "parametros_treinamento.txt").write_text("\n".join(params) + "\n", encoding="utf-8")
    (docs_dir / "dataset_usado.txt").write_text(
        f"processed_dir={processed_dir}\ndataset_dir={dataset_dir}\n"
        f"reference={data_dir / 'referencia_voz.wav'}\n",
        encoding="utf-8",
    )
    observations = [
        "Pacote gerado automaticamente pelo pipeline super_Voz.",
        "O StyleTTS2 nao usa um vocoder externo separado; o decoder/vocoder treinado esta no checkpoint.",
        "Para inferencia, mantenha os pesos auxiliares ASR, JDC e PLBERT junto do config.yml.",
        "data_reference/wavs foi incluido porque train_list.txt e val_list.txt dependem desses audios para retomar o treino.",
        "O projeto oficial StyleTTS2 fornece notebooks de inferencia, nao um inference.py oficial.",
        f"Codigo StyleTTS2 necessario para inferencia: {cfg.get('styletts2_repo', 'https://github.com/yl4579/StyleTTS2.git')}",
    ]
    if missing_aux:
        observations.append("Pesos auxiliares ausentes: " + ", ".join(missing_aux))
    if not (package_dir / "inference" / "inference.py").exists():
        observations.append("inference.py nao existe no projeto; adicione um script de inferencia compativel antes de distribuir o pacote.")
    if not sample_candidates:
        observations.append("Nenhuma amostra gerada foi encontrada para outputs/amostras_geradas.wav.")
    (docs_dir / "observacoes.txt").write_text("\n".join(observations) + "\n", encoding="utf-8")

    return latest

def sync_outputs(
    style_dir: Path,
    dataset_dir: Path,
    cfg: dict,
    s3=None,
    bucket=None,
    checkpoint_state=None,
    training_succeeded: bool = True,
) -> None:
    print("\n" + "="*60)
    if training_succeeded:
        print(" ✅ TREINO FINALIZADO!")
    else:
        print(" ⚠️ TREINO INTERROMPIDO OU COM FALHA!")
    print("="*60)
    print(f"Pacote da voz em: {style_dir / str(cfg.get('voice_package_dir', 'minha_voz_styletts2'))}")
    print(f"Dataset preparado em: {dataset_dir}")
    try:
        materialize_visible_outputs(style_dir, dataset_dir, cfg)
    except OSError as exc:
        print(f"[OUTPUTS][AVISO] Nao foi possivel materializar copias para download: {exc}")
        print("Os arquivos originais continuam em /kaggle/working/StyleTTS2/Models/super_Voz.")
    r2_cfg = cfg.get("cloudflare_r2", {})
    if r2_cfg.get("disable_r2_uploads"):
        print("Nota: upload/sync R2 desativado por disable_r2_uploads=true; downloads R2 continuam permitidos.")
        print("Em Kaggle Commit, /kaggle/working entra nos outputs do notebook.")
        if not training_succeeded:
            print("Nota: o pacote foi sincronizado para recuperacao, mas o treino nao concluiu com sucesso.")
        print("="*60 + "\n")
        return
    output_prefix = r2_cfg.get("output_prefix")
    if s3 and bucket and output_prefix:
        sync_persistent_outputs(s3, bucket, output_prefix, style_dir, dataset_dir, checkpoint_state)
        print(f"Arquivos persistidos no R2: s3://{bucket}/{output_prefix}")
    else:
        print("Nota: R2 sem configuracao completa; arquivos ficaram apenas em /kaggle/working.")
        print("Em Kaggle Commit, /kaggle/working entra nos outputs do notebook.")
    print("="*60 + "\n")

def materialize_visible_outputs(style_dir: Path, dataset_dir: Path, cfg: dict) -> None:
    tb_cfg = cfg.get("terabox", {}) or {}
    if not tb_cfg.get("keep_visible_outputs", True):
        return

    output_root = Path(tb_cfg.get("visible_output_dir") or cfg.get("output_dir") or "/kaggle/working/super_voz_resultados")
    output_root.mkdir(parents=True, exist_ok=True)

    items = {
        style_dir / str(cfg.get("voice_package_dir", "minha_voz_styletts2")): output_root / "minha_voz_styletts2",
        dataset_dir: output_root / "dataset_styletts2",
        Path(cfg.get("output_dir", "/kaggle/working/super_Voz_outputs")): output_root / "outputs",
    }
    for src, dst in items.items():
        if not src.exists():
            continue
        shutil.copytree(src, dst, dirs_exist_ok=True)

    print(f"Pastas visiveis para download no Kaggle: {output_root}")

def config_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []

def materialize_f5_voice_package(
    package_dir: Path,
    f5_cfg: dict,
    f5_library_dir: Path,
    f5_dataset_dir: Path,
    processed_dir: Path,
    base_checkpoint: Path,
    trained_checkpoint: Path | None,
) -> None:
    model_dir = package_dir / "model"
    data_dir = package_dir / "data_reference"
    docs_dir = package_dir / "docs"
    for path in [model_dir, data_dir, docs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    if trained_checkpoint:
        replace_with_hardlink_or_copy(trained_checkpoint, model_dir / trained_checkpoint.name)
        replace_with_hardlink_or_copy(trained_checkpoint, model_dir / f"latest_checkpoint{trained_checkpoint.suffix}")
    replace_with_hardlink_or_copy(base_checkpoint, model_dir / f"base_checkpoint{base_checkpoint.suffix}")
    copy_if_exists(f5_dataset_dir / "vocab.txt", model_dir / "vocab.txt")
    copy_if_exists(f5_dataset_dir / "duration.json", docs_dir / "duration.json")
    copy_if_exists(f5_dataset_dir / "metadata.csv", data_dir / "metadata.csv")

    wavs = sorted(processed_dir.glob("*.wav"))
    if wavs:
        replace_with_hardlink_or_copy(wavs[0], data_dir / "referencia_voz.wav")

    manifest = {
        "schema_version": 1,
        "package_name": package_dir.name,
        "architecture": "F5-TTS",
        "primary_language": "pt-BR",
        "inference_runtime_required": True,
        "base_library": {
            "repo_id": f5_cfg.get("repo_id", "Tharyck/multispeaker-ptbr-f5tts"),
            "huggingface_remote_dir": f5_cfg.get("huggingface_remote_dir", "libraries/f5_tts_ptbr_tharyck"),
            "local_dir": str(f5_library_dir),
            "checkpoint": str(base_checkpoint.relative_to(f5_library_dir)) if f5_library_dir in base_checkpoint.parents else str(base_checkpoint),
        },
        "voice_checkpoint": f"model/{trained_checkpoint.name}" if trained_checkpoint else None,
        "latest_checkpoint": f"model/latest_checkpoint{trained_checkpoint.suffix}" if trained_checkpoint else None,
        "tokenizer": f5_cfg.get("tokenizer", "char"),
        "exp_name": f5_cfg.get("exp_name", "F5TTS_Base"),
        "notes": "Este pacote contem artefatos da voz neural. A inferencia deve ser feita por outro programa com F5-TTS e a biblioteca/base PT-BR.",
    }
    write_json(package_dir / "manifest.json", manifest)
    (package_dir / "README.md").write_text(
        "# super_Voz F5-TTS PT-BR\n\n"
        "Pacote de voz neural treinado/adaptado com F5-TTS PT-BR.\n\n"
        "Este pacote nao executa inferencia sozinho. O programa de inferencia deve carregar o runtime F5-TTS, "
        f"a biblioteca/base `{f5_cfg.get('huggingface_remote_dir', 'libraries/f5_tts_ptbr_tharyck')}` "
        "e o checkpoint desta voz em `model/`.\n",
        encoding="utf-8",
    )

def download_pretrained(style_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    filename = "Models/LibriTTS/epochs_2nd_00020.pth"
    path = hf_hub_download(
        repo_id="yl4579/StyleTTS2-LibriTTS",
        filename=filename,
        local_dir=str(style_dir),
    )
    return Path(path)

def latest_finetune_checkpoint(style_dir: Path) -> Path | None:
    checkpoints = list_valid_finetune_checkpoints(style_dir)
    if checkpoints:
        return checkpoints[-1]
    packaged_latest = style_dir / "minha_voz_styletts2" / "model" / "latest_checkpoint.pth"
    if packaged_latest.exists() and zipfile.is_zipfile(packaged_latest):
        return packaged_latest
    packaged = style_dir / "minha_voz_styletts2" / "model" / "best_model.pth"
    return packaged if packaged.exists() else None

def patch_styletts2_config(style_dir: Path, dataset_dir: Path, cfg: dict) -> Path:
    import yaml

    config_path = style_dir / "Configs" / "config_ft.yml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["log_dir"] = "Models/super_Voz"
    config["epochs"] = int(cfg.get("epochs", 50))
    config["batch_size"] = int(cfg.get("batch_size", 2))
    config["max_len"] = int(cfg.get("max_len", 160))
    config["save_freq"] = int(cfg.get("save_freq", 5))
    config["log_interval"] = int(cfg.get("log_interval", 10))
    config["device"] = "cuda"

    latest_checkpoint = latest_finetune_checkpoint(style_dir)
    if latest_checkpoint:
        config["pretrained_model"] = latest_checkpoint.relative_to(style_dir).as_posix()
        config["load_only_params"] = False
        print(f"[INFO] Retomando treino do checkpoint: {config['pretrained_model']}")
    else:
        config["pretrained_model"] = "Models/LibriTTS/epochs_2nd_00020.pth"
        config["load_only_params"] = True
        print(f"[INFO] Nenhum checkpoint anterior encontrado; usando base: {config['pretrained_model']}")

    data_params = config.setdefault("data_params", {})
    data_params["train_data"] = "Data/train_list.txt"
    data_params["val_data"] = "Data/val_list.txt"
    data_params["root_path"] = str(dataset_dir / "wavs")
    data_params["OOD_data"] = "Data/OOD_texts.txt"

    loss_params = config.setdefault("loss_params", {})
    loss_params["diff_epoch"] = int(cfg.get("diff_epoch", 10))
    loss_params["joint_epoch"] = int(cfg.get("joint_epoch", 999))

    slmadv_params = config.setdefault("slmadv_params", {})
    slmadv_params["batch_percentage"] = float(cfg.get("batch_percentage", 0.125))
    slmadv_params["min_len"] = int(cfg.get("slm_min_len", 120))
    slmadv_params["max_len"] = int(cfg.get("slm_max_len", 220))

    out_path = style_dir / "Configs" / "config_super_voz.yml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return out_path

def patch_pytorch_compatibility(style_dir: Path) -> None:
    models_py = style_dir / "models.py"
    if not models_py.exists():
        return

    print("[INFO] Aplicando patch de compatibilidade PyTorch 2.6+ em models.py...")
    content = models_py.read_text(encoding="utf-8")
    original = content

    replacements = {
        "torch.load(model_path, map_location='cpu')":
            "torch.load(model_path, map_location='cpu', weights_only=False)",
        'torch.load(model_path, map_location="cpu")':
            'torch.load(model_path, map_location="cpu", weights_only=False)',
        "torch.load(ASR_MODEL_PATH, map_location='cpu')":
            "torch.load(ASR_MODEL_PATH, map_location='cpu', weights_only=False)",
        'torch.load(ASR_MODEL_PATH, map_location="cpu")':
            'torch.load(ASR_MODEL_PATH, map_location="cpu", weights_only=False)',
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    if content != original:
        models_py.write_text(content, encoding="utf-8")
        print("✅ Patch PyTorch 2.6+ aplicado.")

def patch_styletts2_oom_safety(style_dir: Path) -> None:
    train_py = style_dir / "train_finetune_accelerate.py"
    if not train_py.exists():
        return

    print("[INFO] Aplicando patch anti-OOM em train_finetune_accelerate.py...")
    content = train_py.read_text(encoding="utf-8")
    original = content

    replacements = {
        "mel_len_st = int(mel_input_length.min().item() / 2 - 1)":
            "mel_len_st = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)",
        "# get clips\n                    mel_len = int(mel_input_length.min().item() / 2 - 1)":
            "# get clips\n                    mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)",
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    if content != original:
        train_py.write_text(content, encoding="utf-8")
        print("✅ Patch anti-OOM aplicado.")

def patch_styletts2_zero_division_safety(style_dir: Path) -> None:
    """Aplica patch para evitar ZeroDivisionError se o validation dataloader for vazio."""
    train_py = style_dir / "train_finetune_accelerate.py"
    if not train_py.exists():
        return

    print("[INFO] Aplicando patch contra ZeroDivisionError em train_finetune_accelerate.py...")
    content = train_py.read_text(encoding="utf-8")
    original = content

    # Garante que iters_test seja pelo menos 1 antes da divisão
    old_log = "logger.info('Validation loss:"
    new_log = "iters_test = max(1, iters_test)\n        logger.info('Validation loss:"
    
    if old_log in content and new_log not in content:
        content = content.replace(old_log, new_log)
        # Também corrigir divisões subsequentes no tensorboard
        content = content.replace("loss_test / iters_test", "loss_test / max(1, iters_test)")
        content = content.replace("loss_align / iters_test", "loss_align / max(1, iters_test)")
        content = content.replace("loss_f / iters_test", "loss_f / max(1, iters_test)")

    if content != original:
        train_py.write_text(content, encoding="utf-8")
        print("✅ Patch contra ZeroDivisionError aplicado.")
    else:
        print("ℹ️ Patch contra ZeroDivisionError já aplicado ou não necessário.")

