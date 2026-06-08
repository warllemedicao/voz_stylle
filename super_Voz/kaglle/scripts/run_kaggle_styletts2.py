#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}


def run(cmd, cwd=None, check=True, display_cmd=None):
    shown = display_cmd if display_cmd is not None else cmd
    print("\n$ " + " ".join(map(str, shown)))
    if cwd:
        print("cwd:", cwd)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


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


def load_r2_env_defaults(r2_cfg: dict) -> dict:
    aliases = {
        "endpoint_url": ["R2_ENDPOINT_URL", "CLOUDFLARE_R2_ENDPOINT_URL", "CLOUDFLARE_ENDPOINT_URL"],
        "access_key_id": ["R2_ACCESS_KEY_ID", "CLOUDFLARE_R2_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"],
        "secret_access_key": ["R2_SECRET_ACCESS_KEY", "CLOUDFLARE_R2_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"],
        "bucket_name": ["R2_BUCKET_NAME", "CLOUDFLARE_R2_BUCKET_NAME"],
        "raw_audio_prefix": ["R2_RAW_AUDIO_PREFIX", "CLOUDFLARE_R2_RAW_AUDIO_PREFIX", "SUPER_VOZ_R2_RAW_AUDIO_PREFIX"],
    }
    out = dict(r2_cfg or {})
    for key, env_names in aliases.items():
        if out.get(key):
            continue
        value = get_env_or_kaggle_secret(env_names)
        if value:
            out[key] = value
    return out


def verify_gpu() -> bool:
    """Verifica se a GPU está disponível e é compatível."""
    print("\n--- Verificando Hardware ---")
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU Detectada: {name} ({mem:.1f} GB)")
            return True
        else:
            print("❌ Nenhuma GPU detectada!")
            print("⚠️ StyleTTS2 requer GPU para treinar sem falhas (SIGSEGV).")
            return False
    except ImportError:
        print("❌ PyTorch não instalado.")
        return False


def torch_packages_for_runtime() -> list[str]:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if major < 7:
                print(
                    f"[INFO] GPU sm_{major}{minor} detectada; fixando Torch 2.5.1 para compatibilidade com P100/K80."
                )
                return ["torch==2.5.1", "torchaudio==2.5.1", "torchvision==0.20.1"]
    except Exception as exc:
        print(f"[AVISO] Nao foi possivel detectar capability CUDA antes do pip install: {exc}")

    return ["torch", "torchaudio", "torchvision"]


def transformer_packages_for_runtime() -> list[str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return ["transformers"]

        major, _minor = torch.cuda.get_device_capability(0)
        if major < 7:
            print(
                "[INFO] Fixando transformers==4.46.3: versões recentes exigem Torch >=2.6 "
                "para carregar checkpoints PyTorch do Hugging Face."
            )
            return ["transformers==4.46.3"]
    except Exception as exc:
        print(f"[AVISO] Nao foi possivel detectar versao/compatibilidade antes do pip install: {exc}")

    return ["transformers"]


def install_ml_runtime_dependencies() -> None:
    print("\n--- Instalando Runtime ML compatível ---")
    torch_packages = torch_packages_for_runtime()
    transformer_packages = transformer_packages_for_runtime()
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        *torch_packages,
        *transformer_packages,
        "accelerate",
    ])


def install_dependencies(style_dir: Path) -> None:
    print("\n--- Instalando Dependências ---")
    
    # No Kaggle, tentamos instalar boto3 se não houver
    run([sys.executable, "-m", "pip", "install", "-q", "boto3"])

    missing_sys = []
    for pkg in ["ffmpeg", "sox", "espeak-ng"]:
        if shutil.which(pkg) is None:
            missing_sys.append(pkg)

    if missing_sys:
        print(f"[INFO] Instalando pacotes de sistema: {missing_sys}")
        # No Kaggle, apt-get precisa de cuidado, mas geralmente funciona
        run(["apt-get", "update"], check=False)
        run(["apt-get", "install", "-y", "ffmpeg", "sox", "libsndfile1", "espeak-ng"], check=False)

    print("[INFO] Verificando/Instalando dependências Python...")
    # Desinstalar onnxruntime comum para evitar conflito com a versão GPU
    run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"], check=False)
    
    # Configuração para instalação rápida do DeepSpeed sem compilação de C++ ops
    os.environ["DS_BUILD_OPS"] = "0"

    torch_packages = torch_packages_for_runtime()
    transformer_packages = transformer_packages_for_runtime()
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        *torch_packages,
        *transformer_packages,
        "accelerate",
        "huggingface_hub",
        "pyyaml",
        "librosa",
        "soundfile",
        "phonemizer",
        "openai-whisper",
        "demucs",
        "onnxruntime-gpu",
        "omegaconf",
        "ptflops",
        "celluloid",
        "rich",
        "matplotlib",
        "deepspeed",
        "pandas",
        "scipy",
        "tqdm",
    ])

    if os.environ.get("SUPER_VOZ_ENABLE_RESEMBLE", "1") != "0":
        # Dependências do Resemble já foram instaladas acima; --no-deps preserva o stack torch/torchaudio do Kaggle.
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-deps", "resemble-enhance"])
    else:
        print("[INFO] Pulando resemble-enhance no Kaggle porque SUPER_VOZ_ENABLE_RESEMBLE=0.")

    requirements = style_dir / "requirements.txt"
    if requirements.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=False)


def install_audio_cleaning_dependencies() -> None:
    print("\n--- Instalando Dependências da Limpeza IA ---")

    run([sys.executable, "-m", "pip", "install", "-q", "boto3"])

    missing_sys = []
    for pkg in ["ffmpeg", "sox", "espeak-ng"]:
        if shutil.which(pkg) is None:
            missing_sys.append(pkg)

    if missing_sys:
        print(f"[INFO] Instalando pacotes de sistema: {missing_sys}")
        run(["apt-get", "update"], check=False)
        run(["apt-get", "install", "-y", "ffmpeg", "sox", "libsndfile1", "espeak-ng"], check=False)

    print("[INFO] Verificando/Instalando dependências Python da limpeza...")
    os.environ["DS_BUILD_OPS"] = "0"
    run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"], check=False)
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "librosa",
        "soundfile",
        "openai-whisper",
        "demucs",
        "onnxruntime-gpu",
        "deepspeed",
        "scipy",
        "tqdm",
    ])

    if os.environ.get("SUPER_VOZ_ENABLE_RESEMBLE", "1") != "0":
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-deps", "resemble-enhance"])
    else:
        print("[INFO] Pulando resemble-enhance no Kaggle porque SUPER_VOZ_ENABLE_RESEMBLE=0.")


def get_r2_client(cfg: dict):
    import boto3
    from botocore.config import Config

    r2_cfg = load_r2_env_defaults(cfg.get("cloudflare_r2", {}))
    cfg["cloudflare_r2"] = r2_cfg
    required = ["endpoint_url", "access_key_id", "secret_access_key", "bucket_name"]
    missing = [key for key in required if not r2_cfg.get(key) or "INSERIR" in str(r2_cfg.get(key))]
    if missing:
        if r2_cfg:
            print(f"[R2][AVISO] Configuracao R2 incompleta; faltando: {', '.join(missing)}")
            if any(key in missing for key in ("access_key_id", "secret_access_key")):
                print(
                    "[R2][DICA] Download do Cloudflare R2 precisa dos Kaggle Secrets "
                    "R2_ACCESS_KEY_ID e R2_SECRET_ACCESS_KEY. Upload R2 continua bloqueado."
                )
        return None, None

    s3 = boto3.client(
        "s3",
        endpoint_url=r2_cfg.get("endpoint_url"),
        aws_access_key_id=r2_cfg.get("access_key_id"),
        aws_secret_access_key=r2_cfg.get("secret_access_key"),
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    return s3, r2_cfg.get("bucket_name")


def download_from_r2(s3, bucket, prefix, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[R2] Sincronizando prefixo '{prefix}' para {local_dir}...")
    
    paginator = s3.get_paginator("list_objects_v2")
    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                
                # Resolve caminho local relativo ao prefixo
                rel_path = os.path.relpath(key, prefix)
                dst_path = local_dir / rel_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                print(f"  - Baixando: {key} -> {dst_path.name}")
                s3.download_file(bucket, key, str(dst_path))
                downloaded += 1
    return downloaded


def upload_to_r2(s3, bucket, prefix, local_dir: Path):
    print(f"[R2] Fazendo upload de {local_dir} para prefixo '{prefix}'...")
    uploaded = 0
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            src_path = Path(root) / file
            rel_path = os.path.relpath(src_path, local_dir)
            key = os.path.join(prefix, rel_path).replace("\\", "/")
            
            print(f"  + Enviando: {src_path.name} -> {key}")
            s3.upload_file(str(src_path), bucket, key)
            uploaded += 1
    return uploaded


def upload_changed_files_to_r2(
    s3,
    bucket,
    prefix: str,
    local_dir: Path,
    state: dict[str, tuple[int, int]] | None = None,
    allowed=None,
) -> int:
    if not local_dir.exists():
        return 0

    state = state if state is not None else {}
    prefix = prefix.strip("/")
    uploaded = 0

    for src_path in sorted(local_dir.rglob("*")):
        if not src_path.is_file():
            continue
        if allowed and not allowed(src_path):
            continue

        stat = src_path.stat()
        rel_path = src_path.relative_to(local_dir).as_posix()
        fingerprint = (stat.st_size, int(stat.st_mtime))
        if state.get(rel_path) == fingerprint:
            continue

        key = f"{prefix}/{rel_path}" if prefix else rel_path
        print(f"[R2] Salvando: {src_path} -> {key}")
        s3.upload_file(str(src_path), bucket, key)
        state[rel_path] = fingerprint
        uploaded += 1

    return uploaded


def get_terabox_ndus(tb_cfg: dict) -> str:
    env_name = tb_cfg.get("ndus_env", "TERABOX_NDUS")
    return os.environ.get(env_name, "") or tb_cfg.get("cookie_ndus", "")


def render_terabox_command(template: list[str], tb_cfg: dict, local_dir: Path, remote_dir: str, ndus: str) -> list[str]:
    values = {
        "cli": tb_cfg.get("cli_path", "terabox-cli"),
        "python": sys.executable,
        "script_dir": str(Path(__file__).resolve().parent),
        "kaggle_dir": str(Path(__file__).resolve().parents[1]),
        "local_dir": str(local_dir),
        "remote_dir": remote_dir,
        "ndus": ndus,
        "restore_share_url": tb_cfg.get("restore_share_url", ""),
        "restore_share_password": tb_cfg.get("restore_share_password", ""),
    }
    return [str(part).format(**values) for part in template]


def run_terabox_command(template: list[str], tb_cfg: dict, local_dir: Path, remote_dir: str, ndus: str, check=False) -> bool:
    if not template:
        return False
    cmd = render_terabox_command(template, tb_cfg, local_dir, remote_dir, ndus)
    restore_password = str(tb_cfg.get("restore_share_password", ""))
    display_cmd = [
        "***" if (arg == ndus and ndus) or (arg == restore_password and restore_password) else arg
        for arg in cmd
    ]
    try:
        run(cmd, check=check, display_cmd=display_cmd)
        return True
    except Exception as exc:
        print(f"[TeraBox][AVISO] Comando falhou: {exc}")
        return False


def setup_terabox(cfg: dict) -> dict | None:
    tb_cfg = cfg.get("terabox", {}) or {}
    if not tb_cfg.get("enabled"):
        return None

    ndus = get_terabox_ndus(tb_cfg)
    if not ndus:
        print("[TeraBox][AVISO] terabox.enabled=true, mas TERABOX_NDUS nao foi encontrado. Sincronizacao TeraBox desativada.")
        return None

    missing_env = [name for name in tb_cfg.get("required_env", []) or [] if not os.environ.get(str(name))]
    if missing_env:
        print(
            "[TeraBox][AVISO] Secrets obrigatorios ausentes: "
            + ", ".join(map(str, missing_env))
            + ". Sincronizacao TeraBox desativada."
        )
        return None

    for install_cmd in tb_cfg.get("install_commands", []) or []:
        if not install_cmd:
            continue
        run([str(part) for part in install_cmd], check=False)

    login_command = tb_cfg.get("login_command", ["{cli}", "login", "--ndus", "{ndus}"])
    if login_command:
        ok = run_terabox_command(login_command, tb_cfg, Path("."), "", ndus, check=False)
        if not ok:
            print("[TeraBox][AVISO] Login falhou. O treino continua sem sincronizacao TeraBox.")
            return None

    print("[TeraBox] Sessao configurada.")
    return tb_cfg


def restore_styletts2_from_candidates(cfg: dict, style_dir: Path) -> int:
    candidates = cfg.get("styletts2_restore_candidates", []) or []
    default_candidates = [
        "/kaggle/input/styllet2",
        "/kaggle/input/styletts2",
        "/kaggle/input/terabox/StyleTTS2",
        "/kaggle/input/terabox/styletts2",
        "/kaggle/input/super-voz/StyleTTS2",
        "/kaggle/input/super-voz/styletts2",
    ]
    copied = 0

    for candidate in [*candidates, *default_candidates]:
        src = find_path_case_insensitive(candidate)
        if not src or not src.exists():
            continue

        nested = first_existing([
            str(src / "StyleTTS2"),
            str(src / "styletts2"),
            str(src / "styllet2"),
        ])
        if nested and (nested / "Models").exists():
            src = nested

        checkpoint_src = first_existing([
            str(src / "Models" / "super_Voz"),
            str(src / "StyleTTS2" / "Models" / "super_Voz"),
            str(src / "styletts2" / "Models" / "super_Voz"),
        ])

        if checkpoint_src:
            dst = style_dir / "Models" / "super_Voz"
            copied += copy_tree_files(checkpoint_src, dst)
            print(f"[RESTORE] Checkpoints restaurados de {checkpoint_src} para {dst}.")
            continue

        if (src / "Models").exists() or (src / "Configs").exists():
            copied += copy_tree_files(src, style_dir)
            print(f"[RESTORE] Estado StyleTTS2 restaurado de {src} para {style_dir}.")

    return copied


def terabox_download_styletts2(tb_cfg: dict | None, style_dir: Path) -> None:
    if not tb_cfg:
        return

    ndus = get_terabox_ndus(tb_cfg)
    remote_dir = tb_cfg.get("remote_styletts2_dir", "/StyleTTS2")
    local_dir = Path(tb_cfg.get("local_styletts2_dir", str(style_dir)))
    command = tb_cfg.get("download_command", [])
    if not command:
        print("[TeraBox] Download remoto nao configurado; use Kaggle Input para restaurar checkpoints.")
        return

    print(f"[TeraBox] Tentando baixar estado remoto {remote_dir} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    run_terabox_command(command, tb_cfg, local_dir, remote_dir, ndus, check=False)
    normalize_restored_styletts2(style_dir)


def normalize_restored_styletts2(style_dir: Path) -> int:
    checkpoint_dir = style_dir / "Models" / "super_Voz"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for nested in [
        style_dir / "StyleTTS2",
        style_dir / "styletts2",
        style_dir / "styllet2",
    ]:
        nested_ckpt = nested / "Models" / "super_Voz"
        if nested_ckpt.exists():
            copied += copy_tree_files(nested_ckpt, checkpoint_dir)

    for checkpoint in sorted(style_dir.rglob("epoch_2nd_*.pth")):
        if checkpoint_dir in checkpoint.parents:
            continue
        dst = checkpoint_dir / checkpoint.name
        if dst.exists() and dst.stat().st_size == checkpoint.stat().st_size:
            continue
        shutil.copy2(checkpoint, dst)
        copied += 1

    if copied:
        print(f"[RESTORE] Checkpoints normalizados em {checkpoint_dir}: {copied} arquivo(s).")
    return copied


def terabox_upload_checkpoints(tb_cfg: dict | None, style_dir: Path) -> bool:
    if not tb_cfg:
        return False

    checkpoint_dir = style_dir / "Models" / "super_Voz"
    if not checkpoint_dir.exists():
        return False

    missing_env = [name for name in tb_cfg.get("upload_required_env", []) or [] if not os.environ.get(str(name))]
    if missing_env:
        print(
            "[TeraBox][AVISO] Upload pulado; secrets ausentes: "
            + ", ".join(map(str, missing_env))
        )
        return False

    ndus = get_terabox_ndus(tb_cfg)
    remote_dir = tb_cfg.get("remote_checkpoint_dir") or tb_cfg.get("remote_styletts2_dir", "/StyleTTS2").rstrip("/") + "/Models/super_Voz"
    command = tb_cfg.get("upload_command", ["{cli}", "upload", "{local_dir}", "{remote_dir}"])

    print(f"[TeraBox] Enviando checkpoints {checkpoint_dir} -> {remote_dir}")
    return run_terabox_command(command, tb_cfg, checkpoint_dir, remote_dir, ndus, check=False)


def start_periodic_terabox_checkpoint_sync(
    tb_cfg: dict | None,
    style_dir: Path,
    interval_seconds: int,
) -> tuple[threading.Event, threading.Thread] | tuple[None, None]:
    if not tb_cfg:
        return None, None

    stop_event = threading.Event()

    def worker() -> None:
        while not stop_event.wait(interval_seconds):
            try:
                terabox_upload_checkpoints(tb_cfg, style_dir)
            except Exception as exc:
                print(f"[TeraBox][AVISO] Falha na sincronizacao periodica: {exc}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print(f"[TeraBox] Sincronizacao periodica de checkpoints ativa a cada {interval_seconds}s.")
    return stop_event, thread


def setup_huggingface(cfg: dict) -> dict | None:
    hf_cfg = cfg.get("huggingface", {}) or {}
    if not hf_cfg.get("enabled"):
        return None

    token_env = str(hf_cfg.get("token_env", "HF_TOKEN"))
    token = get_env_or_kaggle_secret([token_env], required=bool(hf_cfg.get("required", False)))
    bucket_uri = str(hf_cfg.get("bucket_uri", "")).strip().rstrip("/")
    if not token or not bucket_uri.startswith("hf://buckets/"):
        message = f"Configure o secret {token_env} e huggingface.bucket_uri para ativar o upload."
        if hf_cfg.get("required", False):
            raise RuntimeError("[HuggingFace] " + message)
        print("[HuggingFace][AVISO] " + message)
        return None

    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "huggingface_hub"])
    if shutil.which("hf") is None:
        message = "O comando hf nao ficou disponivel; sincronizacao desativada."
        if hf_cfg.get("required", False):
            raise RuntimeError("[HuggingFace] " + message)
        print("[HuggingFace][AVISO] " + message)
        return None

    os.environ["HF_TOKEN"] = token
    repo_fallback_id = bucket_uri.removeprefix("hf://buckets/").strip("/")
    hf_cfg["_repo_fallback_id"] = repo_fallback_id

    create_result = run(
        ["hf", "buckets", "create", bucket_uri, "--exist-ok"],
        check=False,
    )
    if create_result.returncode != 0:
        print(
            "[HuggingFace][AVISO] CLI sem suporte funcional a buckets ou bucket inacessivel; "
            f"usando fallback de repositorio: {repo_fallback_id}"
        )
    else:
        print(f"[HuggingFace] Bucket criado/validado: {bucket_uri}")

    hf_cfg["_sync_lock"] = threading.Lock()
    return hf_cfg


def huggingface_bucket_uri(hf_cfg: dict) -> str:
    return str(hf_cfg["bucket_uri"]).strip().rstrip("/")


def huggingface_repo_fallback_id(hf_cfg: dict) -> str:
    return str(hf_cfg.get("_repo_fallback_id") or huggingface_bucket_uri(hf_cfg).removeprefix("hf://buckets/")).strip("/")


def huggingface_restore_package(hf_cfg: dict | None, package_dir: Path) -> bool:
    if not hf_cfg:
        return False

    package_dir.mkdir(parents=True, exist_ok=True)
    bucket_uri = huggingface_bucket_uri(hf_cfg)
    repo_id = huggingface_repo_fallback_id(hf_cfg)
    commands = [
        ["hf", "buckets", "sync", bucket_uri, str(package_dir)],
        ["hf", "sync", bucket_uri, str(package_dir)],
        ["hf", "download", repo_id, "--local-dir", str(package_dir)],
    ]
    for command in commands:
        result = run(command, check=False)
        if result.returncode == 0:
            print(f"[HuggingFace] Pacote restaurado em {package_dir}.")
            return True
    print("[HuggingFace][AVISO] Nao foi possivel restaurar o pacote do Hugging Face.")
    return False


def run_quiet(cmd, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def restore_huggingface_subdir(hf_cfg: dict | None, remote_dir: str, local_dir: Path) -> bool:
    if not hf_cfg or not remote_dir:
        return False

    repo_id = huggingface_repo_fallback_id(hf_cfg)
    remote_dir = remote_dir.strip("/")
    temp_dir = local_dir.parent / f".hf_restore_{local_dir.name}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "hf",
        "download",
        repo_id,
        "--include",
        f"{remote_dir}/**",
        "--local-dir",
        str(temp_dir),
        "--repo-type",
        "model",
    ]
    print(f"[HuggingFace] Tentando restaurar biblioteca: {repo_id}/{remote_dir} -> {local_dir}")
    result = run_quiet(command)
    restored_root = temp_dir / remote_dir
    if result.returncode != 0 or not restored_root.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("[HuggingFace][AVISO] Biblioteca remota ainda nao existe ou nao pode ser baixada.")
        return False

    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(restored_root), str(local_dir))
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"[HuggingFace] Biblioteca restaurada em: {local_dir}")
    return True


def upload_huggingface_subdir(hf_cfg: dict | None, local_dir: Path, remote_dir: str) -> bool:
    if not hf_cfg or not local_dir.exists() or not remote_dir:
        return False

    repo_id = huggingface_repo_fallback_id(hf_cfg)
    remote_dir = remote_dir.strip("/")
    command = [
        "hf",
        "upload",
        repo_id,
        str(local_dir),
        remote_dir,
        "--repo-type",
        "model",
    ]
    print(f"[HuggingFace] Enviando biblioteca: {local_dir} -> {repo_id}/{remote_dir}")
    result = run(command, check=False)
    if result.returncode != 0:
        print("[HuggingFace][AVISO] Upload da biblioteca F5-TTS PT-BR falhou; treino ainda pode usar cache local.")
        return False
    return True


def ensure_f5_tts_ptbr_library(cfg: dict, hf_cfg: dict | None) -> Path | None:
    f5_cfg = cfg.get("f5_tts_ptbr", {}) or {}
    if not f5_cfg.get("enabled", False):
        return None

    default_root = Path(cfg.get("model_library_root", "/kaggle/working/super_voz_model_library"))
    local_dir = Path(f5_cfg.get("local_dir", str(default_root / "f5_tts_ptbr")))
    remote_dir = str(f5_cfg.get("huggingface_remote_dir", "libraries/f5_tts_ptbr")).strip("/")
    repo_id = str(f5_cfg.get("repo_id", "firstpixel/F5-TTS-pt-br")).strip()

    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"[F5-TTS-PT-BR] Biblioteca local encontrada: {local_dir}")
        return local_dir

    if restore_huggingface_subdir(hf_cfg, remote_dir, local_dir):
        return local_dir

    if not repo_id:
        raise RuntimeError("f5_tts_ptbr.repo_id nao configurado e biblioteca nao foi restaurada do Hugging Face.")

    print(f"[F5-TTS-PT-BR] Baixando biblioteca/base de {repo_id} para {local_dir}")
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "huggingface_hub"])
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"[F5-TTS-PT-BR] Biblioteca/base disponivel em: {local_dir}")
    upload_huggingface_subdir(hf_cfg, local_dir, remote_dir)
    return local_dir


def install_f5_tts_dependencies() -> None:
    packages = [
        "f5-tts",
        "accelerate",
        "safetensors",
        "num2words",
    ]
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *packages])
    install_ml_runtime_dependencies()


def read_voice_metadata_rows(processed_dir: Path) -> list[tuple[str, str]]:
    metadata = processed_dir / "train.txt"
    if not metadata.exists():
        metadata = processed_dir / "metadata.csv"
    if not metadata.exists():
        raise FileNotFoundError(f"Metadata da voz nao encontrado em {processed_dir}")

    rows: list[tuple[str, str]] = []
    with metadata.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "|" if "|" in sample else ","
        reader = csv.reader(f, delimiter=delimiter)
        for raw in reader:
            if not raw or len(raw) < 2:
                continue
            file_id = raw[0].strip()
            text = re.sub(r"\s+", " ", raw[1].strip().lower())
            if file_id and text and text.upper() != "VAZIO":
                rows.append((file_id, text.replace("|", " ")))
    if not rows:
        raise RuntimeError(f"Nenhuma linha valida encontrada em {metadata}")
    return rows


def resolve_processed_audio(processed_dir: Path, file_id: str) -> Path | None:
    candidate = Path(file_id)
    names = [candidate.name]
    if candidate.suffix.lower() not in AUDIO_EXTS:
        names.extend(f"{candidate.name}{ext}" for ext in AUDIO_EXTS)
    for name in names:
        path = processed_dir / name
        if path.exists():
            return path.resolve()
    return None


def write_f5_metadata_csv(processed_dir: Path, out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = read_voice_metadata_rows(processed_dir)
    written = 0
    missing: list[str] = []
    with out_csv.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])
        for file_id, text in rows:
            audio_path = resolve_processed_audio(processed_dir, file_id)
            if not audio_path:
                missing.append(file_id)
                continue
            writer.writerow([audio_path.as_posix(), text])
            written += 1

    if missing:
        preview = ", ".join(missing[:10])
        print(f"[F5-TTS-PT-BR][AVISO] Audios sem arquivo correspondente ignorados: {len(missing)} ({preview})")
    if written == 0:
        raise RuntimeError("Nenhum audio valido foi escrito no metadata.csv do F5-TTS.")
    print(f"[F5-TTS-PT-BR] Metadata CSV criado: {out_csv} ({written} linhas)")
    return written


def f5_checkpoint_path(library_dir: Path, f5_cfg: dict) -> Path:
    subpath = str(f5_cfg.get("checkpoint_subpath", "pt-br/model_last.safetensors")).strip()
    checkpoint = library_dir / subpath
    if checkpoint.exists():
        return checkpoint
    candidates = sorted(
        [*library_dir.rglob("model_last.safetensors"), *library_dir.rglob("model_last.pt"), *library_dir.rglob("*.safetensors")],
        key=lambda path: (path.name != "model_last.safetensors", len(path.parts), path.as_posix()),
    )
    if candidates:
        print(f"[F5-TTS-PT-BR] Checkpoint base detectado: {candidates[0]}")
        return candidates[0]
    raise FileNotFoundError(f"Checkpoint F5-TTS PT-BR nao encontrado em {library_dir}")


def find_f5_checkpoint_dir(dataset_name: str) -> Path | None:
    code = (
        "from importlib.resources import files\n"
        "from pathlib import Path\n"
        f"p = Path(files('f5_tts').joinpath('../../ckpts/{dataset_name}')).resolve()\n"
        "print(p)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        print(f"[F5-TTS-PT-BR][AVISO] Nao foi possivel localizar ckpts do pacote f5_tts: {result.stdout}")
        return None
    path = Path(result.stdout.strip())
    return path if path.exists() else None


def f5_package_path(relative_path: str) -> Path:
    code = (
        "from importlib.resources import files\n"
        "from pathlib import Path\n"
        f"print(Path(files('f5_tts').joinpath('{relative_path}')).resolve())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Nao foi possivel localizar arquivo do pacote f5_tts: {result.stdout}")
    return Path(result.stdout.strip())


def f5_prepared_dataset_dir(dataset_name: str, tokenizer: str) -> Path:
    return f5_package_path(f"../../data/{dataset_name}_{tokenizer}")


def latest_f5_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir or not checkpoint_dir.exists():
        return None
    candidates = [
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".pt", ".pth", ".safetensors"}
        and not path.name.startswith("pretrained_")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def list_f5_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir or not checkpoint_dir.exists():
        return []
    checkpoints = [
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".pt", ".pth", ".safetensors"}
        and not path.name.startswith("pretrained_")
    ]
    return sorted(checkpoints, key=lambda path: (path.stat().st_mtime_ns, path.name))


def prune_f5_uploaded_checkpoints(checkpoint_dir: Path, keep_last: int = 2) -> tuple[int, int]:
    checkpoints = list_f5_checkpoints(checkpoint_dir)
    if keep_last <= 0 or len(checkpoints) <= keep_last:
        return 0, 0

    keep = {path.resolve() for path in checkpoints[-keep_last:]}
    removed = 0
    recovered = 0
    for path in checkpoints:
        if path.resolve() in keep:
            continue
        size = path.stat().st_size
        freed = removable_file_bytes(path)
        path.unlink()
        removed += 1
        recovered += freed
        print(
            "[DISCO][F5] Checkpoint local antigo removido apos upload confirmado: "
            f"{path.name} ({format_bytes(size)}, liberado {format_bytes(freed)})"
        )

    if removed:
        print(f"[DISCO][F5] Retencao local manteve os {keep_last} checkpoint(s) mais recentes e recuperou {format_bytes(recovered)}.")
    return removed, recovered


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


def f5_checkpoint_state(path: Path | None) -> tuple[str, int, int] | None:
    if not path or not path.exists():
        return None
    stat = path.stat()
    return (path.name, stat.st_size, stat.st_mtime_ns)


def sync_f5_voice_checkpoint(
    hf_cfg: dict | None,
    package_dir: Path,
    f5_cfg: dict,
    f5_library_dir: Path,
    f5_dataset_dir: Path,
    processed_dir: Path,
    base_checkpoint: Path,
    checkpoint: Path,
    reason: str,
) -> bool:
    materialize_f5_voice_package(
        package_dir,
        f5_cfg,
        f5_library_dir,
        f5_dataset_dir,
        processed_dir,
        base_checkpoint,
        checkpoint,
    )
    print(f"[F5-TTS-PT-BR] Sincronizando checkpoint ({reason}): {checkpoint.name}")
    return upload_huggingface_subdir(hf_cfg, package_dir, f"voices/{package_dir.name}")


def start_f5_checkpoint_sync(
    hf_cfg: dict | None,
    package_dir: Path,
    f5_cfg: dict,
    f5_library_dir: Path,
    f5_dataset_dir: Path,
    processed_dir: Path,
    base_checkpoint: Path,
    checkpoint_dir: Path,
    poll_interval_seconds: int,
    stable_seconds: int,
    keep_last_checkpoints: int,
) -> tuple[threading.Event, threading.Thread] | tuple[None, None]:
    if not hf_cfg:
        return None, None

    stop_event = threading.Event()

    def worker() -> None:
        last_uploaded = f5_cfg.get("_last_uploaded_checkpoint")
        while not stop_event.wait(poll_interval_seconds):
            try:
                latest = latest_f5_checkpoint(checkpoint_dir)
                current_state = f5_checkpoint_state(latest)
                if not latest or not current_state or current_state == last_uploaded:
                    continue
                if not is_stable_file(latest, min_age_seconds=stable_seconds):
                    continue
                if sync_f5_voice_checkpoint(
                    hf_cfg,
                    package_dir,
                    f5_cfg,
                    f5_library_dir,
                    f5_dataset_dir,
                    processed_dir,
                    base_checkpoint,
                    latest,
                    reason="checkpoint novo durante treino",
                ):
                    last_uploaded = current_state
                    f5_cfg["_last_uploaded_checkpoint"] = current_state
                    prune_f5_uploaded_checkpoints(checkpoint_dir, keep_last=keep_last_checkpoints)
            except Exception as exc:
                print(f"[F5-TTS-PT-BR][AVISO] Falha no monitor de checkpoint: {exc}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print(
        "[F5-TTS-PT-BR] Monitor de checkpoints ativo; "
        f"checagem a cada {poll_interval_seconds}s e upload somente para checkpoint novo estavel."
    )
    return stop_event, thread


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
            "repo_id": f5_cfg.get("repo_id", "firstpixel/F5-TTS-pt-br"),
            "huggingface_remote_dir": f5_cfg.get("huggingface_remote_dir", "libraries/f5_tts_ptbr"),
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
        "a biblioteca/base `libraries/f5_tts_ptbr` e o checkpoint desta voz em `model/`.\n",
        encoding="utf-8",
    )


def run_f5_tts_training(
    cfg: dict,
    hf_cfg: dict | None,
    f5_library_dir: Path,
    processed_dir: Path,
) -> Path:
    f5_cfg = cfg.get("f5_tts_ptbr", {}) or {}
    install_f5_tts_dependencies()

    dataset_name = str(f5_cfg.get("dataset_name", "super_voz_f5_ptbr")).strip()
    tokenizer = str(f5_cfg.get("tokenizer", "char"))
    f5_dataset_dir = f5_prepared_dataset_dir(dataset_name, tokenizer)
    metadata_work_dir = Path(f5_cfg.get("dataset_dir", "/kaggle/working/super_voz_f5_dataset"))
    package_dir = Path(cfg.get("styletts2_dir", "/kaggle/working/StyleTTS2")) / str(
        cfg.get("f5_voice_package_dir", "minha_voz_f5_tts_ptbr")
    )
    if f5_dataset_dir.exists():
        shutil.rmtree(f5_dataset_dir)
    f5_dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata_work_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = metadata_work_dir / "metadata.csv"
    write_f5_metadata_csv(processed_dir, metadata_csv)

    prepare_script = f5_package_path("train/datasets/prepare_csv_wavs.py")
    prepare_cmd = [
        sys.executable,
        str(prepare_script),
        str(metadata_csv),
        str(f5_dataset_dir),
        "--workers",
        str(int(f5_cfg.get("workers", 2))),
    ]
    if bool(f5_cfg.get("prepare_as_pretrain", True)):
        prepare_cmd.append("--pretrain")
    run(prepare_cmd)

    base_checkpoint = f5_checkpoint_path(f5_library_dir, f5_cfg)
    checkpoint_dir = f5_package_path(f"../../ckpts/{dataset_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    finetune_script = f5_package_path("train/finetune_cli.py")
    train_cmd = [
        "accelerate",
        "launch",
        f"--mixed_precision={f5_cfg.get('mixed_precision', 'fp16')}",
        str(finetune_script),
        "--exp_name",
        str(f5_cfg.get("exp_name", "F5TTS_Base")),
        "--dataset_name",
        dataset_name,
        "--learning_rate",
        str(f5_cfg.get("learning_rate", 1e-5)),
        "--batch_size_per_gpu",
        str(int(f5_cfg.get("batch_size_per_gpu", 1600))),
        "--batch_size_type",
        str(f5_cfg.get("batch_size_type", "frame")),
        "--max_samples",
        str(int(f5_cfg.get("max_samples", 32))),
        "--grad_accumulation_steps",
        str(int(f5_cfg.get("grad_accumulation_steps", 1))),
        "--epochs",
        str(int(f5_cfg.get("epochs", 20))),
        "--num_warmup_updates",
        str(int(f5_cfg.get("num_warmup_updates", 100))),
        "--save_per_updates",
        str(int(f5_cfg.get("save_per_updates", 500))),
        "--keep_last_n_checkpoints",
        str(int(f5_cfg.get("keep_last_n_checkpoints", 3))),
        "--last_per_updates",
        str(int(f5_cfg.get("last_per_updates", 100))),
        "--finetune",
        "--pretrain",
        str(base_checkpoint),
        "--tokenizer",
        tokenizer,
    ]

    sync_stop = None
    sync_thread = None
    training_error: Exception | None = None
    keep_last_checkpoints = max(1, int(f5_cfg.get("local_checkpoint_keep_last", 2)))
    try:
        sync_stop, sync_thread = start_f5_checkpoint_sync(
            hf_cfg,
            package_dir,
            f5_cfg,
            f5_library_dir,
            f5_dataset_dir,
            processed_dir,
            base_checkpoint,
            checkpoint_dir,
            max(30, int(f5_cfg.get("checkpoint_sync_interval_seconds", 300))),
            max(5, int(f5_cfg.get("checkpoint_stable_seconds", 30))),
            keep_last_checkpoints,
        )
        run_with_keepalive(
            train_cmd,
            keepalive_interval_seconds=max(30, int(f5_cfg.get("keepalive_interval_seconds", 120))),
        )
    except Exception as exc:
        training_error = exc
        print(f"[F5-TTS-PT-BR][AVISO] Treino interrompido/falhou; tentando sincronizar ultimo checkpoint antes de sair: {exc}")
    finally:
        if sync_stop and sync_thread:
            sync_stop.set()
            sync_thread.join(timeout=30)

    checkpoint_dir = find_f5_checkpoint_dir(dataset_name) or checkpoint_dir
    trained_checkpoint = latest_f5_checkpoint(checkpoint_dir) if checkpoint_dir else None
    if not trained_checkpoint:
        if training_error:
            raise training_error
        raise RuntimeError("Treino F5-TTS terminou, mas nenhum checkpoint da voz foi encontrado.")
    print(f"[F5-TTS-PT-BR] Ultimo checkpoint da voz: {trained_checkpoint}")

    if not is_stable_file(
        trained_checkpoint,
        min_age_seconds=max(5, int(f5_cfg.get("checkpoint_stable_seconds", 30))),
    ):
        print("[F5-TTS-PT-BR][AVISO] Ultimo checkpoint ainda nao parece estavel; tentando sincronizar mesmo assim no encerramento.")
    final_state = f5_checkpoint_state(trained_checkpoint)
    if final_state == f5_cfg.get("_last_uploaded_checkpoint"):
        print("[F5-TTS-PT-BR] Ultimo checkpoint ja sincronizado; upload final duplicado pulado.")
    elif sync_f5_voice_checkpoint(
        hf_cfg,
        package_dir,
        f5_cfg,
        f5_library_dir,
        f5_dataset_dir,
        processed_dir,
        base_checkpoint,
        trained_checkpoint,
        reason="sincronizacao final",
    ):
        f5_cfg["_last_uploaded_checkpoint"] = final_state
        prune_f5_uploaded_checkpoints(checkpoint_dir, keep_last=keep_last_checkpoints)
    print(f"[F5-TTS-PT-BR] Pacote da voz neural pronto em: {package_dir}")
    if training_error:
        raise training_error
    return package_dir


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


def huggingface_sync_package(
    hf_cfg: dict | None,
    package_dir: Path,
    style_dir: Path,
    dataset_dir: Path,
    processed_dir: Path,
    project_dir: Path,
    config_path: Path,
    cfg: dict,
    reason: str = "manual",
) -> bool:
    if not hf_cfg:
        materialize_voice_package(
            package_dir, style_dir, dataset_dir, processed_dir, project_dir, config_path, cfg
        )
        return False

    lock = hf_cfg["_sync_lock"]
    if not lock.acquire(blocking=False):
        return False
    try:
        latest = materialize_voice_package(
            package_dir, style_dir, dataset_dir, processed_dir, project_dir, config_path, cfg
        )
        bucket_uri = huggingface_bucket_uri(hf_cfg)
        repo_id = huggingface_repo_fallback_id(hf_cfg)
        commands = [
            ["hf", "buckets", "sync", str(package_dir), bucket_uri, "--delete"],
            ["hf", "sync", str(package_dir), bucket_uri, "--delete"],
            ["hf", "upload-large-folder", repo_id, str(package_dir), "--repo-type", "model"],
            ["hf", "upload", repo_id, str(package_dir), ".", "--repo-type", "model"],
        ]
        synced = False
        target = bucket_uri
        print(f"[HuggingFace] Iniciando upload do pacote ({reason}).")
        for command in commands:
            result = run(command, check=False)
            if result.returncode == 0:
                synced = True
                if command[1] in {"upload-large-folder", "upload"}:
                    target = f"repo:{repo_id}"
                break

        if not synced:
            print("[HuggingFace][AVISO] Upload do pacote falhou; checkpoints locais foram preservados.")
            report_working_disk("upload Hugging Face falhou")
            return False
        print(f"[HuggingFace] Pacote sincronizado: {package_dir} -> {target}")
        if latest and latest.parent.name == "super_Voz":
            before_cleanup = disk_size([style_dir / "Models" / "super_Voz", package_dir])
            prune_uploaded_checkpoints(style_dir / "Models" / "super_Voz", latest)
            remove_pretrained_base_after_finetune_upload(style_dir)
            cleanup_training_artifacts(style_dir, package_dir)
            after_cleanup = disk_size([style_dir / "Models" / "super_Voz", package_dir])
            recovered = max(0, before_cleanup - after_cleanup)
            print(f"[DISCO] Limpeza apos upload confirmou {format_bytes(recovered)} recuperados.")
            report_working_disk("apos limpeza de checkpoints")
        return True
    finally:
        lock.release()


def start_huggingface_checkpoint_sync(
    hf_cfg: dict | None,
    package_dir: Path,
    style_dir: Path,
    dataset_dir: Path,
    processed_dir: Path,
    project_dir: Path,
    config_path: Path,
    cfg: dict,
    poll_interval_seconds: int,
) -> tuple[threading.Event, threading.Thread] | tuple[None, None]:
    if not hf_cfg:
        return None, None

    stop_event = threading.Event()

    def worker() -> None:
        last_uploaded = hf_cfg.get("_last_uploaded_checkpoint")
        while not stop_event.wait(poll_interval_seconds):
            latest = latest_finetune_checkpoint(style_dir)
            if not latest or latest.parent.name != "super_Voz":
                continue
            current_state = checkpoint_state(latest)
            if current_state == last_uploaded:
                continue
            if not is_stable_checkpoint(latest):
                continue
            if huggingface_sync_package(
                hf_cfg,
                package_dir,
                style_dir,
                dataset_dir,
                processed_dir,
                project_dir,
                config_path,
                cfg,
                reason=f"checkpoint de epoca concluida: {latest.name}",
            ):
                last_uploaded = current_state
                hf_cfg["_last_uploaded_checkpoint"] = current_state

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print(
        "[HuggingFace] Monitor de checkpoints ativo; "
        f"checagem a cada {poll_interval_seconds}s, upload somente para checkpoint novo de epoca."
    )
    return stop_event, thread


def start_periodic_checkpoint_sync(
    s3,
    bucket,
    output_prefix: str,
    style_dir: Path,
    interval_seconds: int,
) -> tuple[threading.Event, threading.Thread, dict[str, tuple[int, int]]]:
    stop_event = threading.Event()
    state: dict[str, tuple[int, int]] = {}
    checkpoint_dir = style_dir / "Models" / "super_Voz"
    prefix = f"{output_prefix.strip('/')}/checkpoints"

    def worker() -> None:
        while not stop_event.wait(interval_seconds):
            try:
                uploaded = upload_changed_files_to_r2(
                    s3,
                    bucket,
                    prefix,
                    checkpoint_dir,
                    state,
                    allowed=lambda p: p.suffix.lower() in {".pth", ".log", ".txt", ".yml", ".yaml"},
                )
                if uploaded:
                    print(f"[R2] Sincronizacao periodica: {uploaded} arquivo(s) enviados.")
            except Exception as exc:
                print(f"[R2][AVISO] Falha na sincronizacao periodica: {exc}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print(f"[R2] Sincronizacao periodica de checkpoints ativa a cada {interval_seconds}s.")
    return stop_event, thread, state


def sync_persistent_outputs(
    s3,
    bucket,
    output_prefix: str,
    style_dir: Path,
    dataset_dir: Path,
    checkpoint_state: dict[str, tuple[int, int]] | None = None,
) -> None:
    output_prefix = output_prefix.strip("/")
    checkpoint_dir = style_dir / "Models" / "super_Voz"

    uploaded_ckpts = upload_changed_files_to_r2(
        s3,
        bucket,
        f"{output_prefix}/checkpoints",
        checkpoint_dir,
        checkpoint_state,
    )
    uploaded_dataset = upload_changed_files_to_r2(
        s3,
        bucket,
        f"{output_prefix}/dataset",
        dataset_dir,
    )
    print(f"[R2] Persistencia concluida: {uploaded_ckpts} checkpoint(s)/log(s), {uploaded_dataset} arquivo(s) de dataset.")


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline Kaggle super_Voz com StyleTTS2.")
    parser.add_argument("--config", default="styletts2_kaggle_config.yml")
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    import yaml

    code_dir = Path(__file__).resolve().parents[1]
    repo_dir = Path("/kaggle/working/Super_voz").resolve()
    if not repo_dir.exists():
        repo_dir = code_dir

    config_path = code_dir / args.config
    if not config_path.exists():
        config_path = Path(args.config).resolve()

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_dir = code_dir
    data_root = Path(cfg.get("repo_dir", str(repo_dir))).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    report_working_disk("inicio do pipeline")
    tts_engine = str(cfg.get("tts_engine", "styletts2")).strip().lower()

    # Configuração de memória
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cfg.get("enable_resemble_enhance", True):
        os.environ["SUPER_VOZ_ENABLE_RESEMBLE"] = "1"
    else:
        os.environ["SUPER_VOZ_ENABLE_RESEMBLE"] = "0"
    print(f"[INFO] SUPER_VOZ_ENABLE_RESEMBLE={os.environ['SUPER_VOZ_ENABLE_RESEMBLE']}")

    # Verifica GPU antes de tudo para evitar SIGSEGV
    if not verify_gpu():
        print("🛑 Abortando devido à falta de GPU. O treino falharia com SIGSEGV.")
        return 1

    huggingface_cfg = setup_huggingface(cfg)
    style_dir = Path(cfg.get("styletts2_dir", "/kaggle/working/StyleTTS2"))
    terabox_cfg = None
    f5_library_dir = None

    if tts_engine == "f5_tts_ptbr":
        install_ml_runtime_dependencies()
        style_dir.mkdir(parents=True, exist_ok=True)
        f5_library_dir = ensure_f5_tts_ptbr_library(cfg, huggingface_cfg)
        if f5_library_dir:
            print(f"[F5-TTS-PT-BR] Biblioteca separada pronta para treino/inferencia: {f5_library_dir}")
        install_audio_cleaning_dependencies()
    else:
        clone_or_pull(cfg.get("styletts2_repo", "https://github.com/yl4579/StyleTTS2.git"), style_dir)
        package_dir = style_dir / str(cfg.get("voice_package_dir", "minha_voz_styletts2"))

        restore_styletts2_from_candidates(cfg, style_dir)
        terabox_cfg = setup_terabox(cfg)
        restored_hf = huggingface_restore_package(huggingface_cfg, package_dir)
        if huggingface_cfg and not restored_hf:
            print("[HuggingFace][AVISO] Pacote remoto nao restaurado; o primeiro upload ocorrera no primeiro checkpoint.")

        patch_pytorch_compatibility(style_dir)
        patch_styletts2_oom_safety(style_dir)
        patch_styletts2_zero_division_safety(style_dir)

        install_dependencies(style_dir)
    
    # Preparar audios locais
    local_raw = data_root / "Audios_brutos"
    local_processed = data_root / "Audios_processados"
    local_raw.mkdir(parents=True, exist_ok=True)
    local_processed.mkdir(parents=True, exist_ok=True)

    s3, bucket = get_r2_client(cfg)

    # SEMPRE buscamos Audios Brutos agora para garantir que limpeza_ia.py rode com as novas otimizações
    if s3:
        r2_cfg = cfg.get("cloudflare_r2", {})
        raw_prefix = r2_cfg.get("raw_audio_prefix")
        if raw_prefix:
            downloaded = download_from_r2(s3, bucket, raw_prefix, local_raw)
            print(f"✅ Audios brutos importados do R2: {downloaded}")
        else:
            print("[R2][AVISO] raw_audio_prefix ausente; pulando download de audios R2.")
    else:
        print("[AVISO] Configuração R2 ausente ou incompleta.")

    if count_files(local_raw, allowed=lambda p: p.suffix.lower() in AUDIO_EXTS) == 0:
        print("[INFO] Verificando candidatos locais/Kaggle Input para audios brutos...")
        raw_candidates = [
            str(data_root / "Audios_brutos"),
            str(project_dir / "Audios_brutos"),
            *(cfg.get("raw_audio_candidates", []) or []),
        ]
        raw_drive = first_existing(raw_candidates)

        if raw_drive:
            copied = copy_tree_files(raw_drive, local_raw, allowed=lambda p: p.suffix.lower() in AUDIO_EXTS)
            print(f"Audios brutos copiados de {raw_drive}: {copied}")

    if count_files(local_raw, allowed=lambda p: p.suffix.lower() in AUDIO_EXTS) == 0:
        discovered = discover_kaggle_audio_dirs()
        for audio_dir in discovered:
            copied = copy_tree_files(audio_dir, local_raw / audio_dir.name, allowed=lambda p: p.suffix.lower() in AUDIO_EXTS)
            print(f"Audios brutos encontrados automaticamente em {audio_dir}: {copied}")
            if copied:
                break

    if count_files(local_raw, allowed=lambda p: p.suffix.lower() in AUDIO_EXTS) == 0:
         print("❌ NENHUM ÁUDIO BRUTO ENCONTRADO! Verifique os caminhos R2 ou Kaggle Input.")
         print("Dica: configure os Kaggle Secrets R2_* ou anexe um Dataset contendo arquivos .wav/.mp3/.flac.")
         return 1

    print("\n[INFO] Iniciando Limpeza IA (necessario para garantir audios consistentes para treino)...")
    run([
        sys.executable,
        str(project_dir / "limpeza_ia.py"),
        "--input_dir", str(local_raw),
        "--output_dir", str(local_processed),
        "--ambiente", "kaggle",
        "--enhancer", "resemble" if cfg.get("enable_resemble_enhance", True) else "auto",
        "--force",
    ], cwd=project_dir)

    if tts_engine == "f5_tts_ptbr":
        if not f5_library_dir:
            raise RuntimeError("tts_engine=f5_tts_ptbr, mas a biblioteca F5-TTS PT-BR nao foi preparada.")
        f5_package_dir = run_f5_tts_training(cfg, huggingface_cfg, f5_library_dir, local_processed)
        cleanup_intermediate_audio(cfg, local_raw, local_processed)
        report_working_disk("apos treino/exportacao F5-TTS PT-BR")
        print("\n✅ Treino F5-TTS PT-BR finalizado! Pacote da voz em:", f5_package_dir)
        return 0

    dataset_dir = Path("/kaggle/working/super_Voz_styletts2_data")
    if cfg.get("cleanup_previous_dataset", True) and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
        print(f"[DISCO] Dataset preparado de execucao anterior removido: {dataset_dir}")
    prepare_cmd = [
        sys.executable,
        str(project_dir / "scripts" / "prepare_styletts2_dataset.py"),
        "--input_dir", str(local_processed),
        "--output_dir", str(dataset_dir),
        "--speaker", str(cfg.get("speaker", "0")),
        "--sample_rate", str(cfg.get("sample_rate", 24000)),
    ]
    if cfg.get("phonemize", True):
        prepare_cmd.extend(["--phonemize", "--phonemizer_language", str(cfg.get("phonemizer_language", "pt-br"))])

    run(prepare_cmd, cwd=project_dir)

    terabox_download_styletts2(terabox_cfg, style_dir)

    # Copiar listas para o StyleTTS2
    (style_dir / "Data").mkdir(parents=True, exist_ok=True)
    for name in ["train_list.txt", "val_list.txt", "OOD_texts.txt"]:
        shutil.copy2(dataset_dir / "Data" / name, style_dir / "Data" / name)

    if latest_finetune_checkpoint(style_dir):
        print("[DISCO] Checkpoint da voz restaurado; download do checkpoint base LibriTTS pulado.")
    elif tts_engine == "f5_tts_ptbr":
        raise RuntimeError(
            "Nenhum checkpoint da voz foi encontrado e o fallback LibriTTS em ingles esta bloqueado. "
            "A biblioteca F5-TTS PT-BR ja foi preparada separadamente; use o runner F5-TTS PT-BR "
            "para iniciar a nova base/voz neural sem contaminar o treino com LibriTTS."
        )
    else:
        download_pretrained(style_dir)
    config_path = patch_styletts2_config(style_dir, dataset_dir, cfg)
    materialize_voice_package(
        package_dir,
        style_dir,
        dataset_dir,
        local_processed,
        project_dir,
        config_path,
        cfg,
    )
    if huggingface_cfg:
        print("[HuggingFace] Upload inicial do pacote pulado; o proximo envio ocorrera no primeiro checkpoint de epoca.")
    cleanup_intermediate_audio(cfg, local_raw, local_processed)
    report_working_disk("apos remover audios intermediarios")

    sync_stop = None
    sync_thread = None
    tb_sync_stop = None
    tb_sync_thread = None
    hf_sync_stop = None
    hf_sync_thread = None
    checkpoint_state = None
    training_succeeded = False
    r2_cfg = cfg.get("cloudflare_r2", {})
    output_prefix = None if r2_cfg.get("disable_r2_uploads") else r2_cfg.get("output_prefix")
    if s3 and bucket and output_prefix:
        interval_seconds = int(cfg.get("r2_sync_interval_seconds", 600))
        sync_stop, sync_thread, checkpoint_state = start_periodic_checkpoint_sync(
            s3,
            bucket,
            output_prefix,
            style_dir,
            max(60, interval_seconds),
        )
    if terabox_cfg:
        tb_interval_seconds = int(terabox_cfg.get("sync_interval_seconds", 600))
        tb_sync_stop, tb_sync_thread = start_periodic_terabox_checkpoint_sync(
            terabox_cfg,
            style_dir,
            max(60, tb_interval_seconds),
        )
    if huggingface_cfg:
        hf_interval_seconds = int(huggingface_cfg.get("sync_interval_seconds", 300))
        hf_sync_stop, hf_sync_thread = start_huggingface_checkpoint_sync(
            huggingface_cfg,
            package_dir,
            style_dir,
            dataset_dir,
            local_processed,
            project_dir,
            config_path,
            cfg,
            max(30, hf_interval_seconds),
        )

    try:
        if not args.skip_train:
            report_working_disk("antes do treinamento")
            run_training_with_progress([
                "accelerate", "launch",
                "--mixed_precision=fp16",
                "--num_processes=1",
                "train_finetune_accelerate.py",
                "--config_path", str(config_path),
            ], cwd=style_dir)
        training_succeeded = True
    finally:
        if sync_stop and sync_thread:
            sync_stop.set()
            sync_thread.join(timeout=30)
        if tb_sync_stop and tb_sync_thread:
            tb_sync_stop.set()
            tb_sync_thread.join(timeout=30)
        if hf_sync_stop and hf_sync_thread:
            hf_sync_stop.set()
            hf_sync_thread.join(timeout=30)
        terabox_upload_checkpoints(terabox_cfg, style_dir)
        huggingface_sync_package(
            huggingface_cfg,
            package_dir,
            style_dir,
            dataset_dir,
            local_processed,
            project_dir,
            config_path,
            cfg,
            reason="checkpoint critico/final antes de encerrar sessao",
        )
        report_working_disk("apos sincronizacao final")
        sync_outputs(style_dir, dataset_dir, cfg, s3, bucket, checkpoint_state, training_succeeded)

    print("\n✅ Treino finalizado! Pacote da voz em:", package_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
