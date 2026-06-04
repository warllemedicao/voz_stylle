#!/usr/bin/env python3
import argparse
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
        for env_name in env_names:
            value = os.environ.get(env_name)
            if value:
                out[key] = value
                break
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
    token = os.environ.get(token_env, "")
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
    hf_cfg["_sync_lock"] = threading.Lock()
    print(f"[HuggingFace] Bucket configurado: {bucket_uri}")
    return hf_cfg


def huggingface_restore_package(hf_cfg: dict | None, package_dir: Path) -> bool:
    if not hf_cfg:
        return False

    package_dir.mkdir(parents=True, exist_ok=True)
    result = run(
        ["hf", "sync", str(hf_cfg["bucket_uri"]), str(package_dir)],
        check=False,
    )
    if result.returncode == 0:
        print(f"[HuggingFace] Pacote restaurado em {package_dir}.")
        return True
    print("[HuggingFace][AVISO] Nao foi possivel restaurar o bucket; o treino continuara sem estado remoto.")
    return False


def prune_uploaded_checkpoints(checkpoint_dir: Path, uploaded_checkpoint: Path) -> int:
    checkpoints = sorted(checkpoint_dir.glob("epoch_2nd_*.pth"))
    removed = 0
    for path in checkpoints:
        if path.name > uploaded_checkpoint.name:
            continue
        path.unlink()
        removed += 1
        print(f"[DISCO] Checkpoint local antigo removido apos upload: {path.name}")
    return removed


def remove_pretrained_base_after_finetune_upload(style_dir: Path) -> None:
    base_checkpoint = style_dir / "Models" / "LibriTTS" / "epochs_2nd_00020.pth"
    if not base_checkpoint.exists():
        return
    base_checkpoint.unlink()
    print(f"[DISCO] Checkpoint base removido apos persistir a voz treinada: {base_checkpoint}")


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
        replace_with_hardlink_or_copy(latest, model_dir / "best_model.pth")
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
        result = run(
            ["hf", "sync", str(package_dir), str(hf_cfg["bucket_uri"]), "--delete"],
            check=False,
        )
        if result.returncode != 0:
            print("[HuggingFace][AVISO] Upload do pacote falhou; checkpoints locais foram preservados.")
            report_working_disk("upload Hugging Face falhou")
            return False
        print(f"[HuggingFace] Pacote sincronizado: {package_dir} -> {hf_cfg['bucket_uri']}")
        if latest and latest.parent.name == "super_Voz":
            prune_uploaded_checkpoints(style_dir / "Models" / "super_Voz", latest)
            remove_pretrained_base_after_finetune_upload(style_dir)
        return True
    finally:
        lock.release()


def start_periodic_huggingface_package_sync(
    hf_cfg: dict | None,
    package_dir: Path,
    style_dir: Path,
    dataset_dir: Path,
    processed_dir: Path,
    project_dir: Path,
    config_path: Path,
    cfg: dict,
    interval_seconds: int,
) -> tuple[threading.Event, threading.Thread] | tuple[None, None]:
    if not hf_cfg:
        return None, None

    stop_event = threading.Event()

    def worker() -> None:
        last_checkpoint = None
        last_sync = time.monotonic()
        while not stop_event.wait(5):
            latest = latest_finetune_checkpoint(style_dir)
            checkpoint_state = None
            if latest and latest.parent.name == "super_Voz":
                stat = latest.stat()
                checkpoint_state = (latest.name, stat.st_size, stat.st_mtime_ns)
            due_for_log = time.monotonic() - last_sync >= interval_seconds
            if checkpoint_state == last_checkpoint and not due_for_log:
                continue
            if huggingface_sync_package(
                hf_cfg, package_dir, style_dir, dataset_dir, processed_dir, project_dir, config_path, cfg
            ):
                last_checkpoint = checkpoint_state
                last_sync = time.monotonic()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print(
        "[HuggingFace] Monitor de checkpoints ativo a cada 5s; "
        f"logs sincronizados no maximo a cada {interval_seconds}s."
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
    ckpt_dir = style_dir / "Models" / "super_Voz"
    checkpoints = sorted(path for path in ckpt_dir.glob("epoch_2nd_*.pth") if zipfile.is_zipfile(path))
    if checkpoints:
        return checkpoints[-1]
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


def sync_outputs(style_dir: Path, dataset_dir: Path, cfg: dict, s3=None, bucket=None, checkpoint_state=None) -> None:
    print("\n" + "="*60)
    print(" ✅ TREINO FINALIZADO!")
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

    style_dir = Path(cfg.get("styletts2_dir", "/kaggle/working/StyleTTS2"))
    clone_or_pull(cfg.get("styletts2_repo", "https://github.com/yl4579/StyleTTS2.git"), style_dir)
    package_dir = style_dir / str(cfg.get("voice_package_dir", "minha_voz_styletts2"))

    restore_styletts2_from_candidates(cfg, style_dir)
    terabox_cfg = setup_terabox(cfg)
    huggingface_cfg = setup_huggingface(cfg)
    restored_hf = huggingface_restore_package(huggingface_cfg, package_dir)
    if huggingface_cfg and huggingface_cfg.get("required", False) and not restored_hf:
        raise RuntimeError("[HuggingFace] Nao foi possivel acessar o bucket obrigatorio antes do treino.")

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

    print("\n[INFO] Iniciando Limpeza IA (necessário para garantir formato StyleTTS2)...")
    run([
        sys.executable,
        str(project_dir / "limpeza_ia.py"),
        "--input_dir", str(local_raw),
        "--output_dir", str(local_processed),
        "--ambiente", "kaggle",
        "--enhancer", "resemble" if cfg.get("enable_resemble_enhance", True) else "auto",
        "--force",
    ], cwd=project_dir)

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
    else:
        download_pretrained(style_dir)
    config_path = patch_styletts2_config(style_dir, dataset_dir, cfg)
    initial_hf_sync = huggingface_sync_package(
        huggingface_cfg,
        package_dir,
        style_dir,
        dataset_dir,
        local_processed,
        project_dir,
        config_path,
        cfg,
    )
    if huggingface_cfg and huggingface_cfg.get("required", False) and not initial_hf_sync:
        raise RuntimeError("[HuggingFace] Nao foi possivel validar o upload do pacote antes do treino.")
    cleanup_intermediate_audio(cfg, local_raw, local_processed)
    report_working_disk("apos remover audios intermediarios")

    sync_stop = None
    sync_thread = None
    tb_sync_stop = None
    tb_sync_thread = None
    hf_sync_stop = None
    hf_sync_thread = None
    checkpoint_state = None
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
        hf_interval_seconds = int(huggingface_cfg.get("sync_interval_seconds", 60))
        hf_sync_stop, hf_sync_thread = start_periodic_huggingface_package_sync(
            huggingface_cfg,
            package_dir,
            style_dir,
            dataset_dir,
            local_processed,
            project_dir,
            config_path,
            cfg,
            max(15, hf_interval_seconds),
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
        )
        report_working_disk("apos sincronizacao final")
        sync_outputs(style_dir, dataset_dir, cfg, s3, bucket, checkpoint_state)

    print("\n✅ Treino finalizado! Pacote da voz em:", package_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
