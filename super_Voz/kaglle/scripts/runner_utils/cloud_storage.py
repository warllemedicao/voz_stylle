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
from .utils import *

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

    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", HF_HUB_COMPAT_PACKAGE])
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

def run_quiet(cmd, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

