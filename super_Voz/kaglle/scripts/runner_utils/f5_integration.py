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
from .cloud_storage import restore_huggingface_subdir, upload_huggingface_subdir
from .environment import install_f5_tts_dependencies

def f5_library_vocab_path(library_dir: Path, f5_cfg: dict) -> Path | None:
    subpath = str(f5_cfg.get("base_vocab_subpath", "vocab.txt")).strip()
    if subpath:
        candidate = library_dir / subpath
        if candidate.exists():
            return candidate

    candidates = sorted(library_dir.rglob("vocab.txt"), key=lambda path: (len(path.parts), path.as_posix()))
    return candidates[0] if candidates else None

def count_f5_vocab_rows(vocab_path: Path) -> int:
    tokens = [line for line in vocab_path.read_text(encoding="utf-8").splitlines() if line]
    return len(tokens) + 1

def validate_f5_library(library_dir: Path, f5_cfg: dict, *, strict: bool = False) -> bool:
    if not library_dir.exists() or not any(library_dir.iterdir()):
        if strict:
            raise RuntimeError(f"Biblioteca F5-TTS PT-BR vazia ou inexistente: {library_dir}")
        return False

    checkpoint_ok = False
    try:
        f5_checkpoint_path(library_dir, f5_cfg)
        checkpoint_ok = True
    except FileNotFoundError as exc:
        if strict:
            raise RuntimeError(str(exc)) from exc

    use_base_vocab = bool(f5_cfg.get("use_base_vocab", False))
    vocab_ok = True
    if use_base_vocab:
        vocab_path = f5_library_vocab_path(library_dir, f5_cfg)
        if not vocab_path:
            if strict:
                raise RuntimeError(
                    f"f5_tts_ptbr.use_base_vocab=true, mas vocab.txt nao foi encontrado em {library_dir}."
                )
            vocab_ok = False
        else:
            rows = count_f5_vocab_rows(vocab_path)
            expected_rows = f5_cfg.get("expected_vocab_rows")
            if expected_rows and rows != int(expected_rows):
                message = (
                    f"vocab.txt da biblioteca F5 gera {rows} linhas de embedding, "
                    f"mas a config espera {int(expected_rows)}."
                )
                if strict:
                    raise RuntimeError(message)
                print(f"[F5-TTS-PT-BR][AVISO] {message}")
                vocab_ok = False

    return checkpoint_ok and vocab_ok

def ensure_f5_tts_ptbr_library(cfg: dict, hf_cfg: dict | None) -> Path | None:
    f5_cfg = cfg.get("f5_tts_ptbr", {}) or {}
    if not f5_cfg.get("enabled", False):
        return None

    default_root = Path(cfg.get("model_library_root", "/kaggle/working/super_voz_model_library"))
    local_dir = Path(f5_cfg.get("local_dir", str(default_root / "f5_tts_ptbr_tharyck")))
    remote_dir = str(f5_cfg.get("huggingface_remote_dir", "libraries/f5_tts_ptbr_tharyck")).strip("/")
    repo_id = str(f5_cfg.get("repo_id", "Tharyck/multispeaker-ptbr-f5tts")).strip()

    if local_dir.exists() and any(local_dir.iterdir()):
        if validate_f5_library(local_dir, f5_cfg):
            print(f"[F5-TTS-PT-BR] Biblioteca local encontrada: {local_dir}")
            return local_dir
        print(f"[F5-TTS-PT-BR][AVISO] Biblioteca local incompatível; baixando novamente: {local_dir}")
        shutil.rmtree(local_dir, ignore_errors=True)

    if restore_huggingface_subdir(hf_cfg, remote_dir, local_dir) and validate_f5_library(local_dir, f5_cfg):
        return local_dir
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"[F5-TTS-PT-BR][AVISO] Biblioteca restaurada incompatível; baixando de {repo_id}.")
        shutil.rmtree(local_dir, ignore_errors=True)

    if not repo_id:
        raise RuntimeError("f5_tts_ptbr.repo_id nao configurado e biblioteca nao foi restaurada do Hugging Face.")

    print(f"[F5-TTS-PT-BR] Baixando biblioteca/base de {repo_id} para {local_dir}")
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", HF_HUB_COMPAT_PACKAGE])
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_kwargs = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
    }
    allow_patterns = config_string_list(f5_cfg.get("download_allow_patterns"))
    if allow_patterns:
        snapshot_kwargs["allow_patterns"] = allow_patterns
        print(f"[F5-TTS-PT-BR] Download limitado aos arquivos: {', '.join(allow_patterns)}")
    snapshot_download(**snapshot_kwargs)
    validate_f5_library(local_dir, f5_cfg, strict=True)
    print(f"[F5-TTS-PT-BR] Biblioteca/base disponivel em: {local_dir}")
    upload_huggingface_subdir(hf_cfg, local_dir, remote_dir)
    return local_dir

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
    subpath = str(f5_cfg.get("checkpoint_subpath", "model_last.safetensors")).strip()
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

def normalize_f5_ema_state_dict(state_dict: dict) -> dict:
    normalized = dict(state_dict)
    tensor_factory = None
    for value in normalized.values():
        if hasattr(value, "new_tensor"):
            tensor_factory = value.new_tensor
            break

    has_ema_prefix = any(str(key).startswith("ema_model.") for key in normalized)
    has_raw_transformer = any(str(key).startswith("transformer.") for key in normalized)
    if has_raw_transformer and not has_ema_prefix:
        normalized = {
            f"ema_model.{key}" if str(key).startswith("transformer.") else key: value
            for key, value in normalized.items()
        }

    if tensor_factory and "initted" not in normalized:
        normalized["initted"] = tensor_factory(True)
    if tensor_factory and "step" not in normalized:
        normalized["step"] = tensor_factory(0)
    return normalized

def apply_f5_base_vocab_to_dataset(f5_library_dir: Path, f5_dataset_dir: Path, f5_cfg: dict) -> None:
    if not bool(f5_cfg.get("use_base_vocab", False)):
        return

    base_vocab = f5_library_vocab_path(f5_library_dir, f5_cfg)
    if not base_vocab:
        raise RuntimeError(
            "f5_tts_ptbr.use_base_vocab=true, mas a biblioteca base nao contem vocab.txt. "
            "Use uma biblioteca F5 completa com vocabulario publicado."
        )

    target_vocab = f5_dataset_dir / "vocab.txt"
    shutil.copy2(base_vocab, target_vocab)
    rows = count_f5_vocab_rows(target_vocab)
    expected_rows = f5_cfg.get("expected_vocab_rows")
    if expected_rows and rows != int(expected_rows):
        raise RuntimeError(
            f"vocab.txt base copiado de {base_vocab} gera {rows} linhas de embedding; "
            f"esperado pela config: {int(expected_rows)}."
        )
    print(
        "[F5-TTS-PT-BR] vocab.txt da biblioteca base aplicado ao dataset: "
        f"{base_vocab} -> {target_vocab} ({rows} linhas de embedding)."
    )

def adapt_f5_text_embedding_to_vocab(state_dict: dict, target_rows: int | None) -> tuple[dict, bool]:
    import torch

    if not target_rows:
        return state_dict, False

    key = "ema_model.transformer.text_embed.text_embed.weight"
    weight = state_dict.get(key)
    if weight is None or not hasattr(weight, "shape") or len(weight.shape) != 2:
        print(f"[F5-TTS-PT-BR][AVISO] Embedding de texto nao encontrado no checkpoint; chave esperada: {key}")
        return state_dict, False

    current_rows = int(weight.shape[0])
    if current_rows == target_rows:
        return state_dict, False

    adapted = dict(state_dict)
    if current_rows > target_rows:
        adapted[key] = weight[:target_rows].clone()
        print(
            "[F5-TTS-PT-BR] Embedding de texto do pretrain ajustado ao vocabulario atual: "
            f"{current_rows} -> {target_rows} linhas."
        )
        return adapted, True

    extra_rows = weight.new_zeros((target_rows - current_rows, int(weight.shape[1])))
    adapted[key] = torch.cat([weight, extra_rows], dim=0)
    print(
        "[F5-TTS-PT-BR] Embedding de texto do pretrain expandido ao vocabulario atual: "
        f"{current_rows} -> {target_rows} linhas."
    )
    return adapted, True

def build_f5_trainer_checkpoint(state_dict: dict, target_vocab_rows: int | None) -> tuple[dict, bool]:
    normalized = normalize_f5_ema_state_dict(state_dict)
    normalized, changed = adapt_f5_text_embedding_to_vocab(normalized, target_vocab_rows)
    return {"ema_model_state_dict": normalized}, changed

def ensure_f5_pretrain_checkpoint(base_checkpoint: Path, checkpoint_dir: Path, target_vocab_rows: int | None = None) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vocab_suffix = f"_vocab{target_vocab_rows}" if target_vocab_rows else ""
    converted = checkpoint_dir / f"pretrained_{base_checkpoint.stem}_ema{vocab_suffix}.pt"
    for stale in checkpoint_dir.glob("pretrained_*"):
        if stale.resolve() == converted.resolve():
            continue
        try:
            stale.unlink()
            print(f"[F5-TTS-PT-BR] Checkpoint pretrain antigo removido para evitar carga incompatível: {stale.name}")
        except Exception as exc:
            print(f"[F5-TTS-PT-BR][AVISO] Nao foi possivel remover pretrain antigo {stale}: {exc}")

    if converted.exists() and converted.stat().st_size > 0:
        print(f"[F5-TTS-PT-BR] Checkpoint pretrain compatível reutilizado: {converted}")
        return converted

    suffix = base_checkpoint.suffix.lower()
    if suffix == ".safetensors":
        print(f"[F5-TTS-PT-BR] Convertendo checkpoint safetensors para formato EMA do trainer: {base_checkpoint}")
        from safetensors.torch import load_file
        import torch

        raw_state = load_file(str(base_checkpoint), device="cpu")
        checkpoint, _ = build_f5_trainer_checkpoint(raw_state, target_vocab_rows)
        save_f5_trainer_checkpoint(checkpoint, converted)
        print(f"[F5-TTS-PT-BR] Checkpoint pretrain compatível criado: {converted}")
        return converted

    if suffix in {".pt", ".pth"}:
        import torch

        loaded = torch.load(base_checkpoint, weights_only=True, map_location="cpu")
        if isinstance(loaded, dict) and "ema_model_state_dict" in loaded:
            checkpoint, changed = build_f5_trainer_checkpoint(loaded["ema_model_state_dict"], target_vocab_rows)
            if not changed:
                print(f"[F5-TTS-PT-BR] Checkpoint pretrain já está em formato trainer: {base_checkpoint}")
                return base_checkpoint
            save_f5_trainer_checkpoint(checkpoint, converted)
            print(f"[F5-TTS-PT-BR] Checkpoint pretrain compatível criado: {converted}")
            return converted
        if isinstance(loaded, dict):
            checkpoint, _ = build_f5_trainer_checkpoint(loaded, target_vocab_rows)
            save_f5_trainer_checkpoint(checkpoint, converted)
            print(f"[F5-TTS-PT-BR] Checkpoint pretrain compatível criado: {converted}")
            return converted

    print(f"[F5-TTS-PT-BR][AVISO] Formato de checkpoint nao reconhecido para conversao; usando original: {base_checkpoint}")
    return base_checkpoint

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

def f5_checkpoint_state(path: Path | None) -> tuple[str, int, int] | None:
    if not path or not path.exists():
        return None
    stat = path.stat()
    return (path.name, stat.st_size, stat.st_mtime_ns)

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
    apply_f5_base_vocab_to_dataset(f5_library_dir, f5_dataset_dir, f5_cfg)

    checkpoint_dir = f5_package_path(f"../../ckpts/{dataset_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint = f5_checkpoint_path(f5_library_dir, f5_cfg)
    target_vocab_rows = f5_dataset_vocab_rows(f5_dataset_dir)
    train_pretrain_checkpoint = ensure_f5_pretrain_checkpoint(base_checkpoint, checkpoint_dir, target_vocab_rows)
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
        str(train_pretrain_checkpoint),
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

def save_f5_trainer_checkpoint(checkpoint: dict, destination: Path) -> None:
    import torch

    try:
        torch.save(checkpoint, destination, _use_new_zipfile_serialization=False)
    except TypeError:
        torch.save(checkpoint, destination)

