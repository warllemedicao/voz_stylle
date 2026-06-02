#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}


def run(cmd, cwd=None, check=True):
    print("\n$ " + " ".join(map(str, cmd)))
    if cwd:
        print("cwd:", cwd)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def run_training_with_progress(cmd, cwd=None) -> None:
    print("\n$ " + " ".join(map(str, cmd)))
    if cwd:
        print("cwd:", cwd)
    print("[TREINO] Iniciando. O console mostrara uma barra por epoca/passo; detalhes ficam em Models/super_Voz/train.log.")

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

    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue

        train_match = train_re.search(line)
        if train_match:
            epoch, epochs, step, steps, loss = train_match.groups()
            step_i = int(step)
            steps_i = max(1, int(steps))
            filled = int(30 * step_i / steps_i)
            bar = "#" * filled + "-" * (30 - filled)
            current_line = f"[TREINO] Epoca {epoch}/{epochs} |{bar}| passo {step}/{steps} loss {loss}"
            print("\r" + current_line, end="", flush=True)
            continue

        val_match = val_re.search(line)
        if val_match:
            if current_line:
                print()
                current_line = ""
            val_loss, dur_loss, f0_loss = val_match.groups()
            print(f"[VALIDACAO] loss {val_loss} | dur {dur_loss} | f0 {f0_loss}")
            continue

        if important_re.search(line):
            if current_line:
                print()
                current_line = ""
            print(line)

    if current_line:
        print()

    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def clone_or_pull(url: str, dest: Path) -> None:
    if dest.exists():
        # Garantir que estamos em uma branch antes de dar pull
        run(["git", "-C", str(dest), "checkout", "main"], check=False)
        run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
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

    # No Linux/Colab, caminhos absolutos começam com /
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


def setup_cuda_memory_env() -> None:
    """Reduz fragmentação de memória CUDA no Colab/PyTorch."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[INFO] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")


def install_dependencies(style_dir: Path) -> None:
    missing_sys = []
    for pkg in ["ffmpeg", "sox", "espeak-ng"]:
        if shutil.which(pkg) is None:
            missing_sys.append(pkg)

    if missing_sys:
        print(f"[INFO] Instalando pacotes de sistema: {missing_sys}")
        run(["apt-get", "update"])
        run(["apt-get", "install", "-y", "ffmpeg", "sox", "libsndfile1", "espeak-ng"])

    print("[INFO] Verificando/Instalando dependências Python...")
    # Desinstalar onnxruntime comum para evitar conflito com a versão GPU
    run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"], check=False)
    
    # Configuração para instalação rápida do DeepSpeed sem compilação de C++ ops
    os.environ["DS_BUILD_OPS"] = "0"

    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "torch",
        "torchaudio",
        "torchvision",
        "accelerate",
        "huggingface_hub",
        "pyyaml",
        "librosa",
        "soundfile",
        "phonemizer",
        "openai-whisper",
        "demucs",
        "boto3",
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
        # Dependências do Resemble já foram instaladas acima; --no-deps preserva o stack torch/torchaudio do Colab.
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-deps", "resemble-enhance"])
    else:
        print("[INFO] Pulando resemble-enhance no Colab porque SUPER_VOZ_ENABLE_RESEMBLE=0.")

    requirements = style_dir / "requirements.txt"
    if requirements.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=False)


def download_pretrained(style_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    filename = "Models/LibriTTS/epochs_2nd_00020.pth"
    path = hf_hub_download(
        repo_id="yl4579/StyleTTS2-LibriTTS",
        filename=filename,
        local_dir=str(style_dir),
        local_dir_use_symlinks=False,
    )
    return Path(path)


def get_r2_client(cfg: dict):
    import boto3
    from botocore.config import Config

    r2_cfg = cfg.get("cloudflare_r2", {})
    if not r2_cfg or "INSERIR" in r2_cfg.get("access_key_id", ""):
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
            # Normaliza separadores para o S3
            key = os.path.join(prefix, rel_path).replace("\\", "/")
            
            print(f"  + Enviando: {src_path.name} -> {key}")
            s3.upload_file(str(src_path), bucket, key)
            uploaded += 1
    return uploaded


def latest_finetune_checkpoint(style_dir: Path) -> Path | None:
    ckpt_dir = style_dir / "Models" / "super_Voz"
    checkpoints = sorted(ckpt_dir.glob("epoch_2nd_*.pth"))
    return checkpoints[-1] if checkpoints else None


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

    # Parâmetros defensivos contra CUDA OOM na fase SLM adversarial.
    # joint_epoch=999 mantém essa fase desligada por padrão no Colab comum,
    # mas estes limites protegem caso ela seja ativada no futuro.
    slmadv_params = config.setdefault("slmadv_params", {})
    slmadv_params["batch_percentage"] = float(cfg.get("batch_percentage", 0.125))
    slmadv_params["min_len"] = int(cfg.get("slm_min_len", 120))
    slmadv_params["max_len"] = int(cfg.get("slm_max_len", 220))

    out_path = style_dir / "Configs" / "config_super_voz.yml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return out_path


def prepare_local_audio(project_dir: Path, cfg: dict) -> tuple[Path, bool]:
    local_raw = project_dir / "Audios_brutos"
    local_processed = project_dir / "Audios_processados"
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
        print("[AVISO] Configuração R2 incompleta ou ausente. Verificando candidatos locais/Drive...")
        raw_candidates = cfg.get("raw_audio_candidates", [])
        raw_drive = first_existing(raw_candidates)

        if raw_drive:
            copied = copy_tree_files(raw_drive, local_raw, allowed=lambda p: p.suffix.lower() in AUDIO_EXTS)
            print(f"Audios brutos importados: {copied}")

    # SEMPRE rodamos a limpeza agora
    if not any(local_raw.rglob("*")):
        error_msg = (
            "\n" + "!" * 60 + "\n"
            " [ERRO] NENHUM ÁUDIO BRUTO ENCONTRADO\n"
            "!" * 60 + "\n"
            "Verifique se você configurou o Cloudflare R2 corretamente ou se os arquivos estão na pasta 'Audios_brutos'.\n"
            "!" * 60
        )
        raise FileNotFoundError(error_msg)

    print("\n[INFO] Iniciando Limpeza IA (necessário para garantir formato StyleTTS2)...")
    run([
        sys.executable,
        "limpeza_ia.py",
        "--input_dir",
        str(local_raw),
        "--output_dir",
        str(local_processed),
        "--ambiente",
        "colab",
        "--enhancer",
        "auto",
        "--force", # Forçamos para garantir que ignore cache de limpeza anterior
    ], cwd=project_dir)

    train_txt = local_processed / "train.txt"
    if not train_txt.exists() or not train_txt.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"train.txt nao encontrado ou vazio: {train_txt}")

    return local_processed, False # Retornamos False para 'imported_processed' para indicar que foi gerado agora


def sync_outputs(style_dir: Path, dataset_dir: Path, cfg: dict) -> None:
    print("\n" + "="*60)
    print(" ✅ PROCESSAMENTO FINALIZADO!")
    print("="*60)
    print(f"Checkpoints em: {style_dir / 'Models' / 'super_Voz'}")
    print(f"Dataset em: {dataset_dir}")
    print("Nota: Sincronização automática com R2/Drive desativada.")
    print("Você pode baixar os arquivos manualmente usando o menu lateral do Colab.")
    print("="*60 + "\n")


def patch_pytorch_compatibility(style_dir: Path) -> None:
    """Aplica patch no StyleTTS2 para compatibilidade com PyTorch 2.6+.

    O PyTorch 2.6 mudou o padrão de torch.load para weights_only=True.
    Os checkpoints do StyleTTS2/ASR carregam objetos Python extras, então as
    chamadas antigas torch.load(..., map_location=...) falham com
    _pickle.UnpicklingError. Como o checkpoint vem do repositório/modelo confiado
    pelo próprio pipeline, forçamos weights_only=False nos carregamentos internos
    do StyleTTS2.
    """
    models_py = style_dir / "models.py"
    if not models_py.exists():
        print("[AVISO] models.py não encontrado para aplicação de patch.")
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
        print("✅ Patch PyTorch 2.6+ aplicado: torch.load(..., weights_only=False).")
    else:
        print("ℹ️ Nenhuma chamada torch.load antiga encontrada em models.py; patch já pode estar aplicado.")


def patch_styletts2_oom_safety(style_dir: Path) -> None:
    """Aplica pequenos patches no treino do StyleTTS2 para reduzir OOM em Colab/T4.

    O train_finetune_accelerate.py já limita o recorte principal por max_len,
    mas a validação e o recorte de referência podem usar trechos longos demais.
    Isso estoura memória no HiFi-GAN/Snake1D em GPUs de 15 GB.
    """
    train_py = style_dir / "train_finetune_accelerate.py"
    if not train_py.exists():
        print("[AVISO] train_finetune_accelerate.py não encontrado para patch OOM.")
        return

    print("[INFO] Aplicando patch anti-OOM in train_finetune_accelerate.py...")
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
        print("✅ Patch anti-OOM aplicado: validação/referência agora respeitam max_len.")
    else:
        print("ℹ️ Patch anti-OOM já estava aplicado ou o script mudou de formato.")


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
            print("Verifique se você selecionou 'GPU' em Ambiente de Execução -> Alterar tipo.")
            return False
    except ImportError:
        print("❌ PyTorch não instalado.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline Colab super_Voz com StyleTTS2.")
    parser.add_argument("--config", default="styletts2_colab_config.yml")
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    import yaml

    project_dir = Path(__file__).resolve().parents[1]
    with (project_dir / args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_cuda_memory_env()

    # Verifica GPU antes de tudo para evitar SIGSEGV
    if not verify_gpu():
        print("🛑 Abortando devido à falta de GPU.")
        return 1

    style_dir = Path(cfg.get("styletts2_dir", "/content/StyleTTS2"))
    clone_or_pull(cfg.get("styletts2_repo", "https://github.com/yl4579/StyleTTS2.git"), style_dir)

    # APLICA PATCHES LOGO APÓS CLONAR/ATUALIZAR O STYLETTS2.
    patch_pytorch_compatibility(style_dir)
    patch_styletts2_oom_safety(style_dir)
    patch_styletts2_zero_division_safety(style_dir)

    install_dependencies(style_dir)

    processed_dir, imported_processed = prepare_local_audio(project_dir, cfg)
    print(f"Dataset processado: {processed_dir}")
    print(f"Limpeza/transcricao pulada: {imported_processed}")

    dataset_dir = Path("/content/super_Voz_styletts2_data")
    prepare_cmd = [
        sys.executable,
        str(project_dir / "scripts" / "prepare_styletts2_dataset.py"),
        "--input_dir",
        str(processed_dir),
        "--output_dir",
        str(dataset_dir),
        "--speaker",
        str(cfg.get("speaker", "0")),
        "--sample_rate",
        str(cfg.get("sample_rate", 24000)),
        "--val_ratio",
        str(cfg.get("val_ratio", 0.05)),
    ]

    if cfg.get("phonemize", True):
        prepare_cmd.extend(["--phonemize", "--phonemizer_language", str(cfg.get("phonemizer_language", "pt-br"))])

    if cfg.get("max_audio_seconds") is not None:
        prepare_cmd.extend(["--max_seconds", str(cfg.get("max_audio_seconds"))])
    if cfg.get("max_text_chars") is not None:
        prepare_cmd.extend(["--max_text_chars", str(cfg.get("max_text_chars"))])

    run(prepare_cmd, cwd=project_dir)

    (style_dir / "Data").mkdir(parents=True, exist_ok=True)
    for name in ["train_list.txt", "val_list.txt", "OOD_texts.txt"]:
        shutil.copy2(dataset_dir / "Data" / name, style_dir / "Data" / name)

    pretrained = download_pretrained(style_dir)
    print(f"Checkpoint base: {pretrained}")

    config_path = patch_styletts2_config(style_dir, dataset_dir, cfg)
    print(f"Config StyleTTS2: {config_path}")

    if not args.skip_train:
        train_script = style_dir / "train_finetune_accelerate.py"
        if not train_script.exists():
            raise FileNotFoundError(f"Script de fine-tuning nao encontrado: {train_script}")

        run_training_with_progress([
            "accelerate",
            "launch",
            "--mixed_precision=fp16",
            "--num_processes=1",
            str(train_script.name),
            "--config_path",
            str(config_path),
        ], cwd=style_dir)

    sync_outputs(style_dir, dataset_dir, cfg)
    print("Pipeline super_Voz finalizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
