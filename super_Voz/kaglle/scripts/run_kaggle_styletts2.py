#!/usr/bin/env python3
import argparse
import sys
import os
import shutil
from pathlib import Path
import yaml

from runner_utils.utils import *
from runner_utils.cloud_storage import *
from runner_utils.environment import *
from runner_utils.f5_integration import *

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}

def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline Kaggle super_Voz com StyleTTS2.")
    parser.add_argument("--config", default="styletts2_kaggle_config.yml")
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

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
