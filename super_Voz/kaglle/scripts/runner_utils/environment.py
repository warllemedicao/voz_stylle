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

def require_commands(commands: list[str]) -> None:
    missing = [command for command in commands if shutil.which(command) is None]
    if missing:
        raise RuntimeError("Comandos obrigatorios ausentes apos instalacao: " + ", ".join(missing))

def require_python_modules(modules: dict[str, str], exact_versions: dict[str, str] | None = None) -> None:
    exact_versions = exact_versions or {}
    missing = []
    for module_name, package_name in modules.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            missing.append(f"{module_name} ({package_name}): {exc}")
            continue
        expected = exact_versions.get(package_name)
        if expected:
            try:
                installed = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                missing.append(f"{module_name} ({package_name}=={expected}, versao indisponivel)")
                continue
            if installed != expected:
                missing.append(f"{package_name}=={expected} requerido, instalado={installed}")
    if missing:
        raise RuntimeError("Dependencias obrigatorias ausentes/incompativeis: " + "; ".join(missing))

def require_python_modules_in_fresh_process(
    modules: dict[str, str],
    exact_versions: dict[str, str] | None = None,
) -> None:
    exact_versions = exact_versions or {}
    payload = json.dumps({"modules": modules, "exact_versions": exact_versions})
    code = r"""
import importlib
import importlib.metadata
import json
import sys

payload = json.loads(sys.argv[1])
missing = []
for module_name, package_name in payload["modules"].items():
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(f"{module_name} ({package_name}): {exc}")
        continue
    expected = payload["exact_versions"].get(package_name)
    if expected:
        try:
            installed = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"{module_name} ({package_name}=={expected}, versao indisponivel)")
            continue
        if installed != expected:
            missing.append(f"{package_name}=={expected} requerido, instalado={installed}")
if missing:
    print("; ".join(missing))
    raise SystemExit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, payload],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Dependencias obrigatorias ausentes/incompativeis: " + result.stdout.strip())

def reexec_after_ml_runtime_install() -> None:
    if os.environ.get("SUPER_VOZ_ML_RUNTIME_REEXECED", "0") == "1":
        return
    os.environ["SUPER_VOZ_ML_RUNTIME_REEXECED"] = "1"
    print("[INFO] Runtime ML atualizado; reiniciando o runner para recarregar Torch/Torchaudio/Torchvision.")
    os.execv(sys.executable, [sys.executable, *sys.argv])

def reexec_after_audio_cleaning_install() -> None:
    if os.environ.get("SUPER_VOZ_AUDIO_DEPS_REEXECED", "0") == "1":
        return
    os.environ["SUPER_VOZ_AUDIO_DEPS_REEXECED"] = "1"
    print("[INFO] Dependencias da limpeza atualizadas; reiniciando o runner para recarregar NumPy/SciPy/Pandas.")
    os.execv(sys.executable, [sys.executable, *sys.argv])

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
    if os.environ.get("SUPER_VOZ_ML_RUNTIME_REEXECED", "0") == "1":
        print("[INFO] Runtime ML ja foi atualizado nesta execucao; validando em processo limpo.")
        require_python_modules_in_fresh_process(ML_RUNTIME_MODULES)
        return

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
        HF_HUB_COMPAT_PACKAGE,
        "accelerate",
    ])
    require_python_modules_in_fresh_process(ML_RUNTIME_MODULES)
    reexec_after_ml_runtime_install()

def verify_audio_cleaning_dependencies(resemble_enabled: bool) -> None:
    modules = {
        "boto3": "boto3",
        "librosa": "librosa",
        "soundfile": "soundfile",
        "whisper": "openai-whisper",
        "demucs": "demucs",
        "onnxruntime": "onnxruntime-gpu",
        "deepspeed": "deepspeed",
        "scipy": "scipy",
        "tqdm": "tqdm",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "tabulate": "tabulate",
        "resampy": "resampy",
    }
    if resemble_enabled:
        modules["resemble_enhance"] = "resemble-enhance"
    require_python_modules_in_fresh_process(modules, RESEMBLE_COMPAT_VERSIONS)

def install_dependencies(style_dir: Path) -> None:
    print("\n--- Instalando Dependências ---")
    if os.environ.get("SUPER_VOZ_AUDIO_DEPS_REEXECED", "0") == "1":
        print("[INFO] Dependencias de audio/limpeza ja foram atualizadas nesta execucao; validando.")
        verify_audio_cleaning_dependencies(os.environ.get("SUPER_VOZ_ENABLE_RESEMBLE", "1") != "0")
        return
    
    # No Kaggle, tentamos instalar boto3 se não houver
    run([sys.executable, "-m", "pip", "install", "-q", "boto3"])

    missing_sys = []
    for pkg in ["ffmpeg", "sox", "espeak-ng"]:
        if shutil.which(pkg) is None:
            missing_sys.append(pkg)

    if missing_sys:
        print(f"[INFO] Instalando pacotes de sistema: {missing_sys}")
        # No Kaggle, apt-get precisa de cuidado, mas geralmente funciona
        run(["apt-get", "update"])
        run(["apt-get", "install", "-y", "ffmpeg", "sox", "libsndfile1", "espeak-ng"])
    require_commands(["ffmpeg", "sox", "espeak-ng"])

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
        HF_HUB_COMPAT_PACKAGE,
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
        "deepspeed",
        "tqdm",
        *RESEMBLE_COMPAT_PACKAGES,
    ])

    if os.environ.get("SUPER_VOZ_ENABLE_RESEMBLE", "1") != "0":
        # --no-deps preserva o stack torch/torchaudio do Kaggle; os pins críticos do Resemble entram acima.
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-deps", "resemble-enhance"])
    else:
        print("[INFO] Pulando resemble-enhance no Kaggle porque SUPER_VOZ_ENABLE_RESEMBLE=0.")

    requirements_strict = Path(__file__).resolve().parents[2] / "requirements-kaggle-strict.txt"
    if requirements_strict.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_strict)])
    
    requirements = style_dir / "requirements.txt"
    if requirements.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)])
    verify_audio_cleaning_dependencies(os.environ.get("SUPER_VOZ_ENABLE_RESEMBLE", "1") != "0")
    reexec_after_audio_cleaning_install()

def install_audio_cleaning_dependencies() -> None:
    print("\n--- Instalando Dependências da Limpeza IA ---")
    if os.environ.get("SUPER_VOZ_AUDIO_DEPS_REEXECED", "0") == "1":
        print("[INFO] Dependencias da limpeza ja foram atualizadas nesta execucao; validando.")
        verify_audio_cleaning_dependencies(os.environ.get("SUPER_VOZ_ENABLE_RESEMBLE", "1") != "0")
        return

    run([sys.executable, "-m", "pip", "install", "-q", "boto3"])

    missing_sys = []
    for pkg in ["ffmpeg", "sox", "espeak-ng"]:
        if shutil.which(pkg) is None:
            missing_sys.append(pkg)

    if missing_sys:
        print(f"[INFO] Instalando pacotes de sistema: {missing_sys}")
        run(["apt-get", "update"])
        run(["apt-get", "install", "-y", "ffmpeg", "sox", "libsndfile1", "espeak-ng"])
    require_commands(["ffmpeg", "sox", "espeak-ng"])

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
        "tqdm",
        *RESEMBLE_COMPAT_PACKAGES,
    ])

    if os.environ.get("SUPER_VOZ_ENABLE_RESEMBLE", "1") != "0":
        # --no-deps preserva o stack torch/torchaudio do Kaggle; os pins críticos do Resemble entram acima.
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-deps", "resemble-enhance"])
    else:
        print("[INFO] Pulando resemble-enhance no Kaggle porque SUPER_VOZ_ENABLE_RESEMBLE=0.")
    verify_audio_cleaning_dependencies(os.environ.get("SUPER_VOZ_ENABLE_RESEMBLE", "1") != "0")
    reexec_after_audio_cleaning_install()

def install_f5_tts_dependencies() -> None:
    packages = [
        "f5-tts",
        "accelerate",
        "safetensors",
        "num2words",
    ]
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *packages])
    require_python_modules(
        {
            "f5_tts": "f5-tts",
            "accelerate": "accelerate",
            "safetensors": "safetensors",
            "num2words": "num2words",
        }
    )
    install_ml_runtime_dependencies()

