import os
import shutil
import subprocess
from pathlib import Path
import sys

# 🛠️ 1. Configurar Repositório e Ambiente
REPO_URL = "https://github.com/warllemedicao/voz_stylle.git"
PROJECT_ROOT = Path("/kaggle/working/Super_voz")
RUNNER_RELATIVE_PATH = Path("super_Voz") / "kaglle" / "scripts" / "run_kaggle_styletts2.py"
os.environ["SUPER_VOZ_ENABLE_RESEMBLE"] = "1"


def has_kaggle_runner(repo_root: Path) -> bool:
    return (repo_root / RUNNER_RELATIVE_PATH).exists()


def find_kaggle_input_repo() -> Path | None:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None
    for runner in input_root.rglob(str(RUNNER_RELATIVE_PATH)):
        return runner.parents[3]
    return None


def setup_project_root() -> Path:
    if PROJECT_ROOT.exists():
        print("--- Atualizando Repositório ---")
        subprocess.run(["git", "-C", str(PROJECT_ROOT), "remote", "set-url", "origin", REPO_URL], check=False)
        fetch = subprocess.run(["git", "-C", str(PROJECT_ROOT), "fetch", "--all"], check=False)
        if fetch.returncode == 0:
            subprocess.run(["git", "-C", str(PROJECT_ROOT), "checkout", "main"], check=False)
            subprocess.run(["git", "-C", str(PROJECT_ROOT), "reset", "--hard", "origin/main"], check=False)
        else:
            print("[AVISO] GitHub indisponivel; usando clone local existente se estiver valido.")
        if has_kaggle_runner(PROJECT_ROOT):
            return PROJECT_ROOT
    else:
        print("--- Clonando Repositório ---")
        try:
            subprocess.run(["git", "clone", REPO_URL, str(PROJECT_ROOT)], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[AVISO] Clone via GitHub falhou ({exc}). Tentando Kaggle Input local.")
        if has_kaggle_runner(PROJECT_ROOT):
            return PROJECT_ROOT

    input_repo = find_kaggle_input_repo()
    if input_repo:
        print(f"--- Usando copia anexada em Kaggle Input: {input_repo} ---")
        shutil.copytree(input_repo, PROJECT_ROOT, dirs_exist_ok=True)
        if has_kaggle_runner(PROJECT_ROOT):
            return PROJECT_ROOT

    raise RuntimeError(
        "Nao foi possivel obter o codigo do super_Voz. "
        "Ative Internet no Kaggle para clonar o GitHub ou anexe um Kaggle Dataset "
        "contendo super_Voz/kaglle/scripts/run_kaggle_styletts2.py."
    )

PROJECT_ROOT = setup_project_root()

# Localiza o subdiretório Kaggle do projeto
PROJECT_DIR = PROJECT_ROOT / "super_Voz" / "kaglle"
if not (PROJECT_DIR / "scripts" / "run_kaggle_styletts2.py").exists():
    raise FileNotFoundError(
        f"Runner Kaggle nao encontrado em {PROJECT_DIR}. "
        f"Verifique se {REPO_URL} foi clonado/atualizado corretamente."
    )
os.chdir(PROJECT_DIR)

print(f"✅ Pronto! Diretório atual: {os.getcwd()}")

# 🚀 2. Iniciar Pipeline
subprocess.run([sys.executable, "scripts/run_kaggle_styletts2.py", "--config", "styletts2_kaggle_config.yml"], check=True)
