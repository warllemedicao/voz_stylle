import os
import subprocess
from pathlib import Path
import sys

# 🛠️ 1. Configurar Repositório e Ambiente
REPO_URL = "https://github.com/warllemedicao/Voz_styllett2.git"
PROJECT_ROOT = Path("/kaggle/working/Super_voz")

if PROJECT_ROOT.exists():
    print("--- Atualizando Repositório ---")
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "pull"], check=False)
else:
    print("--- Clonando Repositório ---")
    subprocess.run(["git", "clone", REPO_URL, str(PROJECT_ROOT)], check=True)

# Localiza o subdiretório do projeto
PROJECT_DIR = PROJECT_ROOT / "super_Voz"
os.chdir(PROJECT_DIR)

print(f"✅ Pronto! Diretório atual: {os.getcwd()}")

# 🚀 2. Iniciar Pipeline
subprocess.run([sys.executable, "scripts/run_kaggle_styletts2.py", "--config", "styletts2_kaggle_config.yml"], check=True)
