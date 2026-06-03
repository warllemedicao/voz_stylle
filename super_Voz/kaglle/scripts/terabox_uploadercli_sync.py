#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_URL = "https://github.com/dnigamer/TeraboxUploaderCLI.git"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("$ " + " ".join(cmd))
    if cwd:
        print("cwd:", cwd)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def env_value(name: str, required: bool = True, default: str = "") -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"Secret/variavel {name} ausente.")
    return value


def ensure_tool(tool_dir: Path) -> None:
    if (tool_dir / "main.py").exists():
        run(["git", "-C", str(tool_dir), "pull"], check=False)
    else:
        tool_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", REPO_URL, str(tool_dir)])

    requirements = tool_dir / "requirements.txt"
    if requirements.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=False)


def ensure_downloader() -> None:
    try:
        import TeraboxDL  # noqa: F401
        return
    except ImportError:
        pass

    run([sys.executable, "-m", "pip", "install", "-q", "terabox-downloader==1.8"])


def write_configs(tool_dir: Path, local_dir: Path, remote_dir: str) -> None:
    secrets = {
        "jstoken": env_value("TERABOX_JS_TOKEN"),
        "cookies": {
            "csrfToken": env_value("TERABOX_CSRF_TOKEN"),
            "browserid": env_value("TERABOX_BROWSER_ID"),
            "lang": os.environ.get("TERABOX_LANG", "en"),
            "ndus": env_value("TERABOX_NDUS"),
            "ndut_fmt": env_value("TERABOX_NDUT_FMT"),
        },
    }
    settings = {
        "directories": {
            "sourcedir": str(local_dir),
            "remotedir": remote_dir,
            "uploadeddir": str(tool_dir / "_uploaded"),
        },
        "files": {
            "movefiles": "false",
            "deletesource": "false",
        },
        "encryption": {
            "enabled": "false",
            "encryptionkey": "",
        },
        "ignoredfiles": [],
        "appearance": {
            "showquota": "true",
        },
    }
    (tool_dir / "secrets.json").write_text(json.dumps(secrets, indent=2), encoding="utf-8")
    (tool_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")


def cookie_header() -> str:
    lang = os.environ.get("TERABOX_LANG", "en")
    ndus = env_value("TERABOX_NDUS")
    return f"lang={lang}; ndus={ndus};"


def candidate_share_urls(share_url: str, share_password: str) -> list[str]:
    urls = [share_url]
    if share_password and "pwd=" not in share_url:
        separator = "&" if "?" in share_url else "?"
        urls.append(f"{share_url}{separator}pwd={share_password}")
    return urls


def extract_archive_if_needed(path: Path, output_dir: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        print(f"[TeraBox] Extraindo ZIP restaurado: {path}")
        with zipfile.ZipFile(path) as archive:
            archive.extractall(output_dir)
        return

    if suffix in {".tar", ".gz", ".tgz", ".bz2", ".xz"}:
        print(f"[TeraBox] Extraindo arquivo restaurado: {path}")
        shutil.unpack_archive(str(path), str(output_dir))


def organize_downloaded_checkpoints(local_dir: Path) -> int:
    checkpoint_dir = local_dir / "Models" / "super_Voz"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for checkpoint in sorted(local_dir.rglob("epoch_2nd_*.pth")):
        if checkpoint_dir in checkpoint.parents:
            continue
        dst = checkpoint_dir / checkpoint.name
        if dst.exists() and dst.stat().st_size == checkpoint.stat().st_size:
            continue
        shutil.copy2(checkpoint, dst)
        copied += 1

    if copied:
        print(f"[TeraBox] Checkpoints copiados para {checkpoint_dir}: {copied}")
    return copied


def download_shared_link(local_dir: Path, share_url: str, share_password: str) -> None:
    ensure_downloader()
    from TeraboxDL import TeraboxDL

    local_dir.mkdir(parents=True, exist_ok=True)
    terabox = TeraboxDL(cookie_header())
    last_error = ""

    for url in candidate_share_urls(share_url, share_password):
        display_url = url.replace(share_password, "***") if share_password else url
        print(f"[TeraBox] Lendo link compartilhado: {display_url}")
        file_info = terabox.get_file_info(url)
        if isinstance(file_info, dict) and "error" in file_info:
            last_error = str(file_info["error"])
            print(f"[TeraBox][AVISO] Falha ao ler link: {last_error}")
            continue
        if not isinstance(file_info, dict):
            last_error = f"resposta inesperada: {type(file_info).__name__}"
            print(f"[TeraBox][AVISO] {last_error}")
            continue

        print(f"[TeraBox] Baixando: {file_info.get('file_name', 'arquivo remoto')}")
        result = terabox.download(file_info, save_path=str(local_dir))
        if isinstance(result, dict) and "error" in result:
            last_error = str(result["error"])
            print(f"[TeraBox][AVISO] Falha no download: {last_error}")
            continue
        if not isinstance(result, dict) or not result.get("file_path"):
            last_error = f"resultado de download inesperado: {result!r}"
            print(f"[TeraBox][AVISO] {last_error}")
            continue

        file_path = Path(result.get("file_path", "")).resolve()
        if file_path.is_file():
            print(f"[TeraBox] Download concluido: {file_path}")
            extract_archive_if_needed(file_path, local_dir)
            organize_downloaded_checkpoints(local_dir)
            return
        last_error = f"arquivo baixado nao encontrado: {file_path}"

    raise RuntimeError(f"Nao foi possivel baixar o link TeraBox. Ultimo erro: {last_error or 'desconhecido'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sincroniza pasta/checkpoints com TeraBox.")
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--tool-dir", default="/kaggle/working/TeraboxUploaderCLI")
    parser.add_argument("--share-url", default="")
    parser.add_argument("--share-password", default="")
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
    if args.action == "download":
        if not args.share_url:
            raise RuntimeError("--share-url e obrigatorio para download TeraBox.")
        download_shared_link(local_dir, args.share_url, args.share_password)
        return 0

    if not local_dir.exists():
        print(f"[TeraBox] Pasta local ausente, nada para enviar: {local_dir}")
        return 0

    if shutil.which("git") is None:
        raise RuntimeError("git ausente; nao foi possivel instalar TeraboxUploaderCLI.")

    tool_dir = Path(args.tool_dir).resolve()
    ensure_tool(tool_dir)
    write_configs(tool_dir, local_dir, args.remote_dir)
    run([sys.executable, "main.py"], cwd=tool_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
