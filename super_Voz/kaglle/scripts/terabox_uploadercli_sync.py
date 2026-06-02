#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload de pasta para TeraBox via TeraboxUploaderCLI.")
    parser.add_argument("action", choices=["upload"])
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--tool-dir", default="/kaggle/working/TeraboxUploaderCLI")
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
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
