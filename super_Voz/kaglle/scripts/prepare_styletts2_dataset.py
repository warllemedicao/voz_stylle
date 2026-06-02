#!/usr/bin/env python3
import argparse
import csv
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.replace("|", " ")


def read_metadata(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if not path.exists():
        raise FileNotFoundError(f"Metadata nao encontrado: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "|" if "|" in sample else ","
        reader = csv.reader(f, delimiter=delimiter)
        for raw in reader:
            if not raw or len(raw) < 2:
                continue
            file_id = raw[0].strip()
            text = clean_text(raw[1] if len(raw) == 2 else raw[1])
            if file_id and text and text.upper() != "VAZIO":
                rows.append((file_id, text))
    return rows


def resolve_audio(input_dir: Path, file_id: str) -> Path | None:
    candidate = Path(file_id)
    names = [candidate.name]
    if candidate.suffix.lower() not in AUDIO_EXTS:
        names.extend(f"{candidate.name}{ext}" for ext in AUDIO_EXTS)

    for name in names:
        path = input_dir / name
        if path.exists():
            return path
    return None


def audio_duration_seconds(path: Path) -> float | None:
    """Retorna duração do áudio. Se falhar, retorna um valor alto para forçar o bloqueio do item."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        return float(result.stdout.strip())
    except Exception:
        # Fallback para soundfile se ffprobe falhar
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return info.duration
        except Exception:
            # Se tudo falhar, retorna um valor absurdo para garantir que seja filtrado pelo max_seconds
            print(f"[AVISO] Falha ao ler duração de {path.name}. O arquivo será ignorado por segurança.")
            return 9999.0


def convert_audio(src: Path, dst: Path, sr: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1",
        "-ar", str(sr),
        "-sample_fmt", "s16",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def phonemize_texts(texts: list[str], language: str) -> list[str]:
    try:
        from phonemizer import phonemize
    except Exception as exc:
        print(f"[AVISO] phonemizer indisponivel, usando texto cru: {exc}")
        return texts

    try:
        return [
            clean_text(item)
            for item in phonemize(
                texts,
                language=language,
                backend="espeak",
                strip=True,
                preserve_punctuation=True,
                with_stress=True,
                njobs=1,
            )
        ]
    except Exception as exc:
        print(f"[AVISO] falha na fonemizacao '{language}', usando texto cru: {exc}")
        return texts


def write_list(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for filename, text, speaker in rows:
            f.write(f"{filename}|{text}|{speaker}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara dataset para StyleTTS2.")
    parser.add_argument("--input_dir", required=True, help="Pasta com WAVs e train.txt/metadata.csv.")
    parser.add_argument("--output_dir", required=True, help="Pasta de saida do dataset StyleTTS2.")
    parser.add_argument("--metadata", default=None, help="Arquivo train.txt ou metadata.csv.")
    parser.add_argument("--speaker", default="0", help="ID/nome do speaker.")
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--phonemize", action="store_true")
    parser.add_argument("--phonemizer_language", default="pt-br")
    parser.add_argument("--max_seconds", type=float, default=12.0)
    parser.add_argument("--max_text_chars", type=int, default=280)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    wav_dir = output_dir / "wavs"
    data_dir = output_dir / "Data"

    metadata = Path(args.metadata).resolve() if args.metadata else input_dir / "train.txt"
    if not metadata.exists():
        alt = input_dir / "metadata.csv"
        metadata = alt if alt.exists() else metadata

    rows = read_metadata(metadata)
    if not rows:
        raise RuntimeError(f"Nenhuma linha valida encontrada em {metadata}")

    prepared: list[tuple[str, str, str]] = []
    missing: list[str] = []
    skipped_long_audio: list[str] = []
    skipped_long_text: list[str] = []

    for idx, (file_id, text) in enumerate(rows):
        src = resolve_audio(input_dir, file_id)
        if src is None:
            missing.append(file_id)
            continue

        if args.max_text_chars and len(text) > args.max_text_chars:
            skipped_long_text.append(f"{file_id}|chars={len(text)}")
            continue

        duration = audio_duration_seconds(src)
        if duration is not None and args.max_seconds and duration > args.max_seconds:
            skipped_long_audio.append(f"{file_id}|seconds={duration:.2f}")
            continue

        dst_name = f"{idx:05d}_{src.stem}.wav"
        convert_audio(src, wav_dir / dst_name, args.sample_rate)
        prepared.append((dst_name, text, args.speaker))

    if not prepared:
        raise RuntimeError(
            "Nenhum audio foi preparado para StyleTTS2. "
            "Aumente max_audio_seconds/max_text_chars ou revise train.txt."
        )

    if args.phonemize:
        phonemes = phonemize_texts([row[1] for row in prepared], args.phonemizer_language)
        prepared = [(row[0], phonemes[idx], row[2]) for idx, row in enumerate(prepared)]

    random.seed(args.seed)
    random.shuffle(prepared)

    val_count = max(1, int(len(prepared) * args.val_ratio)) if len(prepared) > 1 else 0
    val_rows = prepared[:val_count]
    train_rows = prepared[val_count:]
    if not train_rows:
        train_rows, val_rows = prepared, []

    write_list(data_dir / "train_list.txt", train_rows)
    write_list(data_dir / "val_list.txt", val_rows or train_rows[:1])
    write_list(data_dir / "all_list.txt", prepared)
    (data_dir / "OOD_texts.txt").write_text(
        "\n".join(f"{text}|ood" for _, text, _ in prepared),
        encoding="utf-8",
    )

    report = [
        f"metadata={metadata}",
        f"input_dir={input_dir}",
        f"output_dir={output_dir}",
        f"sample_rate={args.sample_rate}",
        f"prepared={len(prepared)}",
        f"train={len(train_rows)}",
        f"val={len(val_rows or train_rows[:1])}",
        f"missing={len(missing)}",
        f"skipped_long_audio={len(skipped_long_audio)}",
        f"skipped_long_text={len(skipped_long_text)}",
        f"max_seconds={args.max_seconds}",
        f"max_text_chars={args.max_text_chars}",
        f"phonemize={args.phonemize}",
    ]
    (output_dir / "prepare_report.txt").write_text("\n".join(report), encoding="utf-8")

    if missing:
        (output_dir / "missing_audio.txt").write_text("\n".join(missing), encoding="utf-8")
        print(f"[AVISO] {len(missing)} audio(s) citados no metadata nao foram encontrados.")
    if skipped_long_audio:
        (output_dir / "skipped_long_audio.txt").write_text("\n".join(skipped_long_audio), encoding="utf-8")
        print(f"[AVISO] {len(skipped_long_audio)} audio(s) longos foram ignorados para evitar OOM.")
    if skipped_long_text:
        (output_dir / "skipped_long_text.txt").write_text("\n".join(skipped_long_text), encoding="utf-8")
        print(f"[AVISO] {len(skipped_long_text)} texto(s) longos foram ignorados para evitar OOM.")

    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
