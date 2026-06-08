#!/usr/bin/env python3
# ============================================================
# limpeza_ia.py — LIMPEZA DE ÁUDIO COM ANÁLISE INTELIGENTE (V9)
# Resemble Enhance GPU: inferência oficial por device="cuda" e tratamento por defeito principal
# ============================================================

import os
import subprocess
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
import librosa
import torch
import torchaudio
import warnings
import soundfile as sf

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO
# ============================================================

CACHE_ANÁLISE = "analise_audio_cache.json"
PROCESSADOS_LOG = "processados.json"
DNSMOS_MODEL_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"

# ============================================================
# UTILITÁRIOS
# ============================================================

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

def check_gpu_enhancer():
    """Verifica se o onnxruntime-gpu está realmente usando a GPU."""
    if not torch.cuda.is_available():
        print("[INFO] GPU não detectada via Torch. Resemble-enhance usará CPU (lento).")
        return

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' not in providers:
            print("[AVISO] onnxruntime-gpu não está usando GPU! Reinstalando...")
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu"], check=True)
            print("[OK] onnxruntime-gpu reinstalado.")
        else:
            print("[OK] Motor GPU (ONNX) confirmado.")
    except Exception as e:
        print(f"[AVISO] Falha ao verificar motor GPU: {e}")


def is_cuda_runtime_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = [
        "no kernel image is available",
        "cudaerrornokernelimagefordevice",
        "not compatible with the current pytorch installation",
        "expected all tensors to be on the same device",
        "cuda error",
    ]
    return any(marker in text for marker in markers)


def select_torch_device() -> str:
    if os.environ.get("SUPER_VOZ_FORCE_CPU", "0") == "1":
        print("[INFO] SUPER_VOZ_FORCE_CPU=1; Whisper/Resemble usarao CPU.")
        return "cpu"

    if not torch.cuda.is_available():
        print("[INFO] Torch CUDA indisponivel; Whisper/Resemble usarao CPU.")
        return "cpu"

    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        probe = torch.ones(1, device="cuda")
        probe = (probe + 1).detach().cpu()
        torch.cuda.synchronize()
        print(f"[OK] Torch CUDA operacional para {name} (sm_{major}{minor}).")
        return "cuda"
    except Exception as exc:
        print(f"[AVISO] Torch CUDA falhou em teste real: {exc}")
        print("[INFO] Whisper/Resemble usarao CPU para evitar falha cudaErrorNoKernelImageForDevice.")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return "cpu"


def load_whisper_safely(model_name: str, device: str):
    import whisper

    try:
        return whisper.load_model(model_name, device=device), device
    except Exception as exc:
        if device == "cuda" and is_cuda_runtime_error(exc):
            print(f"[AVISO] Whisper falhou na GPU: {exc}")
            print("[INFO] Recarregando Whisper em CPU.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return whisper.load_model(model_name, device="cpu"), "cpu"
        raise

# ============================================================
# DNSMOS: NOTA DE QUALIDADE (MÉTRICA DA MICROSOFT)
# ============================================================

class DNSMOS:
    def __init__(self, model_path=None):
        self.model_path = model_path or "dnsmos_model.onnx"
        self.session = None
        self._check_model()

    def _check_model(self):
        if not Path(self.model_path).exists():
            print(f"[INFO] Baixando modelo DNSMOS...")
            try:
                import urllib.request
                urllib.request.urlretrieve(DNSMOS_MODEL_URL, self.model_path)
            except Exception as e:
                print(f"[AVISO] Falha ao baixar DNSMOS: {e}")

        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' not in available_providers:
                providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception as e:
            print(f"[ERRO CRÍTICO] Motor DNSMOS falhou: {e}")

    def score(self, audio: np.ndarray, sr: int) -> dict:
        if self.session is None: return {"ovrl": 0.4, "sig": 0.4, "bak": 0.4} 
        try:
            if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            target_len = 144160
            if len(audio) < target_len: audio = np.pad(audio, (0, target_len - len(audio)))
            else: audio = audio[:target_len]
            audio_input = audio.astype(np.float32)[np.newaxis, :]
            inputs = {self.session.get_inputs()[0].name: audio_input}
            outputs = self.session.run(None, inputs)
            return {
                "sig": (outputs[0][0][0] - 1) / 4,
                "bak": (outputs[0][0][1] - 1) / 4,
                "ovrl": (outputs[0][0][2] - 1) / 4
            }
        except Exception as e:
            return {"ovrl": 0.5, "sig": 0.5, "bak": 0.5}

# ============================================================
# AUDIO ENHANCER: RESEMBLE ENHANCE (SOLUÇÃO V9)
# ============================================================

class AudioEnhancer:
    def __init__(self, enabled: bool = True, device_str: str | None = None):
        self.device_str = device_str or select_torch_device()
        self.device_obj = torch.device(self.device_str)
        self.has_resemble = False
        self._warmup_done = False
        if not enabled:
            print("[INFO] Resemble Enhance desativado; usando limpeza deterministica.")
            return
        try:
            import resemble_enhance
            self.has_resemble = True
            print(f"[INFO] Motor de restauração V9 detectado no device: {self.device_str}")
        except Exception as e:
            print(f"[AVISO] resemble-enhance indisponível: {e}")

    def _warmup(self):
        """Carrega o modelo no device alvo sem rodar dummy inference fora do fluxo oficial."""
        if not self.has_resemble or self._warmup_done or self.device_str == "cpu":
            return

        print(f"[INFO] Carregando Resemble Enhance na GPU ({self.device_str})...")
        try:
            from resemble_enhance.enhancer.inference import load_enhancer

            try:
                load_enhancer.cache_clear()
            except: pass

            with torch.inference_mode():
                load_enhancer(None, self.device_str)

            self._warmup_done = True
            print("[INFO] Resemble Enhance pronto na GPU.")
        except Exception as e:
            print(f"[AVISO] Falha ao carregar Resemble na GPU: {e}")
            print("[INFO] O áudio será preservado se a inferência falhar.")

    def process(self, input_path: Path, output_path: Path, defect: str = "degraded_voice"):
        if not self.has_resemble: return False

        if not self._warmup_done:
            self._warmup()

        try:
            from resemble_enhance.enhancer.inference import denoise, enhance

            # 1. Carregar áudio
            dwav, sr = torchaudio.load(str(input_path))

            # 2. A API oficial espera waveform 1D. Mantemos em CPU e passamos device="cuda";
            # a inferência interna faz resample, chunking e move cada chunk para a GPU.
            if dwav.shape[0] > 1:
                dwav = dwav.mean(dim=0)
            else:
                dwav = dwav.squeeze(0)
            dwav = dwav.cpu().to(torch.float32)

            treatment = self._select_treatment(defect)
            print(f"  [RESEMBLE] Defeito principal: {defect} | Tratamento unico: {treatment}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.inference_mode():
                try:
                    hwav, new_sr = self._run_resemble(denoise, enhance, dwav, sr, treatment, self.device_str)
                except Exception as gpu_err:
                    if self.device_str == "cuda" and is_cuda_runtime_error(gpu_err):
                        print(f"  [AVISO] Falha CUDA em {input_path.name}. Usando Fallback CPU...")
                        hwav, new_sr = self._run_resemble(denoise, enhance, dwav, sr, treatment, "cpu")
                    else:
                        raise gpu_err
            
            # 6. Salvar resultado
            audio_out = hwav.cpu().numpy()
            sf.write(str(output_path), audio_out, new_sr)
            if not self._validar_saida(input_path, output_path):
                print(f"  [AVISO] Saida do enhancer reprovada em {input_path.name}. Preservando original.")
                return False
            return True
        except Exception as e:
            print(f"  [ERRO ENHANCER] {e}")
            return False

    def _select_treatment(self, defect: str) -> str:
        if defect in {"hissing", "background_noise"}:
            return "denoise"
        return "enhance"

    def _run_resemble(self, denoise_fn, enhance_fn, dwav, sr, treatment: str, device: str):
        if treatment == "denoise":
            return denoise_fn(dwav, sr, device=device)
        return enhance_fn(dwav, sr, device=device, nfe=32, solver="midpoint", lambd=0.5)

    def _validar_saida(self, input_path: Path, output_path: Path) -> bool:
        """Evita aceitar audio vazio, distorcido ou com duracao muito diferente."""
        try:
            y_in, sr_in = librosa.load(str(input_path), sr=None, mono=True)
            y_out, sr_out = librosa.load(str(output_path), sr=None, mono=True)
            if y_in.size == 0 or y_out.size == 0:
                return False
            if not np.all(np.isfinite(y_out)):
                return False

            dur_in = y_in.size / float(sr_in)
            dur_out = y_out.size / float(sr_out)
            if dur_in <= 0 or dur_out <= 0:
                return False
            ratio = dur_out / dur_in
            if ratio < 0.70 or ratio > 1.30:
                return False

            rms_in = float(np.sqrt(np.mean(np.square(y_in))) + 1e-9)
            rms_out = float(np.sqrt(np.mean(np.square(y_out))) + 1e-9)
            if rms_out < 1e-5 or rms_out / rms_in < 0.10 or rms_out / rms_in > 8.0:
                return False
            if float(np.max(np.abs(y_out))) > 1.25:
                return False
            return True
        except Exception as e:
            print(f"  [AVISO] Falha ao validar saida do enhancer: {e}")
            return False

# ============================================================
# CLASSE DE ANÁLISE DE ÁUDIO
# ============================================================

class AudioAnalyzer:
    def __init__(self, sr: int = 24000):
        self.sr = sr
        self.mos_tool = DNSMOS()

    def analyze(self, audio_path: str, verbose: bool = False) -> dict:
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        except Exception as e:
            return {"status": "erro", "erro": str(e), "processamento_necessario": True, "score_geral": 0}

        audio = audio.astype(np.float32)
        mos_scores = self.mos_tool.score(audio, sr)
        
        # Heurísticas rápidas
        D = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
        freq_bins = np.fft.rfftfreq(1024, 1/sr)
        hissing_idx = freq_bins > 8000
        razao_hissing = (np.mean(np.abs(D[hissing_idx, :])) / np.mean(np.abs(D))) if np.any(hissing_idx) else 0
        hissing_heu = min(razao_hissing * 5, 1.0)
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=1024, hop_length=256)[0]
        background_noise = min(float(np.mean(flatness)) * 2.5, 1.0)

        problemas = []
        defect_scores = {
            "degraded_voice": max(0.0, 0.6 - float(mos_scores["ovrl"])) / 0.6,
            "background_noise": background_noise,
            "hissing": float(hissing_heu),
        }
        if mos_scores['ovrl'] < 0.6: problemas.append(f"Voz degradada (Nota IA: {mos_scores['ovrl']*5:.1f}/5.0)")
        if background_noise > 0.45: problemas.append("Ruído de fundo detectado")
        if hissing_heu > 0.5: problemas.append("Chiado agudo detectado")

        score_geral = mos_scores['ovrl'] * 0.7 + (1.0 - hissing_heu) * 0.15 + (1.0 - background_noise) * 0.15
        processamento_necessario = len(problemas) > 0 or score_geral < 0.75
        defect_principal = "none"
        if processamento_necessario:
            defect_principal = max(defect_scores, key=defect_scores.get)

        resultado = {
            "status": "sucesso", "audio_path": str(audio_path),
            "processamento_necessario": processamento_necessario,
            "defeito_principal": defect_principal,
            "problemas": problemas, "score_geral": round(score_geral, 3),
            "scores_detalhados": {"dnsmos_ovrl": mos_scores['ovrl'], **defect_scores}
        }
        if verbose: self._imprimir_resultado(resultado)
        return convert_numpy_types(resultado)

    def _imprimir_resultado(self, r: dict):
        print(f"\n--- QUALIDADE: {Path(r['audio_path']).name} ---")
        print(f"Score Geral: {r['score_geral']:.1%}")
        print(f"Defeito principal: {r.get('defeito_principal', 'none')}")
        for p in r['problemas']: print(f"  ❌ {p}")
        if r['processamento_necessario']: print("  ⚡ AÇÃO: Restaurando...")
        else: print("  ✅ AÇÃO: Preservando original.")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ambiente", type=str, choices=["colab", "kaggle", "local"], default="local")
    parser.add_argument("--enhancer", type=str, choices=["auto", "off", "resemble"], default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    print(f"[INFO] Ambiente detectado: {args.ambiente}")
    if args.ambiente in ["colab", "kaggle"]:
        check_gpu_enhancer()
    torch_device = select_torch_device()

    analyzer = AudioAnalyzer()
    enhancer_enabled = args.enhancer == "resemble"
    if args.enhancer == "auto":
        enhancer_enabled = os.environ.get("SUPER_VOZ_ENABLE_RESEMBLE", "1") != "0"
        if args.ambiente in ["colab", "kaggle"] and not enhancer_enabled:
            print("[INFO] Resemble Enhance em modo auto foi desativado neste ambiente.")
            print("[INFO] Motivo: SUPER_VOZ_ENABLE_RESEMBLE=0.")
    enhancer = AudioEnhancer(enabled=enhancer_enabled, device_str=torch_device)
    
    print("[INFO] Carregando Whisper...")
    model, whisper_device = load_whisper_safely("medium", torch_device)

    audio_files = sorted(list(input_dir.glob("*.wav")) + list(input_dir.glob("*.mp3")))
    print(f"\n🚀 Processando {len(audio_files)} arquivos...")
    
    metadata = []
    for idx, audio_path in enumerate(audio_files):
        print(f"[{idx+1}/{len(audio_files)}] {audio_path.name}")
        info = analyzer.analyze(str(audio_path), verbose=True)
        
        file_id = f"voz_{idx:04d}_{audio_path.stem}"
        final_wav = output_dir / f"{file_id}.wav"

        if info["processamento_necessario"]:
            if not enhancer.process(audio_path, final_wav, defect=info.get("defeito_principal", "degraded_voice")):
                print(f"  [INFO] Usando original para {audio_path.name} antes da padronizacao final.")
                shutil.copy2(audio_path, final_wav)
        else:
            shutil.copy2(audio_path, final_wav)

        # Padronização final para StyleTTS2
        try:
            y, sr = librosa.load(str(final_wav), sr=24000, mono=True)
            y = librosa.util.normalize(librosa.effects.trim(y, top_db=25)[0]) * 0.95
            sf.write(str(final_wav), y, 24000, subtype='PCM_16')
            
            print(f"  [WHISPER] Transcrevendo...")
            res = model.transcribe(str(final_wav), language="pt", fp16=(whisper_device == "cuda"))
            text = res["text"].strip()
            print(f"  [TEXTO] {text}")
            if text: metadata.append(f"{file_id}|{text}|{text}")
        except Exception as e:
            print(f"  [ERRO FINAL] {e}")

    with open("train.txt", "w", encoding="utf-8") as f: f.write("\n".join(metadata))
    print(f"✅ Dataset pronto em: {output_dir}")

if __name__ == "__main__":
    main()
