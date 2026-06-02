#!/usr/bin/env python3
# ============================================================
# audio_analyzer.py — ANÁLISE INTELIGENTE DE ÁUDIO
# Detecta: ruído, hissing, sons musicais, silêncios
# ============================================================

import numpy as np
import librosa
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

def convert_numpy_types(obj: Any) -> Any:
    """
    Converte tipos numpy para tipos Python nativos para JSON serialization.
    """
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class AudioAnalyzer:
    """
    Analisa áudio para determinar se precisa de processamento.
    Detecta:
    - Ruído (frequências aleatórias)
    - Hissing (frequências altas acima de 8kHz)
    - Sons musicais (harmônicos bem definidos)
    - Silêncios significativos
    """
    
    def __init__(self, sr: int = 22050, hop_length: int = 256):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = 1024
        
    def analyze(self, audio_path: str, verbose: bool = False) -> Dict:
        """
        Análise completa de um arquivo de áudio.
        Retorna dicionário com diagnóstico.
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        except Exception as e:
            if verbose:
                print(f"[ERRO] Não foi possível carregar {audio_path}: {e}")
            return {
                "status": "erro",
                "erro": str(e),
                "processamento_necessario": True,
                "problemas": ["Erro ao carregar áudio"],
                "score": 0
            }
        
        audio = audio.astype(np.float32)
        
        # Computar spectrograma
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=80,
            power=2.0
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Computar STFT
        D = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # Extrair features
        problemas = []
        scores = {}
        
        # 1. Detecção de Silêncio
        silencio_score = self._detectar_silencio(audio)
        scores['silencio'] = silencio_score
        if silencio_score > 0.5:
            problemas.append("Silêncio significativo detectado")
        
        # 2. Detecção de Ruído
        ruido_score = self._detectar_ruido(mel_db)
        scores['ruido'] = ruido_score
        if ruido_score > 0.6:
            problemas.append("Ruído detectado")
        
        # 3. Detecção de Hissing (assobio)
        hissing_score = self._detectar_hissing(D, sr)
        scores['hissing'] = hissing_score
        if hissing_score > 0.5:
            problemas.append("Hissing (assobio) detectado")
        
        # 4. Detecção de Sons Musicais
        musical_score = self._detectar_musical(mel_db, D, sr)
        scores['musical'] = musical_score
        if musical_score > 0.6:
            problemas.append("Sons musicais detectados")
        
        # 5. Razão de Clipping
        clipping_score = self._detectar_clipping(audio)
        scores['clipping'] = clipping_score
        if clipping_score > 0.1:
            problemas.append("Clipping detectado")
        
        # Score geral de qualidade (0-1, onde 1 é perfeito)
        score_geral = 1.0 - np.mean([
            silencio_score * 0.1,
            ruido_score * 0.3,
            hissing_score * 0.2,
            musical_score * 0.2,
            clipping_score * 0.2
        ])
        
        # Determinar se processamento é necessário
        processamento_necessario = len(problemas) > 0 or score_geral < 0.7
        
        resultado = {
            "status": "sucesso",
            "audio_path": str(audio_path),
            "duracao_segundos": len(audio) / sr,
            "sample_rate": sr,
            "processamento_necessario": processamento_necessario,
            "problemas": problemas,
            "score_geral": round(score_geral, 3),
            "scores_detalhados": {k: round(v, 3) for k, v in scores.items()}
        }
        
        # Converter tipos numpy para JSON serialization
        resultado = convert_numpy_types(resultado)
        
        if verbose:
            self._imprimir_resultado(resultado)
        
        return resultado
    
    def _detectar_silencio(self, audio: np.ndarray) -> float:
        """Detecta porcentagem de silêncio no áudio."""
        # Define silêncio como amplitude < 0.01
        threshold = 0.01
        silencio = np.abs(audio) < threshold
        porcentagem_silencio = np.mean(silencio)
        
        # Score de 0 a 1, onde 1 significa 100% de silêncio
        return min(porcentagem_silencio, 1.0)
    
    def _detectar_ruido(self, mel_db: np.ndarray) -> float:
        """
        Detecta ruído usando análise espectral.
        Ruído tem distribuição plana no espectro.
        """
        # Calcular a variância espectral ao longo do tempo
        # Se é constante, provavelmente é ruído
        media_espectral = np.mean(mel_db, axis=1)
        desvio_espectral = np.std(media_espectral)
        
        # Ruído branco tem desvio baixo (espectro plano)
        # Fala tem picos em bandas específicas
        # Normalizar para 0-1
        ruido_score = max(0, 1 - (desvio_espectral / 10.0))
        
        return min(ruido_score, 1.0)
    
    def _detectar_hissing(self, D: np.ndarray, sr: int) -> float:
        """
        Detecta hissing (assobio) em frequências altas (>8kHz).
        Hissing é muito concentrado em frequências altas.
        """
        # Converter para escala de frequência
        freq_bins = np.fft.rfftfreq(self.n_fft, 1/sr)
        
        # Frequências acima de 8kHz
        hissing_threshold = 8000
        hissing_idx = freq_bins > hissing_threshold
        
        if not np.any(hissing_idx):
            return 0.0
        
        # Energia em frequências altas vs total
        energia_hissing = np.mean(np.abs(D[hissing_idx, :]))
        energia_total = np.mean(np.abs(D))
        
        if energia_total == 0:
            return 0.0
        
        razao_hissing = energia_hissing / energia_total
        
        # Score: quanto mais energia em altas frequências, mais hissing
        # Esperamos menos de 5% de energia em altas frequências
        return min(razao_hissing * 5, 1.0)
    
    def _detectar_musical(self, mel_db: np.ndarray, D: np.ndarray, sr: int) -> float:
        """
        Detecta sons musicais (harmônicos bem definidos).
        Música tem picos espectrais claros.
        """
        # Calcular crista do espectro (razão entre pico e média)
        media_tempo = np.mean(mel_db, axis=1)
        picos = np.max(mel_db, axis=1)
        
        # Razão de pico
        razao_pico = np.mean(picos) - np.mean(media_tempo)
        
        # Música tem razão de pico mais alta
        # Normalizar (esperamos ~10-20dB de razão)
        musical_score = max(0, (razao_pico / 20.0))
        
        # Também verificar periodicidade do espectro (harmônicos)
        diferenca_freq = np.abs(np.diff(np.mean(D, axis=1)))
        periodicidade = np.std(diferenca_freq)
        
        # Harmônicos bem definidos têm periodicidade alta
        harmonica_score = max(0, (periodicidade / 100.0))
        
        return min((musical_score + harmonica_score) / 2, 1.0)
    
    def _detectar_clipping(self, audio: np.ndarray) -> float:
        """Detecta clipping (distorção por saturação)."""
        # Valores muito próximos de 1.0 ou -1.0 indicam clipping
        clipping_threshold = 0.99
        clipped = np.abs(audio) > clipping_threshold
        razao_clipping = np.mean(clipped)
        
        return min(razao_clipping * 10, 1.0)
    
    def _imprimir_resultado(self, resultado: Dict):
        """Imprime resultado da análise de forma legível."""
        print("\n" + "="*60)
        print("ANÁLISE DE ÁUDIO")
        print("="*60)
        
        print(f"Arquivo: {Path(resultado.get('audio_path', 'desconhecido')).name}")
        print(f"Duração: {resultado.get('duracao_segundos', 0):.2f}s")
        print(f"Taxa de amostragem: {resultado.get('sample_rate', 0)} Hz")
        print(f"Score geral: {resultado.get('score_geral', 0):.1%}")
        
        print("\nProblemas detectados:")
        if resultado.get('problemas'):
            for problema in resultado['problemas']:
                print(f"  ⚠️  {problema}")
        else:
            print("  ✅ Nenhum problema detectado")
        
        print("\nScores detalhados:")
        for chave, valor in resultado.get('scores_detalhados', {}).items():
            barra = "█" * int(valor * 20) + "░" * (20 - int(valor * 20))
            print(f"  {chave:12s}: [{barra}] {valor:.1%}")
        
        if resultado.get('processamento_necessario'):
            print("\n⚡ Processamento NECESSÁRIO")
        else:
            print("\n✅ Áudio de qualidade aceitável (pode pular processamento)")
        
        print("="*60 + "\n")
    
    def analisar_batch(self, audio_files: List[str], cache_path: str = None) -> Dict[str, Dict]:
        """
        Analisa múltiplos arquivos com cache opcional.
        """
        resultados = {}
        cache = {}
        
        if cache_path and Path(cache_path).exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        
        for audio_path in audio_files:
            chave = str(Path(audio_path).stat().st_mtime)  # Use mtime como cache key
            
            if chave in cache:
                resultados[str(audio_path)] = cache[chave]
            else:
                resultado = self.analyze(audio_path, verbose=True)
                resultados[str(audio_path)] = resultado
                cache[chave] = resultado
        
        # Salvar cache
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            # Converter tipos numpy antes de salvar
            cache_convertido = {k: convert_numpy_types(v) for k, v in cache.items()}
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_convertido, f, ensure_ascii=False, indent=2)
        
        return resultados


def analisar_audio_simples(audio_path: str) -> Dict:
    """Função simples para análise rápida."""
    analyzer = AudioAnalyzer()
    return analyzer.analyze(audio_path, verbose=True)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python audio_analyzer.py <arquivo_audio> [<arquivo2> ...]")
        sys.exit(1)
    
    analyzer = AudioAnalyzer()
    
    for audio_path in sys.argv[1:]:
        if Path(audio_path).exists():
            analyzer.analyze(audio_path, verbose=True)
        else:
            print(f"[ERRO] Arquivo não encontrado: {audio_path}")
