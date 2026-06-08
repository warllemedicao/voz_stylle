# Registro de Alterações: limpeza_ia.py

## [2026-06-08] Dependencias da limpeza no modo Kaggle F5-TTS PT-BR
- Diagnostico: no modo `tts_engine: "f5_tts_ptbr"`, o runner pulava o instalador legado `install_dependencies(style_dir)`, mas ainda chamava `limpeza_ia.py`.
- Sintomas no Kaggle: `No module named 'onnxruntime'`, `No module named 'resemble_enhance'` e `ModuleNotFoundError: No module named 'whisper'` logo apos `Audios brutos importados do R2`.
- A causa raiz era instalacao ausente das dependencias da limpeza/transcricao, nao erro no R2 nem nos audios.
- `scripts/run_kaggle_styletts2.py` agora possui `install_audio_cleaning_dependencies()` e chama esse bloco tambem no ramo `f5_tts_ptbr` antes de iniciar a Limpeza IA.
- Para diagnosticos futuros, conferir se o log mostra `--- Instalando Dependências da Limpeza IA ---` antes de `[INFO] Iniciando Limpeza IA`.

## [2026-06-08] Fallback Torch/Whisper para P100 incompatível
- Sintoma no Kaggle P100: Whisper falhava em `whisper.load_model("medium")` com `CUDA error: no kernel image is available for execution on the device`.
- A causa era PyTorch ativo sem kernel compativel com `sm_60`; `torch.cuda.is_available()` ainda retornava verdadeiro, entao Whisper tentava carregar na GPU e abortava.
- `limpeza_ia.py` agora roda um teste CUDA real com tensor pequeno antes de escolher o device de Whisper/Resemble.
- Se o teste ou o carregamento do Whisper falhar por erro CUDA de runtime, a limpeza recarrega Whisper em CPU e chama `transcribe(..., fp16=False)`.
- O `AudioEnhancer` recebe o mesmo device seguro. Se Resemble falhar em CUDA por erro de runtime, tenta fallback CPU para aquele arquivo.

## [2026-06-05] Observacao do runner Kaggle
- A limpeza de audio continua independente da retencao de checkpoints, mas o fluxo Kaggle foi ajustado para evitar falso sucesso de treino.
- O runner mantem o checkpoint mais recente em `Models/super_Voz` apos upload e remove apenas checkpoints anteriores quando um checkpoint mais novo ja foi persistido.
- Em caso de falha no `accelerate`, a sincronizacao final pode ocorrer para recuperacao, mas a mensagem final passa a indicar interrupcao/falha do treino.

## [2026-05-31]
- Inicialização do registro de alterações.
- Adicionado suporte ao argumento `--ambiente` (colab, kaggle, local).
- Implementação da função `check_gpu_enhancer()` para forçar reinstalação do `onnxruntime-gpu` caso a GPU não seja detectada pelo motor ONNX.
- Adicionada função `convert_numpy_types()` para garantir serialização JSON correta.
- **Versão V8:** Atualização do `AudioEnhancer` com carregamento explícito de modelos (`load_enhancer`, `load_denoiser`), uso de `torch.inference_mode()`, limpeza de cache e fallback para CPU em caso de erro de device persistente.
- **Melhoria de Visibilidade:** Adicionados logs detalhados para o processo de transcrição (Whisper), permitindo acompanhar o texto gerado em tempo real no console.
- **Integridade de Código:** Removidas duplicações de funções e classes que surgiram durante o merge de versões, garantindo um script limpo e eficiente.

## [2026-05-31] Ajuste de compatibilidade Colab/Kaggle
- Diagnóstico: o erro `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` vem do `resemble-enhance`/PyTorch, não do `onnxruntime-gpu`. No Colab, a instalação `resemble-enhance --no-deps` evita downgrade do PyTorch, mas deixa a biblioteca sujeita a incompatibilidade interna de device.
- O `resemble-enhance` agora fica desativado por padrão em `--ambiente colab` e `--ambiente kaggle` quando `--enhancer auto` é usado.
- Para teste manual, ainda é possível forçar o enhancer com `--enhancer resemble` ou definir `SUPER_VOZ_ENABLE_RESEMBLE=1` antes de rodar o pipeline.
- Adicionada validação da saída do enhancer: áudio vazio, não finito, com duração muito diferente, volume anormal ou pico excessivo é rejeitado. Nesses casos, o script preserva o original e aplica apenas a padronização final para StyleTTS2.
- A limpeza segura continua garantindo 24 kHz, mono, PCM 16-bit, trim de silêncio e normalização antes da transcrição e geração do `train.txt`.

## [2026-05-31] Integração com notebook Colab (V8, superada)
- O notebook `run_colab_super_voz.ipynb` passou temporariamente a definir `SUPER_VOZ_ENABLE_RESEMBLE=0` por padrão, mantendo o modo `--enhancer auto` em rota segura.
- Essa política foi superada pela V9, que reativou Resemble Enhance em GPU por padrão com inferência corrigida.

## [2026-05-31] Versão V9 - Resemble Enhance GPU padrão
- Auphonic API foi descartado por ser serviço pago.
- `--enhancer auto` agora habilita Resemble Enhance por padrão, exceto quando `SUPER_VOZ_ENABLE_RESEMBLE=0`.
- A integração com Resemble foi ajustada para seguir o fluxo oficial: waveform mono 1D permanece em CPU e a chamada recebe `device="cuda"`, deixando a biblioteca fazer resample, chunking e movimentação interna para GPU.
- Removido o warm-up com dummy tensor em GPU, que podia induzir mismatch CPU/CUDA fora da inferência oficial.
- O analisador agora informa `defeito_principal` por áudio.
- Tratamento único por defeito:
  - `hissing` e `background_noise` usam `denoise`;
  - `degraded_voice` usa `enhance`.
- A validação pós-enhancer foi mantida para impedir áudio vazio, não finito, com duração alterada, volume anormal ou pico excessivo.

## [2026-06-02] Integração com visualização de treino
- A limpeza continua exibindo análise, tratamento e transcrição por arquivo.
- Após a limpeza e preparação do dataset, a etapa de treinamento passou a usar uma visualização compacta de progresso no orquestrador Colab/Kaggle.
- Essa mudança evita que a saída fonemizada/verbosa do fluxo de treino polua o console, mantendo os detalhes completos em `Models/super_Voz/train.log`.

## [2026-06-03] Reativação do Resemble no fluxo Kaggle
- O notebook Kaggle `run_kaggle_styletts2.ipynb` voltou a definir `SUPER_VOZ_ENABLE_RESEMBLE=1` por padrão.
- Com isso, `limpeza_ia.py --enhancer auto --ambiente kaggle` não apenas identifica áudios defeituosos: ele tenta reparar com Resemble Enhance antes da padronização final.
- O fluxo ativo continua escolhendo um único tratamento por defeito dominante:
  - `hissing` e `background_noise` usam `denoise`;
  - `degraded_voice` usa `enhance`.
- A padronização final permanece obrigatória em todos os casos: 24 kHz, mono, PCM 16-bit, trim de silêncio e normalização.
- Se o Resemble falhar, der erro de device ou a saída reprovar validação de duração/RMS/pico, o original é preservado e ainda passa pela padronização final segura.
- Essa alteração corrige a regressão em que o Kaggle deixava `SUPER_VOZ_ENABLE_RESEMBLE=0`, fazendo o script copiar o original após detectar defeito e aplicar somente a limpeza determinística.

## [2026-06-03] Ativação forçada no runner Kaggle
- A reativação por `os.environ.setdefault("SUPER_VOZ_ENABLE_RESEMBLE", "1")` não era suficiente quando a sessão Kaggle já herdava `SUPER_VOZ_ENABLE_RESEMBLE=0`.
- O notebook agora usa atribuição direta: `os.environ["SUPER_VOZ_ENABLE_RESEMBLE"] = "1"`.
- `run_kaggle_oneclick.py` também define a variável como `1`.
- `scripts/run_kaggle_styletts2.py` força `SUPER_VOZ_ENABLE_RESEMBLE=1` quando `enable_resemble_enhance: true` no YAML.
- Com o enhancer habilitado, o runner chama `limpeza_ia.py --enhancer resemble`, evitando que `--enhancer auto` respeite um valor antigo `0` herdado do ambiente.

## [2026-06-04] Pesquisa: diagnóstico seguro e ferramentas gratuitas para limpeza estilo Auphonic

Objetivo da pesquisa: evoluir o `limpeza_ia.py` para escolher a menor intervenção possível por arquivo, preservando identidade vocal para treino StyleTTS2. A conclusão principal é que não devemos aplicar uma cadeia pesada em todos os áudios. O fluxo mais seguro é medir o defeito dominante, escolher uma ferramenta específica e validar a saída antes de aceitar o arquivo processado.

### Referências pesquisadas

- Auphonic usa a estratégia de análise automática antes do processamento: Adaptive Leveler, loudness normalization, noise/hum reduction, filtering e speech recognition. Referências:
  - https://eu1.auphonic.com/help/
  - https://us1.auphonic.com/help/algorithms/singletrack.html
  - https://auphonic.com/pricing
- DNSMOS é métrica não intrusiva para avaliar qualidade de fala e ruído, com notas para `SIG`, `BAK` e `OVRL`. Referência:
  - https://arxiv.org/abs/2110.01763
- NISQA é alternativa não intrusiva para MOS e qualidade perceptual de fala/TTS. Referências:
  - https://github.com/gabrielmittag/NISQA
  - https://arxiv.org/abs/2304.09226
- FFmpeg `ebur128`/`loudnorm` mede loudness integrado, LRA e true peak; isso é melhor que normalizar apenas pico/RMS. Referências:
  - https://ffmpeg.org/ffmpeg-filters.html#ebur128
  - https://ffmpeg.org/ffmpeg-filters.html#loudnorm
  - https://github.com/slhck/ffmpeg-normalize
- Silero VAD e pyannote.audio são opções para detectar presença de fala, silêncio, sobreposição e segmentação. Referências:
  - https://github.com/snakers4/silero-vad
  - https://github.com/pyannote/pyannote-audio
- ClipDetect detecta clipping mesmo quando o áudio já foi normalizado depois da distorção. Referência:
  - https://pypi.org/project/clipdetect/
- DeepFilterNet, RNNoise, Resemble Enhance, Demucs, noisereduce, Pedalboard, VoiceFixer e ClearerVoice-Studio são candidatos gratuitos/open source para tratamentos específicos. Referências:
  - https://github.com/Rikorose/DeepFilterNet
  - https://github.com/xiph/rnnoise
  - https://github.com/resemble-ai/resemble-enhance
  - https://github.com/facebookresearch/demucs
  - https://github.com/timsainb/noisereduce
  - https://github.com/spotify/pedalboard
  - https://github.com/haoheliu/voicefixer
  - https://github.com/modelscope/ClearerVoice-Studio

### Estratégia recomendada

1. **Análise antes de restaurar**
   - Medir `DNSMOS` como já fazemos, mantendo `sig`, `bak` e `ovrl`.
   - Adicionar opcionalmente `NISQA` para uma segunda opinião de MOS quando houver GPU/tempo.
   - Medir clipping com `clipdetect` ou heurística própria de amostras saturadas/platôs.
   - Medir LUFS, LRA e true peak com FFmpeg `ebur128` ou `loudnorm`.
   - Medir fala/silêncio com Silero VAD; para casos avançados, pyannote.audio pode detectar sobreposição ou troca de falante.
   - Medir espectro com `librosa`: excesso acima de 8 kHz indica chiado; baixa energia acima de 4-6 kHz pode indicar áudio telefônico/baixa banda; flatness alta sugere ruído de fundo.

2. **Escolher uma ferramenta por defeito dominante**
   - Não aplicar Demucs, denoise, enhancer, compressor e loudnorm em cadeia pesada por padrão.
   - Aplicar tratamento único ou curto, conforme defeito principal.
   - Preservar original quando a análise indicar áudio já bom.

3. **Validar depois do tratamento**
   - Recalcular duração, RMS, pico, true peak, LUFS e DNSMOS.
   - Rejeitar saída com duração muito alterada, RMS anormal, pico excessivo, clipping novo, queda de MOS ou transcrição claramente pior.
   - Se reprovar, copiar original e aplicar apenas padronização final segura para StyleTTS2.

### Matriz de decisão sugerida

| Defeito detectado | Indicadores | Ferramenta preferida | Alternativa | Risco para a voz | Observação |
|---|---|---|---|---|---|
| Áudio bom | DNSMOS alto, sem clipping, LUFS aceitável, VAD estável | Preservar original | Padronização 24 kHz/mono/PCM16 | Baixo | Melhor não restaurar áudio que já serve para treino. |
| Ruído estacionário leve | `bak` baixo, flatness moderada, sem voz degradada | DeepFilterNet | RNNoise | Baixo/médio | Menos agressivo que enhancer completo; bom para ventilador, hiss leve e ambiente constante. |
| Ruído estacionário forte | `bak` muito baixo, hissing alto | Resemble `denoise` | DeepFilterNet forte | Médio | Validar se fricativas e respirações naturais não foram apagadas. |
| Voz degradada | `ovrl` e `sig` baixos, sem música dominante | Resemble `enhance` | VoiceFixer | Médio/alto | Pode alterar timbre; usar só quando o ganho de qualidade justificar. |
| Áudio telefônico/baixa banda | sample rate baixo, baixa energia em alta frequência | Resemble `enhance` | ClearerVoice super-resolution | Alto | Pode "inventar" brilho; validar identidade vocal com cuidado. |
| Música ou fundo musical | detecção de música/energia harmônica persistente | Demucs vocals-only | ClearerVoice separation | Alto | Usar só quando a voz está misturada com música; Demucs pode criar artefatos. |
| Várias vozes/sobreposição | pyannote indica overlapped speech ou speaker change frequente | Rejeitar/filtrar trecho | pyannote segmentation | Baixo | Para treinar uma voz neural, é melhor remover trechos com outro falante do que tentar reparar. |
| Silêncio longo/pausas | VAD mostra muita ausência de fala | Trim por VAD/librosa | FFmpeg silenceremove | Baixo | Não cortar pausas internas naturais demais; apenas bordas e silêncios excessivos. |
| Clipping/distorção | clipdetect positivo, platôs no waveform, true peak alto | Rejeitar se severo | VoiceFixer/declipping experimental | Alto | Clipping severo não é "limpeza"; pode contaminar o treino. |
| Volume inconsistente | LUFS muito baixo/alto, LRA alto, sem ruído grave | ffmpeg-normalize/loudnorm | Pedalboard compressor/limiter | Baixo/médio | Preferir LUFS real a normalização por pico. |
| Sibilância/chiado agudo | excesso espectral acima de 8 kHz | DeepFilterNet ou de-esser leve via Pedalboard/EQ | Resemble denoise | Médio | Evitar apagar `s`, `f`, brilho natural da voz. |
| Reverb/sala | MOS baixo, cauda espectral, fala distante | ClearerVoice/VoiceFixer experimental | Resemble enhance | Alto | Dereverb é arriscado para identidade vocal; só em modo opcional. |

### Stack gratuita recomendada por prioridade

1. **FFmpeg `ebur128`/`loudnorm` ou `ffmpeg-normalize`**
   - Prioridade alta porque o script atual normaliza amplitude, mas ainda não faz LUFS real.
   - Deve entrar no final, depois de qualquer restauração, com true peak conservador.

2. **DeepFilterNet**
   - Melhor candidato para denoise principal gratuito sem depender de uma restauração tão agressiva.
   - Bom para ruído de fundo e chiado quando a voz ainda está preservada.

3. **Silero VAD**
   - Leve e útil para medir proporção de fala, cortar bordas silenciosas e evitar amplificar ruído em trechos sem fala.

4. **ClipDetect ou detector próprio de clipping**
   - Importante para impedir que áudio irrecuperável entre no dataset.
   - Clipping severo deve marcar o arquivo para rejeição, não para enhancement automático.

5. **Pedalboard**
   - Útil para high-pass, compressor, limiter e EQ leve, imitando parte do nivelamento estilo Auphonic.
   - Deve ser usado com parâmetros conservadores.

6. **Demucs**
   - Já é instalado no runner, mas não deve ser padrão.
   - Usar apenas quando houver música/fundo musical claro.

7. **NISQA**
   - Boa segunda métrica de qualidade, especialmente para TTS/naturalidade.
   - Pode ser opcional por custo de instalação/execução.

8. **ClearerVoice-Studio e VoiceFixer**
   - Bons candidatos para modo pesado/experimental.
   - Devem ficar atrás de uma flag porque podem alterar timbre e naturalidade.

### Regra de preservação da identidade vocal

- Se o áudio for aceitável, preservar.
- Se o defeito for leve, aplicar ferramenta leve.
- Se o defeito for grave, tentar ferramenta pesada apenas uma vez.
- Se a saída piorar qualquer métrica crítica ou parecer alterar a duração/timbre, descartar a saída processada.
- Para treino StyleTTS2, é melhor ter menos arquivos bons do que muitos arquivos artificialmente restaurados com identidade vocal instável.
