# Registro de Alterações: limpeza_ia.py

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
