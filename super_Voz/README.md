# super_Voz

Projeto de treinamento TTS baseado em StyleTTS2.

## Fluxo de Execução (Colab/Kaggle)

O projeto usa o **Cloudflare R2** (S3-compatible) para armazenamento persistente de áudios e checkpoints.

1. **Acesso a Dados:**
   - O pipeline baixa automaticamente os áudios do bucket R2 configurado em `styletts2_colab_config.yml` ou `styletts2_kaggle_config.yml`.
2. **Ambiente:** Clona/atualiza este repositório do GitHub.
3. **Dados:** Sincroniza `Audios_brutos` do bucket.
4. **Limpeza OBRIGATÓRIA:** Usa `limpeza_ia.py` (DNSMOS + heurísticas + Resemble Enhance GPU + Whisper + padronização segura) para reparar áudios defeituosos e garantir o formato correto para StyleTTS2 (24 kHz, mono, trim de silêncio, PCM 16-bit).
5. **Conversão:** Converte o dataset para o formato do StyleTTS2 (`wav|texto|speaker`).
6. StyleTTS2: Clona o StyleTTS2 oficial e aplica patches de compatibilidade.
7. Treino: Executa fine-tuning com `accelerate launch` usando uma barra compacta de progresso por epoca/passo.
8. Finalização: Os checkpoints e resultados ficam disponíveis localmente no Colab (`/content`) ou Kaggle (`/kaggle/working`) para download manual.

## Visualização e retomada do treino

Durante o fine-tuning, o console do Colab/Kaggle agora filtra a saída extensa do StyleTTS2 e exibe uma linha compacta:

```text
[TREINO] Epoca 12/50 |############------------------| passo 45/111 loss 0.25300
[VALIDACAO] loss 0.235 | dur 1.076 | f0 1.607
```

Os logs completos continuam sendo gravados em `Models/super_Voz/train.log`.

O pipeline também procura automaticamente o checkpoint mais recente em `Models/super_Voz/epoch_2nd_*.pth`. Se existir, ele retoma o treino a partir dele com o otimizador completo. Se não existir, usa o checkpoint base `Models/LibriTTS/epochs_2nd_00020.pth`.

Para reduzir perda de progresso quando o Colab interromper a sessão por limite de GPU, `save_freq` fica configurado como `1`, salvando um checkpoint ao fim de cada epoca.

## Política do Resemble Enhance

O `resemble-enhance` é o motor gratuito/local de restauração do projeto e fica ativado por padrão no Colab/Kaggle quando há GPU.

Comportamento atual:
- `limpeza_ia.py --enhancer auto` usa Resemble Enhance salvo se `SUPER_VOZ_ENABLE_RESEMBLE=0`.
- O áudio é classificado por defeito principal antes do tratamento.
- Defeitos de chiado/ruído usam tratamento único `denoise`.
- Voz degradada/baixa qualidade usa tratamento único `enhance`.
- A integração GPU usa `device="cuda"` na chamada oficial da biblioteca e mantém o waveform de entrada em CPU para evitar mistura manual de devices.
- Toda saída do enhancer é validada; se reprovar, o áudio original é usado e ainda passa pela padronização final.

## Configuração do Bucket (Cloudflare R2) - APENAS ENTRADA

Para usar este projeto, você deve configurar o Cloudflare R2 para baixar os áudios brutos.

```yaml
cloudflare_r2:
  endpoint_url: "https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
  access_key_id: "SUA_ACCESS_KEY"
  secret_access_key: "SUA_SECRET_KEY"
  bucket_name: "NOME_DO_BUCKET"
  raw_audio_prefix: "Audios_brutos/"
```

### Estrutura do Bucket:
- `Audios_brutos/`: Coloque aqui seus áudios originais para processamento.

**Nota:** A sincronização de áudios já processados foi desativada para garantir que a `limpeza_ia.py` sempre aplique os tratamentos de áudio necessários para evitar erros de validação no StyleTTS2.

## Estrutura de Pastas Recomendada

```text
super_Voz/
  Audios_brutos/       # Áudios originais (mp3, wav, etc)
  Audios_processados/  # Áudios limpos + train.txt (gerado automaticamente via limpeza_ia.py)
  checkpoints/         # Checkpoints salvos durante o treino
  outputs/             # Logs e outros artefatos
```

## Como rodar no Kaggle

1. Crie um novo Notebook no Kaggle.
2. Ative a **GPU** (T4 x2 ou P100).
3. Importe o notebook `run_kaggle_styletts2.ipynb` ou copie as células dele.
4. Ajuste o arquivo `styletts2_kaggle_config.yml` se necessário.
5. Após o treino, baixe os arquivos da pasta `/kaggle/working`.

### Persistência opcional no TeraBox

O Kaggle não monta Google Drive como o Colab. Para usar TeraBox como armazenamento durante o treino, crie um Kaggle Secret chamado `TERABOX_NDUS` com o cookie `ndus` da sessão e configure a seção `terabox` em `styletts2_kaggle_config.yml`. Se você já exportou a pasta do TeraBox como Kaggle Dataset, anexe-a com um nome como `styllet2` ou `styletts2`; o pipeline tenta restaurar `Models/super_Voz` automaticamente antes de escolher o checkpoint.

Como o TeraBox não fornece uma CLI oficial estável equivalente ao `rclone`, o projeto usa comandos configuráveis:

```yaml
terabox:
  enabled: true
  ndus_env: "TERABOX_NDUS"
  cli_path: "/kaggle/working/terabox-cli"
  remote_styletts2_dir: "/StyleTTS2"
  remote_checkpoint_dir: "/StyleTTS2/Models/super_Voz"
  install_commands:
    - ["curl", "-L", "URL_DA_CLI_ESCOLHIDA", "-o", "/kaggle/working/terabox-cli"]
    - ["chmod", "+x", "/kaggle/working/terabox-cli"]
```

Os templates `login_command`, `download_command` e `upload_command` podem ser ajustados conforme a sintaxe da CLI escolhida. O pipeline baixa o estado remoto do StyleTTS2 antes de escolher o checkpoint, envia checkpoints apenas quando uma época gera um checkpoint novo válido ou no upload crítico/final, e remove checkpoints locais antigos somente depois de persistência confirmada.

## Erros Comuns e Correções

- **SIGSEGV (Signal 11):** Geralmente ocorre se o ambiente não detecta a GPU corretamente ou se há incompatibilidade de bibliotecas. O script agora inclui verificações de GPU e patches de compatibilidade para PyTorch 2.6+.
- **Erro do Resemble Enhance `Expected all tensors to be on the same device`:** A integração atual evita pré-mover/resamplear o tensor manualmente para GPU. O waveform entra em CPU e a biblioteca recebe `device="cuda"`. Se ainda falhar, o script faz fallback CPU para aquele arquivo; se a saída reprovar, preserva o original.
- **CUDA Out of Memory (OOM):** StyleTTS2 é extremamente pesado para a GPU T4 (15GB). O projeto possui mitigação robusta de duas formas:
  1. Aplica patches automáticos no `train_finetune_accelerate.py` para limitar o cálculo do tamanho máximo de validação e referência.
  2. Implementa filtros rigorosos na construção do dataset:
     - `max_len: 128` (Tamanho menor no lote).
     - `max_audio_seconds: 10` (Recusa áudios com mais de 10s no dataset de treinamento para não estourar a memória de alinhamento de sequência). Se o script de áudio falhar ao extrair o tempo, a amostra também é recusada (9999s artificial) para proteger o pipeline de surpresas.
     - `batch_size: 2` (O mínimo que o modelo requer devido às camadas de Batch Normalization no discriminador).
- **Interrupção por limite de GPU do Colab:** Reexecute o notebook mantendo a mesma pasta `StyleTTS2` no Drive. O pipeline detecta o último `epoch_2nd_*.pth` e continua o treino, em vez de começar novamente do checkpoint base.

## Observação importante sobre português

O StyleTTS2 oficial foi publicado principalmente com suporte e checkpoints voltados para inglês. Para português, o pipeline abaixo consegue preparar os dados e iniciar fine-tuning, mas a qualidade final depende de fonemização, dataset e compatibilidade do PL-BERT usado.
