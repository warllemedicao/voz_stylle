# run_kaggle_styletts2.ipynb

Este documento resume o estado atual do notebook `run_kaggle_styletts2.ipynb` e das mudancas relacionadas no pipeline Kaggle do `super_Voz`.

## Objetivo

O notebook foi ajustado para rodar o treinamento do StyleTTS2 no Kaggle sem depender do Google Drive, permitindo download de audios do Cloudflare R2 e bloqueando apenas upload/sync de resultados para o R2 na execucao Kaggle.

O motivo principal e evitar taxas de escrita/persistencia no Cloudflare. A estrategia atual e:

- usar Cloudflare R2 ou Kaggle Dataset como entrada de audios;
- sincronizar o pacote completo da voz com Hugging Face Bucket;
- manter no working o checkpoint mais recente necessario para retomar o treino e apagar apenas checkpoints anteriores ja persistidos;
- tentar TeraBox apenas como persistencia opcional, se houver uma CLI configurada e o secret `TERABOX_NDUS`.

## Politica Cloudflare

O notebook cria uma configuracao runtime chamada:

```text
styletts2_kaggle_sem_cloudflare.yml
```

Dentro dessa configuracao, `cloudflare_r2` e preservado para permitir `raw_audio_prefix`, `endpoint_url`, `bucket_name` e credenciais de leitura. O notebook remove apenas `output_prefix` e grava a trava de upload:

```yaml
cloudflare_r2:
  disable_r2_uploads: true
```

Com isso, quando o notebook chama:

```text
python -u scripts/run_kaggle_styletts2.py --config styletts2_kaggle_sem_cloudflare.yml
```

o pipeline pode baixar audios do Cloudflare R2 quando `raw_audio_prefix` estiver configurado, mas nao deve enviar checkpoints, dataset ou resultados para R2.

O arquivo base `styletts2_kaggle_config.yml` ainda pode ficar com:

```yaml
cloudflare_r2:
  endpoint_url: "https://..."
  bucket_name: "super-voz"
  raw_audio_prefix: "super_voz/Audios_Brutos/"
```

A config Kaggle ja vem apontando para o R2 de leitura do projeto. Se quiser sobrescrever sem editar o Git, crie Kaggle Secrets com `R2_ENDPOINT_URL`, `R2_BUCKET_NAME`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` e `R2_RAW_AUDIO_PREFIX`. A config runtime do notebook bloqueia apenas upload/sync para R2.

## Biblioteca F5-TTS PT-BR

O pipeline separa a biblioteca/base F5-TTS PT-BR dos checkpoints da voz neural. A config usa:

```yaml
tts_engine: "f5_tts_ptbr"
model_library_root: "/kaggle/working/super_voz_model_library"
f5_voice_package_dir: "minha_voz_f5_tts_ptbr"
f5_tts_ptbr:
  repo_id: "firstpixel/F5-TTS-pt-br"
  local_dir: "/kaggle/working/super_voz_model_library/f5_tts_ptbr"
  huggingface_remote_dir: "libraries/f5_tts_ptbr"
  dataset_name: "super_voz_f5_ptbr"
  checkpoint_subpath: "pt-br/model_last.safetensors"
  exp_name: "F5TTS_Base"
  tokenizer: "char"
  checkpoint_sync_interval_seconds: 300
  checkpoint_stable_seconds: 30
  local_checkpoint_keep_last: 2
  keepalive_interval_seconds: 120
```

Na primeira execucao, o runner tenta restaurar `libraries/f5_tts_ptbr` do Hugging Face. Se a pasta ainda nao existir, baixa `firstpixel/F5-TTS-pt-br` e envia a biblioteca para essa pasta remota. Os checkpoints/artefatos da voz F5 devem ficar separados no pacote `minha_voz_f5_tts_ptbr`, enquanto o pacote legado `minha_voz_styletts2` continua reservado para StyleTTS2.

Enquanto `tts_engine: "f5_tts_ptbr"` estiver ativo, o fallback LibriTTS em ingles fica bloqueado quando nao houver checkpoint anterior. Isso evita iniciar uma nova voz PT-BR a partir de `yl4579/StyleTTS2-LibriTTS`.

Mesmo no modo F5, a etapa `limpeza_ia.py` continua obrigatoria antes do dataset. Por isso o runner instala dependencias de limpeza em um bloco proprio (`install_audio_cleaning_dependencies()`), separado do instalador legado do StyleTTS2. Esse bloco deve aparecer no log antes de `[INFO] Iniciando Limpeza IA` e cobre `openai-whisper`, `onnxruntime-gpu`, `deepspeed`, `resemble-enhance` quando habilitado, e bibliotecas de audio como `librosa` e `soundfile`.

Em GPU Kaggle P100/K80, o fluxo F5 tambem precisa do pin de runtime ML que antes existia apenas no caminho StyleTTS2. O runner chama `install_ml_runtime_dependencies()` antes da limpeza e depois da instalacao de `f5-tts`; para `sm_<7`, ele fixa `torch==2.5.1`, `torchaudio==2.5.1`, `torchvision==0.20.1` e `transformers==4.46.3`. Se o log mostrar `sm_60 is not compatible with the current PyTorch installation`, a proxima checagem e procurar `--- Instalando Runtime ML compatível ---` e a mensagem `GPU sm_60 detectada; fixando Torch 2.5.1`. A limpeza tambem testa uma operacao CUDA real antes de carregar Whisper; se a GPU falhar com `no kernel image is available`, Whisper/Resemble caem para CPU em vez de abortar.

Este projeto nao executa inferencia texto-para-audio. Ele gera e persiste os arquivos da voz neural; outro programa deve carregar o runtime F5-TTS, a biblioteca/base `libraries/f5_tts_ptbr` e o pacote `voices/minha_voz_f5_tts_ptbr`.

Durante o fine-tuning F5, o runner inicia um monitor de checkpoints. A cada `checkpoint_sync_interval_seconds`, ele procura o checkpoint mais recente em `ckpts/super_voz_f5_ptbr`; se o arquivo for novo e estiver estavel por `checkpoint_stable_seconds`, o pacote parcial da voz e materializado e enviado para `voices/minha_voz_f5_tts_ptbr`. Sem checkpoint novo, nao ha upload. Um keep-alive imprime status a cada `keepalive_interval_seconds` para manter o notebook ativo/visivel durante treinos longos.

Se o processo de treino falhar depois de algum checkpoint local existir, o runner ainda tenta sincronizar o ultimo checkpoint antes de encerrar. Se o monitor ja tiver enviado exatamente esse checkpoint, o upload final duplicado e pulado.

Apos upload confirmado de checkpoint F5, a retencao local remove checkpoints antigos e mantem apenas os `local_checkpoint_keep_last` mais recentes, por padrao 2. Isso evita acumulo de checkpoints no `/kaggle/working`.

## Entrada de audios

O pipeline primeiro tenta baixar audios brutos do Cloudflare R2 quando `cloudflare_r2.raw_audio_prefix` estiver configurado. Se R2 nao estiver configurado ou nao houver download, ele procura audios brutos nestes caminhos:

```text
Audios_brutos
/kaggle/input/super-voz/Audios_brutos
/kaggle/input/super-voz/Audios_Brutos
```

Tambem e possivel anexar um Kaggle Dataset contendo os audios em uma dessas pastas.

## Saidas locais

Durante e depois do treino, os principais artefatos ficam em:

```text
/kaggle/working/StyleTTS2/minha_voz_f5_tts_ptbr
/kaggle/working/super_voz_model_library/f5_tts_ptbr
/kaggle/working/super_voz_f5_dataset
/kaggle/working/StyleTTS2/minha_voz_styletts2
/kaggle/working/super_Voz_styletts2_data
/kaggle/working/super_Voz_outputs
```

Ao final, o notebook informa a pasta local do pacote:

```text
/kaggle/working/StyleTTS2/minha_voz_f5_tts_ptbr
```

O notebook nao cria uma copia nem um ZIP, pois isso pode duplicar varios gigabytes e causar
`No space left on device`. Se nenhuma persistencia externa funcionar, salve uma versao com
`Save Version -> Save & Run All (Commit)` para publicar `/kaggle/working` nos outputs.

## Hugging Face Bucket

Os checkpoints e artefatos da voz sao sincronizados com:

```text
hf://buckets/warllem/Super_voz
```

Adicione este Kaggle Secret:

```text
HF_TOKEN
```

O token precisa de permissao de escrita no bucket ou no repositorio fallback. O runner tenta
restaurar e sincronizar `/kaggle/working/StyleTTS2/minha_voz_styletts2` nesta ordem:

1. `hf buckets sync`, quando a CLI instalada suporta Hugging Face Buckets;
2. `hf sync`, quando esse alias existir;
3. fallback para repositorio Hugging Face usando `hf download`, `hf upload-large-folder` e
   `hf upload`, derivando `warllem/Super_voz` de `hf://buckets/warllem/Super_voz`.

Depois de um upload confirmado por qualquer um desses caminhos, a politica de retencao local e:

1. o checkpoint enviado continua em `Models/super_Voz`, pois ele pode ser o `pretrained_model`
   usado pelo processo de treino em andamento;
2. quando um checkpoint mais novo for enviado com sucesso, os checkpoints anteriores sao removidos;
3. `model/latest_checkpoint.pth` no pacote guarda o ultimo checkpoint valido para retomada externa;
4. `model/best_model.pth` no pacote guarda o melhor checkpoint conhecido pela perda de validacao,
   ou o primeiro checkpoint valido quando ainda nao ha metrica confiavel.

O notebook carrega esse secret com `kaggle_secrets.UserSecretsClient` e grava em
`os.environ`. O runner tambem faz fallback direto para `UserSecretsClient().get_secret("HF_TOKEN")`
quando `HF_TOKEN` nao estiver no ambiente. Assim, o script tambem funciona quando for chamado
diretamente no Kaggle, desde que o secret exista com esse label exato.

Nesta configuracao, a persistencia Hugging Face e obrigatoria. Se `HF_TOKEN` estiver ausente
ou se nenhum backend de upload funcionar, o treino aborta antes de gerar checkpoints. Falha de
restauracao remota antes do treino vira aviso, porque pode ser a primeira execucao ou o backend
fallback pode estar vazio. O runner valida acesso ao Hugging Face no setup, mas nao faz upload
inicial do pacote antes do treino. O monitor checa novos checkpoints em intervalo configuravel
(`huggingface.sync_interval_seconds`, 300s por padrao), e so faz upload quando encontra um
`epoch_2nd_*.pth` novo, valido e estavel. O bloco `finally` ainda faz um upload critico/final
antes de encerrar a sessao.

Essa politica evita uploads por batch, step ou intervalos curtos de tempo. Sem checkpoint novo, a
checagem nao sincroniza o diretorio inteiro. Depois de upload confirmado, o runner remove somente
checkpoints locais anteriores ao ultimo checkpoint persistido e registra o espaco recuperado nos
logs `[DISCO]`.

Se o processo `accelerate` falhar, o bloco final ainda sincroniza o pacote para recuperacao, mas a
saida passa a informar `TREINO INTERROMPIDO OU COM FALHA` em vez de anunciar sucesso.

### Diagnostico de secret indisponivel no Kaggle

Se o log mostrar:

```text
No user secrets exist for kernel id ... and label HF_TOKEN
```

o problema nao esta no runner: o servico do Kaggle informou que o secret `HF_TOKEN` nao existe
ou nao esta disponivel para aquele kernel/notebook. Isso pode acontecer quando o label foi
criado com outro nome, em outro notebook/conta, no lugar errado da interface, ou quando o kernel
nao foi reiniciado depois da criacao do secret.

Antes de rodar o pipeline, valide em uma celula separada:

```python
from kaggle_secrets import UserSecretsClient

for label in ["HF_TOKEN", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]:
    try:
        value = UserSecretsClient().get_secret(label)
        print(label, "OK", "tamanho:", len(value or ""))
    except Exception as exc:
        print(label, "ERRO:", exc)
```

`HF_TOKEN` precisa aparecer como `OK`. Se ainda aparecer erro, recrie o secret em
`Add-ons > Secrets` usando exatamente o label `HF_TOKEN`, salve, reinicie o kernel e rode o
teste novamente.

Para reduzir o uso do `/kaggle/working`, o runner tambem remove `Audios_brutos` e
`Audios_processados` depois que o dataset final e o pacote forem criados. O dataset preparado
da execucao anterior e os WAVs antigos do pacote sao removidos antes de recriar a versao atual.
Mensagens `[DISCO]` mostram o espaco usado e livre durante o pipeline.
No modo atual F5-TTS PT-BR, o checkpoint base LibriTTS nao e baixado. A base PT-BR fica em
`libraries/f5_tts_ptbr` e os artefatos da voz ficam em `voices/minha_voz_f5_tts_ptbr`.

Quando o backend usado for `hf buckets sync` ou `hf sync`, o upload usa `--delete`, portanto o
bucket nao acumula artefatos removidos do pacote. Quando a CLI do Kaggle nao oferece suporte a
buckets/sync, o fallback para repositorio usa `hf upload-large-folder` e nao remove arquivos
remotos antigos automaticamente.

Para baixar a voz em outro computador usando bucket:

```text
hf buckets sync hf://buckets/warllem/Super_voz ./local
```

Se a CLI nao tiver `hf buckets sync`, use o repositorio fallback:

```text
hf download warllem/Super_voz --local-dir ./local
```

No Windows, instale a CLI com:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

Para enviar uma pasta manualmente:

```text
hf buckets sync ./data hf://buckets/warllem/Super_voz --delete
```

Fallback manual para repositorio:

```text
hf upload-large-folder warllem/Super_voz ./data --repo-type model
```

O pacote inclui `manifest.json`, `config.json`, `tokenizer_config.json`, `api_config.json`,
`README.md`, `best_model.pth`, `config.yml`, listas e audios do dataset, metadata, referencia
de voz, requisitos, notebooks oficiais de inferencia, documentacao e os pesos auxiliares
`Utils/ASR`, `Utils/JDC` e `Utils/PLBERT` quando estiverem disponiveis. Esses JSONs documentam
o pacote no estilo Hugging Face, mas o modelo continua sendo StyleTTS2 e nao carrega diretamente
via `transformers.pipeline`.

## TeraBox opcional

O TeraBox foi tratado como uma saida experimental, porque nao ha uma CLI oficial estavel equivalente ao `rclone`.

O notebook tenta carregar um Kaggle Secret chamado:

```text
TERABOX_NDUS
```

Esse secret deve conter o cookie `ndus` da sessao TeraBox. Ele nao deve ser escrito no notebook, no YAML nem no Git.

Se o secret existir, o notebook ativa `terabox.enabled` na config runtime. Se nao existir, o script ainda tenta restaurar checkpoints de um Kaggle Input chamado `styllet2`, `styletts2`, `terabox/StyleTTS2` ou `super-voz/StyleTTS2`.

## Configuracao TeraBox

A secao TeraBox em `styletts2_kaggle_config.yml` ficou assim por padrao:

```yaml
terabox:
  enabled: false
  ndus_env: "TERABOX_NDUS"
  cookie_ndus: ""
  cli_path: "/kaggle/working/terabox-cli"
  remote_styletts2_dir: "/StyleTTS2"
  remote_checkpoint_dir: "/StyleTTS2/Models/super_Voz"
  sync_interval_seconds: 600
  install_commands: []
  login_command:
    - "{cli}"
    - "login"
    - "--ndus"
    - "{ndus}"
  download_command:
    - "{cli}"
    - "download"
    - "{remote_dir}"
    - "{local_dir}"
  upload_command:
    - "{cli}"
    - "upload"
    - "{local_dir}"
    - "{remote_dir}"
```

Os comandos sao templates. Eles podem ser ajustados conforme a sintaxe da CLI ou fork escolhido.

Variaveis disponiveis nos templates:

- `{cli}`: caminho da CLI configurado em `cli_path`;
- `{ndus}`: cookie vindo do secret ou do campo `cookie_ndus`;
- `{remote_dir}`: pasta remota configurada;
- `{local_dir}`: pasta local a sincronizar.

## Fluxo TeraBox no script

Quando TeraBox esta habilitado e o login funciona, o script:

1. instala a CLI usando `install_commands`, se houver comandos configurados;
2. executa `login_command`;
3. tenta baixar o estado remoto de `/StyleTTS2` antes de escolher o checkpoint;
4. inicia sincronizacao periodica dos checkpoints a cada `sync_interval_seconds`;
5. faz upload final dos checkpoints no bloco `finally`, mesmo se o treino parar por erro.

Os checkpoints enviados ficam em:

```text
/StyleTTS2/Models/super_Voz
```

ou no caminho definido por `remote_checkpoint_dir`.

## Retomada de treino

Depois de baixar o estado remoto opcional, o pipeline procura automaticamente:

```text
/kaggle/working/StyleTTS2/Models/super_Voz/epoch_2nd_*.pth
```

Se encontrar checkpoint de fine-tuning, ele retoma com:

```yaml
load_only_params: false
```

No modo legado StyleTTS2, se nao encontrar checkpoint, o fallback antigo era:

```text
Models/LibriTTS/epochs_2nd_00020.pth
```

Esse fallback fica bloqueado quando `tts_engine: "f5_tts_ptbr"` esta ativo.

com:

```yaml
load_only_params: true
```

## Seguranca

O cookie `ndus` e uma credencial de sessao. Ele nao deve ser versionado.

O notebook e o script mascaram o cookie nos logs quando ele aparece como argumento de comando. Mesmo assim, a pratica recomendada e usar Kaggle Secrets.

Se um `ndus` tiver sido compartilhado em chat, arquivo ou notebook publico, ele deve ser considerado exposto. Nesse caso, o ideal e encerrar a sessao no TeraBox, logar novamente e criar um novo secret.

## Arquivos alterados

- `run_kaggle_styletts2.ipynb`: notebook one-click Kaggle com Hugging Face Bucket.
- `scripts/run_kaggle_styletts2.py`: montagem do pacote, restauracao, sincronizacao e retencao de checkpoint.
- `styletts2_kaggle_config.yml`: bucket e pasta do pacote configurados.
- `inference/`: helper de validacao e exemplo de uso dos caminhos do pacote.

## Estado atual da solucao

A solucao atual usa Cloudflare apenas como entrada de audios e Hugging Face Bucket como
persistencia principal dos artefatos da voz.

O TeraBox permanece opcional e desativado por padrao.
