# run_kaggle_styletts2.ipynb

Este documento resume o estado atual do notebook `run_kaggle_styletts2.ipynb` e das mudancas relacionadas no pipeline Kaggle do `super_Voz`.

## Objetivo

O notebook foi ajustado para rodar o treinamento do StyleTTS2 no Kaggle sem depender do Google Drive, permitindo download de audios do Cloudflare R2 e bloqueando apenas upload/sync de resultados para o R2 na execucao Kaggle.

O motivo principal e evitar taxas de escrita/persistencia no Cloudflare. A estrategia atual e:

- usar Cloudflare R2 ou Kaggle Dataset como entrada de audios;
- sincronizar o pacote completo da voz com Hugging Face Bucket;
- manter apenas `minha_voz_styletts2/model/best_model.pth` depois de upload confirmado;
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
/kaggle/working/StyleTTS2/minha_voz_styletts2
/kaggle/working/super_Voz_styletts2_data
/kaggle/working/super_Voz_outputs
```

Ao final, o notebook informa a pasta local do pacote:

```text
/kaggle/working/StyleTTS2/minha_voz_styletts2
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

O token precisa de permissao de escrita no bucket. O runner usa `hf sync` para restaurar o
pacote antes do treino e sincroniza `/kaggle/working/StyleTTS2/minha_voz_styletts2` durante
o treino. Depois de um upload confirmado, remove os `epoch_2nd_*.pth` de
`Models/super_Voz`, mantendo somente `model/best_model.pth` no pacote local.

O notebook carrega esse secret com `kaggle_secrets.UserSecretsClient` e grava em
`os.environ`. O runner tambem faz fallback direto para `UserSecretsClient().get_secret("HF_TOKEN")`
quando `HF_TOKEN` nao estiver no ambiente. Assim, o script tambem funciona quando for chamado
diretamente no Kaggle, desde que o secret exista com esse label exato.

Nesta configuracao, o bucket e obrigatorio. Se `HF_TOKEN` estiver ausente ou o bucket nao
puder ser acessado/criado com `hf buckets create ... --exist-ok`, o treino aborta antes de
gerar checkpoints. O runner verifica novos checkpoints a cada 5 segundos, remove apenas os
checkpoints que ja foram enviados e preserva qualquer checkpoint mais novo criado durante um upload.

Para reduzir o uso do `/kaggle/working`, o runner tambem remove `Audios_brutos` e
`Audios_processados` depois que o dataset final e o pacote forem criados. O dataset preparado
da execucao anterior e os WAVs antigos do pacote sao removidos antes de recriar a versao atual.
Mensagens `[DISCO]` mostram o espaco usado e livre durante o pipeline.
Quando `best_model.pth` ja foi restaurado, o checkpoint base LibriTTS nao e baixado novamente.
No primeiro treinamento, esse checkpoint base e removido depois que o primeiro checkpoint da
voz for sincronizado com sucesso.

O upload usa `--delete`, portanto o bucket nao acumula artefatos removidos do pacote. Para
baixar a voz em outro computador:

```text
hf sync hf://buckets/warllem/Super_voz ./local
```

No Windows, instale a CLI com:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

Para enviar uma pasta manualmente:

```text
hf sync ./data hf://buckets/warllem/Super_voz
```

O pacote inclui `best_model.pth`, `config.yml`, listas e audios do dataset, metadata,
referencia de voz, requisitos, notebooks oficiais de inferencia, documentacao e os pesos
auxiliares `Utils/ASR`, `Utils/JDC` e `Utils/PLBERT` quando estiverem disponiveis.

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

Se nao encontrar, usa o checkpoint base:

```text
Models/LibriTTS/epochs_2nd_00020.pth
```

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
