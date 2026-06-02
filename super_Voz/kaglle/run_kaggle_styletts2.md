# run_kaggle_styletts2.ipynb

Este documento resume o estado atual do notebook `run_kaggle_styletts2.ipynb` e das mudancas relacionadas no pipeline Kaggle do `super_Voz`.

## Objetivo

O notebook foi ajustado para rodar o treinamento do StyleTTS2 no Kaggle sem depender do Google Drive, permitindo download de audios do Cloudflare R2 e bloqueando apenas upload/sync de resultados para o R2 na execucao Kaggle.

O motivo principal e evitar taxas de escrita/persistencia no Cloudflare. A estrategia atual e:

- usar Cloudflare R2 ou Kaggle Dataset como entrada de audios;
- manter checkpoints e resultados em `/kaggle/working`;
- gerar um ZIP final com os artefatos;
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
/kaggle/working/StyleTTS2/Models/super_Voz
/kaggle/working/super_Voz_styletts2_data
/kaggle/working/super_Voz_outputs
```

Ao final, o notebook empacota os resultados em:

```text
/kaggle/working/super_voz_resultados.zip
```

Se nenhuma persistencia externa funcionar, esse ZIP continua disponivel nos outputs do Kaggle, especialmente quando a execucao e feita via `Save Version -> Save & Run All (Commit)`.

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

- `run_kaggle_styletts2.ipynb`: notebook one-click Kaggle com Cloudflare desligado e TeraBox opcional.
- `scripts/run_kaggle_styletts2.py`: adaptador TeraBox, sincronizacao periodica e upload final.
- `styletts2_kaggle_config.yml`: Cloudflare desligado por padrao e secao TeraBox configuravel.
- `README.md`: documentacao resumida da persistencia opcional via TeraBox.

## Estado atual da solucao

A solucao atual evita Cloudflare no Kaggle e preserva uma saida local robusta via ZIP.

O TeraBox ainda depende da escolha de uma CLI comunitaria funcional para Linux/Kaggle. Por isso, a integracao foi feita por comandos configuraveis, em vez de fixar uma URL ou uma ferramenta especifica no codigo.
