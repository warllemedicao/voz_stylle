# Kaggle Voz

Documento atualizado da pasta `super_Voz/kaglle`, que agora concentra os arquivos exclusivos do fluxo Kaggle para evitar conflito com Colab/local.

## Arquivos principais

- `run_kaggle_styletts2.ipynb`: notebook one-click para Kaggle.
- `styletts2_kaggle_config.yml`: configuracao base do pipeline Kaggle.
- `scripts/run_kaggle_styletts2.py`: runner real do treino StyleTTS2 no Kaggle.
- `scripts/prepare_styletts2_dataset.py`: prepara `Audios_processados` para o formato StyleTTS2.
- `scripts/terabox_uploadercli_sync.py`: wrapper de upload TeraBox via ferramenta comunitaria.
- `limpeza_ia.py`: limpeza/transcricao dos audios brutos antes do treino.
- `run_kaggle_oneclick.py`: bootstrap alternativo simples.

## Fluxo do notebook

1. Clona ou atualiza `https://github.com/warllemedicao/voz_stylle.git` em `/kaggle/working/Super_voz`.
2. Localiza `run_kaggle_styletts2.py` dentro de `super_Voz/kaglle/scripts`.
3. Entra em `/kaggle/working/Super_voz/super_Voz/kaglle`.
4. Gera `styletts2_kaggle_sem_cloudflare.yml`, mantendo download R2 permitido e bloqueando upload R2.
5. Executa:

```bash
python -u scripts/run_kaggle_styletts2.py --config styletts2_kaggle_sem_cloudflare.yml
```

## Correcao do erro de runner nao encontrado

O erro:

```text
FileNotFoundError: Runner Kaggle nao encontrado em /kaggle/working/Super_voz/super_Voz/kaglle/scripts/run_kaggle_styletts2.py
```

foi causado pelo notebook clonar o repositorio antigo/errado:

```text
https://github.com/warllemedicao/Voz_styllett2.git
```

O commit anterior foi enviado para:

```text
https://github.com/warllemedicao/voz_stylle.git
```

Como o Kaggle clonava o repositorio errado, a pasta `super_Voz/kaglle` atualizada nao existia no clone novo e o notebook parava antes de carregar secrets ou rodar o pipeline.

Correcao aplicada:

- `run_kaggle_styletts2.ipynb`, `run_kaggle_oneclick.py` e `styletts2_kaggle_config.yml` agora usam `https://github.com/warllemedicao/voz_stylle.git`.
- Quando `/kaggle/working/Super_voz` ja existe, o notebook e o one-click atualizam `origin` com `git remote set-url origin https://github.com/warllemedicao/voz_stylle.git` antes de `fetch/pull`. Isso evita reutilizar um clone antigo apontando para `Voz_styllett2.git`.
- A mensagem de erro do notebook agora lista quais `run_kaggle_styletts2.py` foram encontrados no clone, para diagnosticar rapidamente clone errado ou estrutura divergente.

Depois de atualizar no GitHub, se o Kaggle ainda clonar `Voz_styllett2.git`, o notebook usado no Kaggle esta desatualizado. Importe/cole a versao nova do notebook ou atualize a celula manualmente para `https://github.com/warllemedicao/voz_stylle.git`.

## Correcao do erro `limpeza_ia.py`

O erro observado foi:

```text
/usr/bin/python3: can't open file '/kaggle/working/Super_voz/limpeza_ia.py': [Errno 2] No such file or directory
```

No log mais recente, os secrets R2 funcionaram: a linha `Audios brutos importados do R2: 523` confirma que o download dos audios foi concluido. A falha aconteceu depois, na etapa de limpeza, porque uma versao antiga do runner foi executada a partir de:

```text
/kaggle/working/Super_voz/super_Voz/scripts/run_kaggle_styletts2.py
```

Esse caminho nao e mais o fluxo Kaggle correto. O arquivo de limpeza existe em:

```text
/kaggle/working/Super_voz/super_Voz/kaglle/limpeza_ia.py
```

O runner correto fica em:

```text
/kaggle/working/Super_voz/super_Voz/kaglle/scripts/run_kaggle_styletts2.py
```

Correcoes aplicadas:

- O notebook agora prefere explicitamente `super_Voz/kaglle/scripts/run_kaggle_styletts2.py` e falha cedo se esse arquivo nao existir.
- `run_kaggle_oneclick.py` agora entra em `super_Voz/kaglle`, nao em `super_Voz`.
- O runner chama `limpeza_ia.py` por caminho absoluto, eliminando ambiguidade de `cwd`.
- Foram adicionados wrappers de compatibilidade em `scripts/`, `limpeza_ia.py`, `super_Voz/scripts/` e `super_Voz/limpeza_ia.py`. Se um notebook antigo ainda chamar `super_Voz/scripts/run_kaggle_styletts2.py`, `scripts/run_kaggle_styletts2.py` ou `python limpeza_ia.py` a partir da raiz do clone ou de `super_Voz`, ele sera redirecionado para os arquivos reais em `super_Voz/kaglle`.

O runner separa:

- `code_dir`: pasta do codigo Kaggle, `super_Voz/kaglle`;
- `data_root`: raiz de dados/runtime, por padrao `/kaggle/working/Super_voz`.

Assim, `limpeza_ia.py` e `prepare_styletts2_dataset.py` rodam a partir da pasta correta, mas os dados continuam em:

```text
/kaggle/working/Super_voz/Audios_brutos
/kaggle/working/Super_voz/Audios_processados
```

Com os wrappers, mesmo que o Kaggle ainda mostre `cwd: /kaggle/working/Super_voz`, `cwd: /kaggle/working/Super_voz/super_Voz` ou execute `super_Voz/scripts/run_kaggle_styletts2.py`, a execucao deve cair no fluxo corrigido de `super_Voz/kaglle`. Se o log ainda mostrar erro abrindo `/kaggle/working/Super_voz/limpeza_ia.py`, o clone no Kaggle esta anterior a este commit; limpe `/kaggle/working/Super_voz` ou force `git reset --hard origin/main` antes de rodar.

## Correcao do Resemble desligado no Kaggle

O fluxo de analise/reparo correto exige que audios defeituosos sejam tratados pelo Resemble Enhance antes da padronizacao final. O notebook Kaggle estava definindo:

```python
SUPER_VOZ_ENABLE_RESEMBLE=0
```

Com isso, `limpeza_ia.py --enhancer auto` detectava o defeito, mas o `AudioEnhancer` ficava desligado; o arquivo original era copiado e apenas passava por 24 kHz/mono/PCM16/trim/normalizacao. Isso garantia formato StyleTTS2, mas nao fazia `denoise` ou `enhance` real nos audios ruins.

Correcao aplicada:

- `run_kaggle_styletts2.ipynb` agora define `SUPER_VOZ_ENABLE_RESEMBLE=1` por padrao.
- O runner ja instala `resemble-enhance` quando essa variavel nao e `0`.
- A chamada continua `limpeza_ia.py --enhancer auto`; nesse modo, audios com `hissing` ou `background_noise` usam `denoise`, e `degraded_voice` usa `enhance`.
- Se o Resemble falhar ou a saida for reprovada pelas validacoes de duracao/RMS/pico, o pipeline preserva o original e ainda aplica a padronizacao final segura.

## Entrada de audios

Ordem de busca:

1. Cloudflare R2, se os secrets/configs `R2_*` estiverem disponiveis.
2. `/kaggle/working/Super_voz/Audios_brutos`.
3. `super_Voz/kaglle/Audios_brutos`, se existir.
4. Kaggle Inputs configurados:

```text
/kaggle/input/super-voz/Audios_brutos
/kaggle/input/super-voz/Audios_Brutos
```

5. Descoberta automatica em `/kaggle/input`, procurando pastas com arquivos `.wav`, `.mp3`, `.flac`, `.ogg` ou `.m4a`.

## Saidas

Durante o treino:

```text
/kaggle/working/StyleTTS2/Models/super_Voz
/kaggle/working/super_Voz_styletts2_data
/kaggle/working/super_Voz_outputs
```

Ao final, o notebook empacota:

```text
/kaggle/working/super_voz_resultados.zip
```

Se TeraBox/R2 nao estiverem configurados, esse ZIP e o fallback principal para baixar os resultados pelos outputs do Kaggle.

## Cloudflare R2

No Kaggle, o notebook cria uma config runtime com:

```yaml
cloudflare_r2:
  disable_r2_uploads: true
```

Isso bloqueia upload/sync de checkpoints e resultados para R2, mas preserva o download dos audios brutos quando `raw_audio_prefix` e credenciais existem.

## Correcao do aviso R2 incompleto

O erro atual foi:

```text
[R2][AVISO] Configuracao R2 incompleta; faltando: endpoint_url, access_key_id, secret_access_key
[AVISO] Configuração R2 ausente ou incompleta.
❌ NENHUM ÁUDIO BRUTO ENCONTRADO!
```

A causa foi a config base do Kaggle ter ficado com `cloudflare_r2.endpoint_url` vazio. Isso fazia o runner considerar o R2 incompleto mesmo antes de avaliar as credenciais.

Correcao aplicada:

- `styletts2_kaggle_config.yml` voltou a trazer os dados nao-secretos de leitura: `endpoint_url`, `bucket_name` e `raw_audio_prefix`.
- `access_key_id` e `secret_access_key` continuam vazios no Git de proposito. Eles devem vir dos Kaggle Secrets `R2_ACCESS_KEY_ID` e `R2_SECRET_ACCESS_KEY`.
- O notebook parou de imprimir erro para secrets opcionais de TeraBox e overrides opcionais de R2. Ele so marca como obrigatorios os dois secrets sensiveis de leitura R2.
- O runner agora explica que, quando faltarem `access_key_id` ou `secret_access_key`, o download R2 precisa desses dois Kaggle Secrets. A trava `disable_r2_uploads: true` continua bloqueando apenas upload para Cloudflare; ela nao bloqueia o download dos audios.

Se esses dois secrets R2 nao existirem no kernel Kaggle e tambem nao houver Kaggle Dataset com audios, o pipeline vai falhar corretamente por falta de entrada.

Secrets aceitos:

```text
R2_ENDPOINT_URL
R2_BUCKET_NAME
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_RAW_AUDIO_PREFIX
```

O arquivo `styletts2_kaggle_config.yml` nao deve guardar access key ou secret key. Deixe esses campos vazios no YAML e use os Kaggle Secrets acima. O runner preenche a configuracao a partir das variaveis de ambiente carregadas pelo notebook.

## TeraBox

Nao ha CLI oficial estavel do TeraBox equivalente ao `rclone`. A solucao implementada usa um wrapper configuravel em:

```text
scripts/terabox_uploadercli_sync.py
```

Esse wrapper instala/usa `dnigamer/TeraboxUploaderCLI` no runtime do Kaggle e gera `secrets.json`/`settings.json` temporarios fora do Git. A ferramenta exige tokens da sessao web do TeraBox, nao apenas `ndus`.

Crie estes Kaggle Secrets:

```text
TERABOX_NDUS
TERABOX_JS_TOKEN
TERABOX_CSRF_TOKEN
TERABOX_BROWSER_ID
TERABOX_NDUT_FMT
```

Com esses secrets, o notebook ativa `terabox.enabled` na config runtime. O upload periodico/final usa:

```yaml
terabox:
  upload_command:
    - "{python}"
    - "{script_dir}/terabox_uploadercli_sync.py"
    - "upload"
    - "--local-dir"
    - "{local_dir}"
    - "--remote-dir"
    - "{remote_dir}"
    - "--tool-dir"
    - "{cli}"
```

O destino padrao dos checkpoints e:

```text
/StyleTTS2/Models/super_Voz
```

## Restauracao de checkpoints

Download direto do TeraBox nao ficou como caminho principal, porque as ferramentas comunitarias publicas sao instaveis e muitas focam upload ou links compartilhados. Para retomar treino com confiabilidade:

1. Baixe/exporte a pasta `StyleTTS2` ou `Models/super_Voz` do TeraBox.
2. Crie um Kaggle Dataset com essa pasta.
3. Anexe o dataset ao notebook com nome como `styllet2`, `styletts2` ou `super-voz`.

O runner tenta restaurar automaticamente destes caminhos:

```text
/kaggle/input/styllet2
/kaggle/input/styletts2
/kaggle/input/terabox/StyleTTS2
/kaggle/input/terabox/styletts2
/kaggle/input/super-voz/StyleTTS2
/kaggle/input/super-voz/styletts2
```

Se encontrar `epoch_2nd_*.pth`, retoma o fine-tuning com `load_only_params: false`. Se nao encontrar, usa o pretrained LibriTTS base.

## Como obter os tokens TeraBox

Os projetos publicos consultados documentam que os valores vem da sessao web logada:

- `ndus`: cookie em `Application -> Cookies -> https://www.terabox.com`.
- `jsToken`: parametro em chamadas XHR/API no painel `Network`.
- `csrfToken`, `browserid`, `ndut_fmt`: cookies da mesma sessao.

Nao grave esses valores no notebook, YAML ou Git. Use Kaggle Secrets.

Referencias publicas:

- `dnigamer/TeraboxUploaderCLI`: https://github.com/dnigamer/TeraboxUploaderCLI
- `Pahadi10/terabox-upload-tool`: https://github.com/Pahadi10/terabox-upload-tool

## Estado recomendado

- Use R2 ou Kaggle Dataset para entrada de audios.
- Use `/kaggle/working/super_voz_resultados.zip` como fallback obrigatorio.
- Use TeraBox para upload periodico/final de checkpoints somente quando todos os secrets de sessao estiverem atualizados.
- Use Kaggle Dataset para restaurar checkpoints em novas execucoes.
