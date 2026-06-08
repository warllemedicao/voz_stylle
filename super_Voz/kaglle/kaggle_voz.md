# Kaggle Voz

Documento atualizado da pasta `super_Voz/kaglle`, que agora concentra os arquivos exclusivos do fluxo Kaggle para evitar conflito com Colab/local.

## Arquivos principais

- `run_kaggle_styletts2.ipynb`: notebook one-click para Kaggle.
- `styletts2_kaggle_config.yml`: configuracao base do pipeline Kaggle.
- `scripts/run_kaggle_styletts2.py`: runner real do treino no Kaggle; no modo atual usa F5-TTS PT-BR.
- `scripts/prepare_styletts2_dataset.py`: prepara `Audios_processados` para o formato StyleTTS2.
- `scripts/terabox_uploadercli_sync.py`: wrapper de upload TeraBox via ferramenta comunitaria.
- `limpeza_ia.py`: limpeza/transcricao dos audios brutos antes do treino.
- `run_kaggle_oneclick.py`: bootstrap alternativo simples.

## Fluxo do notebook

1. Clona ou atualiza `https://github.com/warllemedicao/voz_stylle.git` em `/kaggle/working/Super_voz`.
   - Se o GitHub/DNS falhar, reutiliza o clone local quando ele ja contem `super_Voz/kaglle/scripts/run_kaggle_styletts2.py`.
   - Se nao houver clone local valido, procura uma copia anexada em `/kaggle/input` e copia para `/kaggle/working/Super_voz`.
   - Se nao houver internet nem copia local/dataset com o codigo, falha com uma mensagem explicando para ativar Internet no Kaggle ou anexar um Kaggle Dataset com o projeto.
2. Localiza `run_kaggle_styletts2.py` dentro de `super_Voz/kaglle/scripts`.
3. Entra em `/kaggle/working/Super_voz/super_Voz/kaglle`.
4. Gera `styletts2_kaggle_sem_cloudflare.yml`, mantendo download R2 permitido e bloqueando upload R2.
5. Executa:

```bash
python -u scripts/run_kaggle_styletts2.py --config styletts2_kaggle_sem_cloudflare.yml
```

## Modo atual F5-TTS PT-BR

A configuracao atual usa `tts_engine: "f5_tts_ptbr"`. Nesse modo, o runner:

1. baixa/limpa/transcreve os audios;
2. restaura `libraries/f5_tts_ptbr` do Hugging Face ou baixa `firstpixel/F5-TTS-pt-br`;
3. prepara o dataset no formato oficial do F5-TTS;
4. roda fine-tuning usando o checkpoint PT-BR;
5. exporta apenas os artefatos da voz neural para `minha_voz_f5_tts_ptbr`;
6. envia o pacote da voz para `voices/minha_voz_f5_tts_ptbr`.

A inferencia nao faz parte deste projeto. Outro programa deve carregar o runtime F5-TTS, a biblioteca/base `libraries/f5_tts_ptbr` e o pacote da voz em `voices/minha_voz_f5_tts_ptbr`.

Durante o treino F5, um monitor procura checkpoints novos periodicamente. O upload para Hugging Face ocorre somente quando aparece checkpoint novo e estavel; sem checkpoint novo, a checagem nao envia nada. O runner tambem imprime keep-alive no log para reduzir risco de a execucao parecer parada em treinos longos. Se o treino falhar apos gerar checkpoint, ele tenta sincronizar o ultimo checkpoint antes de sair.

## Correcao do erro `Could not resolve host: github.com`

O erro atual foi:

```text
fatal: unable to access 'https://github.com/warllemedicao/voz_stylle.git/': Could not resolve host: github.com
CalledProcessError: Command '['git', 'clone', 'https://github.com/warllemedicao/voz_stylle.git', '/kaggle/working/Super_voz']' returned non-zero exit status 128.
```

A causa foi a correcao anterior que trocou o repositorio antigo por `https://github.com/warllemedicao/voz_stylle.git` e passou a executar `git clone` com `check=True` quando `/kaggle/working/Super_voz` nao existia. Isso resolveu o clone do repositorio errado, mas deixou o bootstrap fragil quando o Kaggle esta sem Internet ativada ou com falha temporaria de DNS.

Correcao aplicada:

- `run_kaggle_styletts2.ipynb` e `run_kaggle_oneclick.py` agora capturam falha de `git clone`/`git fetch`.
- Se ja existir `/kaggle/working/Super_voz` com o runner Kaggle correto, o notebook continua usando esse clone mesmo sem GitHub.
- Se nao existir clone em `/kaggle/working`, o bootstrap procura em `/kaggle/input` uma copia anexada como Kaggle Dataset contendo `super_Voz/kaglle/scripts/run_kaggle_styletts2.py` e copia para `/kaggle/working/Super_voz`.
- Se nenhuma dessas fontes existir, a falha passa a explicar a acao necessaria: ativar Internet no Kaggle ou anexar um dataset com o codigo do projeto.

Para executar a versao mais recente diretamente do GitHub, mantenha **Settings -> Internet** ativado no notebook Kaggle. Para executar sem internet, anexe um Kaggle Dataset contendo este repositorio.

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

- `run_kaggle_styletts2.ipynb` agora define `SUPER_VOZ_ENABLE_RESEMBLE=1` com atribuicao direta, nao `setdefault`, para sobrescrever sessoes Kaggle antigas que ainda tenham `0`.
- `run_kaggle_oneclick.py` tambem define `SUPER_VOZ_ENABLE_RESEMBLE=1`.
- `scripts/run_kaggle_styletts2.py` le `enable_resemble_enhance: true` do YAML e forca `SUPER_VOZ_ENABLE_RESEMBLE=1` antes de instalar dependencias e antes da limpeza.
- A chamada da limpeza passou a usar `limpeza_ia.py --enhancer resemble` quando `enable_resemble_enhance` esta ativo. Assim o reparo nao depende mais de `--enhancer auto` nem de uma variavel de ambiente herdada.
- Nesse modo, audios com `hissing` ou `background_noise` usam `denoise`, e `degraded_voice` usa `enhance`.
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

- `styletts2_kaggle_config.yml` deve continuar trazendo os dados de leitura do Cloudflare R2 usados pelo projeto, incluindo `endpoint_url`, `bucket_name` e `raw_audio_prefix`.
- Nao remova a secao `cloudflare_r2` do YAML. No Kaggle, ela continua sendo a fonte principal para localizar os audios brutos no R2.
- `access_key_id` e `secret_access_key` podem vir dos Kaggle Secrets `R2_ACCESS_KEY_ID` e `R2_SECRET_ACCESS_KEY` ou permanecer no YAML conforme a estrategia atual do projeto.
- O notebook parou de imprimir erro para secrets opcionais de TeraBox e overrides opcionais de R2. Ele so marca como obrigatorios os dois secrets sensiveis de leitura R2.
- O runner agora tenta preencher os campos ausentes primeiro por variaveis de ambiente e depois diretamente via `UserSecretsClient().get_secret(...)` com os labels aceitos. Se ainda faltarem `access_key_id` ou `secret_access_key`, ele explica que o download R2 precisa desses dois Kaggle Secrets. A trava `disable_r2_uploads: true` continua bloqueando apenas upload para Cloudflare; ela nao bloqueia o download dos audios.

Se esses dois secrets R2 nao existirem no kernel Kaggle e tambem nao houver Kaggle Dataset com audios, o pipeline vai falhar corretamente por falta de entrada.

Secrets aceitos:

```text
R2_ENDPOINT_URL
R2_BUCKET_NAME
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_RAW_AUDIO_PREFIX
```

O runner preenche campos ausentes a partir das variaveis de ambiente carregadas pelo notebook ou, quando executado diretamente no Kaggle, via `kaggle_secrets.UserSecretsClient`. Use exatamente os labels acima nos Kaggle Secrets.

Se o Kaggle responder `No user secrets exist for kernel id ... and label ...`, o secret nao
esta disponivel para aquele kernel com o label solicitado. Valide em uma celula separada:

```python
from kaggle_secrets import UserSecretsClient

for label in ["HF_TOKEN", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]:
    try:
        value = UserSecretsClient().get_secret(label)
        print(label, "OK", "tamanho:", len(value or ""))
    except Exception as exc:
        print(label, "ERRO:", exc)
```

`HF_TOKEN` e obrigatorio para a persistencia em Hugging Face. Os dois secrets R2 so sao
necessarios quando as credenciais nao estiverem disponiveis no YAML ou em variaveis de ambiente.

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

No modo legado StyleTTS2, se encontrar `epoch_2nd_*.pth`, retoma o fine-tuning com `load_only_params: false`. No modo atual F5-TTS PT-BR, o fallback LibriTTS em ingles nao e usado.

A retencao local de checkpoints segue uma janela segura: o checkpoint mais recente permanece em
`Models/super_Voz` depois do upload, e o runner so remove checkpoints anteriores quando um checkpoint
mais novo ja foi persistido. Assim o Kaggle economiza disco sem apagar o arquivo que o treino ainda
pode precisar para continuar.

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

## Correcao do erro Torch/Transformers no Kaggle

O erro:

```text
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6
```

nao vem do dataset preparado. No log, `prepared=522`, `train=496`, `val=26` e `missing=0` confirmam que a etapa de dataset passou. A falha acontece no inicio do treino, quando o StyleTTS2 carrega um modelo/checkpoint Hugging Face em formato PyTorch.

Em GPUs antigas do Kaggle, como P100/K80, o runner fixa `torch==2.5.1` para compatibilidade CUDA. Versoes recentes do `transformers` bloqueiam `torch.load` com Torch menor que 2.6 ao carregar checkpoints `.bin`/PyTorch. Como o `requirements.txt` do StyleTTS2 declara `transformers` sem versao, o Kaggle pode instalar uma versao nova demais e quebrar o treino.

Correcao aplicada:

- Para GPU `sm_<7`, o runner mantem `torch==2.5.1`, `torchaudio==2.5.1` e `torchvision==0.20.1`.
- No mesmo caso, o runner fixa `transformers==4.46.3`, evitando o bloqueio novo de `torch.load` durante o carregamento dos modelos auxiliares do StyleTTS2.
- Em GPUs mais novas, o runner continua deixando `torch` e `transformers` sem pin rigido.

Se o erro continuar no Kaggle, limpe o runtime ou force reinstalacao para garantir que uma versao recente demais de `transformers` nao ficou carregada na sessao antiga.
