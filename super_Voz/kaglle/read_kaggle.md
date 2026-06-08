# Read Kaggle

Historico e checklist do fluxo Kaggle do `super_Voz`.

## Contexto

A pasta `super_Voz/kaglle` concentra os arquivos exclusivos do Kaggle para nao conflitar com os arquivos do Colab.

O notebook principal e:

```text
super_Voz/kaglle/run_kaggle_styletts2.ipynb
```

O runner real e:

```text
super_Voz/kaglle/scripts/run_kaggle_styletts2.py
```

## Erro corrigido

### Dependencias da Limpeza IA no modo F5

Falha observada depois do download R2:

```text
[AVISO] Falha ao verificar motor GPU: No module named 'onnxruntime'
[ERRO CRÍTICO] Motor DNSMOS falhou: No module named 'onnxruntime'
[AVISO] resemble-enhance indisponível: No module named 'resemble_enhance'
ModuleNotFoundError: No module named 'whisper'
```

A causa nao era R2 nem falta de audio. O modo atual `tts_engine: "f5_tts_ptbr"` pulava o instalador legado do StyleTTS2, mas ainda executava `limpeza_ia.py`, que precisa de `openai-whisper`, `onnxruntime-gpu` e, quando ativado, `resemble-enhance`.

O runner agora instala as dependencias da limpeza via `install_audio_cleaning_dependencies()` tambem no ramo F5 antes de iniciar a Limpeza IA. Para o proximo erro parecido, confirme no log se aparece:

```text
--- Instalando Dependências da Limpeza IA ---
```

antes de:

```text
[INFO] Iniciando Limpeza IA
```

### P100 incompatível com PyTorch ativo

Falha observada ao carregar Whisper:

```text
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation.
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

A causa e PyTorch instalado sem kernel CUDA para `sm_60`. O runner agora executa `install_ml_runtime_dependencies()` no fluxo F5 antes da limpeza e depois de instalar `f5-tts`, fixando Torch/Transformers para P100/K80. A limpeza tambem testa CUDA de verdade antes de carregar Whisper; se falhar, usa CPU para Whisper/Resemble em vez de abortar.

No proximo log, conferir:

```text
--- Instalando Runtime ML compatível ---
GPU sm_60 detectada; fixando Torch 2.5.1
```

e depois, dentro da limpeza, uma destas linhas:

```text
[OK] Torch CUDA operacional
Whisper/Resemble usarao CPU
```

### Resemble sem `deepspeed`

Falha observada durante restauração de áudio:

```text
[AVISO] Falha ao carregar Resemble na GPU: No module named 'deepspeed'
[ERRO ENHANCER] No module named 'deepspeed'
```

Isso nao quebra Whisper nem a padronizacao final; o script preserva o original e segue. A causa e que `resemble-enhance` e instalado com `--no-deps` para nao trocar Torch/Torchaudio, entao o runner precisa instalar explicitamente dependencias usadas internamente. O instalador da Limpeza IA agora inclui `deepspeed` e define `DS_BUILD_OPS=0`.

### Checkpoint F5 sem envelope EMA

Falha observada no inicio do treino:

```text
RuntimeError: Error(s) in loading state_dict for EMA:
Missing key(s): "initted", "step", "ema_model.transformer..."
Unexpected key(s): "transformer..."
```

A limpeza e o dataset ja tinham passado. A causa era o checkpoint PT-BR `pt-br/model_last.safetensors`: ele contem pesos crus `transformer.*`, mas o trainer do `f5-tts` espera um checkpoint EMA `ema_model.transformer.*` para `--pretrain`.

O runner agora converte o checkpoint antes do treino para:

```text
ckpts/<dataset_name>/pretrained_*_ema.pt
```

Essa conversao prefixa os pesos como `ema_model.transformer.*` e cria os buffers `initted` e `step`, sem alterar o arquivo original da biblioteca. Antes de criar ou reutilizar o convertido, o runner remove `pretrained_*` antigos em `ckpts/<dataset_name>` para impedir que o trainer escolha um `.safetensors` cru deixado por execucao anterior.

No proximo log, procurar:

```text
[F5-TTS-PT-BR] Convertendo checkpoint safetensors para formato EMA do trainer
[F5-TTS-PT-BR] Checkpoint pretrain compatível criado
```

### Checkpoint F5 com embedding de texto incompatível

Falha observada depois da conversao EMA:

```text
RuntimeError: Error(s) in loading state_dict for EMA:
size mismatch for ema_model.transformer.text_embed.text_embed.weight:
copying a param with shape torch.Size([2546, 512]) from checkpoint,
the shape in current model is torch.Size([56, 512])
```

A causa e que o checkpoint PT-BR original pode ter um vocabulario maior que o tokenizer `char` criado para o dataset atual. O runner agora le `vocab.txt` depois de `prepare_csv_wavs.py`, calcula `len(vocab) + 1` linhas e ajusta `ema_model.transformer.text_embed.text_embed.weight` durante a conversao do pretrain. O convertido passa a usar o nome `pretrained_*_ema_vocab<N>.pt`, o que evita reutilizar caches antigos com embedding incompatível.

No proximo log, procurar:

```text
[F5-TTS-PT-BR] Embedding de texto do pretrain ajustado ao vocabulario atual: 2546 -> 56 linhas.
[F5-TTS-PT-BR] Checkpoint pretrain compatível criado
```

### Caminho do `limpeza_ia.py`

O Kaggle falhou com:

```text
/usr/bin/python3: can't open file '/kaggle/working/Super_voz/limpeza_ia.py': [Errno 2] No such file or directory
```

A causa era que o runner procurava `limpeza_ia.py` em:

```text
/kaggle/working/Super_voz/limpeza_ia.py
```

mas o arquivo correto fica em:

```text
/kaggle/working/Super_voz/super_Voz/kaglle/limpeza_ia.py
```

O runner foi ajustado para separar:

```text
code_dir  = /kaggle/working/Super_voz/super_Voz/kaglle
data_root = /kaggle/working/Super_voz
```

Assim, o codigo roda da pasta Kaggle correta, mas os dados continuam em:

```text
/kaggle/working/Super_voz/Audios_brutos
/kaggle/working/Super_voz/Audios_processados
```

## Secrets TeraBox no Kaggle

Para o upload TeraBox funcionar, crie um secret separado para cada item em:

```text
Kaggle Notebook > Add-ons > Secrets
```

Use exatamente estes labels:

```text
TERABOX_NDUS
TERABOX_JS_TOKEN
TERABOX_CSRF_TOKEN
TERABOX_BROWSER_ID
TERABOX_NDUT_FMT
```

Em cada secret:

```text
Label = nome do secret
Value = valor correspondente
```

Nao coloque aspas, espacos extras ou `Label:` dentro do campo.

## Onde pegar os valores no TeraBox

Abra `https://www.terabox.com`, faca login e pressione `F12`.

### Cookies

Va em:

```text
Application > Cookies > https://www.terabox.com
```

Copie:

```text
ndus       -> TERABOX_NDUS
csrfToken  -> TERABOX_CSRF_TOKEN
browserid  -> TERABOX_BROWSER_ID
ndut_fmt   -> TERABOX_NDUT_FMT
```

### jsToken

Va em:

```text
Network > Fetch/XHR
```

Com o painel aberto, clique em uma pasta ou arquivo no TeraBox. Depois clique em uma requisicao parecida com:

```text
list
api/list
home
quota
categorylist
```

Procure em:

```text
Payload
Headers
Query String Parameters
Request URL
```

O campo e:

```text
jsToken -> TERABOX_JS_TOKEN
```

Se a requisicao mostrar campos como estes:

```text
app_id
web
channel
clienttype
jsToken
dp-logid
dir
num
page
```

copie apenas o valor de `jsToken`.

## Seguranca

Os valores `ndus`, `csrfToken`, `browserid`, `ndut_fmt` e `jsToken` sao credenciais reais da sessao TeraBox.

Nao grave esses valores em:

```text
notebook
YAML
Git
Markdown
chat publico
```

Se esses valores forem compartilhados, trate a sessao como exposta. O procedimento seguro e:

1. sair da conta no TeraBox;
2. entrar novamente;
3. copiar novos valores;
4. atualizar os Kaggle Secrets.

## Execucao esperada

Depois de criar os secrets, execute o notebook com:

```text
Internet = On
GPU = On
Run All
```

No log inicial, o esperado e:

```text
Secret TERABOX_NDUS carregado para TERABOX_NDUS.
Secret TERABOX_JS_TOKEN carregado para TERABOX_JS_TOKEN.
Secret TERABOX_CSRF_TOKEN carregado para TERABOX_CSRF_TOKEN.
Secret TERABOX_BROWSER_ID carregado para TERABOX_BROWSER_ID.
Secret TERABOX_NDUT_FMT carregado para TERABOX_NDUT_FMT.
Config runtime: TeraBox ativado.
```

O pipeline tambem deve passar do ponto que falhava antes:

```text
[INFO] Iniciando Limpeza IA...
cwd: /kaggle/working/Super_voz/super_Voz/kaglle
```

## TeraBox no Kaggle

O Kaggle nao tem ferramenta nativa para TeraBox.

A solucao do projeto usa:

```text
super_Voz/kaglle/scripts/terabox_uploadercli_sync.py
```

Esse wrapper usa uma ferramenta comunitaria para fazer upload dos checkpoints com os secrets da sessao.

O TeraBox deve ser tratado como persistencia opcional. Mesmo se o TeraBox falhar, o pipeline deve continuar e gerar o pacote final:

```text
/kaggle/working/super_voz_resultados.zip
```

## Restaurar checkpoints

O caminho mais confiavel para retomar treino continua sendo Kaggle Dataset:

1. exporte ou baixe a pasta `StyleTTS2` ou `Models/super_Voz` do TeraBox;
2. crie um Kaggle Dataset com essa pasta;
3. anexe o dataset no notebook;
4. rode novamente.

O runner tenta restaurar automaticamente de caminhos como:

```text
/kaggle/input/styllet2
/kaggle/input/styletts2
/kaggle/input/terabox/StyleTTS2
/kaggle/input/terabox/styletts2
/kaggle/input/super-voz/StyleTTS2
/kaggle/input/super-voz/styletts2
```

No modo atual `tts_engine: "f5_tts_ptbr"`, o runner nao usa o checkpoint base LibriTTS em ingles. Ele restaura/baixa a biblioteca `libraries/f5_tts_ptbr`, faz o fine-tuning F5-TTS PT-BR e exporta a voz para `voices/minha_voz_f5_tts_ptbr`. A inferencia deve acontecer em outro programa que carregue esses artefatos.

O treino F5 tem monitor de checkpoint: a checagem roda periodicamente, envia para Hugging Face apenas quando encontra checkpoint novo e estavel, e pula uploads quando nada mudou. O log tambem recebe mensagens de keep-alive durante o `accelerate` para manter a execucao visivel no Kaggle.

Durante o treino, o runner nao apaga mais o checkpoint que acabou de enviar. Ele mantem o checkpoint
mais recente em `Models/super_Voz` e remove apenas checkpoints anteriores quando um checkpoint mais
novo ja foi enviado com sucesso. Isso evita que o config aponte para um arquivo removido no meio da
retomada.

## Se der erro

Enviar o trecho do log a partir de:

```text
INICIANDO PIPELINE SUPER_VOZ
```

ate a falha.

Trechos importantes para diagnostico:

```text
[TeraBox][AVISO]
limpeza_ia.py
CalledProcessError
Traceback
```
