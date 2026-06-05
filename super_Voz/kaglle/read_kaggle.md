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

Se encontrar `epoch_2nd_*.pth`, retoma o fine-tuning. Se nao encontrar, usa o checkpoint base do LibriTTS.

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
