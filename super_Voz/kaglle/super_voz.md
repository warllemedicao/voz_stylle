# Histórico do Projeto super_Voz - Combate ao ZeroDivisionError

## Problema Recorrente
O treinamento do StyleTTS2 falha com `ZeroDivisionError: division by zero` no script `train_finetune_accelerate.py`.

## Diagnóstico
Embora tenhamos aplicado um patch matemático para evitar a divisão por zero (`iters_test = max(1, iters_test)`), o fato de o erro persistir ou de a validação resultar em `0` iterações indica que o **Dataset de Validação está sendo totalmente rejeitado** pelo StyleTTS2.

### Possíveis Causas nos Áudios Processados:
1. **Silêncios Longos:** Áudios com muito silêncio no início/fim podem ser filtrados ou causar falhas no alinhamento.
2. **Formato Incompatível:** O StyleTTS2 é extremamente rígido. Ele espera:
   - Sample Rate específico (geralmente 24kHz).
   - Áudio Mono.
   - Bit depth de 16-bit PCM.
   - Sem silêncios excessivos (o modelo tenta alinhar texto -> áudio; se houver áudio sem fala correspondente, ele falha).
3. **Duração:** Áudios muito curtos (< 1s) ou muito longos (> 12s) costumam ser descartados pelo dataloader interno.

## Plano de Ação (30/05/2026)
1. **Documentar Histórico:** Criação deste arquivo `super_voz.md`.
2. **Forçar Reprocessamento:** Remover a busca por `Audios_processados` no config para garantir que o `limpeza_ia.py` rode do zero.
3. **Otimizar `limpeza_ia.py`:** Revisar o script para garantir que ele aplique:
   - Trim de silêncio agressivo.
   - Normalização de volume.
   - Conversão exata para o formato StyleTTS2.

## Melhoria na Qualidade de Áudio (31/05/2026)
Implementação de ferramentas de estado-da-arte para análise e limpeza, focando na qualidade exigida pelo StyleTTS2.

### Novas Tecnologias Integradas:
1. **DNSMOS (Microsoft):** Substituímos a análise manual por uma rede neural que dá notas de 1 a 5 para a qualidade da voz (MOS). Isso evita processar áudios que já estão perfeitos e garante que áudios ruins sejam detectados com precisão.
2. **Resemble Enhance:** Motor gratuito/local de reparo de fala. A integração V9 usa GPU no Colab/Kaggle com `device="cuda"` e tratamento único por defeito principal.
3. **Sistema Híbrido de Análise:** Restauramos as **Heurísticas de Ruído e Assobio (Hissing)** para trabalhar em conjunto com a IA. Agora, o programa reporta exatamente quais defeitos foram encontrados (ex: "Ruído constante", "Chiado agudo"), dando mais transparência ao usuário.

### Impacto no Processo:
- **Segurança:** O programa agora é mais inteligente. Se o `DNSMOS` der uma nota alta, o áudio original é preservado para evitar artefatos de IA.
- **Fidelidade StyleTTS2:** O áudio final é garantido em 24kHz, Mono, 16-bit PCM e normalizado em -1dB, eliminando a principal causa do `ZeroDivisionError`.
- **Robustez de Instalação:** O `onnxruntime-gpu` continua sendo verificado para o DNSMOS. O `resemble-enhance` é instalado por padrão quando `SUPER_VOZ_ENABLE_RESEMBLE` não é `0`, preservando o stack Torch/Torchaudio do ambiente.

## Solução Técnica Final (Versão 8 - 31/05/2026)
Após a Versão 6 ainda apresentar erros de "Device Mismatch" em alguns ambientes Colab, implementamos a **Versão 8**, a mais robusta até agora.

### Melhorias da Versão 8:
1. **Explicit Model Loading:** Agora o script chama `load_enhancer` e `load_denoiser` explicitamente antes de qualquer processamento, garantindo que ambos os modelos internos da biblioteca sejam movidos para a GPU de forma independente.
2. **String-Based Device:** Mudamos a passagem do device de objetos `torch.device` para strings literais (`'cuda'`), seguindo recomendações de compatibilidade da biblioteca.
3. **Cache Management:** Adicionamos `cache_clear()` nos carregadores de modelo para evitar que estados corrompidos de execuções anteriores interfiram no processo.
4. **Fallback Inteligente:** Se, mesmo com todas as precauções, um áudio específico causar erro de device na GPU, o sistema agora captura a exceção, move o tensor para a CPU e processa aquele áudio individualmente em modo de segurança, retornando para a GPU no áudio seguinte. Isso garante que o pipeline nunca trave no meio do caminho.
5. **Contexto de Inferência:** Todas as chamadas agora são encapsuladas em `torch.inference_mode()` para máxima eficiência e segurança de memória.

## Revisão de Compatibilidade do Resemble Enhance (31/05/2026)
O log do Colab mostrou que a incompatibilidade persiste mesmo após warm-up, carregamento explícito e fallback:

`Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

Conclusão técnica naquele momento: a integração V8 do `resemble-enhance` não estava confiável para este projeto. O `onnxruntime-gpu` estava relacionado ao DNSMOS e podia estar correto, mas o erro ocorria dentro do fluxo PyTorch do enhancer. Como o notebook instalava `resemble-enhance --no-deps` para evitar downgrade do PyTorch, a biblioteca carregava, porém a integração manual de device podia ficar incompatível com o CUDA/PyTorch presente no runtime.

### Decisão Operacional V8 (superada pela V9)
- O pipeline Colab/Kaggle deixou temporariamente de usar `resemble-enhance` por padrão.
- Essa decisão foi revertida na V9 após revisar a forma correta de chamar a inferência GPU.
- A guarda de qualidade foi mantida: se o áudio sair vazio, distorcido, com duração alterada ou volume anormal, ele é descartado e o original é preservado.

## Solução Técnica V9 - Resemble Enhance GPU (31/05/2026)
Após nova pesquisa e revisão do código oficial do Resemble Enhance, o projeto abandonou a ideia de usar Auphonic API por ser pago e voltou a usar um motor local/gratuito.

### Decisão Técnica
- Resemble Enhance passa a ser o restaurador padrão em `--enhancer auto`.
- O notebook Colab define `SUPER_VOZ_ENABLE_RESEMBLE=1` por padrão.
- Para desligar o enhancer, defina `SUPER_VOZ_ENABLE_RESEMBLE=0`.
- A instalação usa `resemble-enhance --upgrade --no-deps` depois de instalar explicitamente as dependências necessárias, preservando o stack `torch`/`torchaudio` do Colab/Kaggle.

### Correção do Uso GPU
O erro anterior vinha de uma integração agressiva: o pipeline fazia resample manual, movia waveform/resampler para CUDA e ainda rodava dummy inference. A V9 remove isso.

Fluxo atual:
1. Carrega o áudio com `torchaudio`.
2. Converte para mono 1D.
3. Mantém o waveform em CPU.
4. Chama a API oficial com `device="cuda"`.
5. A própria biblioteca faz resample, chunking e movimentação interna para GPU.
6. Se ocorrer mismatch, tenta fallback CPU apenas naquele arquivo.
7. Valida a saída antes de aceitar.

### Tratamento por Defeito Principal
O `limpeza_ia.py` agora informa um defeito principal por áudio:
- `hissing`: chiado agudo;
- `background_noise`: ruído de fundo, reservado para expansão;
- `degraded_voice`: voz degradada/baixa qualidade.

Cada áudio recebe um único tratamento:
- `hissing` ou `background_noise` -> `denoise`;
- `degraded_voice` -> `enhance`.

Isso evita aplicar uma cadeia ampla de efeitos em todos os arquivos. O objetivo é reparar somente o defeito dominante e reduzir risco de alterar a identidade vocal.

## Arquitetura Atual do Pipeline (31/05/2026)

### 1. Notebook Colab (`run_colab_super_voz.ipynb`)
- Monta o Google Drive.
- Ativa keep-alive.
- Clona ou atualiza este repositório a partir do GitHub.
- Instala dependências mínimas para iniciar (`pyyaml`, `boto3`).
- Verifica GPU e `onnxruntime-gpu`.
- Define `SUPER_VOZ_ENABLE_RESEMBLE=1` por padrão para ativar Resemble Enhance GPU.
- Chama `scripts/run_colab_styletts2.py --config styletts2_colab_config.yml`.

### 2. Orquestrador Colab (`scripts/run_colab_styletts2.py`)
- Configura `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Confirma GPU antes de treinar.
- Clona/atualiza o StyleTTS2.
- Aplica patches no StyleTTS2:
  - compatibilidade `torch.load(..., weights_only=False)` para PyTorch 2.6+;
  - mitigação de OOM em validação/referência;
  - proteção contra `ZeroDivisionError` quando a validação fica vazia.
- Instala bibliotecas Python e pacotes de sistema.
- Instala `resemble-enhance` por padrão, exceto quando `SUPER_VOZ_ENABLE_RESEMBLE=0`.
- Baixa áudios brutos do Cloudflare R2.
- Executa `limpeza_ia.py --ambiente colab --enhancer auto --force`.
- Prepara o dataset StyleTTS2 e inicia o fine-tuning.

### 3. Limpeza e Transcrição (`limpeza_ia.py`)
- Avalia qualidade com DNSMOS e heurística de chiado.
- Decide se o áudio precisa de restauração.
- Em Colab/Kaggle, `--enhancer auto` usa Resemble Enhance quando `SUPER_VOZ_ENABLE_RESEMBLE` não é `0`.
- Informa o `defeito_principal` e escolhe um único tratamento.
- Mesmo com o enhancer ativo, valida a saída antes de aceitar:
  - rejeita áudio vazio;
  - rejeita valores não finitos;
  - rejeita duração muito diferente da original;
  - rejeita volume anormal ou pico excessivo.
- Se a restauração falhar ou for rejeitada, copia o original.
- Sempre aplica padronização final:
  - 24 kHz;
  - mono;
  - PCM 16-bit;
  - trim de silêncio;
  - normalização.
- Transcreve com Whisper e gera `train.txt`.
- No Kaggle com `tts_engine: "f5_tts_ptbr"`, essa etapa tambem exige instalacao propria das dependencias da limpeza. O runner agora chama `install_audio_cleaning_dependencies()` antes de `limpeza_ia.py`; se `onnxruntime`, `resemble_enhance`, `whisper` ou outra biblioteca usada nao ficar disponivel apos instalacao, o processo aborta antes do dataset.
- A politica atual e falhar cedo quando dependencia essencial falha: runtime ML, F5-TTS, limpeza/transcricao, DNSMOS/ONNX, Resemble e comandos de audio sao verificados apos instalacao. TeraBox e restores/uploads alternativos continuam opcionais.
- Em P100/K80, `torch.cuda.is_available()` pode ser verdadeiro mesmo quando o PyTorch instalado nao tem kernel para a arquitetura da GPU. A limpeza agora faz um teste CUDA real antes de carregar Whisper/Resemble; se aparecer `no kernel image is available`, usa CPU nessa etapa em vez de encerrar o pipeline.
- Quando o runner troca o stack `torch/torchaudio/torchvision` para compatibilidade P100, ele valida em subprocesso limpo e reinicia a si mesmo uma vez. Isso evita falso erro de ABI causado por `torch` antigo ja importado no processo que chamou `pip install`.
- Quando o runner instala/downgradeia NumPy/SciPy/Pandas para compatibilidade Resemble, tambem valida em subprocesso limpo e reinicia uma vez. Isso evita falsos erros `numpy.dtype size changed` em `whisper`, `pandas`, `scipy` e `resampy`.
- `huggingface_hub` fica fixado em `>=0.23.2,<1.0` para manter compatibilidade com `transformers==4.46.3`.

### 4. Preparação StyleTTS2
- `prepare_styletts2_dataset.py` converte `Audios_processados` para listas do StyleTTS2.
- Aplica filtros de duração/texto definidos no YAML (`max_audio_seconds`, `max_text_chars`).
- Copia `train_list.txt`, `val_list.txt` e `OOD_texts.txt` para a pasta `Data` do StyleTTS2.

### Comportamento Esperado
- O projeto deve priorizar estabilidade do dataset, não restauração agressiva.
- Em Colab/Kaggle, áudio ruim deve passar pelo `resemble-enhance` automaticamente quando GPU estiver disponível.
- Todo áudio aceito no dataset deve sair no formato StyleTTS2, mesmo quando for preservado original.
- O pipeline deve falhar cedo se não houver GPU, áudios brutos ou `train.txt` válido.
- O enhancer é o motor local padrão, mas nunca deve sobrescrever a padronização segura nem entrar no dataset sem validação.

## Atualização de Continuidade do Treino (02/06/2026)

O Colab pode interromper sessões por limite de uso de GPU antes das 50 epocas. Para reduzir retrabalho:

- Os orquestradores agora procuram o checkpoint mais recente em `Models/super_Voz/epoch_2nd_*.pth`.
- Quando um checkpoint de fine-tuning existe, `pretrained_model` passa a apontar para ele e `load_only_params=False`, preservando pesos e estado do otimizador.
- No modo legado StyleTTS2, quando nao existe checkpoint anterior, o pipeline usava `Models/LibriTTS/epochs_2nd_00020.pth` com `load_only_params=True`. No modo atual F5-TTS PT-BR, esse fallback fica bloqueado.
- `save_freq` foi reduzido para `1`, salvando a cada epoca.
- A chamada de treino agora filtra a saída extensa do StyleTTS2 e mostra uma barra compacta por epoca/passo. As linhas completas continuam em `Models/super_Voz/train.log`.

## Atualização F5-TTS PT-BR (08/06/2026)

O modo atual do Kaggle passou a usar `tts_engine: "f5_tts_ptbr"` para evitar iniciar vozes PT-BR a partir do LibriTTS em ingles.

- A biblioteca/base F5-TTS PT-BR fica separada em `libraries/f5_tts_ptbr_tharyck`.
- A base atual usa `Tharyck/multispeaker-ptbr-f5tts`, persistida em `libraries/f5_tts_ptbr_tharyck`, porque publica `vocab.txt`, `setting.json` e checkpoints compativeis com F5-TTS.
- Os artefatos da voz neural ficam separados em `voices/minha_voz_f5_tts_ptbr`.
- O projeto gera/exporta os arquivos da voz; a inferencia texto-para-audio deve ser feita por outro programa.
- No modo F5, o fallback LibriTTS fica bloqueado.
- O runner baixa apenas os arquivos necessarios da base Tharyck (`model_last.safetensors`, `vocab.txt`, `setting.json`, README e referencias) para evitar puxar todos os checkpoints grandes do repositorio.
- Quando `use_base_vocab: true`, o runner copia o `vocab.txt` da biblioteca base para o dataset F5 depois de `prepare_csv_wavs.py`, mantendo as 2546 linhas do embedding textual (`len(vocab) + 1`) em vez de reduzir a camada para o vocabulario pequeno da voz.
- Quando o checkpoint base PT-BR vem como `.safetensors` de pesos crus (`transformer.*`), o runner cria um checkpoint temporario em formato EMA (`ema_model.transformer.*`) antes do fine-tuning e remove caches `pretrained_*` antigos que poderiam ser escolhidos pelo trainer. Se o embedding de texto do checkpoint divergir do vocabulario ativo, o runner ajusta `ema_model.transformer.text_embed.text_embed.weight` para `len(vocab.txt) + 1` linhas durante a conversao.
- O monitor F5 procura checkpoint novo durante o treino e envia para Hugging Face apenas quando o arquivo novo esta estavel.
- O runner imprime keep-alive periodico no log do Kaggle durante o `accelerate`.
- Se o treino F5 falhar depois de gerar checkpoint local, o runner tenta sincronizar o ultimo checkpoint antes de encerrar.

## Atualização Resemble Enhance no Kaggle (08/06/2026)

Durante a limpeza IA no Kaggle, o Resemble Enhance iniciou corretamente em arquivos com `degraded_voice`, mas alguns reparos falharam logo antes da gravação do WAV com:

```text
[ERRO ENHANCER] only 0-dimensional arrays can be converted to Python scalars
```

Diagnostico: a chamada `enhance` estava dentro do fluxo esperado da biblioteca, mas a borda de gravacao precisava tolerar retornos em formatos diferentes. A causa provavel era `hwav` ou `new_sr` chegando como tensor/array com shape inesperado, em vez de audio mono 1D e sample rate `int`.

Correção aplicada em `limpeza_ia.py`:

- normaliza a saida do Resemble antes de `soundfile.write`;
- converte tensor/array para `float32`;
- mistura canais para mono quando necessario;
- achata o audio para 1D;
- limpa `NaN`/`inf`;
- limita pico acima de 1.0;
- converte `new_sr` explicitamente para `int`;
- passa `tau=0.5` explicitamente em `enhance`;
- se `enhance` falhar por erro interno nao-CUDA, tenta `denoise` conservador antes de preservar o original;
- permite diagnostico com `SUPER_VOZ_DEBUG_ENHANCER=1`.

Se o enhancer ainda devolver saida vazia, escalar ou sample rate invalido, o pipeline continua seguro: rejeita a saida, preserva o original e aplica a padronizacao final 24 kHz/mono/PCM16 antes da transcricao.

Atualizacao do mesmo dia: em logs como `Enhance falhou ... tentando denoise conservador...` seguido de `[WHISPER] Transcrevendo...`, o pipeline nao voltou ao erro antigo completo. O fallback `denoise` foi executado e aceito, mas a causa precisava ser corrigida. A falha do `enhance` vem do CFM do `resemble-enhance`, que chama `float(scipy.optimize.fsolve(...))`; com NumPy 2.x essa conversao de array 1D para escalar falha. O runner agora instala os pins criticos declarados pelo wheel do Resemble (`numpy==1.26.2`, `scipy==1.11.4` e dependencias auxiliares) antes de instalar `resemble-enhance --no-deps`. A limpeza tambem imprime `[RESEMBLE][VERSOES]` para confirmar o ambiente real no Kaggle.

## Atualização de Progresso no Kaggle (03/06/2026)

O StyleTTS2 pode registrar as linhas de progresso apenas em `Models/super_Voz/train.log`, sem repassar essas linhas diretamente para o console do notebook. Para evitar a sensação de travamento:

- `run_kaggle_styletts2.py` agora acompanha `Models/super_Voz/train.log` em tempo real durante o treino.
- Quando encontra linhas `Epoch [...], Step [...], Loss: ...`, o wrapper imprime a barra compacta direto na célula do Kaggle.
- A barra continua mostrando progresso por epoca/passo de treino, não por checkpoint gerado.
- Linhas de validação também são refletidas no console como `[VALIDACAO]`.
- O treino, dataset e parâmetros não foram alterados; a mudança é apenas de visualização/monitoramento.

## Persistência em Hugging Face Bucket (04/06/2026)

Os arquivos da voz neural passam a ser sincronizados com o bucket:

```text
hf://buckets/warllem/Super_voz
```

- O runner tenta restaurar a persistencia Hugging Face antes de escolher o checkpoint de retomada.
- O pacote local fica em `/kaggle/working/StyleTTS2/minha_voz_styletts2`.
- A sincronização tenta `hf buckets sync ... --delete`, depois `hf sync ... --delete`, e por fim
  cai para repositorio Hugging Face com `hf upload-large-folder`/`hf upload` quando a CLI do
  Kaggle nao tiver suporte a buckets.
- Depois que o upload é confirmado, o checkpoint atual permanece em `Models/super_Voz`; apenas checkpoints anteriores ao último checkpoint enviado são apagados.
- O pacote mantém `model/latest_checkpoint.pth` para retomada e `model/best_model.pth` para o melhor modelo conhecido.
- O Hugging Face é obrigatório na configuração Kaggle: sem `HF_TOKEN` ou sem backend de upload funcional, o treino aborta antes de gerar checkpoints locais.
- Falha ao restaurar pacote remoto nao aborta sozinha; o primeiro upload real ocorre no primeiro checkpoint de epoca.
- O runner checa checkpoints novos a cada 300 segundos por padrao e só faz upload quando encontra um checkpoint novo, valido e estavel.
- Se o treino falhar, o bloco final sincroniza o pacote para recuperação, mas a mensagem final informa falha/interrupção em vez de `TREINO FINALIZADO`.
- `Audios_brutos` e `Audios_processados` são removidos depois que o dataset final e o pacote forem criados.
- O dataset preparado de uma execução anterior e os WAVs antigos do pacote são removidos antes de gerar a versão atual.
- O runner informa o uso e espaço livre do `/kaggle/working` nos pontos principais do pipeline.
- No modo atual F5-TTS PT-BR, o checkpoint base LibriTTS nao e baixado.
- No modo legado StyleTTS2, a politica antiga de retencao do checkpoint base continua documentada apenas como historico.
- O pacote inclui `manifest.json`, `config.json`, `tokenizer_config.json`, `api_config.json`, `README.md`, configuração StyleTTS2, dataset preparado, metadata, referência de voz, documentação, requisitos e pesos auxiliares `Utils/ASR`, `Utils/JDC` e `Utils/PLBERT` quando disponíveis.
- Os metadados seguem uma organização parecida com pacotes do Hugging Face, mas o checkpoint continua em formato StyleTTS2 `.pth` e nao e um modelo nativo de `transformers.pipeline`.
- Os WAVs preparados também são necessários para retomar o treinamento, pois `train_list.txt` e `val_list.txt` apontam para esses arquivos.
- O StyleTTS2 não usa um vocoder externo separado; o decoder/vocoder treinado está dentro do checkpoint.
- O projeto oficial fornece notebooks de inferência, não um `inference.py` oficial. Os notebooks `Inference_LibriTTS.ipynb` e `Inference_LJSpeech.ipynb` são incluídos quando disponíveis.

### Como o `/kaggle/working` será usado

`/kaggle/working` é o armazenamento temporário do próprio Kaggle. Se esse disco encher, o treinamento pode falhar com `No space left on device`. O Hugging Face não substitui completamente o disco local durante o treino: o StyleTTS2 ainda precisa ler o código, o dataset final e pelo menos um checkpoint local enquanto está executando.

Fluxo de uso do disco:

1. O notebook valida o `HF_TOKEN` e um backend Hugging Face funcional antes de iniciar o treino.
2. O pacote remoto é restaurado em `/kaggle/working/StyleTTS2/minha_voz_styletts2`.
3. Os áudios brutos e processados existem apenas durante download, limpeza e preparação.
4. O dataset final é criado em `/kaggle/working/super_Voz_styletts2_data`.
5. O pacote inicial é materializado localmente, sem upload Hugging Face antes do treino.
6. `Audios_brutos` e `Audios_processados` são apagados antes do treinamento.
7. Durante o treino, cada checkpoint novo de epoca é detectado, copiado para `model/latest_checkpoint.pth` e enviado ao Hugging Face; `model/best_model.pth` só muda quando a validação melhora ou quando ainda não existe best.
8. Se o upload falhar, o checkpoint local é preservado para não perder o treinamento.

Arquivos que precisam permanecer no working durante o treino:

```text
/kaggle/working/StyleTTS2
/kaggle/working/super_Voz_styletts2_data
/kaggle/working/StyleTTS2/Models/super_Voz/epoch_2nd_*.pth
/kaggle/working/StyleTTS2/minha_voz_styletts2/model/latest_checkpoint.pth
/kaggle/working/StyleTTS2/minha_voz_styletts2/model/best_model.pth
```

O pacote também contém `data_reference/wavs`, mas esses WAVs são criados por hard link para o dataset final quando o sistema de arquivos permite. Assim, eles aparecem em duas pastas sem ocupar o dobro do espaço físico.

No modo atual F5-TTS PT-BR, o maior pico de uso vem da biblioteca/base `Tharyck/multispeaker-ptbr-f5tts`, do dataset preparado e dos checkpoints da voz. O runner limita o download da biblioteca aos arquivos necessarios para nao baixar todos os checkpoints do repositorio. O checkpoint base LibriTTS nao e baixado nesse modo.

Mensagens com prefixo `[DISCO]` mostram o espaço usado e livre no início, antes do treino, depois da limpeza dos intermediários, após falhas de upload e após a sincronização final.

## Observação Operacional Cloudflare/Kaggle (05/06/2026)

Não remova os dados do Cloudflare do arquivo YAML. No fluxo Kaggle atual, a seção
`cloudflare_r2` do `styletts2_kaggle_config.yml` deve continuar contendo os dados
necessários para leitura dos áudios brutos, incluindo `endpoint_url`, `bucket_name` e
`raw_audio_prefix`. Esses dados permitem que o runner baixe novamente os áudios de entrada
quando o Kaggle inicia uma sessão limpa.

A regra operacional atual é:

- Cloudflare R2 continua sendo fonte de entrada dos áudios brutos.
- A configuração runtime do notebook pode bloquear upload para R2 com `disable_r2_uploads: true`.
- Não apagar a seção `cloudflare_r2` do YAML, porque sem ela o pipeline depende apenas de Kaggle Input local.
- Se as credenciais R2 forem usadas no YAML deste projeto, elas devem permanecer disponíveis para o Kaggle conforme a estratégia atual do projeto.

## Leitura de Kaggle Secrets no Runner (05/06/2026)

O runner Kaggle agora tambem tenta ler secrets diretamente com:

```python
from kaggle_secrets import UserSecretsClient
UserSecretsClient().get_secret("NOME_DO_SECRET")
```

Isso vale para `HF_TOKEN` e para os aliases R2 aceitos, como `R2_ACCESS_KEY_ID` e
`R2_SECRET_ACCESS_KEY`. A ordem de leitura e:

1. usar a variavel ja existente em `os.environ`;
2. se ela nao existir, tentar o Kaggle Secret com o mesmo label;
3. se encontrar, gravar o valor em `os.environ` para as proximas etapas;
4. se o secret obrigatorio ainda estiver ausente, falhar cedo com mensagem clara.

Com isso, o notebook pode continuar carregando secrets antes do runner, mas o script tambem
fica protegido quando for executado diretamente no Kaggle.

### Diagnóstico do erro `No user secrets exist`

Se o Kaggle responder:

```text
No user secrets exist for kernel id ... and label HF_TOKEN
```

o runner novo ja esta sendo executado e a tentativa de leitura via `UserSecretsClient` chegou
ao servico de secrets do Kaggle. A falha significa que, para aquele kernel/notebook, o Kaggle
nao encontrou um secret com o label exato `HF_TOKEN`.

Teste minimo antes do pipeline:

```python
from kaggle_secrets import UserSecretsClient

for label in ["HF_TOKEN", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]:
    try:
        value = UserSecretsClient().get_secret(label)
        print(label, "OK", "tamanho:", len(value or ""))
    except Exception as exc:
        print(label, "ERRO:", exc)
```

`HF_TOKEN` e obrigatorio e precisa aparecer como `OK`. Se falhar, corrija no Kaggle:
`Add-ons > Secrets`, label exatamente `HF_TOKEN`, value com o token Hugging Face, salvar,
reiniciar o kernel e executar o teste novamente.

## Simulação do Encerramento na Época 10 (05/06/2026)

Foi informado que uma execução anterior treinou até a `epoch 10` e depois finalizou sozinha por causa de um erro. Pela leitura do runner Kaggle, a época 10 é um ponto sensível porque o YAML usa:

```yaml
diff_epoch: 10
```

Na configuração do StyleTTS2, `diff_epoch` normalmente marca a transição para uma fase mais pesada do treinamento, envolvendo partes adicionais do modelo/perdas. Portanto, uma simulação provável é:

1. O treino rodou normalmente das épocas 1 a 10 com o conjunto inicial de perdas/modelos.
2. Ao entrar na fase após `diff_epoch`, o StyleTTS2 passou a usar componentes mais pesados.
3. Em GPU Kaggle limitada, como P100/T4, isso pode ter causado erro de CUDA, OOM, processo `Killed` ou falha ao carregar algum modelo auxiliar.
4. O wrapper `run_training_with_progress()` detecta que o processo de treino saiu com código diferente de zero e levanta `CalledProcessError`.
5. Mesmo com erro, o bloco `finally` do runner ainda tenta parar os monitores e sincronizar o pacote/checkpoint final com Hugging Face/TeraBox/R2, fazendo parecer que o notebook "finalizou sozinho".

Hipóteses mais fortes para o erro na época 10:

- transição do `diff_epoch: 10` aumentou uso de VRAM e causou OOM;
- versão de dependência incompatível em etapa acionada só depois dessa época;
- checkpoint ou modelo auxiliar ausente/corrompido, exigido apenas nessa fase;
- falha de sincronização ou disco cheio durante materialização/upload do pacote após checkpoint;
- validação ou dataloader com lote inválido ao mudar a fase de treino.

Para confirmar, o trecho mais importante do log é o final de `Models/super_Voz/train.log` junto com as linhas do notebook logo antes de `CalledProcessError`, `CUDA out of memory`, `Killed`, `Traceback` ou `[HuggingFace][AVISO]`.

## Modificações Realizadas
- [x] Criação de `super_voz.md`.
- [x] Upgrade do `limpeza_ia.py` para a **Versão 8** (Explicit Loading + CPU Fallback).
- [x] Documentação das alterações nos arquivos `.md` individuais.
- [x] Desativação segura do `resemble-enhance` por padrão no Colab/Kaggle.
- [x] Guarda de qualidade para impedir que áudio defeituoso do enhancer entre no dataset.
- [x] Atualização do notebook Colab com política explícita do enhancer.
- [x] Documentação da arquitetura atual do pipeline.
- [x] Descarte do Auphonic API por custo.
- [x] Ativação padrão do Resemble Enhance GPU no Colab/Kaggle.
- [x] Ajuste da integração Resemble para seguir o fluxo oficial `device="cuda"`.
- [x] Tratamento único por `defeito_principal`.
- [x] Retomada automática do último checkpoint `epoch_2nd_*.pth`.
- [x] Barra compacta de progresso durante o treinamento.
- [x] Espelhamento do progresso de `train.log` no console do Kaggle.
- [x] Salvamento de checkpoint a cada epoca (`save_freq: 1`).
- [x] Persistência do pacote completo da voz em Hugging Face Bucket.
- [x] Retenção do checkpoint mais recente local e remoção apenas dos anteriores após upload confirmado de checkpoint mais novo.

## Refatoração Modular e Estabilidade do Pipeline Kaggle (09/06/2026)

Com o pipeline crescendo para abranger a transição StyleTTS2 -> F5-TTS, integração de nuvem com R2, TeraBox e Hugging Face, o arquivo `run_kaggle_styletts2.py` havia se tornado um monólito frágil (+3.000 linhas), e o projeto começou a sofrer com instabilidades e atualizações surpresa nas bibliotecas do Kaggle (ex: conflitos bruscos com numpy 2.x e o resemble-enhance). 

### Medidas Aplicadas:
1. **Blindagem de Dependências:** Adição do arquivo `requirements-kaggle-strict.txt` congelando todas as versões validadas (`numpy==1.26.2`, `torch==2.5.1`, `transformers==4.46.3`). Isso interrompe o "Dependency Hell" causado pelas execuções em nuvem.
2. **Refatoração Modular:** O arquivo `run_kaggle_styletts2.py` foi decomposto usando módulos injetados em `scripts/runner_utils/` (`cloud_storage.py`, `environment.py`, `f5_integration.py`, `utils.py`), tornando a manutenção mais segura e ágil.
3. **Erradicação da Raiz do ZeroDivisionError:** Adicionado limite físico `--min_seconds=0.8` no `prepare_styletts2_dataset.py`, deletando sumariamente da lista de treinamento arquivos picotados microscópicos que acabavam passando no Whisper, mas quebravam a validação matemática do StyleTTS2/F5-TTS.
4. **Constantes e Escopo:** Na transição modular, corrigimos o problema de escopo agrupando as constantes de ambiente como `HF_HUB_COMPAT_PACKAGE` dentro do `utils.py`.

## ⚠️ AVISO IMPORTANTE SOBRE COLAB/KAGGLE
O ambiente do Colab e Kaggle **clona este repositório do GitHub**. 
Se as modificações feitas aqui não forem enviadas para o seu GitHub (**git commit** e **git push**), o Colab continuará rodando a versão antiga e o erro persistirá.

**Para que a correção funcione no Colab:**
1. Salve todas as alterações.
2. Faça o `commit` e `push` para o seu repositório.
3. Reinicie a execução no Colab.
