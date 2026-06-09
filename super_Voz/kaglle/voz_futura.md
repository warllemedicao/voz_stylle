# Voz Futura

Historico e conclusoes da conversa sobre uso da voz neural treinada no projeto `super_Voz`.

## Objetivo

Criar uma forma pratica de usar a voz neural treinada para:

- leitura de textos;
- respostas de uma IA;
- possivel integracao com automacao residencial;
- possivel uso em Alexa Skill;
- futuramente empacotar a voz em API, biblioteca ou app proprio.

## Estado Atual da Voz Neural

Foi analisado o projeto `super_Voz` e os artefatos parciais do treinamento no Google Drive.

Arquivos principais encontrados no Drive:

- `Models/super_Voz/epoch_2nd_00019.pth`
- `Models/super_Voz/epoch_2nd_00024.pth`
- `Models/super_Voz/epoch_2nd_00029.pth`
- `Models/super_Voz/train.log`
- `Models/super_Voz/config_super_voz.yml`
- `Data/train_list.txt`
- `Data/val_list.txt`

Conclusoes:

- O treinamento nao finalizou as 50 epocas.
- O ultimo checkpoint realmente retomavel encontrado foi `epoch_2nd_00029.pth`.
- O log avancou ate a epoca 31, passo 80/111, mas sem checkpoint correspondente.
- O treino parece ter sido interrompido por sessao/tempo, nao por erro claro de modelo.
- O checkpoint parcial ja pode ser usado para testes de inferencia, mas ainda nao deve ser considerado produto final.
- No fluxo Kaggle atual, a limpeza de disco mantem sempre o checkpoint mais recente local e apaga
  apenas checkpoints anteriores depois que um checkpoint mais novo foi enviado.
- Se o treino falhar, a sincronizacao final ainda preserva artefatos para recuperacao, mas a saida
  nao deve ser interpretada como treino concluido.

Metricas observadas:

- Dataset de treino: 223 linhas.
- Dataset de validacao: 11 linhas.
- Melhor validation loss observada: `0.228`.
- Ultima validacao completa: `0.235`.
- Ultimo checkpoint: aproximadamente 2.24 GB.

Ponto critico de seguranca:

- Os arquivos YAML do projeto continham credenciais Cloudflare R2 em texto claro.
- Se esses arquivos foram publicados ou compartilhados, as chaves devem ser revogadas/rotacionadas.
- O ideal e usar variaveis de ambiente para credenciais.

## Alexa

Conclusao principal:

Nao e possivel substituir globalmente a voz da Alexa pela voz neural treinada.

O que acontece na pratica:

1. Voce diz: "Alexa, abrir minha skill".
2. A Alexa escuta, entende e roteia o comando usando a voz/sistema da propria Alexa.
3. A skill recebe o texto/intencao.
4. Se a skill responder com texto comum, quem fala e a voz da Alexa/Amazon Polly.
5. Se a skill gerar ou apontar para um audio MP3 com a voz neural, a Alexa toca esse audio.

Ou seja:

- A Alexa continua sendo a interface de entrada.
- A voz da Alexa continua existindo para comandos, erros, abertura de skill, timers, automacoes e respostas do sistema.
- A sua voz neural pode aparecer como audio tocado dentro da skill.

Exemplo:

```text
Usuario: Alexa, abrir Professor Voz.
Alexa: Abrindo Professor Voz.  # voz da Alexa
Usuario: Explique por que a Terra e redonda.
Skill: gera resposta com IA, sintetiza com a voz neural e devolve MP3.
Usuario ouve: explicacao na voz neural treinada.
```

Nao da para fazer:

- "Alexa, explique qualquer coisa" e sempre cair na sua skill.
- Substituir a voz da Alexa inteira.
- Interceptar todos os comandos da casa.
- Trocar a wake word ou as mensagens do sistema pela voz treinada.

O mais proximo seria:

```text
Alexa, peca ao Professor Voz para explicar por que a Terra e redonda.
```

Nesse caso, a skill poderia responder com audio gerado pela voz neural.

## Google Assistant

Conclusao principal:

Nao e possivel usar a voz neural treinada como voz do Google Assistant.

Exemplo:

```text
Usuario: Ok Google, explique por que a Terra e redonda.
Resposta: voz do proprio Google Assistant.
```

O Google Assistant nao oferece hoje um caminho simples para substituir a voz global por uma voz neural propria.

As antigas Conversational Actions do Google Assistant foram encerradas em 13 de junho de 2023, o que reduziu bastante a possibilidade de criar experiencias conversacionais customizadas no Assistant.

Uso recomendado:

- manter Google Assistant apenas para automacao residencial comum;
- usar a voz neural em um app proprio, extensao, bot ou painel separado.

## App Proprio Com IA e Automacao

A ideia mais viavel e criar um app proprio que combine:

- Gemini API para inteligencia/respostas;
- Home Assistant API para controlar a casa;
- StyleTTS2/super_Voz para gerar audio com a voz neural treinada;
- interface propria para texto, microfone e reproducao de audio.

Arquitetura proposta:

```text
Usuario fala ou digita
        ↓
App proprio
        ↓
Backend
        ↓
Gemini API entende/responde
        ↓
Home Assistant API executa comandos da casa
        ↓
StyleTTS2 gera audio com voz neural
        ↓
App toca a resposta na voz treinada
```

Exemplo:

```text
Usuario: Apague a luz da sala e explique a diferenca entre substantivo e adjetivo.

Backend:
- chama Home Assistant para apagar a luz;
- chama Gemini para gerar a explicacao;
- chama StyleTTS2 para gerar o audio com a voz neural.

Resposta falada:
"Apaguei a luz da sala. Substantivo e a palavra que nomeia seres, objetos, lugares ou ideias..."
```

Nesse app, a resposta seria ouvida com a voz neural treinada, nao com a voz do Google, Alexa ou Gemini.

## Gemini API

Quando se fala em "chamar o Gemini", nao significa abrir o site/app do Gemini.

Significa usar a API:

```text
App/Backend -> Gemini API -> resposta em texto
```

O usuario continua dentro do app proprio.

O Gemini e usado "por baixo dos panos", como motor de linguagem.

Observacoes:

- Existe camada gratuita da Gemini Developer API/Google AI Studio, mas com limites.
- Para prototipo e uso leve, pode ser suficiente.
- Para uso intenso, pode exigir plano pago.
- A chave da API nao deve ficar exposta dentro do app; deve ficar no backend.

## Home Assistant

O Home Assistant e recomendado como central de automacao.

Motivo:

- O app proprio nao precisa integrar individualmente cada fabricante.
- O Home Assistant ja integra luzes, tomadas, sensores, cenas, Zigbee, Wi-Fi, Matter, Tuya, MQTT, ESPHome etc.
- O app proprio chama apenas a API do Home Assistant.

Arquitetura:

```text
App proprio -> Backend -> Home Assistant API -> dispositivos da casa
```

Assim, o app pode controlar a casa sem depender do Google Assistant ou Alexa.

## Voz Neural Como API

Primeiro passo tecnico recomendado:

Transformar a voz neural treinada em uma API TTS simples.

Em vez de o app lidar diretamente com StyleTTS2, checkpoints, configs e dependencias, ele chamaria:

```http
POST /tts
{
  "text": "Ola, tudo bem?"
}
```

E receberia:

```text
audio.wav ou audio.mp3
```

Arquitetura:

```text
Texto -> API TTS -> StyleTTS2 -> audio com voz neural
```

Formato inicial sugerido:

```text
super-voz-tts/
  model/
    epoch_2nd_00029.pth
    config_super_voz.yml
  inference.py
  api.py
  requirements.txt
```

Esse formato facilitaria usar a voz em:

- app proprio;
- extensao leitora de texto;
- Alexa Skill via MP3;
- bot de Telegram/WhatsApp;
- painel web;
- automacoes do Home Assistant.

## Sobre Reduzir Para Um Arquivo Unico

Nao existe normalmente "a voz neural" como um arquivo pequeno unico.

Uma voz neural costuma depender de:

- pesos do modelo;
- configuracao;
- tokenizer/fonemizador;
- decoder/vocoder;
- codigo de inferencia;
- dependencias de audio;
- possivelmente modelos auxiliares de pitch, duracao, estilo e linguagem.

No caso atual, o checkpoint `epoch_2nd_00029.pth` tem cerca de 2.24 GB.

Opcao mais realista:

1. Criar um motor minimo de inferencia.
2. Criar uma API local.
3. Manter o modelo carregado em memoria.
4. O app chama a API e recebe audio.
5. Depois otimizar tamanho e velocidade.

Possiveis otimizacoes futuras:

- gerar MP3 em vez de WAV para transporte;
- cachear respostas frequentes;
- empacotar o ambiente via Docker com um servidor FastAPI;
- testar CPU vs GPU;
- exportar as partes pesadas do decodificador F5-TTS / StyleTTS2 para ONNX (isso derrubaria o tamanho do modelo drasticamente e aceleraria a inferência de 2.24GB para algo que roda em tempo real);
- quantizar com ONNX Runtime;
- avaliar TensorRT se usar GPU NVIDIA;
- avaliar CoreML se usar Apple Silicon.

## Como Alexa e Google Funcionam

Alexa e Google Assistant nao funcionam como um unico arquivo de voz local simples.

Fluxo geral:

```text
Usuario fala
        ↓
dispositivo envia/analisa audio
        ↓
servidores da Amazon/Google entendem
        ↓
modelo TTS gera resposta
        ↓
dispositivo toca audio
```

Grande parte do processamento e feita em cloud, com modelos otimizados, cache, infraestrutura proprietaria e hardware especializado.

Por isso, a melhor analogia para o projeto `super_Voz` e:

```text
criar seu proprio servidor TTS
```

E nao tentar colocar tudo diretamente dentro de Alexa/Google Assistant.

## Recomendacao Final

Rota recomendada:

1. Retomar/testar a voz neural a partir do checkpoint `epoch_2nd_00029.pth`.
2. Criar inferencia simples: texto para WAV.
3. Criar API TTS local com FastAPI.
4. Criar app web simples que chama:
   - Gemini API para IA;
   - Home Assistant API para automacao;
   - API TTS para falar com a voz neural.
5. Depois empacotar como app Android/desktop ou extensao.

Arquitetura alvo:

```text
App proprio
  ↓
Backend local
  ├── Gemini API
  ├── Home Assistant API
  └── super_Voz TTS API
        ↓
      audio com voz neural treinada
```

Conclusao:

E viavel criar um assistente proprio que conversa, responde perguntas do dia a dia e controla a casa usando Home Assistant, Gemini e a voz neural treinada. O caminho mais realista nao e substituir Alexa/Google, mas construir uma camada propria por cima do Home Assistant e usar sua voz neural como motor TTS.

## Historico Kaggle F5 09/06/2026

O erro `SIGBUS`/`OSError [Errno 5]` visto durante o upload de `model_last.pt` nao foi causado pelo dataset nem pelo aviso de metadata do Hugging Face. A causa foi concorrencia de I/O: o F5 regrava `model_last.pt` no mesmo caminho, enquanto o monitor podia enviar esse checkpoint vivo e o pacote podia apontar para ele por hardlink.

O runner Kaggle agora trabalha em duas fases: cria snapshot do checkpoint estavel, espera aparecer um checkpoint seguinte, envia somente o snapshot anterior, apaga esse snapshot apos upload confirmado e mantem apenas o checkpoint atual no working. Isso preserva espaco no Kaggle e evita upload do arquivo ainda em uso pelo treino.
