# Registro de Alterações: run_colab_super_voz.ipynb

## [2026-05-31]
- Inicialização do registro de alterações.
- Implementação de verificação de hardware (GPU) e correção automática do `onnxruntime-gpu`.
- Adicionado passo "--- 6. Verificando Hardware e Aceleração ---" no notebook.
- O notebook continua delegando a instalação completa ao `scripts/run_colab_styletts2.py`; a partir desta revisão, esse script não instala `resemble-enhance` por padrão para evitar incompatibilidade PyTorch/CUDA no Colab.
- Para testar o enhancer manualmente no Colab, defina `SUPER_VOZ_ENABLE_RESEMBLE=1` antes de iniciar o pipeline.

## [2026-05-31] Atualização do notebook (V8, superada)
- Adicionada seção visível "Política do Resemble Enhance" na célula markdown inicial.
- A célula one-click agora define `SUPER_VOZ_ENABLE_RESEMBLE=0` por padrão antes de montar Drive e iniciar o pipeline.
- Adicionado passo "--- 7. Política do Resemble Enhance ---" no log do Colab para deixar claro se o enhancer está desligado ou em teste manual.
- Essa política foi superada pela V9, que voltou a ativar Resemble Enhance por padrão com uso GPU corrigido.

## [2026-05-31] Resemble Enhance GPU padrão
- A célula one-click agora define `SUPER_VOZ_ENABLE_RESEMBLE=1` por padrão.
- A seção "Política do Resemble Enhance" foi atualizada para indicar uso GPU com `device="cuda"`.
- O log do passo "--- 7. Política do Resemble Enhance ---" agora informa quando o enhancer está ativo em GPU.
- Para desligar o tratamento por IA e preservar originais, o usuário deve trocar a variável para `0`.

## [2026-06-02] Progresso de treino e retomada
- O script chamado pelo notebook agora mostra uma barra compacta de progresso do fine-tuning por epoca/passo.
- Saídas extensas do StyleTTS2 são filtradas no console; o log completo continua em `Models/super_Voz/train.log`.
- Ao reiniciar o notebook após limite de GPU do Colab, o pipeline procura o último `Models/super_Voz/epoch_2nd_*.pth` e retoma dele automaticamente.
- `save_freq` foi ajustado para `1`, salvando checkpoint a cada epoca para reduzir perda de progresso.
