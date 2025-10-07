# Instalação simples para Google Colab
# Cole este código em uma célula do notebook e execute

!pip install --upgrade pip

# Instalar todas as dependências de uma vez
!pip install "transformers>=4.46,<4.47" "accelerate>=0.34,<0.35" "peft>=0.11.1" "bitsandbytes>=0.43.1" "datasets>=2.20.0" "evaluate>=0.4.2" "rouge-score>=0.1.2" "bert-score>=0.3.13" "sentencepiece>=0.2.0" "einops>=0.8.0" "torch>=2.3.0"

# Verificar instalação
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import transformers
print(f"Transformers version: {transformers.__version__}")

print("✅ Instalação concluída!")
