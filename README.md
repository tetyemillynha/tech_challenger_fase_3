# Tech Challenge — Fine-tuning com AmazonTitles-1.3MM

## Objetivo
Treinar (fine-tuning) um foundation model para, dado um **título de produto** (e uma pergunta), **gerar a descrição completa** aprendida a partir do dataset **AmazonTitles-1.3MM** (`trn.json`).

## Dataset: Seleção e Preparação

### Estrutura Original
O dataset `trn.json` contém produtos da Amazon com os seguintes campos:
- **`uid`**: Identificador único do produto
- **`title`**: Título do produto
- **`content`**: Descrição completa do produto
- **`target_ind`**: Índices de tokens relevantes (não utilizados neste projeto)
- **`target_rel`**: Relevância dos tokens (não utilizados neste projeto)

### Processo de Seleção e Filtragem
O processo de preparação do dataset (`src/01_prepare_dataset.py`) implementa os seguintes critérios de seleção:

1. **Filtros de Qualidade:**
   - Títulos com pelo menos 3 caracteres
   - Descrições com pelo menos 20 caracteres
   - Remoção de entradas com campos vazios ou nulos

2. **Divisão do Dataset:**
   - **Treino**: 10.000 amostras
   - **Validação**: 1.000 amostras  
   - **Teste**: 1.000 amostras
   - **Total**: 12.000 amostras selecionadas aleatoriamente

3. **Formatação dos Dados:**
   - **Prompt em inglês** (obteve melhores resultados):
     ```
     "Based on the product title below, answer the question.
     Question: What is the complete product description? Answer strictly in the format: 'Description: ...' and nothing else.
     Title: [TÍTULO_DO_PRODUTO]"
     ```
   - **Target determinístico**: `"Description: " + content`
   - **Seed fixo** (42) para reprodutibilidade

### Estrutura Final
Cada exemplo no dataset preparado contém:
- **`input_text`**: Prompt formatado com pergunta e título
- **`target_text`**: Descrição formatada com prefixo "Description: "

### Armazenamento
- **Local**: `/content/drive/MyDrive/amazon_ft/cache/prepared_descfmt`
- **Formato**: Dataset HuggingFace com divisões train/validation/test

## Processo de Fine-tuning: Detalhes Técnicos

### Arquitetura Base
- **Modelo**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Parâmetros**: 1.1 bilhão de parâmetros
- **Tipo**: Modelo causal de linguagem otimizado para chat

### Técnica QLoRA (Quantized LoRA)
O fine-tuning utiliza QLoRA para reduzir o uso de memória mantendo a qualidade:

#### Configuração de Quantização (4-bit)
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 quantization
    bnb_4bit_use_double_quant=True,      # Double quantization para melhor compressão
    bnb_4bit_compute_dtype=torch.float16 # Precisão de computação
)
```

#### Configuração LoRA
```python
LoraConfig(
    r=32,                    # Rank do adapter (capacidade de adaptação)
    lora_alpha=64,           # Fator de escala (r * 2 para estabilidade)
    lora_dropout=0.05,       # Dropout para regularização
    task_type="CAUSAL_LM",   # Tipo de tarefa
    target_modules="all-linear", # Todos os módulos lineares
    modules_to_save=["lm_head", "embed_tokens"] # Módulos preservados
)
```

### Parâmetros de Treinamento
```python
TrainingArguments(
    max_steps=1300,                    # Número total de steps
    num_train_epochs=1,                # Uma época completa
    learning_rate=1e-4,                # Taxa de aprendizado
    lr_scheduler_type="cosine",        # Agendador de LR cosseno
    warmup_ratio=0.03,                 # 3% dos steps para warmup
    gradient_accumulation_steps=8,     # Acumulação de gradientes
    per_device_train_batch_size=2,     # Batch size por dispositivo
    logging_steps=20,                  # Log a cada 20 steps
    save_steps=1300,                   # Salvar no final
    save_total_limit=1,               # Manter apenas 1 checkpoint
    bf16=False, fp16=True,            # Precisão mista (fp16 para T4)
    optim="adamw_torch",              # Otimizador AdamW
    report_to="none"                  # Sem logging externo
)
```

### Processo de Formatação
1. **Template de Chat**: Utiliza o template nativo do TinyLlama para formatação
2. **Tokenização**: Máximo de 512 tokens com truncamento
3. **Data Collator**: Language modeling sem MLM (masked language modeling)

### Otimizações de Memória
- **4-bit Quantization**: Reduz uso de memória em ~75%
- **Gradient Accumulation**: Simula batch size maior (efetivo: 16)
- **Mixed Precision**: fp16 para reduzir uso de VRAM
- **Device Map**: Carregamento automático em GPU/CPU conforme disponível

## Código-fonte do Processo de Fine-tuning

### Script Principal: `src/02_sft_qLoRA_tinyllama.py`
### Processo de Merge: `src/03_merge_fp16.py`

## Reprodutibilidade (Colab)
0. **Instala pacotes** (`src/00_colab_install_packages.py`)
1. **Prepare dataset** (`src/01_prepare_dataset.py`)
   - Salva em: `/content/drive/MyDrive/amazon_ft/cache/prepared_descfmt`
2. **Treine QLoRA** (`src/02_sft_qLoRA_tinyllama.py`)
   - Salva adapter em: `/content/drive/MyDrive/amazon_ft/outputs/tinyllama11b_descfmt_lora/adapter`
3. **Merge + Avaliação** (`src/03_merge_fp16.py`)
   - Modelo mergeado (fp16): `/content/drive/MyDrive/amazon_ft/outputs/tinyllama11b_descfmt_merged_fp16`
   - Gera CSV lado a lado e JSON de métricas em `amazon_ft/outputs/`
4. **Demo** (`src/04_eval_quick.py`)
   - Geração interativa (antes vs depois)

## Resultados (execução rápida)
- **ROUGE-1** e **ROUGE-L** ↑ após FT em subset rápido (K=12).  
- Arquivos:
  - `side_by_side_tinyllama_descfmt_merged_quick.csv`
  - `metrics_tinyllama_descfmt_merged_quick.json`
  - (opcional) `improved_examples_tinyllama_descfmt_merged_quick.csv`

## Resultados da Avaliação

### Arquivos de Resultados (`results/`)

Após executar o processo completo de fine-tuning e avaliação, são gerados os seguintes arquivos na pasta `results/`:

#### 1. **Comparação Lado a Lado** (`side_by_side_tinyllama_descfmt_merged_quick.csv`)

Este arquivo CSV contém uma comparação detalhada entre o modelo baseline (original) e o modelo fine-tuned para 12 amostras aleatórias do conjunto de teste.

**Estrutura do arquivo:**
- **`id`**: Índice da amostra no dataset de teste
- **`input_preview`**: Preview do prompt de entrada (truncado para visualização)
- **`reference_preview`**: Preview da descrição de referência (truncado)
- **`baseline`**: Resposta gerada pelo modelo TinyLlama original (sem fine-tuning)
- **`fine_tuned_merged`**: Resposta gerada pelo modelo fine-tuned

**Exemplo de comparação:**
```
Baseline: "Product Title: ATT Iphone 4 16GB BLACK CDMA - Premium Leather Case..."
Fine-tuned: "This is a premium leather case. It is made of the highest quality leather and is designed to fit perfectly to your phone..."
```

#### 2. **Métricas Quantitativas** (`metrics_tinyllama_descfmt_merged_quick.json`)

Arquivo JSON contendo as métricas ROUGE calculadas para ambos os modelos:

```json
{
  "rouge_base": {
    "rouge1": 0.141,    // ROUGE-1 do modelo baseline
    "rouge2": 0.051,    // ROUGE-2 do modelo baseline  
    "rougeL": 0.099,    // ROUGE-L do modelo baseline
    "rougeLsum": 0.103  // ROUGE-Lsum do modelo baseline
  },
  "rouge_ft": {
    "rouge1": 0.199,    // ROUGE-1 do modelo fine-tuned
    "rouge2": 0.054,    // ROUGE-2 do modelo fine-tuned
    "rougeL": 0.144,    // ROUGE-L do modelo fine-tuned
    "rougeLsum": 0.143  // ROUGE-Lsum do modelo fine-tuned
  }
}
```

### Análise dos Resultados

#### Melhorias Quantitativas
- **ROUGE-1**: Aumento de **14.1%** para **19.9%** (+41% de melhoria relativa)
- **ROUGE-L**: Aumento de **9.9%** para **14.4%** (+45% de melhoria relativa)
- **ROUGE-2**: Melhoria marginal de **5.1%** para **5.4%**

#### Melhorias Qualitativas
Observando o arquivo CSV, o modelo fine-tuned demonstra:

1. **Melhor Formatação**: Respostas mais estruturadas e coerentes
2. **Maior Relevância**: Descrições mais específicas sobre os produtos
3. **Menos Hallucination**: Redução de informações irrelevantes ou incorretas
4. **Consistência**: Respostas mais consistentes com o formato solicitado

#### Limitações dos Resultados
- **Subset Pequeno**: Avaliação realizada apenas em 12 amostras (K=12)
- **Dataset Limitado**: Treinamento em apenas 10k amostras do dataset original
- **Métricas Simples**: Apenas ROUGE utilizado (sem avaliação humana)
- **Formato Determinístico**: Target fixo "Description: ..." pode limitar criatividade

### Como Interpretar os Resultados

1. **ROUGE-1**: Mede sobreposição de palavras individuais (melhoria significativa)
2. **ROUGE-L**: Mede sobreposição de sequências mais longas (melhoria moderada)
3. **ROUGE-2**: Mede sobreposição de bigramas (melhoria marginal)

O aumento consistente em ROUGE-1 e ROUGE-L indica que o modelo fine-tuned está gerando descrições mais relevantes e estruturadas, enquanto o ROUGE-2 estável sugere que a melhoria não vem apenas de repetição de frases comuns.

## Limitações
- Subset do dataset (10k de treino; quick eval K=12)
- Alvo determinístico (“Description: …”) para deixar o contraste visível
- Sem avaliação humana; pode ampliar K (50–100) e steps (600–1000) pra resultados melhores


## Como rodar (no Colab)

- GPU: É necessário habilitar GPU no notebook

- Preparar o dataset
`!python 01_prepare_dataset.py`

- Treinar o modelo
`!python 02_sft_qLoRA_tinyllama.py`

- Mergear
`!python 03_merge_fp16.py`

- Avaliar (rápido):
`!python 04_eval_quick.py`
