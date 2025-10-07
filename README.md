# Tech Challenge — Fine-tuning com AmazonTitles-1.3MM

## Objetivo
Treinar (fine-tuning) um foundation model para, dado um **título de produto** (e uma pergunta), **gerar a descrição completa** aprendida a partir do dataset **AmazonTitles-1.3MM** (`trn.json`).

## Dataset
- **Fonte:** AmazonTitles-1.3MM — usar `trn.json`
- **Campos:** `title` (título) e `content` (descrição)
- **Pré-processamento:** construção de pares *input → target*
  - **Input (`input_text`):** prompt em EN
  - **Target (`target_text`):** `"Description: " + content` (formato determinístico)

## Modelo e Técnica
- **Base:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **FT:** QLoRA 4-bit (nf4, double quant), LoRA (r=32, alpha=64, dropout=0.05)
- **Treino:** `max_steps=600`, LR `1e-4`, `cosine`, `warmup_ratio=0.03`,
  `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`,
  `max_length=512`, `bf16` (ou `fp16` no T4)
- **Merge:** após treino, `merge_and_unload()` em **fp16** para avaliação (modelo único)

## Reprodutibilidade (Colab)
1. **Prepare dataset** (`notebooks/01_prepare_dataset.ipynb`)
   - Salva em: `/content/drive/MyDrive/amazon_ft/cache/prepared_descfmt`
2. **Treine QLoRA** (`notebooks/02_sft_qLoRA_tinyllama.ipynb`)
   - Salva adapter em: `/content/drive/MyDrive/amazon_ft/outputs/tinyllama11b_descfmt_lora/adapter`
3. **Merge + Avaliação** (`notebooks/03_merge_and_eval.ipynb`)
   - Modelo mergeado (fp16): `/content/drive/MyDrive/amazon_ft/outputs/tinyllama11b_descfmt_merged_fp16`
   - Gera CSV lado a lado e JSON de métricas em `amazon_ft/outputs/`
4. **Demo** (`notebooks/04_demo_inference.ipynb`)
   - Geração interativa (antes vs depois)

## Resultados (execução rápida)
- **ROUGE-1** e **ROUGE-L** ↑ após FT em subset rápido (K=12).  
- Arquivos:
  - `side_by_side_tinyllama_descfmt_merged_quick.csv`
  - `metrics_tinyllama_descfmt_merged_quick.json`
  - (opcional) `improved_examples_tinyllama_descfmt_merged_quick.csv`

## Limitações
- Subset do dataset (10k de treino; quick eval K=12)
- Alvo determinístico (“Description: …”) para deixar o contraste visível
- Sem avaliação humana; pode ampliar K (50–100) e steps (600–1000) pra resultados melhores


## Como rodar (no Colab)

GPU: É necessário habilitar GPU no notebook

Preparar o dataset
!python prepare_dataset.py

Treinar o modelo
!python sft_qLoRA_tinyllama.py

Mergear
!python merge_fp16.py

Avaliar (rápido):
!python eval_quick.py
