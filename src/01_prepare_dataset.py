# === Preparação do dataset ===
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, json, random
from datasets import Dataset, DatasetDict

DS_DIR = "/content/drive/MyDrive/amazon_ft/cache/prepared_descfmt"

# Localiza o arquivo dataset
if os.path.exists(DS_DIR):
    print("Dataset já existe no Drive:", DS_DIR)
else:
    print("Criando dataset com formato 'Description: ...' em:", DS_DIR)
    candidates = [
        "/content/data/trn.json",
        "/content/drive/MyDrive/projeto_fase_3/trn.json",
        "/content/drive/MyDrive/trn.json",
    ]
    json_path = next((p for p in candidates if os.path.exists(p)), None)
    if json_path is None:
        raise FileNotFoundError(
            "Não encontrou o trn.json. Faça upload em /content/data/trn.json "
            "ou copie para /content/drive/MyDrive/projeto_fase_3/trn.json"
        )

    # Em ingles porque obteve melhores resultados
    QUESTION = "What is the complete product description? Answer strictly in the format: 'Description: ...' and nothing else."
    N_TRAIN, N_VAL, N_TEST = 10000, 1000, 1000

    def build_example(title, content):
        user = (
            "Based on the product title below, answer the question.\n"
            f"Question: {QUESTION}\n"
            f"Title: {title.strip()}"
        )
        target = "Description: " + content.strip()
        return {"input_text": user, "target_text": target}

    # Cria pares de entrada e alvo (input e target)
    pairs = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            title = (ex.get("title") or "").strip()
            content = (ex.get("content") or "").strip()
            if len(title) < 3 or len(content) < 20:
                continue
            pairs.append(build_example(title, content))

    random.seed(42)
    random.shuffle(pairs)
    need = N_TRAIN + N_VAL + N_TEST
    pairs = pairs[:need]

    # Separa em train/val/test
    train_pairs = pairs[:N_TRAIN]
    val_pairs   = pairs[N_TRAIN:N_TRAIN+N_VAL]
    test_pairs  = pairs[N_TRAIN+N_VAL:N_TRAIN+N_VAL+N_TEST]

    # Salva no Drive
    ds = DatasetDict({
        "train": Dataset.from_list(train_pairs),
        "validation": Dataset.from_list(val_pairs),
        "test": Dataset.from_list(test_pairs),
    })
    os.makedirs(DS_DIR, exist_ok=True)
    ds.save_to_disk(DS_DIR)
    
    print("Dataset salvo no Drive:", DS_DIR)