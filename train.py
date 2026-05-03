import torch
import logging
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, classification_report
import numpy as np

# ====================== 配置 ======================
MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"
NUM_LABELS = 10  # 根据意图类别调整
MAX_LEN = 256
EPOCHS = 5
FOLDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================== 工具函数 ======================
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"macro_f1": macro_f1}

# ====================== 5-Fold 训练 ======================
def train_kfold(data_path):
    # 加载清洗后的数据
    df = load_cleaned_data(data_path)
    dataset = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        logger.info(f"===== Fold {fold + 1}/{FOLDS} =====")
        
        train_ds = dataset.select(train_idx)
        val_ds = dataset.select(val_idx)

        def tokenize_func(examples):
            return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN, padding="max_length")
        
        train_tok = train_ds.map(tokenize_func, batched=True)
        val_tok = val_ds.map(tokenize_func, batched=True)
        collator = DataCollatorWithPadding(tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        )

        # 动态类别权重（长尾分布优化）
        class_weights = compute_class_weights(df["label"])
        model.classifier.weight = torch.nn.Parameter(torch.tensor(class_weights, dtype=torch.float32))

        args = TrainingArguments(
            output_dir=f"./output_fold_{fold}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=EPOCHS,
            fp16=True,
            logging_steps=10,
            metric_for_best_model="macro_f1",
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics
        )

        trainer.train()
        logger.info(f"Fold {fold+1} Best Macro-F1: {trainer.evaluate()['eval_macro_f1']:.4f}")

if __name__ == "__main__":
    train_kfold("./data/labeled_data.csv")
