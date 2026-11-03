import math
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, Trainer, TrainingArguments

from extractor.model import get_model
from utils.constants import LABELED_SENTENCE_DIR

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 32
NUM_EPOCHS = 20
WARMUP_RATIO = 0.1


def preprocess(df: pd.DataFrame, id_column="docid", text_column="text"):
    df = df.copy()
    df = df.set_index(id_column)
    df = df.replace(-1, 0)

    # label: array of all symptom columns (except text)
    df["label"] = df.drop([text_column], axis=1).values.tolist()
    df["label"] = df["label"].apply(lambda x: [float(val) for val in x])

    return df[[text_column, "label"]]


def prepare_dataset(data_dir, id_column="docid", text_column="text"):
    data_23 = pd.read_csv(os.path.join(data_dir, "erisk2023_majority.csv"))
    data_24 = pd.read_csv(os.path.join(data_dir, "erisk2024_majority.csv"))

    # 90/10 train/val split over the pooled 2023 + 2024 labeled data
    data = pd.concat([data_23, data_24]).reset_index(drop=True)
    val_df = data.sample(frac=0.1, random_state=SEED)
    train_df = data.drop(val_df.index).reset_index(drop=True)

    train_df = preprocess(train_df, id_column=id_column, text_column=text_column)
    val_df = preprocess(val_df, id_column=id_column, text_column=text_column)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, preserve_index=False),
        }
    )

    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to multi-label predictions
    pred_probas = torch.sigmoid(torch.tensor(logits))
    predictions = (pred_probas > 0.5).int().numpy()

    report = classification_report(
        labels, predictions, output_dict=True, zero_division=0
    )
    return {
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
    }


def train_model(model, dataset):
    # warmup_ratio is rejected by the TrainingArguments of the Kaggle image,
    # so warmup is set in steps: the same value Trainer computes from the ratio
    # (ceil(ratio * total optimizer steps)).
    # With several GPUs, Trainer multiplies the per-device batch by the GPU count
    # (DataParallel), so split BATCH_SIZE across the GPUs to keep the total at 32.
    n_gpu = max(1, torch.cuda.device_count())
    per_device_batch = BATCH_SIZE // n_gpu
    effective_batch = per_device_batch * n_gpu
    steps_per_epoch = math.ceil(len(dataset["train"]) / effective_batch)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = math.ceil(total_steps * WARMUP_RATIO)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=64,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    return trainer


def main(
    data_dir,
    model_path="rafalposwiata/deproberta-large-depression",
    text_column="text",
    id_column="docid",
):
    dataset = prepare_dataset(data_dir, id_column=id_column, text_column=text_column)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column], truncation=True, padding="max_length", max_length=256
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    model = get_model(model_path=model_path, num_labels=21)

    trainer = train_model(model, tokenized_datasets)
    trainer.train()
    # the last step is not always a save step, so keep the final adapter explicitly
    trainer.save_model("./results/final")

    torch.cuda.empty_cache()
    trainer.evaluate()


if __name__ == "__main__":
    main(LABELED_SENTENCE_DIR)
