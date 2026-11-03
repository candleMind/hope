import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.constants import PROCESSED_DATA_DIR

from .model import load_checkpoint


def load_data(data_dir, id_column="doc_no", text_column="doc_text"):
    """
    Load data from the specified directory.
    Assumes that the directory contains CSV files.
    """
    data_files = glob(os.path.join(data_dir, "*.csv"))
    data = []

    for file in data_files:
        df = pd.read_csv(file)
        df["file"] = file
        data.append(df[[id_column, text_column, "file"]])

    all_df = pd.concat(data, ignore_index=True)
    all_df = all_df.dropna(subset=[text_column])  # Remove NaN text
    all_df = all_df[
        all_df[text_column].astype(str).str.len() >= 3
    ]  # Ensure string type
    return all_df


def infer(model, tokenizer, texts, batch_size=32, device="cuda"):
    """
    Run inference on the provided texts using the specified model.
    """
    # eval by batch
    preds = []
    with torch.no_grad():  # Move outside the loop
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i : i + batch_size]

            # Tokenize with optimized settings
            inputs = tokenizer(
                batch_texts,
                padding=True,  # Dynamic padding instead of max_length
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )

            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)

            # Convert to predictions and move to CPU immediately
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()
            preds.append(predictions)

            # Clear GPU cache periodically
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

    # Use concatenate instead of array to handle different batch shapes
    preds = np.concatenate(preds, axis=0)
    return preds


def save_predictions(all_df, output_dir):
    grouped = all_df.groupby("file")

    for file_name, group_df in grouped:
        group_df.to_csv(os.path.join(output_dir, file_name), index=False)
        print(f"Saved {file_name}: {len(group_df)} rows")

    print(f"Total files saved: {len(grouped)}")


def main(checkpoint_path):
    print("Loading model and tokenizer...")
    model = load_checkpoint(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(
        "rafalposwiata/deproberta-large-depression"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Model and tokenizer loaded successfully.")

    print("Loading data...")
    all_df = load_data(PROCESSED_DATA_DIR)
    texts = all_df["doc_text"].tolist()

    if not texts:
        print("No texts to process. Exiting.")
        return
    else:
        print(f"Loaded {len(texts)} texts for inference.")

    print("Running inference...")
    preds = infer(model, tokenizer, texts, device=device)

    columns = [f"query_{i}" for i in range(1, 22)]
    all_df.loc[:, columns] = preds

    save_predictions(all_df, "outputs/extractor_predictions")


if __name__ == "__main__":
    checkpoint_path = "path/to/your/checkpoint"  # Update with your checkpoint path
    main(checkpoint_path)
