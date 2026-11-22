import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


def read_data(data_path, text_column="doc_text", id_column="docid"):
    print(f"Reading data from {data_path}")

    if os.path.isdir(data_path):
        dfs = []
        for file in os.listdir(data_path):
            if file.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(data_path, file))
                    df = df.dropna(subset=[text_column])
                    df[text_column] = df[text_column].astype(str)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading file {file}: {e}")

        concat_dfs = pd.concat(dfs, ignore_index=True)

    elif data_path.endswith(".csv"):
        concat_dfs = pd.read_csv(data_path)
        concat_dfs = concat_dfs.dropna(subset=[text_column])
        concat_dfs[text_column] = concat_dfs[text_column].astype(str)

    # drop text with less than 5 characters
    concat_dfs = concat_dfs[concat_dfs[text_column].str.len() >= 5]
    # drop text that only contains numbers
    concat_dfs = concat_dfs[~concat_dfs[text_column].str.isnumeric()]
    return concat_dfs.reset_index(drop=True)


def encoding(model: SentenceTransformer, device, df, batch_size=64):
    """Encode text data using SentenceTransformer model."""
    print(f"Using device: {device}")

    # Load data and model
    sentences = df["doc_text"].values

    embeddings = model.encode(
        sentences, show_progress_bar=True, batch_size=batch_size, device=device
    )
    print(embeddings.shape)

    return embeddings


def read_statements(symptoms, folder):
    if not folder or folder.startswith("<"):
        raise ValueError("Set LLM_DESCRIPTION_FOLDER in utils/constants.py to the folder of the LLM descriptions.")
    out = {}
    for s in symptoms:
        path = os.path.join(folder, f"{s}.txt")
        with open(path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) != 5:
            raise ValueError(f"{path}: expected 5 statements, found {len(lines)}")
        out[s] = lines
    return out


def encode_statements(encode, statements, symptoms):
    raw = [np.asarray(encode(statements[s])) for s in symptoms]
    avg_c = normalize(np.array([r.mean(axis=0) for r in raw]))
    five = normalize(np.vstack(raw))
    return avg_c, five


def make_st_encoder(model_path, prefix="", device=None, batch_size=64):
    st = SentenceTransformer(model_path, trust_remote_code=True, device=device)
    st.max_seq_length = 256

    def encode(texts):
        return st.encode([prefix + t for t in texts], batch_size=batch_size, show_progress_bar=True)

    return encode
