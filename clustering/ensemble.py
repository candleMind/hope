import os

import numpy as np
import pandas as pd

from clustering.embedding import encode_statements, read_statements
from clustering.kmeans import MIN_KEEP, model_scores
from utils.constants import KMEANS_K, LIST_BDI_SYMPTOMS, NOISE_THRESHOLD, RELEVANT_THRESHOLD

# substring match, as in the original submission (so "me" also hits "some")
FIRST_PERSON = (
    "i ",
    "me",
    "my",
    "mine",
    "myself",
    "i'm",
    "i've",
    "i'll",
    "i'd",
    "im ",
    "am ",
)


def has_first_person(text):
    text = str(text).lower()
    return any(p in text for p in FIRST_PERSON)


def main(symptom_sentence_dir, output_path, id_column="doc_no", first_person_only=True):
    """
    Average per-symptom similarity across embedding models (subdirs of
    symptom_sentence_dir), then merge into one wide CSV:
    `id` + `<symptom>_similarity` for every BDI symptom. Docs missing from a
    symptom's top-k get 0.
    """
    model_dirs = [
        os.path.join(symptom_sentence_dir, d)
        for d in sorted(os.listdir(symptom_sentence_dir))
        if os.path.isdir(os.path.join(symptom_sentence_dir, d))
    ]
    merged = None
    for symptom in LIST_BDI_SYMPTOMS:
        dfs = [pd.read_csv(os.path.join(d, f"{symptom}.csv")) for d in model_dirs]
        if first_person_only:
            dfs = [df[df["sentence"].apply(has_first_person)] for df in dfs]
        sim = (
            pd.concat(dfs)
            .groupby(id_column)["similarity"]
            .mean()
            .rename(f"{symptom}_similarity")
            .reset_index()
        )
        merged = sim if merged is None else merged.merge(sim, on=id_column, how="outer")

    merged = merged.fillna(0).rename(columns={id_column: "id"})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved {len(merged)} rows to {output_path}")


def unsupervised_scores(models, docs_all, thr, K, labels, n, min_keep=MIN_KEEP):
    """models: name -> dict(X=, docs=, avg_c=, five=). Returns (units x n matrix, kept shares) or (None, shares)."""
    frames, shares = [], {}
    for name, m in models.items():
        frame, share = model_scores(m["X"], m["docs"], m["avg_c"], m["five"], thr, K, labels, n, min_keep)
        shares[name] = round(share, 4)
        if frame is None:
            return None, shares
        frames.append(frame.reindex(docs_all).fillna(0.0).values)
    return sum(frames) / len(frames), shares


def run_unsupervised(relevant_file, output_file, encoders, symptoms, statements_dir,
                     thr=NOISE_THRESHOLD, K=KMEANS_K,
                     rel_thres=None, segment=None, text_column="doc_text", id_column="doc_no", min_keep=MIN_KEEP,
                     first_person=False):
    if rel_thres is None:
        rel_thres = RELEVANT_THRESHOLD
    if segment is None:
        from data_processing.text_preprocess import sentence_segmentation as segment
    df = pd.read_csv(relevant_file)
    pred_cols = [c for c in df.columns if c.startswith(("pred_", "query_"))]
    if pred_cols:
        df = df[df[pred_cols].gt(rel_thres).any(axis=1)]
    df = df.dropna(subset=[text_column]).reset_index(drop=True)
    df["sentence"] = df[text_column].astype(str).apply(segment)
    df = df.explode("sentence").dropna(subset=["sentence"]).reset_index(drop=True)
    docs_all = np.unique(df[id_column].to_numpy())
    if first_person:
        df = df[df["sentence"].map(has_first_person)].reset_index(drop=True)
    texts = df["sentence"].astype(str).tolist()
    docs = df[id_column].to_numpy()
    statements = read_statements(symptoms, statements_dir)
    labels = np.repeat(np.arange(len(symptoms)), 5)
    models = {}
    for name, encode in encoders.items():
        avg_c, five = encode_statements(encode, statements, symptoms)
        models[name] = dict(X=encode(texts), docs=docs, avg_c=avg_c, five=five)
    scores, shares = unsupervised_scores(models, docs_all, thr, K, labels, len(symptoms), min_keep)
    if scores is None:
        raise RuntimeError(f"fewer than {min_keep} sentences kept for at least one model: {shares}")
    out = pd.DataFrame(scores, columns=[f"{s}_similarity" for s in symptoms])
    out.insert(0, "id", docs_all)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    out.to_csv(output_file, index=False)
    print(f"Saved {len(out)} units to {output_file}; kept share per model: {shares}")
    return shares


if __name__ == "__main__":
    main("outputs/symptom_sentences", "outputs/unsupervised_preds.csv")
