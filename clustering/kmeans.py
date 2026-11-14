import os

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clustering.noise_filter import keep_mask
from clustering.similarity import symptom_sims
from data_processing.text_preprocess import sentence_segmentation
from utils.constants import CHECKPOINT_PATH, DESCRIPTION_FOLDER, RELEVANT_THRESHOLD


def load_avg_questionnaire_vectors(description_folder, model):
    """
    For each symptom, average all questionnaire vectors to get one vector per symptom.
    Returns:
        - symptom_to_vector: dict of symptom -> avg vector
    """
    symptom_to_vector = {}
    for fname in os.listdir(description_folder):
        if fname.endswith(".txt"):
            symptom = fname.replace(".txt", "")
            with open(os.path.join(description_folder, fname), "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                vectors = model.encode(lines)
                avg_vec = np.mean(vectors, axis=0)
                symptom_to_vector[symptom] = avg_vec
    return symptom_to_vector


def load_depression_model(model_checkpoint):
    """
    Load a fine-tuned depression model from checkpoint using PEFT
    Returns:
        - model: A SentenceTransformer-compatible wrapper for the loaded model
    """
    print(f"Loading depression model from checkpoint: {model_checkpoint}")

    # Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "rafalposwiata/deproberta-large-depression"
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "rafalposwiata/deproberta-large-depression",
        num_labels=21,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    # Load the fine-tuned model with PEFT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(
        base_model, model_checkpoint, device_map="auto"
    ).to(device)

    # Create a wrapper class to make it compatible with SentenceTransformer's encode method
    class ModelWrapper:
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.model.eval()

        def encode(self, sentences, batch_size=8, show_progress_bar=False, device=None):
            if device is None:
                device = self.device

            # Handle both single sentence and list of sentences
            if isinstance(sentences, str):
                sentences = [sentences]

            all_embeddings = []

            # Process in batches
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    # Extract the final hidden state or logits as embeddings
                    # Using the logits as embeddings for simplicity
                    embeddings = outputs.logits.cpu().numpy()
                    all_embeddings.append(embeddings)

            # Concatenate all batch embeddings
            all_embeddings = np.vstack(all_embeddings)
            return all_embeddings

    # Return wrapped model
    wrapped_model = ModelWrapper(model, tokenizer, device)
    return wrapped_model


def kmeans_clustering(embeddings, questionnaire_symptom_vectors, n_clusters=10):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    # Normalize for cosine similarity
    embeddings_norm = normalize(embeddings)
    centroids_norm = normalize(np.array(questionnaire_symptom_vectors))

    kmeans = KMeans(
        n_clusters=n_clusters, init=centroids_norm, n_init=1
    )  # n_init=1 to use provided init
    kmeans.fit(embeddings_norm)
    return kmeans.labels_, kmeans.cluster_centers_, embeddings_norm


def find_symptom_sentences(
    data_path,
    description_folder,
    model_path,
    topk=1000,
    rel_thres=RELEVANT_THRESHOLD,
    use_depression_model=False,
    checkpoint_path=CHECKPOINT_PATH,
    output_dir=None,
    id_column="doc_no",
    text_column="doc_text",
):
    # Load data
    df = pd.read_csv(data_path)

    # filter out rows with  no value >= rel_thres
    pred_cols = [col for col in df.columns if col.startswith(("pred_", "query_"))]
    df = df[df[pred_cols].gt(rel_thres).any(axis=1)].reset_index(drop=True)
    print(f"Filtered to {len(df)} texts with rel_thres > {rel_thres}")

    # split doc_text into sentences
    df["sentence"] = df[text_column].apply(sentence_segmentation)
    df = df.explode("sentence").reset_index(drop=True)
    print(f"Exploded to {len(df)} sentences after sentence segmentation")

    # Load model - either depression model or sentence transformer
    if use_depression_model and checkpoint_path:
        model = load_depression_model(checkpoint_path)
        model_name = "_".join(
            checkpoint_path.split("/")[-2:]
        )  # Use last two parts of path as model name
    else:
        model = SentenceTransformer(model_path, trust_remote_code=True)
        model_name = model_path.split("/")[-1]
    # Encode all sentences
    embeddings = model.encode(
        df["sentence"].tolist(),
        show_progress_bar=True,
        batch_size=16,
        device="cuda",
    )
    # Load avg questionnaire vectors
    symptom_to_vector = load_avg_questionnaire_vectors(description_folder, model)
    if output_dir is None:
        output_dir = "."

    num_clusters = len(symptom_to_vector)
    print(f"Loaded {num_clusters} symptom avg vectors from {description_folder}")

    symptom_names = list(symptom_to_vector.keys())
    symptom_vectors = [symptom_to_vector[name] for name in symptom_names]

    # apply k-means clustering
    labels, cluster_centers, embeddings_norm = kmeans_clustering(
        embeddings, symptom_vectors, n_clusters=num_clusters
    )
    print("k-means labels:", labels.shape)
    # map cluster id to symptom
    cluster_id_to_symptom = {i: symptom for i, symptom in enumerate(symptom_names)}

    # for each cluster, save topk closest sentences to cluster center
    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)

    for cluster_id in range(num_clusters):
        symptom = cluster_id_to_symptom[cluster_id]
        distances = cdist(
            embeddings_norm, cluster_centers[cluster_id].reshape(1, -1), metric="cosine"
        ).flatten()

        # Sort by distance (ascending order - smallest distance first)
        top_idx = np.argsort(distances)[:topk]

        rows = [df.iloc[idx] for idx in top_idx]
        out_df = pd.DataFrame(rows)

        # Add Minkowski distance scores
        out_df["minkowski_distance"] = distances[top_idx]

        # Optional: Convert to similarity (e.g., inverse distance)
        out_df["similarity"] = 1 / (1 + out_df["minkowski_distance"])

        # scale similarity to [0, 1]
        out_df["similarity"] = (out_df["similarity"] - out_df["similarity"].min()) / (
            out_df["similarity"].max() - out_df["similarity"].min()
        )

        out_df = out_df.drop_duplicates(subset=[id_column], keep="first").reset_index(
            drop=True
        )

        out_df[
            [
                id_column,
                text_column,
                "sentence",
                "minkowski_distance",
                "similarity",
            ]
        ].to_csv(f"{output_dir}/{model_name}/{symptom}.csv", index=False)
        print(f"Saved top {len(out_df)} for cluster {cluster_id} ({symptom})")


def main(data_path, description_folder, output_dir, model_path, topk=5000):
    """Main function to run the kmeans-like sentence collection."""
    # Run for modernbert-embed-base
    find_symptom_sentences(
        data_path,
        description_folder,
        model_path=model_path,
        output_dir=output_dir,
        topk=topk,
    )


MIN_KEEP = 500


def fit_kmeans(X, init):
    km = KMeans(n_clusters=len(init), init=init, n_init=1, max_iter=300).fit(X)
    return km.cluster_centers_


def doc_frame(docs, sims, n):
    df = pd.DataFrame(sims, columns=[f"s{j}" for j in range(n)])
    df["doc"] = docs
    return df.groupby("doc").max()


def model_scores(X, docs, avg_c, five, thr, K, labels, n, min_keep=MIN_KEEP):
    X = normalize(np.asarray(X))
    if thr is None:
        keep, share = np.ones(len(X), dtype=bool), 1.0
    else:
        keep, share = keep_mask(X, five, thr)
    if int(keep.sum()) < min_keep:
        return None, share
    init = avg_c if K == n else five
    C = fit_kmeans(X[keep], init)
    S = symptom_sims(X, C, labels, n)
    return doc_frame(np.asarray(docs)[keep], S[keep], n), share


if __name__ == "__main__":
    data_path = "data/labels/relevant_predictions.csv"
    output_dir = "outputs/symptom_sentences"

    # Run for nomic-embed-text-v1.5
    main(
        data_path,
        DESCRIPTION_FOLDER,
        output_dir,
        model_path="nomic-ai/nomic-embed-text-v1.5",
    )

    # run for modernbert-embed-base
    main(data_path, DESCRIPTION_FOLDER, output_dir, model_path="modernbert-embed-base")
