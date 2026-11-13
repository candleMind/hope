import os
import pickle

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier


def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def euclidean_distance(a, b):
    return torch.nn.functional.pairwise_distance(a, b, p=2)


def manhattan_distance(a, b):
    return torch.nn.functional.pairwise_distance(a, b, p=1)


def minkowski_distance(a, b):
    return torch.nn.functional.pairwise_distance(a, b, p=float("inf"))


def similarity(a, b, metric="cosine"):
    if metric == "cosine":
        return cosine_similarity(a, b)
    elif metric == "euclidean":
        return euclidean_distance(a, b)
    elif metric == "manhattan":
        return manhattan_distance(a, b)
    elif metric == "minkowski":
        return minkowski_distance(a, b)
    else:
        raise ValueError(f"Invalid metric: {metric}")


def get_top_similar(
    reranked_df: pd.DataFrame, queries, output_path, aggegrate="max_each"
):
    if aggegrate == "mean":
        reranked_df["similarity"] = reranked_df.filter(like="similarity").mean(axis=1)
        reranked_df = reranked_df.sort_values("similarity", ascending=False)
        reranked_df = reranked_df.drop_duplicates(subset=["docid"], keep="first")
        reranked_df = reranked_df.head(1000)
    elif aggegrate == "max_each":
        results = []
        for i, query in enumerate(queries):
            query_df = reranked_df[["docid", "doc_text", f"similarity_{i}"]]
            query_df = query_df.rename(columns={f"similarity_{i}": "similarity"})
            query_df = query_df.sort_values("similarity", ascending=False)
            # top 10% of each query
            # query_df = query_df.head(int(len(query_df) * 0.1))
            results.append(query_df)

        results = pd.concat(results)
        results = results.sort_values("similarity", ascending=False)
        results = results.drop_duplicates(subset=["docid"], keep="first")
        results = results.head(1000)

    return results


def reranking(
    df: pd.DataFrame,
    embeddings,
    model: SentenceTransformer,
    queries,
    metric="cosine",
):
    # Load query
    query_embedding = model.encode(queries, convert_to_tensor=True)
    print("query_embedding", query_embedding.shape)

    # Calculate similarity by batch
    batch_size = 96
    similarities = []
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i : i + batch_size]
        sim = similarity(
            query_embedding.unsqueeze(1),
            torch.Tensor(batch).unsqueeze(0).to(query_embedding.device),
            metric=metric,
        )
        similarities.append(sim)
    similarities = torch.cat(similarities, dim=1)
    if metric != "cosine":
        # normalize and invert the distance
        similarities = 1 - similarities / similarities.max()

    # similarity for each query
    for i, query in enumerate(queries):
        df[f"similarity_{i}"] = similarities[i].cpu().numpy()

    df["avg_similarity"] = df[[f"similarity_{i}" for i in range(len(queries))]].mean(
        axis=1
    )
    df = df.sort_values("avg_similarity", ascending=False)

    return df


def reranking_tfidf(
    df: pd.DataFrame, tfidf_matrix, vectorizer, queries, metric="cosine"
):
    # Transform queries into TF-IDF vectors
    query_tfidf = vectorizer.transform(queries)
    # to numpy array
    query_tfidf = query_tfidf.toarray()
    tfidf_matrix = tfidf_matrix.toarray()

    print("query_tfidf", query_tfidf.shape)
    print("tfidf_matrix", tfidf_matrix.shape)

    # Calculate similarity by batch
    batch_size = 128
    similarities = []
    for i in range(0, len(tfidf_matrix), batch_size):
        batch = tfidf_matrix[i : i + batch_size]
        sim = similarity(
            torch.Tensor(query_tfidf).unsqueeze(1),
            torch.Tensor(batch).unsqueeze(0),
            metric=metric,
        )
        similarities.append(sim)
    similarities = torch.cat(similarities, dim=1)
    if metric != "cosine":
        # normalize and invert the distance
        similarities = 1 - similarities / similarities.max()

    # similarity for each query
    for i, query in enumerate(queries):
        df[f"similarity_{i}"] = similarities[i].cpu().numpy()

    df["avg_similarity"] = df[[f"similarity_{i}" for i in range(len(queries))]].mean(
        axis=1
    )
    df = df.sort_values("avg_similarity", ascending=False)

    return df


"""
For kNN
"""


def k_means_clustering(embeddings, n_clusters=3, init=None):
    if init:
        print(f"Using custom initialization with shape: {init.shape}")
        kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=42)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_


def classify_knn(centers, labels, sentence_embeddings, k=3):
    print(centers.shape, labels.shape)
    knn = KNeighborsClassifier(n_neighbors=k)
    # convert symptoms to integers
    knn.fit(centers, labels)
    predictions = knn.predict(sentence_embeddings)
    # save knn model
    os.makedirs("checkpoints", exist_ok=True)
    with open(f"checkpoints/knn_model_k{k}.pkl", "wb") as f:
        pickle.dump(knn, f)
    return predictions


def calculate_distances(sentence_embedding, query_embeddings, metric="minkowski"):
    # return distance between sentence embedding and nearest query embedding
    return (
        similarity(
            torch.Tensor([sentence_embedding]),
            torch.Tensor(query_embeddings),
            metric=metric,
        )
        .min()
        .numpy()
    )


def get_top_sentences_for_symptoms(
    df: pd.DataFrame, symptom, query_embeddings, top_k, distance_metric="euclidean"
):
    symptom_df = df[df["symptom"] == symptom].copy()
    symptom_df["distance"] = df["embedding"].apply(
        lambda x: calculate_distances(x, query_embeddings, distance_metric)
    )
    print(symptom_df.head())
    symptom_df["similarity"] = 1 - symptom_df["distance"] / symptom_df["distance"].max()
    # symptom_df = symptom_df.sort_values("similarity", ascending=False)
    # symptom_df = symptom_df.head(top_k)
    symptom_df = symptom_df.nlargest(top_k, "similarity")
    return symptom_df[["docid", "doc_text", "similarity"]]


def inverse_distance(X, Y):
    return 1.0 / (1.0 + cdist(X, Y, metric="euclidean"))


def seed_max(X, seeds):
    return inverse_distance(X, seeds).max(axis=1)


def sym_max(D, labels, n):
    out = np.zeros((D.shape[0], n))
    for j in range(n):
        out[:, j] = D[:, labels == j].max(axis=1)
    return out


def symptom_sims(X, C, labels, n):
    D = inverse_distance(X, C)
    return D if len(C) == n else sym_max(D, labels, n)
