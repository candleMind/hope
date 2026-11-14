import os

import pandas as pd
import torch

from fusion.model import load_checkpoint
from utils.constants import LIST_BDI_SYMPTOMS

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    checkpoint_path,
    supervised_path,
    unsupervised_path,
    run_path,
    run_name,
    hidden_size=64,
    top_k=1000,
    batch_size=4096,
):
    """
    Fuse supervised (query_1..21) and unsupervised (<symptom>_similarity) scores
    per doc_no, then write a TREC run file: `query_i Q0 doc_no rank score run_name`.
    """
    n = len(LIST_BDI_SYMPTOMS)
    sup_cols = [f"query_{i}" for i in range(1, n + 1)]
    # column order must match the one the fusion checkpoint was trained with
    unsup_cols = [f"{s}_similarity" for s in LIST_BDI_SYMPTOMS]

    sup = pd.read_csv(supervised_path).drop_duplicates("doc_no")
    unsup = pd.read_csv(unsupervised_path)
    df = sup.merge(unsup, left_on="doc_no", right_on="id", how="left")
    df[unsup_cols] = df[unsup_cols].fillna(0)

    model = load_checkpoint(checkpoint_path, n, n, n, hidden_size)
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            chunk = df.iloc[i : i + batch_size]
            x_sup = torch.tensor(chunk[sup_cols].values, dtype=torch.float32)
            x_unsup = torch.tensor(chunk[unsup_cols].values, dtype=torch.float32)
            out = model(x_sup.to(device), x_unsup.to(device))
            probs.append(torch.sigmoid(out).cpu())
    probs = torch.cat(probs).numpy()

    os.makedirs(os.path.dirname(run_path) or ".", exist_ok=True)
    with open(run_path, "w") as f:
        for i in range(n):
            top = pd.Series(probs[:, i], index=df["doc_no"]).nlargest(top_k)
            for rank, (doc_no, score) in enumerate(top.items(), 1):
                f.write(f"query_{i + 1} Q0 {doc_no} {rank} {score:.6f} {run_name}\n")
    print(f"Run file saved to {run_path}")
