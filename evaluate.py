import argparse

import numpy as np
import pandas as pd
import pytrec_eval
from sklearn.metrics import precision_recall_fscore_support


def load_run_file(run_file_path):
    """
    Loads a TREC-formatted run file into the dictionary structure required by pytrec_eval.
    """
    preds = {}
    with open(run_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            query_id, _, doc_id, _, score, _ = parts
            score = float(score)
            if query_id not in preds:
                preds[query_id] = {}
            preds[query_id][doc_id] = score
    return preds


def prepare_golds(gold_url):
    """
    Loads and prepares the ground truth data from a CSV file.
    The format is converted to the dictionary structure required by pytrec_eval.
    """
    gold_df = pd.read_csv(gold_url)
    if "text" in gold_df.columns:
        gold_df = gold_df.drop(columns=["text"])
    if "docid" in gold_df.columns:
        gold_df = gold_df.rename(columns={"docid": "doc_id"})

    golds = {}
    for _, row in gold_df.iterrows():
        doc_id = row["doc_id"]
        for col in gold_df.columns[1:]:
            query_id = col
            if query_id not in golds:
                golds[query_id] = {}
            # pytrec_eval expects integer relevance scores
            relevance = int(row[col] > 0)
            if relevance > 0:
                golds[query_id][doc_id] = relevance
    return golds


def call_metrics(preds, gold_url, thres=0.5):
    """
    Evaluates predictions against the ground truth using standard IR metrics.
    Calculates MAP, R-Prec, P@10, NDCG, and binary P/R/F1.
    """
    golds = prepare_golds(gold_url)
    # Ensure all queries in preds exist in golds to avoid errors
    queries_to_eval = {q: preds[q] for q in preds if q in golds}

    evaluator = pytrec_eval.RelevanceEvaluator(golds, {"map", "Rprec", "P_10", "ndcg"})
    result = evaluator.evaluate(queries_to_eval)

    for q in queries_to_eval:
        pred_list, gold_list = [], []
        # Align predictions and ground truth for sklearn metrics
        all_doc_ids = set(golds.get(q, {}).keys()) | set(preds.get(q, {}).keys())

        for docid in sorted(list(all_doc_ids)):
            gold_list.append(golds.get(q, {}).get(docid, 0))
            pred_score = preds.get(q, {}).get(docid, 0)
            pred_list.append(1 if pred_score >= thres else 0)

        p, r, f1, _ = precision_recall_fscore_support(
            gold_list, pred_list, average="binary", zero_division=0
        )
        result[q]["P"], result[q]["R"], result[q]["F1"] = p, r, f1

    # Calculate and add average scores
    avg_scores = {}
    all_metrics = set()
    for q_res in result.values():
        all_metrics.update(q_res.keys())

    for metric in all_metrics:
        avg_scores[metric] = np.mean([result[q].get(metric, 0) for q in result])

    result["AVG"] = avg_scores

    # Format for output
    df_result = pd.DataFrame(result).T
    for col in df_result.columns:
        df_result[col] = df_result[col].apply(lambda x: round(x, 4))

    print("Evaluation Results:")
    print(df_result)
    df_result.to_csv("results.tsv", sep="\t")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate a TREC run file.")
    parser.add_argument(
        "--gold_file",
        type=str,
        required=True,
        help="Path to the ground truth qrels file.",
    )
    parser.add_argument(
        "--run_file",
        type=str,
        required=True,
        help="Path to the TREC-formatted run file to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary classification metrics (P, R, F1).",
    )
    args = parser.parse_args()

    print(f"Evaluating run file: {args.run_file}")
    print(f"Against gold standard: {args.gold_file}")

    preds = load_run_file(args.run_file)
    call_metrics(preds, args.gold_file, thres=args.threshold)


if __name__ == "__main__":
    main()
