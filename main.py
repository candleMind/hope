import argparse
import os

from clustering.embedding import make_st_encoder
from clustering.ensemble import run_unsupervised
from clustering.relevant_filter import filter_relevant_predictions
from data_processing.create_labeled_data import main as create_labeled_data_main
from data_processing.trec_loader import main as trec_loader_main
from evaluate import call_metrics, load_run_file
from extractor.infer import main as extractor_infer_main
from fusion.run import main as fusion_main
from utils.constants import (
    KMEANS_K,
    LIST_BDI_SYMPTOMS,
    LLM_DESCRIPTION_FOLDER,
    NOISE_THRESHOLD,
    NOMIC_TASK_PREFIX,
)


def main():
    parser = argparse.ArgumentParser(description="Run the HOPE pipeline.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model checkpoint for the extractor.",
    )
    parser.add_argument(
        "--fusion_checkpoint",
        type=str,
        required=True,
        help="Path to the trained fusion model checkpoint (.pt).",
    )
    parser.add_argument(
        "--gold_file",
        type=str,
        required=True,
        help="Path to the ground truth qrels file for evaluation.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="hope_run",
        help="A name for the final run file.",
    )
    parser.add_argument(
        "--first_person",
        action="store_true",
        help="First-person sentence filter before the unsupervised step (E68; not in the main configuration).",
    )
    args = parser.parse_args()

    print("--- Starting HOPE Pipeline ---")

    # 1. Data Preparation
    print("\n--- Step 1: Data Preparation ---")
    if not os.path.exists("data/erisk25-t1-processed"):
        print("Running TREC loader...")
        trec_loader_main()
    else:
        print("Processed TREC data found, skipping TREC loader.")

    if not os.path.exists("data/labels/labeled_data_with_text.csv"):
        print("Creating labeled data...")
        create_labeled_data_main()
    else:
        print("Labeled data found, skipping creation.")

    # 2. Supervised Symptom Extraction
    print("\n--- Step 2: Supervised Symptom Extraction ---")
    extractor_output_dir = "outputs/extractor_predictions"
    if not os.path.exists(extractor_output_dir) or not os.listdir(extractor_output_dir):
        print("Running supervised extractor...")
        extractor_infer_main(args.checkpoint_path)
    else:
        print("Extractor predictions found, skipping inference.")

    # 3. Unsupervised Semantic Clustering
    print("\n--- Step 3: Unsupervised Semantic Clustering ---")
    relevant_preds_file = "data/labels/relevant_predictions.csv"
    if not os.path.exists(relevant_preds_file):
        print("Filtering relevant predictions...")
        filter_relevant_predictions(output_file=relevant_preds_file)
    else:
        print("Relevant predictions file found, skipping filter.")

    unsupervised_preds_file = "outputs/unsupervised_preds.csv"
    if not os.path.exists(unsupervised_preds_file):
        encoders = {
            "nomic": make_st_encoder("nomic-ai/nomic-embed-text-v1.5", prefix=NOMIC_TASK_PREFIX),
            "modernbert": make_st_encoder("nomic-ai/modernbert-embed-base"),
        }
        run_unsupervised(
            relevant_preds_file,
            unsupervised_preds_file,
            encoders,
            LIST_BDI_SYMPTOMS,
            statements_dir=LLM_DESCRIPTION_FOLDER,
            thr=NOISE_THRESHOLD,
            K=KMEANS_K,
            first_person=args.first_person,
        )
    else:
        print("Unsupervised predictions found, skipping ensemble.")

    # 4. Score Fusion
    print("\n--- Step 4: Score Fusion ---")
    run_file_path = f"runs/{args.run_name}.txt"
    if not os.path.exists(run_file_path):
        print("Running score fusion...")
        fusion_main(
            args.fusion_checkpoint,
            relevant_preds_file,
            unsupervised_preds_file,
            run_file_path,
            args.run_name,
        )
    else:
        print(f"Run file {run_file_path} found, skipping fusion.")

    # 5. Evaluation
    print("\n--- Step 5: Evaluation ---")
    if os.path.exists(run_file_path):
        call_metrics(load_run_file(run_file_path), args.gold_file)
    else:
        print("No run file found to evaluate.")

    print("\n--- HOPE Pipeline Finished ---")


if __name__ == "__main__":
    main()
