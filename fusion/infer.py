import ast
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from fusion.model import load_checkpoint
from fusion.training import FusionDataset, collate_fn
from utils.constants import LIST_BDI_SYMPTOMS

device = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_test_dataset(supervised_path, unsupervised_path, symptom_columns):
    """
    Prepare test dataset by merging supervised and unsupervised predictions.

    Args:
        supervised_path: Path to supervised predictions CSV
        unsupervised_path: Path to unsupervised predictions CSV
        symptom_columns: List of symptom column names

    Returns:
        Merged DataFrame with labels, supervised predictions, and unsupervised predictions
    """
    # Load supervised predictions
    supervised_df = pd.read_csv(supervised_path)
    supervised_df = supervised_df.loc[
        :, ~supervised_df.columns.str.contains("^Unnamed")
    ]

    # Parse label column if it's a string representation of a list
    if "label" in supervised_df.columns and isinstance(
        supervised_df["label"].iloc[0], str
    ):
        supervised_df["label"] = supervised_df["label"].apply(ast.literal_eval)

    # Create label mappings and extract labels
    labels_mapping = {}
    for i, symptom in enumerate(symptom_columns, 1):
        if "label" in supervised_df.columns:
            supervised_df[symptom] = supervised_df["label"].apply(
                lambda x: x[i - 1] if isinstance(x, list) else x
            )
        labels_mapping[f"pred_{i}"] = f"pred_{symptom}"

    # Rename supervised prediction columns
    supervised_df = supervised_df.rename(columns=labels_mapping)

    # Load unsupervised predictions
    unsupervised_df = pd.read_csv(unsupervised_path)

    # Merge datasets
    merged_df = pd.merge(supervised_df, unsupervised_df, on="id", how="inner")

    return merged_df


def evaluate_model(
    model,
    test_df,
    symptom_columns,
    sup_columns,
    unsup_columns,
    model_name="hybrid",
    batch_size=32,
):
    """
    Evaluate fusion model on test data and save predictions to DataFrame.

    Args:
        model: Trained fusion model
        test_df: DataFrame with test data
        symptom_columns: List of symptom label column names
        sup_columns: List of supervised prediction column names
        unsup_columns: List of unsupervised prediction column names
        model_name: Prefix for prediction columns (default: "hybrid")
        batch_size: Batch size for inference

    Returns:
        Updated DataFrame with predictions
    """
    # Create test dataloader
    test_dataset = FusionDataset(test_df, symptom_columns, sup_columns, unsup_columns)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            sup_output, unsup_output, label = batch
            out = model(sup_output.to(device), unsup_output.to(device))

            # Get model predictions
            probs = torch.sigmoid(out).detach().cpu()
            preds = (probs > 0.5).int()

            all_preds.extend(preds.tolist())
            all_labels.extend(label.tolist())
            all_probs.extend(probs.tolist())

            torch.cuda.empty_cache()

    # Convert to numpy arrays for easier handling
    all_preds = np.array(all_preds)  # (n_samples, n_labels)
    all_labels = np.array(all_labels)  # (n_samples, n_labels)
    all_probs = np.array(all_probs)  # (n_samples, n_labels)

    # Create a copy of test_df to add predictions
    result_df = test_df.copy()

    # Add prediction columns for each label
    for i, label in enumerate(symptom_columns):
        # Binary predictions
        result_df[f"{model_name}_{label}"] = all_preds[:, i]
        # Probability scores
        result_df[f"{model_name}_{label}_prob"] = all_probs[:, i]

    # Calculate and print overall metrics
    print("=== Overall Multi-label Classification Report ===")
    print(
        classification_report(
            all_labels,
            all_preds,
            zero_division=0,
            target_names=symptom_columns,
            digits=4,
        )
    )

    return result_df


def infer(
    checkpoint_path,
    test_supervised_path,
    test_unsupervised_path,
    symptom_columns,
    sup_size,
    unsup_size,
    hidden_size=64,
    output_path=None,
):
    """
    Run inference on test data using a trained fusion model.

    Args:
        checkpoint_path: Path to the trained model checkpoint
        test_supervised_path: Path to test supervised predictions CSV
        test_unsupervised_path: Path to test unsupervised predictions CSV
        symptom_columns: List of symptom column names
        sup_size: Dimension of supervised predictions
        unsup_size: Dimension of unsupervised predictions
        hidden_size: Size of hidden layer in the model
        output_path: Optional path to save predictions CSV

    Returns:
        DataFrame with predictions
    """
    print("Loading model...")
    model = load_checkpoint(
        checkpoint_path=checkpoint_path,
        sup_size=sup_size,
        unsup_size=unsup_size,
        label_size=len(symptom_columns),
        hidden_size=hidden_size,
    )
    model.to(device)
    print("Model loaded successfully.")

    print("Loading test data...")
    test_df = prepare_test_dataset(
        test_supervised_path, test_unsupervised_path, symptom_columns
    )

    # Get column names
    sup_columns = [f"pred_{col}" for col in symptom_columns]
    unsup_columns = [col for col in test_df.columns if col.endswith("_similarity")]

    print(f"Test samples: {len(test_df)}")
    print(f"Supervised columns: {sup_columns}")
    print(f"Unsupervised columns: {unsup_columns}")

    print("Running inference...")
    result_df = evaluate_model(
        model, test_df, symptom_columns, sup_columns, unsup_columns
    )

    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    return result_df


def main():
    """Main inference function"""
    # Configuration
    symptom_columns = LIST_BDI_SYMPTOMS  # 21 for BDI-II

    # Paths (adjust these to your paths)
    checkpoint_path = "results/{checkpoint}.pt"  # Update with your checkpoint path
    test_supervised_path = "outputs/test/{supervised_preds}.csv"
    test_unsupervised_path = "outputs/test/{unsupervised_preds}.csv"
    output_path = "outputs/test/fusion_predictions.csv"

    # Get supervised and unsupervised column counts
    # These should match your training configuration
    sup_size = len(symptom_columns)
    unsup_size = len(symptom_columns)  # one KMeans cluster per BDI symptom

    # Run inference
    result_df = infer(
        checkpoint_path=checkpoint_path,
        test_supervised_path=test_supervised_path,
        test_unsupervised_path=test_unsupervised_path,
        symptom_columns=symptom_columns,
        sup_size=sup_size,
        unsup_size=unsup_size,
        hidden_size=64,
        output_path=output_path,
    )

    print(f"\nInference completed! Results shape: {result_df.shape}")


if __name__ == "__main__":
    main()
