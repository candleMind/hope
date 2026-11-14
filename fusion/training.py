import ast
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fusion.model import get_model
from utils.constants import LIST_BDI_SYMPTOMS

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


class FusionDataset(Dataset):
    """
    Dataset for fusion model that combines supervised and unsupervised predictions.
    """

    def __init__(self, df, label_columns, sup_columns, unsup_columns):
        """
        Args:
            df: DataFrame containing the data
            label_columns: List of ground truth label column names
            sup_columns: List of supervised prediction column names
            unsup_columns: List of unsupervised prediction column names
        """
        self.df = df.reset_index(drop=True)
        self.label_columns = label_columns
        self.sup_columns = sup_columns
        self.unsup_columns = unsup_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract supervised predictions (batch, label_count)
        sup_output = torch.tensor(
            [row[col] for col in self.sup_columns], dtype=torch.float32
        )

        # Extract unsupervised similarities/distances (batch, label_count)
        unsup_output = torch.tensor(
            [row[col] for col in self.unsup_columns], dtype=torch.float32
        )

        # Extract binary labels (batch, label_count)
        binary_label = torch.tensor(
            [row[col] for col in self.label_columns], dtype=torch.float32
        )

        sample = {
            "sup_output": sup_output,
            "unsup_output": unsup_output,
            "binary_label": binary_label,
            "id": row["id"],
        }

        return sample


def collate_fn(batch):
    """Collate function for DataLoader"""
    sup_outputs = torch.stack([item["sup_output"] for item in batch])
    unsup_outputs = torch.stack([item["unsup_output"] for item in batch])
    binary_labels = torch.stack([item["binary_label"] for item in batch])

    return sup_outputs, unsup_outputs, binary_labels


def prepare_dataset(
    supervised_path,
    unsupervised_path,
    symptom_columns,
):
    """
    Prepare dataset by merging supervised and unsupervised predictions.

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


def call_metrics(refs, preds):
    """
    Calculate evaluation metrics.

    Args:
        refs: Ground truth labels
        preds: Predicted labels

    Returns:
        Dictionary of metrics
    """
    result = {}
    result["acc"] = round(accuracy_score(refs, preds), 4)
    result["macro_f1"] = round(
        f1_score(refs, preds, average="macro", zero_division=0), 4
    )
    result["micro_f1"] = round(
        f1_score(refs, preds, average="micro", zero_division=0), 4
    )
    result["weighted_f1"] = round(
        f1_score(refs, preds, average="weighted", zero_division=0), 4
    )

    return result


def val_model(model, val_loader, criterion, best_f1, best_loss, steps, save_dir="./"):
    """
    Validate the model.

    Args:
        model: The fusion model
        val_loader: Validation data loader
        criterion: Loss function
        best_f1: Current best F1 score
        best_loss: Current best loss
        steps: Current training step
        save_dir: Directory to save checkpoints

    Returns:
        Tuple of (avg_val_loss, val_result, best_f1, best_loss)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            sup_output, unsup_output, label = batch
            out = model(sup_output.to(device), unsup_output.to(device))
            loss = criterion(out.squeeze(-1), label.to(device))

            total_loss += loss.item() * len(label)
            all_preds.extend((torch.sigmoid(out) > 0.5).int().detach().cpu().tolist())
            all_labels.extend(label.cpu().tolist())

            torch.cuda.empty_cache()

    avg_val_loss = total_loss / len(all_labels)
    val_result = call_metrics(all_labels, all_preds)
    val_f1 = val_result["macro_f1"]

    # Save best model
    if val_f1 > best_f1 or avg_val_loss < best_loss:
        best_f1 = max(val_f1, best_f1)
        best_loss = min(avg_val_loss, best_loss)
        best_model_name = os.path.join(
            save_dir, f"fusion_{steps}_f1({val_f1:.4f})_loss({avg_val_loss:.4f}).pt"
        )
        torch.save(model.state_dict(), best_model_name)
        print(f"Saved best model: {best_model_name}")

    return avg_val_loss, val_result, best_f1, best_loss


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=0.001,
    save_steps=None,
    scheduler_type="cosine",
    patience=5,
    factor=0.5,
    save_dir="./results",
):
    """
    Train the fusion model.

    Args:
        model: The fusion model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        save_steps: Save checkpoint every N steps (if None, saves once per epoch)
        scheduler_type: Type of learning rate scheduler ('cosine', 'plateau', 'step', 'exponential')
        patience: Patience for ReduceLROnPlateau scheduler
        factor: Learning rate reduction factor
        save_dir: Directory to save checkpoints

    Returns:
        Trained model
    """
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Initialize scheduler
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience, verbose=True
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    elif scheduler_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        scheduler = None
        print(
            f"Warning: Unknown scheduler type '{scheduler_type}'. Training without scheduler."
        )

    model.train()

    best_f1, best_loss, steps = 0.0, 1000.0, 0

    if save_steps is None:
        save_steps = len(train_loader)

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            sup_output, unsup_output, label = batch
            optimizer.zero_grad()

            out = model(sup_output.to(device), unsup_output.to(device))
            loss = criterion(out.squeeze(-1), label.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(label)
            all_preds.extend((torch.sigmoid(out) > 0.5).int().detach().cpu().tolist())
            all_labels.extend(label.cpu().tolist())

            torch.cuda.empty_cache()
            steps += 1

            if steps % save_steps == 0:
                avg_train_loss = total_loss / len(all_labels)
                train_result = call_metrics(all_labels, all_preds)
                avg_val_loss, val_result, best_f1, best_loss = val_model(
                    model, val_loader, criterion, best_f1, best_loss, steps, save_dir
                )
                model.train()

                print(
                    f"Epoch {epoch+1:02d} | Steps {steps} | "
                    f"Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f} | "
                    f"Val Macro F1: {val_result['macro_f1']:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

        # Step scheduler at end of epoch
        if scheduler is not None:
            if scheduler_type == "plateau":
                # For ReduceLROnPlateau, pass the metric to monitor
                if steps % save_steps != 0:
                    _, val_result_epoch, _, _ = val_model(
                        model,
                        val_loader,
                        criterion,
                        best_f1,
                        best_loss,
                        steps,
                        save_dir,
                    )
                    model.train()
                    scheduler.step(val_result_epoch["macro_f1"])
                else:
                    scheduler.step(val_result["macro_f1"])
            else:
                scheduler.step()

    return model


def main(
    symptom_columns,
    train_supervised_path,
    train_unsupervised_path,
    val_supervised_path,
    val_unsupervised_path,
    output_dir="./results",
    num_epochs=100,
    lr=2e-3,
    batch_size=32,
    val_batch_size=64,
    save_steps=50,
    hidden_size=64,
    scheduler_type="cosine",
):
    """
    Main training function.

    Args:
        symptom_columns: List of symptom column names
        train_supervised_path: Path to training supervised predictions CSV
        train_unsupervised_path: Path to training unsupervised predictions CSV
        val_supervised_path: Path to validation supervised predictions CSV
        val_unsupervised_path: Path to validation unsupervised predictions CSV
        output_dir: Directory to save model checkpoints (default: "./results")
        num_epochs: Number of training epochs (default: 100)
        lr: Learning rate (default: 2e-3)
        batch_size: Training batch size (default: 32)
        val_batch_size: Validation batch size (default: 64)
        save_steps: Save checkpoint every N steps (default: 50)
        hidden_size: Hidden layer size (default: 64)
        scheduler_type: Learning rate scheduler type (default: "cosine")
    """
    # Prepare datasets
    train_df = prepare_dataset(
        train_supervised_path, train_unsupervised_path, symptom_columns
    )
    val_df = prepare_dataset(
        val_supervised_path, val_unsupervised_path, symptom_columns
    )

    # Get column names
    sup_columns = [f"pred_{col}" for col in symptom_columns]
    unsup_columns = [col for col in train_df.columns if col.endswith("_similarity")]

    print(f"Symptom columns: {symptom_columns}")
    print(f"Supervised columns: {sup_columns}")
    print(f"Unsupervised columns: {unsup_columns}")

    # Create datasets and dataloaders
    train_dataset = FusionDataset(train_df, symptom_columns, sup_columns, unsup_columns)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_dataset = FusionDataset(val_df, symptom_columns, sup_columns, unsup_columns)
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = get_model(
        sup_size=len(sup_columns),
        unsup_size=len(unsup_columns),
        label_size=len(symptom_columns),
        hidden_size=hidden_size,
    )

    # Train model
    train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=lr,
        save_steps=save_steps,
        scheduler_type=scheduler_type,
        save_dir=output_dir,
    )

    print("Training completed!")


if __name__ == "__main__":
    # Default configuration for standalone execution
    symptom_columns = LIST_BDI_SYMPTOMS

    train_supervised_path = "outputs/train/supervised_preds.csv"
    train_unsupervised_path = (
        "outputs/train/multilabel-questionaire-32_avg_sim_merged_output.csv"
    )
    val_supervised_path = "outputs/val/supervised_preds.csv"
    val_unsupervised_path = (
        "outputs/val/multilabel-questionaire-32_avg_sim_merged_output.csv"
    )
    output_dir = "./results"

    main(
        symptom_columns=symptom_columns,
        train_supervised_path=train_supervised_path,
        train_unsupervised_path=train_unsupervised_path,
        val_supervised_path=val_supervised_path,
        val_unsupervised_path=val_unsupervised_path,
        output_dir=output_dir,
    )
