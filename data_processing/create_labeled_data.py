import os
from glob import glob

import pandas as pd

from data_processing.multilabel_convert import convert_single_to_multilabel


def main():
    # Step 1: read and save raw labels
    os.makedirs("data/labels", exist_ok=True)
    raw_csv = "data/qrels_majority_merged.csv"
    label_csv = "data/labels/erisk2025_majority.csv"
    df_raw = pd.read_csv(raw_csv)
    df_raw.to_csv(label_csv, index=False)

    # Step 2: convert to multi-label
    df_multi = convert_single_to_multilabel(label_csv)

    # Step 3: stream merge labels with text
    # Prepare output file
    output_file = "data/labels/labeled_data_with_text.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Write header
    header = df_multi.columns.tolist() + ["text"]
    pd.DataFrame(columns=header).to_csv(output_file, index=False)

    # Prepare lookup
    doc_ids_set = set(df_multi["doc_id"])
    # Iterate through processed files and append matching rows
    processed_dir = "data/erisk25-t1-processed"
    for csv_file in glob(os.path.join(processed_dir, "*.csv")):
        # Read in chunks to avoid memory overload
        for chunk in pd.read_csv(
            csv_file, usecols=["doc_no", "doc_text"], chunksize=10000
        ):
            matched = chunk[chunk["doc_no"].isin(doc_ids_set)]
            if not matched.empty:
                matched = matched.rename(
                    columns={"doc_no": "doc_id", "doc_text": "text"}
                )
                # Merge labels
                merged = matched.merge(df_multi, on="doc_id")
                # Order columns
                merged = merged[df_multi.columns.tolist() + ["text"]]
                # Append to output
                merged.to_csv(output_file, mode="a", index=False, header=False)


if __name__ == "__main__":
    main()
