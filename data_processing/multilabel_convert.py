import pandas as pd


def convert_single_to_multilabel(input_csv: str) -> pd.DataFrame:
    """
    Reads a single-label qrels CSV and pivots to multi-label format.
    Returns a DataFrame with columns: 'doc_id', 'query_1', ..., 'query_21'.
    """
    df = pd.read_csv(input_csv)

    # Pivot to wide format: rows=doc_id, columns=query, values=relevant
    wide = df.pivot(index="doc_id", columns="query", values="relevant")

    # Rename columns to 'query_{i}'
    wide = wide.rename(columns=lambda x: f"query_{int(x)}")

    # Ensure all query columns from 1 to 21 exist
    for i in range(1, 22):
        col = f"query_{i}"
        if col not in wide.columns:
            wide[col] = False

    # Reorder columns and fill missing with False
    cols = ["doc_id"] + [f"query_{i}" for i in range(1, 22)]
    result = wide.reset_index()[cols]
    result = result.fillna(False)
    return result
