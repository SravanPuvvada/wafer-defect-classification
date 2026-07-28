"""
Data loading utilities for the WM-811K Wafer Map dataset.

The dataset is typically distributed as a single pickle file (LSWMD.pkl)
containing a pandas DataFrame with columns:
    - waferMap: 2D numpy array (die-level pass/fail/pattern values)
    - failureType: label (may be nested array/string depending on source)
    - trianTestLabel: 'Training'/'Test' split (note: typo is in the
      original dataset itself)
    - dieSize, lotName, waferIndex: metadata (not used for modeling)

Only ~172,950 of the ~811,457 wafers have a labeled failureType; the rest
are unlabeled and should be dropped for supervised classification.
"""

import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# --- Compatibility shim for old WM-811K pickle files ---
# LSWMD.pkl was originally saved with an old pandas version (~2018) whose
# internal module layout has since changed (e.g. pandas.indexes ->
# pandas.core.indexes). Modern pandas can't unpickle it without this shim,
# which registers the old module names as aliases pointing at their new
# locations, purely so pickle.load can resolve them.
def _apply_pandas_pickle_compat():
    import pandas.core.indexes as _idx
    sys.modules.setdefault("pandas.indexes", _idx)
    try:
        import pandas.core.indexes.base as _idx_base
        sys.modules.setdefault("pandas.indexes.base", _idx_base)
    except ImportError:
        pass

DEFECT_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "none"
]


def load_raw(pkl_path: str) -> pd.DataFrame:
    """Load the raw WM-811K pickle file into a DataFrame."""
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Could not find {pkl_path}. Download LSWMD.pkl from Kaggle "
            f"(https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) "
            f"and place it in the data/ folder."
        )
    _apply_pandas_pickle_compat()
    with open(pkl_path, "rb") as f:
        # encoding="latin1" is required because LSWMD.pkl was originally
        # pickled under Python 2, whose string encoding differs from
        # Python 3's default. Without this, pickle.load() throws a
        # UnicodeDecodeError on the very first non-ASCII byte it meets.
        df = pickle.load(f, encoding="latin1")
    return df


def _extract_label(failure_type):
    """
    failureType is often stored as a nested array like [['Center']] or
    an empty array [[]] for unlabeled wafers. Normalize to a plain string
    or None.
    """
    try:
        val = failure_type[0][0]
        return val if isinstance(val, str) and val != "" else None
    except (IndexError, TypeError):
        return None


def load_labeled_dataset(pkl_path: str) -> pd.DataFrame:
    """
    Load WM-811K and return only the labeled subset, with a clean
    'label' column (string) ready for use.
    """
    df = load_raw(pkl_path)

    # Normalize label column (handles the nested-array format in the
    # original dataset)
    if "failureType" in df.columns:
        df["label"] = df["failureType"].apply(_extract_label)
    else:
        raise KeyError(
            "Expected column 'failureType' not found. "
            f"Available columns: {list(df.columns)}"
        )

    labeled = df[df["label"].notna()].reset_index(drop=True)
    print(f"Loaded {len(df)} total wafers, {len(labeled)} labeled.")
    return labeled


def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    """Return counts per defect class, sorted descending."""
    return df["label"].value_counts().sort_values(ascending=False)


if __name__ == "__main__":
    # Quick smoke test — run this after placing LSWMD.pkl in data/
    df = load_labeled_dataset("data/LSWMD.pkl")
    print(get_class_distribution(df))