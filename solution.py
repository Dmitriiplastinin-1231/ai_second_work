import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

TARGET_COLUMN = "salary_mean_net"
DEFAULT_TEST_FILE = "test_x.csv"
DEFAULT_PREDICTION = 0.0
MAX_TFIDF_FEATURES = 50000


def _join_text_columns(frame):
    """Join text columns into a single string per row."""
    if isinstance(frame, np.ndarray):
        frame = pd.DataFrame(frame)
    return frame.fillna("").astype(str).agg(" ".join, axis=1)


def _detect_id_column(columns):
    for column in columns:
        column_lower = column.lower()
        if column_lower in {"id", "row_id"}:
            return column
    return None


def _find_train_paths(base_dir, target_col):
    preferred = [
        base_dir / "train.csv",
        base_dir / "train_data.csv",
        base_dir / "train_full.csv",
    ]
    for path in preferred:
        if path.exists():
            return path, None

    train_x = base_dir / "train_x.csv"
    train_y = base_dir / "train_y.csv"
    if train_x.exists() and train_y.exists():
        return train_x, train_y

    for path in base_dir.glob("*.csv"):
        if path.name in {DEFAULT_TEST_FILE, "sample_submission.csv"}:
            continue
        try:
            sample_cols = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        if target_col in sample_cols:
            return path, None
    return None, None


def _load_training_data(train_path, target_path, target_col):
    if train_path is None:
        raise FileNotFoundError("Training data file was not found.")

    if target_path is None:
        train_df = pd.read_csv(train_path)
        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' was not found.")
        y = train_df[target_col]
        x = train_df.drop(columns=[target_col])
        return x, y

    train_x = pd.read_csv(train_path)
    train_y = pd.read_csv(target_path)
    if target_col in train_y.columns:
        y = train_y[target_col]
    elif train_y.shape[1] == 1:
        y = train_y.iloc[:, 0]
    else:
        raise ValueError(
            "Target file must contain a single column or "
            f"'{target_col}', got {train_y.shape[1]} columns."
        )
    return train_x, y


def _build_pipeline(text_cols, num_cols, alpha):
    transformers = []

    if text_cols:
        text_pipeline = Pipeline(
            steps=[
                ("join", FunctionTransformer(_join_text_columns, validate=False)),
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=MAX_TFIDF_FEATURES,
                        ngram_range=(1, 2),
                    ),
                ),
            ]
        )
        transformers.append(("text", text_pipeline, text_cols))

    if num_cols:
        num_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        transformers.append(("num", num_pipeline, num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def train_and_predict(train_df, target, test_df, id_column=None, alpha=1.0):
    id_column = id_column or _detect_id_column(test_df.columns)
    drop_columns = [col for col in [id_column] if col in train_df.columns]
    x_train = train_df.drop(columns=drop_columns)
    x_test = test_df.drop(columns=[id_column]) if id_column else test_df.copy()

    text_cols = [col for col in x_train.columns if x_train[col].dtype == object]
    num_cols = [col for col in x_train.columns if col not in text_cols]

    model = _build_pipeline(text_cols, num_cols, alpha)
    model.fit(x_train, target)

    predictions = model.predict(x_test)
    predictions = np.nan_to_num(
        predictions,
        nan=DEFAULT_PREDICTION,
        posinf=DEFAULT_PREDICTION,
        neginf=DEFAULT_PREDICTION,
    )
    # Salaries cannot be negative; clip as a safeguard even if defaults change.
    predictions = np.clip(predictions, 0, None)

    ids = (
        test_df[id_column]
        if id_column
        else pd.Series(range(len(test_df)), name="ID")
    )
    return ids, predictions


def main():
    parser = argparse.ArgumentParser(
        description="Train salary model and create submission."
    )
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Path to training CSV (auto-discovered if omitted).",
    )
    parser.add_argument(
        "--train-target",
        type=str,
        default=None,
        help="Path to separate target CSV (when features and labels are split).",
    )
    parser.add_argument("--test", type=str, default=None, help="Path to test CSV.")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength (alpha parameter).",
    )
    args = parser.parse_args()

    base_dir = Path(".")
    train_path = Path(args.train) if args.train else None
    target_path = Path(args.train_target) if args.train_target else None

    if train_path is None:
        train_path, target_path = _find_train_paths(base_dir, TARGET_COLUMN)

    test_path = Path(args.test) if args.test else base_dir / DEFAULT_TEST_FILE
    if not test_path.exists():
        candidates = list(base_dir.glob("*test*.csv"))
        if candidates:
            test_path = candidates[0]
        else:
            raise FileNotFoundError("Test data file was not found.")

    x_train, y_train = _load_training_data(train_path, target_path, TARGET_COLUMN)
    test_df = pd.read_csv(test_path)

    id_col = _detect_id_column(test_df.columns)
    ids, predictions = train_and_predict(
        x_train, y_train, test_df, id_col, alpha=args.alpha
    )

    output_id_col = ids.name or id_col or "ID"
    submission = pd.DataFrame({output_id_col: ids, TARGET_COLUMN: predictions})
    submission.to_csv(args.output, index=False)
    print(f"Saved submission to {args.output}")


if __name__ == "__main__":
    main()
