import argparse
import sys
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
NOTEBOOK_CONNECTION_FLAG = "-f"
KAGGLE_INPUT_DIR = Path("/kaggle/input")
KAGGLE_WORKING_DIR = Path("/kaggle/working")
# Jupyter kernels pass connection-related arguments with these prefixes.
NOTEBOOK_ARG_PREFIXES = (
    "--ip",
    "--stdin",
    "--control",
    "--shell",
    "--transport",
    "--iopub",
    "--hb",
    "--Session.",
)


def _join_text_columns(frame):
    """Join text columns into a single string per row."""
    if isinstance(frame, np.ndarray):
        frame = pd.DataFrame(frame)
    return frame.fillna("").astype(str).agg(" ".join, axis=1)


def _detect_id_column(columns):
    """Find a likely ID column name from a list of columns, or None."""
    for column in columns:
        column_lower = column.lower()
        if column_lower in {"id", "row_id"}:
            return column
    return None


def _filter_notebook_args(unknown_args):
    """Filter out Jupyter kernel args and keep only real unknown arguments."""
    unrecognized_args = []
    skip_notebook_arg_value = False
    for index, arg in enumerate(unknown_args):
        if skip_notebook_arg_value:
            skip_notebook_arg_value = False
            continue
        if arg == NOTEBOOK_CONNECTION_FLAG:
            if index + 1 < len(unknown_args):
                next_arg = unknown_args[index + 1]
                if not next_arg.startswith("-"):
                    skip_notebook_arg_value = True
            continue
        if arg.startswith(NOTEBOOK_ARG_PREFIXES):
            continue
        unrecognized_args.append(arg)
    return unrecognized_args


def _candidate_base_dirs(base_dir):
    """Return directories to scan for data files."""
    candidates = [base_dir]
    if KAGGLE_INPUT_DIR.exists():
        candidates.append(KAGGLE_INPUT_DIR)
        candidates.extend(
            path for path in KAGGLE_INPUT_DIR.iterdir() if path.is_dir()
        )
    seen = set()
    ordered = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def _iter_csv_candidates(base_dir):
    """Yield CSV files, using shallow glob for cwd and recursive for others."""
    if base_dir == Path("."):
        return base_dir.glob("*.csv")
    return base_dir.rglob("*.csv")


def _find_train_paths(base_dir, target_col):
    """Locate training data files and return (train_path, target_path)."""
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

    for path in _iter_csv_candidates(base_dir):
        if path.name in {DEFAULT_TEST_FILE, "sample_submission.csv"}:
            continue
        try:
            sample_cols = pd.read_csv(path, nrows=0).columns
        except (
            OSError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ):
            continue
        if target_col in sample_cols:
            return path, None
    return None, None


def _find_training_data(base_dirs, target_col):
    for base_dir in base_dirs:
        train_path, target_path = _find_train_paths(base_dir, target_col)
        if train_path is not None:
            return train_path, target_path
    return None, None


def _find_test_path(base_dirs):
    for base_dir in base_dirs:
        candidate = base_dir / DEFAULT_TEST_FILE
        if candidate.exists():
            return candidate
    for base_dir in base_dirs:
        candidates = sorted(base_dir.glob("*test*.csv"))
        if candidates:
            return candidates[0]
    for base_dir in base_dirs:
        if base_dir == Path("."):
            continue
        candidates = sorted(base_dir.rglob("*test*.csv"))
        if candidates:
            return candidates[0]
    return None


def _load_training_data(train_path, target_path, target_col):
    """Load training features and targets from CSV files."""
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
            "Target file must contain either the column "
            f"'{target_col}' or exactly one column, got "
            f"{train_y.shape[1]} columns."
        )
    return train_x, y


def _build_pipeline(text_cols, num_cols, alpha):
    """Build a preprocessing + Ridge regression pipeline."""
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


def _align_feature_frames(train_df, test_df):
    """Align test columns to train columns, filling missing with NaN."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    missing_cols = [col for col in train_df.columns if col not in test_df.columns]
    for col in missing_cols:
        test_df[col] = np.nan
    extra_cols = [col for col in test_df.columns if col not in train_df.columns]
    if extra_cols:
        test_df = test_df.drop(columns=extra_cols)
    test_df = test_df[train_df.columns]
    return train_df, test_df


def train_and_predict(train_df, target, test_df, id_column=None, alpha=1.0):
    """Fit the model and return (ids, predictions) for the test set."""
    id_column = id_column or _detect_id_column(test_df.columns)
    if id_column and id_column not in test_df.columns:
        id_column = None
    drop_columns = (
        [id_column] if id_column and id_column in train_df.columns else []
    )
    x_train = train_df.drop(columns=drop_columns)
    x_test = test_df.drop(columns=[id_column]) if id_column else test_df.copy()
    x_train, x_test = _align_feature_frames(x_train, x_test)

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
    # Salaries cannot be negative; clip model predictions to zero minimum.
    predictions = np.clip(predictions, 0, None)

    ids = (
        test_df[id_column]
        if id_column
        else pd.Series(range(len(test_df)), name="ID")
    )
    return ids, predictions


def main():
    """CLI entry point for training a salary model and writing submissions."""
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
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Path to output CSV file for predictions.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength (alpha parameter).",
    )
    args, unknown = parser.parse_known_args()
    unrecognized_args = _filter_notebook_args(unknown)
    if unrecognized_args:
        print(
            f"Warning: ignoring unknown arguments: {unrecognized_args}",
            file=sys.stderr,
        )

    base_dir = Path(".")
    base_dirs = _candidate_base_dirs(base_dir)
    train_path = Path(args.train) if args.train else None
    target_path = Path(args.train_target) if args.train_target else None

    if train_path is None:
        train_path, target_path = _find_training_data(
            base_dirs, TARGET_COLUMN
        )

    if train_path is None or not train_path.exists():
        raise FileNotFoundError("Training data file was not found.")
    if target_path is not None and not target_path.exists():
        raise FileNotFoundError("Training target file was not found.")

    test_path = Path(args.test) if args.test else _find_test_path(base_dirs)
    if test_path is None or not test_path.exists():
        raise FileNotFoundError("Test data file was not found.")

    x_train, y_train = _load_training_data(train_path, target_path, TARGET_COLUMN)
    test_df = pd.read_csv(test_path)

    id_col = _detect_id_column(test_df.columns)
    ids, predictions = train_and_predict(
        x_train, y_train, test_df, id_col, alpha=args.alpha
    )

    output_id_col = (
        ids.name
        if ids.name not in (None, "")
        else (id_col or "ID")
    )
    submission = pd.DataFrame({output_id_col: ids, TARGET_COLUMN: predictions})
    output_path = Path(args.output)
    if args.output == "submission.csv" and KAGGLE_WORKING_DIR.exists():
        output_path = KAGGLE_WORKING_DIR / args.output
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
