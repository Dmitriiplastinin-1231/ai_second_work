import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COL = "salary_mean_net"
MAX_TF_IDF_FEATURES = 20000
MIN_SALARY = 0


def list_csv_and_txt_files(data_dir: Path) -> list[Path]:
    patterns = ["*.csv", "*.txt"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(data_dir.glob(pattern))
    return sorted({file.resolve() for file in files})


def detect_delimiter(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            sample = handle.read(4096)
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        return ","


def read_table_columns(path: Path) -> list[str]:
    delimiter = detect_delimiter(path) if path.suffix.lower() == ".txt" else ","
    df = pd.read_csv(path, sep=delimiter, nrows=0)
    return list(df.columns)


def read_table(path: Path) -> pd.DataFrame:
    delimiter = detect_delimiter(path) if path.suffix.lower() == ".txt" else ","
    return pd.read_csv(path, sep=delimiter, low_memory=False)


def discover_train_test(data_dir: Path) -> tuple[Path, Path]:
    data_files = list_csv_and_txt_files(data_dir)
    if not data_files:
        raise FileNotFoundError(f"No .csv/.txt files found in {data_dir}")

    train_path = None
    test_path = None
    columns_map: dict[Path, list[str]] = {}
    for path in data_files:
        columns = read_table_columns(path)
        columns_map[path] = columns
        if TARGET_COL in columns and train_path is None:
            train_path = path
        elif "test" in path.name.lower() and test_path is None:
            test_path = path

    if test_path is None:
        for path, columns in columns_map.items():
            if TARGET_COL not in columns:
                test_path = path
                break

    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "Could not detect train/test files. Ensure train contains "
            f"'{TARGET_COL}' and test does not."
        )

    return train_path, test_path


class TextCombiner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if df.empty:
            return pd.Series("", index=df.index)
        return df.fillna("").astype(str).agg(" ".join, axis=1)


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = [col for col in X_train.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if text_cols:
        transformers.append(
            (
                        "text",
                        Pipeline(
                            [
                                ("combine", TextCombiner()),
                                (
                                    "tfidf",
                                    TfidfVectorizer(
                                max_features=MAX_TF_IDF_FEATURES,
                                ngram_range=(1, 2),
                                min_df=2,
                            ),
                        ),
                    ]
                ),
                text_cols,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns found after preprocessing.")

    preprocess = ColumnTransformer(transformers)

    return Pipeline(
        [
            ("prep", preprocess),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def resolve_id_column(test: pd.DataFrame) -> pd.Series:
    for candidate in ["ID", "id", "index"]:
        if candidate in test.columns:
            return test[candidate]
    return pd.Series(test.index, name="ID")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model and create submission.")
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing train/test files.",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Path for the generated submission file.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    train_path, test_path = discover_train_test(data_dir)
    print(f"Train file: {train_path}")
    print(f"Test file: {test_path}")

    train = read_table(train_path)
    test = read_table(test_path)

    y = train[TARGET_COL].astype(float)
    test_cols = set(test.columns)
    train_features = [col for col in train.columns if col != TARGET_COL]
    shared_cols = set(train_features) & test_cols
    feature_cols = [col for col in train_features if col in shared_cols]
    if not feature_cols:
        raise ValueError("No shared feature columns between train and test.")

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    model = build_pipeline(X_train)
    model.fit(X_train, y)

    preds = model.predict(X_test)
    # Salary cannot be negative in the competition metric.
    preds = np.maximum(preds, MIN_SALARY)

    submission = pd.DataFrame(
        {
            "ID": resolve_id_column(test).values,
            TARGET_COL: preds,
        }
    )

    output_path = Path(args.output).resolve()
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
