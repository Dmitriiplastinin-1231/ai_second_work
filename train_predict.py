import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


TARGET_COL = "salary_mean_net"


def list_data_files(data_dir: Path) -> list[Path]:
    patterns = ["*.csv", "*.txt"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(data_dir.glob(pattern))
    return sorted({file.resolve() for file in files})


def read_table_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".txt":
        df = pd.read_csv(path, sep=None, engine="python", nrows=0)
    else:
        df = pd.read_csv(path, nrows=0)
    return list(df.columns)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".txt":
        return pd.read_csv(path, sep=None, engine="python", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def discover_train_test(data_dir: Path) -> tuple[Path, Path]:
    data_files = list_data_files(data_dir)
    if not data_files:
        raise FileNotFoundError(f"No .csv/.txt files found in {data_dir}")

    train_path = None
    test_path = None
    for path in data_files:
        columns = read_table_columns(path)
        if TARGET_COL in columns:
            train_path = path
        elif "test" in path.name.lower() and test_path is None:
            test_path = path

    if train_path is None:
        for path in data_files:
            columns = read_table_columns(path)
            if TARGET_COL in columns:
                train_path = path
                break

    if test_path is None:
        for path in data_files:
            columns = read_table_columns(path)
            if TARGET_COL not in columns:
                test_path = path
                break

    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "Could not detect train/test files. Ensure train contains "
            f"'{TARGET_COL}' and test does not."
        )

    return train_path, test_path


def combine_text(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series([""] * len(df))
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
                        ("scaler", StandardScaler(with_mean=False)),
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
                        ("combine", FunctionTransformer(combine_text, validate=False)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=50000,
                                ngram_range=(1, 2),
                                min_df=1,
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
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )


def get_id_values(test: pd.DataFrame) -> pd.Series:
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

    train = read_table(train_path)
    test = read_table(test_path)

    if TARGET_COL not in train.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {train_path}.")

    y = train[TARGET_COL].astype(float)
    feature_cols = [col for col in train.columns if col in test.columns and col != TARGET_COL]
    if not feature_cols:
        raise ValueError("No shared feature columns between train and test.")

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    model = build_pipeline(X_train)
    model.fit(X_train, y)

    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)

    submission = pd.DataFrame(
        {
            "ID": get_id_values(test).values,
            TARGET_COL: preds,
        }
    )

    output_path = Path(args.output).resolve()
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
