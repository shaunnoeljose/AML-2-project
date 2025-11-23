#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_slm.py

Train small local models (SLM) to evaluate Socratic responses using the
synthetic hybrid dataset created by hybrid_dataset.py.

- Model 1: evaluation classifier (GOOD / WEAK / CONFUSED)
- Model 2: tone classifier (analytical / curious / frustrated / neutral / playful)

Both are TF-IDF + LogisticRegression pipelines.

Usage:

  python train_slm.py \
    --data-path data/eval_training_data_hybrid.csv \
    --out-dir models/slm_eval

Requirements:
  pip install pandas scikit-learn joblib
"""

import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import joblib


def build_text_column(df: pd.DataFrame) -> pd.Series:
    """
    Combine question + response into a single text field for classification.
    """
    q = df["question"].astype(str)
    r = df["response"].astype(str)
    return (q + " [SEP] " + r).str.strip()


def build_pipeline() -> Pipeline:
    """
    TF-IDF (1–2 grams) + LogisticRegression with class_weight='balanced'.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=50000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    class_weight="balanced",
                    n_jobs=-1,
                    multi_class="auto",
                ),
            ),
        ]
    )


def train_and_eval(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    label_name: str,
) -> Tuple[Pipeline, str]:
    """
    Fit a model and return it plus a text report.
    """
    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    report = classification_report(y_val, y_pred, digits=3)
    cm = confusion_matrix(y_val, y_pred, labels=sorted(np.unique(y_val)))

    report_text = []
    report_text.append(f"\n=== {label_name} classifier report ===")
    report_text.append(report)
    report_text.append("Confusion matrix (rows=true, cols=pred):")
    report_text.append(str(cm))
    report_full = "\n".join(report_text)

    print(report_full)
    return model, report_full


def parse_args():
    p = argparse.ArgumentParser(description="Train SLM evaluation + tone classifiers.")
    p.add_argument(
        "--data-path",
        type=str,
        default="data/eval_training_data_hybrid.csv",
        help="CSV file produced by hybrid_dataset.py",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="models/slm_eval",
        help="Directory to save trained models and reports.",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of data for held-out test set.",
    )
    p.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of data for validation (relative to full dataset).",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading dataset from: {args.data_path}")
    df = pd.read_csv(args.data_path)

    required = ["question", "response", "evaluation", "tone"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Drop rows with missing text/labels
    df = df.dropna(subset=["question", "response", "evaluation", "tone"]).reset_index(drop=True)

    # Build combined text
    X = build_text_column(df).values
    y_eval = df["evaluation"].astype(str).values
    y_tone = df["tone"].astype(str).values

    # First split off test set
    X_trainval, X_test, y_eval_trainval, y_eval_test, y_tone_trainval, y_tone_test = train_test_split(
        X,
        y_eval,
        y_tone,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_eval,  # stratify by evaluation (primary label)
    )

    # Now split train/val from trainval
    val_frac_of_trainval = args.val_size / (1.0 - args.test_size)
    (
        X_train,
        X_val,
        y_eval_train,
        y_eval_val,
        y_tone_train,
        y_tone_val,
    ) = train_test_split(
        X_trainval,
        y_eval_trainval,
        y_tone_trainval,
        test_size=val_frac_of_trainval,
        random_state=args.random_state,
        stratify=y_eval_trainval,
    )

    print(f"[INFO] Dataset sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    # ---------------------------
    # Train EVALUATION classifier
    # ---------------------------
    eval_model, eval_val_report = train_and_eval(
        X_train, X_val, y_eval_train, y_eval_val, label_name="Evaluation (GOOD/WEAK/CONFUSED)"
    )

    # Evaluate on held-out test set
    y_eval_test_pred = eval_model.predict(X_test)
    eval_test_report = "\n".join(
        [
            "\n=== Evaluation classifier – TEST set ===",
            classification_report(y_eval_test, y_eval_test_pred, digits=3),
        ]
    )
    print(eval_test_report)

    # ------------------------
    # Train TONE classifier
    # ------------------------
    tone_model, tone_val_report = train_and_eval(
        X_train, X_val, y_tone_train, y_tone_val, label_name="Tone (analytical/curious/frustrated/neutral/playful)"
    )

    y_tone_test_pred = tone_model.predict(X_test)
    tone_test_report = "\n".join(
        [
            "\n=== Tone classifier – TEST set ===",
            classification_report(y_tone_test, y_tone_test_pred, digits=3),
        ]
    )
    print(tone_test_report)

    # ------------------------
    # Save models + reports
    # ------------------------
    eval_model_path = os.path.join(args.out_dir, "slm_evaluation.joblib")
    tone_model_path = os.path.join(args.out_dir, "slm_tone.joblib")
    meta_path = os.path.join(args.out_dir, "slm_meta.json")
    reports_path = os.path.join(args.out_dir, "training_reports.txt")

    joblib.dump(eval_model, eval_model_path)
    joblib.dump(tone_model, tone_model_path)

    # Small metadata for later integration
    meta = {
        "data_path": os.path.abspath(args.data_path),
        "n_samples": int(len(df)),
        "labels_evaluation": sorted(list(np.unique(y_eval))),
        "labels_tone": sorted(list(np.unique(y_tone))),
        "eval_model_path": os.path.abspath(eval_model_path),
        "tone_model_path": os.path.abspath(tone_model_path),
    }

    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(reports_path, "w", encoding="utf-8") as f:
        f.write(eval_val_report)
        f.write("\n")
        f.write(eval_test_report)
        f.write("\n")
        f.write(tone_val_report)
        f.write("\n")
        f.write(tone_test_report)

    print(f"\n[OK] Saved evaluation model → {eval_model_path}")
    print(f"[OK] Saved tone model       → {tone_model_path}")
    print(f"[OK] Saved metadata         → {meta_path}")
    print(f"[OK] Saved training reports → {reports_path}")


if __name__ == "__main__":
    main()

########Adding evaluaton snippet for integration########

# import joblib
# import os

# from sklearn.pipeline import Pipeline

# eval_model = joblib.load("models/slm_eval/slm_evaluation.joblib")
# tone_model = joblib.load("models/slm_eval/slm_tone.joblib")

# def score_response(question: str, response: str):
#     text = f"{question} [SEP] {response}"
#     eval_label = eval_model.predict([text])[0]
#     tone_label = tone_model.predict([text])[0]
#     return eval_label, tone_label