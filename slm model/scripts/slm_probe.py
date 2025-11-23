#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
slm_probe.py

Quick script to sanity-check the trained SLM models on a few examples.
"""

import os
import argparse
import joblib
import json


def load_models(model_dir: str):
    eval_path = os.path.join(model_dir, "slm_evaluation.joblib")
    tone_path = os.path.join(model_dir, "slm_tone.joblib")
    meta_path = os.path.join(model_dir, "slm_meta.json")

    eval_model = joblib.load(eval_path)
    tone_model = joblib.load(tone_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("[INFO] Loaded models from:", model_dir)
    print("[INFO] Evaluation labels:", meta.get("labels_evaluation"))
    print("[INFO] Tone labels:", meta.get("labels_tone"))

    return eval_model, tone_model


def score_pair(eval_model, tone_model, question: str, response: str):
    text = f"{question} [SEP] {response}"
    eval_label = eval_model.predict([text])[0]
    tone_label = tone_model.predict([text])[0]
    return eval_label, tone_label


def parse_args():
    p = argparse.ArgumentParser(description="Probe SLM evaluation + tone models.")
    p.add_argument(
        "--model-dir",
        type=str,
        default="models/slm_eval",
        help="Directory containing slm_evaluation.joblib / slm_tone.joblib",
    )
    return p.parse_args()


def main():
    args = parse_args()
    eval_model, tone_model = load_models(args.model_dir)

    examples = [
        {
            "question": "Can you explain what a confusion matrix is and why it matters?",
            "response": "A confusion matrix shows true vs predicted labels so we can see where the model gets confused, not just overall accuracy.",
        },
        {
            "question": "How should we handle missing values in a dataset?",
            "response": "I don’t know, maybe just delete everything? I haven’t really thought about it.",
        },
        {
            "question": "We’re designing an app for elderly users. What should we keep in mind?",
            "response": "Honestly this sounds boring, I’d rather just copy whatever another app does.",
        },
        {
            "question": "How can we improve the onboarding flow for new users?",
            "response": "We could run usability tests, watch where users drop off, and simplify the first-time experience into a short guided tour.",
        },
    ]

    for i, ex in enumerate(examples, start=1):
        q = ex["question"]
        r = ex["response"]
        eval_label, tone_label = score_pair(eval_model, tone_model, q, r)

        print("\n----------------------------")
        print(f"Example {i}")
        print("Q:", q)
        print("R:", r)
        print("→ evaluation:", eval_label)
        print("→ tone:      ", tone_label)


if __name__ == "__main__":
    main()
