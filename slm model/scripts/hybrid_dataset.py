# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# hybrid_dataset.py
# - Hybrid data builder for SLM evaluation labels:
#   * Seed a balanced high-quality set via Gemini (quality x tone).
#   * Expand locally with lightweight augmentation (no more API calls).
#   * Optional semantic dedupe.
# - Robust NLTK bootstrap + graceful fallback (no crashes).
# - Clear logs and progress bars.

# Requirements:
#   pip install google-generativeai pandas scikit-learn tqdm nltk
#   (optional) pip install sentence-transformers

# Usage example:
#   python hybrid_dataset.py --api-key "YOUR_KEY" \
#     --model gemini-2.0-flash --per-quality 120 --per-tone 120 --batch-n 6 \
#     --temperature 0.9 --max-calls 180 \
#     --seed-out data/eval_seed.csv \
#     --augment-factor 3 \
#     --semantic-dedupe --dedupe-thresh 0.92 \
#     --final-out data/eval_training_data_hybrid.csv
# """

# import os
# import re
# import csv
# import sys
# import json
# import time
# import math
# import random
# import string
# import argparse
# import traceback
# from typing import List, Dict, Any, Optional, Tuple

# import pandas as pd
# from tqdm import tqdm

# # --- Optional NLP deps (auto-handled) ---
# import nltk
# from nltk.corpus import wordnet as wn
# from nltk.tokenize import word_tokenize

# # --- ML utils ---
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Try sentence-transformers (optional)
# _HAVE_ST = False
# try:
#     from sentence_transformers import SentenceTransformer
#     _HAVE_ST = True
# except Exception:
#     _HAVE_ST = False

# # --- Gemini ---
# try:
#     import google.generativeai as genai
# except Exception as e:
#     print("[WARN] google-generativeai not installed. Install with: pip install google-generativeai")
#     genai = None


# QUALITY_LABELS = ["GOOD", "WEAK", "CONFUSED"]
# TONE_LABELS = ["analytical", "curious", "frustrated", "neutral", "playful"]

# # -------------------------
# # NLTK BOOTSTRAP
# # -------------------------
# def ensure_nltk_data() -> bool:
#     """
#     Ensure NLTK data needed for augmentation is available.
#     Returns True if everything is ready; False if we should skip synonym swaps.
#     """
#     needed = ["punkt_tab", "punkt", "wordnet", "omw-1.4"]
#     ok = True
#     for pkg in needed:
#         try:
#             if pkg == "punkt_tab":
#                 nltk.data.find("tokenizers/punkt_tab/english/")
#             elif pkg == "punkt":
#                 nltk.data.find("tokenizers/punkt") 
#             elif pkg == "wordnet":
#                 nltk.data.find("corpora/wordnet")
#             elif pkg == "omw-1.4":
#                 nltk.data.find("corpora/omw-1.4")
#         except LookupError:
#             try:
#                 print(f"[INFO] Downloading NLTK resource: {pkg} …")
#                 nltk.download(pkg, quiet=True)
#             except Exception as e:
#                 print(f"[WARN] Could not download NLTK resource '{pkg}': {e}")
#                 ok = False

#     # Verify fallback path
#     try:
#         nltk.data.find("tokenizers/punkt_tab/english/")
#     except LookupError:
#         try:
#             nltk.data.find("tokenizers/punkt")
#         except LookupError:
#             ok = False

#     for corp in ["wordnet", "omw-1.4"]:
#         try:
#             nltk.data.find(f"corpora/{corp}")
#         except LookupError:
#             ok = False

#     if ok:
#         print("[OK] NLTK resources are ready.")
#     else:
#         print("[WARN] NLTK resources incomplete. Will degrade augmentation (no synonym swaps).")
#     return ok


# # -------------------------
# # Text helpers
# # -------------------------
# _PUNCT = set(string.punctuation)

# def simple_tokenize(text: str) -> List[str]:
#     return re.findall(r"\w+|\S", text)

# def synonym_swap(text: str, prob: float = 0.12, enable_synonyms: bool = True) -> str:
#     """
#     Replace some content words with WordNet synonyms. If enable_synonyms=False,
#     returns text unchanged (fallback).
#     """
#     if not enable_synonyms:
#         return text

#     try:
#         tokens = word_tokenize(text)
#     except Exception:
#         tokens = simple_tokenize(text)

#     def get_synonym(word: str) -> Optional[str]:
#         synsets = wn.synsets(word)
#         if not synsets:
#             return None
#         lemmas = set()
#         for s in synsets:
#             for l in s.lemmas():
#                 cand = l.name().replace("_", " ")
#                 if cand.lower() != word.lower():
#                     lemmas.add(cand)
#         if not lemmas:
#             return None
#         return random.choice(sorted(list(lemmas)))

#     new_tokens = []
#     for tok in tokens:
#         if tok.isalpha() and random.random() < prob:
#             syn = get_synonym(tok)
#             new_tokens.append(syn if syn else tok)
#         else:
#             new_tokens.append(tok)
#     return " ".join(new_tokens)

# def slight_noise(text: str, char_drop_p: float = 0.02, char_swap_p: float = 0.02) -> str:
#     """
#     Add mild character-level noise without breaking readability.
#     """
#     chars = list(text)
#     # random drops
#     keep = []
#     for c in chars:
#         if random.random() < char_drop_p and c not in "\n":
#             continue
#         keep.append(c)
#     chars = keep

#     # random adjacent swaps
#     i = 0
#     while i < len(chars) - 1:
#         if random.random() < char_swap_p:
#             chars[i], chars[i+1] = chars[i+1], chars[i]
#             i += 2
#         else:
#             i += 1
#     return "".join(chars)

# def mask_tokens(text: str, mask_p: float = 0.06, mask_token: str = "[MASK]") -> str:
#     """
#     Replace a few tokens with [MASK]; keeps structure.
#     """
#     toks = simple_tokenize(text)
#     out = []
#     for t in toks:
#         if t.isalpha() and random.random() < mask_p:
#             out.append(mask_token)
#         else:
#             out.append(t)
#     return " ".join(out)

# def enforce_sentence_style(text: str) -> str:
#     text = re.sub(r"\s+", " ", text).strip()
#     if text and text[-1] not in ".?!":
#         text += "."
#     return text

# # -------------------------
# # Gemini Seeding
# # -------------------------
# SEED_SYSTEM = (
#     "You are a creative data generator for training a classifier that evaluates user responses "
#     "in a Socratic design-thinking assistant.\n"
#     "You will produce rows with fields: question, response, evaluation (GOOD/WEAK/CONFUSED), tone (analytical/curious/frustrated/neutral/playful).\n"
#     "Rules:\n"
#     " - Make responses natural and varied; do NOT follow templates; avoid repetition.\n"
#     " - Ensure labels accurately reflect the response quality and tone.\n"
#     " - Spread topics across product design, research, education, data analysis, creative writing, etc.\n"
#     " - Keep responses 1–4 sentences, concise but realistic.\n"
# )

# SEED_USER_FMT = (
#     "Generate {n} JSON lines. Each line is a JSON object with keys: question, response, evaluation, tone.\n"
#     "Ensure roughly balanced coverage for evaluation={evals} and tone={tones} across this batch.\n"
#     "Return ONLY JSON lines (one object per line), no extra text."
# )

# def configure_gemini(api_key: str, model_name: str, temperature: float = 0.9):
#     if genai is None:
#         raise RuntimeError("google-generativeai is not installed.")
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel(model_name)
#     return model, {"temperature": temperature}

# def chunk_list(lst: List[Any], n: int) -> List[List[Any]]:
#     for i in range(0, len(lst), n):
#         yield lst[i:i+n]

# def call_gemini_jsonl(model, params: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
#     """
#     Ask model for n JSONL rows, parse safely.
#     """
#     prompt = SEED_SYSTEM + "\n" + SEED_USER_FMT.format(
#         n=n, evals=QUALITY_LABELS, tones=TONE_LABELS
#     )
#     resp = model.generate_content([prompt], generation_config=params)
#     text = resp.text or ""
#     rows = []
#     for line in text.splitlines():
#         line = line.strip()
#         if not line:
#             continue
#         # tolerate extra commas / stray text
#         try:
#             obj = json.loads(line)
#             if isinstance(obj, dict) and all(k in obj for k in ("question", "response", "evaluation", "tone")):
#                 rows.append({
#                     "question": enforce_sentence_style(str(obj["question"])),
#                     "response": enforce_sentence_style(str(obj["response"])),
#                     "evaluation": str(obj["evaluation"]).strip().upper(),
#                     "tone": str(obj["tone"]).strip().lower(),
#                 })
#         except Exception:
#             continue
#     return rows

# def seed_with_gemini(api_key: str,
#                      model_name: str,
#                      per_quality: int,
#                      per_tone: int,
#                      batch_n: int,
#                      temperature: float,
#                      max_calls: int) -> pd.DataFrame:
#     """
#     Build a seed set targeting roughly per_quality * len(QUALITY_LABELS) rows and
#     per_tone * len(TONE_LABELS) rows across batches.
#     """
#     target_quality = per_quality * len(QUALITY_LABELS)
#     target_tone = per_tone * len(TONE_LABELS)
#     target_rows = max(target_quality, target_tone)

#     model, params = configure_gemini(api_key, model_name, temperature=temperature)

#     rows: List[Dict[str, Any]] = []
#     calls_made = 0

#     batches = math.ceil(target_rows / batch_n)
#     print(f"[INFO] Target seed rows: ~{target_rows}  ({per_quality}/quality, {per_tone}/tone)")
#     print(f"[INFO] Calling Gemini in {batches} batch(es), {batch_n} per call")

#     for _ in tqdm(range(batches), desc="Gemini seeding"):
#         if calls_made >= max_calls:
#             print("[WARN] Reached --max-calls limit; stopping early.")
#             break

#         for attempt in range(1, 9):  # retry with backoff
#             try:
#                 new_rows = call_gemini_jsonl(model, params, batch_n)
#                 rows.extend(new_rows)
#                 calls_made += 1
#                 break
#             except Exception as e:
#                 # Handle quota-like signals
#                 msg = str(e)
#                 if "429" in msg or "quota" in msg.lower():
#                     sleep_s = min(30.0, 1.5 * attempt + random.random())
#                     print(f"[WARN] Quota/rate limit (attempt {attempt}/8). Sleeping {sleep_s:.1f}s")
#                     time.sleep(sleep_s)
#                 else:
#                     print(f"[WARN] Gemini call failed: {e}")
#                     time.sleep(1.0)
#         else:
#             print("[WARN] Giving up this batch after retries.")

#     if not rows:
#         raise RuntimeError("Seeding produced zero rows. Check API key/quota.")

#     df = pd.DataFrame(rows)
#     # Light normalization
#     df["evaluation"] = df["evaluation"].str.upper().map(lambda x: x if x in QUALITY_LABELS else "GOOD")
#     df["tone"] = df["tone"].str.lower().map(lambda x: x if x in TONE_LABELS else "neutral")
#     return df


# # -------------------------
# # Local augmentation
# # -------------------------
# def augment_response_once(text: str, nltk_ready: bool) -> str:
#     # sequence of small, safe transforms
#     s = text
#     s = synonym_swap(s, prob=0.12, enable_synonyms=nltk_ready)
#     s = slight_noise(s, char_drop_p=0.01, char_swap_p=0.01)
#     s = mask_tokens(s, mask_p=0.05, mask_token="[MASK]")
#     return enforce_sentence_style(s)

# def augment_pair(row: Dict[str, Any], factor: int, nltk_ready: bool) -> List[Dict[str, Any]]:
#     """
#     Duplicate a (question, response, evaluation, tone) pair with small variations
#     of the response. Labels remain the same.
#     """
#     out = []
#     for _ in range(factor):
#         new_resp = augment_response_once(str(row["response"]), nltk_ready=nltk_ready)
#         out.append({
#             "question": row["question"],
#             "response": new_resp,
#             "evaluation": row["evaluation"],
#             "tone": row["tone"],
#         })
#     return out

# def augment_dataframe(df: pd.DataFrame, factor: int, nltk_ready: bool) -> pd.DataFrame:
#     rows = df.to_dict(orient="records")
#     aug_rows: List[Dict[str, Any]] = []
#     for r in tqdm(rows, desc="Augmenting locally"):
#         aug_rows.extend(augment_pair(r, factor=factor, nltk_ready=nltk_ready))
#     return pd.DataFrame(aug_rows)


# # -------------------------
# # Semantic dedupe
# # -------------------------
# def semantic_dedupe(df: pd.DataFrame,
#                     text_cols: Tuple[str, str] = ("question", "response"),
#                     thresh: float = 0.92) -> pd.DataFrame:
#     """
#     Remove near-duplicates based on cosine similarity over combined text.
#     If sentence-transformers is available, it will be used; otherwise TF-IDF.
#     """
#     combo = (df[text_cols[0]].astype(str) + " || " + df[text_cols[1]].astype(str)).tolist()
#     keep_mask = np.ones(len(combo), dtype=bool)

#     if _HAVE_ST:
#         print("[INFO] Semantic dedupe: using Sentence-Transformers (all-MiniLM-L6-v2)")
#         model = SentenceTransformer("all-MiniLM-L6-v2")
#         emb = model.encode(combo, show_progress_bar=True, batch_size=128, normalize_embeddings=True)

#         # Greedy dedupe using previously kept *vectors* (not indices)
#         seen_vecs: list[np.ndarray] = []
#         for i in tqdm(range(len(emb)), desc="Dedupe"):
#             if not keep_mask[i]:
#                 continue

#             ei = emb[i]
#             is_dup = False
#             for prev in seen_vecs:
#                 # emb is already L2-normalized → dot = cosine similarity
#                 sim = float(np.dot(ei, prev))
#                 if sim >= thresh:
#                     keep_mask[i] = False
#                     is_dup = True
#                     break

#             if not is_dup:
#                 seen_vecs.append(ei)
#     else:
#         print("[INFO] Semantic dedupe: using TF-IDF cosine (install sentence-transformers for better recall)")
#         vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
#         X = vec.fit_transform(combo)
#         # Greedy dedupe (chunked to keep memory ok)
#         seen_vectors = []
#         for i in tqdm(range(X.shape[0]), desc="Dedupe"):
#             if not keep_mask[i]:
#                 continue
#             xi = X[i]
#             for xv in seen_vectors:
#                 # cosine(Xi, Xv)
#                 num = xi.multiply(xv).sum()
#                 den = np.sqrt(xi.multiply(xi).sum()) * np.sqrt(xv.multiply(xv).sum())
#                 sim = float(num / (den + 1e-9))
#                 if sim >= thresh:
#                     keep_mask[i] = False
#                     break
#             if keep_mask[i]:
#                 seen_vectors.append(xi)

#     return df.loc[keep_mask].reset_index(drop=True)


# # -------------------------
# # CLI / MAIN
# # -------------------------
# def parse_args():
#     p = argparse.ArgumentParser(description="Hybrid dataset builder (Gemini seed + local augmentation)")
#     p.add_argument("--api-key", type=str, default=os.getenv("GEMINI_API_KEY"),
#                    help="Gemini API key or set GEMINI_API_KEY")
#     p.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model name")
#     p.add_argument("--per-quality", type=int, default=120, help="Rows per quality label in seed (approx)")
#     p.add_argument("--per-tone", type=int, default=120, help="Rows per tone label in seed (approx)")
#     p.add_argument("--batch-n", type=int, default=6, help="Rows requested per Gemini call")
#     p.add_argument("--temperature", type=float, default=0.9, help="Gemini temperature")
#     p.add_argument("--max-calls", type=int, default=180, help="Hard limit on Gemini calls")

#     p.add_argument("--seed-out", type=str, required=True, help="Path to write the seed CSV")
#     p.add_argument("--augment-factor", type=int, default=3, help="Local copies per seed row")

#     p.add_argument("--semantic-dedupe", action="store_true", help="Enable semantic dedupe")
#     p.add_argument("--dedupe-thresh", type=float, default=0.92, help="Cosine similarity threshold for dedupe")

#     p.add_argument("--final-out", type=str, required=True, help="Path to write the final (augmented) CSV")

#     return p.parse_args()


# def main():
#     args = parse_args()

#     if not args.api_key:
#         print("[FATAL] Provide --api-key or set GEMINI_API_KEY")
#         sys.exit(1)

#     os.makedirs(os.path.dirname(os.path.abspath(args.seed_out)), exist_ok=True)
#     os.makedirs(os.path.dirname(os.path.abspath(args.final_out)), exist_ok=True)

#     # 1) Seed via Gemini (balanced-ish)
#     print(f"[INFO] Building seed set with Gemini → {args.seed_out}")
#     try:
#         seed_df = seed_with_gemini(
#             api_key=args.api_key,
#             model_name=args.model,
#             per_quality=args.per_quality,
#             per_tone=args.per_tone,
#             batch_n=args.batch_n,
#             temperature=args.temperature,
#             max_calls=args.max_calls
#         )
#     except Exception as e:
#         print(f"[FATAL] Gemini seeding failed: {e}")
#         traceback.print_exc()
#         sys.exit(1)

#     # Enforce columns + clean
#     seed_df = seed_df[["question", "response", "evaluation", "tone"]].copy()
#     seed_df.dropna(subset=["question", "response"], inplace=True)
#     # Basic uniqueness by pair
#     seed_df.drop_duplicates(subset=["question", "response"], inplace=True)

#     seed_df.to_csv(args.seed_out, index=False, quoting=csv.QUOTE_MINIMAL)
#     print(f"[OK] Seed rows written: {len(seed_df)}  →  {os.path.abspath(args.seed_out)}")

#     # 2) NLTK bootstrap
#     nltk_ready = ensure_nltk_data()

#     # 3) Local augmentation
#     print(f"[INFO] Local augmentation ×{args.augment_factor}")
#     aug_df = augment_dataframe(seed_df, factor=args.augment_factor, nltk_ready=nltk_ready)

#     # 4) Combine seed + augmented
#     final_df = pd.concat([seed_df, aug_df], ignore_index=True)

#     # 5) Optional semantic dedupe
#     if args.semantic_dedupe:
#         print("[INFO] Semantic dedupe enabled")
#         before = len(final_df)
#         final_df = semantic_dedupe(final_df, text_cols=("question", "response"), thresh=args.dedupe_thresh)
#         after = len(final_df)
#         print(f"[OK] Dedupe reduced rows: {before} → {after}  (-{before - after})")

#     # 6) Final shuffle + write
#     final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
#     final_df.to_csv(args.final_out, index=False, quoting=csv.QUOTE_MINIMAL)

#     # 7) Tiny summary
#     print("\n=== Hybrid Dataset Summary ===")
#     print(f"Seed rows:         {len(seed_df)}")
#     print(f"Augmented rows:    {len(aug_df)}")
#     print(f"Final rows:        {len(final_df)}")
#     print(f"Saved to:          {os.path.abspath(args.final_out)}")

#     # Label distributions
#     q_counts = final_df["evaluation"].value_counts().to_dict()
#     t_counts = final_df["tone"].value_counts().to_dict()
#     print("\nLabel balance (evaluation):", q_counts)
#     print("Label balance (tone):       ", t_counts)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hybrid_dataset.py
- Hybrid data builder for SLM evaluation labels:
  * Seed a balanced high-quality set via Gemini (quality x tone).
  * OR reuse an existing seed CSV (skip Gemini entirely).
  * Expand locally with lightweight augmentation (no more API calls).
  * Optional semantic dedupe.
- Robust NLTK bootstrap + graceful fallback (no crashes).
- Clear logs and progress bars.

Requirements:
  pip install google-generativeai pandas scikit-learn tqdm nltk
  (optional) pip install sentence-transformers

Usage examples:

  # Full pipeline (with Gemini seeding)
  python hybrid_dataset.py --api-key "YOUR_KEY" \
    --model gemini-2.0-flash --per-quality 120 --per-tone 120 --batch-n 6 \
    --temperature 0.9 --max-calls 180 \
    --seed-out data/eval_seed.csv \
    --augment-factor 3 \
    --semantic-dedupe --dedupe-thresh 0.92 \
    --final-out data/eval_training_data_hybrid.csv

  # Reuse existing seed (NO Gemini calls)
  python hybrid_dataset.py \
    --seed-in data/eval_seed.csv \
    --seed-out data/eval_seed.csv \
    --augment-factor 3 \
    --semantic-dedupe --dedupe-thresh 0.92 \
    --final-out data/eval_training_data_hybrid.csv
"""

import os
import re
import csv
import sys
import json
import time
import math
import random
import string
import argparse
import traceback
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# --- Optional NLP deps (auto-handled) ---
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

# --- ML utils ---
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Try sentence-transformers (optional)
_HAVE_ST = False
try:
    from sentence_transformers import SentenceTransformer
    _HAVE_ST = True
except Exception:
    _HAVE_ST = False

# --- Gemini ---
try:
    import google.generativeai as genai
except Exception as e:
    print("[WARN] google-generativeai not installed. Install with: pip install google-generativeai")
    genai = None


QUALITY_LABELS = ["GOOD", "WEAK", "CONFUSED"]
TONE_LABELS = ["analytical", "curious", "frustrated", "neutral", "playful"]

# -------------------------
# NLTK BOOTSTRAP
# -------------------------
def ensure_nltk_data() -> bool:
    """
    Ensure NLTK data needed for augmentation is available.
    Returns True if everything is ready; False if we should skip synonym swaps.
    """
    needed = ["punkt_tab", "punkt", "wordnet", "omw-1.4"]
    ok = True
    for pkg in needed:
        try:
            if pkg == "punkt_tab":
                nltk.data.find("tokenizers/punkt_tab/english/")
            elif pkg == "punkt":
                nltk.data.find("tokenizers/punkt")
            elif pkg == "wordnet":
                nltk.data.find("corpora/wordnet")
            elif pkg == "omw-1.4":
                nltk.data.find("corpora/omw-1.4")
        except LookupError:
            try:
                print(f"[INFO] Downloading NLTK resource: {pkg} …")
                nltk.download(pkg, quiet=True)
            except Exception as e:
                print(f"[WARN] Could not download NLTK resource '{pkg}': {e}")
                ok = False

    # Verify fallback path
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            ok = False

    for corp in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{corp}")
        except LookupError:
            ok = False

    if ok:
        print("[OK] NLTK resources are ready.")
    else:
        print("[WARN] NLTK resources incomplete. Will degrade augmentation (no synonym swaps).")
    return ok


# -------------------------
# Text helpers
# -------------------------
_PUNCT = set(string.punctuation)

def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|\S", text)

def synonym_swap(text: str, prob: float = 0.12, enable_synonyms: bool = True) -> str:
    """
    Replace some content words with WordNet synonyms. If enable_synonyms=False,
    returns text unchanged (fallback).
    """
    if not enable_synonyms:
        return text

    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = simple_tokenize(text)

    def get_synonym(word: str) -> Optional[str]:
        synsets = wn.synsets(word)
        if not synsets:
            return None
        lemmas = set()
        for s in synsets:
            for l in s.lemmas():
                cand = l.name().replace("_", " ")
                if cand.lower() != word.lower():
                    lemmas.add(cand)
        if not lemmas:
            return None
        return random.choice(sorted(list(lemmas)))

    new_tokens = []
    for tok in tokens:
        if tok.isalpha() and random.random() < prob:
            syn = get_synonym(tok)
            new_tokens.append(syn if syn else tok)
        else:
            new_tokens.append(tok)
    return " ".join(new_tokens)

def slight_noise(text: str, char_drop_p: float = 0.02, char_swap_p: float = 0.02) -> str:
    """
    Add mild character-level noise without breaking readability.
    """
    chars = list(text)
    # random drops
    keep = []
    for c in chars:
        if random.random() < char_drop_p and c not in "\n":
            continue
        keep.append(c)
    chars = keep

    # random adjacent swaps
    i = 0
    while i < len(chars) - 1:
        if random.random() < char_swap_p:
            chars[i], chars[i+1] = chars[i+1], chars[i]
            i += 2
        else:
            i += 1
    return "".join(chars)

def mask_tokens(text: str, mask_p: float = 0.06, mask_token: str = "[MASK]") -> str:
    """
    Replace a few tokens with [MASK]; keeps structure.
    """
    toks = simple_tokenize(text)
    out = []
    for t in toks:
        if t.isalpha() and random.random() < mask_p:
            out.append(mask_token)
        else:
            out.append(t)
    return " ".join(out)

def enforce_sentence_style(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".?!":
        text += "."
    return text


# -------------------------
# Gemini Seeding
# -------------------------
SEED_SYSTEM = (
    "You are a creative data generator for training a classifier that evaluates user responses "
    "in a Socratic design-thinking assistant.\n"
    "You will produce rows with fields: question, response, evaluation (GOOD/WEAK/CONFUSED), tone (analytical/curious/frustrated/neutral/playful).\n"
    "Rules:\n"
    " - Make responses natural and varied; do NOT follow templates; avoid repetition.\n"
    " - Ensure labels accurately reflect the response quality and tone.\n"
    " - Spread topics across product design, research, education, data analysis, creative writing, etc.\n"
    " - Keep responses 1–4 sentences, concise but realistic.\n"
)

SEED_USER_FMT = (
    "Generate {n} JSON lines. Each line is a JSON object with keys: question, response, evaluation, tone.\n"
    "Ensure roughly balanced coverage for evaluation={evals} and tone={tones} across this batch.\n"
    "Return ONLY JSON lines (one object per line), no extra text."
)

def configure_gemini(api_key: str, model_name: str, temperature: float = 0.9):
    if genai is None:
        raise RuntimeError("google-generativeai is not installed.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    return model, {"temperature": temperature}

def call_gemini_jsonl(model, params: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """
    Ask model for n JSONL rows, parse safely.
    """
    prompt = SEED_SYSTEM + "\n" + SEED_USER_FMT.format(
        n=n, evals=QUALITY_LABELS, tones=TONE_LABELS
    )
    resp = model.generate_content([prompt], generation_config=params)
    text = resp.text or ""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # tolerate extra commas / stray text
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and all(k in obj for k in ("question", "response", "evaluation", "tone")):
                rows.append({
                    "question": enforce_sentence_style(str(obj["question"])),
                    "response": enforce_sentence_style(str(obj["response"])),
                    "evaluation": str(obj["evaluation"]).strip().upper(),
                    "tone": str(obj["tone"]).strip().lower(),
                })
        except Exception:
            continue
    return rows

def seed_with_gemini(api_key: str,
                     model_name: str,
                     per_quality: int,
                     per_tone: int,
                     batch_n: int,
                     temperature: float,
                     max_calls: int) -> pd.DataFrame:
    """
    Build a seed set targeting roughly per_quality * len(QUALITY_LABELS) rows and
    per_tone * len(TONE_LABELS) rows across batches.
    """
    target_quality = per_quality * len(QUALITY_LABELS)
    target_tone = per_tone * len(TONE_LABELS)
    target_rows = max(target_quality, target_tone)

    model, params = configure_gemini(api_key, model_name, temperature=temperature)

    rows: List[Dict[str, Any]] = []
    calls_made = 0

    batches = math.ceil(target_rows / batch_n)
    print(f"[INFO] Target seed rows: ~{target_rows}  ({per_quality}/quality, {per_tone}/tone)")
    print(f"[INFO] Calling Gemini in {batches} batch(es), {batch_n} per call")

    for _ in tqdm(range(batches), desc="Gemini seeding"):
        if calls_made >= max_calls:
            print("[WARN] Reached --max-calls limit; stopping early.")
            break

        for attempt in range(1, 9):  # retry with backoff
            try:
                new_rows = call_gemini_jsonl(model, params, batch_n)
                rows.extend(new_rows)
                calls_made += 1
                break
            except Exception as e:
                # Handle quota-like signals
                msg = str(e)
                if "429" in msg or "quota" in msg.lower():
                    sleep_s = min(30.0, 1.5 * attempt + random.random())
                    print(f"[WARN] Quota/rate limit (attempt {attempt}/8). Sleeping {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                else:
                    print(f"[WARN] Gemini call failed: {e}")
                    time.sleep(1.0)
        else:
            print("[WARN] Giving up this batch after retries.")

    if not rows:
        raise RuntimeError("Seeding produced zero rows. Check API key/quota.")

    df = pd.DataFrame(rows)
    # Light normalization
    df["evaluation"] = df["evaluation"].str.upper().map(lambda x: x if x in QUALITY_LABELS else "GOOD")
    df["tone"] = df["tone"].str.lower().map(lambda x: x if x in TONE_LABELS else "neutral")
    return df


# -------------------------
# Local augmentation
# -------------------------
def augment_response_once(text: str, nltk_ready: bool) -> str:
    # sequence of small, safe transforms
    s = text
    s = synonym_swap(s, prob=0.12, enable_synonyms=nltk_ready)
    s = slight_noise(s, char_drop_p=0.01, char_swap_p=0.01)
    s = mask_tokens(s, mask_p=0.05, mask_token="[MASK]")
    return enforce_sentence_style(s)

def augment_pair(row: Dict[str, Any], factor: int, nltk_ready: bool) -> List[Dict[str, Any]]:
    """
    Duplicate a (question, response, evaluation, tone) pair with small variations
    of the response. Labels remain the same.
    """
    out = []
    for _ in range(factor):
        new_resp = augment_response_once(str(row["response"]), nltk_ready=nltk_ready)
        out.append({
            "question": row["question"],
            "response": new_resp,
            "evaluation": row["evaluation"],
            "tone": row["tone"],
        })
    return out

def augment_dataframe(df: pd.DataFrame, factor: int, nltk_ready: bool) -> pd.DataFrame:
    rows = df.to_dict(orient="records")
    aug_rows: List[Dict[str, Any]] = []
    for r in tqdm(rows, desc="Augmenting locally"):
        aug_rows.extend(augment_pair(r, factor=factor, nltk_ready=nltk_ready))
    return pd.DataFrame(aug_rows)


# -------------------------
# Semantic dedupe
# -------------------------
def semantic_dedupe(df: pd.DataFrame,
                    text_cols: Tuple[str, str] = ("question", "response"),
                    thresh: float = 0.92) -> pd.DataFrame:
    """
    Remove near-duplicates based on cosine similarity over combined text.
    If sentence-transformers is available, it will be used; otherwise TF-IDF.
    """
    combo = (df[text_cols[0]].astype(str) + " || " + df[text_cols[1]].astype(str)).tolist()
    keep_mask = np.ones(len(combo), dtype=bool)

    if _HAVE_ST:
        print("[INFO] Semantic dedupe: using Sentence-Transformers (all-MiniLM-L6-v2)")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(combo, show_progress_bar=True, batch_size=128, normalize_embeddings=True)

        # Greedy dedupe using previously kept *vectors* (not indices)
        seen_vecs: list[np.ndarray] = []
        for i in tqdm(range(len(emb)), desc="Dedupe"):
            if not keep_mask[i]:
                continue

            ei = emb[i]
            is_dup = False
            for prev in seen_vecs:
                # emb is already L2-normalized → dot = cosine similarity
                sim = float(np.dot(ei, prev))
                if sim >= thresh:
                    keep_mask[i] = False
                    is_dup = True
                    break

            if not is_dup:
                seen_vecs.append(ei)
    else:
        print("[INFO] Semantic dedupe: using TF-IDF cosine (install sentence-transformers for better recall)")
        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
        X = vec.fit_transform(combo)
        # Greedy dedupe
        seen_vectors = []
        for i in tqdm(range(X.shape[0]), desc="Dedupe"):
            if not keep_mask[i]:
                continue
            xi = X[i]
            for xv in seen_vectors:
                num = xi.multiply(xv).sum()
                den = np.sqrt(xi.multiply(xi).sum()) * np.sqrt(xv.multiply(xv).sum())
                sim = float(num / (den + 1e-9))
                if sim >= thresh:
                    keep_mask[i] = False
                    break
            if keep_mask[i]:
                seen_vectors.append(xi)

    return df.loc[keep_mask].reset_index(drop=True)


# -------------------------
# CLI / MAIN
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid dataset builder (Gemini seed + local augmentation, or reuse existing seed)"
    )

    # Gemini-related (only needed when NOT using --seed-in)
    p.add_argument("--api-key", type=str, default=os.getenv("GEMINI_API_KEY"),
                   help="Gemini API key or set GEMINI_API_KEY")
    p.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model name")
    p.add_argument("--per-quality", type=int, default=120, help="Rows per quality label in seed (approx)")
    p.add_argument("--per-tone", type=int, default=120, help="Rows per tone label in seed (approx)")
    p.add_argument("--batch-n", type=int, default=6, help="Rows requested per Gemini call")
    p.add_argument("--temperature", type=float, default=0.9, help="Gemini temperature")
    p.add_argument("--max-calls", type=int, default=180, help="Hard limit on Gemini calls")

    # Seed handling
    p.add_argument("--seed-in", type=str, default=None,
                   help="Existing seed CSV to reuse (skips Gemini seeding if provided)")
    p.add_argument("--seed-out", type=str, required=True,
                   help="Path to write the seed CSV (or re-save when using --seed-in)")

    # Augmentation & dedupe
    p.add_argument("--augment-factor", type=int, default=3, help="Local copies per seed row")
    p.add_argument("--semantic-dedupe", action="store_true", help="Enable semantic dedupe")
    p.add_argument("--dedupe-thresh", type=float, default=0.92, help="Cosine similarity threshold for dedupe")

    p.add_argument("--final-out", type=str, required=True, help="Path to write the final (augmented) CSV")

    return p.parse_args()


def main():
    args = parse_args()

    # Make sure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(args.seed_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.final_out)), exist_ok=True)

    # 1) Seed stage: either reuse existing CSV or call Gemini
    if args.seed_in:
        if not os.path.exists(args.seed_in):
            print(f"[FATAL] --seed-in file does not exist: {args.seed_in}")
            sys.exit(1)
        print(f"[INFO] Reusing existing seed CSV → {args.seed_in}")
        seed_df = pd.read_csv(args.seed_in)

    else:
        # No seed_in provided → we must call Gemini and need an API key
        if not args.api_key:
            print("[FATAL] Provide --api-key or set GEMINI_API_KEY (or use --seed-in to reuse an existing seed)")
            sys.exit(1)

        print(f"[INFO] Building seed set with Gemini → {args.seed_out}")
        try:
            seed_df = seed_with_gemini(
                api_key=args.api_key,
                model_name=args.model,
                per_quality=args.per_quality,
                per_tone=args.per_tone,
                batch_n=args.batch_n,
                temperature=args.temperature,
                max_calls=args.max_calls
            )
        except Exception as e:
            print(f"[FATAL] Gemini seeding failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    # 1b) Enforce schema & cleanliness for both paths
    expected_cols = ["question", "response", "evaluation", "tone"]
    missing = [c for c in expected_cols if c not in seed_df.columns]
    if missing:
        print(f"[FATAL] Seed CSV missing required columns: {missing}")
        sys.exit(1)

    seed_df = seed_df[expected_cols].copy()
    seed_df.dropna(subset=["question", "response"], inplace=True)
    seed_df.drop_duplicates(subset=["question", "response"], inplace=True)

    # Normalize labels in case seed-in came from elsewhere
    seed_df["evaluation"] = (
        seed_df["evaluation"]
        .astype(str).str.upper()
        .map(lambda x: x if x in QUALITY_LABELS else "GOOD")
    )
    seed_df["tone"] = (
        seed_df["tone"]
        .astype(str).str.lower()
        .map(lambda x: x if x in TONE_LABELS else "neutral")
    )

    # Write cleaned seed to seed_out (even if reused)
    seed_df.to_csv(args.seed_out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[OK] Seed rows written: {len(seed_df)}  →  {os.path.abspath(args.seed_out)}")

    # 2) NLTK bootstrap
    nltk_ready = ensure_nltk_data()

    # 3) Local augmentation
    print(f"[INFO] Local augmentation ×{args.augment_factor}")
    aug_df = augment_dataframe(seed_df, factor=args.augment_factor, nltk_ready=nltk_ready)

    # 4) Combine seed + augmented
    final_df = pd.concat([seed_df, aug_df], ignore_index=True)

    # 5) Optional semantic dedupe
    if args.semantic_dedupe:
        print("[INFO] Semantic dedupe enabled")
        before = len(final_df)
        final_df = semantic_dedupe(final_df, text_cols=("question", "response"), thresh=args.dedupe_thresh)
        after = len(final_df)
        print(f"[OK] Dedupe reduced rows: {before} → {after}  (-{before - after})")

    # 6) Final shuffle + write
    final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    final_df.to_csv(args.final_out, index=False, quoting=csv.QUOTE_MINIMAL)

    # 7) Tiny summary
    print("\n=== Hybrid Dataset Summary ===")
    print(f"Seed rows:         {len(seed_df)}")
    print(f"Augmented rows:    {len(aug_df)}")
    print(f"Final rows:        {len(final_df)}")
    print(f"Saved to:          {os.path.abspath(args.final_out)}")

    # Label distributions
    q_counts = final_df["evaluation"].value_counts().to_dict()
    t_counts = final_df["tone"].value_counts().to_dict()
    print("\nLabel balance (evaluation):", q_counts)
    print("Label balance (tone):       ", t_counts)


if __name__ == "__main__":
    main()
