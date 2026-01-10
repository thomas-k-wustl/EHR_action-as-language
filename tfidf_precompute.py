import os
import pandas as pd
import yaml
import argparse

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import save_npz


def load_config(yaml_path):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config


def find_valid_prefix(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No valid path_prefix found in config.")

def extract_actions_from_parquet(file_path):
    try:
        new_strings = (
            pd.read_parquet(file_path, columns=["METRIC_NAME"])["METRIC_NAME"]
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)  # collapse multiple spaces
        )
        new_strings = new_strings[new_strings.str.len() > 0]
        return set(new_strings.tolist())
    except Exception as e:
        print(f"Skipping {file_path} due to error: {e}")
        return set()

def tfidf_vectorize_action_texts(yaml_config_path):
    config = load_config(yaml_config_path)
    path_prefix = find_valid_prefix(config["path_prefix"])
    data_path = os.path.join(path_prefix, config["audit_log_cache"])
    wpe_list_path = os.path.join(path_prefix, config["wpe_list"])
    model_name = config.get("model")
    model_cache_path = config.get("model_cache_path")
    save_path = os.path.join(path_prefix, model_cache_path + model_name.split("/")[1],
                             config['HF_model_name'])

    user_id_df = pd.read_csv(wpe_list_path)
    if "exclusion_list" in config and config["exclusion_list"]:
        user_id_df = user_id_df[~user_id_df["idx"].isin(config["exclusion_list"])]

    wpe_idx_list = user_id_df["idx"].tolist()
    valid_action_texts = set()

    file_paths = []
    for wpe_idx in wpe_idx_list:
        for case_or_control in ["case", "control"]:
            file_name = str(wpe_idx) + config["audit_log_cache_file"][case_or_control]
            file_path = os.path.join(data_path, str(wpe_idx), file_name)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                file_paths.append(file_path)

    partial_sets = joblib.Parallel(
        n_jobs=-1,  # or os.cpu_count()
        backend="loky",  # processes (bypasses GIL; good for I/O + CPU parsing)
        prefer="processes",
        batch_size=32,  # send paths in small batches to cut overhead
        pre_dispatch="2*n_jobs"  # don't dispatch everything at once
    )(joblib.delayed(extract_actions_from_parquet)(p) for p in file_paths)

    global_actions = set().union(*partial_sets)
    valid_action_texts = sorted(list(global_actions))

    # During fitting, the vectorizer will extract character n-grams (3–5) across each string, build a vocabulary of all such substrings, and compute IDF weights.
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), dtype=np.float32)
    vectorizer.fit(valid_action_texts)

    A_valid = vectorizer.transform(valid_action_texts)
    # we'll use cosine similarity measure, so store an L2-normalized matrix to skip divisions at query time
    # Then cosine becomes a simple dot during LLM chunk inference: sims = A_valid_l2 @ v_chunk_l2
    A_valid_l2 = normalize(A_valid, norm="l2", axis=1, copy=True)

    # Save to disk
    output_dir = os.path.dirname(save_path)
    joblib.dump(valid_action_texts, os.path.join(output_dir, "valid_action_texts.txt"))
    joblib.dump(valid_action_texts, os.path.join(output_dir, "valid_action_texts.pkl"))
    print(f"[SANITY CHECK] Size of valid action texts: {len(valid_action_texts)}")
    # Save to disk
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer_char3_5.pkl"))
    print(
        f"Saved fitted TF-IDF vectorizer (fitted on global action text set) -- at {os.path.join(output_dir, 'tfidf_vectorizer_char3_5.pkl')}")
    # save_npz(os.path.join(output_dir, "A_valid.npz"), A_valid)
    save_npz(os.path.join(output_dir, "A_valid_l2norm.npz"), A_valid_l2)
    # print(f"Saved transformed global action text set A_valid -- at {os.path.join(output_dir, 'A_valid.npz')}")
    print(f"Saved l2 row-normalized matrix A_valid_l2norm -- at {os.path.join(output_dir, 'A_valid_l2norm.npz')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ACTION_NAME → [ACT_xxx] token map.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to your yaml config file (config_WPE.yaml)")

    args = parser.parse_args()
    tfidf_vectorize_action_texts(args.config_file)



# === Later at LLM inference: load and use ===
# from scipy.sparse import load_npz
# vectorizer = joblib.load("tfidf_vectorizer_char3_5.pkl")
# A_valid_l2 = load_npz("A_valid_l2norm.npz")
# valid_action_texts = joblib.load("valid_action_texts.pkl")
#
# v = vectorizer.transform([predicted_chunk_text])
# from sklearn.preprocessing import normalize
# v_l2 = normalize(v, norm="l2", axis=1)
#
# sims = (A_valid_l2 @ v_l2.T).toarray().ravel()
# topk_idx = np.argsort(sims)[::-1][:k]
# topk_actions = [valid_action_texts[i] for i in topk_idx]