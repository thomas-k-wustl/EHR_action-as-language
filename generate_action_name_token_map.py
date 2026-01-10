import os
import pandas as pd
import yaml
from collections import Counter
import json
import argparse


def load_config(yaml_path):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config


def find_valid_prefix(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No valid path_prefix found in config.")


def generate_action_token_map(yaml_config_path, top_k):
    config = load_config(yaml_config_path)
    path_prefix = find_valid_prefix(config["path_prefix"])
    data_path = os.path.join(path_prefix, config["audit_log_cache"])
    wpe_list_path = os.path.join(path_prefix, config["wpe_list"])

    user_id_df = pd.read_csv(wpe_list_path)
    if "exclusion_list" in config and config["exclusion_list"]:
        user_id_df = user_id_df[~user_id_df["idx"].isin(config["exclusion_list"])]

    wpe_idx_list = user_id_df["idx"].tolist()
    action_counter = Counter() # counter dict

    for wpe_idx in wpe_idx_list:
        for case_or_control in ["case", "control"]:
            file_name = str(wpe_idx) + config["audit_log_cache_file"][case_or_control]
            file_path = os.path.join(data_path, str(wpe_idx), file_name)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                try:
                    df = pd.read_parquet(file_path)
                    if "METRIC_NAME" in df.columns:
                        action_counter.update(df["METRIC_NAME"].dropna().tolist())
                except Exception as e:
                    print(f"Skipping {file_path} due to error: {e}")

    most_common_actions = [x[0] for x in action_counter.most_common(top_k)]
    action_token_map = {action: f"[ACT_{i}]" for i, action in enumerate(most_common_actions)}
    # action_token_map["_RARE_"] = "[ACT_RARE]"

    output_dir = os.path.dirname(wpe_list_path)
    output_file = os.path.join(output_dir, "action_token_map.json")
    with open(output_file, "w") as f:
        json.dump(action_token_map, f, indent=2)

    print(f"Saved action_token_map with {len(action_token_map) } unique actions at {output_file}. Rest of the actions will be tokenized as [ACT_RARE] during data tokenization.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ACTION_NAME → [ACT_xxx] token map.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to your yaml config file (config_WPE.yaml)")
    parser.add_argument("--top_k", type=int, default=500, help="Number of most common actions to map.")

    args = parser.parse_args()
    generate_action_token_map(args.config_file, args.top_k)