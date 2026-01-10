import os
import pandas as pd
from datetime import timedelta
import yaml
import torch
import warnings
import random
warnings.filterwarnings("ignore")
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run EHR Audit Log LLM pipeline")
    parser.add_argument('--config_file', type=str, default="config_WPE.yaml",
                        help="Path to config YAML file (default: config_WPE.yaml)")
    args = parser.parse_args()
    # Load configuration from YAML file
    # config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "config_WPE.yaml"))
    config_path = os.path.abspath(args.config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Define paths
    path_prefix = next((prefix for prefix in config["path_prefix"] if os.path.exists(os.path.normpath(prefix))), "")
    print(path_prefix)
    wpe_list_path = os.path.join(path_prefix, config.get("wpe_list"))

    # Load the WPE list file
    if config.get("n_mini") is not None:
        wpe_table = pd.read_csv(wpe_list_path, parse_dates=["ORDER_DTTM_raw"]).head(config.get("n_mini"))
    else:  # Use full dataset
        wpe_table = pd.read_csv(wpe_list_path, parse_dates=["ORDER_DTTM_raw"])

    # Extract unique WPE IDs
    wpe_ids = wpe_table["idx"].unique().tolist()

    # Shuffle with fixed seed
    seed = config.get("random_seed", 123)  # use config seed if exists
    random.seed(seed)
    random.shuffle(wpe_ids)

    # Split ratios (use your existing ratios)
    train_frac = config.get("train_split", 0.7)
    val_frac = config.get("val_split", 0.1)
    n_total = len(wpe_ids)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    train_ids = wpe_ids[:n_train]
    val_ids = wpe_ids[n_train:n_train + n_val]
    test_ids = wpe_ids[n_train + n_val:]

    # Save splits using torch.save (safer than pickle)
    split_dict = {
        "train": [str(x) for x in train_ids],
        "val": [str(x) for x in val_ids],
        "test": [str(x) for x in test_ids]
    }

    output_dir = os.path.dirname(wpe_list_path)
    output_file = os.path.join(output_dir, "fixed_wpe_splits.pt")
    torch.save(split_dict, output_file)

    print(f"[INFO] Saved deterministic split file at {output_file}")