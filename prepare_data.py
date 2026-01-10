import pickle
from joblib import Parallel, delayed
import time
import os
import pandas as pd
from datetime import timedelta
import yaml
import torch
import warnings
import random
warnings.filterwarnings("ignore")

def generate_fixed_split():
    """
    Generates a deterministic train/validation/test split of unique WPE IDs
    from the WPE list CSV and saves the split to a file.

    This function loads the WPE table, extracts unique WPE IDs, shuffles them
    with a fixed random seed (from config), splits them into train/val/test
    sets according to configured ratios, and saves the split dictionary to
    'fixed_wpe_splits.pt' in the same directory as the WPE list CSV.

    The resulting split file is used to ensure consistent dataset splits
    across all experiments and prevents data leakage or randomness between runs.

    Raises:
        FileNotFoundError: if the WPE list file is not found.
    """
    # Load configuration from YAML file
    config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "config_WPE.yaml"))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Define paths
    path_prefix = next((prefix for prefix in config["path_prefix"] if os.path.exists(os.path.normpath(prefix))), "")
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
    split_dict = {"train": train_ids, "val": val_ids, "test": test_ids}

    output_dir = os.path.dirname(wpe_list_path)
    output_file = os.path.join(output_dir, "fixed_wpe_splits.pt")
    torch.save(split_dict, output_file)

    print(f"[INFO] Saved deterministic split file at {output_file}")

def process_row(row, audit_log_path, data_save_path, config, wpe_table, sample_n):
    # Initialize local counters and lists
    local_case_counter = 1
    local_case_counter_w_logs = 0
    local_control_idx = 0
    local_l_parquet_found = []
    local_l_parquet_notFound = []

    idx = row["idx"]
    order_time = row["ORDER_DTTM_raw"]
    folder_path = os.path.join(audit_log_path, str(idx))
    parquet_file = f"{idx}{config.get('audit_log_file')}"
    full_parquet_path = os.path.join(folder_path, parquet_file)

    save_path_idx = os.path.join(data_save_path, str(idx))
    if not os.path.exists(save_path_idx):
        os.makedirs(save_path_idx)

    # Check if the audit log file exists
    if not os.path.exists(full_parquet_path):
        local_l_parquet_notFound.append(idx)
    else:
        local_l_parquet_found.append(idx)
        # load audit logs
        parquet_data = pd.read_parquet(full_parquet_path)
        parquet_data['S_DIFF_FROM_RAR_TIME'] = round(parquet_data['S_DIFF_FROM_RAR_TIME'], 0)

        # CASE: extract logs in the 1h period before WPE order time
        case_data = parquet_data[(parquet_data["ACCESS_TIME"] >= (order_time - timedelta(minutes=config.get('min_prior')))) &
                                 (parquet_data["ACCESS_TIME"] < order_time)]
        save_path = save_path_idx
        # save_path = os.path.join(save_path_idx, 'case_controls')
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        if len(case_data) != 0:
            local_case_counter_w_logs += 1
            output_file = os.path.join(save_path, f"{idx}_case_{config.get('min_prior')}m.parquet")
            case_data.to_parquet(output_file, index=False)
            # output_file = os.path.join(save_path, f"{idx}_case_1h.csv")
            # case_data.to_csv(output_file, index=False)

        # MATCHED CONTROLS: find any previous activity related to placing an order. Currently using Order List Changed metric.
        control_anchors = parquet_data[(parquet_data["METRIC_ID"] == 17108) &
        # Give enough buffer time of 1 hour to avoid using an audit log event corresponding to CASE
                                       (parquet_data["ACCESS_TIME"] < (order_time - timedelta(minutes=60)))].sort_values("ACCESS_TIME").copy()
        # Ensure these orders don't correspond to another RAR event from the same clinician user.
        # Check if the ordering USER has multiple WPEs recorded in the study period
        if wpe_table["ORDERING_USER_ID"].value_counts().get(row["ORDERING_USER_ID"], 0) > 1:
            other_wpes = wpe_table[(wpe_table["ORDERING_USER_ID"] == row["ORDERING_USER_ID"]) &
                                    (wpe_table["idx"] != idx)]
            # Find fuzzy-matching rows (max 1 min difference)
            within_one_minute = control_anchors['ACCESS_TIME'].apply(
                lambda x: any(abs(x - other_wpes['ORDER_DTTM_raw']) <= timedelta(minutes=1))
            )
            control_anchors = control_anchors[~within_one_minute]
        if sample_n is not None:
            # Randomly sample n control anchors to construct a mini dataset
            if len(control_anchors) > sample_n:
                control_anchors = control_anchors.sample(n=sample_n, random_state=config.get("random_seed"))

        local_control_data = pd.DataFrame()
        for _, anchor_row in control_anchors.iterrows():

            anchor_time = anchor_row["ACCESS_TIME"]
            filtered_data = parquet_data[(parquet_data["ACCESS_TIME"] >= (anchor_time - timedelta(minutes=config.get('min_prior')))) &
                                         (parquet_data["ACCESS_TIME"] < anchor_time)].copy()
            if len(filtered_data) > 0:
                local_control_idx += 1
                try:
                    filtered_data.loc[:, 'control_idx'] = local_control_idx
                except ValueError as e:
                    print(f"Skipping iteration due to error: {e}")
                    print(f"Shape of the dataframe (filtered_data): {filtered_data.shape}")
                    continue
                control_anchors.loc[(control_anchors["ACCESS_INSTANT"] == anchor_row["ACCESS_INSTANT"]) &
                                    (control_anchors["USER_ID"] == anchor_row["USER_ID"]), 'control_idx'] = local_control_idx
                local_control_data = pd.concat([local_control_data, filtered_data], axis=0)

        # parquet_file_ca = os.path.join(save_path, f"{idx}_control_anchors.parquet")
        # if not control_anchors.empty:
        #     control_anchors.to_parquet(parquet_file_ca, index=False)
        # csv_file_ca = os.path.join(save_path, f"{idx}_control_anchors.csv")
        # control_anchors.to_csv(csv_file_ca, index=False)

        parquet_file_c = os.path.join(save_path, f"{idx}_control_{config.get('min_prior')}m.parquet")
        required_cols = {'ACCESS_INSTANT', 'USER_ID', 'ACCESS_TIME', 'METRIC_NAME', 'PAT_ID'}

        if not local_control_data.empty and required_cols.issubset(local_control_data.columns):
            local_control_data.to_parquet(parquet_file_c, index=False)
        # else:
        #     if len(local_control_data)==0:
        #         print(f"Skipping saving {parquet_file_c}: empty dataframe")
        #     else:
        #         print(
        #             f"Skipping saving {parquet_file_c}: missing columns {required_cols - set(local_control_data.columns)}")

        # csv_file_c = os.path.join(save_path, f"{idx}_control_1h.csv")
        # local_control_data.to_csv(csv_file_c, index=False)


    return {
        "case_counter": local_case_counter,
        "case_counter_w_logs": local_case_counter_w_logs,
        "control_idx": local_control_idx,
        "l_parquet_found": local_l_parquet_found,
        "l_parquet_notFound": local_l_parquet_notFound
    }

if __name__ == '__main__':
    # Generate deterministic data split table for downstream use.
    generate_fixed_split()

    # Load configuration from YAML file
    config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "config_WPE.yaml"))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Define paths
    path_prefix = next((prefix for prefix in config["path_prefix"] if os.path.exists(os.path.normpath(prefix))), "")
    wpe_list = os.path.join(path_prefix, config.get("wpe_list"))
    audit_log_path = os.path.join(path_prefix, config.get("audit_log_path"))
    data_save_path = os.path.join(path_prefix, config.get("audit_log_cache"))
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    # Load the main CSV file
    if config.get("n_mini") is not None:
        wpe_table = pd.read_csv(wpe_list, parse_dates=["ORDER_DTTM_raw"]).head(config.get("n_mini"))
    else: # Use full dataset
        wpe_table = pd.read_csv(wpe_list, parse_dates=["ORDER_DTTM_raw"])

    sample_n = config.get("sample_n_controls")

    start_time = time.time()
    results = Parallel(n_jobs=config.get("num_workers", -1))(
        delayed(process_row)(row, audit_log_path, data_save_path, config, wpe_table, sample_n)
        for _, row in wpe_table.iterrows()
    )

    # Aggregate results
    case_counter = sum(r["case_counter"] for r in results)
    case_counter_w_logs = sum(r["case_counter_w_logs"] for r in results)
    control_idx = sum(r["control_idx"] for r in results)
    l_parquet_found = [idx for r in results for idx in r["l_parquet_found"]]
    l_parquet_notFound = [idx for r in results for idx in r["l_parquet_notFound"]]

    with open(os.path.normpath(os.path.join(data_save_path, "../wpe_list/l_parquet_found.pkl")), "wb") as f:
        pickle.dump(l_parquet_found, f)
    with open(os.path.normpath(os.path.join(data_save_path, "../wpe_list/l_parquet_notFound.pkl")), "wb") as f:
        pickle.dump(l_parquet_notFound, f)

    print(f"For {case_counter} WPE cases in WPE list: \
        \n\t{len(l_parquet_notFound)} didn't have any logs saved (X_logs_IDENTIFIED.parquet).\
        \n\t{len(l_parquet_found)} had logs saved (X_logs_IDENTIFIED.parquet).\
        \nOut of {len(l_parquet_found)} WPEs with logs available:\
        \n\t{case_counter_w_logs} had 1+ logs in the previous {config.get('min_prior')}.\
        \n\t{control_idx} controls (with logs) were identified and extracted.")

    elapsed = time.time() - start_time
    print(f"Elapsed time: {int(elapsed)} seconds")