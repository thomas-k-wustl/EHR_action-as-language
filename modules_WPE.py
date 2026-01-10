import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import multiprocessing
from functools import partial
import gc
import sys
import json
import joblib
import re

import pdb
from tqdm import tqdm
import psutil, gc
import logging
# Configure logging to show messages with INFO level or higher
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import pickle
from pickle import UnpicklingError
# from lightning import pytorch as pl
from torch.utils.data import ConcatDataset, random_split, Subset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, SGD, RMSprop, Adagrad, Adadelta
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import traceback
import random
import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  #,AdamW
import bitsandbytes as bnb
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig, prepare_model_for_kbit_training
# prepare_model_for_kbit_training only supported for peft version 0.4.0 or above. currently disabled for argonaute debugging which has peft version 0.3.0


from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from data_WPE import EHRAuditLogDataSet, TokenizedDataSet
from static_features_WPE import build_static_feature_matrix

# Set the environment variable to disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# class EHRAuditLogDataModule(pl.LightningDataModule):
class EHRAuditLogDataModule():
    def __init__(self, yaml_config_path: str, access_config_path: str, n_positions=1024, reset_cache: bool = False,
                 debug: bool = False,
                 provider_type=None, unit_string: str = "session", seed: int = 123):
        """
        Initialize the data module with given configurations and parameters.

        :param yaml_config_path: Path to the YAML configuration file.
        :param batch_size: Batch size for dataloaders.
        :param n_positions: Context length for the model.
        :param reset_cache: Flag to reset cached sequences.
        :param debug: Flag to enable debug mode.
        :param provider_type: Specific provider type to filter data.
        """
        # super().__init__()

        with open(yaml_config_path) as f:
            self.config = yaml.safe_load(f)

        with open(access_config_path) as f:
            self.access_config = yaml.safe_load(f)

        self.path_prefix = next((prefix for prefix in self.config["path_prefix"] if os.path.exists(prefix)), "")

        if self.config.get("custom_tokenization", False):
            action_token_map_path = self.config.get("action_token_map_path", None)
            if action_token_map_path is None:
                raise ValueError("custom_tokenization=True but action_token_map_path not specified in config_WPE.yaml")
            with open(os.path.join(self.path_prefix, action_token_map_path), "r") as f:
                self.action_token_map = json.load(f)
            print(
                f"[INFO] Loaded action_token_map with {len(self.action_token_map)} entries from {action_token_map_path}")
        else:
            self.action_token_map = None

        self.yaml_config_path = yaml_config_path

        self.n_positions = n_positions
        self.reset_cache = reset_cache
        self.provider_type = provider_type
        self.debug = debug
        self.datasets = []
        self.unit_string = unit_string
        self.seed = seed
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.use_vanilla = self.config.get('use_vanilla', False)
        self.case_datasets = None
        self.control_datasets = None


        self.truncated_sequences_count = 0
        self.total_sequences_count = 0

        if self.config['num_workers'] == "dynamic":
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = self.config['num_workers']

    def monitor_memory(self, option='GPU'):
        if option == 'GPU':
            print("\tGPU Memory Usage:")
            print(f"\t\tAllocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            print(f"\t\tCached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
        elif option == 'CPU':
            import psutil

            # Get the memory details
            memory_info = psutil.virtual_memory()

            # Print memory details
            print("\tCPU memroy Usage:")
            print(f"\t\tTotal memory: {memory_info.total / (1024 ** 3):.2f} GB")
            print(f"\t\tAvailable memory: {memory_info.available / (1024 ** 3):.2f} GB")
            print(f"\t\tUsed memory: {memory_info.used / (1024 ** 3):.2f} GB")
            print(f"\t\tMemory percentage: {memory_info.percent}%")

    def prepare_data(self):
        """
        Prepare data by loading datasets from audit logs and tokenizing them.
        Loads datasets in parallel and caches them unless in debug mode.
        """
        # print('Preparing data...')
        logging.info("BEGIN: prepare_data()")
        # Find valid path prefix from config
        path_prefix = next((prefix for prefix in self.config["path_prefix"] if os.path.exists(prefix)), "")

        self.data_path = os.path.join(path_prefix, self.config["audit_log_cache"])
        # self.log_name = self.config["audit_log_file"]
        # shift_sep_min = self.config["sep_min"]["shift"]
        session_sep_min = self.config["sep_min"]["session"]

        def log_load(index, wpe_idx: str, log_every_n=500):
            """
            Load audit logs for a specific WPE.
            :type WPE: object
            """
            if index % log_every_n == 0:
                mem = psutil.virtual_memory()
                print(
                    f"[PROGRESS] {index}/{len(wpe_idx_list)} | WPE: {wpe_idx} | RAM Used: {mem.used / (1024 ** 3):.2f} GB")

            WPE_path = os.path.join(self.data_path, wpe_idx)

            case_log_name = wpe_idx + self.config.get("audit_log_cache_file")['case']
            case_log_path = os.path.join(WPE_path, case_log_name)

            control_log_name = wpe_idx + self.config.get("audit_log_cache_file")['control']
            control_log_path = os.path.join(WPE_path, control_log_name)

            # Skip if log file is missing or empty
            if not os.path.exists(case_log_path) or os.path.getsize(case_log_path) == 0:
                # print(f"Missing/Empty CASE log file for WPE_idx: {wpe_idx}")
                return
            if not os.path.exists(control_log_path) or os.path.getsize(control_log_path) == 0:
                # print(f"Missing/Empty CONTROL log file for WPE_idx: {wpe_idx}")
                return

            # Skip if WPE is in exclusion list
            if wpe_idx in (self.config.get("exclusion_list", []) or []):
                return

            case_dset = EHRAuditLogDataSet(
                yaml_config_path=self.yaml_config_path,
                root_dir=WPE_path,
                session_sep_min=session_sep_min,
                log_name=case_log_name,
                event_type_cols=["METRIC_NAME"],
                cache=WPE_path,
                reset_cache=self.reset_cache,
                unit_string=self.unit_string,
                clean_auditLogs=self.config["clean_auditLogs"],
                num_fields = self.config['num_fields'],
                wpe_id = wpe_idx,
                action_token_map=self.action_token_map
            )

            control_dset = EHRAuditLogDataSet(
                yaml_config_path=self.yaml_config_path,
                root_dir=WPE_path,
                session_sep_min=session_sep_min,
                log_name=control_log_name,
                event_type_cols=["METRIC_NAME"],
                cache=WPE_path,
                reset_cache=self.reset_cache,
                unit_string=self.unit_string,
                clean_auditLogs=self.config["clean_auditLogs"],
                num_fields=self.config['num_fields'],
                wpe_id = wpe_idx,
                action_token_map=self.action_token_map
            )

            case_dset.load()
            control_dset.load()

            return case_dset, control_dset

        if self.config.get("n_mini") is not None:
            # take n cases that were from a temporally distinct period than those used in the SFTpipeline -- only used for additional inference testing afterwards
            if self.config.get("temporal_distinct_testset") is not None:
                og_user_id_df = pd.read_csv(os.path.join(path_prefix, self.config["wpe_list"]))
                og_user_id_df['ORDER_DTTM_raw'] = pd.to_datetime(og_user_id_df['ORDER_DTTM_raw'])
                if self.config.get("temporal_distinct_testset") == 1:
                    user_id_df = og_user_id_df[(og_user_id_df['ORDER_DTTM_raw'] >= pd.to_datetime("2021-06-01")) & \
                                                     (og_user_id_df['ORDER_DTTM_raw'] < pd.to_datetime("2022-06-01"))]
                elif self.config.get("temporal_distinct_testset") == 2:
                    user_id_df = og_user_id_df[(og_user_id_df['ORDER_DTTM_raw'] >= pd.to_datetime("2023-06-01")) & \
                                               (og_user_id_df['ORDER_DTTM_raw'] < pd.to_datetime("2024-06-01"))]

            # take n cases to be used in the SFTpipeline
            else:
                user_id_df = pd.read_csv(os.path.join(path_prefix, self.config["wpe_list"])).head(self.config.get("n_mini"))

        else:
            user_id_df = pd.read_csv(os.path.join(path_prefix, self.config["wpe_list"]))

        # If provider type is not explicitly set with the config.yaml file, run for all users regardless of type.
        if self.provider_type != 'All' and self.provider_type is not None:
            user_id_df = user_id_df.loc[user_id_df["PROV_TYPE"] == self.provider_type]

        if self.debug:
            user_id_df = user_id_df.head(2)

        wpe_idx_list = user_id_df["idx"].tolist()

        # print("user_id_list:", user_id_list)
        print(f"\t# of WPEs: {len(wpe_idx_list)}")
        # print(f"\tProvider type: {self.provider_type if self.provider_type is not None else 'All'}")
        if wpe_idx_list is None or len(wpe_idx_list) == 0:
            raise ValueError("wpe_idx_list is None or Empty. Please ensure it is properly imported in.")

        # load the audit logs in parallel
        print("Loading & preprocessing audit logs before tokenization...")
        if self.reset_cache:
            print("\tLoading & caching audit logs from raw tables...")
        else:
            print("\tLoading from cache...")

        # self.monitor_memory(option='GPU')  # Monitor memory usage
        # self.monitor_memory(option='CPU')

        # Determine number of threads for parallel processing
        # threads = 1 if self.debug else -1 if self.reset_cache else 1
        # threads = -1
        # threads = 4

        # results = joblib.Parallel(n_jobs=self.config.get("num_workers", -1), verbose=1)(
        #     joblib.delayed(log_load)(str(wpe_idx)) for wpe_idx in wpe_idx_list
        # )
        results = joblib.Parallel(n_jobs=self.config.get("num_workers", -1), verbose=1)(
            joblib.delayed(log_load)(i, str(wpe_idx)) for i, wpe_idx in enumerate(wpe_idx_list)
        )
        filtered_results = [r for r in results if r is not None]
        print(f"[INFO] Loaded {len(filtered_results)} valid WPE case-control pairs out of {len(wpe_idx_list)} total.")
        case_datasets, control_datasets = zip(*filtered_results)

        # Filter out None results and add to self.datasets
        self.case_datasets = [dset for dset in case_datasets if dset is not None]
        self.control_datasets = [dset for dset in control_datasets if dset is not None]

        # Serialize full dataset objects for reuse
        os.makedirs(os.path.join(
            self.path_prefix,
            self.config["model_cache_path"] + self.config["model"].split("/")[1],
            self.config['HF_model_name']
        ), exist_ok=True)

        mem_before = psutil.virtual_memory().used / (1024 ** 3)
        gc.collect()
        mem_after = psutil.virtual_memory().used / (1024 ** 3)
        print(f"Memory before GC: {mem_before:.2f} GB → after GC: {mem_after:.2f} GB")
        cache_dir = os.path.join(self.path_prefix, self.config["model_cache_path"] + self.config["model"].split("/")[1],
                                 self.config['HF_model_name'])
        torch.save(self.case_datasets,
                   os.path.join(cache_dir, "cached_case_datasets.pt"))
        print("[INFO] Saved serialized case dataset to disk for reuse.")
        mem_before = psutil.virtual_memory().used / (1024 ** 3)
        gc.collect()
        mem_after = psutil.virtual_memory().used / (1024 ** 3)
        print(f"Memory before GC: {mem_before:.2f} GB → after GC: {mem_after:.2f} GB")
        # torch.save(self.control_datasets,
        #            os.path.join(self.path_prefix, self.config["model_cache_path"] + self.config["model"].split("/")[1],
        #                         self.config['HF_model_name'], "cached_control_datasets.pt"))
        chunk_size = 500  # or smaller based on your RAM
        for i in range(0, len(self.control_datasets), chunk_size):
            chunk = self.control_datasets[i:i + chunk_size]
            torch.save(chunk, os.path.join(cache_dir, f"control_chunk_{i // chunk_size}.pt"))
            if os.path.getsize(os.path.join(cache_dir, f"control_chunk_{i // chunk_size}.pt")) == 0:
                raise RuntimeError(f"Empty chunk file written: {os.path.join(cache_dir, f'control_chunk_{i // chunk_size}.pt')}")
            del chunk
            gc.collect()
        print("[INFO] Saved serialized control dataset to disk for reuse.")

        # print("Loading & preprocessing complete.")
        logging.info("FINISHED: prepare_data()")

        # self.monitor_memory(option='GPU')  # Monitor memory usage
        # self.monitor_memory(option='CPU')
        # print(f"Number of datasets loaded: {len(self.datasets)}")

    def get_input_strings(self):
        input_strings = []
        for dataset in self.datasets:
            if self.unit_string == "session":
                input_strings.extend(dataset.sessionStrings)
            else:
                input_strings.extend(dataset.rowStrings)
            # print(f"Aggregating {self.unit_string} strings from dataset. Current total: {len(input_strings)}")
        return input_strings

    def get_seq_len_distribution(self):
        logging.info("BEGIN: get_seq_len_distribution()...")

        model_name = self.config["model"]
        if self.use_vanilla:
            token_cache_path = os.path.join(self.config["model_cache_path"] + model_name.split("/")[1],
                                            model_name.split("/")[1])
        else:
            token_cache_path = os.path.join(self.config["model_cache_path"] + model_name.split("/")[1],
                                            self.config['HF_model_name'])

        # Try to reload cached dataset objects if available
        case_path = os.path.join(self.path_prefix, token_cache_path, "cached_case_datasets.pt")
        # control_path = os.path.join(self.path_prefix, token_cache_path, "cached_control_datasets.pt")
        if os.path.exists(case_path):
            self.case_datasets = torch.load(case_path)
            # self.control_datasets = torch.load(control_path)
            # Reassemble self.control_datasets from chunked files
            control_chunks = []
            chunk_index = 0
            while True:
                chunk_path = os.path.join(self.path_prefix, token_cache_path, f"control_chunk_{chunk_index}.pt")
                if not os.path.exists(chunk_path):
                    break
                control_chunks.extend(torch.load(chunk_path))
                chunk_index += 1
            self.control_datasets = control_chunks
            print("[INFO] Loaded cached case/control datasets from disk.")

            if self.config.get("temporal_distinct_testset") is not None:
                print(f"[INFO] Calculating distribution for OOS-{self.config.get('temporal_distinct_testset')}")
                test_input_strings = []

                ## Process case datasets

                ### CASE Dataset
                wpe_datasets = list(self.case_datasets)  # Each dataset corresponds to a unique WPE case
                test_wpe = wpe_datasets.copy()

                # Extract the action sequence preceding error from each WPE case.
                test_case_input_strings = [case_dataset.sessionStrings[0] for case_dataset in test_wpe]
                print(
                    f"[INFO] Number of case sequences: {len(wpe_datasets)}"
                    f"\n\tTest: {len(test_case_input_strings)}")
                # Append the case session strings into the session string bins.
                test_input_strings.extend(test_case_input_strings)

                ### CONTROL Dataset
                control_datasets = list(self.control_datasets)
                test_control = control_datasets.copy()

                # Extract the action sequence preceding error from each WPE case.
                # All control sessionStrings for the test cases are kept, which maintains the 1:N ratio.
                test_control_input_strings = [s for control_dataset in test_control for s in
                                              control_dataset.sessionStrings]

                print(
                    f"[INFO] Number of control samples: {len(test_control_input_strings)}"
                    f"\n\tTest: {len(test_control_input_strings)}"
                )

                # Append the control session strings into the session string bins.
                test_input_strings.extend(test_control_input_strings)

                ## Calculate distribution
                test_seq_len_distribution = [len(re.findall(r"\[ACT_\d+\]", s)) for s in test_input_strings]

                test_percentiles = np.percentile(test_seq_len_distribution, [25, 50, 75])

                print(
                    f"Test:\n25th: {test_percentiles[0]}, 50th (median): {test_percentiles[1]}, 75th: {test_percentiles[2]}")
            else:
                print(f"[INFO] Calculating distribution for In-Sample Set")
                # Step 2: Split dataset.sessionStrings into train, validation, and test sets

                train_input_strings = []
                val_input_strings = []
                test_input_strings = []

                ## Process case datasets
                # Since each WPE case has only a single corresponding error sequence,
                # the only way to achieve a 70/10/20 train/val/test split is to split the WPE cases themselves across these sets.
                # That means each split will contain a different set of WPEs, ensuring that error sequences do not overlap across splits.

                split_file = os.path.join(
                    os.path.dirname(os.path.join(self.path_prefix, self.config["wpe_list"])),
                    "fixed_wpe_splits.pt"
                )
                if os.path.exists(split_file):
                    print("[INFO] Using deterministic WPE split from fixed_wpe_splits.pt")
                    split_dict = torch.load(split_file)

                    ### CASE Dataset
                    wpe_datasets = list(self.case_datasets)  # Each dataset corresponds to a unique WPE case
                    assert any(dset.wpe_id in split_dict["train"] for dset in
                               wpe_datasets), "No matching case datasets found!"
                    train_wpe = [dset for dset in wpe_datasets if dset.wpe_id in split_dict["train"]]
                    val_wpe = [dset for dset in wpe_datasets if dset.wpe_id in split_dict["val"]]
                    test_wpe = [dset for dset in wpe_datasets if dset.wpe_id in split_dict["test"]]

                    # Extract the action sequence preceding error from each WPE case.
                    train_case_input_strings = [case_dataset.sessionStrings[0] for case_dataset in train_wpe]
                    val_case_input_strings = [case_dataset.sessionStrings[0] for case_dataset in val_wpe]
                    test_case_input_strings = [case_dataset.sessionStrings[0] for case_dataset in test_wpe]
                    print(
                        f"[INFO] Number of case sequences: {len(wpe_datasets)}\n\tTrain: {len(train_case_input_strings)}"
                        f"\n\tValidation: {len(val_case_input_strings)}"
                        f"\n\tTest: {len(test_case_input_strings)}")
                    # Append the case session strings into the session string bins.
                    train_input_strings.extend(train_case_input_strings)
                    val_input_strings.extend(val_case_input_strings)
                    test_input_strings.extend(test_case_input_strings)

                    ### CONTROL Dataset
                    control_datasets = list(self.control_datasets)
                    train_control = [dset for dset in control_datasets if dset.wpe_id in split_dict["train"]]
                    val_control = [dset for dset in control_datasets if dset.wpe_id in split_dict["val"]]
                    test_control = [dset for dset in control_datasets if dset.wpe_id in split_dict["test"]]

                    train_control_input_strings = [s for control_dataset in train_control for s in
                                                   control_dataset.sessionStrings]
                    val_control_input_strings = [s for control_dataset in val_control for s in
                                                 control_dataset.sessionStrings]
                    test_control_input_strings = [s for control_dataset in test_control for s in
                                                  control_dataset.sessionStrings]

                    print(
                        f"[INFO] Number of control samples: {len(train_control_input_strings) + len(val_control_input_strings) + len(test_control_input_strings)}"
                        f"\n\tTrain (1:{self.config.get('sample_n_controls', 'ALL')}): {len(train_control_input_strings)}"
                        f"\n\tValidation (1:{self.config.get('sample_n_controls', 'ALL')}): {len(val_control_input_strings)}"
                        f"\n\tTest: {len(test_control_input_strings)}")

                    # Append the control session strings into the session string bins.
                    train_input_strings.extend(train_control_input_strings)
                    val_input_strings.extend(val_control_input_strings)
                    test_input_strings.extend(test_control_input_strings)

                    ## Calculate distribution
                    train_seq_len_distribution = [len(re.findall(r"\[ACT_\d+\]", s)) for s in train_input_strings]
                    val_seq_len_distribution = [len(re.findall(r"\[ACT_\d+\]", s)) for s in val_input_strings]
                    test_seq_len_distribution = [len(re.findall(r"\[ACT_\d+\]", s)) for s in test_input_strings]

                    train_percentiles = np.percentile(train_seq_len_distribution, [25, 50, 75])
                    val_percentiles = np.percentile(val_seq_len_distribution, [25, 50, 75])
                    test_percentiles = np.percentile(test_seq_len_distribution, [25, 50, 75])

                    print(
                        f"Train:\n25th: {train_percentiles[0]}, 50th (median): {train_percentiles[1]}, 75th: {train_percentiles[2]}")
                    print(
                        f"Val:\n25th: {val_percentiles[0]}, 50th (median): {val_percentiles[1]}, 75th: {val_percentiles[2]}")
                    print(
                        f"Test:\n25th: {test_percentiles[0]}, 50th (median): {test_percentiles[1]}, 75th: {test_percentiles[2]}")


    def setup(self, demo: bool = False, convert_field_vals_to: str = None, test_unseen: bool = False):
        logging.info("BEGIN: setup()...")
        if demo:
            from datasets import load_dataset
            if test_unseen:
                self.datasets = [load_dataset("ag_news", split="test")["text"]]
            else:
                self.datasets = [load_dataset("ag_news", split="train")["text"]]
            if self.debug:
                print(f"datasets: {self.datasets}")
        else:
            ## IF reset_cache=False and tokenized dataset already saved, directly load it
            model_name = self.config["model"]
            if self.use_vanilla:
                token_cache_path = os.path.join(self.config["model_cache_path"] + model_name.split("/")[1],
                                                model_name.split("/")[1])
            else:
                token_cache_path = os.path.join(self.config["model_cache_path"] + model_name.split("/")[1],
                                                self.config['HF_model_name'])

            access_token = self.access_config["HF_access_token"]
            reset_cache = self.config["reset_cache"]

            train_cache = os.path.join(self.path_prefix, token_cache_path, f"tokenized_dataset_train.pt")
            val_cache = os.path.join(self.path_prefix, token_cache_path, f"tokenized_dataset_val.pt")
            test_cache = os.path.join(self.path_prefix, token_cache_path, f"tokenized_dataset_test.pt")
            # print(train_cache, val_cache, test_cache)
            # print(os.path.exists(p) for p in [train_cache, val_cache, test_cache])
            if not reset_cache and all(os.path.exists(p) for p in [train_cache, val_cache, test_cache]):
                print("setup(): Loading tokenized dataset from cache...")
                self.tokenizer = EHRAuditLogTokenizer(
                    yaml_config_path=self.yaml_config_path,
                    model_name=model_name,
                    cache=token_cache_path,
                    reset_cache=False,
                    access_token=access_token,
                    debug=self.debug,
                    n_positions=self.n_positions
                )
                print("Training set:")
                self.tokenizer.load_tokens_from_cache(tag="_train")
                self.train_dataset = TokenizedDataSet(self.tokenizer.get_tokenized_dataset())

                print("Validation set:")
                self.tokenizer.load_tokens_from_cache(tag="_val")
                self.val_dataset = TokenizedDataSet(self.tokenizer.get_tokenized_dataset())

                print("Test set:")
                self.tokenizer.load_tokens_from_cache(tag="_test")
                tokenized_data = self.tokenizer.get_tokenized_dataset()
                # test_labels = np.load(os.path.join(self.path_prefix, token_cache_path, "test_labels.npy"))
                # test_timedelta_seqs = np.load(os.path.join(self.path_prefix, token_cache_path, "test_timedelta_sequences_aligned.npy"))
                # ## Add error labels and timedeltas as items into the tokenized data object
                # for i in range(len(tokenized_data)):
                #     tokenized_data[i]['error_label'] = test_labels[i]
                #     tokenized_data[i]['time_delta'] = test_timedelta_seqs[i]

                self.test_dataset = TokenizedDataSet(tokenized_data)

                logging.info("Tokenized datasets loaded from cache. Skipping full setup.")

                if os.path.exists(os.path.join(self.path_prefix, token_cache_path, "test_set_static_features.parquet")):
                    static_features = pd.read_parquet(
                        os.path.join(self.path_prefix, token_cache_path, "test_set_static_features.parquet"))
                    print(f"[INFO] Confirmed pre-computed static feature matrix exists in cache."
                          f"\n\tshape: {static_features.shape}"
                          f"\n\tcolumns: {static_features.columns}")
                    del static_features
                else:
                    print(f"[INFO] Static feature matrix does not exist in cache.")
            else:
                print(f"[INFO] reset_cache={reset_cache} | Cached files present? {all(os.path.exists(p) for p in [train_cache, val_cache, test_cache])}")
                ## IF reset_cache=True, run to the full preparation cycle.

                if not self.case_datasets or not self.control_datasets:
                    # self.prepare_data()
                    # Try to reload cached dataset objects if available
                    case_path = os.path.join(self.path_prefix, token_cache_path, "cached_case_datasets.pt")
                    # control_path = os.path.join(self.path_prefix, token_cache_path, "cached_control_datasets.pt")
                    if os.path.exists(case_path):
                        self.case_datasets = torch.load(case_path)
                        # self.control_datasets = torch.load(control_path)
                        # Reassemble self.control_datasets from chunked files
                        control_chunks = []
                        chunk_index = 0
                        while True:
                            chunk_path = os.path.join(self.path_prefix, token_cache_path, f"control_chunk_{chunk_index}.pt")
                            if not os.path.exists(chunk_path):
                                break
                            control_chunks.extend(torch.load(chunk_path))
                            chunk_index += 1
                        self.control_datasets = control_chunks
                        print("[INFO] Loaded cached case/control datasets from disk.")

                    else:
                        self.prepare_data()
                        if self.config["only_data_prep"]:
                            logging.info("INFO: Exiting pipeline after data preparation complete.")
                            sys.exit()
                    # check a few examples
                    # print(f"Examples to check prepare_data() worked properly: \nCASE:\n\n{self.case_datasets[0].sessionStrings[0]}\n\nCONTROL:\n\n{self.control_datasets[0].sessionStrings[0]}")

                    # Token length checking block
                    # The n_positions parameter in the config should be larger than the token length,
                    # to ensure that the entire sequence is preserved and no token near the end of the sequence is truncated
                    # If key information is late (e.g., important audit events near the end of the 1-hour session), they would be lost during training.
                    # self._check_token_lengths()
                    # return #DEBUGGING - to be removed

                if convert_field_vals_to is not None:
                    self.convert_field_vals(convert_field_vals_to)



                # Step 2: Split dataset.sessionStrings into train, validation, and test sets

                train_input_strings = []
                val_input_strings = []
                test_input_strings = []
                # train_wpe_ids = []
                # val_wpe_ids = []
                test_wpe_ids = []
                test_timedelta_seqs = []

                ## Process case datasets
                # Since each WPE case has only a single corresponding error sequence,
                # the only way to achieve a 70/10/20 train/val/test split is to split the WPE cases themselves across these sets.
                # That means each split will contain a different set of WPEs, ensuring that error sequences do not overlap across splits.

                split_file = os.path.join(
                    os.path.dirname(os.path.join(self.path_prefix, self.config["wpe_list"])),
                    "fixed_wpe_splits.pt"
                )
                if os.path.exists(split_file):
                    print("[INFO] Using deterministic WPE split from fixed_wpe_splits.pt")
                    split_dict = torch.load(split_file)

                    ### CASE Dataset
                    wpe_datasets = list(self.case_datasets)  # Each dataset corresponds to a unique WPE case
                    assert any(dset.wpe_id in split_dict["train"] for dset in
                               wpe_datasets), "No matching case datasets found!"
                    train_wpe = [dset for dset in wpe_datasets if dset.wpe_id in split_dict["train"]]
                    val_wpe = [dset for dset in wpe_datasets if dset.wpe_id in split_dict["val"]]
                    test_wpe = [dset for dset in wpe_datasets if dset.wpe_id in split_dict["test"]]
                    # train_case_wpe_ids = [case_dataset.wpe_id for case_dataset in train_wpe]
                    # val_case_wpe_ids = [case_dataset.wpe_id for case_dataset in val_wpe]
                    test_case_wpe_ids = [case_dataset.wpe_id for case_dataset in test_wpe]
                    test_case_timedelta_sequences = [case_dataset.timedelta_sequences[0] for case_dataset in test_wpe]

                    # Extract the action sequence preceding error from each WPE case.
                    train_case_input_strings = [case_dataset.sessionStrings[0] for case_dataset in train_wpe]
                    val_case_input_strings = [case_dataset.sessionStrings[0] for case_dataset in val_wpe]
                    test_case_input_strings = [case_dataset.sessionStrings[0] for case_dataset in test_wpe]
                    print(
                        f"[INFO] Number of case sequences: {len(wpe_datasets)}\n\tTrain: {len(train_case_input_strings)}"
                        f"\n\tValidation: {len(val_case_input_strings)}"
                        f"\n\tTest: {len(test_case_input_strings)}")
                    # Append the case session strings into the session string bins.
                    train_input_strings.extend(train_case_input_strings)
                    val_input_strings.extend(val_case_input_strings)
                    test_input_strings.extend(test_case_input_strings)
                    # train_wpe_ids.extend(train_case_wpe_ids)
                    # val_wpe_ids.extend(val_case_wpe_ids)
                    test_wpe_ids.extend(test_case_wpe_ids)
                    test_timedelta_seqs.extend(test_case_timedelta_sequences)

                    ### CONTROL Dataset
                    control_datasets = list(self.control_datasets)
                    train_control = [dset for dset in control_datasets if dset.wpe_id in split_dict["train"]]
                    val_control = [dset for dset in control_datasets if dset.wpe_id in split_dict["val"]]
                    test_control = [dset for dset in control_datasets if dset.wpe_id in split_dict["test"]]
                    # train_control_wpe_ids = [dataset.wpe_id for dataset in train_control if
                    #                          len(dataset.sessionStrings) > 0]
                    # val_control_wpe_ids = [dataset.wpe_id for dataset in val_control if
                    #                          len(dataset.sessionStrings) > 0]
                    test_control_wpe_ids = [dataset.wpe_id for dataset in test_control if
                                             len(dataset.sessionStrings) > 0]
                    test_control_timedelta_sequences = [TD_seq for control_dataset in test_control for TD_seq in
                                                  control_dataset.timedelta_sequences]
                    # Extract the action sequence preceding error from each WPE case.
                    ## Option 1: randomly sample 1 control
                    # train_input_strings = [random.choice(dataset.sessionStrings) for control_datasets in train_control]
                    # val_input_strings = [random.choice(dataset.sessionStrings) for control_datasets in val_control]
                    # test_input_strings = [s for control_dataset in test_control for s in control_dataset.sessionStrings]
                    ## Option 2: take the most recent control sample (from WPE case event time)
                    # train_control_input_strings = [
                    #     dataset.sessionStrings[-1]
                    #     for dataset in train_control
                    #     if len(dataset.sessionStrings) > 0
                    # ]
                    # val_control_input_strings = [
                    #     dataset.sessionStrings[-1]
                    #     for dataset in val_control
                    #     if len(dataset.sessionStrings) > 0
                    # ]
                    # keep N samples from all controls for a desired 1:N matching
                    sample_n = self.config.get("sample_n_controls")
                    if sample_n is not None:
                        train_control_input_strings = [
                            dataset.sessionStrings[-sample_n:]
                            for dataset in train_control
                            if len(dataset.sessionStrings) > 0
                        ]
                        val_control_input_strings = [
                            dataset.sessionStrings[-sample_n:]
                            for dataset in val_control
                            if len(dataset.sessionStrings) > 0
                        ]
                    else:
                        train_control_input_strings = [s for control_dataset in train_control for s in
                                                      control_dataset.sessionStrings]
                        val_control_input_strings = [s for control_dataset in val_control for s in
                                                       control_dataset.sessionStrings]
                    # All control sessionStrings for the test cases are kept, which maintains the 1:N ratio.
                    # For test WPEs with fewer than N matched controls, all available controls are retained, which would make it 1:M matched (M<N).
                    test_control_input_strings = [s for control_dataset in test_control for s in
                                                  control_dataset.sessionStrings]
                    num_skipped = sum(len(d.sessionStrings) == 0 for d in train_control)
                    num_skipped += sum(len(d.sessionStrings) == 0 for d in val_control)
                    num_skipped += sum(len(d.sessionStrings) == 0 for d in test_control)
                    print(f"Skipped {num_skipped} empty control datasets.")

                    M = len(test_control_input_strings) / max(len(test_input_strings), 1)
                    print(
                        f"[INFO] Number of control samples: {len(train_control_input_strings) + len(val_control_input_strings) + len(test_control_input_strings)}"
                        f"\n\tTrain (1:{self.config.get('sample_n_controls', 'ALL')}): {len(train_control_input_strings)}"
                        f"\n\tValidation (1:{self.config.get('sample_n_controls', 'ALL')}): {len(val_control_input_strings)}"
                        f"\n\tTest: {len(test_control_input_strings)}"
                        f"\n\t\tTarget: 1:{self.config.get('sample_n_controls', 'ALL')} matching"
                        f"\n\t\tReality: ~1:{M:.1f}")

                    # Append the control session strings into the session string bins.
                    train_input_strings.extend(train_control_input_strings)
                    val_input_strings.extend(val_control_input_strings)
                    test_input_strings.extend(test_control_input_strings)
                    # train_wpe_ids.extend(train_control_wpe_ids)
                    # val_wpe_ids.extend(val_control_wpe_ids)
                    test_wpe_ids.extend(test_control_wpe_ids)
                    test_timedelta_seqs.extend(test_control_timedelta_sequences)
                else:
                    self.test_wpe_ids = []
                    raise FileNotFoundError("[ERROR] fixed_wpe_splits.pt not found. Please run generate_fixed_split.py first.")

                # Step 3: Initialize the EHRAuditLogTokenizer once
                model_name = self.config["model"]
                if demo:
                    token_cache_path = os.path.dirname(os.path.abspath(__file__))  # current file's pwd
                else:
                    if self.use_vanilla:
                        token_cache_path = os.path.join(self.config["model_cache_path"] + model_name.split("/")[1],
                                                        model_name.split("/")[1])
                    else:
                        token_cache_path = os.path.join(self.config["model_cache_path"] + model_name.split("/")[1], self.config['HF_model_name'])
                # print(f"\tToken cache path: {token_cache_path}")
                access_token = self.access_config["HF_access_token"]
                reset_cache = self.config["reset_cache"]

                self.tokenizer = EHRAuditLogTokenizer(yaml_config_path=self.yaml_config_path,
                                                     model_name=model_name,
                                                     cache=token_cache_path,
                                                     reset_cache=reset_cache,
                                                     access_token=access_token,
                                                     debug=self.debug,
                                                     n_positions=self.n_positions)

                # Truncate each session string to only the last n_positions tokens before tokenization
                train_input_strings = [self._truncate_to_last_n_tokens(s) for s in train_input_strings if isinstance(s, str) and s.strip()]
                val_input_strings = [self._truncate_to_last_n_tokens(s) for s in val_input_strings if isinstance(s, str) and s.strip()]
                test_input_strings = [self._truncate_to_last_n_tokens(s) for s in test_input_strings if isinstance(s, str) and s.strip()]

                os.makedirs(os.path.join(self.path_prefix, token_cache_path), exist_ok=True)
                np.save(os.path.join(self.path_prefix, token_cache_path, "testset_wpe_ids.npy"), np.array(test_wpe_ids))
                np.save(os.path.join(self.path_prefix, token_cache_path, "test_timedelta_sequences.npy"),
                        np.array(test_timedelta_seqs, dtype=object))

                # print(f"Example truncated session string:\n{train_input_strings[0][:300]}")
                # print(f"#tokens in truncated string: {len(tokenizer.tokenizer.tokenize(train_input_strings[0]))}")

                # Step 4: Run tokenizer on each list

                # DEBUG: Before tokenization
                process = psutil.Process(os.getpid())
                print(
                    f"[BEFORE TOKENIZATION] Job Memory - RSS: {process.memory_info().rss / (1024 ** 3):.2f} GB, VMS: {process.memory_info().vms / (1024 ** 3):.2f} GB")

                print("Training set:")
                self.tokenizer.load(train_input_strings, padding='max_length', tag='_train')
                self.train_dataset = TokenizedDataSet(self.tokenizer.get_tokenized_dataset())

                # DEBUG: After tokenization
                process = psutil.Process(os.getpid())
                print(
                    f"[BEFORE TOKENIZATION] Job Memory - RSS: {process.memory_info().rss / (1024 ** 3):.2f} GB, VMS: {process.memory_info().vms / (1024 ** 3):.2f} GB")

                print("Validation set:")
                self.tokenizer.load(val_input_strings, padding='max_length', tag='_val')
                self.val_dataset = TokenizedDataSet(self.tokenizer.get_tokenized_dataset())

                print("Test set:")
                self.tokenizer.load(test_input_strings, padding='max_length', tag='_test')
                tokenized_data = self.tokenizer.get_tokenized_dataset()

                self.tokenizer.tokenizer.save_pretrained(os.path.join(self.path_prefix, token_cache_path))
                print(f"Saved tokenizer at {os.path.join(self.path_prefix,token_cache_path)}")
                # print(f"[DATASET] Tokenizer length at dataset creation: {len(self.tokenizer.tokenizer)}")

                # Assign binary error labels
                test_labels = [1] * len(test_case_input_strings) + [0] * len(test_control_input_strings)
                # Propagate Timedeltas with the tokenized dataset objects -- for downstream use during inference
                ## Truncate and pad the timedelta sequences to exactly match the truncated&tokenized test sequences.
                reserved_tokens = self.config.get("model_configs", {}).get(self.config.get("model", ""), {}).get(
                    "prompt_reserved_tokens", 64)
                truncate_len = self.n_positions - reserved_tokens
                ### Pad with 0's for the padding positions -- easy to remove later via filtering
                test_timedelta_seqs = [list(s)[-truncate_len:] + [0.0] * max(0, truncate_len - len(s)) for s in test_timedelta_seqs]
                ### Replace the first action's timedelta (np.nan) in each sequence, with an arbitrary large timedelta value
                test_timedelta_seqs = [[999 if np.isnan(x) else x for x in s] for s in
                            test_timedelta_seqs]
                np.save(os.path.join(self.path_prefix, token_cache_path, "test_timedelta_sequences_aligned.npy"),
                        np.array(test_timedelta_seqs))
                ## Add error labels and timedeltas as items into the tokenized data object
                for i in range(len(tokenized_data)):
                    tokenized_data[i]['error_label'] = test_labels[i]
                    tokenized_data[i]['time_delta'] = test_timedelta_seqs[i]

                self.test_dataset = TokenizedDataSet(tokenized_data)
                print(f"[INFO] Token length truncation stats:")
                print(f"   Total sequences: {self.total_sequences_count}")
                print(f"   Sequences truncated: {self.truncated_sequences_count} "
                      f"({(self.truncated_sequences_count / max(1, self.total_sequences_count)) * 100:.2f}%)")

                np.save(os.path.join(self.path_prefix, token_cache_path, "test_labels.npy"), np.array(test_labels))
                print(f"Saved test_labels.npy with {len(test_labels)} entries to match test_dataset.")


                if self.config.get("cache_static_features"):
                    print(f"[INFO] Calculating & Caching static features...")
                    # Construct static feature set while you have access to the exact test datasets
                    # Create a dataframe to store static features
                    df_static_features , wpe_ids = build_static_feature_matrix(test_wpe, test_control)
                    # df_static_features['wpe_id'] = [dset.wpe_id for dset in test_wpe + test_control]
                    df_static_features['wpe_id'] = wpe_ids
                    assert len(df_static_features) == len(test_labels), \
                        f"Mismatch: static features rows ({len(df_static_features)}) vs test labels ({len(test_labels)})"
                    df_static_features['error_label'] = test_labels
                    df_static_features.to_parquet(os.path.join(self.path_prefix, token_cache_path, "test_set_static_features.parquet"))
                    print(f"FINISHED. Static Feature Matrix Columns:\n{df_static_features.columns}")
        # pdb.set_trace()

    def setup_testset(self, checkpoint_path=None, convert_field_vals_to: str = None):
        logging.info("BEGIN: setup_testset()...")

        ## IF reset_cache=False and tokenized dataset already saved, directly load it
        model_name = self.config["model"]
        if self.use_vanilla:
            token_cache_path = os.path.join(self.config["model_cache_path"] + model_name.split("/")[1],
                                            model_name.split("/")[1])
        else:
            token_cache_path = os.path.join(self.config["model_cache_path"] + model_name.split("/")[1],
                                            self.config['HF_model_name'])

        access_token = self.access_config["HF_access_token"]
        reset_cache = self.config["reset_cache"]

        test_cache = os.path.join(self.path_prefix, token_cache_path, f"tokenized_dataset_test.pt")

        if not reset_cache and os.path.exists(test_cache):
            print("setup_testset(): Loading tokenized dataset from cache...")
            self.tokenizer = EHRAuditLogTokenizer(
                yaml_config_path=self.yaml_config_path,
                model_name=model_name,
                cache=token_cache_path,
                reset_cache=False,
                access_token=access_token,
                debug=self.debug,
                n_positions=self.n_positions
            )

            print("Test set:")
            self.tokenizer.load_tokens_from_cache(tag=f"_test")
            tokenized_data = self.tokenizer.get_tokenized_dataset()
            # test_labels = np.load(os.path.join(self.path_prefix, token_cache_path, f"test_labels.npy"))
            # test_timedelta_seqs = np.load(
            #     os.path.join(self.path_prefix, token_cache_path, "test_timedelta_sequences_aligned.npy"))

            # ## Add error labels and timedeltas as items into the tokenized data object
            # for i in range(len(tokenized_data)):
            #     tokenized_data[i]['error_label'] = test_labels[i]
            #     tokenized_data[i]['time_delta'] = test_timedelta_seqs[i]

            self.test_dataset = TokenizedDataSet(tokenized_data)
            # self.test_dataset = TokenizedDataSet(tokenizer.get_tokenized_dataset())

            # for i in range(len(self.test_dataset)):
            #     self.test_dataset[i]["error_label"] = int(test_labels[i])

            logging.info("Tokenized datasets loaded from cache. Skipping full setup.")

            if os.path.exists(os.path.join(self.path_prefix, token_cache_path, f"test_set_static_features.parquet")):
                static_features = pd.read_parquet(
                    os.path.join(self.path_prefix, token_cache_path, f"test_set_static_features.parquet"))
                print(f"[INFO] Confirmed pre-computed static feature matrix exists in cache."
                      f"\n\tshape: {static_features.shape}"
                      f"\n\tcolumns: {static_features.columns}")
                del static_features
            else:
                print(f"[INFO] Static feature matrix does not exist in cache.")
        else:
            print(f"[INFO] reset_cache={reset_cache} | Cached files present? {os.path.exists(test_cache)}")
            ## IF reset_cache=True, run to the full preparation cycle.
            if not self.case_datasets or not self.control_datasets:
                # Try to reload cached dataset objects if available
                case_path = os.path.join(self.path_prefix, token_cache_path, "cached_case_datasets.pt")
                if os.path.exists(case_path):
                    self.case_datasets = torch.load(case_path)
                    # Reassemble self.control_datasets from chunked files
                    control_chunks = []
                    chunk_index = 0
                    while True:
                        chunk_path = os.path.join(self.path_prefix, token_cache_path, f"control_chunk_{chunk_index}.pt")
                        if not os.path.exists(chunk_path):
                            break
                        control_chunks.extend(torch.load(chunk_path))
                        chunk_index += 1
                    self.control_datasets = control_chunks
                    print("[INFO] Loaded cached case/control datasets from disk.")

                else:
                    self.prepare_data()
                    if self.config["only_data_prep"]:
                        logging.info("INFO: Exiting pipeline after data preparation complete.")
                        sys.exit()


            if convert_field_vals_to is not None:
                self.convert_field_vals(convert_field_vals_to)




            test_input_strings = []
            test_wpe_ids = []
            test_timedelta_seqs = []

            ## Process case datasets

            ### CASE Dataset
            wpe_datasets = list(self.case_datasets)  # Each dataset corresponds to a unique WPE case
            test_wpe = wpe_datasets.copy()
            # train_case_wpe_ids = [case_dataset.wpe_id for case_dataset in train_wpe]
            # val_case_wpe_ids = [case_dataset.wpe_id for case_dataset in val_wpe]
            test_case_wpe_ids = [case_dataset.wpe_id for case_dataset in test_wpe]
            test_case_timedelta_sequences = [case_dataset.timedelta_sequences[0] for case_dataset in test_wpe]

            # Extract the action sequence preceding error from each WPE case.
            test_case_input_strings = [case_dataset.sessionStrings[0] for case_dataset in test_wpe]
            print(
                f"[INFO] Number of case sequences: {len(wpe_datasets)}"
                f"\n\tTest: {len(test_case_input_strings)}")
            # Append the case session strings into the session string bins.
            test_input_strings.extend(test_case_input_strings)

            test_wpe_ids.extend(test_case_wpe_ids)
            test_timedelta_seqs.extend(test_case_timedelta_sequences)

            ### CONTROL Dataset
            control_datasets = list(self.control_datasets)
            test_control = control_datasets.copy()
            test_control_wpe_ids = [dataset.wpe_id for dataset in test_control if
                                     len(dataset.sessionStrings) > 0]
            test_control_timedelta_sequences = [TD_seq for control_dataset in test_control for TD_seq in
                                                control_dataset.timedelta_sequences]

            # Extract the action sequence preceding error from each WPE case.
            # All control sessionStrings for the test cases are kept, which maintains the 1:N ratio.
            test_control_input_strings = [s for control_dataset in test_control for s in
                                          control_dataset.sessionStrings]

            num_skipped = sum(len(d.sessionStrings) == 0 for d in test_control)
            print(f"Skipped {num_skipped} empty control datasets.")

            M = len(test_control_input_strings) / max(len(test_input_strings), 1)
            print(
                f"[INFO] Number of control samples: {len(test_control_input_strings)}"
                f"\n\tTest: {len(test_control_input_strings)}"
                f"\n\t\tTarget: 1:{self.config.get('sample_n_controls', 'ALL')} matching"
                f"\n\t\tReality: ~1:{M:.1f}")

            # Append the control session strings into the session string bins.
            test_input_strings.extend(test_control_input_strings)
            test_wpe_ids.extend(test_control_wpe_ids)
            test_timedelta_seqs.extend(test_control_timedelta_sequences)

            # Step 3: Initialize the EHRAuditLogTokenizer once
            model_name = self.config["model"]
            if self.use_vanilla:
                token_cache_path = os.path.join(self.config["model_cache_path"] + model_name.split("/")[1],
                                                model_name.split("/")[1])
            else:
                token_cache_path = os.path.join(self.config["model_cache_path"] + model_name.split("/")[1], self.config['HF_model_name'])
            # print(f"\tToken cache path: {token_cache_path}")
            access_token = self.access_config["HF_access_token"]
            reset_cache = self.config["reset_cache"]

            if checkpoint_path is not None:
                if os.path.isdir(checkpoint_path):
                    logging.info(f"Loading tokenizer from local checkpoint: {checkpoint_path}")
                    loaded_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=False)
                else:
                    repo_id = f"{self.access_config['HF_username']}/{self.config['HF_repo_name']}"
                    logging.info(f"Loading tokenizer from HF checkpoint: {repo_id}")
                    loaded_tokenizer = AutoTokenizer.from_pretrained(
                        repo_id,
                        use_auth_token=self.access_config['HF_access_token'],
                        trust_remote_code=False
                    )

                self.tokenizer = EHRAuditLogTokenizer(loaded_tokenizer=loaded_tokenizer,
                                                      yaml_config_path=self.yaml_config_path,
                                                      model_name=model_name,
                                                      cache=token_cache_path,
                                                      reset_cache=reset_cache,
                                                      access_token=access_token,
                                                      debug=self.debug,
                                                      n_positions=self.n_positions)
            else:
                self.tokenizer = EHRAuditLogTokenizer(yaml_config_path=self.yaml_config_path,
                                                      model_name=model_name,
                                                      cache=token_cache_path,
                                                      reset_cache=reset_cache,
                                                      access_token=access_token,
                                                      debug=self.debug,
                                                      n_positions=self.n_positions)

            # Truncate each session string to only the last n_positions tokens before tokenization
            test_input_strings = [self._truncate_to_last_n_tokens(s) for s in test_input_strings if isinstance(s, str) and s.strip()]

            os.makedirs(os.path.join(self.path_prefix, token_cache_path), exist_ok=True)
            np.save(os.path.join(self.path_prefix, token_cache_path, f"test_wpe_ids.npy"), np.array(test_wpe_ids))
            np.save(os.path.join(self.path_prefix, token_cache_path, f"test_timedelta_sequences.npy"),
                    np.array(test_timedelta_seqs, dtype=object))

            # print(f"Example truncated session string:\n{train_input_strings[0][:300]}")
            # print(f"#tokens in truncated string: {len(tokenizer.tokenizer.tokenize(train_input_strings[0]))}")

            # Step 4: Run tokenizer on each list

            # DEBUG: Before tokenization
            print("Test set:")
            self.tokenizer.load(test_input_strings, padding='max_length', tag=f"_test")
            tokenized_data = self.tokenizer.get_tokenized_dataset()

            # Assign binary error labels
            test_labels = [1] * len(test_case_input_strings) + [0] * len(test_control_input_strings)
            # Propagate Timedeltas with the tokenized dataset objects -- for downstream use during inference
            ## Truncate and pad the timedelta sequences to exactly match the truncated&tokenized test sequences.
            reserved_tokens = self.config.get("model_configs", {}).get(self.config.get("model", ""), {}).get(
                "prompt_reserved_tokens", 64)
            truncate_len = self.n_positions - reserved_tokens
            ### Pad with 0's for the padding positions -- easy to remove later via filtering
            test_timedelta_seqs = [list(s)[-truncate_len:] + [0.0] * max(0, truncate_len - len(s)) for s in
                                   test_timedelta_seqs]
            ### Replace the first action's timedelta (np.nan) in each sequence, with an arbitrary large timedelta value
            test_timedelta_seqs = [[999 if np.isnan(x) else x for x in s] for s in
                                   test_timedelta_seqs]
            np.save(os.path.join(self.path_prefix, token_cache_path, "test_timedelta_sequences_aligned.npy"),
                    np.array(test_timedelta_seqs))
            ## Add error labels and timedeltas as items into the tokenized data object
            for i in range(len(tokenized_data)):
                tokenized_data[i]['error_label'] = test_labels[i]
                tokenized_data[i]['time_delta'] = test_timedelta_seqs[i]

            self.test_dataset = TokenizedDataSet(tokenized_data)

            print(f"[INFO] Token length truncation stats:")
            print(f"   Total sequences: {self.total_sequences_count}")
            print(f"   Sequences truncated: {self.truncated_sequences_count} "
                  f"({(self.truncated_sequences_count / max(1, self.total_sequences_count)) * 100:.2f}%)")

            np.save(os.path.join(self.path_prefix, token_cache_path, f"test_labels.npy"), np.array(test_labels))
            print(f"Saved test_labels.npy with {len(test_labels)} entries to match test_dataset.")


            if self.config.get("cache_static_features"):
                print(f"[INFO] Calculating & Caching static features...")
                # Construct static feature set while you have access to the exact test datasets
                # Create a dataframe to store static features
                df_static_features , wpe_ids = build_static_feature_matrix(test_wpe, test_control)
                # df_static_features['wpe_id'] = [dset.wpe_id for dset in test_wpe + test_control]
                df_static_features['wpe_id'] = wpe_ids
                assert len(df_static_features) == len(test_labels), \
                    f"Mismatch: static features rows ({len(df_static_features)}) vs test labels ({len(test_labels)})"
                df_static_features['error_label'] = test_labels
                df_static_features.to_parquet(os.path.join(self.path_prefix, token_cache_path, f"test_set_static_features.parquet"))
                print(f"FINISHED. Static Feature Matrix Columns:\n{df_static_features.columns}")
    def _truncate_to_last_n_tokens(self, text: str):
        # Just keep last "n_positions" long tokens directly before an order, to keep the most recent context
        # Efficient to do it before tokenization. Addresses issues of logs having different # actions in the preceding time window.
        tokens = self.tokenizer.tokenizer.tokenize(text)
        reserved_tokens = self.config.get("model_configs", {}).get(self.config.get("model", ""), {}).get(
            "prompt_reserved_tokens", 64)
        truncate_len = self.n_positions - reserved_tokens

        self.total_sequences_count += 1
        if len(tokens) > truncate_len:
            self.truncated_sequences_count += 1

        truncated_tokens = tokens[-truncate_len:]
        # print(f"Original length: {len(tokens)}, Truncated to: {len(truncated_tokens)}")

        return self.tokenizer.tokenizer.convert_tokens_to_string(truncated_tokens)

    def train_dataloader(self):
        """
        Returns DataLoader for training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size']['train'],
            num_workers=self.num_workers,
            worker_init_fn=partial(worker_fn, seed=self.seed),  # fix the seed argument for reproducibility
            pin_memory=True,
            # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
            # This can speed up data transfer to the GPU.
            collate_fn=partial(collate_fn),  #n_positions=self.n_positions),
            # fix the n_positions argument for reproducibility
            shuffle=True,  # Shuffle to ensure training data order is random each epoch
        )

    def val_dataloader(self):
        """
        Returns DataLoader for validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size']['val'],
            num_workers=self.num_workers,
            worker_init_fn=partial(worker_fn, seed=self.seed),
            pin_memory=True,
            collate_fn=partial(collate_fn),  #n_positions=self.n_positions),
            shuffle=False,  # No shuffle to ensure consistent evaluation
        )

    def test_dataloader(self):
        """
        Returns DataLoader for testing dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size']['test'],
            num_workers=self.num_workers,
            worker_init_fn=partial(worker_fn, seed=self.seed),
            pin_memory=True,
            collate_fn=partial(collate_fn),  #n_positions=self.n_positions), # investigate
            shuffle=False,  # No shuffle to ensure consistent testing
        )

    def convert_field_vals(self, convert_field_vals_to):
        logging.info(
            "Checkpoint 0: Check whether each possible TIME_DELTA value is mappable to a single token in the Llama-3 tokenizer.")
        timedelta_range = range(0, 301)
        tokenizer = AutoTokenizer.from_pretrained(self.config["model"], token=self.config["HF_access_token"])
        not_single_token_intgers = []
        timedelta_range_token_input_ids = []
        for num in timedelta_range:
            tokenized = tokenizer(str(num), add_special_tokens=False)["input_ids"]
            if len(tokenized) != 1:
                not_single_token_intgers.append(num)
            else:
                timedelta_range_token_input_ids += tokenized
        if len(not_single_token_intgers) == 0:
            print(f"All time delta values <= 5min are mappable to a single token.")
            # print(timedelta_range_token_input_ids)
        else:
            print(
                f"{len(not_single_token_intgers)} time delta values are not mappable to a single token: {not_single_token_intgers}")

        # Init a dataframe to store all user's log
        df_all = pd.DataFrame()
        logging.info(
            "Checkpoint 1: Checking access to raw dataframe column subset from each log file.")

        for dataset in self.datasets:
            # print(f"Showing an example from a single user's log: \n{dataset.df.head()}")
            df_all = pd.concat([df_all, dataset.df], axis=0)


        if convert_field_vals_to == 'numbers':
            logging.info("Checkpoint 2: Find a bunch of integers mappable to a single token in the Llama-3 tokenizer.")

            if 'TIME_DELTA' in df_all.columns:
                uniq_tokens_needed =  df_all.drop(['TIME_DELTA', 'session_ID'], axis=1).nunique().sum()
            else:
                uniq_tokens_needed = df_all.nunique().sum()
            print(f"Total unique field values: {uniq_tokens_needed}")
            print(f"Total unique USER_ID field values: {df_all['USER_ID'].nunique()}")
            print(f"Total unique ACTION_NAME field values: {df_all['ACTION_NAME'].nunique()}")

            # find integers that are not used for timedelta range's token input ids
            available_token_input_ids = list(set(range(0,2000)) - set(timedelta_range_token_input_ids))
            # randomly select enough integers to map field values
            mapped_ints = random.sample(available_token_input_ids, uniq_tokens_needed)
            mapped_ints = [str(i) for i in mapped_ints]
            single_token_map = mapped_ints

            logging.info(
                "Checkpoint 3: Building a mapping dictionary and converting all tabular values to single-token integers.")

            # Get all unique values from dataframe of raw audit log tables
            unique_values = pd.unique(df_all.drop(['TIME_DELTA', 'session_ID'], axis=1).values.ravel())

            # Create a mapping dictionary
            mapping_dict = dict(zip(unique_values, single_token_map))
            df_mapped = df_all.replace(mapping_dict)

            # print(f"Mapped Dataframe (head 5): {df_mapped.head()}")
            # print(f"Mapping dictionary: {mapping_dict}")

            logging.info(
                "Checkpoint 4: Putting back into the dataset module object for easy access from pipeline")

            if len(self.datasets) != df_mapped['USER_ID'].nunique():
                raise ValueError("USER_ID count mismatch between raw and mapped audit log tables versions.")

            for i, dataset in enumerate(self.datasets):
                try:
                    user_id = df_mapped['USER_ID'].unique()[i]
                except KeyError:
                    raise KeyError(
                        "The 'USER_ID' column is missing from the DataFrame. Check num_fields configuration parameter again.")
                dataset.rowStrings, dataset.sessionStrings = self.sessions_to_str(
                    df_mapped[df_mapped['USER_ID'] == user_id],
                    [col for col in df_mapped.columns if col != 'session_ID'])


        elif convert_field_vals_to == 'words':
            logging.info("Checkpoint 2: Find a bunch of people names mappable to a single token in the Llama-3 tokenizer.")

            uniq_tokens_needed = df_all['USER_ID'].nunique()
            print(f"We need at least {uniq_tokens_needed} names to map all audit log USER IDs.")
            # importing a list of people names and checking whether each is single-token mappable
            # List of people names downloaded from: https://catalog.data.gov/dataset/popular-baby-names/resource/02e8f55e-2157-4cb2-961a-2aabb75cbc8b
            # List of people names downloaded from: https://www.kaggle.com/datasets/kaggle/us-baby-names?select=NationalNames.csv
            path = os.path.normpath(os.path.join(os.path.dirname(__file__), "NationalNames.csv"))
            name_list = pd.read_csv(path)
            name_list = name_list['Name'].tolist()


            single_token_names = []
            for name in name_list:
                tokenized = tokenizer(str(name), add_special_tokens=False)["input_ids"]
                if len(tokenized) == 1:  # Check if it maps to a single token
                    single_token_names += tokenized
                if len(single_token_names) == uniq_tokens_needed:
                    print(f"Enough single-token mappable people names have been collected! Stopping search.")
                    break
                else:
                    if name == name_list[-1]:
                        print(f"Not enough peoplenames collected to map all unique values {len(single_token_names)}/{df_all['USER_ID'].nunique()}")

            logging.info(
                "Checkpoint 3: Find a bunch of action verbs mappable to a single token in the Llama-3 tokenizer.")

            uniq_tokens_needed = df_all['ACTION_NAME'].nunique()
            print(f"We need at least {uniq_tokens_needed} action verbs to map all audit log action names")
            # importing a list of action verbs and checking whether each is single-token mappable
            # Downloaded from: https://github.com/glukhman/Learning-English-Past-Tense-RNN
            path = os.path.normpath(os.path.join(os.path.dirname(__file__), "most-common-verbs-english.csv"))
            verb_list = pd.read_csv(path)
            verb_list = verb_list.dropna(how='any')['Word'].tolist()


            single_token_verbs = []
            for verb in verb_list:
                tokenized = tokenizer(str(verb), add_special_tokens=False)["input_ids"]
                if len(tokenized) == 1:  # Check if it maps to a single token
                    single_token_verbs.append(verb)
                if len(single_token_verbs) == uniq_tokens_needed:
                    print(f"Enough single-token mappable action verbs have been collected! Stopping search.")
                    break
                else:
                    if verb == verb_list[-1]:
                        print(
                            f"Not enough verbs collected to map all unique values {len(single_token_verbs)}/{df_all['ACTION_NAME'].nunique()}")

            # single_token_map = {**single_token_names, **single_token_verbs}

    # def _check_token_lengths(self):
    #     from transformers import AutoTokenizer
    #     import matplotlib.pyplot as plt
    #
    #     tokenizer = AutoTokenizer.from_pretrained(self.config['model'], token=self.access_config['HF_access_token'])
    #
    #     sessionStrings = []
    #     for dataset in self.case_datasets:
    #         sessionStrings.extend(dataset.sessionStrings)
    #
    #     lengths = [len(tokenizer.tokenize(text)) for text in sessionStrings]
    #
    #     model_name = self.config["model"]
    #     path_prefix = next((prefix for prefix in self.config["path_prefix"] if os.path.exists(prefix)), "")
    #     save_dir = os.path.join(path_prefix, self.config["project_path"], "diagnostics")
    #     if self.config["n_mini"] is not None:
    #         save_dir = os.path.join(save_dir, f"{self.config['n_mini']}WPEs_{self.config['min_prior']}min_activity")
    #     else:
    #         save_dir = os.path.join(save_dir, self.config['HF_model_name'])
    #     os.makedirs(save_dir, exist_ok=True)
    #     assert os.path.exists(save_dir)
    #
    #     plt.hist(lengths, bins=50)
    #     plt.title('Token Length Distribution of SessionStrings')
    #     plt.xlabel('Token Length')
    #     plt.ylabel('Number of Sequences')
    #     save_plot_path = os.path.join(save_dir, 'token_length_distribution.png')
    #     plt.savefig(save_plot_path)
    #     plt.close()
    #     print(f"Token length distribution plot saved at: {save_plot_path}")
    #
    #     percentiles = [50, 75, 90, 95, 99]
    #     # Save statistics
    #     stats = {
    #         f"Percent > n_positions ({self.n_positions})": sum(l > self.n_positions for l in lengths) / len(
    #             lengths) * 100,
    #         "min": int(np.min(lengths)),
    #         "max": int(np.max(lengths)),
    #         "mean": int(np.mean(lengths)),
    #         "std": int(np.std(lengths)),
    #         "percentiles": {
    #             str(p): int(np.percentile(lengths, p)) for p in percentiles
    #         }
    #     }
    #     save_stats_path = os.path.join(save_dir, 'token_length_stats.txt')
    #     with open(save_stats_path, 'w') as f:
    #         for k, v in stats.items():
    #             if isinstance(v, dict):
    #                 f.write(f"{k}:\n")
    #                 for sub_k, sub_v in v.items():
    #                     f.write(f"  {sub_k}th percentile: {sub_v}\n")
    #             else:
    #                 f.write(f"{k}: {v:.2f}\n")
    #     print(f"Token length statistics saved at: {save_stats_path}")
    #
    #     # Also print for immediate feedback
    #     for k, v in stats.items():
    #         if isinstance(v, dict):
    #             print(f"{k}:")
    #             for sub_k, sub_v in v.items():
    #                 print(f"  {sub_k}th percentile: {sub_v}")
    #         else:
    #             print(f"{k}: {v:.2f}")


    def sessions_to_str(self, df: pd.DataFrame, cols_to_keep: list):
        """
        Combine all rowStrings that are in the same session, using the session_ID as a key. Use
        comma as a delimiter between rowStrings, and a next-line string(\n) denoting
        the end of each session.

        :param df:
        :return: rowStrings (list), sessionStrings (list)
        """
        list_rowStrings = []
        list_sessionStrings = []
        curr_session_ID = None
        field_delimiter = ","
        row_delimiter = "\n"

        # audit_log_headers = field_delimiter.join(cols_to_keep)

        for i, row in df.iterrows():
            if row['session_ID'] != curr_session_ID:
                sessionString = ""
                # sessionString = audit_log_headers + row_delimiter
                curr_session_ID = row['session_ID']

            rowString = field_delimiter.join([str(row[col]) for col in cols_to_keep])
            list_rowStrings.append(rowString)

            sessionString += rowString + row_delimiter

            try:
                if (i == len(df) - 1) or (i < len(df) - 1 and df.loc[i + 1, 'session_ID'] != curr_session_ID):
                    # if current row == last row in session
                    list_sessionStrings.append(sessionString)
            except Exception as e:
                # Print the error trace
                # print("A plotting error occurred:", e)
                print(f"An error occurred: {type(e).__name__}: {e}")
                traceback.print_exc()
                print(f"i: {i}, len(df): {len(df)}")

        return list_rowStrings, list_sessionStrings

class EHRAuditLogTokenizer:
    def __init__(self, loaded_tokenizer=None, yaml_config_path: str = None, model_name="meta-llama/Meta-Llama-3-8B", cache: str = None,
                 access_token: str = None, debug: bool = False, reset_cache: bool = False, n_positions: int = 128):
        """
        Initialize the EHRAuditLogTokenizer with the specified model name.

        Parameters:
        model_name (str): The name of the model to use for tokenization.
        """
        with open(yaml_config_path) as f:
            self.config = yaml.safe_load(f)
        self.model_name = model_name
        self.path_prefix = next((prefix for prefix in self.config["path_prefix"] if os.path.exists(prefix)), "")

        if loaded_tokenizer:
            self.tokenizer = loaded_tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token,
                                                           padding_side=self.config['token_padding_side'])
            if self.config.get("use_delimiter", True):
                # Add special token for row delimiter
                self.tokenizer.add_special_tokens({"additional_special_tokens": ["<ROW>"]})
                # self.tokenizer.special_tokens_map["row_token"] = "<ROW>"

            # Set padding token for the tokenizer.
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<pad>"]})
            # self.tokenizer.pad_token = self.tokenizer.eos_token
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = "<pad>"

            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<FIRST_ROW>"]})
            # self.tokenizer.special_tokens_map["row_token"] = "<FIRST_ROW>"

            if self.config.get("num_fields")>1:
                self.tokenizer.add_special_tokens({"additional_special_tokens": ['[TD_0]', '[TD_10]', '[TD_60]', '[TD_>60]']})



            # Add custom action tokens if custom_tokenization=True
            if self.config.get("custom_tokenization", False):
                action_token_map_path = self.config.get("action_token_map_path", None)
                if action_token_map_path is None:
                    raise ValueError("custom_tokenization=True but action_token_map_path not specified.")
                with open(os.path.join(self.path_prefix, action_token_map_path), "r") as f:
                    action_token_map = json.load(f)
                custom_tokens = list(set(action_token_map.values()))  # Get list of [ACT_xxx] tokens
                self.tokenizer.add_special_tokens({"additional_special_tokens": custom_tokens})
                self.tokenizer.add_special_tokens({"additional_special_tokens": ["[ACT_RARE]"]})
                print(f"[INFO] Added {len(custom_tokens)} custom ACTION_NAME tokens to tokenizer.")

        self.tokenized_dataset = []
        self.debug = debug
        self.reset_cache = reset_cache
        self.n_positions = n_positions
        self.provider_type = self.config.get('provider_type', 'All')



        if cache is not None:
            self.cache = cache
        else:
            self.cache = None


    def tokenize_dataset(self, dataset, padding):
        """
        Tokenize the dataset of sentences.

        Parameters:
        dataset (list): A list of sentences (strings) to tokenize.

        Returns:
        list: A list of tokenized sentences in PyTorch tensor format.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")

        # self.tokenized_dataset = [self.tokenizer(sentence, return_tensors='pt') for sentence in dataset]
        # self.tokenized_dataset = [self.tokenizer(sentence, return_tensors='pt',truncation=True, padding='max_length', max_length=self.n_positions) for sentence in dataset]
        # # Add attention mask extraction
        # self.tokenized_dataset = [{'input_ids': item['input_ids'], 'attention_mask': item['attention_mask']} for item in
        #                           self.tokenized_dataset]

        self.tokenized_dataset = []
        reserved_tokens = self.config.get("model_configs", {}).get(self.config.get("model", ""), {}).get(
            "prompt_reserved_tokens", 64)
        for sentence in dataset:
            tokenized_input = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=padding,
                                             max_length=self.n_positions-reserved_tokens,
                                             add_special_tokens=False # to prevent tokenizer from altering our custom [ACT_xxx] tokens
                                             )
            input_ids = tokenized_input['input_ids'].squeeze(0)
            attention_mask = tokenized_input['attention_mask'].squeeze(0)
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens in the loss
            if self.config.get("use_delimiter", True):
                # Mask out <ROW> tokens from loss computation
                row_token_id = self.tokenizer.convert_tokens_to_ids("<ROW>")
                labels[labels == row_token_id] = -100

            if self.config.get("num_fields") > 1:
                first_row_token_id = self.tokenizer.convert_tokens_to_ids("<FIRST_ROW>")
                labels[labels == first_row_token_id] = -100
                TD_token_ids = self.tokenizer.convert_tokens_to_ids(['[TD_0]', '[TD_10]', '[TD_60]', '[TD_>60]'])
                labels[torch.isin(labels, torch.tensor(TD_token_ids, device=labels.device))] = -100

            # Shift labels by one position to the left
            labels = torch.roll(labels, shifts=-1)
            labels[-1] = -100  # Ignore the last token, which has no corresponding next token

            self.tokenized_dataset.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })

        return self.tokenized_dataset

    def display_tokenized_dataset(self, dataset):
        """
        Display the tokenized dataset.
        """
        if not self.tokenized_dataset:
            raise ValueError("Dataset not tokenized. Call tokenize_dataset() first.")

        for idx, tokenized_input in enumerate(self.tokenized_dataset):
            input_ids = tokenized_input['input_ids'][0]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            print(f"Sentence {idx + 1}:")
            print(f"Original: {dataset[idx]}")
            print(f"Tokens: {tokens}")
            print(f"Token IDs: {input_ids.tolist()}")
            print()
            if self.debug:
                break

    def run_tokenizer(self, dataset, padding, tag=''):
        """
        Tokenize the dataset and cache the tokens.

        Parameters:
        dataset (list): A list of sentences (strings) to tokenize.
        """
        print(f"\tTokenizing with a pretrained tokenizer from {self.model_name}...")
        # self.load_tokenizer()
        self.tokenize_dataset(dataset, padding)
        # self.display_tokenized_dataset()
        if self.cache == os.path.dirname(os.path.abspath(__file__)):
            cache_path = self.cache
        else:
            cache_path = os.path.normpath(os.path.join(self.path_prefix, self.cache))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        if (self.cache is not None and self.reset_cache) or not os.path.exists(os.path.join(cache_path, f"tokenized_dataset{tag}.pt")):
            # with open(os.path.normpath(os.path.join(cache_path, f"tokenized_dataset{tag}.pt")), "wb") as f:
            #     pickle.dump(self.tokenized_dataset, f)
            #     print("\tTokenization complete.")
            torch.save(self.tokenized_dataset, os.path.normpath(os.path.join(cache_path, f"tokenized_dataset{tag}.pt")))
            print("\tTokenization complete & saved.")

    def get_tokenized_dataset(self):
        """
        Retrieve the tokenized dataset.

        Returns:
        list: The tokenized dataset.
        """
        return self.tokenized_dataset

    def load_tokens_from_cache(self, dataset=None, padding=None, tag=''):
        """
        Load the dataset from a cached file.
        """
        if self.cache == os.path.dirname(os.path.abspath(__file__)):
            cache_path = self.cache
        else:
            cache_path = os.path.normpath(os.path.join(self.path_prefix, self.cache))
        # cache_path = os.path.normpath(os.path.join(self.path_prefix, self.cache))
        if not os.path.exists(cache_path):
            raise ValueError("Cache does not exist.")

        # with open(os.path.normpath(os.path.join(cache_path, f"tokenized_dataset{tag}.pt")), "rb") as f:
        #     try:
        #         self.tokenized_dataset = pickle.load(f)
        #     except EOFError:
        #         self.tokenized_dataset = None
        try:
            print(
                f"\tLoading cached tokens generated with pretrained tokenizer associated with {self.model_name}...")
            self.tokenized_dataset = torch.load(
                os.path.normpath(os.path.join(cache_path, f"tokenized_dataset{tag}.pt")))
            print("\tLoading complete.")
        except (EOFError, RuntimeError, UnpicklingError, FileNotFoundError) as e:
            print(
                f"[WARNING] Failed to load cached dataset ({e}). Auto-resetting cache and regenerating dataset.")
            # self.reset_cache = True
            # self.run_tokenizer(dataset, padding, tag)
            # self.reset_cache = True
            # self.run_tokenizer(dataset, padding=padding, tag=tag)
        # print(type(self.tokenized_dataset), type(self.tokenized_dataset[0]))


    def load(self, dataset, padding, tag=''):
        """
        Load the dataset from either a log file or a cache.
        """
        print("Performing tokenization...")
        if self.cache == os.path.dirname(os.path.abspath(__file__)):
            cache_path = self.cache
        else:
            cache_path = os.path.normpath(os.path.join(self.path_prefix, self.cache))
        print(f"\tcache_path: {cache_path}")
        # cache_path = os.path.normpath(os.path.join(self.path_prefix, self.cache))

        cache_file = os.path.join(self.path_prefix, cache_path, f"tokenized_dataset_test.pt")
        # if not self.reset_cache and self.cache and os.path.exists(cache_path):
        if not self.reset_cache and self.cache and os.path.exists(cache_file):
            # print(self.reset_cache)
            self.load_tokens_from_cache(dataset, padding, tag)
        else:
            self.run_tokenizer(dataset, padding, tag)
