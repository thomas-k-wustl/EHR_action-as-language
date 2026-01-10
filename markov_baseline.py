import os
import numpy as np
import pandas as pd
import yaml
import torch
import warnings
import random
warnings.filterwarnings("ignore")
import argparse
import logging
import sys
# Configure logging to show messages with INFO level or higher
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

from transformers import AutoTokenizer, AutoModelForCausalLM
import joblib


current_script_dir = os.path.dirname(os.path.abspath(__file__))

class markovBaseline():
    def __init__(self, yaml_config_path, access_config_path, td_cutoff):
        with open(yaml_config_path) as f:
            self.config = yaml.safe_load(f)
        with open(access_config_path) as f:
            self.access_config = yaml.safe_load(f)

        self.td_cutoff = td_cutoff

        os.environ["HF_TOKEN"] = self.access_config['HF_access_token']
        os.environ["HF_HUB_TOKEN"] = self.access_config['HF_access_token']
        self.access_token = self.access_config.get('access_token')

        # Define paths
        path_prefix = next((prefix for prefix in self.config["path_prefix"] if os.path.exists(os.path.normpath(prefix))), "")
        # print(path_prefix)

        model_cache_path = self.config.get("model_cache_path", current_script_dir)
        model_name = self.config.get("model")

        self.save_dir = os.path.join(path_prefix, model_cache_path + model_name.split("/")[1],
                                self.config['HF_model_name'])

        self.model_dir = os.path.join(path_prefix, model_cache_path + model_name.split("/")[1],
                                      self.config['HF_repo_name'])

        # Check model checkpoint existence before evaluation
        self.loadable_model_checkpoint_path = os.path.join(self.model_dir, "best_eval_loss_checkpoint/")

    def predict_next_action(self):
        logging.info("Begin computing transition matrix")

        logging.info("Loading cached tokenized datasets...")
        train = torch.load(os.path.join(self.model_dir, "tokenized_dataset_train.pt"), map_location='cpu')
        # self.test = torch.load(os.path.join(self.save_dir, "tokenized_dataset_test.pt"), map_location='cpu')
        self.test = torch.load(
            # os.path.join(self.save_dir, f"tokenized_dataset_test{self.config.get('temporal_distinct_testset','')}.pt"),
            # map_location='cpu')
            os.path.join(self.save_dir, "tokenized_dataset_test.pt"),
            map_location='cpu')

        valid_act_token_ids = set(id for tok, id in self.tokenizer.get_vocab().items() if tok.startswith('[ACT_'))
        print(f"[DEBUG] Valid action token ids: {valid_act_token_ids}")
        # Markov transition-based baseline
        skip_N_token = self.config.get("skip_N_token", 10)


        def filter_valid_tokens(item, valid_ids):
            return [tok_tensor.item() for tok_tensor in item['input_ids'] if tok_tensor.item() in valid_ids]

        # # Format it as list of lists (sequences)
        logging.info("Constructing train_sequences (filtered)...")
        train_sequences = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(filter_valid_tokens)(item, valid_act_token_ids) for item in train
        )

        logging.info("Constructing test_sequences (filtered)...")
        self.test_sequences = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(filter_valid_tokens)(item, valid_act_token_ids) for item in self.test
        )
        # print(f"[DEBUG] example test_sequences AFTER KEEPING ONLY [ACT_xx]: {test_sequences[0]}")
        # sys.exit()

        # Extract [ACT_xx] tokens only
        # act_tokens = sorted(set(tok for seq in train_sequences for tok in seq ))
        from itertools import chain
        act_tokens = sorted(set(chain.from_iterable(train_sequences)))
        self.token_to_idx = {tok: i for i, tok in enumerate(act_tokens)}
        self.idx_to_token = {i: tok for tok, i in self.token_to_idx.items()}
        self.N = len(act_tokens)

        # Construct transition matrix
        self.trans_matrix = np.zeros((self.N, self.N), dtype=np.float32)
        for seq in train_sequences:
            for a, b in zip(seq[:-1], seq[1:]):
                if a in self.token_to_idx and b in self.token_to_idx:
                    i, j = self.token_to_idx[a], self.token_to_idx[b]
                    self.trans_matrix[i, j] += 1

        # Normalize matrix rows
        row_sums = self.trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.trans_matrix /= row_sums

        # Compute true-entropy (report) and true-perplexity (unused)
        if not (((isinstance(self.td_cutoff, float)) or (isinstance(self.td_cutoff, int))) and (self.td_cutoff is not None)):
            entropy, true_perplexity = self.compute_true_entropy_and_perplexity(self.trans_matrix)

        if not os.path.exists(os.path.join(self.save_dir, "test_last_action_transition_vectors.csv")):
            self.prepare_error_classification()

        # Evaluate on test set
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        random.seed(self.config.get("random_seed", 123))

        logging.info(f"Begin predicting next action, based on the transition matrix.")

        all_indices = list(range(self.N))

        if ((isinstance(self.td_cutoff, float)) or (isinstance(self.td_cutoff, int))) and (self.td_cutoff is not None):
            td_path = os.path.join(self.save_dir, f"test_timedelta_sequences_aligned.npy")
            timedelta_sequences = np.load(td_path)

            # Compute row-wise entropy over the transition matrix
            row_entropies = np.zeros(self.N)
            for i in range(self.N):
                row = self.trans_matrix[i]
                row_entropies[i] = -np.sum(row * np.log(row + 1e-12))
            user_entropy_values = []

        log_probs = []

        # for seq in self.test_sequences:
        for seq_idx, seq in enumerate(self.test_sequences):
            if len(seq) <= skip_N_token:
                continue
            for i in range(skip_N_token, len(seq)):
                if ((isinstance(self.td_cutoff, float)) or (isinstance(self.td_cutoff, int))) and (self.td_cutoff is not None):
                    td_seq = timedelta_sequences[seq_idx]
                    if td_seq[i] <= self.td_cutoff:
                        continue  # skip this token prediction if timedelta is < td_cutoff -- treating as "auto-generated action"


                prev_token = seq[i - 1] if skip_N_token > 0 else -1
                true_next = seq[i]
                if prev_token in self.token_to_idx:
                    pred_probs = self.trans_matrix[self.token_to_idx[prev_token]]
                    # top5_idx = np.argsort(pred_probs)[-5:][::-1]
                    non_zero_idx = np.flatnonzero(pred_probs)
                    if len(non_zero_idx) >= 5:
                        topk = np.argpartition(pred_probs[non_zero_idx], -5)[-5:]
                        top5_idx = non_zero_idx[topk[np.argsort(pred_probs[non_zero_idx][topk])[::-1]]]
                    else:
                        topk_main = non_zero_idx[np.argsort(pred_probs[non_zero_idx])[::-1]].tolist()
                        remaining = [i for i in all_indices if i not in topk_main]
                        topk_extra = random.sample(remaining, 5 - len(topk_main)) if remaining else []
                        top5_idx = topk_main + topk_extra
                    if ((isinstance(self.td_cutoff, float)) or (isinstance(self.td_cutoff, int))) and (
                            self.td_cutoff is not None):
                        user_entropy_values.append(row_entropies[self.token_to_idx[prev_token]])
                else:
                    # pred_probs = np.full(N, 1.0 / N, dtype=np.float32)
                    top5_idx = random.sample(range(self.N), 5)

                top5_tokens = [self.idx_to_token[idx] for idx in top5_idx]
                total += 1
                if true_next == top5_tokens[0]:
                    correct_top1 += 1
                if true_next in top5_tokens:
                    correct_top5 += 1

                # Compute Perplexity (report) based on cross-entropy
                i = self.token_to_idx.get(prev_token, None)
                j = self.token_to_idx.get(true_next, None)
                if i is not None and j is not None:
                    prob = self.trans_matrix[i][j]
                    if prob > 0:
                        log_probs.append(-np.log(prob))

        if total == 0:
            logging.warning("No valid predictions were made (possibly all test sequences were too short < skip_N_token).")
            return

        acc = correct_top1 / total
        topk_acc = correct_top5 / total
        print(f"Top-1 Accuracy: {correct_top1 / total:.4f}")
        print(f"Top-5 Accuracy: {correct_top5 / total:.4f}")
        print(f"Total Predictions Made: {total}")

        if log_probs:
            cross_entropy = np.mean(log_probs)
            test_perplexity = np.exp(cross_entropy)
            print(f"[RESULT] Cross-Entropy on Test Data (Markov): {cross_entropy:.4f}")
            print(f"[RESULT] Perplexity on Test Data (Markov): {test_perplexity:.4f}")
        else:
            print("[WARNING] No valid test transitions to compute cross-entropy/perplexity.")

        if user_entropy_values:
            entropy = np.mean(user_entropy_values)
            true_perplexity = np.exp(entropy)
            print(f"[RESULT] True Entropy (User-Initiated Tokens Only): {entropy:.4f}")
            print(f"[RESULT] True Perplexity (User-Initiated Tokens Only): {true_perplexity:.4f}")
        else:
            print("[WARNING] No user-initiated tokens to compute user-specific true entropy/perplexity.")


        if ((isinstance(self.td_cutoff, float)) or (isinstance(self.td_cutoff, int))) and (self.td_cutoff is not None):
            results_csv_path = os.path.join(self.save_dir, "markov_inference_results_user_init_actions.csv")
        else:
            results_csv_path = os.path.join(self.save_dir, "markov_inference_results.csv")
        result_col_name = self.config['HF_repo_name'] + '-markov_baseline'

        results_series = pd.Series({
            "Top-1 Accuracy": acc,
            "Top-5 Accuracy": topk_acc,
            "Entropy": entropy,
            "True Perplexity (not used)": true_perplexity,
            "Cross-Entropy (not used)": cross_entropy,
            "Perplexity": test_perplexity
        }, name=result_col_name)

        results_df = results_series.to_frame()  # Converts Series to DataFrame (1 column)
        results_df.to_csv(results_csv_path)
        print(f"[INFO] Evaluation results exported to {results_csv_path}")

    def load_tokenizer(self):
        if self.loadable_model_checkpoint_path is not None and os.path.isdir(self.loadable_model_checkpoint_path):
            logging.info(f"Loading tokenizer from local checkpoint: {self.loadable_model_checkpoint_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.loadable_model_checkpoint_path, trust_remote_code=False)
        else:
            repo_id = f"{self.access_config['HF_username']}/{self.config['HF_repo_name']}"
            logging.info(f"Loading tokenizer from HF checkpoint: {repo_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                use_auth_token=self.access_config['HF_access_token'],
                trust_remote_code=False
            )
        logging.info(
            f"Tokenizer loaded. Tokenizer size: {len(self.tokenizer)}")

    def prepare_error_classification(self):
        logging.info("Preparing downstream error classification input table...")
        # Save 1st-order transition matrix (row: source token, columns: target token probabilities)
        transition_lookup_df = pd.DataFrame(
            self.trans_matrix.copy(),
            index=[self.idx_to_token[i] for i in range(self.N)]
        )
        transition_lookup_df.index.name = "action_token_id"
        transition_lookup_path = os.path.join(self.save_dir, "markov_transition_lookup_table.csv")
        transition_lookup_df.to_csv(transition_lookup_path)
        logging.info("Markov Transition lookup table saved to {transition_lookup_path}")

        # Load error labels from separate file
        # label_path = os.path.join(self.save_dir, f"test{self.config.get('temporal_distinct_testset','')}_labels.npy")
        label_path = os.path.join(self.save_dir, "test_labels.npy")
        error_label_array = np.load(label_path)

        # Construct a table for downstream error classification
        last_token_vectors = []
        error_labels = []

        for i, item in enumerate(self.test):
            seq = self.test_sequences[i]
            if len(seq) == 0:
                continue
            last_token = seq[-1]
            if last_token not in self.token_to_idx:
                continue
            vec = self.trans_matrix[self.token_to_idx[last_token]]
            last_token_vectors.append(vec)
            error_labels.append(error_label_array[i])

        # Combine into DataFrame
        last_action_df = pd.DataFrame(last_token_vectors)
        last_action_df["error_label"] = error_labels

        last_action_table_path = os.path.join(self.save_dir, "test_last_action_transition_vectors.csv")
        last_action_df.to_csv(last_action_table_path, index=False)
        logging.info("Last-action transition vectors with error labels saved to {last_action_table_path}")

    def compute_true_entropy_and_perplexity(self, trans_matrix):
        """
        Compute the true entropy and perplexity of a first-order Markov model.
        """
        eigvals, eigvecs = np.linalg.eig(trans_matrix.T)
        stat_dist = np.real(eigvecs[:, np.isclose(eigvals, 1)])
        stat_dist = stat_dist[:, 0]
        stat_dist = stat_dist / stat_dist.sum()

        eps = 1e-12
        logP = np.log(trans_matrix + eps)
        entropy = -np.sum(stat_dist[:, None] * trans_matrix * logP)
        true_perplexity = np.exp(entropy)

        print(f"[RESULT] True Entropy (Markov): {entropy:.4f}")
        print(f"[RESULT] True Perplexity (Markov): {true_perplexity:.4f}")
        return entropy, true_perplexity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform 1st order Markov transition-based predictions (non-contextual baseline).")
    parser.add_argument('--config_file', type=str, default="config_fullWPE_FB-T3.yaml",
                        help="Path to config YAML file (default: config.yaml)")
    parser.add_argument('--access_config_file', type=str, default="access_config.yaml",
                        help="Path to HF config YAML file (default: access_config.yaml)")
    parser.add_argument('--timedelta_cutoff', type=float, default=None,
                        help="Timedelta cutoff (seconds; float) to filter out consecutive auto-generated actions")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config_file)
    access_config_path = os.path.abspath(args.access_config_file)
    td_cutoff = args.timedelta_cutoff

    markov = markovBaseline(
        yaml_config_path=config_path,
        access_config_path=access_config_path,
        td_cutoff=td_cutoff
    )

    markov.load_tokenizer()
    markov.predict_next_action()


