import numpy as np
import pandas as pd
import os
import argparse
import yaml
import itertools
import torch
from joblib import Parallel, delayed


# from statsmodels.stats.contingency_tables import mcnemar
# from scipy.stats import wilcoxon, rankdata

"""
All reported metrics:
Report raw aggregated score & bootstrapped 95% CI 
"""
def compute_ECE(all_confs, all_corrects, n_bins=10):
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (all_confs > bin_lower) & (all_confs <= bin_upper)
        # prop_in_bin = in_bin.float().mean().item()

        if np.any(in_bin):
            bin_confidence = np.mean(all_confs[in_bin])
            bin_accuracy = np.mean(all_corrects[in_bin])
            bin_weight = np.sum(in_bin) / len(all_confs)
            ece += np.abs(bin_confidence - bin_accuracy) * bin_weight

    return ece
class BootStrap:
    def __init__(self, model_name, all_corrects, all_topk_corrects, nn_all_corrects, nn_topk_corrects, all_entropy, all_ce, all_confidences):
        self.model_name = model_name
        self.corrects = np.array(all_corrects)
        self.topk_corrects = np.array(all_topk_corrects)
        self.nn_corrects = None if nn_all_corrects is None else np.asarray(nn_all_corrects)
        self.nn_topk_corrects = None if nn_topk_corrects is None else np.asarray(nn_topk_corrects)
        self.entropies = np.array(all_entropy)
        self.cross_entropies = np.array(all_ce)
        self.confidences = np.array(all_confidences)
        self.N = len(all_corrects)
        print(f"Model Name: {self.model_name}")

    def compute_stat(self, metric_name=None):
        if metric_name == "accuracy":
            resample = np.random.choice(self.corrects, size=self.N, replace=True)
            return np.mean(resample)
        elif metric_name == "topk_accuracy":
            resample = np.random.choice(self.topk_corrects, size=self.N, replace=True)
            return np.mean(resample)
        elif metric_name == "nn_accuracy" and self.nn_corrects is not None:
            resample = np.random.choice(self.nn_corrects, size=self.N, replace=True)
            return np.mean(resample)
        elif metric_name == "nn_topk_accuracy" and self.nn_topk_corrects is not None:
            resample = np.random.choice(self.nn_topk_corrects, size=self.N, replace=True)
            return np.mean(resample)
        elif metric_name == "entropy":
            resample = np.random.choice(self.entropies, size=self.N, replace=True)
            return np.mean(resample)
        elif metric_name == "perplexity":
            if self.model_name == 'markov':
                valid = self.cross_entropies[~np.isnan(self.cross_entropies)]
                resample = np.random.choice(valid, size=min(self.N, len(valid)), replace=True)
            else:
                resample = np.random.choice(self.cross_entropies, size=self.N, replace=True)
            return np.exp(np.mean(resample))
        elif metric_name == "ece":
            idxs = np.random.choice(self.N, size=self.N, replace=True)
            return compute_ECE(self.confidences[idxs], self.corrects[idxs])
        return None
    # Generalized bootstrap CI function
    def bootstrap_ci(self, metric_name=None, n_bootstrap=1000, n_jobs=-1):
        if metric_name in ["nn_accuracy", "nn_topk_accuracy"]:
            if (metric_name == "nn_accuracy" and self.nn_corrects is None) or \
                    (metric_name == "nn_topk_accuracy" and self.nn_topk_corrects is None):
                print(f"[SKIP] {metric_name} unavailable (no nn_* data).")
                return np.nan, [np.nan, np.nan]

        stats = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_stat)(metric_name=metric_name)
            for _ in range(n_bootstrap)
        )
        stats = np.array(stats)

        # stats = []
        # for _ in range(n_bootstrap):
        #     stats.append(self.compute_stat(metric_name=metric_name))
        if metric_name == "entropy":
            print(f"[Report] Raw Entropy Score: \nMean: {np.mean(self.entropies)}\nMedian: {np.median(self.entropies)}")
        bootstrap_point_estimate = np.mean(stats)
        bootstrap_ci = np.percentile(stats, [2.5, 97.5])
        print(f"[Report] Bootstrap Mean: {bootstrap_point_estimate}")
        print(f"[Report] Bootstrap 95% CI: {bootstrap_ci}")
        return bootstrap_point_estimate, bootstrap_ci

    def compute_raw_stat(self, metric_name=None):
        if metric_name == "accuracy":
            return np.mean(self.corrects)
        elif metric_name == "topk_accuracy":
            return np.mean(self.topk_corrects)
        elif metric_name == "nn_accuracy" and self.nn_corrects is not None:
            return np.mean(self.nn_corrects)
        elif metric_name == "nn_topk_accuracy" and self.nn_topk_corrects is not None:
            return np.mean(self.nn_topk_corrects)
        elif metric_name == "entropy":
            return np.mean(self.entropies)
        elif metric_name == "perplexity":
            if self.model_name == 'markov':
                return np.exp(np.nanmean(self.cross_entropies))
            else:
                return np.exp(np.mean(self.cross_entropies))
        elif metric_name == "ece":
            idxs = np.random.choice(self.N, size=self.N, replace=True)
            return compute_ECE(self.confidences, self.corrects)
        return None
    def run(self, metric_names=None, save_path=None):
        results = {}
        for metric in metric_names:
            print(f"Metric name: {metric}")
            raw_score = self.compute_raw_stat(metric_name=metric)
            print(f"[Report] Raw {metric} Score: {raw_score}")
            bootstrap_mean, bootstrap_ci = self.bootstrap_ci(metric_name=metric)
            results[metric] = [raw_score, bootstrap_mean, bootstrap_ci]
        df = pd.DataFrame.from_dict(results, orient='index')
        df.columns = ['Raw_Score', 'Bootstrap_Mean', 'Bootstrap_95%CI']
        result_save_path = os.path.join(save_path, f"{self.model_name}_report_bootstrapCI.csv")
        df.to_csv(result_save_path)
        print(f"[FINISHED] Model Reporting saved to {result_save_path}")
"""
Performance comparison across models:
"""
class CompareModels:
    def __init__(self, model1: dict=None, model2: dict=None):
        self.model1 = model1
        self.model2 = model2
        print(f"Models: {self.model1['name']} vs {self.model2['name']}")
    def mcnemar_test(self, metric_name='accuracy'):
        # Binary arrays: 1 = correct, 0 = incorrect
        """
        Binary metrics: Accuracy, Accuracy@Top-5
        For reporting, report accuracy score (proportion) and proportion CI method.

        For binary token-level outcome comparison: use McNemar's test (binary paired test)
        -- Not resampling-based. No bootstrap needed.

        p-value to indicate signif diff.
        McNemar's doesn't estimate effect size . You'd interpret the directionality from the raw difference between accuracy scores.
        Optionally compute OR and CI to quantify likelihood of one model being correct over other when they disagree. From contingency table:

        OR = B/C
        B: Model A correct, Model B wrong
        C: Model A wrong, Model B correct
        """
        if metric_name == "accuracy":
            # Construct McNemar contingency table
            A = np.sum((self.model1['corrects'] == 1) & (self.model2['corrects'] == 1))  # both correct
            B = np.sum((self.model1['corrects'] == 1) & (self.model2['corrects'] == 0))  # A correct, B wrong
            C = np.sum((self.model1['corrects'] == 0) & (self.model2['corrects'] == 1))  # A wrong, B correct
            D = np.sum((self.model1['corrects'] == 0) & (self.model2['corrects'] == 0))  # both wrong
        elif metric_name == "topk_accuracy":
            A = np.sum((self.model1['topk_corrects'] == 1) & (self.model2['topk_corrects'] == 1))
            B = np.sum((self.model1['topk_corrects'] == 1) & (self.model2['topk_corrects'] == 0))
            C = np.sum((self.model1['topk_corrects'] == 0) & (self.model2['topk_corrects'] == 1))
            D = np.sum((self.model1['topk_corrects'] == 0) & (self.model2['topk_corrects'] == 0))

        contingency_table = [[A, B],
                            [C, D]]

        result = mcnemar(contingency_table, exact=False, correction=True)
        print(f"Metric name: {metric_name}")
        print(f"[RESULT] McNemar’s test p-value: {result.pvalue:.4f}")

        # # Optionally Compute Odds Ratio and 95% CI
        # if B > 0 and C > 0:
        #     odds_ratio = B / C
        #     log_or = np.log(odds_ratio)
        #     se_log_or = np.sqrt(1/B + 1/C)
        #     z = 1.96  # for 95% CI
        #     ci_lower = np.exp(log_or - z * se_log_or)
        #     ci_upper = np.exp(log_or + z * se_log_or)
        #     print(f"[RESULT] Odds Ratio: {odds_ratio:.3f}")
        #     print(f"[RESULT] 95% CI for OR: [{ci_lower:.3f}, {ci_upper:.3f}]")
        # else:
        #     print("Cannot compute OR/CI: zero count in discordant pair.")
        #     odds_ratio = np.nan
        #     ci_lower, ci_upper = np.nan, np.nan

        # Example interpretation
        # example: The Markov model achieved higher token-level accuracy (46%) than the LLM (38%).
        # McNemar’s test showed that the difference was statistically significant (p < 0.001), with an odds ratio of 1.71 (95% CI: 1.29–2.24),
        # indicating that when the two models disagreed, Markov was 1.71× more likely to make the correct prediction.
        return result.pvalue #odds_ratio, [ci_lower, ci_upper], result.pvalue
    def paired_test(self, metric_name="entropy"):
        assert len(self.model1['entropies']) == len(self.model2['entropies'])
        stat, p_value = wilcoxon(self.model1['entropies'], self.model2['entropies'], zero_method="wilcox", alternative='two-sided')
        print(f"Metric name: {metric_name}")
        print(f"[RESULT] Wilcoxon signed-rank test p-value: {p_value:.4f}")

        def rank_biserial_effect_size(x, y):
            diff = x - y
            ranks = rankdata(np.abs(diff))
            W_plus = np.sum(ranks[diff > 0])
            W_minus = np.sum(ranks[diff < 0])
            return (W_plus - W_minus) / (W_plus + W_minus)

        # Effect size
        r = rank_biserial_effect_size(self.model1['entropies'], self.model2['entropies'])
        print(f"[RESULT] Rank Biserial effect size: {r:.4f}")
        # effect size interpretation
        # range [-1,1]; small (0.1-0.3), moderate (0.3-0.5), large (>0.5)
        # example: The LLM had significantly lower entropy than the Markov model (Wilcoxon p < 0.001), with a large effect size (r = 0.52), indicating consistent entropy reductions across token positions.
        return r, p_value

    def bootstrap_ece_diff(self, n_iter=1000, n_bins=10):
        """
            Step-by-step: Comparing ECE across models using paired bootstrap
                1.	For each model, save:
                •	confidences: the softmax max values per token
                •	corrects: 1 if prediction was correct, 0 if not
            (These lists must be aligned across models — you’ve already ensured this.)
                2.	Paired resampling across both models:
                •	At each iteration, sample the same token indices (with replacement)
                •	Compute "ECE for each model" using that sample
                •	Record the "difference in ECEs"
            """
        assert len(self.model1['confidences']) == len(self.model2['confidences']) == len(self.model1['corrects']) == len(self.model2['corrects'])
        N = len(self.model1['confidences'])
        ece_diffs = []
        for _ in range(n_iter):
            idxs = np.random.choice(N, size=N, replace=True)
            ece1 = compute_ECE(self.model1['confidences'][idxs], self.model1['corrects'][idxs], n_bins)
            ece2 = compute_ECE(self.model2['confidences'][idxs], self.model1['corrects'][idxs], n_bins)
            ece_diffs.append(ece1 - ece2)
        ece_diffs = np.array(ece_diffs)
        mean_diff = np.mean(ece_diffs)
        std_diff = np.std(ece_diffs)
        median_diff = np.median(ece_diffs)
        ci = np.percentile(ece_diffs, [2.5, 97.5])

        print(f"Metric name: ECE")
        print(f"[RESULT] Bootstrap Mean Diff (SD): {mean_diff:.4f} ({std_diff:.4f})")
        print(f"[RESULT] Bootstrap Median Diff: {median_diff:.4f}")
        print(f"[RESULT] 95% CI: {ci}")
        return mean_diff, std_diff, median_diff, ci

    def bootstrap_perplexity_diff(self, n_iter=1000):
        assert len(self.model1['cross_entropies']) == len(self.model2['cross_entropies'])
        N = len(self.model1['cross_entropies'])
        boot_diffs = []
        for _ in range(n_iter):
            idxs = np.random.choice(N, size=N, replace=True)

            perplexity1 = np.exp(np.mean(self.model1['cross_entropies'][idxs]))
            perplexity2 = np.exp(np.mean(self.model2['cross_entropies'][idxs]))
            boot_diffs.append(perplexity1 - perplexity2)

        mean_diff = np.mean(boot_diffs)
        std_diff = np.std(boot_diffs)
        median_diff = np.median(boot_diffs)
        ci = np.percentile(boot_diffs, [2.5, 97.5])

        print(f"Metric name: Perplexity")
        print(f"[RESULT] Bootstrap Mean Diff (SD): {mean_diff:.4f} ({std_diff:.4f})")
        print(f"[RESULT] Bootstrap Median Diff: {median_diff:.4f}")
        print(f"[RESULT] 95% CI: {ci}")
        return mean_diff, std_diff, median_diff, ci

    def run(self, save_path=None):
        # acc_OR, acc_ci, acc_p = self.mcnemar_test(metric_name='accuracy')
        acc_p = self.mcnemar_test(metric_name='accuracy')
        # topk_acc_OR, topk_acc_ci, topk_acc_p = self.mcnemar_test(metric_name='topk_accuracy')
        topk_acc_p = self.mcnemar_test(metric_name='topk_accuracy')
        r, entropy_p = self.paired_test(metric_name='entropy')
        # perplexity_mean_diff, perplexity_std_diff, perplexity_median_diff, perplexity_ci = self.bootstrap_perplexity_diff()
        # ece_mean_diff, ece_std_diff, ece_median_diff, ece_ci = self.bootstrap_ece_diff()

        # results = {'accuracy': [acc_OR, acc_ci, acc_p, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        #            'topk_accuracy': [topk_acc_OR, topk_acc_ci, acc_p, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        #            'entropy': [np.nan, np.nan, np.nan, r, entropy_p, np.nan, np.nan, np.nan, np.nan],
        #            'perplexity': [np.nan, np.nan, np.nan, np.nan, np.nan, perplexity_mean_diff, perplexity_std_diff, perplexity_median_diff, perplexity_ci],
        #            'ece': [np.nan, np.nan, np.nan, np.nan, np.nan, ece_mean_diff, ece_std_diff, ece_median_diff, ece_ci]
        #            }
        results = {'accuracy': [np.nan, acc_p],
                   'topk_accuracy': [np.nan, topk_acc_p],
                   'entropy': [r, entropy_p]}
        df = pd.DataFrame.from_dict(results, orient='index')
        # df.columns = ['McNemar_OR', 'McNemar_95%CI', 'McNemar_P', 'Wilcoxon_r', 'Wilcoxon_P', 'Bootstrap_mean', 'Bootstrap_SD', 'Bootstrap_median', 'Bootstrap_95%CI']
        df.columns = ['Effect Size', 'P-value']
        result_save_path = os.path.join(save_path, f"compare_models_{self.model1['name']}_{self.model2['name']}.csv")
        df.to_csv(result_save_path)
        print(f"[FINISHED] Model comparison saved to {result_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Inference Reporting & Model Comparison")
    parser.add_argument('--config_file', type=str, default="config_WPE.yaml",
                        help="Path to config YAML file (default: config_WPE.yaml)")
    args = parser.parse_args()
    # Load configuration from YAML file
    # config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "config_WPE.yaml"))
    config_path = os.path.abspath(args.config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    path_prefix = next((prefix for prefix in config["path_prefix"] if os.path.exists(prefix)), "")
    model_name = config.get("model")
    model_cache_path = config.get("model_cache_path")
    save_path = os.path.join(path_prefix, model_cache_path + model_name.split("/")[1],
                                   config['HF_model_name'])

    model_cache_dict = {}
    # for model in ['markov', 'llm', 'llm-prompt']:
    # for model in ['llm', 'llm-prompt']:
    for model in ['markov']:
    # for model in ['llm-prompt']:
        model_cache_dict[model] = {'name': model,
                  'corrects': torch.load(os.path.join(save_path, f"{model}_tokens_correct.pt")).numpy(),
                  'topk_corrects': torch.load(os.path.join(save_path, f"{model}_tokens_top5_correct.pt")).numpy(),
                  'nn_corrects': None,
                  'nn_topk_corrects': None,
                  'entropies': torch.load(os.path.join(save_path, f"{model}_tokens_entropy.pt")).numpy(),
                  'cross_entropies': torch.load(os.path.join(save_path, f"{model}_tokens_cross_entropy.pt")).numpy(),
                  'confidences': torch.load(os.path.join(save_path, f"{model}_tokens_confidence.pt")).numpy(),
                  }
        if 'llm' in model:
            model_cache_dict[model]['nn_corrects'] = torch.load(os.path.join(save_path, f"{model}_tokens_nn_correct.pt")).numpy()
            model_cache_dict[model]['nn_topk_corrects'] = torch.load(os.path.join(save_path, f"{model}_tokens_nn_top5_correct.pt")).numpy()


        reporting = BootStrap(model,
                              model_cache_dict[model]['corrects'],
                              model_cache_dict[model]['topk_corrects'],
                              model_cache_dict[model]['nn_corrects'],
                              model_cache_dict[model]['nn_topk_corrects'],
                              model_cache_dict[model]['entropies'],
                              model_cache_dict[model]['cross_entropies'],
                              model_cache_dict[model]['confidences']
                              )
        reporting.run(metric_names=['accuracy', 'topk_accuracy', 'nn_accuracy', 'nn_topk_accuracy', 'entropy', 'perplexity', 'ece'],
                      save_path=save_path)

    # # Run paired statistical tests for each model pair
    # for model_pair in itertools.permutations(model_cache_dict, 2):
    #     model1 = model_pair[0]
    #     model2 = model_pair[1]
    #     comparison = CompareModels(model_cache_dict[model1], model_cache_dict[model2])
    #     comparison.run(save_path=save_path)
