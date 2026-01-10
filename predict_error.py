import os
import numpy as np
import pandas as pd
import torch
import time


from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, average_precision_score, f1_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

import yaml
import argparse
# Set MPLCONFIGDIR to a directory relative to the current script location where you have write permissions.
current_script_dir = os.path.dirname(os.path.abspath(__file__))
MPL_cache_path = os.path.join(current_script_dir, '.config/matplotlib')
if not os.path.exists(MPL_cache_path):
    os.makedirs(MPL_cache_path, exist_ok=True)
os.environ['MPLCONFIGDIR'] = MPL_cache_path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run error prediction pipeline.")
    parser.add_argument('--config_file', type=str, default="config_error_prediction.yaml",
                        help="Path to config YAML file (default: config_error_prediction.yaml)")
    parser.add_argument('--label_rebalancing', type=str, default="class_weighting",
                        help="Specify label rebalancing strategy to use (default: class_weighting) | Options: class_weighting, undersampling, oversampling")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    args = parser.parse_args()
    # Load configuration from YAML file
    # config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "config_WPE.yaml"))
    config_path = os.path.abspath(args.config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # === Configuration ===
    path_prefix = next((prefix for prefix in config["path_prefix"] if os.path.exists(os.path.normpath(prefix))), "")
    model_cache_path = config['model_cache_path']

    # This must be a list now
        ## llama3-WPE_EHRlogs-FB-T3
        ## llama3-WPE_EHRlogs-FB-T3/test1
        ## llama3-WPE_EHRlogs-FB-T3/test2
    LLM_model_names = config['HF_model_name']


    # debug = True if args.debug=='True' else False
    print(f"DEBUG: {args.debug}")

    # custom_labels = {
    #     ("llama3-WPE_EHRlogs-baseline-CUSTOM", "", "Static Features", "Logistic Regression"): "Static-LR",
    #     ("llama3-WPE_EHRlogs-baseline-CUSTOM", "", "Static Features", "XGBoost"): "Static-XGB",
    #     ("llama3-WPE_EHRlogs-baseline-CUSTOM", "", "LLM Embeddings", "Logistic Regression"): "FB-LR",
    #     ("llama3-WPE_EHRlogs-baseline-CUSTOM", "", "LLM Embeddings", "XGBoost"): "FB-XGB",
    #     ("llama3-WPE_EHRlogs-baseline-timedelta-CUSTOM", "", "LLM Embeddings", "Logistic Regression"): "FB+T-LR",
    #     ("llama3-WPE_EHRlogs-baseline-timedelta-CUSTOM", "", "LLM Embeddings", "XGBoost"): "FB+T-XGB",
    #     ("llama3-WPE_EHRlogs-baseline-CUSTOM", "-prompt", "LLM Embeddings", "Logistic Regression"): "FB+P-LR",
    #     ("llama3-WPE_EHRlogs-baseline-CUSTOM", "-prompt", "LLM Embeddings", "XGBoost"): "FB+P-XGB",
    #     ("llama3-WPE_EHRlogs-baseline-timedelta-CUSTOM", "-prompt", "LLM Embeddings", "Logistic Regression"): "FB+TP-LR",
    #     ("llama3-WPE_EHRlogs-baseline-timedelta-CUSTOM", "-prompt", "LLM Embeddings", "XGBoost"): "FB+TP-XGB",
    #     ("llama3-WPE_EHRlogs-baseline", "", "LLM Embeddings", "Logistic Regression"): "WB-LR",
    #     ("llama3-WPE_EHRlogs-baseline", "", "LLM Embeddings", "XGBoost"): "WB-XGB",
    #     ("llama3-WPE_EHRlogs-baseline-timedelta", "", "LLM Embeddings", "Logistic Regression"): "WB+T-LR",
    #     ("llama3-WPE_EHRlogs-baseline-timedelta", "", "LLM Embeddings", "XGBoost"): "WB+T-XGB",
    #     ("llama3-WPE_EHRlogs-baseline", "-prompt", "LLM Embeddings", "Logistic Regression"): "WB+P-LR",
    #     ("llama3-WPE_EHRlogs-baseline", "-prompt", "LLM Embeddings", "XGBoost"): "WB+P-XGB",
    #     ("llama3-WPE_EHRlogs-baseline-timedelta", "-prompt", "LLM Embeddings", "Logistic Regression"): "WB+TP-LR",
    #     ("llama3-WPE_EHRlogs-baseline-timedelta", "-prompt", "LLM Embeddings", "XGBoost"): "WB+TP-XGB",
    # }
    custom_labels = {
        ("llama3-WPE_EHRlogs-FB-T3", "Static Features", "Logistic Regression"): "Static-LR",
        ("llama3-WPE_EHRlogs-FB-T3", "Static Features", "XGBoost"): "Static-XGB",
        ("llama3-WPE_EHRlogs-FB-T3", "Markov Baseline", "Logistic Regression"): "Markov-LR",
        ("llama3-WPE_EHRlogs-FB-T3", "Markov Baseline", "XGBoost"): "Markov-XGB",
        ("llama3-WPE_EHRlogs-FB-T3", "LLM Embeddings", "Logistic Regression"): "BestLLM-LR",
        ("llama3-WPE_EHRlogs-FB-T3", "LLM Embeddings", "XGBoost"): "BestLLM-XGB",
    }

    if isinstance(LLM_model_names, str):
        LLM_model_names = [LLM_model_names]  # in case only 1 model provided
    all_results = []
    all_roc_curves = []
    all_pr_curves = []

    # Plot ROC & PR Curves across all combinations of custom_labels
    fig, ax = plt.subplots(3, 2, figsize=(12, 15))

    np.random.seed(config["random_seed"])

    for LLM_model_name in LLM_model_names:
        save_model_path = os.path.join(path_prefix, model_cache_path + config.get("model").split("/")[1], LLM_model_name)

        # 1. Load static features
        static_df = pd.read_parquet(os.path.join(save_model_path, "test_set_static_features.parquet"))
        static_labels = static_df.pop('error_label').to_numpy()
        static_df.drop(columns=['wpe_id'], inplace=True)
        static_features = static_df.to_numpy()

        # 2. Load 1st-order Markov transition vectors
        markov_df = pd.read_csv(os.path.join(save_model_path, "test_last_action_transition_vectors.csv"))
        markov_labels = markov_df.pop('error_label').to_numpy()
        markov_transition_vectors = markov_df.to_numpy()

        if not args.debug:
            # 3. Load LLM embeddings
            llm_embeddings = torch.load(os.path.join(save_model_path, f"seq_embeddings-prompt.pt")).to(torch.float32).numpy()
                # Shape = (N, D)
                #     N = number of test samples (same as test_set_static_features.parquet rows)
                #     D = hidden size of model (depends on model, e.g., LLaMA3-8B → D=4096)
            LLM_labels = np.load(os.path.join(save_model_path, f"seq_error_labels-prompt.npy"))
                # Shape = (N,)  → binary labels aligned with embeddings


        if args.debug:
            feature_sets = {
                "Static Features": static_features,
                "Markov Baseline": markov_transition_vectors
            }
        else:
            feature_sets = {
                "Static Features": static_features,
                "Markov Baseline": markov_transition_vectors,
                "LLM Embeddings": llm_embeddings
            }


        # === 5-Fold CV ===
        for feature_name, X in feature_sets.items():

            # Perform normalization on static feature matrix. No need to normalize other feature vectors.
            if feature_name == "Static Features":
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                y = static_labels
            elif feature_name == "Markov Baseline":
                y = markov_labels
            elif feature_name == "LLM Embeddings":
                y = LLM_labels

            if args.label_rebalancing == 'class_weighting':
                # Add class weights to the models to penalize misclassification of the minority class.
                neg_count = np.sum(y == 0)
                pos_count = np.sum(y == 1)

                if feature_name == "LLM Embeddings":
                    # slightly different configuration settings for a more scalable model fitting when using high-dimensional LLM embedding feature matrix
                    models = {
                        "Logistic Regression": LogisticRegression(
                            class_weight='balanced',
                            solver='saga',         # optimized for high-dimensional data
                            max_iter=1000,
                            penalty='l2',
                            n_jobs=-1              # parallelize across CPU cores
                        ),
                        "XGBoost": XGBClassifier(
                            scale_pos_weight=neg_count / pos_count,
                            objective='binary:logistic',
                            tree_method='hist',  # faster training on CPUs
                            max_depth=6,         # Reduces overfitting in high-dimensional settings
                            n_estimators=100,    # Reduce if training is slow
                            learning_rate=0.1,
                            random_state=config["random_seed"],
                            verbosity=1)
                    }
                else:
                    models = {
                        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear'),
                        "XGBoost": XGBClassifier(
                            scale_pos_weight=neg_count / pos_count,
                            use_label_encoder=False,
                            random_state=config["random_seed"],
                            verbosity=0)
                    }
            else:
                if feature_name == "LLM Embeddings":
                    # slightly different configuration settings for a more scalable model fitting when using high-dimensional LLM embedding feature matrix
                    models = {
                        "Logistic Regression": LogisticRegression(
                            solver='saga',         # optimized for high-dimensional data
                            max_iter=1000,
                            penalty='l2',
                            n_jobs=-1              # parallelize across CPU cores
                        ),
                        "XGBoost": XGBClassifier(
                            objective='binary:logistic',
                            tree_method='hist',  # faster training on CPUs
                            max_depth=6,         # Reduces overfitting in high-dimensional settings
                            n_estimators=100,    # Reduce if training is slow
                            learning_rate=0.1,
                            random_state=config["random_seed"],
                            verbosity=1)
                    }
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000,
                                                              solver='liblinear'),
                    "XGBoost": XGBClassifier(
                        use_label_encoder=False,
                        random_state=config["random_seed"],
                        verbosity=0)
                }


            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["random_seed"])

            for model_name, model in models.items():
                metrics = {'accuracy': [], 'auroc': [], 'auprc': [], 'precision': [], 'recall': [], 'f1': []}
                fold_y_trues = []
                fold_y_probas = []
                print(f"===================================================\nFeatures: {feature_name}\nModel: {model_name}\n")

                fold = 1
                # perform stratified k-fold cross-validation
                for train_idx, test_idx in skf.split(X, y):

                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    if args.label_rebalancing == 'oversampling':
                        print(f"===================================================\nBefore Oversampling Minority Class (Errors)\nFold: {fold}\n\tClass distribution in TRAIN set: {np.unique(y_train, return_counts=True)}")
                        majority_indices = np.where(y_train == 0)[0]
                        minority_indices = np.where(y_train == 1)[0]
                        oversample_minority = np.random.choice(minority_indices, size=len(majority_indices),
                                                               replace=True)
                        balanced_indices = np.concatenate([majority_indices, oversample_minority])
                        np.random.shuffle(balanced_indices)

                        X_train = X_train[balanced_indices].copy()
                        y_train = y_train[balanced_indices].copy()
                        print(
                            f"AFTER Oversampling Minority Class (Errors)\nFold: {fold}\n\tClass distribution in TRAIN set: {np.unique(y_train, return_counts=True)}\n===================================================")
                    elif args.label_rebalancing == 'undersampling':
                        print(f"===================================================\nBefore Undersampling Majority Class (Non-errors)\nFold: {fold}\n\tClass distribution in TRAIN set: {np.unique(y_train, return_counts=True)}")
                        majority_indices = np.where(y_train == 0)[0]
                        minority_indices = np.where(y_train == 1)[0]
                        undersample_majority = np.random.choice(majority_indices, size=len(minority_indices),
                                                                replace=False)
                        balanced_indices = np.concatenate([minority_indices, undersample_majority])
                        np.random.shuffle(balanced_indices)

                        X_train = X_train[balanced_indices].copy()
                        y_train = y_train[balanced_indices].copy()
                        print(f"AFTER Undersampling Majority Class (Non-errors)\nFold: {fold}\n\tClass distribution in TRAIN set: {np.unique(y_train, return_counts=True)}\n===================================================")

                    start = time.time()
                    model.fit(X_train, y_train)
                    print(f"Training time: {time.time() - start}")
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]

                    print(f"Fold: {fold}\n\tClass distribution in TEST labels: {np.unique(y_test, return_counts=True)}\n\tClass distribution in TEST predictions: {np.unique(y_pred, return_counts=True)}")
                    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                    metrics['auroc'].append(roc_auc_score(y_test, y_proba))
                    # Precision at the default threshold (typically 0.5)
                    metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
                    # Area under the precision-recall curve (AP)
                    metrics['auprc'].append(average_precision_score(y_test, y_proba))
                    metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
                    metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))

                    fold_y_trues.append(y_test)
                    fold_y_probas.append(y_proba)
                    fold += 1

                # Concatenate all test predictions across folds (effectively reconstructing a single dataset that includes all samples)
                flat_y_true = np.concatenate(fold_y_trues)
                flat_y_proba = np.concatenate(fold_y_probas)
                RocCurveDisplay.from_predictions(flat_y_true, flat_y_proba)

                # label = f"{feature_name} ({model_name})"
                label = custom_labels.get((LLM_model_name, feature_name, model_name), "No Label")
                all_roc_curves.append((flat_y_true.copy(), flat_y_proba.copy(), label))
                all_pr_curves.append((flat_y_true.copy(), flat_y_proba.copy(), label)) ## need fold_y_preds.copy() instead?

                # Average results across all folds
                all_results.append({
                    'Run': label,
                    'Feature Set': feature_name,
                    'Model': model_name,
                    'AUROC': round(np.mean(metrics['auroc']), 2),
                    'AUROC_std': round(np.std(metrics['auroc']), 2),
                    'AUPRC': round(np.mean(metrics['auprc']), 2),
                    'AUPRC_std': round(np.std(metrics['auprc']), 2),
                    'Accuracy': round(np.mean(metrics['accuracy']),2),
                    'Accuracy_std': round(np.std(metrics['accuracy']), 2),
                    'Precision': round(np.mean(metrics['precision']),2),
                    'Precision_std': round(np.std(metrics['precision']), 2),
                    'Recall': round(np.mean(metrics['recall']),2),
                    'Recall_std': round(np.std(metrics['recall']), 2),
                    'F1': round(np.mean(metrics['f1']), 2),
                    'F1_std': round(np.std(metrics['f1']), 2),
                })

        # Combined AUROC plot
        ## ROC curves generated by aggregating test set predictions across 5 CV folds. AUROC values shown in the legend may differ slightly from cross-validated performance reported in the table
        fig, ax = plt.subplots(figsize=(7, 7))
        for y_true, y_proba, label in all_roc_curves:
            # Area under the precision-recall curve (AP)
            auc = roc_auc_score(y_true, y_proba)
            RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=label)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random (AUROC = 0.50)', zorder=1)
        ax.set_title(f"Combined AUROC Curves")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        if not os.path.exists(os.path.join(save_model_path, "plots")):
            os.makedirs(os.path.join(save_model_path, "plots"))
        plt.savefig(
            os.path.join(save_model_path, "plots", f"ROC_curve.png"))
        plt.savefig(
            os.path.join(save_model_path, "plots", f"ROC_curve.eps"))
        plt.close()

        # Combined PRC plot
        fig, ax = plt.subplots(figsize=(7, 7))
        for y_true, y_proba, label in all_pr_curves:
            # Compute area under the precision-recall curve
            ap = average_precision_score(y_true, y_proba)

            disp = PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name=label)
            color = disp.line_.get_color()

            # Optimal global threshold using predictions concatenated from all folds
            thresholds = np.linspace(0.0, 1.0, 101)
            f1s = [f1_score(y_true, y_proba >= t, zero_division=0) for t in thresholds]
            best_threshold = thresholds[np.argmax(f1s)]
            best_f1 = np.max(f1s)
            precision_at_best = precision_score(y_true, y_proba >= best_threshold)
            recall_at_best = recall_score(y_true, y_proba >= best_threshold)
            print(f"\n++++++++++++++++++\nRun: {label}\n\tThresh: {best_threshold:.2f}\n\tPrec: {precision_at_best:.2f}\n\tRec: {recall_at_best:.2f}\n\tF1: {best_f1:.2f}\n++++++++++++++++++\n")
            # ax.scatter(recall_at_best, precision_at_best, color=color, label=f"Best F1 (thresh={best_threshold:.1f})")
            ax.scatter(recall_at_best, precision_at_best, color=color, s=40, zorder=3)
        ax.set_title(f"Combined Precision-Recall Curves")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="upper right")
        plt.savefig(
            os.path.join(save_model_path, "plots", f"PR_curve.png"))
        plt.savefig(
            os.path.join(save_model_path, "plots", f"PR_curve.eps"))
        plt.close()

        # Save full results
        results_df = pd.DataFrame(all_results)
        print(results_df)
        # results_df.to_csv(os.path.join(path_prefix, config['results_path'], "results_all_runs.csv"),
        #                   index=False)
        results_df.to_csv(os.path.join(save_model_path, "error_pred_results_all_runs.csv"),
                          index=False)