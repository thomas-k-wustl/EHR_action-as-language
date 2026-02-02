# EHR Action-as-Language Audit Log Foundation Model Workflow

This repository contains a research pipeline for modeling **EHR audit log action sequences** as a **language-like sequence** (“actions as tokens”), with a focus on audit log sequences corresponding to ordering events. The code supports:

- preprocessing and caching time-windowed audit log sequences for ordering events (in this case, WPE cases and matched controls),
- representing audit logs in **word-based** or **field-based / structured** forms,
- optional **custom action-token vocabularies** (e.g., `[ACT_123]`) and structured special tokens for field-based models,
- fine-tuning autoregressive LLMs (e.g., llama 3) with **(Q)LoRA** using an SFT-style trainer,
- evaluation and extraction of token-level metrics (accuracy, top-k, cross-entropy/perplexity, entropy),
- baseline comparisons (Markov transition baseline) and model-performance comparison utilities.

> **Data & privacy**: This repository does **not** include any EHR audit log data. All real data are expected to reside in protected storage under appropriate IRB/DUA approvals. The pipeline is designed to run on de-identified / institution-approved extracts.

---

## Contents

- [Contents](#contents)
- [Conceptual overview](#conceptual-overview)
- [Repository layout](#repository-layout)
- [End-to-end run (typical)](#end-to-end-run-typical)
- [Requirements](#requirements)
- [Configuration](#configuration)

---

## Conceptual overview

EHR audit logs record clinician interactions as granular events (e.g., “Chart Review”, “Order Entry”, “Result Viewed”). Traditional analyses often treat events independently, which can miss workflow context. This project treats an EHR interaction stream as a **sequence** analogous to text:

- Each action becomes a token (either a natural-language action name → subword tokens, or a mapped symbolic token like `[ACT_42]`).
- Optional structured fields (e.g., time-delta buckets) can be encoded as special tokens.
- Autoregressive LLMs are fine-tuned to learn workflow regularities and produce predictive distributions over next actions.
- Evaluation focuses on generalization and uncertainty-related metrics (entropy, calibration-related summaries where applicable).

This codebase is currently oriented around **ordering events** (e.g., 30 minutes prior to the index order), but the preprocessing/tokenization abstractions are reusable for other audit-log sequence tasks.

---

## Repository layout

Below, each Python file is listed in the order it appears in the typical workflow. For each file: **what it does**, **inputs**, and **outputs**. Stages are labeled as **Data preprocessing** vs **Model training / evaluation**.

### Data preprocessing

- `prepare_data.py`  
  **What it does:** Extracts orders and audit logs windows preceding the order from raw audit logs; writes cached parquet files per order; also generates the train val test split file by order ID so orders from the same clinician and similar time point are grouped into the same data split.  
  **Inputs:**  
  - `config_*.yaml` - specifies the experimental design
  - `wpe_list` CSV - list of orders
  - Raw audit logs: `{audit_log_path}/{idx}{audit_log_file}`

  **Outputs:**  
  - Cached case order windows: `{audit_log_cache}/{idx}/{idx}_case_{min_prior}m.parquet`  
  - Cached control order windows: `{audit_log_cache}/{idx}/{idx}_control_{min_prior}m.parquet`  
  - `l_parquet_found.pkl`, `l_parquet_notFound.pkl` (informational; not used elsewhere)

- `generate_action_name_token_map.py`  
  **What it does:** Builds the field-based tokenization, i.e., action → `[ACT_*]` token map when `custom_tokenization: True`.  
  **Inputs:**  
  - `config_*.yaml` (paths)  
  - Cached action sequence parquets in `{audit_log_cache}/{idx}/`
 
  **Outputs:** `action_token_map.json` (typically in the `wpe_list/` folder)

- `tfidf_precompute.py`  
  **What it does:** Fits a character n-gram TF‑IDF model over all actions (used during inference‑time to retrieve the closed valid action to the generated natural text for the word-based model).  
  **Inputs:**  
  - `config_*.yaml` (paths)  
  - Cached action sequence parquets in `{audit_log_cache}/{idx}/`
 
  **Outputs:**  
  - `tfidf_vectorizer_char3_5.pkl`  
  - `A_valid_l2norm.npz`  
  - `valid_action_texts.pkl` / `.txt`

### Model training / evaluation

- `modules_WPE.py`  
  **What it does:** Loads cached action sequence parquets into dataset objects, applies the data split to build train/val/test sequences, tokenizes them, and writes tokenized caches.  
  **Inputs:**  
  - `fixed_wpe_splits.pt`  
  - Cached parquets in `{audit_log_cache}/{idx}/`  
  - `action_token_map.json` (if `custom_tokenization: True`)  
  - `config_*.yaml`, `access_config.yaml`
 
  **Outputs:**  
  - `cached_case_datasets.pt`  
  - `control_chunk_*.pt`  
  - `tokenized_dataset_{train,val,test}.pt`  

- `data_WPE.py`  
  **What it does:** Defines two classes used by `modules_WPE.py`: `EHRAuditLogDataSet` and `TokenizedDataSet`

- `SFTmodules_WPE.py`  
  **What it does:** Fine‑tunes the LLM (SFT/LoRA/QLoRA), evaluates next‑action prediction, and extracts token‑level metrics.  
  **Inputs:**  
  - Tokenized datasets from `modules_WPE.py`  
  - TF‑IDF artifacts from `tfidf_precompute.py` (for word to closest action retrieval)  
  - `config_*.yaml`, `access_config.yaml`
 
  **Outputs:**  
  - Model checkpoints (local or HF)  
  - Evaluation artifacts (per‑token metrics, embeddings, etc.)

- `main_WPE.py`  
  **What it does:** Orchestrates the full pipeline: data module setup → model training → evaluation → result saving.  
  **Inputs:** `config_*.yaml`, `access_config.yaml`  
  **Outputs:** Model checkpoints + evaluation outputs (in configured `results_path`)

- `markov_baseline_updated.py`  
  **What it does:** Markov next‑action baselines for comparison to the LLM.  
  **Inputs:** Tokenized datasets / cached data, `config_*.yaml`  
  **Outputs:** Baseline metrics

- `model_performance_comparison.py`  
  **What it does:** Compares metrics across model runs and generates summary statistics.  
  **Inputs:** Saved evaluation outputs from multiple runs  
  **Outputs:** Comparison tables / reports

---

## End-to-end run (typical)

1) **Prepare cached order windows + split data**
```
python prepare_data.py
```

2) **Optional: build action token map for field-based models (only if `custom_tokenization: True`)**
```
python generate_action_name_token_map.py --config_file config_fullWPE_WB_prompt-T3.yaml --top_k 5000
```

3) **Optional: precompute TF‑IDF artifacts (for word-action retrieval chunk‑level evaluation in `SFTmodules_WPE.py`)**
```
python tfidf_precompute.py --config_file config_fullWPE_WB_prompt-T3.yaml
```

4) **Train + evaluate**
```
python main_WPE.py --config_file config_fullWPE_WB_prompt-T3.yaml
```

5) **Compare runs (optional)**
```
python model_performance_comparison.py --config_file config_fullWPE_WB_prompt-T3.yaml
```

---

## Requirements

### Python
- Python 3.9+ recommended

### Core libraries (indicative)
- `torch`
- `transformers`
- `peft` (LoRA/QLoRA)
- `pandas`, `numpy`
- `pyarrow` (for parquet)
- `joblib`, `tqdm`
- `matplotlib`
- optional: `wandb`


---

## Configuration

### Required: `config_WPE.yaml`
`main_WPE.py` is driven by a YAML config file. The included example:

- `config_fullWPE_WB_prompt-T3.yaml`

### Required: `access_config.yaml` (not present in repo)
`main_WPE.py` requires an `access_config.yaml`, which is used to configure your hugging-face and WANDB tokens:
```yaml
HF_access_token: "YOUR_HF_TOKEN"
# Optional:
# WANDB_API_KEY: "..."
