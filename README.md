# EHR Action-as-Language (Wrong-Patient Error Workflow Modeling)

This repository contains a research pipeline for modeling **EHR audit log action sequences** as a **language-like sequence** (“actions as tokens”), with a focus on *wrong-patient error (WPE)* case–control data. The code supports:

- preprocessing and caching time-windowed audit log sequences for WPE cases and matched controls,
- representing audit logs in **word-based** or **field-based / structured** forms,
- optional **custom action-token vocabularies** (e.g., `[ACT_123]`) and structured special tokens,
- fine-tuning autoregressive LLMs (e.g., Llama 3) with **(Q)LoRA** using an SFT-style trainer,
- evaluation and extraction of token-level metrics (accuracy, top-k, cross-entropy/perplexity, entropy),
- baseline comparisons (Markov transition baseline) and model-performance comparison utilities.

> **Data & privacy**: This repository does **not** include any EHR audit log data. All real data are expected to reside in protected storage under appropriate IRB/DUA approvals. The pipeline is designed to run on de-identified / institution-approved extracts.

---

## Contents

- [Conceptual overview](#conceptual-overview)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Data layout expected by the pipeline](#data-layout-expected-by-the-pipeline)
- [Quickstart](#quickstart)
- [Running experiments](#running-experiments)
- [Outputs](#outputs)
- [Baselines and comparisons](#baselines-and-comparisons)
- [Reproducibility notes](#reproducibility-notes)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Conceptual overview

EHR audit logs record clinician interactions as granular events (e.g., “Chart Review”, “Order Entry”, “Result Viewed”). Traditional analyses often treat events independently, which can miss workflow context. This project treats an interaction stream as a **sequence** analogous to text:

- Each action becomes a token (either a natural-language action name → subword tokens, or a mapped symbolic token like `[ACT_42]`).
- Optional structured fields (e.g., time-delta buckets) can be encoded as special tokens.
- Autoregressive LLMs are fine-tuned to learn workflow regularities and produce predictive distributions over next actions.
- Evaluation focuses on generalization and uncertainty-related metrics (entropy, calibration-related summaries where applicable).

This codebase is currently oriented around **WPE case–control windows** (e.g., 30 minutes prior to the index event), but the preprocessing/tokenization abstractions are reusable for other audit-log sequence tasks.

---

## Repository layout

Top-level scripts/modules included in this repo:

- `main_WPE.py`  
  Primary entry point to run the WPE pipeline (data module setup, training, evaluation, extraction).

- `modules_WPE.py`  
  Core data and tokenization utilities:
  - `EHRAuditLogDataModule` (dataset loading/caching + dataloaders)
  - `EHRAuditLogTokenizer` (special tokens, optional action token maps, truncation/padding)

- `data_WPE.py`  
  Dataset definition (`EHRAuditLogDataSet`) and WPE-specific loading assumptions.

- `SFTmodules_WPE.py`  
  Trainer utilities for SFT-style fine-tuning and evaluation callbacks (e.g., `EHRAuditLogSFTTrainer`).

- `prepare_data.py`  
  Utilities for building cached parquet windows and fixed splits.

- `generate_fixed_split.py`  
  Deterministic train/val/test split generation from a WPE list.

- `generate_action_name_token_map.py`  
  Builds an action → `[ACT_*]` mapping JSON used when `custom_tokenization: True`.

- `markov_baseline.py`, `markov_baseline_updated.py`  
  Markov transition baseline(s) for next-action prediction comparisons.

- `model_performance_comparison.py`  
  Utilities to compare metrics across models/runs and produce summary reports.

- `tfidf_precompute.py`  
  Precompute TF-IDF features (useful for earlier-stage representations or baselines).

> **Note on private/internal modules**: `data_WPE.py` imports `auditlog_split` and `rm_auto_gen_actions`. If these modules are part of your private repository but not included in this snapshot, ensure they are available on `PYTHONPATH`. If they are intentionally not distributed, see “Reproducibility notes” for how to stub/disable those parts.

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

Because this repository is used in protected environments with varying cluster/container setups, dependency installation is typically done via your lab’s environment management (conda, pip, or container images). If you plan to share this repo beyond your lab, consider adding either:
- `requirements.txt`, or
- `environment.yml` (conda), or
- a container recipe.

---

## Configuration

`main_WPE.py` is driven by a YAML config file. The included example:

- `config_fullWPE_WB_prompt-T3.yaml`

### Required: `access_config.yaml`
`main_WPE.py` loads an `access_config.yaml` located next to the script:
```yaml
HF_access_token: "YOUR_HF_TOKEN"
# Optional:
# WANDB_API_KEY: "..."
