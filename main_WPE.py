import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Set CUDA_HOME to ensure DeepSpeed can find CUDA
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["PATH"] = os.environ["CUDA_HOME"] + "/bin:" + os.environ.get("PATH", "")

# Set environment variables for NVIDIA GPU visibility and capabilities
os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
os.environ['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,utility'
current_script_dir = os.path.dirname(os.path.abspath(__file__))

###########################################################################################
## Use below code block for caching while using RIS compute

# Set HF_HOME to a directory where you have write permissions.
# This has to be set, so that the HuggingFace transformers library knows the location of its cache directory.

HF_cache_path = os.path.join(current_script_dir, '.cache', 'huggingface')
if not os.path.exists(HF_cache_path):
    os.makedirs(HF_cache_path, exist_ok=True)
os.environ['HF_HOME'] = HF_cache_path

# Set MPLCONFIGDIR to a directory relative to the current script location where you have write permissions.
MPL_cache_path = os.path.join(current_script_dir, '.config/matplotlib')
if not os.path.exists(MPL_cache_path):
    os.makedirs(MPL_cache_path, exist_ok=True)
os.environ['MPLCONFIGDIR'] = MPL_cache_path
#
# TRANSFORMERS_CACHE_DIR = os.path.join(current_script_dir, '.cache', 'transformers')
# if not os.path.exists(TRANSFORMERS_CACHE_DIR):
#     os.makedirs(TRANSFORMERS_CACHE_DIR)
# os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR
# ###########################################################################################
# ## Else (e.g., on argonaute server where you don't have write permissions to mounted RIS drive)
# # Use your home directory for write permissions
# HF_CACHE = os.path.expanduser("~/.cache/huggingface")
# os.makedirs(HF_CACHE, exist_ok=True)
# os.environ['HF_HOME'] = HF_CACHE
# # Also tell the hub explicitly where to put its files
# os.environ['HF_HUB_CACHE'] = os.path.join(HF_CACHE, "hub")

###########################################################################################
# import lightning.pytorch as pl
import numpy as np
import torch
torch.cuda.empty_cache()
import yaml
import pandas as pd
from modules_WPE import EHRAuditLogDataModule, EHRAuditLogTokenizer
# from modules_blora import BayesianLoRA
from SFTmodules_WPE import EHRAuditLogSFTTrainer
import gc
import warnings
import logging
# Configure logging to show messages with INFO level or higher
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
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

    access_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "access_config.yaml"))
    with open(access_config_path, "r") as f:
        access_config = yaml.safe_load(f)

    if config.get('ignore_warnings'):
        warnings.filterwarnings("ignore")

    path_prefix = next((prefix for prefix in config["path_prefix"] if os.path.exists(prefix)), "")
    print(f"Is the network drive connected? {'Yes' if len(path_prefix) > 0 else 'No'}")
    print(f"\tNetwork drive path mounted within Docker: {path_prefix}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"Multiple GPUs available? {torch.cuda.device_count() > 1}")
    print(f"\tSpecified GPUs to use? {isinstance(config.get('GPU_ID'), list)}")
    print(f"\tUse all available GPUs? {(config.get('GPU_ID') == None)}")
    print(f"\tnum_workers: {config['num_workers']}\n")

    # Retrieve model configurations and other parameters from config file
    # batch_size = config.get("batch_size", 1)
    reset_cache = config.get("reset_cache", False)
    debug = config.get("debug", False)
    format_results = config.get("format_results", False)
    model_name = config.get("model")
    model_configs = config.get("model_configs")
    provider_type = config.get("provider_type", 'All')
    n_positions = model_configs[model_name]['n_positions']
    unit_string = config.get("unit_string", "session")
    access_token = access_config.get("HF_access_token", None)
    seed = config.get("random_seed", 123)
    results_path = config.get("results_path", current_script_dir)
    epochs = config.get("epochs", 2)
    fig_save_path = config.get("fig_save_path", current_script_dir)
    model_cache_path = config.get("model_cache_path", current_script_dir)
    experiment = config.get("experiment", None)

    # Ensure model_name is provided in the configuration
    if model_name is None:
        raise ValueError("Model name must be specified in the configuration file.")
    else:
        print(f"Using base model from HuggingFace: {model_name}")
    print(f"Basic configurations:\n"
          f"\texperiment: {experiment}\n"
          f"\tusing {config['num_fields']} columns from audit logs...\n"
          f"\ttraining batch_size: {config['batch_size']['train']}\n"
          f"\tvalidation batch_size: {config['batch_size']['val']}\n"
          f"\ttesting batch_size: {config['batch_size']['test']}\n\n"
          f"\tsequence length: {n_positions}\n"
          f"\tsearch strategy at generation: {config['search']}\n"
          f"\tuse prompting: {config['use_prompt']}\n"
          )


    # Initialize the data module with provided configurations
    dm = EHRAuditLogDataModule(
        yaml_config_path=config_path,
        access_config_path=access_config_path,
        # batch_size=batch_size,
        reset_cache=reset_cache,
        debug=debug,
        n_positions=n_positions,
        provider_type=provider_type,
        unit_string=unit_string,
        seed=seed
    )




    if experiment == 'setup_tokenized_dataset':
        logging.info("Pre-processing & tokenizing data")
        dm.setup(demo=False)

    if experiment == 'setup_tokenized_OOStest':
        logging.info("Pre-processing & tokenizing unseen test data")

        # Preprocess & tokenize new test data using cached tokenizer
        dm.setup_testset()

    if experiment == 'SFTpipeline':
        logging.info("Pre-processing & tokenizing data")
        dm.setup(demo=False)

        # Initialize the trainer
        save_model_path = os.path.join(path_prefix, model_cache_path + model_name.split("/")[1],
                                          config['HF_model_name'])
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        # print(f"[DEBUG] dataset tokenizer id: {id(dm.tokenizer.tokenizer)}")
        trainer = EHRAuditLogSFTTrainer(
            yaml_config_path=config_path,
            access_config_path=os.path.normpath(os.path.join(os.path.dirname(__file__), "access_config.yaml")),
            model_save_path=save_model_path,
            train_data=dm.train_dataset,
            val_data=dm.val_dataset,
            test_data=dm.test_dataset,
            tokenizer=dm.tokenizer.tokenizer
        )
        trainer.tokenizer = dm.tokenizer.tokenizer
        # print(f"[DEBUG] trainer tokenizer id: {id(trainer.tokenizer)}")

        trainer.model.resize_token_embeddings(len(trainer.tokenizer))
        model_vocab = trainer.model.get_input_embeddings().weight.shape[0]
        tokenizer_vocab = len(trainer.tokenizer)
        assert model_vocab == tokenizer_vocab, f"Model vocab size {model_vocab} != tokenizer size {tokenizer_vocab}"

        # Check model checkpoint existence before evaluation
        loadable_model_checkpoint_path = os.path.join(save_model_path, "best_eval_loss_checkpoint/")
        if not os.path.exists(loadable_model_checkpoint_path) or reset_cache==True:
            trainer.train()
        else:
            if config['load_from_HF']:
                trainer.load_finetuned_model()
            else:
                trainer.load_finetuned_model(loadable_model_checkpoint_path)

        ## Unified method to perform inference evaluation & extractions
        trainer.evaluate_and_extract_all(save_model_path)

    if experiment == 'SFT_test':
        logging.info("Pre-processing & tokenizing unseen test data")

        # Define paths
        save_model_path = os.path.join(path_prefix, model_cache_path + model_name.split("/")[1],
                                       config['HF_model_name'].split("/")[0])
        loadable_cached_checkpoint_path = os.path.join(save_model_path, "best_eval_loss_checkpoint/")

        # Preprocess & tokenize new test data using cached tokenizer
        dm.setup_testset(checkpoint_path=loadable_cached_checkpoint_path)

        # Initialize the trainer
        # print(f"[DEBUG] dataset tokenizer id: {id(dm.tokenizer.tokenizer)}")
        trainer = EHRAuditLogSFTTrainer(
            yaml_config_path=config_path,
            access_config_path=os.path.normpath(os.path.join(os.path.dirname(__file__), "access_config.yaml")),
            model_save_path=save_model_path,
            train_data=None,
            val_data=None,
            test_data=dm.test_dataset,
            tokenizer=dm.tokenizer.tokenizer
        )

        # Check model checkpoint existence before evaluation
        if os.path.exists(loadable_cached_checkpoint_path):
            if config['load_from_HF']:
                trainer.load_finetuned_model()
            else:
                trainer.load_finetuned_model(loadable_cached_checkpoint_path)
        else:
            logging.warning("Model checkpoint doesn't exist. Please specify a valid path.")

        # save_dir = os.path.join(save_model_path, f"test{config.get('temporal_distinct_testset')}")
        save_dir = os.path.join(save_model_path, config.get('HF_model_name').split('/')[1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ## Unified method to perform inference evaluation & extractions
        trainer.evaluate_and_extract_all(save_dir)
