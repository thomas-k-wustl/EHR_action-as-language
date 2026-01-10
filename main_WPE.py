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

def save_evaluation_results(evaluation_results, df_results_per_row_ALL, df_results_per_row_GEN, path_prefix,
                            results_path, model_name, format_results, unit_string):
    # Create a DataFrame to store the evaluation results
    # df_results = pd.DataFrame(evaluation_results)
    df_results = pd.DataFrame([{
        "USER_ID": result["USER_ID"],
        "session_ID": result["session_ID"],
        "avg_perplexity (whole)": result["avg_perplexity"],
        "avg_cross_entropy (whole)": result["avg_cross_entropy"],
        "avg_perplexity (generated)": result["avg_generated_perplexity"],
        "avg_cross_entropy (generated)": result["avg_generated_cross_entropy"],
        "bleu_score (generated)": result["bleu_score"],
        "rouge1 (generated)": result["rouge_scores"]["rouge1"].fmeasure,
        "rouge2 (generated)": result["rouge_scores"]["rouge2"].fmeasure,
        "rougeL (generated)": result["rouge_scores"]["rougeL"].fmeasure,
        "input (first half)": result["input"],
        "reference (latter half)": result["reference"],
        "generated_text": result["generated_text"]
    } for result in evaluation_results])

    df_results_summary = pd.DataFrame({"Mean Perplexity (whole input)": [np.mean(df_results["avg_perplexity (whole)"])],
                                       "Mean Cross Entropy (whole input)": [np.mean(
                                           df_results["avg_cross_entropy (whole)"])],
                                       "Mean Perplexity (generated text)": [np.mean(
                                           df_results["avg_perplexity (generated)"])],
                                       "Mean Cross Entropy (generated text)": [np.mean(
                                           df_results["avg_cross_entropy (generated)"])],
                                       "Mean BLEU score (generated text)": [np.mean(
                                           df_results["bleu_score (generated)"])],
                                       "Mean ROUGE-1 score (generated text)": [
                                           np.mean(df_results["rouge1 (generated)"])],
                                       "Mean ROUGE-2 score (generated text)": [
                                           np.mean(df_results["rouge2 (generated)"])],
                                       "Mean ROUGE-L score (generated text)": [
                                           np.mean(df_results["rougeL (generated)"])],
                                       })

    # Save the model evaluation results to a CSV file
    prediction_output_path = os.path.join(path_prefix, results_path, model_name.split("/")[1])
    if not os.path.exists(prediction_output_path):
        os.makedirs(prediction_output_path)
    # Save detailed results (per session)
    csv_output_path = os.path.join(prediction_output_path,
                                   model_name.split("/")[1] + f'_evaluation_results_per_{unit_string}.csv')
    df_results.to_csv(csv_output_path, index=False)

    # Save summary results (transposed)
    csv_summary_output_path = os.path.join(prediction_output_path,
                                           "SUMMARY_" + model_name.split("/")[
                                               1] + f'_evaluation_results_per_{unit_string}.csv')
    df_results_summary_T = df_results_summary.T
    df_results_summary_T.columns = ["Value"]
    df_results_summary_T.to_csv(csv_summary_output_path, index=True)

    # Save the DataFrame to a Parquet file
    parquet_output_path = os.path.join(prediction_output_path,
                                       model_name.split("/")[1] + f'_evaluation_results_per_{unit_string}.parquet')
    df_results.to_parquet(parquet_output_path, index=False)

    print(
        f"Model prediction evaluation results saved to:\n{csv_output_path}\n{csv_summary_output_path}\n{parquet_output_path}")

    if format_results:
        for k, v in {'ALL': df_results_per_row_ALL, 'GEN': df_results_per_row_GEN}.items():
            # Save token-level perplexity & cross entropy values per each Audit Log row into a separate CSV file
            per_row_csv_output_path = os.path.join(prediction_output_path,
                                                   model_name.split("/")[
                                                       1] + f'_token_Perplexity_and_CrossEntropy_per_row_{k}.csv')
            v.to_csv(per_row_csv_output_path, index=False)
            # Save the DataFrame to a Parquet file
            per_row_parquet_output_path = os.path.join(prediction_output_path,
                                                       model_name.split("/")[
                                                           1] + f'_token_Perplexity_and_CrossEntropy_per_row_{k}.parquet')
            v.to_parquet(per_row_parquet_output_path, index=False)
            print(
                f"Token-level Perplexity score & Cross Entropy values (original tabular audit log structure) saved to:\n{per_row_csv_output_path}\n{per_row_parquet_output_path}")


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

    # prompt_text = (
    #     "Predict next row in CSV with columns: user name, seconds passed, action name.\n"
    #     "Each row is an action in sequence. "
    #     "User name repeats. "
    #     "Seconds passed is integer. "
    #     "Action name column is categorical.\n"
    #     "Input:\n"
    # )
    # prompt_text = (
    #     "Predict the next action name. "
    #     "Action names are categorical and occur in a sequence. "
    #     "Each action is delimited by a newline character. "
    #     "Input:\n"
    # )

    if experiment == 1:
        ## Experiment 1: Predicting with Llama-3 without additional training.
        # Example usage of the data module
        dm.prepare_data()

        # Access session strings created through the data loading pipeline
        input_strings = dm.get_input_strings()


        print(f"\tNumber of {unit_string} Strings: {len(input_strings)}")

        token_cache_path = config["model_cache_path"] + model_name.split("/")[1]
        # Initialize the tokenizer module with the provided model configuration
        tokenizer = EHRAuditLogTokenizer(yaml_config_path=config_path,
                                         model_name=model_name,
                                         cache=token_cache_path,
                                         reset_cache=reset_cache,
                                         access_token=access_token,
                                         n_positions=n_positions,
                                         debug=debug)

        # Run the tokenizer on the dataset
        tokenizer.load(input_strings, tag='_all')
        tokenized_inputs = tokenizer.get_tokenized_dataset()

        # # Show an example to tokenized session as a sanity check
        # if debug:
        #     tokenizer.display_tokenized_dataset(input_strings)

        # Testing token prediction with Llama-3 (without training)
        # Initialize the evaluator
        evaluator = LLMEvaluator(model_name=model_name,
                                 access_token=access_token,
                                 tokenizerWrapper=tokenizer,
                                 debug=debug,
                                 format_results=format_results)

        if debug:
            # in debugging mode, run prediction & evaluashtion for a single session string.
            tokenized_inputs = tokenized_inputs[:1]
            input_strings = input_strings[:1]
            # print(f"tokenized_input: {tokenized_inputs}")
            # print(f"DEBUGGING--Printing the first input string (i.e., session): {input_strings}")
            # print(f"Length of the first tokenized input string: {len(tokenized_inputs)}")
        # Evaluate the model's reconstruction capability
        evaluation_results, df_results_per_row_ALL, df_results_per_row_GEN = evaluator.evaluate(tokenized_inputs,
                                                                                                input_strings)
        print(f"Number of {unit_string} strings evaluated: {len(evaluation_results)}")

        # Save evaluation results
        save_evaluation_results(evaluation_results, df_results_per_row_ALL, df_results_per_row_GEN, path_prefix,
                                results_path, model_name, format_results, unit_string)

    if experiment == 'mapping':
        logging.info("Testing audit log field value mapping")
        dm.setup(demo=False, convert_field_vals_to=config.get("convert_field_vals_to", None))
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

    if experiment == 'train':
        logging.info("Running MLE Training pipeline")
        dm.setup(demo=False, convert_field_vals_to=config.get("convert_field_vals_to", None))
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        # ======================TRAINING=============================

        # Print some information to verify setup
        print(f"\nTrain loader: {len(train_loader)} batches")
        print(f"Validation loader: {len(val_loader)} batches")
        print(f"Test loader: {len(test_loader)} batches\n")


        # Initialize the trainer
        trained_model_path = os.path.join(path_prefix, model_cache_path + model_name.split("/")[1], config['HF_model_name'])
        if not os.path.exists(trained_model_path):
            os.makedirs(trained_model_path)

        print(f"Total epochs: {epochs}")
        print(f"Custom loss: {config.get('custom_loss', None)}")
        if config['custom_loss']:
            print(f"Weighting factor for Levenshtein loss: {config.get('custom_loss_w', None)}")
            print(f"\tDynamic adjustment of weight: {config.get('dynamic_w', False)}")

        if config['use_prompt']:
            prompt = prompt_text
        else:
            prompt = ""

        trainer = LLMTrainer(yaml_config_path=config_path,
                             model_name=model_name, access_token=access_token,
                             train_loader=train_loader, val_loader=val_loader,
                             # debug=debug,
                             GPU_ID=config.get("GPU_ID", None),
                             use_quant=config.get("use_quant", False),
                             optimizer_type=config.get("optimizer_type", None),
                             HF_username=config.get("HF_username", None),
                             HF_model_name=config.get("HF_model_name", None),
                             save_model_at=config.get("save_model_at"),
                             save_model_option=config.get("save_model_when")['option'],
                             save_period=config.get("save_model_when")['period'],
                             model_save_path=trained_model_path,
                             custom_loss=config.get("custom_loss", False),
                             custom_loss_w=config.get("custom_loss_w", 0.5),
                             dynamic_w=config.get("dynamic_w", False),
                             use_prompt=config.get("use_prompt", False),
                             prompt=prompt
                             )

        # Train the model for 2 epochs
        trainer.train(epochs=epochs)

        save_path = os.path.join(path_prefix, results_path, model_name.split("/")[1], config['HF_model_name'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Plot and save the losses
        trainer.plot_losses(save_path=save_path)

        # # Load the best model (load all things saved)
        # trainer.load_model(trained_model_path)

    if experiment == 'demo_train':
        logging.info("Running MLE Training pipeline WITH DEMO DATA")
        prompt_text = (
        ""
        "Input:\n"
        )
        dm.setup(demo=True)
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        # ======================TRAINING=============================

        # Print some information to verify setup
        print(f"\nTrain loader: {len(train_loader)} batches")
        print(f"Validation loader: {len(val_loader)} batches")
        print(f"Test loader: {len(test_loader)} batches\n")


        # Initialize the trainer
        trained_model_path = os.path.join(path_prefix, model_cache_path + model_name.split("/")[1], config['HF_model_name'])
        if not os.path.exists(trained_model_path):
            os.makedirs(trained_model_path)

        print(f"Total epochs: {epochs}")
        print(f"Custom loss: {config.get('custom_loss', None)}")
        if config['custom_loss']:
            print(f"Weighting factor for Levenshtein loss: {config.get('custom_loss_w', None)}")
            print(f"\tDynamic adjustment of weight: {config.get('dynamic_w', False)}")

        if config['use_prompt']:
            prompt = prompt_text
        else:
            prompt = ""

        trainer = LLMTrainer(yaml_config_path=config_path,
                             model_name=model_name, access_token=access_token,
                             train_loader=train_loader, val_loader=val_loader,
                             # debug=debug,
                             GPU_ID=config.get("GPU_ID", None),
                             use_quant=config.get("use_quant", False),
                             optimizer_type=config.get("optimizer_type", None),
                             HF_username=access_config.get("HF_username", None),
                             HF_model_name=config.get("HF_model_name", None),
                             save_model_at=config.get("save_model_at"),
                             save_model_option=config.get("save_model_when")['option'],
                             save_period=config.get("save_model_when")['period'],
                             model_save_path=trained_model_path,
                             custom_loss=config.get("custom_loss", False),
                             custom_loss_w=config.get("custom_loss_w", 0.5),
                             dynamic_w=config.get("dynamic_w", False),
                             use_prompt=config.get("use_prompt", False),
                             prompt=prompt
                             )

        # Train the model for 2 epochs
        trainer.train(epochs=epochs)

        save_path = os.path.join(path_prefix, results_path, model_name.split("/")[1], config['HF_model_name'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Plot and save the losses
        trainer.plot_losses(save_path=save_path)

        # # Load the best model (load all things saved)
        # trainer.load_model(trained_model_path)

    if experiment == 'descriptive':
        logging.info("Calculating sequence length distributions")
        dm.get_seq_len_distribution()

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

        ## Outdated; modular implementation for debugging
        # # Test on an unseen set
        # trainer.evaluate_test_set()
        #
        # # Repeated sequential forward-pass to extract per-token values
        # # Testing modular method approach, but highly inefficient. Once validated, move on to unified approach with evaluate_test_set() for a single forward pass.
        # embeddings = trainer.extract_sequence_embeddings(os.path.join(trained_model_path, "outputs/seq_embeddings.pt"))
        # ce_values = trainer.extract_per_token_cross_entropy(os.path.join(trained_model_path, "outputs/action_entropy.json"))

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

    if experiment == 'SFT_test_debug':
        logging.info("[Debugging Mode] Pre-processing & tokenizing unseen test data")

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

        ## Unified method to perform inference evaluation & extractions
        trainer.debug_evaluate_and_extract_all()