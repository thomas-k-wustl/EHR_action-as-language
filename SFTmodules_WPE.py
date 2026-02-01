import os
import torch
from tqdm import tqdm
from functools import partial

from transformers.integrations import WandbCallback

current_script_dir = os.path.dirname(os.path.abspath(__file__))
TRITON_CACHE_DIR=os.path.join(current_script_dir, '.cache', 'triton')
if not os.path.exists(TRITON_CACHE_DIR):
    os.makedirs(TRITON_CACHE_DIR, exist_ok=True)
os.environ['TRITON_CACHE_DIR'] = TRITON_CACHE_DIR

WANDB_CACHE_DIR=os.path.join(current_script_dir, '.cache', 'wandb')
os.environ["WANDB_CACHE_DIR"] = WANDB_CACHE_DIR
os.environ["WANDB_DATA_DIR"] = WANDB_CACHE_DIR

import logging
# Configure logging to show messages with INFO level or higher
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

import yaml
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, EarlyStoppingCallback
#,AdamW
from transformers import TrainerCallback
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
import bitsandbytes as bnb

from datasets import load_dataset
# from accelerate import Accelerator
from trl import SFTTrainer, SFTConfig
from trl.trainer import ConstantLengthDataset
import wandb
import gc
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import sys
from scipy.sparse import load_npz
import re, joblib
from sklearn.preprocessing import normalize


def collate_fn(batch, tokenizer, config=None):

    input_ids_list = [item['input_ids'] for item in batch]
    attention_masks_list = [item['attention_mask'] for item in batch]
    labels_list = [item['labels'] for item in batch]

    # Baseline (normal padding)
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks_list, batch_first=True, padding_value=0)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    # print(f"[DEBUG] First input sample after prompting: {tokenizer.decode(input_ids[0], skip_special_tokens=False)}")

    out = {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }

    # Optional: include error_label if it exists
    if 'error_label' in batch[0]:
        error_labels = torch.tensor([item['error_label'] for item in batch])
        out['error_label'] = error_labels

    # Optional: include time_delta if it exists
    if 'time_delta' in batch[0]:
        time_deltas = torch.tensor([item['time_delta'] for item in batch])
        out['time_delta'] = time_deltas

    # Debug safety check for collate_fn
    vocab_size = len(tokenizer)
    labels_flat = labels[labels != -100]
    if labels_flat.numel() > 0:
        assert labels_flat.min() >= 0, f"[collate_fn] Found negative label values: min={labels_flat.min()}"
        assert labels_flat.max() < vocab_size, f"[collate_fn] Found label values >= vocab size: max={labels_flat.max()}, vocab_size={vocab_size}"

    return out

def worker_fn(worker_id, seed=0):
    """
    Custom worker function to set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EHRAuditLogSFTTrainer:
    def __init__(self, yaml_config_path,
                 access_config_path,
                 model_save_path,
                 train_data=None,
                 val_data=None,
                 test_data=None,
                 demo=False,
                 tokenizer=None
                 ):

        print(f"\nInitializing Training loop...")
        with open(yaml_config_path) as f:
            self.config = yaml.safe_load(f)

        with open(access_config_path) as f:
            self.access_config = yaml.safe_load(f)

        # Must be set this way, with DDP
        os.environ["HF_TOKEN"] = self.access_config['HF_access_token']
        os.environ["HF_HUB_TOKEN"] = self.access_config['HF_access_token']

        # self.debug = debug
        self.use_quant = self.config.get('use_quant', False)
        self.GPU_ID = self.config.get('GPU_ID', None)
        self.model_name = self.config.get('model', None)
        self.access_token = self.access_config.get('access_token')

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.demo = demo

        self.tokenizer = tokenizer

        # GPU device setup
        # *** MODIFIED for multi-GPU: Use LOCAL_RANK if available; ensures each process spawned by torchrun runs on specific GPU
        if "LOCAL_RANK" in os.environ:
            self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))  # ***
        else:
            if self.GPU_ID is not None:
                # If passed list of GPU IDs for multiple GPUs
                if isinstance(self.GPU_ID, list) and torch.cuda.is_available():
                    self.device = torch.device(f'cuda:{self.GPU_ID[0]}')  # Use the first GPU in the list as primary
                else:  # Use single GPU
                    self.device = torch.device(f'cuda:{self.GPU_ID}') if torch.cuda.is_available() else 'cpu'
            else:  # Use all available GPU
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.use_quant:
            # Apply PEFT with QLoRA
            # Step 1: model quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                use_auth_token=self.access_token,
                torch_dtype=torch.bfloat16,# Set to True if running on other GPUs that support bfloat16
                # torch_dtype=torch.float16, # Set to True if running on Tesla V100 GPUs
                # quantization_config=quantization_config,
                device_map={"": self.device}, # disables model parallelism, forces model on specified GPU
                trust_remote_code=False
                # device_map="auto" # this line  overrides the self.device and uses all to do model parallelism (splitting the model across all avail GPUs, because the model doesn't fully fit on one).
            )


            ## Step 2: applying LoRA
            self.peft_config = LoraConfig(
                r=self.config['peft_config']['r'],  # Low-rank parameter -- to try 1,2,4,8
                lora_alpha=self.config['peft_config']['lora_alpha'],  # Alpha parameter
                target_modules=self.config['peft_config']['target_modules'],  # Target weight matrix(matrices) to apply LoRA.
                # In our case, modules from the self-attention block
                # This is the OG paper best results using W_q and W_v (query and value weight matrices)
                # to try out different combos -- W_q, W_k, W_v, W_o, (W_q, W_v), (W_q, W_k, W_v, W_o)
                # for Llama style models, names: q_proj, v_proj, etc
                # for bloom style mdoels, names: query_key_value
                lora_dropout=self.config['peft_config']['lora_dropout'],  # Dropout rate for LoRA
                bias=self.config['peft_config']['bias'],  # Bias type
                task_type=self.config['peft_config']['task_type']  # Task type
            )
            # self.model = get_peft_model(self.model, self.peft_config)
            # # Debugging print
            # # print(f"Device for model parameters after PEFT: {[p.device for p in self.model.parameters()][:5]}")
            # self.model.print_trainable_parameters()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                use_auth_token=self.access_token,
                torch_dtype=torch.bfloat16, # Set to True if running on other GPUs that support bfloat16
                # torch_dtype=torch.float16 # Set to True if running on Tesla V100 GPUs
                trust_remote_code=False
            )
            # Apply PEFT with LoRA
            self.peft_config = LoraConfig(
                r=self.config['peft_config']['r'],  # Low-rank parameter -- to try 1,2,4,8
                lora_alpha=self.config['peft_config']['lora_alpha'],  # Alpha parameter
                target_modules=self.config['peft_config']['target_modules'],
                lora_dropout=self.config['peft_config']['lora_dropout'],  # Dropout rate for LoRA
                bias=self.config['peft_config']['bias'],  # Bias type
                task_type=self.config['peft_config']['task_type']  # Task type
            )
            # self.model = get_peft_model(self.model, self.peft_config)
            # # Debugging print
            # # print(f"Device for model parameters after PEFT: {[p.device for p in self.model.parameters()][:5]}")
            # self.model.print_trainable_parameters()
            self.model.to(self.device)

        # Ensure use_cache is set to False
        self.model.config.use_cache = False

        self.model_save_path = model_save_path

        # Resize model embeddings to match updated tokenizer vocab size
        self.model.resize_token_embeddings(len(self.tokenizer))

        print(f"Model embedding size: {self.model.get_input_embeddings().weight.shape[0]}")
        print(f"Tokenizer length: {len(self.tokenizer)}")

        # ------------------ Use WandB experiment tracker -----------------------
        from datetime import datetime
        import pytz

        # Get the current date and time as a string
        timezone = pytz.timezone('US/Eastern')
        self.current_datetime = datetime.now(timezone).strftime("%Y%m%d_%H%M")

        ############ WANDB LOGGING ############
        os.environ["WANDB_DATA_DIR"] = os.environ["WANDB_CACHE_DIR"]
        if self.config["log_wandb"]:
            # Disable netrc usage
            os.environ["WANDB_DISABLE_NETRC"] = "true"
            # # Set the API key manually
            # os.environ["WANDB_API_KEY"] = self.access_config['wandb_api_key']
            # os.environ["WANDB_PROJECT"] = 'SFTTraining'
            # os.environ["WANDB_ENTITY"] = 'kairos54-washington-university-in-st-louis'

            wandb.login(key=self.access_config['wandb_api_key'], relogin=True, force=False)

            # start a new wandb run to track this script
            wandb.init(
                project=self.config['HF_repo_name']+self.current_datetime,
                entity=self.access_config['wandb_team_name'],
                name=self.config['HF_repo_name']+self.current_datetime,
                tags=["wpe", "SFT", "llama3"]
            )
            wandb.config.update({
                "epochs": self.config['epochs'],
                "batch_size": self.config['batch_size']['train'],
                "learning_rate": self.config['lr']
            })
        ########################################




    def create_datasets(self, tokenizer, streaming=False):
        dataset = load_dataset(
            "ag_news",
            split="train", # "train[:500]"
            num_proc=self.config['num_workers']
        )
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'text'])

        dataset = dataset.train_test_split(test_size=0.3, seed=self.config['random_seed'])
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

        train_dataset = ConstantLengthDataset(
            tokenizer=tokenizer,
            dataset=train_data,
            dataset_text_field='text',
            infinite=True,
            seq_length=self.config.get("model_configs")[self.model_name]['n_positions'],
        )
        valid_dataset = ConstantLengthDataset(
            tokenizer=tokenizer,
            dataset=valid_data,
            dataset_text_field='text',
            infinite=False,
            seq_length=self.config.get("model_configs")[self.model_name]['n_positions'],
        )
        return train_dataset, valid_dataset

    def run_training(self, train_data, val_data):
        self.training_args = TrainingArguments(
            output_dir=self.model_save_path,
            # eval_strategy="epoch",
            # evaluation_strategy="epoch",
            # save_strategy="epoch",
            evaluation_strategy="steps",  # Change this to "steps" for step-based evaluation
            eval_steps=100,  # Run evaluation every 10 steps (or set it to your desired frequency)
            save_strategy="steps",
            save_steps=200,
            save_total_limit=1,
            logging_steps=50,
            num_train_epochs=self.config['epochs'],
            per_device_train_batch_size=self.config['batch_size']['train'],
            per_device_eval_batch_size=self.config['batch_size']['val'],
            learning_rate=self.config['lr'],
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
            gradient_checkpointing=True,
            bf16=True, # Set to True if running on other GPUs that support bfloat16
            # fp16=False, # Set to True if running on Tesla V100 GPUs
            # max_grad_norm=0.0,  # Prevent AMP unscale error that occurs with setting both fp16=True and torch_dtype=torch.float16 in Tesla V100 GPU environment
            optim='adamw_torch',
            weight_decay=0,
            seed=self.config['random_seed'],
            dataloader_num_workers=self.config['num_workers'],
            load_best_model_at_end=True, # Trainer checkpoints the params with the lowest eval_loss.
            metric_for_best_model='eval_loss', # Trainer checkpoints the params with the lowest eval_loss.
            greater_is_better=False, # Trainer checkpoints the params with the lowest eval_loss.
            run_name=self.config['HF_repo_name']+self.current_datetime,
            report_to="wandb" if self.config["log_wandb"] else [],
            # disable_tqdm=True
            ddp_find_unused_parameters=False, # uses Accelerate
            # auto_find_batch_size=True,  # uses Accelerate
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            peft_config=self.peft_config,
            packing=True, #True this forces longer sequences into each batch and increases GPU memory pressure up front
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2),
                       MinimalImprovementStoppingCallback(threshold=0.005),
                       WandbCallback()]
            # stop training if eval_loss has not improved for 2 consecutive evaluation intervals (evaluation steps).
        )

        # print(f"gradient_accumulation_steps: {self.config.get('gradient_accumulation_steps', 1)}")
        # print(f"World size: {self.trainer.args.world_size}") ## Prints 3, when 3 GPUs used

        self.print_trainable_parameters(self.model)

        print("Training...")

        # Train the model
        print(f"Available GPU Memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
        # training_results = self.trainer.train(resume_from_checkpoint=True)
        # Dynamically find the latest checkpoint, if any, and resume from it
        from glob import glob
        checkpoint_dirs = sorted(
            glob(os.path.join(self.model_save_path, "checkpoint-*")),
            key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else -1,
            reverse=True
        )
        if checkpoint_dirs and self.config.get('resume_training', False):
            print(f"[INFO]: Resuming from existing fine-tuned model checkpoint...")
            latest_checkpoint = checkpoint_dirs[0]
            training_results = self.trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            print(f"[INFO]: Starting fresh training run...")
            training_results = self.trainer.train()

        print("Saving best checkpoint of the model locally.")
        self.trainer.model.save_pretrained(os.path.join(self.model_save_path, "best_eval_loss_checkpoint/"))
        self.tokenizer.save_pretrained(os.path.join(self.model_save_path, "best_eval_loss_checkpoint/"))
        logging.info(f"Saved model & tokenizer to local path {os.path.join(self.model_save_path, 'best_eval_loss_checkpoint/')}")

        if self.trainer.is_world_process_zero():
            print("Saving best checkpoint of the model on HuggingFace")
            self.trainer.model.push_to_hub(
                repo_id=f"{self.access_config['HF_username']}/{self.config['HF_repo_name']}",
                private=True,
                use_auth_token=self.access_config['HF_access_token'],
                exist_ok=True
            )
            self.tokenizer.push_to_hub(
                repo_id=f"{self.access_config['HF_username']}/{self.config['HF_repo_name']}",
                use_auth_token=self.access_config['HF_access_token']
            )
            logging.info(f"Pushed model & tokenizer to Hugging Face repo {self.access_config['HF_username']}/{self.config['HF_repo_name']}")

        self.model = self.trainer.model

    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def train(self, train_data=None, val_data=None):
        if self.demo:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.access_config["HF_access_token"], trust_remote_code=False)
            train_dataset, eval_dataset = self.create_datasets(tokenizer)
            self.run_training(train_dataset, eval_dataset)
        else:
            if self.train_data is None or self.val_data is None:
                raise ValueError("train_data and val_data must be provided for training.")
            print(f"[INFO] Total train sequences: {len(self.train_data)}")
            print(f"[INFO] Total val sequences: {len(self.val_data)}")
            self.run_training(self.train_data, self.val_data)

    def load_finetuned_model(self, checkpoint_path=None):
        # logging.info(f"Loading fine-tuned model from checkpoint: {checkpoint_path}")
        # self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16).to(self.device)
        if checkpoint_path is not None and os.path.isdir(checkpoint_path):
            logging.info(f"Loading tokenizer from local checkpoint: {checkpoint_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=False)
            # Load base model with NO checkpoint
            logging.info(f"Loading base model from HuggingFace: {self.model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                use_auth_token=self.access_token,
                torch_dtype=torch.bfloat16,  # Set to True if running on other GPUs that support bfloat16
                trust_remote_code=False
            )

            #  Immediately resize embeddings to match tokenizer
            model.resize_token_embeddings(len(self.tokenizer))


            logging.info(f"Loading fine-tuned model from local checkpoint: {checkpoint_path}")
            # Local checkpoint
            model = PeftModel.from_pretrained(
                model,
                checkpoint_path,
                is_trainable=self.config.get("peft_is_trainable", False)
            )

        else:
            repo_id = f"{self.access_config['HF_username']}/{self.config['HF_repo_name']}"
            logging.info(f"Loading tokenizer from HF checkpoint: {repo_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                use_auth_token=self.access_config['HF_access_token'],
                trust_remote_code=False
            )

            # Load base model with NO checkpoint
            logging.info(f"Loading base model from HuggingFace: {self.model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                use_auth_token=self.access_token,
                torch_dtype=torch.bfloat16,  # Set to True if running on other GPUs that support bfloat16
                trust_remote_code=False
            )

            #  Immediately resize embeddings to match tokenizer
            model.resize_token_embeddings(len(self.tokenizer))

            logging.info(f"Loading LoRA adapter weights from HF repo: {repo_id}")
            model = PeftModel.from_pretrained(
                model,
                repo_id,
                is_trainable=self.config.get("peft_is_trainable", False),
                use_auth_token=self.access_config['HF_access_token']
            )

        self.model = model.to(self.device)
        logging.info(
            f"Model loaded. Tokenizer size: {len(self.tokenizer)}, Embedding size: {self.model.get_input_embeddings().weight.shape[0]}")
        assert len(self.tokenizer) == self.model.get_input_embeddings().weight.shape[0], "Token vocab size and embedding size must be the same."
    def test_dataloader(self):
        """
        Returns DataLoader for testing dataset.
        """
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.config['batch_size']['test'],
            num_workers=self.config["num_workers"],
            # num_workers=min(2, self.config['num_workers']),
            worker_init_fn=partial(worker_fn, seed=self.config["random_seed"]),
            pin_memory=True,
            collate_fn=partial(collate_fn, tokenizer=self.tokenizer, config=self.config),  #n_positions=self.n_positions), # investigate
            shuffle=False,  # No shuffle to ensure consistent testing
        )

    def evaluate_test_set(self):
        # Only supports accuracy & perplexity measures.
        # Use it for debugging, but will eventually be removed.
        logging.info("Evaluating on test set...")

        # Setup minimal eval args
        eval_args = TrainingArguments(
            output_dir=os.path.join(self.model_save_path, "eval"),
            per_device_eval_batch_size=1,
            do_predict=True,
            report_to=[], #specify [] for debugging #specify "wandb" for reporting
            logging_dir=os.path.join(self.model_save_path, "logs"),
        )

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=eval_args
        )

        # For each of the m sequences in the test set with each sequence n tokens long, it computes n-1 token-level predictions.
        output = trainer.predict(self.test_data)
        logits = output.predictions
        labels = output.label_ids
        # Calculate average token-level cross-entropy loss over the entire test set
        loss = output.metrics.get("test_loss")

        # Compute metrics
        preds = np.argmax(logits, axis=-1)
        mask = labels != -100  # Ignore padding tokens
        # flattened, masked, and compared to ground-truth tokens (also flattened).
        # The final acc is the aggregated accuracy across all valid tokens in all m sequences in the test set.
        acc = accuracy_score(labels[mask].flatten(), preds[mask].flatten())
        # Convert cross-entropy to perplexity by exponentiation. Same aggregation logic as the accuracy.
        ppl = torch.exp(torch.tensor(loss)).item() if loss is not None else None

        print(f"[RESULT] Test Accuracy: {acc:.4f}")
        print(f"[RESULT] Test Perplexity: {ppl:.4f}" if ppl else "[RESULT] Perplexity unavailable")

    def extract_sequence_embeddings(self, save_path=None):
        """
        Extracts the last-token hidden state (sequence embedding) for each test sample.
        Saves to `save_path` if specified.
        """
        self.model.eval()
        embeddings = []
        dataloader = self.test_dataloader()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting sequence embeddings"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1][:, -1, :]  # final token embedding
                embeddings.append(last_hidden.cpu())
        all_embeddings = torch.cat(embeddings, dim=0)
        if save_path:
            torch.save(all_embeddings, save_path)
        return all_embeddings

    def extract_per_token_cross_entropy(self, save_path=None):
        """
        Extracts per-token cross-entropy and per-sequence perplexity.
        Saves to .npy if `save_path` is provided.
        """
        self.model.eval()
        ce_values = []
        dataloader = self.test_dataloader()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting token-level CE"):
                input_ids = batch['input_ids'].to(self.device)
                outputs = self.model(input_ids=input_ids, labels=input_ids)

                logits = outputs.logits[:, :-1, :]  # predict next token
                labels = input_ids[:, 1:]

                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                token_ce = -token_log_probs.squeeze(0).cpu().tolist()
                ce_values.append(token_ce)

        if save_path:
            np.save(os.path.join(save_path, "cross_entropy.npy"), ce_values)

        return ce_values

    def evaluate_and_extract_all(self, save_dir=None):
        """
        TOBE used later, once modular methods are verified.
        Performs a single-pass inference over the test set to compute:
        - Token-level accuracy and perplexity
        - Per-token cross-entropy values
        - Sequence-level embeddings (last token's hidden state)
        - Top-k (k=5) accuracy per token
        - Token-level entropy
        Saves outputs if `save_dir` is provided.
        """
        print(f"[DEBUG] Prompting active: {self.config.get('use_prompt', False)}")
        self.model.eval()
        all_correct_count, all_labels, all_ce, all_ce_values, all_ppls, all_embeddings = [], [], [], [], [], []
        all_confidences, all_corrects, all_topk_corrects = [], [], []
        all_topk_correct_count, all_entropy, all_ece = [], [], []
        all_labels_binary = []
        # time_deltas = []

        all_correct_chunks, all_total_chunks = [], []
        all_chunk_nn_top1, all_chunk_nn_topk = [], []
        all_chunk_ce, all_chunk_entropy, all_chunk_topk_acc, all_chunk_confidence = [], [], [], []

        def split_chunks_by_row(batch_tensor, row_token_id):
            """Splits each sequence in the batch into chunks by <ROW>."""
            chunks = []
            prev = 0
            for i, token in enumerate(batch_tensor):
                if token.item() == row_token_id:
                    if prev < i:
                        chunks.append(batch_tensor[prev:i])
                    prev = i + 1
            return chunks


        # Store by token position (sequence index)
        dataloader = self.test_dataloader()
        # import psutil
        # print(f"[DEBUG] CPU Memory: {psutil.virtual_memory().percent}% used")
        # print(f"[DEBUG] CUDA Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
        with (torch.no_grad()):
            if self.config.get("custom_tokenization") == False:
                if self.config.get("temporal_distinct_testset") is not None:
                    self.vectorizer = joblib.load(os.path.join(save_dir, "../tfidf_vectorizer_char3_5.pkl"))
                    self.A_valid_l2 = load_npz(os.path.join(save_dir, "../A_valid_l2norm.npz"))
                    self.valid_action_texts = joblib.load(os.path.join(save_dir, "../valid_action_texts.pkl"))
                else:
                    self.vectorizer = joblib.load(os.path.join(save_dir, "tfidf_vectorizer_char3_5.pkl"))
                    self.A_valid_l2 = load_npz(os.path.join(save_dir, "A_valid_l2norm.npz"))
                    self.valid_action_texts = joblib.load(os.path.join(save_dir, "valid_action_texts.pkl"))

            def preprocess_text(s: str) -> str:
                s = s.lower().strip()
                return re.sub(r"\s+", " ", s)

            for batch in tqdm(dataloader, desc="Unified Evaluation & Feature Extraction"):

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                max_id = input_ids.max().item()
                model_vocab = self.model.get_input_embeddings().weight.shape[0]
                assert max_id < model_vocab, f"Batch has input_id {max_id} >= model_vocab {model_vocab}"

                from torch.cuda.amp import autocast
                with autocast(dtype=torch.bfloat16):
                # If needed, disable autocast for test loop
                # with torch.cuda.amp.autocast(enabled=False):
                    if self.config.get("use_prompt", False):

                        if self.config.get("instruct_text") is None:
                            # Option 1: Warm-up-prompting experiment: Use input_ids as labels (shifted by skip_N_tokens masking)
                            inputs = input_ids
                            labels = batch['labels'].to(self.device)
                            skip_N_tokens = min(self.config.get("skip_N_tokens", 0), labels.shape[1])
                            if self.config.get("custom_tokenization") == False:
                                # Shift input_ids left by one along dim=1 (sequence dimension)
                                labels = torch.roll(input_ids, shifts=-1, dims=1)
                                # Set last token of each sequence to -100 (no next-token target)
                                labels[:, -1] = -100
                        else:
                            # Option 2: Instruction-prompting experiment
                            self.instruction = self.tokenizer.encode(self.config.get("instruct_text", ""),
                                                                return_tensors="pt").to(self.device)
                            inputs = torch.cat([self.instruction.expand(input_ids.size(0), -1), input_ids], dim=1)
                            # Adjust attention_mask to match prepended instruction
                            if attention_mask is not None:
                                prompt_mask = torch.ones((input_ids.size(0), self.instruction.size(1)), dtype=attention_mask.dtype).to(self.device)
                                attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
                            # Adjust labels to match prpended instruction
                            labels = torch.cat([
                                torch.full((input_ids.size(0), self.instruction.size(1)), -100, dtype=torch.long).to(
                                    self.device),
                                input_ids
                            ], dim=1)

                            if self.config.get("only_instruct", False):
                                # Make the model see [Instruction] + [Tokens to predict]
                                skip_N_tokens = self.instruction.size(1)
                            else:
                                # Make the model see [Instruction] + [Prompt examples] + [Tokens to predict]
                                skip_N_tokens = self.instruction.size(1) + min(self.config.get("skip_N_tokens", 0), labels.shape[1])

                        if skip_N_tokens > 0:
                            labels[:, :skip_N_tokens] = -100
                            # print(
                            #     f"[DEBUG] +Prompt mode active: skip_N_tokens = {skip_N_tokens} → first {skip_N_tokens} tokens set to -100 for masking.")
                            # prompt_tokens = inputs[0, :skip_N_tokens].tolist()
                            # prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=False)
                            # print(f"[DEBUG] Decoded prompt text for first sequence:\n{prompt_text}")

                    else:
                        ## Redundant re-shifting. Tokenized dataset already has correct full input ID seq and left-shifted labels.
                        # # Baseline: standard next-token prediction (shift input_ids by 1)
                        # inputs = input_ids[:, :-1]
                        # labels = input_ids[:, 1:]
                        ## New fix that directly uses batch-stored input_ids and labels which were already prepared and shifted correctly.
                        ## if no prompting, we don't need the first warm-up part of the inputs
                        # inputs = input_ids
                        # labels = batch['labels'].to(self.device)
                        # For non-prompted version only
                        inputs = input_ids[:, self.config.get("skip_N_tokens", 0):]
                        attention_mask = attention_mask[:, self.config.get("skip_N_tokens", 0):]
                        labels = batch['labels'].to(self.device)

                        if self.config.get("custom_tokenization") == False:
                            # Shift input_ids left by one along dim=1 (sequence dimension)
                            labels = torch.roll(input_ids, shifts=-1, dims=1)
                            # Set last token of each sequence to -100 (no next-token target)
                            labels[:, -1] = -100
                        # safe_inputs = input_ids.clone()
                        # safe_inputs[safe_inputs == -100] = self.tokenizer.pad_token_id
                        # safe_labels = labels.clone()
                        # safe_labels[safe_labels == -100] = self.tokenizer.pad_token_id
                        # decoded_input = self.tokenizer.decode(safe_inputs[0], skip_special_tokens=False)
                        # decoded_text = self.tokenizer.decode(safe_labels[0], skip_special_tokens=False)
                        # print(f"[DEBUG] first input:{decoded_input}")
                        # print(f"[DEBUG] first label:{decoded_text}")
                        # row_token_id = self.tokenizer.convert_tokens_to_ids("<ROW>")
                        # print(f"When first loaded: <ROW> count in input_ids = {(input_ids == row_token_id).sum().item()}")
                        # print(f"When first loaded: <ROW> count in labels = {(labels == row_token_id).sum().item()}")


                    # DEBUG: Debug check to catch dataset issues
                    vocab_size = self.model.get_output_embeddings().weight.size(0)
                    assert (inputs >= 0).all() and (inputs < vocab_size).all(), \
                        f"input_ids out of bounds (min {inputs.min().item()}, max {inputs.max().item()})"
                    assert ((labels == -100) | ((labels >= 0) & (labels < vocab_size))).all(), \
                        f"labels out of bounds (min {labels[labels != -100].min().item()}, max {labels[labels != -100].max().item()})"

                    outputs = self.model(input_ids=inputs, attention_mask=attention_mask,
                                         output_hidden_states=False)
                    logits = outputs.logits

                    if self.config.get("use_prompt") and self.config.get("instruct_text"):
                        num_instruction_tokens = self.instruction.size(1)
                        logits = logits[:, num_instruction_tokens:, :]
                        labels = labels[:, num_instruction_tokens:]

                    if self.config.get("use_prompt"):
                        # Skip first N tokens (warm-up window) for fair metric calculation
                        logits = logits[:, self.config.get("skip_N_tokens", 0):, :]

                    labels = labels[:, self.config.get("skip_N_tokens", 0):]
                    # if self.config['timedelta_cutoff'] is not None:
                    #     time_deltas = time_deltas[:, self.config.get("skip_N_tokens", 0):]

                    # # Keep only sequences with at least 1 token after skipping first N tokens
                    # # print(f"[DEBUG] Batch size before masking short sequences: {labels.size(0)}")
                    # valid_lengths = (labels != -100).sum(dim=1)
                    # keep_mask = valid_lengths >= 1  # at least 1 valid token after skipping
                    # if not keep_mask.any():
                    #     # print(f"[DEBUG] All sequences skipped in this batch, due to their length < skip_N_tokens (warm-up sequence)")
                    #     continue
                    # # print(f"[DEBUG] Retained {keep_mask.sum().item()} sequences out of {labels.size(0)}")
                    # logits = logits[keep_mask]
                    # labels = labels[keep_mask]
                    # if self.config['timedelta_cutoff'] is not None:
                    #     time_deltas = time_deltas[keep_mask]

                    ## If using field-based approach, apply logit masking to constrain model output to valid action tokens (e.g., [ACT_xxx])
                    if self.config.get('custom_tokenization') and self.config.get("use_prompt") and self.config.get("instruct_text"):
                        valid_act_token_ids = [id for tok, id in self.tokenizer.get_vocab().items() if tok.startswith('[ACT_')]
                        valid_act_token_ids.append(self.tokenizer.pad_token_id)
                        valid_act_token_ids = torch.tensor(valid_act_token_ids, device=logits.device)
                        vocab_size = logits.size(-1)
                        logit_mask = torch.full((vocab_size,), float('-inf'), device=logits.device)
                        logit_mask[valid_act_token_ids] = 0.0
                        logits = logits + logit_mask.view(1, 1, -1)
                        # DEBUGGING cross entropy unintentionally  being inf
                        unique_label_ids = labels[labels != -100].unique()
                        invalid_labels = [id.item() for id in unique_label_ids if id.item() not in valid_act_token_ids]
                        invalid_tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in invalid_labels]
                        assert len(
                            invalid_labels) == 0, f"Found true label(s) not in valid_act_token_ids:\nIDs: {invalid_labels}\nTokens: {invalid_tokens}"

                    log_probs = F.log_softmax(logits, dim=-1)

                    # Temporary safe indexing to replace -100 with valid token IDs in labels, so that gather() operation can run
                    safe_labels = labels.clone()
                    safe_labels[labels == -100] = self.tokenizer.pad_token_id #assign dummy index

                    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
                    token_ce = -token_log_probs.squeeze(0)  # shape: (seq_len - 1)

                    preds = torch.argmax(logits, dim=-1).squeeze(0)
                    labels_flat = labels.squeeze(0)


                    mask = (labels_flat != -100) & (labels_flat != self.tokenizer.pad_token_id) & (labels_flat != self.tokenizer.convert_tokens_to_ids("<ROW>")) \
                            & (labels_flat != self.tokenizer.convert_tokens_to_ids("<FIRST_ROW>")) \
                            & (labels_flat != self.tokenizer.convert_tokens_to_ids(['[TD_0]', '[TD_10]', '[TD_60]', '[TD_>60]']))
                    correct_count = (preds[mask] == labels_flat[mask]).sum().item()
                    total = mask.sum().item()

                    if (self.config.get("custom_tokenization") == False and self.config.get("use_delimiter")):
                        row_token_id = self.tokenizer.convert_tokens_to_ids("<ROW>")
                        # print(f"row_token_id: {row_token_id}")
                        # print([tok for tok in self.tokenizer.get_vocab().keys() if "ROW" in tok])
                        # print("Is ROW token in labels_flat?", (labels_flat == row_token_id).any().item())
                        # print(f"# row_token_id in labels_flat: {(labels_flat == row_token_id).sum()}")
                        mask = (labels_flat != -100) & (labels_flat != self.tokenizer.pad_token_id)
                        # Derive label chunks based on <ROW>
                        # print(f"labels_flat: {labels_flat}")
                        label_chunks = split_chunks_by_row(labels_flat[mask], row_token_id)
                        # print(f"label_chunks: {label_chunks}")
                        # Align predictions chunk-by-chunk using label chunk sizes
                        pred_chunks = []
                        cursor = 0
                        mask = (labels_flat != -100) & (labels_flat != self.tokenizer.pad_token_id) & (labels_flat != self.tokenizer.convert_tokens_to_ids("<ROW>"))
                        preds_masked = preds[mask]
                        if preds_masked.ndim == 2:
                            preds_flat = preds_masked.reshape(-1)  # Ensures 1D shape
                        else:
                            preds_flat = preds_masked

                        for chunk in label_chunks:
                            chunk_len = len(chunk)
                            pred_chunk = preds_flat[cursor:cursor + chunk_len]
                            pred_chunks.append(pred_chunk)
                            cursor += chunk_len
                        # print(f"[DEBUG] all chunk pairs are same length: {all(len(p) == len(l) for p, l in zip(pred_chunks, label_chunks))}")
                        # print(f"[DEBUG] preds_flat length == total length of label_chunks: {len(preds_flat) == sum(len(chunk) for chunk in label_chunks)}")
                        # print(f"pred_chunks: {pred_chunks}")
                        # print(f"label_chunks: {label_chunks}")

                        # Count fully correct chunks
                        all_matches = [
                            torch.equal(p_chunk, l_chunk)
                            for p_chunk, l_chunk in zip(pred_chunks, label_chunks)
                        ]
                        # print(f"all_matches: {all_matches}")
                        # correct_chunks = sum(all_matches)
                        # total_chunks = len(label_chunks)
                        # all_correct_chunks.append(correct_chunks)
                        # all_total_chunks.append(total_chunks)
                        all_correct_chunks.extend(all_matches)
                        # print(f"[DEBUG] correct_chunks: {correct_chunks}, total_chunks: {total_chunks}")

                        ################ Embedding similarity matching ###############
                        # Decode and normalize text per chunk
                        label_chunk_texts = [
                            preprocess_text(self.tokenizer.decode(chunk.tolist(), skip_special_tokens=True))
                            for chunk in label_chunks
                        ]
                        pred_chunk_texts = [
                            preprocess_text(self.tokenizer.decode(chunk.tolist(), skip_special_tokens=True))
                            for chunk in pred_chunks
                        ]

                        # Vectorize all predicted chunk texts in one shot, then L2-normalize
                        V = self.vectorizer.transform(pred_chunk_texts)  # (num_chunks, vocab)
                        V_l2 = normalize(V, norm="l2", axis=1, copy=True)  # unit norm rows

                        # Cosine similarities via dot product against pre-normalized valid-actions
                        # sims shape: (num_valid_actions, num_chunks)
                        sims = (self.A_valid_l2 @ V_l2.T).toarray()

                        k = 5
                        nn_top1_flags, nn_topk_flags = [], []

                        # for each chunk, rank valid actions by similarity
                        for j in range(sims.shape[1]):
                            s = sims[:, j]
                            ranked_idx = np.argsort(-s)  # descending
                            pred_top1 = self.valid_action_texts[ranked_idx[0]]
                            topk_set = {self.valid_action_texts[i] for i in ranked_idx[:k]}

                            # Compare against ground-truth action text
                            gold = label_chunk_texts[j]

                            nn_top1_flags.append(1 if gold == pred_top1 else 0)
                            nn_topk_flags.append(1 if gold in topk_set else 0)

                        # store for reporting
                        all_chunk_nn_top1.extend(nn_top1_flags)
                        all_chunk_nn_topk.extend(nn_topk_flags)
                        ####################################################################

                        mask = (labels_flat != -100) & (labels_flat != self.tokenizer.pad_token_id) & (
                                    labels_flat != self.tokenizer.convert_tokens_to_ids("<ROW>")) \
                               & (labels_flat != self.tokenizer.convert_tokens_to_ids("<FIRST_ROW>")) \
                               & (labels_flat != self.tokenizer.convert_tokens_to_ids(
                            ['[TD_0]', '[TD_10]', '[TD_60]', '[TD_>60]']))

                    # Compute top-k accuracy (k=5)
                    topk_preds = torch.topk(logits, k=5, dim=-1).indices  # shape: (1, seq_len-1, 5)
                    topk_correct = (topk_preds == labels.unsqueeze(-1)).any(dim=-1).squeeze(0)  # shape: (seq_len-1,)
                    # topk_correct_masked = topk_correct[mask].float().mean().item() # incorrectly averaging per-batch
                    topk_correct_masked = topk_correct[mask].sum().item()
                    all_topk_corrects.extend(topk_correct[mask].cpu().tolist())
                    all_topk_correct_count.append(topk_correct_masked)

                    # if len(all_preds) == 0:
                    #     self.debug_one_sample_prediction(input_ids.cpu(), labels.cpu(), preds.cpu(), topk_preds.cpu())

                    # Compute token-level entropy
                    probs = F.softmax(logits, dim=-1)  # shape: (1, seq_len-1, vocab_size)
                    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).squeeze(0)  # shape: (seq_len-1,)
                    # entropy_masked = entropy[mask].mean().item() # again incorrect batch-level averaging
                    # all_entropy.append(entropy_masked)
                    all_entropy.extend(entropy[mask].cpu().tolist()) # fixed version to store all token-level entropy

                    # Compute expected calibration error (ECE)
                    # ece = self.compute_ECE(logits, labels, mask) # incorrect since batch-averaging
                    # all_ece.append(ece)
                    confidences = F.softmax(logits, dim=-1).max(dim=-1).values.squeeze(0)
                    correct = (preds[mask] == labels_flat[mask]).float()
                    all_confidences.extend(confidences[mask].cpu().tolist())
                    all_corrects.extend(correct.cpu().tolist())

                    if self.config.get("custom_tokenization") == False and self.config.get("use_delimiter"):
                        ce_chunks = []
                        entropy_chunks = []
                        topk_chunk_accs = []
                        confidence_chunks = []
                        cursor = 0

                        for chunk in label_chunks:
                            chunk_len = len(chunk)

                            ce_chunk = np.array(token_ce[mask][cursor:cursor + chunk_len].cpu())
                            ce_chunks.append(np.mean(ce_chunk))

                            entropy_chunk = np.array(entropy[mask][cursor:cursor + chunk_len].cpu())
                            entropy_chunks.append(np.mean(entropy_chunk))

                            conf_chunk = np.array(confidences[mask][cursor:cursor + chunk_len].cpu())
                            confidence_chunks.append(np.mean(conf_chunk))

                            topk_chunk = np.array(topk_correct[mask][cursor:cursor + chunk_len].cpu())
                            topk_chunk_accs.append(np.mean(topk_chunk))

                            cursor += chunk_len

                        all_chunk_ce.extend(ce_chunks)
                        all_chunk_entropy.extend(entropy_chunks)
                        all_chunk_topk_acc.extend(topk_chunk_accs)
                        all_chunk_confidence.extend(confidence_chunks)



                    # # Embedding = last token's hidden state
                    # last_hidden = outputs.hidden_states[-1][:, -1, :]  # shape: (1, hidden_dim)

                    all_correct_count.append(correct_count)
                    all_labels.append(total)
                    # all_ce.append(token_ce.cpu().tolist())
                    # all_ce_values.append(token_ce[mask].mean().item())
                    all_ce_values.extend(token_ce[mask].cpu().tolist()) # fixed version to store all token-level prediction's CE
                    # all_ppls.append(torch.exp(token_ce[mask].mean()).item())
                    # all_embeddings.append(last_hidden.detach().cpu())

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # Optional, for multiprocess GPU use



        if self.config.get("custom_tokenization") == False and self.config.get("use_delimiter"):
            print(f"[DEBUG] Total # of predicted positions: {len(all_correct_chunks)}")
            # print(f"[DEBUG] all_correct_chunks: {all_correct_chunks}")
            # print(f"[DEBUG] all_chunk_ce: {all_chunk_ce}")
            # print(f"[DEBUG] all_chunk_entropy: {all_chunk_entropy}")
            # print(f"[DEBUG] all_chunk_confidence: {all_chunk_confidence}")
            # chunk_acc = sum(all_correct_chunks) / sum(all_total_chunks)
            chunk_acc = np.mean(all_correct_chunks)


            # Compute chunk-level CE, PPL, entropy, top-k acc
            avg_chunk_ce = np.mean(all_chunk_ce)
            # avg_chunk_ppl = np.mean([np.exp(c) for c in all_chunk_ce])
            avg_chunk_ppl = np.exp(avg_chunk_ce)
            avg_chunk_entropy = np.mean(all_chunk_entropy)
            avg_chunk_topk_acc = np.mean(all_chunk_topk_acc)
            try:
                global_ece = self.compute_ECE(all_chunk_confidence, all_correct_chunks)  # weighted sum; no need for SD or CI
            except Exception as e:
                print("ECE computation threw an error", e)
                global_ece = None

            nn_top1_mean = float(np.mean(all_chunk_nn_top1)) if len(all_chunk_nn_top1) else 0.0
            nn_topk_mean = float(np.mean(all_chunk_nn_topk)) if len(all_chunk_nn_topk) else 0.0
            print(f"[RESULT] Action-Level Top-1 Accuracy (TF-IDF mapped): {nn_top1_mean:.4f}")
            print(f"[RESULT] Action-Level Top-{k} Accuracy (TF-IDF mapped): {nn_topk_mean:.4f}")

            print(f"[RESULT] Field-level Chunk Accuracy: {chunk_acc:.8f}")
            print(f"[RESULT] Field-Level Chunk Top-5 Accuracy: {avg_chunk_topk_acc:.4f}")
            print(f"[RESULT] Field-Level Chunk Mean Cross-Entropy: {avg_chunk_ce:.4f} (Perplexity: {avg_chunk_ppl:.4f})")
            print(f"[RESULT] Field-Level Chunk Mean Entropy: {avg_chunk_entropy:.4f}")
            print(f"[RESULT] Field-Level Chunk ECE: {global_ece:.4f}")
            print(f"[RESULT] # top-1 acc metrics across all positions: {len(all_correct_chunks)}")
            print(f"[RESULT] # top-5 acc metrics across all positions: {len(all_chunk_topk_acc)}")
            print(f"[RESULT] # CE across all positions: {len(all_chunk_ce)}")
            print(f"[RESULT] # Entropy metrics across all positions: {len(all_chunk_entropy)}")
            print(f"[RESULT] # Confidence metrics across all positions: {len(all_chunk_confidence)}")

            if self.config["log_wandb"]:
                wandb.log({
                    "test/action_top1_acc": nn_top1_mean,
                    "test/action_topk_acc": nn_topk_mean,
                    "test/chunk_accuracy": chunk_acc,
                    "test/chunk_cross_entropy": avg_chunk_ce,
                    "test/chunk_perplexity": avg_chunk_ppl,
                    "test/chunk_entropy": avg_chunk_entropy,
                    "test/chunk_top5_accuracy": avg_chunk_topk_acc,
                    "test/chunk_ece": global_ece
                })
        else:
            print(f"[DEBUG] Total # of predicted positions: {sum(all_labels)}")
            # top1_acc = sum(all_correct_count) / sum(all_labels)
            top1_acc = np.mean(all_corrects)
            top1_std = np.std(all_corrects)

            # avg_ppl = np.mean(all_ppls)
            avg_ce = np.mean(all_ce_values)
            ce_std = np.std(all_ce_values)

            avg_ppl = np.exp(
                avg_ce)  # exponential of the average cross entropy; perplexity is fundamentally a seq-level metric that captures overall uncertainty. cannot compute SD

            # avg_topk_acc = np.mean(all_topk_correct) #incorrectly averaging batches
            # avg_topk_acc = sum(all_topk_correct_count) / sum(all_labels)
            avg_topk_acc = np.mean(all_topk_corrects)
            topk_std = np.std(all_topk_corrects)

            avg_entropy = np.mean(all_entropy)
            entropy_std = np.std(all_entropy)

            # avg_ece = np.mean(all_ece)
            try:
                global_ece = self.compute_ECE(all_confidences, all_corrects)  # weighted sum; no need for SD or CI
            except Exception as e:
                print("ECE computation threw an error", e)
                global_ece = None

            print(f"[RESULT] Avg Top-1 Accuracy (SD): {top1_acc:.8f} ({top1_std:.8f})")
            print(f"[RESULT] Avg Top-5 Accuracy (SD): {avg_topk_acc:.4f} ({topk_std:.4f})")
            print(f"[RESULT] Avg Cross-Entropy (SD): {avg_ce:.4f} ({ce_std:.4f})")
            print(f"[RESULT] Overall Perplexity: {avg_ppl:.4f}")
            print(f"[RESULT] Avg Entropy (SD): {avg_entropy:.4f} ({entropy_std:.4f})")
            print(f"[RESULT] Overall Expected Calibration Error (ECE): {global_ece:.4f}")
            print(f"[RESULT] # top-1 acc metrics across all positions: {len(all_corrects)}")
            print(f"[RESULT] # top-5 acc metrics across all positions: {len(all_topk_corrects)}")
            print(f"[RESULT] # CE across all positions: {len(all_ce_values)}")
            print(f"[RESULT] # Entropy metrics across all positions: {len(all_entropy)}")
            print(f"[RESULT] # Confidence metrics across all positions: {len(all_confidences)}")


            if self.config["log_wandb"]:
                wandb.log({
                    "test/accuracy": top1_acc,
                    "test/perplexity": avg_ppl,
                    "test/top5_accuracy": avg_topk_acc,
                    "test/entropy": avg_entropy,
                    "test/ece": global_ece
                })

        if save_dir:
            if self.config.get("custom_tokenization"):
                print(f"[SANITY CHECK] Sample label distribution: \n\t0 (non-error):{np.bincount(all_labels_binary)[0]}\n\t1 (error):{np.bincount(all_labels_binary)[1]}")
                if self.config["use_prompt"]:
                    if self.config.get("instruct_text") is None:
                        # torch.save(torch.cat(all_embeddings, dim=0), os.path.join(save_dir, "seq_embeddings-prompt.pt"))
                        # np.save(os.path.join(save_dir, "seq_error_labels-prompt.npy"), np.array(all_labels_binary))
                        # np.save(os.path.join(save_dir, "cross_entropy-prompt.npy"), np.array(all_ce, dtype=object))
                        results_csv_path = os.path.join(save_dir, "llm_inference_results-prompt.csv")
                        result_col_name = self.config['HF_repo_name']+'-prompt'
                        # Save each token-level metric across all tokens in the test set in deterministic order
                        torch.save(torch.tensor(all_corrects, dtype=torch.long), os.path.join(save_dir, "llm-prompt_tokens_correct.pt"))
                        torch.save(torch.tensor(all_topk_corrects, dtype=torch.long), os.path.join(save_dir, "llm-prompt_tokens_top5_correct.pt"))
                        torch.save(torch.tensor(all_ce_values), os.path.join(save_dir, "llm-prompt_tokens_cross_entropy.pt"))
                        torch.save(torch.tensor(all_entropy), os.path.join(save_dir, "llm-prompt_tokens_entropy.pt"))
                        torch.save(torch.tensor(all_confidences), os.path.join(save_dir, "llm-prompt_tokens_confidence.pt"))
                        print(f"[INFO] Token-level metrics cached to {save_dir}")
                    else:
                        torch.save(torch.cat(all_embeddings, dim=0), os.path.join(save_dir, "seq_embeddings-instruction_prompt.pt"))
                        np.save(os.path.join(save_dir, "seq_error_labels-instruction_prompt.npy"), np.array(all_labels_binary))
                        np.save(os.path.join(save_dir, "cross_entropy-instruction_prompt.npy"), np.array(all_ce, dtype=object))
                        results_csv_path = os.path.join(save_dir, "inference_results-instruction_prompt.csv")
                        result_col_name = self.config['HF_repo_name'] + '-instruction_prompt'
                else:
                    # torch.save(torch.cat(all_embeddings, dim=0), os.path.join(save_dir, "seq_embeddings.pt"))
                    # np.save(os.path.join(save_dir, "seq_error_labels.npy"), np.array(all_labels_binary))
                    # np.save(os.path.join(save_dir, "cross_entropy.npy"), np.array(all_ce, dtype=object))
                    results_csv_path = os.path.join(save_dir, "llm_inference_results.csv")
                    result_col_name = self.config['HF_repo_name']
                    # Save each token-level metric across all tokens in the test set in deterministic order
                    torch.save(torch.tensor(all_corrects, dtype=torch.long), os.path.join(save_dir, "llm_tokens_correct.pt"))
                    torch.save(torch.tensor(all_topk_corrects, dtype=torch.long), os.path.join(save_dir, "llm_tokens_top5_correct.pt"))
                    torch.save(torch.tensor(all_ce_values), os.path.join(save_dir, "llm_tokens_cross_entropy.pt"))
                    torch.save(torch.tensor(all_entropy), os.path.join(save_dir, "llm_tokens_entropy.pt"))
                    torch.save(torch.tensor(all_confidences), os.path.join(save_dir, "llm_tokens_confidence.pt"))
                    print(f"[INFO] Token-level metrics cached to {save_dir}")
                results_series = pd.Series({
                    "Avg Top-1 Accuracy (SD)": f"{top1_acc} ({top1_std})",
                    "Avg Top-5 Accuracy (SD)": f"{avg_topk_acc} ({topk_std})",
                    "Avg Cross-Entropy (SD)": f"{avg_ce} ({ce_std})",
                    "Overall Perplexity": f"{avg_ppl}",
                    "Avg Entropy (SD)": f"{avg_entropy} ({entropy_std})",
                    "Overall ECE": f"{global_ece}"
                }, name=result_col_name)
            else:
                if self.config["use_prompt"]:
                    results_csv_path = os.path.join(save_dir, "llm_inference_results-prompt.csv")
                    result_col_name = self.config['HF_repo_name'] + '-prompt'
                    # Save each token-level metric across all tokens in the test set in deterministic order
                    torch.save(torch.tensor(all_correct_chunks, dtype=torch.long),
                               os.path.join(save_dir, "llm-prompt_tokens_correct.pt"))
                    torch.save(torch.tensor(all_chunk_topk_acc, dtype=torch.long),
                               os.path.join(save_dir, "llm-prompt_tokens_top5_correct.pt"))
                    torch.save(torch.tensor(all_chunk_ce),
                               os.path.join(save_dir, "llm-prompt_tokens_cross_entropy.pt"))
                    torch.save(torch.tensor(all_chunk_entropy), os.path.join(save_dir, "llm-prompt_tokens_entropy.pt"))
                    torch.save(torch.tensor(all_chunk_confidence), os.path.join(save_dir, "llm-prompt_tokens_confidence.pt"))
                    torch.save(torch.tensor(all_chunk_nn_top1), os.path.join(save_dir, "llm-prompt_tokens_nn_correct.pt"))
                    torch.save(torch.tensor(all_chunk_nn_topk), os.path.join(save_dir, "llm-prompt_tokens_nn_top5_correct.pt"))
                    print(f"[INFO] Chunk-level metrics cached to {save_dir}")
                else:
                    results_csv_path = os.path.join(save_dir, "llm_inference_results.csv")
                    result_col_name = self.config['HF_repo_name']
                    # Save each token-level metric across all tokens in the test set in deterministic order
                    torch.save(torch.tensor(all_correct_chunks, dtype=torch.long),
                               os.path.join(save_dir, "llm_tokens_correct.pt"))
                    torch.save(torch.tensor(all_chunk_topk_acc, dtype=torch.long),
                               os.path.join(save_dir, "llm_tokens_top5_correct.pt"))
                    torch.save(torch.tensor(all_chunk_ce), os.path.join(save_dir, "llm_tokens_cross_entropy.pt"))
                    torch.save(torch.tensor(all_chunk_entropy), os.path.join(save_dir, "llm_tokens_entropy.pt"))
                    torch.save(torch.tensor(all_chunk_confidence), os.path.join(save_dir, "llm_tokens_confidence.pt"))
                    torch.save(torch.tensor(all_chunk_nn_top1), os.path.join(save_dir, "llm_tokens_nn_correct.pt"))
                    torch.save(torch.tensor(all_chunk_nn_topk), os.path.join(save_dir, "llm_tokens_nn_top5_correct.pt"))
                    print(f"[INFO] Chunk-level metrics cached to {save_dir}")


                results_series = pd.Series({
                    "Action-level NN Accuracy": nn_top1_mean,
                    "Action-level NN Top-5 Accuracy": nn_topk_mean,
                    "Field-level Accuracy (chunked)": chunk_acc,
                    "Field-level Top-5 Accuracy (chunked; relaxed)": avg_chunk_topk_acc,
                    "Field-level Cross-Entropy (chunked)": avg_chunk_ce,
                    "Field-level Perplexity (chunked)": avg_chunk_ppl,
                    "Field-level Entropy (chunked)": avg_chunk_entropy,
                    "Field-level ECE": global_ece
                }, name=result_col_name)



            results_df = results_series.to_frame()
            results_df.to_csv(results_csv_path)
            print(f"[INFO] Evaluation results exported to {results_csv_path}")



        # return top1_acc, avg_ppl, all_ce, all_embeddings, avg_topk_acc, avg_entropy

    

    def compute_ECE(self, all_confs, all_corrects, n_bins=10):
        all_confs = np.array(all_confs)
        all_corrects = np.array(all_corrects)

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

    def debug_one_sample_prediction(self, input_ids, labels, preds, topk_preds):
        # tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.access_config["HF_access_token"])
        tokenizer = self.tokenizer

        decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        decoded_labels = tokenizer.decode(labels[0], skip_special_tokens=False)
        # decoded_preds = tokenizer.decode(preds[0], skip_special_tokens=False)
        # Custom decoding block for predictions, with prompt separation if use_prompt is enabled
        if self.config.get("use_prompt", False):
            # Include prompt portion
            skip_N_tokens = min(self.config.get("skip_N_tokens", 15), preds[0].shape[0])
            prompt_text = tokenizer.decode(input_ids[0][:skip_N_tokens], skip_special_tokens=False)
            pred_text = tokenizer.decode(preds[0][skip_N_tokens:], skip_special_tokens=False)
            decoded_preds = f"[PROMPT]\n{prompt_text}\n[PREDICTION]\n{pred_text}"
        else:
            decoded_preds = tokenizer.decode(preds[0], skip_special_tokens=False)

        print("\n--- DEBUG: Single Sequence Example ---")
        print("[Input text (context)]")
        print(decoded_input)
        print("\n[Ground-truth sequence (labels)]")
        print(decoded_labels)
        print("\n[Model prediction]")
        print(decoded_preds)

        # for i in range(min(len(labels[0]) - 1, 10)):
        # Updated per-token print loop to account for prompting
        skip_N_tokens = min(self.config.get("skip_N_tokens", 15), input_ids.shape[1]) if self.config.get("use_prompt", False) else 0
        for i in range(skip_N_tokens, min(len(labels[0]) - 1, skip_N_tokens + 10)):
            input_tok = tokenizer.decode([input_ids[0][i]])
            label_tok = tokenizer.decode([labels[0][i]])
            # print(f"Step {i + 1}: Input: {input_tok} → Label (next token): {label_tok}")
            print(f"Step {i + 1 - skip_N_tokens}: Input: {input_tok} → Label (next token): {label_tok}")

        print("\n[Top-5 predictions for first few tokens]")
        # topk_tokens = topk_preds[0][:10]  # only first 10 tokens
        topk_tokens = topk_preds[0][skip_N_tokens:skip_N_tokens + 10]  # only first 10 tokens after prompt, if any
        for i, token_ids in enumerate(topk_tokens):
            tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
            # ground_truth_token = tokenizer.convert_ids_to_tokens([labels[0][i].item()])[0]
            ground_truth_token = tokenizer.convert_ids_to_tokens([labels[0][skip_N_tokens + i].item()])[0]
            print(f"Step {i + 1}: {tokens} | Ground truth: {ground_truth_token}")


from transformers import TrainerCallback

class MinimalImprovementStoppingCallback(TrainerCallback):
    def __init__(self, threshold=0.005):
        self.threshold = threshold
        self.best_score = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_score = metrics.get("eval_loss", None)
        if current_score is None:
            return

        if self.best_score is None:
            self.best_score = current_score
        else:
            improvement = self.best_score - current_score
            if improvement >= self.threshold:
                self.best_score = current_score
            else:
                print(f"[Early Stop Triggered] Improvement {improvement:.6f} < threshold {self.threshold}")
                control.should_training_stop = True
class EvalCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} ended, triggering evaluation...")
        control.should_evaluate = True  # Keep this as it informs the trainer
        # Manually trigger evaluation
        trainer = kwargs.get('trainer', None)
        if trainer:
            trainer.evaluate()


class LoggingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):


        logging.info(f"Epoch {state.epoch} has ended")


        # Check if log_history contains the required information
        if len(state.log_history) > 0:
            print("There are logged history")
            last_log = state.log_history[-1]
            # Log epoch training loss if available
            if 'loss' in last_log:
                print("Train loss is in last_log")
                logging.info(f"Epoch {state.epoch}: Training Loss: {last_log['loss']:.4f}")
            # Log epoch validation loss if available
            if 'eval_loss' in last_log:
                print("Val loss is in last_log")
                logging.info(f"Epoch {state.epoch}: Validation Loss: {last_log['eval_loss']:.4f}")
        else:
            print("There is no logged history")


        # Log GPU memory usage
        gc.collect()
        torch.cuda.empty_cache()
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        logging.info(f"Peak GPU memory usage after epoch {state.epoch}: {peak_memory:.2f} GB")
        torch.cuda.reset_peak_memory_stats()  # Reset stats to monitor peak memory for the next epoch
