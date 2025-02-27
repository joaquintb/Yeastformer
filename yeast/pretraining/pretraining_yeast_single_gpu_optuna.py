import datetime
import os
import pickle
import random
import optuna

import numpy as np
import pytz
import torch
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, TrainingArguments, TrainerCallback
from geneformer import GeneformerPretrainer

# -------------------------------
# Global setup: Set random seeds and load dataset/token dictionary once
# -------------------------------
seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

timezone = pytz.timezone("Europe/Madrid")
rootdir = "/home/logs/jtorresb/Geneformer/yeast/pretraining"

# -------------------------------
# Load the gene token dictionary
# -------------------------------
with open("/home/logs/jtorresb/Geneformer/yeast/yeast_data/output/yeast_token_dict.pkl", "rb") as fp:
    token_dictionary = pickle.load(fp)

# -------------------------------
# Load dataset and split into train/validation
# -------------------------------
dataset = load_from_disk("/home/logs/jtorresb/Geneformer/yeast/yeast_data/output/yeast_master_matrix_sgd.dataset")
dataset_split = dataset.train_test_split(test_size=0.05, seed=seed_val)

# IMPORTANT:
# Re-index each split so that they run from 0 to len(split)-1.
train_dataset = dataset_split['train'].select(range(len(dataset_split['train'])))
val_dataset   = dataset_split['test'].select(range(len(dataset_split['test'])))

# -------------------------------
# Re-map the lengths file for the training split
# -------------------------------
with open("/home/logs/jtorresb/Geneformer/yeast/yeast_data/output/yeast_lengths.pkl", "rb") as fp:
    original_lengths = pickle.load(fp)

if hasattr(dataset_split['train'], "_indices"):
    train_old_indices_raw = dataset_split['train']._indices.to_pylist()
    train_old_indices = []
    for idx in train_old_indices_raw:
        if isinstance(idx, dict):
            train_old_indices.append(list(idx.values())[0])
        else:
            train_old_indices.append(idx)
else:
    train_old_indices = list(range(len(dataset_split['train'])))

new_train_lengths = [original_lengths[old_idx] for old_idx in train_old_indices]
new_train_lengths_file = os.path.join(rootdir, "train_lengths_reindexed.pkl")
with open(new_train_lengths_file, "wb") as f:
    pickle.dump(new_train_lengths, f)

# -------------------------------
# Define a custom Optuna pruning callback for the Trainer
# -------------------------------
class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial, metric_name):
        self.trial = trial
        self.metric_name = metric_name

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.metric_name in metrics:
            score = metrics[self.metric_name]
            self.trial.report(score, step=state.global_step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return control

# -------------------------------
# Objective function for Optuna
# -------------------------------
def objective(trial):
    # ----- Hyperparameter suggestions -----
    # Optimizer and scheduler
    max_lr = trial.suggest_loguniform("max_lr", 5e-4, 5e-3)
    warmup_steps = trial.suggest_int("warmup_steps", 50, 500)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 0.1)
    lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts"])
    
    num_layers = trial.suggest_int("num_layers", 2, 6)
    num_embed_dim = trial.suggest_categorical("num_embed_dim", [256, 384, 512])

    if num_embed_dim == 256:
        num_attn_heads = trial.suggest_categorical("num_attn_heads_256", [2, 4, 8])
    elif num_embed_dim == 384:
        num_attn_heads = trial.suggest_categorical("num_attn_heads_384", [2, 3, 4, 6, 8, 12])
    elif num_embed_dim == 512:
        num_attn_heads = trial.suggest_categorical("num_attn_heads_512", [4, 8, 16])

    # Feed-forward layer multiplier (for intermediate size)
    ffn_multiplier = trial.suggest_int("ffn_multiplier", 2, 4)
    
    # Regularization (dropout rates)
    attention_probs_dropout_prob = trial.suggest_float("attention_dropout", 0.1, 0.3, step=0.05)
    hidden_dropout_prob = trial.suggest_float("hidden_dropout", 0.1, 0.3, step=0.05)

    # Other fixed model parameters
    model_type = "bert"
    max_input_size = 512  # Sequence length
    activ_fn = "gelu"
    initializer_range = 0.02
    layer_norm_eps = 1e-12

    # Training parameters
    geneformer_batch_size = 8   # Adjusted for memory constraints
    epochs = 12
    optimizer = "adamw_torch"   # AdamW with bias correction
    gradient_accumulation_steps = 4  # Effective batch size = 32
    fp16 = True  # Mixed precision training

    # -------------------------------
    # Output directories (unique per trial)
    # -------------------------------
    current_date = datetime.datetime.now(tz=timezone)
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
    run_name = (f"{datestamp}_yeastformer_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}_"
                f"E{epochs}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_scheduler_type}_WU{warmup_steps}_O{optimizer}")
    training_output_dir = os.path.join(rootdir, "models", run_name)
    logging_dir = os.path.join(rootdir, "runs", run_name)
    model_output_dir = os.path.join(training_output_dir, "final_model")

    os.makedirs(training_output_dir, exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)

    # -------------------------------
    # Define model configuration with suggested hyperparameters
    # -------------------------------
    config = BertConfig(
        hidden_size=num_embed_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attn_heads,
        intermediate_size=num_embed_dim * ffn_multiplier,
        hidden_act=activ_fn,
        max_position_embeddings=max_input_size,
        initializer_range=initializer_range,
        layer_norm_eps=layer_norm_eps,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        hidden_dropout_prob=hidden_dropout_prob,
        model_type=model_type,
        pad_token_id=token_dictionary.get("<pad>"),
        vocab_size=len(token_dictionary),
        output_attentions=True,
    )

    model = BertForMaskedLM(config).train()
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))

    # -------------------------------
    # Define training arguments (with evaluation enabled for pruning)
    # -------------------------------
    training_args = TrainingArguments(
        learning_rate=max_lr,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=50,
        group_by_length=True,
        length_column_name="length",
        disable_tqdm=False,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        per_device_train_batch_size=geneformer_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        logging_steps=50,
        output_dir=training_output_dir,
        logging_dir=logging_dir,
        fp16=fp16,
        save_strategy="no",
    )

    # -------------------------------
    # Define the trainer using the re-indexed training dataset and the val dataset,
    # and add the pruning callback.
    # -------------------------------
    trainer = GeneformerPretrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        example_lengths_file=new_train_lengths_file,
        token_dictionary=token_dictionary,
    )
    trainer.add_callback(OptunaPruningCallback(trial, metric_name="eval_loss"))

    # Train and evaluate
    trainer.train()
    eval_metrics = trainer.evaluate()

    final_loss = eval_metrics.get("eval_loss")
    if final_loss is None:
        raise ValueError("Evaluation loss not found in metrics.")
    return final_loss

# -------------------------------
# Run Optuna optimization with pruning
# -------------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best trial:", study.best_trial.params)