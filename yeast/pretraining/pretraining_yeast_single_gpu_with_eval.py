import datetime
import os
import pickle
import random

import numpy as np
import pytz
import torch
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, TrainingArguments
from geneformer import GeneformerPretrainer

from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

# -------------------------------
# Plotting eval and train loss in the same graph
# -------------------------------
class CustomTensorBoardCallback(TrainerCallback):
    def __init__(self, tb_writer=None):
        self.tb_writer = tb_writer if tb_writer else SummaryWriter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log both train and eval loss under a unified name."""
        if logs is not None:
            if "loss" in logs:  # Training loss
                self.tb_writer.add_scalar("Loss/train", logs["loss"], state.global_step)
            if "eval_loss" in logs:  # Evaluation loss
                self.tb_writer.add_scalar("Loss/eval", logs["eval_loss"], state.global_step)
            self.tb_writer.flush()

# -------------------------------
# Set random seeds and directories
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
# Model parameters
# -------------------------------
model_type = "bert"
max_input_size = 512  # Sequence length
num_layers = 4        # Fewer layers to avoid overfitting on a smaller dataset
num_attn_heads = 4    # Attention heads
num_embed_dim = 256   # Embedding dimensions
intermed_size = num_embed_dim * 2
activ_fn = "gelu"
initializer_range = 0.02
layer_norm_eps = 1e-12
attention_probs_dropout_prob = 0.1  # More dropout for smaller dataset
hidden_dropout_prob = 0.1  

# -------------------------------
# Training parameters
# -------------------------------
num_examples = 11889  # Total examples in the full dataset
geneformer_batch_size = 8  # Adjusted for memory constraints
epochs = 1
optimizer = "adamw_torch"  # AdamW with bias correction

# Extra training parameters
gradient_accumulation_steps = 4  # Simulate a batch size of 32
fp16 = True                    # Mixed precision for memory savings
learning_rate = 0.0016
warmup_steps = 50
weight_decay = 0.07
lr_scheduler_type = "cosine" 

# -------------------------------
# Output directories setup
# -------------------------------
current_date = datetime.datetime.now(tz=timezone)
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
run_name = (f"{datestamp}_yeastformer_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}"
            f"_E{epochs}_B{geneformer_batch_size}_LR{learning_rate}_LS{lr_scheduler_type}_WU{warmup_steps}_O{optimizer}")
training_output_dir = os.path.join(rootdir, "models", run_name)
logging_dir = os.path.join(rootdir, "runs", run_name)
model_output_dir = os.path.join(training_output_dir, "models")

os.makedirs(training_output_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)

# -------------------------------
# Load the gene token dictionary
# -------------------------------
with open("/home/logs/jtorresb/Geneformer/yeast/yeast_data/output/yeast_token_dict.pkl", "rb") as fp:
    token_dictionary = pickle.load(fp)

# -------------------------------
# Load dataset and split into train/validation
# -------------------------------
dataset = load_from_disk("/home/logs/jtorresb/Geneformer/yeast/yeast_data/output/yeast_master_matrix_sgd.dataset")
dataset_split = dataset.train_test_split(test_size=0.1, seed=seed_val)

# IMPORTANT:
# Hugging Face’s train_test_split preserves the original (non-contiguous) indices.
# Re-index each split so that they run from 0 to len(split)-1.
train_dataset = dataset_split['train'].select(range(len(dataset_split['train'])))
val_dataset   = dataset_split['test'].select(range(len(dataset_split['test'])))

# -------------------------------
# Re-map the lengths file for the training split
# -------------------------------
# Load the original lengths list (each position corresponds to an example in the full dataset)
with open("/home/logs/jtorresb/Geneformer/yeast/yeast_data/output/yeast_lengths.pkl", "rb") as fp:
    original_lengths = pickle.load(fp)

# Get the original indices from the training split.
if hasattr(dataset_split['train'], "_indices"):
    # Convert the pyarrow ChunkedArray to a Python list
    train_old_indices_raw = dataset_split['train']._indices.to_pylist()
    # If elements are dicts, extract the integer value.
    train_old_indices = []
    for idx in train_old_indices_raw:
        if isinstance(idx, dict):
            # Assuming the dict contains a single key-value pair, extract the value.
            # You might need to adjust this if your dict structure is different.
            train_old_indices.append(list(idx.values())[0])
        else:
            train_old_indices.append(idx)
else:
    train_old_indices = list(range(len(dataset_split['train'])))

# Create a new lengths list for the training dataset.
new_train_lengths = [original_lengths[old_idx] for old_idx in train_old_indices]

# # Save the new lengths list to a file so that the trainer can load it.
new_train_lengths_file = os.path.join(rootdir, "train_lengths_reindexed.pkl")
with open(new_train_lengths_file, "wb") as f:
    pickle.dump(new_train_lengths, f)

# -------------------------------
# Define model configuration
# -------------------------------
config = BertConfig(
    hidden_size=num_embed_dim,
    num_hidden_layers=num_layers,
    initializer_range=initializer_range,
    layer_norm_eps=layer_norm_eps,
    attention_probs_dropout_prob=attention_probs_dropout_prob,
    hidden_dropout_prob=hidden_dropout_prob,
    intermediate_size=intermed_size,
    hidden_act=activ_fn,
    max_position_embeddings=max_input_size,
    model_type=model_type,
    num_attention_heads=num_attn_heads,
    pad_token_id=token_dictionary.get("<pad>"),
    vocab_size=len(token_dictionary),
    output_attentions=True
)

model = BertForMaskedLM(config).train()
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))

# -------------------------------
# Define training arguments (with evaluation enabled)
# -------------------------------
training_args = TrainingArguments(
    learning_rate=learning_rate,
    do_train=True,
    do_eval=True,                    # Enable evaluation
    evaluation_strategy="steps",     # Evaluate every X steps
    eval_steps=100,                  # Adjust as needed
    group_by_length=True,
    length_column_name="length",
    disable_tqdm=False,
    lr_scheduler_type=lr_scheduler_type,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    per_device_train_batch_size=geneformer_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=epochs,
    save_strategy="no",              # No checkpoints during training
    logging_steps=250,
    output_dir=training_output_dir,
    logging_dir=logging_dir,
    fp16=fp16,
    report_to="tensorboard",
    # log_level="info"
)

# -------------------------------
# Define the trainer
# -------------------------------
trainer = GeneformerPretrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Pass the validation split here
    example_lengths_file=new_train_lengths_file,  # Use the re-mapped lengths list file
    token_dictionary=token_dictionary,
    callbacks=[CustomTensorBoardCallback()]
)

# -------------------------------
# Train the model and save it
# -------------------------------
trainer.train()
trainer.save_model(model_output_dir)