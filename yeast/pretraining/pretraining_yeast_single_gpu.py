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

# Set random seeds
seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Set timezone and directories
timezone = pytz.timezone("Europe/Madrid")
rootdir = "/home/logs/jtorresb/Geneformer/yeast/pretraining"

# Model parameters
model_type = "bert"
max_input_size = 512  # Sequence length
num_layers = 4  # Avoid overfitting in smaller dataset
num_attn_heads = 4  # Attention heads
num_embed_dim = 256  # Embedding dimensions
intermed_size = num_embed_dim * 2
activ_fn = "gelu"
initializer_range = 0.02
layer_norm_eps = 1e-12
attention_probs_dropout_prob = 0.1  # More dropout to prevent overfitting in smaller dataset
hidden_dropout_prob = 0.1  

# Training parameters
num_examples = 11889  # Num of .arrow rows
geneformer_batch_size = 8  # Adjusted for memory constraints
epochs = 15
optimizer = "adamw_torch"  # Uses AdamW with bias correction

# Extra training parameters
gradient_accumulation_steps = 4  # Simulates batch size of 32
fp16 = True  # Enable mixed precision for memory savings
learning_rate = 0.002
warmup_steps = 50
weight_decay = 0.01
lr_scheduler_type = "cosine"

# Output directories
current_date = datetime.datetime.now(tz=timezone)
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
run_name = f"{datestamp}_yeastformer_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}_E{epochs}_B{geneformer_batch_size}_LR{learning_rate}_LS{lr_scheduler_type}_WU{warmup_steps}_O{optimizer}"
training_output_dir = f"{rootdir}/models/{run_name}/"
logging_dir = f"{rootdir}/runs/{run_name}/"
model_output_dir = os.path.join(training_output_dir, "models/")

os.makedirs(training_output_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)

# Load gene token dictionary
with open("/home/logs/jtorresb/Geneformer/yeast/yeast_data/output/yeast_token_dict.pkl", "rb") as fp:
    token_dictionary = pickle.load(fp)

# Define model configuration
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

training_args = TrainingArguments(
    learning_rate=learning_rate,
    do_train=True,
    do_eval=False,
    group_by_length=True,
    length_column_name="length",
    disable_tqdm=False,
    lr_scheduler_type=lr_scheduler_type,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    per_device_train_batch_size=geneformer_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=epochs,
    save_strategy="steps",
    save_steps=500,
    logging_steps=100,
    output_dir=training_output_dir,
    logging_dir=logging_dir,
    fp16=fp16,
)

# Define trainer
trainer = GeneformerPretrainer(
    model=model,
    args=training_args,
    train_dataset=load_from_disk("/home/logs/jtorresb/Geneformer/yeast/yeast_data/output/yeast_master_matrix_sgd.dataset"),
    example_lengths_file="/home/logs/jtorresb/Geneformer/yeast/yeast_data/output/yeast_lengths.pkl",
    token_dictionary=token_dictionary,
)

trainer.train()