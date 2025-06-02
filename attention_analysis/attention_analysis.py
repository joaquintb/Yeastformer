import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForMaskedLM
from datasets import load_from_disk
import random
import os
from tqdm import tqdm

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

model_path = "/home/logs/jtorresb/Geneformer/yeast/pretraining/models/250304_125959_yeastformer_L3_emb384_SL512_E20_B8_LR0.00115_LSlinear_WU144_Oadamw_torch/models"
token_dict_path = "/home/logs/jtorresb/Geneformer/yeast/yeast_data/output/yeast_token_dict.pkl"

# Load gene token dictionary
with open(token_dict_path, "rb") as fp:
    token_dictionary = pickle.load(fp)

# Load model
model = BertForMaskedLM.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

# Invert the token dictionary (if it maps gene_name -> token_id)
id_to_gene = {v: k for k, v in token_dictionary.items()}

# -------------------------------
# Load dataset
# -------------------------------
dataset = load_from_disk("/home/logs/jtorresb/Geneformer/yeast/yeast_data/output/yeast_master_matrix_sgd.dataset")
# In this example, we use the full dataset (no train-test split)

# -------------------------------
# Collate Function for Padding
# -------------------------------
def collate_fn(samples):
    """
    Pads 'input_ids' sequences to the longest length in the batch.
    Uses 0 as the padding token and generates an attention mask.
    """
    input_ids_list = [torch.tensor(s["input_ids"], dtype=torch.long) for s in samples]

    # Pad sequences
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)

    # Attention mask (1 for real tokens, 0 for padding)
    attention_mask = (padded_input_ids != 0).long()

    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask
    }

# -------------------------------
# DataLoader with Collate Function
# -------------------------------
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

# -------------------------------
# Processing Loop
# -------------------------------
gene_attention_scores = {}
rank_weights = [5, 4, 3, 2, 1]

for batch in tqdm(dataloader, desc="Processing batches"):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)

    last_layer_attentions = outputs.attentions[-1]  # shape: [batch, heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = last_layer_attentions.shape
    device_mask = torch.eye(seq_len, dtype=torch.bool, device=input_ids.device)

    for b in range(batch_size):
        sample_input_ids = input_ids[b].cpu().tolist()
        sample_genes = [id_to_gene.get(t_id, None) for t_id in sample_input_ids]

        for head in range(num_heads):
            att_matrix = last_layer_attentions[b, head]
            att_matrix_masked = att_matrix.masked_fill(device_mask, float('-inf'))

            topk = torch.topk(att_matrix_masked, k=5, dim=1)
            top_indices = topk.indices

            for i, source_gene in enumerate(sample_genes):
                if source_gene is None or source_gene == "PAD":
                    continue
                for rank, target_pos in enumerate(top_indices[i]):
                    target_index = target_pos.item()
                    target_gene = sample_genes[target_index]
                    if target_gene is None or target_gene == "PAD":
                        continue
                    weight = rank_weights[rank]
                    # Safely update nested dictionary
                    inner_dict = gene_attention_scores.setdefault(source_gene, {})
                    inner_dict[target_gene] = inner_dict.get(target_gene, 0) + weight

# -------------------------------
# Save top 5 most important genes for each gene to a text file
# -------------------------------
output_file = "top5_genes.txt"

with open(output_file, "w") as f:
    for source_gene, targets in gene_attention_scores.items():
        # Sort target genes by aggregated score (highest first)
        sorted_targets = sorted(targets.items(), key=lambda x: x[1], reverse=True)
        # Select the top 5 targets
        top5 = sorted_targets[:5]
        # Write the source gene and its top 5 targets to the file
        f.write(f"{source_gene}:\n")
        for rank, (target_gene, score) in enumerate(top5, start=1):
            f.write(f"    {rank}. {target_gene} (score: {score})\n")
        f.write("\n")

print(f"Top 5 genes for each gene have been written to '{output_file}'.")