from datasets import load_from_disk

dataset = load_from_disk("/home/logs/jtorresb/Geneformer/Genecorpus/example_input_files/gene_classification/dosage_sensitive_tfs/gc-30M_sample50k.dataset")

# Split the dataset into 45k for training and 5k for testing
split = dataset.train_test_split(test_size=5000)

train_dataset = split['train']
test_dataset = split['test']

# Save the train and test splits as new datasets
train_dataset.save_to_disk('train_dataset')
test_dataset.save_to_disk('test_dataset')