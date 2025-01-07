from datasets import load_from_disk

dataset = load_from_disk("/home/logs/jtorresb/Geneformer/toy_datasets/toy_test.dataset")

print(dataset)

# View feature columns
print("Columns:", dataset.column_names)

# Inspect first 5 rows
print("Sample Rows:", dataset[:5])

# Check dataset size
print("Number of Rows:", len(dataset))