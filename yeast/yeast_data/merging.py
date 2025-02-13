import os
import pandas as pd

# Directory containing .pcl files
pcl_directory = '/home/logs/jtorresb/Geneformer/yeast/yeast_data/all_pcls'

# Prepare an empty DataFrame to store the master matrix
master_df = pd.DataFrame()

# Dictionary to track the count of each column name
column_count = {}

# Iterate over each .pcl file in the directory
for file_count, file_name in enumerate(os.listdir(pcl_directory)):
    print("Processing file", file_count + 1, "of", len(os.listdir(pcl_directory)))
    print(master_df.shape)
    print(len(column_count))
    print("---------------------")
    if file_name.endswith(".pcl"):
        file_path = os.path.join(pcl_directory, file_name)
        try:
            # Load the .pcl file
            df = pd.read_csv(file_path, sep="\t", index_col=0)

            # Filter out only the experimental columns (exclude 'GWEIGHT', 'NAME', 'IDENTIFIER', 'Description', etc.)
            experiment_columns = [col for col in df.columns if col not in ['GWEIGHT', 'NAME', 'IDENTIFIER', 'Description']]

            # Keep only the rows and columns of interest (experiment columns)
            df = df[experiment_columns]

            # Track and rename duplicates
            for col in df.columns:
                if col in column_count:
                    column_count[col] += 1
                    new_col_name = f"{col}_{column_count[col]}"
                    df.rename(columns={col: new_col_name}, inplace=True)
                else:
                    column_count[col] = 1

            # Merge this dataframe with the master dataframe (join by gene names/rows)
            if master_df.empty:
                master_df = df
            else:
                master_df = master_df.join(df, how='outer')  # 'outer' join to keep all genes

        except Exception as e:
            print(f"Error processing file '{file_name}': {e}")

# Save the master matrix to a new file
master_df.to_csv('/home/logs/jtorresb/Geneformer/yeast/yeast_data/output/unnormalized_yeast_master_matrix_sgd.csv', sep="\t")
print("Master matrix created and saved.")