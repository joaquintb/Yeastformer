import os
import pandas as pd
import dask.dataframe as dd

# Directory containing .pcl files
pcl_directory = '/home/logs/jtorresb/Geneformer/yeast/yeast_data/test_pcls'

# Prepare an empty DataFrame to store the master matrix (this will hold all gene data)
master_df = pd.DataFrame()

# Iterate over each .pcl file in the directory
for file_name in os.listdir(pcl_directory):
    if file_name.endswith(".pcl"):
        file_path = os.path.join(pcl_directory, file_name)
        try:
            # Load the .pcl file using Dask for memory efficiency (chunksize can be adjusted)
            ddf = dd.read_csv(file_path, sep="\t", usecols=lambda col: col not in ['GWEIGHT', 'NAME', 'IDENTIFIER', 'Description'])
            
            # Compute the Dask DataFrame (process it in-memory)
            df = ddf.compute()

            # Ensure columns are of type float32 to optimize memory usage
            df = df.astype('float32')

            # Set 'NAME' as index to use gene names for merging (if not already set)
            if 'NAME' in df.columns:
                df.set_index('NAME', inplace=True)

            # Merge data into the master DataFrame, aligning by gene name (index)
            if master_df.empty:
                master_df = df
            else:
                master_df = master_df.join(df, how='outer', lsuffix=f'_{file_name}')

            print(f"Processed {file_name} - shape: {df.shape}")

        except Exception as e:
            print(f"Error processing file '{file_name}': {e}")

# Save the master matrix to a new file
output_path = '/home/logs/jtorresb/Geneformer/yeast/yeast_data/output/test_matrix.csv'
master_df.to_csv(output_path, sep="\t")
print(f"Master matrix created and saved to {output_path}")
