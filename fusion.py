import pandas as pd
import numpy as np
from tqdm import tqdm

# Load the two CSV files
df1 = pd.read_csv('upp_hr_CLAHE_pre.csv')
df2 = pd.read_csv('hr_sweep1.csv')

# Define the classes to take from df1 and df2
df1_classes = set(range(19, 27))
df2_classes = set(range(29)) - df1_classes

# Create a copy of df1 to store the combined results
combined_df = df1.copy()

# Iterate through each row in chunks of 29 classes
for i in tqdm(range(0, len(df1), 29)):
    for j in range(29):  # Iterate over each class within the chunk
        if j in df1_classes:
            # Use RLE from df1 for specified classes
            combined_df.at[i + j, 'rle'] = df1.at[i + j, 'rle']
        else:
            # Use RLE from df2 for other classes
            combined_df.at[i + j, 'rle'] = df2.at[i + j, 'rle']

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("combined_result.csv", index=False)