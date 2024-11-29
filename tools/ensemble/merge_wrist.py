import pandas as pd
import numpy as np
from tqdm import tqdm

# Load the two CSV files
df1 = pd.read_csv('./output/top.csv')
df2 = pd.read_csv('output_9746.csv')

# 상수 정의
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

# Define the classes to take from df1 and df2
df1_classes = set(CLASSES[i] for i in range(19, 27))  # 'Trapezium'부터 'Pisiform'까지
df2_classes = set(CLASSES[i] for i in range(29)) - df1_classes  # 나머지 클래스들

# Create a copy of df1 to store the combined results
combined_df = df1.copy()

# Iterate through each row in df1
for i in tqdm(range(len(df1))):
    image_name = df1.at[i, 'image_name']
    class_label = df1.at[i, 'class']
    
    if class_label in df1_classes:
        # Use RLE from df1 for specified classes
        combined_df.at[i, 'rle'] = df1.at[i, 'rle']
    else:
        # Find the corresponding row in df2 for the same image_name and class
        matching_row_df2 = df2[(df2['image_name'] == image_name) & (df2['class'] == class_label)]
        
        # If there's a match in df2, replace rle from df2
        if not matching_row_df2.empty:
            combined_df.at[i, 'rle'] = matching_row_df2.iloc[0]['rle']

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("combined_result.csv", index=False)
