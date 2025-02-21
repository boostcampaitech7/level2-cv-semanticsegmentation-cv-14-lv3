import pandas as pd

# Define the classes to sort by
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

# Read the CSV files
file_a = pd.read_csv('/data/output/output1.csv')  # Replace with your first file path
file_b = pd.read_csv('/data/output/output2.csv')  # Replace with your second file path

# Select rows from A.csv where class is "Trapezoid" or "Pisiform"
condition_a = file_a[file_a['class'].isin(['Trapezoid', 'Pisiform'])]

# Check if 'Trapezoid' and 'Pisiform' values exist in A.csv
if condition_a.empty:  # If no rows with 'Trapezoid' or 'Pisiform' in A.csv
    # If A.csv does not have these values, use B.csv for 'Trapezoid' and 'Pisiform'
    condition_a = file_b[file_b['class'].isin(['Trapezoid', 'Pisiform'])]

# Select rows from B.csv for the rest of the classes (excluding 'Trapezoid' and 'Pisiform')
condition_b = file_b[~file_b['class'].isin(['Trapezoid', 'Pisiform'])]

# Combine the selected data from both files
merged_df = pd.concat([condition_a, condition_b], ignore_index=True)

# Convert class_name into a categorical type with the specified order
merged_df['class'] = pd.Categorical(merged_df['class'], categories=CLASSES, ordered=True)

# Sort by image_name first and class_name second
merged_df_sorted = merged_df.sort_values(by=['image_name', 'class'])

# Save the sorted DataFrame to a new CSV file
merged_df_sorted.to_csv('2class_merged_output.csv', index=False)