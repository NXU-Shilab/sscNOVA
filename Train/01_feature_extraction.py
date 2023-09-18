import argparse
import pandas as pd

# Create an argument parser
parser = argparse.ArgumentParser(description='Extract columns from a DataFrame based on the content of another file.')
parser.add_argument('first_file', type=str, help='Path to the first file')
parser.add_argument('second_file', type=str, help='Path to the second file')
parser.add_argument('output_file', type=str, help='Path to the output file')
parser.add_argument('label', type=int, choices=[0, 1], help='Label to add (0 or 1)')
args = parser.parse_args()

# Load data
first_df = pd.read_csv(args.first_file, header=None, names=['content'])
second_df = pd.read_csv(args.second_file, sep='\t')

# Generate the list of column names to extract
columns_to_extract = first_df['content'].tolist()

# Extract the corresponding columns (chrom pos name ref alt)
extracted_columns = second_df.iloc[:, 2:7]
extracted_columns = pd.concat([extracted_columns, second_df[columns_to_extract]], axis=1)

# Add the Label column
extracted_columns['Label'] = args.label

# Save results to file
extracted_columns.to_csv(args.output_file, index=False, sep='\t')
