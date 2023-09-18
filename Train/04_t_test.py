import argparse
import numpy as np
import pandas as pd
import warnings
from scipy.stats import ttest_1samp

# Create an argument parser
parser = argparse.ArgumentParser(description='Load data, make predictions, and save results.')
parser.add_argument('file_name', type=str, help='Path to the data file')
parser.add_argument('upper_threshold', type=float, help='Upper threshold for positive prediction')
parser.add_argument('lower_threshold', type=float, help='Lower threshold for negative prediction')
args = parser.parse_args()

# Load data
data = pd.read_csv(args.file_name, sep='\t')

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Precision loss occurred in moment calculation")

alpha = 0.05
positive_count = 0
negative_count = 0
abandon_count = 0
predictions = []

# Perform a t-test and make predictions
for _, row in data.iterrows():
    values = row.values[-5:].astype(float) 
    t_statistic, p_value = ttest_1samp(values, 0.5)
    
    if np.any(values > args.upper_threshold) and p_value < alpha:
        positive_count += 1
        predictions.append(1)
    elif np.any(values < args.lower_threshold) and p_value < alpha:
        negative_count += 1
        predictions.append(0)
    else:
        abandon_count += 1
        predictions.append(-1)

# Create a DataFrame containing only labels 1 or 0
filtered_data = data.copy()
filtered_data['Label'] = predictions
filtered_data = filtered_data[filtered_data['Label'].isin([0, 1])]

print("Number of positive:", positive_count)
print("Number of negative:", negative_count)
print("Number of abandon:", abandon_count)

# Save results to file
filtered_data.drop(['fold_1_prob', 'fold_2_prob', 'fold_3_prob', 'fold_4_prob', 'fold_5_prob'], axis=1, inplace=True)
output_file = args.file_name.replace('.csv', f'_label_{args.upper_threshold}_{args.lower_threshold}.csv')
filtered_data.to_csv(output_file, index=False, sep='\t')
print(f"Results saved to: {output_file}")
