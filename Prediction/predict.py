import argparse
import pandas as pd
import numpy as np
from keras.models import load_model


# Create an argument parser
parser = argparse.ArgumentParser(description='Load data, make predictions, and save results.')
parser.add_argument('file_name', type=str, help='Path to the data file')
args = parser.parse_args()

# Load data
data = pd.read_csv(args.file_name, sep='\t')
X_test = data.iloc[:, 5:]

# Load model
model_path = f'sscNOVA.h5'
loaded_model = load_model(model_path)

# Make predictions using the loaded model
X_test_fold = np.array(X_test) 
X_test_fold = np.reshape(X_test_fold, (X_test_fold.shape[0], X_test_fold.shape[1], 1))
y_prob_test = loaded_model.predict(X_test_fold)

# Create a DataFrame with only the desired columns
output_data = data.iloc[:, :5]
output_data['probability'] = y_prob_test

# Save results to file
output_file_name = args.file_name.replace('.csv', f'.csv')
output_data.to_csv(output_file_name, index=False, sep='\t')
print(f'Predicted probabilities saved to {output_file_name}')
