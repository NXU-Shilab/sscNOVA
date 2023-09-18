import argparse
import pandas as pd
import joblib


# Create an argument parser
parser = argparse.ArgumentParser(description='Load data, make predictions, and save results.')
parser.add_argument('file_name', type=str, help='Path to the data file')
parser.add_argument('model_name', type=str, help='Name of the model')
args = parser.parse_args()

# Load data
data = pd.read_csv(args.file_name, sep='\t')
X_test = data.iloc[:, 5:]

fold_models = []
predictions = []

# Load model
n_splits = 5
for fold in range(n_splits):
    model_path = f'{args.model_name}_model_fold_{fold+1}.joblib'  
    loaded_model = joblib.load(model_path)  
    fold_models.append(loaded_model)

# Make predictions using the loaded model
for fold, loaded_model in enumerate(fold_models):
    y_prob_test = loaded_model.predict_proba(X_test)[:, 1]
    predictions.append(y_prob_test)

for fold in range(n_splits):
    column_name = f'fold_{fold+1}_prob'
    data[column_name] = predictions[fold]

# Save results to file
output_file_name = args.file_name.replace('.csv', f'_{args.model_name}_probability.csv')
data.to_csv(output_file_name, index=False, sep='\t')
print(f'Predicted probabilities saved to {output_file_name}')
