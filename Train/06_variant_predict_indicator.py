import argparse
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc


# Calculate and update the best metrics, fold index and results
def best_fold_metrics(fold_index, auroc, auprc, recall, fold_results):
    global best_metrics, best_fold_index, best_fold_results
    current_metrics = (auroc, auprc, recall)
    
    if current_metrics > best_metrics:
        best_metrics = current_metrics
        best_fold_index = fold_index
        best_fold_results = fold_results
    
    return best_fold_index, best_fold_results

# Calculate the optimal threshold
def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

# Compute evaluation indicators
def compute_evaluation_indicator(tn, fp, fn, tp, y_true, y_prob):
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    accuracy = round(accuracy, decimal)
    precision = tp / (tp + fp)
    precision = round(precision, decimal)
    recall = tp / (tp + fn)
    recall = round(recall, decimal)
    specificity = tn / (tn + fp)
    specificity = round(specificity, decimal)  
    f1_score = (2 * precision * recall) / (precision + recall)
    f1_score = round(f1_score, decimal)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = auc(fpr, tpr)
    auroc = round(auroc, decimal)
    precision_, recall_ , _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall_, precision_)
    auprc = round(auprc, decimal)

    return accuracy, precision, recall, specificity, f1_score, auroc, auprc

# Create an argument parser
parser = argparse.ArgumentParser(description='Load data, make predictions, and save results.')
parser.add_argument('file_name', type=str, help='Path to the data file')
parser.add_argument('model_name', type=str, help='Name of the model')
parser.add_argument('feature', type=int, help='Number of features')
args = parser.parse_args()

# Load data
data = pd.read_csv(args.file_name, sep='\t')
feature = args.feature
decimal = 3

# X is feature matrix, y is target vector
X_test = data.iloc[:, 5:-1]
y_test = data.iloc[:,-1]

fold_results = []
fold_models = []
predictions = []

best_metrics = (0.0, 0.0, 0.0)
best_fold_index = -1 
best_fold_results = None

# Load model
n_splits = 5
for fold in range(n_splits):
    model_path = f'{args.model_name}_model_fold_{fold}.h5'
    loaded_model = load_model(model_path)
    fold_models.append(loaded_model)

# Load model parameters and evaluation
for fold, loaded_model in enumerate(fold_models):
    cross = ['fold-1', 'fold-2', 'fold-3', 'fold-4', 'fold-5']

    X_test_fold = np.array(X_test)  
    X_test_fold = np.reshape(X_test_fold, (X_test_fold.shape[0], X_test_fold.shape[1], 1))
    
    # testing dataset evaluation
    y_prob_test = loaded_model.predict(X_test_fold)
    threshold_test = Find_Optimal_Cutoff(y_test, y_prob_test)
    y_pred_test = (y_prob_test > threshold_test).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    accuracy, precision, recall, specificity, f1_score, auroc, auprc = compute_evaluation_indicator(tn, fp, fn, tp, y_test, y_prob_test)

    fold_result_test = {'cross': cross[fold], 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'Specificity': specificity, 'F1-Score': f1_score, 'AUC': auroc, 'AUPRC': auprc}
    fold_results.append(fold_result_test)

    fold_results_output = pd.DataFrame({
        'CHROM': data.iloc[:, 0],
        'POS': data.iloc[:, 1],
        'RSID': data.iloc[:, 2],
        'REF': data.iloc[:, 3],
        'ALT': data.iloc[:, 4], 
        'PredictedProbability': y_prob_test.flatten(),
        'PredictedLabel': y_pred_test.flatten(),
        'TrueLabel': y_test
    })

    # Calculate five-fold maximum
    best_fold_index, best_fold_result = best_fold_metrics(fold, auroc, auprc, recall, fold_results_output)
    print(f'HGMD_ClinVar_data fold {fold+1}: the index of the hgmd_clinvar_data with the highest performance is fold {best_fold_index+1}')
    best_fold_result.to_csv(f'{args.model_name}_{feature}_best_fold_result_hgmd.csv', index=False, sep='\t')

    predictions.append(y_prob_test)

for fold in range(n_splits):
    column_name = f'fold_{fold+1}_prob'
    data[column_name] = predictions[fold].flatten()

# Save results to file
output_file_name = args.file_name.replace('.csv', f'_{args.model_name}_probability.csv')
data.to_csv(output_file_name, index=False, sep='\t')
print(f'Predicted probabilities saved to {output_file_name}')

fold_result_df = pd.concat([pd.DataFrame([fold_result]) for fold_result in fold_results])
fold_result_df.to_csv(f'{args.model_name}_{feature}_fold_result_hgmd.csv', header=True, index=False)
