import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib


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

# Compute confusion matrix
def compute_confusion_matrix(y_true, y_pred):
    part = y_pred ^ y_true             
    pcount = np.bincount(part)         
    tp_list = list(y_pred & y_true)    
    fp_list = list(y_pred & ~y_true)   
    TP = tp_list.count(1)              
    FP = fp_list.count(1)              
    TN = pcount[0] - TP                
    FN = pcount[1] - FP               
    return TN, FP, FN, TP

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

print('The program is executing...')

# Create an argument parser
parser = argparse.ArgumentParser(description='Create and train a neural network model.')
parser.add_argument('file_name', type=str, help='Path to the data file')
parser.add_argument('feature', type=int, help='Number of features')
args = parser.parse_args()

# Load data
data = pd.read_csv(args.file_name, sep='\t')
feature = args.feature
decimal = 3

# Random sample and dataset split
data = data.sample(frac=1, random_state=42)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=21)
kf = KFold(n_splits=5, shuffle=True, random_state=31)

# X is feature matrix, y is target vector
X = train_data.iloc[:, 5:-1]
y = train_data.iloc[:,-1]
X_test = test_data.iloc[:, 5:-1]
y_test = test_data.iloc[:,-1]

fold_results_validate = []
fold_results_test = []
fold_results_hgmd = []

best_metrics = (0.0, 0.0, 0.0)
best_fold_index = -1 
best_fold_results = None

# Model train and evaluation
for fold, (train_index, validate_index) in enumerate(kf.split(X)):
    cross = ['fold-1', 'fold-2', 'fold-3', 'fold-4', 'fold-5']

    X_train, X_validate = X.iloc[train_index], X.iloc[validate_index]
    y_train, y_validate = y.iloc[train_index], y.iloc[validate_index]

    # Model train and save model weights
    model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    model.fit(X_train, y_train)
    joblib.dump(model, f'rf_{feature}_model_fold_{fold+1}.joblib')

    # validating dataset evaluation
    y_prob_validate = model.predict_proba(X_validate)[:, 1]
    threshold_validate = Find_Optimal_Cutoff(y_validate, y_prob_validate)
    y_pred_validate = (y_prob_validate > threshold_validate).astype(int)

    tn, fp, fn, tp = compute_confusion_matrix(y_validate, y_pred_validate)
    accuracy, precision, recall, specificity, f1_score, auroc, auprc = compute_evaluation_indicator(tn, fp, fn, tp, y_validate, y_prob_validate)
                    
    fold_result_validate = {'cross': cross[fold], 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'Specificity': specificity, 'F1-Score': f1_score, 'AUC': auroc, 'AUPRC': auprc}
    fold_results_validate.append(fold_result_validate)

    # testing dataset evaluation
    y_prob_test = model.predict_proba(X_test)[:, 1]
    threshold_test = Find_Optimal_Cutoff(y_test, y_prob_test)
    y_pred_test = (y_prob_test > threshold_test).astype(int)

    tn, fp, fn, tp = compute_confusion_matrix(y_test, y_pred_test)
    accuracy, precision, recall, specificity, f1_score, auroc, auprc = compute_evaluation_indicator(tn, fp, fn, tp, y_test, y_prob_test)

    fold_result_test = {'cross': cross[fold], 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'Specificity': specificity, 'F1-Score': f1_score, 'AUC': auroc, 'AUPRC': auprc}
    fold_results_test.append(fold_result_test)

    fold_results_output_test = pd.DataFrame({
        'CHROM': test_data.iloc[:, 0],
        'POS': test_data.iloc[:, 1],
        'RSID': test_data.iloc[:, 2],
        'REF': test_data.iloc[:, 3],
        'ALT': test_data.iloc[:, 4], 
        'PredictedProbability': y_prob_test,
        'PredictedLabel': y_pred_test,
        'TrueLabel': y_test
    })

    # Calculate five-fold maximum
    best_fold_index_test, best_fold_results_test = best_fold_metrics(fold, auroc, auprc, recall, fold_results_output_test)
    print("Test_data fold {}: the index of the test_data with the highest performance is fold {}".format(fold+1, best_fold_index_test+1))
    best_fold_results_test.to_csv('rf_{}_best_fold_results_test.csv'.format(feature), index=False, sep='\t')

# Save results to file
fold_result_validate_df = pd.concat([pd.DataFrame([fold_result_validate]) for fold_result_validate in fold_results_validate])
fold_result_validate_df.to_csv('rf_{}_fold_results_validate.csv'.format(feature), header=True, index=False)

fold_result_test_df = pd.concat([pd.DataFrame([fold_result_test]) for fold_result_test in fold_results_test])
fold_result_test_df.to_csv('rf_{}_fold_results_test.csv'.format(feature), header=True, index=False)

print('The program execution is completed.')
