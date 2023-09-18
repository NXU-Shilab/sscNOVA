import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from torch.utils.data import DataLoader


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
    target = target.cpu().numpy()
    predicted = predicted.cpu().numpy()
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

# Compute confusion matrix
def compute_confusion_matrix(y_true, y_pred):
    y_true = y_true.view(-1).cpu().numpy()  
    y_pred = y_pred.view(-1).cpu().numpy()  
    y_pred = (y_pred > 0.5).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tn, fp, fn, tp

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

# Define Transformer model architecture
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim          
        self.output_dim = output_dim         
        self.hidden_dim = hidden_dim         
        self.num_layers = num_layers         
        self.num_heads = num_heads           
        self.dropout = dropout             
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, activation="gelu", dropout=dropout).to(device)  
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).to(device)  
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, activation="gelu", dropout=dropout).to(device)  
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers).to(device)  
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)     
        self.fc2 = nn.Linear(hidden_dim, output_dim).to(device)    
        self.dropout = nn.Dropout(dropout).to(device)        
        
    def forward(self, x):
        out = self.transformer_encoder(x)  
        out = self.transformer_decoder(out, out)  
        out = self.fc1(out)  
        out = self.fc2(out)  
        out = self.dropout(out)  
        return out

print('The program is executing...')

# Create an argument parser
parser = argparse.ArgumentParser(description='Create and train a neural network model.')
parser.add_argument('file_name', type=str, help='Path to the data file')
parser.add_argument('file2_name', type=str, help='Path to the data file2')
parser.add_argument('feature', type=int, help='Number of features')
args = parser.parse_args()

# Load data
data = pd.read_csv(args.file_name, sep='\t')
pseudo_data = pd.read_csv(args.file2_name, sep='\t')
feature = args.feature
decimal = 3      

# Random sample and dataset split
data = data.sample(frac=1, random_state=42)
pseudo_data = pseudo_data.sample(frac=1, random_state=42)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=21)
train_data = pd.concat([train_data, pseudo_data], ignore_index=True)
kf = KFold(n_splits=5, shuffle=True, random_state=31)

# X is feature matrix, y is target vector
X = train_data.iloc[:, 5:-1]
y = train_data.iloc[:,-1]
X_test = test_data.iloc[:, 5:-1]
y_test = test_data.iloc[:,-1]

fold_results_validate = []
fold_results_test = []

best_metrics = (0.0, 0.0, 0.0)
best_fold_index = -1 
best_fold_results = None

# Add columns to X based on the number of features and set their values to 0
if (feature == 141):
    X['New_Column1'] = 0
    X['New_Column2'] = 0
    X['New_Column3'] = 0
    X_test['New_Column1'] = 0
    X_test['New_Column2'] = 0
    X_test['New_Column3'] = 0
    feature = feature + 3
elif (feature == 40):
    X['New_Column1'] = 0
    X['New_Column2'] = 0
    X_test['New_Column1'] = 0
    X_test['New_Column2'] = 0
    feature = feature + 2
else:
    feature = feature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device) 
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Model train and evaluation
for fold, (train_index, validate_index) in enumerate(kf.split(X)):
    cross = ['fold-1', 'fold-2', 'fold-3', 'fold-4', 'fold-5']
    
    X_train, X_validate = X.iloc[train_index], X.iloc[validate_index]
    y_train, y_validate = y.iloc[train_index], y.iloc[validate_index]
    
    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    X_validate = torch.tensor(X_validate.values, dtype=torch.float32).to(device) 
    y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    y_validate = torch.tensor(y_validate.values, dtype=torch.float32)
    
    seed = 42
    torch.manual_seed(seed)
    input_dim = feature     
    output_dim = 1      
    num_epochs = 50     
    learn_rate = 1e-5   
    batch_size = 32     
    hidden_dim = 32     
    num_heads = 6       
    num_layers = 2      
    dropout = 0.05      

    # Model train and save model weights
    model = TransformerClassifier(input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout)
    model = model.to(device)        
    criterion = nn.BCELoss().to(device)                            
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)       
    
    for epoch in range(num_epochs):
        dataloader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
        for i, (inputs, labels) in enumerate(dataloader):
            labels = labels.unsqueeze(1)            
            optimizer.zero_grad()                   
            outputs = model(inputs)                 
            outputs = torch.sigmoid(outputs)        
            loss = criterion(outputs, labels)       
            optimizer.zero_grad()                   
            loss.backward()                         
            optimizer.step()                        

    torch.save(model, f'tf_pseudo_{feature}_model_fold_{fold+1}.pth')
    
    # validating dataset evaluation
    model.eval()  
    with torch.no_grad():
        outputs = model(X_validate).to(device)  
        y_prob_validate = torch.sigmoid(outputs.view(-1,1)).float().view(-1).cpu()
        threshold_validate = Find_Optimal_Cutoff(y_validate, y_prob_validate)
        threshold_validate = torch.Tensor(threshold_validate)
        y_pred_validate = (y_prob_validate > threshold_validate).to(torch.int)
        
        tn, fp, fn, tp = compute_confusion_matrix(y_validate, y_pred_validate)
        accuracy, precision, recall, specificity, f1_score, auroc, auprc = compute_evaluation_indicator(tn, fp, fn, tp, y_validate, y_prob_validate)
        
        fold_result_validate = {'cross': cross[f], 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'Specificity': specificity, 'F1-Score': f1_score, 'AUC': auroc, 'AUPRC': auprc}
        fold_results_validate.append(fold_result_validate)

    # testing dataset evaluation
    model.eval()  
    with torch.no_grad():
        outputs = model(X_test).to(device)  
        y_prob_test = torch.sigmoid(outputs.view(-1,1)).float().view(-1).cpu()
        threshold_test = Find_Optimal_Cutoff(y_test, y_prob_test)
        threshold_test = torch.Tensor(threshold_test)
        y_pred_test = (y_prob_test > threshold_test).to(torch.int)

        tn, fp, fn, tp = compute_confusion_matrix(y_test, y_prob_test)
        accuracy, precision, recall, specificity, f1_score, auroc, auprc = compute_evaluation_indicator(tn, fp, fn, tp, y_test, y_prob_test)
        
        fold_result_test = {'cross': cross[f], 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'Specificity': specificity, 'F1-Score': f1_score, 'AUC': auroc, 'AUPRC': auprc}
        fold_results_test.append(fold_result_test)

    fold_results_output_test = pd.DataFrame({
        'CHROM': test_data.iloc[:, 0],
        'POS': test_data.iloc[:, 1],
        'RSID': test_data.iloc[:, 2],
        'REF': test_data.iloc[:, 3],
        'ALT': test_data.iloc[:, 4], 
        'PredictedProbability': y_prob_test.flatten(),
        'PredictedLabel': y_pred_test.flatten(),
        'TrueLabel': y_test
    })

    # Calculate five-fold maximum
    best_fold_index_test, best_fold_results_test = best_fold_metrics(f, auroc, auprc, recall, fold_results_output_test)
    print("Test_data fold {}: the index of the test_data with the highest performance is fold {}".format(f+1, best_fold_index_test+1))
    best_fold_results_test.to_csv('cnn_pseudo_{}_best_fold_results_test.csv'.format(feature), index=False, sep='\t')

    f = f + 1

# Save results to file
fold_result_validate_df = pd.concat([pd.DataFrame([fold_result_validate]) for fold_result_validate in fold_results_validate])
fold_result_validate_df.to_csv('tf_pseudo_{}_fold_results_validate.csv'.format(feature), header=True, index=False)

fold_result_test_df = pd.concat([pd.DataFrame([fold_result_test]) for fold_result_test in fold_results_test])
fold_result_test_df.to_csv('tf_pseudo_{}_fold_results_test.csv'.format(feature), header=True, index=False)

print('The program execution is completed.')
