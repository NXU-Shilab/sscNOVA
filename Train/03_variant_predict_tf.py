import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


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

# Create an argument parser
parser = argparse.ArgumentParser(description='Load data, make predictions, and save results.')
parser.add_argument('file_name', type=str, help='Path to the data file')
parser.add_argument('model_name', type=str, help='Name of the model')
parser.add_argument('feature', type=int, help='Number of features')
args = parser.parse_args()

# Load data
data = pd.read_csv(args.file_name, sep='\t')
feature = args.feature
X_test = data.iloc[:, 5:]

fold_models = []
predictions = []

# Add columns to X based on the number of features and set their values to 0
if (feature == 141):
    X_test['New_Column1'] = 0
    X_test['New_Column2'] = 0
    X_test['New_Column3'] = 0
    input_dim = feature + 3
elif (feature == 40):
    X_test['New_Column1'] = 0
    X_test['New_Column2'] = 0
    input_dim = feature + 2
else:
    input_dim = feature

seed = 42
torch.manual_seed(seed)
input_dim = input_dim     
output_dim = 1      
num_epochs = 50     
learn_rate = 1e-5   
batch_size = 32     
hidden_dim = 32     
num_heads = 6       
num_layers = 2      
dropout = 0.05  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device) 

# Load model
n_splits = 5
for fold in range(n_splits):
    model_path = f'{args.model_name}_model_fold_{fold+1}.pth'
    loaded_model = TransformerClassifier(input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout) 
    loaded_model.load_state_dict(torch.load(model_path))  
    loaded_model.eval()  
    fold_models.append(loaded_model)

# Make predictions using the loaded model
for fold, loaded_model in enumerate(fold_models):
    with torch.no_grad():
        outputs = loaded_model(X_test)
        y_prob_test = torch.sigmoid(outputs.view(-1, 1)).float().view(-1).cpu()
        column_name = f'fold_{fold+1}_prob'  
        data[column_name] = y_prob_test 

# Save results to file
output_file_name = args.file_name.replace('.csv', f'_{args.model_name}_probability.csv')
data.to_csv(output_file_name, index=False, sep='\t')
print(f'Predicted probabilities saved to {output_file_name}')
