import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


data = pd.read_excel('compare_newTest.xlsx')

indicators = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'AUC', 'AUPRC']
data_matrix = data.loc[:, indicators].values

data.set_index('Model', inplace=True)

max_auc_index = data['AUC'].idxmax()

auc_column_index = indicators.index('AUC')
auprc_column_index = indicators.index('AUPRC')

plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(data_matrix, cmap='coolwarm', annot=True, fmt='.3f', xticklabels=indicators, yticklabels=data.index, 
            linewidths=0.1, center=0.6, vmin=0.4, vmax=0.7, annot_kws={'color':'black'})

heatmap.add_patch(plt.Rectangle((auc_column_index, data.index.get_loc(max_auc_index)), 2, 1, fill=False, edgecolor='black', lw=1))

plt.xlabel('Evaluation Indicator', labelpad=10, fontsize=12)
plt.ylabel('Model Name', labelpad=10, fontsize=12)

plt.tight_layout()

with PdfPages('compare_newTest.pdf') as pdf:
    pdf.savefig()
    plt.close()

