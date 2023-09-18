import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

data = pd.read_excel('compare_newTest_tools.xlsx')

indicators = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'AUC', 'AUPRC']
data_matrix = data.loc[:, indicators].values

data.set_index('Model', inplace=True)

annot_color = {'color':'black'}

plt.figure(figsize=(15, 6))
sns.heatmap(data_matrix, cmap='coolwarm', annot=True, fmt='.3f', xticklabels=indicators, yticklabels=data.index, 
            linewidths=0.1, center=0.5, vmin=0.0, vmax=1.0, annot_kws=annot_color)

plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)

plt.xlabel('Evaluation Indicators', labelpad=10, fontsize=12)
plt.ylabel('Models', labelpad=10, fontsize=12)

with PdfPages('compare_newTest_tools.pdf') as pdf:
    pdf.savefig()
    plt.close()

