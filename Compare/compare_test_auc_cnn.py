from sklearn.metrics import precision_recall_curve, roc_curve, auc
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd


cnn_test_40 = pd.read_csv("./cnn_test_40.csv", sep='\t')
cnn_test_150 = pd.read_csv("./cnn_test_150.csv", sep='\t')
cnn_test_141= pd.read_csv("./cnn_test_141.csv", sep='\t')

cnn_pseudo_test_40 = pd.read_csv("./cnn_pseudo_test_40.csv", sep='\t')
cnn_pseudo_test_150 = pd.read_csv("./cnn_pseudo_test_150.csv", sep='\t')
cnn_pseudo_test_141= pd.read_csv("./cnn_pseudo_test_141.csv", sep='\t')

cnn_40_prob = cnn_test_40.iloc[:,:-1]
cnn_40_test = cnn_test_40.iloc[:,-1]
cnn_150_prob = cnn_test_150.iloc[:,:-1]
cnn_150_test = cnn_test_150.iloc[:,-1]
cnn_141_prob = cnn_test_141.iloc[:,:-1]
cnn_141_test = cnn_test_141.iloc[:,-1]

cnn_40_pseudo_prob = cnn_pseudo_test_40.iloc[:,:-1]
cnn_40_pseudo_test = cnn_pseudo_test_40.iloc[:,-1]
cnn_150_pseudo_prob = cnn_pseudo_test_150.iloc[:,:-1]
cnn_150_pseudo_test = cnn_pseudo_test_150.iloc[:,-1]
cnn_141_pseudo_prob = cnn_pseudo_test_141.iloc[:,:-1]
cnn_141_pseudo_test = cnn_pseudo_test_141.iloc[:,-1]

prob = [cnn_40_prob, cnn_150_prob, cnn_141_prob, cnn_40_pseudo_prob, cnn_150_pseudo_prob, cnn_141_pseudo_prob]
test = [cnn_40_test, cnn_150_test, cnn_141_test, cnn_40_pseudo_test, cnn_150_pseudo_test, cnn_141_pseudo_test]

model = ['cnn_40', 'cnn_150', 'cnn_141', 'cnn_pseudo_40', 'cnn_pseudo_150', 'sscNOVA']

pdf_pages = PdfPages('compare_test_auc_cnn.pdf')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))

colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'maroon', 'gold', 'black', 'gray', 'brown', 'crimson']

auc_results = []
auprc_results = []
for f, (i, j) in enumerate(zip(test, prob)):

    fpr, tpr, _ = roc_curve(i, j)
    auroc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(i, j)
    auprc = auc(recall, precision)

    auc_results.append((model[f], auroc))
    auprc_results.append((model[f], auprc))

sorted_auc_results = sorted(auc_results, key=lambda x: x[1], reverse=True)

sorted_auc_results_with_colors = zip(colors, sorted_auc_results)

for color, (model_name, auc_value) in sorted_auc_results_with_colors:
    f = model.index(model_name)

    i, j = test[f], prob[f]

    fpr, tpr, _ = roc_curve(i, j)

    ax[0].plot(fpr, tpr, lw=2, alpha=0.3, label='{} (AUC = {:.3f})'.format(model_name, auc_value), color=color)

sorted_auprc_results = sorted(auprc_results, key=lambda x: x[1], reverse=True)

sorted_auprc_results_with_colors = zip(colors, sorted_auprc_results)

for color, (model_name, auprc_value) in sorted_auprc_results_with_colors:
    f = model.index(model_name)

    i, j = test[f], prob[f]

    precision, recall, _ = precision_recall_curve(i, j)
    auprc = auc(recall, precision)

    ax[1].plot(recall, precision, lw=2, alpha=0.3, label='{} (AUPRC = {:.3f})'.format(model_name, auprc_value), color=color)

ax[0].plot([0,1], [0,1], linestyle='--', color='#8b8b8b', label='chance')
ax[1].plot([0,1], [1,0], linestyle='--', color='#8b8b8b', label='chance')
ax[0].legend(loc='lower right', frameon=False)
ax[1].legend(loc='lower left', frameon=False)
ax[0].set_xlabel('False Positive Rate', labelpad=10, fontsize=12)
ax[0].set_ylabel('True Positive Rate', labelpad=10, fontsize=12)
ax[1].set_xlabel('Recall', labelpad=10, fontsize=12)
ax[1].set_ylabel('Precision', labelpad=10, fontsize=12)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')

fig.tight_layout()
pdf_pages.savefig(fig)

pdf_pages.close()
