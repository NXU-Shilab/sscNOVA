from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt

data_141 = pd.read_csv('Feature_random_forest.csv', sep='\t')
data_150 = pd.read_csv('Feature_mutual_info_classif_SelectKBest_150.csv', sep='\t')
data_40 = pd.read_csv('Feature_Sequence_Class.csv', sep='\t')

feature_141 = data_141.iloc[:, 5:-1]
label_141 = data_141.iloc[:,-1]

feature_150 = data_150.iloc[:, 5:-1]
label_150 = data_150.iloc[:,-1]

feature_40 = data_40.iloc[:, 5:-1]
label_40 = data_40.iloc[:,-1]

files = [data_141, data_150, data_40]
features = [feature_141, feature_150, feature_40]
labels = [label_141, label_150, label_40]
tmp = ['141', '150', '40']

pdf_pages = PdfPages('Feature_tsne_3.pdf')

for i in range(len(files)):
    
    feature = features[i]
    label = labels[i]
    
    tsne = TSNE(n_components=2, learning_rate=100)
    X_tsne = tsne.fit_transform(feature)

    if i % 3 == 0:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs[i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, cmap='viridis', s=3)
    
    if i == 0:
        axs[i].set_title(f'Random Forest ( {tmp[i]} feature )')
        axs[i].set_xlabel('TSNE-1')
        axs[i].set_ylabel('TSNE-2')
    elif i == 1:
        axs[i].set_title(f'SelectKBest: mutual_info_classif ( {tmp[i]} feature )')
        axs[i].set_xlabel('TSNE-1')
        axs[i].set_ylabel('TSNE-2')
    else:
        axs[i].set_title(f'Sequence Class ( {tmp[i]} feature )')
        axs[i].set_xlabel('TSNE-1')
        axs[i].set_ylabel('TSNE-2')

    if i % 3 == 2 or i == len(tmp) - 1:
        fig.tight_layout()
        pdf_pages.savefig(fig)

for i in range(3):
        axs[i].set_aspect('equal')
        
pdf_pages.close()
