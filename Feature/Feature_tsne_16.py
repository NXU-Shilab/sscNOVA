from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt

data_21907 = pd.read_csv('Feature_21907.csv', sep='\t')
data_3102 = pd.read_csv('Feature_3102.csv', sep='\t')

data_mutual_1000 = pd.read_csv("Feature_mutual_info_classif_SelectKBest_1000.csv", sep='\t')
data_mutual_800 = pd.read_csv("Feature_mutual_info_classif_SelectKBest_800.csv", sep='\t')
data_mutual_600 = pd.read_csv("Feature_mutual_info_classif_SelectKBest_600.csv", sep='\t')
data_mutual_400 = pd.read_csv("Feature_mutual_info_classif_SelectKBest_400.csv", sep='\t')
data_mutual_200 = pd.read_csv("Feature_mutual_info_classif_SelectKBest_200.csv", sep='\t')

data_f_1000 = pd.read_csv("Feature_f_classif_SelectKBest_1000.csv", sep='\t')
data_f_800 = pd.read_csv("Feature_f_classif_SelectKBest_800.csv", sep='\t')
data_f_600 = pd.read_csv("Feature_f_classif_SelectKBest_600.csv", sep='\t')
data_f_400 = pd.read_csv("Feature_f_classif_SelectKBest_400.csv", sep='\t')
data_f_200 = pd.read_csv("Feature_f_classif_SelectKBest_200.csv", sep='\t')

data_150 = pd.read_csv("Feature_mutual_info_classif_SelectKBest_150.csv", sep='\t')
data_100 = pd.read_csv("Feature_mutual_info_classif_SelectKBest_100.csv", sep='\t')
data_50 = pd.read_csv("Feature_mutual_info_classif_SelectKBest_50.csv", sep='\t')
data_141 = pd.read_csv('Feature_random_forest.csv', sep='\t')

feature_21907 = data_21907.iloc[:, 5:-1]
label_21907 = data_21907.iloc[:,-1]

feature_3102 = data_3102.iloc[:, 5:-1]
label_3102 = data_3102.iloc[:,-1]

feature_mutual_1000 = data_mutual_1000.iloc[:, 5:-1]
label_mutual_1000 = data_mutual_1000.iloc[:,-1]

feature_mutual_800 = data_mutual_800.iloc[:, 5:-1]
label_mutual_800 = data_mutual_800.iloc[:,-1]

feature_mutual_600 = data_mutual_600.iloc[:, 5:-1]
label_mutual_600 = data_mutual_600.iloc[:,-1]

feature_mutual_400 = data_mutual_400.iloc[:, 5:-1]
label_mutual_400 = data_mutual_400.iloc[:,-1]

feature_mutual_200 = data_mutual_200.iloc[:, 5:-1]
label_mutual_200 = data_mutual_200.iloc[:,-1]

feature_f_1000 = data_f_1000.iloc[:, 5:-1]
label_f_1000 = data_f_1000.iloc[:,-1]

feature_f_800 = data_f_800.iloc[:, 5:-1]
label_f_800 = data_f_800.iloc[:,-1]

feature_f_600 = data_f_600.iloc[:, 5:-1]
label_f_600 = data_f_600.iloc[:,-1]

feature_f_400 = data_f_400.iloc[:, 5:-1]
label_f_400 = data_f_400.iloc[:,-1]

feature_f_200 = data_f_200.iloc[:, 5:-1]
label_f_200 = data_f_200.iloc[:,-1]

feature_150 = data_150.iloc[:, 5:-1]
label_150 = data_150.iloc[:,-1]

feature_100 = data_100.iloc[:, 5:-1]
label_100 = data_100.iloc[:,-1]

feature_50 = data_50.iloc[:, 5:-1]
label_50 = data_50.iloc[:,-1]

feature_141 = data_141.iloc[:, 5:-1]
label_141 = data_141.iloc[:,-1]

files = [data_21907, data_3102, data_mutual_1000, data_f_1000, data_mutual_800, data_f_800, data_mutual_600, data_f_600, data_mutual_400, data_f_400, data_mutual_200, data_f_200, data_150, data_100, data_50, data_141]
features = [feature_21907, feature_3102, feature_mutual_1000, feature_f_1000, feature_mutual_800, feature_f_800, feature_mutual_600,  feature_f_600, feature_mutual_400, feature_f_400, feature_mutual_200, feature_f_200, feature_150, feature_100, feature_50, feature_141]
labels = [label_21907, label_3102, label_mutual_1000, label_f_1000, label_mutual_800, label_f_800, label_mutual_600, label_f_600, label_mutual_400, label_f_400, label_mutual_200, label_f_200, label_150, label_100, label_50, label_141]
tmp = ['21907', '3102', '1000', '1000', '800', '800', '600', '600', '400', '400', '200', '200', '150', '100', '50', '141']

pdf_pages = PdfPages('Feature_tsne_16.pdf')

for i in range(len(files)):
    
    feature = features[i]
    label = labels[i]
    
    tsne = TSNE(n_components=2, learning_rate=100)
    X_tsne = tsne.fit_transform(feature)

    if i % 16 == 0:
        fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
    axs[int(i/4)%4,i%4].scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, cmap='viridis', s=3)
    
    if i == 0:
        axs[int(i/4)%4,i%4].set_title(f'Chromatin Profile ( {tmp[i]} feature )')
        axs[int(i/4)%4,i%4].set_xlabel('TSNE-1')
        axs[int(i/4)%4,i%4].set_ylabel('TSNE-2')
    elif i == 1:
        axs[int(i/4)%4,i%4].set_title(f'Immune Feature ( {tmp[i]} feature )')
        axs[int(i/4)%4,i%4].set_xlabel('TSNE-1')
        axs[int(i/4)%4,i%4].set_ylabel('TSNE-2')
    elif i == 15:
        axs[int(i/4)%4,i%4].set_title(f'Random Forest ( 141 feature )')
        axs[int(i/4)%4,i%4].set_xlabel('TSNE-1')
        axs[int(i/4)%4,i%4].set_ylabel('TSNE-2')
    elif i != 1 and i != 13 and i != 15 and i % 2 != 0:
        axs[int(i/4)%4,i%4].set_title(f'SelectKBest: f_classif ( {tmp[i]} feature )')
        axs[int(i/4)%4,i%4].set_xlabel('TSNE-1')
        axs[int(i/4)%4,i%4].set_ylabel('TSNE-2')
    else:
        axs[int(i/4)%4,i%4].set_title(f'SelectKBest: mutual_info_classif ( {tmp[i]} feature )')
        axs[int(i/4)%4,i%4].set_xlabel('TSNE-1')
        axs[int(i/4)%4,i%4].set_ylabel('TSNE-2')

    if i % 16 == 15 or i == len(tmp) - 1:
        fig.tight_layout()
        pdf_pages.savefig(fig)

for i in range(4):
    for j in range(4):
        axs[i,j].set_aspect('equal')
        
pdf_pages.close()
