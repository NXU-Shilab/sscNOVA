from sklearn.ensemble import RandomForestClassifier
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Feature_3102.csv", sep='\t')

feature = data.iloc[:, 5:-1]
label = data.iloc[:,-1]

rfc = RandomForestClassifier(random_state=42)
rfc.fit(feature, label)

importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1] 
selected_feat_names = set()
S = {}

for f in range(feature.shape[1]):
    if importances[indices[f]] >= 0.001:
        selected_feat_names.add(feature.columns[indices[f]])
        S[feature.columns[indices[f]]] = importances[indices[f]]
        print(feature.columns[indices[f]], importances[indices[f]])

imp_fea = pd.Series(S)

selected_features_df = pd.DataFrame(imp_fea, columns=['Importance'])
selected_features_df.index.name = 'Feature'
selected_features_df['Importance'] = selected_features_df['Importance'].apply(lambda x: '{:.6f}'.format(x))
selected_features_df.to_csv('Feature_Importance.txt', sep='\t')

imp_fea = imp_fea.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(25, 25))
bars = ax.barh(range(len(imp_fea)), imp_fea, align='center')
plt.yticks(range(len(imp_fea)), imp_fea.index)
plt.xlabel('Feature Importance', labelpad=10, fontsize=14)
plt.ylabel('Feature Name', labelpad=10, fontsize=14)

plt.xlim(0, max(imp_fea) + 0.001)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.0001, bar.get_y() + bar.get_height() / 2, '{:.6f}'.format(width), ha='left', va='center')

plt.tight_layout()

with PdfPages('Feature_Importance.pdf') as pdf:
    pdf.savefig()
    plt.close()

print(len(selected_feat_names), "features are selected.")
