from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np

data = pd.read_csv("Feature_3102.csv", sep='\t')

feature = data.iloc[:, 5:-1]
label = data.iloc[:,-1]

model = SelectKBest(f_classif, k=200)
feature_new = model.fit_transform(feature, label)

feature = pd.DataFrame(feature)

scores = model.scores_
indices = np.argsort(scores)[::-1]

k_best_list = []

for i in range(200):
    k_best_feature = feature.columns[indices[i]]
    k_best_list.append(k_best_feature)

with open("Feature_f_classif_SelectKBest_200.txt", "w") as file:
    for feature_name in k_best_list:
        file.write(feature_name + "\n")
