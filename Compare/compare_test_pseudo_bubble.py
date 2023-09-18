import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_excel("compare_test_pseudo.xlsx")

df = df.sort_values(by='Recall', ascending=False)

model_names = df.iloc[:, 0].tolist()
auc_values = df.iloc[:, 6].tolist()
auprc_values = df.iloc[:, 7].tolist()
recall_values = df.iloc[:, 3].tolist()

custom_colors = ['#E89C9C', '#9CC29C', '#9C9CE8', '#E89CE8', '#9CE8E8', '#E8E89C', '#A5ABBE', '#A0CBA0', '#C18867', '#D6D6B0', '#D8BFD8', '#E9967A']

plt.figure(figsize=(8, 6))
legend_handles = []  
for i, (auc, auprc, recall, name) in enumerate(zip(auc_values, auprc_values, recall_values, model_names)):
    label = f"{name} (Recall={recall:.3f})"  
    plt.scatter(auc, auprc, s=np.square((recall-0.7))*20000, label=label, color=custom_colors[i % len(custom_colors)])
    handle = plt.Line2D([], [], marker='o', markersize=10, linestyle='None', markeredgewidth=0, label=label, markerfacecolor=custom_colors[i % len(custom_colors)])
    legend_handles.append(handle)  

plt.legend(handles=legend_handles, loc='lower right', fontsize=8)

plt.xlabel('AUC')
plt.ylabel('AUPRC')
plt.xlim(0.75, 1)
plt.ylim(0.75, 1)

plt.savefig('compare_test_pseudo_bubble.pdf', format='pdf')
