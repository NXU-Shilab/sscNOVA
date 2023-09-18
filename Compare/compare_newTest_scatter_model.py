import matplotlib.pyplot as plt

auc_scores = [0.658, 0.652,	0.647, 0.643, 0.626, 0.626, 0.612, 0.609, 0.608, 0.604, 0.592, 0.590]
auprc_scores = [0.580, 0.545, 0.498, 0.516,	0.487, 0.533, 0.485, 0.452,	0.49, 0.477, 0.487, 0.464]
model_labels = ["sscNOVA", "cnn_pseudo_40", "svm_pseudo_40", "svm_pseudo_150", "cnn_pseudo_150", "svm_pseudo_141", "tf_pseudo_150", "rf_pseudo_40", "tf_pseudo_40", "tf_pseudo_141","rf_pseudo_141", "rf_pseudo_150"]
colors = ['#E89C9C', '#9CC29C', '#9C9CE8', '#E89CE8', '#9CE8E8', '#E8E89C', '#A5ABBE', '#A0CBA0', '#C18867', '#B0B0B0', '#D6D6B0', '#E9967A']

fig, ax= plt.subplots(figsize=(8, 6))

scatter = plt.scatter(auc_scores, auprc_scores, c=colors, s=100)

handles = []
for color, label, auc, auprc in zip(colors, model_labels, auc_scores, auprc_scores):
    handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, markeredgewidth=0, label=f"{label} (AUC: {auc:.3f}, AUPRC: {auprc:.3f})")
    handles.append(handle)

legend = ax.legend(handles=handles, loc='upper left', fontsize=8)

ax.set_xlim(0.3, 0.7)
ax.set_ylim(0.3, 0.7)

ax.set_xlabel("AUC")
ax.set_ylabel("AUPRC")

plt.savefig('compare_newTest_scatter_model.pdf', format='pdf')
