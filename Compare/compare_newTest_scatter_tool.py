import matplotlib.pyplot as plt

auc_scores = [0.658, 0.644, 0.559, 0.523, 0.514, 0.510]
auprc_scores = [0.580, 0.529, 0.486, 0.460, 0.41, 0.392]
model_labels = ["sscNOVA", "ExPecto_probability", "deltaSVM_k562", "deltaSVM_hepg2", "deltaSVM_gm12878", "ExPecto_prediction"]
colors = ['#E89C9C', '#9CC29C', '#9C9CE8', '#E89CE8', '#9CE8E8', '#E8E89C']

fig, ax= plt.subplots(figsize=(8, 6))

scatter = plt.scatter(auc_scores, auprc_scores, c=colors, s=100)

handles = []
for color, label, auc, auprc in zip(colors, model_labels, auc_scores, auprc_scores):
    handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f"{label} (AUC: {auc:.3f}, AUPRC: {auprc:.3f})")
    handles.append(handle)

legend = ax.legend(handles=handles, loc='upper left', fontsize=8)

ax.set_xlim(0.3, 0.7)
ax.set_ylim(0.3, 0.7)

ax.set_xlabel("AUC")
ax.set_ylabel("AUPRC")

plt.savefig('compare_newTest_scatter_tool.pdf', format='pdf')
