import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import multilabel_confusion_matrix
from ecg_classification_nn_testing import all_predictions, all_labels, all_predictions_np_transformed, all_labels_np_transformed
from ecg_classification_nn_training import mlb

all_predictions = all_predictions.numpy()
all_labels = all_labels.numpy()

classes = mlb.classes_
mcm = multilabel_confusion_matrix(all_labels, all_predictions)

num_classes = len(mlb.classes_)
TP = sum(mcm[k,1,1] for k in range(num_classes))
TN = sum(mcm[k,0,0] for k in range(num_classes))
FN = sum(mcm[k,1,0] for k in range(num_classes))
FP = sum(mcm[k,0,1] for k in range(num_classes))

fig, ax = plt.subplots(1,5, figsize =(15,3))
vmax = mcm.max()
global_accuracy = 100*(TP+TN)/(TP+TN+FN+FP)
for i, cm in enumerate(mcm):
    ylabels = ["",""]
    xlabels = ["",""]
    if (i==0):
        ylabels = ["Actually Negative", "Actually Positive"]
        xlabels = ["Predicted Negative", "Predicted Positive"]

    sns.heatmap(cm, ax=ax[i], vmin=0, vmax=vmax,cbar=False, square=True,  annot=True, fmt="d",cmap = "OrRd", xticklabels=xlabels, yticklabels=ylabels)
    ax[i].set_title(f"{classes[i]}")

fig.suptitle(f"Confusion Matrices for ECG Classification with a CNN\n"f"Global Accuracy: {global_accuracy}", horizontalalignment = 'center', fontsize=10, y=0.78, x = 0.45)
fig.colorbar(ax[0].collections[0], shrink = 0.5,ax=ax, location ="right")

plt.savefig("confusion_matrix_plot.png")
plt.show()
print(f"global accuracy: {100*(TP+TN)/(TP+TN+FN+FP)}%")





