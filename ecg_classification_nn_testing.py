import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ecg_classification_nn_training import testloader, classifier_network, mlb

clf = classifier_network()
PATH = './ecg_classification_nn.pth'

#load test data and nn from training file
data = iter(testloader)
inputs, labels = next(data)

clf.load_state_dict(torch.load(PATH))

#test the test data with the model
print("testing the model")
clf.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in testloader:
        outputs = clf(inputs)
        sigmoid = nn.Sigmoid()
        probabilities = sigmoid(outputs)
        predictions = (probabilities > 0.9).int()
        
        all_predictions.append(predictions)
        all_labels.append(labels)
all_predictions = torch.cat(all_predictions, dim=0)
all_labels = torch.cat(all_labels, dim = 0)
print("testing complete")

#analyze results

all_predictions_np = all_predictions.numpy()
all_labels_np = all_labels.numpy()

all_predictions_np_transformed = mlb.inverse_transform(all_predictions_np)
all_labels_np_transformed = mlb.inverse_transform(all_labels_np)


