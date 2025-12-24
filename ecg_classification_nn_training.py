import pandas as pd
import numpy as np
import wfdb
import ast

#processing code based on "example_physionet.py" from the dataset

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = '../PTB-XL/'
sampling_rate = 100

Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

try:
    print("attempting to load presaved np array")
    X = np.load("lowresolution_ecg_nparray.npy")
    print("presaved np array loaded successfully")
except OSError:
    print("np array load failed or does not exist, proceeding with hard load")
    X = load_raw_data(Y, sampling_rate, path)
    print("reached saved step")
    np.save("lowresolution_ecg_nparray.npy", X)
    print("saved successfully")

agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dict):
    tmp = []
    for key in y_dict.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

test_fold = 10

X_train = X[np.where(Y.strat_fold != test_fold)]
Y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass


X_test = X[np.where(Y.strat_fold == test_fold)]
Y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass


#encode the data for pytorch to accept
import torch
import joblib
from sklearn import svm
from sklearn.preprocessing import MultiLabelBinarizer

Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

mlb = MultiLabelBinarizer()

Y_train_encoded = mlb.fit_transform(Y_train)
Y_test_encoded = mlb.transform(Y_test)

joblib.dump(mlb, "ecg_classfication_encoder.joblib")
#define a data set
from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self, X_data, Y_data):
        self.X = torch.from_numpy(X_data.astype(np.float32))
        self.Y = torch.from_numpy(Y_data).type(torch.LongTensor)
        self.len = self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    def __len__(self):
        return self.len

train_data = Data(X_train, Y_train_encoded)
test_data = Data(X_test, Y_test_encoded)

#load data with a dataloader
batch_size = 32
trainloader = DataLoader(train_data, batch_size = batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(test_data, batch_size = batch_size, shuffle=True, num_workers=2)
#define neural network
import torch.nn as nn

input_dim = X_train.shape[1]
hidden_layers = 32
output_dim = 5

class classifier_network(nn.Module):
    def __init__(self):
        super(classifier_network, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = input_dim , out_channels=8, kernel_size =3, padding = 1)
        self.conv2 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten() 
        self.LReLU = nn.LeakyReLU()
        self.fc = nn.Linear(1536, 5)

    def forward(self, x):
        x = self.LReLU(self.conv1(x))
        x = self.LReLU(self.conv2(x))
        x = self.LReLU(self.conv3(x))
        x = self.LReLU(self.conv4(x))
        x = self.LReLU(self.conv5(x))
        x = self.flatten(x)
        x = self.fc(x)

        return x

clf = classifier_network()

#train neural network
def main(): 
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)

    epochs = 30
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, classes = data
            optimizer.zero_grad()
            outputs = clf(inputs)
            loss = criterion(outputs, classes.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch: {epoch+1}, loss: {running_loss}")
    print("model trained successfully")

    #save the neural network
    PATH = './ecg_classification_nn.pth'
    torch.save(clf.state_dict(), PATH)

    print("model saved successfully.")
if __name__ == "__main__":
    main()
