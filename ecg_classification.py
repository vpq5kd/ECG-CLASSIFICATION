import pandas as pd
import numpy as np
import wfdb
import ast
import torch
from sklearn import svm
from sklearn.preprocessing import MultiLabelBinarizer

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

X_train = torch.from_numpy(X[np.where(Y.strat_fold != test_fold)])
Y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass


X_test = torch.from_numpy(X[np.where(Y.strat_fold == test_fold)])
Y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass


#encode the data for pytorch to accept
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

mlb = MultiLabelBinarizer()

Y_train_encoded = torch.from_numpy(mlb.fit_transform(Y_train))
Y_test_encoded = torch.from_numpy(mlb.transform(Y_test))




