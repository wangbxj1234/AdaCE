# -*- coding: utf-8 -*-




import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np

data = pd.read_csv("./data.csv")

# The following code is adapted from the GitHub repository of Emadeldeen Eldele:
# https://github.com/emadeldeen24/TS-TCC/blob/main/data_preprocessing/epilepsy/preprocess.py
#############################################
y = data.iloc[:, -1]
x = data.iloc[:, 1:-1]

x = x.to_numpy()
y = y.to_numpy()
y = y - 1

for i, j in enumerate(y):
    if j != 0:
        y[i] = 1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#############################################


#Our part
##
train_data = pd.DataFrame({'sample': X_train.tolist(), 'label': y_train})
train_data['sample'] = train_data['sample'].apply(lambda x: ' '.join(map(str, x)), )
train_data.to_csv("train.csv", index=False)

val_data = pd.DataFrame({'sample': X_val.tolist(), 'label': y_val})
val_data['sample'] = val_data['sample'].apply(lambda x: ' '.join(map(str, x)), )
val_data.to_csv("val.csv", index=False)

test_data = pd.DataFrame({'sample': X_test.tolist(), 'label': y_test})
test_data['sample'] = test_data['sample'].apply(lambda x: ' '.join(map(str, x)), )
test_data.to_csv("test.csv", index=False)
##
