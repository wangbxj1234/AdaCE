# -*- coding: utf-8 -*-


from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np
from datasets import Dataset


# The following code is adapted from the GitHub repository of Emadeldeen Eldele:
# https://github.com/emadeldeen24/TS-TCC/blob/main/data_preprocessing/uci_har/preprocess_har.py
#############################################
data_dir = '/content/drive/MyDrive/UCI HAR Dataset'

subject_data = np.loadtxt(f'{data_dir}/train/subject_train.txt')
# Samples
train_acc_x = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_x_train.txt')
train_acc_y = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_y_train.txt')
train_acc_z = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_z_train.txt')
train_gyro_x = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_x_train.txt')
train_gyro_y = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_y_train.txt')
train_gyro_z = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_z_train.txt')
train_tot_acc_x = np.loadtxt(f'{data_dir}/train/Inertial Signals/total_acc_x_train.txt')
train_tot_acc_y = np.loadtxt(f'{data_dir}/train/Inertial Signals/total_acc_y_train.txt')
train_tot_acc_z = np.loadtxt(f'{data_dir}/train/Inertial Signals/total_acc_z_train.txt')

test_acc_x = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_x_test.txt')
test_acc_y = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_y_test.txt')
test_acc_z = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_z_test.txt')
test_gyro_x = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_x_test.txt')
test_gyro_y = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_y_test.txt')
test_gyro_z = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_z_test.txt')
test_tot_acc_x = np.loadtxt(f'{data_dir}/test/Inertial Signals/total_acc_x_test.txt')
test_tot_acc_y = np.loadtxt(f'{data_dir}/test/Inertial Signals/total_acc_y_test.txt')
test_tot_acc_z = np.loadtxt(f'{data_dir}/test/Inertial Signals/total_acc_z_test.txt')

train_data = np.stack((train_acc_x, train_acc_y, train_acc_z,
                       train_gyro_x, train_gyro_y, train_gyro_z,
                       train_tot_acc_x, train_tot_acc_y, train_tot_acc_z), axis=1)
X_test = np.stack((test_acc_x, test_acc_y, test_acc_z,
                      test_gyro_x, test_gyro_y, test_gyro_z,
                      test_tot_acc_x, test_tot_acc_y, test_tot_acc_z), axis=1)
# labels
train_labels = np.loadtxt(f'{data_dir}/train/y_train.txt')
train_labels -= np.min(train_labels)
y_test = np.loadtxt(f'{data_dir}/test/y_test.txt')
y_test -= np.min(y_test)


X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
#############################################



#First part of our AdaCE
#!pip install huggingface_hub
#!huggingface-cli login

trainset = dict()
trainset["image"] = torch.from_numpy(X_train).view(5881,3, 3, 128)
trainset["label"] = torch.from_numpy(y_train)
train_dataset = Dataset.from_dict(trainset)



testset = dict()
testset["image"] = torch.from_numpy(X_val).view(1471,3, 3, 128)
testset["label"] = torch.from_numpy(y_val)
test_dataset = Dataset.from_dict(testset)
#

#test_dataset.push_to_hub("wbxlala/har3", "test")
#train_dataset.push_to_hub("wbxlala/har3", "train")

