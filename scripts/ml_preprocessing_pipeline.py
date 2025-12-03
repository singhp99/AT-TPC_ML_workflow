import numpy as np
import random
import numpy as np
import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin #for GridSearchCv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from ml_preprocessing_steps import OutlierDetection, UpDownScaling, ReclassifyingLabels, DataLimitation, ScalingData

config_path = Path("../config.json")
with config_path.open() as config_file:
    config = json.load(config_file)
cfg_ml_pipe = config["ml_prep_parameters"]
    
target_size = cfg_ml_pipe["target_size"]
isotope = cfg_ml_pipe["isotope"]

pipeline_1 = Pipeline([
    # ("noise", AttpcNoiseAddition(ratio_noise=0.1)),
    ("outlier",OutlierDetection()), #getting rid of the outliers
    ("sampler", UpDownScaling(target_size,isotope)),
]) #up/down sampler 

# The `pipeline_2` is a data processing pipeline that consists of the following steps:
pipeline_2 = Pipeline([
    ("reclassify",ReclassifyingLabels()),
    ("limiting", DataLimitation(target_size)),
    #("augument", DataAugumentation(target_size=800)),
    ("scaling", ScalingData()),
]) #reclassifying and scaling (w/ concatonated dataset)

"""Pipeline 1""" 
run_min = cfg_ml_pipe["run_min"]
run_max = cfg_ml_pipe["run_max"]
run_num_list = np.arange(run_min, run_max+1)
sum_data = None

for run_num in run_num_list: 
    print(f"Pipeline for Run {run_num}")
    data =  np.load(cfg_ml_pipe["data"] + f"run00{run_num}_data.npy")
    event_lengths = np.load(cfg_ml_pipe["event_lengths"] + f"run00{run_num}_evtlen.npy")
    
    assert data.shape == (event_lengths.size, np.max(event_lengths)+2, 4), f"Array {run_num} has incorrect shape"
    assert len(np.unique(data[:,-1,0])) == event_lengths.size, f"Array Run {run_num} has incorrect event_ids" 
    data_static = pipeline_1.fit_transform((data,event_lengths))
    if sum_data is None: #the first iteration only 
        sum_data = data_static
    else:
        sum_data = np.concatenate((sum_data,data_static),axis=0)

"""Pipeline 2"""
transformed_data = pipeline_2.fit_transform(sum_data)
print()
print(f"The full transformed data shape: {transformed_data.shape}")

mask = np.any((transformed_data[:, :-2, :3] > 1) | (transformed_data[:, :-2, :3] < -1), axis=(1, 2))
indices = np.argwhere(mask)
print(indices)
assert len(indices) == 0, "Points remain that are not within range[-1,1]"


"""Splitting data into train,val,test sets"""
y = transformed_data[:,-2,0] 

data_train_val, data_test = train_test_split(
    transformed_data, test_size=0.2, stratify=y) #need to preserve class distribution 

y_train_val = data_train_val[:, -2, 0]  
data_train, data_val = train_test_split(
    data_train_val, test_size=0.2, stratify=y_train_val) #need to preserve class distribution and ensure we train on all classes

X_train = data_train[:,:-2,:] 
y_train = data_train[:,-2,0] 

X_val = data_val[:,:-2,:] 
y_val = data_val[:,-2,0]  

X_test = data_test[:,:-2,:] 
y_test = data_test[:,-2,0] 


"""Train-val-test sets shape"""
train_shape = X_train.shape
print(f"The training set shape: {train_shape}")

val_shape = X_val.shape
print(f"The validation set shape: {val_shape}")

test_shape = X_test.shape
print(f"The test set shape: {test_shape}")


"""Class distribution for each set"""
_ , counts_train = np.unique(y_train, return_counts=True)
_ , counts_val = np.unique(y_val, return_counts=True)
_ , counts_test = np.unique(y_test, return_counts=True)

print(f"Class distribution for train data is {counts_train}")
print(f"Class distribution for val data is {counts_val}")
print(f"Class distribution for test data is {counts_test}")


"""Saving the sets for training"""
np.save(cfg_ml_pipe["directory_training"] + isotope + "_size" + str(target_size)+ "_train_features", X_train) #FILE CHANGE
np.save(cfg_ml_pipe["directory_training"] + isotope + "_size" + str(target_size)+ "_val_features", X_val) #FILE CHANGE
np.save(cfg_ml_pipe["directory_training"] + isotope + "_size" + str(target_size)+ "_test_features", X_test) #FILE CHANGE

np.save(cfg_ml_pipe["directory_training"] + isotope + "_size" + str(target_size)+ "_train_labels", y_train) #FILE CHANGE
np.save(cfg_ml_pipe["directory_training"] + isotope + "_size" + str(target_size)+ "_val_labels", y_val) #FILE CHANGE
np.save(cfg_ml_pipe["directory_training"] + isotope + "_size" + str(target_size)+ "_test_labels", y_test) #FILE CHANGE