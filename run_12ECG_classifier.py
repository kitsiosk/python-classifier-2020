#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features

def run_12ECG_classifier(data,header_data,classes,model):
    # Array of mean and std used for normalization
    mean = np.asarray(joblib.load('mean.sav'))
    std = np.asarray(joblib.load('std.sav'))

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1,-1)
    # Normalize features to have 0 mean and 1 std
    feats_reshape = feats_reshape - mean.reshape(1,-1)
    feats_reshape = np.divide(feats_reshape, std.reshape(1,-1))
    
    score = model.predict(feats_reshape)
    threshold = 0.1
    
    for i in range(num_classes):
        if score[0][i] >= threshold:
            current_label[i] = 1
        current_score[i] = score[0][i]

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
    filename='finalized_model_auth.sav'
    loaded_model = joblib.load(filename)

    return loaded_model
