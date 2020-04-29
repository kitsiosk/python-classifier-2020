#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features

def run_12ECG_classifier(data,header_data,classes,model):
    nn = model['nn']
    cnn = model['cnn']
    
    # Array of mean and std used for normalization
    mean = np.asarray(joblib.load('mean.sav'))[0:182]
    std = np.asarray(joblib.load('std.sav'))[0:182]

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    features_all = get_12ECG_features(data,header_data)
    features = np.asarray(features_all['features'])
    cnn_features = np.asarray(features_all['cnn_features'])
    feats_reshape = features.reshape(1,-1)
    # Normalize features to have 0 mean and 1 std
    feats_reshape = feats_reshape - mean.reshape(1,-1)
    feats_reshape = np.divide(feats_reshape, std.reshape(1,-1))
    
    cnn_feats_reshape = cnn_features.reshape(1, 7500, 12)
    
    score1 = nn.predict(feats_reshape)
    score2 = cnn.predict(cnn_feats_reshape)
    threshold = 0.1
    # weight for the weighted average ensembling(w for CNN, 1-w for NN)
    w = 0.72
    for i in range(num_classes):
        score = score1[0][i]*(1-w) + score2[0][i]*w
        if score>= threshold:
            current_label[i] = 1
        current_score[i] = score

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
    filename='finalized_model_auth.sav'
    filename_cnn='finalized_model_cnn.sav'
    nn = joblib.load(filename)
    cnn = joblib.load(filename_cnn)
    return {'nn': nn, 'cnn': cnn}
    return nn
