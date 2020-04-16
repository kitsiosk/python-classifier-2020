#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features
from sklearn.preprocessing import normalize

def run_12ECG_classifier(data,header_data,classes,model):
	# Array of mean and std used for normalization
    norm_array = np.array([[5.41927364e+02, 6.18449827e+06, 5.28511638e+02,
            5.65793159e+06, 1.34273763e+02, 4.16362736e+06,
            2.42319167e+04, 2.83173904e+14, -6.31155215e-02,
            -4.75982666e-01, 7.31542705e-01, 1.54435326e+00,
            1.33367747e+05, 6.01657700e+01, 4.62120111e-01,
            5.26006083e+02, 1.59808261e+06, 5.13823906e+02,
            1.15766950e+06, 1.43760699e+02, 1.78784952e+06,
            2.62631836e+04, 2.37026599e+14, 5.26644580e-02,
            -6.28349130e-01, 4.34943770e-01, 9.11368251e-01,
            1.33367747e+05],
           [1.32203861e+02, 1.15856624e+07, 1.66425759e+02,
            7.51984395e+06, 7.20743439e+01, 1.61509566e+07,
            3.18934224e+04, 3.66828901e+15, 1.28195100e+00,
            1.56809428e+00, 4.20396849e+00, 7.19662412e+00,
            4.50828465e+05, 1.90588774e+01, 4.98563049e-01,
            1.18104583e+02, 9.07906279e+06, 1.55582414e+02,
            6.44997676e+06, 6.71081873e+01, 1.51626808e+07,
            2.80868759e+04, 3.37006059e+15, 1.18874739e+00,
            1.38706967e+00, 3.67593878e+00, 5.96220161e+00,
            4.50828465e+05]])
            
    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1,-1)
    # Normalize features to have 0 mean and 1 std
    feats_reshape = feats_reshape - norm_array[0][:]
    feats_reshape = np.divide(feats_reshape, norm_array[1][:])
    
    score = model.predict(feats_reshape)
    threshold = 0.1
    
    for i in range(num_classes):
        if score[0][i] >= threshold:
            current_label[i] = 1
        current_score[i] = score[0][i]

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
    filename='finalized_model_1.sav'
    loaded_model = joblib.load(filename)

    return loaded_model
