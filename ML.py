# -*- coding: utf-8 -*-
"""
@author: vpsora
contact: vigneashwara.solairajapandiyan@utu.fi,vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work
"Adaptive In-situ Monitoring for Laser Powder Bed Fusion:Self-Supervised Learning for Layer Thickness Monitoring Across Scan lengths based on Pyrometry"

@any reuse of this code should be authorized by the code author
"""
# %%
# Libraries to import

import os
import numpy as np
# implementing train-test-split
from sklearn.model_selection import train_test_split
from Utils.plot_roc import *
from Classifiers.XGBoost import *
from Classifiers.Logistic_regression import *
from Classifiers.NavieBayes import *
from Classifiers.QDA import *
from Classifiers.kNN import *
from Classifiers.NeuralNets import *
from Classifiers.SVM import *
from Classifiers.RF import *


directory = 'Preprocessed'
filename = 'PhotodiodeD1'
filename_ = 'ML'

folder_created = os.path.join('Figures/', directory)
print(folder_created)

window_lengths = [0, 100, 250, 500]

''' Plotting functions'''
# Plotting the latent spaces computed from the backbone individually in 2D and 3D on driven by the window lengths
for length in window_lengths:

    # PhotodiodeD1_0_train_embeddings

    train_embeddings = folder_created+'/' + \
        str(filename)+'_'+str(length)+'_train_embeddings' + '.npy'
    Featurespace = np.load(train_embeddings)
    # print(X.dtype())

    # PhotodiodeD1_0_train_labels
    train_labelsname = folder_created+'/' + \
        str(filename)+'_'+str(length)+'_train_labels'+'.npy'
    classspace = np.load(train_labelsname)
    # print(Y.shape())

    X_train, X_test, y_train, y_test = train_test_split(
        Featurespace, classspace, test_size=0.25, random_state=66)

    classes = np.unique(classspace)
    classes = list(classes)

    # folder = os.path.join(path_, 'Contrastive_Classifier')
    folder = os.path.join('Figures/', str(filename_))

    try:
        os.makedirs(folder, exist_ok=True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")

    print(folder)
    folder = folder+'/'

    # RF(X_train, X_test, y_train, y_test, 100, feature_cols, Featurespace, classspace, classes, folder)
    NN(X_train, X_test, y_train, y_test, Featurespace, classspace, classes, folder)
    KNN(X_train, X_test, y_train, y_test, Featurespace,
        classspace, classes, 15, 'distance', folder)
    QDA(X_train, X_test, y_train, y_test,
        Featurespace, classspace, classes, folder)
    NB(X_train, X_test, y_train, y_test, Featurespace, classspace, classes, folder)
    LR(X_train, X_test, y_train, y_test, Featurespace, classspace, classes, folder)
    XGBoost(X_train, X_test, y_train, y_test,
            Featurespace, classspace, classes, folder)
    SVM(X_train, X_test, y_train, y_test,
        Featurespace, classspace, classes, folder)
