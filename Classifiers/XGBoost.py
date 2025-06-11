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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from numpy import sort

from Utils.Helper import *
from Utils.plot_roc import *

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

from numpy import mean
from numpy import std
from sklearn.preprocessing import LabelEncoder

# %%


def XGBoost(X_train, X_test, y_train, y_test, Featurespace, classspace, classes, folder):

    print('Model to be trained is XGBoost')

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)

    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    predictions = model.predict(X_test)
    print("XGBoost Accuracy:", metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    graph_name1 = 'XGBoost'+'_without normalization w/o Opt'
    graph_name2 = 'XGBoost'

    graph_1 = 'XGBoost'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2 = 'XGBoost'+'_Confusion_Matrix'+'_'+'Opt'+'.png'

    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]

    for title, normalize, graphname in titles_options:
        plt.figure(figsize=(20, 10), dpi=400)
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                                     display_labels=classes,
                                                     cmap=plt.cm.pink, xticks_rotation='vertical',
                                                     normalize=normalize, values_format='0.2f')
        # Adjust tick label size (axes labels)
        disp.ax_.tick_params(axis='both', which='major', labelsize=8)  # Reduce font size for tick labels
    
    # Adjust font size inside matrix cells
        for row in disp.text_:
           for text in row:
            text.set_fontsize(6) # Reduce font size for cell annotations
        plt.title(title, size=12)
        graphname = folder+graphname
        plt.savefig(graphname, bbox_inches='tight', dpi=400)
    savemodel = folder+'XGBoost'+'_model'+'.sav'
    joblib.dump(model, savemodel)

    plt.figure(figsize=(20, 10), dpi=400)
    plot_importance(model)
    plt.savefig('Feature_XGBoost.png', bbox_inches='tight', dpi=400)
    pyplot.show()

    thresholds = sort(model.feature_importances_)
