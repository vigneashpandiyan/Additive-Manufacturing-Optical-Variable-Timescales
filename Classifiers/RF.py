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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
from Utils.Helper import *
from Utils.plot_roc import *
import numpy


def RF(X_train, X_test, y_train, y_test, n_estimators, feature_cols, Featurespace, classspace, classes, folder):

    print('Model to be trained is RF')

    model = RandomForestClassifier(n_estimators=n_estimators, oob_score=True)
    model.fit(X_train, y_train)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)

    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # Accuracy of the model

    predictions = model.predict(X_test)
    print("RF Accuracy:", metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    # Plotting of the model

    graph_name1 = 'Random Forest'+'_without normalization w/o Opt'
    graph_name2 = 'Random Forest'

    graph_1 = 'Random Forest'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2 = 'Random Forest'+'_Confusion_Matrix'+'_'+'Opt'+'.png'

    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]

    for title, normalize, graphname in titles_options:
        plt.figure(figsize=(20, 10), dpi=200)
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                                     display_labels=classes,
                                                     cmap=plt.cm.Blues, xticks_rotation='vertical',
                                                     normalize=normalize, values_format='0.2f')

        # disp.ax_.set_title(title)
        # Adjust tick label size (axes labels)
        disp.ax_.tick_params(axis='both', which='major', labelsize=8)  # Reduce font size for tick labels
    
    # Adjust font size inside matrix cells
        for row in disp.text_:
           for text in row:
            text.set_fontsize(6) # Reduce font size for cell annotations
        plt.title(title, size=12)
        graphname = folder+graphname
        plt.savefig(graphname, bbox_inches='tight', dpi=200)
    savemodel = folder+'Random Forest'+'_model'+'.sav'
    joblib.dump(model, savemodel)
