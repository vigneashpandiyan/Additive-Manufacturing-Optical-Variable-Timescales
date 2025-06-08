# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:57:22 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch


The codes in this following script will be used for the publication of the following work

"Classification of Progressive Wear on a Multi-Directional Pin-on-Disc 
Tribometer Simulating Conditions in Human Joints-UHMWPE against CoCrMo 
Using Acoustic Emission and Machine Learning"

@any reuse of this code should be authorized by the first owner, code author
"""
# %%
# Libraries to import
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
print(__doc__)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)

    axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve

    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times

    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score

    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def RecursiveFeatureElimination_Thresholdvalue(Featurenames, support):
    data = pd.DataFrame([])
    a_list = []
    b_list = []

    for Featurename, support in zip(Featurenames, support):

        a_list.append(Featurename)
        b_list.append(support)

    data = pd.DataFrame({'A': a_list, 'C': b_list})
    data = data.loc[data.C == True, :]

    return data


def RecursiveFeatureElimination(Featurespace, classspace, features):

    classspace = np.ravel(classspace)
    Featurespace = pd.DataFrame(Featurespace)
    num_cols = len(list(Featurespace))
    rng = range(1, num_cols + 1)
    Featurenames = ['Feature_' + str(i) for i in rng]
    Featurespace.columns = Featurenames
    feature_cols = list(Featurespace.columns)
    Featurespace.info()
    Featurespace.describe()
    Featurespace.head()

    model = RFE(estimator=LogisticRegression(), n_features_to_select=features, step=10, verbose=5)
    model.fit(Featurespace, classspace)

    support = model.get_support()

    data = RecursiveFeatureElimination_Thresholdvalue(Featurenames, support)

    index_value = data.index.values.tolist()
    my_array = np.array(index_value)
    Featurespace = Featurespace.to_numpy()
    Featurespace = Featurespace[:, my_array]

    return Featurespace
