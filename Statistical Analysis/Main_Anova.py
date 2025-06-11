# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 22:17:37 2025

@author: vpsora
contact: vigneashwara.solairajapandiyan@utu.fi,vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work
"Adaptive In-situ Monitoring for Laser Powder Bed Fusion:Self-Supervised Learning for Layer Thickness Monitoring Across Scan lengths based on Pyrometry"

@any reuse of this code should be authorized by the code author
"""

from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import os
import numpy as np

# %%
sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid': False})
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

# %%
# Get the path of the current working directory
file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
print(total_path)
# %%
# Create a folder to save the data
folder_name = 'Visualization_Plots'
path_ = os.path.join(total_path, folder_name)
print("Name of the folder..", path_)

try:
    os.makedirs(path_, exist_ok=True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")

# %%
# Load the dataset with frequency domain features
data_feature = os.path.join(total_path, 'Featurespace_1000.npy')
print(data_feature)
data_feature = np.load(data_feature)
print(data_feature.shape[1])
data_feature = pd.DataFrame(data_feature)

# Load the dataset with ground truth labels
data_class = os.path.join(total_path, 'classpace_1000.npy')
print(data_class)
data_class = np.load(data_class)
print(data_class.shape)
data_class = pd.DataFrame(data_class)
data_class.columns = ['Categorical']
data_class = pd.DataFrame(data_class)

# Concatenate the feature and class labels
data = pd.concat([data_feature, data_class], axis=1)
# print("Respective windows per category", data.Categorical.value_counts())
minval = min(data.Categorical.value_counts())
# minval=np.round(minval,decimals=-3)
# print("windows of the class: ", minval)
data_1 = pd.concat([data[data.Categorical == cat].head(minval)
                   for cat in data.Categorical.unique()])
# print("Balanced dataset: ", data_1.Categorical.value_counts())

Featurespace = data_1.iloc[:, :-1]
Featurespace = (Featurespace[7])  # kurtosis
classspace = data_1.iloc[:, -1]

# %%

# Ensure you're working with Series
kurtosis_values = Featurespace
class_labels = classspace

# Combine into a DataFrame for grouping
df_anova = pd.DataFrame({
    'kurtosis': kurtosis_values,
    'class_label': class_labels
})

# Group kurtosis values by class
grouped_data = [group['kurtosis'].values for name,
                group in df_anova.groupby('class_label')]

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(*grouped_data)

# Print results
print("=== One-Way ANOVA on Kurtosis ===")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value:     {p_value:.6e}")

if p_value < 0.05:
    print("Result: Statistically significant differences across classes (p < 0.05).")
else:
    print("Result: No statistically significant differences (p ≥ 0.05).")

# %%


# Run Tukey HSD if ANOVA is significant
if p_value < 0.05:
    tukey = pairwise_tukeyhsd(
        endog=df_anova['kurtosis'], groups=df_anova['class_label'], alpha=0.05)
    print("\n=== Tukey HSD Post-hoc Test ===")
    print(tukey)
