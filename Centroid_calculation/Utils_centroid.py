# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 13:35:10 2025

@author: vpsora
"""

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    accuracy_score,
)
import matplotlib.pyplot as plt
import os


def anomaly_detection_centroid(epoch_length, folder_created, filename, threshold=5.0, ideal_label_value=30):
    """
    Performs anomaly detection using Euclidean distance to the '30 um' centroid.
    Includes centroid computation, auto-threshold selection (if None), prediction, evaluation, and visualization.

    Args:
        epoch_length (int): Epoch identifier for the filename.
        folder_created (str): Folder containing the saved .npy files.
        filename (str): Base name of the saved files.
        threshold (float or None): Distance threshold; if None, will be automatically selected from ROC.

    Returns:
        result (dict): Dictionary with centroids, distances, predictions, ground_truth, threshold, and confusion matrix.
    """

    # Load data
    features = np.load(f"{folder_created}/{filename}_TSNE_{epoch_length}.npy")
    labels = np.load(f"{folder_created}/{filename}_label_{epoch_length}.npy")

    # Label mapping
    new_labels = ['10 um', '20 um', '30 um', '40 um', '50 um', '60 um',
                  '70 um', '80 um', '90 um', '100 um', '110 um']
    label_map = dict(zip(range(len(new_labels)), new_labels))

    # Find numeric label for '30 um'
    ideal_label_value = [k for k, v in label_map.items() if v == '30 um'][0]

    # Compute centroids
    unique_labels = np.unique(labels)
    centroids = {label: np.mean(features[labels == label], axis=0) for label in unique_labels}

    if ideal_label_value not in centroids:
        raise ValueError(f"Label for '30 um' ({ideal_label_value}) not found in data.")

    ideal_centroid = centroids[ideal_label_value]

    # Compute distances to ideal centroid
    distances = np.array([euclidean(vec, ideal_centroid) for vec in features])
    ground_truth = labels == ideal_label_value

    # Automatically select threshold if not given
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(ground_truth, -distances)  # Use -distances so "higher" = more normal
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        threshold = -thresholds[best_idx]  # Convert back
        print(f"Auto-selected threshold (Youden's J): {threshold:.4f}")

    # Predict based on threshold
    predictions = distances < threshold

    # Accuracy and confusion matrix
    acc = accuracy_score(ground_truth, predictions)
    cm = confusion_matrix(ground_truth, predictions)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(ground_truth, predictions, target_names=['Anomaly', 'Normal']))

    # Plot 1: Distance histogram with accuracy in title
    plt.figure(figsize=(7, 4),dpi=200)
    plt.hist(distances[ground_truth], bins=30, alpha=0.7, label='Normal (30 µm)', color='green')
    plt.hist(distances[~ground_truth], bins=30, alpha=0.7, label='Anomaly', color='red')
    
    threshold = 7.8
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold = {threshold:.2f}')
    plt.xlabel('Distance to centroid (30 µm)')
    plt.ylabel('Number of windows')
    plt.title(f'Distance Distribution for Anomaly Detection (Window length {1000-epoch_length})\nAccuracy = {acc*100:.2f}%')
    # plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=True)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_created, f"distance_histogram_epoch{epoch_length}.png"))
    plt.show()
    plt.close()


    return {
        "centroids": centroids,
        "distances": distances,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "confusion_matrix": cm,
        "accuracy": acc,
        "threshold": threshold
    }