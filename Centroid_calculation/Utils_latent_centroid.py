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

import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
from scipy.spatial.distance import euclidean


def anomaly_detection_latent_centroid(epoch_length, folder_created, filename, threshold=None, ideal_label_value=30):
    """
    Performs anomaly detection using Euclidean distance to the '30 um' centroid.
    Includes centroid computation, auto-threshold selection (if None), prediction, evaluation, and visualization.

    Returns:
        result (dict): Includes centroids, distances, predictions, threshold, accuracy, and timing stats.
    """

    # Load data
    features = np.load(
        f"{folder_created}/{filename}_{epoch_length}_train_embeddings.npy")
    labels = np.load(
        f"{folder_created}/{filename}_{epoch_length}_train_labels.npy")

    # Label mapping
    new_labels = ['10 um', '20 um', '30 um', '40 um', '50 um', '60 um',
                  '70 um', '80 um', '90 um', '100 um', '110 um']
    label_map = dict(zip(range(len(new_labels)), new_labels))
    ideal_label_value = [k for k, v in label_map.items() if v == '30 um'][0]

    # Compute centroids
    unique_labels = np.unique(labels)
    centroids = {label: np.mean(
        features[labels == label], axis=0) for label in unique_labels}

    if ideal_label_value not in centroids:
        raise ValueError(
            f"Label for '30 um' ({ideal_label_value}) not found in data.")

    ideal_centroid = centroids[ideal_label_value]

    # Compute distances
    start_total = time.time()
    distances = []
    sample_times = []

    for vec in features:
        t0 = time.time()
        d = euclidean(vec, ideal_centroid)
        t1 = time.time()
        distances.append(d)
        sample_times.append(t1 - t0)

    distances = np.array(distances)
    ground_truth = labels == ideal_label_value

    # Threshold auto-selection
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(ground_truth, -distances)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        threshold = -thresholds[best_idx]
        print(f"Auto-selected threshold (Youden's J): {threshold:.4f}")

    # Prediction
    predictions = distances < threshold
    total_time = time.time() - start_total

    # Timing summary
    mean_time_ms = np.mean(sample_times) * 1000
    std_time_ms = np.std(sample_times) * 1000
    print(f"Average processing time per window: {
          mean_time_ms:.4f} ± {std_time_ms:.4f} ms")

    # Accuracy and confusion matrix
    acc = accuracy_score(ground_truth, predictions)
    cm = confusion_matrix(ground_truth, predictions)

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(ground_truth, predictions,
          target_names=['Anomaly', 'Normal']))

    # Plotting
    plt.figure(figsize=(7, 4), dpi=200)
    plt.hist(distances[ground_truth], bins=30, alpha=0.7,
             label='Normal (30 µm)', color='green')
    plt.hist(distances[~ground_truth], bins=30,
             alpha=0.7, label='Anomaly', color='red')
    plt.axvline(threshold, color='black', linestyle='--',
                label=f'Threshold = {threshold:.2f}')
    plt.xlabel('Distance to centroid (30 µm)')
    plt.ylabel('Number of windows')
    plt.title(f'Distance distribution (Window length {
              1000 - epoch_length})\nAccuracy = {acc*100:.2f}%')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=True)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_created,
                f"Distance_histogram_latent_epoch_{epoch_length}.png"))
    plt.show()
    plt.close()

    return {
        "centroids": centroids,
        "distances": distances,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "confusion_matrix": cm,
        "accuracy": acc,
        "threshold": threshold,
        "mean_time_ms": mean_time_ms,
        "std_time_ms": std_time_ms
    }
