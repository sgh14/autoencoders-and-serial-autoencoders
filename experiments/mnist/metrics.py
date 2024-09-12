import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

'''
Purity measures the frequency of data belonging to the same cluster sharing the
same class label, while Accuracy measures the frequency of data from the same
class appearing in a single cluster.
'''

def purity_score(y_true, y_pred):
    # Compute confusion matrix
    matrix = confusion_matrix(y_true, y_pred)

    # Find the maximum values in each row (each class)
    max_in_rows = np.amax(matrix, axis=1)

    # Sum the maximum values found
    purity = np.sum(max_in_rows) / np.sum(matrix)

    return purity


def clustering_accuracy(y_true, y_pred):
    # Compute the confusion matrix
    matrix = confusion_matrix(y_true, y_pred)

    # Use the linear_sum_assignment method to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(-matrix)

    # Calculate the accuracy using the optimal assignment
    accuracy = matrix[row_ind, col_ind].sum() / np.sum(matrix)

    return accuracy


def compute_metrics(
    dataset_small,
    dataset,
    dataset_noisy_small,
    dataset_noisy,
    output_dir,
    n_classes
):
    datasets = [dataset_small, dataset, dataset_noisy_small, dataset_noisy]
    titles = ['few-clean', 'many-clean', 'few-noisy', 'many-noisy']
    results = {'title': [], 'purity': [], 'accuracy': []}
    
    for title, (X_red, y) in zip(titles, datasets):
        # Initialize K-Means with n_classes clusters
        k_means = KMeans(n_clusters=n_classes)
        # Fit K-Means to the reduced data
        clusters = k_means.fit_predict(X_red)
        # Calculate the purity of the resulting clusters
        clusters_purity = purity_score(y, clusters)
        # Calculate the accuracy of the resulting clusters
        clusters_accuracy = clustering_accuracy(y, clusters)

        results['title'].append(title)
        results['purity'].append(clusters_purity)
        results['accuracy'].append(clusters_accuracy)
    
    # Save the results to a .txt file
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'metrics.txt'), sep='\t', index=False)
