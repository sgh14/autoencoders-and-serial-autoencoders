
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
'''
Purity measures the frequency of data belonging to the same cluster sharing the
same class label, while Accuracy measures the frequency of data from the same
class appearing in a single cluster.
'''

def purity_score(y_true, y_pred):
    # Compute confusion matrix
    matrix = confusion_matrix(y_true, y_pred)

    # Find the maximum values in each column (which represents clusters)
    max_in_cols = np.amax(matrix, axis=0)

    # Sum the maximum values found
    purity = np.sum(max_in_cols) / np.sum(matrix)

    return purity


def clustering_accuracy(y_true, y_pred):
    # Compute the confusion matrix
    matrix = confusion_matrix(y_true, y_pred)

    # Use the linear_sum_assignment method to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(-matrix)

    # Calculate the accuracy using the optimal assignment
    accuracy = matrix[row_ind, col_ind].sum() / np.sum(matrix)

    return accuracy


def compute_metrics(X_red, X, Y, n_classes):
    """
    Función para calcular la pureza y el índice de Rand de los clusters de K-Means
    sobre los datos reducidos y, también, el accuracy de las clasificación de KNN
    con 1 vecino de los datos reducidos.
    """
    # Inicializamos la clase de K-Means con n_classes clusters
    k_means = KMeans(n_clusters=n_classes)
    # Ajustamos K-Means a los datos reducidos
    clusters = k_means.fit_predict(X_red)
    # Calculamos la pureza de los clusters resultantes
    clusters_purity = purity_score(Y, clusters)
    # Calculamos el índice de Rand de los clusters resultantes
    clusters_rand_index = metrics.cluster.rand_score(Y, clusters)

    # Inicializamos la clase de KNN con 1 vecino
    knn = KNeighborsClassifier(n_neighbors=1)
    # Definimos la cantidad de datos para entrenamiento y para validación
    fraction = (70*(X_red.shape[0]))//100
    # Dividimos los datos en un conjunto de entrenamiento y otro de validación
    X_red_train, X_red_test = np.split(X_red, [fraction])
    Y_train, Y_test = np.split(Y, [fraction])
    # Ajustamos KNN a los datos reducidos
    knn = knn.fit(X_red_train, Y_train)
    # Realizamos las predicciones sobre el cojunto de validación
    Y_pred = knn.predict(X_red_test)
    # Calculamos el accuracy de las predicciones
    classification_accuracy = metrics.accuracy_score(Y_test, Y_pred)

    return clusters_purity, clusters_rand_index, classification_accuracy