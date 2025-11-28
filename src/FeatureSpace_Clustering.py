import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.thresholding import _cross_entropy
import cv2
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from numpy.linalg import norm

def remove_outliers(objects):
    """
    Removes outliers from input data using Z-score method.

    Parameters:
    - objects: numpy array or list, input data

    Returns:
    - outliers: numpy array, indices of outliers in the input data
    """
    from scipy import stats  # Importing within the function to avoid global imports
    import numpy as np

    z = np.abs(stats.zscore(objects))  # Compute Z-scores for the input data
    threshold = 2  # Threshold for outlier detection
    outliers = np.where(z > threshold)  # Find indices where Z-score exceeds the threshold
    return outliers

def kmeans_clustering(list_of_objects, isPlot=0, certainty=0.60, toSave=0):
    """
    Perform k-means clustering on a list of objects.

    Parameters:
    - list_of_objects: list of objects with image attribute
    - isPlot: flag to determine if plots should be displayed (default: 0)
    - certainty: threshold for certainty level (default: 0.60)
    - toSave: flag to determine if results should be saved (default: 0)

    Returns:
    - list_of_objects: updated list of objects with labels assigned
    - best_n_clusters: optimal number of clusters determined by silhouette score
    """
    nn = len(list_of_objects)
    x = np.zeros((nn, 2))
    image_size = list_of_objects[0].image.shape
    center = image_size[0] // 2
    null_ids = []

    for i in range(nn):
        img = list_of_objects[i].image
        img_denoised = np.uint8(img)
        img_denoised = cv2.fastNlMeansDenoising(img_denoised, None, 3, 21, 3)

        thresholds = np.arange(np.min(img_denoised) + 1.3, np.max(img_denoised) - 1.3, 0.1)
        entropies = [_cross_entropy(img_denoised, t) for t in thresholds]

        if len(entropies) == 0:
            null_ids.append(i)
            continue

        x[i][1] = list_of_objects[i].area
        x[i][0] = np.argmin(entropies) / len(thresholds)

    # Remove null objects
    id = np.arange(0, len(list_of_objects))
    non_null = [i for i in id if i not in null_ids]
    list_of_objects = [list_of_objects[i] for i in non_null]
    x = x[non_null]
    nn = len(list_of_objects)

    # Normalize features
    mean_of_rows = np.mean(x, axis=0)
    min_of_rows = np.min(x, axis=0)
    max_of_rows = np.max(x, axis=0)
    d = max_of_rows - min_of_rows
    normalized_x = (x - mean_of_rows) / d

    # Remove outliers
    outlier_idx, _ = remove_outliers(normalized_x)
    id = np.arange(0, len(list_of_objects))
    non_outlier = [i for i in id if i not in outlier_idx]
    list_of_objects = [list_of_objects[i] for i in non_outlier]
    normalized_x = normalized_x[non_outlier]
    x = x[non_outlier]
    nn = len(list_of_objects)

    # Determine optimal number of clusters using silhouette score
    sil_score_max = -1
    from sklearn.cluster import KMeans
    sil_score = []
    for n_clusters in range(2, 10):
        model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
        labels = model.fit_predict(normalized_x)
        sil_score.append(silhouette_score(normalized_x, labels))

        if sil_score[-1] > sil_score_max:
            sil_score_max = sil_score[-1]
            best_n_clusters = n_clusters

    print("Best number of clusters =", best_n_clusters)

    # Plot silhouette scores
    if isPlot == 1:
        fig1 = plt.figure()
        plt.scatter(range(2, 10), sil_score, color='black')
        plt.plot(range(2, 10), sil_score, color='gray')
        plt.show()

    # Perform k-means clustering with the best number of clusters
    model = KMeans(n_clusters=best_n_clusters, init='k-means++', max_iter=100, n_init=1)
    labels = model.fit_predict(normalized_x)
    list_of_objects = assign_label(list_of_objects, labels, normalized_x, model, certainty)

    # Plot clusters if required
    if isPlot == 1:
        fig = plt.figure()
        for i in range(nn):
            if list_of_objects[i].label == 0:
                cluster0 = plt.scatter(normalized_x[i][0], normalized_x[i][1], marker=',', c='0.6')
            elif list_of_objects[i].label == 1:
                cluster1 = plt.scatter(normalized_x[i][0], normalized_x[i][1], marker='*', c='0.6')
            else:
                rest = plt.scatter(normalized_x[i][0], normalized_x[i][1], marker='v', c='0.2')

        plt.legend((cluster0, cluster1, rest), ('cluster 0', 'cluster 1', 'borderline'))
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1])
        plt.show()

    # Save results if required
    if toSave:
        fig1.savefig("Raw_Figs/silhouette_score.svg", format='svg')
        fig.savefig("Raw_Figs/different_clusters.svg", format='svg')


    return list_of_objects, best_n_clusters

def assign_label(list_of_objects, labels, normalized_x, km, certainty):
    """
    Assign labels to objects based on k-means clustering results and certainty threshold.

    Parameters:
    - list_of_objects: list of objects with features attribute
    - labels: cluster labels assigned by k-means clustering
    - normalized_x: normalized features of objects
    - km: k-means model used for clustering
    - certainty: threshold for certainty level

    Returns:
    - list_of_objects: updated list of objects with assigned labels
    """
    nn = len(list_of_objects)
    centroids = km.cluster_centers_
    n_clusters = len(centroids)
    distance = np.zeros((nn, n_clusters))

    # Calculate distances between objects and cluster centroids
    for k in range(n_clusters):
        row_norm = norm(normalized_x - centroids[k, :], axis=1)
        distance[:, k] = np.square(row_norm)

    # Assign labels based on distance ratios and certainty threshold
    for i in range(nn):
        distance[i, :] / np.max(distance, axis=0)
        list_of_objects[i].features = normalized_x[i]

        # Determine label based on distance ratios and certainty threshold
        if labels[i] == 0 and (distance[i][0] / distance[i][1] < certainty):
            list_of_objects[i].label = 0
        elif labels[i] == 1 and (distance[i][1] / distance[i][0] < certainty):
            list_of_objects[i].label = 1
        else:
            list_of_objects[i].label = 2

    return list_of_objects


def objects_per_cluster(list_of_objects, cluster, isPlot=0):
    """
    Get objects belonging to a specific cluster.

    Parameters:
    list_of_objects (list): List of objects to filter.
    cluster (int): Cluster label to filter objects.
    isPlot (int, optional): Flag to enable plotting. Default is 0.
    toSave (int, optional): Flag to enable saving. Default is 0.

    Returns:
    list: List of objects belonging to the specified cluster.
    """
    # Filter objects based on cluster label
    objects_in_cluster = [d for d in list_of_objects if d.label == cluster]

    # Plot objects if requested
    return objects_in_cluster


def borderline_objects(list_of_objects, cluster, isPlot=0):
    """
    Get borderline objects belonging to a specific cluster.

    Parameters:
    list_of_objects (list): List of objects to filter.
    cluster (int): Cluster label to filter objects.
    isPlot (int, optional): Flag to enable plotting. Default is 0.
    toSave (int, optional): Flag to enable saving. Default is 0.

    Returns:
    list: List of borderline objects belonging to the specified cluster.
    """
    # Filter objects based on cluster label
    objects_in_cluster = [d for d in list_of_objects if d.label == cluster]

    # Plot objects if requested
    return objects_in_cluster


