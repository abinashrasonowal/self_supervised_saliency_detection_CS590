from sklearn.cluster import KMeans
import numpy as np

def cluster_patches(embeddings, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def refine_segmentation(cluster_labels, image_shape):
    # Example of refining the segmentation output, like reshaping or morphological ops
    # This is where you'd implement additional post-processing if necessary.
    return cluster_labels.reshape(image_shape)
