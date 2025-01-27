import numpy as np
from scipy.spatial.distance import cdist
import faiss


def nn_classification(prototypes,prototypes_label,test_images):

    # Calculate the Euclidean distance between each test image and the prototypes
    distances = cdist(test_images,prototypes)
    
    # Find the prototype with the smallest distance for each test image
    closest_prototypes = np.argmin(distances,axis=1)
    
    # Assign the label of the closest prototype to the test image
    predicted_labels = prototypes_label[closest_prototypes]
    
    return predicted_labels

def nn_classification_faiss(prototypes,prototypes_label,test_images):
    
    # Create an index using the prototypes
    index = faiss.IndexFlatL2(prototypes.shape[1])
    index.add(prototypes)

    # Find the k closest prototypes for each test image
    k = 1
    distances, closest_prototypes = index.search(test_images,k)

    # Assign the label of the closest prototype to the test image
    predicted_labels = prototypes_label[closest_prototypes[:,0]]

    return predicted_labels

def compute_accuracy(predicted_labels, true_labels):
    return np.mean(predicted_labels == true_labels)