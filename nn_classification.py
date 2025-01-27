import numpy as np
from scipy.spatial.distance import cdist


def nn_classification(prototypes,prototypes_label,test_images):

    # Calculate the Euclidean distance between each test image and the prototypes
    distances = cdist(test_images,prototypes)
    
    # Find the prototype with the smallest distance for each test image
    closest_prototypes = np.argmin(distances,axis=1)
    
    # Assign the label of the closest prototype to the test image
    predicted_labels = prototypes_label[closest_prototypes]
    
    return predicted_labels

def compute_accuracy(predicted_labels, true_labels):
    return np.mean(predicted_labels == true_labels)