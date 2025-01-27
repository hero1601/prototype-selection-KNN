import numpy as np
from scipy.spatial.distance import cdist

def prototype_selection(images,labels,M,max_iter = 10,tolerance = 1e-4):
    
    images_size,images_features = images.shape
    
    # Randomly select M images from the dataset
    random_indices = np.random.choice(images_size,M,replace=False)
    centroids = images[random_indices]

    for iter in range(max_iter):

        # Assign each image to the closest prototype
        distances = cdist(images,centroids)
        closest_centroids = np.argmin(distances,axis=1)
        
        # Update the prototypes
        new_centroids = np.zeros((M,images_features))
        for i in range(M):
            new_centroids[i] = np.mean(images[closest_centroids==i],axis=0)
        
        # Check for convergence
        if np.linalg.norm(new_centroids-centroids) < tolerance:
            break
        centroids = new_centroids

    prototypes = []
    prototype_labels = []

    for j in range(M):
        # Find the data points closest to each centroid
        closest_point = np.argmin(cdist(centroids[j].reshape(1, -1), images), axis=1)[0]
        prototypes.append(images[closest_point])
        prototype_labels.append(labels[closest_point])

    return np.array(prototypes),np.array(prototype_labels)


def random_selection(images,labels,M):
    images_size,images_features = images.shape
    random_indices = np.random.choice(images_size,M,replace=False)
    return images[random_indices],labels[random_indices]

    