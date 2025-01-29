import faiss
import numpy as np
from scipy.spatial.distance import cdist

def prototype_selection(images,labels,M,max_iter = 10,tolerance = 1e-4):
    
    images_size,images_features = images.shape
    
    # Randomly select M images from the dataset
    random_indices = np.random.choice(images_size,M,replace=False)
    centroids = images[random_indices].astype(np.float64)

    index = faiss.IndexFlatL2(images_features)
    index.add(centroids.astype(np.float64))

    for iter in range(max_iter):

        # Assign each image to the closest prototype
        distances, closest_centroids = index.search(images.astype(np.float64),1)
        
        # Update the prototypes
        new_centroids = np.zeros((M,images_features),dtype=np.float64)
        for i in range(M):
            new_centroids[i] = np.mean(images[closest_centroids[:,0] == i],axis=0)
        
        # print(f"Iteration {iter+1}: {np.linalg.norm(new_centroids-centroids)}")
        
        # Check for convergence
        if np.linalg.norm(new_centroids-centroids) < tolerance:
            break
        centroids = new_centroids
        index.reset()
        index.add(centroids.astype(np.float64))

    # Add images to the FAISS index
    index.reset()
    index.add(images)

    # Perform search for nearest image for each centroid
    distances, indices = index.search(centroids, 1)
    closest_images = indices.flatten()

    # Gather the closest prototypes and their labels
    prototypes = images[closest_images]
    prototype_labels = labels[closest_images]

    return np.array(prototypes),np.array(prototype_labels)


def random_selection(images,labels,M):
    images_size,images_features = images.shape
    random_indices = np.random.choice(images_size,M,replace=False)
    return images[random_indices],labels[random_indices]

    