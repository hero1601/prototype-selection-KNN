import numpy as np

def read_files(image_dim=28):
    train_image_path = 'train-images.idx3-ubyte'
    train_labels_path = 'train-labels.idx1-ubyte'
    test_image_path = 't10k-images.idx3-ubyte'
    test_labels_path = 't10k-labels.idx1-ubyte'

    with open(train_image_path, 'rb') as file:
        train_data = np.frombuffer(file.read(), dtype = np.uint8)

    with open(train_labels_path, 'rb') as file:
        train_labels = np.frombuffer(file.read(), dtype = np.uint8)

    with open(test_image_path, 'rb') as file:
        test_data = np.frombuffer(file.read(), dtype = np.uint8)

    with open(test_labels_path, 'rb') as file:
        test_labels = np.frombuffer(file.read(), dtype = np.uint8)

    train_labels = train_labels[8:]
    test_labels = test_labels[8:]
    train_images = train_data[16:].reshape(-1,image_dim*image_dim)
    test_images = test_data[16:].reshape(-1,image_dim*image_dim)

    return train_images, train_labels, test_images, test_labels