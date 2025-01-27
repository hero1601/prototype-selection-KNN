import numpy as np
import matplotlib.pyplot as plt
from nn_classification import nn_classification, compute_accuracy, nn_classification_faiss
from prototype_selection import prototype_selection, random_selection
from read_files import read_files


def main():
    # Call the function to load data
    train_images, train_labels , test_images , test_labels = read_files(image_dim=28)

    M = [100,500,1000,3000,5000,10000]

    accuracy = []
    random_accuracy = []

    for m in M:
        prototypes, prototype_labels = prototype_selection(train_images, train_labels, m)
        randome_prototypes, random_prototype_labels = random_selection(train_images, train_labels, m)

        predicted_labels = nn_classification_faiss(prototypes, prototype_labels, test_images)
        random_predicted_labels = nn_classification_faiss(randome_prototypes, random_prototype_labels, test_images)

        accuracy.append(compute_accuracy(predicted_labels, test_labels))
        random_accuracy.append(compute_accuracy(random_predicted_labels, test_labels))

    plt.plot(M, accuracy, label="Prototype Selection",color="blue")
    plt.plot(M, random_accuracy, label="Random Selection",color="red")
    plt.xlabel("Number of Prototypes(M)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Prototypes(M)")
    plt.legend()
    plt.show()

    # print(f"Prototypes shape: {prototypes.shape}")
    # plt.imshow(prototypes.reshape(-1,28,28)[5], cmap="gray")
    # plt.title("First Image")
    # plt.show()
    # print(f"Prototype Labels: {prototype_labels[5]}")


if __name__ == "__main__":
    main()