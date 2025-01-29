import numpy as np
import matplotlib.pyplot as plt
from nn_classification import nn_classification, compute_accuracy, nn_classification_faiss
from prototype_selection import prototype_selection, random_selection
from read_files import read_files


def main(train_images, train_labels , test_images , test_labels):
    M = [1000]
    Iterations = 1

    accuracy = []
    random_accuracy = []
    for iter in range(Iterations):
        for m in M:
            prototypes, prototype_labels = prototype_selection(train_images, train_labels, m)
            randome_prototypes, random_prototype_labels = random_selection(train_images, train_labels, m)

            predicted_labels = nn_classification_faiss(prototypes, prototype_labels, test_images)
            random_predicted_labels = nn_classification_faiss(randome_prototypes, random_prototype_labels, test_images)

            accuracy_value = compute_accuracy(predicted_labels, test_labels)
            random_accuracy_value = compute_accuracy(random_predicted_labels, test_labels)

            accuracy.append(accuracy_value*100)
            random_accuracy.append(random_accuracy_value*100)

            print(f"Accuracy for M={m}: {accuracy_value}")
            print(f"Random Accuracy for M={m}: {random_accuracy_value}")
    
    #Calulate mean,standard deviation and confidence interval
    # mean_accuracy = np.mean(accuracy)
    # mean_random_accuracy = np.mean(random_accuracy)
    # std_accuracy = np.std(accuracy, ddof=1)
    # std_random_accuracy = np.std(random_accuracy,ddof=1)
    # confidence_interval_accuracy = 2.045 * std_accuracy / np.sqrt(len(accuracy))
    # confidence_interval_random_accuracy = 2.045 * std_random_accuracy / np.sqrt(len(random_accuracy))

    # print(f"Mean Accuracy: {mean_accuracy}")
    # print(f"Mean Random Accuracy: {mean_random_accuracy}")
    # print(f"Standard Deviation Accuracy: {std_accuracy}")
    # print(f"Standard Deviation Random Accuracy: {std_random_accuracy}")
    # print(f"Confidence Interval Accuracy: {confidence_interval_accuracy}")
    # print(f"Confidence Interval Random Accuracy: {confidence_interval_random_accuracy}")

    # Plot the results
    # plt.plot(M, accuracy, label="Prototype Selection",color="blue")
    # plt.plot(M, random_accuracy, label="Random Selection",color="red")
    # plt.xlabel("Number of Prototypes(M)")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy vs Number of Prototypes(M)")
    # plt.legend()
    # plt.show()



if __name__ == "__main__":
    train_images, train_labels , test_images , test_labels = read_files(image_dim=28)
    main(train_images, train_labels , test_images , test_labels)