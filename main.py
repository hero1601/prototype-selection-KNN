import numpy as np
import matplotlib.pyplot as plt
from prototype_selection import prototype_selection
from read_files import read_files


def main():
    # Call the function to load data
    train_images, train_labels , test_images , test_labels = read_files(image_dim=28)

    prototypes, prototype_labels = prototype_selection(train_images, train_labels, 10)

    # print(f"Prototypes shape: {prototypes.shape}")
    # plt.imshow(prototypes.reshape(-1,28,28)[5], cmap="gray")
    # plt.title("First Image")
    # plt.show()
    # print(f"Prototype Labels: {prototype_labels[5]}")


if __name__ == "__main__":
    main()