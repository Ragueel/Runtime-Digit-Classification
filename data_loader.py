from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# Prepares data for training and testing

DATA_PATH = './data/compressed/'


# Gets all images with their labels
def get_data():
    data = []
    labels = []

    for file in os.listdir(DATA_PATH):
        data.append(get_numpy_from_path(DATA_PATH + file))
        labels.append(int(file[0]))
    data = np.array(data)
    data = data.reshape((len(data), 256, 256))
    return (data, labels)


# Create softmax
def get_softmax_label(num):
    array = np.array(np.zeros(10))
    array[num] = 1
    return array


# Converts image at path to numpy array
def get_numpy_from_path(path):
    image = Image.open(path)

    data = list(image.getdata())
    # print(data[0])
    data = np.array(
        [((arr[0] + arr[1] + arr[2]) / 3) if ((arr[0] + arr[1] + arr[2]) / 3) < 200 else .0 for arr in data])

    array = data / 255
    return array


# Visualize given array of x
def visualize(x, y, output=None):
    pix = x

    pix = pix.reshape((256, 256))

    plt.title('Label is {0}, output is :{1}'.format(y, output))
    plt.imshow(pix, cmap='gray')
    plt.show()
