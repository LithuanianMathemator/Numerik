
import numpy as np
from matplotlib import pyplot as plt

# first excercise

def mean(A):
    return np.mean(A, axis = 0)

def variance(A):
    return np.var(A, axis = 0)

def testfirst():

    imgs = np.fromfile('train-images-idx3-ubyte', dtype=np.uint8)
    imgs = np.reshape(imgs[16:], [-1, 28, 28])

    labs = np.fromfile('train-labels-idx1-ubyte', dtype=np.uint8)
    labs = labs[8:]

    # list for pictures of each number
    numbers = [[] for i in range(10)]

    N = len(imgs)

    # distributing pictures
    for i in range(N):

        numbers[labs[i]].append(imgs[i])

    # slicing for 100 pictures each
    for i in range(10):

        numbers[i] = numbers[i][:100]

    plt.rcParams["figure.figsize"] = (20,3)

    fig = plt.figure(figsize=(10,10))

    for i in range(1, 21):

        if i < 11:
            fig.add_subplot(4, 5, i)
            plt.imshow(mean(numbers[i-1]), cmap = 'gray', interpolation = 'nearest')
            plt.title("Mittelwert für " + str(i) + "!")
            plt.axis('off')
        else:
            fig.add_subplot(4, 5, i)
            plt.imshow(variance(numbers[i-11]), cmap = 'gray', \
            interpolation = 'nearest')
            plt.title("Varianz für " + str(i) + "!")
            plt.axis('off')

    plt.tight_layout()
    plt.show()
