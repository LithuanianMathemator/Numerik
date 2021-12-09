
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt

def mean(A):

    # A as set of pictures

    N = len(A)

    pxl = np.copy(np.int64(A[0]))

    for i in range(1, N):

        pxl += np.int64(A[i])

    b = (1/N) * pxl

    return b

def variance(A):

    # A as set of pictures

    N = len(A)

    # mean for squared difference
    b = mean(A)

    pxl = np.square(np.int64(A[0]) - b)

    for i in range(1, N):

        pxl += np.square(np.int64(A[i]) - b)

    V = (1/N) * pxl

    return V

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

    for i in range(10):

        plt.imshow(mean(numbers[i]), cmap = 'gray')
        plt.title("Mittelwert für " + str(i) + "!")
        plt.show()

        plt.imshow(variance(numbers[i]), cmap = 'gray')
        plt.title("Varianz für " + str(i) + "!")
        plt.show()

        
