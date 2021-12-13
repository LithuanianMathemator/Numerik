

import numpy as np
from matplotlib import pyplot as plt
import random

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

    fig = plt.figure(figsize=(10,10))

    for i in range(1, 21):

        if i < 11:
            fig.add_subplot(4, 5, i)
            plt.imshow(mean(numbers[i-1]), cmap = 'gray', \
            interpolation = 'nearest')
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

# second excercise

def eigenvalues(A):

    B = A - mean(A)

    x = len(B)
    y = len(B[0])
    z = len(B[0][0])

    Y = (B.flatten()).reshape(x, y*z)

    Y = Y.transpose()

    S = Y @ Y.transpose()

    return np.linalg.eig(S)

def singular(A):

    B = A - mean(A)

    x = len(B)
    y = len(B[0])
    z = len(B[0][0])

    Y = (B.flatten()).reshape(x, y*z)

    Y = Y.transpose()

    return np.linalg.svd(Y)

def testsecond():

    imgs = np.fromfile('train-images-idx3-ubyte', dtype=np.uint8)
    imgs = np.reshape(imgs[16:], [-1, 28, 28])
    imgs = np.int64(imgs[:1000])

    eigenvs = np.real(np.sort(eigenvalues(imgs)[0])[::-1])[:50]

    singularvs = np.square(np.sort(np.real(singular(imgs)[1]))[::-1])[:50]

    fig = plt.figure(figsize=(10,5))
    fig.add_subplot(1, 2, 1)
    plt.plot(eigenvs)
    plt.title("Eigenwerte!")
    fig.add_subplot(1, 2, 2)
    plt.plot(singularvs)
    plt.title("Quadrierte Singulärwerte!")

    plt.tight_layout()
    plt.show()

# third excercise

def dsubspace(A, d):
    # find ideal subspace with dimension d

    # slicing matrix to get first d eigenvectors
    eigmatrix = eigenvalues(A)[1]
    eigmatrix = eigmatrix.transpose()
    eigmatrix = eigmatrix[:d]
    eigmatrix = eigmatrix.transpose()

    # b is mean from first function
    b = mean(A).flatten()

    return (eigmatrix,b)

def projection(A, b, x):

    proj = A @ A.transpose() @ (x - b) + b

    return proj

def testthird():

    imgs = np.fromfile('train-images-idx3-ubyte', dtype=np.uint8)
    imgs = np.reshape(imgs[16:], [-1, 28, 28])

    labs = np.fromfile('train-labels-idx1-ubyte', dtype=np.uint8)
    labs = labs[8:]

    # first test: principal components

    img_5 = np.int64(imgs[:1000])

    A, b = dsubspace(img_5, 5)
    b = b.reshape(28,28)

    fig = plt.figure(figsize=(12,5))

    for i in range(5):
        fig.add_subplot(1, 6, i+1)
        plt.imshow((np.real(A).transpose())[i].reshape(28,28), cmap = 'gray', \
        interpolation = 'nearest')
        plt.title("Hauptkomponente " + str(i+1) + "!")
        plt.axis('off')

    fig.add_subplot(1, 6, 6)
    plt.imshow(b, cmap = 'gray', \
    interpolation = 'nearest')
    plt.title("Mittelwert!")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # second test: projection

    tests = [np.int64(random.choice(img_5)) for i in range(4)]

    fig = plt.figure(figsize=(12,5))

    for i in range(4):
        fig.add_subplot(2, 4, i+1)
        plt.imshow(tests[i], cmap = 'gray', \
        interpolation = 'nearest')
        plt.title("Testbild " + str(i+1) + "!")
        plt.axis('off')

        proj = projection(A, b.flatten(), tests[i].flatten())
        proj = proj.reshape(28,28)
        fig.add_subplot(2, 4, i+5)
        plt.imshow(np.real(proj), cmap = 'gray', \
        interpolation = 'nearest')
        plt.title("Projektion " + str(i+1) + "!")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
