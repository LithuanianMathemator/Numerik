
import numpy as np
from matplotlib import pyplot as plt
import random
from tabulate import tabulate

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

# fourth excercise

def kmeans(n, dataset):

    # extracting data from labeled dataset

    data = []

    for i in range(len(dataset)):
        data.append(dataset[i][0])

    A, B = dsubspace(data, 2)

    reduced_data = []

    for i in range(len(dataset)):

        reduced = A.transpose() @ (dataset[i][0].flatten() - B)
        reduced_data.append((reduced, dataset[i][1]))

    # initial representants
    b_1, b_2 = represent(reduced_data)

    j = 0

    while j < n:

        C_1 = []
        C_2 = []

        for i in range(len(reduced_data)):

            a_1 = np.linalg.norm(b_1 - reduced_data[i][0])**2
            a_2 = np.linalg.norm(b_2 - reduced_data[i][0])**2

            if a_1 < a_2:
                C_1.append(reduced_data[i])
            else:
                C_2.append(reduced_data[i])

        meanc1 = [C_1[i][0] for i in range(len(C_1))]
        meanc2 = [C_2[i][0] for i in range(len(C_2))]

        b_1 = mean(meanc1)
        b_2 = mean(meanc2)

        j += 1

    x_1, x_2, y_1, y_2, l_1, l_2 = [], [], [], [], [], []

    for i in range(len(C_1)):
        x_1.append(C_1[i][0][0])
        y_1.append(C_1[i][0][1])
        l_1.append(C_1[i][1])

    for i in range(len(C_2)):
        x_2.append(C_2[i][0][0])
        y_2.append(C_2[i][0][1])
        l_2.append(C_2[i][1])

    return (np.real(x_1), np.real(y_1), np.real(x_2), np.real(y_2), l_1, l_2)


def kmeanstest(n, a, b):

    imgs = np.fromfile('train-images-idx3-ubyte', dtype=np.uint8)
    imgs = np.reshape(imgs[16:], [-1, 28, 28])

    labs = np.fromfile('train-labels-idx1-ubyte', dtype=np.uint8)
    labs = labs[8:]

    # list for pictures of each number
    numbers = [[] for i in range(10)]

    N = len(imgs)

    # distributing pictures
    for i in range(N):

        numbers[labs[i]].append((imgs[i], labs[i]))

    # slicing for 1000 pictures each
    for i in range(10):

        numbers[i] = numbers[i][:1000]

    test = numbers[a] + numbers[b]

    x_1, y_1, x_2, y_2, l_1, l_2 = kmeans(n, test)

    plt.scatter(x_1, y_1, color = 'red')
    plt.scatter(x_2, y_2, color = 'blue')
    plt.title("Scatter für Samples von " + str(a) + \
    " und " + str(b) + "!")
    plt.show()

    # classifying 100 pictures, giving out table:

    second = numbers[a][:100] + numbers[b][:100]

    x_1, y_1, x_2, y_2, l_1, l_2 = kmeans(n, second)

    correct_1_count = 0
    correct_2_count = 0
    false_1_count = 0
    false_2_count = 0

    for i in range(len(l_1)):

        if l_1[i] == a:
            correct_1_count += 1
        else:
            false_2_count += 1

    for i in range(len(l_2)):

        if l_2[i] == b:
            correct_2_count += 1
        else:
            false_1_count += 1

    print(tabulate([[str(a), correct_1_count, false_1_count], \
    [str(b), correct_2_count, false_2_count]], \
    headers=['Ziffer', 'Richtig', 'Falsch']))


def represent(reduced_data):

    reduced_data_1 = [reduced_data[i][0] for i in range(len(reduced_data))]

    b_1 = mean(reduced_data_1[:(len(reduced_data)//2)]).flatten()
    b_2 = mean(reduced_data_1[(len(reduced_data)//2 + 1):]).flatten()

    return b_1, b_2
