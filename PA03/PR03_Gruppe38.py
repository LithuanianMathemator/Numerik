

import numpy as np
from matplotlib import pyplot as plt
import random

# first excercise

def mean(A):
    return np.mean(A, axis = 0).real

def variance(A):
    return np.var(A, axis = 0).real

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
    
def kmeans(set1=0, set2=1, n=1000):

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

    # slicing for 1000 pictures each
    for i in range(10):

        numbers[i] = numbers[i][:n]

    testset = [numbers[set1], numbers[set2]]

    test = testset[0] + testset[1]

    A, b = dsubspace(test, 2)

    reduced_data = []

    for i in range(len(test)):

        reduced = A.transpose() @ (test[i].flatten() - b)
        reduced_data.append(reduced)

    # initial representants
    b_1 = mean(reduced_data[:(len(test)//2)]).flatten()
    b_2 = mean(reduced_data[(len(test)//2 + 1):]).flatten()

    for _ in range(40):

        C_1 = []
        C_2 = []

        for i in range(len(reduced_data)):

            a_1 = np.linalg.norm(b_1 - reduced_data[i])**2
            a_2 = np.linalg.norm(b_2 - reduced_data[i])**2

            if a_1 < a_2:
                C_1.append(reduced_data[i])
            else:
                C_2.append(reduced_data[i])

        b_1 = mean(C_1)
        b_2 = mean(C_2)

    test_numbers = np.int64(np.floor(np.random.rand(100)*1000))
    '''
    test_numbers = \
        [  0,   3,   6,  35,  50,  84, 106, 110, 111, 111, \
         114, 122, 124, 130, 161, 178, 185, 189, 192, 194, \
         200, 200, 223, 228, 240, 242, 246, 258, 269, 272, \
         279, 292, 317, 361, 363, 366, 366, 384, 412, 415, \
         437, 441, 443, 472, 477, 491, 495, 504, 512, 516, \
         519, 522, 528, 530, 531, 532, 546, 549, 549, 550, \
         554, 580, 581, 587, 599, 605, 609, 614, 642, 654, \
         666, 685, 688, 690, 710, 720, 733, 736, 738, 741, \
         762, 770, 774, 782, 783, 822, 841, 849, 873, 875, \
         877, 894, 895, 899, 901, 906, 917, 918, 924, 960]
    '''
    
    set1_right = [[],[]]
    set1_wrong = [[],[]]
    set2_right = [[],[]]
    set2_wrong = [[],[]]

    for i in range(100):
        digit1 = (A.T @ (numbers[set1][test_numbers[i]].flatten() - b)).real
        digit2 = (A.T @ (numbers[set2][test_numbers[i]].flatten() - b)).real

        d1_right = np.linalg.norm(b_1 - digit1)
        d1_wrong = np.linalg.norm(b_2 - digit1)
        d2_right = np.linalg.norm(b_2 - digit2)
        d2_wrong = np.linalg.norm(b_1 - digit2)

        if d1_right < d1_wrong:
            set1_right[0].append(digit1[0])
            set1_right[1].append(digit1[1])
        else:
            set1_wrong[0].append(digit1[0])
            set1_wrong[1].append(digit1[1])
        
        if d2_right < d2_wrong:
            set2_right[0].append(digit2[0])
            set2_right[1].append(digit2[1])
        else:
            set2_wrong[0].append(digit2[0])
            set2_wrong[1].append(digit2[1])

    n1_right = len(set1_right[0])
    n1_wrong = 100 - n1_right
    n2_right = len(set2_right[0])
    n2_wrong = 100 - n2_right
    

    print(str(set1) + ' richtig erkannt: '+ str(n1_right)+'/100')
    print(str(set2) + ' richtig erkannt: '+ str(n2_right)+'/100')

    np.seterr(divide='ignore')
    plt.scatter(set1_right[0], set1_right[1], color='red' , label=str(set1)+' korrekt')
    plt.scatter(set1_wrong[0], set1_wrong[1], color='blue', label=str(set1)+' falsch' )
    plt.scatter(set2_right[0], set2_right[1], color='blue', label=str(set2)+' korrekt', marker='x')
    plt.scatter(set2_wrong[0], set2_wrong[1], color='red' , label=str(set2)+' falsch' , marker='x')
    plt.scatter(b_1[0], b_1[1], color='black', linewidths=2)
    plt.scatter(b_2[0], b_2[1], color='black', linewidths=2, marker='x')
    plt.axline((b_1+b_2)/2, slope=-np.divide((b_1[0]-b_2[0]), (b_1[1]-b_2[1])), color='black')
    plt.title("Scatter für Samples von " + str(set1) + " und " + str(set2) + "!")
    plt.legend()
    plt.table(cellText=[[''                       , 'korrekt'    , 'falsch'     ],  \
                        ['Zuordnung zu '+str(set1), str(n1_right), str(n1_wrong)],  \
                        ['Zuordnung zu '+str(set2), str(n2_right), str(n2_wrong)]], \
              cellLoc='center', \
              bbox=[0, -0.5, 1, 0.3])
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    testfirst()
    testsecond()
    testthird()
    kmeans()
