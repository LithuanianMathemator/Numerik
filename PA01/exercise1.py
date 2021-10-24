import skimage as sk
import numpy as np
import matplotlib as plt
from skimage import io


def func1mean(A, weight = 1):
    # type of A: numpy array
    # weight: 0 = random, 1 = equal, 2 = test of 100 random matrices

    n = len(A)
    m = len(A[0])

    W = np.random.rand(n, m)
    # 'weight matrix'

    if weight == 0:

        summe = 0
        for i in range(0, n):
            summe += sum(W[i])

        for i in range(0, n):
            for j in range(0, m):
                W[i][j] = W[i][j]/summe

        meanA = 0
        for i in range(0, n):
            for j in range(0, m):
                meanA += W[i][j]*A[i][j]

        return meanA

    elif weight == 1:

        meanA = 0
        for i in range(0, n):
            for j in range(0, m):
                meanA += (1/(n*m))*A[i][j]

        return meanA

    elif weight == 2:

        M = []
        N = []

        for i in range(100):

            P = np.random.randint(1, 100, size=2)
            A = np.random.randint(20, size=(P[0], P[1]))

            M.append(abs(np.mean(A) - func1mean(A, 0)))
            N.append(abs(np.mean(A) - func1mean(A, 1)))

        return max(M), max(N)

    else:

        return "C'mon bro!"


def func1median(A, weight = 1):
    # type of A: numpy array
    # weight: 0 = random, 1 = equal, 2 = test of 100 random matrices

    n = len(A)
    m = len(A[0])

    W = np.random.rand(n, m)
    # 'weight matrix'

    if weight == 0:

        summe = 0
        for i in range(0, n):
            summe += sum(W[i])

        for i in range(0, n):
            for j in range(0, m):
                W[i][j] = W[i][j]/summe

        Wflat = np.array(W).flatten()
        Aflat = np.sort(np.array(A).flatten())

        s = 0
        k = 0
        while s < 0.5:
             s += Wflat[k]
             k += 1

        # gewichteter Median aus Signalverarbeitung
        if s == 0.5:
            return (Aflat[k] + Aflat[k+1])/2
        else:
            if k > n*m:
                return Aflat[k-1]
            else:
                return Aflat[k]

    elif weight == 1:

        Aflat = np.sort(np.array(A).flatten())

        print(Aflat)

        if (n*m) % 2 == 0:
            return (Aflat[(n*m)//2-1] + Aflat[(n*m)//2])/2
        else:
            return Aflat[(n*m)//2]


    elif weight == 2:

        M = []
        N = []

        for i in range(100):

            P = np.random.randint(3, 100, size=2)
            A = np.random.randint(20, size=(P[0], P[1]))

            M.append(abs(np.median(A) - func1median(A, 0)))
            N.append(abs(np.median(A) - func1median(A, 1)))

        return max(M), max(N)

    else:

        return "C'mon bro!"

