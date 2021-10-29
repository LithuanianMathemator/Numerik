import skimage as sk
import numpy as np
import matplotlib as plt
from skimage import io


##################### 1. #####################

def func1mean(A, W, weight):
    # type of A: numpy array
    # W: weight matrix as numpy array
    # weight: 0 = random, 1 = equal, 2 = test of 100 random matrices

    n = len(A)
    m = len(A[0])

    if weight == 0: 

        summe = 0
        for i in range(0, n):
            summe += sum(W[i])

        for i in range(0, n):
            for j in range(0, m):
                W[i][j] = float(W[i][j])/float(summe) 
        
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

    elif weight == 2: #test of 100 matrices

        M = []
        N = []

        for i in range(100):

            P = np.random.randint(1, 100, size=2)
            A = np.random.randint(20, size=(P[0], P[1]))
            W = np.random.randint(20, size=(P[0], P[1])) #?

            M.append(abs(np.mean(A) - func1mean(A, W, 0)))
            N.append(abs(np.mean(A) - func1mean(A, W, 1)))

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
    
##################### 2. #####################    
    
def gaussianweight(s, sigma):
    lenW = float(0.5 * (s-1))
    vector = np.linspace(-lenW, lenW, 2*s+1)
    gaussian_vector = np.exp(-0.5 * np.square(vector)/np.square(sigma))
    W = np.outer(gaussian_vector, gaussian_vector)
    return W

def meanfilter(s, W, weight, image):
    #weight: 0 = gaussian, 1 = equal
    
    picture = io.imread(image)
    
    n = len(picture)
    m = len(picture[0])
    
    padded = np.zeros((picture.shape[0]+2*s, picture.shape[1]+2*s))
    padded[s:-s,s:-s] = picture

    newpicture = np.zeros((picture.shape[0],picture.shape[1]))

    if weight == 1: #equal
    
        for i in range(s,n-s):
            for j in range(s, m-s):
                B = padded[i-s:i+s+1, j-s:j+s+1]
                newpicture[i][j] = func1mean(B,W,0)
    
        plt.imshow(newpicture, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    if weight == 0: #gaussian 
        sigma = 3 
        W_gauss = gaussianweight(s,sigma)
        
        for i in range(s,n-s):
            for j in range(s, m-s):
                
                B = padded[i-s:i+s+1, j-s:j+s+1]
                newpicture[i][j] = func1mean(B,W_gauss,0)
        
        plt.imshow(newpicture, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        plt.imshow(W_gauss, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

W = np.random.uniform(20, size=(5, 5))

# image filtering: 0 = gaussian, 1 = equal    
meanfilter(2, W, 0, image = "B1.png")
meanfilter(2, W, 1, image = "B1.png")

meanfilter(2, W, 0, image = "B2.png")
meanfilter(2, W, 1, image = "B2.png")

meanfilter(2, W, 0, image = "C.png")
meanfilter(2, W, 1, image = "C.png")


##################### 3. #####################    
    
    
def func1medianfilter(s, weight=1, bild='B1.png'):

    picture = io.imread(bild)

    n = len(picture)
    m = len(picture[0])

    newpicture = np.zeros(shape=(n, m))

    if weight == 1:

        for i in range(n):
            for j in range(m):

                Eps = []

                for x in range(-s, s+1, 1):
                    for y in range(-s, s+1, 1):

                        a = i + x
                        b = j + y

                        # continuation by mirroring
                        if a < 0 or b < 0 or a > (n-1) or b > (m-1):

                            if a < 0 and b >= 0 and b <= (m-1):
                                Eps.append(picture[a - (2*x)][b])
                            elif a >= 0 and b < 0 and a <= (n-1):
                                Eps.append(picture[a][b - (2*y)])
                            elif a > n and b >= 0 and b <= (m-1):
                                Eps.append(picture[a - (2*x)][b])
                            elif a >= 0 and b > (m-1) and a <= (n-1):
                                Eps.append(picture[a][b - (2*y)])
                            elif a < 0 and b < 0:
                                Eps.append(picture[a - (2*x)][b - (2*y)])
                            elif a < 0 and b > (m-1):
                                Eps.append(picture[a - (2*x)][b - (2*y)])
                            elif a > (n-1) and b < 0:
                                Eps.append(picture[a - (2*x)][b - (2*y)])
                            elif a > (n-1) and b > (m-1):
                                Eps.append(picture[a - (2*x)][b - (2*y)])

                        else:
                            Eps.append(picture[a][b])

                Eps.sort()

                epslen = len(Eps)

                if len(Eps) % 2 == 0:
                    newpicture[i][j] = (1/2) * (int(Eps[(epslen//2)-1])
                                                + int(Eps[epslen//2]))

                else:
                    newpicture[i][j] = Eps[(epslen//2)]

        plt.imshow(newpicture, cmap='gray', interpolation='nearest')

        plt.axis('off')

        plt.tight_layout()
        plt.show()

    if weight == 0:

        sigma = 3
        M = 3 * int(sigma)

        newpicture = np.zeros(shape=(n, m))

        for i in range(n):
            for j in range(m):

                Eps = []

                for x in range(-s, s+1, 1):
                    for y in range(-s, s+1, 1):

                        a = i + x
                        b = j + y

                        if abs(x) <= M and abs(y) <= M:
                            wval = np.exp(-((x+s)**2 + (y+s)**2)/(2*sigma**2))
                        else:
                            wval = 0

                        # continuation by mirroring
                        if a < 0 or b < 0 or a > (n-1) or b > (m-1):

                            if a < 0 and b >= 0 and b <= (m-1):
                                Eps.append([picture[a - (2*x)][b], wval])
                            elif a >= 0 and b < 0 and a <= (n-1):
                                Eps.append([picture[a][b - (2*y)], wval])
                            elif a > n and b >= 0 and b <= (m-1):
                                Eps.append([picture[a - (2*x)][b], wval])
                            elif a >= 0 and b > (m-1) and a <= (n-1):
                                Eps.append([picture[a][b - (2*y)], wval])
                            elif a < 0 and b < 0:
                                Eps.append([picture[a - (2*x)][b - (2*y)],
                                            wval])
                            elif a < 0 and b > (m-1):
                                Eps.append([picture[a - (2*x)][b - (2*y)],
                                            wval])
                            elif a > (n-1) and b < 0:
                                Eps.append([picture[a - (2*x)][b - (2*y)],
                                            wval])
                            elif a > (n-1) and b > (m-1):
                                Eps.append([picture[a - (2*x)][b - (2*y)],
                                            wval])

                        else:
                            Eps.append([picture[a][b], wval])

                summe = 0

                for z in range(len(Eps)):
                    summe += Eps[z][1]

                for z in range(len(Eps)):
                    Eps[z][1] = Eps[z][1]/summe

                Eps.sort(key=lambda x: x[0])

                q = 0
                p = 0
                while q < 0.5:
                    q += Eps[p][1]
                    p += 1

                epslen = len(Eps)

                if q == 0.5:
                    newpicture[i][j] = (1/2) * (int(Eps[p-1][0])
                                                + int(Eps[p][0]))
                else:
                    if p > len(Eps)-1:
                        newpicture[i][j] = int(Eps[len(Eps)-1][0])
                    else:
                        newpicture[i][j] = int(Eps[p][0])

        plt.imshow(newpicture, cmap='gray')

        plt.axis('off')

        plt.tight_layout()
        plt.show()
