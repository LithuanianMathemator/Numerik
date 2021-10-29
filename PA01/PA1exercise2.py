import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

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
                W[i][j] = float(W[i][j])/float(summe) #TO DO
        
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

    if weight == 1: #ungewichtet (gleichgewichtet)
    
        for i in range(s,n-s):
            for j in range(s, m-s):
                #print("for Schleife W:{}".format(W))
                B = padded[i-s:i+s+1, j-s:j+s+1]
                newpicture[i][j] = func1mean(B,W,0)
    
        plt.imshow(newpicture, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    if weight == 0: #gaussian 
        sigma = 3 # TO DO: variation of sigma w.r.t. s ?
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


