from typing import TextIO
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


##################### 1. #####################

def func1mean(A, W=[[0]], weight=1):
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

        for i in range(100):

            sizing = np.random.randint(3, 20, size=2)
            A = np.random.randint(257, size=(sizing[0], sizing[1]))

            M.append(abs(np.mean(A) - func1mean(A)))

        return max(M)

    else:

        return "C'mon bro!"
    
def func1median(A, W=[0], weight=1):
    # type of A: numpy array
    # weight: 0 = random, 1 = equal, 2 = test of 100 random matrices

    n = len(A)
    m = len(A[0])

    if weight == 0:

        medianlist = []

        for i in range(len(A)):
            for j in range(len(A[0])):
                medianlist.append([A[i][j], W[i][j]])

        medianlist.sort(key=lambda x: x[0])

        q = 0
        p = 0
        while q < 0.5:
            q += medianlist[p][1]
            p += 1

        if q == 0.5:
            return (1/2) * (int(medianlist[p-1][0]) + int(medianlist[p][0]))
        else:
            return medianlist[p][0]

    elif weight == 1:

        Aflat = np.sort(np.array(A).flatten())

        if (n*m) % 2 == 0:
            return (Aflat[(n*m)//2-1] + Aflat[(n*m)//2])/2
        else:
            return Aflat[(n*m)//2]

    elif weight == 2:

        M = []

        for i in range(100):

            sizing = np.random.randint(3, 20, size=2)
            A = np.random.randint(257, size=(sizing[0], sizing[1]))

            M.append(abs(np.median(A) - func1median(A)))

        return max(M)

    else:

        return "C'mon bro!"
    
##################### 2. #####################    
    
def gaussianweight(s, sigma):
    vector = np.linspace(-s, s, 2*s+1)
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
                newpicture[i][j] = func1mean(B,W,1)
    
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
        
        #plt.imshow(W_gauss, cmap="gray")
        #plt.axis("off")
        #plt.tight_layout()
        #plt.show()

W = np.random.uniform(20, size=(5, 5))

# image filtering: 0 = gaussian, 1 = equal    
#meanfilter(2, W, 0, image = "PA01/B1.png")
#meanfilter(2, W, 1, image = "PA01/B1.png")

#meanfilter(2, W, 0, image = "PA01/B2.png")
#meanfilter(2, W, 1, image = "PA01/B2.png")

#meanfilter(2, W, 0, image = "PA01/C.png")
#meanfilter(2, W, 1, image = "PA01/C.png")


##################### 3. #####################    

def func1medianfilter(s, weight=0, bild='PA01/C.png'):

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
        plt.title('Medianfilter ohne Gauß, s= ' + s + ', Bild = ' + bild + '!')
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
        plt.title('Medianfilter mit Gauß, s= ' + s + ', Bild = ' + bild + '!')
        plt.show()

    
#func1medianfilter(3, bild='PA01/B2.png', weight=0)

##################### 4. #####################    


# bilateral gaussian filter
def bilateral(image, s=1, rho_s=3, rho_r=75):
    (m, n) = image.shape
    target = np.ndarray(shape=(m, n), dtype=np.uint8)

    W_s = w_s_array(s, rho_s)
    W_r = w_r_array(255, rho_r)
    
    for i in range(m):
        for j in range(n):
            target[i, j] = calculate_pixel(image, i, j, m, n, s, W_s, W_r)
            print((i, j), end='\r')
    return target

def calculate_pixel(image, k, l, m, n, s, W_s, W_r):
    sum = 0
    summands_array = np.ndarray(shape=(2*s+1, 2*s+1), dtype=float)
    for i in range(k-s, k+s+1):
        for j in range(l-s, l+s+1):
            summands_array[i-k+s, j-l+s] = W_s[k-i+s, l-j+s] * W_r[int(image[k, l]) - int(image[c(i, j, m, n)])+255]
    divisor = 0
    for i in range(k-s, k+s+1):
        for j in range(l-s, l+s+1):
                    divisor += summands_array[i-k+s, j-l+s]
    for i in range(k-s, k+s+1):
        for j in range(l-s, l+s+1):
            dividend = summands_array[i-k+s, j-l+s]
            value = ( dividend / divisor ) * int(image[c(i, j, m, n)])
            sum += value
    return sum

# gaussian distance weight
def w_s(x, y, rho_s):
    exp_arg = -(x**2 + y**2) / (2*pow(rho_s,2))
    return np.exp(exp_arg)

# gaussian distance weight, precalculated
def w_s_array(s, rho_s):
    n = 2*s+1
    array = np.ndarray(shape=(n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            x = - s + i
            y = - s + j
            array[i, j] = w_s(x, y, rho_s)
    return array

# gaussian color weight
def w_r(x, rho_r):
    exp_arg = -pow(x,2) / (2*pow(rho_r,2))
    return np.exp(exp_arg)

# gaussian color weight, precalculated
def w_r_array(n, rho_r):
    array = np.ndarray(shape=(2*n+1), dtype=float)
    for x in range(2*n+1):
        array[x] = w_r(x-n, rho_r)
    return array

# clamps values to dimensions -> mirrors at edges
def c(x, y, m, n):
    _x = -abs(abs(x) - (m-1)) + (m-1)
    _y = -abs(abs(y) - (n-1)) + (n-1) 
    return (_x, _y)

def view(image1, image2, title, _cmap='gray'):
    fig, (p1, p2) = plt.subplots(1, 2)
    fig.suptitle(title)
    p1.imshow(image1, cmap=_cmap)
    p2.imshow(image2, cmap=_cmap)
    plt.show()


if __name__ == "__main__":
    #mean

    #median
    
    func1medianfilter(2, 1, 'B1.png')
    func1medianfilter(2, 0, 'B1.png')
    func1medianfilter(2, 1, 'B2.png')
    func1medianfilter(2, 0, 'B2.png')
    func1medianfilter(2, 1, 'C.png')
    func1medianfilter(2, 0, 'C.png')
    

    #bilateral

    print('\n-----\nimage: B1')
    B1_in = io.imread('PA01/B1.png')
    B1_out = bilateral(B1_in, s=2, rho_s=3, rho_r=75)
    view(B1_in, B1_out, 'test')

    print('\nimage: B2')
    B2_in = io.imread('PA01/B1.png')
    B2_out = bilateral(B2_in, s=3, rho_s=3, rho_r=75)
    view(B2_in, B2_out)

    print('\nimage: C')
    C_in = io.imread('PA01/B1.png')
    C_out = bilateral(C_in, s=3, rho_s=3, rho_r=75)
    view(C_in, C_out)

    print('')
