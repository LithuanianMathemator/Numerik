import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from skimage import io
from skimage.color import rgb2gray

def laplaceoperator(n,m):
    # n,m: Bildgröße
    
    l = n * m
    hauptdiag = np.ones(l) * -4
    nebendiag = np.ones(l-1)
    außendiag = np.ones(l-n)

    #for i in range(1,l-1):
    #    if i % 4 == 0:
    #        nebendiag[i] = 0
    for i in range(1,l-1):
        if i % n == 0:
            nebendiag[i-1] = 0
            
    
    diagonalen = [hauptdiag, nebendiag, nebendiag, außendiag, außendiag]
    
    matrix = sparse.diags(diagonalen, [0,-1,1,n,-n])
    return matrix

def seamlessdiff(f, g, pos):

    # pos as tuple

    # reading files and converting [0,1] scale to [0,255] scale for uint8
    pic_f = io.imread(f)                                    # this is f*
    pic_g = io.imread(g)
    gray_f = np.uint8(rgb2gray(pic_f)*255)
    gray_g = np.uint8(rgb2gray(pic_g)*255)


    N_f = len(pic_f)                                        # size of f*
    N_g = len(pic_g)                                        # size of f and g
    M_f = len(pic_f[0])
    M_g = len(pic_g[0])

    # slicing to get matrix of replaced f
    gray_r = gray_f[pos[0]:pos[0]+N_g ,pos[1]:pos[1]+M_g]   # space of f* to be
                                                            # replaced
    r = 5

    gray_r[r:-r, r:-r] = np.full((N_g-(2*r), M_g-(2*r)), 0)
    # np.full((N_g-4, M_g-4), -1)
    # np.zeros((N_g-4, M_g-4))
    # gray_r[0:, 0:] = gray_g

    # vectorizing f and g
    vec_f = gray_r.flatten('F')
    vec_g = gray_g.flatten('F')

    delta = laplaceoperator(N_g, M_g)

    b = delta @ vec_g

    T = delta.transpose()

    print(b)
    print(T.tocsr()[0:1, 0:].multiply(vec_f[0]))

    # tracking of equations that are not needed
    colbool = [True for i in range(len(vec_f))]
    #colbool = [True]*len(vec_f)

    # bringing stuff to the right side
    for i in range(len(vec_f)):
        if vec_f[i] != 0:
            b -= T.tocsr()[i:i+1, 0:].multiply(vec_f[i])
            colbool[i] = False

    remlist = []
    for i in range(len(vec_f)):
        if colbool[i]:
            remlist.append(i)

    b = np.asarray(b)[0]

    # making the LGS smaller
    coldelt = delta.tocsr()[:, remlist]
    newdelt = coldelt.tocsr()[remlist, :]
    print(newdelt.shape)
    newb = b[remlist]
    print(newb)
    first = np.zeros(len(newb))

    # CG to get f
    newf = sparse.linalg.cg(newdelt, newb, x0=first, \
    maxiter=400, M=None, callback=None, atol=None)

    print(newf[0])

    s = 0
    o = np.zeros((M_g-(2*r), N_g-(2*r)))
    for i in range(M_g-(2*r)):
        for j in range(N_g-(2*r)):
            o[i][j] = newf[0][s]
            s+=1

    gray_r[r:-r, r:-r] = o.transpose()

    plt.imshow(gray_f, cmap='gray', interpolation='nearest')

    plt.axis('off')

    plt.tight_layout()
    plt.show()


def seamlessdiff_advanced(f, g):
    pic_f = io.imread(f)
    pic_g = io.imread(g)