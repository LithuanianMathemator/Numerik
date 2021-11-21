
import numpy as np
from scipy import sparse
from scipy import linalg
from scipy.linalg import norm
from matplotlib import pyplot as plt
from skimage import io
from skimage.color import rgb2gray

def laplaceoperator(n,m):
    # n,m: Bildgröße
    
    l = n * m
    hauptdiag = np.ones(l) * -4
    nebendiag = np.ones(l-1)
    außendiag = np.ones(l-n)

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


def D_v(n):
    return sparse.diags([[-1]*n, [1]*(n-1)], [0, 1])

def D_r(n):
    return sparse.diags([[1]*n, [-1]*(n-1)], [0, -1])

def seamlessdiff_advanced(f, g, x, y, r=5, n_iter=40000, verbose=False):
    pic_f = io.imread(f)
    pic_g = io.imread(g)
    (n, m) = (len(pic_g), len(pic_g[0]))
    pic_f_area = pic_f[x:x+n, y:y+m]

    h = np.ndarray(shape=(n-2*r, m-2*r, 3))

    for i in range(3):
        if verbose:
            print('started run', i+1)
        
        f = pic_f_area[:, :, i]
        g = pic_g[:, :, i]

        grad_f_x = D_v(n) @ f
        grad_f_y = f @ D_v(m).transpose()

        grad_g_x = D_v(n) @ g
        grad_g_y = g @ D_v(m).transpose()

        if verbose:
            print('calculated gradients')

        norm_f = norm(np.dstack([grad_f_x, grad_f_y]), axis=2)
        norm_g = norm(np.dstack([grad_g_x, grad_g_y]), axis=2)

        if verbose:
            print('calculated gradient norm')


        v_x = np.where(norm_f > norm_g, grad_f_x, grad_g_x)
        v_y = np.where(norm_f > norm_g, grad_f_y, grad_g_y)

        if verbose:
            print('determined v')

        div_v = D_r(n) @ v_x + v_y @ D_r(m).transpose()

        f_cut = f
        f_cut[r:-r, r:-r] = 0

        vec_div_v = div_v.flatten('F')
        b = vec_div_v - laplaceoperator(n, m) @ f_cut.flatten('F')

        if verbose:
            print('simplified SLE')

        superfluous_rows = [i for i in range(n*m) if ((i+r)%n < 2*r)  or ((i+n*r)%(m*n) < 2*n*r)]
        
        mask = np.ones(n*m, dtype=bool)
        mask[superfluous_rows] = False

        delta = laplaceoperator(n-2*r, m-2*r)
        b_cut = np.delete(b, superfluous_rows, axis=0)

        if verbose:
            print('cut matrices')
        
        (vec_h, info) = linalg.cg(delta, b_cut, x0=np.zeros(len(b_cut)), maxiter=n_iter, atol=None, M=None, callback=None)
        print('.')
        h[:, :, i] = vec_h.reshape(n-2*r, m-2*r, order='F')
        if verbose:
            print('solved SLE')
    np.clip(h, 0, 255, out=h)
    pic_f[x+r:x+n-r, y+r:y+m-r, :] = h

    return pic_f