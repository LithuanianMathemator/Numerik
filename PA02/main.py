
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

def laplace(n, m):

    I_m = sparse.eye(m)
    I_n = sparse.eye(n)
    D_m = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(m, m))
    D_n = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n))

    operator = sparse.kron(I_m, D_n) + sparse.kron(D_m, I_n)

    return operator

def seamlessmatrix(f, g, pos):

    f_blue = np.int32((io.imread(f))[..., 2])
    f_green = np.int32((io.imread(f))[..., 1])
    f_red = np.int32((io.imread(f))[..., 0])

    g_blue = np.int32((io.imread(g))[..., 2])
    g_green = np.int32((io.imread(g))[..., 1])
    g_red = np.int32((io.imread(g))[..., 0])

    red = grayhelp(f_red, g_red, pos)
    green = grayhelp(f_green, g_green, pos)
    blue = grayhelp(f_blue, g_blue, pos)

    result = np.dstack((red, green, blue))

    plt.imshow(result, cmap='gray', interpolation='nearest')

    plt.axis('off')

    plt.tight_layout()
    plt.show()


def grayhelp(f, g, pos):

    # f* and g as matrices
    # pos: tuple to situate g

    r = 37

    N = len(g)
    M = len(g[0])

    # border and f as slice from the whole picture f*
    cut_f = f[pos[0]: pos[0] + N, pos[1]: pos[1] + M]

    # turning everything except the border to zeros
    cut_f[r:-r, r:-r] = 0

    # laplace operator to determine missing border values
    lp_big = laplace(M, N)

    deviation = lp_big @ (cut_f.flatten())

    # determining b by multiplying g with laplace operator and cutting it down
    gradient_g = lp_big @ (g.flatten())

    # correcting deviation
    gradient_g -= deviation

    # getting b by turning back into a matrix, deleting the border and
    # vectorizing again
    b = ((gradient_g.reshape(N, M))[r:-r, r:-r]).flatten()

    lp_small = laplace(M - 2*r, N - 2*r)

    start = np.zeros(len(b))

    new_f = scipy.sparse.linalg.cg(lp_small, b, x0=start,
                                   maxiter=400000,
                                   M=None, callback=None, atol=None)

    replacement = new_f[0].reshape(N-2*r, M-2*r)

    cut_f[r:-r, r:-r] = replacement

    return f



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

if __name__ == "__main__":
    pass