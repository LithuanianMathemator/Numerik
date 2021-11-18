import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from scipy import sparse
import scipy.sparse.linalg


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

    N = len(g)
    M = len(g[0])

    # border and f as slice from the whole picture f*
    cut_f = f[pos[0]: pos[0] + N, pos[1]: pos[1] + M]

    # turning everything except the border to zeros
    cut_f[1:-1, 1:-1] = 0

    # laplace operator to determine missing border values
    lp_big = laplace(M, N)

    deviation = lp_big @ (cut_f.flatten())

    # determining b by multiplying g with laplace operator and cutting it down
    gradient_g = lp_big @ (g.flatten())

    # correcting deviation
    gradient_g -= deviation

    # getting b by turning back into a matrix, deleting the border and
    # vectorizing again
    b = ((gradient_g.reshape(N, M))[1:-1, 1:-1]).flatten()

    lp_small = laplace(M - 2, N - 2)

    start = np.zeros(len(b))

    new_f = scipy.sparse.linalg.cg(lp_small, b, x0=start,
                                        maxiter=40000000000,
                                        M=None, callback=None, atol=None)

    replacement = new_f[0].reshape(N-2, M-2)

    cut_f[1:-1, 1:-1] = replacement

    return f
