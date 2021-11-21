import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.sparse import linalg
from skimage import io
from matplotlib import pyplot as plt

def laplace(n, m):

    I_m = sparse.eye(m)
    I_n = sparse.eye(n)
    D_m = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(m, m))
    D_n = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n))

    operator = sparse.kron(I_m, D_n) + sparse.kron(D_m, I_n)

    return operator

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
        
        v_x = np.ndarray(shape=(n, m))
        v_y = np.ndarray(shape=(n, m))

        for j in range(n):
            for k in range(m):
                if norm_f[j, k] > norm_g[j, k]:
                    v_x[j, k] = grad_f_x[j, k]
                    v_y[j, k] = grad_f_y[j, k]
                else:
                    v_x[j, k] = grad_g_x[j, k]
                    v_y[j, k] = grad_g_y[j, k]
        
        if verbose:
            print('determined v')

        div_v = D_r(n) @ v_x + v_y @ D_r(m).transpose()

        f_cut = f
        f_cut[r:-r, r:-r] = 0

        vec_div_v = div_v.flatten('F')
        b = vec_div_v - laplace(n, m) @ f_cut.flatten('F')

        if verbose:
            print('simplified SLE')

        superfluous_rows = [i for i in range(n*m) if ((i+r)%n < 2*r)  or ((i+n*r)%(m*n) < 2*n*r)]
        
        mask = np.ones(n*m, dtype=bool)
        mask[superfluous_rows] = False

        delta = laplace(n-2*r, m-2*r)
        b_cut = np.delete(b, superfluous_rows, axis=0)

        if verbose:
            print('cut matrices')
        
        (vec_h, info) = linalg.cg(delta, b_cut, x0=np.zeros(len(b_cut)), maxiter=n_iter, atol=None, M=None, callback=None)
        h[:, :, i] = vec_h.reshape(n-2*r, m-2*r, order='F')
        if verbose:
            print('solved SLE')
    np.clip(h, 0, 255, out=h)
    pic_f[x+r:x+n-r, y+r:y+m-r, :] = h
    
    plt.imshow(pic_f, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#seamlessdiff_advanced('PA02/water.jpg', 'PA02/bear.jpg', 0, 0)
seamlessdiff_advanced('PA02/bird.jpg', 'PA02/plane.jpg', 50, 250)
