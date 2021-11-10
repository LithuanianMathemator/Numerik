import numpy as np
from scipy import sparse

def laplaceoperator(n,m):
    # n,m: Bildgröße
    
    l = n * m
    hauptdiag = np.ones(l) * -4
    nebendiag = np.ones(l-1)
    außendiag = np.ones(l-3)

    for i in range(1,l-1):
        if i % 4 == 0:
            nebendiag[i] = 0
            
    
    diagonalen = [hauptdiag, nebendiag, nebendiag, außendiag, außendiag]
    
    matrix = sparse.diags(diagonalen, [0,-1,1,n,-n])
    return matrix.toarray()