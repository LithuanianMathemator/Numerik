import utility as u
import numpy as np

import time

def plot_bilateral():
    # open images
    B1_in = u.B1()
    B2_in = u.B2()
    C_in = u.C()
    
    _t_start = time.time()

    # calculate bilateral filter
    print('image: B1\n')
    B1_out = bilateral(B1_in)
    print('\n\nimage: B2\n')
    B2_out = bilateral(B2_in)
    print('\n\nimage: C\n')
    C_out = bilateral(C_in)
    print('')

    _t_end = time.time()
    #print(_t_end - _t_start)

    # display results
    u.view(B1_in, B1_out)
    u.view(B2_in, B2_out)
    u.view(C_in , C_out)

# bilateral gaussian filter
def bilateral(image, s=1, rho_s=3, rho_r=75):
    (m, n) = image.shape
    target = np.ndarray(shape=(m, n), dtype=np.uint8)
    
    for i in range(m):
        for j in range(n):
            target[i, j] = calculate_pixel(image, i, j, m, n, s, rho_s, rho_r)
            #u.progress(i, j, m, n)             # 253s
            #print((i, j), end='\r')            # 118s
                                                # 104s
    return target

# calculate pixel for bilateral gaussian filter
def calculate_pixel(image, k, l, m, n, s, rho_s, rho_r):
    sum = 0
    for i in range(k-s, k+s+1):
        for j in range(l-s, l+s+1):
            dividend = w_s(k-i, l-j, rho_s) * w_r(int(image[k, l]) - int(image[c(i, j, m, n)]), rho_r)
            divisor = 0
            for u in range(k-s, k+s+1):
                for v in range(l-s, l+s+1):
                    divisor += w_s(k-u, l-v, rho_s) * w_r(int(image[k, l]) - int(image[c(u, v, m, n)]), rho_r)
            value = ( dividend / divisor ) * int(image[c(i, j, m, n)])
            sum += value
    return sum

# gaussian distance weight
def w_s(x, y, rho_s):
    exp_arg = -(x**2 + y**2) / (2*pow(rho_s,2))
    return np.exp(exp_arg)

# gaussian color weight
def w_r(x, rho_r):
    exp_arg = -pow(x,2) / (2*pow(rho_r,2))
    return np.exp(exp_arg)

# clamps values to dimensions -> mirrors at edges
def c(x, y, m, n):
    _x = -abs(abs(x) - (m-1)) + (m-1)
    _y = -abs(abs(y) - (n-1)) + (n-1) 
    return (_x, _y)

plot_bilateral()