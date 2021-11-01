import utility as u
import numpy as np

import time

def plot_bilateral():
    # open images
    B1_in = u.B1()
    B2_in = u.B2()
    C_in = u.C()

    t = time.time()

    # calculate bilateral filter
    #print('image: B1')
    #B1_out = bilateral(B1_in, s=2, rho_s=3, rho_r=75)
    #print('\nimage: B2')
    #B2_out = bilateral(B2_in, s=2, rho_s=4, rho_r=120)
    print('\nimage: C')
    C_out = bilateral(C_in, s=2, rho_s=3, rho_r=75)
    print('')

    print(time.time() - t)

    # display results
    #u.view(B1_in, B1_out)
    #u.view(B2_in, B2_out, 'abc')
    u.view(C_in , C_out, 'title')

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

plot_bilateral()