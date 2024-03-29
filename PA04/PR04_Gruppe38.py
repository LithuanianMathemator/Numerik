
from time import time
import numpy as np
import matplotlib.pyplot as plt


##### 1. #####


def newton(F, dF, x0, delta = 0.0001, epsilon = 0.0001, maxIter = 100):

    '''
    F: function, input is vector
    dF: matrix with input as vector
    x0: starting vector
    delta, epsilon, maxIter: terminating conditions
    '''

    x_k = x0
    k = 0

    while k < maxIter and np.linalg.norm(F(x_k)) >= epsilon:

        if k > 1 and np.linalg.norm(x_k - x_k_1) < delta:
            break

        sol = np.linalg.solve(dF(x_k), -F(x_k))

        x_k_1 = x_k
        x_k = x_k + sol

        k += 1

    return x_k


##### 2. #####


def polynomial2(x):
    return np.array([x[0]**3 - 2*x[0]])

def polderivative(x):
    return np.array([[3*(x[0]**2) - 2]])

def testsecond():

    x_0 = (newton(polynomial2, polderivative, np.array([0.1]),\
     0.0000000001, 0.0000000001, 50)[0], polynomial2(newton(polynomial2,\
     polderivative, np.array([[0.1]]), 0.0000000001, 0.0000000001, 50))[0][0])

    x_1 = (newton(polynomial2, polderivative, np.array([[2]]),\
     0.0000000001, 0.0000000001, 50)[0], polynomial2(newton(polynomial2,\
     polderivative, np.array([[0.1]]), 0.0000000001, 0.0000000001, 50))[0][0])

    x_2 = (newton(polynomial2, polderivative, np.array([[-2]]),\
     0.0000000001, 0.0000000001, 50)[0], polynomial2(newton(polynomial2,\
     polderivative, np.array([[0.1]]), 0.0000000001, 0.0000000001, 50))[0][0])

    x = np.linspace(-2.1, 2.2, num = 100)

    plt.plot(x, polynomial2(np.array([x]))[0], color = 'violet')
    plt.scatter(x_0[0], x_0[1], color = 'green')
    plt.scatter(x_1[0], x_0[1], color = 'green')
    plt.scatter(x_2[0], x_0[1], color = 'green')
    plt.grid()
    plt.title('Nullstellen der Funktion x^3-2x')
    plt.show()


##### 3. #####


def func3(x):
    return np.array([x[0]**2 + x[1]**2 - 6*x[0], (3/4)*np.exp(-x[0]) - x[1]])

def derivative3(x):
    return np.array([[2*x[0] - 6, 2*x[1]],
                     [-(3/4)*np.exp(-x[0]), -1]])

def testthird():
    result = newton(func3, derivative3, np.array([0.08, 0.7]),\
     0.000000001, 0.000000001)
    plt.title(f'Nullstelle der Funktion f(x)=(x1^2+x2^2-6*x1, 3/4*e^(-x1)-x2)^T\n(x0, x1) = ({result[0]:.3f}, {result[1]:.3f})')
    plt.plot(result[0], result[1], marker='x')
    plt.grid()
    plt.show()


##### 4. #####


def einheitswurzel(d, n, n_max, delta, epsilon):
    space = np.linspace(-1, 1, n)
    B_grid = np.array([[np.complex(a, b) for a in space] for b in space])
    B = np.ndarray((n, n), dtype=complex)

    F = lambda z: np.array([z**d-1])
    dF = lambda z: np.array([d*z**(d-1)])

    for x in range(n):
        for y in range(n):
            z = B_grid[x, y]
            try:
                B[x, y] = newton(F, dF, np.array([z]), delta=delta, epsilon=epsilon, maxIter=n_max)
            except np.linalg.LinAlgError:
                B[x, y] = 0
    return B

    

def testfourth():
    n = 512
    d = 3
    n_max = 15
    delta = 0.00001
    epsilon = 0.00001

    B = einheitswurzel(d, n, n_max, delta, epsilon)
    
    plt.imshow(np.angle(B), cmap='hsv')
    plt.grid()
    plt.show()
    

##### Alternative zu 4. #####    
    

def func4(x):
    return np.array([x[0]**3 - 1])


def derivative4(x):
    return np.array([[3*(x[0]**2)]])


def newton3():

    x = np.linspace(-1, 1, num=512)

    nums = np.empty([512, 512], dtype=np.complex128)

    for i in range(512):
        nums[512 - i - 1] = x + 1.0j*x[i]

    res = np.empty([512, 512], dtype=np.complex128)

    for i in range(512):
        for j in range(512):
            res[i][j] = newton(func4, derivative4, np.array([nums[i][j]]),
                               0.00001, 0.00001, 15)[0]

    picture = np.zeros((512, 512))

    for i in range(512):
        for j in range(512):

            picture[i][j] = np.angle(res[i][j])

    plt.imshow(picture)
    plt.show()
    

##### 5. #####


def testfifth():
    n = 512
    d = 5
    n_max_1 = 5
    n_max_2 = 15
    delta = 0.00000000000001
    epsilon = 0.00000000000001

    B1 = einheitswurzel(d, n, n_max_1, delta, epsilon)
    B2 = einheitswurzel(d, n, n_max_2, delta, epsilon)
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('z^5=0')
    ax1.imshow(np.angle(B1), cmap='hsv')
    ax2.imshow(np.angle(B2), cmap='hsv')
    ax1.grid()
    ax2.grid()
    ax1.set_title(f'n_max = {n_max_1}')
    ax2.set_title(f'n_max = {n_max_2}')
    plt.show()

##### 6. #####


def minimize():

    return newton(func6, derivative6, np.array([-1.1, 1.1]), 0.00000000001,
                  0.00000000001, 1000)


def func6(x):
    return np.array([4*(x[0] + 1)**3, 4*(x[1] - 1)**3])


def derivative6(x):
    return np.array([[12*(x[0] + 1)**2, 0],
                     [0, 12*(x[1] - 1)**2]])

def testsixth():
    fun = lambda a, b: (a+1)**4+(b-1)**4
    result = minimize()
    x1 = np.arange(-2, 0, 0.05)
    x2 = np.arange(0, 2, 0.05)
    X1, X2 = np.meshgrid(x1, x2)
    y = np.array([fun(a, b) for a,b in zip(np.ravel(X1), np.ravel(X2))])
    Y = y.reshape(X1.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X1, X2, Y)
    ax.plot(result[0], result[1], fun(result[0], result[1]), marker='|', markerfacecolor='red', markeredgecolor='red', markersize=30)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title(f'Minimierer der Funktion f(x)=(x1 + 1)^4+(x2-1)^4\n(x1, x2) = ({result[0]:.3f}, {result[1]:.3f})')

    plt.show()


##### 7. #####


def mandelbrot_iteration(z, n_max):
    c = z
    for i in range(n_max):
        if z.real*z.real+z.imag*z.imag > 4:
            return i
        z = z*z+c
    return 255

def testseventh():
    n = 1024
    space_x = np.linspace(-1.5, 0.5, n)
    space_y = np.linspace(-1, 1, n)
    B_grid = np.array([[np.complex(a, b) for a in space_x] for b in space_y])
    B = np.ndarray((n, n), dtype=int)

    n_max = 256

    for x in range(n):
        for y in range(n):
            z = B_grid[x, y]
            B[x, y] = mandelbrot_iteration(z, n_max)
    
    plt.imshow(B)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    testsecond()
    testthird()
    testfourth()
    testfifth()
    testsixth()
    testseventh()
