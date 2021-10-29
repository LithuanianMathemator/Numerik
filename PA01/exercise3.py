import skimage as sk
import numpy as np
import matplotlib as plt
from skimage import io


def func1medianfilter(s, weight=1, bild='B1.png'):

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
        plt.show()
