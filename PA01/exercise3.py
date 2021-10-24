def func1medianfilter(s, weight=1, bild='B1.png'):

    picture = io.imread(bild)
    #
    # plt.imshow(picture, cmap='gray', interpolation='nearest')
    #
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    n = len(picture)
    m = len(picture[0])

    print(m)

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
                    picture[i][j] = (1/2) * (int(Eps[(epslen//2)-1])
                                             + int(Eps[epslen//2]))

                else:
                    picture[i][j] = Eps[(epslen//2)]

        plt.imshow(picture, cmap='gray', interpolation='nearest')

        plt.axis('off')

        plt.tight_layout()
        plt.show()
