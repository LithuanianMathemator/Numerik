from skimage import io
import matplotlib.pyplot as plt

# open B1
def B1():
    return openPNG('B1.png')

# open B2
def B2():
    return openPNG('B2.png')

# open C
def C():
    return openPNG('C.png')

# open PNG from path
def openPNG(path):
    try:
        PNG = io.imread(path)
    except:
        try:
            PNG = io.imread('.\PA01.\\' + path)
        except:
            print('Datei ' + path + ' nicht gefunden')
            return
    return PNG

# display image
def view(image1, image2, title, _cmap='gray'):
    fig, (p1, p2) = plt.subplots(1, 2)
    fig.suptitle(title)
    p1.imshow(image1, cmap=_cmap)
    p2.imshow(image2, cmap=_cmap)
    p1.axis('off')
    p2.axis('off')
    plt.show()

# print progress (slow, x2,5)
def progress(i, j, m, n):
    pct_1 = i/(m-1)
    pct_2 = j/(n-1)
    _val1 = (pct_1 * 10)//1
    _val2 = (pct_2 * 10)//1

    print('\033[F', end='')
    print('cols: [', end='')
    for k in range(10):
        if k <= _val1-1:
            print('#', end='')
        else:
            print(' ', end='')
    print('] ' + str(i+1) + '/' + str(m) + '\nrows: [', end='')
    for k in range(10):
        if k <= _val2-1:
            print('#', end='')
        else:
            print(' ', end='')
    print('] ' + str(j+1) + '/' + str(n) + '   ', end='')
