import numpy as np
from matplotlib import pyplot as plt
import math

from threading import Thread

tex = plt.imread("assets/wood.jpg").copy()
tex_w = len(tex[0])
tex_h = len(tex)


def getTexVal(tex, x, y):
    tex_w = len(tex[0])
    tex_h = len(tex)

    if x < 0 or y < 0 or x >= tex_w or y >= tex_h:
        return 0
    return tex[y][x] / 255


def getTexValI(tex, x, y):
    tex_w = len(tex[0])
    tex_h = len(tex)
    x *= tex_w
    y *= tex_h
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = x - x0
    dy = y - y0
    c00 = getTexVal(tex, x0, y0)
    c01 = getTexVal(tex, x0, y1)
    c10 = getTexVal(tex, x1, y0)
    c11 = getTexVal(tex, x1, y1)
    w00 = (1 - dx) * (1-dy)
    w01 = (1 - dx) * dy
    w10 = dx * (1-dy)
    w11 = dx * dy
    return w00 * c00 + w01 * c01 + w10 * c10 + w11 * c11


a = 400
mid = a//2
img_w = a
img_h = a
r = a * 0.4

img = [[np.array([0, 0, 0]) for j in range(img_w)] for i in range(img_h)]

l = np.array([-1., 1., 1.])
l /= np.linalg.norm(l, ord=2)

e = np.array([0, 0, 1])

diff = 0.7
spec = 0.5
spow = 10
ambi = 0.1

# def RenderLineRange(l_start, l_stop):
    # for i in range(l_start, l_stop):

for i in range(img_h):
    for j in range(img_w):
        x = (j-mid) / r
        y = -(i-mid) / r
        if x*x+y*y < 1:
            z = (1-x**2-y**2)**0.5
            n = np.array([x, y, z])
            h = l+e
            h /= np.linalg.norm(h, ord=2)
            tex_u = math.acos(x) / math.acos(-1)
            tex_v = math.acos(y) / math.acos(-1)
            diff_tex_val = getTexValI(tex, tex_u, tex_v)
            c = ambi * np.array([1, 1, 1]) + diff * diff_tex_val * max(0, np.dot(n, l)) + \
                spec * np.array([1, 1, 1]) * (max(0, np.dot(n, h)) ** spow)
            img[i][j] = c


# N = 8
# ths = []
# for i in range(N):
#     l_start = img_h // N * i
#     l_stop = img_h // N * (i+1)
#     th = Thread(target=RenderLineRange, args=(l_start, l_stop, ))
#     ths.append(th)
# for i in ths:
#     i.start()
# for i in ths:
#     i.join()
#     print("+")

plt.imshow(img, cmap='gray', vmin=0, vmax=1)

plt.show()
