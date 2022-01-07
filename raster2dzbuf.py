import numpy as np
from matplotlib import pyplot as plt
import copy
import random


def check(p, j, i):
    p = np.array([[x[0], x[1], 0] for x in p])
    return np.sign(np.cross(p[1]-p[0], p[0]-[i, j, 0])[2]) == np.sign(np.cross(p[2]-p[1], p[1]-[i, j, 0])[2]) == np.sign(np.cross(p[0]-p[2], p[2]-[i, j, 0])[2])


def getz(p, j, i):
    q = [j, i, 0]
    pp = np.array([[x[0], x[1], 0] for x in p])
    w = [np.linalg.norm(np.cross(pp[(i+1) % 3]-q, pp[(i+2) % 3]-q), ord=1)
         for i in range(3)]
    w /= np.sum(w)
    return np.sum([w[i]*p[i][2] for i in range(3)])


ts = []
ts.append(np.array([[10, 10, 1], [70, 20, 1], [20, 70, 1]]))
ts.append(np.array([[20, 20, 0], [50, 20, 3], [20, 50, 3]]))

img_w = 100
img_h = 100

img = [[0 for j in range(img_w)] for i in range(img_h)]
zbuf = [[9 for j in range(img_w)] for i in range(img_h)]
for p in ts:
    c = random.randint(3, 10)
    for i in range(img_h):
        for j in range(img_w):
            if check(p, j, i):
                z = getz(p, j, i)
                if z < zbuf[i][j]:
                    zbuf[i][j] = z
                    img[i][j] = c

plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.subplot(122)
plt.imshow(zbuf, cmap="gray")
plt.show()
