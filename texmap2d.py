import numpy as np
from matplotlib import pyplot as plt
import random
import math

texture = [[i % 2 for j in range(10)] for i in range(10)]


def getTextureOValue(x, y):
    w = len(texture[0])
    h = len(texture)
    if x < 0 or y < 0 or x >= w or y >= h:
        return 0
    return texture[y][x]


def getTextureValue(x, y):
    x0, y0 = math.floor(x), math.floor(y)
    x1, y1 = x0+1, y0+1
    w = [(x1-x)*(y1-y), (x1-x)*(y-y0), (x-x0)*(y1-y), (x-x0)*(y-y0)]
    v = [getTextureOValue(x0, y0), getTextureOValue(
        x0, y1), getTextureOValue(x1, y0), getTextureOValue(x1, y1)]
    return sum(i*j for (i, j) in zip(w, v))


p = np.array([[10, 10, 0, 0, 0], [90, 20, 0, 10, 0], [20, 80, 0, 0, 10]])


def check(p, j, i):
    p = np.array([[x[0], x[1], 0] for x in p])
    return np.sign(np.cross(p[1]-p[0], p[0]-[i, j, 0])[2]) == np.sign(np.cross(p[2]-p[1], p[1]-[i, j, 0])[2]) == np.sign(np.cross(p[0]-p[2], p[2]-[i, j, 0])[2])


def getuv(p, j, i):
    q = [j, i, 0]
    pp = np.array([[x[0], x[1], 0] for x in p])
    w = [np.linalg.norm(np.cross(pp[(i+1) % 3]-q, pp[(i+2) % 3]-q), ord=1)
         for i in range(3)]
    w /= np.sum(w)
    return np.sum([w[i]*p[i][3] for i in range(3)]), np.sum([w[i]*p[i][4] for i in range(3)])


img = [[0 for j in range(100)] for i in range(100)]

for i in range(100):
    for j in range(100):
        if check(p, j, i):
            u, v = getuv(p, j, i)
            img[j][i] = getTextureValue(u, v)

plt.imshow(img)
plt.show()
