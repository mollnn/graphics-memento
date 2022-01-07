import numpy as np
from matplotlib import pyplot as plt


def check(p, j, i):
    return np.sign(np.cross(p[1]-p[0], p[0]-[i, j, 0])[2]) == np.sign(np.cross(p[2]-p[1], p[1]-[i, j, 0])[2]) == np.sign(np.cross(p[0]-p[2], p[2]-[i, j, 0])[2])


def msaa(p, j, i, d=2):
    offsets = [(x+0.5)/d for x in range(d)]
    ans = 0
    for oj in offsets:
        for oi in offsets:
            ans += check(p, j+oj, i+oi)
    return ans/d/d


p = np.array([[10, 10, 0], [30, 20, 0], [20, 30, 0]])

img_w = 50
img_h = 50

img = [[msaa(p, j, i) for j in range(img_w)] for i in range(img_h)]
plt.imshow(img)
plt.show()
