import numpy as np
from matplotlib import pyplot as plt
import random
import math
from mpl_toolkits.mplot3d import Axes3D


def getBezierInterp(p, t):
    if len(p) == 1:
        return p[0]
    return getBezierInterp([p[i]*(1-t) + p[i+1]*t for i in range(len(p)-1)], t)


control_points = np.array([[np.array([i, j, random.random()]) for j in range(4)]for i in range(4)])


div = 20

xs = np.array([[0. for i in range(div)] for j in range(div)])
ys = np.array([[0. for i in range(div)] for j in range(div)])
zs = np.array([[0. for i in range(div)] for j in range(div)])

for i in range(div):
    t = i / div
    q = [getBezierInterp(control_points[j], t) for j in range(4)]
    for j in range(div):
        tt = j /div
        qq = getBezierInterp(q, tt)
        xs[i][j] = qq[0]
        ys[i][j] = qq[1]
        zs[i][j] = qq[2]


fig = plt.figure(1, figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_top_view()

ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='jet')
plt.show()