import numpy as np
from matplotlib import pyplot as plt
import random
import math


def getBezierInterp(p, t):
    if len(p) == 1:
        return p[0]
    return getBezierInterp([p[i]*(1-t) + p[i+1]*t for i in range(len(p)-1)], t)


control_points = [
    np.array([random.random(), random.random()]) for i in range(20)]
control_points.sort(key=lambda x: x[0])

print(control_points)
xs = []
ys = []
div = 100
for u in range(0,len(control_points)-3,3):
    tcp = control_points[u:u+4]
    for i in range(div+1):
        t = i / div
        q = getBezierInterp(tcp, t)
        xs .append(q[0])
        ys .append(q[1])

plt.plot(xs, ys)
plt.scatter([i[0] for i in control_points], [i[1] for i in control_points])
plt.scatter([i[0] for i in control_points[::3]], [i[1] for i in control_points[::3]])
plt.show()
