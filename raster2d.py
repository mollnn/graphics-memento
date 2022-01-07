import numpy as np
from matplotlib import pyplot as plt

p = np.array([[10, 10, 0], [30, 20, 0], [20, 30, 0]])

img_w = 50
img_h = 50

img = [[np.sign(np.cross(p[1]-p[0], p[0]-[i, j, 0])[2]) == np.sign(np.cross(p[2]-p[1], p[1]-[i, j, 0])[2])
        == np.sign(np.cross(p[0]-p[2], p[2]-[i, j, 0])[2]) for j in range(img_w)] for i in range(img_h)]
plt.imshow(img)
plt.show()
