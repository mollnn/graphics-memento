import numpy as np
from matplotlib import pyplot as plt

a = 500
mid = a//2
img_w = a
img_h = a
r = a * 0.4

img = [[0 for j in range(img_w)] for i in range(img_h)]

l = np.array([-1.,1.,1.])
l /= np.linalg.norm(l, ord=2)

e = np.array([0,0,1])

diff = 0.3
spec = 3
spow = 8
ambi = 0.05

for i in range(img_h):
    for j in range(img_w):
        x = (j-mid) / r
        y = -(i-mid) / r
        if x*x+y*y<1:
            z = (1-x**2-y**2)**0.5
            n = np.array([x,y,z])
            h = l+e
            h /= np.linalg.norm(h, ord=2)
            c = ambi + diff * max(0, np.dot(n,l)) + spec *( max(0, np.dot(n,h)) ** spow)
            img[i][j]=c

plt.imshow(img, cmap='gray')
plt.show()