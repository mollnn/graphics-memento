import taichi as ti
import numpy as np
from time import time


ti.init(arch=ti.cpu)

img_h, img_w = 256, 256
img_shape = (img_h, img_w)
img_shape_c = (img_h, img_w, 3)

img = ti.Vector.field(3, ti.f32, img_shape)

@ti.kernel
def draw():
    for i,j in img:
        img[i,j] = [i / 255, j / 255, 0]


gui = ti.GUI(res=img_shape)

while True:
    draw()
    gui.set_image(img)
    gui.show()