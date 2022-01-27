import numpy as np
from matplotlib import pyplot as plt

TEX_MEM_SIZE = 1048576 * 4
tex_mem_used = [0]
tex_mem = np.array([[0, 0, 0] for i in range(TEX_MEM_SIZE)], dtype=np.float32)


def texAlloc(tex_mem, im_filename, used_):
    used= used_[0]
    im= plt.imread(im_filename)
    im=np.array(im)
    if len(im.shape)==2:
        im=np.tile(np.expand_dims(im,2),(1,1,3))
    sz= im.shape[0]*im.shape[1]
    tex_mem[used: used+sz] = im.reshape((-1, 3)) / 255.0
    used += sz
    used_[0]= used
    return used-sz, im.shape[0], im.shape[1]


def render(tex_mem, addr, h, w):
    im = tex_mem[addr: addr+h*w].reshape((h, w, 3))
    plt.imshow(im)
    plt.show()


h1= texAlloc(tex_mem, "assets/ground.jpg", tex_mem_used)
h2= texAlloc(tex_mem, "assets/wood.jpg", tex_mem_used)

render(tex_mem, h1[0], h1[1], h1[2])
render(tex_mem, h2[0], h2[1], h2[2])
