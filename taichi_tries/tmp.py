import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

orig = ti.Vector.field(3,ti.f32,())
dir = ti.Vector.field(3,ti.f32,())
tris = ti.Vector.field(3, ti.f32, (1, 3))

tris.from_numpy(np.array([
    [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
]))

orig.from_numpy(np.array([1,2,3]))
dir.from_numpy(np.array([1,0,0]))

@ti.kernel
def func():
    o = orig[None]
    d = dir[None]
    e1 = tris[0,1] - tris[0,0]
    e2 = tris[0,2] - tris[0,0]
    s = o - tris[0,0]
    s1 = d.cross(e2)
    s2 = s.cross(e1)
    t = s2.dot(e2)
    b1 = s1.dot(s)
    b2 = s2.dot(d)
    q = s1.dot(e1)
    print(q)


func()