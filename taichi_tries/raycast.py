import numpy as np
import taichi as ti
from time import time

ti.init(arch=ti.cuda, debug=True)

NT = 1
NH = NW = 512
CN = CR = CH = 1

tris = ti.Vector.field(3, ti.f32, (NT, 3))
img = ti.Vector.field(3, ti.f32, (NH, NW))
cam_pos = ti.Vector.field(3, ti.f32, ())
cam_gaze = ti.Vector.field(3, ti.f32, ())
cam_top = ti.Vector.field(3, ti.f32, ())
clip_n = ti.field(ti.f32, ())
clip_r = ti.field(ti.f32, ())
clip_h = ti.field(ti.f32, ())
img_w = ti.field(ti.f32, ())
img_h = ti.field(ti.f32, ())

tris.from_numpy(np.array([
    [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
]))
cam_pos .from_numpy(np.array([0, 0, 2]))
cam_gaze .from_numpy(np.array([0, 0, -1]))
cam_top .from_numpy(np.array([0, 1, 0]))
clip_n .from_numpy(np.array(CN))
clip_r .from_numpy(np.array(CR))
clip_h .from_numpy(np.array(CH))
img_w .from_numpy(np.array(NW))
img_h .from_numpy(np.array(NH))
img.from_numpy(np.zeros((NH, NW, 3)))


@ti.func
def checkIntersect(orig, dir, trid):
    o = orig
    d = dir
    e1 = tris[trid, 1] - tris[trid, 0]
    e2 = tris[trid, 2] - tris[trid, 0]
    s = o - tris[trid, 0]
    s1 = d.cross(e2)
    s2 = s.cross(e1)
    t = s2.dot(e2)
    b1 = s1.dot(s)
    b2 = s2.dot(d)
    q = s1.dot(e1)
    return t/q, b1/q, b2/q


@ti.func
def getIntersection(orig, dir):
    # Find nearest intersection of ray(orig,dir) and triangles
    ans_t, ans_b1, ans_b2, ans_obj_id = 2e9, 0., 0., -1
    for i in range(tris.shape[0]):
        t, b1, b2 = checkIntersect(orig, dir, i)
        if t > 0 and t < ans_t and b1 > 0 and b2 > 0 and b1 + b2 < 1:
            ans_t, ans_b1, ans_b2, ans_obj_id = t, b1, b2, i
    return ans_t, ans_b1, ans_b2, ans_obj_id


@ti.func
def generateInitialRay(img_x, img_y):
    cam_handle = cam_gaze[None].cross(cam_top[None])
    canonical_x = (img_x + 0.5) / img_w[None] * 2 - 1
    canonical_y = - (img_y + 0.5) / img_h[None] * 2 + 1
    cam_near_center = cam_pos[None] + cam_gaze[None] * clip_n[None]
    orig = cam_near_center + canonical_x * cam_handle * \
        clip_r[None] + canonical_y * cam_top[None] * clip_h[None]
    dir = orig - cam_pos[None]
    dir = dir.normalized()
    return orig, dir


@ti.kernel
def render():
    for y, x in img:
        orig, dir = generateInitialRay(x, y)
        t, b1, b2, obj_id = getIntersection(orig, dir)
        if obj_id != -1:
            img[y, x] = [(obj_id + 1) * 0.25, 0., 0.]


gui = ti.GUI(res=(NW, NH))

while True:
    stt = time()
    render()
    print("time usage:", time()-stt, " fps:", 1./(time()-stt+1e-9))
    gui.set_image(img)
    gui.show()
