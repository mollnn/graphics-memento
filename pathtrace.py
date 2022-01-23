import numpy as np
import taichi as ti
from time import time as tm
import time
from numpy.linalg import norm


def readobj(filename):
    fp = open(filename, 'r')
    fl = fp.readlines()
    verts = [[]]
    ans = []
    for s in fl:
        a = s.split()
        if len(a)>0:
            if a[0]=='v':
                verts.append([float(a[1]), float(a[2]), float(a[3])])
            elif a[0]=='f':
                b = a[1:]
                b = [i.split('/') for i in b]
                ans.append([verts[int(b[0][0])], verts[int(b[1][0])], verts[int(b[2][0])]])
    return ans



ti.init(arch=ti.cuda, debug=False)

scene = []
scene += readobj('assets/bunny.obj')
scene += readobj('assets/test.obj')
md = np.array(scene)

NT = len(md)
NH = NW = 256
CN = CR = CH = 0.01

mesh_vertices = ti.Vector.field(3, ti.f32, (NT, 3))
mesh_attributes = ti.Vector.field(3, ti.f32, (NT))
img = ti.Vector.field(3, ti.f32, (NH, NW))
cam_pos = ti.Vector.field(3, ti.f32, ())
cam_gaze = ti.Vector.field(3, ti.f32, ())
cam_top = ti.Vector.field(3, ti.f32, ())
clip_n = ti.field(ti.f32, ())
clip_r = ti.field(ti.f32, ())
clip_h = ti.field(ti.f32, ())
img_w = ti.field(ti.f32, ())
img_h = ti.field(ti.f32, ())
light_pos = ti.Vector.field(3, ti.f32, ())
light_int = ti.Vector.field(3, ti.f32, ())


mesh_vertices.from_numpy(md)
nm = [np.cross(p[1]-p[0], p[2]-p[0]) for p in md]
mesh_attributes.from_numpy(np.array([
    i/norm(i) for i in nm
]))
cam_pos .from_numpy(np.array([0., 0.1, 0.15]))
cam_gaze .from_numpy(np.array([0., 0., -1]))
cam_top .from_numpy(np.array([0., 1, 0]))
clip_n .from_numpy(np.array(CN))
clip_r .from_numpy(np.array(CR))
clip_h .from_numpy(np.array(CH))
img_w .from_numpy(np.array(NW))
img_h .from_numpy(np.array(NH))
img.from_numpy(np.zeros((NH, NW, 3)))
light_pos.from_numpy(np.array([0., 1, 3]))
light_int.from_numpy(np.array([10., 10., 10.]))


@ti.func
def checkIntersect(orig, dir, trid):
    o = orig
    d = dir
    e1 = mesh_vertices[trid, 1] - mesh_vertices[trid, 0]
    e2 = mesh_vertices[trid, 2] - mesh_vertices[trid, 0]
    s = o - mesh_vertices[trid, 0]
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
    for i in range(mesh_vertices.shape[0]):
        t, b1, b2 = checkIntersect(orig, dir, i)
        if t > 0 and t < ans_t and b1 > 0 and b2 > 0 and b1 + b2 < 1:
            ans_t, ans_b1, ans_b2, ans_obj_id = t, b1, b2, i
    return ans_t, ans_b1, ans_b2, ans_obj_id


@ti.func
def generateInitialRay(img_x, img_y):
    cam_handle = cam_gaze[None].cross(cam_top[None])
    canonical_x = (img_x + 0.5) / img_w[None] * 2 - 1
    canonical_y = (img_y + 0.5) / img_h[None] * 2 - 1
    cam_near_center = cam_pos[None] + cam_gaze[None] * clip_n[None]
    orig = cam_near_center + canonical_x * cam_handle * \
        clip_r[None] + canonical_y * cam_top[None] * clip_h[None]
    dir = orig - cam_pos[None]
    dir = dir.normalized()
    return orig, dir


@ti.func
def sample_brdf(normal):
    r, theta = 0.0, 0.0
    sx = ti.random() * 2.0 - 1.0
    sy = ti.random() * 2.0 - 1.0
    if sx != 0 or sy != 0:
        if abs(sx) > abs(sy):
            r = sx
            theta = np.pi / 4 * (sy / sx)
        else:
            r = sy
            theta = np.pi / 4 * (2 - sx / sy)
    u = ti.Vector([1.0, 0.0, 0.0])
    if abs(normal[1]) < 1 - 1e-7:
        u = normal.cross(ti.Vector([0.0, 1.0, 0.0]))
    v = normal.cross(u)
    costt, sintt = ti.cos(theta), ti.sin(theta)
    xy = (u * costt + v * sintt) * r
    zlen = ti.sqrt(max(0.0, 1.0 - xy.dot(xy)))
    ans = xy + zlen * normal
    return ans.normalized()


@ti.kernel
def render():
    SPP = 1
    for x, y in img:
        tans = ti.Vector([0., 0., 0.], dt=ti.f32)
        for sp in range(SPP):
            orig, dir = generateInitialRay(x, y)
            NB = 3
            ans = ti.Vector([0., 0., 0.], dt=ti.f32)
            coef = ti.Vector([1., 1., 1.], dt=ti.f32)
            for _ in range(NB):
                t, b1, b2, tid = getIntersection(orig, dir)
                p = orig + dir * t
                if tid != -1 and mesh_attributes[tid].dot(-dir) > 0:
                    brdf = ti.Vector([0.7, 0.7, 0.7], dt=ti.f32)
                    ans += light_int[None] * brdf / (p - light_pos[None]).norm() ** 2 * max(
                        mesh_attributes[tid].dot((light_pos[None]-p).normalized()), 0.) * coef
                    wi = sample_brdf(mesh_attributes[tid])
                    coef *= brdf * max(mesh_attributes[tid].dot(wi), 0.)
                    orig = p + wi * 1e-6
                    dir = wi
            tans += ans
        img[x, y] = tans / SPP



gui = ti.GUI(res=(NW, NH))
frame_id = 0
while True:
    stt = tm()

    cam_pos .from_numpy(
        np.array([0.3*ti.cos(frame_id * 0.02), 0.1, 0.3*ti.sin(frame_id * 0.02)]))
    cam_gaze.from_numpy(
        np.array([0.3*ti.cos(frame_id * 0.02), 0.0, 0.3*ti.sin(frame_id * 0.02)]))
    cam_gaze[None] = -cam_gaze[None].normalized()

    render()
    if frame_id % 10 == 0:
        print("time usage:", tm()-stt, " able fps:", 1./(tm()-stt+1e-9))
    gui.set_image(img)
    gui.show()
    frame_id += 1
