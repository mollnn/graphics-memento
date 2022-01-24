import numpy as np
import taichi as ti
from time import time as tm
import time
from numpy.linalg import norm


def readObject(filename, material_id, offset=[0, 0, 0], scale=1):
    fp = open(filename, 'r')
    fl = fp.readlines()
    verts = [[]]
    ans = []
    for s in fl:
        a = s.split()
        if len(a) > 0:
            if a[0] == 'v':
                verts.append(np.array([float(a[1]), float(a[2]), float(a[3])]))
            elif a[0] == 'f':
                b = a[1:]
                b = [i.split('/') for i in b]
                ans.append(
                    [verts[int(b[0][0])]*scale+offset, verts[int(b[1][0])]*scale+offset, verts[int(b[2][0])]*scale+offset, material_id])
    return ans


ti.init(arch=ti.cuda, debug=False)

scene = []
scene += readObject('assets/cube.obj', 1, offset=[0, -20, 0], scale=10)
scene += readObject('assets/cube.obj', 3, offset=[-20, 0, 0], scale=10)
scene += readObject('assets/cube.obj', 1, offset=[0, 0, -20], scale=10)
scene += readObject('assets/cube.obj', 1, offset=[0, 20, 0], scale=10)
scene += readObject('assets/cube.obj', 0, offset=[0, 14.9, 0], scale=5)
scene += readObject('assets/cube.obj', 4, offset=[20, 0, 0], scale=10)
scene += readObject('assets/cube.obj', 2, offset=[0, 0, 20], scale=10)
scene_material_id = [i[3] for i in scene]
scene = [i[:3] for i in scene]
matattrs = [
    [[0, 0, 0], [4, 4, 4], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0.3, 0.3, 0.3], [0, 0, 0], [0, 0, 0]],
    [[2, 0, 0], [0.8, 0.8, 1.0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0.3, 0.0, 0.0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0.0, 0.3, 0.0], [0, 0, 0], [0, 0, 0]]
]
mesh_desc = np.array(scene)
mesh_material_desc = np.array(scene_material_id)
matattrs_np = np.array(matattrs, dtype=np.float32)

N_TRIANGLES = len(mesh_desc)
IMG_HEIGHT = IMG_WEIGHT = 256
CLIP_N = CLIP_R = CLIP_H = 0.1
N_MATERIALS = 100

mesh_vertices = ti.Vector.field(3, ti.f32, (N_TRIANGLES, 3))
mesh_material_id = ti.field(ti.i32, (N_TRIANGLES))
material_attributes = ti.Vector.field(3, ti.f32, (N_MATERIALS, 4))

img = ti.Vector.field(3, ti.f32, (IMG_HEIGHT, IMG_WEIGHT))
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


mesh_vertices.from_numpy(mesh_desc)
mesh_material_id.from_numpy(mesh_material_desc)
material_attributes.from_numpy(matattrs_np)
cam_pos .from_numpy(np.array([0., 0.1, 0.15]))
cam_gaze .from_numpy(np.array([0., 0., -1]))
cam_top .from_numpy(np.array([0., 1, 0]))
clip_n .from_numpy(np.array(CLIP_N))
clip_r .from_numpy(np.array(CLIP_R))
clip_h .from_numpy(np.array(CLIP_H))
img_w .from_numpy(np.array(IMG_WEIGHT))
img_h .from_numpy(np.array(IMG_HEIGHT))
img.from_numpy(np.zeros((IMG_HEIGHT, IMG_WEIGHT, 3)))
light_pos.from_numpy(np.array([0., 2, 7]))
light_int.from_numpy(np.array([100., 100., 100.]))


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
        if t > 0 and ans_t - t > 1e-3 and b1 > 0 and b2 > 0 and b1 + b2 < 1:
            ans_t, ans_b1, ans_b2, ans_obj_id = t, b1, b2, i
    return ans_t, ans_b1, ans_b2, ans_obj_id


# @ti.func
# def checkVisibility(p, q):
#     d = (q-p).normalized()
#     p1 = p + d * 1e-4
#     thres = (q-p).norm() - 1e-4
#     t, b1, b2, obj = getIntersection(p1, d)
#     return t > thres


@ti.func
def generateInitialRay(img_x, img_y):
    cam_handle = cam_gaze[None].cross(cam_top[None]).normalized()
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
    t = ti.Vector([1, 2, 3], dt=ti.f32)
    ex = t.cross(normal).normalized()
    ey = ex.cross(normal).normalized()
    ez = normal
    r1 = ti.random()
    r2 = ti.random()
    r = ti.sqrt(r1)
    a = 2*3.14159*r2
    x = r*ti.cos(a)
    y = r*ti.sin(a)
    z = ti.sqrt(1-r1)
    w = x*ex + y*ey + z*ez
    return w.normalized()


@ti.kernel
def render():
    SPP = 32
    for x, y in img:
        tans = ti.Vector([0., 0., 0.], dt=ti.f32)
        for sp in range(SPP):
            orig, dir = generateInitialRay(x, y)
            N_BOUNCE = 6    # RR will cause warp divergence
            ans = ti.Vector([0., 0., 0.], dt=ti.f32)
            coef = ti.Vector([1., 1., 1.], dt=ti.f32)
            for _ in range(N_BOUNCE):
                t, bc1, bc2, triangle_id = getIntersection(orig, dir)
                hit_pos = orig + dir * t
                p0 = mesh_vertices[triangle_id, 0]
                p1 = mesh_vertices[triangle_id, 1]
                p2 = mesh_vertices[triangle_id, 2]
                material_id = mesh_material_id[triangle_id]
                material_type_id = material_attributes[material_id, 0][0]
                normal = (p1-p0).cross(p2-p0).normalized()

                if triangle_id != -1 and normal.dot(-dir) > 0:
                    # Implement different materials here
                    if material_type_id == 0:
                        # Area light
                        ans += material_attributes[material_id, 1] * coef
                        break
                    elif material_type_id == 1:
                        # Pure lambert
                        brdf = material_attributes[material_id, 1]
                        # if checkVisibility(hit_pos, light_pos[None]):
                        #     ans += light_int[None] * brdf / (hit_pos - light_pos[None]).norm() ** 2 * max(
                        #         normal.dot((light_pos[None]-hit_pos).normalized()), 0.) * coef
                        wi = sample_brdf(normal)
                        coef *= brdf * 3.14159
                        orig = hit_pos + wi * 1e-4
                        dir = wi
                    elif material_type_id == 2:
                        # Pure specular
                        brdf = material_attributes[material_id, 1]
                        theta = normal.dot(-dir)
                        wi = 2*ti.cos(theta)*normal+dir
                        wi = wi.normalized()
                        coef *= brdf
                        orig = hit_pos + wi * 1e-4
                        dir = wi

                else:
                    break
            tans += ans
        img[x, y] = ti.pow(tans / SPP, 2.2)


gui = ti.GUI(res=(IMG_WEIGHT, IMG_HEIGHT))
frame_id = 0

ina_r = 3.0
ina_h = 1.0

while True:
    stt = tm()
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            break
    if gui.is_pressed(ti.GUI.LEFT, 'a'):
        ina_r -= 0.1
    if gui.is_pressed(ti.GUI.RIGHT, 'd'):
        ina_r += 0.1
    if gui.is_pressed(ti.GUI.UP, 'w'):
        ina_h += 0.1
    if gui.is_pressed(ti.GUI.DOWN, 's'):
        ina_h -= 0.1

    cam_pos .from_numpy(
        np.array([ina_r*ti.cos(frame_id * 0.02), ina_h, ina_r*ti.sin(frame_id * 0.02)]))
    cam_gaze[None] = -cam_pos[None]
    cam_gaze[None] = cam_gaze[None].normalized()

    render()
    gui.set_image(img)
    gui.show()
    if frame_id % 10 == 0:
        print("time usage:", tm()-stt, " able fps:", 1./(tm()-stt+1e-9))
    frame_id += 1
