import numpy as np
import taichi as ti
from matplotlib import pyplot as plt
import random
import math
from numpy.linalg.linalg import norm
import time
import os

# todo  Perf shadow mapping

ti.init(arch=ti.cuda, debug=False)

# Textures

TEX_MEM_SIZE = 1048576 * 8
tex_mem_used = [0]
tex_mem = np.array([[0, 0, 0] for i in range(TEX_MEM_SIZE)], dtype=np.float32)


def texAlloc(tex_mem, im_filename, used_):
    used = used_[0]
    im = plt.imread(im_filename)
    im = np.array(im, dtype=np.float32)
    if np.max(im) > 1.0:
        im /= 255.0
    if len(im.shape) == 2:
        im = np.tile(np.expand_dims(im, 2), (1, 1, 3))
    sz = im.shape[0]*im.shape[1]
    tex_mem[used: used+sz] = im.reshape((-1, 3))
    used += sz
    used_[0] = used
    return used-sz, im.shape[0], im.shape[1]


textures_filename = [
]


# Material


material_attr_vec = [
    [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0, 0, 0], [1, 0, 0]]
]

material_attr_int = [
    [0, -1]
]

material_dict = {
    "default": 0
}

mtl_file_memento = []


# Ai: [illum model, tex1, tex2, ...]
# Av For Blinn-Phong: [Ka, Kd, Ks, [Ns, 0, 0]]


def readMaterial(filename):
    (filepath, _) = os.path.split(filename)
    if filename in mtl_file_memento:
        return
    mtl_file_memento.append(filename)
    fp = open(filename, 'r')
    fl = fp.readlines()
    mtl_vec = []
    mtl_int = []
    mtl_id = -1
    for s in fl:
        a = s.split()
        if len(a) > 0:
            if a[0] == 'newmtl':
                if mtl_vec != []:
                    material_attr_vec.append(mtl_vec)
                    material_attr_int.append(mtl_int)
                mtl_vec = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
                mtl_int = [0, -1]
                mtl_name = filename + "::" + a[1]
                mtl_id = len(material_attr_vec)
                material_dict[mtl_name] = mtl_id
            elif a[0] == 'Ka':
                mtl_vec[0] = list(map(float, a[1:4]))
            elif a[0] == 'Kd':
                mtl_vec[1] = list(map(float, a[1:4]))
            elif a[0] == 'Ks':
                mtl_vec[2] = list(map(float, a[1:4]))
            elif a[0] == 'Ns':
                mtl_vec[3][0] = float(a[1])
            elif a[0] == 'map_Kd':
                a[1] = filepath + '/' + a[1]
                if a[1] not in textures_filename:
                    textures_filename.append(a[1])
                mtl_int[1] = textures_filename.index(a[1])
    if mtl_vec != []:
        material_attr_vec.append(mtl_vec)
        material_attr_int.append(mtl_int)


def readObject(filename, offset=[0, 0, 0], scale=1):
    (filepath, _) = os.path.split(filename)
    fp = open(filename, 'r')
    fl = fp.readlines()
    verts = [[]]
    verts_t = [[]]
    ans = []
    mtl_filename = ""
    mtl_id = 0
    for s in fl:
        a = s.split()
        if len(a) > 0:
            if a[0] == 'mtllib':
                mtl_filename = filepath + '/' + a[1]
                readMaterial(mtl_filename)
            elif a[0] == 'usemtl':
                mtl_name = mtl_filename+"::"+a[1]
                mtl_id = material_dict[mtl_name]
            elif a[0] == 'v':
                verts.append(np.array([float(a[1]), float(a[2]), float(a[3])]))
            elif a[0] == 'vt':
                verts_t.append(np.array([float(a[1]), float(a[2])]))
            elif a[0] == 'f':
                b = a[1:]
                b = [i.split('/') for i in b]
                ans.append(
                    [verts[int(b[0][0])]*scale+offset, verts[int(b[1][0])]*scale+offset, verts[int(b[2][0])]*scale+offset,
                     verts_t[int(b[0][1])] if len(verts_t) > 1 else [0.0, 0.0],
                     verts_t[int(b[1][1])] if len(verts_t) > 1 else [0.0, 0.0],
                     verts_t[int(b[2][1])] if len(verts_t) > 1 else [0.0, 0.0],
                     mtl_id])
    return ans


def normalized(x):
    return x / norm(x)


# Scene

scene = []
scene += readObject('assets/rock.obj',  offset=[0, 0.6, 0], scale=1)
scene += readObject('assets/cube.obj',  offset=[0, -5, 0], scale=5)


scene_vertices = np.array([i[:3] for i in scene])
scene_uvcoords = np.array([i[3:6] for i in scene])
scene_material = np.array([i[6] for i in scene])


n_triangles = scene_vertices.shape[0]
scene_vertices_dev = ti.Vector.field(3, ti.f32, (n_triangles, 3))
scene_uvcoords_dev = ti.Vector.field(2, ti.f32, (n_triangles, 3))
scene_material_dev = ti.field(ti.i32, (n_triangles))
scene_vertices_dev.from_numpy(scene_vertices)
scene_uvcoords_dev.from_numpy(scene_uvcoords)
scene_material_dev.from_numpy(scene_material)


# Commit Texture

N_TEXTURE = len(textures_filename)

textures_desc = [texAlloc(tex_mem, i, tex_mem_used) for i in textures_filename]

tex_mem_dev = ti.Vector.field(3, ti.f32, (TEX_MEM_SIZE))
tex_desc_dev = ti.Vector.field(3, ti.i32, (N_TEXTURE))
tex_mem_dev.from_numpy(tex_mem)
tex_desc_dev.from_numpy(np.array(textures_desc))


# Commit Material

N_MATERIALS = len(material_attr_int)
material_attr_vec_dev = ti.Vector.field(3, ti.f32, (N_MATERIALS, 4))
material_attr_vec_dev.from_numpy(np.array(material_attr_vec, np.float32))
material_attr_int_dev = ti.field(ti.i32, (N_MATERIALS, 2))
material_attr_int_dev.from_numpy(np.array(material_attr_int, np.int32))

# Camera

IMG_WIDTH = 512
IMG_HEIGHT = 512

framebuffer_dev = ti.Vector.field(3, ti.f32, (IMG_WIDTH, IMG_HEIGHT))
framebuffer_z_dev = ti.field(ti.f32, (IMG_WIDTH, IMG_HEIGHT))


camera_pos = np.array([0, 4, 5], dtype=np.float32)
camera_gaze = normalized(np.array([0, -0.5, -1], dtype=np.float32))
camera_grav = np.array([0, -1, 0], dtype=np.float32)
camera_hand = normalized(np.cross(camera_grav, camera_gaze))
camera_up = normalized(np.cross(camera_hand, camera_gaze))


def makeCamera(
    camera_pos, camera_gaze, camera_up, camera_hand, fov, asp, img_w, img_h
):
    clip_n = -0.1
    clip_f = -100
    clip_l = -0.1 * math.tan(fov/2)
    clip_r = -clip_l
    clip_t = clip_r / asp
    clip_b = -clip_t
    camera_pos_vec4 = np.concatenate(
        (camera_pos, np.array([1.0], dtype=np.float32)))
    camera_gaze_vec4 = np.concatenate(
        (camera_gaze, np.array([1.0], dtype=np.float32)))
    camera_up_vec4 = np.concatenate(
        (camera_up, np.array([1.0], dtype=np.float32)))
    camera_hand_vec4 = np.concatenate(
        (camera_hand, np.array([1.0], dtype=np.float32)))

    transform_view = np.concatenate(
        ([camera_hand_vec4], [camera_up_vec4],
         [-camera_gaze_vec4], [[0, 0, 0, 1]])
    ) @ (np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-camera_pos[0], -camera_pos[1], -camera_pos[2], 1]
    ], dtype=np.float32).T)

    transform_proj = np.array([
        [2/(clip_r-clip_l), 0, 0, 0],
        [0, 2/(clip_t-clip_b), 0, 0],
        [0, 0, 2/(clip_n-clip_f), 0],
        [(clip_l+clip_r)/(clip_r-clip_l), (clip_t+clip_b) /
         (clip_t-clip_b), (clip_n+clip_f)/(clip_n-clip_f), 1]
    ], dtype=np.float32).T @ np.array([
        [clip_n, 0, 0, 0],
        [0, clip_n, 0, 0],
        [0, 0, clip_n+clip_f, -clip_n*clip_f],
        [0, 0, 1, 0]
    ], dtype=np.float32)

    transform_viewport = np.array([
        [img_w / 2, 0, 0, img_w/2],
        [0, img_h/2, 0, img_h/2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    transform = transform_viewport @ transform_proj @ transform_view

    return transform, transform_view


fov = 90 / 180 * 3.14159
asp = 1.0

transform, transform_view = makeCamera(
    camera_pos, camera_gaze, camera_up, camera_hand, fov, asp, IMG_WIDTH, IMG_HEIGHT
)

transform_dev = ti.Matrix.field(4, 4, ti.f32, [])
transform_dev.from_numpy(transform)

transform_view_dev = ti.Matrix.field(4, 4, ti.f32, [])
transform_view_dev.from_numpy(transform_view)

camera_pos_dev = ti.Vector.field(3, ti.f32, [])

# Lighting
light_pos = np.array([[6, 6, 6], [0, 5, 0]], dtype=np.float32)
light_int = np.array([[100, 100, 100], [10, 10, 10]], dtype=np.float32)
N_LIGHT = light_pos.shape[0]
light_pos_dev = ti.Vector.field(3, ti.f32, [N_LIGHT])
light_int_dev = ti.Vector.field(3, ti.f32, [N_LIGHT])
light_pos_dev.from_numpy(light_pos)
light_int_dev.from_numpy(light_int)

# !! SHADOW MAPPING
# * Only one map now, for the first light source

SMAP_SIZE = 1024
smap_dev = ti.field(ti.f32, (SMAP_SIZE, SMAP_SIZE))

smap_pos = np.array(light_pos[0], dtype=np.float32)
smap_gaze = np.array([0, -1, 0], dtype=np.float32)
smap_grav = np.array([0, 0, -1], dtype=np.float32)
smap_hand = normalized(np.cross(smap_grav, smap_gaze))
smap_up = normalized(np.cross(smap_hand, smap_gaze))
smap_fov = 2.5
smap_asp = 1.0

smap_transform, smap_transform_view = makeCamera(
    smap_pos, smap_gaze, smap_up, smap_hand, smap_fov, smap_asp, SMAP_SIZE, SMAP_SIZE)

smap_pos_dev = ti.Vector.field(3, ti.f32, [])
smap_gaze_dev = ti.Vector.field(3, ti.f32, [])
smap_grav_dev = ti.Vector.field(3, ti.f32, [])
smap_hand_dev = ti.Vector.field(3, ti.f32, [])
smap_up_dev = ti.Vector.field(3, ti.f32, [])
smap_transform_dev = ti.Matrix.field(4, 4, ti.f32, [])
smap_transform_view_dev = ti.Matrix.field(4, 4, ti.f32, [])

smap_pos_dev.from_numpy(smap_pos)
smap_gaze_dev.from_numpy(smap_gaze)
smap_grav_dev.from_numpy(smap_grav)
smap_hand_dev.from_numpy(smap_hand)
smap_up_dev .from_numpy(smap_up)
smap_transform_dev .from_numpy(smap_transform)
smap_transform_view_dev .from_numpy(smap_transform_view)


@ti.func
def getTexPixel(tex_id, x, y):
    # * x,y must be integer
    tex_addr = tex_desc_dev[tex_id][0]
    h = tex_desc_dev[tex_id][1]
    w = tex_desc_dev[tex_id][2]
    x = min(x, w-1)
    x = max(x, 0)
    y = min(y, h-1)
    y = max(y, 0)
    return tex_mem_dev[tex_addr+w*y+x]


@ti.func
def getTexPixelBI(tex_id, x, y):
    # * x,y can be float
    x0 = ti.cast(ti.floor(x), ti.int32)
    y0 = ti.cast(ti.floor(y), ti.int32)
    x1, y1 = x0 + 1, y0 + 1
    return getTexPixel(tex_id, x0, y0) * (x1-x) * (y1-y) + getTexPixel(tex_id, x0, y1) * (x1-x) * (y-y0) + \
        getTexPixel(tex_id, x1, y0) * (x-x0) * (y1-y) + \
        getTexPixel(tex_id, x1, y1) * (x-x0) * (y-y0)


@ti.func
def getTexColorBI(tex_id, u, v):
    ans = ti.Vector([1., 1., 1.])
    if tex_id >= 0:
        h = tex_desc_dev[tex_id][1]
        w = tex_desc_dev[tex_id][2]
        ans = getTexPixelBI(tex_id, u*w, (1-v)*h)
    return ans


@ti.func
def checksgn(x):
    return x[2] >= 0


@ti.func
def cross(x, y):
    return x.cross(y)


@ti.func
def checkInside(p0, p1, p2, x, y):
    # ! must ensure p0, p1, p2 is in SCREEN SPACE, and .z = 0
    p = ti.Vector([x, y, 0], ti.f32)
    return checksgn(cross(p1-p0, p-p0)) == checksgn(cross(p2-p1, p-p1)) == checksgn(cross(p0-p2, p-p2))


@ti.func
def fragmentShader(u, v, P, n, Wo, material_id, Pl, Il):
    # Only support Blinn-Phong now
    illum_mode = material_attr_int_dev[material_id, 0]
    tex1 = material_attr_int_dev[material_id, 1]
    tex1_color = ti.Vector([1.0, 1.0, 1.0]) if tex1 == - \
        1 else getTexColorBI(tex1, u, v)
    Ka = material_attr_vec_dev[material_id, 0]
    Kd = material_attr_vec_dev[material_id, 1]
    if tex1 != -1:
        Kd = tex1_color
    Ks = material_attr_vec_dev[material_id, 2]
    Ns = material_attr_vec_dev[material_id, 3][0]
    result_ambient = Ka
    result_diffuse = Kd * Il / (Pl - P).dot(Pl - P) * \
        max(0, n.dot((Pl-P).normalized()))
    h = ((Pl-P).normalized() + Wo).normalized()
    result_specular = Ks * Il / \
        (Pl - P).dot(Pl - P) * pow(max(0, n.dot(h)), Ns)
    return result_ambient + result_diffuse + result_specular


@ti.func
def interpZ(p0, p1, p2, x, y, z0, z1, z2):
    # ! must ensure p0, p1, p2, x, y is in SCREEN SPACE, and .z = 0, and z0, z1, z2 is in VIEW SPACE
    # Interp in screen space, and get view space z
    p = ti.Vector([x, y, 0], ti.f32)
    w0 = cross(p1-p, p2-p).norm()
    w1 = cross(p0-p, p2-p).norm()
    w2 = cross(p1-p, p0-p).norm()
    ws = w0 + w1 + w2
    w0, w1, w2 = w0/ws, w1/ws, w2/ws
    return - 1.0 / (w0 / z0 + w1 / z1 + w2 / z2)


@ti.func
def interpV(p0, p1, p2, x, y, z0, z1, z2, v0, v1, v2):
    # ! must ensure p0, p1, p2, x, y is in SCREEN SPACE, and .z = 0, and z0, z1, z2 is in VIEW SPACE
    # Interp in screen space, and get view space z
    p = ti.Vector([x, y, 0], ti.f32)
    w0 = cross(p1-p, p2-p).norm()
    w1 = cross(p0-p, p2-p).norm()
    w2 = cross(p1-p, p0-p).norm()
    ws = w0 + w1 + w2
    w0, w1, w2 = w0/ws, w1/ws, w2/ws
    zp = - 1.0 / (w0 / z0 + w1 / z1 + w2 / z2)
    vp = - (w0 * v0 / z0 + w1 * v1 / z1 + w2 * v2 / z2) * zp
    return vp


@ti.func
def fmod(x, y):
    return x-int(x/y)*y


cache_vertices_vs = ti.Vector.field(3, ti.f32, (n_triangles, 3))
cache_vertices_ss = ti.Vector.field(3, ti.f32, (n_triangles, 3))

@ti.kernel
def renderLightpass():
    # * "camera" in this function refers to LIGHT
    for x, y in smap_dev:
        smap_dev[x, y] = 1e9

    for i in range(n_triangles):
        p0_ws, p1_ws, p2_ws = scene_vertices_dev[i, 0], scene_vertices_dev[i, 1], scene_vertices_dev[i, 2]
        p0_ws4 = ti.Vector([p0_ws[0], p0_ws[1], p0_ws[2], 1], ti.f32)
        p1_ws4 = ti.Vector([p1_ws[0], p1_ws[1], p1_ws[2], 1], ti.f32)
        p2_ws4 = ti.Vector([p2_ws[0], p2_ws[1], p2_ws[2], 1], ti.f32)
        p0_vs4 = smap_transform_view_dev[None] @ p0_ws4
        p1_vs4 = smap_transform_view_dev[None] @ p1_ws4
        p2_vs4 = smap_transform_view_dev[None] @ p2_ws4
        p0_ss4 = smap_transform_dev[None] @ p0_ws4
        p1_ss4 = smap_transform_dev[None] @ p1_ws4
        p2_ss4 = smap_transform_dev[None] @ p2_ws4
        p0_vs = ti.Vector([p0_vs4[0], p0_vs4[1], p0_vs4[2]])
        p1_vs = ti.Vector([p1_vs4[0], p1_vs4[1], p1_vs4[2]])
        p2_vs = ti.Vector([p2_vs4[0], p2_vs4[1], p2_vs4[2]])
        p0_ss = ti.Vector(
            [p0_ss4[0]/p0_ss4[3], p0_ss4[1]/p0_ss4[3], p0_ss4[2]/p0_ss4[3]])
        p1_ss = ti.Vector(
            [p1_ss4[0]/p1_ss4[3], p1_ss4[1]/p1_ss4[3], p1_ss4[2]/p1_ss4[3]])
        p2_ss = ti.Vector(
            [p2_ss4[0]/p2_ss4[3], p2_ss4[1]/p2_ss4[3], p2_ss4[2]/p2_ss4[3]])
        p0_ss[2], p1_ss[2], p2_ss[2] = 0, 0, 0
        cache_vertices_vs[i,0] = p0_vs
        cache_vertices_vs[i,1] = p1_vs
        cache_vertices_vs[i,2] = p2_vs
        cache_vertices_ss[i,0] = p0_ss
        cache_vertices_ss[i,1] = p1_ss
        cache_vertices_ss[i,2] = p2_ss

    for x, y in smap_dev:
        for i in range(n_triangles):
            p0_vs = cache_vertices_vs[i,0]  
            p1_vs = cache_vertices_vs[i,1]  
            p2_vs = cache_vertices_vs[i,2]  
            p0_ss = cache_vertices_ss[i,0]  
            p1_ss = cache_vertices_ss[i,1]  
            p2_ss = cache_vertices_ss[i,2]  
            z_vs = -interpZ(p0_ss, p1_ss, p2_ss, x, y,
                            p0_vs[2], p1_vs[2], p2_vs[2])
            if checkInside(p0_ss, p1_ss, p2_ss, x, y) and -z_vs < smap_dev[x, y] and z_vs < -0.1:
                smap_dev[x, y] = -z_vs


@ti.func
def checkShadow(smap_transform_view, obj_pos):
    obj_pos4 = ti.Vector([obj_pos[0], obj_pos[1], obj_pos[2], 1.0])
    obj_pos_vs4 = smap_transform_view @ obj_pos4
    obj_pos_vs = ti.Vector([obj_pos_vs4[0]/obj_pos_vs4[3], obj_pos_vs4[1]/obj_pos_vs4[3], obj_pos_vs4[2]/obj_pos_vs4[3]])
    dir_vs = - obj_pos_vs / obj_pos_vs[2]
    fx = ti.tan(smap_fov / 2) 
    fy = fx * smap_asp
    dx = dir_vs[0]
    dy = dir_vs[1]
    rx = dx / fx / 2 + 0.5
    ry = dy / fy / 2 + 0.5
    actual_z = -obj_pos_vs[2]
    ans = True
    if rx>0 and rx<1 and ry>0 and ry<1:
        ix = int(rx*SMAP_SIZE)
        iy = int(ry*SMAP_SIZE)
        smap_z = smap_dev[ix, iy] 
        if actual_z - smap_z > 5e-3:
            ans = False
    return ans


@ti.kernel
def render():
    for x, y in framebuffer_dev:
        framebuffer_z_dev[x, y] = 1e9
        framebuffer_dev[x, y] = ti.Vector([0., 0., 0.])
    for i in range(n_triangles):
        p0_ws, p1_ws, p2_ws = scene_vertices_dev[i, 0], scene_vertices_dev[i, 1], scene_vertices_dev[i, 2]
        p0_ws4 = ti.Vector([p0_ws[0], p0_ws[1], p0_ws[2], 1], ti.f32)
        p1_ws4 = ti.Vector([p1_ws[0], p1_ws[1], p1_ws[2], 1], ti.f32)
        p2_ws4 = ti.Vector([p2_ws[0], p2_ws[1], p2_ws[2], 1], ti.f32)
        p0_vs4 = transform_view_dev[None] @ p0_ws4
        p1_vs4 = transform_view_dev[None] @ p1_ws4
        p2_vs4 = transform_view_dev[None] @ p2_ws4
        p0_ss4 = transform_dev[None] @ p0_ws4
        p1_ss4 = transform_dev[None] @ p1_ws4
        p2_ss4 = transform_dev[None] @ p2_ws4
        p0_vs = ti.Vector([p0_vs4[0], p0_vs4[1], p0_vs4[2]])
        p1_vs = ti.Vector([p1_vs4[0], p1_vs4[1], p1_vs4[2]])
        p2_vs = ti.Vector([p2_vs4[0], p2_vs4[1], p2_vs4[2]])
        p0_ss = ti.Vector(
            [p0_ss4[0]/p0_ss4[3], p0_ss4[1]/p0_ss4[3], p0_ss4[2]/p0_ss4[3]])
        p1_ss = ti.Vector(
            [p1_ss4[0]/p1_ss4[3], p1_ss4[1]/p1_ss4[3], p1_ss4[2]/p1_ss4[3]])
        p2_ss = ti.Vector(
            [p2_ss4[0]/p2_ss4[3], p2_ss4[1]/p2_ss4[3], p2_ss4[2]/p2_ss4[3]])
        p0_ss[2], p1_ss[2], p2_ss[2] = 0, 0, 0
        cache_vertices_vs[i,0] = p0_vs
        cache_vertices_vs[i,1] = p1_vs
        cache_vertices_vs[i,2] = p2_vs
        cache_vertices_ss[i,0] = p0_ss
        cache_vertices_ss[i,1] = p1_ss
        cache_vertices_ss[i,2] = p2_ss

    for x, y in framebuffer_dev:
        for i in range(n_triangles):
            p0_vs = cache_vertices_vs[i,0]  
            p1_vs = cache_vertices_vs[i,1]  
            p2_vs = cache_vertices_vs[i,2]  
            p0_ss = cache_vertices_ss[i,0]  
            p1_ss = cache_vertices_ss[i,1]  
            p2_ss = cache_vertices_ss[i,2]  
            z_vs = -interpZ(p0_ss, p1_ss, p2_ss, x, y,
                            p0_vs[2], p1_vs[2], p2_vs[2])
            x_vs = interpV(p0_ss, p1_ss, p2_ss, x, y,
                           p0_vs[2], p1_vs[2], p2_vs[2], p0_vs[0], p1_vs[0], p2_vs[0])
            y_vs = interpV(p0_ss, p1_ss, p2_ss, x, y,
                           p0_vs[2], p1_vs[2], p2_vs[2], p0_vs[1], p1_vs[1], p2_vs[1])
            uv0, uv1, uv2 = scene_uvcoords_dev[i,
                                               0], scene_uvcoords_dev[i, 1], scene_uvcoords_dev[i, 2]
            u = interpV(p0_ss, p1_ss, p2_ss, x, y,
                        p0_vs[2], p1_vs[2], p2_vs[2], uv0[0], uv1[0], uv2[0])
            v = interpV(p0_ss, p1_ss, p2_ss, x, y,
                        p0_vs[2], p1_vs[2], p2_vs[2], uv0[1], uv1[1], uv2[1])
            p_vs = ti.Vector([x_vs, y_vs, z_vs])
            p_vs4 = ti.Vector([x_vs, y_vs, z_vs, 1.0])
            p_ws4 = transform_view_dev[None].inverse() @ p_vs4
            p_ws = ti.Vector([p_ws4[0]/p_ws4[3], p_ws4[1] /
                             p_ws4[3], p_ws4[2]/p_ws4[3]])
            camera_pos = camera_pos_dev[None]
            camera_pos_ws4 = ti.Vector(
                [camera_pos[0], camera_pos[1], camera_pos[2], 1.0])
            camera_pos_vs4 = transform_view_dev[None] @ camera_pos_ws4
            camera_pos_vs = ti.Vector([camera_pos_ws4[0]/camera_pos_ws4[3],
                                      camera_pos_ws4[1]/camera_pos_ws4[3], camera_pos_ws4[2]/camera_pos_ws4[3]])

            if checkInside(p0_ss, p1_ss, p2_ss, x, y) and -z_vs < framebuffer_z_dev[x, y] and z_vs < -0.1:
                answer = ti.Vector([0., 0., 0.])
                for idx_light in range(N_LIGHT):
                    light_pos = light_pos_dev[idx_light]
                    light_int = light_int_dev[idx_light]
                    light_pos_ws4 = ti.Vector(
                        [light_pos[0], light_pos[1], light_pos[2], 1.0])
                    light_pos_vs4 = transform_view_dev[None] @ light_pos_ws4
                    light_pos_vs = ti.Vector([light_pos_ws4[0]/light_pos_ws4[3],
                                              light_pos_ws4[1]/light_pos_ws4[3], light_pos_ws4[2]/light_pos_ws4[3]])
                    material_id = scene_material_dev[i]
                    color = fragmentShader(
                        u, v,
                        p_vs,
                        cross(p1_vs-p0_vs, p2_vs-p0_vs).normalized(),
                        (camera_pos_vs-p_vs).normalized(),
                        material_id,
                        light_pos_vs,
                        light_int
                    )
                    if idx_light>0 or checkShadow(smap_transform_view_dev[None],p_ws):
                        answer += color
                framebuffer_dev[x, y] = answer
                framebuffer_z_dev[x, y] = -z_vs


gui = ti.GUI(res=(IMG_WIDTH, IMG_HEIGHT))
lx, ly = 0.0, 0.0
phi, theta = 0.0, 0.0
while True:

    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            break
    if gui.is_pressed(ti.GUI.LEFT, 'a'):
        camera_pos -= 0.10 * camera_hand
        acc_frame_count = 0
    if gui.is_pressed(ti.GUI.RIGHT, 'd'):
        camera_pos += 0.10 * camera_hand
        acc_frame_count = 0
    if gui.is_pressed(ti.GUI.UP, 'w'):
        camera_pos += 0.10 * camera_gaze
        acc_frame_count = 0
    if gui.is_pressed(ti.GUI.DOWN, 's'):
        camera_pos -= 0.10 * camera_gaze
        acc_frame_count = 0
    if gui.is_pressed('p'):
        camera_pos += 0.10 * camera_up
        acc_frame_count = 0
    if gui.is_pressed('l'):
        camera_pos -= 0.10 * camera_up
        acc_frame_count = 0

    gui.get_event(ti.GUI.MOTION)
    cx, cy = gui.get_cursor_pos()
    dx, dy = cx-lx, cy-ly
    lx, ly = cx, cy

    if gui.is_pressed(ti.GUI.SHIFT) and gui.is_pressed(ti.GUI.LMB):
        camera_pos -= dy * camera_gaze * 3
    elif gui.is_pressed(ti.GUI.LMB):
        camera_pos -= dy * camera_up * 3 + dx * camera_hand * 3
    if gui.is_pressed(ti.GUI.RMB):
        phi += - dx * 6.28 * 0.2
        theta += dy * 3.14 * 0.2

    camera_gaze = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]]) @ \
        np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                 [0, np.sin(theta), np.cos(theta)]]) @ np.array([0., 0., -1.])

    camera_hand = normalized(np.cross(camera_grav, camera_gaze))
    camera_up = normalized(np.cross(camera_hand, camera_gaze))

    camera_pos_vec4 = np.concatenate(
        (camera_pos, np.array([1.0], dtype=np.float32)))
    camera_gaze_vec4 = np.concatenate(
        (camera_gaze, np.array([1.0], dtype=np.float32)))
    camera_up_vec4 = np.concatenate(
        (camera_up, np.array([1.0], dtype=np.float32)))
    camera_hand_vec4 = np.concatenate(
        (camera_hand, np.array([1.0], dtype=np.float32)))

    transform_view = np.concatenate(
        ([camera_hand_vec4], [camera_up_vec4],
         [-camera_gaze_vec4], [[0, 0, 0, 1]])
    ) @ (np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-camera_pos[0], -camera_pos[1], -camera_pos[2], 1]
    ], dtype=np.float32).T)

    transform, transform_view = makeCamera(
        camera_pos, camera_gaze, camera_up, camera_hand, fov, asp, IMG_WIDTH, IMG_HEIGHT
    )

    transform_dev.from_numpy(transform)
    transform_view_dev.from_numpy(transform_view)

    renderLightpass()
    render()

    gui.set_image(framebuffer_dev)

    gui.show()
