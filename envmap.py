import numpy as np
import taichi as ti
from matplotlib import pyplot as plt
import random
import math
from numpy.linalg.linalg import norm
import time
import os

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
    "assets/envmap.jfif"
]


# Material


material_attr_vec = [
    [[0.1,0.1,0.1], [0.2,0.2,0.2], [0.0,0.0,0.0], [10,0,0]]
]

material_attr_int = [
    [0,-1]
]

material_dict = {
    "default": 0
}

mtl_file_memento = []


# Ai: [illum model, tex1, tex2, ...]
# Av For Blinn-Phong: [Ka, Kd, Ks, [Ns, 0, 0]]


def readMaterial(filename):
    (filepath,_) = os.path.split(filename)
    if filename in mtl_file_memento:
        return
    mtl_file_memento.append(filename)
    fp = open(filename, 'r')
    fl = fp.readlines()
    mtl_vec = []
    mtl_int = []
    mtl_id=-1
    for s in fl:
        a = s.split()
        if len(a)>0:
            if a[0] == 'newmtl':
                if mtl_vec != []:
                    material_attr_vec.append(mtl_vec)
                    material_attr_int.append(mtl_int)
                mtl_vec=[[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
                mtl_int=[0,-1]
                mtl_name = filename + "::" + a[1]
                mtl_id=len(material_attr_vec)
                material_dict[mtl_name] = mtl_id
            elif a[0] == 'Ka':
                mtl_vec[0]=list(map(float,a[1:4]))
            elif a[0] == 'Kd':
                mtl_vec[1]=list(map(float,a[1:4]))
            elif a[0] == 'Ks':
                mtl_vec[2]=list(map(float,a[1:4]))
            elif a[0] == 'Ns':
                mtl_vec[3][0]=float(a[1])
            elif a[0] == 'map_Kd':
                a[1] = filepath + '/' + a[1]
                if a[1] not in textures_filename: textures_filename.append(a[1])
                mtl_int[1]=textures_filename.index(a[1])
    if mtl_vec != []:
        material_attr_vec.append(mtl_vec)
        material_attr_int.append(mtl_int)


def readObject(filename, offset=[0, 0, 0], scale=1):
    (filepath,_)  = os.path.split(filename)
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
            if a[0]=='mtllib':
                mtl_filename = filepath + '/' + a[1]
                readMaterial(mtl_filename)
            elif a[0]=='usemtl':
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
scene += readObject('assets/sphere.obj',  offset=[0, 0, 0], scale=1)
scene += readObject('assets/test_r.obj',  offset=[0, -1, 0], scale=1)


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

camera_pos_vec4 = np.concatenate(
    (camera_pos, np.array([1.0], dtype=np.float32)))
camera_gaze_vec4 = np.concatenate(
    (camera_gaze, np.array([1.0], dtype=np.float32)))
camera_up_vec4 = np.concatenate((camera_up, np.array([1.0], dtype=np.float32)))
camera_hand_vec4 = np.concatenate(
    (camera_hand, np.array([1.0], dtype=np.float32)))

clip_n, clip_f, clip_l, clip_r, clip_t, clip_b = -0.2, -100, -0.1, 0.1, 0.1, -0.1

transform_view = np.concatenate(
    ([camera_hand_vec4], [camera_up_vec4], [-camera_gaze_vec4], [[0, 0, 0, 1]])
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
    [IMG_WIDTH / 2, 0, 0, IMG_WIDTH/2],
    [0, IMG_HEIGHT/2, 0, IMG_HEIGHT/2],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

transform = transform_viewport @ transform_proj @ transform_view
light_dir = normalized(np.array([-1., 1., 1.]))

transform_dev = ti.Matrix.field(4, 4, ti.f32, [])
transform_dev.from_numpy(transform)

transform_view_dev = ti.Matrix.field(4, 4, ti.f32, [])
transform_view_dev.from_numpy(transform_view)

camera_pos_dev = ti.Vector.field(3, ti.f32, [])

# Lighting
light_pos = np.array([[6, 6, 6]], dtype=np.float32)
light_int = np.array([[100, 100, 100]], dtype=np.float32) 
N_LIGHT = light_pos.shape[0]
light_pos_dev = ti.Vector.field(3, ti.f32, [N_LIGHT])
light_int_dev = ti.Vector.field(3, ti.f32, [N_LIGHT])
light_pos_dev.from_numpy(light_pos)
light_int_dev.from_numpy(light_int)


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
def getEnvUV(n):
    nn4 = transform_view_dev[None].inverse() @ ti.Vector([n[0],n[1],n[2],0.0])
    nn = ti.Vector([nn4[0],nn4[1],nn4[2]]).normalized()
    theta = ti.asin(nn[1]) + 3.14159 / 2
    cos_theta = ti.sqrt(1.0-nn[1]**2) 
    phi = ti.atan2(nn[0]/cos_theta, nn[2]/cos_theta) + 3.14159
    env_u = phi / 2.0 / 3.14159
    env_v = theta / 3.14159
    return env_u, env_v

@ti.func
def fragmentShader(u, v, P, n, Wo, material_id):
    illum_mode = material_attr_int_dev[material_id, 0]
    tex1 = material_attr_int_dev[material_id, 1]
    tex1_color = ti.Vector([1.0, 1.0, 1.0]) if tex1 == -1 else getTexColorBI(tex1, u, v)
    Ka = material_attr_vec_dev[material_id, 0]
    Kd = material_attr_vec_dev[material_id, 1]
    if tex1 != -1: 
        Kd = tex1_color
    Ks = material_attr_vec_dev[material_id, 2]
    Ns = material_attr_vec_dev[material_id, 3][0]
    # n = (cos t cos i, sin t, cos t sin i)
    Wi = (2*n-Wo).normalized()
    env_u, env_v = getEnvUV(Wi)
    # result_ambient = Ka
    # result_diffuse = Kd * getTexColorBI(0, env_u, env_v) * max(0, n.dot(Wi))
    h = (Wi + Wo).normalized()
    return getTexColorBI(0, env_u, env_v) 


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


@ti.kernel
def render():
    for x, y in framebuffer_dev:
        framebuffer_z_dev[x, y] = 1e9
        framebuffer_dev[x, y] = ti.Vector([0., 0., 0.])
    for x, y in framebuffer_dev:
        for i in range(n_triangles):
            p0_ws, p1_ws, p2_ws = scene_vertices_dev[i,
                                                     0], scene_vertices_dev[i, 1], scene_vertices_dev[i, 2]
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
            uv0, uv1, uv2 = scene_uvcoords_dev[i,
                                               0], scene_uvcoords_dev[i, 1], scene_uvcoords_dev[i, 2]
            p0_ss = ti.Vector( 
                [p0_ss4[0]/p0_ss4[3], p0_ss4[1]/p0_ss4[3], p0_ss4[2]/p0_ss4[3]])
            p1_ss = ti.Vector(
                [p1_ss4[0]/p1_ss4[3], p1_ss4[1]/p1_ss4[3], p1_ss4[2]/p1_ss4[3]])
            p2_ss = ti.Vector(
                [p2_ss4[0]/p2_ss4[3], p2_ss4[1]/p2_ss4[3], p2_ss4[2]/p2_ss4[3]])
            p0_ss[2], p1_ss[2], p2_ss[2] = 0, 0, 0
            z_vs = -interpZ(p0_ss, p1_ss, p2_ss, x, y,
                            p0_vs[2], p1_vs[2], p2_vs[2])
            x_vs = interpV(p0_ss, p1_ss, p2_ss, x, y,
                           p0_vs[2], p1_vs[2], p2_vs[2], p0_vs[0], p1_vs[0], p2_vs[0])
            y_vs = interpV(p0_ss, p1_ss, p2_ss, x, y,
                           p0_vs[2], p1_vs[2], p2_vs[2], p0_vs[1], p1_vs[1], p2_vs[1])
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

            if checkInside(p0_ss, p1_ss, p2_ss, x, y) and -z_vs < framebuffer_z_dev[x, y] and z_vs < clip_n:
                answer = ti.Vector([0., 0., 0.])
                material_id = scene_material_dev[i]
                color = fragmentShader(
                    u, v,
                    p_vs,
                    cross(p1_vs-p0_vs, p2_vs-p0_vs).normalized(),
                    (camera_pos_vs-p_vs).normalized(),
                    material_id,
                )
                answer += color
                framebuffer_dev[x, y] = answer
                framebuffer_z_dev[x, y] = -z_vs
        if framebuffer_z_dev[x,y] > 9e8:
            rx = x /IMG_WIDTH * 2 - 1
            ry = y /IMG_HEIGHT * 2 - 1
            ez = -clip_n
            ex = clip_r
            ey = clip_t
            dir = ti.Vector([-ex*rx, ey*ry, -ez])
            env_u, env_v = getEnvUV(dir)
            framebuffer_dev[x,y] = getTexColorBI(0, env_u, env_v)


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

    transform = transform_viewport @ transform_proj @ transform_view
    light_dir = normalized(np.array([-1., 1., 1.]))

    transform_dev.from_numpy(transform)
    transform_view_dev.from_numpy(transform_view)

    render()
    gui.set_image(framebuffer_dev)
    gui.show()
