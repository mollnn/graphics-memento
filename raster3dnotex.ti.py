import numpy as np
import taichi as ti
from matplotlib import pyplot as plt
import random
import math
from numpy.linalg.linalg import norm
import time

ti.init(arch=ti.cuda, debug=False)


def readObject(filename, material_id, offset=[0, 0, 0], scale=1):
    fp = open(filename, 'r')
    fl = fp.readlines()
    verts = [[]]
    verts_t = [[]]
    ans = []
    for s in fl:
        a = s.split()
        if len(a) > 0:
            if a[0] == 'v':
                verts.append(np.array([float(a[1]), float(a[2]), float(a[3])]))
            elif a[0] == 'vt':
                verts_t.append(np.array([float(a[1]), float(a[2])]))
            elif a[0] == 'f':
                b = a[1:]
                b = [i.split('/') for i in b]
                ans.append(
                    [verts[int(b[0][0])]*scale+offset, verts[int(b[1][0])]*scale+offset, verts[int(b[2][0])]*scale+offset,
                     verts_t[int(b[0][0])] if len(verts_t) > 1 else [0.0, 0.0],
                     verts_t[int(b[1][0])] if len(verts_t) > 1 else [0.0, 0.0],
                     verts_t[int(b[2][0])] if len(verts_t) > 1 else [0.0, 0.0],
                     material_id])
    return ans


def normalized(x):
    return x / norm(x)


scene = []
scene += readObject('assets/bunny.obj', 0, offset=[0, 0, 0], scale=10)
scene += readObject('assets/cube.obj', 0, offset=[0, -1, 0], scale=1)


scene_vertices = np.array([i[:3] for i in scene])
scene_uvcoords = np.array([i[3:6] for i in scene])
scene_material = np.array([i[6] for i in scene])

n_triangles = scene_vertices.shape[0]
scene_vertices_dev = ti.Vector.field(3, ti.f32, (n_triangles, 3))
scene_vertices_dev.from_numpy(scene_vertices)


IMG_WIDTH = 384
IMG_HEIGHT = 256

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

clip_n, clip_f, clip_l, clip_r, clip_t, clip_b = -2, -100, -1, 1, 1, -1

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

light_pos = np.array([6, 6, 6], dtype=np.float32)
light_int = np.array([1.0, 0.5, 0.0], dtype=np.float32) * 100
light_pos_dev = ti.Vector.field(3, ti.f32, [])
light_int_dev = ti.Vector.field(3, ti.f32, [])
light_pos_dev.from_numpy(light_pos)
light_int_dev.from_numpy(light_int)


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
def fragmentShader(P, n, Wo, Ka, Kd, Ks, Ns, Pl, Il):
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
def fmod(x,y):
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
            p_vs = ti.Vector([x_vs, y_vs, z_vs])
            p_vs4 = ti.Vector([x_vs, y_vs, z_vs, 1.0])
            p_ws4 = transform_view_dev[None].inverse() @ p_vs4
            p_ws = ti.Vector([p_ws4[0]/p_ws4[3], p_ws4[1] /
                             p_ws4[3], p_ws4[2]/p_ws4[3]])
            light_pos = light_pos_dev[None]
            light_int = light_int_dev[None]
            light_pos_ws4 = ti.Vector(
                [light_pos[0], light_pos[1], light_pos[2], 1.0])
            light_pos_vs4 = transform_view_dev[None] @ light_pos_ws4
            light_pos_vs = ti.Vector([light_pos_ws4[0]/light_pos_ws4[3],
                                     light_pos_ws4[1]/light_pos_ws4[3], light_pos_ws4[2]/light_pos_ws4[3]])
            camera_pos = camera_pos_dev[None]
            camera_pos_ws4 = ti.Vector(
                [camera_pos[0], camera_pos[1], camera_pos[2], 1.0])
            camera_pos_vs4 = transform_view_dev[None] @ camera_pos_ws4
            camera_pos_vs = ti.Vector([camera_pos_ws4[0]/camera_pos_ws4[3],
                                      camera_pos_ws4[1]/camera_pos_ws4[3], camera_pos_ws4[2]/camera_pos_ws4[3]])
            if checkInside(p0_ss, p1_ss, p2_ss, x, y) and -z_vs < framebuffer_z_dev[x, y] and z_vs < clip_n:
                color = fragmentShader(
                    p_vs,
                    cross(p1_vs-p0_vs, p2_vs-p0_vs).normalized(),
                    (camera_pos_vs-p_vs).normalized(),
                    ti.Vector([0.1, 0.1, 0.2]),
                    ti.Vector([0.5, 0.5, 0.5]),
                    ti.Vector([0.5, 0.5, 0.5]) * 10,
                    10,
                    light_pos_vs,
                    light_int
                )
                framebuffer_dev[x, y] = color
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

    if gui.is_pressed(ti.GUI.CTRL) and gui.is_pressed(ti.GUI.LMB):
        camera_pos -= dy * camera_up + dx * camera_hand
    elif gui.is_pressed(ti.GUI.SHIFT) and gui.is_pressed(ti.GUI.LMB):
        camera_pos -= dy * camera_gaze
    elif gui.is_pressed(ti.GUI.LMB):
        phi -= - dx * 6.28 * 0.1
        theta -= dy * 3.14 * 0.1

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
