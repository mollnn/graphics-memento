import numpy as np
from matplotlib import pyplot as plt, image as mpimg
import random
import math
import copy

from numpy.linalg.linalg import norm


def checksgn(x):
    return x >= 0


def check(p, i, j):
    if i < min(x[0] for x in p):
        return 0
    if i > max(x[0] for x in p):
        return 0
    if j < min(x[1] for x in p):
        return 0
    if j > max(x[1] for x in p):
        return 0
    p = np.array([[x[0], x[1], 0] for x in p])
    return checksgn(np.cross(p[1]-p[0], p[0]-[i, j, 0])[2]) == checksgn(np.cross(p[2]-p[1], p[1]-[i, j, 0])[2]) == checksgn(np.cross(p[0]-p[2], p[2]-[i, j, 0])[2])


def barycentric(p, i, j):
    # p must be 3d and in screen space (i.e. z=0)
    q = np.array([i, j, 0])
    total = norm(np.cross(p[1]-p[0], p[2]-p[0]), ord=2)
    w0 = norm(np.cross(p[1]-q, p[2]-q), ord=2)
    w1 = norm(np.cross(p[2]-q, p[0]-q), ord=2)
    w2 = norm(np.cross(p[0]-q, p[1]-q), ord=2)
    return [w0/total, w1/total, w2/total]


def getTexVal(tex, x, y):
    tex_w = len(tex[0])
    tex_h = len(tex)
    if x < 0 or y < 0 or x >= tex_w or y >= tex_h:
        return np.array([0, 0, 0])
    return tex[y][x] / 255


def getTexValI(tex, x, y):
    tex_w = len(tex[0])
    tex_h = len(tex)
    x *= tex_w
    y *= tex_h
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = x - x0
    dy = y - y0
    c00 = getTexVal(tex, x0, y0)
    c01 = getTexVal(tex, x0, y1)
    c10 = getTexVal(tex, x1, y0)
    c11 = getTexVal(tex, x1, y1)
    w00 = (1 - dx) * (1-dy)
    w01 = (1 - dx) * dy
    w10 = dx * (1-dy)
    w11 = dx * dy
    ans = np.array(w00 * c00 + w01 * c01 + w10 * c10 + w11 * c11)
    return ans


img_w = 512
img_h = 512
clip_n, clip_f, clip_l, clip_r, clip_t, clip_b = -3, -10, -1, 1, 1, -1

img = [[np.array([0., 0, 0]) for j in range(img_w)] for i in range(img_h)]
zbuf = [[abs(clip_f)+1 for j in range(img_w)] for i in range(img_h)]

camera_pos_vs = np.array([0., 1, 2])
camera_gaze = np.array([0., -0.5, -2])
camera_gaze /= np.linalg.norm(camera_gaze, ord=2)
camera_up = [
    np.cross(np.cross(camera_gaze, np.array([0., 1, 0])), camera_gaze)]
camera_up /= np.linalg.norm(camera_up, ord=2)
camera_handle = np.cross(camera_gaze, camera_up)
camera_handle /= np.linalg.norm(camera_handle, ord=2)

camera_pos_vs = np.append(camera_pos_vs, 1)
camera_gaze = np.append(camera_gaze, 0)
camera_up = np.append(camera_up, 0)
camera_handle = np.append(camera_handle, 0)

transform_view_translate = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-camera_pos_vs[0], -camera_pos_vs[1], -camera_pos_vs[2], 1]]).T
transform_view_rotate = np.concatenate(
    ([camera_handle], [camera_up], [-camera_gaze], [[0, 0, 0, 1]]))
transform_view = transform_view_rotate @ transform_view_translate

transform_proj_persp = np.array([[clip_n, 0, 0, 0], [0, clip_n, 0, 0], [
                                0, 0, clip_n+clip_f, -clip_n*clip_f], [0, 0, 1, 0]])
transform_proj_ortho = np.array([[2/(clip_r-clip_l), 0, 0, 0], [0, 2/(clip_t-clip_b), 0, 0], [0, 0, 2/(clip_n-clip_f), 0],
                                [(clip_l+clip_r)/(clip_r-clip_l), (clip_t+clip_b)/(clip_t-clip_b), (clip_n+clip_f)/(clip_n-clip_f), 1]]).T
transform_proj = transform_proj_ortho @ transform_proj_persp

transform_viewport = np.array(
    [[img_w / 2, 0, 0, img_w/2], [0, -img_h/2, 0, img_h/2], [0, 0, 0, 0], [0, 0, 0, 1]])
transform = transform_viewport @ transform_proj @ transform_view



####################################

# mat: {ambi: [r, g, b], diff: "tex", spec: [r, g, b], specpow: p}
# mesh: [[x, y, z, u, v], [x, y, z, u, v], [x, y, z, u, v], [matid]]

mat = [
    {"ambi": np.array([0.1, 0.1, 0.1]), "diff": "assets/wood.jpg",
     "spec": np.array([1, 1, 1]), "specpow": 10}
]

mesh = [
    [[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [1, 0, -1, 0, 1], [0]],
    [[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [-1, 0, -1, 0, 1], [0]],
    [[0, 0, -2, 0, 0], [0, 1, 0, 1, 0], [-1, 0, -1, 0, 1], [0]],
    [[0, 0, -2, 0, 0], [0, 1, 0, 1, 0], [-1, 0, -1, 0, 1], [0]]
]

lights = [
    {"int": np.array([10, 10, 10]), "pos":np.array([1, 1, 2]) },
    {"int": np.array([10, 10, 10]), "pos":np.array([0, 1, 10]) }
]

light_vs_cached = [
    {
        "int": i["int"],
        "pos": (transform_view @ np.array(i["pos"].tolist() + [1]))[:3]
    }
    for i in lights
]


# for each triangle
for tri in mesh:
    mat_id = tri[3][0]

    triangle_vert = [np.array(i[:3] + [1]) for i in tri[:3]]
    triangle_uv = [np.array(i[3:]) for i in tri[:3]]

    ma = mat[mat_id]
    triangle_ambi = ma["ambi"]
    triangle_spec = ma["spec"]
    triangle_specpow = ma["specpow"]
    triangle_tex_filename = ma["diff"]
    triangle_tex = mpimg.imread(triangle_tex_filename).copy()

    vert_ws = triangle_vert
    vert_vs = [transform_view @ i for i in vert_ws]
    vert_ss = [transform @ i for i in vert_ws]
    vert_ss = [i[:3]/i[3] for i in vert_ss]

    for i in range(img_h):
        for j in range(img_w):
            if check(vert_ss, j, i):
                px = j
                py = i
                # Evaluate interpolated z, u, v
                bc = barycentric(vert_ss, px, py)
                z = abs(1 / sum(bc[k] / vert_vs[k][2] for k in range(3)))
                u = -z * sum(triangle_uv[k][0] * bc[k] /
                            vert_vs[k][2] for k in range(3))
                v = -z * sum(triangle_uv[k][1] * bc[k] /
                            vert_vs[k][2] for k in range(3))
                x = (px/img_w-0.5)*clip_r*z/abs(clip_n)
                y = (py/img_h-0.5)*clip_t*z/abs(clip_n)
                p_vs = np.array([x, y, z])

                if z >= zbuf[i][j]:
                    continue

                zbuf[i][j] = z

                # Fragment Shader
                coef_ambi = copy.deepcopy(triangle_ambi)
                coef_spec = copy.deepcopy(triangle_spec)
                coef_specpow = copy.deepcopy(triangle_specpow)
                coef_diff = getTexValI(triangle_tex, u, v)
                camera_pos_vs = np.array([0, 0, 0])

                view_dir_vs = camera_pos_vs - p_vs
                view_dir_vs /= norm(view_dir_vs, ord=2)
                normal_dir_vs = np.cross(
                    vert_vs[1][:3]-vert_vs[0][:3], vert_vs[2][:3]-vert_vs[0][:3])
                normal_dir_vs /= norm(normal_dir_vs, ord=2)

                color = coef_ambi

                for light in light_vs_cached:
                    light_intensity = light["int"]
                    light_pos_vs = light["pos"]   # already in vs!

                    light_dist_vec = light_pos_vs - p_vs
                    light_dir_vs = light_pos_vs - p_vs
                    light_dir_vs /= norm(light_dir_vs, ord=2)
                    half_dir_vs = light_dir_vs + view_dir_vs
                    half_dir_vs /= norm(half_dir_vs, ord=2)

                    color += coef_diff * light_intensity / \
                        light_dist_vec.dot(light_dist_vec) * \
                        max(0, light_dir_vs.dot(normal_dir_vs))
                    color += coef_spec * light_intensity / \
                        light_dist_vec.dot(
                            light_dist_vec) * (max(0, half_dir_vs.dot(normal_dir_vs)) ** coef_specpow)

                img[i][j] = color

plt.imshow(img)
plt.show()
