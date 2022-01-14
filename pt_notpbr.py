import random
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from copy import deepcopy
import math
from tqdm import tqdm
from numba import jit


@jit(nopython=True)
def checkIntersect(orig, dir, p0, p1, p2):
    o = orig
    d = dir
    e1 = p1 - p0
    e2 = p2 - p0
    s = o - p0
    s1 = np.cross(d, e2)
    s2 = np.cross(s, e1)
    t = np.dot(s2, e2)
    b1 = np.dot(s1, s)
    b2 = np.dot(s2, d)
    q = np.dot(s1, e1)
    q += 1e-18
    return t/q, b1/q, b2/q


# * tri = [p0, p1, p2, ambient, diffuse, specular, [spec_pw]]


@jit(nopython=True)
def getIntersection(orig, dir, tris):
    ans_t, ans_b1, ans_b2, ans_obj = 2.0e9, 0., 0., -1
    for idx, tri in enumerate(tris):
        p0, p1, p2 = tri[0], tri[1], tri[2]
        t, b1, b2 = checkIntersect(orig, dir, p0, p1, p2)
        if t > 0 and t < ans_t and b1 > 0 and b2 > 0 and b1 + b2 < 1:
            ans_t, ans_b1, ans_b2, ans_obj = t, b1, b2, idx
    return ans_t, ans_b1, ans_b2, ans_obj


@jit(nopython=True)
def checkVisibility(p, q, tris):
    d = q - p
    d /= norm(d)
    p1 = p + d * 1e-8
    q1 = q - d * 1e-8
    thres = norm(q - p) - 1e-7
    t, b1, b2, obj = getIntersection(p1, d, tris)
    return t > thres


@jit(nopython=True)
def generateInitialRay(cam_pos, cam_gaze, cam_top, clip_n, clip_r, clip_h, img_w, img_h, img_x, img_y):
    cam_handle = np.cross(cam_gaze, cam_top)
    canonical_x = (img_x + 0.5) / img_w * 2 - 1
    canonical_y = - (img_y + 0.5) / img_h * 2 + 1
    cam_near_center = cam_pos + cam_gaze * clip_n
    orig = cam_near_center + canonical_x * cam_handle * \
        clip_r + canonical_y * cam_top * clip_h
    dir = orig - cam_pos
    dir /= norm(dir)
    return orig, dir


@jit(nopython=True)
def getNormal(tri):
    p0, p1, p2 = tri[0], tri[1], tri[2]
    ans = np.cross(p1-p0, p2-p0)
    return ans / norm(ans)


@jit(nopython=True)
def sampleUniformSphere():
    r = np.array([random.random() - 0.5, random.random() -
                 0.5, random.random() - 0.5])
    r /= norm(r)
    return r


@jit(nopython=True)
def naiveSampler(normal):
    x = sampleUniformSphere()
    while np.dot(x, normal) < 0:
        x = sampleUniformSphere()
    pdf = 1.
    return x, pdf


@jit(nopython=True)
def unit(x):
    return x / norm(x)


@jit(nopython=True)
def shade(p, wo, obj, lights, tris):
    ans = np.array([0., 0, 0])
    normal = getNormal(obj)
    # Check if point p can be seen by light
    for light in lights:
        if checkVisibility(p, light[0], tris):
            brdf = obj[3] / math.pi
            ans += light[1] * brdf / (norm(p - light[0]) ** 2) * max(
                np.dot(normal, unit(light[0] - p)), 0.)
    # Bounce more
    PRR = 0.9
    if random.random() < PRR:
        wi, pdf = naiveSampler(normal)
        q_t, q_b1, q_b2, q_obj_id = getIntersection(p + wi * 1e-8, wi, tris)
        if q_obj_id != -1:
            q_obj = tris[q_obj_id]
            q = p + q_t * wi
            Lo = shade(q, -wi, q_obj, lights, tris)
            brdf = obj[3] / math.pi
            ans += Lo * brdf * max(np.dot(normal, wi), 0.) / pdf / PRR
    return ans


@jit(nopython=True)
def renderOneLine(tris, lights, cam_pos, cam_gaze, cam_top, clip_n, clip_r, clip_h, img_w, img_h, SPP, img, i):
    for j in range(img_w):
        tans = np.array([0., 0, 0])
        for k in range(SPP):
            orig, dir = generateInitialRay(
                cam_pos, cam_gaze, cam_top, clip_n, clip_r, clip_h, img_w, img_h, j, i)
            t, b1, b2, o = getIntersection(orig, dir, tris)
            if o != -1:
                u = shade(orig + t * dir, -dir,
                          tris[o], lights, tris) / SPP
                tans += u
        img[i][j] = tans


def render(tris, lights, cam_pos, cam_gaze, cam_top, clip_n, clip_r, clip_h, img_w, img_h, SPP):
    img = np.array([[[0., 0, 0] for j in range(img_w)] for i in range(img_h)])
    seq = list(range(img_h))
    random.shuffle(seq)
    for i in tqdm(seq):
        renderOneLine(tris, lights, cam_pos, cam_gaze, cam_top,
                      clip_n, clip_r, clip_h, img_w, img_h, SPP, img, i)
    return img


def main():
    lights = np.array([
        [[-1., 0.5, 0.], [1., 1, 1]]
    ])

    tris = np.array([
        [[0., 0, -10], [-10., 0, 2],
         [10., 0, 2], [0.5, 0, 0]],
        [[0., 1, -100], [100., 1, 100],
         [-100., 1, 100], [1., 1., 1.]],
        [[-3., 0, -10], [-3., 10, 0],
         [-3., 0, 10], [1., 1., 1.]],
        [[3., 0, 10], [3., 10, 0],
         [3., 0, -10], [1., 1., 1.]],
        [[0., 0, -1], [1., 0, 0],
         [0., 1, -1], [0., 0.5, 0]],
        [[-1., 0, 0], [0., 0, -1],
         [0., 1, -1], [0., 0, 0.5]]
    ])

    cam_pos = np.array([0., 0.6667, 1.3333])
    cam_gaze = np.array([0., -0.5, -1])
    cam_gaze /= norm(cam_gaze)
    cam_top = np.array([0., 1, -0.5])
    cam_top /= norm(cam_top)
    clip_n = 1.
    clip_r = 1.
    clip_h = 1.
    img_w = 128
    img_h = 128
    SPP = 16

    img = render(tris, lights, cam_pos, cam_gaze, cam_top,
                 clip_n, clip_r, clip_h, img_w, img_h, SPP)

    plt.imshow(np.sqrt(img))
    plt.show()


main()
