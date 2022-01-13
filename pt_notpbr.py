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


def getIntersection(orig, dir, tris):
    ans_t, ans_b1, ans_b2, ans_obj = 2.0e9, 0., 0., 0.
    for tri in tris:
        t, b1, b2 = checkIntersect(orig, dir, tri["p0"], tri["p1"], tri["p2"])
        if t > 0 and t < ans_t and b1 > 0 and b2 > 0 and b1 + b2 < 1:
            ans_t, ans_b1, ans_b2, ans_obj = t, b1, b2, tri
    return ans_t, ans_b1, ans_b2, ans_obj


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


def getNormal(obj):
    p0 = obj["p0"]
    p1 = obj["p1"]
    p2 = obj["p2"]
    ans = np.cross(p1-p0, p2-p0)
    return ans / norm(ans)


def sampleUniformSphere():
    r = np.array([random.random() - 0.5, random.random() -
                 0.5, random.random() - 0.5])
    r /= norm(r)
    return r


def naiveSampler(normal):
    x = sampleUniformSphere()
    while np.dot(x, normal) < 0:
        x = sampleUniformSphere()
    pdf = 1.
    return x, pdf


def unit(x):
    return x / norm(x)


def shade(p, wo, obj, lights, tris):
    ans = 0
    normal = getNormal(obj)
    # Check if point p can be seen by light
    for light in lights:
        if checkVisibility(p, light["pos"], tris):
            ans += light["int"] * obj["c"] / (norm(p - light["pos"]) ** 2) * abs(
                np.dot(normal, unit(light["pos"] - p)))
    # Bounce more
    PRR = 0.95
    if random.random() < PRR:
        wi, pdf = naiveSampler(normal)
        q_t, q_b1, q_b2, q_obj = getIntersection(p + wi * 1e-8, wi, tris)
        if q_obj != 0:
            q = p + q_t * wi
            Lo = shade(q, -wi, q_obj, lights, tris)
            brdf = obj["c"]
            ans += Lo * brdf * abs(np.dot(normal, wi)) / pdf / PRR
    return ans


lights = [
    {"pos": np.array([-0.1, 0.2, -0.6]), "int": 0.1}
]

tris = [
    {"p0": np.array([0., 0, -1]),  "p1": np.array([1., 0, 0]),
     "p2": np.array([-1., 0, 0]), "c": 0.4},
    {"p0": np.array([0., 0, -1]),  "p1": np.array([1., 0, 0]),
     "p2": np.array([0., 1, -1]), "c": 0.7},
    {"p0": np.array([0., 0, -1]),  "p1": np.array([-1., 0, 0]),
     "p2": np.array([0., 1, -1]), "c": 0.7}
]

cam_pos = np.array([0., 0.6667, 1.3333])
cam_gaze = np.array([0., -0.5, -1])
cam_gaze /= norm(cam_gaze)
cam_top = np.array([0., 1, -0.5])
cam_top /= norm(cam_top)
clip_n = 1.
clip_r = 1.
clip_h = 1.
img_w = 100
img_h = 100
SPP = 16

img = [[0 for i in range(img_w)] for i in range(img_h)]

for i in tqdm(range(img_h)):
    for j in range(img_h):
        for k in range(SPP):
            orig, dir = generateInitialRay(
                cam_pos, cam_gaze, cam_top, clip_n, clip_r, clip_h, img_w, img_h, j, i)
            t, b1, b2, obj = getIntersection(orig, dir, tris)
            if obj != 0:
                img[i][j] += shade(orig + t * dir, -dir,
                                   obj, lights, tris) / SPP


plt.imshow(np.sqrt(img), cmap="gray", vmax=1)
plt.show()
