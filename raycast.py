import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm


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
    return t/q, b1/q, b2/q


def getIntersection(orig, dir, tris):
    ans_t, ans_b1, ans_b2, ans_obj = 2e9, 0, 0, 0
    for tri in tris:
        t, b1, b2 = checkIntersect(orig, dir, tri["p0"], tri["p1"], tri["p2"])
        if t > 0 and t < ans_t and b1 > 0 and b2 > 0 and b1 + b2 < 1:
            ans_t, ans_b1, ans_b2, ans_obj = t, b1, b2, tri
    return ans_t, ans_b1, ans_b2, ans_obj


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


tris = [
    {"p0": np.array([0, 0, 0]),  "p1": np.array([1, 0, 0]), "p2": np.array([0, 1, 0]), "c": 1},
    {"p0": np.array([-3, 0, -1]),  "p1": np.array([1, 0, -1]), "p2": np.array([0, 1, -1]), "c": 2},
    {"p0": np.array([0, 0, -2]),  "p1": np.array([1, 0, -2]), "p2": np.array([0, 1, -2]), "c": 3}
]

cam_pos = np.array([0, 0, 2])
cam_gaze = np.array([0, 0, -1])
cam_top = np.array([0, 1, 0])
clip_n = 1
clip_r = 1
clip_h = 1
img_w = 100
img_h = 100

img = [[0 for i in range(img_w)] for i in range(img_h)]

for i in range(img_h):
    for j in range(img_h):
        orig, dir = generateInitialRay(
            cam_pos, cam_gaze, cam_top, clip_n, clip_r, clip_h, img_w, img_h, j, i)
        t, b1, b2, obj = getIntersection(orig, dir, tris)
        if obj != 0:
            img[i][j] = obj["c"]

plt.imshow(img, cmap="gray")
plt.show()
