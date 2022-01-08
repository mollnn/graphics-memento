import numpy as np
from matplotlib import pyplot as plt
import random
import math

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


def getInterpZ(p_screenspace, p_viewspace, j, i):
    q = np.array([j, i, 0])
    pp = np.array([np.array([x[0], x[1], 0]) for x in p_screenspace])
    w = [np.linalg.norm(np.cross(pp[(i+1) % 3]-q, pp[(i+2) % 3]-q), ord=1)
         for i in range(3)]
    w /= np.sum(w)+1e-9
    ans = 1/(np.sum([w[i]/(p_viewspace[i][2]+1e-9) for i in range(3)])-1e-9)
    ans = min(ans, max(p[2] for p in p_viewspace))
    ans = max(ans, min(p[2] for p in p_viewspace))
    return -ans


def getInterp(p_screenspace, p_viewspace, vals, j, i):
    q = np.array([j, i, 0])
    pp = np.array([np.array([x[0], x[1], 0]) for x in p_screenspace])
    w = [np.linalg.norm(np.cross(pp[(i+1) % 3]-q, pp[(i+2) % 3]-q), ord=1)
         for i in range(3)]
    w /= np.sum(w)+1e-9
    tans = 1/(np.sum([w[i]/(p_viewspace[i][2]+1e-9) for i in range(3)])-1e-9)
    tans = min(tans, max(p[2] for p in p_viewspace))
    tans = max(tans, min(p[2] for p in p_viewspace))
    ans = np.sum([w[i]*vals[i]/(p_viewspace[i][2]+1e-9)
                 for i in range(3)])-1e-9
    return ans * tans


plt.ion()

angle = 0

while True:
    angle += 0.1

    vertices_worldspace = []
    vertices_worldspace.append(
        [np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 0, 0])])
    vertices_worldspace.append(
        [np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([-1, 0, 0])])
    vertices_worldspace.append(
        [np.array([0, 0, -1]), np.array([0, 1, 0]), np.array([1, 0, 0])])
    vertices_worldspace.append(
        [np.array([0, 0, -1]), np.array([0, 1, 0]), np.array([-1, 0, 0])])

    clip_n, clip_f, clip_l, clip_r, clip_t, clip_b = -3, -10, -1, 1, 1, -1

    img_w = 100
    img_h = 100

    img = [[0 for j in range(img_w)] for i in range(img_h)]
    zbuf = [[abs(clip_f)+1 for j in range(img_w)] for i in range(img_h)]

    camera_pos = np.array([2*math.sin(angle), 4, 2*math.cos(angle)])
    camera_gaze = -camera_pos
    camera_gaze /= np.linalg.norm(camera_gaze, ord=2)
    # camera_up = np.array([0, 1, 0])
    camera_up = [
        np.cross(np.cross(camera_gaze, np.array([0., 1, 0])), camera_gaze)]
    camera_up /= np.linalg.norm(camera_up, ord=2)
    camera_handle = np.cross(camera_gaze, camera_up)
    camera_handle /= np.linalg.norm(camera_handle, ord=2)

    camera_pos = np.append(camera_pos, 1)
    camera_gaze = np.append(camera_gaze, 0)
    camera_up = np.append(camera_up, 0)
    camera_handle = np.append(camera_handle, 0)

    transform_view_translate = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-camera_pos[0], -camera_pos[1], -camera_pos[2], 1]]).T
    transform_view_rotate = np.concatenate(
        ([camera_handle], [camera_up], [-camera_gaze], [[0, 0, 0, 1]]))
    transform_view = transform_view_rotate @ transform_view_translate

    transform_proj_persp = np.array([[clip_n, 0, 0, 0], [0, clip_n, 0, 0], [
                                    0, 0, clip_n+clip_f, -clip_n*clip_f], [0, 0, 1, 0]])
    transform_proj_ortho = np.array([[2/(clip_r-clip_l), 0, 0, 0], [0, 2/(clip_t-clip_b), 0, 0], [0, 0, 2/(clip_n-clip_f), 0],
                                    [(clip_l+clip_r)/(clip_r-clip_l), (clip_t+clip_b)/(clip_t-clip_b), (clip_n+clip_f)/(clip_n-clip_f), 1]]).T
    transform_proj = transform_proj_ortho @ transform_proj_persp

    transform_viewport = np.array(
        [[img_w / 2, 0, 0, img_w/2], [0, img_h/2, 0, img_h/2], [0, 0, 1, 0], [0, 0, 0, 1]])
    transform = transform_viewport @ transform_proj @ transform_view

    light_dir = np.array([-1., 1., 1.])
    light_dir /= np.linalg.norm(light_dir, ord=2)

    light_dir_vs = (transform_view_rotate @ np.append(light_dir, 0))[:3]

    diff = 0.5
    spec = 3
    spow = 8
    ambi = 0.2

    for triangle_ws in vertices_worldspace:
        triangle_ss = [transform @ (np.array(j.tolist() + [1]))
                       for j in triangle_ws]
        triangle_vs = [transform_view @ (np.array(j.tolist() + [1]))
                       for j in triangle_ws]
        normal = np.cross(
            triangle_vs[1][:3]-triangle_vs[0][:3], triangle_vs[2][:3]-triangle_vs[0][:3])
        normal /= np.linalg.norm(normal, ord=2)

        for i in range(img_h):
            for j in range(img_w):
                x = j
                y = img_h - i - 1
                if check([k[:2]/k[3] for k in triangle_ss], x, y):
                    zval = getInterpZ([k[:2]/k[3]
                                       for k in triangle_ss], triangle_vs, x, y)
                    if zval < zbuf[i][j]:
                        zbuf[i][j] = zval
                        vsx = getInterp([k[:2]/k[3] for k in triangle_ss],
                                        triangle_vs, [i[0] for i in triangle_vs], x, y)
                        vsy = getInterp([k[:2]/k[3] for k in triangle_ss],
                                        triangle_vs, [i[0] for i in triangle_vs], x, y)
                        vsz = getInterp([k[:2]/k[3] for k in triangle_ss],
                                        triangle_vs, [i[0] for i in triangle_vs], x, y)
                        vsp = np.array([vsx, vsy, vsz])
                        watch_dir = -vsp
                        watch_dir /= np.linalg.norm(watch_dir, ord=2)
                        half = light_dir_vs + watch_dir
                        half /= np.linalg.norm(half, ord=2)
                        color = ambi + diff * \
                            max(0, np.dot(normal, light_dir_vs)) + \
                            spec * (max(0, np.dot(normal, half)) 1** spow)
                        img[i][j] = color

    plt.subplot(121)
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.subplot(122)
    plt.imshow(zbuf, cmap='gray')
    plt.show()
    plt.pause(0.2)
    plt.clf()  # 清除图像
