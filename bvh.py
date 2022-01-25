import enum
import numpy as np
from pygame import key
import random

import taichi as ti

def vec3(x, y, z):
    return np.array([x, y, z], dtype=np.float32)

def rvec3():
    return vec3(random.random(), random.random(), random.random())

def rvec2():
    return vec3(random.random(), random.random(), 0)


def getAABB(tid, triangle):
    return {
        "p0": vec3(
            min(triangle["p0"][0], triangle["p1"][0], triangle["p2"][0]),
            min(triangle["p0"][1], triangle["p1"][1], triangle["p2"][1]),
            min(triangle["p0"][2], triangle["p1"][2], triangle["p2"][2])
        ),
        "p1": vec3(
            max(triangle["p0"][0], triangle["p1"][0], triangle["p2"][0]),
            max(triangle["p0"][1], triangle["p1"][1], triangle["p2"][1]),
            max(triangle["p0"][2], triangle["p1"][2], triangle["p2"][2])
        ),
        "leaf": True,
        "chl": -1,
        "chr": -1,
        "triangles": [tid]
    }


def mergeAABB_nonleaf(aabb1, aabb2):
    return {
        "p0": vec3(
            min(aabb1["p0"][0], aabb2["p0"][0]),
            min(aabb1["p0"][1], aabb2["p0"][1]),
            min(aabb1["p0"][2], aabb2["p0"][2])
        ),
        "p1": vec3(
            max(aabb1["p1"][0], aabb2["p1"][0]),
            max(aabb1["p1"][1], aabb2["p1"][1]),
            max(aabb1["p1"][2], aabb2["p1"][2])
        ),
        "leaf": False,
        "chl": -1,
        "chr": -1,
        "triangles": []
    }


def mergeAABB(aabb1, aabb2):
    return {
        "p0": vec3(
            min(aabb1["p0"][0], aabb2["p0"][0]),
            min(aabb1["p0"][1], aabb2["p0"][1]),
            min(aabb1["p0"][2], aabb2["p0"][2])
        ),
        "p1": vec3(
            max(aabb1["p1"][0], aabb2["p1"][0]),
            max(aabb1["p1"][1], aabb2["p1"][1]),
            max(aabb1["p1"][2], aabb2["p1"][2])
        ),
        "leaf": True,
        "chl": -1,
        "chr": -1,
        "triangles": aabb1["triangles"] + aabb2["triangles"]
    }


def mergeAABBs(aabb_list):
    ans = aabb_list[0]
    for i in aabb_list[1:]:
        ans = mergeAABB(ans, i)
    return ans


def getAABBs(tidlist, triangles):
    return mergeAABBs([getAABB(tid, triangles[tid]) for tid in tidlist])


def divide(objs, axis, triangles):
    keys = []
    for tid in objs:
        tmp_aabb = getAABB(tid, triangles[tid])
        keys.append([tmp_aabb["p0"][axis], tid])
    keys.sort(key=lambda i: i[0])
    mid = len(keys) // 2
    lefts = keys[:mid]
    rights = keys[mid:]
    return [i[1] for i in lefts], [i[1] for i in rights]


def solve(bvh, p, objs, triangles):
    if len(objs)>3:
        tl, tr = divide(objs, random.randint(0,1), triangles)    # MUST MODITY 1 TO 2 WHEN USED IN 3D
        pl = len(bvh)
        bvh.append({})
        pr = len(bvh)
        bvh.append({})
        solve(bvh, pl, tl, triangles)
        solve(bvh, pr, tr, triangles)
        bvh[p] = mergeAABB_nonleaf(bvh[pl], bvh[pr])
        bvh[p]["chl"] = pl
        bvh[p]["chr"] = pr
    else:
        bvh[p] = getAABBs(objs, triangles)


triangles = [
]

for i in range(10):
    o = rvec2()
    triangles.append({
        "p0": rvec2() * 0.1 + o * 0.9,
        "p1": rvec2() * 0.1 + o * 0.9,
        "p2": rvec2() * 0.1 + o * 0.9
    })


bvh = [{}]
solve(bvh, 0, [i for i in range(len(triangles))], triangles)

for i in bvh:
    print(i)


tri_node_id = {}
node_id_color = {}
for idx, i in enumerate(bvh):
    for j in i["triangles"]:
        tri_node_id[j] = idx

for i in range(len(bvh)):
    node_id_color[i]=random.randint(0, 16777215)

R=512
gui = ti.GUI(res=R)

for idx, i in enumerate(triangles):
    gui.triangle(i["p0"],i["p1"],i["p2"], node_id_color[tri_node_id[idx]])

for i in bvh:
    if i["leaf"]==False:
        gui.rect(i["p0"][:2], i["p1"][:2], radius=random.randint(1,3), color=random.randint(0, 16777215))

gui.show()

while True:
    gui.get_events()