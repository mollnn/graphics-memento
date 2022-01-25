import enum
import numpy as np
import random
import taichi as ti

###################################################
# BVH Builder
###################################################


def bvh_vec3(x, y, z):
    return np.array([x, y, z], dtype=np.float32)


def bvh_rvec3():
    return bvh_vec3(random.random(), random.random(), random.random())


def getAABB(tid, triangle):
    return {
        "p0": bvh_vec3(
            min(triangle["p0"][0], triangle["p1"][0], triangle["p2"][0]),
            min(triangle["p0"][1], triangle["p1"][1], triangle["p2"][1]),
            min(triangle["p0"][2], triangle["p1"][2], triangle["p2"][2])
        ),
        "p1": bvh_vec3(
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
        "p0": bvh_vec3(
            min(aabb1["p0"][0], aabb2["p0"][0]),
            min(aabb1["p0"][1], aabb2["p0"][1]),
            min(aabb1["p0"][2], aabb2["p0"][2])
        ),
        "p1": bvh_vec3(
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
        "p0": bvh_vec3(
            min(aabb1["p0"][0], aabb2["p0"][0]),
            min(aabb1["p0"][1], aabb2["p0"][1]),
            min(aabb1["p0"][2], aabb2["p0"][2])
        ),
        "p1": bvh_vec3(
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


def divideTriangles(objs, axis, triangles):
    keys = []
    for tid in objs:
        tmp_aabb = getAABB(tid, triangles[tid])
        keys.append([tmp_aabb["p0"][axis], tid])
    keys.sort(key=lambda i: i[0])
    mid = len(keys) // 2
    lefts = keys[:mid]
    rights = keys[mid:]
    return [i[1] for i in lefts], [i[1] for i in rights]


def buildBVH_recursive(bvh, p, objs, triangles):
    if len(objs) > 2:
        tl, tr = divideTriangles(objs, random.randint(0, 2), triangles)
        pl = len(bvh)
        bvh.append({})
        pr = len(bvh)
        bvh.append({})
        buildBVH_recursive(bvh, pl, tl, triangles)
        buildBVH_recursive(bvh, pr, tr, triangles)
        bvh[p] = mergeAABB_nonleaf(bvh[pl], bvh[pr])
        bvh[p]["chl"] = pl
        bvh[p]["chr"] = pr
    else:
        bvh[p] = getAABBs(objs, triangles)


def buildBVH(meshs):
    triangles = [
        {"p0": i[0], "p1":i[1], "p2":i[2]} for i in meshs
    ]

    bvh = [{}]
    buildBVH_recursive(bvh, 0, [i for i in range(len(triangles))], triangles)

    res_v = []
    res_d = []
    for i in bvh:
        res_v.append([i["p0"], i["p1"]])

    for i in bvh:
        if i["leaf"]:
            res_d.append([1]+i["triangles"]+[-1]*(3-len(i["triangles"])-1))
        else:
            res_d.append([0, i["chl"], i["chr"]])

    # for i in range(len(bvh)):
    #     print(res_v[i],"\t",res_d[i])

    return len(res_v), np.array(res_v, dtype=np.float32), np.array(res_d, dtype=np.int32)

###################################################

triangles = [
]

for i in range(10):
    o = bvh_rvec3()
    triangles.append([
         bvh_rvec3() * 0.1 + o * 0.9,
         bvh_rvec3() * 0.1 + o * 0.9,
         bvh_rvec3() * 0.1 + o * 0.9
    ])
print(buildBVH(triangles))


