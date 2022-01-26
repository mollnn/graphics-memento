# Path Trace (full version)
#   always sample brdf, bvh, microfacet, texture
# TODO: microfacet
# TODO: texture

from unittest import result
from matplotlib.lines import Line2D
from matplotlib.pyplot import stackplot
import numpy as np
import taichi as ti
from time import time as tm
import time
from numpy.linalg import norm
import random


ti.init(arch=ti.cuda, debug=False)


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


def readObject(filename, material_id, offset=[0, 0, 0], scale=1):
    fp = open(filename, 'r')
    fl = fp.readlines()
    verts = [[]]
    ans = []
    for s in fl:
        a = s.split()
        if len(a) > 0:
            if a[0] == 'v':
                verts.append(np.array([float(a[1]), float(a[2]), float(a[3])]))
            elif a[0] == 'f':
                b = a[1:]
                b = [i.split('/') for i in b]
                ans.append(
                    [verts[int(b[0][0])]*scale+offset, verts[int(b[1][0])]*scale+offset, verts[int(b[2][0])]*scale+offset, material_id])
    return ans


scene = []
scene += readObject('assets/cube.obj', 1, offset=[0, -20, 0], scale=10)
scene += readObject('assets/cube.obj', 3, offset=[-20, 0, 0], scale=10)
scene += readObject('assets/cube.obj', 1, offset=[0, 0, -20], scale=10)
scene += readObject('assets/cube.obj', 1, offset=[0, 20, 0], scale=10)
scene += readObject('assets/cube.obj', 0, offset=[0, 14.9, 0], scale=5)
scene += readObject('assets/cube.obj', 4, offset=[20, 0, 0], scale=10)
scene += readObject('assets/cube.obj', 2, offset=[0, 0, 20], scale=10)
scene += readObject('assets/bunny.obj', 2, offset=[0, 0, 0], scale=10)

scene_material_id = [i[3] for i in scene]
scene = [i[:3] for i in scene]
matattrs = [
    [[0, 0, 0], [4, 4, 4], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0.3, 0.3, 0.3], [0, 0, 0], [0, 0, 0]],
    [[2, 0, 0], [0.8, 0.8, 1.0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0.3, 0.0, 0.0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0.0, 0.3, 0.0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0.12, 0.09, 0.05], [0, 0, 0], [0, 0, 0]]
]
mesh_desc = np.array(scene)
mesh_material_desc = np.array(scene_material_id)
matattrs_np = np.array(matattrs, dtype=np.float32)

bvh_desc = buildBVH(mesh_desc)

N_TRIANGLES = len(mesh_desc)
IMG_HEIGHT = IMG_WIDTH = 512
CLIP_N = CLIP_R = CLIP_H = 0.1
N_MATERIALS = 100
N_BVH_NODES = bvh_desc[0]

mesh_vertices = ti.Vector.field(3, ti.f32, (N_TRIANGLES, 3))
mesh_material_id = ti.field(ti.i32, (N_TRIANGLES))
material_attributes = ti.Vector.field(3, ti.f32, (N_MATERIALS, 4))

img = ti.Vector.field(3, ti.f32, (IMG_HEIGHT, IMG_WIDTH))
cam_pos = ti.Vector.field(3, ti.f32, ())
cam_gaze = ti.Vector.field(3, ti.f32, ())
cam_top = ti.Vector.field(3, ti.f32, ())
clip_n = ti.field(ti.f32, ())
clip_r = ti.field(ti.f32, ())
clip_h = ti.field(ti.f32, ())
img_w = ti.field(ti.f32, ())
img_h = ti.field(ti.f32, ())
bvh_v = ti.Matrix.field(2, 3, ti.f32, (N_BVH_NODES))
bvh_d = ti.Vector.field(3, ti.i32, (N_BVH_NODES))

# Some global temp memory used for BVH-ray intersection
bvhim_stack = ti.field(ti.i32, (IMG_HEIGHT * IMG_WIDTH, 30))
bvhim_result = ti.field(ti.i32, (IMG_HEIGHT * IMG_WIDTH, 100))

bvh_int_cnt = ti.field(ti.f32,(4))

mesh_vertices.from_numpy(mesh_desc)
mesh_material_id.from_numpy(mesh_material_desc)
material_attributes.from_numpy(matattrs_np)
cam_pos .from_numpy(np.array([0., 0.1, 0.15]))
cam_gaze .from_numpy(np.array([0., 0., -1]))
cam_top .from_numpy(np.array([0., 1, 0]))
clip_n .from_numpy(np.array(CLIP_N))
clip_r .from_numpy(np.array(CLIP_R))
clip_h .from_numpy(np.array(CLIP_H))
img_w .from_numpy(np.array(IMG_WIDTH))
img_h .from_numpy(np.array(IMG_HEIGHT))
img.from_numpy(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)))
bvh_v.from_numpy(bvh_desc[1])
bvh_d.from_numpy(bvh_desc[2])


@ti.func
def checkIntersect(orig, dir, trid):
    t = 1.
    b1 = 0.
    b2 = 0.
    q = 1.
    if trid >= 0:
        o = orig
        d = dir
        e1 = mesh_vertices[trid, 1] - mesh_vertices[trid, 0]
        e2 = mesh_vertices[trid, 2] - mesh_vertices[trid, 0]
        s = o - mesh_vertices[trid, 0]
        s1 = d.cross(e2)
        s2 = s.cross(e1)
        t = s2.dot(e2)
        b1 = s1.dot(s)
        b2 = s2.dot(d)
        q = s1.dot(e1)
    return t/q, b1/q, b2/q


@ti.func
def checkIntersectAABB(orig, dir, aabbid):
    aabb_v = bvh_v[aabbid]
    aabb_d = bvh_d[aabbid]
    tx0 = (aabb_v[0, 0]-orig[0]) / (dir[0] + 1e-6)
    tx1 = (aabb_v[1, 0]-orig[0]) / (dir[0] + 1e-6)
    if tx0 > tx1:
        tx0, tx1 = tx1, tx0
    ty0 = (aabb_v[0, 1]-orig[1]) / (dir[1] + 1e-6)
    ty1 = (aabb_v[1, 1]-orig[1]) / (dir[1] + 1e-6)
    if ty0 > ty1:
        ty0, ty1 = ty1, ty0
    tz0 = (aabb_v[0, 2]-orig[2]) / (dir[2] + 1e-6)
    tz1 = (aabb_v[1, 2]-orig[2]) / (dir[2] + 1e-6)
    if tz0 > tz1:
        tz0, tz1 = tz1, tz0
    t0 = max(tx0, ty0, tz0)
    t1 = min(tx1, ty1, tz1)
    return t0 <= t1 and t1 >= 0


@ti.func
def getIntersection(orig, dir, _id):
    # ! _id for each pixel
    # Find nearest intersection of ray(orig,dir) and triangles
    # TODO: BVH Support
    ans_t, ans_b1, ans_b2, ans_obj_id = 2e9, 0., 0., -1

    # Stack element is bvh node id
    # Always focus on the top element
    # Always test intersection with top aabb
    #   If not hit, just pop
    #   If hit
    #      If leaf, just expand it (i.e. pop and push chl, chr)
    #      If nonleaf, add aabb_id to "candidates"
    # Test all triangles in "candidates"
    # * Stack size <= Depth + 1
    # * Candidates size is not limited :( We just assume a limitation

    
    stack_ptr = 0
    result_ptr = 0
    bvhim_stack[_id, stack_ptr] = 0
    stack_ptr += 1
    bvh_int_cnt[0] += 1.0
    bvh_int_tmp = 0.0
    bvh_int_obj_tmp = 0.0

    while stack_ptr > 0:
        cur_aabb = bvhim_stack[_id, stack_ptr-1]
        stack_ptr -= 1
        bvh_int_tmp += 1.0
        if cur_aabb != -1:
            if checkIntersectAABB(orig, dir, cur_aabb):
                if bvh_d[cur_aabb][0] == 0:
                    bvhim_stack[_id, stack_ptr] = bvh_d[cur_aabb][1]
                    stack_ptr += 1
                    bvhim_stack[_id, stack_ptr] = bvh_d[cur_aabb][2]
                    stack_ptr += 1
                else:
                    if bvh_d[cur_aabb][1] != -1:
                        bvhim_result[_id, result_ptr] = bvh_d[cur_aabb][1]
                        result_ptr += 1
                        bvh_int_obj_tmp += 1.0
                    if bvh_d[cur_aabb][2] != -1:
                        bvhim_result[_id, result_ptr] = bvh_d[cur_aabb][2]
                        result_ptr += 1
                        bvh_int_obj_tmp += 1.0
    bvh_int_cnt[1] += bvh_int_tmp
    bvh_int_cnt[2] = max(bvh_int_cnt[2], bvh_int_tmp)
    bvh_int_cnt[3] = max(bvh_int_cnt[3], bvh_int_obj_tmp)

    for i in range(result_ptr):
        t, b1, b2 = checkIntersect(orig, dir, bvhim_result[_id, i])
        if t > 0 and ans_t - t > 1e-3 and b1 > 0 and b2 > 0 and b1 + b2 < 1:
            ans_t, ans_b1, ans_b2, ans_obj_id = t, b1, b2, bvhim_result[_id, i]

    # for i in range(mesh_vertices.shape[0]):
    #     t, b1, b2 = checkIntersect(orig, dir, i)
    #     if t > 0 and ans_t - t > 1e-3 and b1 > 0 and b2 > 0 and b1 + b2 < 1:
    #         ans_t, ans_b1, ans_b2, ans_obj_id = t, b1, b2, i

    return ans_t, ans_b1, ans_b2, ans_obj_id


@ti.func
def generateInitialRay(img_x, img_y):
    cam_handle = cam_gaze[None].cross(cam_top[None]).normalized()
    canonical_x = (img_x + 0.5) / img_w[None] * 2 - 1
    canonical_y = (img_y + 0.5) / img_h[None] * 2 - 1
    cam_near_center = cam_pos[None] + cam_gaze[None] * clip_n[None]
    orig = cam_near_center + canonical_x * cam_handle * \
        clip_r[None] + canonical_y * cam_top[None] * clip_h[None]
    dir = orig - cam_pos[None]
    dir = dir.normalized()
    return orig, dir


@ti.func
def sample_brdf(normal):
    t = ti.Vector([0, 0, 1], dt=ti.f32)
    if normal.dot(t) > 0.9:
        t = ti.Vector([0, 1, 0], dt=ti.f32)
    ex = t.cross(normal).normalized()
    ey = ex.cross(normal).normalized()
    ez = normal
    r1 = ti.random()
    r2 = ti.random()
    r = ti.sqrt(r1)
    a = 2*3.14159*r2
    x = r*ti.cos(a)
    y = r*ti.sin(a)
    z = ti.sqrt(1-r1)
    w = x*ex + y*ey + z*ez
    return w.normalized()


@ti.kernel
def render():
    SPP = 8
    for x, y in img:
        tans = ti.Vector([0., 0., 0.], dt=ti.f32)
        for sp in range(SPP):
            orig, dir = generateInitialRay(x, y)
            N_BOUNCE = 6    # RR will cause warp divergence
            ans = ti.Vector([0., 0., 0.], dt=ti.f32)
            coef = ti.Vector([1., 1., 1.], dt=ti.f32)
            for _ in range(N_BOUNCE):
                t, bc1, bc2, triangle_id = getIntersection(
                    orig, dir, y*IMG_WIDTH+x)

                if triangle_id != -1:
                    hit_pos = orig + dir * t
                    p0 = mesh_vertices[triangle_id, 0]
                    p1 = mesh_vertices[triangle_id, 1]
                    p2 = mesh_vertices[triangle_id, 2]
                    material_id = mesh_material_id[triangle_id]
                    material_type_id = material_attributes[material_id, 0][0]
                    normal = (p1-p0).cross(p2-p0).normalized()
                    if normal.dot(-dir) > 0:

                        # Implement different materials here
                        if material_type_id == 0:
                            # Area light
                            ans += material_attributes[material_id, 1] * coef
                            break
                        elif material_type_id == 1:
                            # Pure lambert
                            brdf = material_attributes[material_id, 1]
                            wi = sample_brdf(normal)
                            coef *= brdf * 3.14159
                            orig = hit_pos + wi * 1e-4
                            dir = wi
                        elif material_type_id == 2:
                            # Pure specular
                            brdf = material_attributes[material_id, 1]
                            wi = 2*normal.dot(-dir)*normal+dir
                            wi = wi.normalized()
                            coef *= brdf
                            orig = hit_pos + wi * 1e-4
                            dir = wi
                    else:
                        break
                else:
                    break
            tans += ans
        img[x, y] = ti.pow(tans / SPP, 2.2) * 1.5


gui = ti.GUI(res=(IMG_WIDTH, IMG_HEIGHT))
frame_id = 500

ina_r = 3.0
ina_h = 1.0

while True:
    stt = tm()
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            break
    if gui.is_pressed(ti.GUI.LEFT, 'a'):
        ina_r -= 0.1
    if gui.is_pressed(ti.GUI.RIGHT, 'd'):
        ina_r += 0.1
    if gui.is_pressed(ti.GUI.UP, 'w'):
        ina_h += 0.1
    if gui.is_pressed(ti.GUI.DOWN, 's'):
        ina_h -= 0.1

    cam_pos .from_numpy(
        np.array([ina_r*ti.cos(frame_id * 0.02), ina_h, ina_r*ti.sin(frame_id * 0.02)]))
    cam_gaze[None] = -cam_pos[None]
    cam_gaze[None] = cam_gaze[None].normalized()

    render()
    ti.sync()
    gui.set_image(img)
    gui.show()
    if frame_id % 10 == 0:
        print("time usage:", tm()-stt, " able fps:", 1./(tm()-stt+1e-9))
        print("average BVH nodes: ", bvh_int_cnt[1] / bvh_int_cnt[0], " max", bvh_int_cnt[2], " maxobj", bvh_int_cnt[3])
    bvh_int_cnt[0] = 0
    bvh_int_cnt[1] = 0
    bvh_int_cnt[2] = 0
    bvh_int_cnt[3] = 0
    frame_id += 10
