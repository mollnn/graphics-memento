# Path Trace (full version)
#   sample mesh light, bvh, microfacet, texture

from importlib_metadata import itertools
import numpy as np
import taichi as ti
from time import time as tm
import time
from numpy.linalg import norm
import random
from matplotlib import pyplot as plt


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
        tmp_aabb = getAABBs(objs, triangles)
        delta = tmp_aabb["p1"] - tmp_aabb["p0"]
        axis = np.argmax(delta)
        # axis = random.randint(0, 2) # pure random axis selection
        tl, tr = divideTriangles(objs, axis, triangles)
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
                if len(b)==3:
                    ans.append(
                        [verts[int(b[0][0])]*scale+offset, verts[int(b[1][0])]*scale+offset, verts[int(b[2][0])]*scale+offset,
                        verts_t[int(b[0][1])] if len(verts_t) > 1 else [0.0, 0.0],
                        verts_t[int(b[1][1])] if len(verts_t) > 1 else [0.0, 0.0],
                        verts_t[int(b[2][1])] if len(verts_t) > 1 else [0.0, 0.0],
                        material_id])
                elif len(b)==4:
                    ans.append(
                        [verts[int(b[0][0])]*scale+offset, verts[int(b[1][0])]*scale+offset, verts[int(b[2][0])]*scale+offset,
                        verts_t[int(b[0][1])] if len(verts_t) > 1 else [0.0, 0.0],
                        verts_t[int(b[1][1])] if len(verts_t) > 1 else [0.0, 0.0],
                        verts_t[int(b[2][1])] if len(verts_t) > 1 else [0.0, 0.0],
                        material_id])
                    ans.append(
                        [verts[int(b[2][0])]*scale+offset, verts[int(b[3][0])]*scale+offset, verts[int(b[0][0])]*scale+offset,
                        verts_t[int(b[2][1])] if len(verts_t) > 1 else [0.0, 0.0],
                        verts_t[int(b[3][1])] if len(verts_t) > 1 else [0.0, 0.0],
                        verts_t[int(b[0][1])] if len(verts_t) > 1 else [0.0, 0.0],
                        material_id])
    return ans


##############################################################
##############################################################
##############################################################


TEX_MEM_SIZE = 1048576 * 4
tex_mem_used = [0]
tex_mem = np.array([[0, 0, 0] for i in range(TEX_MEM_SIZE)], dtype=np.float32)


def texAlloc(tex_mem, im_filename, used_):
    used = used_[0]
    im = plt.imread(im_filename)
    im = np.array(im)
    if len(im.shape) == 2:
        im = np.tile(np.expand_dims(im, 2), (1, 1, 3))
    sz = im.shape[0]*im.shape[1]
    tex_mem[used: used+sz] = im.reshape((-1, 3)) / 255.0
    used += sz
    used_[0] = used
    return used-sz, im.shape[0], im.shape[1]


textures_filename = [
    "assets/ground.jfif"
]

N_TEXTURE = len(textures_filename)

textures_desc = [texAlloc(tex_mem, i, tex_mem_used) for i in textures_filename]

tex_mem_ti = ti.Vector.field(3, ti.f32, (TEX_MEM_SIZE))
tex_desc_ti = ti.Vector.field(3, ti.i32, (N_TEXTURE))
tex_mem_ti.from_numpy(tex_mem)
tex_desc_ti.from_numpy(np.array(textures_desc))


@ti.func
def getTexPixel(tex_id, x, y):
    # * x,y must be integer
    tex_addr = tex_desc_ti[tex_id][0]
    h = tex_desc_ti[tex_id][1]
    w = tex_desc_ti[tex_id][2]
    x = min(x, w-1)
    x = max(x, 0)
    y = min(y, h-1)
    y = max(y, 0)
    return tex_mem_ti[tex_addr+w*y+x]


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
        h = tex_desc_ti[tex_id][1]
        w = tex_desc_ti[tex_id][2]
        ans = getTexPixelBI(tex_id, u*w, (1-v)*h)
    return ans


scene = []
scene += readObject('assets/sponza/sponza.obj', 1, offset=[0, 0, 0], scale=1)
scene += readObject('assets/test.obj', 0, offset=[0, 50, 0], scale=5)

scene_material_id = [i[6] for i in scene]
scene_uv = [i[3:6] for i in scene]
scene = [i[:3] for i in scene]
matattrs = [
    [[0, 0, 0], [100, 100, 100], [0, 0, 0], [0, 0, 0]],
    [[1, 0.9, 3.9], [0.8, 0.8, 0.8], [0, 0, 0], [0, 0, 0]],
    [[2, 0, 0], [0.8, 0.8, 1.0], [0, 0, 0], [0, 0, 0]],
    [[1, 0.9, 3.9], [0.8, 0.2, 0.3], [0, 0, 0], [0, 0, 0]],
    [[1, 0.9, 3.9], [0.3, 0.2, 0.8], [0, 0, 0], [0, 0, 0]],
    [[1, 0.3, 0.3], [0.6, 0.5, 0.2], [0, 0, 0], [0, 0, 0]],
    [[1, 0.9, 0.2], [0.6 * 2, 0.5 * 2, 0.2 * 2], [0, 0, 0], [0, 0, 0]],
]

matattri = [
    [-1],
    [0],
    [-1],
    [-1],
    [-1],
    [-1],
    [-1],
]


mesh_desc = np.array(scene)
mesh_uv_desc = np.array(scene_uv)
print(mesh_uv_desc.shape)
mesh_material_desc = np.array(scene_material_id)
matattrs_np = np.array(matattrs, dtype=np.float32)
matattri_np = np.array(matattri, dtype=np.float32)
light_tids = [(i, norm(np.cross(mesh_desc[i][1]-mesh_desc[i][0], mesh_desc[i][2]-mesh_desc[i][0])))
              for i in range(len(scene)) if matattrs[scene_material_id[i]][0][0] in [0]]  # Emitting material ids here
sum_light_area = sum(i[1] for i in light_tids)
light_tids = [(i[0], i[1]/sum_light_area) for i in light_tids]
light_sampler_tid = np.array([i[0] for i in light_tids])
light_sampler_cdf = np.array(
    list(itertools.accumulate([i[1] for i in light_tids])))
if len(light_sampler_cdf) > 0:
    light_sampler_cdf[-1] = 1.0     # float accumulate correction
N_LIGHT_TRIANGLES = len(light_sampler_cdf)
bvh_desc = buildBVH(mesh_desc)

N_TRIANGLES = len(mesh_desc)
IMG_HEIGHT = IMG_WIDTH = 256
WIDTH_DIV = 4
CLIP_N = CLIP_R = CLIP_H = 0.1
N_MATERIALS = len(matattri)
N_BVH_NODES = bvh_desc[0]

mesh_vertices = ti.Vector.field(3, ti.f32, (N_TRIANGLES, 3))
mesh_uvcoords = ti.Vector.field(2, ti.f32, (N_TRIANGLES, 3))
mesh_material_id = ti.field(ti.i32, (N_TRIANGLES))
material_attributes = ti.Vector.field(3, ti.f32, (N_MATERIALS, 4))
material_exattr = ti.field(ti.i32, (N_MATERIALS, 1))

img = ti.Vector.field(3, ti.f32, (IMG_HEIGHT, IMG_WIDTH))       # original
img_acc = ti.Vector.field(3, ti.f32, (IMG_HEIGHT, IMG_WIDTH))  # accumulated
img_disp = ti.Vector.field(3, ti.f32, (IMG_HEIGHT, IMG_WIDTH))  # post-processed
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
lsp_p = ti.field(ti.f32, (N_LIGHT_TRIANGLES))
lsp_d = ti.field(ti.i32, (N_LIGHT_TRIANGLES))

# Some global temp memory used for BVH-ray intersection
bvhim_stack = ti.field(ti.i32, (IMG_HEIGHT * WIDTH_DIV, 30))
bvhim_result = ti.field(ti.i32, (IMG_HEIGHT * WIDTH_DIV, 200))

bvh_int_cnt = ti.field(ti.f32, (4))

mesh_vertices.from_numpy(mesh_desc)
mesh_uvcoords.from_numpy(mesh_uv_desc)
mesh_material_id.from_numpy(mesh_material_desc)
material_attributes.from_numpy(matattrs_np)
material_exattr.from_numpy(matattri_np)
cam_pos .from_numpy(np.array([0.0, 0.0, 5.0]))
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
lsp_p.from_numpy(light_sampler_cdf)
lsp_d.from_numpy(light_sampler_tid)


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

    #################################################################################

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

    #################################################################################

    # for i in range(mesh_vertices.shape[0]):
    #     t, b1, b2 = checkIntersect(orig, dir, i)
    #     if t > 0 and ans_t - t > 1e-3 and b1 > 0 and b2 > 0 and b1 + b2 < 1:
    #         ans_t, ans_b1, ans_b2, ans_obj_id = t, b1, b2, i

    #################################################################################

    return ans_t, ans_b1, ans_b2, ans_obj_id


@ti.func
def checkVisibility(p, q, _id):
    d = (q-p).normalized()
    p1 = p + d * 1e-4
    thres = (q-p).norm() - 1e-3
    t, b1, b2, obj = getIntersection(p1, d, _id)
    return t > thres


@ti.func
def generateInitialRay(img_x, img_y):
    cam_handle = cam_gaze[None].cross(cam_top[None]).normalized()
    canonical_x = (img_x + 0.5) / img_w[None] * 2 - 1
    canonical_y = (img_y + 0.5) / img_h[None] * 2 - 1
    canonical_y *= -1
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


@ti.func
def sample_light():
    # sample a light
    rnd = ti.random()
    idx = 0
    for i in range(N_LIGHT_TRIANGLES):
        if rnd <= lsp_p[i]:
            idx = i
        else:
            break
    tid = lsp_d[idx]
    # sample a point
    bc0 = ti.random()
    bc1 = ti.random()
    bc2 = ti.random()
    bcs = bc0 + bc1 + bc2
    bc0, bc1, bc2 = bc0 / bcs, bc1/bcs, bc2/bcs
    p0 = mesh_vertices[tid, 0]
    p1 = mesh_vertices[tid, 1]
    p2 = mesh_vertices[tid, 2]
    return tid, bc0*p0 + bc1*p1+bc2*p2


@ti.func
def ggx_d(a, i, o, n):
    h = (i+o).normalized()
    nh = ti.acos(n.dot(h))
    x = a / ti.pow(ti.cos(nh), 2) / (a*a + ti.pow(ti.tan(nh), 2))
    # return 1.0
    return 1.0 / 3.14159 * x * x


@ti.func
def smith_g1(a, v, i, o, n):
    h = (i+o).normalized()
    ans = 0.0
    if v.dot(h) * v.dot(n) > 0:
        x = 1.0 / a / ti.tan(ti.acos(n.dot(v)))
        if a < 1.6:
            ans = (3.535*a + 2.181*a*a) / (1 + 2.276*a + 2.577*a*a)
        else:
            ans = 1.0
    # return 1.0
    return ans


@ti.func
def smith_g(a, i, o, n):
    return smith_g1(a, i, i, o, n) * smith_g1(a, o, i, o, n)


@ti.func
def microfacet_brdf(ad, ag, i, o, n):
    ni = n.dot(i) + 1e-6
    no = n.dot(o) + 1e-6
    return ggx_d(ad, i, o, n) * smith_g(ag, i, o, n) * 1.0 / 4 / ni / no


@ti.kernel
def render():
    SPP = 1
    WIDTH_SEG = IMG_WIDTH // WIDTH_DIV
    for thread_id in range(IMG_HEIGHT * WIDTH_DIV):
        y = thread_id // WIDTH_DIV
        xoff = thread_id % WIDTH_DIV * WIDTH_SEG
        for x0 in range(WIDTH_SEG):
            x = xoff + x0
            tans = ti.Vector([0., 0., 0.], dt=ti.f32)
            for sp in range(SPP):
                x1 = ti.random()
                y1 = ti.random()
                orig, dir = generateInitialRay(x - 0.5 + x1, y - 0.5 + y1)
                N_BOUNCE = 8    # RR will cause warp divergence
                ans = ti.Vector([0., 0., 0.], dt=ti.f32)
                coef = ti.Vector([1., 1., 1.], dt=ti.f32)

                # * Set False after diffuse (sampling light)
                light_source_visible = True

                for _ in range(N_BOUNCE):
                    t, bc1, bc2, triangle_id = getIntersection(
                        orig, dir, thread_id)

                    if triangle_id != -1:
                        hit_pos = orig + dir * t
                        p0 = mesh_vertices[triangle_id, 0]
                        p1 = mesh_vertices[triangle_id, 1]
                        p2 = mesh_vertices[triangle_id, 2]
                        uv0 = mesh_uvcoords[triangle_id, 0]
                        uv1 = mesh_uvcoords[triangle_id, 1]
                        uv2 = mesh_uvcoords[triangle_id, 2]
                        uv = (1-bc1-bc2)*uv0 + bc1*uv1 + bc2*uv2
                        u = uv[0]
                        v = uv[1]
                        material_id = mesh_material_id[triangle_id]
                        material_type_id = material_attributes[material_id, 0][0]
                        normal = (p1-p0).cross(p2-p0).normalized()
                        tex_id = material_exattr[material_id, 0]
                        tex_color = getTexColorBI(tex_id, u, v)
                        if normal.dot(-dir) > 0:
                            # Implement different materials here
                            if material_type_id == 0:
                                # Area light
                                if light_source_visible:  # Add direct contribution of light source
                                    ans += material_attributes[material_id,
                                                               1] * coef * normal.dot(-dir)
                                break
                            elif material_type_id == 1:
                                # Microfacet (GGX_D, SMITH_G, no Fresnel)

                                light_tid, light_pos = sample_light()
                                light_p0 = mesh_vertices[light_tid, 0]
                                light_p1 = mesh_vertices[light_tid, 1]
                                light_p2 = mesh_vertices[light_tid, 2]
                                light_normal = (
                                    light_p1-light_p0).cross(light_p2-light_p0).normalized()

                                brdf_1 = material_attributes[material_id, 1] / 3.14159 * microfacet_brdf(
                                    material_attributes[material_id, 0][1],
                                    material_attributes[material_id, 0][2],
                                    (light_pos-hit_pos).normalized(),
                                    -dir,
                                    normal
                                ) * tex_color

                                if checkVisibility(hit_pos, light_pos, thread_id) and light_normal.dot(hit_pos-light_pos) > 0:
                                    ans += coef * brdf_1 * normal.dot((light_pos-hit_pos).normalized()) * material_attributes[mesh_material_id[light_tid], 1] * \
                                        light_normal.dot(
                                            (hit_pos-light_pos).normalized()) / (hit_pos-light_pos).dot(hit_pos-light_pos) * sum_light_area

                                light_source_visible = False

                                wi = sample_brdf(normal)

                                brdf = material_attributes[material_id, 1] * microfacet_brdf(
                                    material_attributes[material_id, 0][1],
                                    material_attributes[material_id, 0][2],
                                    wi,
                                    -dir,
                                    normal
                                ) * tex_color

                                coef *= brdf * 3.14159
                                orig = hit_pos + wi * 1e-4
                                dir = wi
                            # elif material_type_id == 2:
                            #     # Pure specular
                            #     brdf = material_attributes[material_id, 1]
                            #     wi = 2*normal.dot(-dir)*normal+dir
                            #     wi = wi.normalized()
                            #     coef *= brdf
                            #     orig = hit_pos + wi * 1e-4
                            #     dir = wi
                            #     light_source_visible = True
                        else:
                            break
                    else:
                        break
                tans += ans
            img[x, y] = ti.pow(tans / SPP, 1.0)



def normalized(x):
    return x / norm(x)



@ti.kernel
def post_process(ndiv: ti.i32):
    for y in range(IMG_HEIGHT):
        for x in range(IMG_WIDTH):
            if ndiv==1:
                img_acc[x,y] = img[x,y]
            else:
                img_acc[x,y] += img[x,y]
            img_disp[x,y]=img_acc[x,y] / ndiv

gui = ti.GUI(res=(IMG_WIDTH, IMG_HEIGHT))
frame_id = 0

camera_pos = np.array([0.0, 0.0, 5.0])
camera_gaze = np.array([0.0, 0.0, -1.0])
camera_handle = np.cross(camera_gaze, np.array([0., 1., 0]))
camera_grav = np.array([0., 1., 0])
camera_top = np.cross(camera_handle, camera_gaze)
camera_top /= norm(camera_top)

last_camera_gaze = camera_gaze

acc_frame_count = 0

lx, ly = 0.0, 0.0
phi, theta = 0.0, 0.0

while True:
    stt = tm()

    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            break
    if gui.is_pressed(ti.GUI.LEFT, 'a'):
        camera_pos -= 0.10 * camera_handle
        acc_frame_count = 0
    if gui.is_pressed(ti.GUI.RIGHT, 'd'):
        camera_pos += 0.10 * camera_handle
        acc_frame_count = 0
    if gui.is_pressed(ti.GUI.UP, 'w'):
        camera_pos += 0.10 * camera_gaze
        acc_frame_count = 0
    if gui.is_pressed(ti.GUI.DOWN, 's'):
        camera_pos -= 0.10 * camera_gaze
        acc_frame_count = 0
    if gui.is_pressed('p'):
        camera_pos += 0.10 * camera_top
        acc_frame_count = 0
    if gui.is_pressed('l'):
        camera_pos -= 0.10 * camera_top
        acc_frame_count = 0

    gui.get_event(ti.GUI.MOTION)
    cx, cy = gui.get_cursor_pos()
    dx, dy = cx-lx, cy-ly
    lx, ly = cx, cy

    if gui.is_pressed(ti.GUI.SHIFT) and gui.is_pressed(ti.GUI.LMB):
        camera_pos -= dy * camera_gaze * 3
        acc_frame_count = 0
    elif gui.is_pressed(ti.GUI.LMB):
        camera_pos -= -dy * camera_top * 3 + dx * camera_hand * 3
        acc_frame_count = 0
    if gui.is_pressed(ti.GUI.RMB):
        phi += dx * 6.28 * 0.2
        theta += dy * 3.14 * 0.2
        acc_frame_count = 0

    camera_gaze = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]]) @ \
        np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                 [0, np.sin(theta), np.cos(theta)]]) @ np.array([0., 0., -1.])

    camera_hand = normalized(np.cross(camera_grav, camera_gaze))
    camera_top = normalized(np.cross(camera_hand, camera_gaze))

    if norm(np.array(camera_gaze)-np.array(last_camera_gaze)) > 0.0001:
        acc_frame_count = 0

    last_camera_gaze = camera_gaze

    bvh_int_cnt[0] = 1e-6
    bvh_int_cnt[1] = 0
    bvh_int_cnt[2] = 0
    bvh_int_cnt[3] = 0

    cam_pos.from_numpy(camera_pos)
    cam_gaze.from_numpy(camera_gaze)
    cam_top.from_numpy(camera_top)

    acc_frame_count += 1
    ti.sync()
    render()
    ti.sync()
    post_process(acc_frame_count)
    ti.sync()
    gui.set_image(img_disp)
    gui.show()
    # if frame_id % 10 == 0:
    #     print("time usage:", tm()-stt, " able fps:", 1. /
    #           (tm()-stt+1e-9), "   #triangles", len(mesh_desc))
    # print("average BVH nodes visited: ", bvh_int_cnt[1] / bvh_int_cnt[0],
    #       " max_visited", bvh_int_cnt[2], " max_candidates", bvh_int_cnt[3])

    frame_id += 1
