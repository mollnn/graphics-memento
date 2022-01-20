import numpy as np

def readobj(filename):
    fp = open(filename, 'r')
    fl = fp.readlines()
    verts = [[]]
    ans = []
    for s in fl:
        a = s.split()
        if len(a)>0:
            if a[0]=='v':
                verts.append([float(a[1]), float(a[2]), float(a[3])])
            elif a[0]=='f':
                b = a[1:]
                b = [i.split('/') for i in b]
                ans.append([verts[int(b[0][0])], verts[int(b[1][0])], verts[int(b[2][0])]])
    return ans
