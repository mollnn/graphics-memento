from re import A
import numpy as np
import taichi as ti
from time import time

for ar in [ti.cpu, ti.opengl, ti.cuda]:
    ti.init(arch=ar)

    N = 10**7

    a = ti.field(ti.f32, (N))
    a.from_numpy(np.linspace(0, 1, N))

    @ti.kernel
    def solve() -> ti.f32:
        ans = 0.
        for i in range(N):
            ans += a[i]
        for i in range(N):
            ans += a[i]
        return ans

    t_s = time()
    print(solve())
    print("\033[31m", int((time()-t_s)*1000), "ms", "\033[0m")
