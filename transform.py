import numpy as np
from matplotlib import pyplot as plt

vertices = [[0, 0, 0, 1], [1, 0, 0, 1], [-1, 0, 0, 1], [0, 1, 0, 1], [0, 0, -1, 1]]

cam_pos = np.array([-0.2, 0.2, 2, 1])
gaze_dir = np.array([0.2, -0.2, -1, 0])
gaze_dir /= np.linalg.norm(gaze_dir, ord=1)
up_dir = np.array([0,1,0,0])
left_dir = np.array(np.cross(gaze_dir[:3], up_dir[:3]).tolist()+[0])

mat_view = np.array([left_dir, up_dir, -gaze_dir, [0, 0, 0, 1]]).T @ \
    [[1, 0, 0, -cam_pos[0]], [0, 1, 0, -cam_pos[1]],
        [0, 0, 1, -cam_pos[2]], [0, 0, 0, 1]]

proj_near = -1
proj_far = -5
proj_left = -1
proj_right = 1
proj_top = 1
proj_bottom = -1

mat_p2o = np.array([[proj_near, 0, 0, 0], [0, proj_near, 0, 0], [
                   0, 0, proj_near+proj_far, -proj_near*proj_far], [0, 0, 1, 0]])
mat_ortho = np.array([
    [2/(proj_right-proj_left), 0, 0, -
        (proj_right+proj_left)/(proj_right-proj_left)],
    [0, 2/(proj_top - proj_bottom), 0, -
     (proj_top+proj_bottom)/(proj_top - proj_bottom)],
    [0, 0, 2/(proj_near-proj_far), -(proj_near+proj_far)/(proj_near-proj_far)],
    [0, 0, 0, 1]
])


mat_proj = mat_ortho @ mat_p2o
mat_viewing = mat_proj @ mat_view

transformed_vertices = [mat_viewing @ i for i in vertices]

tv2d = []
for i in transformed_vertices:
    print([j/i[3] for j in i[:3]])
    tv2d.append([j/i[3] for j in i[:2]])

plt.scatter([i[0] for i in tv2d], [i[1] for i in tv2d])
plt.show()