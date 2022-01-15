import sys
import pygame
import numpy as np
from matplotlib import pyplot as plt
from numba import jit

t = 0
p = np.array([[10, 10, 0], [30, 20, 0], [20, 30, 0]])


@jit(nopython=True)
def render(a, p, img_w, img_h):
    img = np.array([[0.0 for j in range(img_w)] for i in range(img_h)])
    for i in range(img_h):
        for j in range(img_w):
            img[i][j] = 1.0 if np.sign(np.cross(p[1]-p[0], p[0]-np.array([j, i, 0.]))[2]) == np.sign(np.cross(p[2]-p[1], p[1]-np.array([j, i, 0.]))[2]) == np.sign(np.cross(p[0]-p[2], p[2]-np.array([j, i, 0.]))[2]) else 0.0
    for x in range(img_w):
        for y in range(img_h):
            a[x][y] = img[y][x] * 255


img_w = 200
img_h = 100
pygame.init()

screen = pygame.display.set_mode((img_w, img_h))
a = pygame.surfarray.pixels3d(screen)
while True:
    render(a, p, img_w, img_h)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            print(x,y)
            p[t][0] = x
            p[t][1] = y
            t += 1
            t %= 3
    pygame.display.flip()
