import sys
import pygame

t = 0


def render(a, img_w, img_h):
    global t
    t += 1
    for i in range(img_w):
        for j in range(img_h):
            a[i][j][0] = j % 255
            a[i][j][1] = t % 255


img_w = 200
img_h = 200
pygame.init()

screen = pygame.display.set_mode((img_w, img_h))
a = pygame.surfarray.pixels3d(screen)
while True:
    render(a, img_w, img_h)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.flip()
