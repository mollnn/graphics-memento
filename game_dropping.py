# Vision controlled game

import cv2
import numpy as np
import taichi as ti
import random
from copy import deepcopy

ti.init(arch=ti.cpu)

cv2.namedWindow('frame')
capture = cv2.VideoCapture(0)
fc = 0
WND_SIZE = 64
THRES = 15
THRES__ = 16
x1, y1, x2, y2 = 0, 0, 500, 0


def mouseEventCallback(event, x, y, flags, param):
    global x1, y1, x2, y2, fc
    if flags == cv2.EVENT_FLAG_LBUTTON:
        x1, y1 = x, y
    if flags == cv2.EVENT_FLAG_MBUTTON:
        fc = 0
    if flags == cv2.EVENT_FLAG_RBUTTON:
        x2, y2 = x, y


def pseudo_normal_random():
    return sum(random.random() for i in range(5)) / 5

cv2.setMouseCallback('frame', mouseEventCallback)
b1w = [False, False]
b2w = [False, False]

bw = [0, 0, 0, 0, 0]

dfa_state = 0

pos = 0.5
posv = 0.0

wndv = 0.0
wnda = 0.0

gui = ti.GUI(res=768)

objs = []

ans = 0

emit_pos = 0


def gravObject():
    global objs
    global wndv
    new_objs = []
    for i in range(len(objs)):
        objs[i][1] -= objs[i][3]
        objs[i][0] -= objs[i][4]
        objs[i][0] += wndv * (0.3 if objs[i][2] < 0 else 0.5 * 0.1 + objs[i][2])
        if objs[i][1] > 0.00:
            new_objs.append(objs[i])
    objs = new_objs


while(True):
    ret, frame = capture.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 1)
    if fc < 2:
        last_img = deepcopy(img)
        avg10 = cv2.mean(img[y1:y1+WND_SIZE, x1:x1+WND_SIZE])
        avg20 = cv2.mean(img[y2:y2+WND_SIZE, x2:x2+WND_SIZE])
    avg1 = cv2.mean(img[y1:y1+WND_SIZE, x1:x1+WND_SIZE])
    avg2 = cv2.mean(img[y2:y2+WND_SIZE, x2:x2+WND_SIZE])
    avg = cv2.mean(img - last_img)

    f1, f2 = np.sum(np.abs(np.array(avg1)-np.array(avg10))
                    ), np.sum(np.abs(np.array(avg2)-np.array(avg20)))
    f = np.sum(np.abs(np.array(last_img, dtype=np.float) -
               np.array(img, dtype=np.float))) / len(img) / len(img[0])

    last_img = deepcopy(img)

    b1 = f1 > THRES
    b2 = f2 > THRES
    b = f < THRES__

    b1w = b1w[1:] + [b1]
    b2w = b2w[1:] + [b2]
    bw = bw[1:] + [b]

    if sum(bw) == len(bw):
        avg10 = cv2.mean(img[y1:y1+WND_SIZE, x1:x1+WND_SIZE])
        avg20 = cv2.mean(img[y2:y2+WND_SIZE, x2:x2+WND_SIZE])

    action = 0
    if sum(b1w) == len(b1w):
        action = 1
    if sum(b2w) == len(b2w):
        action = 2
    if sum(b1w) == 0 and sum(b2w) == 0:
        action = 3

    if dfa_state == 0:
        if action == 1:
            posv += 0.01
            dfa_state = 1
        elif action == 2:
            posv -= 0.01
            dfa_state = 2
    elif dfa_state == 1:
        if action == 3:
            dfa_state = 0
    elif dfa_state == 2:
        if action == 3:
            dfa_state = 0

    pos += posv
    posv *= 0.9

    if random.random() < 0.03:
        emit_pos = random.random() * 0.9

    for _ in range(3):
            for i in range(8):
                objs.append([random.random() * 3 - 1, 1.0, random.randint(1,5), random.random()
                            * 0.02, random.random() * 0.002 - 0.001])
            objs.append([(random.random() * 0.1 + emit_pos)  * 3 - 1, 1.0, 7,
                        random.random() * 0.01 + 0.01, random.random() * 0.002 - 0.001])
            if random.random() < 0.1:
                objs.append([random.random()   * 3 - 1, 1.0, -1, random.random()
                            * 0.03, random.random() * 0.002 - 0.001])
            if random.random() < 0.002:
                o = random.random()
                for i in range(30):
                    objs.append([(random.random() * 0.9 + o * 0.9) * 3 - 1, 1.0, -1, random.random()
                                * 0.03, random.random() * 0.001 - 0.0005])


    wndv += wnda
    wnda *= 0.95
    wnda += pseudo_normal_random() * 0.0002 - 0.0001
    wndv *= 0.99
    gravObject()

    gui.clear(color=0)
    gui.circle((pos, 0.01), radius=15)

    for i in objs:
        if i[2] >= 7:
            gui.circle((i[0], i[1]), radius=2, color=0x5599FF)
        elif i[2] >= 0:
            gui.circle((i[0], i[1]), radius=2 * i[2], color=0xFFFFFF-i[2]*0x333333)
        else:
            gui.circle((i[0], i[1]), radius=int(3 + pseudo_normal_random() * 6),
                       color=0xFF0000 + int(pseudo_normal_random() * 8) * 0x001100)

    for i in objs:
        if i[1] < 0.03 and abs(i[0]-pos) < 0.02:
            if i[2] >= 0:
                ans += i[2]
            else:
                print("score", ans)
                ans = 0
                print("game over! restart")
                objs.clear()
    gui.text("%d" % ans, [0.01, 0.99], 100, 0xFFCC33)

    gui.show()

    cv2.drawMarker(img, (x1, y1), 255*256*256)
    cv2.drawMarker(img, (x1+WND_SIZE, y1), 255*256*256)
    cv2.drawMarker(img, (x1, y1+WND_SIZE), 255*256*256)
    cv2.drawMarker(img, (x1+WND_SIZE, y1+WND_SIZE), 255*256*256)
    cv2.drawMarker(img, (x2, y2), 255*256*256)
    cv2.drawMarker(img, (x2+WND_SIZE, y2), 255*256*256)
    cv2.drawMarker(img, (x2, y2+WND_SIZE), 255*256*256)
    cv2.drawMarker(img, (x2+WND_SIZE, y2+WND_SIZE), 255*256*256)
    cv2.imshow('frame', img)
    # 如果输入q，则退出
    if cv2.waitKey(1) == ord('q'):
        break
    fc += 1
