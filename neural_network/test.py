#!/usr/bin/python3

import numpy as np
import os

import tensorflow as tf

import cv2 as cv

import time

PATTERN_WIDTH = 32
PATTERN_HEIGHT = 32

colors = [
        (128, 128, 128),
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (128, 255, 255)
        ]

check_points_static = [
        (284, 358),
        (311, 356),
        (333, 358),
        (322, 415),
        (354, 415),
        (388, 414),
        (421, 413),
        (456, 414),
        (494, 414),
        (873, 420),
        (907, 421),
        (940, 424),
        (970, 424),
        (998, 426),
        (1028, 429),
        (1011, 368),
        (1034, 370),
        (1057, 372),
        (256, 566),
        (200, 657),
        (141, 775),
        (1097, 571),
        (1156, 658),
        (1214, 772)
        ]

chaoz_vertices = np.float32([
        [879, 529],
        [1014, 596],
        [967, 677],
        [817, 609],
        ])


def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return  nimg.transpose(1, 2, 0).astype(np.uint8)


def Sobel_1_color(gray):
    ddepth = cv.CV_16S
    mix = 0.5

    grad_x = cv.Sobel(gray, ddepth = ddepth, dx = 1, dy = 0)
    grad_y = cv.Sobel(gray, ddepth = ddepth, dx = 0, dy = 1)
    grad_x_abs = cv.convertScaleAbs(grad_x)
    grad_y_abs = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(grad_x_abs, mix, grad_y_abs, mix, 0)

    return grad

def Sobel_3_colors(img):
    b,g,r = cv.split(img)
    mix = 0.7

    grad_r = Sobel_1_color(r)
    grad_g = Sobel_1_color(g)
    grad_b = Sobel_1_color(b)

    grad = cv.addWeighted(grad_r, mix, grad_g, mix, 0)
    grad = cv.addWeighted(grad, 2 * mix, grad_b, mix, 0)

    return grad

def cut_atom(img, pt):
    pt = np.uint16(pt)
    return img[pt[1] - PATTERN_HEIGHT // 2: pt[1] + PATTERN_HEIGHT // 2, pt[0] - PATTERN_WIDTH // 2: pt[0] + PATTERN_WIDTH // 2]

def cut_atoms(img, points):
    imgs = []
    for pt in points:
        atom_img = cut_atom(img, pt)
        imgs.append(atom_img)
    return imgs

def net_check_atoms(model, imgs):
    blob = cv.dnn.blobFromImages(imgs, 1.0 / 255.0, (PATTERN_WIDTH, PATTERN_HEIGHT))
    model.setInput(blob)
    predictions = model.forward()
    types = np.argmax(predictions, axis=1)
    return types

def mark_atoms(img_display, points, types):
    for pt, tp in zip(points, types):
        cv.circle(all_img_display, tuple(pt), 3, colors[tp], 3)


def chaoz_find_circles(img_buf, chaoz_vertices):
    CHAOZ_SIZE = 256
    CHAOZ_ATOM_MIN_RADIUS = 25
    CHAOZ_ATOM_MAX_RADIUS = 30
    chaoz_out = np.float32([[0,0], [CHAOZ_SIZE, 0], [CHAOZ_SIZE, CHAOZ_SIZE], [0, CHAOZ_SIZE]])

    M = cv.getPerspectiveTransform(chaoz_vertices, chaoz_out)
    chaoz = cv.warpPerspective(img_buf, M, (CHAOZ_SIZE, CHAOZ_SIZE))

    chaoz = cv.blur(chaoz, (3, 3))
    chaoz_s = Sobel_3_colors(chaoz)

    circles = cv.HoughCircles(chaoz_s ,cv.HOUGH_GRADIENT,1,minDist=40,
                                param1=35,param2=20,minRadius=CHAOZ_ATOM_MIN_RADIUS,maxRadius=CHAOZ_ATOM_MAX_RADIUS)

    circles = circles[:,:,0:2]
    for i in circles[0]:
        i = tuple(np.uint16(i))
        R = (CHAOZ_ATOM_MIN_RADIUS + CHAOZ_ATOM_MAX_RADIUS) // 2
        cv.circle(chaoz,i,R,(128,0,255),2)
    
    cv.imshow("CHAOZ", chaoz)

    return cv.perspectiveTransform(circles, np.linalg.inv(M))[0]




################################################

model = cv.dnn.readNetFromTensorflow("./puck_model.pb")

all_img_buf = cv.imread("./images2/original/2019-03-02-150600.jpg")

static_atom_img_refs = cut_atoms(all_img_buf, check_points_static)

for filename in os.listdir("./images2/original"):

    all_img_read = cv.imread("./images2/original/" + filename)
    all_img_read = grey_world(all_img_read)

    all_img_display = all_img_read.copy()

    start_time = time.clock()
    #####
    
    cv.copyTo(all_img_read, None, all_img_buf)
 
    static_atom_types = net_check_atoms(model, static_atom_img_refs)

    chaoz_atom_points = chaoz_find_circles(all_img_buf, chaoz_vertices)
    chaoz_atom_imgs = cut_atoms(all_img_buf, chaoz_atom_points)
    chaoz_atom_types = net_check_atoms(model, chaoz_atom_imgs)
 
    #####
    end_time = time.clock()
    print("Processing time: " + str(end_time - start_time) + " seconds")

    mark_atoms(all_img_display, check_points_static, static_atom_types)
    mark_atoms(all_img_display, chaoz_atom_points, chaoz_atom_types)

    #cv.polylines(all_img_display, np.int32([chaoz_vertices]), True, (0,255,0), 1)

    cv.imshow("All Image", all_img_display)

    cv.waitKey()
