#!/usr/bin/python3

import cv2 as cv
import numpy as np
import os

PUCK_SIZE = (32, 32)

path_in = "./images2/original"
path_out = "./images2/output"
path_out_bad = "./images2/output_bad"

path_save_good = "./images2/dataset/puck"
path_save_bad = "./images2/dataset/no_puck"

def extract(img, center, size):
    x0 = center[0] - size[0] // 2 
    x1 = center[0] + size[0] // 2 
    y0 = center[1] - size[1] // 2 
    y1 = center[1] + size[1] // 2 

    if x0 < 0:
        x1 -= x0
        x0 -= x0

    if y0 < 0:
        y1 -= y0
        y0 -= y0

    return img[y0:y1, x0:x1]

def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return  nimg.transpose(1, 2, 0).astype(np.uint8)



pucks = []
no_pucks = []
cadr = 0
for filename in os.listdir(path_in):
    img_in = cv.imread(path_in + "/" + filename)
    img_in = grey_world(img_in)

    (H, W, _) = img_in.shape

    img_out = cv.imread(path_out + "/" + filename)
    img_out = cv.cvtColor(img_out, cv.COLOR_BGR2GRAY)
    img_out = cv.inRange(img_out, 128, 255)

    contours, _ = cv.findContours(img_out, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        center, _ = cv.minEnclosingCircle(contour)
        center = np.asarray(center, dtype=np.int)
        
        dc = (np.random.rand(2)- 0.5) * PUCK_SIZE / 3
        center = center + dc
        center = np.asarray(center, dtype=np.int)
        
        puck = extract(img_in, center, PUCK_SIZE)
        pucks.append(puck)
        #cv.imshow("Puck", puck)
        #cv.waitKey(200)

for filename in os.listdir(path_out_bad):
    img_in = cv.imread(path_in + "/" + filename)
    img_in = grey_world(img_in)

    (H, W, _) = img_in.shape

    img_out = cv.imread(path_out_bad + "/" + filename)
    img_out = cv.cvtColor(img_out, cv.COLOR_BGR2GRAY)
    img_out = cv.inRange(img_out, 128, 255)

    contours, _ = cv.findContours(img_out, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        center, _ = cv.minEnclosingCircle(contour)
        center = np.asarray(center, dtype=np.int)
        
        #dc = (np.random.rand(2)- 0.5) * PUCK_SIZE / 3
        #center = center + dc
        #center = np.asarray(center, dtype=np.int)
        
        no_puck = extract(img_in, center, PUCK_SIZE)
        no_pucks.append(no_puck)
        #cv.imshow("No Puck", no_puck)
        #cv.waitKey(1)

print("Pucks: " + str(len(pucks)))
print("No Pucks: " + str(len(no_pucks)))

i = 0
for puck in pucks:
    cv.imwrite(path_save_good + "/" + str(i) + ".png", puck)
    i += 1

i = 0
for no_puck in no_pucks:
    cv.imwrite(path_save_bad + "/" + str(i) + ".png", no_puck)
    i += 1

