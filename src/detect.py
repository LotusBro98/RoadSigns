#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os

width = 640
height = 480
blurSize = 5
minContourArea = 100
contourApproxEpsInPeri = 0.1
minCircRatio = 0.9
croppedSize = 256
net_size = 32

hsv_ranges = [
	[(0, 110, 0), (10, 255, 255)],
	[(130, 110, 0), (180, 255, 255)],
	[(10, 160, 0), (40, 255, 255)],
	[(100, 120, 0), (120, 255, 255)],
]


CONTOUR_CIRCLE = '2'
CONTOUR_TRIANGLE = '3'
CONTOUR_QUADRANGLE = '4'
CONTOUR_OTHER = '0'

def preprocess(image):
	cv.GaussianBlur(image, (blurSize, blurSize), 1, image)
	hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

	colored = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.uint8)
	for range in hsv_ranges:
		color = cv.inRange(hsv, range[0], range[1])
		cv.bitwise_or(colored, color, colored)

	return colored


def processContour(contour):
	contour = cv.convexHull(contour)

	area = cv.contourArea(contour)
	peri = cv.arcLength(contour, True)

	if area < minContourArea:
		return CONTOUR_OTHER, None, None, contour

	M = cv.moments(contour)
	center = np.float32([M['m10'] / M['m00'], M['m01'] / M['m00']])
	radius = np.sqrt(area / np.pi)

	circRatio = area * 4 * np.pi / peri / peri

	if circRatio > minCircRatio:
		return CONTOUR_CIRCLE, center, radius, contour

	contour = cv.approxPolyDP(contour, peri * contourApproxEpsInPeri, True)
	contour = np.int32(contour)[:, 0]

	if len(contour) == 3:
		return CONTOUR_TRIANGLE, center, radius, contour
	elif len(contour) == 4:
		return CONTOUR_QUADRANGLE, center, radius, contour

	return CONTOUR_OTHER, center, radius, contour


def sliceCircle(image, contour, center, radius):
	M = cv.getAffineTransform(
		np.float32([
			center + (-radius, -radius),
			center + (radius, -radius),
			center + (radius, radius)
		]),
		np.float32([
			(0, 0),
			(croppedSize, 0),
			(croppedSize, croppedSize)
		])
	)

	return cv.warpAffine(image, M, (croppedSize, croppedSize))


def getMaxXY(contour):
	return np.argmax(np.sum(contour, axis=1))


def sliceTriangle(image, contour, center, radius):
	max = getMaxXY(contour)

	M = cv.getAffineTransform(
		np.float32([
			contour[(max + 0) % 3],
			contour[(max + 1) % 3],
			contour[(max + 2) % 3],
		]),
		np.float32([
			(croppedSize, croppedSize),
			(0, croppedSize),
			(croppedSize / 2, 0),
		])
	)

	return cv.warpAffine(image, M, (croppedSize, croppedSize))


def sliceQuadrangle(image, contour, center, radius):
	max = getMaxXY(contour)

	M = cv.getPerspectiveTransform(
		np.float32([
			contour[(max + 0) % 4],
			contour[(max + 1) % 4],
			contour[(max + 2) % 4],
			contour[(max + 3) % 4],
		]),
		np.float32([
			(croppedSize, croppedSize),
			(0, croppedSize),
			(0, 0),
			(croppedSize, 0)
		])
	)

	return cv.warpPerspective(image, M, (croppedSize, croppedSize))


sliceContour = {
	CONTOUR_CIRCLE: sliceCircle,
	CONTOUR_TRIANGLE: sliceTriangle,
	CONTOUR_QUADRANGLE: sliceQuadrangle
}

def dispCircle(image, contour, center, radius):
	cv.circle(image, tuple(np.int32(center)), int(radius), (0, 100, 255), 3)

def dispTriangle(image, contour, center, radius):
	cv.polylines(image, [contour], True, (0, 255, 200), 3)

def dispQuadrangle(image, contour, center, radius):
	cv.polylines(image, [contour], True, (255, 0, 0), 3)

def dispOther(image, contour, center, radius):
	cv.polylines(image, [contour], True, (0, 255, 0), 3)

dispContour = {
	CONTOUR_CIRCLE: dispCircle,
	CONTOUR_TRIANGLE: dispTriangle,
	CONTOUR_QUADRANGLE: dispQuadrangle,
	CONTOUR_OTHER: dispOther
}


model = cv.dnn.readNetFromTensorflow("../neural_network/model.pb")
def net_classify(imgs):
	blob = cv.dnn.blobFromImages(imgs, 1.0 / 255.0, (net_size, net_size))
	model.setInput(blob)
	predictions = model.forward()
	types = np.argmax(predictions, axis=1)
	return types

################################################

sliceI = 0
for filename in os.listdir("../images"):
	image = cv.imread("../images/"+filename)
	image = cv.resize(image, (width, int(image.shape[0] * width / image.shape[1])))
	imageDisp = image.copy()

	binary = preprocess(image)
	cv.imshow("Preprocessed", binary)

	contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		contour = np.int32(contour)[:,0]
		type, center, radius, contour = processContour(contour)
		if type == CONTOUR_OTHER:
			continue

		slice = sliceContour[type](image, contour, center, radius)

		sign_type = net_classify([slice])[0]
		if (sign_type == 0):
			continue
		print(sign_type)

		dispContour[type](imageDisp, contour, center, radius)

		cv.imwrite("../dataset/unsorted/" + str(sliceI) + ".jpg", slice)
		sliceI += 1

	print()

	cv.imshow("Image", imageDisp)
	cv.waitKey()