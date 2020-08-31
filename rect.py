import cv2 as cv
import numpy as np
import math
import sys

def getAngle(p1, p2, p0):
    p1, p2, p0 = p1[0], p2[0], p0[0]
    dx1 = float(p1[0] - p0[0])
    dy1 = float(p1[1] - p0[1])
    dx2 = float(p2[0] - p0[0])
    dy2 = float(p2[1] - p0[1])
    return (dx1*dx2 + dy1*dy2) / math.sqrt((dx1**2 + dy1**2) * (dx2**2 + dy2**2) + 1e-10)

def distanceP2P(p1, p2):
    p1,p2 = p1[0], p2[0]
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# a = (p1.x, p1.y, p2.x, p2.y)
def intersect2Lines(a, b):
    x1, y1, x2, y2 = a[0], a[1], a[2], a[3]
    x3, y3, x4, y4 = b[0], b[1], b[2], b[3]
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d:
        tempX = float((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        tempY = float((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
        return (tempX, tempY)
    return (-1, -1)

def findLargestRect(rects):
    boundedRects = [cv.boundingRect(r) for r in rects]
    max_rect = max(boundedRects, key=lambda r: r[2] and r[3])
    max_index = boundedRects.index(max_rect)
    return (max_index, rects[max_index])

#找到高精度拟合时得到的顶点中 距离小于 低精度拟合得到的四个顶点 maxL的顶点，排除部分顶点的干扰
def betterApprox(max_approx, maxRect, maxL, newApprox):
    for p in max_approx:
        if not (distanceP2P(p, maxRect[0]) > maxL and
                    distanceP2P(p, maxRect[1]) > maxL and
                    distanceP2P(p, maxRect[2]) > maxL and
                    distanceP2P(p, maxRect[3]) > maxL):
            newApprox.append(p)

def computeCorners(corners, lines):
    for i in range(len(lines)):
        cornor = intersect2Lines(lines[i], lines[(i+1) % len(lines)])
        corners.append(cornor)

def drawLines(corners):
    for i in range(len(corners)):
        pt1 = (int(corners[i][0]), int(corners[i][1]))
        pt2 = (int(corners[(i+1) % len(corners)][0]), int(corners[(i+1) % len(corners)][1]))
        cv.line(img, pt1, pt2, (255, 0, 0), 5)

# 找到剩余顶点连线中，边长大于 2 * maxL的四条边作为四边形物体的四条边
def findLines(lines, newApprox, maxL):
    for i in range(len(newApprox)):
        p1 = newApprox[i]
        p2 = newApprox[(i + 1) % len(newApprox)]
        if (distanceP2P(p1,p2) > maxL * 2):
            lines.append((p1[0][0], p1[0][1], p2[0][0], p2[0][1]))

kernel = np.ones((5,5),np.uint8)
kernel_0 = np.zeros_like((3,3),np.uint8)
threshold = 30

img = cv.imread('3.jpg')
img_h, img_w, img_cd = (img.shape)

# img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # img_blur = cv.blur(img_grey, (3,3))
# img_blur = cv.GaussianBlur(img_grey, (3,3), 0)
# img_erode = cv.erode(img_blur, np.ones((5,5),np.uint8), iterations=3)
# edges = cv.Canny(img_erode, threshold, threshold*7, apertureSize=3)
# edges = cv.dilate(edges, np.ones((5,5),np.uint8), iterations=5, borderType=1, borderValue=1)

img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img_blur = cv.blur(img_grey, (3,3))
img_blur = cv.GaussianBlur(img_grey, (3,3), 0)
img_erode = cv.erode(img_blur, np.ones((3,3),np.uint8), iterations=3)
edges = cv.Canny(img_erode, threshold, threshold * 3, apertureSize=3)
edges = cv.dilate(edges, np.ones((3,3),np.uint8), iterations=5, borderType=1, borderValue=1)

contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
rects, hulls = [], []

for c in contours:
    hull = cv.convexHull(c)
    approx = cv.approxPolyDP(hull, cv.arcLength(hull, True) * 0.02, True)
    if len(approx) == 4 and cv.contourArea(approx) > 30000 and cv.isContourConvex(approx):
        maxCos = 0
        for i in range(2,5):
            cos = abs(getAngle(approx[i%4], approx[i-2], approx[i-1]))
            maxCos = max(maxCos, cos)
        if maxCos < 0.3:
            rects.append(approx)
            hulls.append(hull)

if len(rects) == 0:
    raise RuntimeError('no findings')

maxRectIndex, maxRect = findLargestRect(rects)
max_hull = hulls[maxRectIndex]
max_approx = cv.approxPolyDP(max_hull, epsilon=3, closed=True)
maxL = cv.arcLength(max_approx, True) * 0.02
newApprox = []
betterApprox(max_approx, maxRect, maxL, newApprox)
lines = []
findLines(lines, newApprox, maxL)
corners = []
computeCorners(corners, lines)
drawLines(corners)


cv.drawContours(img, approx, -1, (0,255,0), 10)
cv.drawContours(img, maxRect, -1, (0,0,255), 10)

cv.imshow('original', img)
# cv.imshow('greyed', img_grey)
# cv.imshow('blurred', img_blur)
# cv.imshow('eroded', img_erode)
cv.imshow('edges', edges)

cv.waitKey()