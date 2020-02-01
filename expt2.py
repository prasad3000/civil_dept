# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:18:44 2020

@author: hacker 1
"""
import numpy as np
import cv2
import imutils


def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    
    rect = order_points(pts)
    #rect = pts
    (tl, tr, bl, br) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    
    maxwidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    
    maxheight = max(int(heightA), int(heightB))
    
    dst = np.array([
            [0, 0],
            [maxwidth-1, 0],
            [maxwidth-1, maxheight-1],
            [0, maxheight-1]], dtype='float32')
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxwidth, maxheight))
    
    return warped, M, dst

pts = np.array([[109,380],[330,200],[420,200],[630,380]], dtype=np.int32)

screenCnt = np.array([[[104, 328]],[[529, 330]],[[362, 112]],[[294, 113]]])
screenCnt = screenCnt.reshape(4, 2)

image = cv2.imread('masked_image.png')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

warped, M, dst = four_point_transform(orig, pts)
rect = order_points(pts)


##################################################
#pts = np.array([[182,300],[274,200],[473,200],[560,300]], dtype=np.int32)
#dts = np.array([[0,100],[0,0],[380,0],[380,100]], dtype=np.int32)

pts = np.float32([[[ 180.0, 300.0],
                  [ 274.0 ,  200.  ],
                  [ 473.  , 200. ],
                  [ 560. ,300. ]]])
dts = np.float32([[[    0.   ,  100.],
                 [0. ,    0.],
                 [380., 0.],
                 [    380., 100.]]])

(tl, tr, bl, br) = pts

widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

maxwidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

maxheight = max(int(heightA), int(heightB))
    
M = cv2.getPerspectiveTransform(rect, dst)

dst1 = cv2.perspectiveTransform(pts, M)

warped = cv2.warpPerspective(orig, M, (maxwidth, maxheight))
plt.imshow(warped)