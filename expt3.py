# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 18:19:19 2020

@author: hacker 1
"""

import cv2 as cv
import numpy as np

utm = np.float32([[[ 396388.56, 5782566.  ],
                  [ 396477. ,  5782564.  ],
                  [ 396475.  , 5782467.5 ],
                  [ 396386.56 ,5782469.5 ]]])
px = np.float32([[[    0.   ,  0.],
                 [22992. ,    0.],
                 [22992., 25095.],
                 [    0., 25095.]]])

print('src='); print(utm)
print('dst='); print(px)

m1 = cv2.getPerspectiveTransform(utm, px)
print('getPerspectiveTransform='); print(m1)
m2, mask2 = cv2.findHomography(utm, px)
print('findHomography='); print(m2)

print("Checking (should equal to dst)....")

dst1 = cv2.perspectiveTransform(utm, m1);
print('perspectiveTransform(getPerspectiveTransform)='); print(dst1)
dst2 = cv2.perspectiveTransform(utm, m2);
print('perspectiveTransform(findHomography)='); print(dst2)