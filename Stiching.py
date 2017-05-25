#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2017/5/23 11:34
# @Author : YANGz1J
# @Site : 
# @File : test.py
# @Software: PyCharm
import cv2
import numpy as np
import cv2.xfeatures2d as cv
from matplotlib import pyplot as plt
def main():
    img2 = cv2.imread('test1.jpg',cv2.IMREAD_COLOR)
    img1 = cv2.imread('test2.jpg',cv2.IMREAD_COLOR)
    rows1,cols1,channels1=img1.shape
    rows2,cols2,channels2=img2.shape
    # print gakkiimg
    surf = cv.SURF_create(600)
    kp1,des1 = surf.detectAndCompute(img1,None)
    kp2,des2 = surf.detectAndCompute(img2,None)
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2,k=2)
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good.append(m)
    # print len(kp1)
    # print len(kp2)
    img3 = cv2.drawKeypoints(img1,kp1,None,(255,0,0),4)
    img4 = cv2.drawKeypoints(img2,kp2,None,(255,0,0),4)
    # img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:10],flags=2)
    # cv2.namedWindow('Gakki',cv2.WINDOW_NORMAL)
    # cv2.imshow('图1特征点',img3)
    # cv2.imshow('图2特征点',img4)
    # cv2.imshow('3',img5)
    bf = cv2.BFMatcher()
    matchs = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matchs:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    src_pts = 0
    dst_pts = 0
    print(len(good))
    if len(good)>10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
    pts = np.float32([[0, 0], [0, rows1 - 1], [cols1 - 1, rows1 - 1], [cols1 - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img8 = img2
    # img7 = cv2.polylines(img8, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    print(M)
    print(mask)
    img6 = cv2.warpPerspective(img1,M,(cols1+cols2,max(rows1,rows2)))
    img6[0:rows1,0:cols1] = img2

    # cv2.perspectiveTransform()
    img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, [good], None, flags=2)
    plt.figure(1)
    plt.imshow(img6)
    # plt.figure(2)
    # plt.imshow(img7)
    plt.figure(3)
    plt.imshow(img1)
    plt.figure(4)
    plt.imshow(img2)
    plt.show()
if __name__ == '__main__':
    main()
