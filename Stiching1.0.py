#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2017/5/25 16:46
# @Author : YANGz1J
# @Site : 
# @File : Stiching1.0.py
# @Software: PyCharm
import cv2
import numpy as np
import cv2.xfeatures2d as xf2d
from matplotlib import pyplot as plt
global src_rows,src_cols
global dst_rows,dst_cols
global kp1,kp2
global des1,des2
global matches,matchesMask
def readimgs(*imgs):
    read_imgs =[]
    for img in imgs:
        read_img = cv2.imread(img,cv2.IMREAD_COLOR)#读取图片
        b,g,r = cv2.split(read_img)#将图片通道分解
        read_img = cv2.merge([r,g,b])#以rgb重构图片
        read_imgs.append(read_img)
    return read_imgs[0],read_imgs[1]
def FeaturesAndMatch(srcimg,dstimg,hessianThreshold):
    global kp1, kp2
    global des1, des2
    global matches, matchesMask
    surf = xf2d.SURF_create(hessianThreshold)
    kp1,des1 = surf.detectAndCompute(srcimg,None)
    kp2,des2 = surf.detectAndCompute(dstimg,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    Matchimg = cv2.drawMatchesKnn(srcimg, kp1, dstimg, kp2, matches, None, **draw_params)
    plt.figure(1)
    plt.imshow(Matchimg)
def PerspectivetransFormationMatrix(srcimg,dstimg):
    src_pts = np.float32([kp1[m.queryIdx].pt for i, (m, n) in enumerate(matches)]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for i, (m, n) in enumerate(matches)]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    pts = np.float32([[0, 0], [0, src_rows - 1], [src_cols - 1, src_rows - 1], [src_cols - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    dstimg_copy = dstimg.copy()
    pictures = cv2.polylines(dstimg_copy, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    transformedimg = cv2.warpPerspective(srcimg, M, (src_cols + dst_cols, max(src_rows, dst_rows)))
    plt.figure(2)
    plt.imshow(pictures)
    plt.figure(3)
    plt.imshow(transformedimg)
    return pictures,transformedimg
def StichImg(dstimg,transformedimg):
    global src_rows, src_cols
    transformedimg[0:src_rows,0:src_cols] = dstimg
    plt.figure(4)
    plt.imshow(transformedimg)
def StichImgbypyramid(dstimg,transformedimg):
    gpA = Gaussian_pyramid(dstimg)
    gpB = Gaussian_pyramid(transformedimg)
    lpA = Laplacian_pyramid(gpA)
    lpB = Laplacian_pyramid(gpB)
    LS_add(lpA, lpB)
def Gaussian_pyramid(img):
    G = img.copy()
    gp = [G]
    for i in range(5):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp
def Laplacian_pyramid(gp):
    lp = [gp[5]]
    for i in range(5, 1, -1):
        GE = cv2.pyrUp(gp[i])
        h, w, c = gp[i - 1].shape
        L = cv2.subtract(gp[i - 1], GE[0:h, 0:w])
        lp.append(L)
    return lp
def LS_add(lpA,lpB):
    LS = []
    for la, lb in zip(lpA, lpB):
        h, w, c = la.shape
        lb[0:h, 0:w] = la
        LS.append(lb)
    ls_ = LS[0]
    for i in range(1, 5):
        ls_ = cv2.pyrUp(ls_)
        h, w, c = LS[i].shape
        ls_ = cv2.add(ls_[0:h, 0:w], LS[i])
    plt.figure(5)
    plt.imshow(ls_)
def main():
    global src_rows, src_cols
    global dst_rows, dst_cols
    srcimg,dstimg = readimgs('test2.jpg','test1.jpg')
    src_rows,src_cols,c1 = srcimg.shape
    dst_rows,dst_cols,c2 = dstimg.shape
    FeaturesAndMatch(srcimg,dstimg,2000)
    pictures, transformedimg = PerspectivetransFormationMatrix(srcimg,dstimg)
    StichImg(dstimg,transformedimg)
    StichImgbypyramid(dstimg,transformedimg)
    plt.show()
if __name__ == '__main__':
    main()