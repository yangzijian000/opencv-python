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
import time
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
    surf = xf2d.SURF_create(hessianThreshold)#创建SURF对象
    kp1,des1 = surf.detectAndCompute(srcimg,None)#用SURF方法找出源图像特征点
    kp2,des2 = surf.detectAndCompute(dstimg,None)#用SURF方法找出目标图像特征点
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)#创建flann对象，用flann方法进行特征点匹配
    matches = flann.knnMatch(des1, des2, k=2)#找到两个匹配点，一个是距离最优点，另一个是次优点
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:#最优点距离小于0.7倍的次优点视为好的匹配点
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    #画出两张图的匹配点图像
    Matchimg = cv2.drawMatchesKnn(srcimg, kp1, dstimg, kp2, matches, None, **draw_params)
    plt.figure(1)
    plt.imshow(Matchimg)
def PerspectivetransFormationMatrix(srcimg,dstimg):
    #计算透视变换矩阵
    #利用优质的匹配点进行计算，首先要将特征点化坐标找出，化为numpy数组
    src_pts = np.float32([kp1[m.queryIdx].pt for i, (m, n) in enumerate(matches)]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for i, (m, n) in enumerate(matches)]).reshape(-1, 1, 2)
    #进行透视变换矩阵计算，采用RANSAC方法
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    pts = np.float32([[0, 0], [0, src_rows - 1], [src_cols - 1, src_rows - 1], [src_cols - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    dstimg_copy = dstimg.copy()
    #画出将要拼接的部分
    pictures = cv2.polylines(dstimg_copy, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    #进行变换，得到变换后图像
    transformedimg = cv2.warpPerspective(srcimg, M, (src_cols + dst_cols, max(src_rows, dst_rows)))
    plt.figure(2)
    plt.imshow(pictures)
    plt.figure(3)
    plt.imshow(transformedimg)
    return pictures,transformedimg
#进行图像拼接，这个方法是直接进行拼接
def StichImg(dstimg,transformedimg):
    global src_rows, src_cols
    transformedimg[0:src_rows,0:src_cols] = dstimg
    img = cv2.cvtColor(transformedimg,cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img,)
    plt.figure(4)
    plt.imshow(transformedimg)


#用金字塔方法进行拼接，希望能平滑拼接缝隙
def StichImgbypyramid(dstimg,transformedimg,layer):
    gpA = Gaussian_pyramid(dstimg,layer)
    gpB = Gaussian_pyramid(transformedimg,layer)
    lpA = Laplacian_pyramid(gpA,layer)
    lpB = Laplacian_pyramid(gpB,layer)
    LS_add(lpA, lpB,layer)
#首先进行高斯金字塔计算
def Gaussian_pyramid(img,layer):
    G = img.copy()
    gp = [G]
    for i in range(layer):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp
#再通过高斯金字塔计算拉普拉斯金字塔
def Laplacian_pyramid(gp,layer):
    lp = [gp[layer]]
    for i in range(layer, 1, -1):
        GE = cv2.pyrUp(gp[i])
        h, w, c = gp[i - 1].shape
        L = cv2.subtract(gp[i - 1], GE[0:h, 0:w])
        lp.append(L)
    return lp
#将每层金字塔融合再求和
def LS_add(lpA,lpB,layer):
    LS = []
    for la, lb in zip(lpA, lpB):
        h, w, c = la.shape
        lb[0:h, 0:w] = la
        LS.append(lb)
    ls_ = LS[0]
    for i in range(1, layer):
        ls_ = cv2.pyrUp(ls_)
        h, w, c = LS[i].shape
        ls_ = cv2.add(ls_[0:h, 0:w], LS[i])
    plt.figure(5)
    plt.imshow(ls_)
def main():
    starttime = time.time()
    global src_rows, src_cols
    global dst_rows, dst_cols
    srcimg,dstimg = readimgs('test6.jpg','test5.jpg')
    src_rows,src_cols,c1 = srcimg.shape
    dst_rows,dst_cols,c2 = dstimg.shape
    FeaturesAndMatch(srcimg,dstimg,3000)
    pictures, transformedimg = PerspectivetransFormationMatrix(srcimg,dstimg)
    StichImg(dstimg,transformedimg)
    StichImgbypyramid(dstimg,transformedimg,5)
    stoptime = time.time()
    runtime = stoptime - starttime
    print(runtime)
    plt.show()
if __name__ == '__main__':
    main()
