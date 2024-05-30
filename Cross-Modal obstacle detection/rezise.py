import cv2

import os
path = "/data2/liaojunqi/yoloserver/yolov5-master/VOCdevkit/images/val/"#保存jpg图片
path1 = "/data2/liaojunqi/yoloserver/yolov5-master/VOCdevkit/images/val1/"#保存jpg图片
imagelist = os.listdir(path)
for i, txt in enumerate(imagelist):
    img = cv2.imread(path+txt)  # 读图
    height, width = img.shape[:2]  # 获取原图像的水平方向尺寸和垂直方向尺寸。
    res = cv2.resize(img, (640,640))  # dsize=（2*width,2*height）
    cv2.imwrite(path1+txt, res)
