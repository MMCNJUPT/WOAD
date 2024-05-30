# -*- coding:utf8 -*-

# 导入cv2库
import cv2
src = cv2.imread("/data1/liaojunqi/yolov5-masterfusion/VOCdevkit/images/train/000000.png")
view_img=1
if view_img:
#    window_handle= cv2.namedWindow("results", cv2.WINDOW_AUTOSIZE)
    if cv2.getWindowProperty("results", 0) >= 0:
        cv2.imshow("results", src)
        cv2.waitKey(0)
    # import numpy as np
# # a=np.zeros((6,6),dtype=np.uint8)
# # a[0,3]=8
# # for i in a:
# #     print(i)
# a=abs(2-19)
# print(a)
# 打开摄像头
# cap = cv.VideoCapture(0)
#
# b=[[106, 187, 188, 195, 165, 96, 97, 125], [101, 28, 272, 328, 197, 83, 92, 104], [187, 73, 168, 107, 132, 78, 81, 108], [144, 71, 193, 183, 77, 78, 75, 179], [83, 70, 186, 173, 70, 74, 74, 81], [71, 65, 148, 68, 70, 68, 69, 68], [129, 63, 62, 62, 59, 59, 58, 56], [26, 52, 122, 161, 246, 276, 335, 371]]
#
# while (True):
#
#     # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像
#     # hx, frame = cap.read()
#     #
#     # # 如果hx为Flase表示开启摄像头失败，那么就输出"read vido error"并退出程序
#     # if hx is False:
#     #     # 打印报错
#     #     print('read video error')
#     #     # 退出程序
#     #     exit(0)
#     #
#     # # 显示摄像头图像，其中的video为窗口名称，frame为图像
#     # cv.imshow('video', frame)
#     frame=np.zeros((640,640,3),dtype=np.uint8)
#     for i3 in range(8):
#         for j3 in range(8):
#             if b[7-i3][7-j3]<50:
#                 frame[i3*80:(i3+1)*80,j3*80:(j3+1)*80,2]=255#红色
#                 cv.putText(frame,"{}".format("<50"),(int(j3*80),int(i3*80+40)),cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 2)
#             elif b[7-i3][7-j3]<100: 
#                 frame[i3*80:(i3+1)*80,j3*80:(j3+1)*80,1]=255#绿色
#                 cv.putText(frame,"{}".format("<100"),(int(j3*80),int(i3*80+40)),cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 2)
#             elif b[7-i3][7-j3]<150:
#                 frame[i3*80:(i3+1)*80,j3*80:(j3+1)*80,1:3]=255#黄色
#                 cv.putText(frame,"{}".format("<150"),(int(j3*80),int(i3*80+40)),cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 2)
#             elif b[7-i3][7-j3]<200:
#                 frame[i3*80:(i3+1)*80,j3*80:(j3+1)*80,0]=55#棕色
#                 frame[i3*80:(i3+1)*80,j3*80:(j3+1)*80,1]=135
#                 frame[i3*80:(i3+1)*80,j3*80:(j3+1)*80,2]=195
#                 cv.putText(frame,"{}".format("<200"),(int(j3*80),int(i3*80+40)),cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 2)
#             elif b[7-i3][7-j3]<250:
#                 frame[i3*80:(i3+1)*80,j3*80:(j3+1)*80,0]=255#浅紫色
#                 frame[i3*80:(i3+1)*80,j3*80:(j3+1)*80,2]=255
#                 cv.putText(frame,"{}".format("<250"),(int(j3*80),int(i3*80+40)),cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 2)
#             elif b[7-i3][7-j3]<300:
#                 frame[i3*80:(i3+1)*80,j3*80:(j3+1)*80,0]=125#紫色
#                 frame[i3*80:(i3+1)*80,j3*80:(j3+1)*80,2]=125
#                 cv.putText(frame,"{}".format("<300"),(int(j3*80),int(i3*80+40)),cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 2)
#             elif b[7-i3][7-j3]<350:
#
#                 frame[i3*80:(i3+1)*80,j3*80:(j3+1)*80,0:2]=255#浅蓝色
#                 cv.putText(frame,"{}".format("<350"),(int(j3*80),int(i3*80+40)),cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 2)
#             else:
#                 frame[i3*80:(i3+1)*80,j3*80:(j3+1)*80,0]=255#蓝色
#                 cv.putText(frame,"{}".format(">350"),(int(j3*80),int(i3*80+40)),cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 2)
#
#
#
#     cv.imshow("fram",frame)
#     #print(frame)
#     cv.waitKey(0)
#     # 监测键盘输入是否为q，为q则退出程序
#     if cv.waitKey(1) & 0xFF == ord('q'):       # 按q退出
#         break
#
# # 释放摄像头
# cap.release()
#
# # 结束所有窗口
# cv.destroyAllWindows()
