# import os
# def myrename(path):
#     file_list=os.listdir(path)
#     for i,fi in enumerate(file_list):
#         old_name=os.path.join(path,fi)
#         new_name=os.path.join(path,str(i))
#         os.rename(old_name,new_name)
#
# if __name__=="__main__":
#     path="D:/Video/data_aug5"
#     myrename(path)
# import os
#
# # import re
# # import sys
#
# path = r'D:/Video/data_aug5'
#
# filelist = os.listdir(path)
# filetype = '.jpg'

# for file in filelist:
# print(file)

# for file in filelist:
#     Olddir = os.path.join(path, file)
#     # print(Olddir)
#
#     if os.path.isdir(Olddir):
#         continue
#
#     # os.path.splitext("path")：分离文件名与扩展名
#     filename = os.path.splitext(file)[0]
#     filetype = os.path.splitext(file)[1]
#
#     # zfill() 方法返回指定长度的字符串，原字符串右对齐，前面填充0
#     Newdir = os.path.join(path, filename.zfill(6) + filetype)
#     os.rename(Olddir, Newdir)
import os

path = "D:/Video/data_aug5"
filelist = os.listdir(path)
count = 0
for file in filelist:
    print(file)
for file in filelist:
    Olddir = os.path.join(path, file)
    if os.path.isdir(Olddir):
        continue
    filename = os.path.splitext(file)[0]
    filetype = os.path.splitext(file)[1]
    Newdir = os.path.join(path, str(count).zfill(4) + filetype)
    os.rename(Olddir, Newdir)

    count += 1



