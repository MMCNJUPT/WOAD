# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
#import open3d as o3d
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           cv2, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
POINT_FORMATS='npy'  #定义点云文件格式

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except Exception:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path, clouddir,imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', shuffle=False):
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(path, clouddir,imgsz, batch_size,
                                      augment=augment,  # augmentation
                                      hyp=hyp,  # hyperparameters
                                      rect=rect,  # rectangular batches
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=0,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):   #path=data/source/
        p = str(Path(path).resolve())  # os-agnostic absolute path
        # ifilename='roicodeimg'
        # pfilename='sendzdistance'#pointcloud
        # dfilename='senddepth'
        ifilename='ir'
        pfilename='pointcloud'#pointcloud
        dfilename='depth'

        #定义三种文件路径
        p1=p+'/'+ifilename  #rgb
        p2=p+'/'+dfilename  #depth
        p3=p+'/'+pfilename  #pointcloud
        self.p1 = p1
        self.p3 = p3

        #读路径操作
        

        if '*' in p2:   #读取depth所有路径
            files2 = sorted(glob.glob(p2, recursive=True))  # glob
        elif os.path.isdir(p2):
            files2 = sorted(glob.glob(os.path.join(p2, '*.*')))  # dir
        elif os.path.isfile(p2):
            files2 = [p2]  # files
        else:
            raise Exception(f'ERROR: {p2} does not exist')
        clouds = [x for x in files2 if x.split('.')[-1].lower() in IMG_FORMATS]
        ncloud = len(clouds)

        if '*' in p1:   #读取rgb所有路径
            files1 = sorted(glob.glob(p1, recursive=True))  # glob
        elif os.path.isdir(p1):
            files1 = sorted(glob.glob(os.path.join(p1, '*.*')))  # dir
        elif os.path.isfile(p1):
            files1 = [p1]  # files
        else:
            raise Exception(f'ERROR: {p1} does not exist')
        images = [x for x in files1 if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files1 if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.ifiles = images + videos
        self.cfiles = clouds
        #self.pfiles = points
        self.nf = ni + nv  # number of files
        self.cnf = ncloud  # number of files
        #self.pnf = npoint  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto =False #auto
        #验证文件
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        self.ccount = 0
        self.pcount = 0
        return self

    def __next__(self):
        time.sleep(0.5)
        if self.count == self.nf:
            raise StopIteration

        cpath = self.cfiles[self.count]
        num=cpath.split("_")[-2]#.split(".")[0]
        ipath =self.p1+"/ir"+num+".png"
       # num=cpath.split("depth")[-1].split(".")[0]
       # ppath =self.p3+"/zdistance"+num+".npy"
       # ipath =self.p1+"/ir"+num+".png"


        # #处理视频
        # if self.video_flag[self.count]:  #加载video
        #     # Read video
        #     self.mode = 'video'
        #     ret_val, img0 = self.cap.read()
        #     while not ret_val:
        #         self.count += 1
        #         self.cap.release()
        #         if self.count == self.nf:  # last video
        #             raise StopIteration
        #         else:
        #             path = self.files[self.count]
        #             self.new_video(path)
        #             ret_val, img0 = self.cap.read()
        #
        #     self.frame += 1
        #     s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        #else:
            # Read image of rgb
           #  self.count += 1
           #  img0 = cv2.imread(path)  # BGR
           # ####th0=cv2.imread()
           #  assert img0 is not None, f'Image Not Found {path}'
           #  s = f'image {self.count}/{self.nf} {path}: '
        self.count += 1

        #读取图像文件
        img0 = cv2.imread(ipath)  # BGR
        ####th0=cv2.imread()
        assert img0 is not None, f'Image Not Found {ipath}'
        si = f'image {self.count}/{self.nf} {ipath}: '
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        #读取cloud文件
        cloud00 = cv2.imread(cpath)  # BGR
        cloud0=cv2.medianBlur(cloud00, 3)
        # print(f"img  {ipath}")
        # print(f" depth  {cpath}")
        #print(cloud0.shape)
        ####th0=cv2.imread()
        assert cloud0 is not None, f'Image Not Found {cpath}'
        cloud = letterbox(cloud0, self.img_size, stride=self.stride, auto=self.auto)[0]
        # Convert
        cloud = cloud.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        cloud = np.ascontiguousarray(cloud)

        #读取pointcloud文件
        # pcd = o3d.io.read_point_cloud(ppath, remove_nan_points=True, remove_infinite_points=True)
        # lines = np.asarray(pcd.points)
        # z_distance = []
        # for i in range(len(lines)): #求z轴的距离z_distance.append(abs(lines[i][2]))
        #     z_distance.append(abs(lines[i][2]) if abs(lines[i][2]) <400 else 0 )
        # z_distance = np.array(z_distance)
        # z_distance = z_distance.reshape(480, 640) #先reshape成图像大小
        #z_distance=np.load(ppath)
        z_distance=np.zeros((480,640))

        return ipath, img, img0, self.cap, si ,cpath, cloud, cloud0, z_distance

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    # YOLOv5 local webcam dataloader, i.e. `python detect.py --source 0`
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None, s

    def __len__(self):
        return 0


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, new_client_socket, save_dir_roicode,save_dir_senfdepth,save_dir_sendzdistance,sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.p=0
        self.save_dir_roicode=str(save_dir_roicode)
        self.save_dir_senfdepth=str(save_dir_senfdepth)
        self.save_dir_sendzdistance=str(save_dir_sendzdistance)
        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]
        self.temp_data = b''
        n = len(sources)
        self.rois,self.imgs, self.fps, self.frames, self.threads,self.depth ,self.cons, self.z_distance= [None] * n,[None] * n, [0] * n, [0] * n, [None] * n,[None] * n,[None] * n,[None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto =False #auto
        self.new_client_socket=new_client_socket
        self.MAX_BYTES=2200000


        for i, s in enumerate(sources):  # index, source
            self.frames[i] = float('inf')  # infinite stream fallback
            self.fps[i] = 3  # 30 FPS fallback
            temp_data=b''
            flag=0
            l1=0
            head=75
            while True:
                data=self.new_client_socket.recv(self.MAX_BYTES)
                temp_data=temp_data+data
                print(f"len(data1) {len(data)}")
                print(f"len(data) {len(temp_data)}")
                if len(temp_data)>head and flag==0:
                    l1=int(temp_data[0:15].decode())
                    l2=int(temp_data[15:30].decode())
                    l3=int(temp_data[30:45].decode())
                    l4=int(temp_data[45:60].decode())
                    l5=int(temp_data[60:75].decode())

                    print(l1, l2, l3, l4, l5)
                    flag=1
                if len(temp_data)>=l1:
                    self.temp_data = temp_data[l1:]
                    break

            con = temp_data[head:l2+head].decode()
            image = np.asarray(bytearray(temp_data[head+l2:l3+head+l2]), dtype="uint8")
            text2=cv2.imdecode(image,cv2.IMREAD_COLOR)
            image = np.asarray(bytearray(temp_data[head+l2+l3:l3+head+l2+l4]), dtype="uint8")
            text3=cv2.imdecode(image,cv2.IMREAD_COLOR)
            image = np.asarray(bytearray(temp_data[l3+head+l2+l4:l3+head+l2+l4+l5]), dtype="uint8")
            self.cons[i]=list(map(int,con.split(" ")))
            text4=cv2.imdecode(image,0)
            text5=Convert2RGB(text4)



            self.rois[i] =text2
            self.imgs[i] =text3
            self.depth[i]=text5
            self.z_distance[i]=text4*(self.cons[i][-1]/255)
            print(self.cons[i])
            print(self.rois[i].shape)
            print(self.imgs[i].shape)
            print(self.depth[i].shape)
            print(self.z_distance[i].shape)
            print(type(self.z_distance[i]))
            print(f'self.z_distance[i] {self.z_distance[i]}')

            self.imgs[i][self.cons[i][0]:self.cons[i][0]+self.cons[i][2], self.cons[i][1]:self.cons[i][1]+self.cons[i][3]] = self.rois[i]
            cv2.imwrite(self.save_dir_roicode+f"/ir{self.cons[i][-2]}.jpeg", self.imgs[i])
            cv2.imwrite(self.save_dir_senfdepth+f"/depth{self.cons[i][-2]}.jpeg", self.depth[i])
            np.save(self.save_dir_sendzdistance+f"/zdistance{self.cons[i][-2]}.npy", self.z_distance[i])

            self.p=self.p+1
        LOGGER.info('')  # newline

        self.rect = True#np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal


    def __iter__(self):
        self.count = 0
        return self0

    def __next__(self):
  
        i = 0
        flag = 0
        head = 75
        temp_data=self.temp_data
        while True:
            data = self.new_client_socket.recv(self.MAX_BYTES)
            print(len(data))
            if not data:
                break

            temp_data = temp_data + data
            if flag == 0:
                print("flag", flag, len(temp_data))
                l1 = int(temp_data[0:15].decode("ascii"))
                l2 = int(temp_data[15:30].decode("ascii"))
                l3 = int(temp_data[30:45].decode("ascii"))
                l4 = int(temp_data[45:60].decode("ascii"))
                l5 = int(temp_data[60:75].decode("ascii"))
                print(l1, l2, l3, l4, l5)
                # length = temp_data[0:60].decode()
                # length = length.split(sep=' ')
                # print(length)
                # temp_data = temp_data[60:]
                flag = 1
            while len(temp_data) >= 2.5*l1:
                temp_datasource=temp_data[:l1]
                l1source=l1
                l2source=l2
                l3source=l3
                l4source=l4
                l5source=l5
                temp_data=temp_data[l1:]
                l1 = int(temp_data[0:15].decode("ascii"))
                l2 = int(temp_data[15:30].decode("ascii"))
                l3 = int(temp_data[30:45].decode("ascii"))
                l4 = int(temp_data[45:60].decode("ascii"))
                l5 = int(temp_data[60:75].decode("ascii"))
                if len(temp_data)<l1:
                    temp_data=temp_datasource+temp_data
                    l1=l1source
                    l2=l2source
                    l3=l3source
                    l4=l4source
                    l5=l5source
                    break  

            if len(temp_data) >= l1:
                con = temp_data[head:l2+head].decode()
                image = np.asarray(bytearray(temp_data[head+l2:l3+head+l2]), dtype="uint8")
                text2 = cv2.imdecode(image, cv2.IMREAD_COLOR)
                image = np.asarray(bytearray(temp_data[head+l2+l3:l3+head+l2+l4]), dtype="uint8")
                text3 = cv2.imdecode(image, cv2.IMREAD_COLOR)
                image = np.asarray(bytearray(temp_data[l3+head+l2+l4:l3+head+l2+l4+l5]), dtype="uint8")
                text4=cv2.imdecode(image,0)

                self.cons[i] = list(map(int, con.split(" ")))
                text5=Convert2RGB(text4)
                self.rois[i] = text2
                self.imgs[i] = text3
                self.depth[i] = text5
                self.z_distance[i] = text4*(self.cons[i][-1]/255)
                # print(self.cons[i])
                # print(self.rois[i].shape)
                # print(self.imgs[i].shape)
                # print(self.depth[i].shape)
                # print(self.z_distance[i].shape)
                # print(f'self.z_distance[i] {self.z_distance[i]}')
                cv2.imwrite(self.save_dir_senfdepth+f"/depth{self.cons[i][-2]}.jpeg", self.depth[i])
                self.imgs[i][self.cons[i][0]:self.cons[i][0]+self.cons[i][2], self.cons[i][1]:self.cons[i][1]+self.cons[i][3]] = self.rois[i]
                cv2.imwrite(self.save_dir_roicode+f"/ir{self.cons[i][-2]}.jpeg", self.imgs[i])
                np.save(self.save_dir_sendzdistance+f"/zdistance{self.cons[i][-2]}.npy", self.z_distance[i])
                self.p = self.p+1
                flag = 0
                self.temp_data = temp_data[l1:]
                break

        self.count += 1

        # Letterbox

        si = f'image {self.count}/{self.p}: '
        depth0 = self.depth.copy()
        depth = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in depth0]
        depth = np.stack(depth, 0)# Stack
        depth = depth[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW# Convert
        depth = np.ascontiguousarray(depth)


        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]
        img = np.stack(img, 0)# Stack
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW# Convert
        img = np.ascontiguousarray(img)
        #return self.sources, img, img0, None, ''
        z_distance= self.z_distance[i].copy()#用于报距离
        return f"/ir{self.cons[i][-2]}.jpeg", img, img0, 0, si ,f"/depth{self.cons[i][-2]}.jpeg", depth, depth0, z_distance

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
def Convert2RGB(depth, width=480, height=640):
    depth = depth.astype('uint16')
    step = 255
    start_num = 128
    color_length = start_num + 3 * step + 128
    color_data = np.zeros(shape=(width, height, 3), dtype='uint8')
    min_color = np.amin(a=depth)
    max_color = np.amax(a=depth)
    if max_color < 255:
        max_color = 255
    if min_color > 0:
        max_color = 0
    index_new_depth = (depth - min_color)/(max_color - min_color) * color_length
    index_new_depth.astype('uint16')
   # print(f'index_new_depth{index_new_depth}')
    # exit()
   # print(f'depth{depth}')
    # exit()
    temp = index_new_depth.copy()
    # print(temp)
    # print(temp.shape)
    # exit()
    temp1 = np.zeros(shape=(width, height))

    temp1[temp > 0] = 1
    # print(temp1, temp1.shape)
    # exit()
    temp1[temp > start_num] = 2
    temp1[temp > start_num + step] = 3
    temp1[temp > start_num + 2 * step] = 4
    temp1[temp > start_num + 3 * step] = 5
    # temp1[temp1 == 0] = 6

    color_data[:, :, 0] = np.where(temp1 == 1, 127 + temp, color_data[:, :, 0])

    color_data[:, :, 0] = np.where(temp1 == 2, 255, color_data[:, :, 0])
    color_data[:, :, 1] = np.where(temp1 == 2, temp - start_num, color_data[:, :, 1])

    color_data[:, :, 0] = np.where(temp1 == 3, 2 * step + start_num - temp, color_data[:, :, 0])
    color_data[:, :, 1] = np.where(temp1 == 3, 255, color_data[:, :, 1])
    color_data[:, :, 2] = np.where(temp1 == 3, temp - start_num - step, color_data[:, :, 2])

    color_data[:, :, 1] = np.where(temp1 == 4, 3 * step + start_num - temp, color_data[:, :, 1])
    color_data[:, :, 2] = np.where(temp1 == 4, 255, color_data[:, :, 2])

    color_data[:, :, 2] = np.where(temp1 == 5, 128, color_data[:, :, 2])
    color_data[:, :, 2] = np.where(temp1 == 0, 128, color_data[:, :, 2])

    color_data = color_data.astype('uint8')
    color_data[depth == 1] = [127, 0, 170]
    color_data[depth == 2] = [127, 0, 170]
    color_data[depth == 3] = [127, 0, 170]
    color_data[depth == 0] = [0, 0, 0]
    # color_data[index_new_depth == 1] = [127, 0, 170]

    # color_data[index_new_depth == 1] = [127, 0, 170]
    return color_data

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    def __init__(self, path, clouddir,img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic =False #self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None
        self.clouddir=clouddir
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f'{prefix}No images found'

        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # Check cache
        self.label_files = img2label_paths(self.im_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # same hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride
        # Cache images into RAM/disk for faster training (WARNING: large datasets may exceed system resources)
        self.ims = [None] * n
        self.cloudimgs=[None] * n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, bar_format=BAR_FORMAT)
            for i, x in pbar:
                if cache_images == 'disk':
                    gb += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.im_files), bar_format=BAR_FORMAT)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def __len__(self):
        return len(self.im_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        # if mosaic:
        #     # Load mosaic
        #     img, labels = self.load_mosaic(index)
        #     shapes = None
        #
        #     # MixUp augmentation
        #     if random.random() < hyp['mixup']:
        #         img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))
        #
        # else:
        # Load image
        img,cloudim, (h0, w0), (h, w) = self.load_image(index)
        cloudim, _, _ = letterbox(cloudim, 640, auto=False)
        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        #print(img.size)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        #print(labels)
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img, labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        #print(labels)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        cloudim = cloudim.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        cloudim = np.ascontiguousarray(cloudim)
        return torch.from_numpy(img),torch.from_numpy(cloudim) ,labels_out, self.im_files[index], shapes

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
        cloudim=self.cloudimgs[i]
        if im is None:  # not cached in RAM
            cloudfile = self.clouddir + f.split("/")[-1]
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            cloudim0 = cv2.imread(cloudfile)
            cloudim=cv2.medianBlur(cloudim0, 3)
            if r != 1:  # if sizes are not equal
                im = cv2.resize(im,
                                (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA)

            return im,cloudim, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        else:
            return self.ims[i],cloudim, self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
        img4, labels4 = random_perspective(img4, labels4, segments4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img4, labels4

    def load_mosaic9(self, index):
        # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        img9, labels9 = random_perspective(img9, labels9, segments9,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img9, labels9

    @staticmethod
    def collate_fn(batch):
        im,cloud, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0),torch.stack(cloud, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                lb = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(im)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path=DATASETS_DIR / 'coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(str(path) + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path=DATASETS_DIR / 'coco128'):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path=DATASETS_DIR / 'coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = segments[i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('path/to/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith('.zip'):  # path is data.zip
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            ZipFile(path).extractall(path=path.parent)  # unzip
            dir = path.with_suffix('')  # dataset directory == zip name
            return True, str(dir), next(dir.rglob('*.yaml'))  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, 'JPEG', quality=75, optimize=True)  # save
        except Exception as e:  # use OpenCV
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_yaml(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data['path'] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)  # shape(128x80)
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.im_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.im_files), total=dataset.n, desc='HUB Ops'):
                pass

    # Profile
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file) as f:
                x = json.load(f)  # load hyps dict
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

    # Save, print and return
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats
