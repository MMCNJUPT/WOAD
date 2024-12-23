
import argparse
import time
import os
import sys
from pathlib import Path
import pyttsx3
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import onnx
# import onnx.utils
# import onnx.version_converter
# import onnxsim
import socket
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from torchsummary import summary
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import cv2
from pynput import keyboard
isEnd = False
import threading

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    global isEnd
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    nameroicode="roicodeimg"
    save_dir_roicode = increment_path(Path(save_dir) / nameroicode, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir_roicode).mkdir(parents=True, exist_ok=True)  # make dir
    nameroicode="depth"
    save_dir_depth = increment_path(Path(save_dir) / nameroicode, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir_depth).mkdir(parents=True, exist_ok=True)  # make dir
    nameroicode="senddepth"
    save_dir_senfdepth = increment_path(Path(save_dir) / nameroicode, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir_senfdepth).mkdir(parents=True, exist_ok=True)  # make dir
    nameroicode="sendzdistance"
    save_dir_sendzdistance= increment_path(Path(save_dir) / nameroicode, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir_sendzdistance).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    print(weights)

    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # summary(model, input_size=(3, 640, 640))
    #
    # exit()
    # print(model)
    # print(model.stride, model.names, model.pt)#32 ['person', 'stairs'] True
    stride, names, pt = model.stride, model.names, model.pt

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))  # warmup
    #

    socktovoice = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    voiceip="127.0.0.1"
    port=1062
    try:
        socktovoice.connect((voiceip, port))#here is internal IP address ,need no modification
    except:
        pass

    # host='127.0.0.1'
    # port=7788
    # lidarsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # lidarsock.bind((host, port))
    # print('Listening at {}'.format(lidarsock.getsockname()))
    # lidarsock.listen()
    # new_client_socket,client_addr=lidarsock.accept()
    # print(client_addr)
    # webcam=True
    new_client_socket,client_addr=0,0
    # new_client_socket=0
    if webcam:
        view_img = True#check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(new_client_socket, save_dir_roicode,save_dir_senfdepth,save_dir_sendzdistance,source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)#这里是数据加载需要修改的
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    dt, seen = [0.0, 0.0, 0.0], 0
    print(conf_thres, iou_thres)
    detsum=0
    detnum=0
    detsx={}
    detsy={}
    img = cv2.imread("data/testdata/rgb/ir130.png")
    depth = cv2.imread("data/testdata/depth/depth_130_num2.png")
    new_img = np.zeros((480, 1280, 3))
    new_img[:, :640, :] = img[:,:,:]
    new_img[:, 640:, :] = depth[:,:,:]
    cv2.imwrite("result.png", new_img)
    # cv2.imshow("sss",new_img/255)
    # cv2.waitKey(0)
    new_img = new_img
    new_img1 = new_img.transpose((2, 0, 1))[::-1]
    new_img1 = np.ascontiguousarray(new_img1)
    new_img1= np.expand_dims(new_img1, axis=0)
    print(new_img1.shape)
    new_img_in = torch.from_numpy(new_img1).float()/255.0
    pred = model(new_img_in, augment=augment, visualize=visualize)
    for i in pred[0]:
        print(i)
        break
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    print(pred)
    exit()
    for enum , (path, im, im0s, vid_cap, s ,cpath, cloud, cloud0, z_distance) in enumerate(dataset):
        t1 = time_sync()
        # print(path)
        # print(cpath)
        # print(im)
        # print(im[:,114:200,153:250])
        # for i1 in range(3):
        #     for i2 in range(480):
        #         for i3 in range(640):
        #             if im[i1,i2,i3]==0.0:
        #                 im[i1,i2,i3]=120
        # img = im[0].transpose((1, 2, 0))#[::-1]
        # img = np.ascontiguousarray(img)
        # cv2.imwrite("runs/img2.png", img)
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32  model.fp16 =  False,so im.float 32 bits
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        cloud = torch.from_numpy(cloud).to(device)
        cloud = cloud.half() if model.fp16 else cloud.float()  # uint8 to fp16/32
        cloud /= 255  # 0 - 255 to 0.0 - 1.0
        if len(cloud.shape) == 3:
            cloud = cloud[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        xclouds = torch.cat((im, cloud), axis=3)

        # train2=True
        # model_file = 'yolov5slidar3.onnx'
        # torch.onnx.export(
        #     model,
        #     (im,cloud),
        #     model_file,
        #     export_params=True,
        #     opset_version=12,
        #     verbose=False,
        #     training=torch.onnx.TrainingMode.TRAINING if train2 else torch.onnx.TrainingMode.EVAL,
        #     do_constant_folding=not train2,
        #     input_names=['images'],
        #     output_names=['output'],
        #     dynamic_axes={'images': {0: 'batch',2: 'height', 3: 'width'},  # shape(1,3,640,640)
        #                   'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
        #                   }
        # )
        # onnx_model = onnx.load(model_file)
        # onnx.checker.check_model(onnx_model)
        # onnx.save(onnx_model, model_file)
        # print("save")
        # exit()

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(xclouds, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS\
        # print(agnostic_nms)
        # print(max_det)
        # exit()
        for i in pred[0]:
            print(i)
            break
        print(conf_thres, iou_thres, classes, agnostic_nms, max_det)
        #
        exit()
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image print(len(pred))=1
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0,depth0, frame = path, im0s[i].copy(),cloud0[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()# if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):#det是目标的数量
                print(det)
                newdetsx={}
                newdetsy={}
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()#lable type: xyxy
                # print(det)
                detsum=detsum+1
                csave=np.zeros([len(det),2])

                for numdat in range(len(det)):
                    csave[numdat,0]=det[numdat,5]
                    csave[numdat,1]=det[numdat,4]
                    img_x=int((det[numdat,0]+det[numdat,2]).cpu()/2)
                    img_y=int((det[numdat,1]+det[numdat,3]).cpu()/2)
                    newdetsx[numdat]=img_x
                    newdetsy[numdat]=img_y
                updatekeys=[]
                newkeys=[]
                oldkeys=list(detsx.keys())
                update_det_inder = []
                new_det_inder = []
                for idet in range(len(newdetsx)):
                    detx=newdetsx[idet]
                    dety=newdetsy[idet]
                    if len(detsx)>0:
                        flag=0
                        for item in detsx.keys():
                            a=abs(detx-detsx[item])
                            a2=abs(dety-detsy[item])
                            if (a+a2)<30:
                                detsx[item]=detx
                                detsy[item]=dety
                                updatekeys.append(item)
                                if item in oldkeys:
                                    oldkeys.remove(item)
                                flag=1
                                update_det_inder.append(idet)
                                break
                        if flag==0:
                            l=len(detsx)
                            detsx[l]=detx
                            detsy[l]=dety
                            newkeys.append(l)
                            new_det_inder.append(idet)
                    else:
                        detsx[idet]=detx
                        detsy[idet]=dety

                for inderk1, k1 in enumerate(newkeys):

                    detx=detsx[k1]
                    dety=detsy[k1]
                    sumdistance=0
                    distancenumber=0
                    for zi in range(-10,10,1):
                        for zj in range(-10,10,1):
                            try:
                                if z_distance[zj+dety,zi+detx]>0:
                                    sumdistance=sumdistance+z_distance[zj+dety,zi+detx]
                                    distancenumber=distancenumber+1
                            except:
                                continue
                    if distancenumber>0:
                        sumdistance=int(sumdistance/distancenumber)
                    sumdistance=sumdistance/10
                    detinder=new_det_inder[inderk1]
                    cls=int(det[detinder][5].cpu())
                    print("new object in ("+str(dety)+","+str(detx)+")" +" distance "+ str(sumdistance)+"cm"+"  cls: "+str(cls))
                for inderk1, k1 in enumerate(updatekeys):
                    detx=detsx[k1]
                    dety=detsy[k1]
                    sumdistance=0
                    distancenumber=0
                    for zi in range(-10,10,1):
                        for zj in range(-10,10,1):
                            try:
                                if z_distance[zj+dety,zi+detx]>0:
                                    sumdistance=sumdistance+z_distance[zj+dety,zi+detx]
                                    distancenumber=distancenumber+1
                            except:
                                continue
                    if distancenumber>0:
                        sumdistance=int(sumdistance/distancenumber)
                    sumdistance=sumdistance/10
                    detinder=update_det_inder[inderk1]
                    cls=int(det[detinder][5].cpu())
                    if sumdistance<150:
                        print("old object in ("+str(dety)+","+str(detx)+")" +" distance "+ str(sumdistance)+"cm"+"  cls: "+str(cls))




                #文本转语言
                # engine = pyttsx3.init()
                # engine.setProperty("voice","HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ZH-CN_HUIHUI_11.0")
                # engine.say("楼梯在中间")
                # engine.runAndWait()

                img_center=(det[0,0]+det[0,2])
                # print(img_center)
                direction = (img_center-im0.shape[1])/im0.shape[1]
                direction = direction.to("cpu")
                # if direction < 0.1 and direction > -0.1:
                #     print("stairs in the middle")
                #
                # elif direction <= 0.3:
                #     print("the stairs is slightly to the right")
                # elif direction >= -0.3:
                #     print("the stairs is slightly to the left")
                # elif direction > 0.3:
                #     print("the stairs to the right")
                # elif direction < -0.3:
                #     print("the stairs to the left")



                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    centerx=int((xyxy[0]+xyxy[2])/2)
                    centery = int(( xyxy[1]+xyxy[3])/2)
                    sumdistance=0
                    distancenumber=0
                    for zi in range(-10,10,1):
                        for zj in range(-10,10,1):
                            try:
                                if z_distance[zj+centery,zi+centerx]>0:
                                    sumdistance=sumdistance+z_distance[zj+centery,zi+centerx]
                                    distancenumber=distancenumber+1
                            except:
                                continue
                    if distancenumber>0:
                        sumdistance=int(sumdistance/distancenumber)
                    sumdistance=sumdistance/10
                    #print(f'distance  {sumdistance} cm')
                    # print(centerx, centery, cls)



                    try:
                        if cls==0.0:
                            l1=f"1 {sumdistance}"
                            data1 = l1.encode('ascii')#person
                            socktovoice.send(data1)
                        elif cls==1.0:
                            l2=f"2 {sumdistance}"
                            data2 = l2.encode('ascii')#box
                            socktovoice.send(data2)
                        elif cls==2.0:
                            l3=f"3 {sumdistance}"
                            data3 = l3.encode('ascii')#stairs
                            socktovoice.send(data3)
                    except:
                        pass
                    if save_txt:  # Write to file false
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image  true
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            else:
                csave=np.zeros([1])
                csave[0]=5
            #filenum=path.split("\\")[-1].split(".")[0]

            filename=f"runs/label_conf_npy/{enum}.npy"
            #np.save(filename, csave)
            # Stream results
            im0 = annotator.result()
            view_img= True
            #print(imc.shape)
            # if view_img:
            #     cv2.imshow("result", im0)
            #     #cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                elif dataset.mode == 'stream':
                    cv2.imwrite(save_path, im0)
                    cv2.imwrite(str(save_dir_depth)+cpath, depth0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        if isEnd:
            print("esc break")
            break
    print(detsum)

    # # Print results
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # if update:
    #     strip_optimizer(weights)  # update model (to fix SourceChangeWarning)vi datqa


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp97/weights/best.pt', help='model path(s)')#权重文件runs/train/exp54/weights/last.pt
    parser.add_argument('--source', type=str, default=ROOT / 'data/testdata/', help='file/dir/URL/glob, 0 for webcam')#测试数据data/source2/
    parser.add_argument('--data', type=str, default=ROOT / 'data/stairscoco.yaml', help='(optional) dataset.yaml path')#
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))
def keyboard_on_press(key):
    global isEnd
    try:
        print('字母键{0} press'.format(key.char))
    except AttributeError:
        print('特殊键{0} press'.format(key))
        if key == keyboard.Key.esc:
            isEnd = True
            return False
def function(opt):
    global isEnd
    # 创建、启动键盘监听线程,阻塞版本
    listener = keyboard.Listener(on_press=keyboard_on_press)
    # 创建、启动鼠标监听线程
    t2 = threading.Thread(target=main,args=(opt,))
    listener.start()
    t2.start()
    listener.join()
    t2.join()

if __name__ == "__main__":
    opt = parse_opt()
    function(opt)
