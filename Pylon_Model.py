#!/usr/bin/python3.8

import time
import utils.datasets as ud
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import LoadStreams, LoadImages

# Manually set YOLOv5 arguments
# weights = ["/home/alireza/yolo5_light/YOLOv5-Lite/best.pt"]
source = "test6.png"
img_size = 640
conf_thres = 0.45
iou_thres = 0.45
device = ""
view_img = False
save_txt = False
save_conf = False
nosave = False
classes = None
agnostic_nms = False
augment = False
update = False
project = "/home/px4vision/catkin_ws/src/vision_control/nodes/runs/detect"
name = "exp"
exist_ok = False


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Assign values to the Namespace object


class Pylon_Model :
    def __init__(self , weight) :
        self.opt = Config(
        weights=weight,
        source=source,
        img_size=img_size,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        device=device,
        view_img=view_img,
        save_txt=save_txt,
        save_conf=save_conf,
        nosave=nosave,
        classes=classes,
        agnostic_nms=agnostic_nms,
        augment=augment,
        update=update,
        project=project,
        name=name,
        exist_ok=exist_ok
        )
        self.Carrier = self.Model_Device_Initialize()


    def Model_Device_Initialize(self) :
        source, weights, view_img, save_txt, imgsz = (
        self.opt.source,
        self.opt.weights,
        self.opt.view_img,
        self.opt.save_txt,
        self.opt.img_size,
        )
        # Initialize
        set_logging()
        self.device = select_device(self.opt.device)
        self.half = self.device.type != "cpu"  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        if self.half:
            model.half()  # to FP16
        print("Model and Device Initialized and Loaded")
        names = model.module.names if hasattr(model, "module") else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        return model ,  [names , colors]
    


    #Converts Frame to Tensor
    def Convert_Frame_Tensor(self ,Frame) : 

        Frame = ud.letterbox(Frame, (640 , 640), stride=self.stride)[0]
        
        # Convert
        Frame = Frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        Frame = np.ascontiguousarray(Frame)
        img_tensor = torch.from_numpy(Frame).to(self.device)
        img_tensor = img_tensor.half() if self.half else img_tensor.float()  # uint8 to fp16/32
            
        img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        Y_Divider , X_Divider = img_tensor.shape[2:]

        return img_tensor , Frame , Y_Divider , X_Divider



    def RatioC(self ,Frame , Y_Divider , X_Divider) : 
        ratioy = Frame.shape[0]/Y_Divider
        ratiox = Frame.shape[1]/X_Divider
        return ratiox , ratioy

    
    
    def Run_Model(self , Frame) : 
        model , Nac = self.Carrier
        
        inp_img , res_img , Y_Divider , X_Divider = self.Convert_Frame_Tensor(Frame)
        rx , ry = self.RatioC(Frame , Y_Divider , X_Divider)
        t00 = time.time()
    
        pred = model(inp_img,augment=self.opt.augment)[0]
        t01 = time.time()

        print(f"time = {t01 -t00} s")
        total_time = t01 - t00

        pred = non_max_suppression(
                pred,
                self.opt.conf_thres,
                self.opt.iou_thres,
                classes=self.opt.classes,
                agnostic=self.opt.agnostic_nms,
            )
        
        names = Nac[0]
        colors = Nac[1]
        l = []
        if len(pred[0]) > 0:
        # Iterate over the detected objects
            for det in pred[0]:
                # Extract coordinates, confidence scores, and class predictions
                x1, y1, x2, y2, conf, cls_pred = det[:6]  # Exclude cls_conf, as it might not be present

                # Check if the confidence score is above a certain threshold
                if conf > 0.45:
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = int(x1*rx), int(y1*ry), int(x2*rx), int(y2*ry)

                    # Get the label for the predicted class
                    label = names[int(cls_pred)]

                    # Display the bounding box
                    print(f"Detected {label} with confidence {conf:.2f}: ({x1}, {y1}) - ({x2}, {y2})")

                    # Draw the bounding box on the frame
                    cv2.rectangle(Frame, (x1, y1), (x2, y2), colors[0],2)
                    cv2.putText(Frame, f"{label}: {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,2)
                    return [x1 , y1 , x2 , y2 , conf] , total_time
                else:
                    print("Low Confidence - No object detected.")
            else:
                print("No objects detected.")
            
        return [0 , 0 , 0 , 0 , 0] , total_time



