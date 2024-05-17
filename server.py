import grpc
from concurrent import futures
import model_pb2 as pb2
import model_pb2_grpc as pb2_grpc
import cv2
import numpy as np
from Pylon_Model import Pylon_Model
from datetime import datetime
import ultralytics
from ultralytics import YOLO
import time


pylon_model = Pylon_Model("best.pt")
print("yolov5 initialized")
yolo8_model = YOLO("yolov8n.pt")
print("yolov8 initialized")

global img
class VideoStreamServicer(pb2_grpc.VideoStreamServicer):
    global img
    def StreamFrames(self, request_iterator, context):
        for frame in request_iterator:
            # Process the received frame

            print("_________________________________________")
            print(f"Frame {frame.id} recived at {datetime.utcnow()}")
            recived_time = datetime.utcnow()
            frame_data = np.frombuffer(frame.frame_data, dtype=np.uint8)
            id = frame.id
            target = frame.target
            img = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            if target < 0 :
                bbox , run_time = pylon_model.Run_Model(img)
            else :
                t00 = time.time()
                results = yolo8_model(img , show=False , stream=False , classes=[target] , max_det=1)
                t01 = time.time()
                run_time = t01 - t00
                print(f"runtime is {run_time}")
                for result in results :
                    
                    boxes = result.boxes
                    if len(boxes) == 0 :
                        bbox = [0 , 0 , 0 , 0 ,0]
                        Detected = False                                     
                    else :
                        x1 , y1 ,x2,y2  = boxes.xyxy[0]
                        conf = float(boxes.conf)
                        x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
                        bbox = [x1 , y1 , x2 , y2 , conf]
                        Detected = True
                        
                        

            print(f"Result of frame {frame.id} sent at {datetime.utcnow()}")
            sent_time = datetime.utcnow()
            response = pb2.Model_Data(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3], confidence=bbox[4] ,
                                       id = id , server_recived=str(recived_time ), server_sent=str(sent_time)
                                       , model_runtime = run_time , detected = Detected)
            
            yield response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_VideoStreamServicer_to_server(VideoStreamServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
