import grpc
import time
import cv2
import model_pb2 as pb2
import model_pb2_grpc as pb2_grpc
import numpy as np
from datetime import datetime
import json
import csv
import threading

global cv_image

csv_file_path = 'res.csv'

stop_thread = False  # Flag to indicate whether the thread should stop

def write_to_csv(frame_id,client_sent, server_recived , model_runtime ,server_sent , client_recived):
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([frame_id,client_sent, server_recived , model_runtime,server_sent , client_recived])

try:
    open(csv_file_path, 'x')
    write_to_csv("id" , "Client Sent" , "Server Recived" , "Model runtime" , 
                 "Server sent" , "Client Recived")
except FileExistsError:
    pass  # File already exists
        
def read_parameters_from_json(json_file):
    with open(json_file, 'r') as f:
        parameters = json.load(f)
    return parameters

try:
    open(csv_file_path, 'x')
except FileExistsError:
    print("Unable to open csv file")

config = read_parameters_from_json('config.json')

def FrameGenerator() : 
    global cv_image, stop_thread
    i = 0
    print("starting the camera")
    camera_index = config.get("camera")
    delay = config.get("delay")
    video_capture = cv2.VideoCapture(camera_index)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while not stop_thread:
        i+=1
        ret, cv_image = video_capture.read()
        if not ret:
            "no data"
            continue
        # Convert the frame to bytes
        cv_image = cv2.resize(cv_image , (1024 , 768))
        # print(f"frame {i} generated")
    video_capture.release()  # Release the camera when done

def SendFrames(stub) : 
    global cv_image
    id = 0
    while True : 
        try :
            id+=1
            # cv2.imshow("tag" , cv_image)
            # cv2.waitKey(0)
            _, frame_data = cv2.imencode('.jpg', cv_image)
            print("__________________________________________")
            print(f"Frame sent at {datetime.utcnow()}")
            frame = pb2.VideoFrame(frame_data=frame_data.tobytes() , id=id)
            frame_id = frame.id
            client_sent = datetime.utcnow()
            response = stub.StreamFrames(iter([frame]))

            response = stub.StreamFrames(iter([frame]))
            client_recived = datetime.utcnow()
            print(f"Result received at {client_recived}")
            for resp in response:
                # Access fields of the Model_Data message
                x1 = resp.x1
                y1 = resp.y1
                x2 = resp.x2
                y2 = resp.y2
                confidence = resp.confidence
                id = resp.id
                server_recived = resp.server_recived
                server_sent = resp.server_sent
                model_runtime = resp.model_runtime

                write_to_csv(frame_id, client_sent, server_recived ,model_runtime
                            ,  server_sent , client_recived)
                
                # Draw bounding box on the frame
                frame = cv2.imdecode(np.frombuffer(frame.frame_data, np.uint8), cv2.IMREAD_COLOR)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Print or use the received data as needed
                print(f"id : {id}")
                print(f"Bounding box coordinates: ({x1}, {y1}), ({x2}, {y2})")
                print("Confidence:", confidence)
                
                # Show the frame with bounding box
                cv2.imshow('Frame with bounding box', frame)
                cv2.waitKey(1)  # Adjust delay as needed
                break
        except :
            print("gRPC thread is not ready")

def run():
    global stop_thread
    ip_address = config.get('ip_address')
    port = config.get('port')
    channel = grpc.insecure_channel(f'{ip_address}:{port}')
    stub = pb2_grpc.VideoStreamStub(channel)
    SendFrames(stub)
    cv2.destroyAllWindows()  # Close OpenCV windows when done
    stop_thread = True  # Set the flag to stop the thread


if __name__ == "__main__" :
    thread1 = threading.Thread(target=FrameGenerator)

    thread1.start()
    # thread1.join()
    run()
    thread1.join()
