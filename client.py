import grpc
import time
import cv2
import model_pb2 as pb2
import model_pb2_grpc as pb2_grpc
import numpy as np
from datetime import datetime
import json
import csv


csv_file_path = 'res.csv'

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
    pass  # File alr

config = read_parameters_from_json('config.json')

def generate_frames():
    # Replace this with your video capture logic
    id = 0
    print("starting the camera")
    camera_index = config.get("camera")
    delay = config.get("delay")
    video_capture = cv2.VideoCapture(camera_index)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        # Convert the frame to bytes
        frame = cv2.resize(frame , (1024 , 768))
        id+=1
        _, frame_data = cv2.imencode('.jpg', frame)
        print("__________________________________________")
        print(f"Frame sent at {datetime.utcnow()}")
        yield pb2.VideoFrame(frame_data=frame_data.tobytes() , id=id)
        time.sleep(delay)

def send_frames(stub):
    latest_frame = None
    for frame in generate_frames():
        client_sent = datetime.utcnow()
        frame_id = frame.id
        
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

def run():
    
    ip_address = config.get('ip_address')
    port = config.get('port')
    channel = grpc.insecure_channel(f'{ip_address}:{port}')
    stub = pb2_grpc.VideoStreamStub(channel)
    send_frames(stub)

if __name__ == '__main__':
    run()
