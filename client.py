import grpc
import time
import cv2
import model_pb2 as pb2
import model_pb2_grpc as pb2_grpc
import numpy as np
from datetime import datetime
import json

def read_parameters_from_json(json_file):
    with open(json_file, 'r') as f:
        parameters = json.load(f)
    return parameters
config = read_parameters_from_json('config.json')

def generate_frames():
    # Replace this with your video capture logic
    id = 0
    camera_index = config.get("camera")
    fps = config.get("fps")
    video_capture = cv2.VideoCapture(camera_index)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video_capture.set(cv2.CAP_PROP_FPS, fps)
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
        time.sleep(0.1)

def send_frames(stub):
    for frame in generate_frames():
        response = stub.StreamFrames(iter([frame]))
        print(f"Result recived at {datetime.utcnow()}")
        for resp in response:
            # Access fields of the Model_Data message
            x1 = resp.x1
            y1 = resp.y1
            x2 = resp.x2
            y2 = resp.y2
            confidence = resp.confidence
            id = resp.id
            
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
