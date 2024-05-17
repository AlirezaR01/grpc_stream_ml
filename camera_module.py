#Camera Producer

import pika
import cv2
import json
import numpy


def read_parameters_from_json(json_file):
    with open(json_file, 'r') as f:
        parameters = json.load(f)
    return parameters


#RabbitMQ Producer initialization
connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)
RabbitMQ_channel = connection.channel()
# Queue exists, update its settings


Config = read_parameters_from_json("config.json")



# Camera connection & config

camera_index = Config.get("camera")
delay = Config.get("delay")
fps = Config.get("fps")
cap = cv2.VideoCapture(camera_index)

if cap.isOpened() == True:
    print("Connected to camera successfully")
else : 
    print("failed to connect to the camera")

try : 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
except : 
    print("unable to set resolution , proceeding with default value")

while True : 
    ret , frame = cap.read()
    if not ret : 
        print("unable to read frame from camera")
        break
    frame = cv2.resize(frame , (1024 , 768))
    
    #encode and convert image to byte
    frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

    #publish frames on queue
    
    RabbitMQ_channel.basic_publish(exchange='', routing_key='Camera_Queue', body=frame_bytes)



cap.release()
connection.close()





