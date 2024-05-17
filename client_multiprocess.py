#RabbitMQ consumer , it takes data from camera module
#send data to the server side using grpc

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
import pika
import threading


global RabbitMQ_channel
global stub 
global index

csv_file_path = 'res.csv'

stop_thread = False  

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
target = config.get("target")

def FrameCallBack(ch, method, properties, body) :
    # time.sleep(1) 
    global stub , index , RabbitMQ_channel
    index+=1
    print(f"frame{index} has been recived from camera module")
    # try : 
    frame_data = pb2.VideoFrame(frame_data=body , id=index , target=target)
    
    client_sent = datetime.utcnow()
    print("__________________________________________")
    print(f"Frame sent at {client_sent}")
    frame_id = frame_data.id
    response = stub.StreamFrames(iter([frame_data]))
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
        detected = resp.detected

        write_to_csv(frame_id, client_sent, server_recived ,model_runtime
                        ,  server_sent , client_recived)
        
        # Draw bounding box on the frame
        frame = cv2.imdecode(np.frombuffer(frame_data.frame_data, np.uint8), cv2.IMREAD_COLOR)
        print(detected)
        if detected:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Print or use the received data as needed
            print(f"id : {id}")
            print(f"Bounding box coordinates: ({x1}, {y1}), ({x2}, {y2})")
            print("Confidence:", confidence)
        
        # Show the frame with bounding box
        cv2.imshow('Frame with bounding box', frame)
        cv2.waitKey(1)  # Adjust delay as needed
        response = None
        break
    # ch.queue_purge(("Camera_Queue"))
    ch.basic_ack(delivery_tag=method.delivery_tag)
    
    

def run():
    global stub , index , RabbitMQ_channel
    #RabbitMQ intialization
    connection_parameters = pika.ConnectionParameters('localhost')
    connection = pika.BlockingConnection(connection_parameters)
    RabbitMQ_channel = connection.channel()

    # Queue exists, update its settings
    RabbitMQ_channel.queue_declare(queue="Camera_Queue" , arguments={"x-max-length":1 ,"overflow":"drop-head"})
    RabbitMQ_channel.queue_purge(queue='Camera_Queue')



    #gRPC initialization
    ip_address = config.get('ip_address')
    port = config.get('port')
    
    gRPC_channel = grpc.insecure_channel(f'{ip_address}:{port}')
    stub = pb2_grpc.VideoStreamStub(gRPC_channel)
    index = 0

    print("RabbitMQ : Consumer defined Successfully")
    
    RabbitMQ_channel.basic_qos(prefetch_count=1)
    RabbitMQ_channel.basic_consume(queue="Camera_Queue",auto_ack=False, on_message_callback=FrameCallBack)
    print("RabbitMQ : Start consuming from camera module")
    RabbitMQ_channel.start_consuming()
    


    cv2.destroyAllWindows() 



if __name__ == "__main__" :
    
    run()
