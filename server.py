import grpc
from concurrent import futures
import model_pb2 as pb2
import model_pb2_grpc as pb2_grpc
import cv2
import numpy as np
from Pylon_Model import Pylon_Model
from datetime import datetime
model = Pylon_Model("best.pt")
class VideoStreamServicer(pb2_grpc.VideoStreamServicer):
    def StreamFrames(self, request_iterator, context):
        for frame in request_iterator:
            # Process the received frame

            print("_________________________________________")
            print(f"Frame {frame.id} recived at {datetime.utcnow()}")
            frame_data = np.frombuffer(frame.frame_data, dtype=np.uint8)
            id = frame.id
            img = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            bbox = model.Run_Model(img)
            
            
            response = pb2.Model_Data(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3], confidence=bbox[4] , id = id)
            print(f"Result of frame {frame.id} sent at {datetime.utcnow()}")
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
