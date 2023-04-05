
import time
import zmq
import pickle, os

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
count = 1
while True:
    #  Wait for next request from client
    message = socket.recv()
    message = pickle.loads(message)
    print(f"Received request: {count}\n {message}\n")

    #  Send reply back to client
    socket.send(b"nothing")
    print(f"send count:{count}\n")
    count = count + 1
