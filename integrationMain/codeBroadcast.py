import threading
import socket
import sys
import time

hostIP = ''      # localhost
hostPort = 2000  # Arbitrary non-privileged port
numSend = 2      # Number of times code should be transmitted

class Broadcast(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.code = sys.argv[1]                                     # Get command line arguments as code

    def run(self):
        HOST = hostIP
        PORT = hostPort
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)     # UDP, AF_INET - address family IPv4, SOCK_DGRAM - datagram
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allows a socket to bind to an address already in use
        #sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)  # Allows a socket to bind to a port already in use, not needed in windows
        sock.bind((HOST, 0))                                     # Bind to host IP and arbitrary port
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Send broadcast data
        for x in range(numSend):                                        # Broadcast code numSend number of times
            print("Sending...")
            sock.sendto(str.encode(self.code), ('<broadcast>', PORT))   # Send data on port: PORT




thread1 = Broadcast()
thread1.start()