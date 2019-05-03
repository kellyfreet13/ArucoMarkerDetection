import socket
import threading
import serial
import array
from struct import pack
from time import sleep

portLocation = '/dev/ttyACM0'

distance = 0
offset = 0
sensor1 = 0
sensor2 = 0
sensor3 = 0

serialSem = threading.Semaphore()
printSem = threading.Semaphore()
ser = serial.Serial(port=portLocation, baudrate=115200)


class codeListener(threading.Thread):
    # declare as global to share between threads
    global offset
    global distance
    global QR_code
    global condition

    def __init__(self, s, p, c):
        threading.Thread.__init__(self)
        self.code = None
        serialSem = s
        printSem = p
        condition = c

    def payload(self, code, distance, offset, sensor1, sensor2, sensor3):
        ch = 0
        length = 12
        pay = pack('hHHHHH', code, distance, offset, sensor1, sensor2, sensor3)
        a = array.array('b', pay)
        a = a.tobytes()

        for i in range(0, 12):
            ch = (ch + a[i]) % 256

        pay = pack('BBB', 67, 79, 12) + pay + pack('B', ch)
        return pay

    def run(self):
        print("\nCode listener running\n")
        self.recvBroadcastCode()
        self.readSerial()


    def recvBroadcastCode(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)             # UDP socket
        sock.bind(('', 2000))                                               # Bind to localhost on port 2000

        data, senderAddr = sock.recvfrom(1500, 0)                           # Receive broadcast, buffer size 1500
        self.code = int( data.decode('UTF-8') )                                   # Save code
        print("\nBroadcast code received: ", self.code)

        condition.acquire()                                                 # Write QR_code to global
        QR_code = self.code
        condition.release()

        self.writeCode()

    def writeCode(self):
        codePay = self.payload(self.code,0,0,0,0,0)                                      # Pack code as unsigned short
        serialSem.acquire()                                                       # Acquire semaphore lock
        ser.write(codePay)                                                  # Write code to serial
        serialSem.release()                                                       # Release semaphore lock
        print("\nBroadcast code written to serial")

    def readSerial(self):
        msgRecv = b''                                                      # Initialize msgRecv as b'0'
        while(msgRecv == b''):                                             # Read serial until a message is received
            serialSem.acquire()
            msgRecv = ser.read_all()                                        # Read from serial and store it in msgRecv
            serialSem.release()
            if(msgRecv != b''):
                if(msgRecv.find(b'Sensor Reading') != -1):
                    print(msgRecv.find(b'Sensor Reading'))
                    print("\nReceived: ", msgRecv)
                    break
        self.writeDistanceOffset()                                      # Call writeDistanceOffset


    def writeDistanceOffset(self):
        while(distance and offset == 0):                                # Wait to receive distance and offset from aruco thread
            sleep(.5)

        # sensor1 = input("Sensor1: ")
        # sensor2 = input("Sensor2: ")
        # sensor3 = input("Sensor3: ")
        printSem.acquire()
        print("Inputs:\t", distance, offset)
        printSem.release()
        pay = self.payload(self.code, int(distance), int(offset), sensor1, sensor2, sensor3)
        print("Writing payload to serial...")  # print payload
        serialSem.acquire()  # get semaphore lock
        ser.write(pay)  # write payload to serial port
        serialSem.release()  # release semaphore lock
