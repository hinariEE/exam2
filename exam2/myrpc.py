import serial
import time
serdev = '/dev/ttyACM0'                # use the device name you get from `ls /dev/ttyACM*`
s = serial.Serial(serdev, 9600)

s.write(bytes("/clarify/run\r", 'UTF-8'))

s.close()
