import serial
import time

# raspi                xbee
# 1(3.3V)           ---> 1(3.3V)
# 6(GND)          ---> 10(GND)
# 8(TXD / GPIO14) ---> 3(RX)
# 10(RXD / GPIO15)---> 2(TX)

# When use GPIO: '/dev/serial0'
#          USB-serial: '/dev/ttyUSB0'
ser = serial.Serial('/dev/serial0', 9600)
print(ser.portstr)

for i in range(100):
    data = "hello"
    ser.write(bytes(data, "UTF-8"))

ser.close()