import serial
import micropyGPS
import threading
import time

time_zone = 9
O_format = 'dd'
gps = micropyGPS.MicropyGPS(time_zone, O_format)

def rungps():
    s = serial.Serial('/dev/serial0', 9600, timeout=10)
    s.readline() 
    while True:
        sentence = s.readline().decode('utf-8')
        if sentence[0] != '$':
            continue
        for x in sentence:
            gps.update(x)

gpsthread = threading.Thread(target=rungps, daemon=True)
gpsthread.start()

while True:
    if gps.clean_sentences > 20:
        h = gps.timestamp[0] if gps.timestamp[0] < 24 else gps.timestamp[0] - 24
        print('%2d:%02d:%04.1f' % (h, gps.timestamp[1], gps.timestamp[2]))
        print('緯度経度: %2.8f, %2.8f\n' % (gps.latitude[0], gps.longitude[0]))
        time.sleep(3.0)