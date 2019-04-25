import wiringpi
import time

pin = 14
read = 0
write = 1

wiringpi.wiringPiSetupGpio()
wiringpi.pinMode(pin, read)

for i in range(100):
    a = wiringpi.digitalRead(pin)
    print('{}: {}'.format(i, a))
    time.sleep(0.1)