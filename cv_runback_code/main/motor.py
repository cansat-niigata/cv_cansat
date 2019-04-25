import wiringpi
import time


class Motor:

    def __init__(self, left=(2, 3), right=(14, 15)):
        # GPIO config
        self.pins = [left[0], left[1], right[0], right[1]]

        self.direction = {'right': (1, 0, 1, 1),
                          'left': (1, 1, 1, 0),
                          'straight': (1, 0, 1, 0)}

        # GPIO output mode to 1
        wiringpi.wiringPiSetupGpio()
        for pin in self.pins:
            wiringpi.pinMode(pin, 1)

    def stop(self):
        for pin in self.pins:
            wiringpi.digitalWrite(pin, 0)
        time.sleep(0.1)

    def move2motors(self, direction, second):
        print('direction: {}, second: {}s'.format(direction, second))
        for i, pin in enumerate(self.pins):
            wiringpi.digitalWrite(pin, self.direction[direction][i])
        time.sleep(second)

    def to_center(self, x1, x2, half_w, direction, second):
        print('red corn set to center')
        dx = x1 - x2
        second *= (half_w - dx) / dx
        self.move2motors(direction, second)


if __name__ == '__main__':
    motor = Motor(left=(2, 3), right=(14, 15))
    motor.move2motors(direction='straight', second=5)
    motor.move2motors(direction='right', second=2)
    motor.move2motors(direction='left', second=2)
