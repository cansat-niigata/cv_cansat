import camera
from motor import Motor
import numpy as np

motor = Motor(left=(2, 3), right=(14, 15))

current = 0
while True:
    x1, half_w, flag = camera.camera_operation(num=current)
    print(flag)
    if flag == 'ARRIVED':
        break
    else:
        if x1 is None:
            motor.move2motors(direction='right', second=2)
        else:
            if np.abs(x1) > half_w + 30:
                if x1 > half_w + 30:
                    direction = 'right'
                elif x1 < half_w - 30:
                    direction = 'left'

                motor.move2motors(direction, second=0.5)    # cansat rotate a little,
                x2, _ = camera.camera_operation()           # then, check how coordinate was change,
                motor.to_center(x1, x2, half_w, direction)  # then, red corn set to center.

            motor.move2motors(direction='straight', second=10)    # go straight

            current += 1

print('GOAL!!')
