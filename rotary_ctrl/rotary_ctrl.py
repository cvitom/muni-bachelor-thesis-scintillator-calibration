from thorlabs_apt_device import TDC001
from time import sleep, time
from grb_ctrl.grb_ctrl import GrbCtrl
import numpy as np
class RotaryCtrl:
    def __init__(self):
        self.tdc = TDC001("/dev/ttyUSB4")
    
    def init(self):
        self.tdc.identify()
        self.tdc.home()
        while self.tdc.status['homing']:
            sleep(0.1)
        self.steps_per_degree = self.tdc.genmoveparams['backlash_distance']
    
    def move_to(self, angle: float):
        target_raw = int(angle * self.steps_per_degree)
        self.tdc.move_absolute(target_raw)
        move_start_time = time()
        sleep(0.1)
        while np.abs(self.get_position_raw() - target_raw) > 10:
            sleep(0.1)
            if time() - move_start_time > 60:
                print("Move timeout")
                # self.tdc = TDC001("/dev/ttyUSB4")
                self.init()
                sleep(1)
                self.move_to(angle)
                break
    
    def get_position(self):
        return self.get_position_raw() / self.steps_per_degree

    def get_position_raw(self):
        return self.tdc.status.get('position')

if __name__ == "__main__":
    print("Connecting to Thorlabs TDC001...")
    tdc = TDC001()
    tdc.identify()
    print("Homing...")
    tdc.home()
    while tdc.status['homing']:
        sleep(0.1)
    steps_per_degree = tdc.genmoveparams['backlash_distance']

    center_angle_approx = 50

    print(f"Moving to position {center_angle_approx}...")
    tdc.move_absolute(center_angle_approx * steps_per_degree)
    sleep(0.1)
    while tdc.status.get('position') != -720 * steps_per_degree:
        sleep(0.1)

    print("Done")
    