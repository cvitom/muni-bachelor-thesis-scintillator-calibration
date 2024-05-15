import numpy as np
import matplotlib.pyplot as plt

class RotaryCalibGUI:
    def __init__(self):
        self.calib_line = None

    def start(self):
        plt.figure()
        plt.title("Rotary calibration")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Integrated counts")
        self.calib_line, = plt.plot([], [], 'ro-')
        plt.show()
    
    def add_point(self, angle, counts):
        x = self.calib_line.get_xdata().tolist()
        y = self.calib_line.get_ydata().tolist()
        x.append(angle)
        y.append(counts)
        self.calib_line.set_xdata(x)
        self.calib_line.set_ydata(y)
        plt.draw()
        plt.pause(0.01)
    
    