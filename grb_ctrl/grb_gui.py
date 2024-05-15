import os

import matplotlib.animation
os.environ["QT_LOGGING_RULES"] = '*.debug=false'

from matplotlib import pyplot as plt
import numpy as np
from time import time
import matplotlib


class GrbGui:
    def __init__(self):
        self.temp_fig = plt.figure()
        self.temp_ax = self.temp_fig.add_subplot(211)
        self.temp_data = np.nan * np.zeros((2, 200))
        self.temp_line = self.temp_ax.plot(self.temp_data[0, :], self.temp_data[1, :])[0]
        self.target_temp_line = None
        self.target_temp_line_limit_1 = None
        self.target_temp_line_limit_2 = None
        self.target_temp = None
        # self.exp_fig = plt.figure()
        self.exp_ax = self.temp_fig.add_subplot(212)
        self.exp_data = np.zeros(256)
        self.exp_line = self.exp_ax.step(np.arange(256), self.exp_data)[0]
        self.exp_ax.set_yscale('log')
        self.spec_data = np.zeros(256)
        
    def show(self):
        # matplotlib.use("Qt5Agg")
        self.anim = matplotlib.animation.FuncAnimation(self.temp_fig, self._update, interval=1000)
        plt.show()

    def pause(self, time: float):
        plt.pause(time)

    def update_temperature(self, temp: float):
        self.temp_data = np.roll(self.temp_data, 1, axis=1)
        self.temp_data[0, 0] = time()
        self.temp_data[1, 0] = temp
        
    def _update(self, i):
        self.temp_line.set_xdata(self.temp_data[0, :] - time())
        self.temp_line.set_ydata(self.temp_data[1, :])
        self.temp_ax.relim()
        self.temp_ax.autoscale_view()

        if self.target_temp is None and self.target_temp_line is not None:
            self.target_temp_line.remove()
            self.target_temp_line = None
            self.target_temp_line_limit_1.remove()
            self.target_temp_line_limit_1 = None
            self.target_temp_line_limit_2.remove()
            self.target_temp_line_limit_2 = None
        elif self.target_temp is not None:
            if self.target_temp_line is None:
                self.target_temp_line = self.temp_ax.axhline(self.target_temp, color='g', linestyle='--', linewidth=2)
                self.target_temp_line_limit_1 = self.temp_ax.axhline(self.target_temp + 0.4, color='r', linestyle='--', linewidth=2)
                self.target_temp_line_limit_2 = self.temp_ax.axhline(self.target_temp - 0.4, color='r', linestyle='--', linewidth=2)
            else:
                self.target_temp_line.set_ydata(self.target_temp)
                self.target_temp_line_limit_1.set_ydata(self.target_temp + 0.4)
                self.target_temp_line_limit_2.set_ydata(self.target_temp - 0.4)
        if self.spec_data is not None:
            self.exp_line.set_ydata(self.spec_data)
            self.exp_ax.relim()
            if self.spec_data.max() > 0:
                self.exp_ax.autoscale_view()

    def update_target_temperature(self, temp: float):
        self.target_temp = temp

    def update_spectrum(self, spectrum):
        self.spec_data = spectrum
        
