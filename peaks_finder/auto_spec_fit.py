import numpy as np
from matplotlib import pyplot as plt
from .spec_manager import load_spec, SpecInfo, get_file_path
from .peaks_manager import PeakInfo
from typing import List
from scipy.optimize import curve_fit
import os

def e2ch(ch: int | np.ndarray, calibration: np.ndarray, t: float, hv: float) -> float | np.ndarray:
    if calibration.shape[0] == 4:
        ch0 = calibration[0]
        k = calibration[1] + calibration[2] * hv + calibration[3] * t
        return ch0 + k * ch
    elif calibration.shape[0] == 7:
        ch0 = calibration[0]
        k = calibration[1] + calibration[2] * hv + calibration[3] * t + calibration[4] * hv**2 + calibration[5] * hv * t + calibration[6] * t**2
        return ch0 + k * ch
    else:
        raise ValueError('Invalid calibration matrix shape')

class AutoSpecFit:
    def __init__(self, calibration: np.ndarray):
        self.model_line = None
        if calibration.shape[0] == 4:
            self.calibration = calibration
            self.calib_mode = 'linear'
        elif calibration.shape[0] == 7:
            self.calibration = calibration
            self.calib_mode = 'quadratic'
        else:
            raise ValueError('Invalid calibration matrix shape')
        

    def gauss(x, max, mu, sigma, c):
        return (max - c) * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c
    
    def spec_model_function(self, channels, *peak_args, disable_plot=False):
        peak_args = np.array(peak_args)
        peak_args = peak_args.reshape(-1, 3)
        result = np.sum([AutoSpecFit.gauss(channels, peak[0], peak[1], peak[2], 0) for peak in peak_args], axis=0)
        # if self.model_line is not None and not disable_plot:
        #     self.model_line.set_ydata(self.spec_model_function(self.model_line.get_xdata(), *peak_args, disable_plot=True))
        # plt.draw()
        # plt.pause(0.01)
        # print("hi")
        return result

    def auto_spec_fit(self, directory: str, isotope: str, t: float, ch: int, hv: int, energies: np.ndarray) -> List[PeakInfo]:
        filepath = get_file_path(isotope, t, ch, hv, directory)
        print(filepath)
        spec_info = load_spec(isotope, t, ch, hv, directory)
        spec = spec_info.spec
        temp = spec_info.temp_avg
        energies = [0, *energies]
        energies = np.array(energies)


        channels = np.arange(len(spec))
        expected_peak_channels = e2ch(energies, self.calibration, temp, hv)
        for i in range(len(energies)):
            if expected_peak_channels[i] < 46 or expected_peak_channels[i] > 150:
                energies[i] = np.nan
                expected_peak_channels[i] = np.nan
        energies = energies[~np.isnan(energies)]
        expected_peak_channels = expected_peak_channels[~np.isnan(expected_peak_channels)]

        peak_args = np.array([np.ones_like(expected_peak_channels), expected_peak_channels, np.ones_like(expected_peak_channels)]).T.flatten()

        # Create fit bounds
        lower_bounds = np.zeros_like(peak_args)
        upper_bounds = np.ones_like(peak_args) * np.inf
        for i in range(len(energies)):
            lower_bounds[3 * i + 1] = expected_peak_channels[i] - 3
            upper_bounds[3 * i + 1] = expected_peak_channels[i] + 3
            peak_args[3 * i] = spec[int(expected_peak_channels[i])]

        if energies.shape[0] == 0:
            print(f"No peaks found for {filepath}")
            return []

        plt.figure()
        plt.title(filepath)
        model_channels = np.linspace(0, len(spec), 1000)
        plt.ylim([1, np.max(spec) * 5])
        plt.xlabel('Channel')
        plt.ylabel('Counts/second')
        # plt.show(block=False)
        popt, pcov = curve_fit(self.spec_model_function, channels[44:], spec[44:], p0=peak_args, bounds=(lower_bounds, upper_bounds))
        model_spec = self.spec_model_function(model_channels, *popt)
        spec_line = plt.step(channels + 0.5, spec, label='measurement')[0]
        self.model_line = plt.plot(model_channels, model_spec, '--', label='model')[0]
        plt.yscale('log')
        plot_dir = f"plots/{directory}/{isotope}"
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/t_{t}_ch_{ch}_hv_{hv}.png")
        # plt.show(block=False)
        # plt.pause(1)
        plt.close()
        results = []
        for i in range(len(energies)):
            max, mu, sigma = popt[3 * i:3 * i + 3]
            mu_err = np.sqrt(pcov[3 * i + 1, 3 * i + 1])
            results.append(PeakInfo(directory, isotope, t, spec_info.temp_avg, spec_info.temp_std, spec_info.exp_time,
                                    ch, hv, energies[i], mu, sigma, max, mu_err))
        return results



if __name__ == '__main__':
    asf = AutoSpecFit()
    asf.auto_spec_fit(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]))