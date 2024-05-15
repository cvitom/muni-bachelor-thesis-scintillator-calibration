import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import PickEvent, KeyEvent
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import pandas as pd
from scipy import signal, optimize
from argparse import ArgumentParser
import os
from typing import List, Callable
from .spec_manager import load_spec, get_file_dir, get_file_path
from . import spec_manager as sm
from typing import List, Tuple
import math
from .peaks_manager import PeaksManager, PeakInfo
from .auto_spec_fit import AutoSpecFit

def gauss(x, max, mu, sigma, c):
    return (max - c) * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

class PeakPicker:
    
    def __init__(self):
        self.x: np.ndarray = None
        self.counts: np.ndarray = None
        self.spec_info: sm.SpecInfo = None

        self.fig, (ax, ax_fit) = plt.subplots(2, 1)
        plt.subplots_adjust(hspace=0.5)
        self.ax: Axes = ax
        self.ax_fit: Axes = ax_fit
        self.ax.set_xlim(30, 255)
        self.ax.set_yscale('log')
        self.ax.set_xlabel('Channel')
        self.ax.set_ylabel('Counts')
        self.ax_fit.set_yscale('log')
        self.ax_fit.set_xlabel('Channel')
        self.ax_fit.set_ylabel('Counts')
        self.selected_peak_element: Line2D = None
        self.selected_peak_sigma = None
        self.selected_peak_idx = None
        self.fit_with_background = True
        self.energies = []
        self.energy_fit_args: np.ndarray = None # (amplitude, center, sigma, c, fit_halfwidth)
        self.AMPL_IDX = 0
        self.CENTER_IDX = 1
        self.SIGMA_IDX = 2
        self.C_IDX = 3
        self.FIT_HWIDTH_IDX = 4
        self.energy_fit_errors: np.ndarray = None
        self.DEFAULT_FIT_HWIDTH = 3
        self.energy_channels: np.ndarray = None
        self.energy_sigmas = []
        self.energy_peak_idxs = []
        # Indexes of all peaks found in the spectrum (= channel numbers, since the spectrum is 256 channels wide starting from 0)
        self.peak_channels: np.ndarray = None
        self._should_exit = False

        self.selected_energy_idx = None
        self.main_ax_fit_element: Line2D = None
        self.fit_width = 3
        self.x = np.arange(0, 256)

        self.dir = None
        self.isotope = None
        self.temp = None
        self.channel = None
        self.hv = None

    
    def peaks_func(self, channels: np.ndarray, peak_args: np.ndarray):
        ampl = peak_args[:, self.AMPL_IDX]
        nan_mask = np.logical_not(np.isnan(ampl))
        ampl = ampl[nan_mask]
        center = peak_args[:, self.CENTER_IDX][nan_mask]
        sigma = peak_args[:, self.SIGMA_IDX][nan_mask]
        c = peak_args[:, self.C_IDX][nan_mask]

        # compute raw peaks (without background, given by c)
        spec = np.zeros(len(channels))
        for i in range(len(ampl)):
            peak = gauss(channels, ampl[i], center[i], sigma[i], 0)
            spec += peak
        
        # add interpolated background
        if self.fit_with_background:
            background = np.interp(channels, center, c)
            return spec + background
        else:
            return spec

    def find_peaks(self):
        self.peak_channels, self.peak_props = signal.find_peaks(self.counts, prominence=0, width=0.6)

    def run_picker(self):
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        plt.show(block=False)

    def export_peak_fits(self) -> List[PeakInfo]:
        peak_fits = []
        for i in range(len(self.energies)):
            if np.isnan(self.energy_channels[i]):
                continue
            
            peak_fit = PeakInfo(
                dir=self.dir,
                isotope=self.isotope,
                temp=self.temp,
                ch=self.channel,
                hv=self.hv,
                energy=self.energies[i],
                channel=self.energy_fit_args[i, self.CENTER_IDX],
                sigma=self.energy_fit_args[i, self.SIGMA_IDX],
                amplitude=self.energy_fit_args[i, self.AMPL_IDX],
                channel_err=self.energy_fit_errors[i, self.CENTER_IDX],
                temp_avg=self.spec_info.temp_avg,
                temp_std=self.spec_info.temp_std,
                exp_time=self.spec_info.exp_time
            )
            peak_fits.append(peak_fit)
        
        return peak_fits

    def pick(self, dir: str, isotope: str, temp: int, channel: int, hv: int, energies: np.ndarray) -> List[PeakInfo]:
        self._should_exit = False
        
        self.dir = dir
        self.isotope = isotope
        self.temp = temp
        self.channel = channel
        self.hv = hv

        file = sm.get_file_path(isotope, temp, channel, hv, dir)
        self.spec_info = sm.load_spec(isotope, temp, channel, hv, dir)
        self.counts = self.spec_info.spec
        self.energies = energies
        self.energy_channels = np.nan * np.zeros(len(energies))
        self.energy_fit_args = np.nan * np.zeros((len(energies), 5))
        self.energy_fit_errors = np.nan * np.zeros((len(energies), 4))
        self.energy_peak_idxs = np.zeros(len(energies)) * np.nan
        self.selected_energy_idx = None
        self.find_peaks()
        self.fig.suptitle(file)
        self.ax.step(self.x + 0.5, self.counts, label='measurement')
        self.ax.plot(self.x[self.peak_channels], self.counts[self.peak_channels], 'x', color='red', label='peaks', picker=5)
        self.ax.set_ylim(1, None)
        self.ax.set_yscale('log')
        self.ax_fit.set_title("Test title")
        self.ax_fit.set_yscale('log')
        self.ax_fit.set_xlabel('Channel')
        self.ax_fit.set_ylabel('Counts')

        while self._should_exit is False:
            plt.pause(0.1)
        
        return self.export_peak_fits()
        

    @property
    def selected_peak_channel(self):
        if self.selected_peak_idx is None:
            return None
        return self.peak_channels[self.selected_peak_idx]

    def on_pick(self, event: PickEvent):
        peak_idx = event.ind[0]
        self.selected_peak_idx = peak_idx
        if peak_idx in self.energy_peak_idxs:
            self.selected_energy_idx = np.nanargmin(np.abs(self.energy_peak_idxs - peak_idx))
        else:
            self.selected_energy_idx = None
        
        self.update_peak_view()


    def update_peak_view(self):
        peak_idx = self.selected_peak_idx
        center_channel = self.x[self.peak_channels[peak_idx]]
        data_filter = np.arange(np.max((center_channel - 10, 0)), np.min((center_channel + 10, 255)))

        if self.selected_peak_element is not None:
            self.selected_peak_element.remove()
        self.selected_peak_element = self.ax.plot(center_channel, self.counts[self.peak_channels[peak_idx]], 'o', color='green', label='selected peak')[0]

        xmin = center_channel - 10
        xmax = center_channel + 10
        if self.selected_energy_idx is not None:
            hwidth = self.energy_fit_args[self.selected_energy_idx, self.FIT_HWIDTH_IDX]
            xmin = center_channel - hwidth - 2
            xmax = center_channel + hwidth + 2

        self.ax_fit.clear()
        self.ax_fit.set_yscale('log')
        self.ax_fit.set_yscale('log')
        self.ax_fit.set_xlabel('Channel')
        self.ax_fit.set_ylabel('Counts')
        self.ax_fit.step(self.x + 0.5, self.counts, label='measurement')
        self.ax_fit.set_xlim(xmin, xmax)
        self.ax_fit.plot(self.x[self.peak_channels], self.counts[self.peak_channels], 'x', color='red', label='peaks')
        if self.selected_energy_idx is not None:
            fit_x = np.linspace(0, 255, 10000)
            fit_y = self.peaks_func(fit_x, self.energy_fit_args)
            self.ax_fit.plot(fit_x, fit_y, '--', color='orange', label='fit')
            if self.main_ax_fit_element is None:
                self.main_ax_fit_element = self.ax.plot(fit_x, fit_y, '--', color='orange', label='fit')[0]
            else:
                self.main_ax_fit_element.set_xdata(fit_x)
                self.main_ax_fit_element.set_ydata(fit_y)
        ymin = np.max((self.counts[int(xmin):int(xmax)].min(), 0))
        ymax = self.counts[int(xmin):int(xmax)].max() * 1.1
        self.ax_fit.set_ylim(ymin, ymax)
        if self.selected_energy_idx is not None:
            self.ax_fit.set_title(f'Energy: {self.energies[self.selected_energy_idx]} kEv')
        else:
            self.ax_fit.set_title('No energy selected')

        self.fig.canvas.draw()


    def on_enter(self):
        self.counts = None
        self.ax.clear()
        self.ax_fit.clear()
        self.fig.canvas.draw()
        self.main_ax_fit_element = None
        self._should_exit = True

    def on_close(self, event):
        self.on_enter()

    def on_key_press(self, event: KeyEvent):
        if event.key == 'escape':
            self.cancel_pick()
        elif event.key == 'enter':
            self.on_enter()
        elif event.key == '+':
            if self.selected_energy_idx is not None:
                self.energy_fit_args[self.selected_energy_idx, self.FIT_HWIDTH_IDX] += 1
                self.fit_peaks()
                self.update_peak_view()
                print(f'Fit width: {self.energy_fit_args[self.selected_energy_idx, self.FIT_HWIDTH_IDX]}')
            
        elif event.key == '-' and self.fit_width > 2:
            if self.selected_energy_idx is not None:
                self.energy_fit_args[self.selected_energy_idx, self.FIT_HWIDTH_IDX] -= 1
                self.fit_peaks()
                self.update_peak_view()
                print(f'Fit width: {self.energy_fit_args[self.selected_energy_idx, self.FIT_HWIDTH_IDX]}')

        elif event.key == 'n':
            self.cancel_pick()
        elif event.key.isdigit():
            self.update_selected_energy(int(event.key))
        elif event.key == 'q':
            self.close()
            exit(0)
        elif event.key == 'b':
            self.fit_with_background = not self.fit_with_background
            self.fit_peaks()
            self.update_peak_view()

    def update_selected_energy(self, selected_energy_key: int):
        if selected_energy_key == 0:
            if self.selected_energy_idx is not None:
                self.energy_channels[self.selected_energy_idx] = None
                self.energy_fit_args[self.selected_energy_idx, :] = np.nan
                self.energy_peak_idxs[self.selected_energy_idx] = None
                
            self.selected_energy_idx = None
        elif selected_energy_key < len(self.energies):
            self.selected_energy_idx = selected_energy_key - 1
        else:
            self.selected_energy_idx = len(self.energies) - 1

        if self.selected_energy_idx is not None:
            self.energy_channels[self.selected_energy_idx] = self.selected_peak_channel
            self.energy_peak_idxs[self.selected_energy_idx] = self.selected_peak_idx
            if np.isnan(self.energy_fit_args[self.selected_energy_idx, self.FIT_HWIDTH_IDX]):
                self.energy_fit_args[self.selected_energy_idx, self.FIT_HWIDTH_IDX] = self.DEFAULT_FIT_HWIDTH
        

        self.fit_peaks()
        self.update_peak_view()

    def cancel_pick(self):
        if self.selected_peak_element is not None:
            self.selected_peak_element.remove()
        self.selected_peak_sigma = None
        self.selected_peak_element = None
        self.ax_fit.clear()
        self.fig.canvas.draw()
    
    def create_fit_mask(self):
        # initialize local variables
        selected_channels = self.energy_channels
        nan_mask = np.logical_not(np.isnan(selected_channels))
        selected_channels = self.energy_channels[nan_mask]
        fit_hwidths = self.energy_fit_args[:, self.FIT_HWIDTH_IDX][nan_mask]

        # create fit mask
        peak_fit_mask = np.zeros(len(self.x), dtype=bool)
        for i in range(len(selected_channels)):
            x_min = int(np.max((selected_channels[i] - fit_hwidths[i], 0)))
            x_max = int(np.min((selected_channels[i] + fit_hwidths[i], 255)))
            peak_fit_mask = np.logical_or(peak_fit_mask, np.logical_and(self.x >= x_min, self.x <= x_max))

        return peak_fit_mask

    def fit_peaks(self):
        # initialize local variables
        selected_channels = self.energy_channels
        nan_mask = np.logical_not(np.isnan(selected_channels))
        selected_channels = np.array(selected_channels[nan_mask], dtype=int)

        # create fit mask
        peak_fit_mask = self.create_fit_mask()

        # create initial fit args
        init_fit_args = np.zeros((len(selected_channels), 4))
        init_fit_args[:, self.AMPL_IDX] = self.counts[selected_channels]
        init_fit_args[:, self.CENTER_IDX] = selected_channels
        init_fit_args[:, self.SIGMA_IDX] = 2
        init_fit_args[:, self.C_IDX] = 0
 
        # create fit bounds
        fit_bounds_lower = (0, 0, 0, 0) * len(selected_channels)
        fit_bounds_upper = (np.inf, 255, np.inf, np.inf) * len(selected_channels)
        fit_bounds = (fit_bounds_lower, fit_bounds_upper)

        # fit
        fit_func = lambda x, *args: self.peaks_func(x, np.reshape(args, (len(selected_channels), 4)))[peak_fit_mask]
        popt, pcov = optimize.curve_fit(fit_func, self.x, self.counts[peak_fit_mask], p0=init_fit_args.flatten(), bounds=fit_bounds)

        # update fit args
        self.energy_fit_args[nan_mask, :4] = popt.reshape((len(selected_channels), 4))
        print("Fit args: ", self.energy_fit_args)

        # update fit errors
        self.energy_fit_errors[nan_mask, :] = np.sqrt(np.diag(pcov)).reshape((len(selected_channels), 4))

        return self.energy_fit_args

    def close(self):
        plt.close(self.fig)
    
isotope_energies = {
    'am241': [26.3, 59.5], # https://www.gammaspectacular.com/blue/am241-spectrum
    'na22': [511],
    'co60': [75, 209.8],
    'co57': [46, 122],
    'ba133': [31, 81, 356],
    'eu152': [39.9, 121, 244, 344],
    'cs137': [32, 180, 662],
    'ra226': [77, 609]
}


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.description = 'Peak picker'
    argparser.add_argument('-e', '--energies', help='Isotope line energy', type=float, nargs='+')
    argparser.add_argument('-o', '--output', help='Output file name', default='peaks.csv', type=str)
    argparser.add_argument('-ch', '--channel', help='Channel number (0, 1, or 2 (means both)). Both by default', type=int, default=2)
    argparser.add_argument('-t', '--temp', help="Temperature (don't specify for every available temp)", type=float)
    argparser.add_argument('-hv', '--hv', help="High voltage (don't specify for every available HV)", type=int)
    argparser.add_argument('--hv-from', help="HV from", type=int)
    argparser.add_argument('--hv-to', help="HV to", type=int)
    argparser.add_argument('-is', '--isotope', help='Isotope', type=str, required=True)
    argparser.add_argument('-d', '--dir', help='Directory with spectra', type=str, default='data')
    argparser.add_argument('-a', '--auto', help='Auto mode', action='store_true')
    argparser.add_argument('-c', '--calib', help='Calibration file', type=str)
    args = argparser.parse_args()

    energies = args.energies
    output = args.output
    channel = args.channel
    temp = args.temp
    preset_hv = args.hv
    hv_from = args.hv_from
    hv_to = args.hv_to
    isotope = args.isotope
    directory = args.dir
    auto_mode = args.auto
    calib_file = args.calib
    
    if energies is None:
        energies = isotope_energies.get(isotope)
        if energies is None:
            print(f"Unknown isotope {isotope}")
            exit(1)

    peak_channels = []
    peak_sigmas = []
    peak_hvs = []

    if calib_file is not None:
        calib = np.loadtxt(calib_file)

    if auto_mode:
        asf = AutoSpecFit(calib)
    else:
        picker = PeakPicker()
        picker.run_picker()

    temps = []
    if temp is None:
        izotope_dir = os.path.join(directory, isotope)
        temp_filter: Callable[[str], bool] = lambda s: s.startswith('t_')
        temp_dirs = filter(temp_filter, os.listdir(izotope_dir))
        temps = [float(temp_dir[2:]) for temp_dir in temp_dirs]
    else:
        temps = [temp]
    temps.sort()

    peaks_manager = PeaksManager(output)

    for t in temps:
        channels = [channel] if channel != 2 else [0, 1] 
        for ch in channels:
            data_dir = get_file_dir(isotope, t, ch, directory)
            hvs = [preset_hv]

            if preset_hv is None:
                hv_filter: Callable[[str], bool] = lambda s: s.startswith('hv_')
                hv_files = filter(hv_filter, os.listdir(data_dir))
                hvs = np.array([int(hv_file[3:-4]) for hv_file in hv_files])
                if hv_from is not None:
                    hvs = hvs[hvs >= hv_from]
                if hv_to is not None:
                    hvs = hvs[hvs <= hv_to]
            
            hvs.sort()
            for hv in hvs:
                spec = load_spec(isotope, t, ch, hv, directory)
                if auto_mode:
                    peak_fits = asf.auto_spec_fit(directory, isotope, t, ch, hv, energies)
                else:
                    peak_fits = picker.pick(directory, isotope, t, ch, hv, energies)
                for peak_fit in peak_fits:
                    peaks_manager.add_peak(peak_fit)
    
    if not auto_mode:
        picker.close()
    print("Congratulations! You have successfully finished the spectra fitting. Go give yourself an icecream (or two)!")
