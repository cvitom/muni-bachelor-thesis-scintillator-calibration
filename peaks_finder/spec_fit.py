import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, optimize
from typing import Tuple, List

def find_peaks_in_file(filename: str, peaks_approx: np.ndarray, fit_halfwidth=2) -> np.ndarray:
    print(f'Processing {filename}')
    df = pd.read_csv(filename, sep=' ', header=None)
    energy_meas = df.values[:, 0]
    counts_meas = df.values[:, 1]

    meas_peaks = []
    approx_peak_frame = 2
    for peak in peaks_approx:
        meas_peaks.append(np.argmax(counts_meas[peak - approx_peak_frame : peak + approx_peak_frame]) + peak - approx_peak_frame)

    def gauss(x, max, mu, sigma, c):
        return (max - c) * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

    def gauss_fit(x, y, peak, width=2):
        p0 = [x[peak], 1, 0]
        popt, pcov = optimize.curve_fit(
            lambda x, mu, sigma, c: gauss(x, y[peak], mu, sigma, c), 
            x[peak - width : peak + width + 1], 
            y[peak - width : peak + width + 1], 
            p0=p0)
        return popt

    peak_fits = [gauss_fit(energy_meas, counts_meas, peak, fit_halfwidth) for peak in meas_peaks]
    peak_fits_simx = [np.linspace(peak[0] - fit_halfwidth, peak[0] + fit_halfwidth, 100) for peak in peak_fits]
    peak_fits_simy = [gauss(peak_fits_simx[i], counts_meas[meas_peaks[i]], *peak_fits[i]) for i in range(len(peak_fits))]

    plt.figure()
    plt.yscale('log')
    plt.bar(energy_meas, counts_meas, width=1, label='measurement')
    plt.plot(energy_meas[meas_peaks], counts_meas[meas_peaks], 'x', label='measured peaks')
    for i in range(len(peak_fits)):
        plt.plot(peak_fits_simx[i], peak_fits_simy[i], '--', label=f'fit {i}')
    plt.show()

    return np.array([peak[0] for peak in peak_fits])