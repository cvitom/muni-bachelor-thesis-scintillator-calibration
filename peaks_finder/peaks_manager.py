import numpy as np
import pandas as pd
import os
from dataclasses import dataclass

@dataclass
class PeakInfo:
    dir: str
    isotope: str
    temp: int
    temp_avg: float
    temp_std: float
    exp_time: float
    ch: int
    hv: int
    energy: float
    channel: float
    sigma: float
    amplitude: float
    channel_err: float

class PeaksManager:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.df = pd.DataFrame({})
        if os.path.exists(output_file) and os.path.isfile(output_file):
            self.df = pd.read_csv(output_file)
        else:
            self.df = pd.DataFrame(columns=['dir', 'isotope', 'temp', 'temp_avg', 'temp_std', 'exp_time', 'ch', 'hv', 'energy', 'channel', 'sigma', 'amplitude', 'channel_err'])
        
    def add_peak(self, peak_info: PeakInfo):
        # check if peak already exists
        peak_idx = self.df[(self.df['dir'] == peak_info.dir) & (self.df['isotope'] == peak_info.isotope) & (self.df['temp'] == peak_info.temp) & (self.df['ch'] == peak_info.ch) & (self.df['hv'] == peak_info.hv) & (self.df['energy'] == peak_info.energy)].index.values
        if peak_idx.size > 0:
            self.df.loc[peak_idx[0], 'channel'] = peak_info.channel
            self.df.loc[peak_idx[0], 'sigma'] = peak_info.sigma
            self.df.loc[peak_idx[0], 'amplitude'] = peak_info.amplitude
            self.df.loc[peak_idx[0], 'channel_err'] = peak_info.channel_err
        else:
            self.df = self.df.append(peak_info.__dict__, ignore_index=True)
        self.df.to_csv(self.output_file, index=False)
    
    
    