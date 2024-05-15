import numpy as np
import pandas as pd
import os
from .spectrum_manager import SpectrumManager

if __name__ == "__main__":
    files = os.listdir("flight_data_orig")
    for file in files:
        if not file.endswith(".txt"):
            continue
        info_parts = file.split("-")
        isotope = info_parts[0].lower()
        channel = 0 if info_parts[1].lower() == 'ch0' else 1
        hv = int(info_parts[2].replace("hv", ""))
        exp_time = int(info_parts[3].split('.')[0].replace("t", "").replace("s", ""))
        spec_mgr = SpectrumManager('flight-data', isotope)
        
        spec_df = pd.read_csv(f"flight_data_orig/{file}", delim_whitespace=True, header=None)
        spec = spec_df.values[:, 1]
        spec_mgr.save_spectrum(spec, channel, hv, exp_time, [22, 22, 22], 22)
        print(f"{file} \t-> {isotope}, {channel}, {hv}, {exp_time}")