import numpy as np
import pandas as pd
import os

class SpectrumManager:
    def __init__(self, output_dir: str, isotope: str):
        self.output_dir = output_dir
        self.isotope = isotope

    def save_spectrum(self, spectrum: np.ndarray, channel: int, hv: int, total_exp_time: int, temperatures, target_temp: float | str):
        dir = f"{self.output_dir}/{self.isotope}/t_{target_temp}/ch_{channel}"
        filepath = f"{dir}/hv_{hv}.csv"
        os.makedirs(dir, exist_ok=True)

        row = np.concatenate(([total_exp_time], temperatures, spectrum))
        df = None
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df.loc[df.values.shape[0]] = row
        else:
            df = pd.DataFrame(columns=['exp_time', 't_1', 't_2', 't_3', *range(0, 256)])
            df.loc[0] = row
            
        df.to_csv(filepath, index=False)




        