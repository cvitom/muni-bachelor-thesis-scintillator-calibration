import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class SpecInfo:
    filepath: str
    spec: np.ndarray
    temp_avg: float
    temp_std: float
    exp_time: float

def get_file_dir(izotope: str, temp: float | str, ch: int, dir="data"):
    temp_str = temp if isinstance(temp, str) else f'{int(temp)}'
    return f"./{dir}/{izotope}/t_{temp_str}/ch_{ch}"

def get_file_path(izotope: str, temp: float | str, ch: int, hv: int, dir="data"):
    temp_str = temp if isinstance(temp, str) else f'{int(temp)}'
    return f"./{dir}/{izotope}/t_{temp_str}/ch_{ch}/hv_{hv}.csv"

def load_spec_from_filepath(filepath: str) -> SpecInfo:
    df = pd.read_csv(filepath)
    exp_times = df.values[:, 0]
    temps = df.values[:, 1:4]
    total_exposure = np.sum(exp_times)
    spec_exposures = df.values[:, 4:]
    spec = np.sum(spec_exposures * np.tile(exp_times, (spec_exposures.shape[1], 1)).T, axis=0) / total_exposure
    temp_avg = np.average(temps, weights=np.tile(exp_times, (3, 1)).T)
    temp_std = np.sqrt(np.average((temps - temp_avg)**2, weights=np.tile(exp_times, (3, 1)).T))

    spec_info = SpecInfo(
        filepath=filepath,
        spec=spec,
        temp_avg=temp_avg,
        temp_std=temp_std,
        exp_time=total_exposure
    )

    return spec_info

def load_spec(isotope: str, temp: float, ch: int, hv: int, dir='data') -> SpecInfo:
    filepath = get_file_path(isotope, temp, ch, hv, dir)
    return load_spec_from_filepath(filepath)

