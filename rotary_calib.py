from rotary_ctrl.rotary_ctrl import RotaryCtrl
from rotary_ctrl.rotary_calib_gui import RotaryCalibGUI
from grb_ctrl.grb_bridge import GrbBridge
from time import sleep
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

center_angle_approx = 5

angle_min = -180
angle_max = 180

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-o", "--output", help="Output csv file for spectrums", default="rotary_spec.csv")
    arg_parser.add_argument("--hv-from", help="Starting HV value", type=int, default=130)
    arg_parser.add_argument("--hv-to", help="Ending HV value", type=int, default=180)
    arg_parser.add_argument("--hv-step", help="HV step value", type=int, default=10)
    arg_parser.add_argument("--ch", help="Channel number", type=int, default=2)
    arg_parser.add_argument("--step", help="Angle step", type=int, default=5)
    arg_parser.add_argument("-t", "--exp-time", help="Exposure time", type=int, default=50)

    args = arg_parser.parse_args()
    output_file = args.output
    hv_from = args.hv_from
    hv_to = args.hv_to
    hv_step = args.hv_step
    selected_channel = args.ch
    exp_time = args.exp_time
    angle_step = args.step


    print("Initializing rotary controller...")
    encoder = RotaryCtrl()
    encoder.init()
    encoder.move_to(center_angle_approx)
    print("Encoder at center position.")


    print("Initializing GRB controller...")
    grb = GrbBridge()

    def perform_scan():
        print("Starting scan...")
        df = pd.DataFrame(columns=["angle", "temp", "ch", "hv", *range(256)])
        for i, angle in enumerate(np.arange(center_angle_approx + angle_min, center_angle_approx + angle_max + 0.1, angle_step)):
            print(f"Moving to position {angle - center_angle_approx} (exp {i} / {int((angle_max - angle_min) / angle_step)})...")
            encoder.move_to(angle)
            channels = [selected_channel]
            if selected_channel == 2:
                channels = [0, 1]
            for ch in channels:
                for hv in range(hv_from, hv_to + 1, hv_step):
                    grb.set_hv(ch, hv)
                    print(f"Taking exposure for channel {ch} and HV {hv}...")
                    exposure = grb.measure_spectrum(0, exp_time)
                    temps = grb.get_temperatures()
                    df.loc[i] = np.concatenate([[angle - center_angle_approx], [np.average(temps)], [ch], [hv], exposure])
                    df.to_csv(output_file, index=False)
        print("Done")
    
    perform_scan()
    