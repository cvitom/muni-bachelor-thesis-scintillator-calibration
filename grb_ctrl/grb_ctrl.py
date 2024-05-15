from .grb_gui import GrbGui
from .grb_bridge import GrbBridge, save_spectrum
from .temp_ctrl import TempCtrl
import numpy as np
from argparse import ArgumentParser
from .spectrum_manager import SpectrumManager
import threading
from time import sleep
import json

class GrbCtrl:
    def __init__(self):
        self.grb = GrbBridge()
        self.temp_ctrl = None
        self.gui = GrbGui()
        self.temp_log = np.array([])
        self.spec_manager = None
    
    def run_exposure(self, channel: int, total_exp_time: int, target_temp: float, hv: int):
        self.grb.set_hv(channel, hv)
        if not self.temp_any:
            self.wait_for_temperature(target_temp)
        
        spectrum = np.zeros(256)
        if target_temp is None:
            target_temp = "any"

        for i in range(int(total_exp_time / self.one_exp_time)):
            temperatures = self.grb.get_temperatures()
            if not self.temp_any:
                self.temp_ctrl.update_temperature(np.average(temperatures))
                if np.abs(target_temp - np.average(temperatures)) > 0.4:
                    print("Temperature drift detected")
                    self.wait_for_temperature(target_temp)
            self.gui.update_temperature(np.average(temperatures))
            print(f"Channel {channel}, temp {target_temp}, hv {hv}: exp {i} / {int(total_exp_time / self.one_exp_time)}, temperatures: {temperatures}")
            spectrum_one = self.grb.measure_spectrum(channel, self.one_exp_time)
            spectrum = spectrum + spectrum_one
            self.gui.update_spectrum(spectrum)
            if not self.disable_save:
                self.spec_manager.save_spectrum(spectrum_one, channel, hv, self.one_exp_time, temperatures, target_temp)

        self.grb.set_hv(channel, hv)

    def wait_for_temperature(self, target_temp: float):
        self.temp_ctrl.set_target_temperature(target_temp)
        self.gui.update_target_temperature(target_temp)

        while True:
            temperatures = self.grb.get_temperatures()
            avg_temp = np.average(temperatures)

            self.temp_ctrl.update_temperature(avg_temp)
            self.temp_log = np.append(self.temp_log, avg_temp)

            print(temperatures)

            wait_measurements = 40

            self.gui.update_temperature(avg_temp)
            if len(self.temp_log) >= wait_measurements and np.all(np.abs(self.temp_log[-wait_measurements:] - target_temp) < 0.4):
                print("Target temperature reached")
                break
            sleep(0.5)
    
    def request_float_if_none(self, value: float | None, name: str) -> float:
        while value is None:
            try:
                print(f"Enter {name}: ")
                value = float(input())
            except:
                print("Invalid input")
        return value

    def run_exposures(self):
        if self.temp_ctrl is not None:
            self.temp_ctrl.connect()

        if self.temp_any:
            for hv in range(self.hv_from, self.hv_to+1, self.hv_step):
                if self.channel == 2:
                    self.run_exposure(0, self.exp_time, None, hv)
                    self.run_exposure(1, self.exp_time, None, hv)
                else:
                    self.run_exposure(self.channel, self.exp_time, None, hv)
        else:
            for temp in np.arange(self.temp_from, self.temp_to, self.temp_step):
                for hv in range(self.hv_from, self.hv_to+1, self.hv_step):
                    if self.channel == 2:
                        self.run_exposure(0, self.exp_time, temp, hv)
                        self.run_exposure(1, self.exp_time, temp, hv)
                    else:
                        self.run_exposure(self.channel, self.exp_time, temp, hv)
        
        print("All exposures finished")
        self.grb.set_hv(0, 0)
        self.grb.set_hv(1, 0)

    def run(self):
        arg_parser = ArgumentParser()
        arg_parser.add_argument('--port', type=str)
        arg_parser.add_argument('--restart', action='store_true')
        arg_parser.add_argument('--start', action='store_true')
        arg_parser.add_argument('--stop', action='store_true')
        arg_parser.add_argument('--isotope', type=str)
        arg_parser.add_argument('--temp-from', type=float)
        arg_parser.add_argument('--temp-step', type=float)
        arg_parser.add_argument('--temp-to', type=float)
        arg_parser.add_argument('--temp-any', action='store_true')
        arg_parser.add_argument('--disable-save', action='store_true')
        arg_parser.add_argument('--hv-from', type=int)
        arg_parser.add_argument('--hv-to', type=int)
        arg_parser.add_argument('--hv-step', type=int)
        arg_parser.add_argument('--exp-time', type=int)
        arg_parser.add_argument('--one-exp-time', type=int)
        arg_parser.add_argument('-o', type=str)
        arg_parser.add_argument('--channel', type=int, default=0)
        arg_parser.add_argument('--config', type=str)

        args = arg_parser.parse_args()

        config_file = args.config
        config = {}
        if config_file is not None:
            config = json.load(open(config_file))

        port = args.port
        restart_required = args.restart
        start_required = args.start
        stop_required = args.stop
        out_dir = args.o or config.get('out_dir') or './data'
        isotope = args.isotope or config.get('isotope') or 'unknown'
        
        self.hv_from = args.hv_from or config.get('hv_from') or 130
        self.hv_to = args.hv_to or config.get('hv_to') or 180
        self.hv_step = args.hv_step or config.get('hv_step') or 10
        self.exp_time = args.exp_time or config.get('exp_time')

        self.temp_from = args.temp_from or config.get('temp_from')
        self.temp_to = args.temp_to or config.get('temp_to')
        self.temp_step = args.temp_step or config.get('temp_step')
        self.disable_save = args.disable_save or config.get('disable_save')
        self.temp_any = args.temp_any or config.get('temp_any')
        self.one_exp_time = args.one_exp_time or config.get('one_exp_time') or 5
        self.channel = args.channel or config.get('channel')

        if start_required:
            print(self.grb.turn_on())
            exit(0)
        if restart_required:
            print(self.grb.turn_off())
            print(self.grb.turn_on())
            exit(0)
        if stop_required:
            print(self.grb.turn_off())
            exit(0)

        self.spec_manager = SpectrumManager(out_dir, isotope)
        if not self.temp_any:
            if port is None:
                print("Port is not specified")
                exit(1)        

            self.temp_from = self.request_float_if_none(self.temp_from, "temperature - from")
            self.temp_to = self.request_float_if_none(self.temp_to, "temperature - to")
            self.temp_step = self.request_float_if_none(self.temp_step, "temperature - step")
            self.exp_time = self.request_float_if_none(self.exp_time, "exposure time")

            self.temp_ctrl = TempCtrl(port)

        threading.Thread(target=self.run_exposures).start()

        self.gui.show()
                
