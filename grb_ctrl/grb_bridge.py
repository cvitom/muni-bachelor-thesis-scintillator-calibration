# echo-client.py

import socket
import numpy as np
from time import sleep
import pandas as pd

HOST_DEFAULT = "127.0.0.1"  # The server's hostname or IP address
PORT_DEFAULT = 9500  # The port used by the server

class GrbBridge:
    def __init__(self):
        self.HOST = HOST_DEFAULT
        self.PORT = PORT_DEFAULT
        
    def ch2rch(self, ch: int) -> int:
        '''Converts channel number (0 or 1) to readout channel number (0 or 2)'''
        if ch == 0:
            return 0
        elif ch == 1:
            return 2
        else:
            raise ValueError("Invalid channel number")

    def send_cmd(self, cmd, wait_time=0.1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.HOST, self.PORT))
            s.sendall((cmd + "\n").encode())
            sleep(wait_time)
            data = s.recv(1024)
            return data[:-2].decode() # remove trailing \r\n

    def turn_on(self):
        result = self.send_cmd("eps set outputs 35", 1)
        result += self.send_cmd("grb state 0", 1)
        result += self.send_cmd("grb state 1", 1)
        result += self.send_cmd("ping 6 -s 2", 1)
        result += self.send_cmd("ping 7 -s 2", 1)
        result += self.send_cmd("grb state 0", 1)
        result += self.send_cmd("grb state 1", 1)
        result += self.send_cmd("grb mmio 0 w:0x40:0x23", 1)
        result += self.send_cmd("grb mmio 1 w:0x40:0x23", 1)
        return result

    def turn_off(self):
        result = self.set_hv(0, 0)
        result += self.set_hv(1, 0)
        result += self.send_cmd("grb mmio 0 w:0x40:0x20 w:0x40:0x14", 1)
        result += self.send_cmd("grb mmio 1 w:0x40:0x20 w:0x40:0x14", 1)
        result += self.send_cmd("grb reboot 0", 1)
        result += self.send_cmd("grb reboot 1", 1)
        result += self.send_cmd("eps set outputs -", 1)
        return result
    
    def get_temperatures(self,):
        result = self.send_cmd(f"grb temperature 0")
        temp_strs = result.split('\n')
        temps = [float(temp) for temp in temp_strs[:-1]] # last line is blank
        return temps
    
    def measure_spectrum(self, channel: int, exp_time: int) -> np.ndarray:
        rch = self.ch2rch(channel)
        result = self.send_cmd(f"grb spectrum -d {rch} {exp_time}")
        return np.array([float(x) for x in result.split(' ')[1:]])
    
    def set_hv(self, channel: int, hv: int) -> str:
        rch = self.ch2rch(channel)
        result = self.send_cmd(f"grb hv {rch} {hv}")
        return result

def save_spectrum(spectrum, filename):
    df = pd.DataFrame({'channel': np.arange(len(spectrum)), 'counts': spectrum})
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    grb = GrbBridge()

    def run_exposure(channel, total_exp_time: int):
        spectrum = np.zeros(256)
        one_exp_time = 5
        for i in range(int(total_exp_time / one_exp_time)):
            print(f"Exp {i} / {int(total_exp_time / one_exp_time)}")
            spectrum = spectrum + grb.measure_spectrum(channel, one_exp_time)
        save_spectrum(spectrum, f'spectrum_{channel}_{total_exp_time}.csv')
        
    # print(grb.turn_off())
    print(grb.turn_on())
    temps = grb.get_temperatures(24)
    print(temps)
    hvresponse = grb.set_hv(0, 180)
    print(hvresponse)
    spectrum = np.zeros(256)
    # for i in range(20):
    #     print(f"Exp {i} / 20")
    #     spectrum = spectrum + grb.measure_spectrum(1, 5)
    spectrum = grb.measure_spectrum(0, 10)
    # save_spectrum(spectrum, 'spectrum_oneaaa_180_1.csv')
    print(grb.turn_off())