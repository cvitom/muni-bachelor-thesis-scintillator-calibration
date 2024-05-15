from argparse import ArgumentParser
from serial import Serial
from time import sleep

class TempCtrl:
    def __init__(self, port: str):
        self.port = port
    
    def update_temperature(self, t: float) -> None:  
        self.serial.write(f'update_temp {t}\n'.encode())
        response = self.serial.readline()
        if (response != b"OK\r\n"):
            # raise ValueError(f"Device responded unexpectedly: {response}")
            print("Temperature update failed")

    def set_target_temperature(self, t: float) -> None:
        self.serial.write(f'set_target {t}\n'.encode())
        sleep(0.2)
        response = self.serial.readline()
        if (response != b"OK\r\n"):
            print(f"Device responded unexpectedly: {response.decode()}")

    def get_power(self) -> float:
        self.serial.write('get_power\n'.encode())
        response = self.serial.readline()
        return float(response.decode())
    
    def connect(self) -> None:
        self.serial = Serial(self.port, 115200, timeout=1)
        self.serial.readline()

    def disconnect(self) -> None:
        self.serial.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--port', type=str, required=True)
    args = arg_parser.parse_args()

    port = args.port

    temp_ctrl = TempCtrl(port)
    temp_ctrl.connect()
    temp_ctrl.set_target_temperature(24.0)
    temp_ctrl.update_temperature(25.0)
    print(temp_ctrl.get_power())
    temp_ctrl.disconnect()