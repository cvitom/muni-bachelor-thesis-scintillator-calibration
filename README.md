# GRBBeta scintillator calibration

 Welcome to the repo with software for measurement and analysis of calibration
 data for gamma detectors on board of GRBAlpha/GRBBeta nanosatellites. 
 
## Requirements

Before running anything in this repo, you need `python3` and required packages installed on your system.
The quickest way to install the packages is using [pipenv](https://pipenv.pypa.io/en/latest/),
with command `pipenv install`. Every command must be then ran inside `pipenv shell`.
Pipenv automatically creates a virtual python environment, so you don't need to
worry about package conflicts with your system.

The second way is to make sure you have every package from `Pipfile` installed.

## Measurement

The code for control of the detector electronics though `vcom` can be found inside of `grb_ctrl` directory.
Prior to running this code, you must have `vcom` software installed and running on your system.
To start `vcom` in noninteractive mode, use the command `vcom -n -a 15 -b 921600 -d /dev/ttyUSB2`.
Replace `/dev/ttyUSB2` with your own serial port.

To run the measurement, launch `grb_ctrl.py` with python. See the available configuration with
via `python3 grb_ctrl.py --help`. Before starting the measurement, you need to boot the
satellite and enable power to the analog boards. The quickest way to do this is via
`python3 grb_ctrl.py --start`.

Software used for the directional sensitivity measurement can be found in `rotary_ctrl` directory.
Run it with `python3 -m rotary_ctrl.rotary_ctrl {args}`, replacing `{args}` with your configuration.

Fitting the peaks can achieved quickly using software in `peaks_finder` directory. 
Launch it with `python3 -m peaks_finder.peaks_finder {args}`, replacing `{args}` with your configuration.

Sample configuration for each of these is in `.vscode/launch.json` file.
## Data analysis

Data and results from our calibration can be downloaded from [here](https://owncloud.cesnet.cz/index.php/s/uj82wuxiYYigzHS)

All the notebooks used for the data analysis can be found in `notebooks` directory.

## temp_ctrl

Within the `temp_ctrl` directory is a platformio project, with simple code for the ESP32 devkit
controlling the Peltier element.