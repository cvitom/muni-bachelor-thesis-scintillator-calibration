import os
os.environ["QT_LOGGING_RULES"] = '*.debug=false'

from grb_ctrl.grb_ctrl import GrbCtrl

if __name__ == "__main__":
    grb_ctrl = GrbCtrl()
    grb_ctrl.run()
    