import os
import argparse
import numpy as np
import pandas as pd
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()
    exp_dir = args.dir

    total_exp_time = 0
    total_files_count = 0
    for root, dirs, files in os.walk(exp_dir):
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(root, file))
                exp_times = df.values[:, 0]
                total_exp_time += np.sum(exp_times)
                total_files_count += 1
    print("EXP TIME: ", total_exp_time)
    print("FILES COUNT: ", total_files_count)
            
    
