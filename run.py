from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from spectral_analysis import start_analysis
from config import CONFIG


def initialise_process(directory: Path, filename: str = None, plot: bool = True):
    if not filename:
        file_list = [x for x in directory.glob("I_40MHz_res*") if x.is_file()]
        vapour_densities = [
            int(x.name.split('I_40MHz_res_')[1].replace(".dpt", "").replace("0.", "")) for
            x in
            file_list]
        vapour_densities.sort()
        file_nums = [vapour_densities[idx] for idx in
                     np.linspace(0, len(vapour_densities) - 1, 5, dtype=int)]
        base_name = "I_40MHz_res_0."
        f_names = [f"{base_name}{file_num}.dpt" for file_num in file_nums]
    else:
        f_names = [filename]
    print(f_names)
    start = time.time()
    for f_name in f_names:
        f_path = directory / f_name
        df = pd.read_csv(f_path, names=["wavenumber", "intensity"], sep="\t")
        df.columns = ["wavenumber", "intensity"]
        df.sort_values(by="wavenumber", inplace=True)
        CONFIG.NU_MIN = df["wavenumber"].iloc[0] - 5
        CONFIG.NU_MAX = df["wavenumber"].iloc[-1] + 5
        start_analysis(df=df, x_name="wavenumber", y_name="intensity", f_path=f_path)
    end = time.time()
    print(f"\nTotal time taken by the program: {round(end - start, 3)}")
    if plot:
        plt.show()



if __name__ == "__main__":
    parser = ArgumentParser(description="Runner script which initialises the process of "
                                        "gas identification and density estimation")
    parser.add_argument('directory', type=Path, help="Directory of the FTIR data files")
    parser.add_argument('--filename', required=False, help="Name of the file to process")
    parser.add_argument('--plot', action='store_true', help="Flag to show plots")
    args = parser.parse_args()
    initialise_process(args.directory, args.filename, args.plot)
