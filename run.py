from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import time

from spectral_analysis import start_analysis
from config import CONFIG


def initialise_process(directory: Path, filename: str = None, plot: bool = True,
                       compute_baseline: bool = False, lbfgs: bool = False):
    if not filename:
        file_list = [x for x in directory.glob("*.dpt") if x.is_file()]
    else:
        file_list = [directory / filename]
    start = time.time()
    for file in file_list:
        print(f"\nProcessing file: {file.name}")
        df = pd.read_csv(file, names=["wavenumber", "intensity"], sep="\t")
        df.columns = ["wavenumber", "intensity"]
        df.sort_values(by="wavenumber", inplace=True)
        CONFIG.NU_MIN = df["wavenumber"].iloc[0] - 1
        CONFIG.NU_MAX = df["wavenumber"].iloc[-1] + 1
        start_analysis(df=df, x_name="wavenumber", y_name="intensity", f_path=file,
                       compute_baseline=compute_baseline, lbfgs_fit=lbfgs)
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
    parser.add_argument('--compute_baseline', action='store_true',
                        help="Compute a new baseline if the flag is used, else use the "
                             "previous one.")
    parser.add_argument('--lbfgs', action='store_true',
                        help='Fit Pseudo-Voigt profile to absorption peaks using L-BFGS-B'
                             ' algorithm. If the flag is not provided, '
                             'the default non-linear least squares algorithm is used.')
    args = parser.parse_args()
    print(args.lbfgs)
    initialise_process(args.directory, args.filename, args.plot, args.compute_baseline,
                       args.lbfgs)
