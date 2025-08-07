import pandas as pd
import numpy as np
import numpy.typing as npt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from typing import Union, Any

from utils import create_plot, StatManager, rmse
from config import hyper_parameters
import analysing_engine as ae


def get_regions_of_interest(values: npt.NDArray[np.float64],
                            threshold: Union[float, np.floating[Any]], x: pd.Series,
                            y: pd.Series, x_name: str, y_name: str, filename: str):
    """Get the points below the threshold. This assumes that the wavenumers are sorted!! very imp!"""
    low_sd_chains = {}
    counter = 0
    for i, value in enumerate(values):
        if value <= threshold and counter not in low_sd_chains:
            low_sd_chains[counter] = [i]
        if value > threshold and counter in low_sd_chains:
            low_sd_chains[counter].append(i)
            counter += 1
    print(low_sd_chains)

    regions = []
    for value in low_sd_chains.values():
        regions.append(
            pd.DataFrame({x_name: x[value[0]: value[1]], y_name: y[value[0]: value[1]]})
        )

    axv_args = [{"args": (x.iloc[val[0]], x.iloc[val[1] - 1]),
                 "kwargs": dict(color='green', alpha=0.4)} for _, val in
                low_sd_chains.items()]
    create_plot(plot_args=[{"args": (x, y)}], figure_args=dict(figsize=(10, 8)),
                y_label=y_name, x_label=x_name, vspan_args=axv_args,
                title=f"{filename} with low SD regions highlighted")

    return regions


def process_data(df: pd.DataFrame, x_name: str, y_name: str, f_path: Path):
    window_size, gauss_sigma = 101, 100

    f_sd_name = f_path.name.replace(".dpt", f"_sd_{window_size}.csv")
    f_sd_path = f_path.parent / 'sd_files' / f_sd_name
    if not f_sd_path.is_file():
        df_stats = StatManager(window_size=window_size)
        sd_vals = df_stats.find_statistic_symmetrically(df["wavenumber"], df["intensity"],
                                                        statistic='std')
        pd.DataFrame(data={x_name: df[x_name], "intensity_sd": sd_vals}).to_csv(f_sd_path,
                                                                                index=False)
        print(f"Saving rolling standard deviations of {y_name} with a window size of "
              f"{window_size} to path {f_sd_path}")
    else:
        print(f"Rolling standard deviations file found. Reading it from {f_sd_path}")
        sd_vals = pd.read_csv(f_sd_path)["intensity_sd"].to_numpy()

    sd_threshold = np.median(sd_vals)
    smoothed_sd = gaussian_filter1d(sd_vals, sigma=gauss_sigma)

    # plotting the original data
    plot_args = [{"args": (df[x_name], df[y_name])}]
    create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)),
                title=f"Water-vapor concentration: {f_path.name}", x_label=x_name,
                y_label=y_name)

    # plotting the standard deviation of y values against x
    plot_args = [{"args": (df[x_name], sd_vals), "kwargs": dict(label="SD")},
                 {"args": (df[x_name], smoothed_sd),
                  "kwargs": dict(label="Gauss smoothed SD")}]
    h_line_args = [{"args": (sd_threshold,),
                    "kwargs": dict(color='black', linestyle='--', label='SD threshold')}]
    create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)), legend=True,
                title=f"Water-vapor concentration: {f_path.name} standard deviation plot",
                y_label=f"{y_name} SD", x_label=x_name, hline_args=h_line_args)

    print(sd_threshold)
    return smoothed_sd, sd_threshold


def start_analysis(df: pd.DataFrame, x_name: str, y_name: str, f_path: Path):
    df.sort_values(by=x_name, inplace=True)
    smoothed_sd, sd_threshold = process_data(df, x_name, y_name, f_path)

    regions = get_regions_of_interest(values=smoothed_sd, threshold=sd_threshold,
                                      x=df[x_name],
                                      y=df[y_name], filename=f_path.name, x_name=x_name,
                                      y_name=y_name)

    for i in range(1, len(regions)):
        x = regions[i][x_name].reset_index(drop=True)
        y = regions[i][y_name].reset_index(drop=True)

        p_h_params = dict(PEAK_PROMINENCE=hyper_parameters["PEAK_PROMINENCE"],
                          PEAK_WLEN=hyper_parameters["PEAK_WLEN"],
                          AVG_WINDOW_SIZE=hyper_parameters["AVG_WINDOW_SIZE"])
        peaks, left_bases, right_bases = ae.peak_finding_process(x, y, p_h_params, i,
                                                                 plots=True)

        k_h_params = dict(NON_PEAK_KNOTS=hyper_parameters["NON_PEAK_KNOTS"])
        knot_vector, non_peak_regions = ae.peak_and_knot_placement_process(
            x, peaks, left_bases, right_bases, k_h_params, i
        )

        b_h_params = dict(**hyper_parameters["BASELINE"])
        baseline_fit = ae.baseline_estimation_process(x, -y, b_h_params, i)
        y_corrected = y + baseline_fit

        y_bkg, y_peak = ae.curve_and_peak_fitting_process(x, -y, peaks, left_bases,
                                                          right_bases, knot_vector, False,
                                                          i)
        without_b_corr_error = rmse(y, -(y_bkg + y_peak))
        print(f"\nRMSE without prior baseline correction: {without_b_corr_error}")

        _, y_peak = ae.curve_and_peak_fitting_process(x, -y_corrected, peaks, left_bases,
                                                          right_bases, knot_vector, True,
                                                          i)
        with_b_corr_error = rmse(y, -(y_peak + baseline_fit))
        print(f"\nRMSE with prior baseline correction: {with_b_corr_error}")


        break
