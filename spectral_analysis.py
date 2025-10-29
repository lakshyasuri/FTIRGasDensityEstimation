import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.typing as npt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from typing import Union, Any

from utils import create_plot, find_statistic_symmetrically, rmse, \
    molecules_per_cm3_to_ppm, bootstrap_ci_calculation, update_config
from config import CONFIG
import analysing_engine as ae


def get_regions_of_interest(values: npt.NDArray[np.float64],
                            threshold: Union[float, np.floating[Any]], x: pd.Series,
                            y: pd.Series, x_name: str,
                            y_name: str, filename: str, baseline):
    """Get the points below the threshold. This assumes that the wavenumers are sorted!! very imp!"""
    low_sd_chains = {}
    counter = 0
    for i, value in enumerate(values):
        if value <= threshold and counter not in low_sd_chains:
            low_sd_chains[counter] = [i]
        if value > threshold and counter in low_sd_chains:
            low_sd_chains[counter].append(i)
            counter += 1
        if i == len(values) - 1 and counter in low_sd_chains and \
                len(low_sd_chains[counter]) == 1:
            low_sd_chains[counter].append(i)
    print(low_sd_chains)

    regions, discarded_regions = [], []
    h_line_args = []
    for i, value in enumerate(low_sd_chains.values()):
        if len(value) > 1:
            print(np.mean(y[value[0]: value[1]]))
            # if np.mean(y[value[0]: value[1]]) < \
            #         CONFIG.hyper_parameters.REGION_THRESHOLD:
            #     discarded_regions.append((x[value[0]], x[value[1]]))
            # else:
            regions.append((value[0], value[1]))
            # h_line_args.append({"args": (np.mean(baseline[value[0]: value[1]]),),
            #                     "kwargs": dict(linestyle='--',
            #                                    label=f'Region {i} baseline mean')})

    print(f"\nDiscarded regions due to close proximity to zero "
          f"intensity: \n{discarded_regions}")
    axv_args = [{"args": (x.iloc[start], x.iloc[end - 1]),
                 "kwargs": dict(color='green', alpha=0.4)} for start, end in regions]
    axv_args[0]["kwargs"]["label"] = "Low SD region"
    create_plot(plot_args=[{"args": (x, y)}], figure_args=dict(figsize=(10, 8)),
                y_label=rf"${y_name.capitalize()}\ SD\ (a.u.)$",
                x_label=rf'${x_name.capitalize()}\ (cm^{{-1}})$',
                vspan_args=axv_args,
                hline_args=[{"args": (CONFIG.hyper_parameters.REGION_THRESHOLD,),
                             "kwargs": {"linestyle": "--", "color": "black",
                                        "label": "Intensity threshold"}}],
                title=f"{filename} with low SD regions highlighted", legend=True,
                y_lim=(-4e-5, 4e-5))
    return regions


def process_data(df: pd.DataFrame, x_name: str, y_name: str, f_path: Path,
                 to_discard: list[tuple[int]]):
    x, y, = [], []
    for i, (left, right) in enumerate(to_discard):
        if i != len(to_discard) - 1:
            y.extend(df[y_name].iloc[right: to_discard[i + 1][0]].values)
            x.extend(df[x_name].iloc[right: to_discard[i + 1][0]].values)
        else:
            y.extend(df[y_name].iloc[right: df.shape[0]].values)
            x.extend(df[x_name].iloc[right: df.shape[0]].values)
    x, y = np.array(x), np.array(y)

    y_avg = find_statistic_symmetrically(x, y, window_size=101,
                                         statistic='mean', assume_sorted=True)
    y_avg_smooth = gaussian_filter1d(y_avg, sigma=300)
    y_avg_smooth_scaled = (y_avg_smooth - min(y_avg_smooth)) / (
            max(y_avg_smooth) - min(y_avg_smooth))
    y_avg_gradient = np.diff(y_avg_smooth_scaled)
    window_size = len(x[x <= x[0] + 50])
    half = window_size // 2
    y_avg_gradient_padded = np.pad(y_avg_gradient, (half, half), "symmetric")
    y_gradient_variance = [np.std(y_avg_gradient_padded[i: i + window_size]) for i in
                           range(len(y_avg_gradient))]

    curr_region, straight_regions = [], []
    min_length = 5
    for i in range(len(y_gradient_variance)):
        if y_gradient_variance[i] < CONFIG.hyper_parameters.ALPHA_COEFF_GRAD_STD:
            curr_region.append(i)
        else:
            if len(curr_region) >= min_length:
                straight_regions.append((curr_region[0], curr_region[-1] + 1))
            curr_region = []
    if len(curr_region) >= min_length:
        straight_regions.append((curr_region[0], curr_region[-1] + 1))
    straight_regions_x = [(x[start], x[end - 1]) for start, end in straight_regions]
    print(f"\nSelecting relatively straight regions from the absorption coefficient "
          "spectrum after discarding the low intensity regions: "
          f"\n{[f'{x[left]} to {x[right - 1]}' for left, right in straight_regions]}")

    axv_args = [{"args": (x[start], x[end - 1]),
                 "kwargs": dict(color='green', alpha=0.4)} for start, end in
                straight_regions]
    axv_args[0]["kwargs"]["label"] = "shortlisted region"
    plot_args = [
        {"args": (df[x_name], df[y_name])}
        # {"args": (df[x_name], y_avg)},
        # {"args": (x, y_avg_smooth)}
    ]
    create_plot(plot_args=plot_args,
                figure_args=dict(figsize=(10, 8)),
                y_label=rf"${y_name.capitalize()}\ SD\ (a.u.)$",
                x_label=rf'${x_name.capitalize()}\ (cm^{{-1}})$',
                vspan_args=axv_args,
                title=f"{f_path.name} with shortlisted regions highlighted", legend=True,
                y_lim=(-4e-5, 4e-5)
                )
    return straight_regions_x


def start_analysis(df: pd.DataFrame, x_name: str, y_name: str, f_path: Path,
                   compute_baseline: bool, least_squares_fit: bool):
    # df = df[df[x_name] <= 8000]
    (alpha,
     _lambda, discarded_regions) = ae.baseline_estimation_process(df[x_name], -df[y_name],
                                                                  CONFIG.hyper_parameters.BASELINE,
                                                                  0, compute_baseline,
                                                                  file_name=f_path.name)
    df["absorption"] = alpha
    y_name = "absorption"
    if _lambda:
        CONFIG.hyper_parameters.BASELINE.BEST_LAM = _lambda
        update_config(CONFIG)

    regions = process_data(df, x_name, y_name, f_path, discarded_regions)

    raw_peaks, co2_peaks, h2o_peaks = [], [], []
    common_peaks, unassigned_peaks = [], []
    voigt_params, rmse_vals = [], []
    co2_concentration, h2o_concentration = [], []
    co2_concentration_2, h2o_concentration_2 = [], []

    for i, (start, end) in enumerate(regions):
        print(F"\n=================== REGION {i} ======================= ")
        left_idx = np.searchsorted(df[x_name], start, "left")
        right_idx = np.searchsorted(df[x_name], end, "right")
        x = df[left_idx: right_idx][x_name].reset_index(drop=True)
        y = df[left_idx: right_idx][y_name].reset_index(drop=True)
        print(f"\nRegion start and end points: {x.iloc[0]} to {x.iloc[-1]}")

        p_h_params = dict(PEAK_PROMINENCE=CONFIG.hyper_parameters.PEAK_PROMINENCE,
                          PEAK_WLEN=CONFIG.hyper_parameters.PEAK_WLEN,
                          AVG_WINDOW_SIZE=CONFIG.hyper_parameters.AVG_WINDOW_SIZE)
        peaks, left_bases, right_bases = ae.peak_finding_process(x, y, p_h_params,
                                                                 i, f_path.name,
                                                                 plots=True)
        raw_peaks.append(len(peaks))

        _, y_peak, peak_params = ae.curve_and_peak_fitting_process(x, y, peaks,
                                                                   left_bases,
                                                                   right_bases,
                                                                   f_path.name, i,
                                                                   least_squares_fit)

        voigt_params.append(len(peak_params) * 4)
        voigt_fit_rmse = rmse(y, y_peak)
        print(voigt_fit_rmse)
        tss = np.sum(np.square(y - np.mean(y)))
        rss = np.sum(np.square(y - y_peak))
        print(round(1 - (rss / tss), 4))
        rmse_vals.append(voigt_fit_rmse)

        (peak_params, co2_indices, h2o_indices,
         overlap_indices, unmatched_indices) = ae.hitran_matching_process(
            peak_params, x, y, peaks, i, f_path.name, y_peak)
        co2_peaks.append(len(co2_indices))
        h2o_peaks.append(len(h2o_indices))
        common_peaks.append(len(overlap_indices))
        unassigned_peaks.append(len(unmatched_indices))

        (co2_concs, h2o_concs,
         co2_concs_2, h2o_concs_2) = ae.concentration_estimation_process(peak_params,
                                                                         co2_indices,
                                                                         h2o_indices,
                                                                         x, y, peaks)
        co2_concentration.extend(co2_concs)
        h2o_concentration.extend(h2o_concs)
        co2_concentration_2.extend(co2_concs_2)
        h2o_concentration_2.extend(h2o_concs_2)
        break

    print(f"\n=========== FINAL DIAGNOSTICS ================")
    print(f"\nNo. of prominent drops for each region: {raw_peaks}. "
          f"\nTotal: {sum(raw_peaks)}")
    print(f"\nNo. of CO2 drops for each region: {co2_peaks}. \nTotal: {sum(co2_peaks)}")
    print(f"\nNo. of H2O drops for each region: {h2o_peaks}. \nTotal: {sum(h2o_peaks)}")
    print(f"\nNo. of common drops for each region: {common_peaks}. "
          f"\nTotal: {sum(common_peaks)}")
    print(f"\nNo. of unmatched drops for each region: {unassigned_peaks}. "
          f"\nTotal: {sum(unassigned_peaks)}")
    print(f"\nNo. of Voigt parameters for each region: {voigt_params}. "
          f"\nTotal: {sum(voigt_params)}")
    print(f"\nRMSE values for Voigt fit in each region: {rmse_vals}. "
          f"\nAverage: {round(np.mean(rmse_vals), 3)}. Sum: {sum(rmse_vals)}")

    co2_mean, co2_lower, co2_upper = bootstrap_ci_calculation(co2_concentration_2)
    h2o_mean, h2o_lower, h2o_upper = bootstrap_ci_calculation(h2o_concentration_2)
    print("FWHM CO2 mean with confidence intervals: \n", co2_lower, co2_mean, co2_upper)
    print("FWHM H2O mean with confidence intervals: \n", h2o_lower, h2o_mean, h2o_upper)

    print("FWHM method: \n", np.mean(co2_concentration_2),
          np.mean(h2o_concentration_2))
    print("FWHM method in ppm: \n",
          molecules_per_cm3_to_ppm(np.mean(co2_concentration_2)),
          molecules_per_cm3_to_ppm(np.mean(h2o_concentration_2)))
    print("Area method: \n", np.mean(co2_concentration),
          np.mean(h2o_concentration))
    print("Area method in ppm: \n",
          molecules_per_cm3_to_ppm(np.mean(co2_concentration)),
          molecules_per_cm3_to_ppm(np.mean(h2o_concentration)))
