import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

import utils
from config import CONFIG
import analysing_engine as ae


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

    y_avg = utils.find_statistic_symmetrically(x, y, window_size=101,
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
    utils.create_plot(plot_args=plot_args,
                      figure_args=dict(figsize=(10, 8)),
                      y_label=rf"${y_name.capitalize()}\ SD\ (a.u.)$",
                      x_label=rf'${x_name.capitalize()}\ (cm^{{-1}})$',
                      vspan_args=axv_args,
                      title=f"{f_path.name} with shortlisted regions highlighted",
                      legend=True,
                      y_lim=(-4e-5, 4e-5)
                      )
    return straight_regions_x


def start_analysis(df: pd.DataFrame, x_name: str, y_name: str, f_path: Path,
                   compute_baseline: bool, least_squares_fit: bool):
    (alpha,
     _lambda, discarded_regions) = ae.baseline_estimation_process(df[x_name], -df[y_name],
                                                                  CONFIG.hyper_parameters.BASELINE,
                                                                  0, compute_baseline,
                                                                  file_name=f_path.name)
    df["absorption"] = alpha
    y_name = "absorption"
    if _lambda:
        CONFIG.hyper_parameters.BASELINE.BEST_LAM = _lambda
        utils.update_config(CONFIG)

    regions = process_data(df, x_name, y_name, f_path, discarded_regions)

    raw_peaks, co2_peaks, h2o_peaks = [], [], []
    rmse_vals = []
    co2_concentration, h2o_concentration = [], []

    for i, (start, end) in enumerate(regions):
        print(F"\n=================== REGION {i} ======================= ")
        left_idx = np.searchsorted(df[x_name], start, "left")
        right_idx = np.searchsorted(df[x_name], end, "right")
        x = df[left_idx: right_idx][x_name].reset_index(drop=True)
        y = df[left_idx: right_idx][y_name].reset_index(drop=True)
        print(f"\nRegion start and end points: {x.iloc[0]} to {x.iloc[-1]}")

        p_h_params = dict(PEAK_PROMINENCE=CONFIG.hyper_parameters.PEAK_PROMINENCE,
                          PEAK_WLEN=CONFIG.hyper_parameters.PEAK_WLEN)
        peaks, left_bases, right_bases = ae.peak_finding_process(x, y, p_h_params,
                                                                 i, f_path.name,
                                                                 plots=True)
        raw_peaks.append(len(peaks))

        (x_peaks_plot, y_peaks_plot,
         peak_params, rmse_value) = ae.curve_and_peak_fitting_process(x, y,
                                                                      peaks,
                                                                      left_bases,
                                                                      right_bases,
                                                                      f_path.name,
                                                                      i,
                                                                      least_squares_fit)

        print(f"\nPseudo-Voigt fit RMSE value for this region: {round(rmse_value, 4)}")
        rmse_vals.append(rmse_value)

        strong_co2_lines, strong_h2o_lines = ae.hitran_matching_process(peak_params, x,
                                                                        peaks)
        co2_concs, h2o_concs = ae.peak_assignment_and_ambiguity_resolution(
            strong_co2_lines, strong_h2o_lines,
            peak_params, x, y, x_peaks_plot, y_peaks_plot, region=i, filename=f_path.name)

        co2_peaks.append(len(co2_concs))
        h2o_peaks.append(len(h2o_concs))

        co2_concentration.extend(co2_concs)
        h2o_concentration.extend(h2o_concs)
        print(F"\n=================== REGION {i} END ======================= ")

    print(f"\n=========== FINAL DIAGNOSTICS ================")
    print(f"\nNo. of initial prominent peaks for each region: {raw_peaks}. "
          f"\nTotal: {sum(raw_peaks)}")
    print(f"\nNo. of strong CO2 peaks for each region: {co2_peaks}. "
          f"\nTotal: {sum(co2_peaks)}")
    print(f"\nNo. of strong H2O peaks for each region: {h2o_peaks}. "
          f"\nTotal: {sum(h2o_peaks)}")
    print(f"\nTotal RMSE value for Pseudo-Voigt fit in all regions: "
          f"{round(sum(rmse_vals), 4)}.")

    co2_mean, co2_lower, co2_upper = utils.bootstrap_ci_calculation(co2_concentration)
    h2o_mean, h2o_lower, h2o_upper = utils.bootstrap_ci_calculation(h2o_concentration)
    print("\nMean CO2 concentration: ", round(co2_mean, 3), " ppm")
    print(f"\nMean H2O concentration: {round(h2o_mean, 3)} pmm")
    print("\nCO2 mean concentration with confidence intervals (alpha=0.05): \n",
          f"lower bound: {round(co2_lower, 3)}, mean: {round(co2_mean, 3)}, "
          f"upper bound: {round(co2_upper, 3)} ppm")
    print("\nH2O mean concentration with confidence intervals (alpha=0.05): \n",
          f"lower bound: {round(h2o_lower, 3)}, mean: {round(h2o_mean, 3)}, "
          f"upper bound {round(h2o_upper, 3)} ppm")
