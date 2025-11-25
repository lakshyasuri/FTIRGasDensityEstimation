import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

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

    straight_regions_x, straight_regions_y = [], []
    for start, end in straight_regions:
        straight_regions_x.append((x[start], x[end - 1]))
        straight_regions_y.extend(y[start: end])
    straight_regions_y = np.array(straight_regions_y)
    wlen = len(df[df[x_name] <= df[x_name].iloc[0] + CONFIG.hyper_parameters.PEAK_WLEN])
    peaks, _ = find_peaks(straight_regions_y, distance=wlen)
    prominences = straight_regions_y[peaks]
    prominences = prominences[prominences > 0]
    peak_treshold = np.quantile(prominences, CONFIG.hyper_parameters.PEAK_PROMINENCE)
    print(peak_treshold)

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
    return straight_regions_x, peak_treshold


def start_analysis(df: pd.DataFrame, x_name: str, y_name: str, f_path: Path,
                   compute_baseline: bool, lbfgs_fit: bool):
    (alpha,
     _lambda, discarded_regions) = ae.baseline_estimation_process(df[x_name], -df[y_name],
                                                                  CONFIG.hyper_parameters.BASELINE,
                                                                  compute_baseline,
                                                                  file_name=f_path.name)
    df["absorption"] = alpha
    y_name = "absorption"
    if _lambda:
        CONFIG.hyper_parameters.BASELINE.BEST_LAM = _lambda
        utils.update_config(CONFIG)

    regions, peak_threshold = process_data(df, x_name, y_name, f_path, discarded_regions)

    raw_peaks, co2_peaks, h2o_peaks = [], [], []
    other_co2_peaks, other_h2o_peaks = [], []
    ambiguous_peaks, noisy_peaks = [], []
    rmse_vals = []
    co2_concentration, h2o_concentration = [], []
    secondary_peak_usage = {}

    for i, (start, end) in enumerate(regions):
        print(F"\n=================== REGION {i} ======================= ")
        left_idx = np.searchsorted(df[x_name], start, "left")
        right_idx = np.searchsorted(df[x_name], end, "right")
        x = df[left_idx: right_idx][x_name].reset_index(drop=True)
        y = df[left_idx: right_idx][y_name].reset_index(drop=True)
        print(f"\nRegion start and end points: {x.iloc[0]} to {x.iloc[-1]}")

        (peaks, left_bases,
         right_bases) = ae.peak_finding_process(x, y,
                                                CONFIG.hyper_parameters.PEAK_WLEN,
                                                peak_threshold, i,
                                                f_path.name,
                                                plots=True)
        raw_peaks.append(len(peaks))

        (x_peaks_plot, y_peaks_plot,
         peak_params, rmse_value) = ae.curve_and_peak_fitting_process(x, y,
                                                                      peaks,
                                                                      left_bases,
                                                                      right_bases,
                                                                      lbfgs_fit)

        print(f"\nPseudo-Voigt fit RMSE value for this region: {round(rmse_value, 4)}")
        rmse_vals.append(rmse_value)

        (strong_co2_lines, strong_h2o_lines,
         weak_co2_lines, weak_h2o_lines,
         weak_ambiguous) = ae.hitran_matching_process(peak_params, x, peaks, region=i,
                                                      filename=f_path.name)

        (co2_concs, h2o_concs, count_other_co2,
         count_other_h2o, count_ambiguous, count_noise,
         secondary_peaks_used) = ae.peak_assignment_and_ambiguity_resolution(
            strong_co2_lines, strong_h2o_lines, weak_co2_lines, weak_h2o_lines,
            weak_ambiguous, peak_params, peaks, x, y, x_peaks_plot, y_peaks_plot,
            region=i, filename=f_path.name)

        co2_peaks.append(len(co2_concs))
        h2o_peaks.append(len(h2o_concs))
        other_co2_peaks.append(count_other_co2)
        other_h2o_peaks.append(count_other_h2o)
        ambiguous_peaks.append(count_ambiguous)
        noisy_peaks.append(count_noise)
        secondary_peak_usage[i] = secondary_peaks_used

        co2_concentration.extend(co2_concs)
        h2o_concentration.extend(h2o_concs)
        if secondary_peaks_used["co2"]:
            print(f"\nSecondary peaks for CO2 used to estimate its concentration "
                  f"because no best peaks were found in region {i}!!")
        if secondary_peaks_used["h2o"]:
            print(f"\nSecondary peaks for H2O used to estimate its concentration "
                  f"because no best peaks were found in region {i}!!")
        print(f"\n=================== REGION {i} END ======================= ")

    print(f"\n=========== FINAL DIAGNOSTICS ================")
    print(f"\nNo. of initial prominent peaks for each region: {raw_peaks}. "
          f"\nTotal: {sum(raw_peaks)}")
    print(f"\nNo. of best CO2 peaks for each region: {co2_peaks}. "
          f"\nTotal: {sum(co2_peaks)}")
    print(f"\nNo. of best H2O peaks for each region: {h2o_peaks}. "
          f"\nTotal: {sum(h2o_peaks)}")
    print(f"\nNo. of secondary (needs further analysis) CO2 peaks for each region: "
          f"{other_co2_peaks}. \nTotal: {sum(other_co2_peaks)}")
    print(f"\nNo. of secondary (needs further analysis) H2O peaks for each region: "
          f"{other_h2o_peaks}. \nTotal: {sum(other_h2o_peaks)}")
    print(f"\nNo. of ambiguous peaks for each region: {ambiguous_peaks}. "
          f"\nTotal: {sum(ambiguous_peaks)}")
    print(f"\nNo. of congested or noisy peaks for each region: {noisy_peaks}. "
          f"\nTotal: {sum(noisy_peaks)}")
    print(f"\nTotal RMSE value for Pseudo-Voigt fit in all regions: "
          f"{round(sum(rmse_vals), 4)}.")

    print_str_co2, print_str_h2o = None, None
    for region, value in secondary_peak_usage.items():
        if value["co2"]:
            print_str_co2 = (
                f"Secondary peaks for CO2 used to estimate its concentration in "
                f"region {region}") if print_str_co2 is None else f"{print_str_co2}, {region}"
        if value["h2o"]:
            print_str_h2o = (
                f"Secondary peaks for H2O used to estimate its concentration in "
                f"region {region}") if print_str_h2o is None else f"{print_str_h2o}, {region}"
    if print_str_co2 is not None:
        print_str_co2 = f"{print_str_co2}. Concentration could be incorrect!"
    if print_str_h2o is not None:
        print_str_h2o = f"{print_str_h2o}. Concentration could be incorrect!"

    co2_mean, co2_lower, co2_upper = utils.bootstrap_ci_calculation(co2_concentration)
    h2o_mean, h2o_lower, h2o_upper = utils.bootstrap_ci_calculation(h2o_concentration)
    if print_str_co2 is not None:
        print(f"\n{print_str_co2}")
    print("Mean CO2 concentration: ", round(co2_mean, 3), " ppm")
    if print_str_h2o is not None:
        print(f"\n{print_str_h2o}")
    print(f"Mean H2O concentration: {round(h2o_mean, 3)} ppm")
    print("\nCO2 mean concentration with confidence intervals (alpha=0.05): \n",
          f"lower bound: {round(co2_lower, 3)}, mean: {round(co2_mean, 3)}, "
          f"upper bound: {round(co2_upper, 3)} ppm")
    print("\nH2O mean concentration with confidence intervals (alpha=0.05): \n",
          f"lower bound: {round(h2o_lower, 3)}, mean: {round(h2o_mean, 3)}, "
          f"upper bound {round(h2o_upper, 3)} ppm")
