import pandas as pd
import numpy as np
import numpy.typing as npt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from typing import Union, Any

from utils import create_plot, find_statistic_symmetrically, rmse, \
    molecules_per_cm3_to_ppm, bootstrap_ci_calculation
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
    baseline_means = []
    for i, value in enumerate(low_sd_chains.values()):
        if len(value) > 1:
            if np.mean(y[value[0]: value[1]]) < \
                    CONFIG.hyper_parameters.REGION_THRESHOLD:
                discarded_regions.append((x[value[0]], x[value[1]]))
            else:
                regions.append((value[0], value[1]))
            # h_line_args.append({"args": (np.mean(baseline[value[0]: value[1]]),),
            #                     "kwargs": dict(linestyle='--',
            #                                    label=f'Region {i} baseline mean')})
            baseline_means.append(np.mean(y[value[0]: value[1]]))

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
                title=f"{filename} with low SD regions highlighted", legend=True)
    return regions


def process_data(df: pd.DataFrame, x_name: str, y_name: str, f_path: Path):
    window_size, gauss_sigma = 201, 200

    f_sd_name = f_path.name.replace(".dpt", f"_sd_{window_size}.csv")
    f_sd_dir = f_path.parent / 'sd_files'
    f_sd_dir.mkdir(parents=True, exist_ok=True)
    f_sd_path = f_sd_dir / f_sd_name
    if not f_sd_path.is_file():
        sd_vals = find_statistic_symmetrically(df["wavenumber"], df["intensity"],
                                               window_size=window_size,
                                               statistic='std', assume_sorted=True)
        pd.DataFrame(data={x_name: df[x_name], "intensity_sd": sd_vals}).to_csv(f_sd_path,
                                                                                index=False)
        print(f"Saving rolling standard deviations of {y_name} with a window size of "
              f"{window_size} to path {f_sd_path}")
    else:
        print(f"Rolling standard deviations file found. Reading it from {f_sd_path}")
        sd_vals = pd.read_csv(f_sd_path)["intensity_sd"].to_numpy()

    # sd_threshold = np.median(sd_vals)
    sd_threshold = np.mean(sd_vals)
    smoothed_sd = gaussian_filter1d(sd_vals, sigma=gauss_sigma)

    # plotting the original data
    # plot_args = [{"args": (df[x_name], df[y_name])}]
    # create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)),
    #             title=f"Water-vapor concentration: {f_path.name}", x_label=x_name,
    #             y_label=y_name)

    # plotting the standard deviation of y values against x
    plot_args = [{"args": (df[x_name], sd_vals), "kwargs": dict(label="SD")},
                 {"args": (df[x_name], smoothed_sd),
                  "kwargs": dict(label="Gauss smoothed SD")}]
    h_line_args = [{"args": (sd_threshold,),
                    "kwargs": dict(color='black', linestyle='--', label='SD threshold')}]

    # inset plot settings and args
    inset_settings = dict(width="100%", height="100%", loc="upper left",
                          bbox_to_anchor=(0.15, 0.65, 0.3, 0.3),
                          bbox_transform=True)
    inset_df = df[(df[x_name] >= 6840) & (df[x_name] <= 6860)]
    inset_args = [{"args": (inset_df[x_name], sd_vals[inset_df.index])},
                  {"args": (inset_df[x_name], smoothed_sd[inset_df.index])}]
    inset_h_line_args = [{"args": (sd_threshold,),
                          "kwargs": dict(color='black', linestyle='--')}]
    create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)),
                legend={"loc": "upper right"},
                title=f"Water-vapor concentration: {f_path.name} standard deviation plot",
                y_label=rf"${y_name.capitalize()}\ SD\ (a.u.)$",
                x_label=rf'${x_name.capitalize()}\ (cm^{{-1}})$',
                hline_args=h_line_args, inset_settings=inset_settings,
                inset_args=inset_args, inset_hline_args=inset_h_line_args)

    print(sd_threshold)
    print(np.median(sd_vals))
    return smoothed_sd, sd_threshold


def start_analysis(df: pd.DataFrame, x_name: str, y_name: str, f_path: Path):
    y_baseline, bspline = ae.baseline_estimation_process(df[x_name], -df[y_name],
                                                         CONFIG.hyper_parameters.BASELINE,
                                                         0,
                                                         file_name=f_path.name)
    smoothed_sd, sd_threshold = process_data(df, x_name, y_name, f_path)
    regions = get_regions_of_interest(values=smoothed_sd, threshold=sd_threshold,
                                      x=df[x_name], y=df[y_name], filename=f_path.name,
                                      x_name=x_name, y_name=y_name, baseline=-y_baseline)

    raw_peaks, co2_peaks, h2o_peaks = [], [], []
    common_peaks, unassigned_peaks = [], []
    voigt_params, rmse_vals = [], []
    co2_concentration, h2o_concentration = [], []
    co2_concentration_2, h2o_concentration_2 = [], []

    for i, (start, end) in enumerate(regions):
        print(F"\n=================== REGION {i} ======================= ")
        x = df[start: end][x_name].reset_index(drop=True)
        y = df[start: end][y_name].reset_index(drop=True)
        print(f"\nRegion start and end points: {x.iloc[0]} to {x.iloc[-1]}")
        y_base = y_baseline[start: end]

        p_h_params = dict(PEAK_PROMINENCE=CONFIG.hyper_parameters.PEAK_PROMINENCE,
                          PEAK_WLEN=CONFIG.hyper_parameters.PEAK_WLEN,
                          AVG_WINDOW_SIZE=CONFIG.hyper_parameters.AVG_WINDOW_SIZE)
        peaks, left_bases, right_bases = ae.peak_finding_process(x, y, p_h_params,
                                                                 -y_base,
                                                                 i, f_path.name,
                                                                 plots=True)
        raw_peaks.append(len(peaks))

        k_h_params = dict(NON_PEAK_KNOTS=CONFIG.hyper_parameters.NON_PEAK_KNOTS)
        knot_vector, non_peak_regions = ae.peak_and_knot_placement_process(
            x, peaks, left_bases, right_bases, k_h_params, i
        )

        y_corrected = y + y_base
        _, y_peak, peak_params = ae.curve_and_peak_fitting_process(x, -y_corrected, peaks,
                                                                   left_bases,
                                                                   right_bases,
                                                                   knot_vector, True,
                                                                   f_path.name, i)
        voigt_params.append(len(peak_params) * 4)
        voigt_fit_rmse = rmse(y_corrected, -y_peak)
        rmse_vals.append(voigt_fit_rmse)

        (peak_params, co2_indices, h2o_indices,
         overlap_indices, unmatched_indices) = ae.hitran_matching_process(
            peak_params, x,
            -y_corrected, peaks, i,
            f_path.name,
            None, y_peak)
        co2_peaks.append(len(co2_indices))
        h2o_peaks.append(len(h2o_indices))
        common_peaks.append(len(overlap_indices))
        unassigned_peaks.append(len(unmatched_indices))

        (co2_concs, h2o_concs,
         co2_concs_2, h2o_concs_2) = ae.concentration_estimation_process(peak_params,
                                                                         co2_indices,
                                                                         h2o_indices,
                                                                         bspline,
                                                                         x, y)
        co2_concentration.extend(co2_concs)
        h2o_concentration.extend(h2o_concs)
        co2_concentration_2.extend(co2_concs_2)
        h2o_concentration_2.extend(h2o_concs_2)

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

    print("FWHM method in ppm: \n",
          molecules_per_cm3_to_ppm(np.mean(co2_concentration_2)),
          molecules_per_cm3_to_ppm(np.mean(h2o_concentration_2)))
    print("Area method: \n", np.mean(co2_concentration),
          np.mean(h2o_concentration))
    print("Area method in ppm: \n",
          molecules_per_cm3_to_ppm(np.mean(co2_concentration)),
          molecules_per_cm3_to_ppm(np.mean(h2o_concentration)))
