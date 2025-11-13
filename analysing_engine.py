import sys
from types import SimpleNamespace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pybaselines.utils import ParameterWarning
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences, savgol_filter
from pybaselines import Baseline
import numpy.typing as npt
import time
from typing import List, Tuple, Any, Union, Dict
from scipy.optimize import minimize, least_squares
from pathlib import Path
import warnings
import multiprocessing
import json

import utils
from HITRAN import fetch_data, get_hitran_strength_threshold
from config import CONFIG


def baseline_wrapper(kwargs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit, params = kwargs.get('baseline_fitter').aspls(
            data=kwargs.get("data"), lam=kwargs.get("lam"),
            max_iter=kwargs.get("max_iter"),
            tol=kwargs.get("tol")
        )
        for warning in w:
            if issubclass(warning.category, ParameterWarning):
                print(warning.message, warning.category, kwargs.get("lam"))
                warnings.warn(f"{warning.category}: {warning.message}")
                return None

    if params["tol_history"].size > 0 and params["tol_history"][-1] > kwargs.get("tol"):
        warnings.warn(f"Baseline fit with lambda {kwargs.get("lam")} did not "
                      f" converge: \n{params["tol_history"]}")
        return None
    return fit


def find_peaks_and_filter(x: Union[pd.Series, np.ndarray],
                          y: Union[pd.Series, np.ndarray],
                          window_len, peak_threshold):
    """Find peaks and filter. Prominence based on the height of peaks"""
    wlen = len(x[x <= x[0] + window_len])
    peaks, _ = find_peaks(y, distance=wlen)
    prominences = y[peaks]
    prominences = prominences.to_numpy() if isinstance(prominences,
                                                       pd.Series) else prominences
    peak_thresh = np.quantile(prominences, peak_threshold)
    filtered_p_idx = np.where(prominences >= peak_thresh)[0]
    filtered_peaks = peaks[filtered_p_idx]
    filtered_peaks.sort()
    filtered_p_proms = prominences[filtered_p_idx]
    return filtered_peaks, filtered_p_proms


def peak_finding_process(x: pd.Series,
                         y: pd.Series,
                         hyper_params: dict, n_region: int,
                         filename: str, plots: bool = True):
    peaks, prominences = find_peaks_and_filter(x, y,
                                               hyper_params["PEAK_WLEN"],
                                               hyper_params["PEAK_PROMINENCE"])
    print(
        f"\nNumber of prominent peaks - {len(peaks)}; {round(len(peaks) / len(x) * 100, 2)} %")

    left_bases, right_bases, to_remove = [], [], []
    dy = np.gradient(y)
    zero_crossings_minima = np.where((dy[:-1] <= 0) & (dy[1:] > 0))[0]
    for i, peak in enumerate(peaks):
        l_bs = zero_crossings_minima[zero_crossings_minima < peak]
        r_bs = zero_crossings_minima[zero_crossings_minima > peak]
        if l_bs.size == 0 or r_bs.size == 0:
            to_remove.append(i)
            continue
        left_bases.append(l_bs[-1])
        right_bases.append(r_bs[0])
    print(f"\nNumber of peaks without bases: {len(to_remove)}\n{peaks[to_remove]}")
    if to_remove:
        peaks = np.delete(peaks, to_remove)
    left_bases = np.array(left_bases, dtype=int)
    right_bases = np.array(right_bases, dtype=int)

    # if plots:
    # plot_args = [{"args": (x, y)}, {"args": (x[peaks], y[peaks], 'x'),
    #                                 "kwargs": {"label": "Absorption maximum"}},
    # {"args": (x[left_bases], y[left_bases], 'ro'),
    #  "kwargs": {"label": "left end point", "ms": 4}},
    # {"args": (x[right_bases], y[right_bases], 'go'),
    #  "kwargs": {"label": "right end point", "ms": 4}}
    # ]

    # # inset plot settings and args
    # inset_settings = dict(width="100%", height="100%", loc="upper left",
    #                       bbox_to_anchor=(0.1, 0.65, 0.5, 0.35),
    #                       bbox_transform=True)
    # inset_idx = x[(x > 6210) & (x <= 6224)].index
    # inset_left_bases = np.intersect1d(inset_idx, left_bases)
    # inset_right_bases = np.intersect1d(inset_idx, right_bases)
    # inset_peaks = np.intersect1d(inset_idx, peaks)
    #
    # inset_args = [{"args": (x[inset_idx], y[inset_idx])},
    #               {"args": (x[inset_peaks], y[inset_peaks], 'x')},
    #               {"args": (x[inset_left_bases], y[inset_left_bases], 'ro'),
    #                "kwargs": {"ms": 4}},
    #               {"args": (x[inset_right_bases], y[inset_right_bases], 'go'),
    #                "kwargs": {"ms": 4}}]

    # utils.create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)),
    #                   legend=True,
    #                   title=f"Region {n_region}: prominent absorption peaks for {filename}",
    #                   x_label=r'$Wavenumber\ (cm^{-1})$',
    #                   y_label=r"$Absorption Coefficient\ (cm^{-1})$",
    #                   y_lim=(-2e-6, None))
    # y_lim=(None, 0.1),
    # inset_settings=inset_settings, inset_args=inset_args)
    return peaks, left_bases, right_bases


def get_low_standard_deviation_regions(x: Union[pd.Series, npt.NDArray[float]],
                                       y: Union[pd.Series, npt.NDArray[float]],
                                       file_name: str):
    window_size, gauss_sigma = 201, 200
    _directory = Path(__file__).resolve().parent / "intermediate_data_files"
    file_path = _directory / file_name
    if not file_path.is_file():
        _directory.mkdir(exist_ok=True)
        sd_vals = utils.find_statistic_symmetrically(x, y, window_size=window_size,
                                                     statistic='std', assume_sorted=True)
        pd.DataFrame(data={"wavenumber": x, "intensity_sd": sd_vals}).to_csv(file_path,
                                                                             index=False)
        print(f"Saving rolling standard deviations of intensity with a window size of "
              f"{window_size} to path {file_path}")
    else:
        print(f"Rolling standard deviations for intensity file found. "
              f"Reading it from {file_path}")
        sd_vals = pd.read_csv(file_path)["intensity_sd"].to_numpy()

    sd_threshold = np.median(sd_vals)
    smoothed_sd = gaussian_filter1d(sd_vals, sigma=gauss_sigma)

    low_sd_chains, counter = {}, 0
    for i, value in enumerate(smoothed_sd):
        if value <= sd_threshold and counter not in low_sd_chains:
            low_sd_chains[counter] = [i]
        if value > sd_threshold and counter in low_sd_chains:
            low_sd_chains[counter].append(i)
            counter += 1
        if i == len(smoothed_sd) - 1 and counter in low_sd_chains and \
                len(low_sd_chains[counter]) == 1:
            low_sd_chains[counter].append(i)
    return smoothed_sd, low_sd_chains, sd_threshold


def low_intensity_filtering(y: Union[pd.Series, npt.NDArray[float]],
                            low_sd_regions: Dict[int, List], intensity_threshold: float):
    regions, discarded_regions = [], []
    for i, value in enumerate(low_sd_regions.values()):
        if len(value) > 1:
            print(np.mean(y[value[0]: value[1]]))
            if np.mean(y[value[0]: value[1]]) < intensity_threshold:
                discarded_regions.append((value[0], value[1]))
            else:
                regions.append((value[0], value[1]))
    return regions, discarded_regions


def baseline_r_12_metric_calculation(x: npt.NDArray[float],
                                     y_smoothed: npt.NDArray[float],
                                     baseline: npt.NDArray[float],
                                     normalization_region: Tuple[int, int]):
    y_corrected = y_smoothed - baseline
    y_normal_region = y_corrected[normalization_region[0]: normalization_region[1]]
    y_normal_mean = np.mean(y_normal_region)
    y_normalized = y_corrected / y_normal_mean

    wlen = len(x[x <= x[0] + 1])
    minima_idx, _ = find_peaks(-y_normalized, distance=wlen)
    prominences = peak_prominences(-y_normalized, minima_idx)[0]
    peak_thresh = np.quantile(prominences, 0.5)
    filtered_minima_idx = minima_idx[np.where(prominences >= peak_thresh)[0]]

    y_no_peaks = y_normalized[filtered_minima_idx]
    y_mean = np.mean(y_normalized[normalization_region[0]: normalization_region[1]])
    peak_region_mask = np.ones(y_normalized.size, dtype=bool)
    peak_region_mask[filtered_minima_idx] = False
    y_peaks = y_normalized[peak_region_mask]

    sum_y_peaks = np.sum(y_peaks)
    sum_y_no_peaks = np.sum(y_no_peaks)
    t_penalty = (max(y_no_peaks) - min(y_no_peaks)) / y_mean
    m1 = (np.log(len(y_peaks)) / sum_y_peaks) + (
            sum_y_no_peaks * t_penalty / np.log(len(y_no_peaks)))
    m2 = sum_y_peaks / (sum_y_no_peaks * t_penalty + sum_y_peaks)
    r_12 = m1 / m2
    return r_12


def baseline_estimation_process(x: Union[pd.Series, npt.NDArray[float]],
                                y: Union[pd.Series, npt.NDArray[float]],
                                hyper_params: SimpleNamespace,
                                n_region: int, compute_baseline: bool,
                                file_name: str = None):
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    # making y positive for easier understanding
    offset = abs(min(y)) + max(y)
    y_positive = y + offset
    y_smoothed = savgol_filter(y_positive, window_length=100, polyorder=2)

    if compute_baseline or hyper_params.BEST_LAM is None:
        file_name = file_name.replace(".dpt", f"_intensity_rolling_sd.csv")
        y_sd, low_sd_regions, _ = get_low_standard_deviation_regions(x, -y, file_name)
        good_regions, discarded_regions = low_intensity_filtering(-y, low_sd_regions,
                                                                  hyper_params.REGION_THRESHOLD)
        good_regions_x = [f"{x[val[0]]} to {x[val[1] - 1]}" for val in good_regions]
        discarded_regions_x = [f"{x[val[0]]} to {x[val[1] - 1]}" for val in
                               discarded_regions]
        normalization_region = good_regions[0]

        print(f"\nLow standard deviation regions in the "
              f"original spectrum: \n{good_regions_x}")
        print(f"\nDiscarded low SD regions due to close proximity to zero "
              f"intensity (for baseline estimation): \n{discarded_regions_x}")
        print(hyper_params.REGION_THRESHOLD)
        print(f"\nRegion chosen for normalization during "
              f"baseline estimation: \n{good_regions_x[0]}")

        plot_args = [
            {"args": (x, -y),
             "kwargs": dict(label="Original spectrum")}]
        baseline_fitter = Baseline(x, assume_sorted=True)
        lambda_vals = np.logspace(start=13, stop=15, num=30)
        print(f"\nComputing a new baseline. Smoothness (lambda) parameter values: "
              f"\n{lambda_vals}")
        rmse_vals, baseline_fits = {}, {}

        with multiprocessing.Pool(processes=8) as pool:
            results = pool.map(baseline_wrapper,
                               [{'baseline_fitter': baseline_fitter, 'data': y_smoothed,
                                 'lam': lam, 'max_iter': 1000,
                                 'tol': 7e-4} for lam in lambda_vals])

        for i, result in enumerate(results):
            if result is not None:
                lam = lambda_vals[i]
                baseline_fits[lam] = result
                rmse_vals[lam] = baseline_r_12_metric_calculation(x, y_smoothed, result,
                                                                  normalization_region)
                plot_args.append({"args": (x, -(result - offset)),
                                  "kwargs": {"label": rf"$\lambda$={lam}"}})

        min_rmse_lam = min(rmse_vals, key=rmse_vals.get)
        print(f"\nR\u00b9\u00b2 metric values for candidate lambdas: "
              f"\n{json.dumps(utils.serialize_json(rmse_vals), indent=1)}")
        print(f"Optimal lambda value: {min_rmse_lam}")
        best_baseline = baseline_fits[min_rmse_lam]
        best_baseline = best_baseline - offset
    else:
        min_rmse_lam = None
        baseline_fitter = Baseline(x, assume_sorted=True)
        func_args = {'baseline_fitter': baseline_fitter, 'data': y_smoothed,
                     'lam': hyper_params.BEST_LAM, 'max_iter': 1000,
                     'tol': 7e-4}
        best_baseline = baseline_wrapper(func_args)
        if best_baseline is None:
            raise Exception(f"Baseline fit with lambda {hyper_params.BEST_LAM} did not "
                            f" converge.")
        best_baseline = best_baseline - offset
        plot_args = [{"args": (x, -y), "kwargs": dict(label="Original spectrum")},
                     {"args": (x, -best_baseline),
                      "kwargs": {"label": rf"$\lambda$={hyper_params.BEST_LAM}"}}
                     ]

    utils.create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)),
                      legend={"loc": "upper right"},
                      title=f"Baseline estimation for humidity level: {file_name}",
                      x_label=r'$Wavenumber\ (cm^{-1})$', y_label=r"$Intensity\ (a.u.)$")

    alpha = ((best_baseline / y) - 1) * (1 - 0.999) / 400

    curr_region, low_intensity_regions = [], []
    for i in range(len(best_baseline)):
        if -best_baseline[i] <= hyper_params.REGION_THRESHOLD:
            curr_region.append(i)
        else:
            if len(curr_region) != 0:
                low_intensity_regions.append((curr_region[0], curr_region[-1] + 1))
            curr_region = []
    if len(curr_region) != 0:
        low_intensity_regions.append((curr_region[0], curr_region[-1] + 1))
    low_intensity_regions_x = [f"{x[val[0]]} to {x[val[1] - 1]}" for val in
                               low_intensity_regions]
    print(f"\nDiscarded low intensity regions due to close proximity to zero "
          f"intensity: \n{low_intensity_regions_x}")
    return alpha, min_rmse_lam, low_intensity_regions


def curve_and_peak_fitting_process(x: Union[pd.Series, np.ndarray],
                                   y: Union[pd.Series, np.ndarray],
                                   peaks: npt.NDArray[np.int64],
                                   left_bases: npt.NDArray[np.int64],
                                   right_bases: npt.NDArray[np.int64],
                                   filename: str, n_region: int,
                                   least_sqaures_fit: bool):
    start_time = time.time()
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    bounds: List[Tuple[Any, Any]] = []

    # setting bounds for faster convergence
    peak_params, optimal_params = [], []
    multiplier = 2 * np.sqrt(2 * np.log(2))
    rmse_vals = []
    plot_args = [
        {"args": (x, y), "kwargs": dict(label="Baseline-corrected spectrum")}
    ]
    voigt_p_list, no_fits, noise, duplicates = [], [], [], []
    x_peaks_plot, y_peaks_plot = [], []
    discarded = False
    fitted_centres = set()

    for i in range(len(peaks)):
        peak_region = slice(left_bases[i], right_bases[i])
        area = np.trapezoid(
            y[peak_region] - np.minimum(y[left_bases[i]], y[right_bases[i]]),
            x[peak_region])

        A0 = area
        mu0 = x[peaks[i]]
        sigma0 = (x[right_bases[i]] - x[left_bases[i]]) / 6
        fwhm0 = sigma0 * multiplier
        # gamma0 = sigma0
        ratio0 = 0.5
        # peak_params.append([A0, mu0, sigma0, gamma0])
        peak_params.append([ratio0, A0, mu0, fwhm0])

        mu0_min = x[left_bases[i]]
        mu0_max = x[right_bases[i]]
        l_idx = np.searchsorted(x, mu0_min, "left")
        r_idx = np.searchsorted(x, mu0_max, "right")
        x_local, y_local = x[l_idx: r_idx], y[l_idx: r_idx]
        area_max = 1
        if least_sqaures_fit:
            try:
                min_result = least_squares(
                    fun=utils.loss_function_vector, jac=utils.jacobian_least_squares,
                    x0=peak_params[-1], args=(x_local, y_local, 1), ftol=1e-15,
                    xtol=1e-15,
                    gtol=1e-15, max_nfev=10000,
                    # bounds=([1e-10, mu0_min, 1e-5, 1e-5],
                    #         [np.inf, mu0_max, 1, 1]),
                    bounds=([0, 1e-15, mu0_min, 1e-8],
                            [1, area_max, mu0_max, 1])
                )
            except ValueError as e:
                if "outside of provided bounds" in str(e):
                    warnings.warn(f"Found an abnormal peak at centre {mu0}. "
                                  f"Fitting routine threw the error: {str(e)}. "
                                  f"Ignoring this peak and moving ahead")
                    voigt_p_list.append({"discard": True, "centre": mu0})
                    no_fits.append(mu0)
                    continue
                else:
                    raise
        else:
            min_result = minimize(
                fun=utils.loss_function, jac=utils.jacobian_of_loss, x0=peak_params[-1],
                args=(x_local, y_local, 1), method='L-BFGS-B',
                # bounds=[(1e-10, None), (x[left_bases[i]], x[right_bases[i]]), (1e-8, 1),
                #         (1e-8, 1)],
                bounds=[(0, 1), (1e-15, area_max), (mu0_min, mu0_max), (1e-8, 1)],
                options={'maxiter': 10000, 'ftol': 1e-15, 'disp': True,
                         'gtol': 1e-15}
            )
        if not min_result.success:
            raise Exception(f"Optimization warning: {min_result}")
        optimal_params.append(min_result.x)

        l_idx_eval = l_idx - 33 if l_idx >= 33 else l_idx
        r_idx_eval = r_idx + 34 if r_idx <= len(x) - 34 else r_idx
        x_eval = x[l_idx_eval: r_idx_eval]
        y_eval = y[l_idx_eval: r_idx_eval]

        y_hat = utils.pseudo_voigt_profile(x_eval, min_result.x[0], min_result.x[1],
                                           min_result.x[2], min_result.x[3])

        rmse_vals.append(utils.rmse(y_eval, y_hat))
        v_params = {"ratio": min_result.x[0], "area": min_result.x[1],
                    "centre": min_result.x[2], "fwhm": min_result.x[3], "discard": False}
        if min_result.x[1] == 0:
            v_params["discard"] = True
            no_fits.append(min_result.x[2])
        elif min_result.x[3] < 0.08 or min_result.x[3] > 0.27:
            v_params["discard"] = True
            noise.append((min_result.x[2], float(min_result.x[3])))
        if min_result.x[2] in fitted_centres:
            v_params["discard"] = True
            duplicates.append(min_result.x[2])
        voigt_p_list.append(v_params)

        # if not v_params['discard']:
        x_peaks_plot.append(x_eval)
        y_peaks_plot.append(y_hat)
        plot_dict = {"args": (x_eval, y_hat),
                     "kwargs": dict(linestyle='--', color='orange')}
        if i == 0:
            plot_dict['kwargs']['label'] = "Fitted Pseudo-Voigt Profile"
        if v_params["discard"]:
            discarded = True
            plot_dict['kwargs']['color'] = 'red'
        plot_args.append(plot_dict)
        fitted_centres.add(min_result.x[2])

    if discarded:
        for arg in plot_args:
            if arg['kwargs'].get('color') == 'red':
                arg['kwargs']['label'] = "Discarded Peaks - FWHM Issue"
                break
    end_time = time.time()

    print(f"\n{len(no_fits)} abnormal or peaks with 0 area discarded: "
          f"\n{np.round(no_fits, 3)}")
    print(f"\n{len(noise)} drops discarded due to FWHM < 0.08 or FWHM > 0.25: "
          f"\n{np.round([val[0] for val in noise], 3)}")
    print(f"\n{len(duplicates)} drops discarded due duplicate fit: "
          f"\n{np.round(duplicates, 3)}")

    print(f"total time taken for fitting {round(end_time - start_time, 3)} seconds")

    # utils.create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)), legend=True,
    #                   title=f"Region {n_region}: Voigt profiles fit for {filename}",
    #                   x_label=r'$Wavenumber\ (cm^{-1})$',
    #                   y_label=r"$Absorption\ (cm^{-1})$")
    # utils.create_plot(plot_args=[{"args": (x[peaks], rmse_vals)}], scatter=True)
    return x_peaks_plot, y_peaks_plot, voigt_p_list, sum(rmse_vals)


def get_full_width_at_half_max(sigma: float, gamma: float):
    fwhm_gauss = 2 * sigma * np.sqrt(2 * np.log(2))
    fwhm_loren = 2 * gamma
    return (0.5343 * fwhm_loren) + np.sqrt(0.2169 * fwhm_loren ** 2 + fwhm_gauss ** 2)


def hitran_matching_process(peak_params: List[dict],
                            x: Union[pd.Series, npt.NDArray[float]],
                            peaks: npt.NDArray[int]):
    nu_exp = np.array([v["centre"] for v in peak_params])
    diff = abs(x[peaks] - nu_exp)
    # utils.create_plot(plot_args=[{"args": (x[peaks], diff)}],
    #                   title="Voigt fit centre differences against observed wavenumbers",
    #                   x_label=r'$Wavenumber\ (cm^{-1})$', y_label='residuals',
    #                   scatter=True)

    nu_hitran_co2, nu_hitran_h2o = fetch_data()
    h2o_s_threshold = get_hitran_strength_threshold(df=nu_hitran_h2o, gas_name='h2o')
    co2_s_threshold = get_hitran_strength_threshold(df=nu_hitran_co2, gas_name='co2')

    strong_co2_lines, weak_co2_lines = {}, {}
    strong_h2o_lines, weak_h2o_lines = {}, {}
    no_matches, weak_ambiguous = [], {}
    for i, nu in enumerate(nu_exp):
        if peak_params[i]["discard"]:
            continue
        low = nu - CONFIG.hyper_parameters.HITRAN_CENTRE_THRESHOLD
        high = nu + CONFIG.hyper_parameters.HITRAN_CENTRE_THRESHOLD

        start_idx_co2 = np.searchsorted(nu_hitran_co2["wavenumber"], low, "left")
        end_idx_co2 = np.searchsorted(nu_hitran_co2["wavenumber"], high, "right")
        start_idx_h2o = np.searchsorted(nu_hitran_h2o["wavenumber"], low, "left")
        end_idx_h2o = np.searchsorted(nu_hitran_h2o["wavenumber"], high, "right")

        potential_matches_co2 = nu_hitran_co2[start_idx_co2: end_idx_co2]
        potential_matches_h2o = nu_hitran_h2o[start_idx_h2o: end_idx_h2o]
        strong_line_co2, strong_line_h2o = False, False

        if potential_matches_co2.empty and potential_matches_h2o.empty:
            no_matches.append(nu)
        else:
            if not potential_matches_co2.empty:
                max_strength_idx_co2 = potential_matches_co2["strength"].idxmax()
                nearest_nu_hitran_co2 = potential_matches_co2["wavenumber"][
                    max_strength_idx_co2]
                line_strength = potential_matches_co2["strength"][max_strength_idx_co2]
                if isinstance(co2_s_threshold, pd.DataFrame):
                    region_nu_idx = np.searchsorted(co2_s_threshold["wavenumber"],
                                                    nearest_nu_hitran_co2, "left") - 1
                    condition_co2 = (
                            line_strength >= co2_s_threshold["mean_strength"].iloc[
                        region_nu_idx])
                else:
                    condition_co2 = (line_strength >= co2_s_threshold)
                if condition_co2:
                    if nu not in strong_co2_lines:
                        strong_line_co2 = True
                        strong_co2_lines[nu] = {"peak_idx": i,
                                                "hitran_nu": nearest_nu_hitran_co2,
                                                "line_strength": line_strength}
                        peak_params[i]["line_strength"] = line_strength
                    else:
                        # duplicate fit because it's very close to another peak
                        continue
                else:
                    weak_co2_lines[nu] = {"peak_idx": i,
                                          "hitran_nu": nearest_nu_hitran_co2,
                                          "line_strength": line_strength}
            if not potential_matches_h2o.empty:
                max_strength_idx_h2o = potential_matches_h2o["strength"].idxmax()
                nearest_nu_hitran_h2o = potential_matches_h2o["wavenumber"][
                    max_strength_idx_h2o]
                line_strength = potential_matches_h2o["strength"][
                    max_strength_idx_h2o]
                if isinstance(h2o_s_threshold, pd.DataFrame):
                    region_nu_idx = np.searchsorted(h2o_s_threshold["wavenumber"],
                                                    nearest_nu_hitran_h2o, "left") - 1
                    condition_h2o = (
                            line_strength >= h2o_s_threshold["mean_strength"].iloc[
                        region_nu_idx])
                else:
                    condition_h2o = (line_strength >= h2o_s_threshold)
                if condition_h2o:
                    if nu not in strong_h2o_lines:
                        strong_line_h2o = True
                        strong_h2o_lines[nu] = {"peak_idx": i,
                                                "hitran_nu": nearest_nu_hitran_h2o,
                                                "line_strength": line_strength}
                        peak_params[i]["line_strength"] = line_strength
                else:
                    weak_h2o_lines[nu] = {"peak_idx": i,
                                          "hitran_nu": nearest_nu_hitran_h2o,
                                          "line_strength": line_strength}
            if not potential_matches_h2o.empty and not potential_matches_co2.empty:
                if strong_line_co2 and strong_line_h2o:
                    # print(
                    #     f"Strong ambiguous - CO2: {nu} {strong_co2_lines[nu]["line_strength"]} "
                    #     f"{(peak_params[i]['area'] / strong_co2_lines[nu]["line_strength"]):.3e}, "
                    #     f"H2O: {nu} {strong_h2o_lines[nu]["line_strength"]}")
                    strong_co2_lines[nu]["ambiguous"] = True
                elif strong_line_co2:
                    if weak_h2o_lines[nu]["line_strength"] >= strong_co2_lines[nu][
                        "line_strength"]:
                        # print(f"strong co2 val: "
                        #       f"{strong_co2_lines[nu]["line_strength"]}, {nu}")
                        # print(f"weak h20 val: "
                        #       f"{weak_h2o_lines[nu]["line_strength"]}, {nu}")
                        strong_h2o_lines[nu] = weak_h2o_lines[nu]
                        strong_co2_lines[nu]["ambiguous"] = True
                elif strong_line_h2o:
                    if weak_co2_lines[nu]["line_strength"] >= strong_h2o_lines[nu][
                        "line_strength"]:
                        # print(f"strong h2o val: {strong_h2o_lines[nu]["line_strength"]}, "
                        #       f"{nu}")
                        # print(f"weak co2 val: {weak_co2_lines[nu]["line_strength"]}, "
                        #       f"{nu}")
                        strong_h2o_lines[nu]["ambiguous"] = True
                else:
                    # print(f"Weak ambiguous - CO2: {nu} "
                    #       f"{weak_co2_lines[nu]["line_strength"]}, H2O: {nu} "
                    #       f"{weak_h2o_lines[nu]["line_strength"]}")
                    weak_ambiguous[nu] = weak_co2_lines[nu]
                    del weak_co2_lines[nu]
                    del weak_h2o_lines[nu]

    weak_co2_centres = list(weak_co2_lines.keys())
    weak_h2o_centres = list(weak_h2o_lines.keys())

    print(f"\n{len(weak_co2_centres)} weak CO2 peaks found: "
          f"\n{np.round(weak_co2_centres, 3)}")
    print(f"\n{len(weak_h2o_centres)} weak H2O peaks found: "
          f"\n{np.round(weak_h2o_centres, 3)}")
    print(f"\n{len(weak_ambiguous)} weak H2O or CO2 (ambiguous) peaks found: "
          f"\n{np.round(list(weak_ambiguous.keys()), 3)}")
    print(f"\n{len(no_matches)} unmatched peaks found: \n{np.round(no_matches, 3)}")

    return strong_co2_lines, strong_h2o_lines


def peak_assignment_and_ambiguity_resolution(strong_co2_lines: Dict[float, dict],
                                             strong_h2o_lines: Dict[float, dict],
                                             peak_params: List[dict], x: pd.Series,
                                             y: pd.Series, x_peaks_plot: list,
                                             y_peaks_plot: list, region: int,
                                             filename: str):
    ambiguous_peaks, blended_peaks = {}, {}
    co2_concs, h2o_concs = [], []
    strong_co2_centres, strong_h2o_centres = [], []
    x_co2_vals, y_co2_vals = [], []
    x_h2o_vals, y_h2o_vals = [], []
    for centre, value in strong_co2_lines.items():
        concentration = peak_params[value["peak_idx"]]["area"] / value["line_strength"]
        concentration_ppm = utils.molecules_per_cm3_to_ppm(concentration)
        if value.get("ambiguous") and \
                strong_h2o_lines[centre]["line_strength"] > value["line_strength"]:
            ambiguous_peaks[centre] = value
        else:
            if hasattr(CONFIG.hyper_parameters, "CO2_PPM_LOWER") and hasattr(
                    CONFIG.hyper_parameters, "CO2_PPM_UPPER"):
                if concentration_ppm < CONFIG.hyper_parameters.CO2_PPM_LOWER or \
                        concentration_ppm > CONFIG.hyper_parameters.CO2_PPM_UPPER:
                    blended_peaks[centre] = value
        if centre not in blended_peaks and centre not in ambiguous_peaks:
            strong_co2_centres.append(centre)
            co2_concs.append(concentration_ppm)
            params = peak_params[value["peak_idx"]]
            x_co2_vals.append(centre)
            y_co2_vals.append(
                utils.pseudo_voigt_profile(centre, params['ratio'], params['area'],
                                           centre, params['fwhm']))
            print(
                f"CO2: line_strength: {value['line_strength']:.3e}, "
                f"Centre: {round(centre, 2)}, "
                f"Conc.: {concentration:.3e} molecules/cm3, "
                f"{round(concentration_ppm, 3)} ppm")

    for centre, value in strong_h2o_lines.items():
        if value.get("ambiguous"):
            ambiguous_peaks[centre] = value
        if centre not in ambiguous_peaks and centre not in blended_peaks:
            concentration = peak_params[value["peak_idx"]]["area"] / value[
                "line_strength"]
            concentration_ppm = utils.molecules_per_cm3_to_ppm(concentration)
            strong_h2o_centres.append(centre)
            h2o_concs.append(concentration_ppm)
            params = peak_params[value["peak_idx"]]
            x_h2o_vals.append(centre)
            y_h2o_vals.append(
                utils.pseudo_voigt_profile(centre, params['ratio'], params['area'],
                                           centre, params['fwhm']))
            print(
                f"H2O: line_strength: {value['line_strength']:.3e}, "
                f"Centre: {round(centre, 2)}, "
                f"Conc.: {concentration:.3e} molecules/cm3, "
                f"{round(concentration_ppm, 3)} ppm")

    print(f"\n{len(strong_co2_centres)} strong unique CO2 peaks found: "
          f"\n{np.round(strong_co2_centres, 3)}")
    print(f"\n{len(strong_h2o_centres)} strong unique H2O peaks found: "
          f"\n{np.round(strong_h2o_centres, 3)}")
    print(f"\n{len(ambiguous_peaks)} strong H2O or CO2 (ambiguous) peaks found: "
          f"\n{np.round(list(ambiguous_peaks.keys()), 3)}")
    print(f"\n{len(blended_peaks)} possibly water blended or noisy peaks found: "
          f"\n{np.round(list(blended_peaks), 3)}")

    plot_args = [
        {"args": (x, y), "kwargs": {"label": "Absorption Coefficient"}}
    ]
    if len(x_co2_vals) != 0:
        plot_args.append({"args": (x_co2_vals, y_co2_vals, 'gx'),
                          "kwargs": {"ms": 8, "label": "CO2 absorption peak"}})
    if len(x_h2o_vals) != 0:
        plot_args.append({"args": (x_h2o_vals, y_h2o_vals, 'rx'),
                          "kwargs": {"ms": 8, "label": "H2O absorption peak"}})

    discarded_label_added = False
    for i in range(len(x_peaks_plot)):
        plt_args = {"args": (x_peaks_plot[i], y_peaks_plot[i]),
                    "kwargs": {"linestyle": "--", "alpha": 0.8, "color": 'orange'}}
        if i == 0:
            plt_args['kwargs']['label'] = "Estimated Pseudo-Voigt profiles"
        if peak_params[i].get('discard'):
            plt_args["kwargs"]["color"] = "black"
            if not discarded_label_added:
                plt_args["kwargs"]["label"] = "Discarded Pseudo-Voigt profiles"
                discarded_label_added = True
        plot_args.append(plt_args)

    utils.create_plot(plot_args=plot_args, figure_args={"figsize": (10, 8)},
                      title=f"Region {region}: Spectrum with Pseudo-Voigt fits and "
                            f"HITRAN matches with a tolerance of +- {CONFIG.hyper_parameters.HITRAN_CENTRE_THRESHOLD} "
                            f"for {filename}",
                      x_label=r'$Wavenumber\ (cm^{-1})$',
                      y_label=r"$Absorption\ Coefficient\ (cm^{-1})$", legend=True,
                      y_lim=(-2e-6, None))
    return co2_concs, h2o_concs
