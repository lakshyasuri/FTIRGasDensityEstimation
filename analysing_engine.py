import sys
from types import SimpleNamespace
import matplotlib.pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pybaselines.utils import ParameterWarning
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import legend
from scipy.ndimage import label, gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences, savgol_filter
from pybaselines import Baseline
import numpy.typing as npt
import time
from typing import List, Tuple, Any, Union, Dict
from scipy.optimize import minimize, least_squares
from scipy.sparse import dia_matrix, diags, linalg
from scipy.sparse.linalg import spsolve
from pathlib import Path
from scipy.interpolate import BSpline, CubicSpline
from scipy.stats import t
import warnings
import multiprocessing
import json

import utils
from HITRAN import fetch_data, get_regional_mean_strength
from config import CONFIG, directory


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
    """Find peaks and filter. Prominence based on distance from local averages"""
    wlen = len(x[x <= x[0] + window_len])
    wlen_2 = len(x[x <= x[0] + 0.3])
    peaks, _ = find_peaks(y, distance=wlen)
    prominences = peak_prominences(y, peaks, wlen_2)[0]

    prominences_2 = y[peaks]
    prominences_2 = prominences_2.to_numpy() if isinstance(prominences_2,
                                                           pd.Series) else prominences_2
    # peak_thresh = np.quantile(prominences, peak_threshold)
    peak_thresh_2 = np.quantile(prominences_2, peak_threshold)
    # filtered_p_idx = np.where(prominences >= peak_thresh)[0]
    filtered_p_idx_2 = np.where(prominences_2 >= peak_thresh_2)[0]
    # filtered_peaks = peaks[filtered_p_idx]
    filtered_peaks_2 = peaks[filtered_p_idx_2]
    # common_peaks = np.array(list(set(filtered_peaks) & set(filtered_peaks_2)))
    common_peaks = filtered_peaks_2
    common_peaks.sort()
    filtered_p_proms = prominences[filtered_p_idx_2]
    return common_peaks, filtered_p_proms


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
        prominences = np.delete(prominences, to_remove)
    left_bases = np.array(left_bases, dtype=int)
    right_bases = np.array(right_bases, dtype=int)

    if plots:
        plot_args = [{"args": (x, y)}, {"args": (x[peaks], y[peaks], 'x'),
                                        "kwargs": {"label": "Absorption minimum"}},
                     # {"args": (x[peaks], prominences, 'x'),
                     #  "kwargs": {"label": "Peak prominence"}},
                     # {"args": (x, baseline),
                     #  "kwargs": {"label": f"Baseline estimate", "color": "red"}},
                     {"args": (x[left_bases], y[left_bases], 'ro'),
                      "kwargs": {"label": "left end point", "ms": 4}},
                     {"args": (x[right_bases], y[right_bases], 'go'),
                      "kwargs": {"label": "right end point", "ms": 4}}
                     ]
        # inset plot settings and args
        inset_settings = dict(width="100%", height="100%", loc="upper left",
                              bbox_to_anchor=(0.1, 0.65, 0.5, 0.35),
                              bbox_transform=True)
        inset_idx = x[(x > 6210) & (x <= 6224)].index
        inset_left_bases = np.intersect1d(inset_idx, left_bases)
        inset_right_bases = np.intersect1d(inset_idx, right_bases)
        inset_peaks = np.intersect1d(inset_idx, peaks)

        inset_args = [{"args": (x[inset_idx], y[inset_idx])},
                      {"args": (x[inset_peaks], y[inset_peaks], 'x')},
                      {"args": (x[inset_left_bases], y[inset_left_bases], 'ro'),
                       "kwargs": {"ms": 4}},
                      {"args": (x[inset_right_bases], y[inset_right_bases], 'go'),
                       "kwargs": {"ms": 4}}]

        utils.create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)),
                          legend=True,
                          title=f"Region {n_region}: prominent absorption drops for {filename}",
                          x_label=r'$Wavenumber\ (cm^{-1})$',
                          y_label=r"$Absorption Coefficient\ (cm^{-1})$")
        # y_lim=(None, 0.1),
        # inset_settings=inset_settings, inset_args=inset_args)
    return peaks, left_bases, right_bases


def spectrum_linear_expansion(x: npt.NDArray[float], y: npt.NDArray[float]):
    N = len(x)
    omega = N // 20
    x_omega, y_omega = x[-omega:], y[-omega:]
    print(len(x_omega))
    reg = LinearRegression(n_jobs=-1).fit(x_omega[:, np.newaxis], y_omega)
    y_hat = reg.predict(x_omega[:, np.newaxis])
    print(reg.coef_)

    extended_len = N // 5
    diff = x[1] - x[0]
    end_val = x[-1] + (diff * extended_len)
    x_extended = np.linspace(start=x[-1], stop=end_val, num=extended_len)
    y_linear = reg.predict(x_extended[:, np.newaxis])

    # FWHM
    max_y = max(y)
    height = max(y) - y_linear[len(y_linear) // 2]
    gauss_width, gauss_height = (x_extended[-1] - x_extended[0]) / 8, height
    gauss_centre_1 = x_extended[extended_len // 3]
    gauss_height_1 = max_y - y_linear[extended_len // 3]
    gauss_centre_2 = x_extended[2 * extended_len // 3]
    gauss_height_2 = max_y - y_linear[2 * extended_len // 3]
    gauss_centre, gauss_sigma = np.mean(x_extended), gauss_width / 2.35482
    y_gauss_peak_1 = gauss_height_1 * np.exp(
        -0.5 * ((x_extended - gauss_centre_1) / gauss_sigma) ** 2)
    y_gauss_peak_2 = gauss_height_2 * np.exp(
        -0.5 * ((x_extended - gauss_centre_2) / gauss_sigma) ** 2)
    y_gauss = y_gauss_peak_1 + y_gauss_peak_2

    y_extended = y_linear + y_gauss

    plot_args = [
        {"args": (np.concatenate([x, x_extended]),
                  np.concatenate([y, y_extended]))}
    ]
    utils.create_plot(
        plot_args=plot_args,
        figure_args=dict(figsize=(10, 8)),
        title="Original Spectrum with Linear Regression at the End",
        x_label=r'$Wavenumber\ (cm^{-1})$',
        y_label=r"$Intensity\ (a.u.)$"
    )

    return x_extended, y_linear, y_gauss


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
    # print(len(filtered_minima_idx))

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
    # return r_12

    # utils.create_plot(plot_args=[{"args": (x, y_normalized)},
    #                        {"args": (x[filtered_minima_idx],
    #                                  y_normalized[filtered_minima_idx], 'x')}])
    # plt.show()
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

    n = len(hyper_params.LAM)

    if compute_baseline or hyper_params.BEST_LAM is None:
        # x_extended, y_linear, y_peak = spectrum_linear_expansion(x, y_smoothed)
        # y_extended = y_linear + y_peak
        # y_full = np.concatenate([y_smoothed, y_extended])
        # x_full = np.concatenate([x, x_extended])
        # plot_args = [
        #     {"args": (x_full, -y_full), "kwargs": dict(label="Original spectrum")}]
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
        # extended_length = len(x_extended)

        with multiprocessing.Pool(processes=8) as pool:
            results = pool.map(baseline_wrapper,
                               [{'baseline_fitter': baseline_fitter, 'data': y_smoothed,
                                 'lam': lam, 'max_iter': 1000,
                                 'tol': 7e-4} for lam in lambda_vals])

        for i, result in enumerate(results):
            if result is not None:
                lam = lambda_vals[i]
                baseline_fits[lam] = result
                # baseline_extended = result[-extended_length:]
                rmse_vals[lam] = baseline_r_12_metric_calculation(x, y_smoothed, result,
                                                                  normalization_region)
                # rmse_vals[lam] = rmse(y_obs=y_linear, y_pred=baseline_extended)
                # plot_args.append({"args": (x_full, -result),
                #                   "kwargs": {"label": rf"$\lambda$={lam}"}})
                plot_args.append({"args": (x, -(result - offset)),
                                  "kwargs": {"label": rf"$\lambda$={lam}"}})

        min_rmse_lam = min(rmse_vals, key=rmse_vals.get)
        print(f"\nR\u00b9\u00b2 metric values for candidate lambdas: "
              f"\n{json.dumps(utils.serialize_json(rmse_vals), indent=1)}")
        print(f"Optimal lambda value: {min_rmse_lam}")
        # best_baseline = baseline_fits[min_rmse_lam][: len(y)]
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
    n_peaks = len(peaks)

    # setting bounds for faster convergence
    peak_params, optimal_params = [], []
    multiplier = 2 * np.sqrt(2 * np.log(2))
    for i in range(len(peaks)):
        peak_region = slice(left_bases[i], right_bases[i])
        area = np.trapezoid(
            y[peak_region] - np.minimum(y[left_bases[i]], y[right_bases[i]]),
            x[peak_region])
        # height * width
        # A0 = y[peaks[i]]
        # A0 = max(area, y[peaks[i]])
        A0 = area
        mu0 = x[peaks[i]]
        # sd = width / 6 -> (3 on either side of the mean)
        # sigma0 = (x[right_bases[i]] - x[left_bases[i]]) / 4
        sigma0 = (x[right_bases[i]] - x[left_bases[i]]) / 6
        fwhm0 = sigma0 * multiplier
        # fwhm0 = 0.1
        gamma0 = sigma0
        ratio0 = 0.5
        # peak_params.append([A0, mu0, sigma0, gamma0])
        peak_params.append([ratio0, A0, mu0, fwhm0])

        # mu0_min = x[left_bases[i]] - 0.3
        # mu0_max = x[right_bases[i]] + 0.3
        mu0_min = x[left_bases[i]]
        mu0_max = x[right_bases[i]]
        l_idx = np.searchsorted(x, mu0_min, "left")
        r_idx = np.searchsorted(x, mu0_max, "right")
        x_local, y_local = x[l_idx: r_idx], y[l_idx: r_idx]
        area_max = y[peaks[i]]

        # bounds.extend([(1e-11, None), (mu0_min, mu0_max), (1e-5, 1), (1e-5, 1)])
        bounds.extend([(0, 1), (1e-11, area_max), (mu0_min, mu0_max), (1e-8, 1)])

        if least_sqaures_fit:
            min_result = least_squares(
                fun=utils.loss_function_vector,
                jac=utils.jacobian_least_squares,
                x0=peak_params[-1],
                args=(x_local, y_local, 1),
                ftol=1e-15,
                xtol=1e-15,
                gtol=1e-15,
                max_nfev=10000,
                # bounds=([1e-10, mu0_min, 1e-5, 1e-5],
                #         [np.inf, mu0_max, 1, 1]),
                bounds=([0, 1e-15, mu0_min, 1e-8],
                        [1, area_max, mu0_max, 1])
            )
        else:
            min_result = minimize(
                fun=utils.loss_function,
                jac=utils.jacobian_of_loss,
                x0=peak_params[-1],
                args=(x_local, y_local, 1),
                method='L-BFGS-B',
                # bounds=[(1e-10, None), (x[left_bases[i]], x[right_bases[i]]), (1e-8, 1),
                #         (1e-8, 1)],
                bounds=[(0, 1), (1e-15, area_max), (mu0_min, mu0_max), (1e-8, 1)],
                options={'maxiter': 10000, 'ftol': 1e-15, 'disp': True,
                         'gtol': 1e-15}
            )

        if not min_result.success:
            raise Exception(f"Optimization warning: {min_result}")

        optimal_params.append(min_result.x)

    # init_params = np.ravel(peak_params)
    #
    # print(f"total number of parameters: {init_params.shape}")
    #
    # def callback(params):
    #     print("Callback params:", params[:5])
    #
    # if least_sqaures_fit:
    #     upper_bounds, lower_bounds = [], []
    #     for b in bounds:
    #         ub = b[1] if b[1] is not None else np.inf
    #         lb = b[0] if b[0] is not None else -np.inf
    #         upper_bounds.append(ub)
    #         lower_bounds.append(lb)
    #
    #     min_result = least_squares(
    #         fun=utils.loss_function_vector,
    #         jac=utils.jacobian_least_squares,
    #         x0=init_params,
    #         args=(x, y, n_peaks),
    #         bounds=(lower_bounds, upper_bounds),
    #         callback=callback
    #     )
    # else:
    #     min_result = minimize(
    #         fun=utils.loss_function,
    #         jac=utils.utils.jacobian_of_loss,
    #         x0=init_params,
    #         args=(x, y, n_peaks),
    #         method='L-BFGS-B',
    #         bounds=bounds,
    #         options={'maxiter': 1000, 'ftol': 1e-5, 'disp': True},
    #         callback=callback
    #     )
    #
    # if not min_result.success:
    #     raise Exception(f"Optimization warning: {min_result}")
    #
    # optimal_params = min_result.x
    y_bkg = np.zeros_like(x)
    optimal_params = np.array(optimal_params)
    voigt_params = optimal_params.reshape(n_peaks, 4)

    voigt_p_list = []
    y_peak = np.zeros_like(x)
    no_fits, noise = [], []
    for (ratio, amp, centre, fwhm) in voigt_params:
        y_peak += utils.pseudo_voigt_profile(x, ratio, amp, centre, fwhm)
        v_params = {
            "area": amp,
            "centre": centre,
            "ratio": ratio,
            "fwhm": fwhm,
            "discard": False
        }
        print(v_params)
        # fwhm = get_full_width_at_half_max(sigma, gamma)
        # print(fwhm)
        # v_params["fwhm"] = fwhm
        if amp == 0:
            v_params["discard"] = True
            no_fits.append(centre)
        elif fwhm < 0.08 or fwhm > 0.27:
            v_params["discard"] = True
            noise.append((centre, float(fwhm)))
        voigt_p_list.append(v_params)

    y_fit = y_bkg + y_peak
    end_time = time.time()

    print(f"\n{len(no_fits)} drops discarded due to area being 0: \n{no_fits}")
    print(f"\n{len(noise)} drops discarded due to FWHM < 0.08 or FWHM > 0.25: \n{noise}")

    print(f"total time taken for fitting {round(end_time - start_time, 3)} seconds")

    plot_args = [
        {"args": (x, y), "kwargs": dict(label="Baseline-corrected spectrum")},
        {"args": (x, y_fit), "kwargs": dict(label="Fitted Voigt profiles",
                                            linestyle='--')}
        # {"args": (x, -y_peak),
        #  "kwargs": dict(label="Voigt peaks", linestyle='--')},
        # {"args": (x[peaks], y[peaks], 'x'),
        #  "kwargs": dict(label="Original prominent peaks")}
    ]

    # inset plot settings and args
    # inset_settings = dict(width="100%", height="100%", loc="upper left",
    #                       bbox_to_anchor=(0.1, 0.65, 0.5, 0.35),
    #                       bbox_transform=True)
    # inset_idx = np.where((x > 6512) & (x <= 6528))[0]
    # inset_peaks = np.intersect1d(inset_idx, peaks)
    #
    # inset_args = [{"args": (x[inset_idx], -y[inset_idx])},
    #               {"args": (x[inset_idx], -y_fit[inset_idx])},
    #               {"args": (x[inset_peaks], -y[inset_peaks], 'x')}]

    utils.create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)), legend=True,
                      title=f"Region {n_region}: Voigt profiles fit for {filename}",
                      x_label=r'$Wavenumber\ (cm^{-1})$',
                      y_label=r"$Absorption\ (cm^{-1})$")
    utils.create_plot(plot_args=[{"args": (x, y - y_fit)}])
    # inset_args=inset_args, inset_settings=inset_settings)
    return y_bkg, y_peak, voigt_p_list


def get_full_width_at_half_max(sigma: float, gamma: float):
    fwhm_gauss = 2 * sigma * np.sqrt(2 * np.log(2))
    fwhm_loren = 2 * gamma
    return (0.5343 * fwhm_loren) + np.sqrt(0.2169 * fwhm_loren ** 2 + fwhm_gauss ** 2)


def hitran_matching_process(peak_params: List[dict],
                            x: Union[pd.Series, npt.NDArray[float]],
                            y: Union[pd.Series, npt.NDArray[float]],
                            peaks: npt.NDArray[int], region: int, filename: str,
                            y_peaks: npt.NDArray[float] = None):
    nu_exp = np.array([v["centre"] for v in peak_params])
    diff = abs(x[peaks] - nu_exp)
    utils.create_plot(plot_args=[{"args": (x[peaks], diff)}],
                      title="Voigt fit centre differences against observed wavenumbers",
                      x_label=r'$Wavenumber\ (cm^{-1})$', y_label='residuals',
                      scatter=True)

    nu_hitran_co2, nu_hitran_h2o = fetch_data()
    mean_co2_df, mean_h2o_df = get_regional_mean_strength(co2_df=nu_hitran_co2,
                                                          h2o_df=nu_hitran_h2o)

    # match_dict_co2, match_dict_h2o = {}, {}
    strong_co2_lines, weak_co2_lines = {}, {}
    strong_h2o_lines, weak_h2o_lines = {}, {}
    no_matches, strong_ambiguous, weak_ambiguous = [], [], []
    for i, nu in enumerate(nu_exp):
        if peak_params[i]["discard"]:
            continue
        # print(nu, x[peaks[i]])
        low = nu - CONFIG.RESOLUTION
        high = nu + CONFIG.RESOLUTION

        start_idx_co2 = np.searchsorted(nu_hitran_co2["wavenumber"], low, "left")
        end_idx_co2 = np.searchsorted(nu_hitran_co2["wavenumber"], high, "right")
        start_idx_h2o = np.searchsorted(nu_hitran_h2o["wavenumber"], low, "left")
        end_idx_h2o = np.searchsorted(nu_hitran_h2o["wavenumber"], high, "right")

        potential_matches_co2 = nu_hitran_co2[start_idx_co2: end_idx_co2]
        potential_matches_h2o = nu_hitran_h2o[start_idx_h2o: end_idx_h2o]
        strong_line_co2, strong_line_h2o = False, False

        if potential_matches_co2.empty and potential_matches_h2o.empty:
            no_matches.append((i, nu))
        else:
            if not potential_matches_co2.empty:
                max_strength_idx_co2 = potential_matches_co2["strength"].idxmax()
                nearest_nu_hitran_co2 = potential_matches_co2["wavenumber"][
                    max_strength_idx_co2]
                region_nu_idx = np.searchsorted(mean_co2_df["wavenumber"],
                                                nearest_nu_hitran_co2, "left") - 1
                line_strength = potential_matches_co2["strength"][max_strength_idx_co2]
                if round(nu, 4) == round(6250.99923687, 4):
                    print(nu)
                    print(potential_matches_co2)
                    print(mean_co2_df.iloc[region_nu_idx])
                if line_strength >= mean_co2_df["mean_strength"].iloc[region_nu_idx]:
                    strong_line_co2 = True
                    strong_co2_lines[i] = [nu, nearest_nu_hitran_co2, line_strength]
                    peak_params[i]["line_strength"] = line_strength
                else:
                    weak_co2_lines[i] = [nu, nearest_nu_hitran_co2, line_strength]
            if not potential_matches_h2o.empty:
                max_strength_idx_h2o = potential_matches_h2o["strength"].idxmax()
                nearest_nu_hitran_h2o = potential_matches_h2o["wavenumber"][
                    max_strength_idx_h2o]
                region_nu_idx = np.searchsorted(mean_h2o_df["wavenumber"],
                                                nearest_nu_hitran_h2o, "left") - 1
                line_strength = potential_matches_h2o["strength"][max_strength_idx_h2o]
                if line_strength >= mean_h2o_df["mean_strength"].iloc[region_nu_idx]:
                    strong_line_h2o = True
                    strong_h2o_lines[i] = [nu, nearest_nu_hitran_h2o, line_strength]
                    peak_params[i]["line_strength"] = line_strength
                else:
                    weak_h2o_lines[i] = [nu, nearest_nu_hitran_h2o, line_strength]
            if not potential_matches_h2o.empty and not potential_matches_co2.empty:
                if strong_line_co2 and strong_line_h2o:
                    print(f"Strong ambiguous - CO2: {strong_co2_lines[i][0]} "
                          f"{strong_co2_lines[i][2]}, H2O: {strong_h2o_lines[i][0]} "
                          f"{strong_h2o_lines[i][2]}")
                    # print(strong_co2_lines[i][2],
                    #       strong_h2o_lines[i][2])
                    del strong_co2_lines[i]
                    del strong_h2o_lines[i]
                    strong_ambiguous.append(i)
                elif strong_line_co2:
                    if weak_h2o_lines[i][2] >= strong_co2_lines[i][2]:
                        print(
                            f"strong co2 val: {strong_co2_lines[i][2]}, {strong_co2_lines[i][0]}")
                        print(
                            f"weak h20 val: {weak_h2o_lines[i][2]}, {weak_h2o_lines[i][0]}")
                        del strong_co2_lines[i]
                    del weak_h2o_lines[i]
                elif strong_line_h2o:
                    # print(
                    #     f"strong h2o val: {strong_h2o_lines[i][2]}, {strong_h2o_lines[i][0]}")
                    # print(f"weak co2 val: {weak_co2_lines[i][2]}, {weak_co2_lines[i][0]}")
                    # print(f"strong h20 val: {strong_h2o_lines[i][2]}, {strong_h2o_lines[i][0]}")
                    del weak_co2_lines[i]
                else:
                    del weak_co2_lines[i]
                    del weak_h2o_lines[i]
                    weak_ambiguous.append(i)

    indices_co2_strong = list(strong_co2_lines.keys())
    indices_h2o_strong = list(strong_h2o_lines.keys())
    indices_co2_weak = list(weak_co2_lines.keys())
    indices_h2o_weak = list(weak_h2o_lines.keys())
    unmatch_indices = [val[0] for val in no_matches]
    unmatched_peaks = peaks[unmatch_indices]

    strong_co2_lines_keys = sorted(strong_co2_lines, key=strong_co2_lines.get)
    for key in strong_co2_lines_keys:
        print(strong_co2_lines[key][0], strong_co2_lines[key][1],
              f"{strong_co2_lines[key][2]:.3e}")

    print(f"\nStrong CO2 absorption peaks matched with HITRAN's nu values with a "
          f"tolerance of +- {CONFIG.RESOLUTION}: \n{strong_co2_lines}")
    print(f"\nStrong H2O absorption peaks matched with HITRAN's nu values with a "
          f"tolerance of +- {CONFIG.RESOLUTION}: \n{strong_h2o_lines}")
    print(f"\nWeak CO2 absorption peaks matched with HITRAN's nu values with a "
          f"tolerance of +- {CONFIG.RESOLUTION}: \n{weak_co2_lines}")
    print(f"\nWeak H2O absorption peaks matched with HITRAN's nu values with a "
          f"tolerance of +- {CONFIG.RESOLUTION}: \n{weak_h2o_lines}")
    print(f"\nUnmatched absorption peaks with a tolerance of "
          f"+= {CONFIG.RESOLUTION}: \n{no_matches}")

    # overlap_peak_keys = [i for i in match_indices_co2 if i in match_indices_h2o]
    # unique_co2_keys = [i for i in match_indices_co2 if i not in match_indices_h2o]
    # unique_h2o_keys = [i for i in match_indices_h2o if i not in match_indices_co2]
    unique_co2_peaks = peaks[indices_co2_strong]
    unique_h2o_peaks = peaks[indices_h2o_strong]
    unique_co2_voigt_vals, unique_co2_x_vals = [], []
    unique_h2o_voigt_vals, unique_h2o_x_vals = [], []
    unique_co2_amp_vals = []
    for key in indices_co2_strong:
        params = peak_params[key]
        y_voigt = utils.pseudo_voigt_profile(x=params["centre"], ratio=params["ratio"],
                                             amplitude=params["area"],
                                             centre=params['centre'],
                                             fwhm=params['fwhm'])
        unique_co2_voigt_vals.append(y_voigt)
        unique_co2_amp_vals.append(params['area'])
        unique_co2_x_vals.append(params["centre"])
    unique_co2_voigt_vals = np.array(unique_co2_voigt_vals)

    for key in indices_h2o_strong:
        params = peak_params[key]
        y_voigt = utils.pseudo_voigt_profile(x=params["centre"], ratio=params["ratio"],
                                             amplitude=params["area"],
                                             centre=params['centre'],
                                             fwhm=params['fwhm'])
        unique_h2o_voigt_vals.append(y_voigt)
        unique_h2o_x_vals.append(params["centre"])
    unique_h2o_voigt_vals = np.array(unique_h2o_voigt_vals)

    print(f"\n{len(indices_co2_strong)} strong unique CO2 peaks found: "
          f"\n{nu_exp[indices_co2_strong]}")
    print(f"\n{len(indices_co2_weak)} weak unique CO2 peaks found: "
          f"\n{nu_exp[indices_co2_weak]}")
    print(f"\n{len(indices_h2o_strong)} strong unique H2O peaks found: "
          f"\n{nu_exp[indices_h2o_strong]}")
    print(f"\n{len(indices_h2o_weak)} weak unique H2O peaks found: "
          f"\n{nu_exp[indices_h2o_weak]}")
    print(f"\n{len(strong_ambiguous)} strong ambiguous H2O and CO2 peaks found: "
          f"\n{nu_exp[strong_ambiguous]}")
    print(f"\n{len(weak_ambiguous)} weak ambiguous H2O and CO2 peaks found: "
          f"\n{nu_exp[weak_ambiguous]}")
    print(f"\n{len(no_matches)} unmatched peaks found: \n{nu_exp[no_matches]}")

    plot_args = [
        {"args": (x, y), "kwargs": {"label": "Absorption Coefficient"}},
        {"args": (unique_co2_x_vals, unique_co2_voigt_vals, 'gx'),
         "kwargs": {"ms": 8, "label": "CO2 absorption peaks"}},
        {"args": (unique_h2o_x_vals, unique_h2o_voigt_vals, 'rx'),
         "kwargs": {"ms": 8, "label": "H2O absorption peaks"}},
        {"args": (unique_co2_x_vals, unique_co2_amp_vals, 'go'),
         "kwargs": {"ms": 8, "label": "CO2 absorption peak area"}},
        {"args": (x, y_peaks),
         "kwargs": {"label": "Estimated Voigt profiles", "linestyle": "--",
                    "alpha": 0.8}}
    ]

    # inset plot settings and args
    # inset_settings = dict(width="100%", height="100%", loc="upper left",
    #                       bbox_to_anchor=(0.1, 0.65, 0.5, 0.35),
    #                       bbox_transform=True)
    # inset_idx = np.where((x > 6231) & (x <= 6255))[0]
    # inset_peaks_co2 = np.intersect1d(inset_idx, unique_co2_peaks)
    # inset_peaks_h2o = np.intersect1d(inset_idx, unique_h2o_peaks)
    #
    # inset_args = [{"args": (x[inset_idx], -y[inset_idx])},
    #               {"args": (x[inset_idx], -y_peaks[inset_idx]),
    #                "kwargs": dict(linestyle="--")},
    #               {"args": (x[inset_peaks_co2], -y[inset_peaks_co2], 'gx'),
    #                "kwargs": dict(ms=7)},
    #               {"args": (x[inset_peaks_h2o], -y[inset_peaks_h2o], 'rx'),
    #                "kwargs": dict(ms=7)}]

    utils.create_plot(plot_args=plot_args, figure_args={"figsize": (10, 8)},
                      title=f"Region {region}: Spectrum fit with Voigt centres matched "
                            f"against HITRAN with a tolerance of +- {CONFIG.RESOLUTION} for {filename}",
                      x_label=r'$Wavenumber\ (cm^{-1})$',
                      y_label=r"$Absorption\ Coefficient\ (cm^{-1})$", legend=True,
                      # inset_settings=inset_settings, inset_args=inset_args,
                      y_lim=(None, None))

    nu_obs = np.array([val[0] for val in strong_co2_lines.values()])
    nu_hit = np.array([val[1] for val in strong_co2_lines.values()])
    # plot_args = [{"args": (nu_obs, (nu_obs - nu_hit))}]
    # utils.create_plot(plot_args=plot_args, figure_args={"figsize": (10, 8)},
    #             title=f"Region {region}: Nu residual plot with a match tolerance of +- {CONFIG.RESOLUTION}",
    #             x_label="Nu observed (Voigt centre)", y_label="Residual", legend=False,
    #             scatter=True)
    print(f"\n mean residual value: \n{np.mean(np.abs(nu_obs - nu_hit))}")
    return peak_params, indices_co2_strong, indices_h2o_strong, strong_ambiguous, unmatch_indices


def concentration_estimation_process(peak_params: List[dict], co2_keys: list[int],
                                     h2o_keys: list[int], x: pd.Series, y: pd.Series,
                                     peaks: npt.NDArray):
    conc_per_peak_co2 = [peak_params[key]["area"] / peak_params[key]["line_strength"] \
                         for key in co2_keys]
    conc_per_peak_h2o = [peak_params[key]["area"] / peak_params[key]["line_strength"] \
                         for key in h2o_keys]
    conc_per_peak_co2_2, conc_per_peak_h2o_2 = [], []

    for key in co2_keys:
        params = peak_params[key]
        absorp_sigma = params["line_strength"] / params["fwhm"]
        # alpha = y[peaks[key]]
        alpha = utils.pseudo_voigt_profile(x=params['centre'], ratio=params['ratio'],
                                           amplitude=params['area'],
                                           centre=params['centre'],
                                           fwhm=params['fwhm'])
        conc_per_peak_co2_2.append(alpha / absorp_sigma)
        print(
            f"CO2: line_strength: {params['line_strength']:.3e}, "
            f"centre: {round(params['centre'], 2)}, "
            f"FWHM conc.: {(alpha / absorp_sigma):.3e}, "
            f"Area conc.: {(peak_params[key]["area"] /
                            peak_params[key]["line_strength"]):.3e}")
    for key in h2o_keys:
        params = peak_params[key]
        # alpha = y[peaks[key]]
        alpha = utils.pseudo_voigt_profile(x=params['centre'], ratio=params['ratio'],
                                           amplitude=params['area'],
                                           centre=params['centre'],
                                           fwhm=params['fwhm'])
        absorp_sigma = params["line_strength"] / params["fwhm"]
        conc_per_peak_h2o_2.append(alpha / absorp_sigma)
        print(
            f"H2O: line_strength: {params['line_strength']:.3e}, "
            f"centre: {round(params['centre'], 2)}, "
            f"FWHM conc.: {(alpha / absorp_sigma):.3e}, "
            f"Area conc.: {(peak_params[key]["area"] /
                            peak_params[key]["line_strength"]):.3e}")
    return conc_per_peak_co2, conc_per_peak_h2o, conc_per_peak_co2_2, conc_per_peak_h2o_2

# plt.show()
# from sklearn.linear_model import LinearRegression
# y = [val[1] for val in match_dict.values()]
# x = [val[0] for val in match_dict.values()]
# X = np.array(x)[:, np.newaxis]
# Y = np.array(y)[:, np.newaxis]
# model = LinearRegression().fit(X, Y)
# y_hat = model.predict(X)
# print(model.score(X, Y))
# plt.scatter(x, y)
# plt.plot(x, y_hat, linestyle="--")
# plt.figure()
# plt.scatter(x, Y - y_hat)
# plt.show()

# if __name__ == "__main__":
#     import pickle
#
#     with open("/Users/lakshya/PycharmProjects/FTIRGasDensityEstimation/voigt_params",
#               "r") as f:
#         peak_params = json.load(f)
#     with open("/Users/lakshya/PycharmProjects/FTIRGasDensityEstimation/peak_indices",
#               "rb") as f:
#         peaks = pickle.load(f)
#     region_df = pd.read_csv(
#         "/Users/lakshya/PycharmProjects/FTIRGasDensityEstimation/test.csv")
#     peaks = np.array(peaks, dtype=int)
#     hitran_matching_process(peak_params, region_df["wavenumber"], -region_df["intensity"],
#                             peaks)
