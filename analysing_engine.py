from typing import Union
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from pybaselines import Baseline
import numpy.typing as npt
import time
from typing import List, Tuple, Any
from scipy.optimize import minimize

from utils import StatManager, create_plot, create_splines_pipeline, loss_function, \
    jacobian_of_loss, voigt_profile


def find_peaks_and_filter(x: Union[pd.Series, np.ndarray],
                          y: Union[pd.Series, np.ndarray], y_avgs: np.ndarray, window_len,
                          peak_threshold):
    """Find peaks and filter. Prominence based on distance from local averages"""
    wlen = len(x[x <= x[0] + window_len])
    peaks, _ = find_peaks(-y, distance=wlen)

    diffs = y_avgs[peaks] - y[peaks]
    diffs = diffs.to_numpy() if isinstance(diffs, pd.Series) else diffs
    peak_thresh = np.quantile(diffs, peak_threshold)
    filtered_p_idx = np.where(diffs >= peak_thresh)[0]
    filtered_peaks = peaks[filtered_p_idx]
    filtered_p_proms = diffs[filtered_p_idx]
    return filtered_peaks, filtered_p_proms


def peak_finding_process(x: Union[pd.Series, np.ndarray],
                         y: Union[pd.Series, np.ndarray],
                         hyper_params: dict, n_region: int,
                         plots: bool = True):
    w_len = hyper_params["AVG_WINDOW_SIZE"]
    x_stat_manager = StatManager(window_size=w_len)
    local_avgs = x_stat_manager.find_statistic_symmetrically(x, y)
    peaks, prominences = find_peaks_and_filter(x, y, local_avgs,
                                               hyper_params["PEAK_WLEN"],
                                               hyper_params["PEAK_PROMINENCE"])
    print(
        f"\nNumber of prominent peaks - {len(peaks)}; {round(len(peaks) / len(x) * 100, 2)} %")

    left_bases, right_bases, to_remove = [], [], []
    dy = np.gradient(y)
    zero_crossings_maxima = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0] + 1
    for i, peak in enumerate(peaks):
        l_bs = zero_crossings_maxima[zero_crossings_maxima < peak]
        r_bs = zero_crossings_maxima[zero_crossings_maxima > peak]
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

    if plots:
        plot_args = [{"args": (x, y)}, {"args": (x[peaks], y[peaks], 'x'),
                                        "kwargs": {"label": "Inverted peak"}},
                     {"args": (x[peaks], prominences, 'x'),
                      "kwargs": {"label": "Peak prominence"}},
                     {"args": (x, local_avgs),
                      "kwargs": {"label": f"local avg, wlen: {w_len}"}},
                     {"args": (x[left_bases], y[left_bases], 'ro'),
                      "kwargs": {"label": "left end point", "ms": 4}},
                     {"args": (x[right_bases], y[right_bases], 'go'),
                      "kwargs": {"label": "right end point", "ms": 4}}
                     ]
        create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)), legend=True,
                    title=f"Region {n_region}: prominent peaks", x_label="Wavenumber",
                    y_label="Intensity")
    return peaks, left_bases, right_bases


def peak_and_knot_placement_process(x: pd.Series, peaks: np.ndarray,
                                    left_bases: np.ndarray, right_bases: np.ndarray,
                                    hyper_params: dict, n_region: int,
                                    plots: bool = True):
    knot_vector = []
    non_peak_regions = []
    for i in range(len(peaks)):
        # region before the first peak
        if i == 0 and left_bases[i] > 0:
            non_peak_regions.append((0, left_bases[i] - 1))
        if i + 1 < len(peaks):
            if right_bases[i] == left_bases[i + 1] or right_bases[i] + 1 == left_bases[
                i + 1]:
                continue
            non_peak_regions.append((right_bases[i] + 1, left_bases[i + 1] - 1))
        if i == len(peaks) - 1 and right_bases[i] < len(x) - 1:
            non_peak_regions.append((right_bases[i] + 1, len(x) - 1))

    # determining the amount of knots a region gets based on its proportional size
    non_p_lengths = np.array([_np[1] - _np[0] + 1 for _np in non_peak_regions], dtype=int)
    total_np_length = sum(non_p_lengths)
    total_knots = hyper_params["NON_PEAK_KNOTS"]

    # extra logic to handle regions here that are quite small compared to the total length
    raw_allocations = total_knots * (non_p_lengths / total_np_length)
    floors = np.floor(raw_allocations).astype(int)
    remainders = raw_allocations - floors

    allocated = floors.copy()
    remaining = total_knots - allocated.sum()

    if remaining > 0:
        extra_indices = np.argsort(remainders)[::-1][:remaining]
        allocated[extra_indices] += 1

    # uniformly place knots in the non-peak regions
    for i, _np in enumerate(non_peak_regions):
        n_knots = allocated[i]
        if n_knots != 0:
            x_positions = np.linspace(x.iloc[_np[0]], x.iloc[_np[1]], n_knots)
            knot_vector.extend(x_positions)
    if left_bases[0] != 0:
        knot_vector.append(x.iloc[0])
    if right_bases[-1] != len(x) - 1:
        knot_vector.append(x.iloc[-1])

    knot_vector = np.sort(np.unique(knot_vector))
    print(f"\nThe total number of knots placed: {len(knot_vector)}")
    print(f"\nWavenumbers where knots are placed: \n{knot_vector}")
    return knot_vector, non_peak_regions


def baseline_estimation_process(x: Union[pd.Series, np.ndarray],
                                y: Union[pd.Series, np.ndarray], hyper_params: dict,
                                n_region: int):
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    baseline_fitter = Baseline(x, assume_sorted=True)
    baseline_fit, _ = baseline_fitter.mixture_model(y, num_knots=hyper_params["N_KNOTS"],
                                                    tol=1e-6, max_iter=3000,
                                                    lam=hyper_params["LAM"])
    baseline_fit_1, _ = baseline_fitter.arpls(y, lam=9e11, max_iter=1000, tol=1e-6)
    plot_args = [
        {"args": (x, -y), "kwargs": dict(label="Estimated curve")},
        {"args": (x, -baseline_fit), "kwargs": dict(label="Spline baseline")},
        {"args": (x, -baseline_fit_1), "kwargs": dict(label="ARPLS baseline")},
        {"args": (x, baseline_fit - y),
         "kwargs": dict(label="Baseline corrected spectrum")}
    ]
    create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)),
                legend=True, title=f"Region {n_region}: Baseline with the original data",
                x_label="Wavenumber", y_label="Estimated Intensity")
    return baseline_fit


def curve_and_peak_fitting_process(x: Union[pd.Series, np.ndarray],
                                   y: Union[pd.Series, np.ndarray],
                                   peaks: npt.NDArray[np.int64],
                                   left_bases: npt.NDArray[np.int64],
                                   right_bases: npt.NDArray[np.int64],
                                   knot_vector: np.ndarray, baseline_corrected: bool,
                                   n_region: int):
    start_time = time.time()
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    x_ready = x[:, np.newaxis]
    splines_pipeline = create_splines_pipeline(knot_vector, 3, 'continue')
    X_basis = splines_pipeline.fit_transform(x_ready)
    if not baseline_corrected:
        n_spline_coeffs = X_basis.shape[1]
        splines_beta_init = np.linalg.lstsq(X_basis, y, rcond=None)[0]
        bounds: List[Tuple[Any, Any]] = [(None, None)] * n_spline_coeffs
    else:
        n_spline_coeffs = 0
        splines_beta_init = []
        bounds: List[Tuple[Any, Any]] = []
    n_peaks = len(peaks)

    # setting bounds for faster convergence
    peak_params = []
    for i in range(len(peaks)):
        # height * width
        A0 = y[peaks[i]] * (x[right_bases[i]] - x[left_bases[i]])
        mu0 = x[peaks[i]]
        # sd = width / 6 -> (3 on either side of the mean)
        sigma0 = (x[right_bases[i]] - x[left_bases[i]]) / 6
        gamma0 = sigma0
        peak_params.append([A0, mu0, sigma0, gamma0])

        center_min = x[left_bases[i]] - 0.3
        center_max = x[right_bases[i]] + 0.3
        bounds.extend([(0, None), (center_min, center_max), (1e-7, None), (1e-7, None)])

    init_params = np.concatenate([splines_beta_init, np.ravel(peak_params)])

    print(f"length of the knot vector: {len(knot_vector)}")
    print(f"shape of the spline basis: {X_basis.shape}")
    print(f"the number of spline coefficients are: {n_spline_coeffs}")
    print(f"shape of the splines beta coefficients is: {len(splines_beta_init)}")
    print(f"total number of parameters: {init_params.shape}")

    min_result = minimize(
        fun=loss_function,
        jac=jacobian_of_loss,
        x0=init_params,
        args=(X_basis, x, y, n_peaks, baseline_corrected),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-6}
    )

    if not min_result.success:
        print(f"Optimization warning: {min_result}")

    optimal_params = min_result.x
    if not baseline_corrected:
        spline_coeffs = optimal_params[: n_spline_coeffs]
        y_bkg = X_basis @ spline_coeffs
    else:
        y_bkg = np.zeros_like(x)
    voigt_params = optimal_params[n_spline_coeffs:].reshape(n_peaks, 4)

    voigt_p_list = []
    y_peak = np.zeros_like(x)
    for (A, mu, sigma, gamma) in voigt_params:
        y_peak += voigt_profile(x, A, mu, sigma, gamma)
        voigt_p_list.append({
            "area": A,
            "center": mu,
            "sigma": sigma,
            "gamma": gamma
        })
    y_fit = y_bkg + y_peak
    end_time = time.time()

    print(f"total time taken for fitting {round(end_time - start_time, 3)} seconds")

    plot_args = [{"args": (x, -(y - y_bkg)), "kwargs": dict(label="original spectrum")},
                 {"args": (x, -(y_fit - y_bkg)),
                  "kwargs": dict(label="bkg + voigt peaks", linestyle='--')},
                 {"args": (x[peaks], -(y - y_bkg)[peaks], 'x'),
                  "kwargs": dict(label="prominent peaks")}]
    create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)), legend=True,
                title=f"Region {n_region}: Fit with a Splines background and Voigt "
                      f"peaks. Prior basline correction: {baseline_corrected}",
                x_label="Wavenumber",
                y_label="Intensity")

    return y_bkg, y_peak
