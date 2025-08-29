import matplotlib.pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.pyplot import legend
from scipy.ndimage import label
from scipy.signal import find_peaks
from pybaselines import Baseline
import numpy.typing as npt
import time
from typing import List, Tuple, Any, Union
from scipy.optimize import minimize
from scipy.sparse import dia_matrix, diags, linalg
from scipy.sparse.linalg import spsolve
from pathlib import Path
from scipy.interpolate import BSpline, CubicSpline
from scipy.stats import t

from utils import find_statistic_symmetrically, create_plot, create_splines_pipeline, \
    loss_function, jacobian_of_loss, voigt_profile, molecules_per_cm3_to_ppm, \
    calculate_standard_errors_pspline
from HITRAN import fetch_data
from config import CONFIG


def find_peaks_and_filter(x: Union[pd.Series, np.ndarray],
                          y: Union[pd.Series, np.ndarray], baseline: np.ndarray[float],
                          window_len, peak_threshold):
    """Find peaks and filter. Prominence based on distance from local averages"""
    wlen = len(x[x <= x[0] + window_len])
    peaks, _ = find_peaks(-y, distance=wlen)

    diffs = baseline[peaks] - y[peaks]
    diffs = diffs.to_numpy() if isinstance(diffs, pd.Series) else diffs
    peak_thresh = np.quantile(diffs, peak_threshold)
    filtered_p_idx = np.where(diffs >= peak_thresh)[0]
    filtered_peaks = peaks[filtered_p_idx]
    filtered_p_proms = diffs[filtered_p_idx]
    return filtered_peaks, filtered_p_proms


def peak_finding_process(x: pd.Series,
                         y: pd.Series,
                         hyper_params: dict, baseline: np.ndarray[float], n_region: int,
                         filename: str, plots: bool = True):
    w_len = hyper_params["AVG_WINDOW_SIZE"]
    # local_avgs = find_statistic_symmetrically(x, y, w_len, assume_sorted=True)
    peaks, prominences = find_peaks_and_filter(x, y, baseline,
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
        prominences = np.delete(prominences, to_remove)
    left_bases = np.array(left_bases, dtype=int)
    right_bases = np.array(right_bases, dtype=int)

    if plots:
        plot_args = [{"args": (x, y)}, {"args": (x[peaks], y[peaks], 'x'),
                                        "kwargs": {"label": "Absorption minimum"}},
                     # {"args": (x[peaks], prominences, 'x'),
                     #  "kwargs": {"label": "Peak prominence"}},
                     {"args": (x, baseline),
                      "kwargs": {"label": f"Baseline estimate", "color": "red"}},
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
                      {"args": (x[inset_idx], baseline[inset_idx]),
                       "kwargs": {"color": "red"}},
                      {"args": (x[inset_left_bases], y[inset_left_bases], 'ro'),
                       "kwargs": {"ms": 4}},
                      {"args": (x[inset_right_bases], y[inset_right_bases], 'go'),
                       "kwargs": {"ms": 4}}]

        create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)), legend=True,
                    title=f"Region {n_region}: prominent absorption drops for {filename}",
                    x_label=r'$Wavenumber\ (cm^{-1})$',
                    y_label=r"$Intensity\ (a.u.)$",
                    y_lim=(None, 0.1),
                    inset_settings=inset_settings, inset_args=inset_args)
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
    return knot_vector, non_peak_regions


def baseline_estimation_process(x: Union[pd.Series, np.ndarray],
                                y: Union[pd.Series, np.ndarray], hyper_params: dict,
                                n_region: int, file_name: str = None):
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    baseline_fitter = Baseline(x, assume_sorted=True)
    n = len(hyper_params["LAM"])
    plot_args = [{"args": (x, -y), "kwargs": dict(label="Original spectrum")}]
    for i in range(n):
        # baseline_fit, _ = baseline_fitter.mixture_model(y,
        #                                                 num_knots=hyper_params["N_KNOTS"],
        #                                                 tol=1e-6, max_iter=3000,
        #                                                 lam=hyper_params["LAM"][i])
        baseline_fit, params = baseline_fitter.pspline_airpls(y,
                                                              lam=hyper_params["LAM"][i],
                                                              tol=hyper_params["TOL"],
                                                              num_knots=hyper_params[
                                                                  "N_KNOTS"])
        if params["tol_history"][-1] > hyper_params["TOL"]:
            raise Exception(f"Baseline fit did not  converge: \n{params["tol_history"]}")
        plot_args.append({"args": (x, -baseline_fit),
                          "kwargs": dict(
                              label=rf"P-spline baseline: $\lambda$={hyper_params["LAM"][i]}, N_KNOTS: {hyper_params["N_KNOTS"]}")})


    bspline = BSpline(params["knots"], params["coef"], 3)

    residuals = y - baseline_fit
    df_eff, se_coeff = calculate_standard_errors_pspline(residuals, params["basis"],
                                                         params["weights"],
                                                         hyper_params["LAM"][0], len(x))


    t_vals = params["coef"] / se_coeff
    plt.figure(figsize=(20, 15))
    plt.scatter(params["coef"], t_vals)
    plt.xlabel("Spline baseline coefficients")
    plt.ylabel("t-statistic")
    plt.xticks(np.linspace(min(params["coef"]), max(params['coef']), 13))
    plt.yticks(np.linspace(min(t_vals), max(t_vals), 13))
    plt.title(f"Student t-statistic values of the estimated baseline "
              f"coefficients with {round(len(x) - df_eff, 2)} degrees of freedom")
    plt.grid()
    plt.figure()

    p_vals = 2 * (1 - t.cdf(np.abs(t_vals), df=(len(x) - df_eff)))
    idx = np.where(p_vals > 0.05)[0]

    plt.figure(figsize=(10, 10))
    plt.scatter(params["coef"], p_vals + 1e-5)
    plt.xlabel("Spline baseline coefficients")
    plt.ylabel("p-values")
    # plt.ylim((0, 0.6))
    plt.xticks(np.linspace(min(params["coef"]), max(params['coef']), 13))
    # plt.yticks(np.linspace(0, 0.75, 5))
    plt.title(f"P-values of the estimated baseline "
              f"coefficients with {round(len(x) - df_eff, 2)} degrees of freedom")
    plt.axhline(0.05, linestyle='--', color='green', label='p-value of 0.05')
    plt.grid()
    plt.legend()
    plt.figure()
    print(p_vals)
    print(t_vals)
    print(se_coeff)
    print(len(x)- df_eff)

    t_crit = t.ppf(1 - 0.05/2, len(x) - df_eff)
    print(t_crit)
    print(params['basis'].shape, df_eff)
    print(idx)
    lower_ci = params['coef'] - t_crit * se_coeff
    upper_ci = params['coef'] + t_crit * se_coeff

    create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)),
                legend={"loc": "upper right"},
                title=f"Baseline estimation with tolerance {hyper_params["TOL"]} for "
                      f"humidity level: {file_name}",
                x_label=r'$Wavenumber\ (cm^{-1})$', y_label=r"$Intensity\ (a.u.)$")

    import statsmodels.api as sm
    # w_residuals = np.sqrt(W) * (y - baseline_fit)
    # print(w_residuals)
    # # plt.figure()
    # fig = sm.qqplot(residuals, line='45')
    # plt.plot()
    # # alpha = ((baseline_fit/y) - 1) * (1 - 0.999/400)
    # # plt.hist(y - baseline_fit, bins=30, density=True, alpha=0.6)
    # # plt.figure()
    # # plt.plot()
    return baseline_fit, bspline


def curve_and_peak_fitting_process(x: Union[pd.Series, np.ndarray],
                                   y: Union[pd.Series, np.ndarray],
                                   peaks: npt.NDArray[np.int64],
                                   left_bases: npt.NDArray[np.int64],
                                   right_bases: npt.NDArray[np.int64],
                                   knot_vector: np.ndarray, baseline_corrected: bool,
                                   filename: str, n_region: int):
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

        bounds.extend([(0, None), (center_min, center_max), (1e-5, 1), (1e-5, 1)])

    init_params = np.concatenate([splines_beta_init, np.ravel(peak_params)])

    print(f"total number of parameters: {init_params.shape}")

    def callback(params):
        print("Callback params:", params[:5])

    min_result = minimize(
        fun=loss_function,
        jac=jacobian_of_loss,
        x0=init_params,
        args=(X_basis, x, y, n_peaks, baseline_corrected),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-5, 'disp': True},
        callback=callback
    )

    if not min_result.success:
        raise Exception(f"Optimization warning: {min_result}")

    optimal_params = min_result.x
    if not baseline_corrected:
        spline_coeffs = optimal_params[: n_spline_coeffs]
        y_bkg = X_basis @ spline_coeffs
    else:
        y_bkg = np.zeros_like(x)
    voigt_params = optimal_params[n_spline_coeffs:].reshape(n_peaks, 4)

    voigt_p_list = []
    y_peak = np.zeros_like(x)
    no_fits, noise = [], []
    for (A, mu, sigma, gamma) in voigt_params:
        y_peak += voigt_profile(x, A, mu, sigma, gamma)
        v_params = {
            "area": A,
            "center": mu,
            "sigma": sigma,
            "gamma": gamma,
            "discard": False
        }
        fwhm = get_full_width_at_half_max(sigma, gamma)
        v_params["fwhm"] = fwhm
        if A == 0:
            v_params["discard"] = True
            no_fits.append(mu)
        elif fwhm < 0.08 or fwhm > 0.25:
            v_params["discard"] = True
            noise.append((mu, float(fwhm)))
        voigt_p_list.append(v_params)

    y_fit = y_bkg + y_peak
    end_time = time.time()

    print(f"\n{len(no_fits)} drops discarded due to area being 0: \n{no_fits}")
    print(f"\n{len(noise)} drops discarded due to FWHM < 0.08 or FWHM > 0.3: \n{noise}")

    print(f"total time taken for fitting {round(end_time - start_time, 3)} seconds")

    plot_args = [
        {"args": (x, -y), "kwargs": dict(label="Baseline-corrected spectrum")},
        {"args": (x, -y_fit), "kwargs": dict(label="Fitted Voigt profiles")},
        # {"args": (x, -y_peak),
        #  "kwargs": dict(label="Voigt peaks", linestyle='--')},
        {"args": (x[peaks], -y[peaks], 'x'),
         "kwargs": dict(label="Original prominent drops")}]

    # inset plot settings and args
    inset_settings = dict(width="100%", height="100%", loc="upper left",
                          bbox_to_anchor=(0.1, 0.65, 0.5, 0.35),
                          bbox_transform=True)
    inset_idx = np.where((x > 6512) & (x <= 6528))[0]
    inset_peaks = np.intersect1d(inset_idx, peaks)

    inset_args = [{"args": (x[inset_idx], -y[inset_idx])},
                  {"args": (x[inset_idx], -y_fit[inset_idx])},
                  {"args": (x[inset_peaks], -y[inset_peaks], 'x')}]

    create_plot(plot_args=plot_args, figure_args=dict(figsize=(10, 8)), legend=True,
                title=f"Region {n_region}: Voigt profiles fit for {filename}",
                x_label=r'$Wavenumber\ (cm^{-1})$', y_label=r"$Intensity\ (a.u.)$",
                y_lim=(None, 0.035),
                inset_args=inset_args, inset_settings=inset_settings)

    return y_bkg, y_peak, voigt_p_list


def get_full_width_at_half_max(sigma: float, gamma: float):
    fwhm_gauss = 2 * sigma * np.sqrt(2 * np.log(2))
    fwhm_loren = 2 * gamma
    return (0.5343 * fwhm_loren) + np.sqrt(0.2169 * fwhm_loren ** 2 + fwhm_gauss ** 2)


def hitran_matching_process(peak_params: List[dict],
                            x: Union[pd.Series, npt.NDArray[float]],
                            y: Union[pd.Series, npt.NDArray[float]],
                            peaks: npt.NDArray[int], region: int, filename: str,
                            y_bkg: npt.NDArray[float] = None,
                            y_peaks: npt.NDArray[float] = None):
    nu_exp = np.array([v["center"] for v in peak_params])
    fetch_data()
    hitran_data_path_1 = Path(CONFIG.HITRAN_DATA_DIR) / CONFIG.HITRAN_DATA_NAME_1
    hitran_data_path_2 = Path(CONFIG.HITRAN_DATA_DIR) / CONFIG.HITRAN_DATA_NAME_2

    nu_hitran_co2 = pd.read_csv(hitran_data_path_1, header=None, sep=r'\s+',
                                usecols=[1, 2])
    nu_hitran_co2.columns = ["wavenumber", "strength"]
    nu_hitran_co2 = nu_hitran_co2.sort_values(by="wavenumber")

    nu_hitran_h2o = pd.read_csv(hitran_data_path_2, header=None, sep=r'\s+',
                                usecols=[1, 2])
    nu_hitran_h2o.columns = ["wavenumber", "strength"]
    nu_hitran_h2o = nu_hitran_h2o.sort_values(by="wavenumber")

    match_dict_co2, match_dict_h2o = {}, {}
    no_matches = []
    for i, nu in enumerate(nu_exp):
        if peak_params[i]["discard"]:
            continue
        low = nu - CONFIG.RESOLUTION
        high = nu + CONFIG.RESOLUTION

        start_idx_co2 = np.searchsorted(nu_hitran_co2["wavenumber"], low, "left")
        end_idx_co2 = np.searchsorted(nu_hitran_co2["wavenumber"], high, "right")
        start_idx_h2o = np.searchsorted(nu_hitran_h2o["wavenumber"], low, "left")
        end_idx_h2o = np.searchsorted(nu_hitran_h2o["wavenumber"], high, "right")

        potential_matches_co2 = nu_hitran_co2[start_idx_co2: end_idx_co2]
        potential_matches_h2o = nu_hitran_h2o[start_idx_h2o: end_idx_h2o]

        if potential_matches_co2.empty and potential_matches_h2o.empty:
            no_matches.append((i, nu))
        if potential_matches_co2.shape[0] > 0:
            # diffs = abs(nu - potential_matches_co2["wavenumber"])
            # nearest_nu_hitran = potential_matches_co2["wavenumber"].iloc[np.argmin(diffs)]
            max_strength_idx = potential_matches_co2["strength"].idxmax()
            nearest_nu_hitran = potential_matches_co2["wavenumber"][max_strength_idx]
            match_dict_co2[i] = [nu, nearest_nu_hitran]
            peak_params[i]["line_strength"] = potential_matches_co2["strength"] \
                [max_strength_idx]
        if potential_matches_h2o.shape[0] > 0:
            # diffs = abs(nu - potential_matches_h2o["wavenumber"])
            # nearest_nu_hitran = potential_matches_h2o["wavenumber"].iloc[np.argmin(diffs)]
            max_strength_idx = potential_matches_h2o["strength"].idxmax()
            nearest_nu_hitran = potential_matches_h2o["wavenumber"][max_strength_idx]
            match_dict_h2o[i] = [nu, nearest_nu_hitran]
            peak_params[i]["line_strength"] = potential_matches_h2o["strength"] \
                [max_strength_idx]

    match_indices_co2 = list(match_dict_co2.keys())
    match_indices_h2o = list(match_dict_h2o.keys())
    matched_peaks_co2 = peaks[match_indices_co2]
    matched_peaks_h2o = peaks[match_indices_h2o]

    unmatch_indices = [val[0] for val in no_matches]
    unmatched_peaks = peaks[unmatch_indices]
    print(f"\nVoigt peak centres that matched with HITRAN's CO2 nu values with a "
          f"tolerance of +- {CONFIG.RESOLUTION}: \n{match_dict_co2}")
    print(f"\nVoigt peak centres that matched with HITRAN's H2O nu values with a "
          f"tolerance of +- {CONFIG.RESOLUTION}: \n{match_dict_h2o}")
    print(f"\nUnmatched Voigt peak centres with a "
          f"tolerance of += {CONFIG.RESOLUTION}: \n{no_matches}")

    overlap_peak_keys = [i for i in match_indices_co2 if i in match_indices_h2o]
    unique_co2_keys = [i for i in match_indices_co2 if i not in match_indices_h2o]
    unique_h2o_keys = [i for i in match_indices_h2o if i not in match_indices_co2]
    unique_co2_peaks = peaks[unique_co2_keys]
    unique_h2o_peaks = peaks[unique_h2o_keys]
    unique_co2_voigt_vals, unique_co2_x_vals = [], []
    unique_h2o_voigt_vals, unique_h2o_x_vals = [], []
    for key in unique_co2_keys:
        params = peak_params[key]
        y_voigt = voigt_profile(params["center"], params["area"], params["center"],
                                params["sigma"], params["gamma"])
        unique_co2_voigt_vals.append(y_voigt)
        unique_co2_x_vals.append(params["center"])
    unique_co2_voigt_vals = np.array(unique_co2_voigt_vals)

    for key in unique_h2o_keys:
        params = peak_params[key]
        y_voigt = voigt_profile(params["center"], params["area"], params["center"],
                                params["sigma"], params["gamma"])
        unique_h2o_voigt_vals.append(y_voigt)
        unique_h2o_x_vals.append(params["center"])
    unique_h2o_voigt_vals = np.array(unique_h2o_voigt_vals)

    print(f"\n{len(unique_co2_keys)} unique CO2 peaks found: \n{nu_exp[unique_co2_keys]}")
    print(f"\n{len(unique_h2o_keys)} unique H2O peaks found: \n{nu_exp[unique_h2o_keys]}")
    print(
        f"\n{len(overlap_peak_keys)} common H2O and CO2 peaks found: \n{nu_exp[overlap_peak_keys]}")

    if y_bkg is not None and y_peaks is not None:
        y_b_corr = y - y_bkg
        plot_args = [
            {"args": (x, -y_b_corr), "kwargs": {"label": "Baseline-corrected spectrum"}},
            {"args": (x[matched_peaks_co2], -y_b_corr[matched_peaks_co2], 'gx'),
             "kwargs": {"label": "Matched CO2 peaks"}},
            {"args": (x[unmatched_peaks], -y_b_corr[unmatched_peaks], 'rx'),
             "kwargs": {"label": "Unmatched peaks"}},
            {"args": (x, -y_peaks),
             "kwargs": {"label": "Estimated Voigt profiles", "linestyle": "--"}}]
    else:
        plot_args = [
            {"args": (x, -y), "kwargs": {"label": "Baseline-corrected spectrum"}},
            {"args": (unique_co2_x_vals, -unique_co2_voigt_vals, 'gx'),
             "kwargs": {"ms": 8, "label": "CO2 absorption drops"}},
            {"args": (unique_h2o_x_vals, -unique_h2o_voigt_vals, 'rx'),
             "kwargs": {"ms": 8, "label": "H2O absorption drops"}}]
        if y_peaks is not None:
            plot_args.append({"args": (x, -y_peaks),
                              "kwargs": {"label": "Estimated Voigt profiles",
                                         "linestyle": "--"}})

    # inset plot settings and args
    inset_settings = dict(width="100%", height="100%", loc="upper left",
                          bbox_to_anchor=(0.1, 0.65, 0.5, 0.35),
                          bbox_transform=True)
    inset_idx = np.where((x > 6231) & (x <= 6255))[0]
    inset_peaks_co2 = np.intersect1d(inset_idx, unique_co2_peaks)
    inset_peaks_h2o = np.intersect1d(inset_idx, unique_h2o_peaks)

    inset_args = [{"args": (x[inset_idx], -y[inset_idx])},
                  {"args": (x[inset_idx], -y_peaks[inset_idx]),
                   "kwargs": dict(linestyle="--")},
                  {"args": (x[inset_peaks_co2], -y[inset_peaks_co2], 'gx'),
                   "kwargs": dict(ms=7)},
                  {"args": (x[inset_peaks_h2o], -y[inset_peaks_h2o], 'rx'),
                   "kwargs": dict(ms=7)}]

    create_plot(plot_args=plot_args, figure_args={"figsize": (10, 8)},
                title=f"Region {region}: Spectrum fit with Voigt centres matched "
                      f"against HITRAN with a tolerance of +- {CONFIG.RESOLUTION} for {filename}",
                x_label=r'$Wavenumber\ (cm^{-1})$',
                y_label=r"$Baseline-corrected\ intensity\ (a.u.)$", legend=True,
                # inset_settings=inset_settings, inset_args=inset_args,
                y_lim=(None, None))

    nu_obs = np.array([val[0] for val in match_dict_co2.values()])
    nu_hit = np.array([val[1] for val in match_dict_co2.values()])
    # plot_args = [{"args": (nu_obs, (nu_obs - nu_hit))}]
    # create_plot(plot_args=plot_args, figure_args={"figsize": (10, 8)},
    #             title=f"Region {region}: Nu residual plot with a match tolerance of +- {CONFIG.RESOLUTION}",
    #             x_label="Nu observed (Voigt centre)", y_label="Residual", legend=False,
    #             scatter=True)
    print(f"\n mean residual value: \n{np.mean(np.abs(nu_obs - nu_hit))}")
    return peak_params, unique_co2_keys, unique_h2o_keys, overlap_peak_keys, unmatch_indices


def concentration_estimation_process(peak_params: List[dict], co2_keys: list[int],
                                     h2o_keys: list[int], baseline_bspline: BSpline,
                                     x: pd.Series, y: pd.Series):
    conc_per_peak_co2 = [peak_params[key]["area"] / peak_params[key]["line_strength"] \
                         for key in co2_keys]
    conc_per_peak_h2o = [peak_params[key]["area"] / peak_params[key]["line_strength"] \
                         for key in h2o_keys]
    conc_per_peak_co2_2, conc_per_peak_h2o_2 = [], []
    cs = CubicSpline(x, y, extrapolate=True)

    for key in co2_keys:
        params = peak_params[key]
        I_0 = -baseline_bspline(params["center"])
        I = cs(params["center"])
        alpha = ((I_0 / I) - 1) * (1 - 0.999) / 400
        absorp_sigma = params["line_strength"] / params["fwhm"]
        conc_per_peak_co2_2.append(alpha / absorp_sigma)

    for key in h2o_keys:
        params = peak_params[key]
        I_0 = -baseline_bspline(params["center"])
        I = cs(params["center"])
        alpha = ((I_0 / I) - 1) * (1 - 0.999) / 400
        absorp_sigma = params["line_strength"] / params["fwhm"]
        conc_per_peak_h2o_2.append(alpha / absorp_sigma)
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
