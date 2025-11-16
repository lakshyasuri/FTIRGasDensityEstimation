import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import numpy.typing as npt
import numpy as np
import pandas as pd
from scipy.sparse import diags, dia_matrix, linalg
from scipy.sparse.linalg import spsolve
from scipy.special import wofz
from sklearn.preprocessing import SplineTransformer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from types import SimpleNamespace


def create_plot(plot_args: List[Dict], figure_args: dict = None,
                legend: Union[bool, dict] = False,
                title: str = "", x_label: str = "", y_label: str = "",
                vline_args: List[Dict] = None, hline_args: List[Dict] = None,
                vspan_args: List[Dict] = None, x_lim: Tuple = (), y_lim: Tuple = (),
                scatter: bool = False, inset_settings: dict = None,
                inset_args: List[dict] = None, inset_hline_args: List[dict] = None,
                fill_between_args: List[dict] = None):
    if hline_args is None:
        hline_args = []
    if vline_args is None:
        vline_args = []
    if figure_args is None:
        figure_args = {}
    if vspan_args is None:
        vspan_args = []
    if inset_args is None:
        inset_args = []
    if inset_settings is None:
        inset_settings = {}
    if inset_hline_args is None:
        inset_hline_args = []
    if fill_between_args is None:
        fill_between_args = []

    fig, ax = plt.subplots(**figure_args)
    if inset_settings:
        if "bbox_transform" in inset_settings and inset_settings["bbox_transform"]:
            inset_settings["bbox_transform"] = ax.transAxes
        axins = inset_axes(ax, **inset_settings)
        for arg in inset_args:
            axins.plot(*arg["args"], **arg.get("kwargs", {}))
        for i_h_arg in inset_hline_args:
            axins.axhline(*i_h_arg["args"], **i_h_arg.get("kwargs", {}))
        axins.grid()
    if scatter:
        for args in plot_args:
            ax.scatter(*args["args"], **args.get("kwargs", {}))
    else:
        for args in plot_args:
            ax.plot(*args["args"], **args.get("kwargs", {}))
        for args in fill_between_args:
            ax.plot(*args["args"], **args.get("kwargs", {}))
    for h_args in hline_args:
        ax.axhline(*h_args["args"], **h_args.get("kwargs", {}))
    for v_args in vline_args:
        ax.axvline(*v_args["args"], **v_args.get("kwargs", {}))
    for v_s_args in vspan_args:
        ax.axvspan(*v_s_args["args"], **v_s_args.get("kwargs", {}))
    if legend:
        if isinstance(legend, dict):
            ax.legend(**legend)
        else:
            ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_lim:
        ax.set_xlim(*x_lim)
    if y_lim:
        ax.set_ylim(*y_lim)
    ax.grid()
    # plt.show()


def find_statistic_symmetrically(x: Union[pd.Series, np.ndarray],
                                 y: Union[pd.Series, np.ndarray], window_size: int,
                                 statistic: str = "mean", assume_sorted=False):
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    if not assume_sorted:
        sorted_idx = np.argsort(x)
        x, y = x[sorted_idx], y[sorted_idx]

    local_statistic = []
    half = window_size // 2
    n = len(x)
    dtype = float

    for i in range(n):
        end = x[i] + half
        start = x[i] - half
        left = np.searchsorted(x, start, "left")
        right = np.searchsorted(x, end, "right")
        l_half, r_half = np.arange(left, i), np.arange(i + 1, right)
        l_half_len, r_half_len = len(l_half), len(r_half)

        if i + l_half_len > n - 1:
            last_till_i = n - i - 1  # exclusive of i
            local_cluster = np.concatenate((range(-last_till_i - 1, 0),
                                            range(-(l_half_len - last_till_i), 0),
                                            l_half))
        elif i < r_half_len:
            local_cluster = np.concatenate((range(i + 1),
                                            range(r_half_len - i),
                                            r_half))
        else:
            l_idx = np.searchsorted(x, start, "left")
            r_idx = np.searchsorted(x, end, "right")
            local_cluster = np.arange(l_idx, r_idx)

        y_vals = y[local_cluster]
        if statistic == 'mean':
            stat = np.mean(y_vals)
        elif statistic == 'std':
            stat = np.std(y_vals)
        elif statistic == 'min':
            stat = local_cluster[np.argmin(y_vals)]
            dtype = int
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
        local_statistic.append(stat)
    return np.array(local_statistic, dtype=dtype)


def gauss_and_lorentz_normal(x: npt.NDArray[np.float64], fwhm: float, centre: float):
    fwhm = max(fwhm, 1e-8)
    centre_diff_squared = np.square(x - centre)
    a_gauss = (2 / fwhm) * (np.sqrt(np.log(2) / np.pi))
    b_gauss = 4 * np.log(2) / (fwhm ** 2)
    gauss_normal = a_gauss * np.exp(-b_gauss * centre_diff_squared)

    lorentz_normal = (1 / np.pi) * (
            (fwhm / 2) / (centre_diff_squared + np.square(fwhm / 2)))

    return gauss_normal, lorentz_normal


def pseudo_voigt_profile(x: Union[npt.NDArray[np.float64], float], ratio: float,
                         amplitude: float,
                         centre: float, fwhm: float):
    gauss_normal, lorentz_normal = gauss_and_lorentz_normal(x=x, fwhm=fwhm, centre=centre)
    pseudo_voigt = amplitude * (ratio * gauss_normal + (1 - ratio) * lorentz_normal)
    return pseudo_voigt


def pseudo_voigt_derivates(x: Union[npt.NDArray[np.float64], float], ratio: float,
                           amplitude: float,
                           centre: float, fwhm: float):
    gauss_normal, lorentz_normal = gauss_and_lorentz_normal(x=x, fwhm=fwhm, centre=centre)
    fwhm = max(fwhm, 1e-8)
    # dV/d_ratio
    d_ratio = amplitude * (gauss_normal - lorentz_normal)

    # dV/d_amplitude
    d_amp = ratio * gauss_normal + (1 - ratio) * lorentz_normal

    # dV/d_centre
    # d_gauss/d_centre
    d_guass_centre_partial = (8 * np.log(2) / (fwhm ** 2)) * (x - centre) * gauss_normal
    # d_lorentz/d_centre
    d_lorentz_centre_partial = (4 * np.pi * (x - centre) / fwhm) * np.square(
        lorentz_normal)
    d_centre = amplitude * (
            ratio * d_guass_centre_partial + (1 - ratio) * d_lorentz_centre_partial)

    # dV/d_fwhm
    # d_gauss/d_fwhm
    t1 = -gauss_normal / fwhm
    t2 = (8 * np.log(2) / (fwhm ** 2)) * np.square(x - centre) * gauss_normal / fwhm
    d_gauss_fwhm_partial = t1 + t2
    # d_lorentz/d_fwhm
    t3 = lorentz_normal / fwhm
    t4 = -np.pi * np.square(lorentz_normal)
    d_lorentz_fwhm_partial = t3 + t4
    d_fwhm = amplitude * (
            ratio * d_gauss_fwhm_partial + (1 - ratio) * d_lorentz_fwhm_partial)

    return d_ratio, d_amp, d_centre, d_fwhm


def voigt_profile(x: npt.NDArray[np.float64], A: float, mu: float, sigma: float,
                  gamma: float):
    sigma = max(sigma, 1e-8)
    gamma = max(gamma, 1e-8)
    z = ((x - mu) + 1J * gamma) / (sigma * np.sqrt(2))
    return (A * wofz(z).real) / (sigma * np.sqrt(2 * np.pi))


def voigt_derivatives(x: npt.NDArray[np.float64], A: float, mu: float, sigma: float,
                      gamma: float):
    sigma = max(sigma, 1e-8)
    gamma = max(gamma, 1e-8)
    xc = x - mu
    z = (xc + 1J * gamma) / (sigma * np.sqrt(2))
    wofz_out = wofz(z)

    V = wofz_out.real / (sigma * np.sqrt(2 * np.pi))
    # dV/dA
    dA = V
    # dV/dmu
    dmu = ((xc * wofz_out.real) - (gamma * wofz_out.imag)) / (
            (sigma ** 3) * np.sqrt(2 * np.pi))
    # dV/dsigma
    dsigma = (((xc ** 2 - gamma ** 2 - sigma ** 2) * wofz_out.real -
               2 * xc * gamma * wofz_out.imag + gamma * sigma * np.sqrt(2 / np.pi)) /
              (sigma ** 4 * np.sqrt(2 * np.pi)))
    # dV/dgamma
    dgamma = -(sigma * np.sqrt(
        2 / np.pi) - xc * wofz_out.imag - gamma * wofz_out.real) / (
                     sigma ** 3 * np.sqrt(2 * np.pi))

    return dA, A * dmu, A * dsigma, A * dgamma


def create_splines_pipeline(knot_vector: npt.NDArray, degree: int = 3,
                            extrapolation: str = 'continue'):
    if len(knot_vector.shape) < 2:
        knot_vector = knot_vector[:, np.newaxis]

    return SplineTransformer(degree=degree, extrapolation=extrapolation,
                             knots=knot_vector)


def evaluate_model(beta: npt.NDArray, x: npt.NDArray, n_peaks: int):
    y_hat = np.zeros_like(x)
    peak_params = beta.reshape(n_peaks, 4)

    for (ratio, amp, centre, fwhm) in peak_params:
        y_hat += pseudo_voigt_profile(x=x, ratio=ratio, amplitude=amp, centre=centre,
                                      fwhm=fwhm)
    return y_hat


def loss_function(beta: npt.NDArray, x: npt.NDArray, y: npt.NDArray, n_peaks: int):
    y_hat = evaluate_model(beta, x, n_peaks)
    return np.sum(np.square(y - y_hat))


def loss_function_vector(beta: npt.NDArray, x: npt.NDArray, y: npt.NDArray, n_peaks: int):
    return y - evaluate_model(beta, x, n_peaks)


def jacobian_of_loss(beta: npt.NDArray, x: npt.NDArray, y: npt.NDArray, n_peaks: int):
    y_hat = evaluate_model(beta, x, n_peaks)
    residuals = y - y_hat
    gradient = np.zeros_like(beta)

    # peak gradients
    peak_params = beta.reshape(n_peaks, 4)
    jacobian_peak = []  # final J_voigt shape -> (n, 4*n_peaks)
    for i, (ratio, amp, centre, fwhm) in enumerate(peak_params):
        idx = i * 4
        d_ratio, d_amp, d_centre, d_fwhm = pseudo_voigt_derivates(x=x, ratio=ratio,
                                                                  amplitude=amp,
                                                                  centre=centre,
                                                                  fwhm=fwhm)
        # jacobian_peak.append(np.column_stack([dA, dmu, dsigma, dgamma]))
        gradient[idx] = -2 * np.sum(residuals * d_ratio)
        gradient[idx + 1] = -2 * np.sum(residuals * d_amp)
        gradient[idx + 2] = -2 * np.sum(residuals * d_centre)
        gradient[idx + 3] = -2 * np.sum(residuals * d_fwhm)
    # J_voigt = np.concatenate(jacobian_peak, axis=1)
    # gradient[n_spline_coeffs: ] = -2 * (J_voigt.T @ residuals)
    return gradient


def jacobian_least_squares(beta: npt.NDArray, x: npt.NDArray, y: npt.NDArray,
                           n_peaks: int):
    n_points = len(x)
    jac = np.zeros((n_points, 4 * n_peaks))
    peak_params = beta.reshape(n_peaks, 4)
    for i, (ratio, amp, centre, fwhm) in enumerate(peak_params):
        idx = i * 4
        d_ratio, d_amp, d_centre, d_fwhm = pseudo_voigt_derivates(x=x, ratio=ratio,
                                                                  amplitude=amp,
                                                                  centre=centre,
                                                                  fwhm=fwhm)
        jac[:, idx] = -d_ratio
        jac[:, idx + 1] = -d_amp
        jac[:, idx + 2] = -d_centre
        jac[:, idx + 3] = -d_fwhm
    return jac


def rmse(y_obs: Union[npt.NDArray, pd.Series],
         y_pred: npt.NDArray):
    return np.sqrt(np.mean((y_obs - y_pred) ** 2))


def molecules_per_cm3_to_ppm(N_gas: float):
    # at 1 atm, 298 K (molecules/cm³)
    return N_gas / 2.46e13


def calculate_standard_errors_pspline(residuals, B, w, lam, data_length):
    n, k = B.shape
    print(n, data_length)
    D = diags([1, -2, 1], [0, 1, 2], shape=(k - 2, k)).tocsc()
    W = dia_matrix((w, 0), shape=(data_length, data_length)).tocsr()

    P = lam * D.T @ D
    M = B.T @ W @ B
    LHS = M + P
    H = spsolve(LHS, M)
    df_eff = np.trace(H.toarray())

    w_rss = residuals.T @ W @ residuals
    res_var = w_rss / (n - df_eff)
    LHS_inv = linalg.inv(LHS)
    COV = res_var * LHS_inv

    SE_coeff = np.sqrt(COV.diagonal())
    return df_eff, SE_coeff


def bootstrap_ci_calculation(values, alpha=0.05, n_boot=1000):
    rng = np.random.default_rng(None)
    n = len(values)
    means = [rng.choice(values, n, replace=True).mean() for _ in range(n_boot)]
    return np.mean(values), np.quantile(means, alpha / 2), np.quantile(means,
                                                                       1 - alpha / 2)


def convert_namespace_to_dict(namespace):
    if isinstance(namespace, SimpleNamespace):
        return {k: convert_namespace_to_dict(v) for k, v in namespace.__dict__.items()}
    elif isinstance(namespace, list):
        return [convert_namespace_to_dict(i) for i in namespace]
    else:
        return namespace


def update_config(config: SimpleNamespace):
    with open("config.json", "w") as f:
        json.dump(convert_namespace_to_dict(config), f, indent=1)


def serialize_json(_object: dict):
    new_object = {}
    for key, value in _object.items():
        new_key = str(key)
        if isinstance(value, (list, np.ndarray)):
            new_value = ",".join(value)
        else:
            new_value = str(value)
        new_object[new_key] = new_value
    return new_object
