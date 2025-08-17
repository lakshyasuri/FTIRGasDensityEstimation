import sys

import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import numpy.typing as npt
import numpy as np
import pandas as pd
from scipy.special import wofz
from sklearn.preprocessing import SplineTransformer


def create_plot(plot_args: List[Dict], figure_args: dict = None, legend: bool = False,
                title: str = "", x_label: str = "", y_label: str = "",
                vline_args: List[Dict] = None, hline_args: List[Dict] = None,
                vspan_args: List[Dict] = None, x_lim: Tuple = (), y_lim: Tuple = (),
                scatter: bool = False):
    if hline_args is None:
        hline_args = []
    if vline_args is None:
        vline_args = []
    if figure_args is None:
        figure_args = {}
    if vspan_args is None:
        vspan_args = []

    plt.figure(**figure_args)
    if scatter:
        for args in plot_args:
            plt.scatter(*args["args"], **args.get("kwargs", {}))
    else:
        for args in plot_args:
            plt.plot(*args["args"], **args.get("kwargs", {}))
    for h_args in hline_args:
        plt.axhline(*h_args["args"], **h_args.get("kwargs", {}))
    for v_args in vline_args:
        plt.axvline(*v_args["args"], **v_args.get("kwargs", {}))
    for v_s_args in vspan_args:
        plt.axvspan(*v_s_args["args"], **v_s_args.get("kwargs", {}))
    if legend:
        plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_lim:
        plt.xlim(*x_lim)
    if y_lim:
        plt.ylim(*y_lim)
    plt.grid()
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


class StatManager:
    def __init__(self, window_size: int):
        # self.symmetric_indices = []
        self.window_size = window_size

    def __find_symmetric_indices(self, x: np.ndarray):
        symmetric_indices = []
        half = self.window_size // 2
        n = len(x)

        for i in range(n):
            end = x[i] + half
            start = x[i] - half
            left = np.searchsorted(x, start, "left")
            right = np.searchsorted(x, end, "right")
            # l_half = np.where((x >= start) & (x < x[i]))[0]
            # r_half = np.where((x > x[i]) & (x <= end))[0]
            l_half, r_half = np.arange(left, i), np.arange(i + 1, right)
            l_half_len, r_half_len = len(l_half), len(r_half)
            # l_half_len, r_half_len = i - left, right - i - 1
            # print(l_half, r_half, l_half_2, r_half_2)

            if i + l_half_len > n - 1:
                print("l_half >")
                last_till_i = n - i - 1  # exclusive of i
                local_cluster = np.concatenate((range(-last_till_i - 1, 0),
                                                range(-(l_half_len - last_till_i), 0),
                                                l_half))
            elif i < r_half_len:
                print("i < r_half")
                local_cluster = np.concatenate((range(i + 1),
                                                range(r_half_len - i),
                                                r_half))
            else:
                print("neither")
                l_idx = np.searchsorted(x, start, "left")
                r_idx = np.searchsorted(x, end, "right")
                local_cluster = np.arange(l_idx, r_idx)
                # local_cluster = np.where((x >= start) & (x <= end))[0]
                # print(local_cluster[0], local_cluster[-1])
                # print(local_cluster_2[0], local_cluster_2[-1])
            # self.symmetric_indices.append(local_cluster)
            symmetric_indices.append(local_cluster)
            sys.exit()
        return symmetric_indices

    # calculating a statistic for every point using a symmetric window
    def find_statistic_symmetrically(self, x: Union[pd.Series, np.ndarray],
                                     y: Union[pd.Series, np.ndarray],
                                     override_window=None, statistic='mean',
                                     assume_sorted=False):
        if isinstance(x, pd.Series):
            x = x.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        if not assume_sorted:
            sorted_idx = np.argsort(x)
            x, y = x[sorted_idx], y[sorted_idx]

        # if not self.symmetric_indices or override_window is not None:
        #     if override_window is not None:
        #         self.window_size = override_window
        #         self.symmetric_indices = []
        #     self.__find_symmetric_indices(x)

        local_statistic = []
        symmetric_indices = self.__find_symmetric_indices(x)
        for indices in symmetric_indices:
            y_vals = y[indices]
            dtype = float

            if statistic == 'mean':
                stat = np.mean(y_vals)
            elif statistic == 'std':
                stat = np.std(y_vals)
            elif statistic == 'min':
                stat = indices[np.argmin(y_vals)]
                dtype = int
            local_statistic.append(stat)
        return np.array(local_statistic, dtype=dtype)


def voigt_profile(x: npt.NDArray[np.float64], A: float, mu: float, sigma: float,
                  gamma: float):
    z = ((x - mu) + 1J * gamma) / (sigma * np.sqrt(2))
    return (A * wofz(z).real) / (sigma * np.sqrt(2 * np.pi))


def voigt_derivatives(x: npt.NDArray[np.float64], A: float, mu: float, sigma: float,
                      gamma: float):
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


def evaluate_model(beta: npt.NDArray, spline_basis: npt.NDArray, x: npt.NDArray,
                   n_peaks: int, baseline_corrected: bool):
    if not baseline_corrected:
        n_spline_coeffs = spline_basis.shape[1]
        spline_params = beta[:n_spline_coeffs]
        y_hat = spline_basis @ spline_params
    else:
        n_spline_coeffs = 0
        y_hat = np.zeros_like(x)
    peak_params = beta[n_spline_coeffs:].reshape(n_peaks, 4)

    for (A, mu, sigma, gamma) in peak_params:
        y_hat += voigt_profile(x, A, mu, sigma, gamma)
    return y_hat


def loss_function(beta: npt.NDArray, spline_basis: npt.NDArray, x: npt.NDArray,
                  y: npt.NDArray, n_peaks: int, baseline_corrected: bool):
    y_hat = evaluate_model(beta, spline_basis, x, n_peaks, baseline_corrected)
    return np.sum(np.square(y - y_hat))


def jacobian_of_loss(beta: npt.NDArray, spline_basis: npt.NDArray, x: npt.NDArray,
                     y: npt.NDArray, n_peaks: int, baseline_corrected: bool):
    y_hat = evaluate_model(beta, spline_basis, x, n_peaks, baseline_corrected)
    residuals = y - y_hat
    gradient = np.zeros_like(beta)

    # spline gradients
    if not baseline_corrected:
        n_spline_coeffs = spline_basis.shape[1]
        gradient[:n_spline_coeffs] = -2 * (spline_basis.T @ residuals)
    else:
        n_spline_coeffs = 0

    # peak gradients
    peak_params = beta[n_spline_coeffs:].reshape(n_peaks, 4)
    jacobian_peak = []  # final J_voigt shape -> (n, 4*n_peaks)
    for i, (A, mu, sigma, gamma) in enumerate(peak_params):
        idx = n_spline_coeffs + i * 4
        dA, dmu, dsigma, dgamma = voigt_derivatives(x, A, mu, sigma, gamma)
        # jacobian_peak.append(np.column_stack([dA, dmu, dsigma, dgamma]))
        gradient[idx] = -2 * np.sum(residuals * dA)
        gradient[idx + 1] = -2 * np.sum(residuals * dmu)
        gradient[idx + 2] = -2 * np.sum(residuals * dsigma)
        gradient[idx + 3] = -2 * np.sum(residuals * dgamma)
    # J_voigt = np.concatenate(jacobian_peak, axis=1)
    # gradient[n_spline_coeffs: ] = -2 * (J_voigt.T @ residuals)
    return gradient


def rmse(y_obs: npt.NDArray, y_pred: npt.NDArray):
    return np.sqrt(np.mean((y_obs - y_pred) ** 2))
