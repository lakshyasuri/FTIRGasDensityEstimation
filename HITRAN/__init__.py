import pandas as pd
from hapi import db_begin, fetch
from pathlib import Path
import numpy as np
from typing import Union

from config import CONFIG


# choose the hitran nu with the strongest line strength within +- 0.08
# If FWHM < 0.08 or 0.1 then ignore. Or if > 0.3 then definitely ignore

# a good number of unique lines is 10 to 20.

# S/fwhm = sigma
# N = alpha/sigma
# d= 400 cm
# R = 0.999
# I = original spectrum, I0 = baseline


# or, simply N = Area/S because N*(integration sigma) = integration alpha => N*S = Area
# 10^16

# distribution of N as well

def fetch_data():
    directory = Path(__file__).resolve().parent
    hitran_dir = directory / CONFIG.HITRAN_DATA_DIR
    gas_path_1 = hitran_dir / CONFIG.HITRAN_DATA_NAME_1
    gas_path_2 = hitran_dir / CONFIG.HITRAN_DATA_NAME_2
    if not gas_path_1.is_file():
        db_begin(str(hitran_dir))
        hitran_dir.mkdir(exist_ok=True)
        fetch('CO2', 2, 1, CONFIG.NU_MIN, CONFIG.NU_MAX,
              ParameterGroups=["Standard"],
              Parameters=["molec_id", "local_iso_id", "nu", "sw", "gamma_air",
                          "gamma_self", "delta_air"])
    if not gas_path_2.is_file():
        db_begin(str(hitran_dir))
        hitran_dir.mkdir(exist_ok=True)
        fetch('H2O', 1, 1, CONFIG.NU_MIN, CONFIG.NU_MAX,
              ParameterGroups=["Standard"],
              Parameters=["molec_id", "local_iso_id", "nu", "sw", "gamma_air",
                          "gamma_self", "delta_air"]
              )

    co2_data = pd.read_csv(gas_path_1, header=None, sep=r'\s+',
                           usecols=[1, 2])
    co2_data.columns = ["wavenumber", "strength"]
    co2_data = co2_data.sort_values(by="wavenumber")

    h2o_data = pd.read_csv(gas_path_2, header=None, sep=r'\s+',
                           usecols=[1, 2])
    h2o_data.columns = ["wavenumber", "strength"]
    h2o_data = h2o_data.sort_values(by="wavenumber")
    return co2_data, h2o_data


def _calculate_threshold_based_strength(df: pd.DataFrame, quantile: Union[float | str]):
    data_dict = {"wavenumber": [], "mean_strength": []}
    current, end = 0, len(df["wavenumber"]) - 1
    while current <= end:
        current_value = df["wavenumber"].iloc[current]
        idx = np.searchsorted(df["wavenumber"], current_value + 100, "right")
        sliced_data = df["strength"].iloc[current: idx]
        if quantile == 'half_max':
            value = max(sliced_data) / 2
        elif quantile == "mean":
            value = np.mean(sliced_data)
        else:
            value = np.quantile(sliced_data, quantile)
        data_dict["wavenumber"].append(current_value)
        data_dict["mean_strength"].append(value)
        current = idx
    return pd.DataFrame(data_dict)


def get_regional_mean_strength(h2o_df: pd.DataFrame):
    directory = Path(__file__).resolve().parent
    hitran_dir = directory / CONFIG.HITRAN_DATA_DIR
    intensity_threshold = CONFIG.HITRAN_H2O_S_THRESHOLD
    h2o_path = hitran_dir / f"H2O_{intensity_threshold}_{CONFIG.NU_MIN}_{CONFIG.NU_MAX}.csv"
    # if not co2_path.is_file():
    #     mean_df_co2 = _calculate_mean_strength(co2_df, quantile_threshold)
    #     mean_df_co2.to_csv(co2_path, index=False)
    # else:
    #     mean_df_co2 = pd.read_csv(co2_path)

    if not h2o_path.is_file():
        mean_df_h2o = _calculate_threshold_based_strength(h2o_df, intensity_threshold)
        mean_df_h2o.to_csv(h2o_path, index=False)
    else:
        mean_df_h2o = pd.read_csv(h2o_path)

    return mean_df_h2o


if __name__ == "__main__":
    co2_data, h2o_data = fetch_data()
    get_regional_mean_strength(h2o_data)
