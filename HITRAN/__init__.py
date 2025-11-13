import pandas as pd
from hapi import db_begin, fetch
from pathlib import Path
import numpy as np
from typing import Union, Literal

from config import CONFIG


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


def _calculate_threshold_based_strength(df: pd.DataFrame, quantile: str | float):
    data_dict = {"wavenumber": [], "mean_strength": []}
    current, end = 0, len(df["wavenumber"]) - 1
    while current <= end:
        current_value = df["wavenumber"].iloc[current]
        idx = np.searchsorted(df["wavenumber"], current_value + 100, "right")
        sliced_data = df["strength"].iloc[current: idx]
        try:
            if quantile == 'half_max':
                value = max(sliced_data) / 2
            elif quantile == "mean":
                value = np.mean(sliced_data)
            else:
                value = np.quantile(sliced_data, float(quantile))
        except Exception:
            msg = ("The value of config key 'STATISTIC' can only be of 'mean', "
                   "'half_max', or a numeric value between 0 to 1")
            raise ValueError(msg)
        data_dict["wavenumber"].append(current_value)
        data_dict["mean_strength"].append(value)
        current = idx
    return pd.DataFrame(data_dict)


def get_hitran_strength_threshold(df: pd.DataFrame, gas_name: Literal[
    "co2", "h2o"]) -> pd.DataFrame | float:
    if gas_name.lower() not in ("co2", "h2o"):
        raise ValueError("The molecule name can only be either CO2 or H2O")

    threshold_config = getattr(CONFIG.hyper_parameters,
                               f"HITRAN_{gas_name.upper()}_S_THRESHOLD")
    statistic_threshold = getattr(threshold_config, "STATISTIC", None)
    if statistic_threshold is not None:
        directory = Path(__file__).resolve().parent
        hitran_dir = directory / CONFIG.HITRAN_DATA_DIR
        intensity_threshold = statistic_threshold
        data_path = hitran_dir / f"{gas_name.upper()}_{intensity_threshold}_{CONFIG.NU_MIN}_{CONFIG.NU_MAX}.csv"

        if not data_path.is_file():
            statistic_df = _calculate_threshold_based_strength(df, intensity_threshold)
            statistic_df.to_csv(data_path, index=False)
        else:
            statistic_df = pd.read_csv(data_path)

        return statistic_df
    else:
        return float(threshold_config.HARD)


if __name__ == "__main__":
    co2_data, h2o_data = fetch_data()
    get_hitran_strength_threshold(h2o_data)
