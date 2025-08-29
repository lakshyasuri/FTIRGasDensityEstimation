from hapi import db_begin, fetch
from pathlib import Path

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
    hitran_dir = Path(CONFIG.HITRAN_DATA_DIR)
    gas_path_1 = hitran_dir / CONFIG.HITRAN_DATA_NAME_1
    gas_path_2 = hitran_dir / CONFIG.HITRAN_DATA_NAME_2
    if not gas_path_1.is_file():
        db_begin(CONFIG.HITRAN_DATA_DIR)
        hitran_dir.mkdir(exist_ok=True)
        fetch('CO2', 2, 1, CONFIG.NU_MIN, CONFIG.NU_MAX)
    if not gas_path_2.is_file():
        db_begin(CONFIG.HITRAN_DATA_DIR)
        hitran_dir.mkdir(exist_ok=True)
        fetch('H2O', 1, 1, CONFIG.NU_MIN, CONFIG.NU_MAX)
