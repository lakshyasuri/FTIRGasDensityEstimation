from hapi import db_begin, fetch
from pathlib import Path

from config import CONFIG


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
