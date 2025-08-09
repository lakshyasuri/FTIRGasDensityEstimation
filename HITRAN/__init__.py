from hapi import db_begin, fetch
from pathlib import Path

from config import CONFIG


def fetch_data():
    hitran_dir = Path(CONFIG.HITRAN_DATA_DIR)
    gas_path = hitran_dir / CONFIG.HITRAN_DATA_NAME
    print(gas_path)
    if not gas_path.is_file():
        db_begin(CONFIG.HITRAN_DATA_DIR)
        hitran_dir.mkdir(exist_ok=True)
        fetch('CO2', 2, 1, CONFIG.NU_MIN, CONFIG.NU_MAX)
