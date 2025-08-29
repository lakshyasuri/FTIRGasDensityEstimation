from types import SimpleNamespace

CONFIG = SimpleNamespace(
    hyper_parameters=SimpleNamespace(
        REGION_THRESHOLD=0.01,
        AVG_WINDOW_SIZE=31,
        PEAK_WLEN=0.03,  # 1, 0.15 (denser)
        PEAK_PROMINENCE=0.8,  # 0.8
        CLUSTER_MAX_DISTANCE=8,
        CLUSTER_MIN_PEAKS=5,
        NON_PEAK_KNOTS=50,
        MIN_KNOT_SPACING=1,
        TAU_VALS=[0.05, 0.03, 0.02, 0.015, 0.01, 0.005, 0.003, 0.002, 0.001],
        BASELINE={
            "N_KNOTS": 200,
            "LAM": [3000],
            "TOL": 0.057
        }
    ),
    RESOLUTION=0.08,
    HITRAN_DATA_DIR='HITRAN_data',
    HITRAN_DATA_NAME_1="CO2.data",
    HITRAN_DATA_NAME_2="H2O.data",
    NU_MIN=None,
    NU_MAX=None
)
