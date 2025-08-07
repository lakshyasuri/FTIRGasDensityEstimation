hyper_parameters = {
    "AVG_WINDOW_SIZE": 31,
    "PEAK_WLEN": 0.3,  # 1, 0.15 (denser)
    "PEAK_PROMINENCE": 0.85,  # 0.8
    "CLUSTER_MAX_DISTANCE": 8,
    "CLUSTER_MIN_PEAKS": 5,
    "NON_PEAK_KNOTS": 75,
    "MIN_KNOT_SPACING": 1,
    "TAU_VALS": [0.05, 0.03, 0.02, 0.015, 0.01, 0.005, 0.003, 0.002, 0.001],
    "BASELINE": {
        "N_KNOTS": 75,
        "LAM": 1e6
    }

}
