"""Shared configuration for M/B Dalaray XGBoost models."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DALUYAN_ROOT = Path(
    "C:/Users/frpag/MyApps/Fred_Vault/Fred_s Vault/Last_semester/Thesis/Daluyan_V2"
)
BACKEND_DATA = DALUYAN_ROOT / "backend" / "data"
PARQUET_DIR = BACKEND_DATA / "parquet"
DUCKDB_PATH = str(BACKEND_DATA / "daluyan.duckdb")

MODELS_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = MODELS_ROOT / "artifacts"
FEATURE_STORE_DIR = MODELS_ROOT / "artifacts" / "feature_store"

# Ensure output directories exist
ARTIFACTS_DIR.mkdir(exist_ok=True)
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
BATTERY_CAPACITY_KWH = 160.0
MAX_PASSENGERS = 40
SAMPLE_RATE_HZ = 1
SAFETY_SOC_THRESHOLD = 15.0  # % — minimum SOC before flagging range limit

# ── Station Registry ──────────────────────────────────────────────────────
# Ordered east → west along the Pasig River
STATIONS = [
    {"name": "Napindan",   "order": 0, "lat": 14.55713, "lon": 121.06836, "active": True},
    {"name": "Guadalupe",  "order": 1, "lat": 14.56811, "lon": 121.04792, "active": True},
    {"name": "Hulo",       "order": 2, "lat": 14.56794, "lon": 121.03364, "active": True},
    {"name": "Valenzuela", "order": 3, "lat": 14.57400, "lon": 121.02578, "active": False},
    {"name": "Lambingan",  "order": 4, "lat": 14.58731, "lon": 121.01847, "active": True},
    {"name": "Sta Ana",    "order": 5, "lat": 14.58236, "lon": 121.01167, "active": False},
    {"name": "PUP",        "order": 6, "lat": 14.59603, "lon": 121.01072, "active": True},
    {"name": "Quinta",     "order": 7, "lat": 14.59572, "lon": 120.98142, "active": True},
    {"name": "Escolta",    "order": 8, "lat": 14.59653, "lon": 120.97761, "active": True},
]

STATION_NAMES = [s["name"] for s in STATIONS]
STATION_ORDER = {s["name"]: s["order"] for s in STATIONS}
STATION_BY_NAME = {s["name"]: s for s in STATIONS}

# ── Temporal Split (v2: 80/10/10) ────────────────────────────────────────
# Train: Sep 30 2025 – Jan 31 2026 (~80%, 429 segments)
# Val:   Feb 01 – Feb 12 2026      (~10%, 69 segments)
# Test:  Feb 13 – Feb 21 2026      (~10%, 54 segments)
TRAIN_END = "2026-01-31"
VAL_END = "2026-02-12"

# ── Feature Lists ─────────────────────────────────────────────────────────

# Model 1A: Trip-level SOC prediction (16 features — SHAP-pruned from 31 + ridership)
# Pruned via Experiment 6 (SHAP feature pruning, Feb 28 2026):
#   Top-15 by mean |SHAP| on validation set reduced test MAE by 6.19%.
#   Removed 16 low-importance features that added noise to the small dataset.
# v8 ridership update (Mar 13 2026):
#   Added passengers_on_board after real ridership data (493/836 segs).
#   SHAP rank #13/16, Test MAE: 0.8604% -> 0.8440% (-1.90%).
TRIP_FEATURES = [
    "soc_per_km_capacity",            # start_soc / (route_distance_km + 0.01)
    "direction_encoded",
    "speed_distance_interaction",     # target_speed * route_distance_km
    "route_distance_km",
    "departure_station_encoded",
    "wave_component_along_route",     # wave height * cos(wave_dir - route_heading)
    "target_speed",
    "hop_count",
    "arrival_station_encoded",
    "energy_rate_proxy",              # target_speed_squared * hop_count (drag * distance proxy)
    "start_soc",
    "hour_of_day",
    "tide_height_m",                  # predicted tide height above MLLW (meters)
    "wave_height_m",
    "current_component_along_route",  # ocean current projected along route (m/s, +ve = with travel)
    "passengers_on_board",            # v8: real ridership data (493 observed + median imputation)
]
TRIP_TARGET = "soc_delta"

# Model 1B: Real-time range estimation (25 features -- removed current_soc to fix mid-trip echo)
# v6 SHAP re-prune (experiment_1b_shap_reprune_v2.py, Feb 28 2026):
#   Added 3 trip-so-far consumption features from Exp 8-11.
#   SHAP re-rank on 28 candidates (25 old + 3 new) selected top-25.
#   Dropped 3 low-importance: soc_rate_60s, soc_rate_120s, wave_height_m.
#   Added 3 new: empirical_power_per_km, soc_consumed_so_far, empirical_soc_per_km.
#   Test MAE: 1.139% -> 1.136% (-6.47% vs production).
#   Val-test gap improved: +0.514% -> +0.386% (better generalization).
# Note: route_avg_soc_delta (Exp 9) excluded — high SHAP but causes overfitting.
# v8 ridership update (Mar 13 2026):
#   Added passengers_on_board (SHAP rank #6/26). Test MAE neutral (+0.08%).
#   Kept per user preference: real data improves imputation baseline.
REALTIME_FEATURES = [
    # Trip progress (top-3 by SHAP)
    "distance_remaining_km",
    "trip_progress_fraction",
    "elapsed_time_s",
    # Route context
    "direction_encoded",
    "arrival_station_encoded",
    "hop_count",
    "departure_station_encoded",
    # Current state (current_soc removed — caused mid-trip echo, model predicted delta≈0)
    "current_motor_power",
    "current_battery_power",
    # Rolling windows
    "avg_speed_60s",
    "avg_power_60s",
    "avg_power_30s",
    "soc_rate_30s",
    "soc_rate_std_60s",             # SOC rate std dev over 60s (consumption stability)
    # External
    "wind_speed_kn",
    "wind_component_along_route",
    "temperature_c",
    # Physics
    "power_speed_ratio",
    # v3 current proxy feature (telemetry-derived)
    "rpm_per_speed",                # avg(port_rpm, stbd_rpm) / (speed + eps)
    # v3 tide features
    "tide_height_m",                # predicted tide height above MLLW (meters)
    "hours_since_high_tide",        # hours elapsed since most recent high tide
    # v6 trip-so-far consumption features (Exp 8)
    "empirical_power_per_km",       # cumulative motor kW-s / (trip_km + eps)
    "soc_consumed_so_far",          # start_soc - current_soc (% consumed this trip)
    "empirical_soc_per_km",         # soc_consumed / (trip_km + eps) (%/km this trip)
    # v8 ridership (real per-leg passenger counts from Excel data)
    "passengers_on_board",          # 493 observed + median imputation, SHAP rank #6
]
REALTIME_TARGET = "soc_remaining_delta"

# Model 2: Motor anomaly detection — raw features
ANOMALY_RAW_FEATURES = [
    "port_motor_power",
    "stbd_motor_power",
    "motor_power_combined",
    "port_rpm",
    "stbd_rpm",
    "speed_over_ground",
    "battery_power",
]

# Model 2: Motor anomaly detection — all features (raw + derived)
ANOMALY_FEATURES = ANOMALY_RAW_FEATURES + [
    "port_stbd_power_ratio",
    "port_stbd_power_diff",
    "port_stbd_rpm_ratio",
    "port_stbd_rpm_diff",
    "port_power_per_rpm",
    "stbd_power_per_rpm",
    "combined_power_per_speed",
    "power_speed_squared_ratio",
]

# Features to predict for reconstruction-error anomaly detection
ANOMALY_TARGETS = [
    "port_motor_power",
    "stbd_motor_power",
    "port_rpm",
    "stbd_rpm",
]

# Model 3: Battery anomaly detection — raw features (per side)
# Source: OneAries BMS (IP_3 = port, IP_4 = stbd), available in all 34-field rows
BATTERY_RAW_FEATURES = [
    "port_cell_v_spread",        # gCellBalance (V) — cell voltage spread
    "stbd_cell_v_spread",
    "port_cell_t_max",           # gMaxCellTemperature (°C)
    "stbd_cell_t_max",
    "port_cell_t_min",           # gMinCellTemperature (°C)
    "stbd_cell_t_min",
    "port_pack_voltage",         # gPackVoltage (V)
    "stbd_pack_voltage",
    "port_pack_current",         # gCurrent (A, +ve=charging, -ve=discharging)
    "stbd_pack_current",
    "port_soc",                  # gStateOfCharge (%)
    "stbd_soc",
    "port_power",                # gPower (kW)
    "stbd_power",
    "port_avg_temperature",      # gAverageTemperature (°C)
    "stbd_avg_temperature",
]

BATTERY_DERIVED_FEATURES = [
    "port_cell_t_spread",        # max - min cell temperature per side
    "stbd_cell_t_spread",
    "cell_v_spread_combined",    # max(port, stbd) cell voltage spread
    "cell_t_spread_combined",    # max(port, stbd) cell temp spread
    "port_stbd_soc_diff",       # |port_soc - stbd_soc| — pack imbalance
    "port_stbd_v_diff",         # |port_pack_voltage - stbd_pack_voltage|
    "port_stbd_power_diff",     # |port_power - stbd_power|
]

BATTERY_FEATURES = BATTERY_RAW_FEATURES + BATTERY_DERIVED_FEATURES

# Reconstruction targets for battery anomaly detection
BATTERY_TARGETS_CHARGING = [
    "port_cell_v_spread",        # cell voltage balance during charge
    "stbd_cell_v_spread",
    "port_pack_current",         # charging current profile
    "stbd_pack_current",
]

BATTERY_TARGETS_OPERATIONAL = [
    "port_cell_v_spread",        # cell voltage balance under load
    "stbd_cell_v_spread",
    "port_cell_t_spread",        # thermal balance under load
    "stbd_cell_t_spread",
]

BATTERY_ANOMALY_CONFIG = {
    "n_trials": 100,
    "n_startup_trials": 15,
}

# ── Raw CSV paths ─────────────────────────────────────────────────────────
RAW_CSV_DIR = BACKEND_DATA / "raw"

# BMS device identifiers in raw CSVs
DEVICE_PORT_BMS = "device/OneAries_IP_3_ID_49"
DEVICE_STBD_BMS = "device/OneAries_IP_4_ID_49"

# ── Monotonic Constraints (v2, pruned for SHAP top-15) ───────────────────
# Physics-enforced relationships for Model 1A trip prediction.
# +1 = feature increasing should increase prediction (more SOC consumed)
# -1 = feature increasing should decrease prediction
#  0 = no constraint
# Note: Only features in TRIP_FEATURES need entries here.
# get_monotonic_constraint_tuple() defaults absent features to 0.
TRIP_MONOTONIC_CONSTRAINTS = {
    "hop_count": 1,                  # more hops = more energy
    "route_distance_km": 1,          # longer route = more energy
    "energy_rate_proxy": 1,          # drag * distance proxy
    "speed_distance_interaction": 1, # speed * distance
    # v4: current WITH travel direction reduces consumption
    "current_component_along_route": -1,
}

# ── Per-Direction Configuration (v4) ──────────────────────────────────
# Directional models omit direction_encoded (constant within each model)
TRIP_FEATURES_DIRECTIONAL = [f for f in TRIP_FEATURES if f != "direction_encoded"]


def get_monotonic_constraint_tuple(feature_list, constraint_dict):
    """Build XGBoost monotone_constraints tuple from feature list and dict."""
    return tuple(constraint_dict.get(f, 0) for f in feature_list)


# ── Ensemble Configuration (v2) ─────────────────────────────────────────
N_ENSEMBLE_SEEDS = 5
ENSEMBLE_SEEDS = [42, 137, 256, 512, 1024]

# ── Data Augmentation Configuration (v2) ─────────────────────────────────
AUGMENTATION = {
    "gaussian_noise": {
        "n_copies": 3,          # number of augmented copies
        "noise_std": 0.02,      # relative to each feature's std dev
    },
    "cmixup": {
        "alpha": 0.2,           # mixup interpolation strength (small = close to original)
        "sigma": 2.0,           # similarity bandwidth for C-Mixup pairing
        "n_synthetic": 1.0,     # multiplier of original dataset size
    },
}

# ── Optuna Search Space (v2) ────────────────────────────────────────────
OPTUNA_CONFIG = {
    "n_trials": 500,             # v10.5: bumped from 300 for deeper search
    "n_startup_trials": 50,     # random trials before TPE kicks in (scaled with n_trials)
    "search_space": {
        "max_depth": (3, 10),
        "learning_rate": (0.005, 0.3),       # log-scale
        "n_estimators": (200, 5000),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "colsample_bylevel": (0.5, 1.0),
        "min_child_weight": (1, 30),
        "reg_alpha": (1e-4, 10.0),           # log-scale
        "reg_lambda": (1e-4, 10.0),          # log-scale
        "gamma": (0.0, 5.0),
        "max_delta_step": (0, 5),
    },
}

# ── XGBoost Default Configs (v2 starting points) ────────────────────────

XGBOOST_TRIP_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "seed": 42,
}

XGBOOST_REALTIME_PARAMS = {
    **XGBOOST_TRIP_PARAMS,
    "min_child_weight": 10,
    "reg_lambda": 2.0,
}

XGBOOST_ANOMALY_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "seed": 42,
}

# ── Random Forest Configuration ───────────────────────────────────────
RF_OPTUNA_CONFIG = {
    "n_trials": 150,        # RF converges faster than XGB
    "n_startup_trials": 20,
}

# ── Anomaly Alternative Architectures ─────────────────────────────────
ANOMALY_RF_CONFIG = {"n_trials": 150, "n_startup_trials": 20}
ANOMALY_AE_CONFIG = {"optuna_trials": 80, "max_epochs": 300, "patience": 25}
ANOMALY_IF_CONFIG = {"n_trials": 100, "n_startup_trials": 15}

# ── LSTM/MLP Configuration ────────────────────────────────────────────
LSTM_CONFIG = {
    "model_1a": {
        "type": "mlp",         # 1A is non-sequential (1 row per trip)
        "optuna_trials": 100,
        "max_epochs": 500,
        "patience": 30,
    },
    "model_1b": {
        "type": "lstm",        # 1B has temporal 1Hz sequences per trip
        "optuna_trials": 100,  # LSTM is expensive; 100 for fair comparison
        "max_epochs": 200,
        "patience": 20,
    },
}
