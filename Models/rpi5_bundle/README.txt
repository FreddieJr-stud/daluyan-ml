M/B Dalaray Digital Shadow - RPi5 Inference Bundle (v5-tide)
=============================================================

Install: pip install -r requirements.txt

Models (XGBoost production):
  - soc_trip_model*.json: Trip-level SOC consumption (5-model ensemble, 31 features)
  - soc_trip_model_q{10,90}.json: Quantile regression bounds for uncertainty intervals
  - soc_realtime_model*.json: Real-time SOC range estimator (5-model ensemble, 33 features)
  - anomaly_*.json: Motor anomaly reconstruction models (4 targets, 15 features)

Architecture comparison (bundled for benchmarking):
  - rf_*.joblib: Random Forest models (requires scikit-learn + joblib)
  - predictions/mlp_*.csv, lstm_*.csv: Pre-computed MLP/LSTM predictions (no PyTorch needed)

Config:
  - conformal_*.json: Split conformal prediction calibration
  - feature names, metadata, and anomaly thresholds in config/

Tide model:
  - inference/features/tide.py: Harmonic tidal prediction (8 constituents, Schureman 1958)
  - Recalibrated Feb 26, 2026 against 382 tidetime.org observations (Val RMSE = 0.128 m)

Inference scripts in inference/ are standalone (xgboost + numpy only).
No internet connection required — tide features computed from bundled harmonic model.
