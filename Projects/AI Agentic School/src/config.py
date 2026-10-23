"""Configuration settings for Ukraine Air Raid Alerts Predictor."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"

# Default datasets
DEFAULT_OFFICIAL_DATA = RAW_DATA_DIR / "official_data_en.csv"
DEFAULT_VOLUNTEER_DATA = RAW_DATA_DIR / "volunteer_data_en.csv"

# Preprocessing & Target Settings
TARGET_COL = "duration_min"
MIN_DURATION_MINUTES = 0.5  # Filter out instantaneous sirens (< 30 seconds)
MAX_DURATION_MINUTES = 1440.0  # Filter out extreme anomalies (> 24 hours)

# Time Series Validation Settings
TEST_SPLIT_DAYS = 28  # Last 4 weeks reserved for test set
N_CV_SPLITS = 5  # TimeSeriesSplit folds on train set
RANDOM_STATE = 42

# Ukraine Oblast Adjacency Map for Spatial Features
NEIGHBORING_OBLASTS = {
    "Kyiv City": ["Kyivska oblast"],
    "Kyivska oblast": [
        "Kyiv City", "Chernihivska oblast", "Poltavska oblast",
        "Cherkaska oblast", "Vinnytska oblast", "Zhytomyrska oblast"
    ],
    "Chernihivska oblast": ["Kyivska oblast", "Poltavska oblast", "Sumska oblast"],
    "Sumska oblast": ["Chernihivska oblast", "Poltavska oblast", "Kharkivska oblast"],
    "Kharkivska oblast": [
        "Sumska oblast", "Poltavska oblast", "Dnipropetrovska oblast",
        "Donetska oblast", "Luhanska oblast"
    ],
    "Luhanska oblast": ["Kharkivska oblast", "Donetska oblast"],
    "Donetska oblast": [
        "Kharkivska oblast", "Dnipropetrovska oblast", "Zaporizka oblast", "Luhanska oblast"
    ],
    "Zaporizka oblast": ["Dnipropetrovska oblast", "Donetska oblast", "Khersonska oblast"],
    "Khersonska oblast": ["Mykolaivska oblast", "Dnipropetrovska oblast", "Zaporizka oblast"],
    "Mykolaivska oblast": [
        "Odeska oblast", "Kirovohradska oblast", "Dnipropetrovska oblast", "Khersonska oblast"
    ],
    "Odeska oblast": ["Vinnytska oblast", "Kirovohradska oblast", "Mykolaivska oblast"],
    "Kirovohradska oblast": [
        "Cherkaska oblast", "Poltavska oblast", "Dnipropetrovska oblast",
        "Mykolaivska oblast", "Odeska oblast", "Vinnytska oblast"
    ],
    "Poltavska oblast": [
        "Chernihivska oblast", "Sumska oblast", "Kharkivska oblast",
        "Dnipropetrovska oblast", "Kirovohradska oblast", "Cherkaska oblast", "Kyivska oblast"
    ],
    "Dnipropetrovska oblast": [
        "Poltavska oblast", "Kharkivska oblast", "Donetska oblast",
        "Zaporizka oblast", "Khersonska oblast", "Mykolaivska oblast", "Kirovohradska oblast"
    ],
    "Cherkaska oblast": [
        "Kyivska oblast", "Poltavska oblast", "Kirovohradska oblast", "Vinnytska oblast"
    ],
    "Vinnytska oblast": [
        "Zhytomyrska oblast", "Kyivska oblast", "Cherkaska oblast",
        "Kirovohradska oblast", "Odeska oblast", "Chernivetska oblast", "Khmelnytska oblast"
    ],
    "Zhytomyrska oblast": [
        "Kyivska oblast", "Vinnytska oblast", "Khmelnytska oblast", "Rivnenska oblast"
    ],
    "Rivnenska oblast": [
        "Zhytomyrska oblast", "Khmelnytska oblast", "Ternopilska oblast", "Volynska oblast"
    ],
    "Volynska oblast": ["Rivnenska oblast", "Lvivska oblast"],
    "Lvivska oblast": [
        "Volynska oblast", "Rivnenska oblast", "Ternopilska oblast",
        "Ivano-Frankivska oblast", "Zakarpatska oblast"
    ],
    "Zakarpatska oblast": ["Lvivska oblast", "Ivano-Frankivska oblast"],
    "Ivano-Frankivska oblast": [
        "Lvivska oblast", "Zakarpatska oblast", "Ternopilska oblast", "Chernivetska oblast"
    ],
    "Ternopilska oblast": [
        "Rivnenska oblast", "Lvivska oblast", "Ivano-Frankivska oblast",
        "Chernivetska oblast", "Khmelnytska oblast"
    ],
    "Chernivetska oblast": [
        "Ivano-Frankivska oblast", "Ternopilska oblast", "Khmelnytska oblast", "Vinnytska oblast"
    ],
    "Khmelnytska oblast": [
        "Rivnenska oblast", "Zhytomyrska oblast", "Vinnytska oblast",
        "Chernivetska oblast", "Ternopilska oblast"
    ],
}

# Model Hyperparameters
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}
