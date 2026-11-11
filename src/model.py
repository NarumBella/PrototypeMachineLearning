"""Training dan evaluasi model regresi produktivitas sawit."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .data import FITUR_KATEGORI, FITUR_NUMERIK, TARGET

TAHUN_TEST = 2023  # data tahun 2023-2024 dipakai sebagai data testing


def split_by_time(d: pd.DataFrame, tahun_test: int = TAHUN_TEST):
    train = d[d.Year < tahun_test]
    test = d[d.Year >= tahun_test]
    return train, test


def _prep(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, FITUR_NUMERIK),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FITUR_KATEGORI),
    ])


def build_models() -> dict:
    return {
        "Linear Regression": Pipeline([("prep", _prep(True)), ("model", LinearRegression())]),
        "Decision Tree": Pipeline([("prep", _prep(False)),
                                   ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42))]),
        "Random Forest": Pipeline([("prep", _prep(False)),
                                   ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                                   n_jobs=-1, random_state=42))]),
    }


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE (%)": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def train_all(d: pd.DataFrame):
    """Latih semua model; kembalikan model terlatih, tabel metrik, dan prediksi data test."""
    train, test = split_by_time(d)
    X_tr, y_tr = train[FITUR_NUMERIK + FITUR_KATEGORI], train[TARGET]
    X_te, y_te = test[FITUR_NUMERIK + FITUR_KATEGORI], test[TARGET]

    hasil, prediksi, models = [], test[["Block_ID", "Tanggal", "Plant_Status", TARGET]].copy(), {}
    # pembanding paling sederhana: produktivitas bulan yang sama tahun lalu
    hasil.append({"Model": "Baseline (sama dengan tahun lalu)", **metrics(y_te, test["lag12"])})
    for nama, pipe in build_models().items():
        pipe.fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        prediksi[nama] = p
        hasil.append({"Model": nama, **metrics(y_te, p)})
        models[nama] = pipe
    return models, pd.DataFrame(hasil), prediksi, (len(train), len(test))


def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    names = pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = pipe.named_steps["model"].feature_importances_
    return pd.DataFrame({"Fitur": names, "Importance": imp}).sort_values("Importance", ascending=False)
