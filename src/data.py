"""Memuat dataset produktivitas sawit dan membentuk fitur untuk model."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

STATUS_URUT = ["TBM", "TM-1", "TM-2", "TM-3", "TM-Utama", "TM-Tua"]
NAMA_BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

# fitur yang dipakai model
FITUR_NUMERIK = ["Plant_Age_Years", "Palms_per_Ha", "Area_Ha", "Month_sin", "Month_cos",
                 "lag1", "lag2", "lag3", "lag12", "roll3"]
FITUR_KATEGORI = ["Plant_Status", "Seed_Type"]
TARGET = "Productivity_t_Ha"


def load_productivity() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "productivity_monthly.csv", encoding="utf-8-sig")
    df["Tanggal"] = pd.to_datetime(dict(year=df.Year, month=df.Month_Number, day=1))
    return df.sort_values(["Block_ID", "Tanggal"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur musiman (bulan siklikal) dan lag produktivitas per blok."""
    df = df.copy()
    df["Month_sin"] = np.sin(2 * np.pi * df.Month_Number / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df.Month_Number / 12)
    g = df.groupby("Block_ID")[TARGET]
    for k in (1, 2, 3, 12):
        df[f"lag{k}"] = g.shift(k)
    df["roll3"] = df[["lag1", "lag2", "lag3"]].mean(axis=1)
    return df


def model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Baris yang dipakai untuk training/testing: tanaman sudah menghasilkan dan lag lengkap."""
    d = add_features(df)
    d = d[d.Plant_Status != "TBM"]
    return d.dropna(subset=["lag1", "lag2", "lag3", "lag12"]).reset_index(drop=True)


def load_leaf_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Hara daun per blok per tahun, digabung dengan rata-rata produktivitas tahunan blok."""
    leaf = pd.read_csv(DATA_DIR / "leaf_sampling_event.csv", encoding="utf-8-sig")
    member = pd.read_csv(DATA_DIR / "kcd_block_membership.csv", encoding="utf-8-sig")
    hara = ["N_pct", "P_pct", "K_pct", "Ca_pct", "Mg_pct"]
    lb = member.merge(leaf[["KCD_Event_ID"] + hara], on="KCD_Event_ID")
    lb = lb.rename(columns={"Sampling_Year": "Year"})
    tahunan = (df[df.Plant_Status != "TBM"]
               .groupby(["Block_ID", "Year"], as_index=False)[TARGET].mean())
    return lb.merge(tahunan, on=["Block_ID", "Year"])
