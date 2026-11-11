"""Prototipe web analisis produktivitas kebun kelapa sawit (Streamlit)."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.data import (FITUR_KATEGORI, FITUR_NUMERIK, NAMA_BULAN, STATUS_URUT, TARGET,
                      add_features, load_leaf_annual, load_productivity, model_table)
from src.model import TAHUN_TEST, feature_importance, train_all

st.set_page_config(page_title="Analisis Produktivitas Sawit", layout="wide")

TEAL = "#004858"


@st.cache_data
def data():
    return load_productivity()


@st.cache_resource
def trained():
    return train_all(model_table(data()))


df = data()
halaman = st.sidebar.radio("Halaman", ["Ringkasan", "Data", "Eksplorasi (EDA)", "Model", "Prediksi"])
st.sidebar.caption("Prototipe Kelompok 1 · Pembelajaran Mesin · Politeknik Caltex Riau")

# ---------------------------------------------------------------- Ringkasan
if halaman == "Ringkasan":
    st.title("Prediksi Produktivitas TBS Kelapa Sawit per Blok")
    st.write(
        "Prototipe ini memprediksi **produktivitas tandan buah segar (TBS) bulanan per blok kebun** "
        "(ton/ha) dari umur tanaman, status tanaman, jenis bibit, kerapatan tanam, pola musiman, "
        "dan produktivitas bulan-bulan sebelumnya. Hasilnya dapat dipakai pengelola kebun untuk "
        "memperkirakan panen dan menandai blok yang produktivitasnya menurun."
    )
    tm = df[df.Plant_Status != "TBM"]
    c = st.columns(4)
    c[0].metric("Observasi bulanan", f"{len(df):,}".replace(",", "."))
    c[1].metric("Blok kebun", df.Block_ID.nunique())
    c[2].metric("Periode", f"{df.Year.min()}–{df.Year.max()}")
    c[3].metric("Rata-rata produktivitas TM", f"{tm[TARGET].mean():.2f} t/ha/bln")

    st.subheader("Alur penelitian")
    st.markdown(
        "1. **Data acquisition**: dataset blok kebun sawit Indonesia (Mendeley Data, CC BY 4.0).\n"
        "2. **EDA**: pola umur tanaman, musim, jenis bibit, dan hara daun.\n"
        "3. **Cleaning**: baris TBM (belum berbuah, produktivitas 0) dikeluarkan dari model.\n"
        "4. **Feature engineering**: bulan siklikal (sin/cos), lag 1/2/3/12 bulan, rata-rata 3 bulan.\n"
        f"5. **Training**: data {df.Year.min()}–{TAHUN_TEST - 1}; **testing**: {TAHUN_TEST}–{df.Year.max()} (dibagi menurut waktu).\n"
        "6. **Evaluasi**: MAE, RMSE, R², MAPE, dibandingkan dengan baseline.\n"
        "7. **Implementasi**: halaman Prediksi pada web ini."
    )
    st.caption("Sumber data: Mimboro, P. (2026). Long-Term Block-Level Oil Palm Productivity Dataset with Leaf "
               "Nutrient and Soil Fertility Measurements from an Indonesian Plantation. Mendeley Data, V2. "
               "doi:10.17632/93k3733p6x.2")

# ---------------------------------------------------------------- Data
elif halaman == "Data":
    st.title("Data")
    c1, c2, c3 = st.columns(3)
    blok = c1.multiselect("Blok", sorted(df.Block_ID.unique()))
    tahun = c2.slider("Tahun", int(df.Year.min()), int(df.Year.max()), (int(df.Year.min()), int(df.Year.max())))
    status = c3.multiselect("Status tanaman", STATUS_URUT)
    v = df[df.Year.between(*tahun)]
    if blok:
        v = v[v.Block_ID.isin(blok)]
    if status:
        v = v[v.Plant_Status.isin(status)]
    st.caption(f"{len(v):,} baris".replace(",", "."))
    st.dataframe(v.drop(columns=["Tanggal"]), width="stretch", height=420)

    st.subheader("Arti kolom")
    st.table(pd.DataFrame({
        "Kolom": ["Plant_Age_Years", "Plant_Status", "Area_Ha", "Productive_Palms", "Palms_per_Ha",
                  "Seed_Type", "Productivity_t_Ha", "FFB_Production_t"],
        "Keterangan": ["Umur tanaman (tahun)", "TBM = belum menghasilkan; TM-1..TM-3 = awal berbuah; "
                       "TM-Utama = produktif; TM-Tua = tua", "Luas blok (ha)", "Jumlah pohon produktif",
                       "Kerapatan tanam (pohon/ha)", "Asal bibit (PPKS, PPKS/Sofc, Lonsum)",
                       "Produktivitas TBS bulanan (ton/ha), target model",
                       "Produksi TBS (ton) = produktivitas × luas, tidak dipakai sebagai fitur"],
    }))

# ---------------------------------------------------------------- EDA
elif halaman == "Eksplorasi (EDA)":
    st.title("Eksplorasi Data")
    tm = df[df.Plant_Status != "TBM"]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Produktivitas menurut umur tanaman")
        a = tm.groupby("Plant_Age_Years", as_index=False)[TARGET].mean()
        st.altair_chart(alt.Chart(a).mark_line(point=True, color=TEAL).encode(
            x=alt.X("Plant_Age_Years:Q", title="Umur tanaman (tahun)"),
            y=alt.Y(f"{TARGET}:Q", title="Rata-rata t/ha/bulan")), width="stretch")
        st.caption("Naik sampai puncak di umur 6–9 tahun, lalu turun setelah umur sekitar 18 tahun.")
    with c2:
        st.subheader("Pola musiman")
        b = tm.groupby("Month_Number", as_index=False)[TARGET].mean()
        b["Bulan"] = b.Month_Number.map(lambda m: NAMA_BULAN[m - 1])
        st.altair_chart(alt.Chart(b).mark_bar(color=TEAL).encode(
            x=alt.X("Bulan:N", sort=NAMA_BULAN, title=None),
            y=alt.Y(f"{TARGET}:Q", title="Rata-rata t/ha/bulan")), width="stretch")
        st.caption("Panen rendah di awal tahun dan tinggi pada pertengahan tahun.")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Sebaran per status tanaman")
        st.altair_chart(alt.Chart(tm).mark_boxplot(color=TEAL).encode(
            x=alt.X("Plant_Status:N", sort=STATUS_URUT, title=None),
            y=alt.Y(f"{TARGET}:Q", title="t/ha/bulan")), width="stretch")
    with c4:
        st.subheader("Tren tahunan per jenis bibit")
        t = tm.groupby(["Year", "Seed_Type"], as_index=False)[TARGET].mean()
        st.altair_chart(alt.Chart(t).mark_line(point=True).encode(
            x=alt.X("Year:O", title="Tahun"), y=alt.Y(f"{TARGET}:Q", title="Rata-rata t/ha/bulan"),
            color=alt.Color("Seed_Type:N", title="Bibit")), width="stretch")

    st.subheader("Riwayat produktivitas per blok")
    pilih = st.selectbox("Blok", sorted(df.Block_ID.unique()))
    h = df[df.Block_ID == pilih]
    st.altair_chart(alt.Chart(h).mark_line(color=TEAL).encode(
        x=alt.X("Tanggal:T", title=None), y=alt.Y(f"{TARGET}:Q", title="t/ha/bulan"),
        tooltip=["Tanggal:T", "Plant_Status", "Plant_Age_Years", alt.Tooltip(f"{TARGET}:Q", format=".3f")]),
        width="stretch")
    info = h.iloc[-1]
    st.caption(f"Tahun tanam {info.Planting_Year} · bibit {info.Seed_Type} · luas {info.Area_Ha} ha · "
               f"{info.Palms_per_Ha} pohon/ha")

    st.subheader("Hara daun dan produktivitas tahunan")
    leaf = load_leaf_annual(df)
    hara = st.radio("Unsur hara", ["N_pct", "P_pct", "K_pct", "Ca_pct", "Mg_pct"], horizontal=True)
    r = leaf[[hara, TARGET]].corr().iloc[0, 1]
    st.altair_chart(alt.Chart(leaf).mark_circle(size=50, color=TEAL, opacity=0.6).encode(
        x=alt.X(f"{hara}:Q", title=f"{hara} daun", scale=alt.Scale(zero=False)),
        y=alt.Y(f"{TARGET}:Q", title="Rata-rata t/ha/bulan pada tahun sampling")),
        width="stretch")
    st.caption(f"{len(leaf)} pasangan blok-tahun (2015–2022). Korelasi Pearson = {r:.2f}. "
               "Data hara belum dipakai di model; dapat menjadi pengembangan berikutnya.")

    st.subheader("Korelasi fitur numerik")
    k = add_features(tm)[FITUR_NUMERIK + [TARGET]].corr().reset_index().melt("index")
    st.altair_chart(alt.Chart(k).mark_rect().encode(
        x=alt.X("index:N", title=None), y=alt.Y("variable:N", title=None),
        color=alt.Color("value:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1]), title="r"),
        tooltip=["index", "variable", alt.Tooltip("value:Q", format=".2f")]), width="stretch")

# ---------------------------------------------------------------- Model
elif halaman == "Model":
    st.title("Model dan Evaluasi")
    models, hasil, prediksi, (n_tr, n_te) = trained()
    ribuan = lambda n: f"{n:,}".replace(",", ".")
    st.write(f"Data training **{ribuan(n_tr)}** baris ({df.Year.min()}–{TAHUN_TEST - 1}), "
             f"data testing **{ribuan(n_te)}** baris ({TAHUN_TEST}–{df.Year.max()}). "
             "Pembagian menurut waktu, sehingga model diuji pada bulan-bulan yang belum pernah dilihat.")
    st.write("Fitur: " + ", ".join(f"`{f}`" for f in FITUR_NUMERIK + FITUR_KATEGORI))
    st.dataframe(hasil.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}", "MAPE (%)": "{:.1f}"})
                 .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#d7ebe9")
                 .highlight_max(subset=["R2"], color="#d7ebe9"),
                 width="stretch", hide_index=True)
    st.caption("Baseline = produktivitas bulan yang sama tahun lalu. Model yang baik harus mengalahkan baseline.")

    nama = st.selectbox("Lihat detail model", list(models))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Aktual vs prediksi (data testing)")
        sc = prediksi.rename(columns={TARGET: "Aktual", nama: "Prediksi"})
        garis = pd.DataFrame({"x": [0, sc[["Aktual", "Prediksi"]].max().max()]})
        st.altair_chart(
            alt.Chart(sc).mark_circle(size=25, opacity=0.5, color=TEAL).encode(
                x=alt.X("Aktual:Q", title="Aktual (t/ha)"), y=alt.Y("Prediksi:Q", title="Prediksi (t/ha)"))
            + alt.Chart(garis).mark_line(color="#888", strokeDash=[4, 4]).encode(x="x:Q", y="x:Q"),
            width="stretch")
    with c2:
        st.subheader("Kepentingan fitur")
        if nama == "Linear Regression":
            pipe = models[nama]
            names = [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
            fi = pd.DataFrame({"Fitur": names, "Importance": np.abs(pipe.named_steps["model"].coef_)})
            fi = fi.sort_values("Importance", ascending=False)
            st.caption("Nilai mutlak koefisien (fitur numerik sudah distandarkan).")
        else:
            fi = feature_importance(models[nama])
        st.altair_chart(alt.Chart(fi.head(12)).mark_bar(color=TEAL).encode(
            x=alt.X("Importance:Q", title=None), y=alt.Y("Fitur:N", sort="-x", title=None)),
            width="stretch")

    st.subheader("Prediksi per blok pada periode testing")
    b = st.selectbox("Blok", sorted(prediksi.Block_ID.unique()), key="blok_model")
    pb = prediksi[prediksi.Block_ID == b].rename(columns={TARGET: "Aktual"})[["Tanggal", "Aktual", nama]]
    st.altair_chart(alt.Chart(pb.melt("Tanggal", var_name="Seri", value_name="t/ha")).mark_line(point=True).encode(
        x=alt.X("Tanggal:T", title=None), y="t/ha:Q",
        color=alt.Color("Seri:N", scale=alt.Scale(range=["#1f1f1f", TEAL]))), width="stretch")

# ---------------------------------------------------------------- Prediksi
else:
    st.title("Prediksi Produktivitas Bulan Berikutnya")
    models, hasil, _, _ = trained()
    terbaik = hasil[hasil.Model != "Baseline (sama dengan tahun lalu)"].sort_values("RMSE").Model.iloc[0]
    nama = st.selectbox("Model", list(models), index=list(models).index(terbaik))

    blok = st.selectbox("Blok", sorted(df.Block_ID.unique()))
    h = df[(df.Block_ID == blok)].sort_values("Tanggal")
    akhir = h.iloc[-1]
    target_tgl = akhir.Tanggal + pd.DateOffset(months=1)
    seri = h.set_index("Tanggal")[TARGET]
    st.write(f"Data terakhir blok **{blok}**: {NAMA_BULAN[akhir.Month_Number - 1]} {akhir.Year}. "
             f"Prediksi untuk **{NAMA_BULAN[target_tgl.month - 1]} {target_tgl.year}**. "
             "Nilai di bawah terisi otomatis dari riwayat blok dan boleh diubah.")

    c1, c2, c3 = st.columns(3)
    umur = c1.number_input("Umur tanaman (tahun)", 0, 40, int(target_tgl.year - akhir.Planting_Year))
    status = c2.selectbox("Status tanaman", STATUS_URUT[1:], index=STATUS_URUT[1:].index(akhir.Plant_Status)
                          if akhir.Plant_Status in STATUS_URUT[1:] else 3)
    bibit = c3.selectbox("Jenis bibit", sorted(df.Seed_Type.unique()),
                         index=sorted(df.Seed_Type.unique()).index(akhir.Seed_Type))
    c4, c5, c6 = st.columns(3)
    luas = c4.number_input("Luas blok (ha)", 1, 100, int(akhir.Area_Ha))
    kerapatan = c5.number_input("Kerapatan (pohon/ha)", 50.0, 200.0, float(akhir.Palms_per_Ha))
    bulan = c6.selectbox("Bulan", list(range(1, 13)), index=target_tgl.month - 1,
                         format_func=lambda m: NAMA_BULAN[m - 1])

    def riwayat(k):
        t = target_tgl - pd.DateOffset(months=k)
        return float(seri.get(t, np.nan)) if t in seri.index else 0.0

    c7, c8, c9, c10 = st.columns(4)
    l1 = c7.number_input("Produktivitas 1 bulan lalu", 0.0, 6.0, riwayat(1), format="%.3f")
    l2 = c8.number_input("2 bulan lalu", 0.0, 6.0, riwayat(2), format="%.3f")
    l3 = c9.number_input("3 bulan lalu", 0.0, 6.0, riwayat(3), format="%.3f")
    l12 = c10.number_input("Bulan yang sama tahun lalu", 0.0, 6.0, riwayat(12), format="%.3f")

    x = pd.DataFrame([{
        "Plant_Age_Years": umur, "Palms_per_Ha": kerapatan, "Area_Ha": luas,
        "Month_sin": np.sin(2 * np.pi * bulan / 12), "Month_cos": np.cos(2 * np.pi * bulan / 12),
        "lag1": l1, "lag2": l2, "lag3": l3, "lag12": l12, "roll3": (l1 + l2 + l3) / 3,
        "Plant_Status": status, "Seed_Type": bibit,
    }])
    y = max(float(models[nama].predict(x)[0]), 0.0)
    # pembanding: rata-rata blok berstatus sama pada bulan yang sama (pola musiman ikut diperhitungkan)
    rata_tm = df[(df.Plant_Status == status) & (df.Month_Number == bulan)][TARGET].mean()

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Produktivitas prediksi", f"{y:.3f} t/ha")
    m2.metric("Perkiraan produksi blok", f"{y * luas:.1f} ton TBS")
    m3.metric(f"Rata-rata {status} bulan {NAMA_BULAN[bulan - 1]}", f"{rata_tm:.3f} t/ha",
              delta=f"{y - rata_tm:+.3f} t/ha")
    if y < 0.8 * rata_tm:
        st.warning("Prediksi di bawah 80% rata-rata blok berstatus sama pada bulan yang sama. Blok ini perlu "
                   "diperiksa (pemupukan, pemeliharaan, atau rencana peremajaan).")

    grafik = h[["Tanggal", TARGET]].tail(24).assign(Seri="Aktual")
    titik = pd.DataFrame({"Tanggal": [target_tgl], TARGET: [y], "Seri": ["Prediksi"]})
    st.altair_chart(
        alt.Chart(grafik).mark_line(point=True, color="#1f1f1f").encode(
            x=alt.X("Tanggal:T", title=None), y=alt.Y(f"{TARGET}:Q", title="t/ha/bulan"))
        + alt.Chart(titik).mark_point(size=160, filled=True, color=TEAL, shape="diamond").encode(
            x="Tanggal:T", y=f"{TARGET}:Q"),
        width="stretch")
    st.caption("Garis = 24 bulan terakhir data aktual; titik = hasil prediksi bulan berikutnya.")
