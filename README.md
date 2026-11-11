# Prototipe Prediksi Produktivitas TBS Kelapa Sawit per Blok

Prototipe web analisis untuk projek mata kuliah **Pembelajaran Mesin** (Kelompok 1, 3 TI F,
Politeknik Caltex Riau). Prototipe ini menjadi gambaran dan acuan awal; model, fitur, dan
tampilannya masih akan dikembangkan.

Ketentuan projek yang dipenuhi: bidang **perkebunan**, data **tabular**, **bukan** analisis
sentimen, dan ada **implementasi web**.

## Masalah yang diangkat

Pengelola kebun perlu memperkirakan panen tandan buah segar (TBS) setiap blok untuk
merencanakan tenaga panen dan angkutan, serta menandai blok yang produktivitasnya menurun
sebagai kandidat pemeliharaan atau peremajaan (sejalan dengan program peremajaan BPDP).

**Tugas ML:** regresi, memprediksi `Productivity_t_Ha` (ton TBS per hektare per bulan) satu
bulan ke depan untuk setiap blok.

## Dataset

Mimboro, P. (2026). *Long-Term Block-Level Oil Palm Productivity Dataset with Leaf Nutrient and
Soil Fertility Measurements from an Indonesian Plantation*. Mendeley Data, V2.
https://doi.org/10.17632/93k3733p6x.2 (lisensi CC BY 4.0).

| File di `data/` | Isi |
|---|---|
| `productivity_monthly.csv` | **Dataset utama**: 6.552 baris, 42 blok, bulanan 2012–2024 |
| `leaf_sampling_event.csv` | Hara daun (N, P, K, Ca, Mg, B), 256 sampel 2015–2022 |
| `kcd_block_membership.csv` | Penghubung sampel daun ke blok |
| `data_dictionary.csv`, `README_dataset.md` | Penjelasan kolom dan struktur dataset asli |

## Alur dan hasil sementara

1. **EDA**: produktivitas naik sampai puncak di umur 6–9 tahun lalu turun; panen rendah pada
   Januari dan tinggi pada pertengahan tahun.
2. **Cleaning**: baris TBM (belum berbuah, produktivitas 0) dikeluarkan dari model.
3. **Feature engineering**: bulan siklikal (sin/cos), lag produktivitas 1, 2, 3, dan 12 bulan,
   rata-rata 3 bulan terakhir, ditambah umur, status tanaman, jenis bibit, luas, kerapatan.
   `FFB_Production_t` tidak dipakai karena sama dengan produktivitas × luas (kebocoran target).
4. **Split menurut waktu**: training 2012–2022, testing 2023–2024.
5. **Evaluasi** pada data testing:

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Baseline (sama dengan tahun lalu) | 0,330 | 0,447 | 0,185 | 26,8% |
| Linear Regression | **0,193** | **0,288** | **0,662** | **16,4%** |
| Decision Tree | 0,276 | 0,378 | 0,418 | 21,6% |
| Random Forest | 0,258 | 0,347 | 0,508 | 20,2% |

Semua model mengalahkan baseline. Linear Regression sementara paling baik.

## Halaman web

- **Ringkasan**: tujuan, ukuran data, alur penelitian.
- **Data**: tabel dengan filter blok, tahun, dan status tanaman.
- **Eksplorasi (EDA)**: umur tanaman, pola musiman, status, jenis bibit, riwayat per blok,
  hara daun, korelasi.
- **Model**: tabel metrik, aktual vs prediksi, kepentingan fitur, prediksi per blok.
- **Prediksi**: pilih blok, input terisi otomatis dari riwayat (boleh diubah), keluar prediksi
  produktivitas dan produksi bulan berikutnya beserta peringatan bila di bawah rata-rata.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka http://localhost:8501.

## Struktur

```
app.py               tampilan web (Streamlit)
src/data.py          memuat data dan membentuk fitur
src/model.py         training, evaluasi, kepentingan fitur
data/                dataset (CC BY 4.0)
.streamlit/          tema tampilan
```

## Rencana pengembangan

- [ ] Menambahkan fitur hara daun (N, P, K, Mg) ke model, digabung per blok dan tahun.
- [ ] Mencoba Gradient Boosting atau XGBoost dan tuning hyperparameter dengan validasi
      berbasis waktu (`TimeSeriesSplit`).
- [ ] Klasifikasi blok "perlu perhatian" (produktivitas rendah dibanding blok seumur).
- [ ] Notebook EDA dan laporan untuk dikumpulkan.
- [ ] Deploy ke Streamlit Community Cloud.

## Rujukan

- Ang, Y., Shafri, H. Z. M., Lee, Y. P., Bakar, S. A., Abidin, H., dkk. (2022). Oil palm yield
  prediction across blocks from multi-source data using machine learning and deep learning.
  *Earth Science Informatics*, 15(4), 2349–2367. https://doi.org/10.1007/s12145-022-00882-9
- Kamaruddin, M. A. (2025). Prediction of oil palm yield using machine learning: comparison of
  linear and non-linear algorithms with multivariate time series data. *Journal of Oil Palm
  Research*. https://doi.org/10.21894/jopr.2025.0044
- Urva, G., & Desriyati, W. (2026). Evaluasi komparatif Neural Network dan Random Forest untuk
  prediksi produktivitas tandan buah segar kelapa sawit berbasis fitur musiman. *Jurnal
  Algoritma*, 23(1). https://doi.org/10.33364/algoritma/v.23-1.3352
- Rashid, M., Bari, B. S., Yusup, Y., Kamaruddin, M. A., & Khan, N. (2021). A comprehensive review
  of crop yield prediction using machine learning approaches with special emphasis on palm oil
  yield prediction. *IEEE Access*, 9, 63406–63439. https://doi.org/10.1109/access.2021.3075159
