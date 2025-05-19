import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def bersihkan_kolom_persen(series):
    return series.str.replace('%', '', regex=False).astype(float)

@st.cache_data
def muat_data():
    try:
        df = pd.read_csv("covid_19_indonesia_time_series_all.csv")
        df.columns = df.columns.str.strip()

        df.rename(columns={
            'Total Cases': 'total_kasus',
            'Total Deaths': 'total_kematian',
            'Total Recovered': 'total_sembuh',
            'Population Density': 'kepadatan_penduduk',
            'Case Fatality Rate': 'rasio_fatalitas_kasus',
            'Location': 'nama_wilayah'
        }, inplace=True)

        if 'rasio_fatalitas_kasus' in df.columns:
            df['rasio_fatalitas_kasus'] = bersihkan_kolom_persen(df['rasio_fatalitas_kasus'].astype(str))

        return df
    except FileNotFoundError:
        st.error("⚠️ File tidak ditemukan. Periksa kembali nama file dan lokasi.")
        return None

df = muat_data()

if df is not None:
    # Prediksi Total Kasus (Supervised Learning)
    st.header('Perkiraan Total Kasus COVID-19')

    fitur_prediksi = ['total_kematian', 'total_sembuh', 'kepadatan_penduduk', 'rasio_fatalitas_kasus']
    target_prediksi = 'total_kasus'

    if all(col in df.columns for col in fitur_prediksi + [target_prediksi]):
        X = df[fitur_prediksi].fillna(df[fitur_prediksi].mean())
        y = df[target_prediksi].fillna(df[target_prediksi].mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        st.subheader('Masukkan Data untuk Prediksi')
        input_data = {}
        for fitur in fitur_prediksi:
            input_data[fitur] = st.number_input(f'{fitur.replace("_", " ").title()}', value=float(X[fitur].mean()))

        input_df = pd.DataFrame([input_data])
        hasil_prediksi = model.predict(input_df)
        st.success(f'Perkiraan Total Kasus: **{hasil_prediksi[0]:,.2f}**')
    else:
        st.warning("⚠️ Kolom yang diperlukan tidak ditemukan.")

    # Clustering Wilayah (Unsupervised Learning)
    st.header('Pengelompokan Wilayah Berdasarkan Data Kasus')

    fitur_clustering = ['total_kasus', 'total_kematian', 'total_sembuh', 'kepadatan_penduduk']
    if all(col in df.columns for col in fitur_clustering):
        data_cluster = df[fitur_clustering].fillna(df[fitur_clustering].mean())

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_cluster)

        n_clusters = st.slider('Jumlah Klaster', 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['kelompok'] = kmeans.fit_predict(data_scaled)

        st.subheader("Hasil Clustering")
        if 'nama_wilayah' in df.columns:
            st.write(df[['nama_wilayah', 'kelompok']].dropna().head())
        else:
            st.write(df[['kelompok']].dropna().head())

        st.subheader("Visualisasi Pengelompokan")
        fitur_x = st.selectbox("Pilih Fitur X", fitur_clustering)
        fitur_y = st.selectbox("Pilih Fitur Y", fitur_clustering, index=1)

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=fitur_x, y=fitur_y, hue='kelompok', palette='viridis', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Data yang diperlukan untuk clustering belum lengkap.")

    # Mockup Dashboard
    st.header("Dashboard Interaktif")

    # Grafik Tren Harian
    st.subheader("Perkembangan Kasus Harian")
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        harian = df.groupby('Date', as_index=False)['total_kasus'].sum()

        fig2, ax2 = plt.subplots()
        ax2.plot(harian['Date'], harian['total_kasus'])
        ax2.set_title("Grafik Tren Kasus Harian")
        ax2.set_xlabel("Tanggal")
        ax2.set_ylabel("Total Kasus")
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat grafik: {e}")

    # Peta Interaktif
    st.subheader("Peta Distribusi Cluster")
    if all(col in df.columns for col in ['Latitude', 'Longitude', 'kelompok']):
        df_map = df[['Latitude', 'Longitude', 'kelompok']].dropna()
        st.map(df_map.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}))
    else:
        st.info("ℹ️ Data lokasi tidak lengkap untuk menampilkan peta.")

    # Analisis Risiko Wilayah
    st.subheader("Ringkasan Risiko COVID-19")
    if 'total_kasus' in df.columns:
        def evaluasi_risiko(jumlah):
            if jumlah > 100000:
                return 'Tinggi'
            elif jumlah > 50000:
                return 'Sedang'
            else:
                return 'Rendah'

        df['risiko'] = df['total_kasus'].apply(evaluasi_risiko)
        if 'nama_wilayah' in df.columns:
            st.write(df[['nama_wilayah', 'total_kasus', 'risiko']].dropna().head())
        else:
            st.write(df[['total_kasus', 'risiko']].dropna().head())
    else:
        st.info("ℹ️ Data kasus belum tersedia.")

else:
    st.warning("⚠️ Gagal memuat data. Periksa kembali file dan lokasinya.")
