import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def clean_percentage_column(series):
    return series.str.replace('%', '', regex=False).astype(float)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("covid_19_indonesia_time_series_all.csv")
        df.columns = df.columns.str.strip()

        df.rename(columns={
            'Total Cases': 'jumlah_total_kasus',
            'Total Deaths': 'jumlah_kematian',
            'Total Recovered': 'jumlah_sembuh',
            'Population Density': 'densitas_populasi',
            'Case Fatality Rate': 'tingkat_fatalitas_kasus',
            'Location': 'lokasi'
        }, inplace=True)

        if 'tingkat_fatalitas_kasus' in df.columns:
            df['tingkat_fatalitas_kasus'] = clean_percentage_column(df['tingkat_fatalitas_kasus'].astype(str))

        return df
    except FileNotFoundError:
        st.error("File tidak ditemukan. Mohon pastikan path-nya benar.")
        return None

df = load_data()

if df is not None:
    # Prediksi Kasus (Supervised Learning)
    st.header('Prediksi Jumlah Total Kasus COVID-19')

    fitur_regresi = ['jumlah_kematian', 'jumlah_sembuh', 'densitas_populasi', 'tingkat_fatalitas_kasus']
    target_regresi = 'jumlah_total_kasus'

    if all(col in df.columns for col in fitur_regresi + [target_regresi]):
        X = df[fitur_regresi].fillna(df[fitur_regresi].mean())
        y = df[target_regresi].fillna(df[target_regresi].mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        st.subheader('Input untuk Prediksi')
        input_data = {}
        for fitur in fitur_regresi:
            input_data[fitur] = st.number_input(f'{fitur.replace("_", " ").title()}', value=float(X[fitur].mean()))

        input_df = pd.DataFrame([input_data])
        hasil_prediksi = model.predict(input_df)
        st.success(f'Prediksi Jumlah Total Kasus: **{hasil_prediksi[0]:,.2f}**')
    else:
        st.warning("Kolom yang dibutuhkan tidak ditemukan.")

    # Clustering Lokasi (Unsupervised Learning)
    st.header('Pengelompokan Lokasi (Clustering)')

    fitur_cluster = ['jumlah_total_kasus', 'jumlah_kematian', 'jumlah_sembuh', 'densitas_populasi']
    if all(col in df.columns for col in fitur_cluster):
        data_cluster = df[fitur_cluster].fillna(df[fitur_cluster].mean())

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_cluster)

        n_clusters = st.slider('Pilih Jumlah Cluster', 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(data_scaled)

        st.subheader("Tabel Hasil Clustering")
        if 'lokasi' in df.columns:
            st.write(df[['lokasi', 'cluster']].dropna().head())
        else:
            st.write(df[['cluster']].dropna().head())

        st.subheader("Visualisasi Clustering")
        fitur_x = st.selectbox("Fitur X", fitur_cluster)
        fitur_y = st.selectbox("Fitur Y", fitur_cluster, index=1)

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=fitur_x, y=fitur_y, hue='cluster', palette='viridis', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Kolom untuk clustering tidak lengkap.")

        
    # BAGIAN 3: Mockup Dashboard
    st.header("Mockup Dashboard")

    # Grafik Tren Harian
    st.subheader("Tren Jumlah Kasus Harian")
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        harian = df.groupby('Date', as_index=False)['jumlah_total_kasus'].sum()

        fig2, ax2 = plt.subplots()
        ax2.plot(harian['Date'], harian['jumlah_total_kasus'])
        ax2.set_title("Tren Jumlah Total Kasus Harian")
        ax2.set_xlabel("Tanggal")
        ax2.set_ylabel("Total Kasus")
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Gagal memuat grafik tren harian: {e}")

    # Peta Interaktif Clustering
    st.subheader("Peta Interaktif Hasil Clustering")
    if all(col in df.columns for col in ['Latitude', 'Longitude', 'cluster']):
        df_map = df[['Latitude', 'Longitude', 'cluster']].dropna()
        st.map(df_map.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}))
    else:
        st.info("Data koordinat tidak lengkap untuk peta.")

    # Ringkasan Risiko Wilayah
    st.subheader("Ringkasan Risiko Wilayah")
    if 'jumlah_total_kasus' in df.columns:
        def risiko(jumlah):
            if jumlah > 100000:
                return 'Tinggi'
            elif jumlah > 50000:
                return 'Sedang'
            else:
                return 'Rendah'

        df['risiko'] = df['jumlah_total_kasus'].apply(risiko)
        if 'lokasi' in df.columns:
            st.write(df[['lokasi', 'jumlah_total_kasus', 'risiko']].dropna().head())
        else:
            st.write(df[['jumlah_total_kasus', 'risiko']].dropna().head())
    else:
        st.info("Data jumlah kasus tidak tersedia.")

else:
    st.warning("❌ Data tidak berhasil dimuat. Pastikan path dan nama file benar.")
