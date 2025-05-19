import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="COVID-19 Clustering Dashboard", layout="wide")
st.title("Analisis COVID-19 Indonesia: Prediksi dan Clustering")

@st.cache_data
def load_data():
    df = pd.read_csv("covid_19_indonesia_time_series_all.csv")
    return df

df = load_data()

# Data Cleaning dan Preprocessing
df = df[['Date', 'Location', 'Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']]
df.dropna(inplace=True)

# Konversi tanggal
df['Date'] = pd.to_datetime(df['Date'])

# Sidebar untuk navigasi
analysis_type = st.sidebar.selectbox("Pilih Jenis Analisis", ["Visualisasi Data", "Prediksi Kasus", "Clustering Wilayah"])

if analysis_type == "Visualisasi Data":
    st.header("Visualisasi Data COVID-19 Indonesia")
    
    unique_locations = df['Location'].unique()
    selected_location = st.selectbox("Pilih Lokasi", unique_locations)
    location_data = df[df['Location'] == selected_location]
    
    st.subheader(f"Tren Kasus Harian di {selected_location}")
    
    fig, ax = plt.subplots(figsize=(10,4))
    daily_cases = location_data.groupby("Date").sum()['Total Cases']
    daily_cases.plot(ax=ax, color='red')
    ax.set_ylabel("Total Cases")
    ax.set_xlabel("Date")
    st.pyplot(fig)
    
    # Menampilkan data mentah
    st.subheader("Data Mentah")
    st.dataframe(location_data.sort_values("Date", ascending=False))

elif analysis_type == "Prediksi Kasus":
    st.header("Prediksi Total Kasus COVID-19 (Supervised Learning)")
    
    # Membuat fitur tambahan
    df['Case Fatality Rate'] = df['Total Deaths'] / df['Total Cases']
    df['Recovery Rate'] = df['Total Recovered'] / df['Total Cases']
    
    # Memilih fitur dan target
    features = df[['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate', 'Recovery Rate']]
    target = df['Total Cases']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Model Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Evaluasi
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Evaluasi Model")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")
    
    # Visualisasi hasil prediksi vs aktual
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual Total Cases')
    ax.set_ylabel('Predicted Total Cases')
    ax.set_title('Actual vs Predicted Total Cases')
    st.pyplot(fig)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': features.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x='Importance', y='Feature', data=importance, ax=ax)
    ax.set_title('Feature Importance')
    st.pyplot(fig)

elif analysis_type == "Clustering Wilayah":
    st.header("Clustering Wilayah Berdasarkan Data COVID-19 (Unsupervised Learning)")
    
    # Persiapan data untuk clustering
    cluster_features = df.groupby('Location')[['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']].mean()
    
    # Standardisasi
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)
    
    # KMeans Clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    cluster_features['Cluster'] = clusters
    
    # Koordinat untuk peta (contoh beberapa lokasi)
    koordinat = pd.DataFrame({
        'Location': [
            'Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur',
            'Bali', 'Sumatera Utara', 'Kalimantan Timur', 'Sulawesi Selatan'
        ],
        'lat': [
            -6.2088, -6.9039, -7.1508, -7.2564,
            -8.4095, 3.5932, 0.5383, -5.1477
        ],
        'lon': [
            106.8456, 107.6186, 110.4439, 112.7688,
            115.1889, 98.6722, 116.4194, 119.4327
        ]
    })
    
    # Gabungkan data clustering dengan koordinat
    map_df = cluster_features.reset_index().merge(koordinat, on='Location')
    
    # Visualisasi peta
    st.subheader("Peta Interaktif Clustering Wilayah")
    fig_map = px.scatter_mapbox(
        map_df,
        lat="lat", lon="lon",
        hover_name="Location",
        color="Cluster",
        size="Total Cases",
        zoom=4,
        height=500,
        mapbox_style="carto-positron"
    )
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Ringkasan cluster
    st.subheader("Ringkasan Risiko Wilayah Berdasarkan Cluster")
    
    # Hitung statistik untuk setiap cluster
    cluster_summary = cluster_features.groupby('Cluster').agg({
        'Total Cases': ['mean', 'median', 'max'],
        'Total Deaths': ['mean', 'median', 'max'],
        'Population Density': ['mean', 'median', 'max']
    })
    
    st.dataframe(cluster_summary)
    
    # Visualisasi boxplot untuk setiap fitur per cluster
    st.subheader("Distribusi Fitur per Cluster")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.boxplot(x='Cluster', y='Total Cases', data=cluster_features.reset_index(), ax=axes[0, 0])
    sns.boxplot(x='Cluster', y='Total Deaths', data=cluster_features.reset_index(), ax=axes[0, 1])
    sns.boxplot(x='Cluster', y='Total Recovered', data=cluster_features.reset_index(), ax=axes[1, 0])
    sns.boxplot(x='Cluster', y='Population Density', data=cluster_features.reset_index(), ax=axes[1, 1])
    
    axes[0, 0].set_title('Total Cases per Cluster')
    axes[0, 1].set_title('Total Deaths per Cluster')
    axes[1, 0].set_title('Total Recovered per Cluster')
    axes[1, 1].set_title('Population Density per Cluster')
    
    plt.tight_layout()
    st.pyplot(fig)