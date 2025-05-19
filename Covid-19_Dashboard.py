import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Configure page
st.set_page_config(page_title="COVID-19 Clustering Dashboard", layout="wide")
st.title("COVID-19 Analysis Dashboard for Indonesia")

@st.cache_data
def load_data():
    df = pd.read_csv("covid_19_indonesia_time_series_all.csv")
    return df

df = load_data()

# Data preprocessing
df = df[['Date', 'Location', 'Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']]
df.dropna(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Case Fatality Rate'] = df['Total Deaths'] / df['Total Cases']

# Sidebar for navigation
st.sidebar.title("Navigation")
analysis_type = st.sidebar.radio("Select Analysis Type", ["Data Exploration", "Clustering Analysis", "Prediction Model"])

if analysis_type == "Data Exploration":
    st.header("Data Exploration")
    
    # Location selection
    unique_locations = df['Location'].unique()
    selected_location = st.selectbox("Select Location", unique_locations)
    location_data = df[df['Location'] == selected_location]
    
    # Daily trends plot
    st.subheader(f"Daily Case Trends in {selected_location}")
    fig, ax = plt.subplots(figsize=(10, 4))
    daily_cases = location_data.groupby("Date").sum()['Total Cases']
    daily_cases.plot(ax=ax, color='red')
    ax.set_ylabel("Total Cases")
    ax.set_xlabel("Date")
    st.pyplot(fig)
    
    # Show raw data
    st.subheader("Raw Data")
    st.dataframe(location_data.sort_values("Date", ascending=False))

elif analysis_type == "Clustering Analysis":
    st.header("Clustering Analysis of Indonesian Regions")
    
    # Prepare data for clustering
    cluster_features = df.groupby('Location')[['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']].mean()
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    cluster_features['Cluster'] = clusters
    
    # Coordinates for Indonesian provinces
    coordinates = pd.DataFrame({
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
    
    # Merge with cluster data
    map_df = cluster_features.reset_index().merge(coordinates, on='Location')
    
    # Interactive map
    st.subheader("Interactive Cluster Map of Regions")
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
    
    # Cluster summary
    st.subheader("Region Risk Summary by Cluster")
    st.dataframe(cluster_features.sort_values("Cluster"))

elif analysis_type == "Prediction Model":
    st.header("COVID-19 Case Prediction Model")
    
    # Prepare data for prediction
    prediction_data = df.groupby('Location').agg({
        'Total Cases': 'max',
        'Total Deaths': 'max',
        'Total Recovered': 'max',
        'Population Density': 'mean',
        'Case Fatality Rate': 'mean'
    }).reset_index()
    
    # Feature selection
    features = st.multiselect(
        "Select Features for Prediction Model",
        ['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate'],
        default=['Total Deaths', 'Total Recovered', 'Population Density']
    )
    
    if len(features) < 1:
        st.warning("Please select at least one feature")
    else:
        X = prediction_data[features]
        y = prediction_data['Total Cases']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        st.subheader("Model Performance")
        st.write(f"Mean Absolute Error: {mae:,.2f}")
        
        # Feature importance
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))
        
        # Prediction interface
        st.subheader("Make Predictions")
        input_data = {}
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(features):
            if i % 2 == 0:
                input_data[feature] = col1.number_input(f"Enter {feature}", min_value=0.0, value=X[feature].mean())
            else:
                input_data[feature] = col2.number_input(f"Enter {feature}", min_value=0.0, value=X[feature].mean())
        
        if st.button("Predict Total Cases"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            st.success(f"Predicted Total Cases: {prediction[0]:,.0f}")