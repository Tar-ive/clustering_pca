import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Set page config
st.set_page_config(page_title="Seoul Bike Clustering", layout="wide")

# Load data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("SeoulBikeData.csv", encoding='cp949')
    except UnicodeDecodeError:
        try:
            return pd.read_csv("SeoulBikeData.csv", encoding='euc-kr')
        except UnicodeDecodeError:
            return pd.read_csv("SeoulBikeData.csv", encoding='iso-2022-kr')

data = load_data()

# Title
st.title("Seoul Bike Data Clustering and Visualization")

# Display raw data
st.subheader("Raw Data")
st.write(data.head())

# Preprocess data
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# Feature selection
st.subheader("Feature Selection")
selected_features = st.multiselect("Select features for clustering:", numeric_columns, default=numeric_columns[:3])

if len(selected_features) < 2:
    st.warning("Please select at least two features for clustering.")
    st.stop()

# Prepare data for clustering
X = data[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
st.subheader("Clustering")
clustering_algorithm = st.selectbox("Select clustering algorithm:", ["K-Means", "DBSCAN", "Agglomerative"])

if clustering_algorithm == "K-Means":
    n_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)
    model = KMeans(n_clusters=n_clusters, random_state=42)
elif clustering_algorithm == "DBSCAN":
    eps = st.slider("Select epsilon value:", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    min_samples = st.slider("Select minimum samples:", min_value=2, max_value=10, value=5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
else:  # Agglomerative
    n_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)
    model = AgglomerativeClustering(n_clusters=n_clusters)

cluster_labels = model.fit_predict(X_scaled)

# Add cluster labels to the dataframe
data['Cluster'] = cluster_labels

# Visualize clusters
st.subheader("Cluster Visualization")
if len(selected_features) >= 3:
    fig = px.scatter_3d(data, x=selected_features[0], y=selected_features[1], z=selected_features[2],
                        color='Cluster', hover_data=selected_features)
else:
    fig = px.scatter(data, x=selected_features[0], y=selected_features[1], color='Cluster', hover_data=selected_features)

st.plotly_chart(fig)

# Display basic statistics and insights
st.subheader("Cluster Statistics")
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
for cluster in range(n_clusters):
    st.write(f"Cluster {cluster}:")
    cluster_data = data[data['Cluster'] == cluster]
    st.write(cluster_data[selected_features].describe())
    st.write("---")

# Feature importance (for K-Means only)
if clustering_algorithm == "K-Means":
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': np.abs(model.cluster_centers_).mean(axis=0)
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    fig_importance = px.bar(feature_importance, x='Feature', y='Importance')
    st.plotly_chart(fig_importance)

# Cluster size comparison
st.subheader("Cluster Size Comparison")
cluster_sizes = data['Cluster'].value_counts().sort_index()
fig_sizes = px.pie(values=cluster_sizes.values, names=cluster_sizes.index, title="Cluster Sizes")
st.plotly_chart(fig_sizes)

# Download clustered data
st.subheader("Download Clustered Data")
csv = data.to_csv(index=False)
b64 = BytesIO()
b64.write(csv.encode())
b64.seek(0)
st.download_button(
    label="Download CSV",
    data=b64,
    file_name="seoul_bike_clustered.csv",
    mime="text/csv"
)

# Correlation heatmap
st.subheader("Correlation Heatmap")
corr_matrix = data[selected_features].corr()
fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto")
st.plotly_chart(fig_corr)

# Cluster profiles
st.subheader("Cluster Profiles")
for cluster in range(n_clusters):
    st.write(f"Cluster {cluster} Profile:")
    cluster_profile = data[data['Cluster'] == cluster][selected_features].mean()
    fig_profile = go.Figure(data=[go.Bar(x=cluster_profile.index, y=cluster_profile.values)])
    fig_profile.update_layout(title=f"Average Feature Values for Cluster {cluster}")
    st.plotly_chart(fig_profile)

# Silhouette Score
if clustering_algorithm != "DBSCAN":
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    st.subheader("Clustering Evaluation")
    st.write(f"Silhouette Score: {silhouette_avg:.4f}")

# Conclusion
st.subheader("Conclusion")
st.write(f"""
This clustering analysis provides insights into the patterns and groupings within the Seoul Bike dataset using the {clustering_algorithm} algorithm. 
By examining the cluster visualizations, statistics, and profiles, we can identify distinct groups of bike rental patterns.
These insights can be used to optimize bike-sharing services, predict demand, and improve overall system efficiency.

Key observations:
1. The clustering algorithm and its parameters significantly impact the results.
2. The number and shape of clusters vary depending on the chosen algorithm.
3. Feature importance (for K-Means) highlights which factors have the most influence on cluster formation.
4. Cluster sizes show the distribution of data points across different groups.
5. Correlation heatmap reveals relationships between selected features.
6. Cluster profiles provide a summary of characteristics for each group.

To further improve this analysis, consider:
- Fine-tuning the parameters for each clustering algorithm
- Incorporating more advanced feature engineering techniques
- Analyzing temporal patterns within clusters
- Integrating external data sources (e.g., weather data, events) for richer insights
""")
