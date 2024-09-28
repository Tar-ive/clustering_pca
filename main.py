import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

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

# PCA
st.subheader("Dimensionality Reduction")
use_pca = st.checkbox("Use PCA for visualization", value=False)
if use_pca:
    n_components = min(3, len(selected_features))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    st.write(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # Create a DataFrame with PCA components
    pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['Cluster'] = 'Not Assigned'  # Placeholder for cluster labels

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
if use_pca:
    pca_df['Cluster'] = cluster_labels

# Visualize clusters
st.subheader("Cluster Visualization")
if use_pca:
    if n_components >= 3:
        fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Cluster', hover_data=['Cluster'])
    else:
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', hover_data=['Cluster'])
else:
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

# Feature importance analysis
st.subheader("Feature Importance Analysis")

# Function to calculate feature importance
def get_feature_importance(X, labels, method='random_forest'):
    if method == 'random_forest':
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, labels)
        importance = rf.feature_importances_
    elif method == 'mutual_info':
        importance = mutual_info_classif(X, labels)
    else:
        raise ValueError("Invalid method. Choose 'random_forest' or 'mutual_info'.")
    
    return importance

# Calculate feature importance
if clustering_algorithm == "K-Means":
    importance = get_feature_importance(X_scaled, cluster_labels, method='random_forest')
    importance_method = "Random Forest"
elif clustering_algorithm == "DBSCAN":
    importance = get_feature_importance(X_scaled, cluster_labels, method='mutual_info')
    importance_method = "Mutual Information"
else:  # Agglomerative
    importance = get_feature_importance(X_scaled, cluster_labels, method='random_forest')
    importance_method = "Random Forest"

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': importance
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Display feature importance
st.write(f"Feature Importance (using {importance_method}):")
fig_importance = px.bar(feature_importance, x='Feature', y='Importance', title=f"Feature Importance ({importance_method})")
st.plotly_chart(fig_importance)

# Interpretation of feature importance
st.write("""
Feature importance analysis helps us understand which features contribute most to the clustering results. 
The higher the importance score, the more influential the feature is in determining the cluster assignments.

Interpretation:
1. Features with high importance scores are the primary drivers of the clustering results.
2. Features with low importance scores have less impact on the cluster assignments.
3. This analysis can help in feature selection for future iterations of the clustering process.
4. Different clustering algorithms may yield different feature importance results.
""")

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

# Cluster Evaluation Metrics
st.subheader("Cluster Evaluation Metrics")

if clustering_algorithm != "DBSCAN":
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
    
    st.write(f"Silhouette Score: {silhouette_avg:.4f}")
    st.write(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
    st.write(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    
    st.write("""
    Interpretation of Cluster Evaluation Metrics:
    
    1. Silhouette Score:
       - Range: -1 to 1
       - Higher values indicate better-defined clusters
       - A score close to 1 suggests that the data point is well-matched to its own cluster and poorly-matched to neighboring clusters
    
    2. Calinski-Harabasz Index:
       - Also known as the Variance Ratio Criterion
       - Higher values indicate better-defined clusters
       - It measures the ratio of between-cluster dispersion to within-cluster dispersion
    
    3. Davies-Bouldin Index:
       - Lower values indicate better clustering
       - It measures the average similarity between each cluster and its most similar cluster
    
    These metrics help evaluate the quality of the clustering results. However, it's important to note that different metrics may favor different aspects of clustering, so it's beneficial to consider multiple metrics when assessing the results.
    """)
else:
    st.write("Cluster evaluation metrics are not applicable for DBSCAN as it doesn't require a pre-defined number of clusters.")

# PCA Explanation
if use_pca:
    st.subheader("PCA Explanation")
    st.write("""
    Principal Component Analysis (PCA) is a dimensionality reduction technique that helps visualize high-dimensional data in lower dimensions.
    It works by finding the directions (principal components) along which the data varies the most.
    The first principal component (PC1) accounts for the most variance in the data, followed by PC2, and so on.
    
    By using PCA, we can:
    1. Visualize high-dimensional data in 2D or 3D plots
    2. Reduce noise and compress data while preserving most of the important information
    3. Identify the most important features that contribute to the variation in the data
    
    The "Explained variance ratio" shows how much of the total variance in the data is explained by each principal component.
    Higher values indicate that the component captures more information from the original features.
    """)

# Conclusion
st.subheader("Conclusion")
st.write(f"""
This clustering analysis provides insights into the patterns and groupings within the Seoul Bike dataset using the {clustering_algorithm} algorithm. 
By examining the cluster visualizations, statistics, profiles, feature importance, and evaluation metrics, we can identify distinct groups of bike rental patterns and understand the factors that contribute most to these patterns.

Key observations:
1. The clustering algorithm and its parameters significantly impact the results.
2. The number and shape of clusters vary depending on the chosen algorithm.
3. Feature importance analysis highlights which factors have the most influence on cluster formation.
4. Cluster sizes show the distribution of data points across different groups.
5. Correlation heatmap reveals relationships between selected features.
6. Cluster profiles provide a summary of characteristics for each group.
7. PCA helps visualize high-dimensional data and identify the most important features.
8. Cluster evaluation metrics provide quantitative measures of clustering quality.

To further improve this analysis, consider:
- Fine-tuning the parameters for each clustering algorithm
- Incorporating more advanced feature engineering techniques
- Analyzing temporal patterns within clusters
- Integrating external data sources (e.g., weather data, events) for richer insights
- Exploring other dimensionality reduction techniques (e.g., t-SNE, UMAP) for comparison with PCA
- Conducting a more in-depth analysis of feature importance across different clustering algorithms
- Comparing the performance of different clustering algorithms using the evaluation metrics
""")
