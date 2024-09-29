# Project Overview
## What does this project do?
This project analyzes and clusters the Seoul Bike rental dataset to uncover patterns in bike rental behaviors in Seoul. Using machine learning and data visualization techniques, it provides insights into various aspects of bike rentals, such as the influence of weather conditions, date, and time on bike usage.

## How does it do it?
Here's a step-by-step breakdown of how the project works:

1. Data Loading:
The project begins by loading the Seoul Bike dataset (SeoulBikeData.csv). It uses pandas to read the CSV file and handles automatic encoding detection to ensure compatibility with different file encodings.

2. Feature Selection:
Users can select the features (columns) they want to include in the clustering analysis. This step is crucial for focusing the analysis on relevant aspects of the data.

3. Dimensionality Reduction (Optional):
For better visualization, users can apply Principal Component Analysis (PCA). PCA reduces the dimensionality of the data, making it easier to visualize in 2D or 3D plots.

4. Clustering Algorithms:
The project supports three clustering algorithms:
K-Means: Partitions the data into a specified number of clusters.
DBSCAN (Density-Based Spatial Clustering of Applications with Noise): Identifies clusters based on the density of data points.
Agglomerative Clustering: Performs hierarchical clustering based on a bottom-up approach.
Users can choose the clustering algorithm and set parameters such as the number of clusters for K-Means or the epsilon value for DBSCAN.

5. Cluster Visualization:
The clustered data is visualized using plotly. The project provides 2D and 3D scatter plots showing the different clusters, allowing users to interpret the results visually.

6. Feature Importance Analysis:
After clustering, the project identifies which features (variables) are important for driving the formation of each cluster. This helps users understand the key factors influencing bike rental patterns.

7. Cluster Evaluation Metrics:
The project provides evaluation metrics (where applicable) to assess the quality of the clusters:
Silhouette Score: Measures the cohesion and separation of clusters.
Calinski-Harabasz Index: Evaluates the ratio of between-cluster dispersion to within-cluster dispersion.
Davies-Bouldin Index: Assesses the average similarity between clusters.

8. Downloadable Results:
Users can download the clustered data as a CSV file for further analysis or reporting.

9. Interactive User Interface:
The project uses streamlit to provide an interactive web interface. Users can select features, choose clustering algorithms, adjust parameters, and visualize results—all through an intuitive web app.

# Conclusion
By following these steps, the project enables users to explore and analyze the Seoul Bike dataset, providing valuable insights into bike rental patterns and behavior in Seoul.

# Installation
## Clone this repository:

```
git clone https://github.com/Tar-ive/clustering_pca
cd seoulbikeclustering 
```

## Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate
````

## Install the required packages:

```
pip install -r requirements.txt
```
# Usage
Ensure you have the `SeoulBikeData.csv` file in the project directory.

# Run the Streamlit app:

streamlit run main.py
Open your web browser and go to http://localhost:5000 to interact with the application.

# Running Locally
To run the application locally:

Make sure you have completed the installation steps above.

Open a terminal or command prompt.

Navigate to the project directory.

Activate your virtual environment if you created one:

`source venv/bin/activate`  # On Windows, use `venv\Scripts\activate`
Run the Streamlit app:

streamlit run main.py
Your default web browser should automatically open with the application. If not, open a browser and go to http://localhost:5000.

Interact with the application by selecting features, choosing clustering algorithms, and exploring the visualizations.

# Data
The SeoulBikeData.csv file contains information about bike rentals in Seoul, including various features such as date, temperature, humidity, wind speed, and more. Make sure this file is present in the project directory before running the application.

# Contributing
Contributions to improve the project are welcome. Please follow these steps:

# Fork the repository
## Create a new branch
```
(git checkout -b feature/improvement)
 ```
 
## Make your changes
Commit your changes 
```
(git commit -am 'Add new feature')
```
## Push to the branch 
```
(git push origin feature/improvement)
```
## Create a new Pull Request

# License
This project is licensed under the MIT License.

## If you have any questions or issues, please open an issue on the GitHub repository.

