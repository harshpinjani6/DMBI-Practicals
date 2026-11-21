import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

columns = ["Area", "Perimeter", "Compactness", "Length", "Width", "Asymmetry", "Groove", "Variety"]
df = pd.read_csv("seeds_dataset.txt", sep=r"\s+", header=None, names=columns)
print(df.head())
print("Dataset Shape:", df.shape)

X = df[columns[:7]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Original Data Shape:", X.shape)
print("Standardized Data Shape:", X_scaled.shape)

inertia_values = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia_values.append(model.inertia_)

plt.plot(range(1, 11), inertia_values, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
print(df[["Area", "Perimeter", "Cluster"]].head())
print("\nCluster Counts:")
print(df["Cluster"].value_counts().sort_index())

silhouette = silhouette_score(X_scaled, df["Cluster"])
inertia = kmeans.inertia_
db_index = davies_bouldin_score(X_scaled, df["Cluster"])
print("Silhouette Score:", round(silhouette, 4))
print("Inertia:", round(inertia, 4))
print("Davies-Bouldin Index:", round(db_index, 4))

plt.figure(figsize=(8, 5))
plt.scatter(df["Area"], df["Perimeter"], c=df["Cluster"], cmap="viridis", s=40)
plt.xlabel("Area")
plt.ylabel("Perimeter")
plt.title("K-Means Clustering Results")
plt.colorbar(label="Cluster")
plt.show()

cluster_means = df.groupby("Cluster")[columns[:7]].mean()
print("Mean Characteristics of Each Cluster:")
print(cluster_means.round(3))
