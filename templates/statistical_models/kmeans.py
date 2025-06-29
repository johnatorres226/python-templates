# ==============================================================================
# K-MEANS CLUSTERING TEMPLATE
# ==============================================================================
# Purpose: Unsupervised clustering analysis with K-Means
# Replace 'your_data.csv' with your dataset
# Update column names to match your data
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Load your data
# Replace 'your_data.csv' with your actual dataset
df = pd.read_csv('your_data.csv')

print("K-Means Clustering Analysis")
print("="*50)

# ================================
# 1. DATA PREPARATION
# ================================
print("1. Data Preparation")
print("-" * 20)

# Display basic info
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Handle missing values
df = df.dropna()

# Select only numeric features for clustering
numeric_df = df.select_dtypes(include=[np.number])
print(f"Using {numeric_df.shape[1]} numeric features for clustering")

if numeric_df.shape[1] < 2:
    print("Error: Need at least 2 numeric features for clustering")
    exit()

# Remove any potential target column if it exists
# Replace 'target_column' with your target variable name, or set to None
target_column = 'target_column'  # Set to None if no target

if target_column and target_column in numeric_df.columns:
    X = numeric_df.drop(target_column, axis=1)
    true_labels = numeric_df[target_column]
    has_true_labels = True
    print(f"Removed target column: {target_column}")
else:
    X = numeric_df
    has_true_labels = False
    print("No target column specified - pure unsupervised clustering")

print(f"Final feature count: {X.shape[1]}")

# ================================
# 2. FEATURE SCALING
# ================================
print("\n2. Feature Scaling")
print("-" * 20)

# Standardize features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features standardized (mean=0, std=1)")
print("Note: Feature scaling is important for K-Means!")

# ================================
# 3. DETERMINING OPTIMAL K
# ================================
print("\n3. Determining Optimal Number of Clusters")
print("-" * 20)

# Range of K values to test
k_range = range(2, min(21, len(X_scaled)))  # Test up to 20 clusters or n_samples
inertias = []
silhouette_scores = []

print("Testing different values of K...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate inertia (within-cluster sum of squares)
    inertias.append(kmeans.inertia_)
    
    # Calculate silhouette score
    sil_score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(sil_score)
    
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")

# Find optimal K using elbow method
# Calculate the rate of change in inertia
if len(inertias) > 2:
    # Calculate second derivative to find elbow
    inertia_diff = np.diff(inertias)
    inertia_diff2 = np.diff(inertia_diff)
    elbow_k = np.argmax(inertia_diff2) + 3  # +3 due to second derivative and 0-indexing
    elbow_k = min(elbow_k, max(k_range))
else:
    elbow_k = 3

# Find optimal K using silhouette score
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]

print(f"\nOptimal K suggestions:")
print(f"Elbow method: K = {elbow_k}")
print(f"Silhouette score: K = {optimal_k_silhouette}")
print(f"Best silhouette score: {max(silhouette_scores):.3f}")

# Choose final K (prioritize silhouette score)
optimal_k = optimal_k_silhouette

# ================================
# 4. FINAL K-MEANS CLUSTERING
# ================================
print("\n4. Final K-Means Clustering")
print("-" * 20)

# Apply K-Means with optimal K
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X_scaled)

print(f"Applied K-Means with K = {optimal_k}")
print(f"Final inertia: {final_kmeans.inertia_:.2f}")
print(f"Final silhouette score: {silhouette_score(X_scaled, cluster_labels):.3f}")

# Add cluster labels to original dataframe
df_clustered = df.copy()
df_clustered['Cluster'] = cluster_labels

# Cluster distribution
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
print(f"\nCluster distribution:")
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")

# ================================
# 5. CLUSTER ANALYSIS
# ================================
print("\n5. Cluster Analysis")
print("-" * 20)

# Calculate cluster centers in original scale
cluster_centers_scaled = final_kmeans.cluster_centers_
cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)

# Create cluster centers dataframe
centers_df = pd.DataFrame(cluster_centers_original, 
                         columns=X.columns,
                         index=[f'Cluster_{i}' for i in range(optimal_k)])

print("Cluster Centers (original scale):")
print(centers_df.round(3))

# Calculate cluster statistics
print("\nCluster Statistics:")
cluster_stats = df_clustered.groupby('Cluster')[X.columns].agg(['mean', 'std']).round(3)
print(cluster_stats)

# ================================
# 6. CLUSTER VALIDATION
# ================================
print("\n6. Cluster Validation")
print("-" * 20)

# Silhouette analysis for each cluster
sil_samples = silhouette_samples(X_scaled, cluster_labels)

print("Silhouette scores by cluster:")
for i in range(optimal_k):
    cluster_sil_scores = sil_samples[cluster_labels == i]
    print(f"Cluster {i}: {cluster_sil_scores.mean():.3f} "
          f"(min: {cluster_sil_scores.min():.3f}, max: {cluster_sil_scores.max():.3f})")

# Within-cluster sum of squares (WCSS) for each cluster
wcss_per_cluster = []
for i in range(optimal_k):
    cluster_points = X_scaled[cluster_labels == i]
    cluster_center = cluster_centers_scaled[i]
    wcss = np.sum((cluster_points - cluster_center) ** 2)
    wcss_per_cluster.append(wcss)
    print(f"Cluster {i} WCSS: {wcss:.2f}")

# ================================
# 7. VISUALIZATIONS
# ================================
print("\n7. Creating Visualizations")
print("-" * 20)

# Set up the plotting area
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('K-Means Clustering Analysis Results', fontsize=16, fontweight='bold')

# Plot 1: Elbow Method
axes[0, 0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=6)
axes[0, 0].axvline(x=elbow_k, color='red', linestyle='--', label=f'Elbow K={elbow_k}')
axes[0, 0].set_xlabel('Number of Clusters (K)')
axes[0, 0].set_ylabel('Inertia (WCSS)')
axes[0, 0].set_title('Elbow Method for Optimal K')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Silhouette Score
axes[0, 1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=6)
axes[0, 1].axvline(x=optimal_k_silhouette, color='red', linestyle='--', 
                  label=f'Best K={optimal_k_silhouette}')
axes[0, 1].set_xlabel('Number of Clusters (K)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Score vs Number of Clusters')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Cluster Visualization (2D PCA)
if X_scaled.shape[1] > 2:
    # Use PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    
    scatter = axes[0, 2].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                cmap='viridis', alpha=0.6)
    
    # Plot cluster centers
    centers_pca = pca.transform(cluster_centers_scaled)
    axes[0, 2].scatter(centers_pca[:, 0], centers_pca[:, 1], 
                      c='red', marker='x', s=200, linewidths=3, label='Centers')
    
    axes[0, 2].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)')
    axes[0, 2].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)')
    axes[0, 2].set_title('Clusters in PCA Space')
    axes[0, 2].legend()
    plt.colorbar(scatter, ax=axes[0, 2])

else:
    # Direct 2D visualization
    scatter = axes[0, 2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, 
                                cmap='viridis', alpha=0.6)
    axes[0, 2].scatter(cluster_centers_scaled[:, 0], cluster_centers_scaled[:, 1], 
                      c='red', marker='x', s=200, linewidths=3, label='Centers')
    axes[0, 2].set_xlabel(X.columns[0])
    axes[0, 2].set_ylabel(X.columns[1])
    axes[0, 2].set_title('Clusters (Scaled Features)')
    axes[0, 2].legend()
    plt.colorbar(scatter, ax=axes[0, 2])

# Plot 4: Silhouette Analysis
from matplotlib.patches import Rectangle
y_lower = 10
colors = plt.cm.viridis(np.linspace(0, 1, optimal_k))

for i in range(optimal_k):
    cluster_sil_scores = sil_samples[cluster_labels == i]
    cluster_sil_scores.sort()
    
    size_cluster_i = cluster_sil_scores.shape[0]
    y_upper = y_lower + size_cluster_i
    
    axes[1, 0].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_scores,
                           facecolor=colors[i], edgecolor='black', alpha=0.7)
    
    axes[1, 0].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

axes[1, 0].set_xlabel('Silhouette Score')
axes[1, 0].set_ylabel('Cluster')
axes[1, 0].set_title('Silhouette Analysis')
axes[1, 0].axvline(x=silhouette_score(X_scaled, cluster_labels), color='red', 
                  linestyle='--', label='Average Score')
axes[1, 0].legend()

# Plot 5: Cluster Size Distribution
axes[1, 1].bar(cluster_counts.index, cluster_counts.values, 
              color=['blue', 'orange', 'green', 'red', 'purple'][:optimal_k])
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Number of Samples')
axes[1, 1].set_title('Cluster Size Distribution')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Plot 6: Feature Importance (Cluster Center Ranges)
feature_ranges = []
for col in X.columns:
    col_centers = centers_df[col].values
    feature_range = col_centers.max() - col_centers.min()
    feature_ranges.append(feature_range)

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Range': feature_ranges
}).sort_values('Range', ascending=True)

axes[1, 2].barh(range(len(feature_importance_df)), feature_importance_df['Range'])
axes[1, 2].set_yticks(range(len(feature_importance_df)))
axes[1, 2].set_yticklabels(feature_importance_df['Feature'])
axes[1, 2].set_xlabel('Range Across Cluster Centers')
axes[1, 2].set_title('Feature Discriminative Power')
axes[1, 2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# ================================
# 8. CLUSTER INTERPRETATION
# ================================
print("\n8. Cluster Interpretation")
print("-" * 20)

print("Cluster Characteristics:")
for i in range(optimal_k):
    cluster_data = df_clustered[df_clustered['Cluster'] == i]
    print(f"\nCluster {i} (n={len(cluster_data)}):")
    
    # Find defining characteristics (features with highest/lowest values)
    cluster_means = cluster_data[X.columns].mean()
    overall_means = df[X.columns].mean()
    
    # Features where this cluster is most different from overall mean
    differences = ((cluster_means - overall_means) / overall_means).abs()
    top_features = differences.nlargest(3)
    
    for feature in top_features.index:
        cluster_val = cluster_means[feature]
        overall_val = overall_means[feature]
        if cluster_val > overall_val:
            print(f"  • High {feature}: {cluster_val:.2f} vs {overall_val:.2f} (overall)")
        else:
            print(f"  • Low {feature}: {cluster_val:.2f} vs {overall_val:.2f} (overall)")

# ================================
# 9. CLUSTER QUALITY ASSESSMENT
# ================================
print("\n9. Cluster Quality Assessment")
print("-" * 20)

# Calculate various clustering metrics
overall_silhouette = silhouette_score(X_scaled, cluster_labels)
print(f"Overall Silhouette Score: {overall_silhouette:.3f}")

# Interpretation guidelines
if overall_silhouette > 0.7:
    print("✓ Excellent clustering structure")
elif overall_silhouette > 0.5:
    print("✓ Good clustering structure")
elif overall_silhouette > 0.25:
    print("⚠ Weak clustering structure")
else:
    print("✗ Poor clustering structure")

# Calculate Calinski-Harabasz Index (Variance Ratio Criterion)
try:
    from sklearn.metrics import calinski_harabasz_score
    ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
    print(f"Calinski-Harabasz Score: {ch_score:.2f}")
except ImportError:
    print("Calinski-Harabasz Score: Not available")

# ================================
# 10. ASSIGNING NEW DATA TO CLUSTERS
# ================================
print("\n10. Assigning New Data to Clusters")
print("-" * 20)

# Example of how to assign new data to clusters
new_data_example = X.iloc[:5].copy()  # Use first 5 samples as example

# Apply the same scaling and clustering
new_data_scaled = scaler.transform(new_data_example)
new_cluster_labels = final_kmeans.predict(new_data_scaled)

print("Example cluster assignments for new data:")
for i, cluster in enumerate(new_cluster_labels):
    print(f"Sample {i+1}: Assigned to Cluster {cluster}")

# Calculate distances to cluster centers
distances = final_kmeans.transform(new_data_scaled)
print("\nDistances to cluster centers:")
for i, dist_row in enumerate(distances):
    closest_cluster = np.argmin(dist_row)
    closest_distance = dist_row[closest_cluster]
    print(f"Sample {i+1}: Closest to Cluster {closest_cluster} (distance: {closest_distance:.2f})")

print("\nK-Means clustering analysis completed!")
print("="*50)

# ================================
# SUMMARY & RECOMMENDATIONS
# ================================
print("\nSUMMARY & RECOMMENDATIONS:")
print(f"• Identified {optimal_k} clusters in the data")
print(f"• Silhouette score: {overall_silhouette:.3f}")
print(f"• Largest cluster: {cluster_counts.max()} samples")
print(f"• Smallest cluster: {cluster_counts.min()} samples")

print("\nK-Means Assumptions & Limitations:")
print("• Assumes spherical clusters of similar size")
print("• Sensitive to initialization (use multiple runs)")
print("• Requires preprocessing (scaling, outlier removal)")
print("• K must be specified in advance")

print("\nNext Steps:")
print("• Validate clusters with domain expertise")
print("• Consider alternative clustering methods if assumptions violated")
print("• Use clusters as features for supervised learning")
print("• Perform cluster-specific analyses")

print("\nRemember to:")
print("1. Replace 'your_data.csv' with your dataset")
print("2. Update 'target_column' or set to None if no target")
print("3. Ensure features are numeric and scaled")
print("4. Validate cluster assignments make business sense")
print("5. Consider outlier removal before clustering")
