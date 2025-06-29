# ==============================================================================
# PRINCIPAL COMPONENT ANALYSIS (PCA) TEMPLATE
# ==============================================================================
# Purpose: Dimensionality reduction and data visualization with PCA
# Replace 'your_data.csv' with your dataset
# Update column names to match your data
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load your data
# Replace 'your_data.csv' with your actual dataset
df = pd.read_csv('your_data.csv')

print("Principal Component Analysis (PCA)")
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

# Separate features and target (if available)
# Replace 'target_column' with your target variable, or set to None if no target
target_column = 'target_column'  # Set to None if no target

if target_column and target_column in df.columns:
    feature_columns = [col for col in df.columns if col != target_column]
    has_target = True
    y = df[target_column]
    print(f"Target variable: {target_column}")
else:
    feature_columns = df.columns.tolist()
    has_target = False
    print("No target variable specified - unsupervised PCA")

# Select only numeric features for PCA
numeric_df = df[feature_columns].select_dtypes(include=[np.number])
print(f"Using {numeric_df.shape[1]} numeric features for PCA")

if numeric_df.shape[1] < 2:
    print("Error: Need at least 2 numeric features for PCA")
    exit()

# ================================
# 2. FEATURE SCALING
# ================================
print("\n2. Feature Scaling")
print("-" * 20)

# Standardize features (critical for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

print("Features standardized (mean=0, std=1)")
print("Note: Feature scaling is essential for PCA!")

# ================================
# 3. INITIAL PCA ANALYSIS
# ================================
print("\n3. Initial PCA Analysis")
print("-" * 20)

# Fit PCA with all components
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# Calculate explained variance
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("Explained variance by component:")
for i in range(min(10, len(explained_variance_ratio))):  # Show first 10 components
    print(f"PC{i+1}: {explained_variance_ratio[i]:.3f} ({explained_variance_ratio[i]*100:.1f}%)")

print(f"\nCumulative explained variance:")
for threshold in [0.80, 0.90, 0.95, 0.99]:
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    print(f"{threshold*100:.0f}% variance: {n_components} components")

# ================================
# 4. OPTIMAL NUMBER OF COMPONENTS
# ================================
print("\n4. Determining Optimal Components")
print("-" * 20)

# Find elbow point (where explained variance drops significantly)
# Calculate second derivative to find elbow
if len(explained_variance_ratio) > 2:
    second_derivative = np.diff(explained_variance_ratio, 2)
    elbow_point = np.argmax(second_derivative) + 2  # +2 due to second derivative
    print(f"Elbow point suggests: {elbow_point} components")

# Choose components for 95% variance
optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
optimal_components = min(optimal_components, len(explained_variance_ratio))
print(f"Components for 95% variance: {optimal_components}")

# ================================
# 5. FINAL PCA WITH OPTIMAL COMPONENTS
# ================================
print("\n5. Final PCA Transformation")
print("-" * 20)

# Apply PCA with optimal number of components
pca_optimal = PCA(n_components=optimal_components)
X_pca = pca_optimal.fit_transform(X_scaled)

print(f"Reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} dimensions")
print(f"Retained variance: {cumulative_variance[optimal_components-1]:.3f}")

# Create DataFrame with principal components
pc_columns = [f'PC{i+1}' for i in range(optimal_components)]
pca_df = pd.DataFrame(X_pca, columns=pc_columns)

if has_target:
    pca_df[target_column] = y.values

# ================================
# 6. COMPONENT ANALYSIS
# ================================
print("\n6. Principal Component Analysis")
print("-" * 20)

# Get component loadings (eigenvectors)
loadings = pca_optimal.components_.T * np.sqrt(pca_optimal.explained_variance_)
loadings_df = pd.DataFrame(loadings, 
                          columns=[f'PC{i+1}' for i in range(optimal_components)],
                          index=numeric_df.columns)

print("Top loadings for first 3 principal components:")
for i in range(min(3, optimal_components)):
    pc_name = f'PC{i+1}'
    top_loadings = loadings_df[pc_name].abs().sort_values(ascending=False).head(5)
    print(f"\n{pc_name} (explains {explained_variance_ratio[i]*100:.1f}% variance):")
    for feature, loading in top_loadings.items():
        sign = '+' if loadings_df.loc[feature, pc_name] > 0 else '-'
        print(f"  {sign} {feature}: {abs(loading):.3f}")

# ================================
# 7. VISUALIZATIONS
# ================================
print("\n7. Creating Visualizations")
print("-" * 20)

# Set up the plotting area
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Principal Component Analysis Results', fontsize=16, fontweight='bold')

# Plot 1: Scree Plot (Explained Variance)
axes[0, 0].plot(range(1, len(explained_variance_ratio[:20]) + 1), 
               explained_variance_ratio[:20], 'bo-', linewidth=2, markersize=6)
axes[0, 0].set_xlabel('Principal Component')
axes[0, 0].set_ylabel('Explained Variance Ratio')
axes[0, 0].set_title('Scree Plot')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Cumulative Explained Variance
axes[0, 1].plot(range(1, len(cumulative_variance[:20]) + 1), 
               cumulative_variance[:20], 'ro-', linewidth=2, markersize=6)
axes[0, 1].axhline(y=0.95, color='green', linestyle='--', label='95% Variance')
axes[0, 1].axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')
axes[0, 1].set_xlabel('Number of Components')
axes[0, 1].set_ylabel('Cumulative Explained Variance')
axes[0, 1].set_title('Cumulative Explained Variance')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: First Two Principal Components
if optimal_components >= 2:
    if has_target and y.nunique() < 10:  # Categorical target
        scatter = axes[0, 2].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=axes[0, 2])
    else:
        axes[0, 2].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    
    axes[0, 2].set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% variance)')
    axes[0, 2].set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}% variance)')
    axes[0, 2].set_title('First Two Principal Components')
    axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Component Loadings Heatmap
if optimal_components <= 10:  # Only if reasonable number of components
    sns.heatmap(loadings_df.iloc[:, :min(optimal_components, 10)].T, 
               annot=False, cmap='RdBu_r', center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Component Loadings Heatmap')
    axes[1, 0].set_xlabel('Original Features')
    axes[1, 0].set_ylabel('Principal Components')

# Plot 5: Top Loadings for PC1
if optimal_components >= 1:
    top_loadings_pc1 = loadings_df['PC1'].abs().sort_values(ascending=True).tail(10)
    axes[1, 1].barh(range(len(top_loadings_pc1)), top_loadings_pc1.values)
    axes[1, 1].set_yticks(range(len(top_loadings_pc1)))
    axes[1, 1].set_yticklabels(top_loadings_pc1.index)
    axes[1, 1].set_xlabel('Absolute Loading')
    axes[1, 1].set_title('Top Feature Loadings for PC1')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

# Plot 6: 3D Plot (if 3+ components)
if optimal_components >= 3:
    ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
    if has_target and y.nunique() < 10:
        scatter = ax_3d.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.6)
    else:
        ax_3d.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.6)
    
    ax_3d.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
    ax_3d.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
    ax_3d.set_zlabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)')
    ax_3d.set_title('First Three Principal Components')
else:
    # Remove empty subplot
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()

# ================================
# 8. PCA FOR CLASSIFICATION (if target available)
# ================================
if has_target and y.nunique() < 10:  # Classification task
    print("\n8. PCA for Classification")
    print("-" * 20)
    
    # Compare classification performance: original vs PCA features
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_pca, X_test_pca, _, _ = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train classifiers
    clf_original = LogisticRegression(random_state=42, max_iter=1000)
    clf_pca = LogisticRegression(random_state=42, max_iter=1000)
    
    clf_original.fit(X_train, y_train)
    clf_pca.fit(X_train_pca, y_train)
    
    # Make predictions
    y_pred_original = clf_original.predict(X_test)
    y_pred_pca = clf_pca.predict(X_test_pca)
    
    # Compare performance
    acc_original = accuracy_score(y_test, y_pred_original)
    acc_pca = accuracy_score(y_test, y_pred_pca)
    
    print(f"Original features ({X_scaled.shape[1]} dims): {acc_original:.3f} accuracy")
    print(f"PCA features ({X_pca.shape[1]} dims): {acc_pca:.3f} accuracy")
    print(f"Dimensionality reduction: {(1 - X_pca.shape[1]/X_scaled.shape[1])*100:.1f}%")
    
    performance_diff = acc_pca - acc_original
    if performance_diff >= -0.05:  # Within 5% of original performance
        print("✓ PCA maintains classification performance while reducing dimensions")
    else:
        print("⚠ PCA significantly reduces classification performance")

# ================================
# 9. INVERSE TRANSFORM EXAMPLE
# ================================
print("\n9. Inverse Transform Example")
print("-" * 20)

# Demonstrate reconstruction of data from principal components
sample_idx = 0
original_sample = X_scaled[sample_idx:sample_idx+1]
pca_sample = X_pca[sample_idx:sample_idx+1]

# Inverse transform
reconstructed_sample = pca_optimal.inverse_transform(pca_sample)

# Calculate reconstruction error
reconstruction_error = np.mean((original_sample - reconstructed_sample) ** 2)
print(f"Sample reconstruction error: {reconstruction_error:.6f}")

# Show original vs reconstructed for first few features
print("\nOriginal vs Reconstructed (first 5 features):")
for i in range(min(5, len(numeric_df.columns))):
    feature_name = numeric_df.columns[i]
    original_val = original_sample[0, i]
    reconstructed_val = reconstructed_sample[0, i]
    print(f"{feature_name}: {original_val:.3f} → {reconstructed_val:.3f}")

# ================================
# 10. FEATURE IMPORTANCE IN PC SPACE
# ================================
print("\n10. Feature Importance in Principal Component Space")
print("-" * 20)

# Calculate feature importance based on contribution to top PCs
feature_importance = np.zeros(len(numeric_df.columns))

for i in range(min(3, optimal_components)):  # Top 3 PCs
    pc_weight = explained_variance_ratio[i]
    feature_importance += np.abs(loadings_df.iloc[:, i].values) * pc_weight

# Create importance dataframe
importance_df = pd.DataFrame({
    'Feature': numeric_df.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("Top 10 most important features in PC space:")
print(importance_df.head(10))

# ================================
# 11. MAKING TRANSFORMATIONS ON NEW DATA
# ================================
print("\n11. Applying PCA to New Data")
print("-" * 20)

# Example of how to apply PCA to new data
new_data_example = numeric_df.iloc[:5].copy()  # Use first 5 samples as example

# Apply the same scaling and PCA transformation
new_data_scaled = scaler.transform(new_data_example)
new_data_pca = pca_optimal.transform(new_data_scaled)

print("Example PCA transformation for new data:")
print("Original shape:", new_data_example.shape)
print("PCA shape:", new_data_pca.shape)
print("First sample PC values:", new_data_pca[0][:min(5, optimal_components)])

print("\nPCA analysis completed!")
print("="*50)

# ================================
# SUMMARY & RECOMMENDATIONS
# ================================
print("\nSUMMARY & RECOMMENDATIONS:")
print(f"• Reduced dimensionality from {X_scaled.shape[1]} to {optimal_components} features")
print(f"• Retained {cumulative_variance[optimal_components-1]*100:.1f}% of original variance")
print("• PCA assumes linear relationships between variables")
print("• Feature scaling is crucial for meaningful PCA results")
print("• Principal components are linear combinations of original features")
print("• Use PCA for visualization, noise reduction, and feature extraction")

print("\nWhen to use PCA:")
print("• High-dimensional data with multicollinearity")
print("• Visualization of complex datasets")
print("• Preprocessing for machine learning algorithms")
print("• Data compression and storage optimization")

print("\nRemember to:")
print("1. Replace 'your_data.csv' with your dataset")
print("2. Update 'target_column' or set to None if no target")
print("3. Ensure features are numeric for PCA")
print("4. Always scale features before applying PCA")
print("5. Interpret principal components based on feature loadings")
