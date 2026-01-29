"""
Segmentation and clustering functions for tabular data analysis.
Provides various clustering algorithms to segment datasets.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from app.models.choices import ModelType


def perform_clustering(df, model_type, config, feature_columns=None):
    """
    Perform clustering on a dataset using the specified algorithm.

    Args:
        df: pandas DataFrame with the data
        model_type: ModelType enum value (KMEANS, DBSCAN, etc.)
        config: dict with clustering parameters
        feature_columns: list of columns to use for clustering (None = use all numeric)

    Returns:
        dict with clustering results including labels, metrics, and model info
    """
    # Select features for clustering
    if feature_columns is None:
        # Use all numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        feature_columns = numeric_df.columns.tolist()
    else:
        numeric_df = df[feature_columns].select_dtypes(include=[np.number])

    if len(numeric_df.columns) == 0:
        raise ValueError("No numeric columns found for clustering")

    # Handle missing values
    X = numeric_df.fillna(numeric_df.mean())

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform clustering based on model type (support both enum and string values)
    if model_type == ModelType.KMEANS or model_type == 'kmeans':
        labels, model = _kmeans_clustering(X_scaled, config)
    elif model_type == ModelType.DBSCAN or model_type == 'dbscan':
        labels, model = _dbscan_clustering(X_scaled, config)
    elif model_type == ModelType.HIERARCHICAL or model_type == 'hierarchical':
        labels, model = _hierarchical_clustering(X_scaled, config)
    elif model_type == ModelType.GAUSSIAN_MIXTURE or model_type == 'gmm':
        labels, model = _gaussian_mixture_clustering(X_scaled, config)
    elif model_type == ModelType.MEAN_SHIFT or model_type == 'meanshift':
        labels, model = _mean_shift_clustering(X_scaled, config)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {model_type}")

    # Calculate clustering metrics
    n_clusters = len(np.unique(labels[labels >= 0]))  # Exclude noise points (-1)

    metrics = {}
    if n_clusters > 1 and len(labels[labels >= 0]) > 0:
        # Only calculate metrics if we have multiple clusters and non-noise points
        valid_mask = labels >= 0
        if np.sum(valid_mask) > n_clusters:
            metrics = {
                'silhouette_score': float(silhouette_score(X_scaled[valid_mask], labels[valid_mask])),
                'davies_bouldin_score': float(davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask])),
                'calinski_harabasz_score': float(calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask])),
            }

    # Calculate cluster statistics
    cluster_stats = _calculate_cluster_stats(df, labels, feature_columns)

    # Perform dimensionality reduction for visualization
    pca = PCA(n_components=min(2, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    # Build result dictionary
    result = {
        'labels': labels.tolist(),
        'n_clusters': int(n_clusters),
        'n_noise_points': int(np.sum(labels == -1)),
        'feature_columns': feature_columns,
        'metrics': metrics,
        'cluster_stats': cluster_stats,
        'pca_coordinates': X_pca.tolist(),
        'pca_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
    }

    # Add model-specific information
    if model_type == ModelType.KMEANS:
        result['cluster_centers'] = model.cluster_centers_.tolist()
        result['inertia'] = float(model.inertia_)
    elif model_type == ModelType.GAUSSIAN_MIXTURE:
        result['means'] = model.means_.tolist()
        result['covariances_type'] = config.get('covariance_type', 'full')

    return result


def _kmeans_clustering(X, config):
    """K-Means clustering"""
    n_clusters = config.get('n_clusters', 3)
    max_iter = config.get('max_iter', 300)
    random_state = config.get('random_state', 42)

    model = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        random_state=random_state,
        n_init=10
    )
    labels = model.fit_predict(X)
    return labels, model


def _dbscan_clustering(X, config):
    """DBSCAN clustering"""
    eps = config.get('eps', 0.5)
    min_samples = config.get('min_samples', 5)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model


def _hierarchical_clustering(X, config):
    """Hierarchical/Agglomerative clustering"""
    n_clusters = config.get('n_clusters', 3)
    linkage = config.get('linkage', 'ward')

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    labels = model.fit_predict(X)
    return labels, model


def _gaussian_mixture_clustering(X, config):
    """Gaussian Mixture Model clustering"""
    n_components = config.get('n_clusters', 3)
    covariance_type = config.get('covariance_type', 'full')
    random_state = config.get('random_state', 42)

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state
    )
    labels = model.fit_predict(X)
    return labels, model


def _mean_shift_clustering(X, config):
    """Mean Shift clustering"""
    bandwidth = config.get('bandwidth', None)

    model = MeanShift(bandwidth=bandwidth)
    labels = model.fit_predict(X)
    return labels, model


def _calculate_cluster_stats(df, labels, feature_columns):
    """Calculate statistics for each cluster"""
    cluster_stats = {}

    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            cluster_name = 'noise'
        else:
            cluster_name = f'cluster_{cluster_id}'

        mask = labels == cluster_id
        cluster_data = df[mask][feature_columns]

        stats = {
            'size': int(np.sum(mask)),
            'percentage': float(np.sum(mask) / len(labels) * 100),
            'feature_means': {},
            'feature_stds': {},
        }

        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(cluster_data[col]):
                stats['feature_means'][col] = float(cluster_data[col].mean())
                stats['feature_stds'][col] = float(cluster_data[col].std())

        cluster_stats[cluster_name] = stats

    return cluster_stats


def determine_optimal_clusters(df, feature_columns=None, max_clusters=10, method='elbow', max_k=None):
    """
    Determine the optimal number of clusters using elbow method or silhouette analysis.

    Args:
        df: pandas DataFrame
        feature_columns: list of columns to use (None = all numeric)
        max_clusters: maximum number of clusters to test
        method: 'elbow' or 'silhouette'
        max_k: backward compatibility alias for max_clusters

    Returns:
        dict with optimal k and scores for each k
    """
    # Handle backward compatibility
    if max_k is not None:
        max_clusters = max_k
    # Prepare data
    if feature_columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
        feature_columns = numeric_df.columns.tolist()
    else:
        numeric_df = df[feature_columns].select_dtypes(include=[np.number])

    X = numeric_df.fillna(numeric_df.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = []
    k_range = range(2, min(max_clusters + 1, len(X)))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        if method == 'elbow':
            scores.append({
                'k': k,
                'score': float(kmeans.inertia_)
            })
        elif method == 'silhouette':
            score = silhouette_score(X_scaled, labels)
            scores.append({
                'k': k,
                'score': float(score)
            })

    # Determine optimal k
    if method == 'elbow':
        # Use the elbow method (find the point of maximum curvature)
        # Simple heuristic: look for largest drop in inertia
        optimal_k = 3  # default
        if len(scores) > 1:
            drops = [scores[i]['score'] - scores[i+1]['score']
                    for i in range(len(scores)-1)]
            optimal_idx = np.argmax(drops)
            optimal_k = scores[optimal_idx]['k']
    else:  # silhouette
        # Choose k with highest silhouette score
        optimal_idx = np.argmax([s['score'] for s in scores])
        optimal_k = scores[optimal_idx]['k']

    return {
        'optimal_k': optimal_k,
        'scores': scores,
        'method': method
    }
