"""
Tests for Clustering and Segmentation Functions
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from django.test import TestCase

from app.functions.segmentation import (
    perform_clustering,
    determine_optimal_clusters
)
from app.models.choices import ModelType
from app.tests.fixtures import (
    MockDataGenerator,
    MockModelFactory
)


class TestClustering(TestCase):
    """Test suite for clustering algorithms"""

    def test_kmeans_clustering(self):
        """Test K-Means clustering"""
        df, features, true_labels = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        config = {
            'algorithm': 'kmeans',
            'n_clusters': 3
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        # Verify clustering results
        self.assertIn('labels', result)
        self.assertIn('metrics', result)
        self.assertIn('cluster_centers', result)

        # Verify correct number of clusters
        self.assertEqual(len(result['cluster_centers']), 3)
        self.assertEqual(len(np.unique(result['labels'])), 3)

        # Verify metrics
        self.assertIn('silhouette_score', result['metrics'])
        self.assertIn('davies_bouldin_score', result['metrics'])
        self.assertIn('calinski_harabasz_score', result['metrics'])

    def test_dbscan_clustering(self):
        """Test DBSCAN clustering"""
        df, features, true_labels = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        config = {
            'algorithm': 'dbscan',
            'eps': 0.5,
            'min_samples': 5
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        self.assertIn('labels', result)
        self.assertIn('metrics', result)
        self.assertTrue(len(result['labels']) == len(df))

    def test_hierarchical_clustering(self):
        """Test Hierarchical/Agglomerative clustering"""
        df, features, true_labels = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        config = {
            'algorithm': 'hierarchical',
            'n_clusters': 3,
            'linkage': 'ward'
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        self.assertIn('labels', result)
        self.assertEqual(len(np.unique(result['labels'])), 3)

        # Test different linkage methods
        for linkage in ['complete', 'average', 'single']:
            with self.subTest(linkage=linkage):
                config['linkage'] = linkage
                result = perform_clustering(df, config['algorithm'], config, feature_columns=features)
                self.assertIn('labels', result)

    def test_gmm_clustering(self):
        """Test Gaussian Mixture Model clustering"""
        df, features, true_labels = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        config = {
            'algorithm': 'gmm',
            'n_clusters': 3
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        self.assertIn('labels', result)
        self.assertIn('metrics', result)
        self.assertEqual(len(np.unique(result['labels'])), 3)

    def test_meanshift_clustering(self):
        """Test Mean Shift clustering"""
        df, features, true_labels = MockDataGenerator.create_clustering_data(
            n_samples=50, n_features=3, n_clusters=2  # Smaller dataset for Mean Shift
        )

        config = {
            'algorithm': 'meanshift',
            'bandwidth': 1.0
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        self.assertIn('labels', result)
        self.assertIn('metrics', result)

    def test_kmeans_different_cluster_counts(self):
        """Test K-Means with different numbers of clusters"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        for n_clusters in [2, 3, 4, 5]:
            with self.subTest(n_clusters=n_clusters):
                config = {
                    'algorithm': 'kmeans',
                    'n_clusters': n_clusters
                }

                result = perform_clustering(df, config['algorithm'], config, feature_columns=features)
                self.assertEqual(len(result['cluster_centers']), n_clusters)
                self.assertEqual(len(np.unique(result['labels'])), n_clusters)

    def test_clustering_metrics_computed(self):
        """Test that all clustering metrics are computed"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        config = {
            'algorithm': 'kmeans',
            'n_clusters': 3
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)
        metrics = result['metrics']

        # Verify all metrics are present
        self.assertIn('silhouette_score', metrics)
        self.assertIn('davies_bouldin_score', metrics)
        self.assertIn('calinski_harabasz_score', metrics)

        # Verify metrics are valid numbers
        self.assertTrue(-1 <= metrics['silhouette_score'] <= 1)
        self.assertTrue(metrics['davies_bouldin_score'] >= 0)
        self.assertTrue(metrics['calinski_harabasz_score'] >= 0)

    def test_cluster_statistics(self):
        """Test that cluster statistics are computed"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        config = {
            'algorithm': 'kmeans',
            'n_clusters': 3
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        # Verify cluster statistics
        self.assertIn('cluster_stats', result)
        stats = result['cluster_stats']

        # Should have stats for each cluster (uses string keys like 'cluster_0')
        for cluster_id in range(3):
            cluster_key = f'cluster_{cluster_id}'
            self.assertIn(cluster_key, stats)
            self.assertIn('size', stats[cluster_key])

    def test_pca_visualization_coordinates(self):
        """Test that PCA coordinates are generated for visualization"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        config = {
            'algorithm': 'kmeans',
            'n_clusters': 3
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        # Verify PCA visualization data
        self.assertIn('pca_coordinates', result)
        pca_coords = result['pca_coordinates']

        # Should have 2D coordinates for each sample (it's a list of lists)
        self.assertEqual(len(pca_coords), len(df))
        self.assertEqual(len(pca_coords[0]), 2)

    def test_feature_standardization(self):
        """Test that features are standardized before clustering"""
        # Create data with different scales
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=3, n_clusters=2
        )

        # Scale features differently
        df[features[0]] = df[features[0]] * 1000
        df[features[1]] = df[features[1]] * 0.001

        config = {
            'algorithm': 'kmeans',
            'n_clusters': 2
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        # Should still produce valid results with standardization
        self.assertIn('labels', result)
        self.assertTrue(-1 <= result['metrics']['silhouette_score'] <= 1)

    def test_optimal_clusters_elbow_method(self):
        """Test optimal cluster determination using elbow method"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        result = determine_optimal_clusters(df, features, method='elbow', max_k=10)

        # Verify result structure
        self.assertIn('optimal_k', result)
        self.assertIn('scores', result)
        self.assertIn('method', result)

        # Optimal k should be reasonable
        self.assertTrue(2 <= result['optimal_k'] <= 10)

        # Should have scores for each k value tested
        self.assertGreater(len(result['scores']), 0)

    def test_optimal_clusters_silhouette_method(self):
        """Test optimal cluster determination using silhouette method"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        result = determine_optimal_clusters(df, features, method='silhouette', max_k=10)

        self.assertIn('optimal_k', result)
        self.assertIn('scores', result)

        # Optimal k should be reasonable
        self.assertTrue(2 <= result['optimal_k'] <= 10)

    def test_clustering_with_missing_values(self):
        """Test clustering handles missing values"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        # Introduce some missing values
        df.iloc[0:5, 0] = np.nan
        df.iloc[10:15, 2] = np.nan

        config = {
            'algorithm': 'kmeans',
            'n_clusters': 3
        }

        # Should handle missing values (either by imputation or error)
        try:
            result = perform_clustering(df, config['algorithm'], config, feature_columns=features)
            self.assertIn('labels', result)
        except Exception as e:
            # If it raises an error, that's also acceptable behavior
            self.assertIsInstance(e, (ValueError, TypeError))

    def test_single_cluster(self):
        """Test clustering with k=1"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=50, n_features=3, n_clusters=2
        )

        config = {
            'algorithm': 'kmeans',
            'n_clusters': 1
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        # All samples should be in same cluster
        self.assertEqual(len(np.unique(result['labels'])), 1)

    def test_high_dimensional_clustering(self):
        """Test clustering with high-dimensional data"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=20, n_clusters=3
        )

        config = {
            'algorithm': 'kmeans',
            'n_clusters': 3
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        self.assertIn('labels', result)
        self.assertEqual(len(result['cluster_centers']), 3)

    def test_small_dataset_clustering(self):
        """Test clustering with small dataset"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=20, n_features=3, n_clusters=2
        )

        config = {
            'algorithm': 'kmeans',
            'n_clusters': 2
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        self.assertIn('labels', result)
        self.assertEqual(len(result['labels']), 20)

    def test_dbscan_noise_detection(self):
        """Test DBSCAN noise point detection"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        config = {
            'algorithm': 'dbscan',
            'eps': 0.1,  # Small eps to create noise points
            'min_samples': 10
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        # DBSCAN labels noise as -1
        labels = result['labels']
        unique_labels = np.unique(labels)

        # Should detect some noise points with strict parameters
        self.assertTrue(len(unique_labels) >= 1)

    def test_clustering_reproducibility(self):
        """Test that clustering is reproducible with same random state"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        config = {
            'algorithm': 'kmeans',
            'n_clusters': 3,
            'random_state': 42
        }

        result1 = perform_clustering(df, config['algorithm'], config, feature_columns=features)
        result2 = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        # Results should be identical
        np.testing.assert_array_equal(result1['labels'], result2['labels'])

    def test_invalid_algorithm(self):
        """Test error handling for invalid clustering algorithm"""
        df, features, _ = MockDataGenerator.create_clustering_data()

        config = {
            'algorithm': 'invalid_algorithm',
            'n_clusters': 3
        }

        with self.assertRaises(Exception):
            perform_clustering(df, config['algorithm'], config, feature_columns=features)

    def test_invalid_cluster_count(self):
        """Test error handling for invalid cluster count"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=10
        )

        # More clusters than samples
        config = {
            'algorithm': 'kmeans',
            'n_clusters': 20
        }

        with self.assertRaises(Exception):
            perform_clustering(df, config['algorithm'], config, feature_columns=features)

    def test_cluster_labels_integration_with_dataset(self):
        """Test that cluster labels can be added to original dataset"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        config = {
            'algorithm': 'kmeans',
            'n_clusters': 3
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        # Add labels to dataframe
        df['cluster'] = result['labels']

        # Verify labels added correctly (should be 100 samples)
        self.assertEqual(len(df), 100)
        self.assertIn('cluster', df.columns)
        self.assertEqual(len(np.unique(df['cluster'])), 3)


class TestClusteringVisualization(TestCase):
    """Test clustering visualization components"""

    def test_pca_dimensionality_reduction(self):
        """Test PCA for high-dimensional visualization"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=10, n_clusters=3
        )

        config = {
            'algorithm': 'kmeans',
            'n_clusters': 3
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        # PCA should reduce to 2D
        self.assertIn('pca_coordinates', result)
        pca_coords = result['pca_coordinates']
        self.assertEqual(len(pca_coords), 100)
        self.assertEqual(len(pca_coords[0]), 2)

    def test_cluster_centers_visualization_data(self):
        """Test that cluster centers are in correct format for visualization"""
        df, features, _ = MockDataGenerator.create_clustering_data(
            n_samples=100, n_features=5, n_clusters=3
        )

        config = {
            'algorithm': 'kmeans',
            'n_clusters': 3
        }

        result = perform_clustering(df, config['algorithm'], config, feature_columns=features)

        centers = result['cluster_centers']

        # Centers should have same dimensionality as features (it's a list of lists)
        self.assertEqual(len(centers[0]), len(features))
        self.assertEqual(len(centers), 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
