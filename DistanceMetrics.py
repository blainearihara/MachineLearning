# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 14:09:40 2025

@author: blain
"""

import numpy as np
import matplotlib.pyplot as plt

class DistanceMeasures:
    """
    Implementation of common distance and similarity measures used in machine learning
    """
    
    def __init__(self):
        pass
    
    # =============================================================================
    # MINKOWSKI DISTANCE FAMILY
    # =============================================================================
    
    def euclidean_distance(self, x1, x2):
        """
        Euclidean Distance (L2 norm): sqrt(sum((xi - yi)^2))
        Most common distance measure. Represents straight-line distance.
        Good for: Continuous features, when all dimensions are equally important
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def manhattan_distance(self, x1, x2):
        """
        Manhattan Distance (L1 norm, Taxicab): sum(|xi - yi|)
        Distance if you can only move along grid lines (like NYC streets).
        Good for: High-dimensional data, robust to outliers
        """
        return np.sum(np.abs(x1 - x2))
    
    def minkowski_distance(self, x1, x2, p):
        """
        Minkowski Distance: (sum(|xi - yi|^p))^(1/p)
        General form where p=1 gives Manhattan, p=2 gives Euclidean
        Good for: When you want to control sensitivity to large differences
        """
        return np.sum(np.abs(x1 - x2) ** p) ** (1/p)
    
    def chebyshev_distance(self, x1, x2):
        """
        Chebyshev Distance (L∞ norm): max(|xi - yi|)
        Maximum difference along any dimension.
        Good for: When the worst-case difference matters most
        """
        return np.max(np.abs(x1 - x2))
    
    # =============================================================================
    # ANGULAR/DIRECTIONAL MEASURES
    # =============================================================================
    
    def cosine_similarity(self, x1, x2):
        """
        Cosine Similarity: (x1 · x2) / (||x1|| * ||x2||)
        Measures angle between vectors, independent of magnitude.
        Range: [-1, 1] where 1 = same direction, -1 = opposite, 0 = orthogonal
        Good for: Text analysis, high-dimensional sparse data
        """
        dot_product = np.dot(x1, x2)
        magnitude_x1 = np.sqrt(np.sum(x1 ** 2))
        magnitude_x2 = np.sqrt(np.sum(x2 ** 2))
        
        if magnitude_x1 == 0 or magnitude_x2 == 0:
            return 0
        return dot_product / (magnitude_x1 * magnitude_x2)
    
    def cosine_distance(self, x1, x2):
        """
        Cosine Distance: 1 - cosine_similarity
        Converts cosine similarity to distance measure.
        Range: [0, 2] where 0 = identical, 2 = opposite
        """
        return 1 - self.cosine_similarity(x1, x2)
    
    def angular_distance(self, x1, x2):
        """
        Angular Distance: arccos(cosine_similarity) / π
        Actual angle between vectors normalized to [0, 1]
        Good for: When you need the actual angle measurement
        """
        cos_sim = self.cosine_similarity(x1, x2)
        cos_sim = np.clip(cos_sim, -1, 1)  # Handle numerical errors
        return np.arccos(cos_sim) / np.pi
    
    # =============================================================================
    # PROBABILISTIC/STATISTICAL MEASURES
    # =============================================================================
    
    def correlation_distance(self, x1, x2):
        """
        Correlation Distance: 1 - |correlation_coefficient|
        Based on Pearson correlation coefficient.
        Good for: When linear relationship matters more than absolute values
        """
        # Calculate means
        mean_x1 = np.mean(x1)
        mean_x2 = np.mean(x2)
        
        # Calculate correlation coefficient
        numerator = np.sum((x1 - mean_x1) * (x2 - mean_x2))
        denominator = np.sqrt(np.sum((x1 - mean_x1) ** 2) * np.sum((x2 - mean_x2) ** 2))
        
        if denominator == 0:
            return 1  # Maximum distance if no variation
        
        correlation = numerator / denominator
        return 1 - abs(correlation)
    
    def bhattacharyya_distance(self, x1, x2):
        """
        Bhattacharyya Distance: -ln(sum(sqrt(xi * yi)))
        Measures similarity between probability distributions.
        Good for: Comparing histograms, probability distributions
        Note: Assumes x1, x2 are probability distributions (sum to 1)
        """
        # Ensure non-negative values (for probability distributions)
        x1_norm = np.maximum(x1, 1e-10)
        x2_norm = np.maximum(x2, 1e-10)
        
        # Normalize to ensure they're valid probability distributions
        x1_norm = x1_norm / np.sum(x1_norm)
        x2_norm = x2_norm / np.sum(x2_norm)
        
        bc_coefficient = np.sum(np.sqrt(x1_norm * x2_norm))
        return -np.log(bc_coefficient + 1e-10)  # Add small value to avoid log(0)
    
    def kl_divergence(self, x1, x2):
        """
        Kullback-Leibler Divergence: sum(xi * log(xi / yi))
        Measures how one probability distribution differs from another.
        Note: Not symmetric, not a true distance metric
        Good for: Information theory, comparing distributions
        """
        # Ensure non-negative and normalized
        x1_norm = np.maximum(x1, 1e-10)
        x2_norm = np.maximum(x2, 1e-10)
        x1_norm = x1_norm / np.sum(x1_norm)
        x2_norm = x2_norm / np.sum(x2_norm)
        
        return np.sum(x1_norm * np.log(x1_norm / x2_norm))
    
    def jensen_shannon_distance(self, x1, x2):
        """
        Jensen-Shannon Distance: sqrt(JS_divergence)
        Symmetric version of KL divergence, proper distance metric.
        Good for: Comparing probability distributions symmetrically
        """
        # Normalize to probability distributions
        x1_norm = np.maximum(x1, 1e-10)
        x2_norm = np.maximum(x2, 1e-10)
        x1_norm = x1_norm / np.sum(x1_norm)
        x2_norm = x2_norm / np.sum(x2_norm)
        
        # Calculate M = (P + Q) / 2
        m = (x1_norm + x2_norm) / 2
        
        # Calculate JS divergence
        js_div = 0.5 * self.kl_divergence(x1_norm, m) + 0.5 * self.kl_divergence(x2_norm, m)
        
        return np.sqrt(js_div)
    
    # =============================================================================
    # CATEGORICAL/BINARY MEASURES
    # =============================================================================
    
    def hamming_distance(self, x1, x2):
        """
        Hamming Distance: sum(xi != yi)
        Number of positions where vectors differ.
        Good for: Categorical data, binary strings, DNA sequences
        """
        return np.sum(x1 != x2)
    
    def jaccard_similarity(self, x1, x2):
        """
        Jaccard Similarity: |intersection| / |union|
        For binary data: |A ∩ B| / |A ∪ B|
        Good for: Set similarity, binary features, text analysis
        """
        # Convert to binary (assuming non-zero means present)
        x1_binary = (x1 != 0).astype(int)
        x2_binary = (x2 != 0).astype(int)
        
        intersection = np.sum(x1_binary & x2_binary)
        union = np.sum(x1_binary | x2_binary)
        
        if union == 0:
            return 1  # Both sets empty = identical
        return intersection / union
    
    def jaccard_distance(self, x1, x2):
        """
        Jaccard Distance: 1 - jaccard_similarity
        """
        return 1 - self.jaccard_similarity(x1, x2)
    
    def dice_coefficient(self, x1, x2):
        """
        Dice Coefficient (Sørensen-Dice): 2 * |intersection| / (|A| + |B|)
        Similar to Jaccard but gives more weight to intersection.
        Good for: Image segmentation, binary classification evaluation
        """
        x1_binary = (x1 != 0).astype(int)
        x2_binary = (x2 != 0).astype(int)
        
        intersection = np.sum(x1_binary & x2_binary)
        total = np.sum(x1_binary) + np.sum(x2_binary)
        
        if total == 0:
            return 1  # Both empty
        return 2 * intersection / total
    
    # =============================================================================
    # SPECIALIZED MEASURES
    # =============================================================================
    
    def mahalanobis_distance(self, x1, x2, cov_matrix):
        """
        Mahalanobis Distance: sqrt((x1-x2)^T * Σ^(-1) * (x1-x2))
        Accounts for covariance between features.
        Good for: When features have different scales/correlations
        """
        diff = x1 - x2
        inv_cov = np.linalg.inv(cov_matrix)
        return np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
    
    def canberra_distance(self, x1, x2):
        """
        Canberra Distance: sum(|xi - yi| / (|xi| + |yi|))
        Weighted version of Manhattan distance.
        Good for: When small values should have equal weight as large values
        """
        numerator = np.abs(x1 - x2)
        denominator = np.abs(x1) + np.abs(x2)
        
        # Handle division by zero
        mask = denominator != 0
        distance = 0
        if np.any(mask):
            distance = np.sum(numerator[mask] / denominator[mask])
        
        return distance
    
    def bray_curtis_distance(self, x1, x2):
        """
        Bray-Curtis Distance: sum(|xi - yi|) / sum(xi + yi)
        Normalized version of Manhattan distance.
        Good for: Ecological data, compositional data
        """
        numerator = np.sum(np.abs(x1 - x2))
        denominator = np.sum(x1 + x2)
        
        if denominator == 0:
            return 0 if numerator == 0 else 1
        
        return numerator / denominator

# =============================================================================
# DEMONSTRATION AND COMPARISON
# =============================================================================

def demonstrate_distances():
    """
    Demonstrate different distance measures with various data types
    """
    dm = DistanceMeasures()
    
    # Test data
    print("=" * 60)
    print("DISTANCE MEASURES DEMONSTRATION")
    print("=" * 60)
    
    # 1. Continuous data
    print("\n1. CONTINUOUS DATA:")
    x1 = np.array([1.0, 2.0, 3.0, 4.0])
    x2 = np.array([2.0, 3.0, 4.0, 5.0])
    print(f"Vector 1: {x1}")
    print(f"Vector 2: {x2}")
    
    print(f"Euclidean:     {dm.euclidean_distance(x1, x2):.4f}")
    print(f"Manhattan:     {dm.manhattan_distance(x1, x2):.4f}")
    print(f"Chebyshev:     {dm.chebyshev_distance(x1, x2):.4f}")
    print(f"Cosine Sim:    {dm.cosine_similarity(x1, x2):.4f}")
    print(f"Correlation:   {dm.correlation_distance(x1, x2):.4f}")
    
    # 2. High-dimensional sparse data (like text)
    print("\n2. SPARSE/TEXT DATA:")
    doc1 = np.array([3, 0, 1, 0, 2, 0, 1, 0])  # word counts
    doc2 = np.array([1, 0, 0, 1, 1, 0, 2, 0])
    print(f"Document 1: {doc1}")
    print(f"Document 2: {doc2}")
    
    print(f"Euclidean:     {dm.euclidean_distance(doc1, doc2):.4f}")
    print(f"Cosine Sim:    {dm.cosine_similarity(doc1, doc2):.4f}")
    print(f"Jaccard Sim:   {dm.jaccard_similarity(doc1, doc2):.4f}")
    
    # 3. Binary data
    print("\n3. BINARY DATA:")
    bin1 = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    bin2 = np.array([1, 1, 0, 1, 0, 0, 1, 1])
    print(f"Binary 1: {bin1}")
    print(f"Binary 2: {bin2}")
    
    print(f"Hamming:       {dm.hamming_distance(bin1, bin2):.4f}")
    print(f"Jaccard Sim:   {dm.jaccard_similarity(bin1, bin2):.4f}")
    print(f"Dice Coeff:    {dm.dice_coefficient(bin1, bin2):.4f}")
    
    # 4. Probability distributions
    print("\n4. PROBABILITY DISTRIBUTIONS:")
    prob1 = np.array([0.4, 0.3, 0.2, 0.1])
    prob2 = np.array([0.2, 0.4, 0.3, 0.1])
    print(f"Prob 1: {prob1}")
    print(f"Prob 2: {prob2}")
    
    print(f"Bhattacharyya: {dm.bhattacharyya_distance(prob1, prob2):.4f}")
    print(f"KL Divergence: {dm.kl_divergence(prob1, prob2):.4f}")
    print(f"JS Distance:   {dm.jensen_shannon_distance(prob1, prob2):.4f}")
    
    # 5. Visualization of different distance measures
    print("\n5. DISTANCE BEHAVIOR COMPARISON:")
    
    # Create test points
    center = np.array([0, 0])
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    distances_euclidean = []
    distances_manhattan = []
    distances_cosine = []
    
    test_points = []
    for angle in angles:
        point = np.array([2*np.cos(angle), 2*np.sin(angle)])
        test_points.append(point)
        distances_euclidean.append(dm.euclidean_distance(center, point))
        distances_manhattan.append(dm.manhattan_distance(center, point))
        distances_cosine.append(dm.cosine_distance(center, point))
    
    print("Point          | Euclidean | Manhattan | Cosine")
    print("-" * 50)
    for i, point in enumerate(test_points):
        print(f"({point[0]:5.2f},{point[1]:5.2f}) | {distances_euclidean[i]:8.3f} | "
              f"{distances_manhattan[i]:8.3f} | {distances_cosine[i]:7.3f}")

def compare_distance_sensitivity():
    """
    Compare how different distances react to outliers
    """
    dm = DistanceMeasures()
    
    print("\n" + "=" * 60)
    print("OUTLIER SENSITIVITY COMPARISON")
    print("=" * 60)
    
    # Normal case
    x1 = np.array([1, 2, 3, 4, 5])
    x2 = np.array([1, 2, 3, 4, 6])  # Small difference in last element
    
    print(f"Normal case:")
    print(f"x1 = {x1}")
    print(f"x2 = {x2}")
    print(f"Euclidean: {dm.euclidean_distance(x1, x2):.3f}")
    print(f"Manhattan: {dm.manhattan_distance(x1, x2):.3f}")
    print(f"Chebyshev: {dm.chebyshev_distance(x1, x2):.3f}")
    
    # With outlier
    x2_outlier = np.array([1, 2, 3, 4, 20])  # Large difference in last element
    
    print(f"\nWith outlier:")
    print(f"x1 = {x1}")
    print(f"x2 = {x2_outlier}")
    print(f"Euclidean: {dm.euclidean_distance(x1, x2_outlier):.3f}")
    print(f"Manhattan: {dm.manhattan_distance(x1, x2_outlier):.3f}")
    print(f"Chebyshev: {dm.chebyshev_distance(x1, x2_outlier):.3f}")
    
    print(f"\nObservation:")
    print("- Euclidean: Moderately sensitive to outliers (squared differences)")
    print("- Manhattan: Less sensitive to outliers (absolute differences)")
    print("- Chebyshev: Only cares about the maximum difference")

if __name__ == "__main__":
    demonstrate_distances()
    compare_distance_sensitivity()
    
    print("\n" + "=" * 60)
    print("WHEN TO USE EACH DISTANCE MEASURE")
    print("=" * 60)
    
    recommendations = {
        "Euclidean": "Continuous features, normally distributed data, equal feature importance",
        "Manhattan": "High dimensions, presence of outliers, sparse data",
        "Cosine": "Text analysis, high-dimensional sparse data, direction matters more than magnitude",
        "Correlation": "Time series, when linear relationship matters",
        "Hamming": "Categorical data, binary strings, DNA sequences",
        "Jaccard": "Set similarity, binary features, recommendation systems",
        "Mahalanobis": "Correlated features, different feature scales",
        "Bhattacharyya": "Comparing probability distributions, histograms"
    }
    
    for measure, use_case in recommendations.items():
        print(f"{measure:15}: {use_case}")