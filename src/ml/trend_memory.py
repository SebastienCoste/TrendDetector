import numpy as np
import time
from typing import List, Dict, Any, Optional
from collections import deque
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class DynamicTrendMemory:
    """Dynamic trend memory with clustering and time-based decay"""

    def __init__(self, max_clusters: int = 20, memory_size: int = 10000, 
                 time_decay_hours: int = 24):
        self.max_clusters = max_clusters
        self.memory_size = memory_size
        self.time_decay_hours = time_decay_hours
        
        # Store trend history
        self.trend_history = deque(maxlen=memory_size)
        self.recent_trend_history = deque(maxlen=memory_size)

        # Centroids for each trend type
        self.trend_centroids = {
            'upward': [],
            'downward': [], 
            'neutral': []
        }
        
        logger.info(f"TrendMemory initialized: max_clusters={max_clusters}, memory_size={memory_size}")

    def reset_recent_history(self) -> None:
        self.recent_trend_history = deque(maxlen=self.memory_size)

    def add_trend_sample(self, embedding: np.ndarray, trend: str, timestamp: float) -> None:
        """Add a new trend sample to memory"""
        sample = {
            'embedding': embedding,
            'trend': trend,
            'timestamp': timestamp
        }
        self.trend_history.append(sample)
        self.recent_trend_history.append(sample)
        logger.info(f"Added sample: trend={trend}, timestamp={timestamp}")

    def get_similar_trends(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar trends using cosine similarity"""
        if not self.trend_history:
            return []
            
        similarities = []
        current_time = time.time()
        
        for sample in self.trend_history:
            # Calculate cosine similarity
            dot_product = np.dot(query_embedding, sample['embedding'])
            norms = np.linalg.norm(query_embedding) * np.linalg.norm(sample['embedding'])
            similarity = dot_product / norms if norms > 0 else 0
            
            # Apply time decay
            time_diff_hours = (current_time - sample['timestamp']) / 3600
            time_decay = np.exp(-time_diff_hours / self.time_decay_hours)
            weighted_similarity = similarity * time_decay
            
            similarities.append({
                'trend': sample['trend'],
                'similarity': weighted_similarity,
                'timestamp': sample['timestamp']
            })
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def get_trend_distances(self, query_embedding: np.ndarray) -> Dict[str, float]:
        """Get distances to trend centroids"""
        distances = {}
        
        for trend_type, centroids in self.trend_centroids.items():
            if not centroids:
                distances[trend_type] = float('inf')
                continue
                
            # Find distance to nearest centroid
            min_distance = float('inf')
            for centroid in centroids:
                distance = np.linalg.norm(query_embedding - centroid)
                min_distance = min(min_distance, distance)
            
            distances[trend_type] = min_distance
            
        return distances

    def update_trend_boundaries(self) -> None:
        """Update trend centroids using clustering"""
        if len(self.trend_history) < 10:
            return
            
        logger.info("Updating trend boundaries...")
        
        # Group samples by trend type
        trend_embeddings = {
            'upward': [],
            'downward': [],
            'neutral': []
        }
        
        for sample in self.trend_history:
            trend_embeddings[sample['trend']].append(sample['embedding'])
        
        # Update centroids for each trend type
        for trend_type, embeddings in trend_embeddings.items():
            if len(embeddings) < 2:
                continue
                
            embeddings_array = np.array(embeddings)
            
            # Determine number of clusters
            n_clusters = min(self.max_clusters, len(embeddings) // 2)
            n_clusters = max(1, n_clusters)
            
            try:
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(embeddings_array)
                
                # Store centroids
                self.trend_centroids[trend_type] = kmeans.cluster_centers_.tolist()
                
                logger.debug(f"Updated {trend_type} centroids: {n_clusters} clusters")
                
            except Exception as e:
                logger.warning(f"Failed to update centroids for {trend_type}: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        trend_counts = {'upward': 0, 'downward': 0, 'neutral': 0}
        
        for sample in self.trend_history:
            trend_counts[sample['trend']] += 1
            
        centroid_counts = {
            trend: len(centroids) 
            for trend, centroids in self.trend_centroids.items()
        }
        
        return {
            'total_samples': len(self.trend_history),
            'trend_distribution': trend_counts,
            'centroid_counts': centroid_counts,
            'memory_utilization': len(self.trend_history) / self.memory_size
        }