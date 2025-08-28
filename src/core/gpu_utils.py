import logging
from typing import Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("GPU libraries loaded successfully")
except ImportError as e:
    GPU_AVAILABLE = False
    logger.warning(f"GPU libraries not available: {e}")
    cp = None

class GPUManager:
    """Manages GPU operations with CPU fallback"""

    def __init__(self, gpu_enabled: bool = True, device_id: int = 0,
                 fallback_cpu: bool = True):
        self.gpu_enabled = gpu_enabled and GPU_AVAILABLE
        self.device_id = device_id
        self.fallback_cpu = fallback_cpu

        if self.gpu_enabled:
            try:
                cp.cuda.Device(device_id).use()
                self.device = cp.cuda.Device(device_id)
                logger.info(f"GPU initialized: Device {device_id}")
            except Exception as e:
                logger.error(f"GPU initialization failed: {e}")
                if fallback_cpu:
                    self.gpu_enabled = False
                    logger.info("Falling back to CPU")
                else:
                    raise

    def to_gpu(self, array: np.ndarray) -> Any:
        """Convert numpy array to GPU array"""
        if self.gpu_enabled:
            try:
                return cp.asarray(array)
            except Exception as e:
                logger.warning(f"GPU conversion failed: {e}")
                if self.fallback_cpu:
                    return array
                raise
        return array

    def to_cpu(self, array: Any) -> np.ndarray:
        """Convert GPU array to numpy array"""
        if self.gpu_enabled and hasattr(array, 'get'):
            try:
                return array.get()
            except Exception as e:
                logger.warning(f"CPU conversion failed: {e}")
        return np.asarray(array)

    def cosine_similarity(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute cosine similarity with GPU acceleration"""
        if self.gpu_enabled:
            try:
                X_gpu = self.to_gpu(X)
                Y_gpu = self.to_gpu(Y)

                # Normalize vectors
                X_norm = X_gpu / cp.linalg.norm(X_gpu, axis=1, keepdims=True)
                Y_norm = Y_gpu / cp.linalg.norm(Y_gpu, axis=1, keepdims=True)

                # Compute similarity
                similarity = cp.dot(X_norm, Y_norm.T)
                return self.to_cpu(similarity)
            except Exception as e:
                logger.warning(f"GPU cosine similarity failed: {e}")
                if self.fallback_cpu:
                    from sklearn.metrics.pairwise import cosine_similarity
                    return cosine_similarity(X, Y)
                raise
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(X, Y)

    def kmeans_fit(self, X: np.ndarray, n_clusters: int, **kwargs) -> Any:
        """Fit KMeans with GPU acceleration"""
        if self.gpu_enabled:
            try:
                from cuml.cluster import KMeans
                X_gpu = self.to_gpu(X)
                kmeans = KMeans(n_clusters=n_clusters, **kwargs)
                kmeans.fit(X_gpu)
                return kmeans
            except Exception as e:
                logger.warning(f"GPU KMeans failed: {e}")
                if self.fallback_cpu:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters, **kwargs)
                    kmeans.fit(X)
                    return kmeans
                raise
        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, **kwargs)
            kmeans.fit(X)
            return kmeans

    @property
    def is_gpu_enabled(self) -> bool:
        """Check if GPU is enabled and available"""
        return self.gpu_enabled

    def get_memory_info(self) -> Optional[dict]:
        """Get GPU memory information"""
        if self.gpu_enabled:
            try:
                mempool = cp.get_default_memory_pool()
                return {
                    "used_bytes": mempool.used_bytes(),
                    "total_bytes": mempool.total_bytes(),
                }
            except Exception:
                return None
        return None

# Global GPU manager instance
gpu_manager: Optional[GPUManager] = None

def initialize_gpu(gpu_config: dict) -> GPUManager:
    """Initialize global GPU manager"""
    global gpu_manager
    gpu_manager = GPUManager(
        gpu_enabled=gpu_config.get("enabled", True),
        device_id=gpu_config.get("device_id", 0),
        fallback_cpu=gpu_config.get("fallback_cpu", True)
    )
    return gpu_manager

def get_gpu_manager() -> GPUManager:
    """Get global GPU manager instance"""
    if gpu_manager is None:
        raise RuntimeError("GPU manager not initialized")
    return gpu_manager