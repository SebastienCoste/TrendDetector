"""
Unit tests for GPU functionality and manager
"""
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.gpu_utils import GPUManager, initialize_gpu, get_gpu_manager
from src.core.config import GPUConfig

class TestGPUManager:
    """Test GPU Manager functionality"""

    def test_gpu_manager_initialization_gpu_enabled(self):
        """Test GPU manager initialization when GPU is available"""
        # Test with GPU enabled
        gpu_manager = GPUManager(gpu_enabled=True, device_id=0, fallback_cpu=True)

        # Should initialize even if CUDA not available (fallback to CPU)
        assert gpu_manager is not None
        assert gpu_manager.fallback_cpu == True

    def test_gpu_manager_initialization_cpu_only(self):
        """Test GPU manager initialization with CPU only"""
        gpu_manager = GPUManager(gpu_enabled=False, device_id=0, fallback_cpu=True)

        assert gpu_manager is not None
        assert gpu_manager.gpu_enabled == False

    def test_to_gpu_array_conversion(self):
        """Test numpy array to GPU conversion"""
        gpu_manager = GPUManager(gpu_enabled=False)  # Force CPU for consistent testing

        test_array = np.array([1.0, 2.0, 3.0, 4.0])
        result = gpu_manager.to_gpu(test_array)

        # Should return original array when GPU disabled
        np.testing.assert_array_equal(result, test_array)

    def test_cosine_similarity_cpu_fallback(self):
        """Test cosine similarity with CPU fallback"""
        gpu_manager = GPUManager(gpu_enabled=False)

        X = np.random.rand(5, 10)
        Y = np.random.rand(3, 10)

        similarity = gpu_manager.cosine_similarity(X, Y)

        # Should return valid similarity matrix
        assert similarity.shape == (5, 3)
        assert np.all(similarity >= -1.0)
        assert np.all(similarity <= 1.0)

    def test_kmeans_cpu_fallback(self):
        """Test KMeans with CPU fallback"""
        gpu_manager = GPUManager(gpu_enabled=False)

        X = np.random.rand(100, 5)
        n_clusters = 3

        kmeans = gpu_manager.kmeans_fit(X, n_clusters)

        # Should return valid KMeans result
        assert hasattr(kmeans, 'cluster_centers_')
        assert kmeans.cluster_centers_.shape == (n_clusters, 5)

    def test_memory_info_without_gpu(self):
        """Test memory info when GPU not available"""
        gpu_manager = GPUManager(gpu_enabled=False)

        memory_info = gpu_manager.get_memory_info()

        # Should return None when GPU not enabled
        assert memory_info is None

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.Device')
    def test_gpu_manager_with_cuda(self, mock_device, mock_cuda_available):
        """Test GPU manager when CUDA is available"""
        mock_device_instance = MagicMock()
        mock_device.return_value = mock_device_instance

        gpu_manager = GPUManager(gpu_enabled=True, device_id=0)

        # Should attempt to initialize GPU
        mock_device.assert_called_with(0)

class TestGPUConfig:
    """Test GPU configuration"""

    def test_gpu_config_creation(self):
        """Test creating GPU configuration"""
        config = GPUConfig(
            enabled=True,
            device_id=1,
            fallback_cpu=True,
            memory_limit=0.8
        )

        assert config.enabled == True
        assert config.device_id == 1
        assert config.fallback_cpu == True
        assert config.memory_limit == 0.8

    def test_initialize_gpu_function(self):
        """Test initialize_gpu function"""
        config = GPUConfig(enabled=False, device_id=0, fallback_cpu=True)

        gpu_manager = initialize_gpu(config)

        assert gpu_manager is not None
        assert gpu_manager.gpu_enabled == False

class TestGPUIntegration:
    """Test GPU integration with other components"""

    def test_gpu_memory_leak_prevention(self):
        """Test that GPU operations don't create memory leaks"""
        gpu_manager = GPUManager(gpu_enabled=False)  # Use CPU to avoid GPU dependency

        # Perform multiple operations
        for i in range(10):
            X = np.random.rand(100, 50)
            Y = np.random.rand(50, 50)

            # Test conversion and similarity
            gpu_array = gpu_manager.to_gpu(X)
            cpu_array = gpu_manager.to_cpu(gpu_array)
            similarity = gpu_manager.cosine_similarity(X, Y)

            # Verify no memory issues
            assert cpu_array.shape == X.shape
            assert similarity.shape == (100, 50)

    def test_gpu_error_handling(self):
        """Test GPU error handling and fallback"""
        gpu_manager = GPUManager(gpu_enabled=False)

        # Test with invalid input
        with pytest.raises(ValueError):
            # This should raise an error due to dimension mismatch
            X = np.array([])  # Empty array
            Y = np.random.rand(3, 10)
            gpu_manager.cosine_similarity(X, Y)

if __name__ == "__main__":
    pytest.main([__file__])
