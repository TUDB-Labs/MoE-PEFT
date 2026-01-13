"""Tests for entropy utility functions in moe_peft.common.moe_utils."""

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from moe_peft.common.moe_utils import (  # noqa: E402
    renyi_entropy,
    shannon_entropy,
    tsallis_entropy,
)


class TestTsallisEntropy:
    """Test cases for Tsallis entropy function."""

    def test_tsallis_with_q_equal_1(self):
        """Test Tsallis entropy with q=1.0 (Shannon entropy case)."""
        # Create a uniform distribution
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        entropy = tsallis_entropy(p, q=1.0, normalize=True)
        # For uniform distribution, normalized entropy should be close to 1
        assert torch.isclose(entropy, torch.tensor([1.0]), atol=1e-5)

    def test_tsallis_with_non_uniform_distribution(self):
        """Test Tsallis entropy with non-uniform distribution."""
        # Create a non-uniform distribution
        p = torch.tensor([[0.7, 0.2, 0.05, 0.05]], dtype=torch.float32)
        entropy = tsallis_entropy(p, q=1.5, normalize=True)
        # Entropy should be less than 1 for non-uniform distribution
        assert entropy < 1.0
        assert entropy > 0.0

    def test_tsallis_normalization(self):
        """Test normalization behavior."""
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        normalized = tsallis_entropy(p, q=2.0, normalize=True)
        unnormalized = tsallis_entropy(p, q=2.0, normalize=False)
        # Normalized should be different from unnormalized
        assert not torch.isclose(normalized, unnormalized)

    def test_tsallis_device_and_dtype_matching(self):
        """Test that output matches input device and dtype."""
        # Test with float64
        p_float64 = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float64)
        entropy_float64 = tsallis_entropy(p_float64, q=1.5, normalize=True)
        assert entropy_float64.dtype == torch.float64

        # Test with float32
        p_float32 = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        entropy_float32 = tsallis_entropy(p_float32, q=1.5, normalize=True)
        assert entropy_float32.dtype == torch.float32

        # Test with GPU if available
        if torch.cuda.is_available():
            p_cuda = p_float32.cuda()
            entropy_cuda = tsallis_entropy(p_cuda, q=1.5, normalize=True)
            assert entropy_cuda.device == p_cuda.device

    def test_tsallis_no_input_mutation(self):
        """Test that input tensor is not mutated."""
        p_original = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        p_copy = p_original.clone()
        _ = tsallis_entropy(p_original, q=1.5, normalize=True)
        # Input should not be modified
        assert torch.allclose(p_original, p_copy)

    def test_tsallis_batch_processing(self):
        """Test batch processing with multiple samples."""
        # Create batch of distributions
        p = torch.tensor(
            [[0.25, 0.25, 0.25, 0.25], [0.5, 0.3, 0.1, 0.1]], dtype=torch.float32
        )
        entropy = tsallis_entropy(p, q=1.0, normalize=True)
        # Should return one entropy value per row
        assert entropy.shape[0] == 2
        # First row is uniform, should have higher entropy
        assert entropy[0] > entropy[1]


class TestRenyiEntropy:
    """Test cases for Rényi entropy function."""

    def test_renyi_with_a_equal_1(self):
        """Test Rényi entropy with a=1.0 (Shannon entropy case)."""
        # Create a uniform distribution
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        entropy = renyi_entropy(p, a=1.0, normalize=True)
        # For uniform distribution, normalized entropy should be close to 1
        assert torch.isclose(entropy, torch.tensor([1.0]), atol=1e-5)

    def test_renyi_with_non_uniform_distribution(self):
        """Test Rényi entropy with non-uniform distribution."""
        # Create a non-uniform distribution
        p = torch.tensor([[0.7, 0.2, 0.05, 0.05]], dtype=torch.float32)
        entropy = renyi_entropy(p, a=2.0, normalize=True)
        # Entropy should be less than 1 for non-uniform distribution
        assert entropy < 1.0
        assert entropy > 0.0

    def test_renyi_normalization(self):
        """Test normalization behavior."""
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        normalized = renyi_entropy(p, a=2.0, normalize=True)
        unnormalized = renyi_entropy(p, a=2.0, normalize=False)
        # Normalized should be different from unnormalized
        assert not torch.isclose(normalized, unnormalized)

    def test_renyi_device_and_dtype_matching(self):
        """Test that output matches input device and dtype."""
        # Test with float64
        p_float64 = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float64)
        entropy_float64 = renyi_entropy(p_float64, a=2.0, normalize=True)
        assert entropy_float64.dtype == torch.float64

        # Test with float32
        p_float32 = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        entropy_float32 = renyi_entropy(p_float32, a=2.0, normalize=True)
        assert entropy_float32.dtype == torch.float32

        # Test with GPU if available
        if torch.cuda.is_available():
            p_cuda = p_float32.cuda()
            entropy_cuda = renyi_entropy(p_cuda, a=2.0, normalize=True)
            assert entropy_cuda.device == p_cuda.device

    def test_renyi_no_input_mutation(self):
        """Test that input tensor is not mutated."""
        p_original = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        p_copy = p_original.clone()
        _ = renyi_entropy(p_original, a=2.0, normalize=True)
        # Input should not be modified
        assert torch.allclose(p_original, p_copy)

    def test_renyi_batch_processing(self):
        """Test batch processing with multiple samples."""
        # Create batch of distributions
        p = torch.tensor(
            [[0.25, 0.25, 0.25, 0.25], [0.5, 0.3, 0.1, 0.1]], dtype=torch.float32
        )
        entropy = renyi_entropy(p, a=2.0, normalize=True)
        # Should return one entropy value per row
        assert entropy.shape[0] == 2
        # First row is uniform, should have higher entropy
        assert entropy[0] > entropy[1]

    def test_renyi_return_type(self):
        """Test that function has correct return type annotation."""
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        result = renyi_entropy(p, a=2.0, normalize=True)
        # Should return a torch.Tensor
        assert isinstance(result, torch.Tensor)


class TestShannonEntropy:
    """Test cases for Shannon entropy function."""

    def test_shannon_uniform_distribution(self):
        """Test Shannon entropy with uniform distribution."""
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        entropy = shannon_entropy(p, normalize=True)
        # For uniform distribution, normalized entropy should be close to 1
        assert torch.isclose(entropy, torch.tensor([1.0]), atol=1e-5)

    def test_shannon_equivalence_with_tsallis(self):
        """Test that Shannon entropy equals Tsallis with q=1.0."""
        p = torch.tensor([[0.3, 0.3, 0.2, 0.2]], dtype=torch.float32)
        shannon = shannon_entropy(p, normalize=True)
        tsallis = tsallis_entropy(p, q=1.0, normalize=True)
        # Should be equivalent
        assert torch.allclose(shannon, tsallis, atol=1e-6)

    def test_shannon_equivalence_with_renyi(self):
        """Test that Shannon entropy equals Rényi with a=1.0."""
        p = torch.tensor([[0.3, 0.3, 0.2, 0.2]], dtype=torch.float32)
        shannon = shannon_entropy(p, normalize=True)
        renyi = renyi_entropy(p, a=1.0, normalize=True)
        # Should be equivalent
        assert torch.allclose(shannon, renyi, atol=1e-6)


class TestEntropyEdgeCases:
    """Test edge cases for entropy functions."""

    def test_numerical_stability_with_small_values(self):
        """Test numerical stability with very small probability values."""
        # Distribution with very small values
        p = torch.tensor([[0.0001, 0.0001, 0.4999, 0.4999]], dtype=torch.float32)
        # Should not produce NaN or inf
        tsallis = tsallis_entropy(p, q=1.5, normalize=True)
        renyi = renyi_entropy(p, a=2.0, normalize=True)
        shannon = shannon_entropy(p, normalize=True)

        assert not torch.isnan(tsallis).any()
        assert not torch.isinf(tsallis).any()
        assert not torch.isnan(renyi).any()
        assert not torch.isinf(renyi).any()
        assert not torch.isnan(shannon).any()
        assert not torch.isinf(shannon).any()

    def test_entropy_with_different_dimensions(self):
        """Test entropy functions with different last dimension sizes."""
        # Test with different numbers of experts/classes
        for n in [2, 4, 8, 16]:
            p = torch.ones((1, n), dtype=torch.float32) / n
            entropy_t = tsallis_entropy(p, q=1.0, normalize=True)
            entropy_r = renyi_entropy(p, a=1.0, normalize=True)
            # Uniform distribution should give normalized entropy close to 1
            assert torch.isclose(entropy_t, torch.tensor([1.0]), atol=1e-5)
            assert torch.isclose(entropy_r, torch.tensor([1.0]), atol=1e-5)

    def test_entropy_indices_beyond_common_range(self):
        """Test with entropy indices beyond [0, 2] range."""
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        # Test with q > 2 for Tsallis
        entropy_t = tsallis_entropy(p, q=3.0, normalize=True)
        assert not torch.isnan(entropy_t).any()
        assert not torch.isinf(entropy_t).any()

        # Test with a > 2 for Rényi
        entropy_r = renyi_entropy(p, a=3.0, normalize=True)
        assert not torch.isnan(entropy_r).any()
        assert not torch.isinf(entropy_r).any()
