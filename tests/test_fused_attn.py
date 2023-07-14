import os
import unittest

import torch

import fused_attention
from tests.naive_implementation import attention

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class TestFusedAttn(unittest.TestCase):
    def run_simple_test(self, head_dim, chunk_size):
        torch.random.manual_seed(179)

        batch_size = 1
        num_heads = 2
        num_chunks = 2

        num_features = num_heads * head_dim
        sequence_len = num_chunks * chunk_size

        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        queries = torch.randn(batch_size, sequence_len, num_features, device="cuda")
        keys = torch.randn(batch_size, sequence_len, num_features, device="cuda")
        values = torch.randn(batch_size, sequence_len, num_features, device="cuda")

        expected = attention(queries, keys, values, head_dim)
        (actual,) = fused_attention.attention_forward(
            head_dim, chunk_size, queries, keys, values, inference=True
        )

        torch.testing.assert_close(
            actual, expected, rtol=0.01, atol=0.01
        )  # tolerance is smaller than default values because of the half precision

    def test_simple_h16_c16(self):
        self.run_simple_test(16, 16)

    def test_simple_h16_c32(self):
        self.run_simple_test(16, 32)

    def test_simple_h32_c16(self):
        self.run_simple_test(32, 16)

    def test_simple_h32_c32(self):
        self.run_simple_test(32, 32)

    def test_simple_h32_c64(self):
        self.run_simple_test(32, 64)

    def test_simple_h64_c16(self):
        self.run_simple_test(64, 16)

    def test_simple_h64_c32(self):
        self.run_simple_test(64, 32)

    def test_simple_h64_c64(self):
        self.run_simple_test(64, 64)

    def test_simple_h64_c128(self):
        self.run_simple_test(64, 128)

    def test_simple_h128_c32(self):
        self.run_simple_test(128, 32)

    def test_simple_h128_c64(self):
        self.run_simple_test(128, 64)

    def test_simple_h128_c128(self):
        self.run_simple_test(128, 128)
