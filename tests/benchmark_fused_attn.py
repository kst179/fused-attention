import torch
import time

import fused_attention

from naive_implementation import attention

def bench():
    batch_size = 4
    num_heads = 40
    head_dim = 128
    chunk_size = 128
    num_features = num_heads * head_dim
    sequence_len = 2048
    num_batchs = 128
    
    torch.random.manual_seed(179)

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    queries = torch.randn(batch_size, sequence_len, num_features, device="cuda")
    keys = torch.randn(batch_size, sequence_len, num_features, device="cuda")
    values = torch.randn(batch_size, sequence_len, num_features, device="cuda")

    time.sleep(10) # cooldown gpu after tensors' initialization

    start = time.time()
    for i in range(num_batchs):
        _ = attention(queries, keys, values, head_dim)
    torch.cuda.synchronize()
    end = time.time()

    naive_ms = (end - start) / num_batchs * 1000

    time.sleep(20) # cooldown gpu after calculations

    start = time.time()
    for _ in range(num_batchs):
        _ = fused_attention.attention_forward(head_dim, chunk_size, queries, keys, values)
    torch.cuda.synchronize()
    end = time.time()

    fused_ms = (end - start) / num_batchs * 1000

    print(f"Naive {naive_ms:.4f} ms")
    print(f"Fused {fused_ms:.4f} ms")

if __name__ == "__main__":
    bench()
