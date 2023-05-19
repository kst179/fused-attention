# Fused attention

## Description
Experimental project, implementation of forward multihead attention pass in single CUDA kernel, without additional VRAM usage for scores calculation.

## The Idea

Attention mechanism is widely used in contemprorary large language models  (BERT, GPT, LLaMA, etc) as well as in some computer vision networks (like ViT and similar). It is a main part of the transformer architecture, in which almost half of the layers are attention layers. So optimizing one could give an opportunity to make large transformers faster (and reduce heat and CO2 emission). Usualy we calculate multihead attention as follows:

$$Q = XW_q; ~K = XW_k; ~V = XW_v$$

$$S_h = \text{softmax}\left( \frac{Q_hK_h^T}{\sqrt H} \right), ~h = 1\dots R$$

$$Y_h = S_hV_h$$

$$Z = YW_o$$

Here 
* $L$ is sequence length (or set size), $F$ is number of features, $R$ is number of heads in multihead attention, $H$ is number of features in the single head; $H \times R = F$
* $X \in \mathbb{R}^{L \times F}$ &mdash; the input matrix
* $W_q, W_k, W_v, W_o \in \mathbb{R}^{F \times F}$ &mdash; query, keys, values and output weight matrices
* $Q_h, K_h, V_h, Y_h \in \mathbb{R}^{L \times H}$ &mdash; submatrices of $Q, K, V, Y$, so-called heads 
* $S_h \in \mathbb{R}^{L \times L}$ &mdash; attention scores for the single head 
* $Y \in \mathbb{R}^{L \times F}$ &mdash; concatenated heads matrix
* $Z \in \mathbb{R}^{L \times F}$ &mdash; output matrix

Time complexity of this operation is $O(LF^2 + FL^2):$ 4 multiplications by weight matrices (each costs $LF^2$) and $2 \times R$ matrix multplications of complexity $HL^2$ for scores calculation and weighted values aggregation. 

Memory complexity is $O(LF + F^2 + R L^2)$, matrices $X, Y, Z,$ have size $L \times F$, weigths have size $F \times F$ and scores have size $L \times L$ for each of $R$ heads.

Here we can already see the problem: despite the fact the features size $F$ is the fixed parameter which allows us to manage the width of the network and the capability of understanding language, the $L$ part is just the length of the context to consider, and the attention mechanism requires $O(L^2)$ of time/memory, assuming $F$ and $R$ is fixed. So if we analyse some chat message, and a whole book the time/memory would grow quadraticly. For example RNNs does not have such problem, amount of calculations are grown linearly. There are some euristics to achieve linear-time calculated attentions but they are usually less powerful of keeping context. Yet there is a way to calculate vanilla attention with $O(L^2)$ time but only $O(L)$ memory usage. Also given less memory read-write operations, we can achieve faster calculations (by reducing constant under the $O(\bullet)$ notation).

 For the big values of $L$ and many heads[^1] in the attention layer the score matrix is the biggest one in these calculations, and despite the fact that matrix multiplications is the most complex calculation here we still waste too much time for reading and writing theese matrices from and to GPU VRAM.

So the idea is to make fused kernel for the intermediate operation $Y = \text{attention}(Q, K, V),$ without storing the full tensor $S$ in the global memory. It is possible if we suppose that head dim $H$ is fixed and small enough[^2] split the queries/keys matrices into chunks of size $C \times H$, where $C$ is usualy also small. Then we multiply these chunks to calculate scores, and aggregate running softmax. 

Most tricky part here is to calculate partial softmax function without storing all the scores. We can describe the whole $\text{attention}(Q, K, V)$ function as 

$$Y_{ij} = \left\lbrace \text{softmax}(QK^T)V\right\rbrace \_{ij} = \sum_{l = 1}^L \left( \frac{\exp (\hat{S}\_{il})}{\sum_{k = 1\dots L}\exp(\hat{S}\_{ik})}\right) V\_{lj} $$

here $\hat{S} = \frac{QK^T}{\sqrt H}$, the unnormalized scores (before softmax). We can rewrite this formula as follows:

$$Y_{ij} = \frac{\sum_{l = 1\dots L} \exp (\hat{S}\_{il}) V_{lj}}{\sum_{l = 1\dots L}\exp(\hat{S}\_{il})} $$

So we can calculate attention in stream-like manner (without storing any calculations along axis $l$), all we need to do is to aggregate the numerator and denominator of this fracture. But there is a problem with sum of exponentials: we cannot calculate it straightforward because some of the exponents could be quite large and if we store them in the half precision they would just overflow. To solve this problem we can use the idea of log-sum-exp function (which by the way is usualy implemented in classic softmax as well): lets represent a sum of exponentials  $$\sum_{i=1}^N \exp(s_i) = \exp(s_{\max})\sum_{i=1}^N\exp(s_i - s_{\max}); ~s_{\max} = \max_{i=1\dots N}s_i$$
while $s_{\max}$ can be big enough to its exponent would owerflow a half precision float, we can still hold its value without calculating the exponent directly. On the other hand $s_i - s_{\max} \leqslant 0, \forall i = 1\dots N$ , so $\exp(s_i - s_{\max}) \in (0, 1]$, and we can calculate $s_{\text{sum}} = \sum_{i=1\dots N}\exp(s_i - s_{\max})$ directly and store it. So sum of exponentials can be represented as a pair of numbers $\left(s_{\max}; s_{\text{sum}} \right),$ and recovered by calculating 
$$\sum_{i=1}^N \exp(s_i) = \exp(s_{\max}) \cdot s_{\text{sum}}$$

We can also represent a numerator in that manner, if we select $s_{\max} = \max_{i=1\dots N}s_i$ and $s_{\text{sum}} = \sum_{i=1\dots N} \exp(s_i - s_{\max})v_i$ then
$$\sum_{i=1}^N \exp(s_i)v_i = \exp(s_{\max})\sum_{i=1}^N \exp(s_i - s_{\max})v_i = \exp(s_{\max}) \cdot s_{\text{sum}}$$

which is up to indices resembles the numerator in the formula above.

Suppose we have 2 sets of numbers: $a_{1\dots N}$ and $b_{1\dots M}$, and we have precalculated sums-of-exponents of them, represented as pairs $(a_{\max}, a_{\text{sum}}), (b_{\max}, b_{\text{sum}})$, then we can represent the sum:
$\sum_{i = 1\dots N} \exp(a_i) + \sum_{j=1\dots M} \exp(b_j)$ as the pair of numbers $(c_{\max}, c_{\text{sum}})$, where
$$c_{\max} = \max(a_{\max}, b_{max})$$
$$c_{\text{sum}} = \exp(a_{\max} - c_{\max})a_{\text{sum}} + \exp(b_{\max} - c_{\max})b_{\text{sum}}$$

Using this idea we can calculate numericaly stable representations of numerator/denominator of the softmax fracture on the small chunks of data (which can be stored in fast on-chip shared memory) and then aggregate them into the full formula, without storing intermediate results. Also we can avoid calculating $\exp(s_{\max})$ because it cancels out in the final fracture:

$$Y_{ij} = \frac{\cancel{\exp(\max_l (\hat{S}\_{il}))} \sum_{l = 1\dots L} \exp (\hat{S}\_{il} - \max_l (\hat{S}\_{il})) V_{lj}}{\cancel{\exp(\max_l (\hat{S}\_{il}))}\sum_{l = 1\dots L}\exp(\hat{S}\_{il} - \max_l (\hat{S}\_{il}))} $$

## Installation

Impemented function can be used as a torch extention or can be added to CUDA/C++ program directly, by including `include/fused_attn.cuh` file into the project.

Minimal dependencies are [**pytorch**](https://pytorch.org/get-started/locally/) and compatible [**CUDA Toolkit**](https://developer.nvidia.com/cuda-downloads) (for C++ projects only CUDA is needed).

To install it as torch extention use:

```bash
$ python setup.py install --user
```

Then you can use the extention in following way:
```python
import torch        # should import torch first
from fused_attn import attention_forward

head_dim = 128      # head dim should be a power of 2 and between 16 and 128
chunk_size = 128    # chunk size should also be a power of 2 greater than 16 but less than 2*head_dim,
                    # also if head_dim=128, chunk_size=16 is prohibited (due to the implementation)
                    # I believe the best choice is to use chunk_size == head_dim

q, k, v = ...       # Tensors should be of shape (batch_size, sequence_len, feature_size)
                    # sequence_len should be divisible by chunk_size (if not then should be padded with zeroes), feature_size - by head_dim

m = ...             # optional mask of size (batch_size, sequence_len, sequence_len) filled with zeroes if 
                    # query-key pair should not be muted and -inf else 

output = attention_forward(head_dim, chunk_size, q, k, v, m)
```

There also a CMake project which just builds a simple test that everything is working. You can build and run it with
```bash
$ mkdir build 
$ cd build 
$ cmake ..
$ make
$ ./test
```

## Tests & Benchmarks

Following implementation was tested against the pytorch naive impementation, and looks to work correctly (despite the differences in ~1% which should be just half precision loss). You can run tests with the following command:

```bash
$ python -m unittest
```
Also the algorithm was tested in perfomance on the simple benchmark on the **NVIDIA GeForce RTX 3050 Laptop** and **NVIDIA A100-PCIE-40GB** on the random tensors (values are from standard normal distribution) with the following parameters:
```
sequence_len=2048
feature_size=5120
head_dim=128
chunk_size=128
num_heads=40
```
The results of the benchmark are in the table below

| GPU      | Batch size | Algorithm       | Time per batch | Speedup[^3] | Additional memory[^4] |
|----------|------------|-----------------|---------------:|------------:|----------------------:|
| RTX 3050 | 4          | Naive           |        83.4 ms |           - |               1280 Mb |
|          |            | Fused           |        58.6 ms |         42% |                  0 Mb |
| A100     | 4          | Naive           |        14.6 ms |           - |               1280 Mb |
|          |            | Fused           |         6.2 ms |        135% |                  0 Mb |
| A100     | 16         | Naive           |        41.7 ms |           - |               5120 Mb |
|          |            | Fused           |        23.4 ms |         78% |                  0 Mb |

## Further improvements

This project was done j4f, in a several weekends. The idea just came to my mind and I was haunted by an obsessive thought to try to implement it and to measure how much perfomance will be. Now, when my interest has been satisfied for a while, I am not sure, if I will continue this little project but if somebody (includes me) will be interested in future improvemns, here is check list what could also be done:

- [x] implement fused forward layer
- [x] run benchmarks on the different gpu archetictures, check the perfomance gain on the high-end gpu
- [ ] perform full benchmark analysis on different parameters, add some visualization
- [ ] create a wrapper for this layer (add input/output linear layers) in pytorch
- [ ] implement a fused input (linear qkv layers + optional rotational encodings for queries and keys) and fused output (linear output layer + residual connection + normalization) to increase the perfomance of the whole attention block
- [ ] save scores tensor in fused layer for backward pass. It will be slower but allow us to obtain a trainable attention which is yet little bit faster then naive approach.
- [ ] implement fused backward layer (without storing scores), it also would be slower, because we need to recalculate score matrix in the backward pass, but will save us a lot of memory on training (so we could increase the batch size for example).
- [ ] implement the same layers but for quantized inputs/weights to speedup the whole network, using 8bit or 4bit quantization, and fit big LLMs in small GPUs at inference.
- [ ] build a full-sized transformer using this layer, check perfomance gain on real task.
- [ ] research another approaches to make NNs smaller and faster, and capable of running on the teapot, I know I am not the only one who want to talk to my teapot...

[^1]: For example in facebook's LLaMA-7B `L` (context length) is usualy equal to 2048 and `R` (number of heads) is 40 and this is a smallest model in the LLaMA family.

[^2]: Typicaly head dim `H` is one of $64$ or $128$. 

[^3]: Speedup calculated as (time_naive / time_fused - 1) * 100%, and shows how much of perfomance (batches / ms) gained relative to the naive implementation (so the 0% speedup is the same, 100% speedup is twice faster and so on).

[^4]: Memory used in attention function calculation to store score tensor if using half precision, qkv and output tensors are not considered there.
