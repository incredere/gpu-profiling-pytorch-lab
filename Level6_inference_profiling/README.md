# Level 6 — Inference Profiling: TinyLlama on T4

This lab moves from **training** profiling (Levels 1–5) to **inference** profiling.
We profile TinyLlama-1.1B on a Tesla T4, separating the two phases every
autoregressive LLM goes through and measuring where time and memory actually go.

## What this lab teaches

LLM inference has fundamentally different performance characteristics from training:

1. **Prefill** — the model processes the entire prompt in parallel. This is a
   large batched matmul, similar to a training forward pass. It's **compute-bound**.

2. **Decode** — the model generates tokens one at a time, reading all model weights
   for each single output token. This is **memory-bandwidth-bound** at batch=1
   because arithmetic intensity drops to ~1 FLOP/byte — far below the GPU's ridge point.

Understanding this split is the foundation for every inference optimization:
KV caching, continuous batching, speculative decoding, and quantization all
target different parts of this picture.

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA Tesla T4 (Turing, 16 GB HBM2, 320 GB/s) |
| Model | TinyLlama-1.1B-Chat-v1.0 (22 layers, 32 heads, d=2048, FP16) |
| Framework | PyTorch + `torch.profiler` with CUDA activity |
| Environment | Kaggle (T4) or Google Colab |

## Experiments

### Experiment 1 — Prefill vs decode latency

Manually separate prefill and decode to measure each phase independently,
avoiding `model.generate()` framework overhead.

**Metrics:**
- **TTFT** (Time to First Token) — prefill latency
- **TBT** (Time Between Tokens) — per-token decode latency

### Experiment 2 — Prefill scaling vs prompt length

Measure prefill latency at prompt lengths from 16 to 512 tokens. Expect
near-linear scaling at short lengths, sub-linear once GPU compute saturates.

### Experiment 3 — Decode latency vs KV cache length

Each decode step reads the entire KV cache via the attention kernel. As the
cache grows, decode slows. We measure this from 32 to 1024 cached tokens.

### Experiment 4 — KV cache memory scaling

Measure actual GPU memory consumed by the KV cache at different sequence lengths.
Compare to the theoretical formula:

```
KV bytes per token = 2 (K+V) × num_layers × num_kv_heads × head_dim × 2 (FP16)
```

### Experiment 5 — End-to-end profiling with `torch.profiler`

Profile `model.generate()` to capture a Kineto trace covering both phases.
Categorize GPU kernels (matmul, attention, normalization, elementwise) and
export the trace for Perfetto visualization.

### Experiment 6 — Throughput vs generation length

Measure end-to-end throughput at generation lengths from 16 to 256 tokens.
Since prefill is amortized over more tokens, effective throughput should
converge toward the decode-limited rate.

### Experiment 7 — Decode roofline analysis

Compute the arithmetic intensity of batch=1 decode and compare to the T4's
FP16 ridge point (~203 FLOPs/byte). Confirms decode is deeply memory-bound
and calculates actual memory bandwidth utilization.

## Results

| Metric | Value |
|---|---:|
| Prefill (TTFT), 14-token prompt | ~35 ms |
| Decode (TBT), single step | ~33.5 ms |
| Decode throughput (batch=1) | ~29 tok/s |
| KV cache per token (theoretical) | 22 KB |
| Decode arithmetic intensity | 1.0 FLOPs/byte |
| T4 FP16 ridge point | 203 FLOPs/byte |
| Memory bandwidth utilization | ~20% |

### Prefill scaling

| Prompt length | Latency |
|---:|---:|
| 16 tokens | ~37 ms |
| 64 tokens | ~38 ms |
| 128 tokens | ~40 ms |
| 256 tokens | ~55 ms |
| 512 tokens | ~120 ms |

Prefill is sub-linear up to ~128 tokens (GPU not fully utilized), then
scales steeply as compute saturates.

### Decode TBT vs KV cache length

| KV cache | TBT |
|---:|---:|
| 32 tokens | ~33.5 ms |
| 256 tokens | ~35.2 ms |
| 1024 tokens | ~33.5 ms |

TBT stays in a narrow ~33–35 ms band at these cache sizes. The attention kernel's
share of decode is small relative to weight loading at 1.1B parameters — KV cache
read cost only dominates at much longer contexts.

## Takeaways

**Prefill is compute-bound, decode is memory-bound.** This is the single most
important fact about LLM inference performance. Prefill processes all prompt
tokens in parallel through large matrix multiplications that saturate CUDA cores.
Decode loads the entire model weight matrix for each single output token — throughput
is gated by HBM bandwidth, not compute.

**KV cache memory scales linearly with sequence length.** Each token adds a fixed
amount of memory. For a model with GQA (grouped query attention), `num_kv_heads`
can be much smaller than `num_attention_heads`, reducing cache size proportionally.
TinyLlama uses GQA with 4 KV heads (vs 32 attention heads), so its cache is
relatively compact at ~22 KB/token. At 2048 tokens the cache uses ~45 MB.

**Decode TBT increases with context length** because the attention kernel reads
a larger KV cache. At short sequences this is negligible; at long contexts it
becomes the dominant cost.

**Batch=1 decode wastes the GPU.** Arithmetic intensity ~1 FLOP/byte is 200x
below the ridge. This is why production serving systems use continuous batching —
serving N requests simultaneously raises AI by N×, moving toward the GPU's
compute roof.

## Files

- `inference_profiling.ipynb` — runnable Kaggle/Colab notebook (T4 runtime)
- `tinyllama_inference.pt.trace.json` — Kineto trace (generated by notebook), open in [perfetto.ui](https://ui.perfetto.dev)
- `inference_profiling_plots.png` — summary plots (generated by notebook)
- `README.md` — this document
