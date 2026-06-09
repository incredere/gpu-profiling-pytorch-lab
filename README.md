# ML Accelerator Profiling Labs

Hands-on GPU profiling labs using PyTorch Profiler, mixed precision benchmarking, and Perfetto trace analysis. Built to develop practical GPU performance engineering skills on real training workloads.

**Author:** Vishwas Somashekara Reddy
**Stack:** PyTorch · PyTorch Profiler · Perfetto · Google Colab · Kaggle · NVIDIA CUDA

---

## Results summary

| Lab | Model | Hardware | Metric | Result |
| --- | --- | --- | --- | --- |
| Level 1 | FeedForward (FashionMNIST) | T4 (Colab) | Self CPU time total | 80.160 ms |
| Level 1 | FeedForward (FashionMNIST) | T4 (Colab) | Self CUDA time total | 287.420 µs |
| Level 2 | FeedForward (FashionMNIST) | T4 (Colab) | FP32 elapsed time | 1.743 sec |
| Level 2 | FeedForward (FashionMNIST) | T4 (Colab) | AMP elapsed time | 1.273 sec |
| Level 2 | FeedForward (FashionMNIST) | T4 (Colab) | Wallclock speedup (AMP vs FP32) | **1.37x** |
| Level 2 | FeedForward (FashionMNIST) | T4 (Colab) | CUDA time FP32 → AMP | 17.631 ms → 8.476 ms (**2.08x**) |
| Level 3 | GPT-2 style (6L, d=512, 8h) | T4 (Colab) | Matmul share of GPU time | **53.50%** |
| Level 3 | GPT-2 style (6L, d=512, 8h) | T4 (Colab) | Attention share of GPU time | **6.77%** |
| Level 3 | GPT-2 style (6L, d=512, 8h) | T4 (Colab) | Adam optimizer share of GPU time | 20.12% |
| Level 3 | GPT-2 style (6L, d=512, 8h) | T4 (Colab) | Backward / forward ratio (CPU-side) | **1.99x** |
| Level 4 | GPT-2 style (6L, d=512, 8h) | T4 (Colab) | FFN matmul AI / achieved | 146 FLOPs/byte → 1.63 TFLOPS (**compute-bound**) |
| Level 4 | GPT-2 style (6L, d=512, 8h) | T4 (Colab) | Attention QK^T AI / achieved | 16 FLOPs/byte → 0.40 TFLOPS (memory-bound) |
| Level 4 | GPT-2 style (6L, d=512, 8h) | T4 (Colab) | LayerNorm AI / achieved | 1 FLOP/byte → 0.11 TFLOPS (memory-bound) |
| Level 4 | GPT-2 style (6L, d=512, 8h) | T4 (Colab) | Elementwise AI / achieved | 0.08 FLOP/byte → 0.003 TFLOPS (memory-bound) |
| Level 5 | GPT-2 style (DDP 2x T4) | 2x T4 (Kaggle) | NCCL % of GPU time | **44.1%** |
| Level 5 | GPT-2 style (DDP 2x T4) | 2x T4 (Kaggle) | Compute-comm overlap | 47.2% |
| Level 5 | GPT-2 style (DDP 2x T4) | 2x T4 (Kaggle) | Scaling efficiency | **68.0%** |
| Level 5 | GPT-2 style (DDP 2x T4) | 2x T4 (Kaggle) | Effective throughput | **1.36x** (ideal: 2.0x) |
| Level 6 | TinyLlama-1.1B (inference) | T4 (Kaggle) | Prefill (TTFT), 14-token prompt | ~35 ms |
| Level 6 | TinyLlama-1.1B (inference) | T4 (Kaggle) | Decode (TBT), batch=1 | ~33.5 ms |
| Level 6 | TinyLlama-1.1B (inference) | T4 (Kaggle) | Decode throughput (batch=1) | **~29 tok/s** |
| Level 6 | TinyLlama-1.1B (inference) | T4 (Kaggle) | Decode arithmetic intensity | 1.0 FLOPs/byte (**memory-bound**) |
| Level 6 | TinyLlama-1.1B (inference) | T4 (Kaggle) | Memory bandwidth utilization | ~20% |

> CUDA-only speedup in Level 2 (2.08x) is higher than wallclock speedup (1.37x) because CPU overhead and data loading are constant across both runs. The GPU kernel efficiency gain from AMP is the real story.
>
> In Level 3, matmul dominates GPU time at short sequence lengths — "attention is the bottleneck" is a long-context story. The O(T²) attention term only catches the O(T·d²) linear term at much longer sequences than we used here (seq_len=128).
>
> In Level 6, batch=1 decode achieves only ~20% of T4's memory bandwidth. Arithmetic intensity is 1.0 FLOP/byte — 200x below the FP16 ridge point. This is why production serving systems use continuous batching.

---

## Lab progression

### Level 1 — GPU profiling with PyTorch Profiler

**Goal:** Understand GPU execution timeline and operator-level performance.

- Trains a feedforward network on FashionMNIST (batch size 256, 1 epoch)
- Captures CPU + CUDA execution traces using `torch.profiler`
- Exports `.pt.trace.json` for Perfetto visualization
- Identifies top operators by CUDA time: `aten::linear`, backward pass, optimizer step

**Key finding:** Backward pass consistently takes longer than forward pass. CPU→GPU data transfer latency is minimal relative to compute.

📁 [`Level1_gpu_profiling_pytorch/`](./Level1_gpu_profiling_pytorch)

---

### Level 2 — Mixed precision profiling (FP32 vs AMP)

**Goal:** Quantify the performance impact of mixed precision training.

- Runs identical workload under FP32 baseline and PyTorch AMP (FP16)
- Captures separate trace files: `fp32.pt.trace.json`, `amp.pt.trace.json`
- Compares elapsed time, CUDA kernel time, and memory usage

**Key finding:** AMP cuts CUDA kernel time by 2.08x by running eligible ops in FP16. Wallclock speedup is 1.37x — lower than CUDA speedup because CPU and data loading overhead is unchanged.

📁 [`Level2_mixed_precision_profiling/`](./Level2_mixed_precision_profiling)

---

### Level 3 — Transformer profiling (GPT-2 style on T4)

**Goal:** Identify where GPU time actually goes during one training step of a decoder-only transformer, and back every claim with numbers parsed directly from a Kineto trace.

- 6-layer GPT-2 style model (d_model=512, 8 heads × 64 dim, d_ff=2048, vocab=50257)
- Batch 8, seq_len 128, Tesla T4, PyTorch `torch.profiler` with CUDA activity
- Captures a Kineto trace across 3 active steps (266.65 ms of GPU kernel time)
- Buckets 1,405 GPU kernels by role and cross-checks against CPU-side annotations

**Key findings:**

- **Matmul dominates, attention does not.** cuBLAS sgemm kernels account for 53.50% of GPU time. Memory-efficient attention (CUTLASS FMHA, fwd + bwd) is only 6.77%.
- **Backward runs 1.99x forward, for the right reason.** Each linear layer launches two sgemms in backward (grad w.r.t. activations + grad w.r.t. weights) versus one in forward. The kernel table confirms it: `sgemm_*_tn` + `sgemm_*_nt` (backward) ≈ 2x `sgemm_*_nn` (forward).
- **Adam is not free.** ~20% of GPU time is spent in `multi_tensor_apply` kernels. Fused / foreach optimizers exist for a reason.

📁 [`Level3_transformer_profiling/`](./Level3_transformer_profiling)

---

### Level 4 — Kernel roofline analysis (GPT-2 on T4)

**Goal:** Move beyond *where* GPU time goes (Level 3) to *whether each operation is using the GPU efficiently*. Apply the Roofline model — the standard framework GPU performance engineers use to decide what to optimize.

- For each kernel category, computes FLOPs from the model architecture and bytes from tensor shapes
- Cross-references against Level 3's measured kernel times to compute achieved performance
- Classifies each kernel against the T4's compute roof (8.1 TFLOPS FP32) and memory roof (320 GB/s)
- Produces a roofline plot showing every kernel relative to the hardware ceiling

**Key findings:**

- **FFN matmul: AI = 146 FLOPs/byte, achieved 1.63 TFLOPS (20% of FP32 peak).** Compute-bound — would benefit most from Tensor Cores via FP16/BF16.
- **Attention QK^T: AI = 16, just below FP32 ridge of 25.** Memory-bound at S=128; shifts compute-bound at longer sequences (this is why FlashAttention matters more at long context).
- **LayerNorm: AI = 1, 1.4% of peak.** Severely memory-bound — the fix is kernel fusion, not faster math.
- **Elementwise: AI = 0.08, 0.03% of peak.** Catastrophically memory-bound — same fix.

**The big idea:** as compute scales faster than bandwidth, more ops become memory-bound. This is why every new GPU generation (Volta → Hopper → Blackwell) ships with more memory bandwidth, not just more FLOPs.

📁 [`Level4_kernel_roofline/`](./Level4_kernel_roofline)

---

### Level 5 — Distributed training profiling (DDP on 2x T4)

**Goal:** Profile a real DDP training run and measure the cost of GPU-to-GPU gradient synchronization. Answer the question: when you add a second GPU, how much of the speedup do you actually get?

- Same GPT-2 model from Level 3, profiled on single GPU (baseline) vs DDP on 2x T4
- Separates compute time from NCCL AllReduce communication time
- Measures gradient bucketing overlap and scaling efficiency
- 76 MB of gradient data AllReduced per step (~5.4 NCCL calls matching 25 MB default bucket size)

**Key findings:**

- **NCCL communication consumes 44.1% of GPU time.** On PCIe-connected T4s without NVLink, gradient sync is the dominant cost for a 19M-param model.
- **Gradient bucketing achieves 47% compute-communication overlap.** Without overlap, DDP would take 104.5 ms/step. With DDP's bucketed AllReduce, actual step time is 81.25 ms — overlap saved 23.25 ms per step.
- **Scaling efficiency is 68% — 2 GPUs give 1.36x throughput, not 2x.** The compute-to-communication ratio is only 1.3:1 at this model size. For efficient scaling you want >10:1.
- **At 7B+ parameters, scaling efficiency climbs above 90%.** Compute grows as O(params × tokens) while AllReduce grows as O(params). This is why distributed training only makes economic sense at scale.

📁 [`Level5_distributed_training/`](./Level5_distributed_training)

---

### Level 6 — Inference profiling (TinyLlama-1.1B on T4)

**Goal:** Profile LLM inference end-to-end, separating the two phases every autoregressive model goes through — prefill (compute-bound) vs decode (memory-bandwidth-bound) — and measuring where time and memory actually go.

- TinyLlama-1.1B-Chat in FP16 on Tesla T4
- Manually separates prefill and decode to measure TTFT and TBT independently
- Measures prefill scaling (16–512 tokens), decode TBT vs KV cache length (32–1024 tokens)
- Profiles `model.generate()` with `torch.profiler` and exports a Kineto trace
- Computes KV cache memory scaling (measured vs theoretical) and decode roofline analysis

**Key findings:**

- **Prefill is compute-bound, decode is memory-bound.** Prefill processes all prompt tokens in parallel through large matmuls. Decode loads the entire 2.2 GB weight matrix for each single output token — throughput is gated by HBM bandwidth, not compute.
- **Decode arithmetic intensity is ~1 FLOP/byte — 200x below the T4 FP16 ridge point (203).** Batch=1 decode achieves only ~20% of T4's memory bandwidth (~29 tok/s measured vs ~145 tok/s theoretical max).
- **KV cache scales linearly at ~22 KB/token.** TinyLlama uses GQA with 4 KV heads (vs 32 attention heads), keeping the cache compact. At 2048 tokens the cache uses ~45 MB.
- **Decode TBT is roughly constant (~33–35 ms) across KV cache sizes up to 1024.** At this model size, weight loading dominates; KV cache read cost only becomes significant at much longer contexts.

📁 [`Level6_inference_profiling/`](./Level6_inference_profiling)

---

## How to run

All labs run on **Google Colab** (free T4 GPU) or **Kaggle** (free 2x T4 for Level 5). No local setup needed.

1. Open the `.ipynb` notebook in the relevant folder
2. Click **Runtime → Change runtime type → T4 GPU** (or GPU T4 x2 on Kaggle for Level 5)
3. Run all cells
4. Download the `.pt.trace.json` output
5. Open [Perfetto UI](https://ui.perfetto.dev) and drag in the trace file to visualize the GPU timeline

---

## Tools used

| Tool | Purpose |
| --- | --- |
| PyTorch Profiler | Capturing CPU + CUDA execution traces |
| Perfetto | Visualizing GPU execution timelines |
| PyTorch AMP | Mixed precision training (FP16/FP32) |
| NCCL | GPU-to-GPU communication for DDP |
| Google Colab | Free T4 GPU environment |
| Kaggle | Free 2x T4 GPU environment |
| nvidia-smi | GPU hardware validation |
