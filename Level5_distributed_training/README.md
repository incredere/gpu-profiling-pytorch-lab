# Level 5 — Distributed Training Profiling (DDP on 2×T4)

This lab profiles a real DDP (DistributedDataParallel) training run and measures the cost of GPU-to-GPU gradient synchronization. It answers the question: **when you add a second GPU, how much of the speedup do you actually get?**

## What this lab teaches

DDP is the standard way to scale training across GPUs. Under the hood:

1. **Forward pass:** each GPU computes independently on its own data shard — no communication
2. **Backward pass:** gradients are synchronized via NCCL AllReduce — this is the communication cost
3. **Optimizer step:** each GPU updates its local copy independently — no communication

The key insight: communication can **overlap** with backward compute. PyTorch DDP buckets gradients (default 25 MB per bucket) and starts AllReduce for earlier layers while later layers are still computing backward. Whether this overlap is effective determines scaling efficiency.

## Experiment setup

- **Model:** GPT-2 small (same as Level 3/4 — 6L, 8H, 512d, ~19M params)
- **Hardware:** 2x NVIDIA T4 on Kaggle (PCIe connected)
- **Backend:** NCCL (GPU-to-GPU communication)
- **Profiler:** PyTorch Profiler with CUDA activity tracing
- **Batch size:** 8 per GPU (16 total with DDP)
- **Profiled steps:** 10 active steps after warmup

## Results

| Metric | Single GPU | DDP (2xT4) |
|--------|-----------|------------|
| Step time (CPU-side) | 55.26 ms | 81.25 ms |
| GPU kernel time per step | 67.93 ms | 111.77 ms |
| Compute per step | 67.93 ms | 62.53 ms |
| NCCL communication per step | — | 49.24 ms |
| NCCL as % of GPU time | — | **44.1%** |
| Compute-comm overlap | — | **47.2%** |
| Scaling efficiency | 100% | **68.0%** |
| Effective throughput | 1.0x | **1.36x** (ideal: 2.0x) |

### GPU time breakdown

**Single GPU** (679.31 ms total, 4,391 kernels across 10 steps):

| Category | Time (ms) | % of GPU time |
|----------|-------:|----------:|
| Compute (matmul) | 318.60 | 46.9% |
| Optimizer (Adam) | 160.79 | 23.7% |
| Elementwise | 135.38 | 19.9% |
| Compute (attention) | 42.43 | 6.2% |
| Other | 22.12 | 3.3% |

**DDP Rank 0** (1,117.73 ms total, 5,068 kernels across 10 steps):

| Category | Time (ms) | % of GPU time |
|----------|-------:|----------:|
| Communication (NCCL) | 492.41 | **44.1%** |
| Compute (matmul) | 259.09 | 23.2% |
| Elementwise | 157.87 | 14.1% |
| Optimizer (Adam) | 149.01 | 13.3% |
| Compute (attention) | 36.74 | 3.3% |
| Other | 22.61 | 2.0% |

## Key findings

**1. NCCL communication consumes 44% of GPU time.** For a 19M-param model, 76 MB of gradient data must be AllReduced every step (~5.4 NCCL calls per step, matching the 25 MB default bucket size). On PCIe-connected T4s without NVLink, this is the dominant cost.

**2. Gradient bucketing achieves 47% compute-communication overlap.** Without overlap, DDP would take 104.5 ms/step (single GPU time + full NCCL time). With DDP's bucketed AllReduce overlapping backward compute, the actual step time is 81.25 ms — overlap saved 23.25 ms per step. Partial but not sufficient to hide the communication cost entirely.

**3. Scaling efficiency is 68% — 2 GPUs give 1.36x throughput, not 2x.** This is the honest reality of distributing a small model across consumer GPUs: the compute-to-communication ratio is unfavorable. Each step does ~62 ms of compute but needs ~49 ms of communication — the ratio is only 1.3:1. For efficient scaling you want this ratio above 10:1.

**4. At larger model sizes, the picture changes dramatically.** For a 7B-param model with large batch sizes, compute time grows as O(params x tokens) while AllReduce grows as O(params). The compute-to-communication ratio improves to 50:1 or higher, pushing scaling efficiency above 90%. This is why distributed training only makes economic sense at scale — and why the infrastructure work to support it matters.

## What to look for in the Perfetto traces

Open the single-GPU and DDP traces side-by-side in [Perfetto UI](https://ui.perfetto.dev):

- **NCCL AllReduce kernels appear as a separate CUDA stream** during the backward pass
- **Overlap is visible** where NCCL kernels run concurrently with backward sgemm kernels on different streams
- **Idle gaps** between backward end and optimizer start reveal communication that couldn't be hidden behind compute
- **Compare kernel counts:** DDP has 5,068 kernels vs single GPU's 4,391 — the difference (~677) is almost entirely NCCL calls and DDP synchronization overhead

## How to run

**On Kaggle (recommended — free 2xT4):**
1. Create a new Kaggle notebook
2. **Settings -> Accelerator -> GPU T4 x2**
3. Upload `distributed_training_profiling.ipynb`
4. Run all cells (~5 min)
5. Download traces from the Output tab

**On any 2-GPU machine:**
1. `pip install torch transformers`
2. Open notebook in Jupyter, run all cells
3. Traces are saved to `./log_single_gpu/` and `./log_ddp_rank0/`

## Files

- `distributed_training_profiling.ipynb` — the main notebook (single-GPU baseline + DDP profiling + analysis)
- `README.md` — this document