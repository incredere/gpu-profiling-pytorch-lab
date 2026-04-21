# ML Accelerator Profiling Labs

Hands-on GPU profiling labs using PyTorch Profiler, mixed precision benchmarking,
and Perfetto trace analysis. Built to develop practical GPU performance engineering
skills on real training workloads.

**Author:** Vishwas Somashekara Reddy  
**Stack:** PyTorch · PyTorch Profiler · Perfetto · Google Colab · NVIDIA CUDA

---

## Results summary

| Lab | Model | Hardware | Key metric | Result |
|-----|-------|----------|-----------|--------|
| Level 1 | FeedForward (FashionMNIST) | T4 (Colab) | Self CUDA time total | 287.420 µs |
| Level 1 | FeedForward (FashionMNIST) | T4 (Colab) | Self CPU time total | 80.160 ms |
| Level 2 | FeedForward (FashionMNIST) | T4 (Colab) | FP32 → AMP speedup | _fill in from your run_ |
| Level 2 | FeedForward (FashionMNIST) | T4 (Colab) | Top CUDA kernel | _fill in_ |

---

## Lab progression

### Level 1 — GPU profiling with PyTorch Profiler
**Goal:** Understand GPU execution timeline and operator-level performance.

- Trains a feedforward network on FashionMNIST (batch size 256, 1 epoch)
- Captures CPU + CUDA execution traces using `torch.profiler`
- Exports `.pt.trace.json` for Perfetto visualization
- Identifies top operators by CUDA time: `aten::linear`, backward pass, optimizer step

**Key finding:** Backward pass consistently takes longer than forward pass.
CPU→GPU data transfer latency is minimal relative to compute.

📁 [`Level1_gpu_profiling_pytorch/`](./Level1_gpu_profiling_pytorch/)

---

### Level 2 — Mixed precision profiling (FP32 vs AMP)
**Goal:** Quantify the performance impact of mixed precision training.

- Runs identical workload under FP32 baseline and PyTorch AMP
- Captures separate trace files for each: `fp32.pt.trace.json`, `amp.pt.trace.json`
- Compares step time, GPU memory usage, and kernel execution patterns

**Key finding:** AMP reduces compute time by automatically casting safe ops to FP16.
GPU kernels execute faster; memory footprint decreases.

📁 [`Level2_mixed_precision_profiling/`](./Level2_mixed_precision_profiling/)

---

## How to run

All labs run on **Google Colab** (free T4 GPU). No local setup needed.

1. Open the `.ipynb` notebook in the relevant folder
2. Click **Runtime → Change runtime type → T4 GPU**
3. Run all cells
4. Download the `.pt.trace.json` output
5. Open [Perfetto UI](https://ui.perfetto.dev) and drag in the trace file

---

## What's next

- **Level 3** — Transformer profiling (GPT-2): attention kernels, LayerNorm, KV ops
- **Level 4** — Operator-level breakdown: kernel time % by op type
- **Level 5** — Distributed training: DDP communication vs compute overlap

---

## Tools used

| Tool | Purpose |
|------|---------|
| PyTorch Profiler | Capturing CPU + CUDA execution traces |
| Perfetto | Visualizing GPU execution timelines |
| PyTorch AMP | Mixed precision training (FP16/FP32) |
| Google Colab | Free T4 GPU environment |
| nvidia-smi | GPU hardware validation |
