# GPU Profiling using PyTorch Profiler and Perfetto

[Open Notebook in Colab](https://colab.research.google.com/github/incredere/gpu-profiling-pytorch-lab/blob/main/gpu_profiling_lab.ipynb)

---

## Overview

This project demonstrates hands-on GPU profiling of a deep learning training workload using **PyTorch Profiler** and **Perfetto**.

The objective of this lab is to understand how machine learning models utilize GPU compute resources and identify performance bottlenecks using trace-based profiling tools.

This project was executed using **Google Colab GPU environment**.

---

## Tools Used

- PyTorch
- PyTorch Profiler
- Google Colab GPU
- NVIDIA CUDA
- Perfetto Trace Viewer
- nvidia-smi

---

## Workflow

1. Train neural network on GPU
2. Capture profiler traces using torch.profiler
3. Generate profiling trace file (.pt.trace.json)
4. Visualize execution timeline using Perfetto UI
5. Analyze compute bottlenecks and GPU utilization

---

## Model Used

Feedforward neural network trained on **FashionMNIST dataset**.

Architecture:

Input → Linear (1024) → ReLU → Linear (512) → ReLU → Output layer (10 classes)

---

## What was analyzed

- Forward pass compute time
- Backward pass compute time
- Optimizer step timing
- CPU to GPU data transfer latency
- GPU kernel execution timeline
- Operator level performance (example: aten::linear)
- GPU utilization patterns

---

## Key Observations

- Matrix multiplication operations (aten::linear) consumed majority of compute time
- Backward pass required more compute time compared to forward pass
- GPU utilization remained consistent during training steps
- CPU to GPU data transfer latency was minimal
- Profiler traces show execution spans for forward, backward, and optimization steps

---

## Screenshots

### GPU Information

![GPU](images/gpu_info.png)

---

### Perfetto Timeline Visualization

![Perfetto](images/perfetto_trace.png)

---

### Profiler Output Summary

![Profiler](images/profiler_output.png)

---

## Repository Structure
