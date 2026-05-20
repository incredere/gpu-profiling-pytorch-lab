# Level 4 — Kernel Roofline Analysis (GPT-2 on T4)

This lab moves beyond *where* GPU time goes (Level 3) to **whether each operation is using the GPU efficiently**. We use the **Roofline model** — the standard mental framework that GPU performance engineers use to identify optimization opportunities.

## What this lab teaches

Every GPU kernel is bottlenecked by one of two things:

1. **Compute** — the math units (CUDA cores, Tensor Cores) are saturated
2. **Memory bandwidth** — the math units are starved waiting for data from HBM

The roofline model lets you tell which, by plotting two numbers per kernel:
- **Arithmetic intensity** = FLOPs ÷ bytes accessed (FLOPs/byte)
- **Achieved performance** = FLOPs ÷ measured time (FLOPS)

Where each kernel sits versus the GPU's hardware "roof" decides what optimization will help.

## T4 hardware roof

| Metric | Value |
|---|---:|
| Peak FP32 compute | 8.1 TFLOPS |
| Peak FP16 (Tensor Cores) | 65 TFLOPS |
| HBM memory bandwidth | 320 GB/s |
| Ridge point (FP32) | 25.3 FLOPs/byte |
| Ridge point (FP16 Tensor Cores) | 203 FLOPs/byte |

Below the ridge → memory-bound. Above the ridge → compute-bound.

## Results — GPT-2 small kernels on T4

Computed from the model architecture (B=8, S=128, D=512, 6 layers) and the kernel times measured in Level 3.

| Kernel | Arithmetic intensity | Achieved (FP32) | % of FP32 peak | Verdict |
|---|---:|---:|---:|---|
| Matmul (FFN) | **146.3** FLOPs/byte | 1.63 TFLOPS | 20.1% | **Compute-bound** (well above ridge) |
| Attention (QK^T) | 16.0 FLOPs/byte | 0.40 TFLOPS | 5.0% | **Memory-bound** (just below ridge) |
| LayerNorm | 1.0 FLOPs/byte | 0.11 TFLOPS | 1.4% | **Severely memory-bound** |
| Elementwise (residuals) | 0.08 FLOPs/byte | 0.003 TFLOPS | 0.03% | **Catastrophically memory-bound** |

## Takeaways

**Matmuls (FFN) — compute-bound at 20% of FP32 peak.** Above the ridge point by a large margin, so they're using the GPU correctly — they just aren't using the *most powerful* units. Switching to FP16/BF16 engages Tensor Cores, raising the theoretical roof from 8.1 TFLOPS to 65 TFLOPS.

**Attention (QK^T) — barely memory-bound at this sequence length.** AI of 16 sits just below the FP32 ridge of 25. At larger sequence lengths (S ≥ 2048), attention's FLOPs grow faster than its memory footprint and it shifts toward compute-bound. This is why FlashAttention's tiling is critical at long contexts — it keeps intermediate matrices in SRAM and raises the effective AI.

**LayerNorm — severely memory-bound at 1.4% of peak.** AI = 1 FLOP/byte. The kernel isn't slow because the code is bad; it's slow because every byte loaded only generates one FLOP of work. *Speeding up the math wouldn't help.* The fix is **kernel fusion** — combining LayerNorm with the next op so data stays in registers instead of round-tripping through HBM.

**Elementwise — catastrophically memory-bound at 0.03% of peak.** Two reads + one write per FLOP. Same fix as LayerNorm: fusion. PyTorch 2's `torch.compile` does this automatically via the inductor backend.

## Summary

> On T4 with this GPT-2 config, matmuls are compute-bound and would benefit most from Tensor Cores (mixed precision). Attention sits near the ridge — its bottleneck depends on sequence length. LayerNorm and elementwise ops are memory-bound and benefit from kernel fusion, not faster math. As compute scales (Volta → Hopper → Blackwell), the memory ridge moves up, so more ops become memory-bound — which is why every new GPU generation also ships with more memory bandwidth.

## How to run

This lab is **pure analysis** — no GPU needed. The notebook reuses the kernel times from Level 3.

1. Open `kernel_roofline_analysis.ipynb` in Jupyter or Colab
2. Run all cells
3. The roofline plot is saved as `roofline_plot.png`

## Files

- `kernel_roofline_analysis.ipynb` — the analysis notebook (with full derivation and roofline plot)
- `roofline_plot.png` — generated roofline visualization
- `README.md` — this document
