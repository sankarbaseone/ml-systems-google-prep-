# Day 6 — MFU Optimization

**Device:** T4 GPU (Kaggle 2x T4)
**Date:** April 13, 2026
**Notebook:** day6_mfu_optimization.ipynb

---

## Objective

Take MFU from 7% (Day 5 baseline) to 30%+ using:
1. Accurate FLOP counting
2. Batch size scaling
3. Mixed precision (float16)

---

## Task 1 — Accurate MFU Measurement

### Formula Used
```
MFU = (model_flops * batch_size) / (step_time * peak_tflops * 1e12)
model_flops = 2 * input_dim * hidden * 2 layers * 3 (fwd + bwd + optimizer)
T4 peak = 8.1 TFLOPS (float32)
```

### Results

| Hidden | Batch | Time(s) | TFLOPS | MFU% |
|--------|-------|---------|--------|------|
| 1024 | 256 | 0.0017 | 1.8595 | 22.96% |
| 1024 | 1024 | 0.0038 | 3.3476 | **41.33%** |
| 2048 | 256 | 0.0030 | 2.1651 | 26.73% |
| 2048 | 1024 | 0.0089 | 2.9084 | 35.91% |
| 4096 | 256 | 0.0095 | 1.3546 | 16.72% |
| 4096 | 1024 | 0.0288 | 1.7866 | 22.06% |

### Key Observations

1. **Best MFU: 41.33%** at hidden=1024, batch=1024
2. MFU grows consistently with batch size across all hidden sizes
3. hidden=1024 outperforms hidden=4096 at same batch — cache efficiency wins
4. Day 5 MFU was 7% — today's best is 41.33% — **6x improvement**

### Why Batch Size Drives MFU

- Small batches = GPU compute units idle between ops (waiting for data)
- Large batches = continuous data flow through compute units
- Batch size is "free" performance — no new hardware needed
- Production training always uses largest batch that fits in memory
- This is why Google TPU pods use enormous batch sizes (millions of tokens)

---

## Task 2 — Mixed Precision (float16 vs float32)

### Results

| Precision | Avg Time | Throughput | Speedup |
|-----------|----------|-----------|---------|
| float32 | 0.0092s | 111,306 samples/sec | baseline |
| float16 | 0.0025s | **415,695 samples/sec** | **3.7x faster** |

### Why float16 is 3.7x Faster

T4 GPU has two compute modes:

| Mode | Peak TFLOPS | Used by |
|------|-------------|---------|
| float32 (CUDA cores) | 8.1 TFLOPS | Default JAX ops |
| float16 (Tensor Cores) | 65 TFLOPS | Mixed precision ops |

- Tensor Cores are specialized circuits for float16 matrix multiply
- 65 / 8.1 = **8x theoretical speedup** — we got 3.7x (real-world overhead included)
- Memory savings: float16 uses half the VRAM → larger batch fits → more MFU gain

### Mixed Precision Best Practice

```python
# Forward/backward in float16 (speed)
# Optimizer state in float32 (numerical stability)
class MLPHalf(nn.Module):
    dtype: any = jnp.float16
    def __call__(self, x):
        x = x.astype(self.dtype)
        # ... layers in float16
        return x.astype(jnp.float32)  # cast back for loss
```

---

## Combined Optimization Impact

| Optimization | Metric Before | Metric After | Gain |
|-------------|--------------|--------------|------|
| Batch 256 → 1024 | MFU 7% | MFU 41% | **6x** |
| float32 → float16 | 111k samples/sec | 415k samples/sec | **3.7x** |
| Combined potential | 83k samples/sec (Day 4) | 415k+ samples/sec | **5x+** |

---

## Critical Insight for Google Interviews

**Q: "How would you improve training throughput on a GPU cluster?"**

**A (with data):**
1. Profile MFU first — if below 30%, hardware is underutilized
2. Increase batch size — free performance, no hardware change
3. Enable mixed precision — 3-8x speedup using Tensor Cores
4. Check memory bandwidth vs compute bound — different fixes for each
5. Measure after each change — don't guess, benchmark

**The gap between 7% MFU and 41% MFU is not a hardware problem.**
**It's an optimization problem. That's what ML Systems Engineers solve.**

---

## Production Numbers (for context)

| System | MFU | Reference |
|--------|-----|-----------|
| Our Day 5 baseline | 7% | This repo |
| Our Day 6 optimized | 41% | This repo |
| PaLM (Google) | ~46% | Chowdhery et al. 2022 |
| GPT-3 (OpenAI) | ~21% | Brown et al. 2020 |
| Production target | 30-60% | Industry standard |

We're now in the same ballpark as production systems.

---

## Next Steps (Day 7)

- Launch flagship project: GPU vs TPU Training Optimization
- Apply float16 + large batch to full transformer training step
- Measure MFU on TPU with same optimizations
- Compare GPU optimized vs TPU optimized side by side

---

## Files

- Notebook: `notebooks/day6_mfu_optimization.ipynb`
- Notes: `notes/day6_notes.md`
