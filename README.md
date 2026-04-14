# 🚀 ML Systems Google Prep
### 3-Month Sprint → Google ML Systems Engineer (TPU/GPU Focus)

[![Progress](https://img.shields.io/badge/Progress-Day%206%2F90-blue)](https://github.com/sankarbaseone/ml-systems-google-prep-)
[![Hardware](https://img.shields.io/badge/Hardware-TPU%20v5e--1%20%7C%20T4%20GPU-orange)](https://github.com/sankarbaseone/ml-systems-google-prep-)
[![Framework](https://img.shields.io/badge/Framework-JAX%20%7C%20XLA%20%7C%20PyTorch-green)](https://github.com/sankarbaseone/ml-systems-google-prep-)

---

## 🎯 Mission

Transition into a **Google ML Systems Engineer** role specializing in AI HPC Infrastructure, TPU/GPU optimization, and distributed ML systems — in 3–6 months.

**Background:** 40-year-old engineer with HPC, GPU clusters, distributed systems, and LLM fine-tuning experience.

---

## 🏗️ Target Skills

| Skill | Status |
|-------|--------|
| JAX + XLA fundamentals | ✅ Done |
| JIT compilation & tracing | ✅ Done |
| vmap / pmap parallelism | ✅ Done |
| GPU vs TPU benchmarking | ✅ Done |
| Performance engineering (MFU) | ✅ Done |
| MFU optimization | ✅ Done |
| Distributed ML systems | 🔲 Upcoming |
| Large-scale training system design | 🔲 Upcoming |

---

## 📅 Daily Progress Log

### ✅ Day 1 — JAX on TPU: First Contact
**Device:** v5e-1 TPU (Colab)

| Experiment | Result |
|------------|--------|
| JIT Eager | 0.563s |
| JIT Warm-up | 0.585s |
| JIT Cached | 0.0011s (**512× faster**) |
| vmap output shape | (4,) ✅ |

**Key Insight:** JIT cached execution is 512× faster than eager. XLA compiles Python functions into a single TPU kernel — zero interpreter overhead.

📓 Notebook: [`day1_jax_tpu.ipynb`](./notebooks/day1_jax_tpu.ipynb)
📝 Notes: [`day1_notes.md`](./notes/day1_notes.md)

---

### ✅ Day 2 — JIT Internals + vmap/pmap Mental Model
**Device:** T4 GPU (Colab, TPU quota exhausted)

| Experiment | Result |
|------------|--------|
| JIT same shape (cached) | 0.0006s |
| JIT new shape (recompile) | 0.0409s |
| bad_relu (Python if) | ❌ TracerBoolConversionError |
| good_relu (jnp.where) | ✅ [0. 2. 0. 4.] |
| Loop (64 vectors) | 0.6237s |
| vmap (64 vectors) | 0.2774s (**2.2× faster**) |

**Key Insights:**
- JIT recompiles on shape change → always use **static shapes** in production
- Python control flow breaks tracing → use `jnp.where` not `if`
- vmap eliminates Python loop overhead → XLA vectorizes at compiler level

📓 Notebook: [`day2_jax_tpu.ipynb`](./notebooks/day2_jax_tpu.ipynb)
📝 Notes: [`day2_notes.md`](./notes/day2_notes.md)

---

### ✅ Day 3 — GPU vs TPU Head-to-Head Benchmark
**Devices:** T4 GPU vs v5e-1 TPU (identical JAX code)

| Matrix Size | T4 GPU | v5e-1 TPU | Speedup |
|-------------|--------|-----------|---------|
| 1024×1024 | 0.0010s | 0.0001s | **10×** |
| 2048×2048 | 0.0035s | 0.0002s | **17×** |
| 4096×4096 | 0.0225s | 0.0009s | **25×** |

**Key Insights:**
- TPU advantage **grows with matrix size** — systolic array stays fully utilized
- GPU becomes memory-bound at large sizes
- TPU wins structurally for transformer training (all matmuls)
- Small-scale benchmarks underestimate TPU advantage by 2–3×

📓 Notebook: [`day3_gpu_vs_tpu.ipynb`](./notebooks/day3_gpu_vs_tpu.ipynb)
📝 Notes: [`day3_notes.md`](./notes/day3_notes.md)

---

### ✅ Day 4 — Training Loop Benchmark
**Device:** T4 GPU (Kaggle)

| Metric | Result |
|--------|--------|
| Min step time | 0.0028s |
| Avg step time | 0.0031s |
| Throughput | **83,411 samples/sec** |
| Final loss | 0.005057 ✓ converging |

**Key Insight:** JIT-compiled training step with proper warmup. Loss converging correctly. GPU baseline established for flagship project comparison.

📓 Notebook: [`day4_training_benchmark.ipynb`](./notebooks/day4_training_benchmark.ipynb)
📝 Notes: [`day4_notes.md`](./notes/day4_notes.md)

---

### ✅ Day 5 — Memory Profiling + Bottleneck Analysis
**Device:** T4 GPU (Kaggle)

**Batch Size Scaling (hidden=2048):**

| Batch Size | Throughput |
|------------|-----------|
| 64 | 29,697 samples/sec |
| 256 | 64,423 samples/sec |
| 1024 | 107,530 samples/sec |

**Hidden Size vs Compute Efficiency (batch=256):**

| Hidden Size | Params | TFLOPS | Status |
|-------------|--------|--------|--------|
| 512 | 787K | 0.55 | compute-bound |
| 1024 | 2.1M | 0.60 | **peak efficiency** |
| 2048 | 6.3M | 0.49 | memory-bound |
| 4096 | 21M | 0.43 | memory-bound |

**🚨 MFU = 7%** (T4 theoretical peak = 8.1 TFLOPS)

**Key Insights:**
- Peak efficiency at hidden=1024, drops beyond due to cache overflow
- Memory bandwidth bottleneck beyond hidden=1024
- 93% of GPU sitting idle — this is the optimization target
- Production transformers target 30–60% MFU
- Closing 7% → 30% MFU = **4× effective speedup without new hardware**

📓 Notebook: [`day5_memory_profiling.ipynb`](./notebooks/day5_memory_profiling.ipynb)
📝 Notes: [`day5_notes.md`](./notes/day5_notes.md)

---

### ✅ Day 6 — MFU Optimization
**Device:** T4 GPU (Kaggle)

**MFU Across Batch Sizes and Model Sizes:**

| Hidden | Batch | Time(s) | TFLOPS | MFU% |
|--------|-------|---------|--------|------|
| 1024 | 256 | 0.0017 | 1.86 | 22.96% |
| 1024 | 1024 | 0.0038 | 3.35 | **41.33%** ← peak |
| 2048 | 256 | 0.0030 | 2.17 | 26.73% |
| 2048 | 1024 | 0.0089 | 2.91 | 35.91% |
| 4096 | 256 | 0.0095 | 1.35 | 16.72% |
| 4096 | 1024 | 0.0288 | 1.79 | 22.06% |

**Mixed Precision Results (hidden=2048, batch=1024):**

| Precision | Throughput | Speedup |
|-----------|-----------|---------|
| float32 | 111,306 samples/sec | baseline |
| float16 | **415,695 samples/sec** | **3.7×** |

**🚀 MFU Journey: 7% → 41.33% (6× improvement)**

**Key Insights:**
- Larger batch size = GPU compute units stay fed = higher MFU
- float16 unlocks T4 Tensor Cores: 65 TFLOPS vs 8.1 TFLOPS float32
- float16 is 3.7× faster — no hardware change needed
- We're now in production range: PaLM achieved ~46% MFU
- **The hardware was always capable. We just weren't using it.**

📓 Notebook: [`day6_mfu_optimization.ipynb`](./notebooks/day6_mfu_optimization.ipynb)
📝 Notes: [`day6_notes.md`](./notes/day6_notes.md)

---

## 🔲 Upcoming

| Day | Topic | Status |
|-----|-------|--------|
| Day 1 | JAX + TPU setup, JIT fundamentals | ✅ Done |
| Day 2 | JIT internals, vmap/pmap | ✅ Done |
| Day 3 | GPU vs TPU benchmark | ✅ Done |
| Day 4 | Training loop benchmark | ✅ Done |
| Day 5 | Memory profiling + MFU | ✅ Done |
| Day 6 | MFU optimization | ✅ Done |
| Day 7 | Flagship project launch | 🟡 Upcoming |
| Week 2 | Distributed training + pmap | 🔲 Upcoming |
| Week 3 | XLA compiler deep dive | 🔲 Upcoming |
| Week 4 | System design for large-scale training | 🔲 Upcoming |

---

## 🛠️ Stack

```
JAX + XLA          — Primary ML framework
Flax + Optax       — Neural network + optimizer
PyTorch            — GPU baseline comparison
Google Colab       — TPU v5e-1
Kaggle Notebooks   — T4 GPU (2x)
GCP Compute Engine — Cloud GPU/TPU (upcoming)
Python 3.12        — Runtime
GitHub             — Daily progress tracking
```

---

## 📊 Flagship Project

### "GPU vs TPU Training Optimization"
> End-to-end benchmark of a transformer training step on T4 GPU vs v5e-1 TPU.
> Measuring: throughput, MFU, memory bandwidth utilization, bottlenecks.

**Current Results:**
- GPU MFU baseline: **7%** → optimized: **41.33%**
- GPU Throughput: **83,411** → float16: **415,695 samples/sec**
- TPU matmul speedup: **25× at 4096×4096**

**Status:** 🟡 Optimization complete — Flagship project launching Day 7

---

## 📈 LinkedIn Progress

| Day | Post Topic | Impressions |
|-----|-----------|-------------|
| Day 1 | JAX + TPU setup | — |
| Day 2 | JIT recompilation traps | — |
| Day 3 | GPU vs TPU: 25× speedup | 1,467+ |
| Day 4 | Training loop: 83K samples/sec | — |
| Day 5 | MFU = 7%: Found the real bottleneck | — |
| Day 6 | MFU 7% → 41%: 6× without new hardware | — |

---

## 🔗 Follow Along

- 💼 [LinkedIn](https://www.linkedin.com/in/sankar-panneer-selvam-54820565/) — Daily posts with benchmark results
- 🐙 [GitHub](https://github.com/sankarbaseone/ml-systems-google-prep-) — Daily notebooks + notes

---

## ⚡ Why This Repo Exists

Google ML Systems interviews require you to have **real numbers, real benchmarks, and real system intuition** — not textbook knowledge. Every notebook here is a proof of work.

> *"Don't tell me you understand TPUs. Show me your benchmark."*

---

*Updated daily. Follow the journey.*
