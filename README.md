# 🚀 ML Systems Google Prep
### 3-Month Sprint → Google ML Systems Engineer (TPU/GPU Focus)

[![Progress](https://img.shields.io/badge/Progress-Day%203%2F90-blue)](https://github.com/sankarbaseone/ml-systems-google-prep-)
[![Hardware](https://img.shields.io/badge/Hardware-TPU%20v5e--1%20%7C%20T4%20GPU-orange)](https://github.com/sankarbaseone/ml-systems-google-prep-)
[![Framework](https://img.shields.io/badge/Framework-JAX%20%7C%20XLA%20%7C%20PyTorch-green)](https://github.com/sankarbaseone/ml-systems-google-prep-)

---

## 🎯 Mission

Transition into a **Google ML Systems Engineer** role specializing in AI HPC Infrastructure, TPU/GPU optimization, and distributed ML systems — in 3–6 months.


---

## 🏗️ Target Skills

| Skill | Status |
|-------|--------|
| JAX + XLA fundamentals | 🟡 In Progress |
| JIT compilation & tracing | ✅ Done |
| vmap / pmap parallelism | ✅ Done |
| GPU vs TPU benchmarking | ✅ Done |
| Distributed ML systems | 🔲 Upcoming |
| Performance engineering (MFU) | 🔲 Upcoming |
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

📓 Notebook: [`day1_jax_tpu.ipynb`](./day1_jax_tpu.ipynb)
📝 Notes: [`day1_notes.md`](./day1_notes.md)

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

📓 Notebook: [`day2_jax_tpu.ipynb`](./day2_jax_tpu.ipynb)
📝 Notes: [`day2_notes.md`](./day2_notes.md)

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

📓 Notebook: [`day3_gpu_vs_tpu.ipynb`](./day3_gpu_vs_tpu.ipynb)
📝 Notes: [`day3_notes.md`](./day3_notes.md)

---

## 🔲 Upcoming

| Day | Topic |
|-----|-------|
| Day 4 | Real training loop benchmark (loss + backward + optimizer) |
| Day 5 | Memory profiling + bottleneck analysis |
| Day 6 | MFU (Model FLOP Utilization) measurement |
| Day 7 | Flagship project: GPU vs TPU Training Optimization |

---

## 🛠️ Stack

```
JAX + XLA          — Primary ML framework
PyTorch            — GPU baseline comparison  
Google Colab       — TPU v5e-1 + T4 GPU
Python 3.12        — Runtime
GitHub             — Daily progress tracking
```

---

## 📊 Flagship Project (Week 1 → )

### "GPU vs TPU Training Optimization"
> End-to-end benchmark of a transformer training step on T4 GPU vs v5e-1 TPU.
> Measuring: throughput, MFU, memory bandwidth utilization, bottlenecks.

**Status:** 🟡 Baseline data collection in progress

---

## 🔗 Follow Along

- 💼 [LinkedIn](https://linkedin.com) — Daily posts with benchmark results
- 🐙 [GitHub](https://github.com/sankarbaseone/ml-systems-google-prep-) — Daily notebooks + notes

---

## ⚡ Why This Repo Exists

Google ML Systems interviews require you to have **real numbers, real benchmarks, and real system intuition** — not textbook knowledge. Every notebook here is a proof of work.

> *"Don't tell me you understand TPUs. Show me your benchmark."*

---

*Updated daily. Follow the journey.*
