## GPU vs TPU Matmul Benchmark

### GPU (T4) Results
Device: CudaDevice(id=0)

Size         Min(s)       Avg(s)
------------------------------------
1024x1024     0.0010       0.0018
2048x2048     0.0035       0.0037
4096x4096     0.0225       0.0226


### TPU (v5e-1) Results
Device: TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)

Size         Min(s)       Avg(s)
------------------------------------
1024x1024     0.0001       0.0002
2048x2048     0.0002       0.0003
4096x4096     0.0009       0.0009


### Observations

1. Which is faster at each size?

- 1024x1024 → TPU is ~10× faster
- 2048x2048 → TPU is ~10–12× faster
- 4096x4096 → TPU is ~20–25× faster

👉 TPU is faster at ALL sizes in this benchmark.


2. At what size does TPU pull ahead?

- TPU is already faster even at small size (1024)
- Gap increases significantly as size grows
- Largest gain observed at 4096x4096


3. Why does TPU outperform GPU?

Key reasons:

- TPU uses systolic array architecture optimized for matrix multiplication
- Higher compute throughput for dense linear algebra
- Better utilization at larger batch sizes
- XLA compiler optimizes execution (fusion + memory layout)

System-level explanation:

- Small sizes → overhead matters more (GPU closer)
- Large sizes → compute dominates → TPU wins heavily

Memory vs Compute Insight:

- GPU becomes partially memory-bound at larger sizes
- TPU keeps high compute utilization due to specialized hardware
- TPU benefits from optimized data flow in systolic arrays

Conclusion:

TPUs significantly outperform GPUs for large, dense matrix operations due to specialized hardware design and compiler-driven optimizations. This makes TPUs ideal for large-scale training workloads with static shapes.
Implication for ML Systems design: always profile at target matrix size. Small-scale benchmarks underestimate TPU advantage by 2-3x.
