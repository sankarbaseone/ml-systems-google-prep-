## Day 5 — Memory Profiling + Bottleneck Analysis

### Batch Size Scaling (hidden=2048)
| Batch | Throughput |
|-------|-----------|
| 64    | 29,697/sec |
| 256   | 64,423/sec |
| 1024  | 107,530/sec |

### Hidden Size vs TFLOPS (batch=256)
| Hidden | Params | TFLOPS |
|--------|--------|--------|
| 512    | 787K   | 0.55   |
| 1024   | 2.1M   | 0.60   |
| 2048   | 6.3M   | 0.49   |
| 4096   | 21M    | 0.43   |

### Key Finding
- Peak efficiency at hidden=1024
- MFU = 7% (T4 theoretical peak = 8.1 TFLOPS)
- Memory-bound beyond hidden=1024
- Gap to close: 7% → 30-60% MFU = ML Systems Engineer's job
