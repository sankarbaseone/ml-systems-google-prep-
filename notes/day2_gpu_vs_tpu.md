# Day 2 — JAX Internals

## Task 1 — JIT
First: 0.1151s  
Second: 0.0006s  
New shape: 0.0788s  

Insight: JIT caches by shape + dtype.

## Task 2 — Control Flow
good_relu: [0. 2. 0. 4.]
Error: truth value ambiguous

Insight: No Python control flow — use jnp.where.

## Task 3 — vmap
Loop: 0.6237s  
vmap: 0.2774s  

Insight: Compiler-level batching → better performance.
