# OpenTail.Stingray — Inference State Architecture Specification

## Overview

This specification formalizes the **4-Tier Inference State Architecture** for `OpenTail.Stingray`. It defines ownership, mutability, and thread-safety contracts across all execution levels.

```text
                     MODEL (IModel)
             Immutable weights, architecture, vocab
                        │
                        ▼
                   RUNTIME (IInferenceRuntime)
             Execution resources, backends, thread pools
                        │
                        ▼
                   SESSION (IInferenceSession)
             Mutable inference state, token history, position
                        │
                        ▼
               LOGICAL KV (IKvSequence)
                        │
                        ▼
               PHYSICAL KV (IKvCache)
                        │
                        ▼
               MEMORY PAGES (KvPageId)
```

---

## 1. Ownership & Mutability Matrix

| State Element | Primary Owner | Mutability | Shared Scope |
| :--- | :--- | :--- | :--- |
| **Model Weights & Architecture** | `IModel` | Immutable | Shared across all sessions |
| **Tokenizer & Vocabulary** | `IModel` / `IForwardPass` | Immutable | Shared across all sessions |
| **Physical Page Pool & Allocator** | `IInferenceRuntime` (`IKvCache`) | Thread-Safe Mutable | Shared pool across runtime |
| **Active Session Registry** | `IInferenceRuntime` (`InMemorySessionManager`) | Thread-Safe Mutable | Runtime instance scope |
| **Token History & Position** | `IInferenceSession` | Single-Writer Mutable | Isolated per session |
| **Logical KV Page Table** | `IInferenceSession` (`IKvSequence`) | Single-Writer Mutable | Shared on zero-copy fork (CoW) |
| **Sampling Configuration** | `SamplingParams` | Immutable Per-Request | Generation request scope |
| **RNG State** | `SamplingParams` / `Random` | Single-Writer Mutable | Generation request scope |
| **Session Snapshots** | `ISessionStore` (`FileSessionStore`) | Serializable DTO | Persistent store |

---

## 2. Tier Responsibilities

### Tier 1: Model (`IModel`)
- **Role**: Shared, read-only tensor weights, neural network architecture, and model parameters.
- **Rule**: Must never store session-specific state or generation progress.

### Tier 2: Runtime (`IInferenceRuntime`)
- **Role**: Top-level environment managing global execution resources:
  - Physical `IKvCache` allocator and page pools.
  - Active `InMemorySessionManager` session registry.
  - Thread pools, device buffers, and backend context.
- **Rule**: Disposing a runtime disposes all active sessions and releases physical memory pools without destroying shared model weights.

### Tier 3: Session (`IInferenceSession`)
- **Role**: Encapsulates single-tenant mutable inference state:
  - Token history (`TokenHistory`), current position (`TokenCount`), and state (`SessionState`).
  - Logical sequence handle (`IKvSequence`).
  - Checkpoints (`SessionCheckpoint`) and snapshots (`InferenceSessionSnapshot`).
- **Rule**: Single-writer thread safety (`SemaphoreSlim(1,1)`). Operations on a session mutate only its isolated state.

### Tier 4: Logical & Physical KV (`IKvSequence` / `IKvCache`)
- **Role**:
  - `IKvSequence`: Logical token-to-page mapping, supporting $O(\text{pages})$ zero-copy page table forking and Copy-on-Write page duplication.
  - `IKvCache`: Thread-safe physical page manager with lock-free Compare-And-Swap (`Interlocked.CompareExchange`) allocation loops.

---

## 3. Concurrency & Thread-Safety Contracts

1. **Model & Vocab Sharing**: Read-only operations against `IModel` / `IForwardPass` are multi-thread safe.
2. **Session Mutation**: `IInferenceSession` enforces single-writer execution via an internal `SemaphoreSlim(1,1)` lock per session. Concurrent queries (`TokenCount`, `State`) read under lightweight state spinlocks.
3. **Copy-on-Write Page Isolation**: When a session is forked (`Fork()`), child and parent sequences share physical page reference counts (`RetainPage`). Modifying an unaligned shared page triggers `PerformCopyOnWrite`, allocating a private page before mutation.

---

## 4. OpenTail Policy vs. Stingray Execution Mechanics Boundary

To maintain clear system architecture, responsibility is partitioned cleanly between **OpenTail** (application policy) and **Stingray** (execution engine):

```text
OpenTail (Application Policy)
  ├── Task / Agent Assignment
  ├── Request Routing & Model Selection
  ├── User Concurrency & Priority Rules
  └── High-level Session Orchestration
        │
        ▼
Stingray (Execution Engine)
  ├── Single-Session & Batched Execution
  ├── Physical KV Cache & Page Pool Allocation
  ├── Continuous Batching & Step Scheduling
  └── Tensor Kernel Execution (CPU/CUDA/Vulkan)
```

Stingray focuses strictly on execution mechanics and does not attempt to embed application-level agent/policy schedulers.

---

## 5. Convergence of Execution Modes

`OpenTail.Stingray` supports two execution models over common model weights, KV allocation, and runtime semantics:

1. **Single-Session Mode (`IInferenceSession` → `IForwardPass`)**: Ideal for direct, low-latency single-tenant workflows, tool calling, and speculative decoding.
2. **Batched Mode (`ContinuousBatchingEngine` → `IBatchedForwardPass`)**: Optimized for high-throughput multi-tenant request batching.

Both execution paths operate against shared `IKvCache` allocators and `IKvSequence` page tables, guaranteeing semantic consistency across execution modes.

---

## 6. Snapshot & Persistence Semantics

- **`InferenceSessionSnapshot`**: Represents a **Token-History / Replay Snapshot** (`Tokens`, `Position`, `ModelId`).
- **Restoration Behavior**: Restoring a session from a snapshot reinstates token history, allocates logical sequence capacity, and immediately pre-fills the model KV cache so the session is inference-ready, avoiding fragile binary KV state serialization.

