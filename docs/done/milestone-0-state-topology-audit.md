# Milestone 0 — state topology audit

**Status:** code-evidence audit plus retained-cache fake-backend seam; real CPU-model equivalence still required  
**Date:** 2026-08-02  
**Scope:** the current OpenTail.Stingray fork, not the SharpInference-derived design

This is the evidence record for the first gate in the session-native inference runtime plan. It
separates capabilities which exist today from capabilities that are merely plausible. A checkmark
in the main plan must only follow an executable seam test, not this static audit.

## Findings

| State/backend | Current owner and concrete type | Append/resume | Rewind | Current-head fork | Export/import | Coverage and position model | Session status |
|---|---|---|---|---|---|---|---|
| CPU dense | `ForwardPass`; `PagedKvCache` | Yes: `PrefillWithCache` / `BatchForwardMulti` accept an explicit cache. | Exact soft `TruncateTo`; pages remain allocated. | Yes, page-aligned COW through `ForkSharedPrefix`; engine exposes it through `CapturePrefix` / `ForkPrefix`. | Not implemented. | Full unless SnapKV compacts. `Length` is physical; `LogicalLength` retains the RoPE position after compaction. | **Reference lane.** |
| CUDA dense | `CudaForwardPass`; internal `CudaSequenceKvCache` | Yes: continuous batching binds one cache per row. | Existing owned-cache rewind; no general public external-cache rewind contract. | No ref-counted device-page fork capability. | Not implemented. | `Length` is physical. `EvictedCount` maps logical position to physical slot after SnapKV eviction. Device tensors require the CUDA owner. | Later conformance lane. |
| Vulkan dense | Vulkan forward paths, not `IBatchedForwardPass` | No independently-owned batch cache surface found. | Architecture-specific only. | Not found. | Not implemented. | Backend/device dependent. | Out of first slice. |
| CPU SnapKV | `PagedKvCache` plus `SnapKvSelector` | Append is supported after compaction. | Only the state-specific soft rewind; no session promise yet. | Explicitly refused for a shared-prefix cache. | Not implemented. | Windowed/compacted: logical and physical lengths diverge. | Capability-gated later. |
| TurboQuant/KVarN | Owned `ForwardPass` compressed/ring state | Not a safe per-sequence batching/session contract. | Partial rewind has compressed-region limits. | No current fork capability. | Codec-specific future work. | Composite compressed and FP32-window state. | Out of reference lane. |
| GDN/recurrent | Hybrid forward passes and snapshot facilities | State-specific. | Snapshot boundaries exist for continuation flows. | Not assumed. | Composite-state future work. | Recurrent state is not ordinary KV. | Separate conformance track. |
| MTP/speculative | Forward pass verification/snapshot paths | Commit and rollback exist internally. | Existing snapshot/rewind paths. | Not assumed. | Composite state required. | Draft/rollback metadata is part of correctness. | Separate conformance track. |
| Vision | Vision pipeline plus decoder state | Not audited as a session unit. | Decoder dependent. | Not assumed. | Media identity plus decoder state required. | Model specific. | Optional milestone only. |

## Code evidence

- `IBatchedForwardPass` makes an `ISequenceKvCache` opaque to the batcher while allowing the
  backend to create, prefill and decode against it. This is the correct backend-neutral boundary.
- `ContinuousBatchingEngine` owns its active and prefilling cache handles privately. It allocates
  at admission and disposes in `DropPrefilling`, `RetireSeq`, immediate-stop activation, fatal
  cleanup, and engine disposal. Therefore it cannot presently return a completed hot state to a
  session owner.
- CPU `PagedKvCache` owns native page pools. `ForkSharedPrefix` is exact for full pages, retains
  the source pool, and uses copy-on-write if a shared page is rewritten. This is a strong building
  block but only for page-aligned CPU dense prefixes.
- CPU `PagedKvCache` reports both physical `Length` and absolute `LogicalLength`; SnapKV sets them
  apart. Session identity and cursor logic must preserve both concepts rather than treating a
  retained cache length as a transcript length.
- CUDA `CudaSequenceKvCache` owns per-layer device tensors and has an `EvictedCount` logical to
  physical mapping. It is neither portable across backend instances nor safe to serialize as raw
  pointers. It must remain behind an explicit capability contract.
- Only `ForwardPass` and `CudaForwardPass` currently implement `IBatchedForwardPass`. No Vulkan
  implementation was found in the current source tree.

## Reference-lane decision

The first executable session slice is limited to CPU dense `PagedKvCache`, F32 KV, greedy
sampling, no SnapKV, no TurboQuant, no recurrent/GDN state, no speculative decoding and no
media. It has one generation writer per session. Exact continuation is the only acceptable result
for this lane.

## Remaining seam spike

- [x] Add a narrowly-scoped lifecycle path that accepts an externally-owned CPU cache.
- [ ] Run one bounded greedy turn through the real `ContinuousBatchingEngine`.
- [x] Detach the cache at retirement without disposing it or stalling the worker.
- [x] Re-admit that exact cache and append a turn.
- [ ] Compare its greedy output to a full replay.
- [x] Cancel during generation and prove the cache returns to the logical turn start.
- [ ] Capture the API, disposal and threading result in ADR-0001 before exposing session storage.

## Consequences for implementation

The next code change must add lifecycle ownership, not another parallel cache implementation. It
must provide exactly-one final disposition for every admitted cache: retained by the session,
released by the engine, or faulted and released. The API must carry the logical cursor separately
from physical cache occupancy, and it must report an explicit refusal for unsupported cache
families rather than attempting a generic export or fork.
