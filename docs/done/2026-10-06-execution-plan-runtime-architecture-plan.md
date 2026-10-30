# Stingray: Make ExecutionPlan the Centre of Runtime Architecture

## 1. Objective & Target Architecture

Refactor Stingray so that execution policy is resolved once into an immutable, inspectable `ExecutionPlan`, and all runtime entry points (`ModelContext`, Server `InferenceEngineLoader`, and CLI `RunCommand`) consume that plan rather than rediscovering execution policy.

```
                External Model Data (GGUF, SafeTensors, Ollama)
                                      │
                                      ▼
                                 ModelPackage
                                      │
                                      ▼
                               ModelDescription
                     (semantic identity + immutable facts)
                                      │
                                      ▼
                              ExecutionPlanner
                     ├─ model requirements & planning facts
                     ├─ unified ExecutionRequest
                     ├─ hardware/backend capabilities
                     ├─ candidate evaluation
                     ├─ existing TierPlanner
                     └─ existing ForwardPassSelection
                                      │
                                      ▼
                           immutable ExecutionPlan
                           (strongly typed, no "auto")
                                      │
                                      ▼
                               RuntimeInstance
                 (Engine resource owner; wrapped in Server by ModelRuntime)
                                      │
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                      State       Scheduler    Executor
                                                   │
                                                   ▼
                                         ArchitectureFactory
                                                   │
                                                   ▼
                                               ForwardPass
```

*Note:* `State` and `Scheduler` are future runtime seams. The immediate implementation stops at:
`ModelPackage` $\rightarrow$ `ModelDescription` $\rightarrow$ `ExecutionPlanner` $\rightarrow$ `ExecutionPlan` $\rightarrow$ `RuntimeInstance` $\rightarrow$ existing `InferenceEngine` / `ContinuousBatchingEngine`.

---

## 2. The Four Distinct Conceptual Boundaries

Stingray strictly distinguishes four separate questions:

| Concept | Question Answered | Key Responsibilities | What It Must NOT Contain |
| :--- | :--- | :--- | :--- |
| **`ModelPackage`** | *What files/components belong together?* | Compositional identity (weights, tokenizer, chat template, vision projector, draft model, configuration, licenses); digest-aware identity. | Machine-specific execution choices; model architecture parsing; hardware policy. Location paths are external to identity. |
| **`ModelDescription`** | *What is this model?* | Semantic identity (architecture ID, semantic family, state model, MoE, multimodal capabilities, verified backends, admission status) **plus** intrinsic planning facts (layer counts, head dims, Rope params, quantization info, GDN/MLA flags). | Machine-specific execution choices (no backend, GPU layers, context size, KV dtype). |
| **`ExecutionPlan`** | *How will this model run on this machine for this request?* | Concrete resolved runtime policy (backend, device, placement, GPU/CPU layer split, context size, KV dtype, batching mode, speculation mode, memory budget). Genuinely immutable collections (`ImmutableArray`). | Unresolved `"auto"` options; delegate instances; native handles; model weight copying; mutable collections. |
| **`RuntimeInstance`** | *What actual resources have been allocated to execute that plan?* | Concrete allocated resources (backend instances, forward pass, KV state memory, buffers, inference engine). Wrapped by `ModelRuntime` in the server layer. | Secondary policy selection; fallback logic; re-planning; reading policy from `Model.Parameters`. |

---

## 3. Package Layer Principles & Ollama Compatibility

1. **Logical Component Boundary, Not Physical Archive:**
   `ModelPackage` describes which external files constitute one model (e.g. single GGUF; GGUF + `mmproj`; GGUF + draft model; SafeTensors directory + tokenizer; or Ollama manifest + content-addressed blobs).
2. **Path Does Not Participate in Package Identity:**
   ```csharp
   public sealed record ModelPackageIdentity(
       string? ContentDigest,
       ModelFormat Format,
       ImmutableArray<ModelPackageComponentIdentity> Components);

   public interface IModelPackage
   {
       ModelPackageIdentity Identity { get; }
       string PrimaryPath { get; }
       ...
   }
   ```
   `PrimaryPath` is diagnostic location metadata, not part of identity equality.
   * **Digest Contract:** When content/component digests are available, identical package content at different paths produces the same `ModelPackageIdentity`. When no digest is available, identity is explicitly provisional; path remains location metadata and is never silently promoted to semantic identity.
   * For Ollama packages, the manifest digest is primary.
3. **Do NOT Invent a New Physical Stingray Package Format:**
   Stingray will not create a new `stingray-manifest` or blob layout. Where practical, Stingray understands and consumes the Ollama package representation (`manifest`, `blobs/sha256-...`) essentially as-is.
4. **No Internal Dependency on Ollama:**
   Ollama compatibility is an adapter representation (`IModelPackage`), not the core Stingray model architecture. Stingray can consume Ollama-compatible packages without Ollama being installed or running.
5. **Stingray-Specific Sidecar Metadata (`stingray.json`):**
   Stable Stingray-specific knowledge (architecture interpretation, admission status, verification evidence, verified backends, known limitations) is stored in a separate sidecar keyed to the package digest, without modifying the underlying weights.
   * **Crucial Rule:** `stingray.json` is strictly advisory/cached evidence. It **never grants admission and never overrides `ArchitectureRegistry` or executable policy**.
   * The sidecar contains versioning (`schema_version`, `model_digest`, `verification_profile_version`) to ensure stale evidence is invalidated across software releases.
6. **No Model Weights Shipped in Stingray:**
   Stingray packages/executables never bundle model weights. External model data remains stored on the user's system.
7. **Package Ownership vs. Reference:**
   Stingray distinguishes imported/reference packages (read-only, no lifecycle ownership) from Stingray-managed packages. Stingray does not assume unreferenced Ollama blobs survive external GC.
8. **No Model Manager Scope Creep:**
   No registry integration, downloads, authentication, or automatic updating in this phase. The scope is strictly *reading/understanding packages* to produce `ModelDescription`.

---

## 4. Fundamental Runtime Invariants

1. **The Planner Decides. The Plan Records. The Runtime Obeys:**
   Once an `ExecutionPlan` has been produced, runtime construction must never make another execution-policy decision.
   * `RuntimeInstance` **may**: validate model identity against the plan, verify required hardware, allocate memory, construct the forward pass via `ArchitectureDescriptor.ConstructForwardPass`, and instantiate the engine.
   * `RuntimeInstance` **must NOT**: choose another backend, choose GPU layer counts, invoke `TierPlanner`, invoke `ForwardPassSelection`, silently fall back to CPU, modify context size or KV dtype, or reinterpret `"auto"`.
   * `RuntimeInstance` **must NOT read policy from `Model.Parameters`**: It uses `Model` solely for tensors, metadata, tokenizer, and architecture identity.
2. **Explicit Failure Over Silent Fallback:**
   If the plan cannot be executed on the current machine (e.g. Vulkan device missing or insufficient VRAM), `RuntimeInstance.Create(plan)` throws an explicit `PlanNotExecutableException`. Re-planning is an explicit, separate operation via `ExecutionPlanner.Plan(...)`.
3. **`ExecutionRequest` Captures All Execution-Affecting Inputs:**
   Every input capable of changing backend selection, layer placement, forward-pass kind, state layout, KV configuration, batching support, speculative execution, or execution resources must be represented in `ExecutionRequest` and resolved into `ExecutionPlan`.
   * Includes: backend, GPU layer pin, context size, KV dtype, TurboQuant enable/mode/head dim, FlashAttention, Rope frequency base/scale overrides, execution thread counts, batch mode/max batch size, speculation mode, SnapKV budget, modality execution mode.
   * No execution-affecting values may remain hidden in `Model.Parameters`, `ContextParams`, environment variables, or CLI-only state after planning.
4. **`LoadFromPlan()` Has No Configuration Escape Hatch:**
   `InferenceEngineLoader.LoadFromPlan(plan)` executes the plan directly via `RuntimeInstance.Create(plan, model)`. It does not accept an unrestricted `baseOptions` that could override planned settings. `Load(options)` converges onto the same pipeline by translating `options` into `ExecutionRequest`, running `ExecutionPlanner`, and calling `RuntimeInstance.Create(plan)`.
5. **Candidate Evaluation Belongs in the Planner, Not Runtime:**
   `ExecutionPlanner` may evaluate candidate backends/splits and reject them (e.g. evaluating Vulkan $\rightarrow$ finding an unsupported architecture $\rightarrow$ evaluating CPU $\rightarrow$ selecting CPU). The final plan records one authoritative choice. Runtime never substitutes backends.
6. **Model-Aware Backend Auto-Selection:**
   Backend selection does not merely check "is GPU available". It cross-references model requirements, `ArchitectureDescriptor` restrictions, backend capabilities, and established policy.
7. **`ModelContext` Owns the Plan for Its Runtime:**
   `ModelContext` owns the `ExecutionPlan` for the runtime it represents. Any change to execution-affecting configuration (such as `maxBatchSize` for continuous batching) requires a new plan and runtime, and never mutates or reconfigures an existing plan.
8. **Clean Integration with Server-Side `ModelRuntime`:**
   `RuntimeInstance` integrates with the existing multi-model runtime architecture without duplicating responsibilities:
   ```
   ModelRuntimeManager
       │
       ▼
   ModelRuntime
       │
       ├── Model / ModelPackage
       ├── ExecutionPlan
       └── RuntimeInstance
              ├── backend resources
              ├── forward pass
              └── inference engine
   ```
   * `RuntimeInstance` lives in the Engine/runtime layer and owns the concrete resources required by one `ExecutionPlan`.
   * `ModelRuntime` remains in `OpenTail.Stingray.Server` and owns residency/lifetime around that runtime.
   * `ModelRuntimeManager` continues to own acquisition, eviction, resource admission, and disposal.
   * Do not move `ModelRuntime`/`ModelRuntimeManager` into Engine merely to satisfy the conceptual diagram.
9. **Eliminate Frontend-Specific Execution Policy:**
   Eliminate `ForwardPassFrontend.Cli` vs `ForwardPassFrontend.Server` divergence. All frontends map to `ExecutionRequest` $\rightarrow$ `ExecutionPlanner` $\rightarrow$ identical execution policy.
10. **Genuine Collection Immutability:**
    All collections in `ExecutionPlan` and its sub-plans are `ImmutableArray<T>` or defensively copied. Mutating planner temporary lists after plan creation cannot alter the plan.

---

## 5. Work Breakdown Structure (All 10 Chunks Completed)

The implementation across all 10 chunks was completed and verified on 2026-10-07. Detailed deliverable descriptions, acceptance checklist, implementation summary, and closure enforcement contracts have been split verbatim into [../done/2026-10-06-execution-plan-runtime-architecture-completed-chunks.md](../done/2026-10-06-execution-plan-runtime-architecture-completed-chunks.md).

- [x] **Chunk 1: `ModelPackage` Abstraction & Sidecar Metadata** — Established logical component boundary and package representations (`LooseGgufModelPackage`, `SafeTensorsModelPackage`, `OllamaModelPackage`) with advisory `StingraySidecarMetadata`.
- [x] **Chunk 2: `ModelDescription` with Intrinsic Planning Facts** — Extracted immutable planning facts snapshot in a single pass without disk re-reading.
- [x] **Chunk 3: `ExecutionPlan` Schema v2 & Structural Validation** — Refactored schema v2 with strongly typed sub-plans, immutable collections, and NativeAOT source-generated JSON.
- [x] **Chunk 4: Authoritative `ExecutionPlanner`** — Centralized runtime policy orchestration in `ExecutionPlanner.Plan(...)`, eliminating frontend-specific divergence.
- [x] **Chunk 5: Plan-Driven `ArchitectureLoadContext`** — Connected architecture factories directly to the resolved plan.
- [x] **Chunk 6: `RuntimeInstance` Resource Boundary (Integrated with Server `ModelRuntime`)** — Deterministic runtime resource allocator and engine construction without silent fallbacks.
- [x] **Chunk 7: Migrate `ModelContext`** — Plan-driven engine construction via `RuntimeInstance.Create(plan, _model)`.
- [x] **Chunk 8: Migrate Server (`InferenceEngineLoader`)** — `LoadFromPlan()` directly instantiates `RuntimeInstance` without re-evaluating policy.
- [x] **Chunk 9: Migrate CLI (`RunCommand` and `StaticPlanCommand`)** — CLI arguments map to `ExecutionRequest` and execute via `RuntimeInstance`.
- [x] **Chunk 10: Bypass Elimination, Contract Tests & Parity Verification** — Dead policy eliminated; cross-frontend parity verified in `FrontendPlanParityTests.cs`.

Full chunk deliverables, acceptance criteria checklist, and closure enforcement verification are archived in [docs/done/2026-10-06-execution-plan-runtime-architecture-completed-chunks.md](../done/2026-10-06-execution-plan-runtime-architecture-completed-chunks.md).
