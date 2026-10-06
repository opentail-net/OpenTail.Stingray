# Stingray: Make ExecutionPlan the Centre of Runtime Architecture

## 1. Objective & Target Architecture

Refactor Stingray so that execution policy is resolved once into an immutable, inspectable `ExecutionPlan`, and all runtime entry points (`ModelContext`, Server `InferenceEngineLoader`, and CLI `RunCommand`) consume that plan rather than rediscovering execution policy.

```
                         Model file / package
                                  │
                                  ▼
                         ModelDescription
                                  │
                                  ▼
                         ArchitectureProfile
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  ExecutionPlanner   │
                       │                     │
                       │ model requirements  │
                       │ runtime request     │
                       │ backend capability  │
                       │ memory constraints  │
                       └──────────┬──────────┘
                                  │
                                  ▼
                         immutable ExecutionPlan (No "auto")
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
        CLI execution       Server execution       .NET API
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  ▼
                         RuntimeInstance
                                  │
                     ┌────────────┼────────────┐
                     ▼            ▼            ▼
                  State       Scheduler     Executor
                     │                         │
                     └────────────┬────────────┘
                                  ▼
                         ExecutionEngine
                                  │
                                  ▼
                         Architecture Factory (ConstructForwardPass)
                                  │
                                  ▼
                            ForwardPass
```

The immediate implementation stops at `ExecutionPlan -> RuntimeInstance -> existing InferenceEngine / ContinuousBatchingEngine`.

---

## 2. Fundamental Invariants

1. **No Runtime Policy Re-evaluation:**
   Once an `ExecutionPlan` has been produced, runtime construction must never make another execution-policy decision.
   * Runtime construction **may**:
     * Open/load the model file;
     * Validate that the model matches the plan (architecture, format, identity);
     * Verify that required hardware is still available;
     * Allocate memory and backend context;
     * Construct the selected forward pass via `ArchitectureDescriptor.ConstructForwardPass`;
     * Create the selected inference engine;
     * Initialise caches and state.
   * Runtime construction **must not**:
     * Choose another backend;
     * Choose another GPU layer count;
     * Call `TierPlanner.Plan()`;
     * Call `ForwardPassSelection.SelectPass()`;
     * Silently fall back from Vulkan/CUDA to CPU;
     * Change context size, KV dtype, speculation mode, or batching mode;
     * Reinterpret `"auto"`.
2. **Explicit Failure Over Silent Fallback:**
   If the plan cannot be executed on the current machine (e.g. Vulkan device missing or insufficient VRAM), `RuntimeInstance.Create(plan)` must fail with an explicit `PlanNotExecutableException`. Re-planning is an explicit, separate operation via `ExecutionPlanner.Plan(...)`.
3. **The Core Triad Separation:**
   * **`ModelDescription`**: What the model is (semantic identity, state requirements, modalities).
   * **`ExecutionPlan`**: How this specific model and request will run (resolved, immutable, inspectable contract).
   * **`RuntimeInstance`**: The allocated runtime resources executing that plan.

---

## 3. Work Breakdown Structure (10 Manageable Chunks)

### Chunk 1: Planning Contracts & Structural Foundation
* **Goal:** Introduce clean semantic records in `src/OpenTail.Stingray.Engine/Planning/` without changing existing runtime behaviour.
* **Deliverables:**
  * `ModelDescription(ModelIdentity Identity, ModelSemanticDescription Semantics, ModelCapabilitySummary Capabilities, ModelResourceSummary Resources)`
  * `ModelIdentity(string ArchitectureId, ModelFormat Format, string ModelPath, string? ContentFingerprint)`
  * `ModelSemanticDescription(ForwardPassFamily ForwardPassFamily, ModelStateKind StateKind, bool IsMoE, bool SupportsImageInput, bool SupportsAudioInput)`
  * `ModelStateKind` enum: `None`, `Kv`, `Recurrent`, `Hybrid`
  * `ModelPlan(ModelDescription Model, ArchitecturePlan Architecture, StateRequirements State, ModalityRequirements Modalities, QuantizationRequirements Quantization)`
  * `BackendPlan(string Backend, string? DeviceId, bool FullOffload)`
  * `PlacementPlan(int GpuLayers, int CpuLayers, long GpuWeightBytes, long CpuWeightBytes, long GpuKvBytes, long EstimatedTotalGpuBytes)`
  * `StatePlan(ModelStateKind Kind, string CacheStrategy, string KvDtype, int ContextSize)`
  * `BatchingPlan(string Mode, int MaxBatchSize, bool Enabled)`
  * `SpeculationPlan(string Mode, bool Enabled)`
  * `ModalityPlan(bool ImageInput, bool AudioInput, string? ProjectorPath)`
  * `MemoryPlan(long EstimatedRamBytes, long EstimatedVramBytes, long AvailableRamBytes, long AvailableVramBytes)`
  * `PlanProvenance(int PlannerSchemaVersion, string PlannerVersion)`
  * `ExecutionRequest` & `BackendCapabilities` records.

### Chunk 2: Make `ExecutionPlan` the Complete Resolved Object
* **Goal:** Refactor existing `ExecutionPlan.cs` into schema version 2 with all sub-plans, NativeAOT source-generated JSON, and structural validation.
* **Deliverables:**
  * Refactor `ExecutionPlan` to hold `ModelPlan Model`, `ForwardPassKind ForwardPassKind`, `BackendPlan BackendPlan`, `PlacementPlan Placement`, `StatePlan State`, `BatchingPlan Batching`, `SpeculationPlan Speculation`, `ModalityPlan Modality`, `MemoryPlan Memory`, and `PlanProvenance Provenance`.
  * Retain backward-compatible property accessors (`ModelPath`, `Backend`, `GpuLayers`, `ContextSize`, etc.).
  * Add `ExecutionPlanValidator`:
    * Enforces `Backend != "auto"`.
    * Enforces `GpuLayers >= 0`, `CpuLayers >= 0`, `GpuLayers + CpuLayers == TotalLayers`.
    * Enforces `ContextSize > 0`.
    * Enforces cross-field consistency (e.g. Vulkan passes require Vulkan backend; continuous batching requires batch-capable forward pass and `MaxBatchSize > 1`).
  * Extend `ExecutionPlanJsonContext` with NativeAOT-safe JSON serialization for all sub-plans.

### Chunk 3: Create Authoritative `ExecutionPlanner`
* **Goal:** Move all planning policy into `ExecutionPlanner.cs`.
* **Deliverables:**
  * `ExecutionPlanner.Plan(ModelPlan model, ExecutionRequest request, BackendCapabilities capabilities)`
  * Workflow:
    1. Resolve semantic architecture via `ArchitectureRegistry.Resolve()`.
    2. Normalize concrete backend (resolve `"auto"` to `cuda`, `vulkan`, or `cpu`).
    3. Run `TierPlanner.Plan(...)` once to establish layer placement (`PlacementPlan`).
    4. Call `ForwardPassSelection.SelectPass(backend, gpuLayers)` once to establish `ForwardPassKind`.
    5. Resolve batching, TurboQuant, and memory budgets.
    6. Construct and validate immutable `ExecutionPlan`.
  * Turn `ExecutionPlanBuilder.Build(...)` into a thin compatibility facade calling `ExecutionPlanner`.

### Chunk 4: Make `ArchitectureLoadContext` Plan-Driven
* **Goal:** Connect architecture factories directly to the resolved plan.
* **Deliverables:**
  * Add `public required ExecutionPlan Plan { get; init; }` to `ArchitectureLoadContext`.
  * Derive existing context properties (`Decision`, `Backend`, `GpuLayers`, `Placement`) directly from `Plan`.
  * In `CommonForwardPassFactory`, branch on `ctx.Plan.ForwardPassKind`.
  * Maintain `ArchitectureDescriptor.ConstructForwardPass(context)` as the single construction seam (invoking `ApplyLoadSetup` then `CreateForwardPass`).

### Chunk 5: Introduce `RuntimeInstance`
* **Goal:** Create the runtime resource allocation and engine construction boundary.
* **Deliverables:**
  * `RuntimeInstance.Create(ExecutionPlan plan, Model model)`:
    * Validates model architecture, format, and path against `plan.Model`.
    * Verifies that the planned backend is operational.
    * Allocates required backend instances (`CudaBackend`, `VulkanBackend`, `CpuBackend`).
    * Builds `ArchitectureLoadContext` and calls `descriptor.ConstructForwardPass(loadCtx)`.
    * Instantiates `ContinuousBatchingEngine` if `plan.Batching.Mode == "continuous"`, else `InferenceEngine`.
    * Manages owned disposables.
    * Throws `PlanNotExecutableException` on mismatch or hardware unavailability.

### Chunk 6: Migrate `ModelContext`
* **Goal:** Replace `BuildForwardPass(...)` policy rediscovery in `ModelContext`.
* **Deliverables:**
  * Expose `public ExecutionPlan ExecutionPlan { get; }` on `ModelContext`.
  * Refactor `CreateDefaultEngine()` and `CreateContinuousBatchingEngine()` to request an `ExecutionPlan` from `ExecutionPlanner` and instantiate via `RuntimeInstance.Create(plan, _model)`.
  * Eliminate the 300-line private backend/selection switch in `ModelContext`.

### Chunk 7: Migrate Server (`InferenceEngineLoader`)
* **Goal:** Ensure `LoadFromPlan()` never rediscovers policy.
* **Deliverables:**
  * Update `LoadFromPlan(plan, baseOptions)`:
    * Validate model identity.
    * Call `RuntimeInstance.Create(plan, model)`.
    * **Never call `Load(options)` from `LoadFromPlan()`.**
  * Update `Load(options)`:
    * Map options to `ExecutionRequest`.
    * Call `ExecutionPlanner.Default.Plan(...)`.
    * Call `RuntimeInstance.Create(plan, model)`.

### Chunk 8: Migrate CLI (`RunCommand` and `StaticPlanCommand`)
* **Goal:** Unify all CLI execution paths onto `ExecutionPlanner` and `RuntimeInstance`.
* **Deliverables:**
  * In `RunCommand.cs`: map CLI arguments to `ExecutionRequest`, invoke `ExecutionPlanner`, and execute via `RuntimeInstance`.
  * `--explain` prints the exact `ExecutionPlan` being executed.
  * In `StaticPlanCommand.cs`: invoke `ExecutionPlanner` directly.

### Chunk 9: Remove Bypasses & Dead Policy Switches
* **Goal:** Guarantee that no frontend or runtime code bypasses the planner.
* **Deliverables:**
  * Eliminate direct calls to `ForwardPassSelection.SelectPass()` outside `ExecutionPlanner` and unit tests.
  * Eliminate direct calls to `TierPlanner.Plan()` outside `ExecutionPlanner` and unit tests.
  * Eliminate direct forward-pass constructors outside the architecture factories.

### Chunk 10: Contract Tests, Parity Verification & Documentation
* **Goal:** Validate end-to-end correctness and update architecture documentation.
* **Deliverables:**
  * `ExecutionPlannerTests`: Verify deterministic mapping of requests to concrete plans.
  * `ExecutionPlanRuntimeTests`: Prove runtime executes the exact planned forward pass and placement without re-planning.
  * `PlanNotExecutableTests`: Verify explicit failure when hardware cannot satisfy the plan (no silent fallback).
  * Cross-Frontend Parity Tests: Verify that CLI, Server, and `ModelContext` produce identical plans for identical inputs.
  * Documentation updates in `docs/3-product-and-runtime/`.

---

## 4. Acceptance Criteria Checklist

- [ ] Exactly one authoritative execution planner (`ExecutionPlanner`).
- [ ] `ExecutionPlan` contains concrete resolved choices (no `"auto"`).
- [ ] CLI, Server, and `ModelContext` obtain plans from the same planner.
- [ ] `RuntimeInstance` executes an existing plan without re-planning.
- [ ] `LoadFromPlan()` does not convert the plan back into options and call `Load()`.
- [ ] `ForwardPassSelection` and `TierPlanner` are planner-owned.
- [ ] `ArchitectureDescriptor` factories are construction-only.
- [ ] `ModelContext` contains no backend/placement policy.
- [ ] Server loader contains no secondary planning path.
- [ ] CLI does not instantiate concrete forward-pass classes directly.
- [ ] Runtime cannot silently substitute CPU/Vulkan/CUDA for the planned backend.
- [ ] Plan JSON serializes/deserializes with NativeAOT source generation.
- [ ] Plan validation catches structurally inconsistent plans.
- [ ] Model identity is verified before executing a plan.
- [ ] Parity tests confirm identical plans across all frontends.
