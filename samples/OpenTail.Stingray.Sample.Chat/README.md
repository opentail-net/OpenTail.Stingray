# OpenTail.Stingray.Sample.Chat

> **Audience:** Package Consumer  
> **Task:** Multi-turn interactive streaming chat in C#  
> **API Contract:** `Model.Load`, `CreateContext`, `InteractiveExecutor`, `ChatSession`

---

## 1. What This Sample Demonstrates

This is the canonical consumer sample showing how to use the high-level `OpenTail.Stingray` library API in a .NET application:
- **Model Lifecycle:** Loading GGUF weights with `Model.Load` and creating isolated execution contexts with `CreateContext`.
- **Stateful Multi-Turn Chat:** Managing conversation history automatically via `ChatSession`.
- **Real-Time Streaming:** Consuming tokens asynchronously via `await foreach (var piece in session.ChatAsync(...))`.
- **Path-Free Model Resolution:** Seamless integration with `ModelHome.ResolveModelPath(...)` when models are pre-installed via `stingray setup chat`.
- **Cooperative Cancellation:** Handling `Ctrl+C` cleanly without crashing the process.

---

## 2. Prerequisites & Asset Setup

You can either pass an explicit path to any GGUF model, or use the pre-installed default:

```bash
# Option A: Install the default verified model (Qwen2.5 0.5B Instruct, 469 MB)
stingray setup chat

# Option B: Pull any other model from Hugging Face
stingray pull -r Qwen/Qwen2.5-0.5B-Instruct-GGUF
```

---

## 3. How to Run

### Path-Free Invocation (uses installed default)
```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Chat -c Release
```

### With an Explicit Model Path
```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Chat -c Release -- \
    -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### Piped Input (Single-Turn / Scripting)
```bash
echo "In one sentence, why do developers write unit tests?" | \
    dotnet run --project samples/OpenTail.Stingray.Sample.Chat -c Release
```

### Available CLI Flags
- `-m, --model <path>`: Path to GGUF model file (or set `STINGRAY_MODEL` environment variable).
- `-s, --system <prompt>`: Custom system instructions for the assistant.
- `-c, --ctx-size <tokens>`: Maximum context window size (default: 2048).
- `--temp <float>`: Sampling temperature (default: 0.7).
