# OpenTail.Stingray.Sample.ToolCall

Demonstrates **agentic tool calling** using the OpenTail.Stingray library directly:

1. Tool schemas are passed to the model's Jinja chat template as a `tools` list
2. The model outputs `<tool_call>{"name":"…","arguments":{…}}</tool_call>` blocks
3. The sample parses those blocks, executes the tools locally, and feeds the results back as `role="tool"` messages
4. A second model turn uses the tool results to produce a natural-language final answer

Works with any model whose chat template supports tool calling (Qwen3, Mistral-Nemo, etc.).

## Run

```bash
# CPU (default)
dotnet run --project samples/OpenTail.Stingray.Sample.ToolCall -c Release -- \
    -m models/Qwen3-8B-Q4_K_M.gguf

# CUDA — auto-detect how many layers fit in VRAM
dotnet run --project samples/OpenTail.Stingray.Sample.ToolCall -c Release -- \
    -m models/Qwen3-35B-A3B-Q4_K_M.gguf --backend cuda -g -1

# CUDA — force all layers on GPU (will OOM if model doesn't fit)
dotnet run --project samples/OpenTail.Stingray.Sample.ToolCall -c Release -- \
    -m models/Qwen3-35B-A3B-Q4_K_M.gguf --backend cuda -g 999
```

Custom question:

```bash
dotnet run --project samples/OpenTail.Stingray.Sample.ToolCall -c Release -- \
    -m models/Qwen3-8B-Q4_K_M.gguf \
    -p "What's the weather in Berlin and what is (100 - 37) * 2?"
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-m` / `--model` | `STINGRAY_MODEL` env | Path to GGUF model file |
| `-p` / `--prompt` | built-in demo | User question to ask |
| `--temp` | `0.6` | Sampling temperature |
| `--backend` | `cpu` | Compute backend: `cpu` or `cuda` |
| `-g` | `0` | GPU layer count: `0`=CPU-only, `-1`=auto (profile VRAM), `N`=N layers on GPU |

## Available tools (demo)

| Tool | Description |
|------|-------------|
| `get_weather` | Returns weather for a named city (canned data for demo) |
| `calculate` | Evaluates arithmetic expressions via a built-in recursive-descent parser |

## Message flow

```
Turn 1 ──────────────────────────────────────────────────────────
  User:   "What is the weather in Paris and 42 * 17?"
  Prompt: Jinja template with tool schemas injected into system prompt

  Model → "<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>
           <tool_call>{"name":"calculate","arguments":{"expression":"42 * 17"}}</tool_call>"

Tool execution ───────────────────────────────────────────────────
  get_weather("Paris")  → "Partly cloudy, 18 °C."
  calculate("42 * 17")  → "714"

Turn 2 ──────────────────────────────────────────────────────────
  Messages now contain the assistant tool_call + two role="tool" results
  Model → "The weather in Paris is partly cloudy at 18 °C, and 42 × 17 = 714."
```

## Extending

Add a new tool in two steps:

1. **Schema** — add another `MakeFunctionTool(...)` entry to `toolDefinitions`
2. **Implementation** — add a case to `ExecuteTool(...)` and a handler function

For GPU inference, swap `CpuBackend` + `ForwardPass` for `VulkanBackend` + `GpuForwardPass` (add the `OpenTail.Stingray.Vulkan` project reference) — the rest of the sample is unchanged.

## Prerequisites

- The .NET 10 SDK, a 64-bit x86 CPU with AVX2, and a GGUF chat model whose template supports tool calls. The run below used `Qwen3-8B-Q4_K_M.gguf` (about 5 GB; roughly 8 GB of RAM).

## Expected output

The model asks for three tool calls, the sample runs them, and a second turn writes the answer. The run behind this README (Ryzen 7 5700G, CPU, 38 s in total):

```text
User: What is the weather like in Paris and Tokyo, and what is 42 times 17?
[model called 3 tool(s)]
  tool 'get_weather' -> Partly cloudy, 18 °C. Light westerly breeze.
  tool 'get_weather' -> Clear skies, 24 °C. Humid.
  tool 'calculate' -> 714
Assistant: The weather in Paris is partly cloudy with a temperature of 18 °C ... 42 times 17 equals **714**.
```

## Known limitations

- The weather tool returns canned data for a few cities; it demonstrates the loop, not a real service.
- Tool-call parsing depends on the model emitting the `<tool_call>` format its template defines; a model that does not will produce no tool calls.
- The CUDA commands above were **not run** (there is no NVIDIA GPU on the development machine); only the CPU command is verified.
- Uses engine internals (`GgufModel`, `ToolGrammar`, `TierPlanner`), not the public package contract.

## Verification

- **Run against real weights, 2026-10-11:** Qwen3-8B Q4_K_M on the CPU backend, default prompt, as shown; the arithmetic is correct (42 x 17 = 714). One run.
- CUDA paths: unverified. Not built against the packed package (it is a contributor example).
