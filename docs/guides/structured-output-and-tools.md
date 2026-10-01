# Guide: structured JSON output and tool calling

**You get:** two ways to make a model's output something a program can use.

- **Structured output:** the whole answer is forced to match a JSON Schema, so it always parses.
- **Tool calling:** you describe functions the model may call, it answers with a call and
  arguments, your code runs the function.

**You need:** a chat model with a tool-aware chat template (Qwen3, Qwen3-Coder, Gemma 4, Llama 3
and DeepSeek families are the ones the constrained mode supports). Start with a model you already
run: [chat guide](chat.md).

## Structured JSON output

```bash
stingray -m models/Qwen3-4B-Q4_K_M.gguf --temp 0 \
  -p "Extract the city and country from: I flew to Lyon, France last week." \
  --json-schema '{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"}},"required":["city","country"]}'
```

For a longer schema use `--json-schema-file schema.json` (alias `--jf`).

Rules to know:

- The root must be an **object schema with at least one property**.
- Plain object schemas (properties, types, `required`, `enum`) are the intended use. **Not enforced:** `$ref`, `oneOf`/`anyOf`,
  `pattern`, `minLength`/`maxLength`, `minimum`/`maximum`. Using them does not fail; that part just
  stops being constrained, so validate the result in your own code.
- `--json-schema-ordered` makes the model emit properties in the order you declared them. That
  lets a program that reads the stream act on an early field before a later, longer one finishes.

## Tool calling

Describe your tools in a JSON file in the OpenAI format:

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Current weather for a city",
      "parameters": {
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"]
      }
    }
  }
]
```

```bash
stingray -m models/Qwen3-8B-Q4_K_M.gguf --tools tools.json -p "What is the weather in Paris?"
```

On a single-prompt run, the parsed tool calls are printed after generation. Add `--tool-grammar` to
constrain the arguments to your schemas, so required keys cannot be dropped, only declared keys and
enum values appear and the value types match. It needs `--tools` and a model family with
constraint support, and is off by default.

The CLI shows the model's call. **Running the tool and feeding the result back is your code's job.**
For the full loop (call, run the tool, send the result as a `tool` message, get the final answer)
see [samples/OpenTail.Stingray.Sample.ToolCall](../../samples/OpenTail.Stingray.Sample.ToolCall).

## Over HTTP

The server's `/v1/chat/completions` accepts the usual OpenAI `tools` and structured-output
request fields. See the [serving guide](serve-an-api.md).

## If the output is wrong

A model can still choose the wrong tool or invent a value that fits the schema but is false.
Constraints guarantee the **shape**, not the truth. Always validate and check the result.
