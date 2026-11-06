// OpenTail.Stingray library sample: the one-call Stingray.CreateAsync facade.
//
//   dotnet run --project samples/OpenTail.Stingray.Sample.QuickChat -c Release -- \
//       "Why do developers write unit tests?"
//
// Stingray.CreateAsync resolves "qwen2.5-0.5b" (installed by `stingray setup chat`, or
// auto-downloaded and SHA-256-verified here if it isn't yet), loads it, and hands back a ready
// chat session — no explicit Model.Load / CreateContext / InteractiveExecutor / ChatSession
// wiring. See OpenTail.Stingray.Sample.Chat for interactive multi-turn chat built from those
// lower-level types directly, when you need more control than this facade exposes.

using OpenTail.Stingray;

string prompt = args.Length > 0 ? string.Join(' ', args) : "In one sentence, why do developers write unit tests?";

Console.Error.Write("loading qwen2.5-0.5b ... ");
await using var ai = await Stingray.CreateAsync("qwen2.5-0.5b");
Console.Error.WriteLine($"{ai.Model.Architecture} (ctx {ai.Context.ContextSize})");

await foreach (string chunk in ai.ChatAsync(prompt))
{
    Console.Out.Write(chunk);
    Console.Out.Flush();
}
Console.WriteLine();
