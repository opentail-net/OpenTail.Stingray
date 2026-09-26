// An OpenAI-compatible chat endpoint (/v1/chat/completions) inside your own ASP.NET app.
//   dotnet run -- path/to/model.gguf
using OpenTail.Stingray.Server;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddOpenTailStingray(builder.Configuration, o => o.ModelPath = args[0]);

var app = builder.Build();
app.MapOpenTailStingray();
app.Run("http://localhost:5080");
