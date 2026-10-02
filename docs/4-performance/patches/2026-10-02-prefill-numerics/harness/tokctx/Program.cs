using OpenTail.Stingray.Core;
var model = GgufModel.Open(args[0]);
var tok = GgufTokenizer.FromGgufModel(model);
foreach (var spec in args.Skip(1))
{
    var p = spec.Split('|'); // file:pos:tokA:tokB
    var ids = tok.Encode(File.ReadAllText(p[0])).Take(1200).ToArray();
    int pos = int.Parse(p[1]);
    string ctx = tok.Decode(ids.Skip(Math.Max(0, pos - 40)).Take(41).ToArray());
    string next = tok.Decode(ids.Skip(pos + 1).Take(6).ToArray());
    Console.WriteLine($"--- {Path.GetFileName(p[0])} pos {pos}\ncontext ending at pos: «{ctx.Replace("\n", "\n")}»\nactual next: «{next.Replace("\n", "\n")}»  A/D top: «{tok.Decode([int.Parse(p[2])])}»  E top: «{tok.Decode([int.Parse(p[3])])}»");
}
