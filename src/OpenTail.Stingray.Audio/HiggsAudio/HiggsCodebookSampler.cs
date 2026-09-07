namespace OpenTail.Stingray.Audio.HiggsAudio;

/// <summary>
/// Real per-codebook delay-pattern state machine for Higgs Audio TTS's AR generation, ported
/// from `sampler.cpp`'s `HiggsCodebookSampler::step`/`HiggsSamplerState` (not guessed). Real
/// reserved audio-vocab ids: `BocId=1024` ("beginning of codebook" placeholder), `EocId=1025`
/// ("end of codebook" signal), `StopCode=-1` (sentinel emitted once generation is done --
/// NEVER a valid codec code, callers must not feed it to <see cref="HiggsCodecDecoder"/>).
///
/// <para>Real staggered delay: at step N (0-indexed), only codebooks `[0, N]` (clamped to
/// `numCodebooks-1`) carry a real sampled code; every codebook beyond that is overwritten with
/// `BocId` before being returned -- structurally similar to MusicGen/Parler-TTS's delay pattern
/// but with no un-delay transform needed at the end (this codec's per-frame code layout is
/// consumed directly, delay only matters during generation). Real stopping: if codebook 0's
/// sampled id is `EocId`, an `eocCountdown` of `numCodebooks-2` further steps begins (or
/// generation ends immediately if `numCodebooks&lt;=2`); once it reaches 0, every subsequent
/// `Step` call returns all-`StopCode` without running the model at all (matches the real
/// reference's own short-circuit).</para>
///
/// <para>This class only implements the real STATE MACHINE (delay masking + stop detection) --
/// per-codebook selection is currently greedy argmax via the caller's
/// <see cref="HiggsArStepper.SampleFromHidden"/>/<see cref="HiggsArStepper.Step"/>, NOT the real
/// reference's temperature/top-p/top-k sampling (`sample_codebook_row`, not yet ported --
/// same "known gap" discipline as this project's other argmax-only stand-ins).</para>
/// </summary>
public sealed class HiggsCodebookSampler
{
    public const int BocId = 1024;
    public const int EocId = 1025;
    public const int StopCode = -1;

    private readonly int _numCodebooks;
    private int _delayCount;
    private int? _eocCountdown;

    public bool GenerationDone { get; private set; }
    public int[] LastCodes { get; private set; }

    public HiggsCodebookSampler(int numCodebooks)
    {
        if (numCodebooks <= 0) throw new ArgumentOutOfRangeException(nameof(numCodebooks));
        _numCodebooks = numCodebooks;
        LastCodes = new int[numCodebooks];
    }

    /// <summary>Applies the real delay mask and stop-detection to one step's raw per-codebook
    /// argmax codes (from <see cref="HiggsArStepper.SampleFromHidden"/>), advancing internal
    /// state. Returns the codes to actually use (masked with <see cref="BocId"/> where the delay
    /// hasn't unlocked that codebook yet, or all <see cref="StopCode"/> if generation is
    /// already done).</summary>
    public int[] Step(int[] rawCodes)
    {
        if (rawCodes.Length != _numCodebooks) throw new ArgumentException("rawCodes length must equal numCodebooks.");

        if (GenerationDone)
        {
            var stopped = new int[_numCodebooks];
            Array.Fill(stopped, StopCode);
            return stopped;
        }

        var codes = (int[])rawCodes.Clone();
        if (_delayCount < _numCodebooks)
        {
            int nextCodebook = _delayCount + 1;
            if (nextCodebook < _numCodebooks)
                for (int cb = nextCodebook; cb < _numCodebooks; cb++) codes[cb] = BocId;
            _delayCount++;
        }
        else if (_eocCountdown is int countdown)
        {
            countdown--;
            _eocCountdown = countdown;
            if (countdown <= 0) GenerationDone = true;
        }
        else if (codes[0] == EocId)
        {
            if (_numCodebooks <= 2) GenerationDone = true;
            else _eocCountdown = _numCodebooks - 2;
        }

        if (!GenerationDone) LastCodes = codes;
        return codes;
    }
}
