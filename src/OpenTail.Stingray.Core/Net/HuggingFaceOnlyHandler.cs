using System.Net;
using System.Net.Http.Headers;

namespace OpenTail.Stingray.Core.Net;

/// <summary>
/// The policy of <see cref="ExternalHttpClient"/> as an <see cref="HttpMessageHandler"/>, for the large-file download path that must keep an
/// <see cref="HttpClient"/> signature: external access is re-evaluated before every request, only HTTPS to Hugging Face hosts is contacted, redirects are
/// followed by hand so every hop is host-checked, and an <c>Authorization</c> header survives only on requests to the Hub itself (never to the CDN a download
/// redirects to). The inner handler must not follow redirects on its own.
/// </summary>
public sealed class HuggingFaceOnlyHandler : DelegatingHandler
{
    private const int MaxRedirects = 5;
    private readonly Func<string, string?> _env;

    public HuggingFaceOnlyHandler(HttpMessageHandler inner, Func<string, string?>? env = null) : base(inner)
    {
        _env = env ?? (n => Environment.GetEnvironmentVariable(n));
    }

    /// <summary>A handler over a real socket handler with redirect-following off.</summary>
    public static HuggingFaceOnlyHandler CreateDefault() =>
        new(new SocketsHttpHandler { AllowAutoRedirect = false, AutomaticDecompression = DecompressionMethods.None });

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        for (int hop = 0; ; hop++)
        {
            var d = ExternalAccess.Evaluate(_env);
            if (!d.Allowed) throw new ExternalAccessDeniedException(d.Reason, d.HowToEnable);
            var uri = request.RequestUri ?? throw new InvalidOperationException("request has no URI");
            if (!ExternalAccess.IsAllowedHost(uri.Host) || uri.Scheme != Uri.UriSchemeHttps)
                throw new ExternalAccessDeniedException($"{uri.Scheme}://{uri.Host} is not a Hugging Face host", null);
            if (!ExternalAccess.MayReceiveToken(uri.Host)) request.Headers.Authorization = null;

            var response = await base.SendAsync(request, cancellationToken).ConfigureAwait(false);
            if ((int)response.StatusCode is >= 300 and < 400 && response.Headers.Location is { } loc)
            {
                response.Dispose();
                if (hop >= MaxRedirects) throw new HttpRequestException($"more than {MaxRedirects} redirects from {uri.Host}");
                var next = new HttpRequestMessage(HttpMethod.Get, loc.IsAbsoluteUri ? loc : new Uri(uri, loc));
                next.Headers.Range = request.Headers.Range;
                foreach (var ua in request.Headers.UserAgent) next.Headers.UserAgent.Add(ua);
                next.Headers.Authorization = request.Headers.Authorization;   // dropped above unless the next host is the Hub
                request = next;
                continue;
            }
            return response;
        }
    }
}
