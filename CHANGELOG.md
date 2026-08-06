# Changelog

All notable user-visible changes are recorded here. Git tags and MinVer provide package versions;
this file provides the human-facing map that a raw commit graph cannot.

## Unreleased

### Added

- CPU execution for the published dense Llama/Mistral SafeTensors profile (F32/F16/BF16), with
  explicit capability boundaries; GGUF remains the recommended quantized deployment format.
- Recommended deployment profiles for CPU-only, CUDA dense, Vulkan, hybrid MoE, and local-server use.
- Guidance for the live model × backend × dtype × batching × speculation capability report.
- A release-quality matrix and retained CI test transcripts for package publication.

### Changed

- Capability reports explicitly distinguish experimental retained-session internals from a supported
  restart-continuation product feature.

## Release notes policy

- Add entries when a change alters user-visible behaviour, compatibility, performance claims,
  configuration defaults, or operational requirements.
- At a release tag, move relevant entries into a version/date heading and link the test receipt.
- Keep entries factual: name the backend/model scope and important limitation.
