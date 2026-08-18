# Third-party notices for OpenTail.Stingray

OpenTail-authored source code is licensed under the MIT License in
[`LICENSE`](LICENSE).

OpenTail.Stingray also contains, derives from, references, or depends upon
third-party software and upstream projects whose copyright and license notices
are reproduced below.

These notices do not alter the license of OpenTail-authored work.

## SharpInference — MIT

OpenTail.Stingray is derived in part from SharpInference and contains source
code originating from SharpInference that has subsequently been modified and
extended by OpenTail.

- Upstream: <https://github.com/pekkah/SharpInference>
- Copyright (c) 2026 Pekka Heikura
- License: MIT

MIT License

Copyright (c) 2026 Pekka Heikura

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## TensorSharp — BSD 3-Clause

The MXFP4 decoding implementation in `src/OpenTail.Stingray.Cpu` is derived from
TensorSharp's managed quantized operations:

- Upstream: <https://github.com/zhongkaifu/TensorSharp>
- Copyright (c) 2026, Zhongkai Fu
- License: BSD 3-Clause

BSD 3-Clause License

Copyright (c) 2026, Zhongkai Fu

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## llama.cpp / ggml — MIT

OpenTail.Stingray uses the GGUF format and quantization layouts maintained by the
llama.cpp / ggml project as a compatibility reference. The CPU dequantizers and
GGUF type-layout definitions are maintained against that reference.

In addition, the byte-level BPE pre-tokenizer split patterns in
`src/OpenTail.Stingray.Core/PreTokenizerPatterns.cs` are **ported from** llama.cpp's
`src/llama-vocab.cpp` (the `llm_tokenizer_bpe` regex table), and are therefore
derived source rather than a format reference.

No llama.cpp or ggml binary is bundled with OpenTail.Stingray.

- Upstream: <https://github.com/ggml-org/llama.cpp>
- Reference build used for parity evidence: `b8585-cpu`
- Copyright (c) 2023-2026 The ggml authors
- License: MIT

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## vLLM — Apache License 2.0

OpenTail.Stingray's continuous-batching and paged-KV-cache design was reviewed
against vLLM's published architecture. No vLLM source files or binaries are
included in OpenTail.Stingray: its cache implementation and kernels are
independently written. This notice is retained with distributed source and
binaries to preserve the Apache-2.0 attribution requested for that upstream
reference.

- Project: vLLM
- Upstream: <https://github.com/vllm-project/vllm>
- License: Apache License, Version 2.0

Apache License

Version 2.0, January 2004

<http://www.apache.org/licenses/>

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

"License" shall mean the terms and conditions for use, reproduction, and
distribution as defined by Sections 1 through 9 of this document.

"Licensor" shall mean the copyright owner or entity authorized by the
copyright owner that is granting the License.

"Legal Entity" shall mean the union of the acting entity and all other
entities that control, are controlled by, or are under common control with
that entity. For the purposes of this definition, "control" means (i) the
power, direct or indirect, to cause the direction or management of such entity,
whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or
more of the outstanding shares, or (iii) beneficial ownership of such entity.

"You" (or "Your") shall mean an individual or Legal Entity exercising
permissions granted by this License.

"Source" form shall mean the preferred form for making modifications,
including but not limited to software source code, documentation source, and
configuration files.

"Object" form shall mean any form resulting from mechanical transformation or
translation of a Source form, including but not limited to compiled object code,
generated documentation, and conversions to other media types.

"Work" shall mean the work of authorship, whether in Source or Object form,
made available under the License, as indicated by a copyright notice that is
included in or attached to the work.

"Derivative Works" shall mean any work, whether in Source or Object form, that
is based on (or derived from) the Work and for which the editorial revisions,
annotations, elaborations, or other modifications represent, as a whole, an
original work of authorship. For the purposes of this License, Derivative Works
shall not include works that remain separable from, or merely link (or bind by
name) to the interfaces of, the Work and Derivative Works thereof.

"Contribution" shall mean any work of authorship, including the original
version of the Work and any modifications or additions to the Work or
Derivative Works thereof, that is intentionally submitted to Licensor for
inclusion in the Work by the copyright owner or by an individual or Legal Entity
authorized to submit on behalf of the copyright owner. For the purposes of this
definition, "submitted" means any form of electronic, verbal, or written
communication sent to the Licensor or its representatives for inclusion in the
Work, excluding communication that is conspicuously marked or otherwise
designated in writing by the copyright owner as "Not a Contribution."

"Contributor" shall mean Licensor and any individual or Legal Entity on behalf
of whom a Contribution has been received by Licensor and subsequently
incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of this
License, each Contributor hereby grants to You a perpetual, worldwide,
non-exclusive, no-charge, royalty-free, irrevocable copyright license to
reproduce, prepare Derivative Works of, publicly display, publicly perform,
sublicense, and distribute the Work and such Derivative Works in Source or
Object form.

3. Grant of Patent License. Subject to the terms and conditions of this License,
each Contributor hereby grants to You a perpetual, worldwide, non-exclusive,
no-charge, royalty-free, irrevocable (except as stated in this section) patent
license to make, have made, use, offer to sell, sell, import, and otherwise
transfer the Work, where such license applies only to those patent claims
licensable by such Contributor that are necessarily infringed by their
Contribution(s) alone or by combination of their Contribution(s) with the Work
to which such Contribution(s) was submitted. If You institute patent litigation
against any entity alleging that the Work or a Contribution incorporated within
the Work constitutes direct or contributory patent infringement, then any patent
licenses granted to You under this License for that Work shall terminate as of
the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the Work or
Derivative Works thereof in any medium, with or without modifications, and in
Source or Object form, provided that You meet the following conditions:

  (a) You must give any other recipients of the Work or Derivative Works a copy
  of this License; and

  (b) You must cause any modified files to carry prominent notices stating that
  You changed the files; and

  (c) You must retain, in the Source form of any Derivative Works that You
  distribute, all copyright, patent, trademark, and attribution notices from
  the Source form of the Work, excluding those notices that do not pertain to
  any part of the Derivative Works; and

  (d) If the Work includes a "NOTICE" text file as part of its distribution,
  then any Derivative Works that You distribute must include a readable copy of
  the attribution notices contained within such NOTICE file, excluding those
  notices that do not pertain to any part of the Derivative Works, in at least
  one of the following places: within a NOTICE text file distributed as part of
  the Derivative Works; within the Source form or documentation, if provided
  along with the Derivative Works; or, within a display generated by the
  Derivative Works, if and wherever such third-party notices normally appear.
  The contents of the NOTICE file are for informational purposes only and do
  not modify the License. You may add Your own attribution notices within
  Derivative Works that You distribute, alongside or as an addendum to the
  NOTICE text from the Work, provided that such additional attribution notices
  cannot be construed as modifying the License.

You may add Your own copyright statement to Your modifications and may provide
additional or different license terms and conditions for use, reproduction, or
distribution of Your modifications, or for any such Derivative Works as a
whole, provided Your use, reproduction, and distribution of the Work otherwise
complies with the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise, any
Contribution intentionally submitted for inclusion in the Work by You to the
Licensor shall be under the terms and conditions of this License, without any
additional terms or conditions. Notwithstanding the above, nothing herein shall
supersede or modify the terms of any separate license agreement you may have
executed with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade names,
trademarks, service marks, or product names of the Licensor, except as required
for reasonable and customary use in describing the origin of the Work and
reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or agreed to in
writing, Licensor provides the Work (and each Contributor provides its
Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied, including, without limitation, any warranties
or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
PARTICULAR PURPOSE. You are solely responsible for determining the
appropriateness of using or redistributing the Work and assume any risks
associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory, whether in
tort (including negligence), contract, or otherwise, unless required by
applicable law (such as deliberate and grossly negligent acts) or agreed to in
writing, shall any Contributor be liable to You for damages, including any
direct, indirect, special, incidental, or consequential damages of any character
arising as a result of this License or out of the use or inability to use the
Work (including but not limited to damages for loss of goodwill, work stoppage,
computer failure or malfunction, or any and all other commercial damages or
losses), even if such Contributor has been advised of the possibility of such
damages.

9. Accepting Warranty or Additional Liability. While redistributing the Work or
Derivative Works thereof, You may choose to offer, and charge a fee for,
acceptance of support, warranty, indemnity, or other liability obligations
and/or rights consistent with this License. However, in accepting such
obligations, You may act only on Your own behalf and on Your sole responsibility,
not on behalf of any other Contributor, and only if You agree to indemnify,
defend, and hold each Contributor harmless for any liability incurred by, or
claims asserted against, such Contributor by reason of your accepting any such
warranty or additional liability.

END OF TERMS AND CONDITIONS

## Runtime NuGet dependencies — MIT

The following packages are runtime dependencies of one or more distributable
OpenTail.Stingray packages. They remain under their respective upstream MIT terms.
The dependency graph is resolved by NuGet at build/install time; package
versions may vary where the project intentionally uses a floating version.

| Component | Copyright / upstream attribution |
| --- | --- |
| Microsoft.ML.Tokenizers | Copyright (c) .NET Foundation and Contributors |
| Microsoft.Extensions.Logging.Abstractions | © Microsoft Corporation. All rights reserved. |
| System.Numerics.Tensors | © Microsoft Corporation. All rights reserved. |
| Vortice.Vulkan | Copyright (c) Amer Koleci and Contributors |
| Spectre.Console.Cli | Patrik Svensson, Phil Scott, Nils Andresen, Cédric Luthi, Frank Ray |

The MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Kokoro TTS References (KOKORO-GPT2, kokoro.cpp, kokoro-infer) — MIT / Apache-2.0

The native C# Text-to-Speech (TTS) architecture, G2P phonemizer frontend, and PLBERT/AdaIN neural vocoder graph in src/OpenTail.Stingray.Audio reference the following C++ implementations:

- **KOKORO-GPT2**: <https://github.com/Himanshu040604/KOKORO-GPT2>
  - Copyright (c) 2026 Himanshu & Contributors
  - License: Apache 2.0 / MIT
- **kokoro.cpp**: <https://github.com/Zackriya-Solutions/kokoro.cpp>
  - Copyright (c) 2026 Zackriya Solutions & Contributors
  - License: MIT
- **kokoro-infer / kokoro-server**: <https://github.com/remsnet/kokoro-server>
  - Copyright (c) 2026 Contributors
  - License: MIT

## Piper (VITS) TTS — MIT

The native C# Piper VITS Text-to-Speech (TTS) architecture, character/IPA token intersperser, normalizing flow, and HiFi-GAN MRF neural vocoder in src/OpenTail.Stingray.Audio/Piper reference the Piper TTS project:

- **Piper**: <https://github.com/rhasspy/piper>
- Copyright (c) 2022 Michael Hansen
- License: MIT

MIT License

Copyright (c) 2022 Michael Hansen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## F5-TTS & CrispASR — MIT

The native C# F5-TTS Flow-Matching Diffusion Transformer (DiT), ConvNeXtV2 text encoder, 100-channel mel extractor, and Vocos neural vocoder in src/OpenTail.Stingray.Audio/F5TTS reference SWivid/F5-TTS and CrispASR:

- **SWivid/F5-TTS**: <https://github.com/SWivid/F5-TTS>
  - Copyright (c) 2024 Yushen Chen and Contributors
  - License: MIT
- **CrispASR**: <https://github.com/thewh1teagle/crispasr>
  - Copyright (c) 2026 CrispASR Contributors
  - License: MIT

MIT License

Copyright (c) 2024 Yushen Chen and F5-TTS Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Chatterbox-Turbo & Resemble AI — MIT

The native C# Chatterbox-Turbo Text-to-Speech (TTS) architecture, autoregressive acoustic language model, discrete speech token generator, and conditional neural audio decoder in src/OpenTail.Stingray.Audio/Chatterbox reference Chatterbox-turbo-cpp and Resemble AI:

- **Chatterbox-turbo-cpp**: <https://github.com/DDATT/Chatterbox-turbo-cpp>
  - Copyright (c) 2026 DDATT
  - License: MIT
- **Resemble AI Chatterbox**: <https://github.com/resemble-ai/chatterbox>
  - Copyright (c) Resemble AI
  - License: MIT

MIT License

Copyright (c) 2026 DDATT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## MeloTTS & MeloTTS.cpp — MIT / Apache-2.0

The native C# MeloTTS multilingual VITS Text-to-Speech (TTS) architecture, tone and language ID conditioning, phone-level context feature fusion, and 44.1kHz HiFi-GAN neural vocoder in src/OpenTail.Stingray.Audio/MeloTTS reference MeloTTS (MyShell.ai) and MeloTTS.cpp (Intel):

- **MeloTTS**: <https://github.com/myshell-ai/MeloTTS>
  - Copyright (c) 2023 MyShell.ai
  - License: MIT
- **MeloTTS.cpp**: <https://github.com/intel/MeloTTS.cpp>
  - Copyright (C) 2024-2025 Tong Qiu, Vincent Liu, Intel Corporation
  - License: Apache 2.0

MIT License

Copyright (c) 2023 MyShell.ai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## OpenAI Whisper & whisper.cpp — MIT

The native C# Whisper automatic speech recognition (ASR) architecture, 80/128-channel 16kHz log-mel spectrogram extractor, audio transformer encoder with Conv1D downsampling, and cross-attention autoregressive transformer decoder in src/OpenTail.Stingray.Audio/Whisper reference OpenAI Whisper and whisper.cpp:

- **OpenAI Whisper**: <https://github.com/openai/whisper>
  - Copyright (c) OpenAI
  - License: MIT
- **whisper.cpp**: <https://github.com/ggerganov/whisper.cpp>
  - Copyright (c) 2023-2026 The ggml authors
  - License: MIT

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
