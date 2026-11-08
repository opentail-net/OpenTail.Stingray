# 2026-11-01-plan-4 - completed items 2026-11

Ticked items moved from [2026-11-01-plan-4.md](../2026-11-01-plan-4.md) on 2026-11-08. Each keeps its text and nested lines. Items with open follow-up text stay in the source doc.

- [x] **C2** *(closed 2026-11-01. Last part, the transcription `model` field, now validates the model and answers 503 `model_not_installed` with a setup hint when its weights are missing (`ValidateTranscriptionModel`, `TranscriptionModelFieldTests`; plan 5 E3, PRs 143 and 165). `language` is validated by `RequestParameterValidation.ValidateLanguageCode`. Earlier state, kept for the record: partly done 2026-11-01: speech `speed` outside 0.25..4 is now a 400, `language` must be a two or three letter code, rerank `top_n` below 1 is a 400, each tested and mutation-checked; speech `model` is NOT ignored, it picks Melo, Chatterbox, F5 or Piper by name; what remains is the transcription `model` field, which is ignored, a behaviour question for the owner)* The audio routes ignore `model` (the route always builds a `WhisperPipeline`) and do not validate `language`; `/v1/audio/speech` replaces a non-positive `speed`
  with 1.0; `/v1/rerank` has no `top_n` range check. Done when each is either refused with a 400 that names the field or documented as ignored with the reason. A `model`
  that selects a pipeline is a behaviour change and needs the owner.
