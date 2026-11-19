# Fish Audio Drama 3 — optional voice module (plan only)

This is a plan. Nothing here is implemented. Drama 3 stays out of the installer, the required model list, and the default voice catalog until the open questions below are settled.

Checked against the published pages on 2026-10-17.

## What the model actually is

Drama 3 and S1 are different models. This plan is for Drama 3 only.

| | Drama 3 | S1 / S1-mini |
|---|---|---|
| Weights | [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) | S1 (4B) is proprietary. [fishaudio/s1-mini](https://huggingface.co/fishaudio/s1-mini) is 0.5B |
| Licence | Fish Audio Community License, Version 1.0 (13 October 2026) | S1-mini is CC-BY-NC-SA-4.0 and gated on Hugging Face |
| Role here | The optional plug-in | Not a substitute. S1-mini's licence is non-commercial and share-alike, and the repo asks the user to share contact details |

Fish Audio's own install note (search snippet of [speech.fish.audio/install](https://speech.fish.audio/install/)) is:

```text
hf download fishaudio/drama-3 --local-dir checkpoints/drama-3
```

A full read of that install page on the same day returned the older S2 instructions (including a 24 GB GPU line). That 24 GB figure is not a Drama 3 requirement. The source of record for the weights is the Hugging Face repository above. Code lives at [github.com/fishaudio/fish-speech](https://github.com/fishaudio/fish-speech). The launch post is [fish.audio/blog/fish-audio-drama-3](https://fish.audio/blog/fish-audio-drama-3/) (13 October 2026): 35B MoE, about 3B active, 83 languages. The model card rounds the size to about 36B.

If `codec.pth` is missing, Fish points at the S2-Pro codec (`hf download fishaudio/s2-pro`). Those weights are under the Fish Audio Research License, not the Community License. This plan does not download them. See open questions.

## Licence (restrictive, so off by default)

Text: [LICENSE](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE) ([raw](https://huggingface.co/fishaudio/drama-3/raw/main/LICENSE)). Title: FISH AUDIO COMMUNITY LICENSE AGREEMENT, Version 1.0, last updated 13 October 2026. The licensor is 39 AI, INC.

- Research and non-commercial use is free (section II).
- A community commercial licence (III(a)) applies only while the user's and its affiliates' annual revenue stays under USD 500,000 (trailing 12 months, all sources). Over that line the user must email business@fish.audio within 30 days and gets 90 days to move off the community terms (III(b)).
- A Hosted Service is not licensed at any revenue (III(c)): letting a third party generate speech from their own text, audio, or voice samples through an API, SDK, host, or a site whose primary purpose is TTS, voiceover, voice generation, or voice cloning. The licence's own exceptions are generating speech for content the user creates (narration, dubbing) and delivering the finished audio, and an agency or studio that delivers only finished output. This app narrates a book the user chose and writes audio files. That is the narration exception, not a licence to expose the model as a voice API.
- Attribution (IV(a)): if the product uses the materials, show "Built with Fish Audio" and ship the licence plus the notice "These Fish Audio Materials are licensed under the Fish Audio Community License, Version 1.0. Copyright © 39 AI, INC. All Rights Reserved."
- Use limits (IV(b)): no imitating an identifiable person without prior consent (a minor: parent or guardian; a deceased person: the rights holder); no impersonation to deceive, defraud, or harass; no sexual content of a minor or of a real person's voice without consent; no stripping watermarks. Training or distilling a foundational or speech model from the materials or their output is also banned (detection models excepted).
- The model definition covers Drama 3 and, only for use with Drama 3, the audio codec including the copy shipped with S2-Pro. The S2-Pro weights themselves are not Fish Audio Materials.

The in-app text the user accepts is the upstream licence file, not a summary. The summary above is for this plan.

## Off until the user accepts

- The module is absent from the required tiers in `voxprint-app-models.json` (`8gb`, `16gb`, and the `any` rows that setup downloads). `on_demand` is true.
- No download, no catalog row, and no engine load until a local acceptance record exists. The record stores the licence version (1.0), the UTC time, and that the user accepted the text of that file.
- The CLI does not download on a bare `models download`. A separate flag is required, for example `voxprint models download fish-drama-3 --accept-fish-audio-license`, and only after the same record is written. Refusing the flag leaves the weights untouched.
- The window shows the licence text and an accept control before it asks MDL to fetch. Closing the window is a refusal.

## Where the weights come from

voxprint-common 0.2.4 has no field named `external-source`. The download path that matches this plan is one mirror and no second entry.

In `voxprint-app-models.json` each file's `mirrors` array is a single object:

```json
{
  "source": "huggingface",
  "repo": "fishaudio/drama-3",
  "filename": "<path inside the repo>",
  "revision": "<pinned commit>"
}
```

MDL turns that into `https://huggingface.co/fishaudio/drama-3/resolve/<revision>/<file>` (`voxprint_common.download.huggingface_url`). That is the official repository Fish names. It is not a VOXPrint mirror and not a re-upload.

Do not add a ModelScope, Kaggle, or `Mitroshenkov87/voxprint-mirror-*` entry. Do not call `public_mirror_urls` (that adds the suite Kaggle dataset). A descriptor, if one is used instead of the app manifest, has the same shape: `mirrors: [{ "kind": "hf", "ref": "fishaudio/drama-3", "revision": "<commit>" }]` and nothing else. `kind: "url"` is only for that same official https base.

`is_gated_source` in voxprint-common 0.2.4 lists only the pyannote repos. `fishaudio/drama-3` is not in that set. If every URL were gated, MDL refuses the model and asks for a suite mirror. This plan does not add one. See open questions.

`infra/external_models.py` is a different feature (a folder of weights the user already has). It is not the download path.

Hashes and sizes are not pinned here. They are copied from the pinned revision at implementation time. This plan does not invent them.

## Integration points

- **Engine.** `core.narration.TTSEngine` is the narrator's adapter: `sample_rate`, `tag`, `synthesize(text) -> float32 mono`, `close()`. `core.tts_engine.Qwen3AdapterEngine` stays the Qwen path. Drama 3 gets its own adapter behind the same protocol. Narration selects it from the voice record, not by replacing Qwen.
- **Catalog.** A voice-catalog entry (see `infra/voice_catalog.py` and `core/voice_library.py`) appears only after acceptance and a finished download. It is not a stock voice. Levi and Miriam stay the stock voices. The entry carries the Fish licence id and the attribution line. This app's `voice.json` consent fields stay as they are: the Fish rule about imitating a real person is additional, not a replacement.
- **CLI.** The flag above, on the existing `models download` command (`cli.py`). `infra/mdl_gate.py` still only checks `ready` and links blobs. It does not download and it does not grow a second downloader.
- **VRAM.** The app manifest tiers that setup installs are `8gb`, `16gb`, and `any`. Drama 3 is in none of them. [MarkTechPost](https://www.marktechpost.com/2026/10/14/fish-audio-drama-3-open-weight-tts-model/) (14 October 2026) and [this install write-up](https://ccleaks.com/news/fish-audio-drama-3-open-weights-oct-2026) both attribute a floor of one NVIDIA GPU with at least 48 GB in FP8 to the Fish Speech install guide, with no CPU, MPS, or ROCm path. A direct read of [speech.fish.audio/install](https://speech.fish.audio/install/) on 17 October 2026 still returned the older S2 page (its 24 GB line is S2, not Drama 3). A 16 GB machine does not fit the full FP8 weights. There is no verified official quant to put in a 16 GB tier. The plug-in reports that it does not fit, and does not start a download, when the detected VRAM is below the pinned floor.

## Open questions

1. Exact file list, byte sizes, and SHA-256 for a pinned `fishaudio/drama-3` commit. Not invented here.
2. Whether Fish publishes an official quant that fits 16 GB. Until that exists, the module is unavailable on that tier.
3. The S2-Pro codec is a different licence. Confirm whether Drama 3 runs without those weights. If it does not, the codec needs its own acceptance step and its own single official URL, or the module stays unavailable.
4. If Hugging Face later gates `fishaudio/drama-3`, MDL 0.2.4 will refuse a gated-only model. The answer is not a VOXPrint re-upload. The module stays off until MDL can fetch that one upstream with the user's own acceptance.
5. Where the "Built with Fish Audio" line is shown (narration metadata, the about box, or both) so section IV(a) is met without a UI redesign before the shared layout lands.
6. Revenue over USD 500,000 is the user's duty under III(b). The app does not know the user's revenue and must not pretend to.
