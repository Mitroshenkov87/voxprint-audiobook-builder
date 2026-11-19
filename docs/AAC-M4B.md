# AAC / M4B: patents (please read)

> The **M4B (AAC)** export exists only as a convenience for **Apple Books compatibility**. The **AAC codec is patent-encumbered** and not fully open. **This project does not provide a patent licence** for it.
> **You are solely responsible for any legal compliance** (patent licensing, royalties, distribution rules that apply to you) when you choose this format. The default formats (Opus, MP3 per chapter, FLAC, WAV) are not affected.

The same text (localized into English, Russian and German) is shown in the program when M4B is selected, and in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

**Hide or disable AAC.** One switch, three ways (highest priority first):
* environment variable `VOXPRINT_ENABLE_AAC=0` (`1` forces it on);
* `%LOCALAPPDATA%\Voxprint\state\features.json` containing `{"aac_m4b": false}`;
* the default `AAC_DEFAULT` in `infra/features.py` (a distributor can set it to `False`).

When disabled the M4B entry disappears from the window and the narrator refuses to produce M4B/AAC even if asked programmatically. The encoders themselves (`aac`, `libopus`, `libmp3lame`) come from the ffmpeg build in use; the export checks for them
before synthesis starts and stops with a clear message if one is missing (whether the pinned ffmpeg build contains all three has **not** been verified yet).
