# OpenVoice (tone-colour converter only)

Unmodified copies of the inference modules from [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice)
at commit `74a1d147b17a8c3092dd5430504bd83ef6c7eb23`:

`attentions.py`, `commons.py`, `mel_processing.py`, `models.py`, `modules.py`, `transforms.py`.

Licence: MIT (`LICENSE` in this folder, also `licenses/openvoice.txt`). Free for commercial use.

The text-to-speech frontend, watermark and base-speaker checkpoints are not included. Voxprint uses only the
V2 tone-colour converter (`converter/checkpoint.pth`) to change timbre while the source spectrogram keeps the
speaker's timing and intonation. See `docs/REVOICE.md`.
