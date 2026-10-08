# Third-party notices

Voxprint AI Audiobook Builder (c) Aleksandr Mitroshenkov, built with AI assistance. Version 0.1.4.

Voxprint stands on the open-source projects and models listed below. Every component keeps its own licence;
the full licence texts are in the `licenses/` folder next to this file. This file is generated from
`credits.json` by `tools/gen_notices.py` - edit `credits.json`, not this file.

## Models (downloaded on first start; not part of the installer)

### Qwen3-TTS-12Hz-1.7B-Base / 0.6B-Base

* Purpose: Base text-to-speech models that your voice is trained on
* Licence: Apache-2.0
* Project: <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base>
* Status: downloaded on first start
* Licence text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)

### Qwen3-ForcedAligner-0.6B

* Purpose: Neural forced aligner: matches every word of the text to a moment in the recording
* Licence: Apache-2.0
* Project: <https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B>
* Status: downloaded on first start
* Licence text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)

### Qwen3-ASR-1.7B / 0.6B

* Purpose: Speech recognition: transcribes recordings without a text and checks recordings, samples and the spoken consent (1.7B on GPUs with 8 GB of video memory or more, else 0.6B)
* Licence: Apache-2.0
* Project: <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
* Status: downloaded on first start
* Licence text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)

### MMS-300M forced aligner (MahmoudAshraf/mms-300m-1130-forced-aligner)

* Purpose: Optional backup aligner model (only if the optional ctc-forced-aligner package is installed). NON-COMMERCIAL licence!
* Licence: CC-BY-NC-4.0
* Project: <https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner>
* Status: optional, not bundled
* Licence text: [`licenses/CC-BY-NC-4.0.txt`](licenses/CC-BY-NC-4.0.txt)

### SAGE FRED-T5 distilled 95M (ai-forever/sage-fredt5-distilled-95m)

* Purpose: Optional Russian spelling and punctuation proposer for book preparation (downloaded on request, about 365 MB; every proposal is checked by a validator)
* Licence: MIT
* Project: <https://huggingface.co/ai-forever/sage-fredt5-distilled-95m>
* Status: downloaded on first start
* Licence text: [`licenses/sage-fredt5.txt`](licenses/sage-fredt5.txt)

### DNSMOS P.835 (microsoft/DNS-Challenge, sig_bak_ovr.onnx)

* Purpose: Predicts the perceived quality (MOS) of a voice sample for the automatic voice check (about 1.2 MB, runs on the CPU). Credit: Reddy, Gopal, Cutler, "DNSMOS P.835", ICASSP 2022, Microsoft; used unmodified
* Licence: CC-BY-4.0 (model) / MIT (reference code)
* Project: <https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS>
* Status: downloaded on first start
* Licence text: [`licenses/CC-BY-4.0.txt`](licenses/CC-BY-4.0.txt), [`licenses/dnsmos-code.txt`](licenses/dnsmos-code.txt)

### DeepFilterNet3 (deep-filter 0.5.6, Rikorose/DeepFilterNet)

* Purpose: Optional noise clean-up of a noisy training recording (downloaded on request from the Train window, about 27 MB, runs on the CPU; the program embeds the DeepFilterNet3 model). Credit: Schröter, Rosenkranz, Escalante-B., Maier, "DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement", INTERSPEECH 2023; used unmodified
* Licence: MIT OR Apache-2.0 (MIT chosen)
* Project: <https://github.com/Rikorose/DeepFilterNet>
* Status: downloaded on first start
* Licence text: [`licenses/deepfilternet.txt`](licenses/deepfilternet.txt)

### OpenVoice V2 tone-colour converter (myshell-ai/OpenVoice)

* Purpose: Direct re-voice: converts a recording into a library voice while keeping timing and intonation (part of the standard model download, about 130 MB; NVIDIA GPU when CUDA is available). MyShell OpenVoice V2; inference code and weights used unmodified
* Licence: MIT
* Project: <https://github.com/myshell-ai/OpenVoice>
* Status: downloaded on first start
* Licence text: [`licenses/openvoice.txt`](licenses/openvoice.txt)

### Gemma 4 12B Instruct (GGUF Q4_K_M, unsloth/gemma-4-12b-it-GGUF, base google/gemma-4-12B-it)

* Purpose: Optional AI text model for literary translation and preparing the text for narration (downloaded on request from the Narrate window, about 7.1 GB, needs about 10 GB of video memory). Gemma 4 by Google DeepMind; quantized GGUF by Unsloth; used unmodified
* Licence: Apache-2.0
* Project: <https://huggingface.co/google/gemma-4-12B-it>
* Status: downloaded on first start
* Licence text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)

### Opus-MT translation models (Helsinki-NLP/opus-mt-tc-big-en-zle, tc-big-zle-en, tc-big-zle-de, tc-big-de-zle; opus-mt-de-en, en-de, ru-en, en-ru)

* Purpose: Offline machine translation of a book before narration (English, Russian, German; Russian <-> German directly). Opus-MT tc-big models, CC-BY-4.0, about 480 MB per direction; the 2020 Opus-MT models (about 300 MB) for English <-> German and as a fallback. Credit: OPUS-MT, University of Helsinki / Helsinki-NLP - J. Tiedemann, M. Aulamo, D. Bakshandaeva, M. Boggia, S.-A. Grönroos, T. Nieminen, A. Raganato, Y. Scherrer, R. Vázquez, S. Virpioja: "Democratizing neural machine translation with OPUS-MT", Language Resources and Evaluation 58 (2024); Tiedemann and Thottingal (2020). Machine translation quality varies.
* Licence: CC-BY-4.0 / Apache-2.0
* Project: <https://huggingface.co/Helsinki-NLP>
* Status: downloaded on first start
* Licence text: [`licenses/CC-BY-4.0.txt`](licenses/CC-BY-4.0.txt), [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)

## Libraries and programs shipped with Voxprint

### llama.cpp (llama-server, build b11476, Vulkan)

* Purpose: Runs the optional AI text model (pre-built llama-server program, downloaded together with it, about 33 MB)
* Licence: MIT
* Project: <https://github.com/ggml-org/llama.cpp>
* Status: downloaded on first start
* Licence text: [`licenses/llama.cpp.txt`](licenses/llama.cpp.txt)

### SentencePiece

* Purpose: Text tokenizer of the Opus-MT translation models
* Licence: Apache-2.0
* Project: <https://github.com/google/sentencepiece>
* Status: bundled
* Licence text: [`licenses/sentencepiece.txt`](licenses/sentencepiece.txt)

### qwen-tts

* Purpose: Qwen3-TTS model code (speech synthesis and the audio tokenizer)
* Licence: Apache-2.0
* Project: <https://github.com/QwenLM/Qwen3-TTS>
* Status: bundled
* Licence text: [`licenses/qwen-tts.txt`](licenses/qwen-tts.txt)

### qwen-asr

* Purpose: Runs the Qwen3 forced aligner and Qwen3-ASR speech recognition
* Licence: Apache-2.0
* Project: <https://github.com/QwenLM/Qwen3-ASR>
* Status: bundled
* Licence text: [`licenses/qwen-asr.txt`](licenses/qwen-asr.txt)

### qwen-omni-utils

* Purpose: Audio/media helpers required by qwen-asr
* Licence: Apache-2.0
* Project: <https://github.com/QwenLM/Qwen2-VL>
* Status: bundled
* Licence text: [`licenses/qwen-omni-utils.txt`](licenses/qwen-omni-utils.txt)

### PyTorch (torch)

* Purpose: Neural-network engine
* Licence: BSD-3-Clause (plus bundled third-party components)
* Project: <https://pytorch.org>
* Status: bundled
* Licence text: [`licenses/pytorch.txt`](licenses/pytorch.txt)

### torchaudio

* Purpose: Audio processing for PyTorch
* Licence: BSD-2-Clause
* Project: <https://github.com/pytorch/audio>
* Status: bundled
* Licence text: [`licenses/torchaudio.txt`](licenses/torchaudio.txt)

### Transformers

* Purpose: Loads and runs the models
* Licence: Apache-2.0
* Project: <https://github.com/huggingface/transformers>
* Status: bundled
* Licence text: [`licenses/transformers.txt`](licenses/transformers.txt)

### PEFT

* Purpose: LoRA voice adapters: training, saving and merging
* Licence: Apache-2.0
* Project: <https://github.com/huggingface/peft>
* Status: bundled
* Licence text: [`licenses/peft.txt`](licenses/peft.txt)

### Accelerate

* Purpose: Device and memory management
* Licence: Apache-2.0
* Project: <https://github.com/huggingface/accelerate>
* Status: bundled
* Licence text: [`licenses/accelerate.txt`](licenses/accelerate.txt)

### huggingface_hub

* Purpose: Downloads the models
* Licence: Apache-2.0
* Project: <https://github.com/huggingface/huggingface_hub>
* Status: bundled
* Licence text: [`licenses/huggingface_hub.txt`](licenses/huggingface_hub.txt)

### safetensors

* Purpose: Safe model weight files
* Licence: Apache-2.0
* Project: <https://github.com/huggingface/safetensors>
* Status: bundled
* Licence text: [`licenses/safetensors.txt`](licenses/safetensors.txt)

### bitsandbytes

* Purpose: 8-bit optimizer that saves video memory
* Licence: MIT
* Project: <https://github.com/bitsandbytes-foundation/bitsandbytes>
* Status: optional, not bundled
* Licence text: [`licenses/bitsandbytes.txt`](licenses/bitsandbytes.txt)

### PySide6 / Qt for Python

* Purpose: The window and all interface elements
* Licence: LGPL-3.0-only (also available as GPL-2.0/3.0 or commercial; used under LGPL-3.0)
* Project: <https://pyside.org>
* Status: bundled
* Licence text: [`licenses/qt-lgpl-3.0.txt`](licenses/qt-lgpl-3.0.txt), [`licenses/gpl-3.0.txt`](licenses/gpl-3.0.txt)
* Note: Used unmodified and dynamically linked. Prefer the folder-type build (build.bat onedir) so the Qt/PySide6 libraries can be replaced. Source: https://code.qt.io and https://pyside.org.

### NumPy

* Purpose: Numerical arrays
* Licence: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0
* Project: <https://numpy.org>
* Status: bundled
* Licence text: [`licenses/numpy.txt`](licenses/numpy.txt)

### SciPy

* Purpose: Signal processing (filtering, resampling)
* Licence: BSD-3-Clause
* Project: <https://scipy.org>
* Status: bundled
* Licence text: [`licenses/scipy.txt`](licenses/scipy.txt)

### librosa

* Purpose: Audio analysis
* Licence: ISC
* Project: <https://librosa.org>
* Status: bundled
* Licence text: [`licenses/librosa.txt`](licenses/librosa.txt)

### python-soundfile

* Purpose: Reading and writing WAV/FLAC files (uses libsndfile, LGPL, shipped in the wheel)
* Licence: BSD-3-Clause
* Project: <https://github.com/bastibe/python-soundfile>
* Status: bundled
* Licence text: [`licenses/soundfile.txt`](licenses/soundfile.txt)

### pydub

* Purpose: Opening MP3 and other audio formats
* Licence: MIT
* Project: <http://pydub.com>
* Status: bundled
* Licence text: [`licenses/pydub.txt`](licenses/pydub.txt)

### imageio-ffmpeg

* Purpose: Provides the ffmpeg program used to decode audio formats
* Licence: BSD-2-Clause (wrapper)
* Project: <https://github.com/imageio/imageio-ffmpeg>
* Status: bundled
* Licence text: [`licenses/imageio-ffmpeg.txt`](licenses/imageio-ffmpeg.txt)

### FFmpeg (executable inside imageio-ffmpeg)

* Purpose: Audio decoding (separate program, run as a subprocess)
* Licence: GPL-3.0 build (--enable-gpl --enable-version3, verified in the Windows 7.1 binary of imageio-ffmpeg 0.6.0)
* Project: <https://ffmpeg.org>
* Status: bundled
* Licence text: [`licenses/gpl-3.0.txt`](licenses/gpl-3.0.txt)
* Note: Separate executable, not linked into Voxprint. Source code: https://ffmpeg.org/download.html#get-sources (build scripts of the binary: https://github.com/imageio/imageio-binaries). The GPL text is in licenses/gpl-3.0.txt.

### einops

* Purpose: Tensor reshaping used by the models
* Licence: MIT
* Project: <https://github.com/arogozhnikov/einops>
* Status: bundled
* Licence text: [`licenses/einops.txt`](licenses/einops.txt)

### ONNX Runtime

* Purpose: Neural-network runtime required by qwen-tts
* Licence: MIT
* Project: <https://onnxruntime.ai>
* Status: bundled
* Licence text: [`licenses/onnxruntime.txt`](licenses/onnxruntime.txt)

### nagisa

* Purpose: Japanese word segmentation required by qwen-asr
* Licence: MIT
* Project: <https://github.com/taishi-i/nagisa>
* Status: bundled
* Licence text: [`licenses/nagisa.txt`](licenses/nagisa.txt)

### pysox

* Purpose: Python wrapper required by qwen-tts (the SoX program itself is not shipped)
* Licence: BSD-3-Clause
* Project: <https://github.com/rabitt/pysox>
* Status: bundled
* Licence text: [`licenses/sox.txt`](licenses/sox.txt)

### ru-normalizr

* Purpose: Spelling out Russian numbers and abbreviations
* Licence: MIT
* Project: <https://github.com/NickZaitsev/ru-normalizr>
* Status: bundled
* Licence text: [`licenses/ru-normalizr.txt`](licenses/ru-normalizr.txt)

### rutextnorm

* Purpose: Backup Russian text normalization
* Licence: MIT
* Project: <https://github.com/shigabeev/russian_tts_normalization>
* Status: bundled
* Licence text: [`licenses/rutextnorm.txt`](licenses/rutextnorm.txt)

### pymorphy3 (+ pymorphy3-dicts-ru)

* Purpose: Russian morphology (word forms)
* Licence: MIT
* Project: <https://github.com/no-plagiarism/pymorphy3>
* Status: bundled
* Licence text: [`licenses/pymorphy3.txt`](licenses/pymorphy3.txt), [`licenses/pymorphy3-dicts-ru.txt`](licenses/pymorphy3-dicts-ru.txt)

### num2words

* Purpose: Numbers to words
* Licence: LGPL-2.1
* Project: <https://github.com/savoirfairelinux/num2words>
* Status: bundled
* Licence text: [`licenses/num2words.txt`](licenses/num2words.txt)
* Note: Pure-Python library used unmodified; the LGPL allows replacing it with another version (it is a normal Python package).

### roman

* Purpose: Roman numerals
* Licence: ZPL-2.1
* Project: <https://github.com/zopefoundation/roman>
* Status: bundled
* Licence text: [`licenses/roman.txt`](licenses/roman.txt)

### eng_to_ipa

* Purpose: English words in Russian text
* Licence: MIT (per the GitHub repository; the PyPI package has no licence field)
* Project: <https://github.com/mphilli/English-to-IPA>
* Status: bundled
* Licence text: [`licenses/eng_to_ipa.txt`](licenses/eng_to_ipa.txt)

### packaging

* Purpose: Version comparison for updates
* Licence: Apache-2.0 OR BSD-2-Clause
* Project: <https://github.com/pypa/packaging>
* Status: bundled
* Licence text: [`licenses/packaging-apache.txt`](licenses/packaging-apache.txt), [`licenses/packaging-bsd.txt`](licenses/packaging-bsd.txt)

### tqdm

* Purpose: Progress bars
* Licence: MPL-2.0 AND MIT
* Project: <https://tqdm.github.io>
* Status: bundled
* Licence text: [`licenses/tqdm.txt`](licenses/tqdm.txt), [`licenses/MPL-2.0.txt`](licenses/MPL-2.0.txt)

### ctc-forced-aligner

* Purpose: Optional backup aligner; not bundled, installed by the user on request
* Licence: BSD-2-Clause (upstream repository MahmoudAshraf97/ctc-forced-aligner)
* Project: <https://github.com/MahmoudAshraf97/ctc-forced-aligner>
* Status: optional, not bundled
* Licence text: [`licenses/ctc-forced-aligner.txt`](licenses/ctc-forced-aligner.txt)
* Note: The PyPI package of this name is published from a fork whose licence metadata is empty; check it before installing.

## Assets

### Tabler Icons

* Purpose: The fingerprint icon of the application (modified)
* Licence: MIT
* Project: <https://tabler.io/icons>
* Status: bundled
* Licence text: [`licenses/tabler-icons.txt`](licenses/tabler-icons.txt)

### LJ Speech 1.1 (public-domain speech; reader Linda Johnson, LibriVox; compiled by Keith Ito)

* Purpose: Training speech of the optional fully open example voice (not part of the program; the voice is downloaded separately)
* Licence: Public domain (CC0-1.0)
* Project: <https://keithito.com/LJ-Speech-Dataset/>
* Status: optional, not bundled
* Licence text: [`licenses/CC0-1.0.txt`](licenses/CC0-1.0.txt)

## Build tools (not shipped, except the PyInstaller bootloader and the Inno Setup installer runtime)

### FFmpeg LGPL build (BtbN/FFmpeg-Builds, downloaded when needed)

* Purpose: Audio decoding. Used only if no working ffmpeg is installed on the computer; downloaded once and checked by a pinned checksum
* Licence: LGPL-2.1-or-later (no GPL components)
* Project: <https://github.com/BtbN/FFmpeg-Builds>
* Status: downloaded on first start
* Licence text: [`licenses/ffmpeg-lgpl-2.1.txt`](licenses/ffmpeg-lgpl-2.1.txt)
* Note: Separate executable, not linked into Voxprint. Source code: https://ffmpeg.org/download.html#get-sources (build scripts: https://github.com/BtbN/FFmpeg-Builds).

### uv

* Purpose: Installs packages when building (not shipped)
* Licence: MIT OR Apache-2.0
* Project: <https://github.com/astral-sh/uv>
* Status: build time
* Licence text: [`licenses/uv-mit.txt`](licenses/uv-mit.txt), [`licenses/uv-apache.txt`](licenses/uv-apache.txt)

### PyInstaller

* Purpose: Packs the program into an application (its bootloader is embedded; the exception allows distribution under any licence)
* Licence: GPL-2.0-or-later with the bootloader exception
* Project: <https://pyinstaller.org>
* Status: build time
* Licence text: [`licenses/pyinstaller.txt`](licenses/pyinstaller.txt)

### Inno Setup

* Purpose: Creates the installer
* Licence: Inno Setup License (permissive, custom)
* Project: <https://jrsoftware.org/isinfo.php>
* Status: build time
* Licence text: [`licenses/inno-setup.txt`](licenses/inno-setup.txt)

## Important compliance notes

* **Qt / PySide6 (LGPL-3.0).** Qt for Python is used unmodified and is linked dynamically (separate DLL/shared
  files). You may replace these libraries with your own build of PySide6/Qt: use the folder-type build
  (`build.bat onedir`), where the libraries are ordinary files. The LGPL-3.0 and GPL-3.0 texts are in
  `licenses/qt-lgpl-3.0.txt` and `licenses/gpl-3.0.txt`; source code: <https://code.qt.io> and <https://pyside.org>.
* **FFmpeg (GPL-3.0 build).** The ffmpeg program bundled inside the `imageio-ffmpeg` wheel is a GPL build
  (`--enable-gpl --enable-version3`, verified in the Windows 7.1 binary of imageio-ffmpeg 0.6.0). It is a separate
  executable started as a subprocess; it is not linked into Voxprint. Source code:
  <https://ffmpeg.org/download.html#get-sources>; build scripts of the binary:
  <https://github.com/imageio/imageio-binaries>. If you do not want to ship a GPL binary, remove it and ship an
  LGPL build of ffmpeg or require ffmpeg on `PATH`.
* **AAC / M4B export - patents.** The optional M4B (AAC) export exists only as a convenience for Apple Books
  compatibility. The AAC codec is patent-encumbered. Voxprint does not provide a patent licence for it, and this
  project's licence (and the licences listed here) do not grant one. **You are solely responsible for any legal
  compliance** (patent licensing, royalties, distribution rules in your country or for your use) when you choose
  this format. The default formats (Opus, MP3 per chapter, FLAC, WAV) are not affected by this note. You can hide and
  disable the AAC option completely: set the environment variable `VOXPRINT_ENABLE_AAC=0` or put
  `{"aac_m4b": false}` into `%LOCALAPPDATA%\Voxprint\state\features.json`. The encoders (`aac`, `libopus`,
  `libmp3lame`) come from the ffmpeg build in use; their presence in the pinned build is checked at run time, and the
  export stops with a clear message if one is missing.
* **soynlp (GPLv3) is intentionally excluded.** `qwen-asr` imports it only inside the Korean-language branch of its
  forced aligner. Voxprint does not install or bundle it; therefore **Korean alignment is unavailable**.
* **Non-commercial model.** The optional backup aligner model `MahmoudAshraf/mms-300m-1130-forced-aligner` is
  licensed CC-BY-NC-4.0 (non-commercial). It is used only when the user installs the optional `ctc-forced-aligner`
  package. The default Qwen models are Apache-2.0.
* **PyInstaller** is GPL-2.0-or-later with a special exception that allows distributing the embedded bootloader as
  part of programs under any licence (`licenses/pyinstaller.txt`). **Inno Setup** is distributed under its own
  permissive licence (`licenses/inno-setup.txt`).
* Where a licence is stated as verified, it was read from the package metadata (PyPI), the project's repository
  (GitHub) or the model card (Hugging Face) on 2026-10-03. Packages pulled in indirectly are listed in the
  appendix of the build-time version of this file.
