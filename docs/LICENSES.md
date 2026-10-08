# Licences and third-party components

Voxprint stands on open-source software; the single source of truth is `credits.json`. From it come the **About** list, `THIRD_PARTY_NOTICES.md`
(shipped by the installer together with the `licenses\` folder of full licence texts) and the build-time appendix with the licences of all installed
packages (`tools\gen_notices.py --with-installed`, called by `build.bat`). Voxprint's *own* source code is under the **Apache License 2.0** (see "Licence of Voxprint" below). Important points:
* **Qt / PySide6 - LGPL-3.0.** Used unmodified and dynamically linked. To let users replace the Qt/PySide6 libraries, prefer `build.bat onedir`
  (the libraries are ordinary files); `--onefile` makes this harder. The LGPL/GPL texts and the source pointers (<https://code.qt.io>, <https://pyside.org>) are included.
* **FFmpeg is a GPL-3.0 build.** The ffmpeg binary inside the `imageio-ffmpeg` wheel was built with `--enable-gpl --enable-version3` (verified in the Windows 7.1 binary of
  imageio-ffmpeg 0.6.0). It is a separate executable started as a subprocess, but you are distributing a GPL binary: keep `licenses\gpl-3.0.txt` and the source link,
  or ship an LGPL ffmpeg build / require ffmpeg on `PATH`.
* **AAC (M4B) is patent-encumbered.** Voxprint provides no patent licence for it; the option is a convenience for Apple Books, is off by default, carries a disclaimer and can be disabled completely (see
  [AAC / M4B](AAC-M4B.md)). The same notice is in `THIRD_PARTY_NOTICES.md`.
* **soynlp (GPLv3) is excluded.** `qwen-asr` imports it lazily, only for Korean. It is not in `requirements.txt` and the build excludes it (`--exclude-module soynlp`),
  so **Korean alignment is unavailable**.
* **CC-BY-NC-4.0 model.** The optional backup aligner model `MahmoudAshraf/mms-300m-1130-forced-aligner` (used only if you install the optional `ctc-forced-aligner`) is
  non-commercial. The default Qwen models and libraries are Apache-2.0 / MIT / BSD-style.
* PyInstaller is GPL-2.0-or-later **with a bootloader exception** (apps may use any licence); Inno Setup has its own permissive licence.

### Licence of Voxprint
* **Source code: Apache License 2.0** - Copyright 2026 Aleksandr Mitroshenkov. Full text in [`LICENSE`](../LICENSE), attribution in [`NOTICE`](../NOTICE).
  You may use, modify and redistribute the code, also commercially, under the terms of that licence (keep the notices; it includes a patent grant). *(The choice may still change to MIT before the first public release.)*
* **Models and voices have their own licences** - the Apache licence of the code does *not* cover them:
  * the models Voxprint downloads (Qwen3-TTS, Qwen3-ASR, the optional text models, and the optional OpenVoice V2 converter, MIT) keep the licences of their authors;
  * every **voice** carries its own licence in `voice.json` (see *My voices and licences*). Voices you train are `custom/personal-only` until you decide otherwise;
  * voice packages are not part of this repository or the installer; a voice marked **"test use only"** (`custom/test-use-only`) may be used to try the program only: do not publish audio made with it and do not use it commercially.
* Third-party components: `THIRD_PARTY_NOTICES.md` and `licenses\`.

## Licence of Voxprint
* **Source code: Apache License 2.0** - Copyright 2026 Aleksandr Mitroshenkov. Full text in [`LICENSE`](../LICENSE), attribution in [`NOTICE`](../NOTICE).
  You may use, modify and redistribute the code, also commercially, under the terms of that licence (keep the notices; it includes a patent grant). *(The choice may still change to MIT before the first public release.)*
* **Models and voices have their own licences** - the Apache licence of the code does *not* cover them:
  * the models Voxprint downloads (Qwen3-TTS, Qwen3-ASR, the optional text models, and the optional OpenVoice V2 converter, MIT) keep the licences of their authors;
  * every **voice** carries its own licence in `voice.json` (see *My voices and licences*). Voices you train are `custom/personal-only` until you decide otherwise;
  * voice packages are not part of this repository or the installer; a voice marked **"test use only"** (`custom/test-use-only`) may be used to try the program only: do not publish audio made with it and do not use it commercially.
* Third-party components: `THIRD_PARTY_NOTICES.md` and `licenses\`.

### Terms of use
The End User Agreement of this program is [`legal/EULA-audiobook-builder.md`](legal/EULA-audiobook-builder.md) (shown by the installer as plain text). It is built from [`legal/EULA-TEMPLATE.md`](legal/EULA-TEMPLATE.md), the common template of all Voxprint programs; a draft for the future Movie Dubber is [`legal/EULA-movie-dubber-DRAFT.md`](legal/EULA-movie-dubber-DRAFT.md). **Not legal advice; the author is not a lawyer.**
