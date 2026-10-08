# Voxprint for Linux (experimental) - test checklist

Short manual test for a real Linux laptop (Ubuntu / Debian / AnduinOS). The CI already checks the install, the unit tests
and a headless start on a clean Ubuntu 24.04 runner; this list covers what CI cannot: the real desktop, audio and the GPU.
Mark each line OK / FAIL and note the message. Logs: `~/.local/share/voxprint/logs/`.

## 1. Install
- [ ] `curl -fsSL https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/v0.1.0-beta/install-voxprint-linux.sh -o install-voxprint-linux.sh`
- [ ] `sha256sum install-voxprint-linux.sh` equals the hash in the release notes / `SHA256SUMS-linux.txt`
- [ ] `bash install-voxprint-linux.sh --check` lists the missing system packages (or says OK)
- [ ] `bash install-voxprint-linux.sh --install-deps` finishes; the last lines are `SELFTEST_IMPORTS OK`, `SELFTEST_TEXT OK` and "Voxprint is installed"
- [ ] With an NVIDIA GPU: `~/.local/share/voxprint/install.json` shows a `cu...` backend, not `cpu`; `voxprint --selftest-imports` prints `cuda available: True`
- [ ] Without an NVIDIA GPU / with `--cpu`: the CPU build is installed (`cuda available: False`) - expected

## 2. Start
- [ ] Voxprint appears in the application menu with the icon; it starts from there
- [ ] `voxprint` in a terminal starts it (if the command is not found: log out and in, or run `~/.local/bin/voxprint`)
- [ ] The Studio window shows the text readably (fonts), buttons react, language switch (en / ru / de) works
- [ ] Wayland session: the window opens. If not or the text is blurry: `QT_QPA_PLATFORM=xcb voxprint` (note which one works)
- [ ] Closing the window ends the process (`pgrep -f voxprint/app/main.py` is empty)

## 3. First run and models
- [ ] The first-run model download starts (about 5-8 GB), shows progress, can be cancelled and resumed
- [ ] Models land in `~/.local/share/voxprint/models`; "open folder" buttons open the file manager

## 4. Narration (the main function)
- [ ] Narrate: choose a small `.txt` / `.epub`, a voice (download one under "Voices" or train one) and start
- [ ] Preview playback works (sound through PipeWire / PulseAudio)
- [ ] The result (mp3 / m4b) is written under `~/.local/share/voxprint/Projects/Audiobooks/` and plays in an ordinary player
- [ ] Translation card: a Russian text narrated as English downloads the Opus-MT model and produces the translated text file
- [ ] On a GPU: speed is clearly better than CPU; on CPU it works, only slowly (a short sentence is enough)

## 5. Update and removal
- [ ] Running the install script again updates the program and keeps models and voices
- [ ] `bash install-voxprint-linux.sh --uninstall` removes launcher, menu entry, program and environment but keeps `~/.local/share/voxprint/models` and `voices`

## Report back
For every FAIL: the step, what you saw, and the files `~/.local/share/voxprint/logs/*.txt`, `~/.local/share/voxprint/install.json`,
the output of `lsb_release -d`, `echo $XDG_SESSION_TYPE`, `nvidia-smi | head -4`.
