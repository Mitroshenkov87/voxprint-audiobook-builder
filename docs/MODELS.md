# Models, mirrors and backups

Where Voxprint gets its models, how existing copies are reused, the download mirrors and how to back up. Licences of the models: [LICENSES.md](LICENSES.md).

## Reusing what is already on your computer
Before downloading a Qwen model Voxprint looks for a complete copy left by another app and uses it **in place - read-only, nothing
is copied, moved, changed or deleted** (`core/model_locator.py`). Where it looks (confidence: high = verified in code/docs, medium = derived, not checked on a real install):

| Source | Location | Confidence |
|---|---|---|
| Standard Hugging Face cache (also what **Alexandria** uses) | `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub` (Windows: `%USERPROFILE%\.cache\huggingface\hub`); layout `models--Qwen--<Name>/{blobs,refs/main,snapshots/<sha>}` | high |
| Alexandria installed via **Pinokio** (Pinokio sets `HF_HOME=./cache/HF_HOME`) | `<pinokio>\api\alexandria-audiobook.git\cache\HF_HOME\hub` (folder name inferred, so `api\*\cache\...` is scanned), or `<pinokio>\cache\HF_HOME\hub` | medium |
| Alexandria in Docker | volume inside Docker - not reachable from the host (use `VOXPRINT_MODEL_DIRS` with a bind mount) | high |
| ModelScope cache | `%MODELSCOPE_CACHE%` or `~/.cache/modelscope`, then `hub/models/<Owner>/<Name>` (dots in the name become `___`) | medium |
| Manual downloads (`--local_dir`) | `~/models`, `./models`, the working folder: `<Name>`, `<Owner>/<Name>`, `<Owner>--<Name>`; a previous Voxprint install | medium |
| Your own list | `VOXPRINT_MODEL_DIRS` (paths separated by `;` on Windows) - searched first. `VOXPRINT_NO_EXTERNAL_MODELS=1` switches reuse off | - |

A copy is accepted only if it is **complete** (valid `config.json`, every file of the repository incl. `speech_tokenizer/`, consistent and non-truncated
`*.safetensors`, all shards, resolvable symlinks, no `*.incomplete`) and matches the **verified revision** from `infra/verified_manifest.json`
(HF snapshot folder name = commit sha; folders without a sha are accepted only when the weight-file sizes equal the verified commit).
On a mismatch the verified revision is downloaded. The updater never touches reused copies.

**Download mirror.** If Hugging Face is slow or unreachable (6 s probe; skipped when you set your own `HF_ENDPOINT`) or fails, Voxprint downloads from
**ModelScope** (modelscope.cn, org `Qwen`; same repository ids, byte-identical file sizes - verified 2026-10-03) and confirms the revision by file sizes.
Downloads resume after an interruption. `VOXPRINT_NO_MIRROR=1` disables the mirror.

**Backup mirrors on Hugging Face.** As the very last source, models with permissive licences are also kept under `Mitroshenkov87/voxprint-mirror-*` (Qwen3 TTS / ASR / aligner, SAGE, and the four Opus-MT translation models `opus-mt-ru-en`, `en-ru`, `de-en`, `en-de` by Helsinki-NLP, CC-BY-4.0 / Apache-2.0). Order: models folder -> original repository -> mirror. The files are byte-identical to the pinned commit of the original, every file is checked against the SHA-256 in `infra/model_mirrors.json` (a mismatch deletes it), and each mirror's model card names the original, the licence and the commit. `VOXPRINT_NO_HF_MIRROR=1` disables it.

**Components.** The same idea applies to Python packages and tools (`infra/env_probe.py`, read-only): installed and current / proven compatible -> reused;
missing -> the newest verified version is installed into **Voxprint's own environment** (its venv / `packages/` overlay), which is auto-updated.
An *outdated component in your own environment* (system Python, Pinokio, conda) is **never touched silently**: a one-click prompt offers to upgrade it
and says exactly what changes. *Upgrade* -> pip upgrades it there (smoke-tested, rolled back on failure); *Not now* and still compatible -> used as is;
*Not now* and too old -> Voxprint uses its own copy and your environment stays untouched. torch: any build is reusable, but its CUDA flavor must fit the driver.
ffmpeg: a system one is used only if `ffmpeg -version` works, otherwise a pinned LGPL build (sha256, staging, smoke test, atomic swap, rollback).
**Unsloth is deliberately not used**: as of 2026-10-03 it has no Qwen3-TTS fine-tuning support and its dependency pins conflict with the verified set;
Voxprint trains with its own LoRA loop.

## Backup, restore and existing models
**Settings (gear) -> Models and voices.**
* **Back up models and voices...** - pick any folder or drive (external disk, NAS share). Voxprint copies to `<folder>\Voxprint-backup\`: every complete model it uses (its own models folder plus complete copies from the Hugging Face cache of other
  programs: TTS bases, forced aligner, clean-up models), the pinned ffmpeg build if it was downloaded, and - with *Include my voices* ticked (default) - the voice library. Before anything is written you see the size and the free space of the target;
  if it will not fit, you get a clear message ("needs X, Y free") and **nothing** is copied. Progress and Cancel are shown in the dialog.
* **Resumable, skips identical files.** Each file is written as `name.part` and renamed when complete; a cancelled / crashed run loses at most that file and the next run continues. A file is skipped when the size and modification time match
  (or, if the time differs, the SHA-256 matches). `voxprint-backup.json` (the manifest) lists every file with size, SHA-256 and time; an interrupted run never removes items the manifest already vouched for. The backup is plain files: browse it, zip it, copy it by hand.
* **Restore from backup...** - pick the folder that contains `Voxprint-backup` (or that folder itself). Every file is verified against the manifest hash (the copy is read back); models are staged in `<name>.restoring` and swapped in only when complete, so a half-restored model is
  never used. A model that differs from the backup is replaced after the new copy is verified; a **voice that exists with different content is never overwritten** (it is reported). Files already identical are skipped.
* **Existing models folder** - name a folder with models from a previous install (an old `...\Voxprint\models` folder, a Voxprint backup, a Hugging Face cache). Before downloading anything, Voxprint **imports** a complete copy from there
  (`infra/existing_models.py`, ahead of the read-only reuse above): a **hard link** if the folder is on the same drive (instant, no extra space), otherwise a **copy**; verified by SHA-256 (against the backup manifest when there is one, else by reading the copy back);
  the same completeness and revision rules as for any foreign copy apply, and a failed or corrupt import falls back to the normal download. The source folder is never modified. Storage: `state\existing_models_dir.txt` (override: `VOXPRINT_EXISTING_MODELS`);
  `VOXPRINT_NO_EXTERNAL_MODELS` does not switch off this explicitly chosen folder. Choosing a folder in Settings offers to import right away (nothing is downloaded); the installer page sets the same file and the first start imports.
Limits: resume works per file (a half-copied multi-GB file starts over); not verified on real removable media / NTFS hard links / a NAS (tests use temporary folders and fake disk usage).
