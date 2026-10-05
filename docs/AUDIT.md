# Code audit — 2026-10-05

Short automated pass before the next installer build. Tools: ruff (undefined names), bandit (high), pip-audit, focused pytest.

## Fixed in this pass

- `core/lora_trainer.py`: nested tokenize/encode helpers bound the model via default arguments so a later `del` in `finally` no longer confuses the linter (and stays safe if a callback outlives the `try`).
- `core/asr_dataset.py`, `core/translate.py`: SHA-1 used only as a cache key; marked `usedforsecurity=False` (bandit B324).
- `infra/portable.py`: download helpers bind the model folder as a default so a stalled thread cannot write into the next model's folder.

## Findings left as-is (accepted)

| Tool | Item | Why left |
|------|------|----------|
| bandit B615 | Hugging Face `snapshot_download` without a pinned revision | Downloads already go through our own SHA-256 checks and mirrors; pinning every upstream tip is a separate product decision. |
| pip-audit | `transformers==4.57.6` — several PYSEC advisories; fixes point at 5.x | Jumping to transformers 5.x needs a TTS/Qwen compatibility check first; not bumped in this pass. |
| ruff E501 / UP* | Thousands of style/typing nits | Project historically uses longer lines and `Optional`/`List`; mass auto-fix would drown the real diff. |

## Tests run

- `tests/test_ai_disclosure.py`, `test_workspace.py`, `test_workspace_ui.py`, `test_auto_repair.py`, `test_models_folder.py`, `test_pauses_optional.py`, `test_voice_naming.py`, `test_keep_awake.py`, `test_voice_index_cache.py` — passed (one skipped).

## Product checklist covered by recent commits

1. Shell stdlib + metadata — done
2. Optional pauses (default off) — done
3. Prep-tick / synth progress — done
4. Voice name + folder rename — done
5. My voices Export / Open folder — done
6. Keep-awake — done
7. Book working folder + clean-up dialog — done
8. Installer models-folder page — done
9. Auto-repair — done
10. Voice index cache WinError 32 — done
11. AI disclosure checkbox (default off) — done

Installer rebuild is **not** started; wait for an explicit go-ahead.
