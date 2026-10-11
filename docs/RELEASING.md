# Releasing

Published tags stay where they are. `v1.0.0-rc.2` is a published release; do not move that tag and do not rebuild it from this tree.

`BUILD.json` holds `build`, `codename`, and `offset`. The number for this RC is the explicit `build` field (**1002**). A release tag wins over that field when the tag ends in `+<number>` or `-build<number>` (`v1.0.0-rc.4+1005`, `v1.0.0-rc.4-build1002`). A version-only tag such as `v1.0.0-rc.3` does not carry a number. `VOXPRINT_BUILD` overrides both. `offset` is history only: it is not added to `GITHUB_RUN_NUMBER`, and a failed CI run does not consume a number. Change `build` only when cutting the next release. The codename is a separate edit and is not set by the number. Details: [BUILDING.md](BUILDING.md).

## Codenames

The next codename must be a unique Torah/Tanakh Hebrew word, chosen from the suite codename registry. That registry is kept in private notes and is not copied into this repository.

Before changing `codename`, check it against the names already used in this repository. The check reads `CHANGELOG.md` and `docs/RELEASE-NOTES-*.md`. It fails when `BUILD.json`'s codename belongs to an earlier build, when two builds share one codename, or when one build is given two different spellings. The codename of the newest build in that history may stay in `BUILD.json` until the next release replaces it. A name that does not appear in the history yet is free as far as this repository is concerned; it still has to be free in the suite codename registry.

```
python tools/codenames.py
```

The same check runs in the unit tests (`tests/test_codenames.py`).
