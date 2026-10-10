# Releasing

Published tags stay where they are. `v1.0.0-rc.2` is a published release; do not move that tag and do not rebuild it from this tree.

`BUILD.json` holds `offset` and `codename`. Those two fields, together with the version, are the source of truth for the current build. Change them only when cutting the next release. Build numbers are described in [BUILDING.md](BUILDING.md).

## Codenames

The next codename must be a unique Torah/Tanakh Hebrew word, chosen from the suite codename registry. That registry is kept in private notes and is not copied into this repository.

Before changing `codename`, check it against the names already used in this repository. The check reads `CHANGELOG.md` and `docs/RELEASE-NOTES-*.md`. It fails when `BUILD.json`'s codename belongs to an earlier build, when two builds share one codename, or when one build is given two different spellings. The codename of the newest build in that history may stay in `BUILD.json` until the next release replaces it. A name that does not appear in the history yet is free as far as this repository is concerned; it still has to be free in the suite codename registry.

```
python tools/codenames.py
```

The same check runs in the unit tests (`tests/test_codenames.py`).
