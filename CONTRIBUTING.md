# Contributing to Voxprint

Thanks for helping! Small, focused pull requests are the easiest to review.

1. **Fork** the repository and create a branch; one topic per PR.
2. **Set up and run the tests** (no GPU, no network needed): `QT_QPA_PLATFORM=offscreen python -m pytest` - details in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Add a test with every behaviour change; keep the suite green.
3. **Open a PR**: say what changed and how you tested it; add an English screenshot for UI changes; note user-visible changes in `CHANGELOG.md` ("Unreleased").
4. **Rules:** voices only with the owner's consent (never commit other people's recordings or voices); no secrets or tokens; UI texts in `locales/{en,ru,de}.json`; keep Windows and Linux working; keep the README short (details go to `docs/`).
5. **Licence:** the code is **Apache-2.0** ([`LICENSE`](LICENSE)); contributions are accepted under the same licence. The End User Agreement of the program is in [docs/legal/](docs/legal/EULA-audiobook-builder.md).

Bugs: open an issue with your OS, GPU/driver, what you did and the log lines (see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#reporting-bugs)). Full guide for contributors and agents: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md), [AGENTS.md](AGENTS.md).
