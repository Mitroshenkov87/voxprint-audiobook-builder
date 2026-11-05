# Feature map

One row per old-UI feature and per CLI feature. Status is `mapped`, `CLI-only`, or `dropped`. Nothing in this table is `dropped`.

The main screen is section Narrate: Choose a book, a full-width voice, and chips for pace, voices, format, split into chapters, background music, and reading manner, then Create audiobook. Those chips and the expert panel share one settings state. Language, text preparation, sound quality, the AI label, and the fine technical parameters stay expert fields. Voice training is three numbered groups on the train window. The Steps wizard is still missing. Quick preview, compare, checks, voice details, and Build a voice for other apps sit under More options.

## Missing shell pieces

Pinned tag `v0.2.5` (Yitro, commit `8a94b531cb4f3526425a80a86a1e64624b5de606`) includes Sections (a tab row), Steps (a wizard), `set_voices`, full `input_path`, Pause and resume, a player with a chapter list, a light theme, and `ui.geometry`. UI freeze (Aleksandr, 15:15): do not wire new polish yet. Still not in this pin:

| Piece | Designed use |
| --- | --- |
| `set_voices` groups | Male and female groups in the voice list |
| Voice preview | A short preview button on a voice row |
| Download-on-pick badge | Mark voices that download the first time they are chosen |

Until Sections and Steps are wired in this app, the frozen windows (`ui/main_window.py`, `ui/voices_window.py`, `ui/narrate_window.py`, `ui/revoice_window.py`, `ui/settings_dialog.py`) still implement those rows. The shell Settings command opens that settings dialog. Voice-consent sentences stay the plain `consent.*` text. Use a flat voice list for now.

## Old UI

| Feature | Where it was | Where it lives now | Status |
| --- | --- | --- | --- |
| Studio home | Studio cards | Section: Narrate (the window opens here) | mapped |
| Narrate a book | Studio card, Narrate window | Section: Narrate | mapped |
| Train your voice | Studio card, Train window | Steps: Train voice | mapped |
| My voices | Studio card, My voices | Section: Voices | mapped |
| Re-voice | Studio card, Re-voice window | Section: Re-voice | mapped |
| Choose a book (TXT, Markdown, FB2, EPUB, .vxbook) | Narrate, Choose book | Section: Narrate, drop “Choose a book” | mapped |
| Narrator voice | Narrate voice combo | Section: Narrate, full-width voice dropdown | mapped |
| Download a catalog voice when narration starts | Narrate voice list | Section: Voices, action Download | mapped |
| Start narration | Narrate, Start | Section: Narrate, action Create audiobook | mapped |
| Pause narration | Narrate, Pause | Section: Narrate, action Pause | mapped |
| Resume narration | Narrate, Resume | Section: Narrate, action Resume | mapped |
| Cancel narration | Narrate, Cancel | Section: Narrate, action Cancel | mapped |
| Progress and time left | Narrate progress | Section: Narrate, progress line | mapped |
| Listen to finished parts | Mini player | Section: Narrate, player | mapped |
| Chapter list while listening | Narrate player hint | Section: Narrate, player chapters | mapped |
| Read chapter titles aloud | Narrate checkbox | Expert field: Read chapter titles | mapped |
| One voice or each speaker | Narrate speakers switch | Section: Narrate, chip Voices (one voice, or different voices by role) | mapped |
| Male voice | Narrate speaker card | Section: Roles, male voice | mapped |
| Female voice | Narrate speaker card | Section: Roles, female voice | mapped |
| Second male voice | Narrate speaker card | Section: Roles, second male voice | mapped |
| Second female voice | Narrate speaker card | Section: Roles, second female voice | mapped |
| Pin a character to a voice | Narrate character row | Section: Roles, character pin | mapped |
| Speaker marks preview | Narrate, preview marks | Section: Roles, action Preview marks | mapped |
| Edit speaker-mark prompt files | Narrate, edit prompts | Section: Roles, action Edit prompts | mapped |
| .vxbook roles and cast | Narrate opens the archive | Section: Roles, book cast | mapped |
| Keep an author stress mark (U+0301) | Prepare text leaves it in place | Expert field: Prepare text (no stress step is added; the speech model does not read marks) | mapped |
| Prepare text | Narrate, Prepare the text | Expert field: Prepare the text | mapped |
| Prepare step: layout | Narrate prepare | Expert field: Fix the layout | mapped |
| Prepare step: page noise | Narrate prepare | Expert field: Remove page noise | mapped |
| Prepare step: quotes | Narrate prepare | Expert field: Fix quotes | mapped |
| Prepare step: links | Narrate prepare | Expert field: Read links as words | mapped |
| Prepare step: headings | Narrate prepare | Expert field: Read headings | mapped |
| Prepare step: numbers | Narrate prepare | Expert field: Read numbers | mapped |
| Prepare step: abbreviations | Narrate prepare | Expert field: Expand abbreviations | mapped |
| Prepare step: letter yo | Narrate prepare | Expert field: Restore the letter yo | mapped |
| Russian typo model | Narrate prepare path | Expert field: Typo cleanup | mapped |
| AI text preparation (markup) | Narrate, AI text model | Expert field: AI text preparation | mapped |
| Sample of the prepared text | Narrate sample pane | Section: Narrate, prepared-text sample | mapped |
| Translate, or keep the book’s language | Narrate translate row | Expert field: Language | mapped |
| Literary translation | Narrate checkbox | Expert field: Literary translation | mapped |
| Download translation models | Narrate button | Action: Manage models | mapped |
| Download the text model | Narrate button | Action: Manage models | mapped |
| Spoken AI disclosure | Narrate checkbox | Expert field: Say this is AI | mapped |
| One Opus file | Narrate formats | Chip Format Opus with Split into chapters Off; expert Audiobook file follows | mapped |
| MP3 chapters | Narrate formats | Chip Format MP3 with Split into chapters On; expert Audiobook file follows | mapped |
| One MP3 file | Narrate formats | Chip Format MP3 with Split into chapters Off; expert Audiobook file follows | mapped |
| Apple Books (M4B) | Narrate formats | Chip Format “M4B audiobook with chapters”; one AAC file with ffmpeg chapter markers (`FORMAT_M4B`) | mapped |
| Opus chapters | Narrate, other formats | Chip Format Opus with Split into chapters On; expert Audiobook file follows | mapped |
| FLAC chapters | Narrate, other formats | Expert field: Audiobook file | mapped |
| WAV chapters | Narrate, other formats | Expert field: Audiobook file | mapped |
| Sound quality Compact, Standard, High | Narrate quality | Expert field: Opus bitrate, MP3 bitrate, AAC bitrate (Standard is the live default) | mapped |
| Opus, MP3, and AAC bitrates | Narrate advanced rates | Expert field: Opus bitrate, MP3 bitrate, AAC bitrate | mapped |
| Fragment check | Narrate checkbox | Expert field: Fragment check | mapped |
| Background sound for a .vxbook | Settings soundscape | Chip Background music. On writes “when the book asks”; Off writes off. Expert Background sound follows | mapped |
| Reading style | Settings and Narrate | Chip Reading manner. Calm writes scripture, Expressive writes fiction. Automatic, dialogue, speed, and pauses stay expert fields | mapped |
| Reading speed | Settings | Expert field: Speed | mapped |
| Pause after a comma | Settings | Expert field: Comma pause | mapped |
| Pause mid-sentence | Settings | Expert field: Mid-sentence pause | mapped |
| Pause after a sentence | Settings | Expert field: Sentence pause | mapped |
| Pause after a paragraph | Settings | Expert field: Paragraph pause | mapped |
| Pause after a chapter | Settings | Expert field: Chapter pause | mapped |
| Reset narration defaults | Settings button | Expert field: Reset on Pace or Voices | mapped |
| Read ordinals | Settings checkbox | Expert field: Read ordinals | mapped |
| Working folder | Narrate | Expert field: Working folder | mapped |
| Speech model | Narrate model row | Expert field: Speech model | mapped |
| Batch size | Narrate / GPU policy | Expert field: Batch size | mapped |
| Graphics memory tier | Narrate / GPU policy | Expert field: Memory | mapped |
| Thermal pause | GPU policy | Expert field: Pause when the computer gets hot | mapped |
| Pace Calm, Natural, Brisk | Narrate reading style and speed | Section: Narrate, chip Pace | mapped |
| Recording plus a transcript | Train, audio and text | Step 1: The recording | mapped |
| No transcript (recognise the speech) | Train checkbox and warning | Step 1: The recording, no transcript | mapped |
| Folder of clips | Train folder button | Step 1: The recording, clip folder | mapped |
| Voice name | Train field | Step 3: Consent and name | mapped |
| Voice type (male, female, child, other) | Train combo | More options | mapped |
| Age group | Train, more about the voice | More options | mapped |
| Speaker, organisation, prepared-by, description, link | Train, more about the voice | More options | mapped |
| Preset Fast | Train presets | Step 2: How to train | mapped |
| Preset Balanced | Train presets | Step 2: How to train | mapped |
| Preset Maximum | Train presets | Step 2: How to train | mapped |
| Preset Manual (epochs, rank, alpha, learning rate, accumulation) | Train advanced | Step 2 quality choice; the numbers open under More options | mapped |
| Time estimate | Train label | Step 2: Quality | mapped |
| Clip length | Train control | More options | mapped |
| Quick preview | Train, Quick preview | More options, action Quick preview | mapped |
| Compare two preview variants | Train preview | More options, action Compare | mapped |
| Use these preview settings | Train preview | More options, action Use these settings | mapped |
| Check the voice after training | Train checkbox | More options | mapped |
| Consent read from the spoken statement | Train consent card | Step 3: Consent and save | mapped |
| Consent chosen by hand | Train consent card | Step 3: Consent and save | mapped |
| No statement, private use only | Train consent card | Step 3: Consent and save | mapped |
| Consent scopes (commercial, public non-commercial, private only) | Train consent card | Step 3: Consent and save | mapped |
| Keep the spoken consent clip | Train checkbox | Step 3: Consent and save | mapped |
| Create dataset only | Train button | More options, action Create dataset | mapped |
| Create voice (LoRA) | Train button | Step 3: Consent and name, action Train the voice | mapped |
| Build the universal model | Train button; My voices export | More options, action Build a voice for other apps | mapped |
| Open the result folder | Train button | Step 3: Consent and save, action Open folder | mapped |
| Preview a voice | My voices card | Section: Voices, action Preview | mapped |
| Narrate with this voice | My voices card | Section: Narrate, voice taken from Voices | mapped |
| Edit voice details and licence | My voices card | Section: Voices, action Edit | mapped |
| Export a voice zip | My voices card | Section: Voices, action Export | mapped |
| Open a voice folder | My voices card | Section: Voices, action Open folder | mapped |
| Delete a voice | My voices card | Section: Voices, action Delete | mapped |
| Import a voice folder | My voices | Section: Voices, action Import folder | mapped |
| Import a voice zip | My voices | Section: Voices, action Import zip | mapped |
| Voice catalog | My voices | Section: Voices, catalog list | mapped |
| Download a catalog voice | My voices | Section: Voices, action Download | mapped |
| Licence badge on a voice | My voices card | Section: Voices, licence line | mapped |
| Record from the microphone | Re-voice | Section: Re-voice, action Record | mapped |
| Play a take | Re-voice | Section: Re-voice, action Play | mapped |
| Add and remove audio files | Re-voice | Section: Re-voice | mapped |
| Recognition language | Re-voice advanced | Section: Re-voice, language | mapped |
| Transcribe to a text book | Re-voice | Section: Re-voice, action To text | mapped |
| Direct re-voice with a library voice | Re-voice | Section: Re-voice, action Direct | mapped |
| Download the recognition model | Re-voice button | Action: Manage models | mapped |
| Edit the recognised text | Re-voice text | Section: Re-voice, text | mapped |
| Hand the text to narration | Re-voice button | Section: Narrate, book set from Re-voice | mapped |
| UI language | Settings | Action: Settings, language | mapped |
| Window transparency | Settings | Action: Settings, transparency | mapped |
| Network interface | Settings | Action: Settings, network interface | mapped |
| Check for updates | Settings | Action: Settings, Check for updates | mapped |
| Open the models folder | Settings | Action: Settings, Open models folder | mapped |
| Open the data and log folder | Settings | Action: Settings, Open data folder | mapped |
| About and third-party notices | Settings, About | Action: Settings, About | mapped |
| Check and repair | Settings | Action: Settings, Check and repair | mapped |
| Save a diagnostic zip | Settings | Action: Settings, Diagnostics | mapped |
| Preload models | Settings | Action: Settings, Preload | mapped |
| Speech-recognition model choice | Settings | Action: Settings, recognition model | mapped |
| Optional runtime components | Settings, Components | Action: Manage models | mapped |
| Manage models | Settings, MDL | Action: Manage models | mapped |
| Projects folder: open, change, default | Settings | Action: Settings, projects folder | mapped |
| Back up models and voices | Settings | Action: Settings, Back up | mapped |
| Leave models or voices out of a backup | Settings checkboxes | Action: Settings, Back up | mapped |
| Restore a backup | Settings | Action: Settings, Restore | mapped |
| Use models from a folder without copying | Settings checkbox | Action: Settings, link models | mapped |
| Import an existing models folder | Settings | Action: Settings, Import models | mapped |
| Privacy note | Studio footer and first-run note | Action: Settings, About | mapped |
| Splash and the six-click picture | Startup | Unchanged startup splash | mapped |
| Repair the install before the window | Startup | Unchanged, before the shell | mapped |
| Block launch until required models are ready | Model dialog | Action: Manage models, then the unavailable line | mapped |

## CLI

| Feature | Where it was | Where it lives now | Status |
| --- | --- | --- | --- |
| `narrate` | CLI `narrate` | unchanged CLI | CLI-only |
| `narrate` book (TXT, Markdown, FB2, EPUB, .vxbook) | CLI positional | unchanged CLI | CLI-only |
| `narrate --voice` | CLI flag | unchanged CLI | CLI-only |
| `narrate --out` | CLI flag | unchanged CLI | CLI-only |
| `narrate --format` | CLI flag | unchanged CLI | CLI-only |
| `narrate --pauses` | CLI flag | unchanged CLI | CLI-only |
| `narrate --no-pauses` | CLI flag | unchanged CLI | CLI-only |
| `narrate --pause-comma` | CLI flag | unchanged CLI | CLI-only |
| `narrate --pause-mid` | CLI flag | unchanged CLI | CLI-only |
| `narrate --pause-sentence` | CLI flag | unchanged CLI | CLI-only |
| `narrate --pause-paragraph` | CLI flag | unchanged CLI | CLI-only |
| `narrate --pause-chapter` | CLI flag | unchanged CLI | CLI-only |
| `narrate --speed` | CLI flag | unchanged CLI | CLI-only |
| `narrate --style` | CLI flag | unchanged CLI | CLI-only |
| `narrate --ordinals` | CLI flag | unchanged CLI | CLI-only |
| `narrate --no-ordinals` | CLI flag | unchanged CLI | CLI-only |
| `narrate --yo` | CLI flag | unchanged CLI | CLI-only |
| `narrate --no-yo` | CLI flag | unchanged CLI | CLI-only |
| `narrate --ai-disclosure` | CLI flag | unchanged CLI | CLI-only |
| `narrate --speakers` | CLI flag | unchanged CLI | CLI-only |
| `narrate --male-voice` | CLI flag | unchanged CLI | CLI-only |
| `narrate --female-voice` | CLI flag | unchanged CLI | CLI-only |
| `narrate --male2-voice` | CLI flag | unchanged CLI | CLI-only |
| `narrate --female2-voice` | CLI flag | unchanged CLI | CLI-only |
| `narrate --character` | CLI flag | unchanged CLI | CLI-only |
| `narrate --speaker-marks` | CLI flag | unchanged CLI | CLI-only |
| `narrate --no-soundscape` | CLI flag | unchanged CLI | CLI-only |
| `narrate --work-dir` | CLI flag | unchanged CLI | CLI-only |
| `train` | CLI `train` | unchanged CLI | CLI-only |
| `train` audio or clip folder | CLI positional | unchanged CLI | CLI-only |
| `train --text` | CLI flag | unchanged CLI | CLI-only |
| `train --name` | CLI flag | unchanged CLI | CLI-only |
| `train --type` | CLI flag | unchanged CLI | CLI-only |
| `train --out` | CLI flag | unchanged CLI | CLI-only |
| `train --license` | CLI flag | unchanged CLI | CLI-only |
| `train --consent` | CLI flag | unchanged CLI | CLI-only |
| `train --speaker` | CLI flag | unchanged CLI | CLI-only |
| `train --language` | CLI flag | unchanged CLI | CLI-only |
| `voices list` | CLI `voices list` | unchanged CLI | CLI-only |
| `voices export` | CLI `voices export` | unchanged CLI | CLI-only |
| `voices catalog` | CLI `voices catalog` | unchanged CLI | CLI-only |
| `voices download` | CLI `voices download` | unchanged CLI | CLI-only |
| `diag` | CLI `diag` | unchanged CLI | CLI-only |
| `diag --out` | CLI flag | unchanged CLI | CLI-only |
| `status` | CLI `status` | unchanged CLI | CLI-only |
| `capabilities` | CLI alias of `status` | unchanged CLI | CLI-only |
| `backup` | CLI `backup` | unchanged CLI | CLI-only |
| `backup --out` | CLI flag | unchanged CLI | CLI-only |
| `backup --no-models` | CLI flag | unchanged CLI | CLI-only |
| `backup --no-voices` | CLI flag | unchanged CLI | CLI-only |
| `restore` | CLI `restore` | unchanged CLI | CLI-only |
| `restore --from` | CLI flag | unchanged CLI | CLI-only |
| `restore --link` | CLI flag | unchanged CLI | CLI-only |
| `models list` | CLI `models list` | unchanged CLI | CLI-only |
| `models download` | CLI `models download` | unchanged CLI | CLI-only |
| `revoice` | CLI `revoice` | unchanged CLI | CLI-only |
| `revoice` audio | CLI positional | unchanged CLI | CLI-only |
| `revoice --out` | CLI flag | unchanged CLI | CLI-only |
| `revoice --title` | CLI flag | unchanged CLI | CLI-only |
| `revoice --language` | CLI flag | unchanged CLI | CLI-only |
| `speakers` | CLI `speakers` | unchanged CLI | CLI-only |
| `speakers --out` | CLI flag | unchanged CLI | CLI-only |
| `check` | CLI `check` | unchanged CLI | CLI-only |
| `repair` | CLI alias of `check` | unchanged CLI | CLI-only |
| `prepare` | CLI `prepare` | unchanged CLI | CLI-only |
| `prepare --out` | CLI flag | unchanged CLI | CLI-only |
| `prepare --report` | CLI flag | unchanged CLI | CLI-only |
| `prepare --language` | CLI flag | unchanged CLI | CLI-only |
| `prepare --steps` | CLI flag | unchanged CLI | CLI-only |
| `prepare --no-rules` | CLI flag | unchanged CLI | CLI-only |
| `prepare --yo` | CLI flag | unchanged CLI | CLI-only |
| `prepare --no-yo` | CLI flag | unchanged CLI | CLI-only |
| `prepare --typos` | CLI flag | unchanged CLI | CLI-only |
| `prepare --no-typos` | CLI flag | unchanged CLI | CLI-only |
| `prepare --llm` / `--markup` | CLI flags | unchanged CLI | CLI-only |
| `prepare --work-dir` | CLI flag | unchanged CLI | CLI-only |
| `translate` | CLI `translate` | unchanged CLI | CLI-only |
| `translate --to` | CLI flag | unchanged CLI | CLI-only |
| `translate --from` | CLI flag | unchanged CLI | CLI-only |
| `translate --out` | CLI flag | unchanged CLI | CLI-only |
| `translate --literary` | CLI flag | unchanged CLI | CLI-only |
| `translate --work-dir` | CLI flag | unchanged CLI | CLI-only |
| `settings list` | CLI `settings list` | unchanged CLI | CLI-only |
| `settings get` | CLI `settings get` | unchanged CLI | CLI-only |
| `settings set` | CLI `settings set` | unchanged CLI | CLI-only |
| `bench` | CLI `bench` | unchanged CLI | CLI-only |
| `bench --voice` | CLI flag | unchanged CLI | CLI-only |
| `bench --modes` | CLI flag | unchanged CLI | CLI-only |
| `bench --out` | CLI flag | unchanged CLI | CLI-only |
| `bench --install-graphs` | CLI flag | unchanged CLI | CLI-only |
| `--json` | CLI flag | unchanged CLI | CLI-only |
| `--yes` | CLI flag | unchanged CLI | CLI-only |
| `--dry-run` | CLI flag `--dry-run` | unchanged CLI | CLI-only |
| `--version` | CLI flag | unchanged CLI | CLI-only |
