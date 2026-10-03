# Was Voxprint macht

**Voxprint AI Audiobook Builder** ({{studio.tagline}}) ist eine Windows-11-Anwendung, die zwei Dinge vollständig auf Ihrem eigenen Computer erledigt:

1. **Sie bringt einer Computerstimme bei, wie eine bestimmte Person zu klingen.** Sie geben eine Aufnahme einer Stimme (Ihrer eigenen oder der einer Person, die zugestimmt hat) – am besten 5 bis 15 Minuten – und, falls vorhanden, den gelesenen Text. Voxprint richtet Text und Ton aufeinander aus, schneidet die Aufnahme in saubere Stücke und trainiert ein kleines Stimm-Add-on (einen *Adapter*, technisch LoRA) für das Sprachmodell **Qwen3-TTS**.
2. **Sie liest ganze Bücher mit dieser Stimme vor.** Sie wählen ein Buch (TXT, FB2 oder EPUB) und eine Stimme aus Ihrer Bibliothek und erhalten ein fertiges Hörbuch mit Kapiteln – eine Opus-Datei, eine MP3 pro Kapitel und so weiter.

Es gibt keine Kommandozeile, keinen Browser und kein Benutzerkonto. Aufnahmen, Texte, Stimmen und Ergebnisse bleiben auf dem Computer. Das Programm geht nur ins Internet, um Modelle herunterzuladen, nach Updates zu suchen und – wenn Sie es verlangen – Stimmen aus dem Stimmen-Repository zu laden.

![Das Studio – der erste Bildschirm von Voxprint (dunkles Design, deutsche Oberfläche).](@studio_home)

> **Beta-Software.** Version 0.1.0 wurde auf einem Rechner (Windows mit NVIDIA RTX 4090) und mit im Wesentlichen einem Sprecher von Anfang bis Ende getestet. Rechnen Sie mit Ecken und Kanten; bewahren Sie eine Kopie Ihrer Aufnahmen und Stimmen auf (siehe „{{backup.title}}“ in Kapitel 8).

## Was Sie brauchen

| | Minimum | Empfohlen |
|---|---|---|
| Betriebssystem | Windows 11 24H2 (Build 26100), 64 Bit | Windows 11 26H2 |
| Grafikkarte | NVIDIA mit etwa 6 GB Videospeicher (VRAM) | NVIDIA mit 16 GB VRAM oder mehr |
| Ohne NVIDIA-Karte | Der Datensatz wird trotzdem erstellt; das Training läuft auf dem Prozessor und dauert viele Stunden (das Programm warnt Sie) | – |
| Speicherplatz | etwa 12 GB (Modelle etwa 7 GB + Programm), plus etwa 4,2 GB für das optionale Universalmodell | eine SSD |
| Internet | einmalig, um die Modelle herunterzuladen (etwa 7 GB) | – |

## So funktioniert es in fünf Schritten

1. Lesen Sie einen bekannten Text laut vor und nehmen Sie ihn auf (5–15 Minuten).
2. Wählen Sie die Aufnahme und die Textdatei – oder nur die Aufnahme, wenn Sie keinen Text haben.
3. Voxprint richtet den Text mit einem neuronalen Aligner an der Aufnahme aus und schneidet sie in saubere Fragmente – den *Datensatz*.
4. Es trainiert auf Ihrer Grafikkarte einen kleinen Stimm-Adapter.
5. Die Stimme erscheint unter **{{studio.voices_title}}**. Nun können Sie damit **{{studio.narrate_title}}**.

> **Verwenden Sie Stimmen verantwortungsvoll.** Eine Stimme ist ein personenbezogenes Datum und ein Persönlichkeitsrecht ihres Inhabers. Verwenden Sie eine Stimme nur mit ausdrücklicher Erlaubnis der Person, der sie gehört, und im Rahmen der Lizenz der Stimme. Siehe Kapitel 11.

# Installation

## Das Installationsprogramm

Starten Sie **Voxprint-Setup.exe** (Inno Setup, Installation für alle Benutzer des Rechners, etwa 1,8 GB, weil die PyTorch-Bibliotheken enthalten sind). Der Assistent ist auf Englisch, Russisch und Deutsch verfügbar. Nach der Seite für den Installationsordner gibt es eine optionale Seite **„Vorhandene Modelle (optional)“**: Wenn Sie bereits heruntergeladene Modelle einer früheren Installation oder aus einer Sicherung haben, wählen Sie diesen Ordner; sonst lassen Sie das Feld leer. Das Installationsprogramm kopiert nichts – es merkt sich nur den Ordner, und beim ersten Start importiert das Programm die Modelle von dort, statt sie herunterzuladen.

Stille Installation für Administratoren: `Voxprint-Setup.exe /VERYSILENT /ModelsDir="D:\old\models"`.

> Wenn das Administratorkonto, das das Installationsprogramm ausführt, ein anderes ist als das Konto, mit dem Sie Voxprint benutzen, gehört der Datenordner dem Administrator. Legen Sie den Ordner mit vorhandenen Modellen dann später unter „{{ui.settings_title}}“ fest (Kapitel 8).

## Erster Start

Beim ersten Start bereitet sich Voxprint vor: Die Statuszeile zeigt „{{ui.prefetch_start}}“ Es lädt die benötigten Modelle herunter (etwa 7 GB, einmalig). Ist Hugging Face langsam oder nicht erreichbar, wechselt Voxprint automatisch zum ModelScope-Spiegel. Der Download kann unterbrochen werden und läuft an der gleichen Stelle weiter. Modelle, die schon auf dem Computer liegen (zum Beispiel im Hugging-Face-Cache oder bei Alexandria/Pinokio), werden ohne erneuten Download nur lesend weiterverwendet.

Ist alles fertig, zeigt die Statuszeile „{{ui.prefetch_done}}“ Beim ersten Start erscheint ein Datenschutzhinweis; eine kurze Erinnerung („{{ui.footer_privacy}}“) bleibt am unteren Rand der Fenster.

## Wo was gespeichert wird

| Was | Wo |
|---|---|
| Programmdaten, Modelle, Protokolle, Stimmenbibliothek | `%LOCALAPPDATA%\Voxprint\` (`models\`, `logs\`, `voices\`, `state\`) |
| Hörbücher | `Documents\Voxprint\Audiobooks` (änderbar unter „{{narr.advanced}}“ im Vertonungsfenster) |
| Trainingsergebnisse | neben Ihrer Aufnahme: `<Name der Aufnahme>_Voxprint\dataset` und `output\<Stimmenname>` |

Den Modellordner und den Daten- und Protokollordner öffnen Sie mit Schaltflächen unter „{{ui.settings_title}}“.

# Schnellstart

**Ich möchte nur ein Buch hören (noch keine eigene Stimme).** Öffnen Sie **{{studio.voices_title}}**, drücken Sie **{{voices.repo_button}}** (oder warten Sie einfach: Online-Stimmen erscheinen von selbst in der Liste) und laden Sie die „Offene Universalstimme“ (Englisch, Lizenz CC0 – für jede Nutzung frei). Gehen Sie zurück ins Studio, öffnen Sie **{{studio.narrate_title}}**, wählen Sie eine TXT-/FB2-/EPUB-Datei und die Stimme und drücken Sie **{{narr.start}}**.

**Ich möchte meine eigene Stimme.** Nehmen Sie sich beim Vorlesen des Aufnahmeskripts auf (Abschnitt 6.1), öffnen Sie **{{studio.train_title}}**, wählen Sie Aufnahme und Skripttext, lassen Sie die Trainingsqualität auf „{{preset.balanced}}“ und drücken Sie **{{ui.btn_lora}}**. Bestätigen Sie nach dem Training die Einverständniskarte. Die Stimme steht nun unter **{{studio.voices_title}}** und in **{{studio.narrate_title}}** bereit.

# Das Studio (Startbildschirm)

Das Studio ist das erste Fenster. Es hat keine eigenen Einstellungen – es ist ein Menü aus drei großen Karten und einem Zahnrad.

![Das Studio.](@studio_home)

| Element | Was es tut und wann Sie es benutzen |
|---|---|
| Titel und Untertitel | „Voxprint AI Audiobook Builder – {{studio.tagline}}“. Nur zur Information. |
| Karte **{{studio.narrate_title}}** | {{studio.narrate_desc}} Öffnet das Vertonungsfenster. Das ist die Hauptkarte. Ist die Bibliothek leer, steht auf der Karte: „{{studio.narrate_no_voice}}“ Während ein Buch vertont wird, steht dort „{{studio.narrate_running}}“ |
| Karte **{{studio.train_title}}** | {{studio.train_desc}} Öffnet das Trainingsfenster. Während des Trainings steht dort „{{studio.train_running}}“ |
| Karte **{{studio.voices_title}}** | {{studio.voices_desc}} Unten auf der Karte zeigt ein Zähler, wie viele Stimmen Sie haben (zum Beispiel „Stimmen in der Bibliothek: 3“). |
| Zahnrad (oben rechts) | Öffnet **{{ui.settings_title}}** (Kapitel 8). |
| Hinweis unten | „{{ui.footer_privacy}}“ |

Jedes andere Fenster hat oben links eine Schaltfläche **{{nav.back}}**, die Sie hierher zurückbringt. Schwere Arbeit (Training, Vertonung) läuft im Hintergrund weiter, wenn Sie zurückgehen.

# Ein Buch vertonen

Öffnen Sie dieses Fenster über die Karte **{{studio.narrate_title}}**. {{narr.intro}}

![Das Vertonungsfenster, obere Hälfte: Buch, Stimme und Textvorbereitung (Demonstrationsstimme).](@narrate_top)

## Schritt 1 – das Buch

| Bedienelement | Was es tut |
|---|---|
| **{{narr.choose_book}}** | Öffnet einen Dateidialog. Unterstützt werden: `.txt` (UTF-8 oder cp1251; Überschriften wie „Kapitel 1“, „Chapter 1“ oder „Глава 2“ werden zu Kapiteln), `.fb2` und `.fb2.zip`, `.epub`. Titel, Autor und Cover werden gelesen, wenn vorhanden. |
| Dateiname und Infozeile | Nach der Auswahl zeigt Voxprint den Titel und „3 Kapitel · etwa 7 Min. Audio“ – Kapitelzahl und erwartete Länge. Vor der Auswahl steht dort „{{narr.no_book}}“ |

Lässt sich die Datei nicht öffnen, erscheint eine Meldung wie „{{err.book_unsupported}}“ oder „{{err.book_empty}}“ (siehe Kapitel „Fehlerbehebung“).

## Schritt 2 – die Stimme

Die Auswahlliste unter **{{narr.voice}}** zeigt die Stimmen Ihrer Bibliothek. Online-Stimmen, die noch nicht installiert sind, erscheinen als „<Name> (herunterladen, <Größe>)“; sie werden beim Start automatisch geladen und geprüft. Unter der Liste zeigt ein farbiges **Lizenz-Badge**, was Sie mit dem Ergebnis tun dürfen (siehe Kapitel 7 und das Glossar in Kapitel 9):

* grün – kommerzielle Nutzung erlaubt (zum Beispiel „CC-BY-4.0 · kommerzielle Nutzung erlaubt“ und „Kommerziell“);
* bernsteinfarben – eingeschränkt („Nur persönliche Nutzung · …“, „Nur privat“, „Öffentlich, nichtkommerziell“).

Unter dem Badge wiederholt ein Hinweis die Einschränkung, zum Beispiel: „{{narr.voice_personal_note}}“ Haben Sie noch gar keine Stimmen, sagt das Fenster „{{narr.no_voice}}“ und bietet die Schaltflächen **{{studio.train_title}}** und **{{studio.voices_title}}** an.

## Schritt 3 – Text vorbereiten

„{{narr.prep_hint}}“ Nichts wird zur Prüfung angezeigt, und Ihre Buchdatei wird nie verändert. Alle regelbasierten Schritte sind standardmäßig angehakt; nehmen Sie den Haken bei dem heraus, was Sie nicht wollen.

| Kontrollkästchen | Was es tut |
|---|---|
| {{prep.layout}} | {{prep.layout_d}} |
| {{prep.noise}} | {{prep.noise_d}} |
| {{prep.quotes}} | {{prep.quotes_d}} |
| {{prep.links}} | {{prep.links_d}} |
| {{prep.headings}} | {{prep.headings_d}} |
| {{prep.numbers}} | {{prep.numbers_d}} |
| {{prep.abbrev}} | {{prep.abbrev_d}} |
| {{prep.spellfix}} | {{prep.spellfix_d}} Es braucht einen einmaligen Download (eine Schaltfläche **{{prep.model_download}}** erscheint; das Modell ist etwa 365 MB groß) und gilt nur für russische Bücher. |
| ▸ {{prep.more}} | Ausgegraute Ideen für spätere Versionen: Pausen und Zeichensetzung per KI, russische Betonungszeichen, Übersetzung vor der Vertonung, verschiedene Stimmen für Figuren. Sie sind mit „{{prep.later_tag}}“ markiert und tun noch nichts. |

Die Regeln decken Russisch und Englisch vollständig ab. Für andere Sprachen (zum Beispiel deutsche Bücher) werden nur die sprachneutralen Schritte angewendet – Layout, Fußnoten, Anführungszeichen, Links.

## Schritt 4 – Ausgabeformat und Qualität

![Das Vertonungsfenster, untere Hälfte: Ausgabeformat, Qualität, Fortschritt und Mini-Player.](@narrate_bottom)

**{{narr.format}}** – wählen Sie eines:

| Format | Was Sie erhalten | Wo es abspielbar ist |
|---|---|---|
| {{fmt.opus_single}} | eine Datei `Autor - Titel.opus` mit Kapitelmarken | {{fmt.opus_single_where}} |
| {{fmt.mp3_chapters}} | ein Ordner mit `NN - Kapitel.mp3`, eine Wiedergabeliste und ein Cover | {{fmt.mp3_chapters_where}} |
| {{fmt.m4b}} | eine `.m4b`-Datei mit Kapiteln | {{fmt.m4b_where}} |

Bei M4B erscheint ein gelber Patenthinweis: „{{narr.aac_note}}“ Kurz gesagt: AAC ist patentbehaftet, Voxprint gewährt dafür keine Patentlizenz, und für die Einhaltung der Gesetze bei Wahl dieses Formats sind allein Sie verantwortlich. Brauchen Sie Apple Books nicht, nutzen Sie es nicht.

**▸ {{narr.other_formats}}** (eingeklappt) enthält weitere Formate: {{fmt.m4b_opus}}, {{fmt.opus_chapters}}, {{fmt.mp3_single}}, {{fmt.flac_chapters}}, {{fmt.wav_chapters}}.

**{{narr.quality}}** – drei Schaltflächen, die die Bitrate festlegen (die Datenmenge pro Sekunde Ton – mehr Daten bedeuten besseren Klang und größere Dateien):

| Schaltfläche | Bedeutung | Opus / MP3 / AAC | Etwa pro Stunde Audio |
|---|---|---|---|
| {{narr.preset_compact}} | {{narr.preset_compact_d}} | 24 / 64 / 48 kbit/s | 11 / 29 / 22 MB |
| {{narr.preset_standard}} | {{narr.preset_standard_d}} | 32 / 96 / 64 kbit/s | 14 / 43 / 29 MB |
| {{narr.preset_high}} | {{narr.preset_high_d}} | 48 / 128 / 96 kbit/s | 22 / 58 / 43 MB |

**{{narr.pauses}}** – ein Regler mit fünf Stufen, von *{{narr.pauses_1}}* über *{{narr.pauses_3}}* (Standard) bis *{{narr.pauses_5}}*. Voxprint schneidet den Text an jedem Komma, Satzende, jeder Auslassung, jedem Gedankenstrich, Absatz, Szenenwechsel und Kapitelende und fügt dort **Stille exakter Länge** ein, sodass die Pausen hörbar sind, egal wie das Stimmmodell den Text phrasiert. Bei *{{narr.pauses_3}}* folgen auf einen Satz 430 ms, auf einen Absatz 1,05 s und auf ein Kapitel 1,9 s – etwas länger als in früheren Versionen; jede Art hat ihre eigene Länge (Komma 240 ms, Auslassung 680 ms, Gedankenstrich 320 ms, Szenenwechsel 1,9 s), der Regler skaliert alle. Die Wahl wird gespeichert. Ändert man sie für ein bereits vertontes Buch, werden die gespeicherten Fragmente neu zusammengefügt statt erneut gesprochen, solange die Textteile gleich bleiben.

**▸ {{narr.advanced}}** (eingeklappt) enthält: die genauen Bitraten je Format (ändern Sie eine, wird die Qualität „benutzerdefiniert“: keine Schaltfläche ist hervorgehoben); **{{narr.choose_folder}}** – den Ordner, in dem das Hörbuch abgelegt wird; das Kontrollkästchen **{{narr.speak_titles}}**; und das schreibgeschützte Feld „{{narr.sample}}“, das zeigt, wie der erste geänderte Absatz nach der Vorbereitung aussieht.

## Starten, pausieren, anhalten

| Bedienelement | Was es tut |
|---|---|
| **{{narr.start}}** | Startet die Vertonung. Beim ersten Start wird das Stimmmodell geladen („{{narr.loading_model}}“). |
| **{{narr.pause}}** / **{{narr.resume}}** | Pause greift nach dem aktuellen Fragment; danach heißt dieselbe Schaltfläche „Fortsetzen“. |
| **{{ui.cancel}}** | Beendet den Auftrag. Fertige Fragmente bleiben gespeichert. |
| Fortschrittsbalken und Statuszeile | z. B. „Fragment 3 von 8 · noch etwa 7 Min.“ – wie viele Stücke (Fragmente) des Buches fertig sind und wie lange es etwa noch dauert. |
| **{{ui.open_folder}}** | Erscheint am Ende („{{ui.ready}}“): öffnet den Ordner mit dem fertigen Hörbuch. |

**Fortsetzbar.** Jedes fertige Fragment wird auf der Festplatte gespeichert. Haben Sie pausiert, abgebrochen, ist das Programm abgestürzt oder haben Sie es geschlossen, drücken Sie später erneut **{{narr.start}}**: „{{narr.resuming_plain}}“ Nach erfolgreichem Abschluss werden die gespeicherten Fragmente gelöscht.

**Zuhören, während es entsteht.** Sobald die ersten Fragmente fertig sind, erscheint der **Mini-Player** („{{player.title_live}}“): eine Schaltfläche „{{player.play}}“ / „{{player.pause}}“, ein Schieberegler und die Zeit. „{{player.live_status_plain}}“ Er folgt neuen Fragmenten und wartet, wenn er die Vertonung eingeholt hat. Ist der Auftrag zu Ende, wird das gesamte Ergebnis seine Wiedergabeliste.

Nach erfolgreichem Abschluss sehen Sie „{{narr.done}}“ und je nach Lizenz der Stimme einen Hinweis, zum Beispiel: „{{narr.done_private_reminder}}“

## Geschwindigkeit

Die Vertonung nutzt **gebündelte Erzeugung (Batch)**: Mehrere Fragmente durchlaufen das Stimmmodell in einem Aufruf statt einzeln. Die Batch-Größe richtet sich nach dem freien Videospeicher (bis zu 12) und wird automatisch halbiert, wenn der Speicher knapp wird; scheitert ein Batch aus anderem Grund, fällt Voxprint auf Fragment-für-Fragment zurück – das Bündeln kann also nie dazu führen, dass ein Buch scheitert. Fertige Fragmente speichert ein Hilfsthread, während die Grafikkarte schon den nächsten Batch erzeugt.

Gemessen auf einer RTX 4090 (12 englische Fragmente, 80 s Audio, offene Stimme): Der frühere Modus brauchte einen Echtzeitfaktor (RTF) von **3,05** (243 s Arbeit für 80 s Ton); mit Batch 12 sind es **0,31** (25 s) – etwa **10-mal schneller**. RTF heißt „Sekunden Arbeit pro Sekunde Audio“: 0,31 bedeutet, dass eine Stunde Hörbuch etwa 19 Minuten dauert. Richtwerte bei RTF 0,31–0,5: 5 Stunden Audio ≈ 1,6–2,5 Stunden Arbeit; 20 Stunden ≈ 6–10 Stunden. Auf Karten mit 12–16 GB ist der Batch kleiner, die Geschwindigkeit also geringer.

# Eigene Stimme trainieren

## Die Stimme aufnehmen

Die Qualität der Stimme hängt vor allem von der Aufnahme ab. Planen Sie **5 bis 15 Minuten** saubere Sprache (unter 2 Minuten löst eine Warnung aus; unter 10 Sekunden wird abgelehnt). Das Programm enthält keinen Rekorder – nehmen Sie ein beliebiges Aufnahmegerät oder eine Handy-App und speichern Sie als WAV, FLAC, MP3, M4A oder OGG.

**Vor der Aufnahme:** ein ruhiger Raum, das Mikrofon etwa eine Handbreit vom Mund entfernt, keine Klang-„Verbesserungen“ (Rauschunterdrückung, Equalizer, Hall), gleichmäßige Lautstärke.

**Das Aufnahmeskript.** Das Repository enthält ein fertiges Skript, `docs/voice-script-ru-v3.txt` (Russisch; es hat auch englische und deutsche Passagen). Es besteht aus 18 Blöcken: Begrüßung, Zahlen und Daten, Laute der Sprache, neutrales Lesen, Ruhe, Freude, Traurigkeit, Ärger, Überraschung, Flüstern und Lautes, drei kurze Geschichten, optional Englisch und ein Fachbegriff-Block, ein optionaler Block nur für Erwachsene und – ganz am Ende – die Einverständniserklärung. Das Skript selbst nennt drei einfache Regeln:

1. **Lesen Sie nur gewöhnliche Zeilen.** Alles in eckigen Klammern und alle Zeilen mit `===` sind Hinweise – nicht vorlesen.
2. **Eine Zeile – ein Satz.** Lesen, atmen, eine Sekunde schweigen, dann die nächste Zeile.
3. **Verhaspelt? Eine Sekunde schweigen und denselben Satz noch einmal von vorn lesen.** Die Aufnahme muss nicht gestoppt werden.

Weitere Hinweise aus dem Skript: Gefühle müssen nicht gespielt werden – es genügt, die Stimme ein wenig zu verändern, und wenn es nicht klappt, lesen Sie in normaler Stimme. Mit „OPTIONAL“ gekennzeichnete Blöcke dürfen übersprungen werden. Genau diese Skriptdatei wählen Sie im Trainingsfenster als „Text“.

> Voxprint ist darauf ausgelegt, wiederholt gelesene Sätze zu verkraften: Beim Abgleich der Aufnahme mit dem Skript wird für jede Zeile die beste Lesung behalten. Dennoch gilt: Je sauberer die Aufnahme, desto besser die Stimme.

**Die Einverständniserklärung (Block 18).** Beenden Sie die Aufnahme mit **einem** der Einverständnissätze – dem, der für Sie zutrifft: *kommerziell* (A), *öffentlich nichtkommerziell* (B) oder *nur privat* (C), auf Russisch, Englisch oder Deutsch. Sagen Sie anstelle der Wörter in eckigen Klammern Ihren eigenen Namen und das heutige Datum. Lesen Sie ihn nur, wenn die Stimme Ihre ist oder Sie die Erlaubnis des Inhabers haben. Einzelheiten in Abschnitt 6.6.

## Das Trainingsfenster

Öffnen Sie es über die Karte **{{studio.train_title}}**. {{ui.subtitle}}

![Das Trainingsfenster mit eingeschaltetem Modus „kein Text“ (englische Oberfläche).](@train_notranscript)

| Bedienelement | Was es tut und wann Sie es benutzen |
|---|---|
| **{{ui.choose_audio}}** | Wählen Sie die Stimmaufnahme (die Datei lässt sich auch ins Fenster ziehen). Vor der Auswahl: „{{ui.audio_none}}“ |
| **{{ui.choose_text}}** | Wählen Sie die Textdatei, die Sie gelesen haben (`.txt`, UTF-8; zum Beispiel das Aufnahmeskript). Vor der Auswahl: „{{ui.text_none}}“ Im Modus „kein Text“ wird diese Zeile zu „{{asr.choose_script}}“ (optional). |
| Kontrollkästchen „{{asr.checkbox}}“ | Nur Audio, ohne Text: siehe Abschnitt 6.4. |
| **{{preset.label}}** | Auswahlliste der Trainingsstufen: siehe Abschnitt 6.3. Darunter zeigt das Programm eine kurze Beschreibung und eine Schätzung der Trainingszeit für Ihre Grafikkarte. |
| ▶ **{{preset.advanced}}** | Zeigt die Zahlen von Hand (nur bei „{{preset.manual}}“ änderbar). |
| **{{consent.title}}** | Wie die Erlaubnis des Stimminhabers festgehalten wird: siehe Abschnitt 6.6. |
| **{{ui.btn_lora}}** | Die Hauptschaltfläche: richtet die Aufnahme aus, schneidet sie, trainiert die Stimme und legt sie in Ihrer Bibliothek ab. |
| **{{preview.button}}** und Kontrollkästchen **{{preview.compare}}** | Ein schneller Hör- und Vergleichstest vor dem langen Lauf: Abschnitt 6.5. |
| Kontrollkästchen „{{check.checkbox}}“ | Nach dem Training lässt Voxprint die neue Stimme einen Testsatz sprechen und prüft Tonhöhe und Verständlichkeit automatisch. Eingeschaltet lassen. |
| Auswahl des Stimmtyps | Optional: „{{ui.voice_type_male}}“, „{{ui.voice_type_female}}“, „{{ui.voice_type_child}}“, „{{ui.voice_type_other}}“ oder „{{ui.voice_type_none}}“. Wird mit der Stimme gespeichert. |
| Beschreibungsfeld | „{{ui.voice_desc_placeholder}}“ |
| **{{ui.btn_merge}}** | Führt den trainierten Adapter zu einem eigenständigen Modellordner zusammen, der in jeder App mit Qwen3-TTS funktioniert (Abschnitt 6.7). |
| **{{ui.btn_dataset}}** | Bereitet nur den Datensatz vor (ausgerichtete und geschnittene Clips), ohne Training. Nützlich, wenn Sie woanders trainieren wollen. |
| Fortschrittsbalken und Etappen | {{stage.model}}, {{stage.align}}, {{stage.slice}}, {{stage.train}}, {{stage.save}} – die laufende Etappe ist hervorgehoben. Während ein Auftrag läuft, erscheint **{{ui.cancel}}**. |
| Statuszeile | Zu Beginn: „{{ui.status_idle}}“ |
| Zahnrad (oben rechts) | Öffnet „{{ui.settings_title}}“. |

Ist der Auftrag beendet, zeigt die Statuszeile „{{ui.voice_registered}}“ und eine Schaltfläche **{{ui.open_folder}}** erscheint. Der Adapterordner ist `…\output\<Stimmenname>` (einige zehn MB); der Datensatz liegt in `…\<Name der Aufnahme>_Voxprint\dataset`.

Im Standardmodus (mit Text) stehen die beiden Schaltflächen **{{ui.choose_audio}}** und **{{ui.choose_text}}** untereinander, jeweils mit dem Namen der gewählten Datei, und der Block „kein Text“ mit der Warnung ist ausgeblendet. *(Für diesen Standardzustand gibt es keinen eigenen Screenshot; das Bild oben zeigt den Modus „kein Text“, der alle übrigen Bedienelemente enthält.)*

## Trainingsstufen

Unter **{{preset.label}}** gibt es vier Stufen. Zu jeder zeigt das Programm, was sie bedeutet, und eine **geschätzte Zeit**, zum Beispiel „Geschätzte Trainingszeit: etwa 2 Min. auf NVIDIA GeForce RTX 4090 (5 Durchläufe, Adaptergröße 32, etwa 71 Clips)“. Lesen Sie die Schätzung als „ungefähr“.

| Stufe | Bedeutung |
|---|---|
| **{{preset.fast}}** | {{preset.desc_fast}} |
| **{{preset.balanced}}** | {{preset.desc_balanced}} |
| **{{preset.maximum}}** | {{preset.desc_maximum}} |
| **{{preset.manual}}** | {{preset.desc_manual}} |

Unter **{{preset.advanced}}** sehen Sie die Zahlen hinter einer Stufe:

| Feld | In einfachen Worten |
|---|---|
| {{preset.adv_epochs}} | Wie oft das Training Ihre ganze Aufnahme durchläuft. Mehr Durchläufe: näher an der Stimme, aber zu viele können sie „brabbeln“ lassen. |
| {{preset.adv_rank}} | Wie groß das Stimm-Add-on ist. Größer – mehr Details, mehr Speicher. |
| {{preset.adv_alpha}} | Wie stark das Add-on das Basismodell beeinflusst. |
| {{preset.adv_lr}} | Wie groß jeder Lernschritt ist. Die empfindlichste Zahl: nicht erhöhen, wenn Sie den Grund nicht kennen. |
| {{preset.adv_accum}} | Wie viele kleine Schritte zusammengezählt werden, bevor das Modell aktualisiert wird (spart Speicher). |

„{{preset.adv_hint}}“ Keine Stufe erhöht jemals die Lernrate. In unseren Tests bedeutete eine niedrigere „Trainingszahl“ (Loss) keine bessere Stimme – hören Sie das Ergebnis immer an.

## Kein Text? (nur Audio)

Haken Sie „{{asr.checkbox}}“ an, und das Programm erkennt die Sprache selbst (Qwen3-ASR, offline; das Erkennungsmodell, etwa 1,9 GB, wird bei der ersten Nutzung geladen). Die Textzeile bleibt als **optionales Skript** („{{asr.choose_script}}“, oder ziehen Sie eine `.txt` ins Fenster): Wählen Sie den vorgelesenen Text, wird jedes erkannte Stück tolerant damit abgeglichen (Versprecher und wiederholte Zeilen werden berücksichtigt), sodass der richtige Text statt des erkannten verwendet wird. Außerdem haben Sie:

* **{{ui.choose_audio}}** – jetzt können Sie **viele Dateien** auf einmal wählen; die Zeile zeigt „N Audiodatei(en) gewählt“;
* **{{asr.choose_folder}}** – einen ganzen Ordner mit Clips, samt Unterordnern;
* Ziehen und Ablegen von Dateien ins Fenster.

Jede Datei wird einzeln erkannt; lange Aufnahmen werden an Pausen in Stücke von bis zu 14 s geschnitten; alles wird zu einem Datensatz zusammengeführt und auf gleiche Lautstärke gebracht. Automatisch übersprungen werden: Clips unter 1,5 s oder über 20 s, leere oder unplausible Erkennung, Duplikate und Clips mit schlechtem Klang. Am Ende meldet das Fenster zum Beispiel „N Datei(en) erkannt: X von Y Clips behalten = … von … Min. Audio“.

> „{{asr.warning}}“

Deshalb müssen Sie „{{asr.confirm}}“ anhaken, bevor die Schaltflächen funktionieren (der Haken wird bei jedem Einschalten des Modus zurückgesetzt).

## Schnelle Vorschau und Vergleich zweier Varianten

Vor einem langen Lauf können Sie Einstellungen günstig testen. **{{preview.button}}** trainiert einen sehr kurzen Adapter an wenigen Clips (höchstens 12 Clips und 4 Durchläufe, etwa höchstens 3 Minuten) und erzeugt dann eine Probe von etwa 10 Sekunden, die Sie im Player anhören können. Die Schätzung steht neben der Schaltfläche: „{{preview.estimate_plain}}“

![Schnelle Vorschau zweier Varianten: Player und automatische Messwerte (englische Oberfläche).](@train_preview)

* Ist **{{preview.compare}}** angehakt, entstehen zwei Proben: A mit den aktuellen Einstellungen und B mit einem größeren Adapter und dem 1,5-fachen der Durchläufe.
* Jede Variante erhält eine Zeile automatischer Messwerte: die Länge der Probe, „Erkennungsfehler“ (wie viele Wörter ein automatischer Zuhörer beim Wiedererkennen der Probe falsch verstanden hat – je niedriger, desto besser) und „Tonhöhe gegenüber Ihrer Aufnahme“ in Halbtönen (wie viel höher oder tiefer die Stimme als Ihre ist – am besten nahe 0). Danach folgt ein Urteil: „{{check.verdict_good}}“ oder eine Warnung.
* **{{preview.play}}** spielt eine Variante ab. **{{preview.use}}** übernimmt ihre Zahlen in die Stufe „{{preset.manual}}“; drücken Sie dann **{{ui.btn_lora}}** für den vollen Lauf.
* Eine Vorschau-Stimme ist vorläufig und wird **nicht** zu **{{studio.voices_title}}** hinzugefügt. Die Messwerte sind ein Hinweis, kein Urteil: Hören Sie immer selbst zu.

## Die Einverständniskarte

Jede Stimme trägt einen Vermerk darüber, was ihr Inhaber erlaubt. Wählen Sie unter **{{consent.title}}**, wie er festgelegt wird:

| Option | Wann Sie sie benutzen |
|---|---|
| {{consent.mode_auto}} | Sie haben die Aufnahme mit einem der drei Einverständnissätze beendet (Block 18 des Skripts). Nach dem Training erkennt das Programm die Erklärung und zeigt das Ergebnis. |
| {{consent.mode_manual}} | Sie tippen den Namen der sprechenden Person (Platzhalter „{{consent.name_placeholder}}“) und wählen den Umfang selbst. Nur für die eigene Stimme oder mit Erlaubnis des Inhabers. |
| {{consent.mode_none}} | Keine Erklärung. Die Stimme wird als *nur privat* gekennzeichnet. |

Das Kontrollkästchen **{{consent.save_clip}}** speichert die gesprochene Erklärung (die letzten 45 Sekunden der Aufnahme) neben der Stimme.

Nach dem Training im automatischen Modus erscheint die **Einverständniskarte**:

![Die Einverständniskarte nach dem Training: erkannter Name, Datum und Umfang sowie die Schaltfläche „Bestätigen“ (englische Oberfläche).](@train_consent)

Sie zeigt den *erkannten* Namen, das Datum und den Umfang sowie den erkannten Satz. Prüfen Sie ihn, ändern Sie bei Bedarf den Umfang in der Auswahlliste und drücken Sie **{{consent.confirm}}**. Ist die Erklärung undeutlich oder fehlt sie, gilt die strengste Stufe – „{{consent.badge_private_only}}“. Das ist ein Vermerk dessen, was gesagt wurde, **keine Rechtsberatung**.

| Umfang in der Auswahlliste | Badge | Bedeutung | Gespeicherte Lizenz |
|---|---|---|---|
| {{consent.scope_commercial}} | {{consent.badge_commercial}} | {{consent.tip_commercial}} | CC-BY-4.0 |
| {{consent.scope_public_noncommercial}} | {{consent.badge_public_noncommercial}} | {{consent.tip_public_noncommercial}} | CC-BY-NC-4.0 |
| {{consent.scope_private_only}} | {{consent.badge_private_only}} | {{consent.tip_private_only}} | custom/personal-only |

## Das Universalmodell (optional)

Der kleine Adapter (einige zehn MB) funktioniert nur in wenigen Apps (zum Beispiel Alexandria). **{{ui.btn_merge}}** führt ihn zu einem eigenständigen Modellordner (`merged_model`) zusammen, der in **jeder** App mit Qwen3-TTS funktioniert. Voxprint prüft den freien Speicherplatz und fragt zuerst nach („{{ui.merge_confirm_title}}“); es wird nie automatisch erstellt. Solange keine Stimme erstellt wurde, ist die Schaltfläche deaktiviert. Einzelheiten stehen in `USAGE.txt` im Ordner.

# Meine Stimmen

Öffnen Sie das Fenster über die Karte **{{studio.voices_title}}**. {{voices.intro}}

![Meine Stimmen: drei Stimmenkarten (Demonstrationsstimmen) mit Lizenz- und Umfang-Badges.](@voices)

Jede Stimme ist eine Karte:

| Element | Bedeutung |
|---|---|
| Name und zwei Badges | Das **Lizenz-Badge** (grün „… · kommerzielle Nutzung erlaubt“ oder bernsteinfarben „Nur persönliche Nutzung · …“) und das **Umfang-Badge** ({{consent.badge_commercial}}, {{consent.badge_public_noncommercial}}, {{consent.badge_private_only}}). |
| Infozeile | Sprache · Minuten Sprache im Training · Epochen (Durchläufe) · Stimmtyp · Autor · (bei Online-Stimmen) Größe. |
| Beschreibung | Was der Autor geschrieben hat. Bei einer Stimme „nur zum Testen“ steht zusätzlich eine Warnung. |
| **{{voices.preview}}** | Spielt eine kurze Probe der Stimme. **{{voices.stop}}** hält sie an. Hat die Stimme keine Probe: „{{voices.no_preview}}“ |
| **{{voices.narrate}}** | Öffnet das Vertonungsfenster mit bereits gewählter Stimme. |
| **{{voices.edit}}** | Öffnet „{{voices.edit_title}}“: {{voices.field_name}}, {{voices.field_author}}, {{voices.field_license}}, {{voices.field_type}}, {{voices.field_description}}. Drücken Sie **{{voices.save}}**. |
| **{{voices.delete}}** | Löscht die Stimme nach Rückfrage von diesem Computer: „{{voices.delete_confirm_plain}}“ |

Die Fußzeile des Fensters wiederholt: „{{voices.rights_note}}“

Ist die Bibliothek leer, sagt das Fenster „{{voices.empty}}“ und bietet **{{voices.empty_train}}** an.

## Online-Stimmen und das Stimmen-Repository

![Stimmen, die online verfügbar, aber noch nicht installiert sind, haben eine Schaltfläche „Herunterladen“ (Demonstrationsdaten; englische Oberfläche).](@voices_online)

* Stimmen aus dem Online-Index erscheinen von selbst als Karten mit dem Zustand „{{voices.remote_state}}“, ihrer Lizenz, der Größe und einer Schaltfläche **{{voices.remote_download}}**. Der Download wird per SHA-256-Prüfsumme kontrolliert und kann fortgesetzt werden.
* **{{voices.remote_refresh}}** lädt die Liste neu. Ohne Verbindung zeigt Voxprint die zuletzt bekannte Liste („{{voices.repo_offline}}“).
* **{{voices.repo_button}}** öffnet den Dialog „{{voices.repo_title}}“ mit **{{voices.repo_refresh}}**, einem Feld für die Adresse des Index (`index.json`) und **{{voices.repo_download}}**.

## Eine Stimme importieren

**{{voices.import}}** bietet zwei Wege: **{{voices.import_folder}}** (ein Adapterordner) und **{{voices.import_zip}}** (ein Stimmpaket). Archive werden auf unsichere Pfade und Größenlimits geprüft. Eine Stimme ohne angegebene Lizenz gilt als „nur persönliche Nutzung“.

## Lizenzen und die zwei Stimmen, die Voxprint anbietet

| Lizenz | Kommerzielle Nutzung |
|---|---|
| CC0-1.0, CC-BY-4.0, CC-BY-SA-4.0 | erlaubt (Namensnennung bzw. Weitergabe unter gleichen Bedingungen beachten, wenn die Lizenz es verlangt) |
| CC-BY-NC-4.0, CC-BY-NC-SA-4.0 | nicht erlaubt |
| custom/personal-only (Standard) | nicht erlaubt; Ergebnisse bleiben auf Ihrem Computer |
| custom/test-use-only | nicht erlaubt; nur zum Testen, keine Veröffentlichung |

* **Offene Universalstimme** (englisch: *Open universal voice*, russisch: *Открытый универсальный голос*) – eine englische Stimme für alle, die keine eigene Aufnahme haben. Nur mit dem gemeinfreien LJ-Speech-Datensatz trainiert (Sprecherin Linda Johnson, LibriVox); Lizenz **CC0-1.0** – für jede Nutzung frei, auch kommerziell; die Nennung des LJ-Speech-Datensatzes (Keith Ito) ist erwünscht.
* **Александр (Alexander)** – eine männliche russische Stimme, die erste mit Voxprint trainierte. Lizenz: **nur persönliche Nutzung** – keine öffentliche Nutzung der Ausgabe, keine kommerziellen Projekte (Umfang „{{consent.badge_private_only}}“).

Beide werden getrennt (nicht im Installationsprogramm enthalten) aus dem Stimmen-Repository geladen.

> Die Bilder hier und in Kapitel 5 verwenden Demonstrationsstimmen; auf den Screenshots trägt die Stimme „Alexander“ noch eine frühere Test-Lizenzbezeichnung. In der veröffentlichten Stimmenliste lautet ihr Badge „Nur persönliche Nutzung“, wie hier beschrieben.

Das Badge ist eine Information, keine Rechtsberatung. Die Lizenz gilt nur für das Stimmenmodell.

# Einstellungen

Öffnen Sie **{{ui.settings_title}}** mit dem Zahnrad (oben rechts im Studio und in den anderen Fenstern).

![Einstellungen (englische Oberfläche).](@settings)

| Bedienelement | Was es tut |
|---|---|
| **{{ui.language}}** | Die Oberflächensprache: English, Deutsch, Русский. Sie ändert sich sofort in allen Fenstern. |
| **{{auto.button}}** | Wählt alle empfohlenen automatischen Optionen und lädt die nötigen Modelle (siehe unten). |
| **{{ui.btn_update}}** | Prüft, ob neuere verifizierte Versionen von Komponenten oder Modellen existieren, und installiert sie (mit Sicherheitsprüfung und automatischem Zurückrollen). Voxprint prüft außerdem einmal pro Woche unauffällig. Meldungen: „{{upd.up_to_date}}“, „{{upd.restart}}“ |
| **{{ui.settings_models_folder}}** | Öffnet den Ordner mit den heruntergeladenen Modellen. |
| **{{ui.settings_data_folder}}** | Öffnet den Daten- und Protokollordner – schicken Sie das Protokoll dem Entwickler, wenn ein Fehler wiederkehrt. |
| **{{backup.title}}** – Kontrollkästchen **{{backup.include_voices}}** | Ob Ihre Stimmenbibliothek zur Sicherung gehört (standardmäßig an). |
| **{{backup.btn_backup}}** | Kopiert Modelle (und Stimmen) in einen gewählten Ordner oder auf ein Laufwerk. Zeigt vorab Größe und freien Platz; fortsetzbar; identische Dateien werden übersprungen; jede Kopie wird geprüft. Fortschrittsbalken und eine Schaltfläche **{{ui.cancel}}** erscheinen. |
| **{{backup.btn_restore}}** | Stellt aus einem Ordner `Voxprint-backup` wieder her. Stimmen, die schon mit anderem Inhalt existieren, werden nie überschrieben. |
| **{{existing.title}}** | {{existing.hint}} **{{existing.choose}}** wählt den Ordner, **{{existing.clear}}** vergisst ihn. |
| **{{ui.settings_repair}}** | {{ui.settings_repair_tip}}. Ihre Stimmen und Modelle bleiben erhalten. |
| **{{ui.about}}** | Version, die Idee, Funktionsweise, Open-Source-Komponenten mit ihren Lizenzen, ein Link zum GitHub-Repository und Hinweise zu Drittsoftware. |
| **{{about.btn_close}}** | Schließt den Dialog. |

Bemerkt das Programm eine beschädigte Installation, bietet ein Banner die Reparatur an („{{ui.repair_offer_plain}}“). Hilft die Reparatur in der installierten Version nicht, starten Sie das Voxprint-Installationsprogramm erneut – Stimmen und Modelle bleiben erhalten.

Findet das Programm eine ältere Komponente in einer Umgebung, die ihm nicht gehört (zum Beispiel Ihr eigenes Python), fragt es zuerst: „{{upg.title}}“ mit **{{upg.btn_upgrade}}** oder **{{upg.btn_later}}**; Ihre Umgebung ändert es nie stillschweigend.

## {{auto.button}}

Ab Werk ist alles, was automatisch laufen kann, **bereits gewählt** und mit einem Stern und dem Wort „{{auto.recommended}}“ markiert: Wenn Sie nichts anfassen, arbeitet Voxprint mit maximaler Qualität. Die Schaltfläche **{{auto.button}}** in den Einstellungen wählt diese Optionen erneut (falls Sie welche abgeschaltet haben) und lädt die dafür nötigen Modelle mit Fortschrittsbalken. Die einzelnen Kontrollkästchen bleiben in ihren Fenstern.

* Textaufbereitung („Buch vertonen“): die sieben regelbasierten Schritte – Layout, Fußnotenmarken und Seitenzahlen, Anführungszeichen und Striche, Links, Kapitelüberschriften, Zahlen in Worten, Abkürzungen.
* KI-Korrektur von Tippfehlern (russische Bücher; einmaliger Download von etwa 365 MB).
* Trainingsfenster: „{{check.checkbox}}“ und „{{preview.compare}}“ (beide brauchen das Spracherkennungsmodell, etwa 1,9 GB, einmal geladen).
* Immer aktiv, ohne Schalter: Ausrichtung des Textes auf das Audio mit Plausibilitätsprüfung, Audio-Qualitätsfilter, Erkennungsfilter des Nur-Audio-Modus.
* Nicht enthalten, weil es sie noch nicht gibt: Zeichensetzungsmodell, Betonungszeichen, Übersetzung, Sprecherrollen.

Die Trainingsvorgabe bleibt absichtlich bei „{{preset.balanced}}“: In unseren Tests ließ längeres Training mit höherer Lernrate die Stimme nuscheln, mehr ist dort also nicht besser. Stattdessen wird nach „{{preview.compare}}“ die bessere der beiden Varianten als „{{auto.recommended}}“ markiert (zuerst das Urteil, dann weniger Erkennungsfehler, dann die kleinere Tonhöhenabweichung; ohne klaren Unterschied Variante A, die günstiger ist). Die Entscheidung bleibt bei Ihnen: Hören Sie immer hin.

# Glossar

<dl class="glossary">

<dt>Adapter (LoRA)</dt><dd>Eine kleine Add-on-Datei (einige zehn MB), die dem großen Sprachmodell eine bestimmte Stimme „beibringt“. Technisch LoRA – Low-Rank Adaptation. Die Schaltfläche „{{ui.btn_lora}}“ erzeugt ihn.</dd>

<dt>Universalmodell</dt><dd>Der zu einem vollständigen Modellordner von etwa 4 GB zusammengeführte Adapter. Er funktioniert in jeder App mit Qwen3-TTS.</dd>

<dt>Qwen3-TTS</dt><dd>Das offene Sprachsynthese-Modell, mit dem Voxprint spricht. Das Basismodell steht unter der Apache-2.0-Lizenz.</dd>

<dt>Datensatz</dt><dd>Das Trainingsmaterial, das Voxprint aus Ihrer Aufnahme schneidet: kurze Clips (3–12 s) mit ihren Texten, ein Referenzclip und ein Bericht.</dd>

<dt>Ausrichtung (Alignment)</dt><dd>Feststellen, in welcher Sekunde der Aufnahme jedes Wort des Textes gesprochen wird. Dadurch lässt sich an Pausen schneiden, nie mitten im Wort.</dd>

<dt>Clip / Fragment</dt><dd>Ein kurzes Stück der Aufnahme im Datensatz. Im Vertonungsfenster bedeutet „Fragment 3 von 8“ ein Stück des Buches.</dd>

<dt>Chunk</dt><dd>Ein Stück des Buchtextes von etwa einem Satz oder Satzteil, das dem Stimmmodell auf einmal übergeben wird. Die Oberfläche nennt es „Fragment“. Fertige Chunks werden gespeichert – dadurch ist die Vertonung fortsetzbar.</dd>

<dt>Batch (Stapel)</dt><dd>Mehrere Chunks, die das Stimmmodell zusammen in einem Aufruf erzeugt. Größere Batches sind viel schneller, brauchen aber mehr Videospeicher (bis zu 12 gleichzeitig).</dd>

<dt>RTF (Echtzeitfaktor)</dt><dd>Sekunden Rechenzeit pro Sekunde erzeugtem Audio. 0,31 bedeutet, dass eine Stunde Hörbuch etwa 19 Minuten dauert. Je niedriger, desto schneller.</dd>

<dt>Epoche / Durchlauf</dt><dd>Ein vollständiger Durchgang des Trainings durch Ihre gesamte Aufnahme (Oberfläche: „{{preset.adv_epochs}}“). Die Stimmenliste zeigt, wie viele Epochen eine Stimme trainiert wurde.</dd>

<dt>Trainingsstufe (Preset)</dt><dd>Ein fertiger Satz Trainingszahlen: {{preset.fast}}, {{preset.balanced}}, {{preset.maximum}}, {{preset.manual}}.</dd>

<dt>Rang, Alpha, Lernrate, Gradientenakkumulation</dt><dd>Technische Trainingszahlen unter „{{preset.advanced}}“; siehe Abschnitt 6.3.</dd>

<dt>Vorschau (Preview)</dt><dd>Unter „{{studio.voices_title}}“: eine kurze Probe einer Stimme (Schaltfläche „{{voices.preview}}“). Im Trainingsfenster: die <i>schnelle Vorschau</i> – ein kurzes Probetraining mit einer Probe von etwa 10 s und automatischen Prüfungen; zu Ihren Stimmen wird nichts hinzugefügt.</dd>

<dt>Erkennungsfehler (WER)</dt><dd>Der Anteil der Wörter, die ein automatischer Zuhörer (Qwen3-ASR) beim Abschreiben der erzeugten Probe falsch versteht. Je niedriger, desto besser.</dd>

<dt>Tonhöhe, Halbton</dt><dd>Wie hoch die Stimme ist. Ein Halbton ist der kleinste Schritt am Klavier. „Tonhöhe gegenüber Ihrer Aufnahme: +1,2 Halbtöne“ heißt: Die Stimme ist etwas höher als das Original.</dd>

<dt>Einverständnis</dt><dd>Die Erlaubnis des Stimminhabers, am Ende der Aufnahme gesprochen oder von Hand eingegeben und mit der Stimme gespeichert. Ein Vermerk, keine Rechtsberatung.</dd>

<dt>Umfang (Scope)</dt><dd>Was das Einverständnis erlaubt: „{{consent.badge_commercial}}“, „{{consent.badge_public_noncommercial}}“ oder „{{consent.badge_private_only}}“.</dd>

<dt>Lizenz-Badge / Umfang-Badge</dt><dd>Farbige Kennzeichen auf der Stimmenkarte und im Vertonungsfenster. Grün: kommerzielle Nutzung erlaubt. Bernstein: eingeschränkt. Das Badge ist eine Information, keine Rechtsberatung.</dd>

<dt>Stimmtyp</dt><dd>Eine optionale Kennzeichnung der Stimme: männlich, weiblich, Kind oder andere. Wird mit der Stimme gespeichert.</dd>

<dt>Mini-Player</dt><dd>Der kleine Player im Vertonungsfenster, der den fertigen Teil des Buches abspielt, während der Rest noch entsteht.</dd>

<dt>Bitrate (kbit/s)</dt><dd>Wie viele Daten eine Sekunde Ton braucht. Höher bedeutet bessere Qualität und größere Dateien. Sprache braucht wenig: Opus mit 32 kbit/s reicht völlig.</dd>

<dt>Opus, MP3, AAC / M4B, FLAC, WAV</dt><dd>Audioformate. Opus: klein, offen, frei. MP3: läuft überall. AAC im M4B-Container: für Apple Books (patentbehaftet). FLAC und WAV: verlustfrei, große Dateien für Archiv und Bearbeitung.</dd>

<dt>Kapitelmarken</dt><dd>Lesezeichen innerhalb einer einzelnen Audiodatei, mit denen ein Player von Kapitel zu Kapitel springen kann.</dd>

<dt>Videospeicher (VRAM)</dt><dd>Der Speicher der Grafikkarte. Er begrenzt die Größe von Modell und Batch.</dd>

<dt>Fortsetzen (Resume)</dt><dd>Eine angehaltene Vertonung oder einen Download an der Stelle weiterführen, an der er stehen blieb.</dd>

<dt>Prüfsumme (SHA-256)</dt><dd>Ein Fingerabdruck einer Datei, der beweist, dass sie unbeschädigt heruntergeladen oder kopiert wurde.</dd>

<dt>Spiegel (ModelScope)</dt><dd>Ein alternativer Download-Server, den Voxprint nutzt, wenn Hugging Face langsam oder nicht erreichbar ist.</dd>

<dt>Reparieren (Repair)</dt><dd>Prüft die eigene Umgebung von Voxprint erneut und behebt, was defekt ist.</dd>

</dl>

# Fehlerbehebung und häufige Fragen

**Welche Meldungen kann ich bekommen, und was tue ich?**

| Meldung | Was zu tun ist |
|---|---|
| {{err.audio_short}} | Nehmen Sie länger auf – 5–15 Minuten. |
| {{err.audio_unreadable}} | Wandeln Sie die Aufnahme in WAV oder MP3 um. |
| {{err.mismatch_length}} | Prüfen Sie, ob Sie die richtige Audio- und Textdatei gewählt und den ganzen Text gelesen haben. |
| {{err.mismatch_extra}} | Schneiden Sie die überzählige Sprache am Anfang oder Ende der Aufnahme ab oder wählen Sie den richtigen Text. |
| {{err.all_dropped}} | Prüfen Sie Mikrofon und Pegel; nehmen Sie an einem ruhigeren Ort neu auf. |
| {{err.oom}} | Schließen Sie andere Programme, die die Grafikkarte nutzen; Voxprint versucht es mit geringerer Last erneut („{{progress.oom_retry}}“). Als letzter Ausweg: **{{ui.retry_cpu}}**. |
| {{err.download_failed}} | Prüfen Sie Ihre Internetverbindung und starten Sie neu; der Download läuft an der gleichen Stelle weiter. |
| {{err.narration_chunk}} | Drücken Sie erneut **{{narr.start}}** – fertige Fragmente bleiben erhalten. |
| {{err.narration_no_ffmpeg}} | Einstellungen → **{{ui.settings_repair}}**, oder wählen Sie WAV. |
| {{err.book_unsupported}} | Wandeln Sie das Buch in TXT, FB2 oder EPUB um. |
| {{err.book_unsafe}} | Die Datei wirkt beschädigt oder zu groß; versuchen Sie eine andere Kopie. |
| {{warn.no_gpu}} | Training ist ohne NVIDIA-Karte möglich, aber extrem langsam. |
| {{warn.loss_low}} | Die Stimme kann brabbeln. Wählen Sie **{{preset.balanced}}**, nehmen Sie mehr Text auf und hören Sie die schnelle Vorschau an. |

**Die Stimme klingt höher als meine.** Eine leichte Tonhöhen-Drift nach oben (bei einer männlichen Stimme wurden etwa +2…+3 Halbtöne beobachtet) ist ein bekannter Effekt dieser Art von Training. Die automatische Prüfung warnt oberhalb von +4. Probieren Sie die schnelle Vorschau mit beiden Varianten und nehmen Sie die näherliegende.

**Die Stimme brabbelt oder hört nicht auf.** Zu viele Durchläufe oder zu hohe Lernrate. Wählen Sie „{{preset.balanced}}“ und erhöhen Sie die Lernrate nie von Hand.

**Kann ich anhalten und später weitermachen?** Vertonung: ja, immer (Abschnitt 5.5). Downloads und Sicherungen: ja. Training: Abbrechen ist möglich, ein voller Lauf beginnt aber von vorn; nutzen Sie vorher die schnelle Vorschau.

**Darf ich meine Stimme für ein veröffentlichtes Hörbuch verwenden?** Nur wenn der Einverständnis-Umfang es erlaubt („{{consent.badge_commercial}}“ zum Verkaufen; „{{consent.badge_public_noncommercial}}“ zum kostenlosen Veröffentlichen). Ohne Einverständniserklärung trainierte Stimmen sind *nur privat*. Die Lizenz können Sie unter **{{voices.edit}}** ändern, wenn die Stimme Ihnen gehört.

**Schickt Voxprint meine Aufnahmen irgendwohin?** Nein. Ins Internet gehen nur der Download von Modellen, die Update-Prüfung und – wenn Sie das Repository öffnen – das Abrufen des Stimmenindex und der von Ihnen gewählten Stimmen. Keine Telemetrie, kein Konto.

**Welche Sprachen werden unterstützt?** Die Oberfläche: English, Deutsch, Русский. Die Textvorbereitung deckt Russisch und Englisch vollständig ab. Das Sprachmodell kann weitere Sprachen, sie sind aber nicht getestet.

**Wo ist das Protokoll?** Schaltfläche „{{ui.settings_data_folder}}“ in den Einstellungen. Schicken Sie die Datei von dort, wenn ein Fehler wiederkehrt.

**Ist die Option AAC / M4B sicher zu verwenden?** Sie ist optional und standardmäßig aus gutem Grund ausgeschaltet: AAC ist patentbehaftet, und für die Einhaltung der Gesetze sind Sie verantwortlich (Abschnitt 5.4).

# Lizenzen und Verantwortung

* **Der eigene Quellcode von Voxprint steht unter der Apache License 2.0** (Copyright 2026 Aleksandr Mitroshenkov; siehe die Dateien `LICENSE` und `NOTICE`).
* **Die Apache-Lizenz des Codes gilt nicht für Modelle und Stimmen.** Die Modelle, die Voxprint herunterlädt (Qwen3-TTS, Qwen3-ASR und die optionalen Textmodelle), behalten die Lizenzen ihrer Autoren. Jede Stimme hat ihre eigene Lizenz, als Badge angezeigt und in `voice.json` gespeichert.
* Komponenten von Drittanbietern samt Lizenzen sind unter „{{ui.about}}“ sowie in `THIRD_PARTY_NOTICES.md` und im Ordner `licenses` der Installation aufgeführt. Qt/PySide6 wird unter der LGPL-3.0 verwendet; das mitgelieferte ffmpeg ist ein eigenes Programm mit eigener Lizenz; der AAC-Codec ist patentbehaftet.
* **Verwenden Sie Stimmen verantwortungsvoll.** Eine Stimme ist ein personenbezogenes Datum und ein Persönlichkeitsrecht. Verwenden Sie eine Stimme nur mit ausdrücklicher Erlaubnis ihres Inhabers und im Rahmen ihrer Lizenz; keine Nachahmung von Personen, kein Betrug, keine irreführenden Inhalte; kennzeichnen Sie synthetische Sprache, wo dies vorgeschrieben ist. Für die rechtmäßige Verwendung der von Ihnen erzeugten Stimmen und Audios sind Sie verantwortlich.

> **Dieses Handbuch und die Lizenz-Badges sind Informationen, keine Rechtsberatung.** Die Einverständniskarte hält fest, was eine sprechende Person gesagt hat; sie ersetzt nicht das Recht Ihres Landes.

*Voxprint AI Audiobook Builder 0.1.0 (Beta). Handbuch vom 3. Oktober 2026.*

*Voxprint AI Audiobook Builder ist ein unabhängiges Projekt und steht in keiner Verbindung zu anderen Produkten oder Diensten mit ähnlichen Namen.*
