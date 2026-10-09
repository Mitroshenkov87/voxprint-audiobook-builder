# Was Voxprint macht

**Voxprint AI Audiobook Builder** ({{studio.tagline}}) ist eine Windows-11-Anwendung, die zwei Dinge vollständig auf Ihrem eigenen Computer erledigt:

1. **Sie bringt einer Computerstimme bei, wie eine bestimmte Person zu klingen.** Sie geben eine Aufnahme einer Stimme (Ihrer eigenen oder der einer Person, die zugestimmt hat) – am besten 5 bis 15 Minuten – und, falls vorhanden, den gelesenen Text. Voxprint richtet Text und Ton aufeinander aus, schneidet die Aufnahme in saubere Stücke und trainiert ein kleines Stimm-Add-on (einen *Adapter*, technisch LoRA) für das Sprachmodell **Qwen3-TTS**.
2. **Sie liest ganze Bücher mit dieser Stimme vor.** Sie wählen ein Buch (TXT, FB2 oder EPUB) und eine Stimme aus Ihrer Bibliothek und erhalten ein fertiges Hörbuch mit Kapiteln – eine Opus-Datei, eine MP3 pro Kapitel und so weiter.

Sie brauchen keine Kommandozeile, keinen Browser und kein Benutzerkonto (eine Kommandozeile gibt es für Skripte und Agenten – Kapitel 11). Aufnahmen, Texte, Stimmen und Ergebnisse bleiben auf dem Computer. Das Programm geht nur ins Internet, um Modelle herunterzuladen, nach Updates zu suchen und – wenn Sie es verlangen – Stimmen aus dem Stimmen-Repository zu laden.

![Das Studio – der erste Bildschirm von Voxprint (Build 667).](@studio_home)

> **Beta-Software.** Version 0.1.3 (Build 667 „Menuchah“) wurde auf einem Rechner (Windows 11 mit NVIDIA RTX 4090) und mit im Wesentlichen einem Sprecher von Anfang bis Ende getestet. Rechnen Sie mit Ecken und Kanten; bewahren Sie eine Kopie Ihrer Aufnahmen und Stimmen auf (siehe „{{backup.title}}“ in Kapitel 8).

## Was Sie brauchen

| | Minimum | Empfohlen |
|---|---|---|
| Betriebssystem | Windows 11 24H2 (Build 26100), 64 Bit | Windows 11 26H2 |
| Grafikkarte | NVIDIA mit etwa 6 GB Videospeicher (VRAM) | NVIDIA mit 16 GB VRAM oder mehr |
| Ohne NVIDIA-Karte | Der Datensatz wird trotzdem erstellt; das Training läuft auf dem Prozessor und dauert viele Stunden (das Programm warnt Sie) | – |
| Speicherplatz | etwa 30 GB für die Installationsart *Vollständig* (Modelle etwa 25 GB + Komponenten etwa 3 GB), plus etwa 4,2 GB für das optionale Universalmodell | eine SSD |
| Internet | während der Installation (die Bibliotheken) und einmalig für die Modelle (zusammen etwa 28 GB bei *Vollständig*; Spracherkennung 1.7B auf Grafikkarten ab 8 GB Grafikspeicher) | – |

## So funktioniert es in fünf Schritten

1. Lesen Sie einen bekannten Text laut vor und nehmen Sie ihn auf (5–15 Minuten).
2. Wählen Sie die Aufnahme und die Textdatei – oder nur die Aufnahme, wenn Sie keinen Text haben.
3. Voxprint richtet den Text mit einem neuronalen Aligner an der Aufnahme aus und schneidet sie in saubere Fragmente – den *Datensatz*.
4. Es trainiert auf Ihrer Grafikkarte einen kleinen Stimm-Adapter.
5. Die Stimme erscheint unter **{{studio.voices_title}}**. Nun können Sie damit **{{studio.narrate_title}}**.

> **Verwenden Sie Stimmen verantwortungsvoll.** Eine Stimme ist ein personenbezogenes Datum und ein Persönlichkeitsrecht ihres Inhabers. Verwenden Sie eine Stimme nur mit ausdrücklicher Erlaubnis der Person, der sie gehört, und im Rahmen der Lizenz der Stimme. Siehe Kapitel 12.

## Screenshots und Bezeichnungen in der Oberfläche

Alle Screenshots in diesem Handbuch stammen aus Build 667 mit **englischer** Oberfläche. Im Text heißen die Bedienelemente so wie in der deutschen Oberfläche. Wenn Sie auf einem Bild eine Beschriftung suchen, die Sie bei sich sehen, hilft diese Tabelle: links die Beschriftung auf dem Screenshot, rechts dieselbe in der deutschen Oberfläche (gleich lautende Beschriftungen sind weggelassen). Die Sprache stellen Sie unter „{{ui.settings_title}}“ → „{{ui.language}}“ um.

| Auf dem Screenshot (englische Oberfläche) | In der deutschen Oberfläche |
|---|---|
| {{en:studio.narrate_title}} | {{studio.narrate_title}} |
| {{en:studio.train_title}} | {{studio.train_title}} |
| {{en:studio.voices_title}} | {{studio.voices_title}} |
| {{en:studio.revoice_title}} | {{studio.revoice_title}} |
| {{en:narr.choose_book}} | {{narr.choose_book}} |
| {{en:narr.translate_title}} | {{narr.translate_title}} |
| {{en:llm.title}} | {{llm.title}} |
| {{en:llm.prepare}} | {{llm.prepare}} |
| {{en:narr.prep_title}} | {{narr.prep_title}} |
| {{en:narr.format}} | {{narr.format}} |
| {{en:narr.quality}} | {{narr.quality}} |
| {{en:narr.other_formats}} | {{narr.other_formats}} |
| {{en:narr.pauses_enable}} | {{narr.pauses_enable}} |
| {{en:narr.pauses}} | {{narr.pauses}} |
| {{en:narr.advanced}} | {{narr.advanced}} |
| {{en:narr.start}} | {{narr.start}} |
| {{en:voices.import}} | {{voices.import}} |
| {{en:voices.repo_button}} | {{voices.repo_button}} |
| {{en:voices.remote_refresh}} | {{voices.remote_refresh}} |
| {{en:voices.preview}} | {{voices.preview}} |
| {{en:voices.narrate}} | {{voices.narrate}} |
| {{en:voices.export}} | {{voices.export}} |
| {{en:voices.open_folder}} | {{voices.open_folder}} |
| {{en:voices.delete}} | {{voices.delete}} |
| {{en:voices.remote_download}} | {{voices.remote_download}} |
| {{en:ui.choose_audio}} | {{ui.choose_audio}} |
| {{en:ui.choose_text}} | {{ui.choose_text}} |
| {{en:asr.checkbox}} | {{asr.checkbox}} |
| {{en:preset.label}} | {{preset.label}} |
| {{en:preset.advanced}} | {{preset.advanced}} |
| {{en:consent.title}} | {{consent.title}} |
| {{en:ui.btn_lora}} | {{ui.btn_lora}} |
| {{en:preview.button}} | {{preview.button}} |
| {{en:preview.compare}} | {{preview.compare}} |
| {{en:train.checks_options}} | {{train.checks_options}} |
| {{en:ui.voice_more}} | {{ui.voice_more}} |
| {{en:ui.btn_merge}} | {{ui.btn_merge}} |
| {{en:ui.btn_dataset}} | {{ui.btn_dataset}} |
| {{en:ui.settings_title}} | {{ui.settings_title}} |
| {{en:ui.language}} | {{ui.language}} |
| {{en:ui.transparency}} | {{ui.transparency}} |
| {{en:ui.net_iface}} | {{ui.net_iface}} |
| {{en:asrmodel.label}} | {{asrmodel.label}} |
| {{en:preload.option}} | {{preload.option}} |
| {{en:narrset.title}} | {{narrset.title}} |
| {{en:narrset.comma}} | {{narrset.comma}} |
| {{en:narrset.mid}} | {{narrset.mid}} |
| {{en:narrset.sentence}} | {{narrset.sentence}} |
| {{en:narrset.paragraph}} | {{narrset.paragraph}} |
| {{en:narrset.chapter}} | {{narrset.chapter}} |
| {{en:narrset.speed}} | {{narrset.speed}} |
| {{en:narrset.style}} | {{narrset.style}} |
| {{en:narrset.defaults}} | {{narrset.defaults}} |
| {{en:ui.btn_update}} | {{ui.btn_update}} |
| {{en:ui.settings_models_folder}} | {{ui.settings_models_folder}} |
| {{en:ui.settings_data_folder}} | {{ui.settings_data_folder}} |
| {{en:diag.button}} | {{diag.button}} |
| {{en:modules.title}} | {{modules.title}} |
| {{en:projects.title}} | {{projects.title}} |
| {{en:backup.title}} | {{backup.title}} |
| {{en:backup.include_models}} | {{backup.include_models}} |
| {{en:backup.include_voices}} | {{backup.include_voices}} |
| {{en:backup.link_models}} | {{backup.link_models}} |
| {{en:backup.btn_backup}} | {{backup.btn_backup}} |
| {{en:backup.btn_restore}} | {{backup.btn_restore}} |
| {{en:existing.title}} | {{existing.title}} |
| {{en:autorepair.title}} | {{autorepair.title}} |
| {{en:backup.btn_restore_folder}} | {{backup.btn_restore_folder}} |
| {{en:autorepair.button}} | {{autorepair.button}} |
| {{en:autorepair.stop}} | {{autorepair.stop}} |
| {{en:ui.about}} | {{ui.about}} |

# Installation

## Das Installationsprogramm

Starten Sie **Voxprint-Setup-online.exe** (Inno Setup, Installation für alle Benutzer des Rechners, etwa 34 MB; Build 667 hat 35.235.971 Bytes und die SHA-256 `c145a9fc84c736d655fbbe9bfd5c1cc94794b1be7fb8e9a5eec293e6f1578e42`). Es ist ein *Online*-Installer: Die Bibliotheken (PyTorch und der Rest) werden während der Installation von PyTorch/PyPI geladen. Das Installationsprogramm ist nicht signiert, daher warnt Windows SmartScreen – wählen Sie *Weitere Informationen → Trotzdem ausführen* erst, wenn die SHA-256 mit der auf der Release-Seite übereinstimmt. Der Assistent ist auf Englisch, Russisch und Deutsch verfügbar; die dort gewählte Sprache wird zur Oberflächensprache des Programms.

Die Seite **„Installationsart“** bietet zwei Möglichkeiten. **Vollständig** (Standard) lädt beim ersten Start ohne weiteren Klick alles – die Komponenten und alle Modelle, etwa 28 GB; der Assistent zeigt die Gesamtgröße und prüft vorher den freien Platz. **Schnell** installiert nur das Programm; beim ersten Start öffnet sich das Fenster **{{modules.title}}**, das denselben vollständigen Download anbietet und ihn mit „Herunterladen“ startet. In keinem Modus wird etwas weggelassen.

Nach der Seite für den Installationsordner gibt es eine optionale Seite **„Vorhandene Modelle (optional)“**: Wenn Sie bereits heruntergeladene Modelle einer früheren Installation oder aus einer Sicherung haben, wählen Sie diesen Ordner; sonst lassen Sie das Feld leer. Das Installationsprogramm kopiert nichts – es merkt sich nur den Ordner, und beim ersten Start importiert das Programm die Modelle von dort, statt sie herunterzuladen.

Stille Installation für Administratoren: `Voxprint-Setup-online.exe /VERYSILENT /Mode=full /ModelsDir="D:\old\models"` (`/Mode=quick` für *Schnell*).

> Wenn das Administratorkonto, das das Installationsprogramm ausführt, ein anderes ist als das Konto, mit dem Sie Voxprint benutzen, gehört der Datenordner dem Administrator. Legen Sie den Ordner mit vorhandenen Modellen dann später unter „{{ui.settings_title}}“ fest (Kapitel 8).

## Erster Start

Beim ersten Start bereitet sich Voxprint vor. Mit dem Online-Installer öffnet sich das Fenster **{{modules.title}}** und arbeitet ohne Klick: Eine Zeile zeigt, was gerade geladen wird („{{modules.now_download}}“), danach die Prüfsummenprüfung, ein Balken den Gesamtfortschritt. Schritt 1 bringt die Programmkomponenten zusammen mit dem Textkorrektur-Modell SAGE und den Übersetzungsmodellen, Schritt 2 direkt danach ALLE Modelle (Stimme, Ausrichtung, Spracherkennung) in Ihren Modellordner. Sonst zeigt die Statuszeile „{{ui.prefetch_start}}“ und Voxprint lädt die benötigten Modelle (bis etwa 15 GB, einmalig). Ist Hugging Face langsam oder nicht erreichbar, wechselt Voxprint automatisch zum ModelScope-Spiegel. Der Download kann unterbrochen werden und läuft an der gleichen Stelle weiter. Modelle, die schon auf dem Computer liegen (zum Beispiel im Hugging-Face-Cache oder bei Alexandria/Pinokio), werden ohne erneuten Download nur lesend weiterverwendet. Die mitgelieferte Stimme *Tirzah* (Kapitel 7) gehört zu diesem Download.

Ist alles fertig, zeigt die Statuszeile „{{ui.prefetch_done}}“ Beim ersten Start erscheint ein Datenschutzhinweis; eine kurze Erinnerung („{{ui.footer_privacy}}“) bleibt am unteren Rand der Fenster.

## Wo was gespeichert wird

| Was | Wo |
|---|---|
| Programmdaten, Modelle, Protokolle, Stimmenbibliothek | `%LOCALAPPDATA%\Voxprint\` (`models\`, `logs\`, `voices\`, `state\`) |
| Projekte: Hörbücher, Stimmtrainings, Neu vertonen | der Projektordner `%LOCALAPPDATA%\Voxprint\Projects` mit `Audiobooks\<Buch>`, `Voices\<Stimmenname>_Voxprint` und `Re-voice`; eine Verknüpfung *Voxprint Projects* liegt in „Dokumente“. Änderbar unter **{{projects.title}}** in „{{ui.settings_title}}“. |
| Trainingsergebnisse | `Projects\Voices\<Stimmenname>_Voxprint`: der `dataset` und der Adapter in `output\<Stimmenname>` |

Den Modellordner und den Daten- und Protokollordner öffnen Sie mit Schaltflächen unter „{{ui.settings_title}}“.

# Schnellstart

**Ich möchte nur ein Buch hören (noch keine eigene Stimme).** Öffnen Sie **{{studio.voices_title}}**, drücken Sie **{{voices.repo_button}}** (oder warten Sie einfach: Online-Stimmen erscheinen von selbst in der Liste) und laden Sie die „Offene Universalstimme“ (Englisch, Lizenz CC0 – für jede Nutzung frei). Gehen Sie zurück ins Studio, öffnen Sie **{{studio.narrate_title}}**, wählen Sie eine TXT-/FB2-/EPUB-Datei und die Stimme und drücken Sie **{{narr.start}}**.

**Ich möchte meine eigene Stimme.** Nehmen Sie sich beim Vorlesen des Aufnahmeskripts auf (Abschnitt 6.1), öffnen Sie **{{studio.train_title}}**, wählen Sie Aufnahme und Skripttext, lassen Sie die Trainingsqualität auf „{{preset.balanced}}“ und drücken Sie **{{ui.btn_lora}}**. Bestätigen Sie nach dem Training die Einverständniskarte. Die Stimme steht nun unter **{{studio.voices_title}}** und in **{{studio.narrate_title}}** bereit.

# Das Studio (Startbildschirm)

Das Studio ist das erste Fenster. Es hat keine eigenen Einstellungen – es ist ein Menü aus vier großen Karten und einem Zahnrad.

![Das Studio.](@studio_home)

| Element | Was es tut und wann Sie es benutzen |
|---|---|
| Titel und Untertitel | „Voxprint AI Audiobook Builder – {{studio.tagline}}“. Nur zur Information. |
| Karte **{{studio.narrate_title}}** | {{studio.narrate_desc}} Öffnet das Vertonungsfenster. Das ist die Hauptkarte. Ist die Bibliothek leer, steht auf der Karte: „{{studio.narrate_no_voice}}“ Während ein Buch vertont wird, steht dort „{{studio.narrate_running}}“ |
| Karte **{{studio.train_title}}** | {{studio.train_desc}} Öffnet das Trainingsfenster. Während des Trainings steht dort „{{studio.train_running}}“ |
| Karte **{{studio.voices_title}}** | {{studio.voices_desc}} Unten auf der Karte zeigt ein Zähler, wie viele Stimmen Sie haben (zum Beispiel „Stimmen in der Bibliothek: 1“ – die mitgelieferte Stimme). |
| Karte **{{studio.revoice_title}}** | {{studio.revoice_desc}} |
| Zahnrad (oben rechts) | Öffnet **{{ui.settings_title}}** (Kapitel 8). |
| Hinweis unten | „{{ui.footer_privacy}}“ |

Jedes andere Fenster hat oben links eine Schaltfläche **{{nav.back}}**, die Sie hierher zurückbringt. Schwere Arbeit (Training, Vertonung) läuft im Hintergrund weiter, wenn Sie zurückgehen.

# Ein Buch vertonen

Öffnen Sie dieses Fenster über die Karte **{{studio.narrate_title}}**. {{narr.intro}}

![Das Vertonungsfenster, obere Hälfte: Buch, Stimme, Textvorbereitung, Übersetzung und das optionale KI-Textmodell. Dieser Screenshot von Build 667 zeigt Boaz; diese Stimme wird nicht mehr installiert. Tirzah ist die mitgelieferte Stimme.](@narrate_top)

## Schritt 1 – das Buch

| Bedienelement | Was es tut |
|---|---|
| **{{narr.choose_book}}** | Öffnet einen Dateidialog. Unterstützt werden: `.txt` (UTF-8 oder cp1251; Überschriften wie „Kapitel 1“, „Chapter 1“ oder „Глава 2“ werden zu Kapiteln), `.fb2` und `.fb2.zip`, `.epub`. Titel, Autor und Cover werden gelesen, wenn vorhanden. |
| Dateiname und Infozeile | Nach der Auswahl zeigt Voxprint den Titel und „3 Kapitel · etwa 7 Min. Audio“ – Kapitelzahl und erwartete Länge. Vor der Auswahl steht dort „{{narr.no_book}}“ |

Lässt sich die Datei nicht öffnen, erscheint eine Meldung wie „{{err.book_unsupported}}“ oder „{{err.book_empty}}“ (siehe Kapitel „Fehlerbehebung“).

## Schritt 2 – die Stimme

Die Auswahlliste unter **{{narr.voice}}** zeigt die Stimmen Ihrer Bibliothek. Online-Stimmen, die noch nicht installiert sind, erscheinen als „<Name> (herunterladen, <Größe>)“; sie werden beim Start automatisch geladen und geprüft. Unter der Liste zeigt ein farbiges **Lizenz-Badge**, was Sie mit dem Ergebnis tun dürfen (siehe Kapitel 7 und das Glossar in Kapitel 9):

* grün – kommerzielle Nutzung erlaubt (zum Beispiel „CC0-1.0 · kommerzielle Nutzung erlaubt“ und „Kommerziell“ bei den mitgelieferten Stimmen);
* bernsteinfarben – eingeschränkt („Nur persönliche Nutzung · …“, „Nur privat“, „Öffentlich, nichtkommerziell“).

Unter dem Badge wiederholt ein Hinweis die Einschränkung, zum Beispiel: „{{narr.voice_personal_note}}“ Haben Sie noch gar keine Stimmen, sagt das Fenster „{{narr.no_voice}}“ und bietet die Schaltflächen **{{studio.train_title}}** und **{{studio.voices_title}}** an.

## Schritt 3 – Text vorbereiten

„{{narr.prep_hint}}“ Nichts wird zur Prüfung angezeigt, und Ihre Buchdatei wird nie verändert.

| Kontrollkästchen | Was es tut |
|---|---|
| {{prep.one}} | {{prep.one_d}} Eine Schaltfläche **{{prep.model_download}}** erscheint, wenn das russische Tippfehler-Modell noch nicht auf diesem Computer ist (etwa 365 MB). |

Die Regeln decken Russisch und Englisch vollständig ab. Für andere Sprachen (zum Beispiel deutsche Bücher) werden nur die sprachneutralen Schritte angewendet – Layout, Fußnoten, Anführungszeichen, Links.

### Optional – Buch übersetzen

Die Karte **{{narr.translate_title}}** hat ein Kontrollkästchen (*{{narr.translate_check}}*) und eine Liste der Zielsprachen. Voxprint erkennt die Sprache des Buchs selbst und übersetzt es satzweise **offline auf Ihrem Computer** mit den offenen Opus-MT-Modellen (Englisch ↔ Russisch, Englisch ↔ Deutsch; Russisch ↔ Deutsch läuft über Englisch, langsamer und ungenauer). Die Modelle sind etwa 300 MB pro Richtung groß und werden einmal heruntergeladen, mit SHA-256-Prüfung, wenn Sie **{{prep.model_download}}** drücken. Eine Grafikkarte wird genutzt, falls vorhanden, sonst der Prozessor. Kapitel, Kapitelüberschriften, Absätze, Gedichtzeilen, Szenenwechsel und damit die Pausen bleiben erhalten; danach läuft die übliche Textvorbereitung für die neue Sprache und das Buch wird darin gesprochen – wählen Sie für das beste Ergebnis eine Stimme der Zielsprache.

* Der übersetzte Text wird neben dem Hörbuch als `translation_xx.txt` gespeichert (im Ordner `Titel (xx)`). Öffnen, lesen und bearbeiten Sie ihn; starten Sie dieselbe Vertonung erneut, wird **Ihre bearbeitete Fassung** gesprochen. Löschen Sie die Datei, um neu zu übersetzen. Übersetzte Sätze werden zwischengespeichert, ein abgebrochener Auftrag übersetzt also keinen Satz doppelt.
* *{{narr.translate_note}}*
* Maschinelle Übersetzung ist keine menschliche Übersetzung: Namen, Redewendungen, Zeiten und Geschlecht können falsch sein; kurze Überschriften leiden am meisten. Gedichte verlieren den Reim.

### Optional – das KI-Textmodell

Die Karte **{{llm.title}}** gehört zum optionalen Textmodell Gemma 4 12B (etwa 7 GB mit der llama.cpp-Laufzeit; es wird von der Installationsart *Vollständig* oder mit `voxprint models download llm` geladen). Ist es installiert, steht auf der Karte „{{llm.ready}}“ Sie bietet zwei Kontrollkästchen: „{{llm.literary}}“ und „{{llm.prepare}}“. „{{llm.note}}“ **{{llm.prompts}}** öffnet den Ordner mit den Anweisungen für das Modell; geänderte Dateien werden statt der eingebauten verwendet.

> Bekanntes Problem in Build 667: Die beiden Kontrollkästchen dieser Karte reagieren möglicherweise nicht auf einen Klick, obwohl das Modell geladen ist. Die regelbasierte Textvorbereitung (Schritt 3) ist nicht betroffen. Das wird in einem der nächsten Builds behoben.

## Schritt 4 – Ausgabeformat und Qualität

![Das Vertonungsfenster, untere Hälfte: Textvorbereitung, Ausgabeformat, Qualität, exakte Pausen und die Start-Schaltfläche.](@narrate_bottom)

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

**{{narr.pauses_enable}}** – standardmäßig aus. Eingeschaltet wird der Text zusätzlich an **jedem** Komma geschnitten und dort Stille exakter Länge eingefügt; die Pausen sind am deutlichsten, aber die Stimme kann kurze Wörter verschlucken. Nur dann ist der Regler **{{narr.pauses}}** aktiv: fünf Stufen von *{{narr.pauses_1}}* über *{{narr.pauses_3}}* (Standard) bis *{{narr.pauses_5}}*; bei *{{narr.pauses_3}}* folgen auf einen Satz 430 ms und auf einen Absatz 1050 ms, und der Regler skaliert jede Art von Pause. Ändert man ihn für ein bereits vertontes Buch, werden die gespeicherten Fragmente neu zusammengefügt statt erneut gesprochen, solange die Textteile gleich bleiben.

**{{narr.check_chunks}}** – standardmäßig aus. Nach der Synthese wird jedes Fragment per Spracherkennung nachgehört; Fragmente, die Wörter auslassen oder wiederholen, werden neu erzeugt.

### Pausen und Lesetempo

Seit Version 0.1.3 folgt jede Vertonung der Zeichensetzung und der Gliederung des Textes – auch ohne das Häkchen oben. Voxprint schneidet den Text satzweise und an starken Übergängen innerhalb eines Satzes (Gedankenstrich, Doppelpunkt, Semikolon, ein Komma vor „und“ / „aber“), schneidet bei jedem gesprochenen Stück die eigene Stille ab und fügt Stille exakter Länge ein: Komma 0,25 s, starke Grenze 0,40 s, Satz 0,60 s, Absatz oder Vers 1,00 s, Kapiteltitel, Kapitelende oder Szenenwechsel 2,00 s. Einfache Kommas bleiben innerhalb eines Stücks, und kein Stück ist kürzer als 20 Zeichen, damit die Stimme keine Wörter verschluckt.

Das **Lesetempo passt sich dem Text an**: Lange, kommareiche und beschreibende Sätze sowie bibelartige Texte werden etwas langsamer gelesen, Dialoge und kurze Zeilen im eigenen Tempo der Stimme. Die fünf Pausenlängen, ein globales **{{narrset.speed}}** (70–130 %) und der **{{narrset.style}}** („{{narrset.style_auto}}“, „{{narrset.style_scripture}}“, „{{narrset.style_fiction}}“, „{{narrset.style_dialogue}}“) werden unter „{{ui.settings_title}}“ → **{{narrset.title}}** (Kapitel 8) eingestellt und gelten ab der nächsten Vertonung. Das Tempo wird nach der Synthese per Zeitdehnung mit erhaltener Tonhöhe angewendet; Änderungen an Pausen oder Tempo sprechen fertige Fragmente daher nie neu – der nächste Lauf fügt sie nur neu zusammen.

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

**Das Aufnahmeskript.** Das Repository enthält fertige Skripte im Ordner `docs/recording-scripts/`, jeweils eigenständig auf Russisch, Englisch und Deutsch geschrieben (TXT und PDF, dunkel und zum Drucken). Jedes hat 17 Blöcke: Begrüßung, die Geschichte der Idee, Zahlen und Daten, fließende Sätze für die Laute der Sprache, neutrales Lesen, Ruhe, Freude, Traurigkeit, Ärger, Überraschung, Flüstern und Lautes, drei kurze Computergeschichten, optional eine Geschichte über einen missratenen Tag mit milden Kraftausdrücken (ohne Flüche), optional Fachbegriffe und – ganz am Ende – die Einverständniserklärung in der Sprache der Datei. Die PDFs enthalten Lesehinweise (Tempo, Pausen, Gefühle, Pausen zum Ausruhen). Das Skript selbst nennt drei einfache Regeln:

1. **Lesen Sie nur gewöhnliche Zeilen.** Alles in eckigen Klammern und alle Zeilen mit `===` sind Hinweise – nicht vorlesen.
2. **Eine Zeile – ein Satz.** Lesen, atmen, eine Sekunde schweigen, dann die nächste Zeile.
3. **Verhaspelt? Eine Sekunde schweigen und denselben Satz noch einmal von vorn lesen.** Die Aufnahme muss nicht gestoppt werden.

Weitere Hinweise aus dem Skript: Gefühle müssen nicht gespielt werden – es genügt, die Stimme ein wenig zu verändern, und wenn es nicht klappt, lesen Sie in normaler Stimme. Mit „OPTIONAL“ gekennzeichnete Blöcke dürfen übersprungen werden. Genau diese Skriptdatei wählen Sie im Trainingsfenster als „Text“.

> Voxprint ist darauf ausgelegt, wiederholt gelesene Sätze zu verkraften: Beim Abgleich der Aufnahme mit dem Skript wird für jede Zeile die beste Lesung behalten. Dennoch gilt: Je sauberer die Aufnahme, desto besser die Stimme.

**Die Einverständniserklärung (Block 18).** Beenden Sie die Aufnahme mit **einem** der Einverständnissätze – dem, der für Sie zutrifft: *kommerziell* (A), *öffentlich nichtkommerziell* (B) oder *nur privat* (C), auf Russisch, Englisch oder Deutsch. Sagen Sie anstelle der Wörter in eckigen Klammern Ihren eigenen Namen und das heutige Datum. Lesen Sie ihn nur, wenn die Stimme Ihre ist oder Sie die Erlaubnis des Inhabers haben. Einzelheiten in Abschnitt 6.6.

## Das Trainingsfenster

Öffnen Sie es über die Karte **{{studio.train_title}}**. {{ui.subtitle}}

![Das Trainingsfenster, oberer Teil: Audio und Text, Trainingsqualität, Einverständnis und die Hauptschaltfläche.](@train_top)

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
| ▸ **{{train.checks_options}}** – Kontrollkästchen „{{check.checkbox}}“ | Nach dem Training lässt Voxprint die neue Stimme einen Testsatz sprechen und prüft Tonhöhe und Verständlichkeit automatisch. Eingeschaltet lassen. |
| „{{ui.voice_gender_none}}“ und „{{ui.voice_age_none}}“, ▸ **{{ui.voice_more}}** | Optionale Angaben zur Stimme. Das Geschlecht („{{ui.voice_type_male}}“, „{{ui.voice_type_female}}“, „{{ui.voice_type_child}}“, „{{ui.voice_type_other}}“) wird in voice.json gespeichert und an den Ordnernamen der trainierten Stimme angehängt (`anna_male`, `anna_female`, `anna_unspecified`). Das Programm schlägt männlich oder weiblich anhand der Tonhöhe Ihrer Aufnahme vor; ändern Sie es, falls es nicht stimmt. „{{ui.voice_more}}“ enthält weitere optionale Felder für voice.json. |
| Beschreibungsfeld | „{{ui.voice_desc_placeholder}}“ |
| **{{ui.btn_merge}}** | Führt den trainierten Adapter zu einem eigenständigen Modellordner zusammen, der in jeder App mit Qwen3-TTS funktioniert (Abschnitt 6.7). |
| **{{ui.btn_dataset}}** | Bereitet nur den Datensatz vor (ausgerichtete und geschnittene Clips), ohne Training. Nützlich, wenn Sie woanders trainieren wollen. |
| Fortschrittsbalken und Etappen | {{stage.model}}, {{stage.align}}, {{stage.slice}}, {{stage.train}}, {{stage.save}} – die laufende Etappe ist hervorgehoben. Während ein Auftrag läuft, erscheint **{{ui.cancel}}**. |
| Statuszeile | Zu Beginn: „{{ui.status_idle}}“ |
| Zahnrad (oben rechts) | Öffnet „{{ui.settings_title}}“. |

![Das Trainingsfenster, unterer Teil: schnelle Vorschau, Name und Beschreibung, Geschlecht und Alter, Universalmodell, Datensatz und die Fortschrittsstufen.](@train_bottom)

Ist der Auftrag beendet, zeigt die Statuszeile „{{ui.voice_registered}}“ und eine Schaltfläche **{{ui.open_folder}}** erscheint. Der Trainingsordner ist `Projects\Voices\<Stimmenname>_Voxprint` im Projektordner: der Datensatz liegt in `dataset`, der Adapter (einige zehn MB) in `output\<Stimmenname>`.

Die Bilder zeigen den Standardmodus mit Text: **{{ui.choose_audio}}** und **{{ui.choose_text}}** untereinander, jeweils mit dem Namen der gewählten Datei. Das Kontrollkästchen „{{asr.checkbox}}“ schaltet in den Modus „nur Audio“ (Abschnitt 6.4): Die Textzeile wird zum optionalen Skript, und ein Block mit einer Warnung erscheint.

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

![Meine Stimmen (Build 667): Tirzah ist die mitgelieferte Stimme (CC0-1.0, schreibgeschützt). Das Bild zeigt außerdem Boaz (diese Stimme wird nicht mehr installiert) und die noch nicht geladene Offene Universalstimme.](@voices)

Jede Stimme ist eine Karte:

| Element | Bedeutung |
|---|---|
| Name und zwei Badges | Das **Lizenz-Badge** (grün „… · kommerzielle Nutzung erlaubt“ oder bernsteinfarben „Nur persönliche Nutzung · …“) und das **Umfang-Badge** ({{consent.badge_commercial}}, {{consent.badge_public_noncommercial}}, {{consent.badge_private_only}}). |
| Infozeile | Sprache · Minuten Sprache im Training · Epochen (Durchläufe) · Stimmtyp · Autor · (bei Online-Stimmen) Größe. |
| Beschreibung | Was der Autor geschrieben hat. Bei einer Stimme „nur zum Testen“ steht zusätzlich eine Warnung. |
| **{{voices.preview}}** | Spielt eine kurze Probe der Stimme. **{{voices.stop}}** hält sie an. Hat die Stimme keine Probe: „{{voices.no_preview}}“ |
| **{{voices.narrate}}** | Öffnet das Vertonungsfenster mit bereits gewählter Stimme. |
| **{{voices.edit}}** | Öffnet „{{voices.edit_title}}“: {{voices.field_name}}, {{voices.field_author}}, {{voices.field_license}}, {{voices.field_gender_age}}, {{voices.field_description}}. Drücken Sie **{{voices.save}}**. |
| **{{voices.export}}** | Speichert die Stimme für einen anderen Computer: „{{voices.export_small_plain}}“ (die trainierte Stimme; ein anderes Voxprint importiert sie in einer Sekunde) oder „{{voices.export_full_plain}}“ (das Universalmodell). |
| **{{voices.open_folder}}** | Öffnet den Ordner der Stimme. |
| **{{voices.delete}}** | Löscht die Stimme nach Rückfrage von diesem Computer: „{{voices.delete_confirm_plain}}“ Mitgelieferte Stimmen sind schreibgeschützt: **{{voices.edit}}** und **{{voices.delete}}** sind ausgegraut („{{voices.bundled_tip}}“). |

Die Fußzeile des Fensters wiederholt: „{{voices.rights_note}}“

Ist die Bibliothek leer, sagt das Fenster „{{voices.empty}}“ und bietet **{{voices.empty_train}}** an.

## Online-Stimmen und das Stimmen-Repository

Die dritte Karte im Bild oben, die *Offene Universalstimme*, ist so eine Online-Stimme: „{{voices.remote_state}}“ und eine Schaltfläche **{{voices.remote_download}}**.

* Stimmen aus dem Online-Index erscheinen von selbst als Karten mit dem Zustand „{{voices.remote_state}}“, ihrer Lizenz, der Größe und einer Schaltfläche **{{voices.remote_download}}**. Der Download wird per SHA-256-Prüfsumme kontrolliert und kann fortgesetzt werden.
* **{{voices.remote_refresh}}** lädt die Liste neu. Ohne Verbindung zeigt Voxprint die zuletzt bekannte Liste („{{voices.repo_offline}}“).
* **{{voices.repo_button}}** öffnet den Dialog „{{voices.repo_title}}“ mit **{{voices.repo_refresh}}**, einem Feld für die Adresse des Index (`index.json`) und **{{voices.repo_download}}**.

## Eine Stimme importieren

**{{voices.import}}** bietet zwei Wege: **{{voices.import_folder}}** (ein Adapterordner) und **{{voices.import_zip}}** (ein Stimmpaket). Archive werden auf unsichere Pfade und Größenlimits geprüft. Eine Stimme ohne angegebene Lizenz gilt als „nur persönliche Nutzung“.

## Lizenzen und die Stimme, die Voxprint anbietet

| Lizenz | Kommerzielle Nutzung |
|---|---|
| CC0-1.0, CC-BY-4.0, CC-BY-SA-4.0 | erlaubt (Namensnennung bzw. Weitergabe unter gleichen Bedingungen beachten, wenn die Lizenz es verlangt) |
| CC-BY-NC-4.0, CC-BY-NC-SA-4.0 | nicht erlaubt |
| custom/personal-only (Standard) | nicht erlaubt; Ergebnisse bleiben auf Ihrem Computer |
| custom/test-use-only | nicht erlaubt; nur zum Testen, keine Veröffentlichung |

* **Tirzah** (weiblich) – die offene **russische** Stimme, die mit Voxprint kommt (Teil des Standard-Modelldownloads, etwa eine halbe Stunde Sprache). Nur mit der gemeinfreien LibriVox-Aufnahme *Степные сказки (Stepnyia skazki)* von Grigori Danilewski trainiert, gelesen von Anastasiia Solokha. Lizenz **CC0-1.0** – für jede Nutzung frei, auch kommerziell; eine Namensnennung ist erwünscht. Der Name stammt von Voxprint und bedeutet keine Billigung durch die Sprecherin. In der Bibliothek ist sie schreibgeschützt („{{voices.bundled_note_plain}}“). Eine Hörprobe – Genesis 1,1–2,3 (russische Synodalübersetzung, gemeinfrei), ebenfalls CC0 – liegt im Repository-Ordner `samples/`. *Boaz* kam mit 0.1.3 und wird nicht mehr installiert.
* **Asher** (männlich) und **Noa** (weiblich) – optionale russische Katalogstimmen, nicht Teil des Standard-Downloads. Sie erscheinen im Stimmenkatalog der App, sobald ihre Pakete veröffentlicht sind. Asher ist mit *Teachings of Christ* von Leo Tolstoi trainiert, gelesen von Vladimir Anyanov; Noa mit *Izbrannye* von Scholem Alejchem, gelesen von Hanna Ponomarenko. Lizenz **CC0-1.0**.
* **Offene Universalstimme** (englisch: *Open universal voice*, russisch: *Открытый универсальный голос*) – eine englische Stimme für alle, die keine eigene Aufnahme haben. Nur mit dem gemeinfreien LJ-Speech-Datensatz trainiert (Sprecherin Linda Johnson, LibriVox); Lizenz **CC0-1.0** – für jede Nutzung frei, auch kommerziell; die Nennung des LJ-Speech-Datensatzes (Keith Ito) ist erwünscht.

Die Offene Universalstimme wird getrennt aus dem Stimmen-Repository geladen.

> Die Bilder in diesem Handbuch zeigen die mitgelieferten Stimmen. Stimmen, die Sie selbst trainieren, sind standardmäßig **nur für den persönlichen Gebrauch** (Umfang „{{consent.badge_private_only}}“): keine öffentliche Nutzung der Ausgabe, keine kommerziellen Projekte.

Das Badge ist eine Information, keine Rechtsberatung. Die Lizenz gilt nur für das Stimmenmodell.

# Einstellungen

Öffnen Sie **{{ui.settings_title}}** mit dem Zahnrad (oben rechts im Studio und in den anderen Fenstern).

![Einstellungen (Build 667; Projektpfad und Netzwerkadresse sind unkenntlich gemacht).](@settings)

| Bedienelement | Was es tut |
|---|---|
| **{{ui.language}}** | Die Oberflächensprache: English, Deutsch, Русский, Українська, Latviešu. Sie ändert sich sofort in allen Fenstern. |
| **{{ui.transparency}}** | Wie durchscheinend die Fenster sind: „{{ui.transparency_default}}“, „{{ui.transparency_more}}“ oder „{{ui.transparency_off}}“. |
| **{{ui.net_iface}}** | Über welchen Netzwerkadapter die Downloads laufen. „{{ui.net_iface_auto}}“: Kann ein Download keine Verbindung aufbauen (VPN, ungewöhnlicher Adapter), werden die anderen Schnittstellen probiert und die funktionierende gemerkt. |
| **{{asrmodel.label}}** | Welches Spracherkennungsmodell verwendet wird. „{{asrmodel.auto}}“ nimmt das 1.7B-Modell auf Karten ab etwa 8 GB Grafikspeicher, sonst 0.6B. Die Zeile unter dem Kontrollkästchen zeigt das verwendete Modell. |
| **{{preload.option}}** | Lädt nach dem Start die installierten Sprachmodelle im Hintergrund in den Arbeitsspeicher, damit eine Vertonung sofort beginnt. Daneben steht der verfügbare RAM. |
| **{{narrset.title}}** | {{narrset.comma}}, {{narrset.mid}}, {{narrset.sentence}}, {{narrset.paragraph}}, {{narrset.chapter}} – die fünf Pausenlängen in Sekunden (Standard 0,25 / 0,40 / 0,60 / 1,00 / 2,00 s); **{{narrset.speed}}** (70–130 %; 100 % ist das eigene Tempo der Stimme); **{{narrset.style}}**. Siehe Abschnitt 5.4. |
| **{{narrset.defaults}}** | Stellt die Standardpausen und das Standardtempo wieder her. |
| **{{ui.btn_update}}** | Prüft, ob neuere verifizierte Versionen von Komponenten oder Modellen existieren, und installiert sie (mit Sicherheitsprüfung und automatischem Zurückrollen). Voxprint prüft außerdem einmal pro Woche unauffällig. Meldungen: „{{upd.up_to_date}}“, „{{upd.restart}}“ |
| **{{ui.settings_models_folder}}** | Öffnet den Ordner mit den heruntergeladenen Modellen. |
| **{{ui.settings_data_folder}}** | Öffnet den Daten- und Protokollordner. |
| **{{diag.button}}** | Speichert die Protokolldateien, Angaben zu diesem Computer (System, GPU, Speicher) und die Einstellungen als eine ZIP-Datei; die Namen Ihrer Dateien werden entfernt. Hängen Sie sie an eine Fehlermeldung an. |
| **{{modules.title}}** | Öffnet das Komponenten-Fenster: was installiert ist, Updates und der vollständige Download. |
| **{{projects.title}}** | Wo neue Projekte (Hörbücher, Stimmtrainings, Neu vertonen) gespeichert werden. **{{projects.open}}**, **{{projects.change}}** (am besten ein lokales Laufwerk mit viel Platz, das nicht mit der Cloud synchronisiert wird), **{{projects.default}}**. Bestehende Projekte bleiben, wo sie sind. |
| **{{backup.title}}** – **{{backup.include_models}}**, **{{backup.include_voices}}** | Was in eine Sicherung kommt (beides standardmäßig an). |
| **{{backup.link_models}}** | Beim Wiederherstellen die Modelle vom Sicherungslaufwerk lesen, statt sie zu kopieren. Das Laufwerk muss angeschlossen bleiben. |
| **{{backup.btn_backup}}** | Kopiert Modelle (und Stimmen) in einen gewählten Ordner oder auf ein Laufwerk. Zeigt vorab Größe und freien Platz; fortsetzbar; identische Dateien werden übersprungen; jede Kopie wird geprüft. Fortschrittsbalken und eine Schaltfläche **{{ui.cancel}}** erscheinen. |
| **{{backup.btn_restore}}**, **{{backup.btn_restore_folder}}** | Stellt aus einem Ordner `Voxprint-backup` wieder her und prüft jede Datei per Prüfsumme. Stimmen, die schon mit anderem Inhalt existieren, werden nie überschrieben; beschädigte oder fehlende Dateien werden wie üblich geladen. |
| **{{existing.title}}** | {{existing.hint}} **{{existing.choose}}** wählt den Ordner, **{{existing.clear}}** vergisst ihn. |
| **{{autorepair.title}}** – **{{autorepair.button}}** | {{autorepair.desc}} Siehe Abschnitt 8.2. |
| **{{ui.about}}** | Version, die Idee, Funktionsweise, Open-Source-Komponenten mit ihren Lizenzen, ein Link zum GitHub-Repository und Hinweise zu Drittsoftware. |

Die unterste Zeile zeigt die Version, zum Beispiel *Voxprint AI Audiobook Builder 0.1.3-beta · build 667 "Menuchah"*.

Bemerkt das Programm eine beschädigte Installation, bietet ein Banner die Reparatur an („{{ui.repair_offer_plain}}“). Hilft **{{autorepair.button}}** nicht, starten Sie das Voxprint-Installationsprogramm erneut – Stimmen und Modelle bleiben erhalten.

Findet das Programm eine ältere Komponente in einer Umgebung, die ihm nicht gehört (zum Beispiel Ihr eigenes Python), fragt es zuerst: „{{upg.title}}“ mit **{{upg.btn_upgrade}}** oder **{{upg.btn_later}}**; Ihre Umgebung ändert es nie stillschweigend.

## Maximale Qualität ab Werk

Ab Werk ist alles, was automatisch laufen kann, **bereits gewählt** und mit einem Stern und dem Wort „{{auto.recommended}}“ markiert: Wenn Sie nichts anfassen, arbeitet Voxprint mit maximaler Qualität. Die dafür nötigen Modelle gehören zum Download beim ersten Start, es ist nichts extra anzuklicken. Die einzelnen Kontrollkästchen bleiben in ihren Fenstern.

* Textaufbereitung („Buch vertonen“): ein Schalter, „{{prep.one}}“, standardmäßig an (Layout, Fußnoten, Anführungszeichen, Links, Überschriften, Zahlen, Abkürzungen und russische Tippfehler, wenn das Modell heruntergeladen ist).
* Übersetzung des Buches vor dem Vertonen (Englisch, Russisch, Deutsch; Opus-MT-Modelle, etwa 300 MB je Richtung, einmal geladen). Aus, bis Sie sie einschalten.
* Trainingsfenster: „{{check.checkbox}}“ und „{{preview.compare}}“ (beide brauchen das Spracherkennungsmodell, etwa 1,9 GB, einmal geladen).
* Immer aktiv, ohne Schalter: Ausrichtung des Textes auf das Audio mit Plausibilitätsprüfung, Audio-Qualitätsfilter, Erkennungsfilter des Nur-Audio-Modus.
* Nicht enthalten, weil es sie noch nicht gibt: ein Zeichensetzungsmodell. Russische Betonungszeichen werden nicht gesetzt, weil das Sprachmodell sie nicht liest; der Buchstabe jo gehört zu „Text vorbereiten“. Sprechermarken sind eine eigene Option auf der Karte des KI-Textmodells.

Die Trainingsvorgabe bleibt absichtlich bei „{{preset.balanced}}“: In unseren Tests ließ längeres Training mit höherer Lernrate die Stimme nuscheln, mehr ist dort also nicht besser. Stattdessen wird nach „{{preview.compare}}“ die bessere der beiden Varianten als „{{auto.recommended}}“ markiert (zuerst das Urteil, dann weniger Erkennungsfehler, dann die kleinere Tonhöhenabweichung; ohne klaren Unterschied Variante A, die günstiger ist). Die Entscheidung bleibt bei Ihnen: Hören Sie immer hin.

## Prüfen und reparieren

**{{autorepair.button}}** prüft das Programm, seine Komponenten und jede Modelldatei einzeln anhand ihrer SHA-256-Prüfsumme und lädt nur, was fehlt oder beschädigt ist. Ihre Stimmen und Bücher bleiben unberührt. Während der Prüfung zeigen ein Fortschrittsbalken und eine Schrittzeile (zum Beispiel *Step 8 of 21: Checking Qwen3-ASR-0.6B: model.safetensors*, deutsch „Schritt 8 von 21: …“), wo sie steht, und die Schaltfläche wird zu **{{autorepair.stop}}**; eine angehaltene Prüfung behält, was fertig ist.

![Prüfen und reparieren während der Prüfung: Fortschrittsbalken und aktueller Schritt.](@repair_running)

Am Ende erscheint eine Zusammenfassung, zum Beispiel *Check finished: 24 checked, 0 repaired or downloaded, 0 failed.* (deutsch „Prüfung beendet: …“). Was nicht repariert werden konnte, steht darunter.

![Prüfen und reparieren abgeschlossen: die Zusammenfassung.](@repair_done)

> Bekanntes Problem in Build 667: Während der Prüfung wird die rechte Spalte der Einstellungen gestaucht, und die Erklärtexte überlappen sich (erstes Bild). Das ist nur optisch und wird in einem der nächsten Builds behoben.

Dieselbe Prüfung läuft ohne Fenster als `Voxprint.exe --auto-repair` (Kapitel 11).

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

<dt>Prüfen und reparieren</dt><dd>Prüft jede Komponenten- und Modelldatei per Prüfsumme und lädt nur Fehlendes oder Beschädigtes („{{ui.settings_title}}“ → „{{autorepair.button}}“ oder <code>--auto-repair</code>).</dd>

<dt>Exakte Pausen</dt><dd>Stille fester Länge, die Voxprint zwischen den gesprochenen Stücken einfügt – nach einem Komma, einem Satz, einem Absatz, einem Kapitel –, unabhängig davon, wie das Stimmmodell den Text phrasiert.</dd>

<dt>Mitgelieferte Stimmen</dt><dd>Tirzah: die offene russische Stimme (CC0-1.0), die mit Voxprint kommt und in der Bibliothek schreibgeschützt ist. Boaz kam mit 0.1.3 und wird nicht mehr installiert.</dd>

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
| {{err.narration_no_ffmpeg}} | Einstellungen → **{{autorepair.button}}**, oder wählen Sie WAV. |
| {{err.book_unsupported}} | Wandeln Sie das Buch in TXT, FB2 oder EPUB um. |
| {{err.book_unsafe}} | Die Datei wirkt beschädigt oder zu groß; versuchen Sie eine andere Kopie. |
| {{warn.no_gpu}} | Training ist ohne NVIDIA-Karte möglich, aber extrem langsam. |
| {{warn.loss_low}} | Die Stimme kann brabbeln. Wählen Sie **{{preset.balanced}}**, nehmen Sie mehr Text auf und hören Sie die schnelle Vorschau an. |

**Die Stimme klingt höher als meine.** Eine leichte Tonhöhen-Drift nach oben (bei einer männlichen Stimme wurden etwa +2…+3 Halbtöne beobachtet) ist ein bekannter Effekt dieser Art von Training. Die automatische Prüfung warnt oberhalb von +4. Probieren Sie die schnelle Vorschau mit beiden Varianten und nehmen Sie die näherliegende.

**Die Stimme brabbelt oder hört nicht auf.** Zu viele Durchläufe oder zu hohe Lernrate. Wählen Sie „{{preset.balanced}}“ und erhöhen Sie die Lernrate nie von Hand.

**Kann ich anhalten und später weitermachen?** Vertonung: ja, immer (Abschnitt 5.5). Downloads und Sicherungen: ja. Training: Abbrechen ist möglich, ein voller Lauf beginnt aber von vorn; nutzen Sie vorher die schnelle Vorschau.

**Darf ich meine Stimme für ein veröffentlichtes Hörbuch verwenden?** Nur wenn der Einverständnis-Umfang es erlaubt („{{consent.badge_commercial}}“ zum Verkaufen; „{{consent.badge_public_noncommercial}}“ zum kostenlosen Veröffentlichen). Ohne Einverständniserklärung trainierte Stimmen sind *nur privat*. Die Lizenz können Sie unter **{{voices.edit}}** ändern, wenn die Stimme Ihnen gehört.

**Schickt Voxprint meine Aufnahmen irgendwohin?** Nein. Ins Internet gehen nur der Download von Modellen, die Update-Prüfung und – wenn Sie das Repository öffnen – das Abrufen des Stimmenindex und der von Ihnen gewählten Stimmen. Keine Telemetrie, kein Konto.

**Welche Sprachen werden unterstützt?** Die Oberfläche: English, Deutsch, Русский, Українська, Latviešu. Die Textvorbereitung deckt Russisch und Englisch vollständig ab. Das Sprachmodell kann weitere Sprachen, sie sind aber nicht getestet.

**Wo ist das Protokoll?** Schaltfläche „{{ui.settings_data_folder}}“ in den Einstellungen. Schicken Sie die Datei von dort, wenn ein Fehler wiederkehrt.

**Ist die Option AAC / M4B sicher zu verwenden?** Sie ist optional und standardmäßig aus gutem Grund ausgeschaltet: AAC ist patentbehaftet, und für die Einhaltung der Gesetze sind Sie verantwortlich (Abschnitt 5.4).

# Kommandozeile

Alles in diesem Kapitel ist optional: Die Fenster können dasselbe. Die Kommandozeile ist für Skripte, Server und KI-Agenten gedacht und nutzt dieselbe Verarbeitung wie die Fenster. Die vollständige Referenz ist `docs/CLI.md`, die Datei für Agenten `docs/AGENTS.md` (beide im Repository, auf Englisch). Die Meldungen der Kommandozeile selbst sind englisch.

## Wo das Programm liegt und die Regeln

Das installierte Programm ist `C:\Program Files\Voxprint\Voxprint.exe` (ein anderer Ordner, wenn Sie einen gewählt haben; `where.exe Voxprint` findet es). In einem Quellcode-Checkout verwenden Sie `python main.py`. Unten steht `voxprint` für beides.

* Nichts fragt nach; `--yes` wird von jedem Befehl akzeptiert.
* `--json` gibt den Fortschritt als JSON-Zeilen aus und endet mit einem Ergebnisobjekt (`type: result`, `ok`, `exit_code`, `outputs`, `warnings`, `error`, `hint`).
* `voxprint --help` und `voxprint <Befehl> --help` zeigen kopierfertige Beispiele; `voxprint --version` gibt zum Beispiel `Voxprint 0.1.3-beta build 667 "Menuchah"` aus.
* Setzen Sie `VOXPRINT_LANG=en`, wenn Sie die Meldungen der Verarbeitung maschinell auswerten.

| Exit-Code | Bedeutung |
|---|---|
| 0 | Erfolg |
| 1 | Unerwarteter Fehler – `voxprint diag` speichern |
| 2 | Fehlende oder unbekannte Argumente |
| 3 | Buch, Audio, Text, Stimme oder Sicherung fehlt oder ist nicht lesbar |
| 4 | Ein Modell oder eine Komponente ist nicht installiert, oder der Download ist fehlgeschlagen |
| 5 | Grafikspeicher reicht nicht – beim Training mit `--force-cpu` wiederholen |
| 6 | Abgebrochen – denselben Befehl erneut ausführen, um fortzusetzen |

## Ein Buch vertonen

```
voxprint narrate BOOK --voice ID_OR_NAME --out DIR [--format mp3,m4b,opus,...]
    [--pause-comma S] [--pause-mid S] [--pause-sentence S] [--pause-paragraph S]
    [--pause-chapter S] [--speed X] [--style auto|scripture|fiction|dialogue]
    [--pauses] [--ai-disclosure] [--json]
```

Das Buch ist eine TXT-, FB2-, `.fb2.zip`- oder EPUB-Datei; das Ergebnis landet in `<DIR>/<Buchtitel>/`; Standardformat ist eine Opus-Datei. Die Pausen-Flags (Sekunden), `--speed` (0,7–1,3) und `--style` ersetzen für einen Lauf die Werte aus „{{ui.settings_title}}“ → „{{narrset.title}}“; `--pauses` schaltet das Schneiden an jedem Komma ein. Wird derselbe Befehl erneut ausgeführt, überspringt er bereits fertige Fragmente, und Änderungen an Pausen oder Tempo sprechen sie nie neu.

```
voxprint narrate genesis.txt --voice tirzah --out ./audiobooks --format mp3 --json
```

## Eine Stimme trainieren

```
voxprint train AUDIO [--text SCRIPT] [--name NAME] [--type male|female|child|other]
    [--out DIR] [--consent none|auto|commercial|public_noncommercial|private_only]
    [--speaker NAME] [--license ID] [--language CODE] [--force-cpu] [--json]
```

Mit `--text` wird die Aufnahme am Skript ausgerichtet; ohne (eine Datei oder ein Ordner mit Clips) baut die Spracherkennung den Datensatz. `--out` ist der übergeordnete Ordner; jede Stimme bekommt ihren eigenen Ordner `<Stimmenname>_Voxprint`. Ohne `--consent` wird die Stimme als *nur privat* gespeichert; `--consent auto` liest die gesprochene Erklärung, die anderen Werte halten einen selbst bestätigten Umfang fest, und `--license` erlaubt nie mehr als das Einverständnis. Trainieren Sie nur eine Stimme, die Sie nutzen dürfen.

```
voxprint train recording.wav --text script.txt --name Anna --type female --consent auto
```

## Sichern und wiederherstellen

```
voxprint backup --out DIR [--no-models] [--no-voices] [--json]
voxprint restore --from DIR [--link] [--json]
```

`backup` kopiert die Modelle und die Stimmenbibliothek nach `<DIR>/Voxprint-backup/` mit dem Verzeichnis `voxprint-backup.json` (Version, Build, Größe und SHA-256 jeder Datei). Dateien, die dort schon mit gleicher Größe und gleichem Hash liegen, werden übersprungen, sodass eine unterbrochene Kopie weiterläuft; reicht der Platz nicht, wird nichts geschrieben. `restore` kopiert die Sicherung in die normalen Ordner und prüft jede Datei; eine beschädigte oder fehlende Datei wird genannt und später wie üblich geladen. `--link` lässt die Modelle auf dem Sicherungslaufwerk (es muss angeschlossen bleiben) und kopiert nur die Stimmen. Fehler enden mit Code 3. Es sind dieselben Aufträge wie die Schaltflächen unter „{{ui.settings_title}}“ → „{{backup.title}}“.

## Die Installation prüfen

```
voxprint status --json                 # Version, GPU, Stimmen, Formate, Modelle
voxprint models download required      # TTS, Ausrichtung und Spracherkennung, falls sie fehlen
Voxprint.exe --verify-install          # schnelle Installationsprüfung mit Ursachen-Codes
Voxprint.exe --auto-repair             # Prüfen und reparieren: jede Datei per SHA-256
voxprint diag --out report.zip         # Diagnosebericht
```

`status` (auch `capabilities`) gibt ein JSON-Objekt aus und lädt nichts herunter – führen Sie es vor `train` oder `narrate` aus. `--auto-repair` ist derselbe Auftrag wie „{{ui.settings_title}}“ → **{{autorepair.button}}** (Abschnitt 8.2): eine Zeile je geprüftem Element, nur beschädigte oder fehlende Dateien werden geladen, Exit-Code 0, wenn alles in Ordnung ist; der Bericht steht auch in `logs\auto_repair.txt`. `--verify-install` schreibt `logs\verify_install.txt`. `diag` entspricht **{{diag.button}}**.

# Lizenzen und Verantwortung

* **Der eigene Quellcode von Voxprint steht unter der Apache License 2.0** (Copyright 2026 Aleksandr Mitroshenkov; siehe die Dateien `LICENSE` und `NOTICE`).
* **Die Apache-Lizenz des Codes gilt nicht für Modelle und Stimmen.** Die Modelle, die Voxprint herunterlädt (Qwen3-TTS, Qwen3-ASR und die optionalen Textmodelle), behalten die Lizenzen ihrer Autoren. Jede Stimme hat ihre eigene Lizenz, als Badge angezeigt und in `voice.json` gespeichert.
* Komponenten von Drittanbietern samt Lizenzen sind unter „{{ui.about}}“ sowie in `THIRD_PARTY_NOTICES.md` und im Ordner `licenses` der Installation aufgeführt. Qt/PySide6 wird unter der LGPL-3.0 verwendet; das mitgelieferte ffmpeg ist ein eigenes Programm mit eigener Lizenz; der AAC-Codec ist patentbehaftet.
* **Verwenden Sie Stimmen verantwortungsvoll.** Eine Stimme ist ein personenbezogenes Datum und ein Persönlichkeitsrecht. Verwenden Sie eine Stimme nur mit ausdrücklicher Erlaubnis ihres Inhabers und im Rahmen ihrer Lizenz; keine Nachahmung von Personen, kein Betrug, keine irreführenden Inhalte; kennzeichnen Sie synthetische Sprache, wo dies vorgeschrieben ist. Für die rechtmäßige Verwendung der von Ihnen erzeugten Stimmen und Audios sind Sie verantwortlich.

> **Dieses Handbuch und die Lizenz-Badges sind Informationen, keine Rechtsberatung.** Die Einverständniskarte hält fest, was eine sprechende Person gesagt hat; sie ersetzt nicht das Recht Ihres Landes.

*Voxprint AI Audiobook Builder 0.1.3 (Beta), Build 667 „Menuchah“. Handbuch vom 9. Oktober 2026.*

*Voxprint AI Audiobook Builder ist ein unabhängiges Projekt und steht in keiner Verbindung zu anderen Produkten oder Diensten mit ähnlichen Namen.*
