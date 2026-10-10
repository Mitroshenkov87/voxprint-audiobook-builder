; Voxprint installer (Inno Setup 6.x). Build with:  ISCC installer\Voxprint.iss   (or through build.bat)
; For the --onedir variant:  ISCC /DONEDIR installer\Voxprint.iss
; Windows 11 x64 only. The models are not part of the installer: the app downloads them itself on first start
; (after the installation, Voxprint can be started right away with the --prefetch flag).
; Wizard page "Setup type" (decided 2026-10-08; infra/setup_mode.py): exactly two choices, written to state\install_mode.txt.
;  * Full (default): the first start downloads EVERYTHING without a further click - the components and ALL models including the
;    optional ones (about FullModelsMB + FullRuntimeMB below, from the pinned sizes); the page checks the free space first.
;  * Quick: only the program is installed; the first start opens the Components window, which OFFERS the same complete download.
;  Nothing is cut in either mode: the end result is identical.  Silent: /Mode=full (default) or /Mode=quick.
; Wizard page "Models folder" (after the install folder): where the voice models (~15 GB) live. Default = the current location
; %LOCALAPPDATA%\Voxprint\models; the user may pick another folder or drive.  Only paths are REMEMBERED; the installer copies no model.
; Two cases (decided 2026-10-05):
;  * a normal folder (empty, or with models in place): it becomes the models folder (%LOCALAPPDATA%\Voxprint\state\models_dir.txt,
;    read by infra/paths.py: models_dir).  The app (1) keeps using models already complete in the default Local folder, (2) picks
;    up models the chosen folder already holds (Voxprint layout, Hugging Face cache), (3) downloads everything missing INTO it.
;    Program Files is refused (the app runs without admin rights and could not write there).
;  * a Voxprint BACKUP (the folder holds voxprint-backup.json, or Voxprint-backup\voxprint-backup.json, or it is the models\ /
;    voices\ folder of a backup; BackupRootOf = infra/paths.py: backup_root_of): it is a RESTORE SOURCE, never the models folder.
;    models_dir.txt is removed (the default Local folder stays the live store) and the backup root goes to
;    state\existing_models_dir.txt; on the first start the app restores models, voices and ffmpeg from it into the normal folders
;    (infra/existing_models.py: restore_backup).  The backup (often on an external drive) is only read.
; Voices and settings stay in %LOCALAPPDATA%\Voxprint.
; Silent: /ModelsFolder="D:\Voxprint models".  The older /ModelsDir="D:\old\models" (a folder to IMPORT models from, written to
; state\existing_models_dir.txt) still works.  Wizard languages: English, Russian, German.
;
; ONLINE variant:  ISCC /DONLINE /DONEDIR /DManifestUrl=<https://.../manifest-beta.json> installer\Voxprint.iss  ->  Output\Voxprint-Setup-online.exe
; A small installer (a few MB + vc_redist): it embeds build\online\voxprint-fetch.exe (tools/online_fetch.py), which downloads the
; payload parts listed in the manifest (release assets, each < 2 GiB, verified by SHA-256, resumable, parts that are already
; installed are skipped) and unpacks them into {app}.  One UAC prompt (PrivilegesRequired=admin) covers everything; the
; user data stays in %LOCALAPPDATA%\Voxprint.  /Manifest=<url or file> overrides the baked-in manifest (tests, mirrors).
; PORTABLE SETUP FOLDER (online variant + /DPORTABLE, docs/THIN-INSTALLER.md; NOT in the default build yet): an optional wizard page "Keep a portable setup folder" (default
; Documents\Voxprint Portable).  When ticked, voxprint-fetch keeps ALL components (and the models this PC needs) in that folder with
; manifest.json + SHA256SUMS.txt, installs from it, and the folder is remembered (state\portable_dir.txt).  A later run finds the folder
; (/FromFolder=<dir>, the remembered one, Documents\Voxprint Portable, or next to the installer) and installs without internet; with
; internet only the changed parts are fetched.  Silent: /Portable=1 [/PortableDir=<dir>] [/PortableModels=none|auto|all] [/FromFolder=<dir>].
; THIN variant (add /DTHIN to the ONLINE command; see docs/THIN-INSTALLER.md): only the components with role "core" (the small
; UI shell built by build_thin.bat) are installed -> Output\Voxprint-Setup-thin.exe; the app downloads the heavy runtime modules
; (PyTorch ...) itself on its first start (infra/modules.py, window "Components").

#define AppName "Voxprint"
; Name shown to the user (wizard, Start menu, Apps list). AppName stays technical: it is the install folder and the data folder name.
#define AppDisplayName "Voxprint AI Audiobook Builder"
#define AppVersion "1.0.0-rc"
; Windows version resources are four numbers. The pre-release suffix stays in AppVersion.
#define AppVersionInfo "1.0.0"
#define AppExe "Voxprint.exe"
; CI build number and codename (tools/build_number.py; build_online.ps1 passes /DAppBuild= /DAppCodename=); 0 = local build
#ifndef AppBuild
#define AppBuild "0"
#endif
#ifndef AppCodename
#define AppCodename ""
#endif
; Pinned sizes of the complete download in MiB (infra/setup_mode.py: full_sizes; tests/test_setup_modes.py keeps them in step)
#define FullModelsMB "25383"
#define FullRuntimeMB "2895"
#ifndef ManifestUrl
#define ManifestUrl "https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/latest/download/manifest-stable.json"
#endif
; The newest release's manifest (build_online.ps1 passes it for GitHub builds): the fallback when a file of the pinned set fails,
; and the first choice when "Try the latest component versions" is ticked (silent: /Latest=1).  Empty = pinned only.
#ifndef LatestManifestUrl
#define LatestManifestUrl ""
#endif

[Setup]
AppId={{6F1D2B7A-3C54-4E0B-9A41-7B5E0C9D2F18}
AppName={#AppDisplayName}
AppVersion={#AppVersion}
VersionInfoVersion={#AppVersionInfo}.{#AppBuild}
VersionInfoProductVersion={#AppVersionInfo}.{#AppBuild}
#if AppBuild != "0"
AppVerName={#AppDisplayName} {#AppVersion} build {#AppBuild} {#AppCodename}
#endif
AppPublisher=Voxprint
DefaultDirName={autopf}\{#AppName}
; One Start menu folder "Voxprint" for every Voxprint program (Audiobook Builder, Movie Dubber); each program removes only its
; own shortcut on uninstall, and Windows / Inno remove the folder only when it is empty.  UsePreviousGroup=no moves the entry of
; builds up to 703 (folder "Voxprint AI Audiobook Builder") into it; [InstallDelete] removes the old one.
DefaultGroupName=Voxprint
UsePreviousGroup=no
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExe}
SetupIconFile=..\assets\voxprint-setup.ico
OutputDir=Output
#ifdef ONLINE
#ifdef THIN
OutputBaseFilename=Voxprint-Setup-thin
ExtraDiskSpaceRequired=800000000
#else
OutputBaseFilename=Voxprint-Setup-online
ExtraDiskSpaceRequired=4500000000
#endif
#else
OutputBaseFilename=Voxprint-Setup
#endif
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; End User Agreement (plain text generated from docs/legal/EULA-audiobook-builder.md by tools/gen_eula_txt.py)
LicenseFile=..\docs\legal\EULA-audiobook-builder.txt
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Windows 11 24H2+ (build 26100). Older versions are refused: [Messages] before the wizard, and
; InitializeSetup again with the detected build number (English, Russian, German).
MinVersion=10.0.26100
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"

[Messages]
english.WinVersionTooLowError=Voxprint AI Audiobook Builder requires Windows 11 24H2 or newer (build 26100). This version of Windows is older and is not supported. Setup will exit.
russian.WinVersionTooLowError=Voxprint AI Audiobook Builder требует Windows 11 24H2 или новее (сборка 26100). Эта версия Windows старше и не поддерживается. Установка будет завершена.
german.WinVersionTooLowError=Voxprint AI Audiobook Builder benötigt Windows 11 24H2 oder neuer (Build 26100). Diese Windows-Version ist älter und wird nicht unterstützt. Die Installation wird beendet.

[CustomMessages]
english.RunPrefetch=Start %1 now
russian.RunPrefetch=Запустить %1 сейчас
german.RunPrefetch=%1 jetzt starten
english.ModePageCaption=Setup type
russian.ModePageCaption=Тип установки
german.ModePageCaption=Installationsart
english.ModePageDescription=When should the components and models be downloaded?
russian.ModePageDescription=Когда скачивать компоненты и модели?
german.ModePageDescription=Wann sollen die Komponenten und Modelle geladen werden?
english.ModePageSubCaption=Both choices end with the same complete program; only the moment of the download differs.
russian.ModePageSubCaption=Оба варианта дают одну и ту же полную программу; отличается только момент загрузки.
german.ModePageSubCaption=Beide Varianten ergeben dasselbe vollständige Programm; nur der Zeitpunkt des Downloads ist anders.
english.ModeFull=Full (recommended): on the first start everything is downloaded without further questions - the components and ALL models, including the optional ones (about %1 GB)
russian.ModeFull=Полная (рекомендуется): при первом запуске всё скачивается без лишних вопросов - компоненты и ВСЕ модели, включая дополнительные (около %1 ГБ)
german.ModeFull=Vollständig (empfohlen): Beim ersten Start wird alles ohne weitere Fragen geladen - die Komponenten und ALLE Modelle, auch die optionalen (ca. %1 GB)
english.ModeQuick=Quick: install only the program now; on the first start the Components window offers the same complete download (about %1 GB)
russian.ModeQuick=Быстрая: сейчас установить только программу; при первом запуске окно «Компоненты» предложит ту же полную загрузку (около %1 ГБ)
german.ModeQuick=Schnell: jetzt nur das Programm installieren; beim ersten Start bietet das Fenster „Komponenten" denselben vollständigen Download an (ca. %1 GB)
english.ModeNoSpace=Not enough free space for the full download on %1: about %2 GB are needed, %3 GB are free.%n%nChoose a models folder on another drive (Back), free some space, or choose Quick setup.
russian.ModeNoSpace=Недостаточно места для полной загрузки на %1: нужно около %2 ГБ, свободно %3 ГБ.%n%nВыберите папку моделей на другом диске (Назад), освободите место или выберите быструю установку.
german.ModeNoSpace=Nicht genug freier Speicher für den vollständigen Download auf %1: etwa %2 GB werden benötigt, %3 GB sind frei.%n%nWählen Sie einen Modellordner auf einem anderen Laufwerk (Zurück), schaffen Sie Platz oder wählen Sie die schnelle Installation.
english.VcRedistStatus=Installing Microsoft Visual C++ components...
russian.VcRedistStatus=Устанавливаю компоненты Microsoft Visual C++…
german.VcRedistStatus=Microsoft Visual C++-Komponenten werden installiert...
english.WinBuildTooOld=%1 requires Windows 11 24H2 or newer (build 26100). This PC is Windows build %2, which is older. Setup will exit.
russian.WinBuildTooOld=%1 требует Windows 11 24H2 или новее (сборка 26100). На этом ПК сборка Windows %2 — она старше. Установка будет завершена.
german.WinBuildTooOld=%1 benötigt Windows 11 24H2 oder neuer (Build 26100). Dieser PC hat Windows-Build %2 und ist damit älter. Die Installation wird beendet.
english.GpuRequired=Voxprint needs an NVIDIA GeForce RTX 40-series or newer GPU. Detected: %1.
russian.GpuRequired=Voxprint нужна видеокарта NVIDIA GeForce RTX 40-й серии или новее. Обнаружено: %1.
german.GpuRequired=Voxprint benötigt eine NVIDIA GeForce RTX der 40er-Serie oder neuer. Erkannt: %1.
english.GpuNone=none
russian.GpuNone=нет
german.GpuNone=keine
english.ModelsPageCaption=Models folder
russian.ModelsPageCaption=Папка моделей
german.ModelsPageCaption=Modellordner
english.ModelsPageDescription=Where should the models (all of them: about %1 GB) be stored?
russian.ModelsPageDescription=Где хранить модели (все вместе: около %1 ГБ)?
german.ModelsPageDescription=Wo sollen die Modelle (alle zusammen: ca. %1 GB) gespeichert werden?
english.ModelsPageSubCaption=Keep the default or choose another folder or drive. An empty folder becomes the models folder (models are downloaded into it); models it already holds are used. A folder with a Voxprint backup (voxprint-backup.json) is only read: on the first start its models and voices are restored into the default folder. Your voices and settings stay in the user profile.
russian.ModelsPageSubCaption=Оставьте папку по умолчанию или выберите другую папку или диск. Пустая папка становится папкой моделей (модели скачиваются в неё); уже имеющиеся в ней модели используются. Папка с резервной копией Voxprint (voxprint-backup.json) только читается: при первом запуске модели и голоса восстанавливаются из неё в папку по умолчанию. Ваши голоса и настройки остаются в профиле пользователя.
german.ModelsPageSubCaption=Behalten Sie den Standard oder wählen Sie einen anderen Ordner oder ein anderes Laufwerk. Ein leerer Ordner wird zum Modellordner (die Modelle werden dorthin geladen); vorhandene Modelle werden verwendet. Ein Ordner mit einer Voxprint-Sicherung (voxprint-backup.json) wird nur gelesen: Beim ersten Start werden Modelle und Stimmen daraus in den Standardordner wiederhergestellt. Ihre Stimmen und Einstellungen bleiben im Benutzerprofil.
english.ModelsPagePrompt=Models folder:
russian.ModelsPagePrompt=Папка моделей:
german.ModelsPagePrompt=Modellordner:
english.ModelsPageBadFolder=The folder %1 cannot be created or is not writable. Choose another folder.
russian.ModelsPageBadFolder=Папку %1 не удаётся создать или в неё нельзя записывать. Выберите другую папку.
german.ModelsPageBadFolder=Der Ordner %1 kann nicht angelegt werden oder ist nicht beschreibbar. Wählen Sie einen anderen Ordner.
english.ModelsPageProtected=The folder %1 is inside Program Files: the program runs without administrator rights and could not download models there. Choose another folder.
russian.ModelsPageProtected=Папка %1 находится в Program Files: программа работает без прав администратора и не сможет скачивать туда модели. Выберите другую папку.
german.ModelsPageProtected=Der Ordner %1 liegt in Program Files: Das Programm läuft ohne Administratorrechte und könnte dort keine Modelle speichern. Wählen Sie einen anderen Ordner.
english.ModelsPageBackupFound=The folder %1 contains a Voxprint backup. It will not be used as the models folder: on the first start its models and voices are copied into %2 (the backup itself is only read).
russian.ModelsPageBackupFound=В папке %1 найдена резервная копия Voxprint. Она не станет папкой моделей: при первом запуске модели и голоса будут скопированы из неё в %2 (сама копия только читается).
german.ModelsPageBackupFound=Der Ordner %1 enthält eine Voxprint-Sicherung. Er wird nicht als Modellordner verwendet: Beim ersten Start werden Modelle und Stimmen nach %2 kopiert (die Sicherung selbst wird nur gelesen).

english.LatestCheck=Try the latest component versions (falls back to the verified ones)
russian.LatestCheck=Попробовать последние версии компонентов (с откатом на проверенные)
german.LatestCheck=Neueste Komponentenversionen versuchen (sonst die geprüften)
english.OnlineStatus=Downloading and unpacking the Voxprint components (the download can be resumed if it is interrupted)...
russian.OnlineStatus=Загрузка и распаковка компонентов Voxprint (при обрыве загрузку можно продолжить)...
german.OnlineStatus=Voxprint-Komponenten werden geladen und entpackt (bei einem Abbruch kann der Download fortgesetzt werden)...
english.OnlineFailed=The download of the Voxprint components failed:%n%n%1%n%nCheck the internet connection and run the setup again: finished parts are kept and the download continues where it stopped.
russian.OnlineFailed=Не удалось загрузить компоненты Voxprint:%n%n%1%n%nПроверьте подключение к интернету и запустите установку снова: готовые части сохранены, загрузка продолжится с места остановки.
german.OnlineFailed=Der Download der Voxprint-Komponenten ist fehlgeschlagen:%n%n%1%n%nPrüfen Sie die Internetverbindung und starten Sie das Setup erneut: Fertige Teile bleiben erhalten, der Download wird fortgesetzt.
english.OnlineStalled=The downloader stopped responding.
russian.OnlineStalled=Загрузчик перестал отвечать.
german.OnlineStalled=Der Downloader antwortet nicht mehr.

english.PortablePageCaption=Portable setup folder (optional)
russian.PortablePageCaption=Переносимая папка установки (необязательно)
german.PortablePageCaption=Portabler Setup-Ordner (optional)
english.PortablePageDescription=Keep everything the setup downloads, for an offline reinstall.
russian.PortablePageDescription=Сохранить всё, что скачивает установка, для переустановки без интернета.
german.PortablePageDescription=Alles behalten, was das Setup lädt, für eine Neuinstallation ohne Internet.
english.PortableCheck=Keep a portable setup folder (all components and models, for offline reinstall)
russian.PortableCheck=Хранить переносимую папку установки (все компоненты и модели, для переустановки без интернета)
german.PortableCheck=Portablen Setup-Ordner behalten (alle Komponenten und Modelle, für Neuinstallation ohne Internet)
english.PortableInfo=The setup downloads ALL components and the models this PC needs into this folder (even those already on the PC) with a checksum list, and installs from it. Later you can run the installer again, or copy the folder to another PC, and install without internet; when the internet is available, only newer parts are downloaded. This needs about 3 GB for the program and up to 10 GB more for the models. No desktop shortcuts are created.
russian.PortableInfo=Установка скачает ВСЕ компоненты и нужные этому ПК модели в эту папку (даже те, что уже есть на ПК) со списком контрольных сумм и установит из неё. Позже можно запустить установщик снова или скопировать папку на другой ПК и установить без интернета; при наличии интернета докачиваются только более новые части. Нужно около 3 ГБ для программы и до 10 ГБ для моделей. Ярлыки на рабочем столе не создаются.
german.PortableInfo=Das Setup lädt ALLE Komponenten und die für diesen PC nötigen Modelle in diesen Ordner (auch bereits vorhandene) mit einer Prüfsummenliste und installiert daraus. Später können Sie das Setup erneut starten oder den Ordner auf einen anderen PC kopieren und ohne Internet installieren; ist Internet vorhanden, werden nur neuere Teile geladen. Erforderlich sind etwa 3 GB für das Programm und bis zu 10 GB für die Modelle. Es werden keine Desktop-Verknüpfungen erstellt.
english.PortableFoundInfo=A setup folder was found: %1%n%nThe setup installs from it. Without internet nothing is downloaded; with internet only newer parts are fetched.
russian.PortableFoundInfo=Найдена папка установки: %1%n%nУстановка пойдёт из неё. Без интернета ничего не скачивается; при наличии интернета докачиваются только более новые части.
german.PortableFoundInfo=Ein Setup-Ordner wurde gefunden: %1%n%nDas Setup installiert daraus. Ohne Internet wird nichts geladen; mit Internet werden nur neuere Teile geholt.
english.PortablePrompt=Setup folder:
russian.PortablePrompt=Папка установки:
german.PortablePrompt=Setup-Ordner:
english.PortableBadFolder=The folder %1 cannot be used (it cannot be created or written to). Choose another folder.
russian.PortableBadFolder=Папку %1 нельзя использовать (её нельзя создать или записать в неё). Выберите другую папку.
german.PortableBadFolder=Der Ordner %1 kann nicht verwendet werden (er kann nicht angelegt oder beschrieben werden). Wählen Sie einen anderen Ordner.

english.ModelsSiblingFound=Voxprint AI Movie Dubber is installed on this PC. Both programs use one models folder, so every model is downloaded only once.
russian.ModelsSiblingFound=На этом ПК установлен Voxprint AI Movie Dubber. Обе программы используют одну папку моделей, поэтому каждая модель скачивается только один раз.
german.ModelsSiblingFound=Voxprint AI Movie Dubber ist auf diesem PC installiert. Beide Programme nutzen einen Modellordner, daher wird jedes Modell nur einmal geladen.
english.UninstallModelsQuestion=No other Voxprint program uses the models in%n%1%n%nDelete these models too (about 15-28 GB)? Choose No to keep them for a later install.
russian.UninstallModelsQuestion=Модели в папке%n%1%nбольше не использует ни одна программа Voxprint.%n%nУдалить и их (около 15-28 ГБ)? Нажмите «Нет», чтобы оставить их для следующей установки.
german.UninstallModelsQuestion=Kein anderes Voxprint-Programm verwendet die Modelle in%n%1%n%nDiese Modelle ebenfalls löschen (etwa 15-28 GB)? Wählen Sie Nein, um sie für eine spätere Installation zu behalten.
english.PortableMissing=The setup folder %1 (from /FromFolder) was not found or has no manifest.json.
russian.PortableMissing=Папка установки %1 (из /FromFolder) не найдена или в ней нет manifest.json.
german.PortableMissing=Der Setup-Ordner %1 (aus /FromFolder) wurde nicht gefunden oder enthält keine manifest.json.

; No desktop shortcut on purpose (no [Tasks] section at all, so no task can be selected, silent or not): only the Start menu entry below + the Apps list entry.
[InstallDelete]
; an upgrade removes a desktop shortcut that an earlier version of this installer may have created (all users / current user)
Type: files; Name: "{commondesktop}\{#AppDisplayName}.lnk"
Type: files; Name: "{userdesktop}\{#AppDisplayName}.lnk"
Type: files; Name: "{commondesktop}\{#AppName}.lnk"
Type: files; Name: "{userdesktop}\{#AppName}.lnk"
; builds up to 703 had their own Start menu folder; from 704 on the entry is in the shared "Voxprint" folder
Type: files; Name: "{commonprograms}\{#AppDisplayName}\{#AppDisplayName}.lnk"
Type: dirifempty; Name: "{commonprograms}\{#AppDisplayName}"
Type: files; Name: "{userprograms}\{#AppDisplayName}\{#AppDisplayName}.lnk"
Type: dirifempty; Name: "{userprograms}\{#AppDisplayName}"

[Files]
#ifdef ONLINE
; the program itself is downloaded by voxprint-fetch.exe (see [Code]); only the downloader and the notices are inside
Source: "..\build\online\voxprint-fetch.exe"; DestDir: "{tmp}"; Flags: dontcopy
#else
#ifdef ONEDIR
Source: "..\dist\Voxprint\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#else
Source: "..\dist\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
#endif
#endif
; Third-party licences (LGPL/GPL/Apache etc.): the component list and the full licence texts.
Source: "..\build\notices\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\licenses\*"; DestDir: "{app}\licenses"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\credits.json"; DestDir: "{app}"; Flags: ignoreversion
; Optional: put vc_redist.x64.exe into installer\redist and it will be installed silently (PyTorch needs it).
#ifexist "redist\vc_redist.x64.exe"
Source: "redist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
#endif


[Icons]
Name: "{group}\{#AppDisplayName}"; Filename: "{app}\{#AppExe}"

[Run]
; (the Visual C++ runtime is installed from [Code]: InstallVcRedist)
Filename: "{app}\{#AppExe}"; Parameters: "--prefetch"; Description: "{cm:RunPrefetch,{#AppDisplayName}}"; Flags: nowait postinstall skipifsilent

[Code]
var
  ModelsPage: TWizardPage;
  ModelsEdit: TNewEdit;
  ModePage: TInputOptionWizardPage;
#ifdef ONLINE
  LatestCheck: TNewCheckBox;
#endif
#ifdef PORTABLE
  PortablePage: TWizardPage;
  PortableCheck: TNewCheckBox;
  PortableEdit: TNewEdit;
  PortableBrowse: TNewButton;
  PortableFound: String;
#endif

{ Hidden CI switch: /SKIPGPUCHECK. Not shown in the wizard. Smoke jobs pass it; users do not. }
function ParamSwitch(const Name: String): Boolean;
var
  I: Integer;
  S: String;
begin
  Result := False;
  for I := 1 to ParamCount do
  begin
    S := Uppercase(Trim(ParamStr(I)));
    if (S = '/' + Name) or (S = '-' + Name) then
    begin
      Result := True;
      Exit;
    end;
  end;
end;

function RPos(const Sub, S: String): Integer;
var
  I, N: Integer;
begin
  Result := 0;
  N := Length(Sub);
  for I := Length(S) - N + 1 downto 1 do
    if Copy(S, I, N) = Sub then
    begin
      Result := I;
      Exit;
    end;
end;

function CapMeets(const Cap: String): Boolean;
var
  P, Major, Minor, Code: Integer;
begin
  Result := False;
  P := Pos('.', Cap);
  if P < 2 then Exit;
  Val(Trim(Copy(Cap, 1, P - 1)), Major, Code);
  if Code <> 0 then Exit;
  Val(Trim(Copy(Cap, P + 1, 8)), Minor, Code);
  if Code <> 0 then Exit;
  Result := (Major > 8) or ((Major = 8) and (Minor >= 9));
end;

{ WMI has no compute capability. Accept GeForce RTX 40-series and newer, plus Ada / Blackwell names. }
function NameIsRtx40(const Name: String): Boolean;
var
  U, Digits: String;
  I, Series: Integer;
begin
  U := Uppercase(Name);
  Result := (Pos('ADA', U) > 0) or (Pos('BLACKWELL', U) > 0);
  if Result then Exit;
  I := Pos('RTX', U);
  if I = 0 then Exit;
  I := I + 3;
  while (I <= Length(U)) and (U[I] = ' ') do
    I := I + 1;
  Digits := '';
  while (I <= Length(U)) and (U[I] >= '0') and (U[I] <= '9') do
  begin
    Digits := Digits + U[I];
    I := I + 1;
  end;
  if Length(Digits) < 4 then Exit;
  Series := StrToIntDef(Copy(Digits, 1, 2), 0);
  Result := Series >= 40;
end;

function RunHiddenToFile(const CmdLine, OutFile: String): Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec(ExpandConstant('{cmd}'), '/C ' + CmdLine + ' > "' + OutFile + '" 2>nul',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

function NextGpuLine(var Text: String; var Line: String): Boolean;
var
  P: Integer;
begin
  Result := Text <> '';
  if not Result then Exit;
  P := Pos(#10, Text);
  if P = 0 then
  begin
    Line := Text;
    Text := '';
  end
  else
  begin
    Line := Copy(Text, 1, P - 1);
    Delete(Text, 1, P);
  end;
  Line := Trim(Line);
end;

function GpuFromSmi(const Text: String; var Detected: String): Boolean;
var
  Rest, Line, Name, Cap: String;
  P, BestMaj, BestMin, Maj, Min, Code, Dot: Integer;
  Any: Boolean;
begin
  Result := False;
  BestMaj := -1;
  BestMin := -1;
  Any := False;
  Rest := Text;
  while NextGpuLine(Rest, Line) do
  begin
    if Line = '' then Continue;
    P := RPos(',', Line);
    if P < 2 then Continue;
    Line := Trim(Copy(Line, 1, P - 1));
    P := RPos(',', Line);
    if P < 2 then Continue;
    Cap := Trim(Copy(Line, P + 1, 16));
    Name := Trim(Copy(Line, 1, P - 1));
    if (Length(Name) >= 2) and (Name[1] = '"') and (Name[Length(Name)] = '"') then
      Name := Copy(Name, 2, Length(Name) - 2);
    Dot := Pos('.', Cap);
    if Dot < 2 then Continue;
    Val(Copy(Cap, 1, Dot - 1), Maj, Code);
    if Code <> 0 then Continue;
    Val(Copy(Cap, Dot + 1, 8), Min, Code);
    if Code <> 0 then Continue;
    if (not Any) or (Maj > BestMaj) or ((Maj = BestMaj) and (Min > BestMin)) then
    begin
      Any := True;
      BestMaj := Maj;
      BestMin := Min;
      Detected := Name;
      Result := CapMeets(Cap);
    end;
  end;
end;

function GpuFromNames(const Text: String; var Detected: String): Boolean;
var
  Rest, Line, Nvidia: String;
begin
  Result := False;
  Nvidia := '';
  Rest := Text;
  while NextGpuLine(Rest, Line) do
  begin
    if (Pos('NVIDIA', Uppercase(Line)) = 0) and (Pos('GEFORCE', Uppercase(Line)) = 0) then Continue;
    if Nvidia = '' then Nvidia := Line;
    if NameIsRtx40(Line) then
    begin
      Detected := Line;
      Result := True;
      Exit;
    end;
  end;
  if Nvidia <> '' then Detected := Nvidia;
end;

function GpuOk(var Detected: String): Boolean;
var
  OutFile: String;
  Raw: AnsiString;
begin
  Detected := CustomMessage('GpuNone');
  OutFile := ExpandConstant('{tmp}\voxprint-gpu.txt');
  if RunHiddenToFile('nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader', OutFile)
     and LoadStringFromFile(OutFile, Raw) and (Trim(String(Raw)) <> '') then
  begin
    Result := GpuFromSmi(String(Raw), Detected);
    if Detected = '' then Detected := CustomMessage('GpuNone');
    Exit;
  end;
  if RunHiddenToFile('powershell.exe -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"', OutFile)
     and LoadStringFromFile(OutFile, Raw) then
  begin
    Result := GpuFromNames(String(Raw), Detected);
    if Detected = '' then Detected := CustomMessage('GpuNone');
    Exit;
  end;
  Result := False;
end;

function InitializeSetup(): Boolean;
var
  V: TWindowsVersion;
  Detected: String;
begin
  Result := True;
  GetWindowsVersionEx(V);
  if V.Build < 26100 then
  begin
    MsgBox(FmtMessage(CustomMessage('WinBuildTooOld'), ['{#AppDisplayName}', IntToStr(V.Build)]), mbError, MB_OK);
    Result := False;
    Exit;
  end;
  if ParamSwitch('SKIPGPUCHECK') then Exit;
  if not GpuOk(Detected) then
  begin
    MsgBox(FmtMessage(CustomMessage('GpuRequired'), [Detected]), mbError, MB_OK);
    Result := False;
  end;
end;

{ Callback of the Browse button on the models page (True = the dialog may create a new folder). }
procedure ModelsBrowseClick(Sender: TObject);
var
  Dir: String;
begin
  Dir := ModelsEdit.Text;
  if BrowseForFolder(CustomMessage('ModelsPagePrompt'), Dir, True) then
    ModelsEdit.Text := Dir;
end;

{ The default models folder of the app (infra/paths.py: default_models_dir). }
function DefaultModelsDir(): String;
begin
  Result := ExpandConstant('{localappdata}\Voxprint\models');
end;

{ One string value of the shared settings file %LOCALAPPDATA%\Voxprint\state\suite.json (infra/suite_settings.py; the Movie
  Dubber reads and writes the same file).  '' when the file, the key or a string value is missing (null = the default folder).
  Our program writes it as ASCII JSON (\uXXXX escapes); a value with raw non-ASCII bytes is not decoded here and counts as ''. }
function SuiteJsonString(const Key: String): String;
var
  Raw: AnsiString;
  S, H: String;
  P, I, N: Integer;
  C: Char;
begin
  Result := '';
  if not LoadStringFromFile(ExpandConstant('{localappdata}\Voxprint\state\suite.json'), Raw) then Exit;
  S := String(Raw);
  P := Pos('"' + Key + '"', S);
  if P = 0 then Exit;
  N := Length(S);
  I := P + Length(Key) + 2;
  while (I <= N) and ((S[I] = ' ') or (S[I] = #9) or (S[I] = #13) or (S[I] = #10) or (S[I] = ':')) do
    I := I + 1;
  if (I > N) or (S[I] <> '"') then Exit;
  I := I + 1;
  while I <= N do
  begin
    C := S[I];
    if C = '"' then Exit;
    if C = '\' then
    begin
      I := I + 1;
      if I > N then Break;
      C := S[I];
      if C = 'u' then
      begin
        H := Copy(S, I + 1, 4);
        Result := Result + Chr(StrToIntDef('$' + H, 63));
        I := I + 4;
      end
      else
        Result := Result + C;
    end
    else if Ord(C) > 127 then
    begin
      Result := '';
      Exit;
    end
    else
      Result := Result + C;
    I := I + 1;
  end;
  Result := '';
end;

{ True when Voxprint AI Movie Dubber is installed: an Apps list entry with "Movie Dubber" in its name, or its key in the models
  folder's .users.json (infra/models_users.py). }
function UninstallEntryFound(RootKey: Integer; const Needle: String): Boolean;
var
  Names: TArrayOfString;
  I: Integer;
  Name: String;
begin
  Result := False;
  if not RegGetSubkeyNames(RootKey, 'Software\Microsoft\Windows\CurrentVersion\Uninstall', Names) then Exit;
  for I := 0 to GetArrayLength(Names) - 1 do
    if RegQueryStringValue(RootKey, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\' + Names[I], 'DisplayName', Name) and
       (Pos(Needle, Lowercase(Name)) > 0) then
    begin
      Result := True;
      Exit;
    end;
end;

function MovieDubberInstalled(const ModelsFolder: String): Boolean;
var
  Raw: AnsiString;
begin
  Result := UninstallEntryFound(HKLM64, 'movie dubber') or UninstallEntryFound(HKLM32, 'movie dubber') or
            UninstallEntryFound(HKCU, 'movie dubber');
  if (not Result) and (ModelsFolder <> '') and LoadStringFromFile(AddBackslash(ModelsFolder) + '.users.json', Raw) then
    Result := Pos('"movie-dubber"', String(Raw)) > 0;
end;

{ What the page shows first: /ModelsFolder=, else the shared models folder of the Voxprint programs (suite.json), else the folder
  remembered by an earlier install, else the default.  The same order as infra/paths.py: configured_models_dir. }
function InitialModelsDir(): String;
var
  S: String;
  Lines: TArrayOfString;
begin
  Result := RemoveBackslash(Trim(ExpandConstant('{param:ModelsFolder|}')));
  if Result <> '' then Exit;
  Result := RemoveBackslash(Trim(SuiteJsonString('models_dir')));
  if (Length(Result) >= 3) and ((Copy(Result, 2, 2) = ':\') or (Copy(Result, 1, 2) = '\\')) then Exit;
  S := ExpandConstant('{localappdata}\Voxprint\state\models_dir.txt');
  if FileExists(S) and LoadStringsFromFile(S, Lines) and (GetArrayLength(Lines) > 0) then
  begin
    Result := RemoveBackslash(Trim(Lines[0]));
    if Result <> '' then Exit;
  end;
  Result := DefaultModelsDir();
end;

{ The folder must be creatable and writable (a read-only USB stick, a typo ...). }
function FolderUsable(const Dir: String): Boolean;
var
  Probe: String;
begin
  Result := False;
  if (Dir = '') or (not ForceDirectories(Dir)) then Exit;
  Probe := AddBackslash(Dir) + '.voxprint-write-test';
  Result := SaveStringToFile(Probe, 'x', False);
  if Result then DeleteFile(Probe);
end;

{ True if D is a Voxprint backup: it holds voxprint-backup.json, or it is a Voxprint-backup folder with models / model / voices
  inside (a backup without its manifest is still no models folder).  The same rules as infra/paths.py: _looks_like_backup. }
function LooksLikeBackup(const D: String): Boolean;
begin
  Result := FileExists(AddBackslash(D) + 'voxprint-backup.json');
  if (not Result) and (CompareText(ExtractFileName(D), 'Voxprint-backup') = 0) then
    Result := DirExists(AddBackslash(D) + 'models') or DirExists(AddBackslash(D) + 'model') or
              DirExists(AddBackslash(D) + 'voices');
end;

{ The Voxprint backup behind a chosen folder, '' if it is none (the same rules as infra/paths.py: backup_root_of): the folder is a
  backup, or it holds a Voxprint-backup backup, or it is the models / model / voices / tools folder of a backup.
  A backup is a RESTORE SOURCE: it never becomes the models folder (the app restores it into the default folder). }
function BackupRootOf(const Dir: String): String;
var
  D, N: String;
begin
  Result := '';
  D := RemoveBackslash(Trim(Dir));
  if D = '' then Exit;
  N := Lowercase(ExtractFileName(D));
  if LooksLikeBackup(D) then
    Result := D
  else if LooksLikeBackup(AddBackslash(D) + 'Voxprint-backup') then
    Result := AddBackslash(D) + 'Voxprint-backup'
  else if ((N = 'models') or (N = 'model') or (N = 'voices') or (N = 'tools')) and LooksLikeBackup(ExtractFileDir(D)) then
    Result := ExtractFileDir(D);
end;

{ True for a folder below Program Files (the setup runs as admin, the app does not: it could not write there). }
function InProgramFiles(const Dir: String): Boolean;
var
  D: String;
begin
  D := Lowercase(AddBackslash(Dir));
  Result := (Pos(Lowercase(AddBackslash(ExpandConstant('{commonpf64}'))), D) = 1) or
            (Pos(Lowercase(AddBackslash(ExpandConstant('{commonpf32}'))), D) = 1);
end;

#ifdef PORTABLE
{ A setup folder made by an earlier run: /FromFolder=<dir>, the one remembered in state\portable_dir.txt, Documents\Voxprint Portable,
  or a folder next to the installer (copied together with it).  A folder counts if it has a manifest.json. }
function IsSetupFolder(const Dir: String): Boolean;
begin
  Result := (Dir <> '') and FileExists(AddBackslash(Dir) + 'manifest.json');
end;

function FindPortableFolder(): String;
var
  S: String;
  Lines: TArrayOfString;
begin
  Result := RemoveBackslash(Trim(ExpandConstant('{param:FromFolder|}')));
  if IsSetupFolder(Result) then Exit;
  S := ExpandConstant('{localappdata}\Voxprint\state\portable_dir.txt');
  if FileExists(S) and LoadStringsFromFile(S, Lines) and (GetArrayLength(Lines) > 0) then
  begin
    Result := RemoveBackslash(Trim(Lines[0]));
    if IsSetupFolder(Result) then Exit;
  end;
  Result := ExpandConstant('{userdocs}\Voxprint Portable');
  if IsSetupFolder(Result) then Exit;
  Result := RemoveBackslash(ExtractFilePath(ExpandConstant('{srcexe}'))) + '\Voxprint Portable';
  if IsSetupFolder(Result) then Exit;
  Result := RemoveBackslash(ExtractFilePath(ExpandConstant('{srcexe}')));
  if IsSetupFolder(Result) then Exit;
  Result := '';
end;

procedure PortableCheckClick(Sender: TObject);
begin
  PortableEdit.Enabled := PortableCheck.Checked;
  PortableBrowse.Enabled := PortableCheck.Checked;
end;

procedure PortableBrowseClick(Sender: TObject);
var
  Dir: String;
begin
  Dir := PortableEdit.Text;
  if BrowseForFolder(CustomMessage('PortablePrompt'), Dir, True) then
    PortableEdit.Text := Dir;
end;

{ The folder to use ('' = the option is off).  /FromFolder=<dir> forces it (strictly offline, see RunOnlineDownload). }
function PortableDir(): String;
begin
  Result := '';
  if Trim(ExpandConstant('{param:FromFolder|}')) <> '' then Result := PortableFound
  else if PortableCheck.Checked then Result := RemoveBackslash(Trim(PortableEdit.Text));
end;

procedure CreatePortablePage();
var
  Info, Prompt: TNewStaticText;
  Dir: String;
  Silent: Boolean;
begin
  PortableFound := FindPortableFolder();
  PortablePage := CreateCustomPage(ModelsPage.ID, CustomMessage('PortablePageCaption'), CustomMessage('PortablePageDescription'));
  PortableCheck := TNewCheckBox.Create(PortablePage);
  PortableCheck.Parent := PortablePage.Surface;
  PortableCheck.Left := 0;
  PortableCheck.Top := 0;
  PortableCheck.Width := PortablePage.SurfaceWidth;
  PortableCheck.Caption := CustomMessage('PortableCheck');
  PortableCheck.OnClick := @PortableCheckClick;
  Info := TNewStaticText.Create(PortablePage);
  Info.Parent := PortablePage.Surface;
  Info.WordWrap := True;
  Info.AutoSize := False;
  Info.Left := 0;
  Info.Top := ScaleY(28);
  Info.Width := PortablePage.SurfaceWidth;
  Info.Height := ScaleY(120);
  if PortableFound <> '' then
    Info.Caption := FmtMessage(CustomMessage('PortableFoundInfo'), [PortableFound])
  else
    Info.Caption := CustomMessage('PortableInfo');
  Prompt := TNewStaticText.Create(PortablePage);
  Prompt.Parent := PortablePage.Surface;
  Prompt.Left := 0;
  Prompt.Top := ScaleY(156);
  Prompt.Caption := CustomMessage('PortablePrompt');
  PortableEdit := TNewEdit.Create(PortablePage);
  PortableEdit.Parent := PortablePage.Surface;
  PortableEdit.Left := 0;
  PortableEdit.Top := ScaleY(174);
  PortableEdit.Width := PortablePage.SurfaceWidth - ScaleX(96);
  Dir := Trim(ExpandConstant('{param:PortableDir|}'));
  if Dir = '' then Dir := PortableFound;
  if Dir = '' then Dir := ExpandConstant('{userdocs}\Voxprint Portable');
  PortableEdit.Text := Dir;
  PortableBrowse := TNewButton.Create(PortablePage);
  PortableBrowse.Parent := PortablePage.Surface;
  PortableBrowse.Left := PortableEdit.Width + ScaleX(8);
  PortableBrowse.Top := PortableEdit.Top - ScaleY(1);
  PortableBrowse.Width := ScaleX(88);
  PortableBrowse.Height := PortableEdit.Height + ScaleY(2);
  PortableBrowse.Caption := WizardForm.DirBrowseButton.Caption;
  PortableBrowse.OnClick := @PortableBrowseClick;
  { off by default; on when a setup folder exists or the silent switch asks for it }
  Silent := (ExpandConstant('{param:Portable|0}') = '1') or (Trim(ExpandConstant('{param:FromFolder|}')) <> '');
  PortableCheck.Checked := Silent or (PortableFound <> '');
  PortableCheckClick(nil);
end;

#endif

{ MiB as whole GB for the texts (rounded). }
function GbText(const Mb: Int64): String;
begin
  Result := IntToStr((Mb + 512) div 1024);
end;

{ The setup type: 'full' (default) or 'quick' (infra/setup_mode.py reads state\install_mode.txt). }
function SetupMode(): String;
begin
  if ModePage.Values[1] then Result := 'quick' else Result := 'full';
end;

{ Free MiB on the drive of Dir (-1 = unknown). }
function FreeMb(const Dir: String): Int64;
var
  Free, Total: Int64;
begin
  Result := -1;
  if GetSpaceOnDisk64(AddBackslash(ExtractFileDrive(Dir)), Free, Total) then
    Result := Free div (1024 * 1024);
end;

{ Full setup: room for the models on the models drive and for the components (runtime) in the user profile (5 % margin).
  Returns '' or the message.  Silent installs only log it (a script may free the space before the first start). }
function SpaceProblem(const ModelsDir: String): String;
var
  ModelsDrive, RtDrive: String;
  Need, Have: Int64;
begin
  Result := '';
  ModelsDrive := ExtractFileDrive(ModelsDir);
  RtDrive := ExtractFileDrive(ExpandConstant('{localappdata}'));
  Need := StrToInt64('{#FullModelsMB}');
  if CompareText(ModelsDrive, RtDrive) = 0 then Need := Need + StrToInt64('{#FullRuntimeMB}');
  Need := Need + Need div 20;
  Have := FreeMb(ModelsDir);
  if (Have >= 0) and (Have < Need) then
  begin
    Result := FmtMessage(CustomMessage('ModeNoSpace'), [ModelsDrive, GbText(Need), GbText(Have)]);
    Exit;
  end;
  if CompareText(ModelsDrive, RtDrive) <> 0 then
  begin
    Need := StrToInt64('{#FullRuntimeMB}');
    Need := Need + Need div 20;
    Have := FreeMb(ExpandConstant('{localappdata}'));
    if (Have >= 0) and (Have < Need) then
      Result := FmtMessage(CustomMessage('ModeNoSpace'), [RtDrive, GbText(Need), GbText(Have)]);
  end;
end;

{ Page after the install folder: the models folder (default = the current Local location).  Nothing is copied here.
  A plain custom page, NOT CreateInputDirPage (that one behaved badly in silent installs; found in the final installer test). }
procedure InitializeWizard();
var
  Info, Prompt, Sibling: TNewStaticText;
  Browse: TNewButton;
begin
  ModelsPage := CreateCustomPage(wpSelectDir, CustomMessage('ModelsPageCaption'),
    FmtMessage(CustomMessage('ModelsPageDescription'), [GbText(StrToInt64('{#FullModelsMB}'))]));
  Info := TNewStaticText.Create(ModelsPage);
  Info.Parent := ModelsPage.Surface;
  Info.WordWrap := True;
  Info.AutoSize := False;
  Info.Left := 0;
  Info.Top := 0;
  Info.Width := ModelsPage.SurfaceWidth;
  Info.Height := ScaleY(110);              { room for six lines of the Russian text }
  Info.Caption := CustomMessage('ModelsPageSubCaption');
  Prompt := TNewStaticText.Create(ModelsPage);
  Prompt.Parent := ModelsPage.Surface;
  Prompt.Left := 0;
  Prompt.Top := ScaleY(118);
  Prompt.Caption := CustomMessage('ModelsPagePrompt');
  ModelsEdit := TNewEdit.Create(ModelsPage);
  ModelsEdit.Parent := ModelsPage.Surface;
  ModelsEdit.Left := 0;
  ModelsEdit.Top := ScaleY(136);
  ModelsEdit.Width := ModelsPage.SurfaceWidth - ScaleX(96);
  ModelsEdit.Text := InitialModelsDir();
  Browse := TNewButton.Create(ModelsPage);
  Browse.Parent := ModelsPage.Surface;
  Browse.Left := ModelsEdit.Width + ScaleX(8);
  Browse.Top := ModelsEdit.Top - ScaleY(1);
  Browse.Width := ScaleX(88);
  Browse.Height := ModelsEdit.Height + ScaleY(2);
  Browse.Caption := WizardForm.DirBrowseButton.Caption;
  Browse.OnClick := @ModelsBrowseClick;
  if MovieDubberInstalled(ModelsEdit.Text) then
  begin
    Sibling := TNewStaticText.Create(ModelsPage);
    Sibling.Parent := ModelsPage.Surface;
    Sibling.WordWrap := True;
    Sibling.AutoSize := False;
    Sibling.Left := 0;
    Sibling.Top := ScaleY(200);
    Sibling.Width := ModelsPage.SurfaceWidth;
    Sibling.Height := ScaleY(34);
    Sibling.Caption := CustomMessage('ModelsSiblingFound');
  end;
#ifdef ONLINE
  if '{#LatestManifestUrl}' <> '' then
  begin
    LatestCheck := TNewCheckBox.Create(ModelsPage);
    LatestCheck.Parent := ModelsPage.Surface;
    LatestCheck.Left := 0;
    LatestCheck.Top := ScaleY(176);
    LatestCheck.Width := ModelsPage.SurfaceWidth;
    LatestCheck.Height := ScaleY(20);
    LatestCheck.Caption := CustomMessage('LatestCheck');
    LatestCheck.Checked := ExpandConstant('{param:Latest|0}') = '1';
  end;
#endif
  { the setup type comes right after the models folder (its free-space check needs that folder) }
  ModePage := CreateInputOptionPage(ModelsPage.ID, CustomMessage('ModePageCaption'), CustomMessage('ModePageDescription'),
    CustomMessage('ModePageSubCaption'), True, False);
  ModePage.Add(FmtMessage(CustomMessage('ModeFull'), [GbText(StrToInt64('{#FullModelsMB}') + StrToInt64('{#FullRuntimeMB}'))]));
  ModePage.Add(FmtMessage(CustomMessage('ModeQuick'), [GbText(StrToInt64('{#FullModelsMB}') + StrToInt64('{#FullRuntimeMB}'))]));
  if Lowercase(Trim(ExpandConstant('{param:Mode|full}'))) = 'quick' then
    ModePage.SelectedValueIndex := 1
  else
    ModePage.SelectedValueIndex := 0;
#ifdef PORTABLE
  CreatePortablePage();
#endif
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Dir, Backup, Problem: String;
begin
  Result := True;
  if (CurPageID = ModePage.ID) and (SetupMode() = 'full') then
  begin
    Dir := RemoveBackslash(Trim(ModelsEdit.Text));
    if BackupRootOf(Dir) <> '' then Dir := DefaultModelsDir();     { a backup is restored into the default folder }
    Problem := SpaceProblem(Dir);
    if Problem <> '' then
    begin
      if WizardSilent then
        Log(Problem)
      else
      begin
        MsgBox(Problem, mbError, MB_OK);
        Result := False;
      end;
    end;
  end;
  if CurPageID = ModelsPage.ID then
  begin
    Dir := RemoveBackslash(Trim(ModelsEdit.Text));
    if Dir = '' then
    begin
      Dir := DefaultModelsDir();                  { an emptied field means "the default" }
      ModelsEdit.Text := Dir;
    end;
    Backup := BackupRootOf(Dir);
    if Backup <> '' then
      { a backup is only read (it may sit on a read-only or external drive): no write test, just say what will happen }
      SuppressibleMsgBox(FmtMessage(CustomMessage('ModelsPageBackupFound'), [Backup, DefaultModelsDir()]), mbInformation, MB_OK, IDOK)
    else if InProgramFiles(Dir) then
    begin
      MsgBox(FmtMessage(CustomMessage('ModelsPageProtected'), [Dir]), mbError, MB_OK);
      Result := False;
    end
    else if not FolderUsable(Dir) then
    begin
      MsgBox(FmtMessage(CustomMessage('ModelsPageBadFolder'), [Dir]), mbError, MB_OK);
      Result := False;
    end;
  end;
#ifdef PORTABLE
  if CurPageID = PortablePage.ID then
  begin
    Dir := PortableDir();
    if (Dir <> '') and (Trim(ExpandConstant('{param:FromFolder|}')) = '') and not FolderUsable(Dir) then
    begin
      MsgBox(FmtMessage(CustomMessage('PortableBadFolder'), [Dir]), mbError, MB_OK);
      Result := False;
    end;
  end;
#endif
end;

#ifdef ONLINE
function VxTick(): Cardinal;
external 'GetTickCount@kernel32.dll stdcall';

{ Online variant: run voxprint-fetch.exe in the background and show its progress.  The downloader writes three lines to a
  status file (state running/done/error, permille, message) plus a heartbeat counter in line 4 every second or two; a counter
  that stops changing for 120 s means that the process died. }
function ReadFetchStatus(const StatusFile: String; var State: String; var Permille: Integer; var Msg, Beat: String): Boolean;
var
  Lines: TArrayOfString;
begin
  Result := False;
  if FileExists(StatusFile) and LoadStringsFromFile(StatusFile, Lines) and (GetArrayLength(Lines) >= 3) then
  begin
    State := Trim(Lines[0]);
    Permille := StrToIntDef(Trim(Lines[1]), 0);
    Msg := Trim(Lines[2]);
    if GetArrayLength(Lines) >= 4 then Beat := Trim(Lines[3]) else Beat := '';
    Result := True;
  end;
end;

function RunOnlineDownload(): String;
var
  Manifest, Cache, StatusFile, Exe, Params, State, Msg, Beat, LastBeat, Err, Folder: String;
  Pm, ResultCode: Integer;
  Page: TOutputProgressWizardPage;
  LastChange: Cardinal;
  Finished: Boolean;
begin
  Result := '';
  ExtractTemporaryFile('voxprint-fetch.exe');
  Manifest := Trim(ExpandConstant('{param:Manifest|}'));
  if Manifest = '' then Manifest := '{#ManifestUrl}';
  Cache := ExpandConstant('{localappdata}\Voxprint\setup-cache');
  ForceDirectories(Cache);
  StatusFile := Cache + '\status.txt';
  DeleteFile(StatusFile);
  Exe := ExpandConstant('{tmp}\voxprint-fetch.exe');
  Params := '--manifest ' + AddQuotes(Manifest) + ' --dest ' + AddQuotes(ExpandConstant('{app}')) +
            ' --cache ' + AddQuotes(Cache) + ' --status ' + AddQuotes(StatusFile);
  if '{#LatestManifestUrl}' <> '' then
  begin
    { pinned first (default) with the newest release as the fallback, or the other way round when the box is ticked }
    Params := Params + ' --latest-manifest ' + AddQuotes('{#LatestManifestUrl}');
    if (LatestCheck <> nil) and LatestCheck.Checked then
      Params := Params + ' --prefer latest';
  end;
#ifdef THIN
  { THIN installer: only the shell (role "core") is installed here; the app downloads the runtime modules itself }
  Params := Params + ' --role core';
#endif
#ifdef PORTABLE
  { Portable setup folder: keep ALL components (+ the models this PC needs) there and install from it.  /FromFolder= is strictly
    offline; otherwise the folder is a cache first and the internet only supplies what is missing or newer. }
  Folder := PortableDir();
  if (Folder = '') and (Trim(ExpandConstant('{param:FromFolder|}')) <> '') then
  begin
    Result := FmtMessage(CustomMessage('PortableMissing'), [Trim(ExpandConstant('{param:FromFolder|}'))]);
    Exit;
  end;
  if Folder <> '' then
  begin
    if Trim(ExpandConstant('{param:FromFolder|}')) <> '' then
      Params := Params + ' --from-folder ' + AddQuotes(Folder)
    else
      Params := Params + ' --portable ' + AddQuotes(Folder) + ' --portable-all --models ' + ExpandConstant('{param:PortableModels|auto}');
  end;
#endif
  Page := CreateOutputProgressPage(CustomMessage('OnlineStatus'), '');
  Page.Show;
  Err := '';
  try
    Page.SetText(CustomMessage('OnlineStatus'), '');
    Page.SetProgress(0, 1000);
    if not Exec(Exe, Params, '', SW_HIDE, ewNoWait, ResultCode) then
      Err := SysErrorMessage(ResultCode)
    else
    begin
      Finished := False;
      LastBeat := '';
      LastChange := VxTick;
      while not Finished do
      begin
        Sleep(300);
        if ReadFetchStatus(StatusFile, State, Pm, Msg, Beat) then
        begin
          Page.SetText(CustomMessage('OnlineStatus'), Msg);
          Page.SetProgress(Pm, 1000);
          if Beat <> LastBeat then
          begin
            LastBeat := Beat;
            LastChange := VxTick;
          end;
          if State = 'done' then Finished := True
          else if State = 'error' then
          begin
            Err := Msg;
            Finished := True;
          end;
        end;
        if (not Finished) and (VxTick - LastChange > 120000) then
        begin
          Err := CustomMessage('OnlineStalled');
          Finished := True;
        end;
      end;
    end;
  finally
    Page.Hide;
  end;
  if Err <> '' then
    Result := FmtMessage(CustomMessage('OnlineFailed'), [Err])
  else
    DelTree(Cache, True, True, True);
end;

{ The download runs before the files are copied: a failure here stops the setup with a message (a non-zero exit code when
  silent) and the wizard goes back, so that the user can retry; the cache keeps the finished parts for the next try. }
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := RunOnlineDownload();
end;
#endif

{ The Visual C++ runtime (PyTorch needs it): installed quietly.  Exit codes that are NOT errors: 0 installed, 1638 a newer
  version is already present, 3010 / 1641 installed (a restart is pending).  Nothing is shown to the user either way; an
  unexpected code only goes to the setup log. }
{ True when a Visual C++ 2015-2022 runtime of version 14.29 or newer (what current PyTorch builds need) is already installed. }
function VcRuntimePresent(): Boolean;
var
  Installed, Major, Minor: Cardinal;
  Key: String;
begin
  Result := False;
  Key := 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64';
  if RegQueryDWordValue(HKLM64, Key, 'Installed', Installed) and (Installed = 1) and
     RegQueryDWordValue(HKLM64, Key, 'Major', Major) and RegQueryDWordValue(HKLM64, Key, 'Minor', Minor) then
    Result := (Major > 14) or ((Major = 14) and (Minor >= 29));
end;

procedure InstallVcRedist();
var
  Exe: String;
  Rc: Integer;
begin
  Exe := ExpandConstant('{tmp}\vc_redist.x64.exe');
  if not FileExists(Exe) then Exit;
  if VcRuntimePresent() then
  begin
    Log('The Visual C++ runtime is already installed - not installing it again');
    Exit;
  end;
  WizardForm.StatusLabel.Caption := CustomMessage('VcRedistStatus');
  if Exec(Exe, '/install /quiet /norestart', '', SW_HIDE, ewWaitUntilTerminated, Rc) then
  begin
    if (Rc <> 0) and (Rc <> 1638) and (Rc <> 3010) and (Rc <> 1641) then
      Log('vc_redist.x64.exe finished with exit code ' + IntToStr(Rc) + ' (ignored)');
  end
  else
    Log('vc_redist.x64.exe could not be started: ' + SysErrorMessage(Rc) + ' (ignored)');
end;

{ UI language code of the app (core/i18n.py LANGS) for the wizard language. }
function AppLanguageCode(): String;
begin
  if ActiveLanguage = 'russian' then
    Result := 'ru'
  else if ActiveLanguage = 'german' then
    Result := 'de'
  else
    Result := 'en';
end;

{ Shared models folder: models\.users.json lists the Voxprint programs that use it (infra/models_users.py; the Movie Dubber
  reads and writes the same file).  Setup adds our key; the app adds it again at every start, so a failure here only delays it.
  The file is one small ASCII JSON object, so a plain text edit is enough. }
procedure RegisterModelsUser(Folder: String);
var
  F, S, Inner: String;
  Raw: AnsiString;
  OpenPos, ClosePos, I: Integer;
begin
  if Folder = '' then Exit;
  ForceDirectories(Folder);
  F := AddBackslash(Folder) + '.users.json';
  S := '';
  if LoadStringFromFile(F, Raw) then
    S := Trim(String(Raw));
  if Pos('"audiobook-builder"', S) > 0 then Exit;
  OpenPos := Pos('{', S);
  ClosePos := 0;
  for I := Length(S) downto 1 do
    if S[I] = '}' then
    begin
      ClosePos := I;
      Break;
    end;
  if (OpenPos = 0) or (ClosePos < OpenPos) then
    S := '{"audiobook-builder": true}'
  else
  begin
    Inner := Trim(Copy(S, OpenPos + 1, ClosePos - OpenPos - 1));
    if Inner = '' then
      S := '{"audiobook-builder": true}'
    else
      S := '{"audiobook-builder": true, ' + Inner + '}';
  end;
  SaveStringToFile(F, AnsiString(S), False);
end;

{ The shared settings file (state\suite.json) gets the models folder chosen above (null = default) and the UI language if it has
  none yet: Voxprint.exe --sync-suite-settings (infra/suite_settings.py) - run as the user who started the setup, so it is that
  user's %LOCALAPPDATA%.  A failure is harmless: the app copies the same values at its next start. }
procedure SyncSuiteSettings();
var
  Rc: Integer;
begin
  if FileExists(ExpandConstant('{app}\{#AppExe}')) then
    if not ExecAsOriginalUser(ExpandConstant('{app}\{#AppExe}'), '--sync-suite-settings', ExpandConstant('{app}'), SW_HIDE,
                              ewWaitUntilTerminated, Rc) then
      Log('suite settings were not synced; the app does it at its next start');
end;

{ Only paths are remembered (UTF-8 files in state\); no model is copied.  models_dir.txt = the models folder (deleted when it is the
  default, so the app follows its own default); existing_models_dir.txt = an optional folder to import models from (/ModelsDir=).
  A chosen folder with a Voxprint backup is NOT written to models_dir.txt: its root goes to existing_models_dir.txt and the app
  restores it into the default folder on the first start (infra/existing_models.py). }
procedure CurStepChanged(CurStep: TSetupStep);
var
  Dir, StateDir, Backup: String;
  Lines: TArrayOfString;
begin
  if CurStep = ssPostInstall then InstallVcRedist();
#ifdef PORTABLE
  if CurStep = ssPostInstall then
  begin
    Dir := PortableDir();
    if Dir <> '' then
    begin
      { the program installs its modules from this folder (infra/portable.py reads this file); its models/ are an "existing models folder" }
      StateDir := ExpandConstant('{localappdata}\Voxprint\state');
      ForceDirectories(StateDir);
      SetArrayLength(Lines, 1);
      Lines[0] := Dir;
      SaveStringsToUTF8File(StateDir + '\portable_dir.txt', Lines, False);
      if (Trim(ExpandConstant('{param:ModelsDir|}')) = '') and DirExists(AddBackslash(Dir) + 'models') then
      begin
        Lines[0] := Dir;
        SaveStringsToUTF8File(StateDir + '\existing_models_dir.txt', Lines, False);
      end;
    end;
  end;
#endif
  if CurStep = ssPostInstall then
  begin
    StateDir := ExpandConstant('{localappdata}\Voxprint\state');
    ForceDirectories(StateDir);
    SetArrayLength(Lines, 1);
    Dir := RemoveBackslash(Trim(ModelsEdit.Text));
    Backup := BackupRootOf(Dir);
    if Backup <> '' then
    begin
      DeleteFile(StateDir + '\models_dir.txt');          { the live models stay in the default folder }
      Lines[0] := Backup;
      SaveStringsToUTF8File(StateDir + '\existing_models_dir.txt', Lines, False);
    end
    else if (Dir = '') or (CompareText(Dir, DefaultModelsDir()) = 0) then
      DeleteFile(StateDir + '\models_dir.txt')
    else
    begin
      Lines[0] := Dir;
      SaveStringsToUTF8File(StateDir + '\models_dir.txt', Lines, False);
    end;
    Dir := Trim(ExpandConstant('{param:ModelsDir|}'));
    if Dir <> '' then
    begin
      Lines[0] := Dir;
      SaveStringsToUTF8File(StateDir + '\existing_models_dir.txt', Lines, False);
    end;
    { register this program in the live models folder (a backup or the default choice = the default folder) }
    Dir := RemoveBackslash(Trim(ModelsEdit.Text));
    if (Backup <> '') or (Dir = '') then
      Dir := DefaultModelsDir();
    RegisterModelsUser(Dir);
    SyncSuiteSettings();
    Lines[0] := SetupMode();                               { Full / Quick (infra/setup_mode.py); no BOM }
    SaveStringsToUTF8FileWithoutBOM(StateDir + '\install_mode.txt', Lines, False);
    { the wizard language becomes the app's UI language, unless the user already chose one (core/i18n.py reads state\language) }
    if not FileExists(StateDir + '\language') then
    begin
      Lines[0] := AppLanguageCode();
      SaveStringsToUTF8FileWithoutBOM(StateDir + '\language', Lines, False);
    end;
  end;
end;

{ Uninstall: remove our key with the program itself (Voxprint.exe --unregister-models-user --out FILE).  FILE gets the number of
  OTHER programs still using the models and the models folder.  -1 (helper failed) means: never offer to delete. }
var
  OtherModelUsers: Integer;
  SharedModelsDir: String;

procedure UnregisterModelsUser;
var
  Rc: Integer;
  OutFile: String;
  Lines: TArrayOfString;
begin
  OtherModelUsers := -1;
  SharedModelsDir := '';
  OutFile := AddBackslash(GetTempDir()) + 'voxprint-models-users.txt';
  DeleteFile(OutFile);
  if not FileExists(ExpandConstant('{app}\{#AppExe}')) then Exit;
  if Exec(ExpandConstant('{app}\{#AppExe}'), '--unregister-models-user --out "' + OutFile + '"', ExpandConstant('{app}'),
          SW_HIDE, ewWaitUntilTerminated, Rc) and (Rc = 0) then
    if LoadStringsFromFile(OutFile, Lines) and (GetArrayLength(Lines) >= 2) then
    begin
      OtherModelUsers := StrToIntDef(Trim(Lines[0]), -1);
      SharedModelsDir := RemoveBackslash(Trim(Lines[1]));
    end;
  DeleteFile(OutFile);
end;

{ The models are deleted only when no other Voxprint program uses them AND the user says yes (default: keep; a silent uninstall
  with /SUPPRESSMSGBOXES keeps them).  Only the default folder is ever offered: a folder the user chose stays untouched. Voices,
  settings and projects in %LOCALAPPDATA%\Voxprint are never deleted here. }
{ runtime\.users.json: remove our key (bookkeeping only; the runtime is not shared yet, nothing is deleted because of it). }
procedure UnregisterRuntimeUser;
var
  Rc: Integer;
begin
  if FileExists(ExpandConstant('{app}\{#AppExe}')) then
    if not Exec(ExpandConstant('{app}\{#AppExe}'), '--unregister-runtime-user', ExpandConstant('{app}'), SW_HIDE,
                ewWaitUntilTerminated, Rc) then
      Log('runtime users file not updated');
end;

#ifdef ONLINE
// Online builds: the program files were downloaded after setup (they are not in the uninstall log), so DeleteProgramFolder
// removes them at usPostUninstall (it replaces the former UninstallDelete entry for the whole program folder) but never
// touches a sibling Voxprint program installed inside the program folder: a sub-folder with its own uninstaller, or a Movie
// Dubber folder.  The data folder in LOCALAPPDATA (models, voices, state, suite.json) is never deleted here.
function IsSiblingFolder(const Path: String): Boolean;
var
  F: TFindRec;
begin
  Result := Pos('dubber', Lowercase(ExtractFileName(Path))) > 0;
  if (not Result) and FindFirst(AddBackslash(Path) + 'unins*.exe', F) then
  begin
    Result := True;
    FindClose(F);
  end;
end;

procedure DeleteProgramFolder;
var
  F: TFindRec;
  Dir, Path: String;
begin
  Dir := ExpandConstant('{app}');
  if FindFirst(AddBackslash(Dir) + '*', F) then
  begin
    try
      repeat
        if (F.Name <> '.') and (F.Name <> '..') then
        begin
          Path := AddBackslash(Dir) + F.Name;
          if (F.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0 then
          begin
            if not IsSiblingFolder(Path) then
              DelTree(Path, True, True, True);
          end
          else
            DeleteFile(Path);
        end;
      until not FindNext(F);
    finally
      FindClose(F);
    end;
  end;
  RemoveDir(Dir);
end;
#endif

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
  begin
    UnregisterModelsUser;
    UnregisterRuntimeUser;
  end;
#ifdef ONLINE
  if CurUninstallStep = usPostUninstall then
    DeleteProgramFolder;
#endif
  if CurUninstallStep = usPostUninstall then
    if (OtherModelUsers = 0) and (SharedModelsDir <> '') and DirExists(SharedModelsDir) and
       (CompareText(SharedModelsDir, DefaultModelsDir()) = 0) then
      if SuppressibleMsgBox(FmtMessage(CustomMessage('UninstallModelsQuestion'), [SharedModelsDir]),
         mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
        DelTree(SharedModelsDir, True, True, True);
end;
