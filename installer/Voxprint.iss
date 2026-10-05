; Voxprint installer (Inno Setup 6.x). Build with:  ISCC installer\Voxprint.iss   (or through build.bat)
; For the --onedir variant:  ISCC /DONEDIR installer\Voxprint.iss
; Windows 11 x64 only. The models are not part of the installer: the app downloads them itself on first start
; (after the installation, Voxprint can be started right away with the --prefetch flag).
; Wizard page "Models folder" (after the install folder): where the voice models (~7 GB) live. Default = the current location
; %LOCALAPPDATA%\Voxprint\models; the user may pick another folder or drive.  Only the path is REMEMBERED
; (%LOCALAPPDATA%\Voxprint\state\models_dir.txt, read by infra/paths.py: models_dir); the installer copies no model.  The app then
; (1) keeps using models that are already complete in the default Local folder, (2) picks up models the chosen folder already holds
; (Voxprint layout, Hugging Face cache, Voxprint backup), (3) downloads everything missing INTO the chosen folder.  Voices and
; settings stay in %LOCALAPPDATA%\Voxprint.  Program Files is refused (the app runs without admin rights and could not write there).
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
#define AppVersion "0.1.0"
#define AppExe "Voxprint.exe"
#ifndef ManifestUrl
#define ManifestUrl "https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/latest/download/manifest-stable.json"
#endif

[Setup]
AppId={{6F1D2B7A-3C54-4E0B-9A41-7B5E0C9D2F18}
AppName={#AppDisplayName}
AppVersion={#AppVersion}
AppPublisher=Voxprint
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppDisplayName}
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
; Windows 11 24H2+ (build 26100). Older versions are not blocked hard - see [Code]: only a soft warning.
MinVersion=10.0.22000
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"

[CustomMessages]
english.RunPrefetch=Start %1 and set up the models (downloads about 7 GB once, or imports your existing models folder)
russian.RunPrefetch=Запустить %1 и подготовить модели (однократная загрузка ~7 ГБ или импорт из вашей папки с моделями)
german.RunPrefetch=%1 starten und die Modelle einrichten (einmaliger Download von ca. 7 GB oder Import aus Ihrem Modellordner)
english.VcRedistStatus=Installing Microsoft Visual C++ components...
russian.VcRedistStatus=Устанавливаю компоненты Microsoft Visual C++…
german.VcRedistStatus=Microsoft Visual C++-Komponenten werden installiert...
english.WinBuildWarning=%1 is designed for Windows 11 (build 26100 and newer). Your build is %2: the program may not work correctly.%n%nContinue the installation?
russian.WinBuildWarning=%1 рассчитан на Windows 11 (сборка 26100 и новее). У вас сборка %2: программа может работать некорректно.%n%nПродолжить установку?
german.WinBuildWarning=%1 ist für Windows 11 ausgelegt (Build 26100 und neuer). Ihr Build ist %2: Das Programm läuft eventuell nicht korrekt.%n%nInstallation fortsetzen?
english.UninstallDataQuestion=Also delete the downloaded models and logs (%1)? They take several gigabytes.
russian.UninstallDataQuestion=Удалить также скачанные модели и журналы (%1)? Они занимают несколько гигабайт.
german.UninstallDataQuestion=Auch die heruntergeladenen Modelle und Protokolle (%1) löschen? Sie belegen mehrere Gigabyte.
english.ModelsPageCaption=Models folder
russian.ModelsPageCaption=Папка моделей
german.ModelsPageCaption=Modellordner
english.ModelsPageDescription=Where should the voice models (about 7 GB) be stored?
russian.ModelsPageDescription=Где хранить голосовые модели (около 7 ГБ)?
german.ModelsPageDescription=Wo sollen die Sprachmodelle (ca. 7 GB) gespeichert werden?
english.ModelsPageSubCaption=Keep the default or choose another folder or drive. Models already downloaded to the default folder are used as they are; if the chosen folder already holds models (for example from a previous install or a Voxprint backup), they are picked up; anything missing is downloaded into the chosen folder. Your voices and settings stay in the user profile.
russian.ModelsPageSubCaption=Оставьте папку по умолчанию или выберите другую папку или диск. Модели, уже скачанные в папку по умолчанию, используются как есть; если в выбранной папке уже есть модели (например, от прошлой установки или резервной копии Voxprint), они будут подхвачены; недостающие скачиваются в выбранную папку. Ваши голоса и настройки остаются в профиле пользователя.
german.ModelsPageSubCaption=Behalten Sie den Standard oder wählen Sie einen anderen Ordner oder ein anderes Laufwerk. Bereits in den Standardordner geladene Modelle werden weiter verwendet; enthält der gewählte Ordner schon Modelle (z. B. von einer früheren Installation oder einer Voxprint-Sicherung), werden sie übernommen; Fehlendes wird in den gewählten Ordner geladen. Ihre Stimmen und Einstellungen bleiben im Benutzerprofil.
english.ModelsPagePrompt=Models folder:
russian.ModelsPagePrompt=Папка моделей:
german.ModelsPagePrompt=Modellordner:
english.ModelsPageBadFolder=The folder %1 cannot be created or is not writable. Choose another folder.
russian.ModelsPageBadFolder=Папку %1 не удаётся создать или в неё нельзя записывать. Выберите другую папку.
german.ModelsPageBadFolder=Der Ordner %1 kann nicht angelegt werden oder ist nicht beschreibbar. Wählen Sie einen anderen Ordner.
english.ModelsPageProtected=The folder %1 is inside Program Files: the program runs without administrator rights and could not download models there. Choose another folder.
russian.ModelsPageProtected=Папка %1 находится в Program Files: программа работает без прав администратора и не сможет скачивать туда модели. Выберите другую папку.
german.ModelsPageProtected=Der Ordner %1 liegt in Program Files: Das Programm läuft ohne Administratorrechte und könnte dort keine Modelle speichern. Wählen Sie einen anderen Ordner.

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

#ifdef ONLINE
[UninstallDelete]
; the downloaded program files are not in the uninstall log (the setup did not copy them): remove the folder
Type: filesandordirs; Name: "{app}"
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
#ifdef PORTABLE
  PortablePage: TWizardPage;
  PortableCheck: TNewCheckBox;
  PortableEdit: TNewEdit;
  PortableBrowse: TNewButton;
  PortableFound: String;
#endif

function InitializeSetup(): Boolean;
var
  V: TWindowsVersion;
begin
  Result := True;
  GetWindowsVersionEx(V);
  if V.Build < 26100 then
    Result := MsgBox(FmtMessage(CustomMessage('WinBuildWarning'), ['{#AppDisplayName}', IntToStr(V.Build)]),
      mbConfirmation, MB_YESNO) = IDYES;
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

{ What the page shows first: /ModelsFolder=, else the folder remembered by an earlier install, else the default. }
function InitialModelsDir(): String;
var
  S: String;
  Lines: TArrayOfString;
begin
  Result := RemoveBackslash(Trim(ExpandConstant('{param:ModelsFolder|}')));
  if Result <> '' then Exit;
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

{ Page after the install folder: the models folder (default = the current Local location).  Nothing is copied here.
  A plain custom page, NOT CreateInputDirPage (that one behaved badly in silent installs; found in the final installer test). }
procedure InitializeWizard();
var
  Info, Prompt: TNewStaticText;
  Browse: TNewButton;
begin
  ModelsPage := CreateCustomPage(wpSelectDir, CustomMessage('ModelsPageCaption'), CustomMessage('ModelsPageDescription'));
  Info := TNewStaticText.Create(ModelsPage);
  Info.Parent := ModelsPage.Surface;
  Info.WordWrap := True;
  Info.AutoSize := False;
  Info.Left := 0;
  Info.Top := 0;
  Info.Width := ModelsPage.SurfaceWidth;
  Info.Height := ScaleY(90);
  Info.Caption := CustomMessage('ModelsPageSubCaption');
  Prompt := TNewStaticText.Create(ModelsPage);
  Prompt.Parent := ModelsPage.Surface;
  Prompt.Left := 0;
  Prompt.Top := ScaleY(98);
  Prompt.Caption := CustomMessage('ModelsPagePrompt');
  ModelsEdit := TNewEdit.Create(ModelsPage);
  ModelsEdit.Parent := ModelsPage.Surface;
  ModelsEdit.Left := 0;
  ModelsEdit.Top := ScaleY(116);
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
#ifdef PORTABLE
  CreatePortablePage();
#endif
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Dir: String;
begin
  Result := True;
  if CurPageID = ModelsPage.ID then
  begin
    Dir := RemoveBackslash(Trim(ModelsEdit.Text));
    if Dir = '' then
    begin
      Dir := DefaultModelsDir();                  { an emptied field means "the default" }
      ModelsEdit.Text := Dir;
    end;
    if InProgramFiles(Dir) then
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

{ Only paths are remembered (UTF-8 files in state\); no model is copied.  models_dir.txt = the models folder (deleted when it is the
  default, so the app follows its own default); existing_models_dir.txt = an optional folder to import models from (/ModelsDir=). }
procedure CurStepChanged(CurStep: TSetupStep);
var
  Dir, StateDir: String;
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
    if (Dir = '') or (CompareText(Dir, DefaultModelsDir()) = 0) then
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
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\Voxprint');
    if DirExists(DataDir) then
      if MsgBox(FmtMessage(CustomMessage('UninstallDataQuestion'), [DataDir]),
         mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
