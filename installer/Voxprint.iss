; Voxprint installer (Inno Setup 6.x). Build with:  ISCC installer\Voxprint.iss   (or through build.bat)
; For the --onedir variant:  ISCC /DONEDIR installer\Voxprint.iss
; Windows 11 x64 only. The models are not part of the installer: the app downloads them itself on first start
; (after the installation, Voxprint can be started right away with the --prefetch flag).
; Optional wizard page "Existing models": the folder with models from a previous install / a Voxprint backup is only
; REMEMBERED (written to %LOCALAPPDATA%\Voxprint\state\existing_models_dir.txt); the installer copies nothing. On the first start
; the program imports the models from there (hard link on the same drive, else copy, checked by checksum) before downloading.
; Silent installs can pass it with /ModelsDir="D:\old\models". Wizard languages: English, Russian, German.
;
; ONLINE variant:  ISCC /DONLINE /DONEDIR /DManifestUrl=<https://.../manifest-beta.json> installer\Voxprint.iss  ->  Output\Voxprint-Setup-online.exe
; A small installer (a few MB + vc_redist): it embeds build\online\voxprint-fetch.exe (tools/online_fetch.py), which downloads the
; payload parts listed in the manifest (release assets, each < 2 GiB, verified by SHA-256, resumable, parts that are already
; installed are skipped) and unpacks them into {app}.  One UAC prompt (PrivilegesRequired=admin) covers everything; the
; user data stays in %LOCALAPPDATA%\Voxprint.  /Manifest=<url or file> overrides the baked-in manifest (tests, mirrors).

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
OutputBaseFilename=Voxprint-Setup-online
ExtraDiskSpaceRequired=4500000000
#else
OutputBaseFilename=Voxprint-Setup
#endif
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
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
english.ModelsPageCaption=Existing models (optional)
russian.ModelsPageCaption=Готовые модели (необязательно)
german.ModelsPageCaption=Vorhandene Modelle (optional)
english.ModelsPageDescription=Do you already have downloaded models from a previous install?
russian.ModelsPageDescription=Есть ли у вас уже скачанные модели от прошлой установки?
german.ModelsPageDescription=Haben Sie bereits heruntergeladene Modelle einer früheren Installation?
english.ModelsPageSubCaption=If so, choose the folder that holds them (for example the old Voxprint models folder or a Voxprint backup). The setup copies nothing: on the first start the program imports the models from there (linked on the same drive, copied otherwise, checked by checksum) instead of downloading them. Leave the field empty to skip this step.
russian.ModelsPageSubCaption=Если да, выберите папку с ними (например, старую папку models от Voxprint или резервную копию Voxprint). Установщик ничего не копирует: при первом запуске программа сама импортирует модели оттуда (ссылкой на том же диске, иначе копированием, с проверкой контрольной суммы) вместо загрузки. Оставьте поле пустым, чтобы пропустить этот шаг.
german.ModelsPageSubCaption=Wenn ja, wählen Sie den Ordner mit den Modellen (z. B. den alten Voxprint-Modellordner oder eine Voxprint-Sicherung). Das Setup kopiert nichts: Beim ersten Start importiert das Programm die Modelle von dort (auf demselben Laufwerk verknüpft, sonst kopiert, per Prüfsumme kontrolliert) statt sie herunterzuladen. Lassen Sie das Feld leer, um diesen Schritt zu überspringen.
english.ModelsPagePrompt=Folder with models:
russian.ModelsPagePrompt=Папка с моделями:
german.ModelsPagePrompt=Ordner mit Modellen:
english.ModelsPageBadFolder=The folder %1 does not exist. Choose an existing folder or clear the field to skip this step.
russian.ModelsPageBadFolder=Папка %1 не существует. Выберите существующую папку или очистите поле, чтобы пропустить этот шаг.
german.ModelsPageBadFolder=Der Ordner %1 existiert nicht. Wählen Sie einen vorhandenen Ordner oder leeren Sie das Feld, um diesen Schritt zu überspringen.

english.OnlineStatus=Downloading and unpacking the Voxprint components (the download can be resumed if it is interrupted)...
russian.OnlineStatus=Загрузка и распаковка компонентов Voxprint (при обрыве загрузку можно продолжить)...
german.OnlineStatus=Voxprint-Komponenten werden geladen und entpackt (bei einem Abbruch kann der Download fortgesetzt werden)...
english.OnlineFailed=The download of the Voxprint components failed:%n%n%1%n%nCheck the internet connection and run the setup again: finished parts are kept and the download continues where it stopped.
russian.OnlineFailed=Не удалось загрузить компоненты Voxprint:%n%n%1%n%nПроверьте подключение к интернету и запустите установку снова: готовые части сохранены, загрузка продолжится с места остановки.
german.OnlineFailed=Der Download der Voxprint-Komponenten ist fehlgeschlagen:%n%n%1%n%nPrüfen Sie die Internetverbindung und starten Sie das Setup erneut: Fertige Teile bleiben erhalten, der Download wird fortgesetzt.
english.OnlineStalled=The downloader stopped responding.
russian.OnlineStalled=Загрузчик перестал отвечать.
german.OnlineStalled=Der Downloader antwortet nicht mehr.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

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
Name: "{autodesktop}\{#AppDisplayName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
#ifexist "redist\vc_redist.x64.exe"
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "{cm:VcRedistStatus}"; Flags: waituntilterminated
#endif
Filename: "{app}\{#AppExe}"; Parameters: "--prefetch"; Description: "{cm:RunPrefetch,{#AppDisplayName}}"; Flags: nowait postinstall skipifsilent

[Code]
var
  ModelsPage: TWizardPage;
  ModelsEdit: TNewEdit;

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

{ Callback of the Browse button on the models page. }
procedure ModelsBrowseClick(Sender: TObject);
var
  Dir: String;
begin
  Dir := ModelsEdit.Text;
  if BrowseForFolder(CustomMessage('ModelsPagePrompt'), Dir, False) then
    ModelsEdit.Text := Dir;
end;

{ Optional page after the install folder: a folder with models from a previous install. Nothing is copied here.
  A plain custom page, NOT CreateInputDirPage: that one refuses an empty field ("You must enter a full path"), which broke
  both the silent install without /ModelsDir and "leave the field empty to skip" (found in the final installer test). }
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
  ModelsEdit.Text := ExpandConstant('{param:ModelsDir|}');
  Browse := TNewButton.Create(ModelsPage);
  Browse.Parent := ModelsPage.Surface;
  Browse.Left := ModelsEdit.Width + ScaleX(8);
  Browse.Top := ModelsEdit.Top - ScaleY(1);
  Browse.Width := ScaleX(88);
  Browse.Height := ModelsEdit.Height + ScaleY(2);
  Browse.Caption := WizardForm.DirBrowseButton.Caption;
  Browse.OnClick := @ModelsBrowseClick;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Dir: String;
begin
  Result := True;
  if CurPageID = ModelsPage.ID then
  begin
    Dir := Trim(ModelsEdit.Text);
    if (Dir <> '') and not DirExists(Dir) then
    begin
      MsgBox(FmtMessage(CustomMessage('ModelsPageBadFolder'), [Dir]), mbError, MB_OK);
      Result := False;
    end;
  end;
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
  Manifest, Cache, StatusFile, Exe, Params, State, Msg, Beat, LastBeat, Err: String;
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

{ Only the path is remembered (UTF-8 file); the app imports the models on its first start. }
procedure CurStepChanged(CurStep: TSetupStep);
var
  Dir, StateDir: String;
  Lines: TArrayOfString;
begin
  if CurStep = ssPostInstall then
  begin
    Dir := Trim(ModelsEdit.Text);
    if Dir <> '' then
    begin
      StateDir := ExpandConstant('{localappdata}\Voxprint\state');
      ForceDirectories(StateDir);
      SetArrayLength(Lines, 1);
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
