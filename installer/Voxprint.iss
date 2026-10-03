; Voxprint installer (Inno Setup 6.x). Build with:  ISCC installer\Voxprint.iss   (or through build.bat)
; For the --onedir variant:  ISCC /DONEDIR installer\Voxprint.iss
; Windows 11 x64 only. The models are not part of the installer: the app downloads them itself on first start
; (after the installation, Voxprint can be started right away with the --prefetch flag).
; Optional wizard page "Existing models": the folder with models from a previous install / a Voxprint backup is only
; REMEMBERED (written to %LOCALAPPDATA%\Voxprint\state\existing_models_dir.txt); the installer copies nothing. On the first start
; the program imports the models from there (hard link on the same drive, else copy, checked by checksum) before downloading.
; Silent installs can pass it with /ModelsDir="D:\old\models". Wizard languages: English, Russian, German.

#define AppName "Voxprint"
; Name shown to the user (wizard, Start menu, Apps list). AppName stays technical: it is the install folder and the data folder name.
#define AppDisplayName "Voxprint AI Audiobook Builder"
#define AppVersion "0.1.0"
#define AppExe "Voxprint.exe"

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
OutputBaseFilename=Voxprint-Setup
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

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
#ifdef ONEDIR
Source: "..\dist\Voxprint\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#else
Source: "..\dist\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
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
