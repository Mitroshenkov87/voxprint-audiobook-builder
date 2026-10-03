; Voxprint installer (Inno Setup 6.x). Build with:  ISCC installer\Voxprint.iss   (or through build.bat)
; For the --onedir variant:  ISCC /DONEDIR installer\Voxprint.iss
; Windows 11 x64 only. The models are not part of the installer: the app downloads them itself on first start
; (after the installation, Voxprint can be started right away with the --prefetch flag).

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
; NOTE: the installer wizard texts below are Russian (single-language installer for now; English/German wizard texts are on the roadmap).
MinVersion=10.0.22000
CloseApplications=yes

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "Создать значок на рабочем столе"; GroupDescription: "Дополнительно:"

[Files]
#ifdef ONEDIR
Source: "..\dist\Voxprint\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#else
Source: "..\dist\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
#endif
; Third-party licences (LGPL/GPL/Apache etc.): the component list and the full licence texts.
Source: "..\build\notices\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
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
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Устанавливаю компоненты Microsoft Visual C++…"; Flags: waituntilterminated
#endif
Filename: "{app}\{#AppExe}"; Parameters: "--prefetch"; Description: "Запустить {#AppDisplayName} и скачать модели (рекомендуется, ~7 ГБ, один раз)"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
var
  V: TWindowsVersion;
begin
  Result := True;
  GetWindowsVersionEx(V);
  if V.Build < 26100 then
    Result := MsgBox('{#AppDisplayName} рассчитан на Windows 11 26H2 (сборка 26100 и новее). ' +
      'У вас сборка ' + IntToStr(V.Build) + ': программа может работать некорректно.' + #13#10 +
      'Продолжить установку?', mbConfirmation, MB_YESNO) = IDYES;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\Voxprint');
    if DirExists(DataDir) then
      if MsgBox('Удалить также скачанные модели и журналы (' + DataDir + ')? Они занимают несколько гигабайт.',
         mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
