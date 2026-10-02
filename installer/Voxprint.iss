; Установщик Voxprint (Inno Setup 6.x). Сборка:  ISCC installer\Voxprint.iss   (или через build.bat)
; Для варианта --onedir:  ISCC /DONEDIR installer\Voxprint.iss
; Только Windows 11 x64. Модели не входят в установщик: приложение само скачивает их при первом запуске
; (после установки предлагается сразу запустить Voxprint с флагом --prefetch).

#define AppName "Voxprint"
#define AppVersion "0.1.0"
#define AppExe "Voxprint.exe"

[Setup]
AppId={{6F1D2B7A-3C54-4E0B-9A41-7B5E0C9D2F18}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Voxprint
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExe}
OutputDir=Output
OutputBaseFilename=Voxprint-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Windows 11 24H2+ (сборка 26100). Более старые версии не блокируем жёстко - см. [Code]: только мягкое предупреждение.
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
; Необязательно: положите vc_redist.x64.exe в installer\redist, и он будет установлен тихо (нужен PyTorch).
#ifexist "redist\vc_redist.x64.exe"
Source: "redist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
#endif

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
#ifexist "redist\vc_redist.x64.exe"
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Устанавливаю компоненты Microsoft Visual C++…"; Flags: waituntilterminated
#endif
Filename: "{app}\{#AppExe}"; Parameters: "--prefetch"; Description: "Запустить Voxprint и скачать модели (рекомендуется, ~7 ГБ, один раз)"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
var
  V: TWindowsVersion;
begin
  Result := True;
  GetWindowsVersionEx(V);
  if V.Build < 26100 then
    Result := MsgBox('Voxprint рассчитан на Windows 11 26H2 (сборка 26100 и новее). ' +
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
